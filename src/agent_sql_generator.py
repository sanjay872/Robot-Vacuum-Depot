from __future__ import annotations
from typing import TypedDict, Optional
import os
import json
import re

import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# 1. STATE DEFINITION
# =====================================================

class AgentState(TypedDict, total=False):
    question: str
    sql: Optional[str]
    df: Optional[pd.DataFrame]
    error: Optional[str]


# =====================================================
# 2. POSTGRES ENGINE SETUP
# =====================================================

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "admin")
PG_DB = os.getenv("PG_DB", "postgres")

DATABASE_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

engine = create_engine(DATABASE_URL)


# =====================================================
# 3. DYNAMIC SCHEMA NAME DETECTION
# =====================================================

def get_active_schema_name() -> str:
    """
    Automatically determine which schema to use.
    Selects the first non-system schema.
    """
    q = """
    SELECT DISTINCT table_schema
    FROM information_schema.tables
    WHERE table_type='BASE TABLE'
      AND table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema
    LIMIT 1;
    """
    df = pd.read_sql_query(q, engine)
    if df.empty:
        raise RuntimeError("No active user schema found.")

    return df.iloc[0]["table_schema"]


# =====================================================
# 4. SCHEMA JSON BUILDER
# =====================================================

def generate_schema_info() -> tuple[str, dict, str]:
    """
    Returns schema_name, schema_dict, and schema_json (str).
    """
    schema_name = get_active_schema_name()

    q = f"""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '{schema_name}'
    ORDER BY table_name, ordinal_position;
    """

    df = pd.read_sql_query(q, engine)

    schema_dict = {}
    for tbl in df["table_name"].unique():
        sub = df[df["table_name"] == tbl]
        schema_dict[tbl] = {
            row["column_name"]: row["data_type"]
            for _, row in sub.iterrows()
        }

    schema_json = json.dumps(schema_dict, indent=2)

    return schema_name, schema_dict, schema_json


# =====================================================
# 5. LLM PROMPT (Uses {schema_name} and {schema_json})
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
You are an expert SQL generator.

Your job is to convert the user's natural-language question
into a single correct PostgreSQL SELECT query.

============================================================
RULES (STRICT)
============================================================

1. **Use ONLY the tables and columns shown in the schema JSON.**
   Do NOT invent tables or columns.

2. **Always prefix tables with the schema name:**
       {schema_name}.table_name

3. **Do NOT prefix column names with the schema.**

4. **Join tables ONLY on columns that exist in multiple tables.**
   (You must infer relationships from schema_json — no hallucination.)

5. **Use WHERE filters only on columns existing in schema_json.**

6. If the question implies:
   - Aggregation → Use GROUP BY, ORDER BY
   - Comparison → Use appropriate numeric columns
   - Filtering → Match natural-language keywords to correct TEXT columns

7. Output ONLY the SQL (no markdown, no comments, no explanation).

============================================================
ACTIVE SCHEMA NAME:
{schema_name}

AVAILABLE TABLES & COLUMNS (GROUND TRUTH):
{schema_json}
============================================================
"""
    ),
    ("human",
     "Convert the following question into SQL:\n\n{question}\n\nSQL:")
])

sql_chain = sql_prompt | llm | StrOutputParser()


# =====================================================
# 6. CLEAN SQL OUTPUT
# =====================================================

def clean_sql_output(raw: str) -> str:
    sql = raw.strip()
    if sql.startswith("```"):
        sql = sql.split("```")[1]
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.rstrip(";").strip()

    if not sql.lower().startswith("select"):
        raise ValueError(f"Not a SELECT query:\n{sql}")

    return sql


# =====================================================
# 7. STATIC SHAPE VALIDATION
# =====================================================

def sql_shape_error(sql: str) -> Optional[str]:
    lower = sql.lower()

    if not lower.startswith("select"):
        return "SQL must start with SELECT."

    # prevent harmful statements
    if any(x in lower for x in ["update ", "delete ", "insert ", "drop "]):
        return "Only SELECT queries allowed."

    # parentheses balance
    depth = 0
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return "Unbalanced parentheses."
    if depth != 0:
        return "Unbalanced parentheses."

    return None


# =====================================================
# 8. SEMANTIC SCHEMA VALIDATION
# =====================================================

def schema_semantic_error(sql: str, schema_dict: dict, schema_name: str) -> Optional[str]:
    lower = sql.lower()

    # detect referenced tables
    used_tables = set()
    for table in schema_dict.keys():
        if f"{schema_name.lower()}.{table.lower()}" in lower:
            used_tables.add(table)

    # map columns → tables
    col_map = {}
    for table, cols in schema_dict.items():
        for col in cols.keys():
            col_map.setdefault(col.lower(), set()).add(table)

    bad_cols = []
    for col in col_map.keys():
        pattern = rf"\\b{col}\\b"
        if re.search(pattern, lower):
            if used_tables and used_tables.isdisjoint(col_map[col]):
                bad_cols.append(col)

    if bad_cols:
        return f"Columns not valid for the referenced tables: {bad_cols}"

    return None


# =====================================================
# 9. SQL GENERATION NODE
# =====================================================

def node_generate_sql(state: AgentState) -> AgentState:
    question = state.get("question")
    if not question:
        return {**state, "error": "No question provided."}

    schema_name, schema_dict, schema_json = generate_schema_info()

    last_error = None
    current_question = question

    for attempt in range(3):
        try:
            raw = sql_chain.invoke({
                "question": current_question,
                "schema_json": schema_json,
                "schema_name": schema_name
            })

            cleaned = clean_sql_output(raw)

            shape_err = sql_shape_error(cleaned)
            semantic_err = schema_semantic_error(cleaned, schema_dict, schema_name)

            if not shape_err and not semantic_err:
                return {**state, "sql": cleaned, "error": None}

            # retry
            last_error = " | ".join(e for e in [shape_err, semantic_err] if e)

            current_question = (
                f"{question}\n\n"
                f"The SQL you generated was invalid:\n{last_error}\n"
                f"Invalid SQL:\n{cleaned}\n\n"
                f"Try again using schema name '{schema_name}'."
            )

        except Exception as e:
            last_error = str(e)

    return {**state, "sql": None, "error": f"SQL generation failed: {last_error}"}


# =====================================================
# 10. SQL EXECUTION NODE
# =====================================================

def node_run_sql(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    sql = state.get("sql")
    if not sql:
        return {**state, "error": "No SQL to execute."}

    try:
        df_pl = pl.read_database(sql, connection=engine)
        df_pd = df_pl.to_pandas()
        return {**state, "df": df_pd, "error": None}
    except Exception as e:
        return {**state, "error": f"SQL execution failed: {e}"}


# =====================================================
# 11. BUILD LANGGRAPH APP
# =====================================================

builder = StateGraph(AgentState)
builder.add_node("generate_sql", node_generate_sql)
builder.add_node("run_sql", node_run_sql)

builder.set_entry_point("generate_sql")
builder.add_edge("generate_sql", "run_sql")
builder.add_edge("run_sql", END)

sql_agent_app = builder.compile()
