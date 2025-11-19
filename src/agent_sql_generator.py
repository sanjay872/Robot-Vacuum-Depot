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
PG_DB = os.getenv("PG_DB", "robot_vacuum")

DATABASE_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    "?options=-csearch_path=robot_vacuum"
)

engine = create_engine(DATABASE_URL)


# =====================================================
# 3. SCHEMA LOADING
# =====================================================

def generate_schema_json() -> str:
    """
    Loads the live PostgreSQL schema and returns it as JSON.
    """
    query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'robot_vacuum'
    ORDER BY table_name, ordinal_position;
    """

    df = pd.read_sql_query(query, engine)
    schema: dict[str, dict[str, str]] = {}

    for table in df["table_name"].unique():
        table_cols = df[df["table_name"] == table]
        schema[table] = {
            row["column_name"]: row["data_type"]
            for _, row in table_cols.iterrows()
        }

    return json.dumps(schema, indent=2)


# =====================================================
# 4. LLM + PROMPT
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a PostgreSQL expert. Convert natural language into a correct SQL query.

STRICT RULES:
1. Use ONLY these tables (schema robot_vacuum):
      - orders
      - customer
      - product
      - review
      - manufacturer
      - warehouse
      - shipment
   (exact list depends on actual schema; use only what is in the schema JSON below)

2. Use only columns that exist in the schema JSON.
3. Table names MUST be referenced as:
        robot_vacuum.orders
        robot_vacuum.customer
        robot_vacuum.product
        robot_vacuum.review
        robot_vacuum.manufacturer
        robot_vacuum.warehouse
   (only if they exist in the schema).

4. Column references NEVER include the schema:
        orders.productid       ✔
        robot_vacuum.orders.productid   ✘

5. ALWAYS produce a single SELECT query.
6. Include proper JOIN conditions when using multiple tables.
7. Never hallucinate columns.
8. delayed deliveries → LOWER(orders.deliverystatus) LIKE '%delayed%'
9. Chicago → orders.deliveryzipcode LIKE '606%'

LIVE DATABASE SCHEMA (use this as ground truth):
{schema_json}

Return ONLY the SQL. No commentary. No markdown. No backticks.
"""
     ),
    ("human",
     "Convert this question into SQL:\n\n{question}\n\nSQL:")
])

sql_chain = sql_prompt | llm | StrOutputParser()


# =====================================================
# 5. SQL CLEANING + BASIC SHAPE CHECKS
# =====================================================

def clean_sql_output(raw: str) -> str:
    """
    Strip markdown fences, backticks, trailing semicolons, and extra text.
    """
    sql = raw.strip()

    # Remove code fences
    if sql.startswith("```"):
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1]
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.replace("`", "").strip()

    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()

    if not sql.lower().startswith("select"):
        raise ValueError(f"LLM returned invalid SQL (must start with SELECT): {sql}")

    return sql


def _sql_shape_error(sql: str) -> Optional[str]:
    """
    Cheap static checks on the SQL shape. Returns an error string if something
    looks obviously wrong, otherwise None.
    """
    s = sql.strip()
    lower = s.lower()

    if not lower.startswith("select"):
        return "SQL must start with SELECT."

    if " delete " in lower or " update " in lower or " insert " in lower:
        return "Only SELECT queries are allowed."

    # Parentheses balance
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return "Unbalanced parentheses."
    if depth != 0:
        return "Unbalanced parentheses."

    if "count()" in lower:
        return "COUNT() is missing an argument; use COUNT(*) or COUNT(col)."

    return None


# =====================================================
# 6. SCHEMA SEMANTIC CHECK — CATCH BAD COLUMNS
# =====================================================

def _schema_semantic_error(sql: str, schema_dict: dict) -> Optional[str]:
    """
    Check if columns used in the SQL are compatible with the tables used.

    If a column is referenced but none of the tables in FROM/JOIN contain it,
    we flag it as a semantic schema error.

    Example: manufacturerid used, but only robot_vacuum.review is in FROM.
    """
    lower = sql.lower()

    # 1) Detect which tables are used in the query
    used_tables: set[str] = set()
    for table_name in schema_dict.keys():
        t_lower = table_name.lower()
        if f"robot_vacuum.{t_lower}" in lower or re.search(rf"\\b{re.escape(t_lower)}\\b", lower):
            used_tables.add(table_name)

    # If we couldn't detect tables, don't block on this check
    if not used_tables:
        return None

    # 2) Build column -> set(tables) mapping from schema
    col_to_tables: dict[str, set[str]] = {}
    for tname, cols in schema_dict.items():
        for col_name in cols.keys():
            col_to_tables.setdefault(col_name.lower(), set()).add(tname)

    # 3) For each known column name, see if it appears in SQL.
    bad_cols: list[str] = []
    for col_name, tables_having in col_to_tables.items():
        pattern = rf"\\b{re.escape(col_name.lower())}\\b"
        if re.search(pattern, lower):
            # Column used: ensure at least one used table actually has it
            if used_tables.isdisjoint(tables_having):
                bad_cols.append(col_name)

    if bad_cols:
        return (
            "Some columns are referenced in the SQL but do not exist in any of the "
            f"tables used in FROM/JOIN: {sorted(set(bad_cols))}. "
            "You likely forgot to JOIN the appropriate table(s) from the schema "
            "or selected the wrong source table."
        )

    return None


# =====================================================
# 7. MAIN SQL GENERATION NODE (WITH RETRIES)
# =====================================================

def node_generate_sql(state: AgentState) -> AgentState:
    question = state.get("question")
    if not question:
        return {**state, "error": "No question provided."}

    try:
        schema_json = generate_schema_json()
        schema_dict = json.loads(schema_json)
    except Exception as e:
        return {**state, "error": f"Failed to load database schema: {e}"}

    last_error = None
    current_question = question

    for attempt in range(1, 4):
        try:
            raw_sql = sql_chain.invoke({
                "question": current_question,
                "schema_json": schema_json,
            })

            cleaned = clean_sql_output(raw_sql)

            shape_err = _sql_shape_error(cleaned)
            semantic_err = _schema_semantic_error(cleaned, schema_dict)

            if shape_err is None and semantic_err is None:
                # Looks good
                return {**state, "sql": cleaned, "error": None}

            # Compose error message
            msgs = []
            if shape_err:
                msgs.append(shape_err)
            if semantic_err:
                msgs.append(semantic_err)
            combined_msg = " | ".join(msgs)
            last_error = combined_msg

            # Ask LLM to fix it with detailed feedback
            current_question = (
                f"{question}\n\n"
                f"The previous SQL you generated was invalid because: {combined_msg}\n"
                f"Here is the invalid SQL:\n{cleaned}\n\n"
                "Please regenerate a correct PostgreSQL SELECT query that strictly follows the schema, "
                "uses only existing columns, and includes proper joins between the relevant tables."
            )

        except Exception as e:
            last_error = str(e)
            current_question = (
                f"{question}\n\n"
                f"Previous SQL generation attempt failed with error: {e}.\n"
                "Please try again with a correct PostgreSQL SELECT query."
            )

    # If we reach here, all attempts failed
    return {
        **state,
        "sql": None,
        "error": f"SQL generation failed after multiple attempts. Last error: {last_error}",
    }


# =====================================================
# 8. SQL EXECUTION NODE
# =====================================================

def node_run_sql(state: AgentState) -> AgentState:
    if state.get("error"):
        # If there is already an error, do not run SQL
        return state

    sql = state.get("sql")
    if not sql:
        return {**state, "error": "No SQL to execute."}

    try:
        df_pl = pl.read_database(sql, connection=engine)
        df_pd = df_pl.to_pandas()

        if df_pd.empty:
            return {**state, "df": df_pd, "error": "SQL executed successfully but returned no rows."}

        return {**state, "df": df_pd, "error": None}

    except Exception as e:
        # Execution-level error (e.g., permissions, type mismatch, etc.)
        return {**state, "error": f"SQL execution failed: {e}"}


# =====================================================
# 9. BUILD GRAPH
# =====================================================

builder = StateGraph(AgentState)
builder.add_node("generate_sql", node_generate_sql)
builder.add_node("run_sql", node_run_sql)

builder.set_entry_point("generate_sql")
builder.add_edge("generate_sql", "run_sql")
builder.add_edge("run_sql", END)

sql_agent_app = builder.compile()


# =====================================================
# 10. LOCAL TEST
# =====================================================

if __name__ == "__main__":
    sample = "Among all manufacturers, who has the best average review rating for their products?"
    result = sql_agent_app.invoke({"question": sample})
    print("SQL:", result.get("sql"))
    print("Error:", result.get("error"))
    df_res = result.get("df")
    if isinstance(df_res, pd.DataFrame):
        print(df_res.head())
