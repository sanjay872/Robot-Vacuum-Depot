from __future__ import annotations
from typing import TypedDict, Optional
import os
import json
import pandas as pd
from sqlalchemy import text

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
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
# 2. POSTGRES + SQLALCHEMY ENGINE
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
# 3. DYNAMIC SCHEMA LOADING
# =====================================================

def generate_schema_json() -> str:
    """
    Reads schema metadata dynamically from PostgreSQL
    and returns a JSON structure for LLM prompt.
    """
    query = """
    SELECT 
        table_name,
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'robot_vacuum'
    ORDER BY table_name, ordinal_position;
    """

    df = pd.read_sql_query(query, engine)

    schema = {}
    for table in df["table_name"].unique():
        cols = df[df["table_name"] == table]
        schema[table] = {
            row["column_name"]: row["data_type"]
            for _, row in cols.iterrows()
        }

    return json.dumps(schema, indent=2)


# =====================================================
# 4. LLM SETUP + UPDATED PROMPT
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

sql_prompt = ChatPromptTemplate.from_messages([
("system",
"""
You are a PostgreSQL SQL expert. Always follow these rules:

1. Fully qualify ONLY table names:
      robot_vacuum.orders
      robot_vacuum.product
      robot_vacuum.customer
   → NEVER write robot_vacuum.table.column  
     (this is invalid).

   Correct:
      robot_vacuum.orders
      orders.productid

   Incorrect:
      robot_vacuum.orders.productid

2. Use table aliases when helpful.

3. ONLY generate a single SELECT query.
   - No comments
   - No non-SELECT SQL

4. Use only columns present in the schema.

5. For delayed deliveries:
      LOWER(orders.deliverystatus) LIKE '%delayed%'

6. For Chicago ZIP codes:
      orders.deliveryzipcode LIKE '606%'

---------------------------------------------------------
LIVE DATABASE SCHEMA:
{schema_json}
---------------------------------------------------------

Return ONLY the SQL query.
""")
,
("human", "Question: {question}\nSQL:")
])

sql_chain = sql_prompt | llm | StrOutputParser()


def clean_sql_output(raw: str) -> str:
    """Cleans markdown fences and validates SELECT-only."""
    sql = raw.strip()

    if sql.lower().startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    sql = sql.strip().rstrip(";")

    if not sql.lower().lstrip().startswith("select"):
        raise ValueError(f"LLM returned non-SELECT SQL:\n{sql}")

    return sql


# =====================================================
# 5. LANGGRAPH NODES
# =====================================================

def node_generate_sql(state: AgentState) -> AgentState:
    question = state.get("question")
    if not question:
        return {**state, "error": "No question provided."}

    try:
        schema_json = generate_schema_json()

        raw_sql = sql_chain.invoke({
            "question": question,
            "schema_json": schema_json
        })

        sql = clean_sql_output(raw_sql)

        return {**state, "sql": sql, "error": None}
    except Exception as e:
        return {**state, "error": f"SQL generation failed: {e}"}


def node_run_sql(state: AgentState) -> AgentState:
    sql = state.get("sql")

    if not sql:
        return state

    if state.get("error"):
        return state

    try:
        df = pd.read_sql_query(text(sql), engine)
        return {**state, "df": df, "error": None}
    except Exception as e:
        return {**state, "error": f"SQL execution failed: {e}"}


# =====================================================
# 6. BUILD LANGGRAPH
# =====================================================

builder = StateGraph(AgentState)

builder.add_node("generate_sql", node_generate_sql)
builder.add_node("run_sql", node_run_sql)

builder.set_entry_point("generate_sql")
builder.add_edge("generate_sql", "run_sql")
builder.add_edge("run_sql", END)

sql_agent_app = builder.compile()


# =====================================================
# 7. LOCAL TEST
# =====================================================

if __name__ == "__main__":
    sample_question = "Which products have the highest number of delayed deliveries in Chicago?"

    state = {
        "question": sample_question
    }

    result = sql_agent_app.invoke(state)

    print("Generated SQL:")
    print(result.get("sql"))

    print("\nError:", result.get("error"))

    df = result.get("df")
    if isinstance(df, pd.DataFrame):
        print("\nResult:")
        print(df.head())
    else:
        print("No DataFrame returned.")
