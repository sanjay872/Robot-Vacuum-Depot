# agent_code_generation.py — FINAL D VERSION (Row-count Aware)

import json
import pandas as pd
import numpy as np
from typing import TypedDict, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
import os

load_dotenv()  # Automatically loads .env from current directory


class VizState(TypedDict):
    df_json: str
    question: Optional[str]
    viz_code: Optional[str]
    error: Optional[str]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ================================================================
# FIXED PROMPT WITH ROW-COUNT LOGIC + REAL COLUMNS ONLY
# ================================================================
viz_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior Python data visualization engineer.

Your job is to generate CLEAN, EXECUTABLE Matplotlib code using REAL column names.
The DataFrame is available as `df`.

STRICT RULES:
---------------------------------------------------------
❌ NEVER generate placeholder column names like:
    'timestamp_column', 'numeric_column', 'category_column'
❌ NEVER use select_dtypes placeholders
❌ NEVER ask the user to replace anything
❌ NEVER output backticks or comments
❌ NEVER use Streamlit

✔ ALWAYS use REAL columns only from the provided list {columns}
✔ ALWAYS return only valid executable code
✔ ALWAYS call:
       plt.figure()
       ...
       plt.tight_layout()

---------------------------------------------------------
DATETIME NORMALIZATION (MANDATORY):
---------------------------------------------------------
Identify datetime-like columns:
- Columns with dtype integer AND values > 1e9 (epoch timestamps)
- Columns that pandas can parse as dates
Then convert:
    if v > 1e12 → unit='ns'
    elif v > 1e10 → unit='ms'
    else → unit='s'
Sort the DataFrame by that datetime column.

---------------------------------------------------------
ROW COUNT LOGIC (MANDATORY):
---------------------------------------------------------
Use len(df):

If len(df) == 0:
    Create a table-like fallback using ax.table.

If len(df) == 1:
    • If time-series → scatter plot with a single point
    • If categorical → bar with 1 bar
    • If numeric only → hist with 1 bucket

If len(df) == 2:
    • If time-series → line plot with markers
    • If numeric → scatter

If len(df) >= 3:
    • Use appropriate chart type normally.

---------------------------------------------------------
CHART TYPE DETECTION:
---------------------------------------------------------
Priority:
1. If question mentions: plot, chart, line, trend → LINE if datetime present
2. If question mentions: distribution, percentage, proportion → PIE
3. If categorical + numeric → BAR
4. If 2 numeric → SCATTER
5. If 1 numeric → HIST
6. Else → TABLE fallback

---------------------------------------------------------
RETURN ONLY EXECUTABLE PYTHON CODE.
"""
    ),
    (
        "human",
        """
User question: {question}

Columns: {columns}
Preview rows: {preview}

Generate only Python code using REAL column names.
"""
    )
])

viz_chain = viz_prompt | llm | StrOutputParser()


def node_generate_viz(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df_json = state["df_json"]
        df = pd.read_json(df_json)

        preview = df.head(5).to_dict(orient="records")
        columns = list(df.columns)

        code = viz_chain.invoke({
            "question": state.get("question", ""),
            "columns": columns,
            "preview": preview
        })

        return {
            "viz_code": code,
            "question": state.get("question"),
            "df_json": df_json,
            "error": None,
        }

    except Exception as e:
        return {
            "viz_code": None,
            "question": state.get("question"),
            "df_json": state.get("df_json"),
            "error": str(e),
        }


builder = StateGraph(VizState)
builder.add_node("generate_viz", node_generate_viz)
builder.set_entry_point("generate_viz")
builder.add_edge("generate_viz", END)

viz_agent_app = builder.compile()

__all__ = ["viz_agent_app"]
