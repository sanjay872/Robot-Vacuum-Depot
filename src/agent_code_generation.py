# ----------------------------------------------------
# Agent 2 — Visualization Code Generator (Standalone Code)
# ----------------------------------------------------

from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List

import pandas as pd
from io import StringIO

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


class VizState(TypedDict, total=False):
    df_json: str          # JSON string of df (records)
    question: Optional[str]
    viz_code: Optional[str]
    error: Optional[str]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

viz_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior Python data visualization engineer.

Environment:
- The code you generate will be executed in a Python REPL.
- A global variable `_df_json` (string) is already defined there.
  It contains the JSON representation of the DataFrame in records format.

Your job:
- Generate a COMPLETE, STANDALONE Python script that:
  - Imports necessary modules.
  - Reconstructs `df` from `_df_json`.
  - depend on the data and given user question determine the type of chart that is best to visualise it, generate code for that.
  - Builds exactly ONE visualization (matplotlib or plotly).
  - Does NOT rely on any external state except `_df_json`.

REQUIRED STEPS IN THE GENERATED CODE:
1. Import the modules you need, e.g.:
   - json
   - from io import StringIO
   - import pandas as pd
   - import matplotlib.pyplot as plt
   - (optionally) import plotly.express as px

2. Rebuild the DataFrame from `_df_json`, for example:
   - Use `pd.read_json(StringIO(_df_json))`
   or
   - Use `json.loads(_df_json)` and then `pd.DataFrame(...)`

3. Use the reconstructed `df` to create ONE visualization that best answers
   the user's question, based on:
   - Column names
   - Data types (numeric, datetime-like, categorical)
   - Question intent (trend, comparison, distribution, proportion, etc.)

MATPLOTLIB RULES:
- Always call `plt.figure()` before plotting.
- You may use `plt.plot`, `plt.bar`, `plt.hist`, `plt.scatter`, `plt.pie`, etc.
- Prefer readable axis labels and a title.
- You may call `plt.tight_layout()`.
- Do NOT call `plt.show()` (the caller will handle rendering).

PLOTLY RULES (optional):
- If you use plotly, assign the figure to a variable named `fig`, e.g.:
    fig = px.bar(...)
- Do NOT call `fig.show()`.

SAFETY RULES:
- Do NOT read or write files.
- Do NOT use eval, exec, os, subprocess, sys, input, or network calls.
- Do NOT wrap code in markdown or backticks.
- Output MUST be pure Python code only.

If the question is ambiguous, choose a reasonable default chart (e.g. bar chart
for category vs numeric, line chart for time vs numeric, histogram for a single numeric column).
"""
    ),
    (
        "human",
        """
User question:
{question}

Columns:
{columns}

Column dtypes:
{dtypes}

Numeric columns:
{numeric_columns}

Datetime-like columns:
{datetime_columns}

Categorical columns:
{categorical_columns}

Data preview (first 5 rows):
{preview}

Generate the COMPLETE Python script now:
"""
    )
])

viz_chain = viz_prompt | llm | StrOutputParser()


def node_generate_viz(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df_json = state["df_json"]
        question = state.get("question", "")

        # Load df locally only for schema analysis
        df = pd.read_json(StringIO(df_json))

        columns: List[str] = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]

        datetime_columns = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
            or "date" in col.lower()
            or "time" in col.lower()
        ]

        categorical_columns = [
            col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
            and df[col].nunique(dropna=True) <= 30
        ]

        preview = df.head(5).to_dict(orient="records")

        code = viz_chain.invoke({
            "question": question,
            "columns": columns,
            "dtypes": dtypes,
            "numeric_columns": numeric_columns,
            "datetime_columns": datetime_columns,
            "categorical_columns": categorical_columns,
            "preview": preview,
        })

        return {
            "viz_code": code,
            "question": question,
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
