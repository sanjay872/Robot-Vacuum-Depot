# agent_code_generation.py

import json
import pandas as pd
from typing import TypedDict, Optional, Dict, Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from langgraph.graph import StateGraph, END

# ----------------------------------------
#  AGENT 2 STATE
# ----------------------------------------

class VizState(TypedDict):
    df_json: str           # required input
    question: Optional[str]
    viz_code: Optional[str]
    error: Optional[str]


# ----------------------------------------
#  LLM FOR GENERATING VISUALIZATION CODE
# ----------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

viz_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior data visualization engineer.

You receive a pandas DataFrame (as JSON) and must generate ONLY Python code
(using pandas, plotly, or matplotlib) that visualizes the data effectively.

Rules:
- MUST return ONLY Python code. No explanation.
- MUST assume the DataFrame will be named `df` in the Streamlit environment.
- Add all required imports for the code to run.
- Choose the BEST visualization based on column types.
- If categorical + numeric → bar chart.
- If numeric + numeric → scatter or line.
- If only numeric → histogram.
- If many columns → table preview.
- NEVER import pandas (already available).
- You MAY import matplotlib.pyplot or plotly.express.
- Keep code self-contained.
"""
    ),
    ("human", "DataFrame columns: {columns}\nSample rows: {preview}\nGenerate visualization code:"),
])

viz_chain = viz_prompt | llm | StrOutputParser()


# ----------------------------------------
#  NODES
# ----------------------------------------

def node_generate_viz(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df_json = state["df_json"]
        df = pd.read_json(df_json)

        preview = df.head(5).to_dict(orient="records")
        columns = list(df.columns)

        code = viz_chain.invoke({
            "columns": columns,
            "preview": preview,
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


# ----------------------------------------
#  GRAPH DEFINITION
# ----------------------------------------

builder = StateGraph(VizState)
builder.add_node("generate_viz", node_generate_viz)
builder.set_entry_point("generate_viz")
builder.add_edge("generate_viz", END)

viz_agent_app = builder.compile()

# Expose the chain for Streamlit
__all__ = ["viz_agent_app"]
