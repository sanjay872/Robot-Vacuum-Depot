# -------------------------------------------------------
# Agent 3 — Code Validator Agent (FINAL FIXED VERSION)
# -------------------------------------------------------

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langgraph.graph import StateGraph, END
import json


# -----------------------------
# Agent 3 State
# -----------------------------
class ValidatorState(TypedDict, total=False):
    code: str
    df_json: str
    is_valid: Optional[bool]
    feedback: Optional[str]


# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# -----------------------------
# Validator Prompt (FIXED)
# -----------------------------
validator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a strict Python visualization code validator.

You receive:
1. df_json (the dataframe as JSON)
2. Python visualization code.

Your job:
- Validate whether the code is syntactically correct.
- Don't worry about the dataframe (df) declaration
- Ensure df columns referenced in the code actually exist.
- Ensure the code would generate a chart/table (matplotlib, plotly, seaborn allowed).
- DO NOT EXECUTE CODE — reason only.

Your output MUST be valid JSON:
  "is_valid": "true" or "false",
  "feedback": "short explanation"

ONLY output JSON. Nothing else.
"""
     ),
    ("human",
     "DataFrame JSON:\n{df_json}\n\nVisualization Code:\n```\n{code}\n```\n\nReturn ONLY the JSON:")
])

validator_chain = validator_prompt | llm | StrOutputParser()


# -----------------------------
# Node (FIXED)
# -----------------------------
def node_validate(state: ValidatorState):
    print(state)
    raw_json = validator_chain.invoke({
        "df_json": state["df_json"],
        "code": state["code"]
    })

    try:
        parsed = json.loads(raw_json)
        return {
            "is_valid": parsed.get("is_valid", False),
            "feedback": parsed.get("feedback", "")
        }
    except Exception:
        return {
            "is_valid": False,
            "feedback": "Validator output invalid JSON. Raw: " + raw_json
        }


# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(ValidatorState)
builder.add_node("validate", node_validate)
builder.set_entry_point("validate")
builder.add_edge("validate", END)

validator_app = builder.compile()
