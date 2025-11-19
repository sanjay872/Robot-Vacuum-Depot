# -------------------------------------------------------
# Agent 3 — Simple Syntax Validator (Stable Version)
# -------------------------------------------------------

from typing import TypedDict, Optional
import ast
from langgraph.graph import StateGraph, END


class ValidatorState(TypedDict, total=False):
    code: str
    df_json: str
    is_valid: Optional[bool]
    feedback: Optional[str]


def node_validate(state: ValidatorState):
    """
    Very simple, robust validator:
    - Checks ONLY Python syntax.
    - If code parses, it's considered valid.
    - Column mismatches / runtime issues are handled by Agent 4.
    """
    code = state.get("code", "")

    if not code.strip():
        return {
            "is_valid": False,
            "feedback": "No code provided to validate."
        }

    try:
        ast.parse(code)
        # If parsing succeeds, we trust it. Runtime errors will be caught by Agent 4.
        return {
            "is_valid": True,
            "feedback": "Syntax is valid. Ready to execute."
        }
    except SyntaxError as e:
        return {
            "is_valid": False,
            "feedback": f"Python syntax error: {e}"
        }
    except Exception as e:
        return {
            "is_valid": False,
            "feedback": f"Unexpected error during validation: {e}"
        }


# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(ValidatorState)
builder.add_node("validate", node_validate)
builder.set_entry_point("validate")
builder.add_edge("validate", END)

validator_app = builder.compile()
