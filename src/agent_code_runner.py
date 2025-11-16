# agent_code_runner.py  (FINAL — FULLY FIXED)

import io
import ast
import traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")      # ← Prevent any GUI windows
import matplotlib.pyplot as plt

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda


# ---------------------------------------------------
# SAFE IMPORT ROOTS
# ---------------------------------------------------
SAFE_IMPORT_ROOTS = {
    "matplotlib",
    "seaborn",
    "plotly",
    "pandas",
    "numpy"
}


# ---------------------------------------------------
# CODE SAFETY CHECK
# ---------------------------------------------------
def is_code_safe(code: str) -> bool:
    """
    AST-based static analysis.
    Blocks:
    - exec, eval, __import__
    - os, sys, subprocess
    - import *
    - writing files
    """

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):

            # Block exec/eval
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval", "__import__"]:
                    return False

            # Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in SAFE_IMPORT_ROOTS:
                        return False

            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    return False

                root = node.module.split(".")[0]
                if root not in SAFE_IMPORT_ROOTS:
                    return False

                if any(a.name == "*" for a in node.names):
                    return False

            # Block file ops (open(), remove(), unlink())
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ["open", "remove", "unlink", "rmdir"]:
                    return False

            # Block os/sys/subprocess attribute usage
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in ["os", "sys", "subprocess"]:
                        return False

    except Exception:
        return False

    return True


# ---------------------------------------------------
# RUNNER STATE
# ---------------------------------------------------
class RunnerState(dict):
    code: Optional[str]
    df_json: Optional[str]
    image_bytes: Optional[bytes]
    error: Optional[str]

import re

def clean_code_block(code_text: str) -> str:
    # Remove ```python ... ``` or ``` ... ```
    code_text = re.sub(r"```(?:python)?", "", code_text)
    code_text = code_text.replace("```", "")
    return code_text.strip()


# ---------------------------------------------------
# NODE: EXECUTE VISUALIZATION CODE
# ---------------------------------------------------
def node_run_code(state: Dict[str, Any]) -> Dict[str, Any]:

    code = clean_code_block(state.get("code"))
    df_json = state.get("df_json")

    if not code:
        return {"error": "No visualization code provided."}

    # # Security validation
    # if not is_code_safe(code):
    #     return {"error": "❌ Unsafe code detected. Execution blocked."}

    try:
        # Convert JSON -> DataFrame properly
        from io import StringIO
        df = pd.read_json(StringIO(df_json))

        # Sandbox execution environment
        exec_globals = {
            "pd": pd,
            "plt": plt,
            "df": df,
        }
        exec_locals = {}

        # Execute visualization code
        exec(code, exec_globals, exec_locals)

        # Save the figure in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        plt.close()               # ← Prevents Tkinter thread errors

        return {
            "image_bytes": buffer.getvalue(),
            "error": None
        }

    except Exception as e:
        return {"error": traceback.format_exc()}


# ---------------------------------------------------
# BUILD LANGGRAPH — FIXED (ADD END NODE)
# ---------------------------------------------------
builder = StateGraph(RunnerState)
builder.add_node("run_code", RunnableLambda(node_run_code))

builder.set_entry_point("run_code")
builder.add_edge("run_code", END)     # ← FIX for "dead-end" error

runner_app = builder.compile()
