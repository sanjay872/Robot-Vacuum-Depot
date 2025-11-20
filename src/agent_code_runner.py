# ---------------------------------------------------------
# Agent 4 — FINAL ROBUST VERSION (Matplotlib + Plotly Safe Execution)
# ---------------------------------------------------------

import io
import ast
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import json
from io import StringIO
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()  # Automatically loads .env from current directory


# ---------------------------------------------------------
# SAFE IMPORT ROOTS — Everything else is BLOCKED
# ---------------------------------------------------------
SAFE_IMPORT_ROOTS = {
    "matplotlib",
    "plotly",
    "seaborn",
    "pandas",
    "numpy"
}



# ---------------------------------------------------------
# 1. CLEAN CODE BLOCKS / REMOVE FENCES
# ---------------------------------------------------------
def clean_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
    code = code.replace("```python", "").replace("```", "")
    return code.strip()


# ---------------------------------------------------------
# 2. SAFETY CHECKER — BLOCKS DANGEROUS CODE
# ---------------------------------------------------------
DANGEROUS_KEYWORDS = [
    "open(", "os.", "subprocess", "shutil", "sys.",
    "eval(", "exec(", "__import__", "input("
]

def is_code_safe(code: str) -> bool:
    # Keyword scan
    for kw in DANGEROUS_KEYWORDS:
        if kw in code:
            return False

    # AST inspection
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # block exec/eval
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval", "open"]:
                    return False

            # block imports outside safe list
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in SAFE_IMPORT_ROOTS:
                        return False

            if isinstance(node, ast.ImportFrom):
                if not node.module:
                    return False
                if node.module.split(".")[0] not in SAFE_IMPORT_ROOTS:
                    return False

            # block attribute access to os, sys, etc.
            if isinstance(node, ast.Attribute):
                root = node.attr.lower()
                if root in ["system", "popen", "remove", "unlink"]:
                    return False

        return True

    except Exception:
        return False


# ---------------------------------------------------------
# Runner State
# ---------------------------------------------------------
class RunnerState(TypedDict, total=False):
    code: Optional[str]
    df_json: Optional[str]
    image_bytes: Optional[bytes]
    error: Optional[str]


# ---------------------------------------------------------
# 3. SAFE DataFrame loader
# ---------------------------------------------------------
def safe_load_df(df_json: str):
    try:
        return pd.read_json(StringIO(df_json))
    except Exception:
        # fallback for record-style json
        try:
            return pd.DataFrame(json.loads(df_json))
        except Exception:
            return pd.DataFrame()


# ---------------------------------------------------------
# 4. EXECUTION NODE
# ---------------------------------------------------------
def node_run_code(state: RunnerState):

    raw_code = state.get("code", "")
    df_json = state.get("df_json", "")

    if not raw_code:
        return {"error": "No visualization code provided."}

    # Clean fences (Agent 2 sometimes returns accidental formatting)
    code = clean_code_fences(raw_code)

    # SAFETY FIRST
    if not is_code_safe(code):
        return {"error": "❌ Unsafe or disallowed code detected. Execution blocked."}

    # Load DataFrame
    df = safe_load_df(df_json)

    # 👇 ADD THIS
    print("==== DataFrame Preview ====")
    print(df.head())
    print("==== DataFrame dtypes ====")
    print(df.dtypes)
    
    # ------------------------------------------------------------------
    # Prepare isolated REPL execution environments
    # ------------------------------------------------------------------
    exec_globals = {
        "pd": pd,
        "plt": plt,
        "px": px,
        "df": df,
    }
    exec_locals = {}

    # Reset any previous figure
    plt.close("all")

    # ------------------------------------------------------------------
    # Execute the generated code SAFELY
    # ------------------------------------------------------------------
    try:
        exec(code, exec_globals, exec_locals)

        # ------------------------------------------------------------------
        # Figure Extraction Logic (Matplotlib OR Plotly)
        # ------------------------------------------------------------------

        # 1️⃣ PLOTLY FIGURE?
        if "fig" in exec_globals and hasattr(exec_globals["fig"], "to_image"):
            try:
                img_bytes = exec_globals["fig"].to_image(format="png")
                return {"image_bytes": img_bytes, "error": None}
            except Exception:
                pass

        if "fig" in exec_locals and hasattr(exec_locals["fig"], "to_image"):
            try:
                img_bytes = exec_locals["fig"].to_image(format="png")
                return {"image_bytes": img_bytes, "error": None}
            except Exception:
                pass

        # 2️⃣ MATPLOTLIB FIGURE?
        fig = plt.gcf()
        if fig and fig.get_axes():
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plt.close(fig)
            return {"image_bytes": buffer.getvalue(), "error": None}

        # 3️⃣ NO FIGURE PRODUCED
        return {"error": "Code executed successfully but produced no figure."}

    except Exception as e:
        return {"error": traceback.format_exc()}


# ---------------------------------------------------------
# Build langgraph app
# ---------------------------------------------------------
builder = StateGraph(RunnerState)
builder.add_node("run_code", RunnableLambda(node_run_code))
builder.set_entry_point("run_code")
builder.add_edge("run_code", END)

runner_app = builder.compile()
