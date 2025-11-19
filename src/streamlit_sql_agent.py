import streamlit as st
import pandas as pd

from agent_sql_generator import sql_agent_app
from agent_code_generation import viz_agent_app
from agent_code_validator import validator_app
from agent_code_runner import runner_app

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="AI SQL → Viz Pipeline (Auto)", layout="wide")
st.title("🤖 AI SQL → Table / Chart (Fully Automated Pipeline)")


# =====================================================
# 1. LLM-BASED INTENT CLASSIFIER (TABLE vs CHART)
# =====================================================

intent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
"You are an intent classifier for analytics questions.

Your job:
- Decide if the user wants a CHART visualization or a TABLE/TEXT answer.

CHART indicators (very important):
- Any request involving comparison across categories
- Words like: plot, chart, graph, visualize, draw, show trends
- Words like: compare, comparison, vs, versus
- Phrases like: average by X, cost efficiency, grouped by, distribution
- Any question asking to evaluate or compare metrics across groups
- Any question involving categories + numeric values
- Any question where a bar chart or line chart would naturally be used

TABLE indicators:
- Listing entities
- Ranking entities
- “Which”, “who”, “what” without comparison intent
- Pure textual lookup or selection

Output:
Return EXACTLY one word:
    chart
or
    table
No punctuation. No explanation."
"""
    ),
    (
        "human",
        "User question: {question}\n\nYour answer (chart/table only):"
    )
])

intent_chain = intent_prompt | intent_llm | StrOutputParser()


def classify_intent(question: str) -> str:
    """Returns 'chart' or 'table' based on LLM classifier."""
    try:
        raw = intent_chain.invoke({"question": question})
        intent = raw.strip().lower()
        if "chart" in intent:
            return "chart"
        return "table"
    except Exception:
        # Failsafe: default to table
        return "table"


# =====================================================
# 2. SESSION STATE HELPER
# =====================================================

def ensure_session_keys():
    defaults = {
        "df_json": None,
        "sql_code": None,
        "viz_code": None,
        "validation": None,
        "image_bytes": None,
        "last_intent": None,
        "last_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_session_keys()


# =====================================================
# 3. USER INPUT
# =====================================================

question = st.text_input("Ask a question about the robot vacuum database:")

with st.expander("📦 Session State Snapshot (debug)"):
    st.json({k: (str(v)[:200] + ("..." if len(str(v)) > 200 else "")) for k, v in st.session_state.items()})


# =====================================================
# 4. MAIN BUTTON — FULL PIPELINE
# =====================================================

if st.button("Run AI Query"):

    st.session_state.last_error = None
    st.session_state.image_bytes = None

    if not question.strip():
        st.warning("Please enter a natural language question first.")
    else:
        # ------------------------------------------
        # STEP 1: Agent 1 — SQL + DataFrame
        # ------------------------------------------
        st.subheader("🔵 Step 1 — Generating SQL and DataFrame")

        sql_state = sql_agent_app.invoke({"question": question})

        sql_code = sql_state.get("sql")
        df = sql_state.get("df")
        error = sql_state.get("error")

        with st.expander("🔍 Generated SQL (Agent 1)"):
            st.code(sql_code or "No SQL generated.", language="sql")

        if error:
            st.error(f"Agent 1 error: {error}")
            st.session_state.last_error = error
            st.stop()

        if not isinstance(df, pd.DataFrame):
            st.error("Agent 1 did not return a valid DataFrame.")
            st.session_state.last_error = "No DataFrame from Agent 1."
            st.stop()

        if df.empty:
            st.warning("Query executed but returned no rows.")
        st.dataframe(df)

        st.session_state.df_json = df.to_json(orient="records")
        st.session_state.sql_code = sql_code

        # ------------------------------------------
        # STEP 2: Classify intent (TABLE vs CHART)
        # ------------------------------------------
        st.subheader("🧠 Step 2 — Classifying Intent (Table vs Chart)")
        intent = classify_intent(question)
        st.session_state.last_intent = intent

        st.write(f"**Classifier decision:** `{intent}`")

        if intent == "table":
            st.success("📄 Using TABLE/TEXT output — showing DataFrame above.")
            st.stop()  # No need to run viz pipeline

        # ------------------------------------------
        # STEP 3: Agent 2 — Generate Visualization Code
        # ------------------------------------------
        st.subheader("🟣 Step 3 — Generating Visualization Code (Agent 2)")

        viz_state = viz_agent_app.invoke({
            "df_json": st.session_state.df_json,
            "question": question,
        })

        viz_code = viz_state.get("viz_code")
        viz_error = viz_state.get("error")

        with st.expander("🎨 Visualization Code (Agent 2)"):
            st.code(viz_code or "", language="python")

        if viz_error:
            st.error(f"Agent 2 error: {viz_error}")
            st.session_state.last_error = viz_error
            st.stop()

        if not viz_code:
            st.error("Agent 2 returned empty visualization code.")
            st.session_state.last_error = "Empty viz code from Agent 2."
            st.stop()

        st.session_state.viz_code = viz_code

        # ------------------------------------------
        # STEP 4: Agent 3 — Validate Code
        # ------------------------------------------
        st.subheader("🟠 Step 4 — Validating Visualization Code (Agent 3)")

        validation = validator_app.invoke({
            "code": st.session_state.viz_code,
            "df_json": st.session_state.df_json,
        })

        st.session_state.validation = validation

        with st.expander("🧪 Validator Output (Agent 3)"):
            st.json(validation)

        if validation.get("is_valid"):
            st.success("✅ Visualization code is VALID.")
        else:
            st.error("❌ Visualization code is NOT valid.")
            st.write(validation.get("feedback", "No feedback provided."))
            st.session_state.last_error = "Code marked invalid by Agent 3."
            # Fallback: keep table only
            st.info("Showing table above as fallback.")
            st.stop()

        # ------------------------------------------
        # STEP 5: Agent 4 — Execute Code
        # ------------------------------------------
        st.subheader("🟢 Step 5 — Executing Visualization Code (Agent 4)")

        run_result = runner_app.invoke({
            "code": st.session_state.viz_code,
            "df_json": st.session_state.df_json,
        })

        if run_result.get("error"):
            st.error("Execution error from Agent 4:")
            st.text(run_result["error"])
            st.session_state.image_bytes = None
            st.session_state.last_error = run_result["error"]
            st.info("Showing table above as fallback.")
        else:
            image_bytes = run_result.get("image_bytes")
            st.session_state.image_bytes = image_bytes

            if image_bytes:
                st.success("✅ Visualization executed successfully.")
                st.subheader("📊 Final Visualization Output")
                st.image(image_bytes, caption="Rendered by multi-agent pipeline", use_column_width=True)
            else:
                st.warning("Agent 4 did not return an image. Showing table only as fallback.")
                st.session_state.last_error = "No image_bytes from Agent 4."


# =====================================================
# 5. OPTIONAL: Show last successful visualization
# =====================================================
if st.session_state.get("image_bytes"):
    with st.expander("🖼 Last Rendered Visualization"):
        st.image(st.session_state.image_bytes, caption="Last successful output", use_column_width=True)
