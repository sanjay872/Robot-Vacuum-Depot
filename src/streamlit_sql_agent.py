import streamlit as st
import pandas as pd

from agent_sql_generator import sql_agent_app
from agent_code_generation import viz_agent_app
from agent_code_validator import validator_app
from agent_code_runner import runner_app

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.markdown("""
    <style>
        .stChatMessage { max-width: 70%; }
        .st-emotion-cache-4oy321 { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI SQL + Visualization Agent")


# ----------------------------------------------------------
# INTENT CLASSIFIER (chart vs table)
# ----------------------------------------------------------
intent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Classify analytics questions into:
- 'chart' → comparison, distribution, trends, grouped metrics
- 'table' → lookup, ranking, listing

Return ONLY: chart  OR  table.
"""
    ),
    ("human", "{question}")
])

intent_chain = intent_prompt | intent_llm | StrOutputParser()


def classify_intent(question):
    try:
        label = intent_chain.invoke({"question": question}).strip().lower()
        return "chart" if "chart" in label else "table"
    except:
        return "table"


# ----------------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------------
user_query = st.chat_input("Ask your data question…")

if user_query:
    # Show user bubble
    st.chat_message("user").write(user_query)

    # Start assistant bubble
    assistant = st.chat_message("assistant")
    with assistant:

        # Step 1 — SQL Agent
        assistant.write("🔵 **Step 1: Generating SQL & Querying Data…**")
        sql_state = sql_agent_app.invoke({"question": user_query})

        if sql_state.get("error"):
            assistant.error(sql_state["error"])
            st.stop()

        sql_code = sql_state["sql"]
        df = sql_state["df"]

        assistant.code(sql_code, language="sql")

        if df is None or df.empty:
            assistant.warning("Query executed but returned no rows.")
            st.stop()

        assistant.write("### 📄 Query Result")
        assistant.dataframe(df)

        df_json = df.to_json(orient="records")


        # Step 2 — Intent Classification
        assistant.write("🧠 **Step 2: Understanding Your Intent…**")
        intent = classify_intent(user_query)
        assistant.write(f"**Intent:** `{intent}`")

        if intent == "table":
            assistant.success("📄 Table output selected — done!")
            st.stop()


        # Step 3 — Viz Code Generation
        assistant.write("🟣 **Step 3: Generating Visualization Code…**")
        viz_state = viz_agent_app.invoke({
            "df_json": df_json,
            "question": user_query
        })

        if viz_state.get("error"):
            assistant.error(viz_state["error"])
            st.stop()

        viz_code = viz_state["viz_code"]
        assistant.code(viz_code, language="python")


        # Step 4 — Code Validation
        assistant.write("🟠 **Step 4: Validating Code…**")
        validation = validator_app.invoke({
            "code": viz_code,
            "df_json": df_json
        })

        if not validation.get("is_valid"):
            assistant.error("❌ Visualization code is invalid.")
            assistant.write(validation.get("feedback"))
            st.stop()

        assistant.success("✔ Code is valid.")


        # Step 5 — Execute Visualization
        assistant.write("🟢 **Step 5: Executing Code…**")
        result = runner_app.invoke({
            "code": viz_code,
            "df_json": df_json
        })

        if result.get("error"):
            assistant.error(result["error"])
            st.stop()

        img = result.get("image_bytes")
        if img:
            assistant.image(img, caption="📊 Final Visualization", use_column_width=True)
        else:
            assistant.warning("Code ran, but no image produced.")
