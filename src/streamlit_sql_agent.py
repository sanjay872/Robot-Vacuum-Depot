import streamlit as st
import pandas as pd
from agent_sql import agent_app    # import your LangGraph agent
from agent_sql import AgentState   # your TypedDict

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Robot Vacuum NL → SQL Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Robot Vacuum NL → SQL AI Agent")
st.write("Ask a natural language question about your robot vacuum dataset.")


# -----------------------------------------
# INPUT BOX
# -----------------------------------------
question = st.text_input("Enter your question:", placeholder="e.g., Which products have the most delayed deliveries in Chicago?")

run_button = st.button("Run Query")


# -----------------------------------------
# PROCESSING
# -----------------------------------------
if run_button:

    if not question.strip():
        st.error("Please enter a question first.")
        st.stop()

    with st.spinner("Running NL → SQL agent..."):

        initial_state: AgentState = {
            "question": question,
            "sql": None,
            "df": None,
            "error": None,
        }

        result = agent_app.invoke(initial_state)

        # Extract
        sql = result.get("sql")
        df = result.get("df")
        error = result.get("error")


    # -----------------------------------------
    # DISPLAY SQL
    # -----------------------------------------
    st.subheader("📝 Generated SQL")
    if sql:
        st.code(sql, language="sql")
    else:
        st.warning("No SQL generated.")


    # -----------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------
    if error:
        st.subheader("⚠️ Error")
        st.error(error)
    else:
        st.subheader("📊 Query Results")

        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df)
        elif isinstance(df, pd.DataFrame) and df.empty:
            st.info("Query executed successfully but returned 0 rows.")
        else:
            st.warning("No DataFrame returned.")


# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This interface uses:
    - LangGraph for NL → SQL reasoning  
    - GPT-4o-mini for SQL generation  
    - SQLAlchemy for DB execution  
    - Streamlit for the UI  
    """)

    st.write("Built for **Robot Vacuum Analytics**.")