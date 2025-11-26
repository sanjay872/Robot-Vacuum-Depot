# 🤖 AI-Powered SQL Query & Visualization System
Fully Automated Multi-Agent Analytics Pipeline

PostgreSQL • Polars • LangChain • LangGraph • Streamlit • PythonREPLTool

# 📌 Overview

This project implements an end-to-end AI analytics assistant capable of:

- Understanding natural-language questions

- Generating correct SQL queries

- Executing SQL against a PostgreSQL database

- Creating visualization code (Matplotlib/Plotly)

- Validating the generated code

- Executing the code safely in a sandbox

- Displaying tables/charts inside Streamlit

- The system uses a five-agent LangGraph architecture with strict validation, safety guards, and fully automated reasoning.

This implementation meets all assignment requirements:

✔ PostgreSQL schema designed in 3NF

✔ CSV loading with Polars

✔ AI agents for SQL, visualization, validation, execution

✔ LangGraph used for agent orchestration

✔ Streamlit UI for final output

✔ Modular, maintainable code across multiple files

# 🧠 Multi-Agent Architecture

Your system is composed of five specialized agents running in a LangGraph workflow:

## Agent 1 - SQL Generator

Input: Natural-language question
Output: SQL query + DataFrame

Capabilities:

Dynamically reads the live DB schema

Converts NL → valid PostgreSQL SQL

Ensures syntactic + semantic correctness

Uses Polars + PostgreSQL for fast execution

📄 File: src/agent_sql_generator.py

## Agent 2 - Visualization Code Generator

Input: DataFrame + user question
Output: Pure, safe Matplotlib code

Capabilities:

Detects numeric vs category columns

Detects timestamps and converts automatically

Avoids placeholders and invalid references

Selects correct chart type:

Line, bar, pie, scatter, histogram

Generates PythonREPL-ready code

📄 File: src/agent_code_generation.py

## Agent 3 - Visualization Code Validator

Input: Visualization code + DataFrame JSON
Output: JSON verdict {is_valid, feedback}

Capabilities:

Ensures no missing columns

Ensures executable Python syntax

Ensures safe visualization-only operations

JSON-only output (strict schema)

📄 File: src/agent_code_validator.py

## Agent 4 - Secure Code Runner (Python REPL)

Input: Validated Python code
Output: PNG image bytes

Capabilities:

Sandboxed execution via PythonREPLTool

No filesystem or OS access

Extracts Matplotlib/Plotly figures

Returns chart as PNG for Streamlit

📄 File: src/agent_code_runner.py

## Agent 5 - Streamlit Application (UI Layer)

Input: User natural-language question
Output: Complete analytics response

UI Features:

ChatGPT-style interface

Shows:

Generated SQL

DataFrame

Visualization code

Validator result

Final chart/table

Fully automated pipeline

📄 File: src/streamlit_sql_agent.py

# 🔐 Environment Variables (.env Setup)

The system uses python-dotenv.

Create a .env file inside src/.

A template is provided:

📄 .env.example

```bash
PG_HOST=YOUR_HOST
PG_PORT=YOUR_PORT
PG_USER=YOUR_USER_NAME
PG_PASSWORD=YOUR_PASSWORD
PG_DB=YOUR_DB
OPENAI_API_KEY=your_api_key_here
```

# 🛠 Installation & Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux

2️⃣ Install Dependencies
pip install -r src/requirements.txt

## Database Setup
3️⃣ Create PostgreSQL database + schema
CREATE DATABASE robot_vacuum;
CREATE SCHEMA robot_vacuum;

4️⃣ Load CSV Using Polars

Run:

📄 src/create_table_and_load_CSV.ipynb

This will:

Create all tables (3NF)

Load CSV using Polars

Insert into PostgreSQL

🚀 Running the Full Application

From project root:
```bash
streamlit run src/streamlit_sql_agent.py
```

Streamlit automatically launches in the browser.

💬 Example Questions to Try

Text/Table Output
```
Which warehouses are below restock threshold?
Which manufacturers have the highest average review rating?
Which ZIP code has the most delayed deliveries?
List customers who placed more than 5 orders.
```


Chart Output
```
Plot monthly revenue trends over time.
Show the distribution of delivery statuses as a pie chart.
Compare average shipping cost by carrier.
Plot average review rating by manufacturer.
```

# 🔒 Security Measures

This system includes robust safeguards:

❌ No exec / eval

❌ No filesystem access

❌ No subprocess commands

✔ Python REPL sandbox

✔ Whitelisted imports

✔ Strict code validator

✔ Safe Matplotlib extraction
