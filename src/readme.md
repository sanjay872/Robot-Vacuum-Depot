# 🤖 AI-Powered SQL Query & Visualization System

Multi-Agent LangGraph + Streamlit Application
PostgreSQL • Polars • LangChain • LangGraph • Streamlit

# 📌 Overview

## This project implements a fully automated AI analytics pipeline using:

- PostgreSQL for relational database storage

- Polars for fast CSV ingestion

- LangChain + LangGraph for orchestrating multiple AI agents

- Five-agent architecture for query → SQL → visualization → validation → execution

- Streamlit for final UI output

    The system takes any natural-language question, converts it into a valid SQL query, retrieves data from PostgreSQL, generates visualization code, validates and executes that code, and displays the final result (chart/table) inside a Streamlit application.

- This project satisfies all requirements of the assignment:

    ✔ Create PostgreSQL tables in 3NF

    ✔ Load CSV into PostgreSQL using Polars

    ✔ Build 5 AI agents

    ✔ Use LangGraph for agent orchestration

    ✔ Display final output using Streamlit UI

    ✔ Ensure modular code across multiple files

# 🧠 Multi-Agent Architecture

Your system uses a 5-agent LangGraph pipeline:

## Agent 1 — SQL Generator

Input: Natural-language question
Output: SQL + pandas DataFrame

Reads DB schema dynamically

Converts natural-language question → SQL

Ensures syntactic correctness

Executes SQL using Polars + PostgreSQL

Returns DataFrame

📄 File: src/agent_sql_generator.py

## Agent 2 — Visualization Code Generator

Input: DataFrame + user question
Output: Safe, executable Matplotlib code

Features:

Row-count–aware (1-point, 2-point, 3+ handling)

Auto-detects datetimes

Only uses real column names

Chooses correct chart type (line, bar, pie, scatter, hist)

No placeholders, no invalid code

📄 File: src/agent_code_generation.py

## Agent 3 — Code Validator

Input: Visualization code + DataFrame JSON
Output: JSON verdict + feedback

Ensures:

Only real DataFrame columns used

Code is syntactically valid

No unsafe operations

Output strictly JSON (no markdown)

📄 File: src/agent_code_validator.py

## Agent 4 — Secure Code Runner

Input: Validated visualization code
Output: PNG image bytes

Features:

Blocks unsafe imports (os, subprocess, exec, eval, etc.)

Executes code in sandboxed namespace

Captures Matplotlib output as PNG

Prevents filesystem and system access

📄 File: src/agent_code_runner.py

## Agent 5 — Streamlit App (UI Layer)

Input: User question
Output: Full pipeline execution with UI

UI Displays:

Generated SQL

DataFrame preview

Visualization code

Validation results

Final chart/table

📄 File: src/streamlit_sql_agent.py


# 🔐 Environment Variables (.env Setup)

The system loads environment variables using python-dotenv.

Create .env in src/:

```
PG_HOST=localhost
PG_PORT=5433
PG_USER=postgres
PG_PASSWORD=admin
PG_DB=robot_vacuum

OPENAI_API_KEY=sk-xxxx
```

Refer .env.example for variable name:

```
PG_HOST=localhost
PG_PORT=5433
PG_USER=postgres
PG_PASSWORD=admin
PG_DB=robot_vacuum
OPENAI_API_KEY=your_api_key_here
```

# 🛠 Installation & Setup
1️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

2️⃣ Install dependencies

```bash
pip install -r src/requirements.txt
```

3️⃣ Set up PostgreSQL

Create database:

```sql
CREATE DATABASE robot_vacuum;
CREATE SCHEMA robot_vacuum;
```

4️⃣ Load CSV (Polars + SQL)

Open the notebook and run it - src/create_table_and_load_CSV.ipynb

This will:

Create all tables in 3NF

Load CSV using Polars

Insert into PostgreSQL

## 🚀 Running the Full Application

From project root:

```bash
streamlit run src/streamlit_sql_agent.py
```

The UI opens in the browser.

- Pipeline steps: 
    - Enter natural-language question
    - Generate SQL
    - Preview DataFrame
    - Generate visualization code
    - Validate
    - Execute safely
    - View final chart/table

# 💡 Example Questions

Text/Table Output Examples:

```
Which warehouses are below restock threshold?

Which manufacturers have highest average review rating?

Which ZIP code has the most delayed deliveries?

Chart Output Examples:

Plot monthly revenue trends over time

What is the percentage distribution of delivery statuses?

Compare average shipping cost by carrier

Plot average review rating by manufacturer
```

# 🔒 Security Measures

- No exec/eval in the system

- No uncontrolled imports

- Code runner is sandboxed

- No filesystem writes

- No shell commands

# 🧾 Submission Requirements — All Satisfied

✔ src/ directory included

✔ doc/ directory with README

✔ requirements.txt included

✔ .env.example included

✔ Jupyter notebook included

✔ Agents separated into multiple files

✔ Streamlit UI included

✔ Fully functional multi-agent architecture