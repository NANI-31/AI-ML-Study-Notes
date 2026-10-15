import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="28.1 SQL Databases - LangChain Systems",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for database telemetry
st.markdown("""
<style>
    .db-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-sql {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class SQLDatabaseBridgeSimulator:
    """
    Simulates SQLAlchemy database connection management,
    schema introspection, query generation, and tool binding for LangChain.
    """
    def __init__(self):
        self.mock_schema = {
            "customers": ["id INT PRIMARY KEY", "name VARCHAR(100)", "tier VARCHAR(20)", "balance NUMERIC(10,2)"],
            "orders": ["id INT PRIMARY KEY", "customer_id INT REFERENCES customers(id)", "order_date DATE", "total_usd NUMERIC(10,2)", "status VARCHAR(30)"],
            "order_items": ["id INT PRIMARY KEY", "order_id INT REFERENCES orders(id)", "sku VARCHAR(50)", "quantity INT", "unit_price NUMERIC(10,2)"]
        }
        self.sample_rows = {
            "customers": [{"id": 101, "name": "Acme Corp", "tier": "Enterprise", "balance": 4500.00}],
            "orders": [{"id": 901, "customer_id": 101, "order_date": "2026-10-01", "total_usd": 1250.00, "status": "completed"}],
            "order_items": [{"id": 1, "order_id": 901, "sku": "AI-SEAT-LIC", "quantity": 5, "unit_price": 250.00}]
        }

    def execute_mock_query(self, query: str) -> Dict[str, Any]:
        normalized = query.strip().upper()
        if "CUSTOMERS" in normalized and "ORDERS" in normalized:
            df = pd.DataFrame([
                {"customer_name": "Acme Corp", "order_count": 14, "total_spend": 17500.00},
                {"customer_name": "Global Logistics", "order_count": 8, "total_spend": 9200.00}
            ])
            return {"status": "SUCCESS", "rows": 2, "data": df, "execution_ms": 14.2}
        elif "CUSTOMERS" in normalized:
            df = pd.DataFrame(self.sample_rows["customers"])
            return {"status": "SUCCESS", "rows": len(df), "data": df, "execution_ms": 6.8}
        else:
            df = pd.DataFrame(self.sample_rows["orders"])
            return {"status": "SUCCESS", "rows": len(df), "data": df, "execution_ms": 8.1}


# --- LESSON HEADER ---
st.title("28.1 SQL Databases: Foundations, SQLAlchemy & Tool Integration")
st.markdown("""
Relational databases remain the backbone of enterprise data infrastructure. Connecting LangChain to SQL databases
allows AI models to query structured tables, perform mathematical aggregations, and derive real-time insights
from live production systems. This module covers relational database fundamentals, SQLAlchemy engine integration,
connection pooling, SQL tool toolkits, and agent interaction architectures.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: SQL Database and LangChain Tool Integration", divider="rainbow")
st.markdown("""
Connecting an LLM to a relational database requires a multi-stage translation bridge.
The database schema is inspected using SQLAlchemy reflection, converted into prompt metadata,
and passed to the model. The model emits dialect-specific SQL, which is executed via a sandboxed connection
wrapper that returns tabular result sets formatted as Markdown or structured JSON.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SQL_Databases_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_user_query {
            label = "Natural Language Interface";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            UserPrompt [label="User Natural Query\\n('Show top 5 enterprise accounts by order spend')", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_langchain_sql {
            label = "LangChain SQL Toolkit & Database Abstraction";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            SQLDatabaseWrapper [label="SQLDatabase Utility\\n(Schema reflection, sample row previews, table whitelisting)", fillcolor="#1E293B", color="#60A5FA"];
            SQLToolkit [label="SQL Database Toolkit\\n(list_tables, get_schema, query_sql, query_checker)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_agent_core {
            label = "SQL Agent Orchestration (ReAct / Tools)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ModelCore [label="Chat Model (e.g. GPT-4o)\\n(Schema-aware SQL query synthesis)", fillcolor="#0369A1", color="#38BDF8"];
            QueryChecker [label="Query Validation Tool\\n(Validates syntax, limits rows, blocks write operations)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_engine_db {
            label = "Database Engine & Infrastructure";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            SQLAlchemyEngine [label="SQLAlchemy Engine (Engine / Connection Pool)\\n(Dialect: PostgreSQL / MySQL / Snowflake / SQLite)", fillcolor="#065F46", color="#10B981"];
            RelationalDB [label="Production Relational DB\\n(Read-only credentials + schema catalogs)", fillcolor="#065F46", color="#10B981"];
        }

        UserPrompt -> ModelCore [label="1. Ingest prompt"];
        SQLDatabaseWrapper -> ModelCore [label="2. Inject table schemas"];
        ModelCore -> SQLToolkit [label="3. Request schema / run query"];
        SQLToolkit -> QueryChecker [label="4. Synthesize dialect query"];
        QueryChecker -> SQLAlchemyEngine [label="5. Execute validated SQL"];
        SQLAlchemyEngine -> RelationalDB [label="6. Query tables"];
        RelationalDB -> SQLAlchemyEngine [label="7. Return tabular rows"];
        SQLAlchemyEngine -> ModelCore [label="8. Formatted result text"];
        ModelCore -> UserPrompt [label="9. Natural language answer"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: SQL DATABASE BASICS ---
st.header("2. SQL Database Basics: Relational Data Models in Generative AI", divider="rainbow")
st.markdown("""
Relational databases enforce fixed schemas, foreign key relationships, ACID transactions, and deterministic SQL querying.
Unlike vector databases that return semantic approximations, SQL databases return exact rows, numerical summations,
and precise aggregations. When integrating with LLMs, the model must understand table structures, primary/foreign keys,
and column constraints to avoid generating invalid joins or column references.
""")

col_db1, col_db2 = st.columns(2)
with col_db1:
    st.markdown("### Core Relational Concepts for AI")
    st.markdown("""
    - **Entity-Relationship Models**: Tables represent distinct business entities (customers, invoices, products) connected via keys.
    - **Exact Aggregations**: Grouping, counting, summing, and filtering are performed deterministically by the database engine, avoiding LLM arithmetic errors.
    - **Schema Constraints**: Nullability, unique constraints, and data types (VARCHAR, INT, NUMERIC, TIMESTAMP) guide model SQL synthesis.
    """)
with col_db2:
    st.markdown("### Common Enterprise SQL Dialects")
    st.markdown("""
    - **PostgreSQL**: Advanced indexing, JSONB columns, window functions.
    - **Snowflake / BigQuery / Redshift**: Cloud data warehouses optimized for large-scale analytical aggregations.
    - **MySQL / MariaDB**: Widespread transaction processing systems.
    - **SQLite**: Lightweight serverless database, ideal for testing and edge applications.
    """)

# --- SECTION 3: SQLALCHEMY INTEGRATION ---
st.header("3. SQLAlchemy Integration: Dialect Agnostic Database Connections", divider="rainbow")
st.markdown("""
LangChain interfaces with relational databases using **SQLAlchemy**, Python's premier database toolkit and Object-Relational Mapper (ORM).
SQLAlchemy abstracts database-specific dialect differences, connection pooling, and credential handling,
enabling LangChain's `SQLDatabase` wrapper to communicate seamlessly with any supported database engine.
""")

st.code("""
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

# 1. Create SQLAlchemy Engine with Connection Pool Configuration
engine = create_engine(
    "postgresql://readonly_agent:secure_token@db.prod.internal:5432/analytics",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# 2. Instantiate LangChain SQLDatabase Abstraction
db = SQLDatabase(
    engine=engine,
    include_tables=["customers", "orders", "order_items"],
    sample_rows_in_table_info=2
)

# Inspect detected dialect and tables
print(f"Dialect: {db.dialect}")
print(f"Managed Tables: {db.get_usable_table_names()}")
""", language="python")

# --- SECTION 4: DATABASE CONNECTIONS ---
st.header("4. Database Connections: Whitelisting, Security & Connection Pooling", divider="rainbow")
st.markdown("""
Exposing an entire enterprise database to an LLM without filters causes prompt context exhaustion and severe security risks.
Production deployments must enforce **table whitelisting**, **sample row limits**, and **dedicated read-only connection strings**.
""")

col_sec1, col_sec2 = st.columns(2)
with col_sec1:
    st.markdown("### Security Configuration Best Practices")
    st.markdown("""
    - **Strict Table Whitelisting**: Use `include_tables=[...]` to expose only tables necessary for the agent's task, hiding sensitive tables (users, passwords, tokens).
    - **Sample Row Injection**: Set `sample_rows_in_table_info=2` to show the LLM real example values for categorical columns without leaking database contents.
    - **Read-Only Credentials**: The database user in the connection URI must strictly lack `INSERT`, `UPDATE`, `DELETE`, and `DROP` privileges.
    """)
with col_sec2:
    st.code("""
# Safe Database Initializer Pattern
from langchain_community.utilities import SQLDatabase

def get_sandboxed_sql_db() -> SQLDatabase:
    # Dedicated read-only connection URI
    uri = "postgresql://ro_agent:pass@db.prod:5432/warehouse"
    
    return SQLDatabase.from_uri(
        database_uri=uri,
        include_tables=["sales_q3", "product_catalog"],
        sample_rows_in_table_info=2,
        max_string_length=300  # Truncate large text fields
    )
    """, language="python")

# --- SECTION 5: SQL TOOLS ---
st.header("5. SQL Tools: The SQLDatabaseToolkit and Tool Primitives", divider="rainbow")
st.markdown("""
LangChain provides `SQLDatabaseToolkit`, a standardized suite of tools that gives agents the capability to inspect
schemas, execute queries, and validate syntax:
1. `sql_db_list_tables`: Returns comma-separated list of usable table names.
2. `sql_db_schema`: Retrieves DDL schemas and sample rows for designated tables.
3. `sql_db_query`: Executes a SQL query and returns results as text.
4. `sql_db_query_checker`: Uses an LLM to double-check SQL syntax for common errors before execution.
""")

st.code("""
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Retrieve toolkit tools
tools = toolkit.get_tools()

for t in tools:
    print(f"Tool: {t.name} -> {t.description[:80]}...")
""", language="python")

# --- SECTION 6: SQL AGENTS ---
st.header("6. SQL Agents: ReAct Decision Loops and Natural Language Database Querying", divider="rainbow")
st.markdown("""
A SQL agent combines a chat model, the `SQLDatabaseToolkit`, and a reasoning loop (ReAct).
Instead of guessing table names, the agent inspects available tables, retrieves relevant schemas,
synthesizes a dialect-correct query, executes it, and formats the output into a clear natural language answer.
""")

st.code("""
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools",
    top_k=10,  # Enforce row limit on all generated queries
    max_iterations=8
)

# Execute inquiry
result = agent_executor.invoke({
    "input": "Which customer spent the most on Enterprise tier orders in Q3?"
})
print(result["output"])
""", language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Hardened Read-Only SQL Database Engine", divider="rainbow")
st.markdown("""
Below is an end-to-end production implementation featuring connection pool management,
query timeout enforcement, and schema reflection with LangChain.
""")

st.code("""
# sql_database_integration.py
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

class EnterpriseDatabaseBridge:
    def __init__(self, db_uri: str, allowed_tables: list[str]):
        # Production connection engine with strict pool timeouts
        self.engine = create_engine(
            db_uri,
            pool_size=10,
            max_overflow=5,
            pool_timeout=15,
            pool_recycle=3600
        )
        self.sql_db = SQLDatabase(
            engine=self.engine,
            include_tables=allowed_tables,
            sample_rows_in_table_info=2
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=self.sql_db, llm=self.llm)
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=False,
            agent_type="openai-tools",
            top_k=25
        )

    def query(self, natural_language_query: str) -> str:
        response = self.agent.invoke({"input": natural_language_query})
        return response["output"]
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: SQL Database Bridge & Schema Inspector Simulator", divider="rainbow")
st.markdown("""
Inspect relational table schemas, sample database rows, and test simulated SQL query executions.
""")

bridge = SQLDatabaseBridgeSimulator()

col_sc, col_res = st.columns([1, 1])

with col_sc:
    st.subheader("📋 Relational Schema Inspector")
    selected_tbl = st.selectbox("Inspect Database Table:", list(bridge.mock_schema.keys()))
    with st.container(border=True):
        st.markdown(f"**Table:** `public.{selected_tbl}`")
        for col_def in bridge.mock_schema[selected_tbl]:
            st.markdown(f"- <code>{col_def}</code>", unsafe_allow_html=True)
            
    query_input = st.text_area(
        "Simulate SQL Query Execution:",
        value=f"SELECT * FROM {selected_tbl} LIMIT 5;"
    )
    exec_btn = st.button("🚀 Execute Query via SQLAlchemy Bridge", type="primary")

with col_res:
    st.subheader("📊 Query Results & Telemetry")
    if exec_btn:
        res = bridge.execute_mock_query(query_input)
        st.markdown(
            f"<div class='db-card'><span class='badge-sql'>QUERY STATUS: {res['status']}</span>"
            f"<br><br><b>Rows Returned:</b> {res['rows']} | <b>Execution Latency:</b> {res['execution_ms']} ms</div>",
            unsafe_allow_html=True
        )
        st.dataframe(res["data"], use_container_width=True)
    else:
        st.info("Select a table, modify the SQL statement, and execute to view tabular result sets.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why should SQLAlchemy be used rather than raw DB-API drivers in LangChain?"):
    st.markdown("""
    **Answer**: SQLAlchemy provides dialect-agnostic database abstraction, automatic schema reflection, connection pooling,
    and parameter binding. By using SQLAlchemy, LangChain's `SQLDatabase` works seamlessly across PostgreSQL, MySQL, SQLite,
    Oracle, and Snowflake without rewriting low-level connection code or SQL introspection queries.
    """)

with st.expander("Q2: What is the purpose of `sample_rows_in_table_info` in `SQLDatabase`?"):
    st.markdown("""
    **Answer**: Setting `sample_rows_in_table_info=2` injects 2 real sample data rows into the schema prompt provided to the LLM.
    This informs the model about expected column formats (e.g. date format strings `'YYYY-MM-DD'`, categorical values `'Enterprise'` vs `'retail'`),
    drastically reducing SQL errors caused by guessing column value conventions.
    """)

with st.expander("Q3: How do you prevent SQL injection when agents query relational databases?"):
    st.markdown("""
    **Answer**: Use dedicated database roles granted strictly read-only permissions (`SELECT`), enforce query AST parsing (rejecting `DROP`, `UPDATE`,
    `DELETE`), and use parameter binding for user-supplied filtering values. Never run SQL agents under administrative database accounts.
    """)

with st.expander("Q4: What tools are included in `SQLDatabaseToolkit`?"):
    st.markdown("""
    **Answer**: The toolkit contains four primary tools: `sql_db_list_tables` (lists accessible tables), `sql_db_schema` (retrieves schema and sample rows),
    `sql_db_query` (executes validated SQL and returns results), and `sql_db_query_checker` (uses an LLM to check query syntax for common errors before execution).
    """)

with st.expander("Q5: How does `create_sql_agent` prevent unbounded result sets from crashing server memory?"):
    st.markdown("""
    **Answer**: `create_sql_agent` injects a default `top_k` parameter (e.g. 10 or 25). The system prompt instructs the agent to append `LIMIT top_k`
    to every generated query, preventing queries like `SELECT * FROM orders` from pulling millions of rows into application memory.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Connecting Under Read-Write User)**: Providing the agent credentials with write or DDL privileges. A hallucinated or injected query can drop production tables.")
    st.warning("**Trap 2 (Exposing Unbounded Schemas)**: Passing 100 enterprise tables into the prompt without `include_tables`. This exhausts token limits and confuses the model.")
with col_trap2:
    st.error("**Trap 3 (Missing Connection Pool Limits)**: Creating new SQLAlchemy engines per user request without connection pooling, exhausting database connection slots under traffic.")
    st.warning("**Trap 4 (Ignoring Dialect Quirks)**: Assuming generic SQL works everywhere. Date functions, string concatenations, and limit clauses differ significantly across PostgreSQL, Oracle, and SQL Server.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What Python library powers LangChain's relational database connections?")
        st.caption("Flip for answer:")
        st.info("**Back**: SQLAlchemy (provides dialect abstraction, schema reflection, and connection pooling).")
    with st.container(border=True):
        st.markdown("**Front**: What parameter in `SQLDatabase` limits which tables the agent sees?")
        st.caption("Flip for answer:")
        st.info("**Back**: `include_tables=['table1', 'table2']`.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What tool in `SQLDatabaseToolkit` runs the final SQL statement?")
        st.caption("Flip for answer:")
        st.info("**Back**: `sql_db_query`.")
    with st.container(border=True):
        st.markdown("**Front**: Why should sample rows be injected into the schema prompt?")
        st.caption("Flip for answer:")
        st.info("**Back**: To demonstrate real column values, casing, and format conventions to the model.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural differences between relational SQL and vector stores.", value=True)
c2 = st.checkbox("I can connect LangChain to SQL databases using SQLAlchemy engines and SQLDatabase.", value=True)
c3 = st.checkbox("I know how to restrict agent access using include_tables and read-only connection credentials.", value=True)
c4 = st.checkbox("I understand the four tools provided by SQLDatabaseToolkit.", value=True)
c5 = st.checkbox("I can build an autonomous database query agent using create_sql_agent.", value=True)
c6 = st.checkbox("I know how top_k limits prevent unbounded memory consumption during SQL execution.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 28.2: SQL Agents.")
