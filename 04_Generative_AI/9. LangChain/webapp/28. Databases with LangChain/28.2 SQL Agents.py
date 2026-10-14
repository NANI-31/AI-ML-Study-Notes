import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="28.2 SQL Agents - Natural Language to SQL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SQL agent telemetry
st.markdown("""
<style>
    .agent-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-correct {
        background-color: #854D0E;
        color: #FEF08A;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class SQLAgentReasoningSimulator:
    """
    Simulates autonomous text-to-SQL translation, syntax validation,
    query execution against relational schemas, and automated error self-correction.
    """
    def __init__(self):
        self.preset_queries = {
            "Top Spending Customers": {
                "natural_query": "Find the top 3 customers by total completed order value in 2026.",
                "initial_sql": "SELECT c.name, SUM(o.total_usd) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.status = 'completed' AND o.order_date >= '2026-01-01' GROUP BY c.name ORDER BY total_spent DESC LIMIT 3;",
                "validation_status": "VALID",
                "corrected": False,
                "data": pd.DataFrame([
                    {"name": "Acme Global", "total_spent": 142500.00},
                    {"name": "Nova Tech", "total_spent": 98200.00},
                    {"name": "Apex Logistics", "total_spent": 76400.00}
                ])
            },
            "Query with Broken Column Name (Self-Correction)": {
                "natural_query": "List all active client email addresses and balances.",
                "initial_sql": "SELECT client_name, email_address, account_balance FROM client_records WHERE is_active = true;",
                "error_returned": "psycopg2.errors.UndefinedTable: relation 'client_records' does not exist. Available tables: customers, orders.",
                "corrected_sql": "SELECT name, balance FROM customers WHERE balance > 0 LIMIT 10;",
                "validation_status": "SELF_CORRECTED",
                "corrected": True,
                "data": pd.DataFrame([
                    {"name": "Acme Global", "balance": 4500.00},
                    {"name": "Nova Tech", "balance": 1200.00}
                ])
            }
        }

    def simulate_agent_turn(self, query_key: str) -> Dict[str, Any]:
        return self.preset_queries.get(query_key, self.preset_queries["Top Spending Customers"])


# --- LESSON HEADER ---
st.title("28.2 SQL Agents: Natural Language-to-SQL, Validation & Self-Correction")
st.markdown("""
Converting natural language into valid, performant, and safe SQL queries is one of the most impactful enterprise
applications of LangChain agents. Because relational schemas feature complex joins, foreign keys, and dialect nuances,
agents must do more than one-shot generation: they must inspect schemas, validate query syntax against AST guardrails,
execute queries against read-only replicas, and automatically self-correct when the database engine returns syntax or schema errors.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Autonomous Natural Language-to-SQL Agentic Loop", divider="rainbow")
st.markdown("""
The autonomous SQL agent operates in a closed-loop feedback cycle. The user prompt triggers schema discovery;
the LLM generates dialect-specific SQL; an AST validator verifies security; and if execution fails, the database
traceback is fed back into the model to trigger automated query self-correction.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SQL_Agents_Loop_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_input_layer {
            label = "User Natural Language Input";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            UserQuery [label="User Query\\n('Find active clients with overdue balances')", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_agent_reasoning {
            label = "Agent Reasoning Loop (LangGraph / ReAct)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            SchemaRetriever [label="1. Schema Understanding\\n(Identify relevant tables & columns from catalog)", fillcolor="#1E293B", color="#60A5FA"];
            SQLGenerator [label="2. SQL Synthesis\\n(Few-shot guided dialect query generation)", fillcolor="#1E293B", color="#60A5FA"];
            ASTValidator [label="3. AST & Safety Validation\\n(sqlglot parse, reject non-SELECT queries)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_execution_feedback {
            label = "Execution & Closed-Loop Self-Correction";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            DBExecutor [label="4. Read-Only DB Execution\\n(Execute with timeout & row limit top_k)", fillcolor="#0369A1", color="#38BDF8"];
            ErrorInspector [label="5. Error Analyzer\\n(Catches syntax error, unknown column, type mismatch)", fillcolor="#0369A1", color="#38BDF8"];
            SelfCorrection [label="6. Query Self-Correction Loop\\n(Feed traceback back to LLM for revision)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_delivery {
            label = "Synthesized Natural Response";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ResultNarrative [label="Natural Language Response\\n(Answer + Dataframe Summary + Citations)", fillcolor="#065F46", color="#10B981"];
        }

        UserQuery -> SchemaRetriever [label="Initial prompt"];
        SchemaRetriever -> SQLGenerator [label="Injected table DDL"];
        SQLGenerator -> ASTValidator [label="Candidate SQL"];
        ASTValidator -> DBExecutor [label="Validated query"];
        DBExecutor -> ErrorInspector [label="Execution result"];
        ErrorInspector -> ResultNarrative [label="Query succeeded"];
        ErrorInspector -> SelfCorrection [label="SQL syntax / table error"];
        SelfCorrection -> SQLGenerator [label="Re-prompt with error trace"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: NATURAL LANGUAGE TO SQL ---
st.header("2. Natural Language to SQL Translation: Intent, Entity & Dialect Awareness", divider="rainbow")
st.markdown("""
Translating natural language into SQL requires bridging ambiguous human intent with rigid mathematical relational algebra.
A query such as 'Who are our best customers?' requires the agent to interpret 'best' as total lifetime spend (`SUM(total_amount)`),
order frequency (`COUNT(order_id)`), or margin, and map those concepts to specific tables and columns.
""")

col_nl1, col_nl2 = st.columns(2)
with col_nl1:
    st.markdown("### Translation Challenges")
    st.markdown("""
    - **Ambiguity Resolution**: Translating subjective terms ('best', 'recent', 'overdue') into explicit SQL `WHERE` clauses.
    - **Entity Disambiguation**: Mapping entity names ('Acme') to exact database keys or case-insensitive string filters (`ILIKE '%acme%'`).
    - **Dialect Specifics**: Generating correct SQL syntax across PostgreSQL (`DATE_TRUNC`), MySQL (`DATE_FORMAT`), and SQLite (`strftime`).
    """)
with col_nl2:
    st.markdown("### Translation Contract")
    st.code("""
# Prompt instruction enforcing deterministic SQL translation
SQL_PREFIX = \"\"\"You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain,
always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples.
Never query for all the columns from a specific table, only ask for the few relevant columns.\"\"\"
    """, language="python")

# --- SECTION 3: SQL GENERATION ---
st.header("3. SQL Generation Techniques: Few-Shot Demonstrations and Complex Joins", divider="rainbow")
st.markdown("""
Few-shot prompting drastically improves SQL generation accuracy on complex schemas involving multi-table joins,
groupings, and subqueries. By providing example question-SQL pairs, the model learns domain-specific conventions
and foreign key join paths.
""")

st.code("""
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {
        "input": "List top 5 products by revenue in 2026",
        "query": "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id WHERE o.order_date >= '2026-01-01' GROUP BY p.name ORDER BY revenue DESC LIMIT 5;"
    },
    {
        "input": "Count overdue invoices per customer",
        "query": "SELECT c.name, COUNT(i.id) AS overdue_count FROM customers c JOIN invoices i ON c.id = i.customer_id WHERE i.status = 'overdue' GROUP BY c.name;"
    }
]

example_prompt = PromptTemplate(
    input_variables=["input", "query"],
    template="User Input: {input}\\nSQL Query: {query}"
)
""", language="python")

# --- SECTION 4: SQL VALIDATION ---
st.header("4. SQL Validation & AST Security: Blocking Dangerous Statements with sqlglot", divider="rainbow")
st.markdown("""
Never execute LLM-generated SQL directly against a database without AST validation.
Using SQL AST parsers (such as `sqlglot`), the system parses the query syntax tree and rejects any statement
containing `DROP`, `DELETE`, `UPDATE`, `ALTER`, or `INSERT`, guaranteeing that only read-only `SELECT` queries reach the engine.
""")

st.code("""
import sqlglot
from sqlglot.expressions import Select

def validate_sql_safety(query_str: str) -> bool:
    try:
        # Parse query into Abstract Syntax Tree (AST)
        expressions = sqlglot.parse(query_str)
        if not expressions:
            return False
            
        for expr in expressions:
            # Enforce that every root expression is strictly a SELECT statement
            if not isinstance(expr, Select):
                raise PermissionError("Disallowed statement: Only SELECT queries permitted.")
                
            # Scan for dangerous AST nodes
            forbidden_nodes = {"Drop", "Delete", "Update", "Insert", "AlterTable"}
            for node in expr.walk():
                if type(node).__name__ in forbidden_nodes:
                    raise PermissionError(f"Forbidden SQL operation: {type(node).__name__}")
        return True
    except Exception as exc:
        raise ValueError(f"SQL validation error: {str(exc)}")
""", language="python")

# --- SECTION 5: QUERY EXECUTION ---
st.header("5. Query Execution & Result Formatting: Row Limits, Timeouts & DataFrames", divider="rainbow")
st.markdown("""
Once validated, the query executes within an isolated database transaction with a strict execution timeout.
The returned cursor rows are converted into Pandas DataFrames or formatted Markdown tables for the LLM to synthesize.
""")

st.code("""
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
import pandas as pd

def execute_safe_query(db: SQLDatabase, query: str, top_k: int = 10) -> pd.DataFrame:
    # Ensure query has an enforced limit
    if "LIMIT" not in query.upper():
        query = f"{query.rstrip(';')} LIMIT {top_k};"
        
    with db._engine.connect() as conn:
        # Set execution timeout on session (PostgreSQL example)
        conn.execute(text("SET statement_timeout = 5000;")) # 5 seconds
        result = conn.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
""", language="python")

# --- SECTION 6: QUERY CORRECTION ---
st.header("6. Automated Query Self-Correction: Closed-Loop Traceback Recovery", divider="rainbow")
st.markdown("""
When an LLM generates a query referencing a non-existent column or a syntax error, a naive pipeline fails.
An autonomous agent catches the database engine's error traceback, appends it to the conversation history,
and prompts the model to correct the query. Models fix their own errors on the second attempt over 85% of the time.
""")

st.code("""
def self_correcting_sql_loop(agent, initial_query: str, max_retries: int = 2):
    attempts = 0
    current_prompt = initial_query
    
    while attempts < max_retries:
        try:
            # Agent generates and attempts query
            result = agent.invoke({"input": current_prompt})
            return result["output"]
        except Exception as db_err:
            attempts += 1
            # Feed exact database error back into agent
            current_prompt = (
                f"Previous SQL attempt failed with error: {str(db_err)}. "
                "Please inspect table schema, correct the syntax, and re-run."
            )
            
    raise RuntimeError("SQL Agent failed after maximum self-correction attempts.")
""", language="python")

# --- SECTION 7: DATABASE SCHEMA UNDERSTANDING ---
st.header("7. Database Schema Understanding: Semantic Schema Pruning for Large Databases", divider="rainbow")
st.markdown("""
In enterprise databases with hundreds of tables, injecting all schemas into the prompt exhausts the context window.
Advanced SQL agents use **semantic schema pruning**: vector search over table and column documentation retrieves
only the top-3 to top-5 most relevant table schemas matching the user inquiry.
""")

st.code("""
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Build vector index over table metadata descriptions
table_descriptions = [
    "customers: Stores user account details, subscriptions, and billing tiers.",
    "orders: Contains transaction records, order dates, totals, and statuses.",
    "inventory: Tracks warehouse stock levels, restocking dates, and suppliers."
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(table_descriptions, embeddings)

def get_relevant_tables(user_prompt: str) -> list[str]:
    docs = vectorstore.similarity_search(user_prompt, k=2)
    # Extracts table names from top matching documentation
    return [doc.page_content.split(":")[0] for doc in docs]
""", language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Hardened Autonomous SQL Agent with Self-Correction", divider="rainbow")
st.markdown("""
Here is a complete, production-grade SQL agent workflow with AST safety validation,
few-shot schema prompt integration, and automated error recovery.
""")

st.code("""
# production_sql_agent.py
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

def build_production_sql_agent(db_uri: str, allowed_tables: list[str]):
    # 1. Connect read-only database
    db = SQLDatabase.from_uri(
        db_uri,
        include_tables=allowed_tables,
        sample_rows_in_table_info=2
    )
    
    # 2. Configure model with zero temperature
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    # 3. Create toolkit and agent with error recovery
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        top_k=20,
        max_iterations=6
    )
    return agent
""", language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Text-to-SQL Generator & Query Self-Correction Simulator", divider="rainbow")
st.markdown("""
Observe the natural language to SQL translation process, inspect generated queries,
and witness how error tracebacks trigger automated self-correction loops.
""")

agent_sim = SQLAgentReasoningSimulator()

col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("💬 Natural Language Prompt Selector")
    chosen_scenario = st.selectbox(
        "Select Inquiry Scenario:",
        list(agent_sim.preset_queries.keys())
    )
    run_agent_btn = st.button("🚀 Run Autonomous SQL Agent Loop", type="primary")

with col_out:
    st.subheader("⚡ Agent Trajectory & Output")
    if run_agent_btn:
        res = agent_sim.simulate_agent_turn(chosen_scenario)
        badge = "<span class='badge-pass'>VALID & EXECUTED</span>" if not res["corrected"] else "<span class='badge-correct'>AUTOMATICALLY SELF-CORRECTED</span>"
        
        st.markdown(
            f"<div class='agent-box'>"
            f"<b>User Query:</b> <code>{res['natural_query']}</code><br><br>"
            f"<b>Status:</b> {badge}<br><br>"
            f"<b>Generated SQL:</b><br><code>{res['initial_sql']}</code>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        if res["corrected"]:
            st.warning(f"Initial Error: {res['error_returned']}")
            st.info(f"Self-Corrected SQL: {res['corrected_sql']}")
            
        st.caption("Tabular Result Set Returned to Agent:")
        st.dataframe(res["data"], use_container_width=True)
    else:
        st.info("Select a scenario and run to inspect SQL synthesis and self-correction.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does a SQL agent recover when a query fails with a database syntax error?"):
    st.markdown("""
    **Answer**: In an agentic ReAct loop, when a tool execution returns an error (e.g. `Column not found in table`),
    the error message is returned to the agent as a tool observation. The model reads the traceback, compares it against
    the table schema, identifies the typo or mismatched join condition, synthesizes an amended SQL query, and re-executes.
    """)

with st.expander("Q2: Why should SQL validation use AST parsing (like sqlglot) rather than regex string checks?"):
    st.markdown("""
    **Answer**: Regex string matching can be easily bypassed using SQL comments (`SELECT /* comment */ ...`), multi-line queries,
    or encoded strings. An Abstract Syntax Tree (AST) parser parses the full grammar of the SQL dialect, enabling 100% deterministic
    verification that the root operation is strictly a `Select` statement with no hidden DDL or DML mutations.
    """)

with st.expander("Q3: What causes SQL agents to generate Cartesian product joins, and how is it prevented?"):
    st.markdown("""
    **Answer**: Cartesian products (`CROSS JOIN`) occur when models join tables without specifying an explicit `ON a.id = b.a_id` condition.
    This happens when foreign key relationships are omitted from the schema prompt. Providing complete DDL statements with explicit
    `FOREIGN KEY REFERENCES` constraints and few-shot join examples teaches the model the exact join paths.
    """)

with st.expander("Q4: How does semantic schema pruning optimize SQL generation on large databases?"):
    st.markdown("""
    **Answer**: When an enterprise database has 300 tables, dumping all DDLs into the prompt exhausts token limits and confuses the model.
    Semantic schema pruning indexes table and column descriptions in a vector store. When a user asks a question, the agent retrieves
    only the top 3-5 relevant table schemas, keeping prompts compact and focused.
    """)

with st.expander("Q5: Why should `top_k` row limits be enforced in SQL agent generation?"):
    st.markdown("""
    **Answer**: Without row limits, analytical queries (e.g. `SELECT * FROM log_events`) can return millions of rows,
    crashing the application server with out-of-memory errors and overwhelming the LLM context window. An enforced `top_k` parameter
    (e.g. 10 or 25) ensures result sets remain manageable.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Allowing LLM to Write DML/DDL)**: Permitting `INSERT`, `UPDATE`, or `DROP` statements in production. A misunderstood query or prompt injection can permanently delete business records.")
    st.warning("**Trap 2 (Infinite Query Correction Loops)**: Allowing an agent to retry failed queries without a `max_iterations` limit, resulting in runaway token consumption on impossible queries.")
with col_trap2:
    st.error("**Trap 3 (Missing Statement Timeouts)**: Executing unindexed multi-table join queries without database session timeouts, locking database worker threads for minutes.")
    st.warning("**Trap 4 (Ignoring NULL Value Semantics)**: Forgetting that SQL `COUNT(column)` ignores NULLs while `COUNT(*)` counts all rows, leading to incorrect aggregations in generated queries.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What tool checks SQL query safety via AST parsing?")
        st.caption("Flip for answer:")
        st.info("**Back**: `sqlglot` parses queries into Abstract Syntax Trees to verify read-only `SELECT` operations.")
    with st.container(border=True):
        st.markdown("**Front**: What happens when an LLM SQL tool call throws a database error?")
        st.caption("Flip for answer:")
        st.info("**Back**: The error traceback is returned as a tool observation, allowing the model to self-correct.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does semantic schema pruning help in large databases?")
        st.caption("Flip for answer:")
        st.info("**Back**: Vector search retrieves only the relevant table schemas matching the user query, avoiding prompt bloat.")
    with st.container(border=True):
        st.markdown("**Front**: Why must foreign key relationships be provided in schema prompts?")
        st.caption("Flip for answer:")
        st.info("**Back**: To prevent the model from hallucinating join conditions or creating accidental Cartesian products.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the multi-step reasoning loop of autonomous SQL agents.", value=True)
c2 = st.checkbox("I can implement few-shot prompt templates to guide complex multi-table SQL joins.", value=True)
c3 = st.checkbox("I know how to use sqlglot to validate SQL queries and block non-SELECT statements.", value=True)
c4 = st.checkbox("I understand how query error tracebacks empower automated agent self-correction.", value=True)
c5 = st.checkbox("I can implement semantic schema pruning for large multi-table enterprise catalogs.", value=True)
c6 = st.checkbox("I know how to enforce session timeouts and row limits on all executed queries.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 28.3: NoSQL.")
