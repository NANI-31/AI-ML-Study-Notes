import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="27.1 Database Integration - LangGraph Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-db {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class DatabaseIntegrationSimulator:
    """
    Simulates database routing between state checkpointers
    and domain-specific user/business relational databases.
    """
    def __init__(self):
        self.mock_user_db = {
            "users": [
                {"id": 101, "name": "Alice Chen", "tier": "Enterprise", "balance_usd": 8500},
                {"id": 102, "name": "Bob Miller", "tier": "Standard", "balance_usd": 1200},
                {"id": 103, "name": "Carol Danvers", "tier": "Enterprise", "balance_usd": 24000}
            ]
        }
        self.checkpoint_ledger = [
            {"thread_id": "thread-usr-101", "checkpoint_id": "chk-001", "step": 1, "size_bytes": 1420},
            {"thread_id": "thread-usr-101", "checkpoint_id": "chk-002", "step": 2, "size_bytes": 3150},
            {"thread_id": "thread-usr-102", "checkpoint_id": "chk-001", "step": 1, "size_bytes": 1280}
        ]

    def query_user_db(self, user_id: int) -> Optional[Dict[str, Any]]:
        for u in self.mock_user_db["users"]:
            if u["id"] == user_id:
                return u
        return None

    def query_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        return [c for c in self.checkpoint_ledger if c["thread_id"] == thread_id]

db_sim = DatabaseIntegrationSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Polyglot Persistence, Relational Data, and Checkpoint Separation Topology", divider="rainbow")

st.markdown("""
Integrating databases with LangGraph applications requires understanding the strict operational boundary between 
**Checkpoint Persistence** (where graph execution history and thread states live) and **Domain/User Databases** 
(where enterprise business records, customer accounts, and transactions live). 

A production LangGraph architecture implements polyglot persistence: utilizing high-throughput relational databases 
like PostgreSQL for durable checkpoints, in-memory caches like Redis for low-latency session scratchpads, 
and domain-specific SQL/NoSQL databases queried securely by tools using parameterized, read-only connections.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DatabaseBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_graph {
            label="1. LangGraph Execution Runtime";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            agent_node [label="Agent Reasoning Node\\nConstructs SQL Queries / Evaluates Data", fillcolor="#0F172A"];
            tool_sql [label="SQL Tool Executor\\nParameterized Query Runner", fillcolor="#1E293B"];
            tool_nosql [label="NoSQL Tool Executor\\nMongoDB Document Queries", fillcolor="#1E293B"];
        }

        subgraph cluster_checkpointer_db {
            label="2. Checkpoint Persistence Storage (System Managed)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            pg_checkpointer [label="PostgreSQL (PostgresSaver)\\ncheckpoints & checkpoint_writes Tables\\nThread Isolation & Time Travel DAG", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            redis_checkpointer [label="Redis (RedisSaver)\\nIn-Memory Cache & Session TTL", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        subgraph cluster_domain_db {
            label="3. Domain & User Business Databases (Application Managed)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            app_postgres [label="PostgreSQL (Enterprise OLTP)\\nUsers, Invoices, Billing, Products", fillcolor="#0F172A"];
            app_mongo [label="MongoDB / DynamoDB (NoSQL Store)\\nUnstructured Profiles & Event Logs", fillcolor="#0F172A"];
        }

        agent_node -> tool_sql [label="Generates Query"];
        agent_node -> tool_nosql [label="Fetches Profile"];

        agent_node -> pg_checkpointer [label="Commit Superstep Snapshots"];
        agent_node -> redis_checkpointer [label="Ephemeral Cache"];

        tool_sql -> app_postgres [label="Parameterized Read-Only SQL"];
        tool_nosql -> app_mongo [label="Document Filter"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: SQL Databases
# ==============================================================================
st.header("2. SQL Databases: Schema Reflection, Parameterized Querying, and Read-Only Connection Sandboxing", divider="rainbow")

st.markdown("""
Enabling agents to query SQL databases requires implementing strict security boundaries to prevent SQL injection 
or accidental data corruption:

1. **Schema Reflection**: Provide the agent with sanitized schema definitions (table names, column types, foreign keys) 
   rather than allowing `SELECT *` exploration.
2. **Parameterized Prepared Statements**: Tools must execute queries via parameterized query interfaces 
   (e.g., `execute("SELECT * FROM users WHERE id = :id", {"id": user_id})`) rather than concatenating raw text.
3. **Read-Only Credentials**: Connect database tools through a dedicated PostgreSQL user with only `SELECT` privileges 
   and a strict statement execution timeout (`SET statement_timeout = '3000ms'`).
""")

st.code('''from langchain_core.tools import tool
from sqlalchemy import create_engine, text

# Dedicated read-only engine with statement timeout
engine = create_engine(
    "postgresql://agent_ro:readonly_pwd@pg.internal:5432/corp_data",
    connect_args={"options": "-c statement_timeout=3000"}
)

@tool
def query_financial_db(account_id: int) -> str:
    """Safely retrieves customer account balances using parameterized SQL."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT name, balance FROM accounts WHERE id = :id"),
            {"id": account_id}
        ).fetchone()
        if not result:
            return "No account found."
        return f"Account {result.name}: Balance = {result.balance} USD"
''', language="python")

# ==============================================================================
# SECTION 3: NoSQL Databases
# ==============================================================================
st.header("3. NoSQL Databases: Document Stores, Unstructured State, and Flexible Schemas", divider="rainbow")

st.markdown("""
Document databases (such as MongoDB, Couchbase, or DynamoDB) are ideal for managing unstructured agent data:
- **Long-Term Memory Stores**: Storing rich, nested user preference profiles that evolve over time without requiring database schema migrations.
- **Multimodal State Blobs**: Storing large JSON objects, tool outputs, and document extraction trees.
- **Session Caches**: Key-value fast lookup of session context.
""")

# ==============================================================================
# SECTION 4: PostgreSQL
# ==============================================================================
st.header("4. PostgreSQL: High-Throughput Checkpointing, Row-Level Locks, and Transaction Pooling with PgBouncer", divider="rainbow")

st.markdown("""
PostgreSQL is the gold standard for enterprise LangGraph checkpointers via `PostgresSaver`. 
Under high concurrency, optimizing Postgres performance requires:

- **Row-Level Thread Locking**: Postgres checkpointer queries acquire row-level locks on `(thread_id, checkpoint_id)` 
  during superstep writes, ensuring that concurrent nodes serialize deterministically without deadlocks.
- **Transaction Pooling**: Deploying **PgBouncer** or **AWS RDS Proxy** in transaction pooling mode 
  ensures that hundreds of concurrent serverless or Kubernetes pods can share a pool of 20-50 physical database connections.
""")

# ==============================================================================
# SECTION 5: Redis
# ==============================================================================
st.header("5. Redis: Low-Latency Ephemeral Checkpointing, In-Memory Caching, and Pub/Sub Event Queues", divider="rainbow")

st.markdown("""
Redis provides sub-millisecond persistence and caching for real-time agent workloads:
- **`RedisSaver`**: Ideal for low-latency customer support chat where session data expires after 24 hours.
- **Semantic Caching**: Caching previous LLM query completions using Redis vector similarity.
- **Pub/Sub Streaming**: Distributing Server-Sent Events (SSE) across multiple load-balanced API servers.
""")

# ==============================================================================
# SECTION 6: Checkpoint Databases
# ==============================================================================
st.header("6. Checkpoint Databases: Dedicated Persistence Engines, Retention TTLs, and Partition Pruning", divider="rainbow")

st.markdown("""
A critical best practice in production LangGraph engineering is deploying a **dedicated database cluster** 
for checkpoints rather than sharing the primary application business database.
""")

with st.expander("🔍 View Checkpoint Database vs Application Database Topology", expanded=True):
    db_topo_dot = """
    digraph DBTopology {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        graph_app [label="LangGraph Application Cluster\\nStateless Kubernetes Pods", fillcolor="#0F172A"];

        subgraph cluster_chk {
            label="Checkpoint Database (High Write IOPS / Append-Only)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chk_pg [label="PostgresSaver Dedicated Instance\\n- Daily Time-Based Partitioning\\n- 30-Day TTL Pruning Policy\\n- Append-Only Write Workload", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_biz {
            label="Application Business Database (High Read / Normalized)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            biz_pg [label="Corporate OLTP Database\\n- Accounts, Users, Transactions\\n- Strict ACID Relations\\n- Read-Only Sandboxed Access for Agent", fillcolor="#312E81"];
        }

        graph_app -> chk_pg [label="Automatic State Checkpoints (High Churn)"];
        graph_app -> biz_pg [label="Explicit Tool Queries (Read-Only)"];
    }
    """
    st.graphviz_chart(db_topo_dot, use_container_width=True)

# ==============================================================================
# SECTION 7: User Databases
# ==============================================================================
st.header("7. User Databases: Isolating Business Relational Entities, RBAC Profiles, and Cross-Session User Profiles", divider="rainbow")

st.markdown("""
While checkpoints store ephemeral runtime execution state (e.g., active conversation history and intermediate loop variables), 
**User Databases** store persistent business entities across sessions:

- **User Accounts & RBAC Clearance**: Determining which tools and documents a user can access.
- **Entity Knowledge Graphs**: Structured records of customer relationships and past purchases.
- **Cross-Session Long-Term Memory**: Extracted user preferences synthesized from previous conversations.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Dual-Database Graph Architecture", divider="rainbow")

st.markdown("""
The following complete module implements a dual-database architecture: using PostgreSQL for state checkpointing 
while querying business customer records via parameterized tools.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # In prod: PostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class DatabaseAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: int
    retrieved_balance: float

def query_customer_node(state: DatabaseAgentState) -> dict:
    cid = state.get("customer_id", 0)
    # Simulated parameterized SQL query: SELECT balance FROM customers WHERE id = :cid
    mock_db = {101: 8500.0, 102: 1200.0, 103: 24000.0}
    bal = mock_db.get(cid, 0.0)
    
    msg = AIMessage(content=f"Database record retrieved: Customer #{cid} balance is ${bal:,.2f} USD.")
    return {
        "messages": [msg],
        "retrieved_balance": bal
    }

def create_database_agent(checkpointer):
    builder = StateGraph(DatabaseAgentState)
    builder.add_node("query_customer", query_customer_node)
    
    builder.add_edge(START, "query_customer")
    builder.add_edge("query_customer", END)
    
    return builder.compile(checkpointer=checkpointer)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Database Routing Simulator & Query Execution Sandbox", divider="rainbow")

st.markdown("""
Simulate enterprise polyglot database lookups, query customer accounts, and inspect checkpoint storage ledgers.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Business Relational Query Sandbox")
    selected_user = st.selectbox("Select Customer Account ID", [101, 102, 103], index=0)
    
    if st.button("Execute Parameterized Query", type="primary"):
        record = db_sim.query_user_db(selected_user)
        if record:
            st.success(f"Customer Found: **{record['name']}**")
            st.json(record)
        else:
            st.error("Account not found.")

with col_wb2:
    st.subheader("Checkpoint Storage Ledger")
    selected_thread = st.selectbox("Select Thread ID to Audit", ["thread-usr-101", "thread-usr-102"])
    chks = db_sim.query_checkpoints(selected_thread)
    
    st.markdown(f"**Found {len(chks)} Checkpoint Records for Thread `{selected_thread}`**")
    df_chks = pd.DataFrame(chks)
    st.dataframe(df_chks, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: Why should an agent's checkpoint database be kept separate from the application's business database?"):
    st.markdown("""
    **Answer:**  
    1. **Workload Isolation**: Checkpointers write large state payloads at every superstep, creating high write IOPS and disk churn that could degrade customer-facing transaction performance.  
    2. **Retention Policies**: Checkpoint databases require aggressive time-based partitioning and TTL pruning (e.g., dropping data after 30 days), whereas business data requires permanent relational consistency.  
    3. **Schema Migrations**: Decoupling prevents LangGraph library updates from impacting application relational schemas.
    """)

with st.expander("Q2: How do you prevent an autonomous agent from executing malicious or destructive SQL queries?"):
    st.markdown("""
    **Answer:**  
    1. Connect via dedicated read-only credentials (`GRANT SELECT ON ...`).  
    2. Configure strict statement timeouts (`statement_timeout = 3000ms`) to prevent DoS attacks.  
    3. Enforce parameterized query tools rather than raw string execution.  
    4. Gate any mutation operations (INSERT/UPDATE/DELETE) behind human approval via `interrupt()`.
    """)

with st.expander("Q3: When should you choose RedisSaver over PostgresSaver?"):
    st.markdown("""
    **Answer:**  
    Choose `RedisSaver` for low-latency conversational agents where state is transient, query speed is critical (sub-millisecond), 
    and sessions expire after hours. Choose `PostgresSaver` when ACID durability, permanent audit trails, time-travel debugging, 
    and long-term human approval workflows are required.
    """)

with st.expander("Q4: What is the role of PgBouncer in high-concurrency LangGraph deployments?"):
    st.markdown("""
    **Answer:**  
    When hundreds of worker pods execute agent threads simultaneously, each pod opens a pool of connections. 
    Without a connection pooler, Postgres exhausts its `max_connections` limit. PgBouncer pools transactions, 
    allowing thousands of agent threads to share a small set of physical database connections.
    """)

with st.expander("Q5: How can NoSQL document stores be utilized for cross-session long-term memory?"):
    st.markdown("""
    **Answer:**  
    NoSQL stores (like MongoDB or DynamoDB) allow agents to read and update hierarchical user profiles 
    (e.g., preferences, communication style, past purchase patterns) as flexible JSON documents without requiring fixed relational schema migrations.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Granting DDL or DML Privileges to Agent Database Roles**  
Allowing an agent database tool to execute with table creation or modification privileges introduces critical prompt injection risks 
where an injected prompt can drop tables or overwrite financial records. Always use read-only roles.
""")

st.error("""
**Trap 2: Checkpoint Table Disk Bloat Without Automated Pruning**  
Deploying `PostgresSaver` in high-throughput environments without time-based table partitioning and automated TTL pruning 
causes checkpoint tables to grow into hundreds of gigabytes within weeks, exhausting disk space.
""")

st.warning("""
**Trap 3: Unbounded SELECT Queries in SQL Tools**  
Allowing queries like `SELECT * FROM orders` without `LIMIT 20` clauses returns millions of rows, overflowing the agent's context window 
and causing out-of-memory errors.
""")

st.warning("""
**Trap 4: Forgetting Query Statement Timeouts**  
Failing to set `statement_timeout` on database connections allows accidental cartesian product queries to hang worker threads indefinitely.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Polyglot Persistence</h4>
        <p>Using different database technologies for different tasks: Postgres for durable checkpoints, Redis for caching, and domain SQL for business data.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>PostgresSaver</h4>
        <p>The standard PostgreSQL checkpointer providing ACID-compliant snapshot storage and row-level thread concurrency.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Read-Only Connection Sandbox</h4>
        <p>A database connection profile restricted strictly to SELECT queries with statement execution timeouts to prevent data corruption.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>PgBouncer Transaction Pooling</h4>
        <p>Middleware multiplexing thousands of client connections onto a small pool of PostgreSQL database worker processes.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architectural separation between checkpoint storage and domain business databases.", value=True)
chk_2 = st.checkbox("I know how to build secure, read-only SQL query tools with parameterized inputs.", value=True)
chk_3 = st.checkbox("I understand how to configure PostgresSaver with connection pooling and PgBouncer.", value=True)
chk_4 = st.checkbox("I know when to choose RedisSaver for low-latency session persistence.", value=True)
chk_5 = st.checkbox("I can implement retention policies and table partitioning to prevent checkpoint bloat.", value=True)
chk_6 = st.checkbox("I understand how to utilize NoSQL document stores for cross-session long-term memory.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Database integration and polyglot persistence mastery confirmed! Ready for Vector Databases.")
