import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="14.2 Checkpointers - LangGraph Curriculum",
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
    .badge-backend {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-postgres {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class CheckpointerComparisonEngine:
    """
    Compares and simulates the 3 primary checkpointer backends:
    MemorySaver, SqliteSaver, and PostgresSaver.
    """
    def __init__(self):
        self.backends = {
            "MemorySaver": {
                "storage_type": "Volatile RAM (dict)",
                "persistence_durability": "None (lost on restart)",
                "concurrency_support": "Single process only",
                "setup_overhead": "Zero setup required",
                "recommended_for": "Unit tests, prototypes, tutorials"
            },
            "SqliteSaver": {
                "storage_type": "Single-file on-disk DB",
                "persistence_durability": "Full disk durability",
                "concurrency_support": "File lock limited",
                "setup_overhead": "Minimal (file path)",
                "recommended_for": "Edge appliances, local CLI tools"
            },
            "PostgresSaver": {
                "storage_type": "Relational ACID Cluster",
                "persistence_durability": "Enterprise multi-AZ replication",
                "concurrency_support": "High concurrent connection pool",
                "setup_overhead": "PostgreSQL instance & setup() tables",
                "recommended_for": "Production cloud microservices"
            }
        }

    def simulate_save_and_retrieve(self, backend: str, thread_id: str, payload_size_kb: int) -> Dict[str, Any]:
        info = self.backends[backend]
        latency = 0.5 if backend == "MemorySaver" else (4.2 if backend == "SqliteSaver" else 12.8)
        return {
            "backend": backend,
            "thread_id": thread_id,
            "latency_ms": latency + round(float(np.random.uniform(0.1, 1.5)), 2),
            "durability": info["persistence_durability"],
            "storage_medium": info["storage_type"],
            "bytes_written": payload_size_kb * 1024
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Checkpointer Architecture and Database Storage Engines", divider="rainbow")

st.markdown("""
A **Checkpointer** is the persistence backend plugged into LangGraph at compile time.
LangGraph defines a clean `BaseCheckpointSaver` interface, allowing developers to switch between
in-memory development environments and high-throughput enterprise databases (Postgres, SQLite, Redis)
without modifying a single line of graph logic.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    checkpointer_blueprint_dot = """
    digraph CheckpointerBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        BASE_INTERFACE [label="BaseCheckpointSaver (Abstract Base Class)\\n- put(config, checkpoint, metadata, new_versions)\\n- get_tuple(config)\\n- list(config, filter, before, limit)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_implementations {
            label = "Concrete Storage Engines";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            MEMORY_SAVER [label="MemorySaver\\nIn-Memory Dict\\nZero Setup / Non-Durable", fillcolor="#1E1E38", color="#8B5CF6"];
            SQLITE_SAVER [label="SqliteSaver\\nOn-Disk SQLite DB\\nEmbedded / Edge Persistent", fillcolor="#431407", color="#FB923C"];
            POSTGRES_SAVER [label="PostgresSaver\\nProduction PostgreSQL Cluster\\nACID / Scalable / Multi-Node", fillcolor="#064E3B", color="#34D399"];
        }

        subgraph cluster_graph {
            label = "Graph Compilation & Invocation";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            COMPILE_STEP [label="builder.compile(checkpointer=chosen_saver)", fillcolor="#0F172A", color="#F59E0B"];
            INVOKE_STEP [label="app.invoke(input, config={'configurable': {'thread_id': '...'}})", fillcolor="#0F172A", color="#34D399"];
            COMPILE_STEP -> INVOKE_STEP;
        }

        BASE_INTERFACE -> MEMORY_SAVER;
        BASE_INTERFACE -> SQLITE_SAVER;
        BASE_INTERFACE -> POSTGRES_SAVER;

        POSTGRES_SAVER -> COMPILE_STEP [label="Injected at compile time"];
    }
    """
    st.graphviz_chart(checkpointer_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Design Invariant:</b> The graph topology never cares where state is stored.
The exact same compiled graph can use <code>MemorySaver</code> in pytest unit tests,
<code>SqliteSaver</code> in local debugging, and <code>PostgresSaver</code> in Amazon EKS production clusters.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: IN-MEMORY CHECKPOINTER
# ==============================================================================
st.header("2. In-Memory Checkpointer: The MemorySaver Primitive for Prototyping and Testing", divider="rainbow")

st.markdown("""
`MemorySaver` is the simplest checkpointer:
""")

st.code("""
from langgraph.checkpoint.memory import MemorySaver

# Zero-configuration in-memory saver
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
""", language="python")

st.markdown("""
- **Best for**: Unit testing, local Jupyter notebooks, and quick proofs-of-concept.
- **Limitation**: State lives in the Python process RAM. Terminating the process deletes all checkpoints.
""")

# ==============================================================================
# SECTION 3: DATABASE-BACKED PERSISTENCE
# ==============================================================================
st.header("3. Database-Backed Persistence: SqliteSaver for Local and Embedded Environments", divider="rainbow")

st.markdown("""
`SqliteSaver` provides on-disk persistence without needing an external database server:
""")

st.code("""
from langgraph.checkpoint.sqlite import SqliteSaver

# Saves checkpoints to local SQLite file
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = builder.compile(checkpointer=checkpointer)
    # Checkpoints survive process restarts!
""", language="python")

# ==============================================================================
# SECTION 4: PRODUCTION PERSISTENCE
# ==============================================================================
st.header("4. Production Persistence: Enterprise PostgresSaver with Connection Pooling", divider="rainbow")

st.markdown("""
`PostgresSaver` is the enterprise standard for production deployments:
""")

st.code("""
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://agent_user:secure_pass@db.internal:5432/agent_state"

# 1. Initialize checkpointer
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 2. Automatically create necessary checkpoint tables if not present
    checkpointer.setup()
    
    # 3. Compile graph
    app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 5: CHECKPOINT CONFIGURATION
# ==============================================================================
st.header("5. Checkpoint Configuration: Setting thread_id, checkpoint_ns, and Metadata", divider="rainbow")

st.markdown("""
When interacting with checkpointers, configuration is passed via `RunnableConfig`:
- **`thread_id`**: The primary partition key separating conversation sessions.
- **`checkpoint_ns`**: Optional namespace for nested subgraph state isolation.
- **`checkpoint_id`**: Specific historical snapshot ID (used for Time Travel).
""")

st.code("""
config = {
    "configurable": {
        "thread_id": "org_acme_ticket_9912",
        "checkpoint_ns": "billing_subgraph"
    }
}
""", language="python")

# ==============================================================================
# SECTION 6: CHECKPOINT STORAGE
# ==============================================================================
st.header("6. Checkpoint Storage: Schema Tables, Blob Serialization, and Pruning Policies", divider="rainbow")

st.markdown("""
Under the hood, `PostgresSaver` maintains two core tables:
1. **`checkpoints`**: Stores serialized state payloads, channel versions, and timestamps.
2. **`checkpoint_blobs`**: Stores large objects separated from metadata for query efficiency.
3. **`checkpoint_writes`**: Stores pending intermediate writes during supersteps.
""")

table_df = pd.DataFrame([
    {
        "Table Name": "checkpoints",
        "Primary Keys": "(thread_id, checkpoint_ns, checkpoint_id)",
        "Contents": "Metadata, timestamps, serialized state channels dictionary"
    },
    {
        "Table Name": "checkpoint_blobs",
        "Primary Keys": "(thread_id, checkpoint_ns, channel, version)",
        "Contents": "Heavy binary payloads and serialized message contents"
    },
    {
        "Table Name": "checkpoint_writes",
        "Primary Keys": "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)",
        "Contents": "Intermediate node writes pending superstep commit"
    }
])
st.table(table_df)

# ==============================================================================
# SECTION 7: CHECKPOINT RETRIEVAL
# ==============================================================================
st.header("7. Checkpoint Retrieval: Querying Snapshots via get_state() and get_state_history()", divider="rainbow")

st.markdown("""
Checkpointers provide two inspection methods:
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("#### `get_state(config)`")
    st.markdown("Returns the **latest** state snapshot for the specified thread:")
    st.code("""
latest_snapshot = app.get_state(config)
print(latest_snapshot.values)
print(latest_snapshot.next)
""", language="python")

with col_r2:
    st.markdown("#### `get_state_history(config)`")
    st.markdown("Returns an iterator of **all historical** snapshots for the thread:")
    st.code("""
for snapshot in app.get_state_history(config):
    print(snapshot.config["configurable"]["checkpoint_id"])
    print(snapshot.created_at)
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Postgres-Ready Enterprise Checkpointer Workflow", divider="rainbow")

st.markdown("""
Below is the production implementation of an enterprise graph configured with connection pool parameters
and state inspection methods:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class EnterpriseOrderState(MessagesState):
    order_id: str
    order_status: str

# 2. Checkpointer Initialization
# In production, replace MemorySaver with PostgresSaver.from_conn_string(DB_URI)
checkpointer = MemorySaver()

# 3. Node Definitions
def order_processor_node(state: EnterpriseOrderState) -> dict:
    order_id = state.get("order_id", "ORD-UNKNOWN")
    ack = AIMessage(content=f"Order {order_id} processed and validated.")
    return {
        "messages": [ack],
        "order_status": "FULFILLED"
    }

# 4. Assemble Graph
builder = StateGraph(EnterpriseOrderState)
builder.add_node("process", order_processor_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

app = builder.compile(checkpointer=checkpointer)

# 5. Invocation and Retrieval
thread_config = {"configurable": {"thread_id": "cust_order_7741"}}
app.invoke({"order_id": "ORD-9920", "messages": [HumanMessage(content="Submit order")]}, config=thread_config)

# 6. Verify Persistence
snapshot = app.get_state(thread_config)
assert snapshot.values["order_status"] == "FULFILLED"
print("Order persisted successfully with checkpoint ID:", snapshot.config["configurable"]["checkpoint_id"])
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Backend Checkpointer Simulator & Retrieval Inspector", divider="rainbow")

st.markdown("""
Compare the storage medium, latency, and durability profiles of `MemorySaver`, `SqliteSaver`,
and `PostgresSaver`. Simulate write and retrieval cycles in real time.
""")

backend_engine = CheckpointerComparisonEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Backend Configuration")
    chosen_backend = st.selectbox(
        "Select Checkpointer Implementation:",
        ["MemorySaver", "SqliteSaver", "PostgresSaver"],
        index=2
    )
    thread_input = st.text_input("Thread ID:", value="tenant_acme_thread_102")
    payload_size = st.slider("Simulated State Payload Size (KB):", min_value=1, max_value=250, value=25)
    
    test_btn = st.button("🚀 Test Write & Retrieve Cycle", type="primary")

with wb_col2:
    st.subheader("📊 Performance & Storage Telemetry")
    if test_btn:
        res = backend_engine.simulate_save_and_retrieve(chosen_backend, thread_input, payload_size)
        badge_class = "badge-postgres" if chosen_backend == "PostgresSaver" else "badge-backend"
        
        st.markdown(f"### Backend: <span class='{badge_class}'>{chosen_backend}</span>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Round-Trip Latency", f"{res['latency_ms']} ms")
        with c2:
            st.metric("Bytes Committed", f"{res['bytes_written']} bytes")

        st.info(f"**Storage Medium**: {res['storage_medium']}")
        st.write(f"**Persistence Durability**: {res['durability']}")
    else:
        st.info("Select a checkpointer backend and click 'Test Write & Retrieve Cycle' to benchmark.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What are the primary differences between MemorySaver, SqliteSaver, and PostgresSaver?", expanded=False):
    st.markdown("""
    **Answer**:
    - `MemorySaver`: In-memory Python dictionary. Zero setup, fast, non-durable. Lost on process restart.
    - `SqliteSaver`: Embedded on-disk relational file. Durable, but limited concurrency due to SQLite database file locks.
    - `PostgresSaver`: Distributed ACID relational database. Enterprise durability, multi-AZ replication, and handles thousands of concurrent worker threads.
    """)

with st.expander("Q2: Why must checkpointer.setup() be called when deploying PostgresSaver?", expanded=False):
    st.markdown("""
    **Answer**:
    `checkpointer.setup()` executes database DDL migrations. It creates the required `checkpoints`, `checkpoint_blobs`,
    and `checkpoint_writes` tables along with indexes on `(thread_id, checkpoint_id)`.
    Without running `setup()`, the first write operation will fail with table missing errors.
    """)

with st.expander("Q3: How does checkpoint_ns enable nested subgraphs?", expanded=False):
    st.markdown("""
    **Answer**:
    When a parent graph invokes a child subgraph, both graphs can share the same `thread_id` while isolating their internal
    channel states using distinct `checkpoint_ns` namespaces (e.g. `""` for parent and `"research_subgraph"` for child).
    """)

with st.expander("Q4: Can you prune or garbage-collect old checkpoints in PostgreSQL?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes. In high-volume systems, older checkpoints can be pruned using periodic cron jobs that delete checkpoints
    older than N days while retaining the latest checkpoint per `thread_id`.
    """)

with st.expander("Q5: What is the purpose of the checkpoint_writes table?", expanded=False):
    st.markdown("""
    **Answer**:
    `checkpoint_writes` acts as a write-ahead staging area. When multiple nodes execute concurrently within a superstep,
    their intermediate return deltas are written to `checkpoint_writes` until all nodes complete, after which the merged
    checkpoint is committed atomically.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Omitting checkpointer.setup() on Fresh Databases")
    st.markdown("""
    *Mechanism*: Connecting `PostgresSaver` to a new PostgreSQL instance without running `setup()`.  
    *Consequence*: Every graph execution crashes with `UndefinedTable: relation "checkpoints" does not exist`.  
    *Fix*: Always call `checkpointer.setup()` during application boot.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): SQLite Concurrency Lock Timeouts in Microservices")
    st.markdown("""
    *Mechanism*: Using `SqliteSaver` in a multi-threaded web server handling 50 concurrent requests.  
    *Consequence*: Requests crash with `sqlite3.OperationalError: database is locked`.  
    *Fix*: Upgrade to `PostgresSaver` for concurrent write workloads.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Connection Pool Starvation")
    st.markdown("""
    *Mechanism*: Instantiating a new `PostgresSaver` connection on every single API request without connection pooling.  
    *Consequence*: Database runs out of client connections (`too many clients already`).  
    *Fix*: Use a shared singleton `PostgresSaver` with a managed connection pool (`psycopg_pool`).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Checkpoint Table Unbounded Growth")
    st.markdown("""
    *Mechanism*: Persisting thousands of turns per day without an automated data retention/pruning strategy.  
    *Consequence*: Checkpoint storage size balloons to hundreds of gigabytes over time.  
    *Fix*: Implement automated pruning jobs to archive historical checkpoints.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**BaseCheckpointSaver**\\nThe abstract LangGraph interface implemented by all checkpointers.")
with col_fc2:
    st.info("**PostgresSaver**\\nThe enterprise-grade checkpointer for persistent state in production PostgreSQL clusters.")
with col_fc3:
    st.info("**checkpointer.setup()**\\nThe method that auto-creates tables and indexes in relational checkpointer databases.")
with col_fc4:
    st.info("**checkpoint_ns**\\nNamespace key used to isolate channel state for nested subgraphs.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural differences between MemorySaver, SqliteSaver, and PostgresSaver.", value=True)
c2 = st.checkbox("I know how to configure and initialize PostgresSaver with connection strings.", value=True)
c3 = st.checkbox("I understand why checkpointer.setup() is mandatory on fresh databases.", value=True)
c4 = st.checkbox("I know how to query thread history using get_state() and get_state_history().", value=True)
c5 = st.checkbox("I understand the schema roles of checkpoints, checkpoint_blobs, and checkpoint_writes.", value=True)
c6 = st.checkbox("I know how to configure connection pools to prevent database starvation.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 14.2 'Checkpointers' with full architectural and practical comprehension!")
