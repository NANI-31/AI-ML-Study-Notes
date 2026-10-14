import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="14.1 What is Persistence - LangGraph Curriculum",
    page_icon="💾",
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
    .badge-ckpt {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-thread {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #9333EA;
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
class PersistenceSimulatorEngine:
    """
    Simulates checkpoint creation, thread indexing, state snapshots,
    and superstep commit lifecycles.
    """
    def __init__(self):
        self.threads: Dict[str, List[Dict[str, Any]]] = {}

    def run_superstep(self, thread_id: str, node_executed: str, state_delta: Dict[str, Any]) -> Dict[str, Any]:
        if thread_id not in self.threads:
            self.threads[thread_id] = []

        history = self.threads[thread_id]
        checkpoint_id = f"1ef_{len(history)+1:04d}_{hex(np.random.randint(1000, 9999))[2:]}"
        
        # Calculate snapshot
        prev_values = history[-1]["values"] if history else {"messages": [], "step_count": 0}
        new_values = dict(prev_values)
        new_values["step_count"] = new_values.get("step_count", 0) + 1
        for k, v in state_delta.items():
            if k == "messages":
                new_values["messages"] = new_values.get("messages", []) + [v]
            else:
                new_values[k] = v

        snapshot = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "step": len(history) + 1,
            "node": node_executed,
            "values": new_values,
            "next": ["tools"] if node_executed == "agent" else ["agent"] if node_executed == "tools" else []
        }
        history.append(snapshot)
        return snapshot

    def get_thread_snapshots(self, thread_id: str) -> List[Dict[str, Any]]:
        return self.threads.get(thread_id, [])


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Persistence Architecture and Checkpoint Lifecycle", divider="rainbow")

st.markdown("""
**Persistence** is the mechanism by which LangGraph saves the complete graph state at every **superstep**.
Rather than keeping execution variables in volatile RAM, LangGraph serializes state snapshots into a **Checkpointer**
(Postgres, SQLite, or Redis) indexed by **`thread_id`**.
This architecture enables four enterprise capabilities:
1. **Multi-Turn Chat**: Seamless memory across stateless HTTP requests.
2. **Human-in-the-Loop**: Pausing execution for hours until human approval is submitted.
3. **Time Travel**: Rewinding to earlier steps to debug or fork execution.
4. **Fault Tolerance**: Resuming long-running workflows after server crashes.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph PersistenceArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CLIENT_REQ [label="Incoming Request\\nconfig = {'configurable': {'thread_id': 'session_abc'}}", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_checkpointer {
            label = "Checkpointer Storage Engine (Postgres / SQLite)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            LOAD_SNAPSHOT [label="1. Load Latest Checkpoint\\nSELECT * FROM checkpoints WHERE thread_id = 'session_abc'\\nORDER BY step DESC LIMIT 1", fillcolor="#2E1065", color="#C084FC"];
            COMMIT_SNAPSHOT [label="4. Commit Superstep Snapshot\\nINSERT INTO checkpoints (thread_id, checkpoint_id, state_blob, next_nodes)", fillcolor="#064E3B", color="#34D399"];
        }

        subgraph cluster_graph_runtime {
            label = "LangGraph Execution Engine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            RESTORE_STATE [label="2. Restore In-Memory State\\nChannels populated with previous values + deltas", fillcolor="#0F172A", color="#38BDF8"];
            EXECUTE_NODES [label="3. Execute Active Superstep Nodes\\nNodes return state deltas", fillcolor="#1E1E38", color="#8B5CF6"];

            RESTORE_STATE -> EXECUTE_NODES;
        }

        CLIENT_REQ -> LOAD_SNAPSHOT;
        LOAD_SNAPSHOT -> RESTORE_STATE [label="Hydrates state"];
        EXECUTE_NODES -> COMMIT_SNAPSHOT [label="Emits snapshot"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Guarantee:</b> LangGraph guarantees that every superstep synchronization barrier is atomic.
If a worker fails mid-step, the checkpointer rolls back to the previous snapshot, guaranteeing zero state corruption.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: WHY PERSISTENCE MATTERS
# ==============================================================================
st.header("2. Why Persistence Matters: Durability, Stateful Multi-Turn Agents & Fault Recovery", divider="rainbow")

st.markdown("""
Without persistence, an AI agent is purely **ephemeral**:
- If a user sends a follow-up message 5 minutes later, all context is lost.
- If a cloud server restarts while an agent is executing a 10-minute research loop, the entire job vanishes.
- If an agent performs a destructive action, developers cannot inspect the exact historical state that triggered it.

With persistence enabled:
1. **Stateless APIs Become Stateful**: FastAPI or Next.js endpoints can spin up, load the thread snapshot, execute one turn, commit, and exit.
2. **Crash Resilience**: Failed workflows resume from the exact last successful node.
3. **Compliance Audit Trails**: Every decision, tool call, and response is stored with cryptographic timestamps.
""")

# ==============================================================================
# SECTION 3: CHECKPOINTS
# ==============================================================================
st.header("3. Checkpoints: State Snapshots at Superstep Synchronization Boundaries", divider="rainbow")

st.markdown("""
What is a **checkpoint**?
A checkpoint is an immutable record created at the conclusion of every superstep.
It contains:
- **`v`**: Checkpoint schema version.
- **`ts`**: ISO-8601 timestamp.
- **`id`**: Unique hierarchical checkpoint ID.
- **`channel_values`**: The serialized dictionary of state values.
- **`channel_versions`**: Monotonic version counters for each channel.
""")

st.code("""
# Anatomy of an internal LangGraph checkpoint dictionary:
{
    "v": 1,
    "ts": "2026-10-13T14:15:22.891Z",
    "id": "1ef88219-482a-60b2-8001-9921bc98a001",
    "channel_values": {
        "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")],
        "user_id": "usr_9921",
        "retry_count": 0
    },
    "channel_versions": {
        "messages": 2,
        "user_id": 1,
        "retry_count": 1
    }
}
""", language="python")

# ==============================================================================
# SECTION 4: STATE SNAPSHOTS
# ==============================================================================
st.header("4. State Snapshots: Inspecting Values, Metadata, Next Nodes, and Writes", divider="rainbow")

st.markdown("""
At runtime, you can inspect a thread's current checkpoint using `graph.get_state(config)`.
The returned `StateSnapshot` object exposes:
""")

snapshot_df = pd.DataFrame([
    {
        "Snapshot Field": "values",
        "Type": "Dict[str, Any]",
        "Purpose": "The current state channel values (messages, custom fields, data)."
    },
    {
        "Snapshot Field": "next",
        "Type": "Tuple[str, ...]",
        "Purpose": "The names of the next node(s) scheduled to execute in the next superstep."
    },
    {
        "Snapshot Field": "config",
        "Type": "RunnableConfig",
        "Purpose": "The exact configuration (thread_id, checkpoint_id) corresponding to this snapshot."
    },
    {
        "Snapshot Field": "metadata",
        "Type": "Dict[str, Any]",
        "Purpose": "Operational details: step number, source node, run ID, and timestamps."
    },
    {
        "Snapshot Field": "created_at",
        "Type": "str",
        "Purpose": "Timestamp when this checkpoint was written to storage."
    }
])
st.table(snapshot_df)

# ==============================================================================
# SECTION 5: THREADS
# ==============================================================================
st.header("5. Threads: Logical Conversation Partitions and Multi-User Isolation", divider="rainbow")

st.markdown("""
A **Thread** is a logical sequence of checkpoints representing a single conversation or workflow instance.
Each thread maintains its own isolated timeline.
Two different users accessing the same LangGraph deployment operate on distinct threads with zero cross-contamination.
""")

with st.expander("🔍 View Multi-Thread State Isolation Flow", expanded=True):
    thread_dot = """
    digraph ThreadIsolation {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CHECKPOINT_DB [label="Checkpointer Database Storage", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_t1 {
            label = "Thread 1: thread_id='user_alice'";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#34D399";
            fontcolor = "#34D399";

            T1_S1 [label="Step 1: Hello from Alice", fillcolor="#064E3B", color="#34D399"];
            T1_S2 [label="Step 2: AI reply to Alice", fillcolor="#064E3B", color="#34D399"];
            T1_S1 -> T1_S2;
        }

        subgraph cluster_t2 {
            label = "Thread 2: thread_id='user_bob'";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#C084FC";
            fontcolor = "#C084FC";

            T2_S1 [label="Step 1: Hello from Bob", fillcolor="#2E1065", color="#C084FC"];
            T2_S2 [label="Step 2: AI reply to Bob", fillcolor="#2E1065", color="#C084FC"];
            T2_S1 -> T2_S2;
        }

        CHECKPOINT_DB -> T1_S1 [label="thread_id partition"];
        CHECKPOINT_DB -> T2_S1 [label="thread_id partition"];
    }
    """
    st.graphviz_chart(thread_dot, use_container_width=True)

# ==============================================================================
# SECTION 6: THREAD IDS
# ==============================================================================
st.header("6. Thread IDs: Session Indexing, Tenant Separation, and Routing Keys", divider="rainbow")

st.markdown("""
The `thread_id` is passed via the `config` dictionary on every invocation:
""")

st.code("""
# Standard pattern for session-based invocation
config = {
    "configurable": {
        "thread_id": "customer_support_thread_4920",
        "tenant_id": "acme_corp"
    }
}

# Invoking graph with thread configuration
response = app.invoke(user_input, config=config)
""", language="python")

# ==============================================================================
# SECTION 7: CHECKPOINT LIFECYCLE
# ==============================================================================
st.header("7. Checkpoint Lifecycle: Read, Step Execution, Commit, and Resumption", divider="rainbow")

st.markdown("""
During each superstep, the checkpoint lifecycle follows 4 deterministic phases:
1. **Read**: The engine loads the latest checkpoint matching `thread_id`.
2. **Execute**: Scheduled nodes run in parallel, returning state deltas.
3. **Reconcile**: Channel reducers merge deltas with existing state.
4. **Commit**: A new checkpoint containing updated values and `next` nodes is committed to the database.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Checkpointed Workflow with Snapshot Inspection", divider="rainbow")

st.markdown("""
Below is an executable implementation compiling a multi-turn graph with `MemorySaver`
and demonstrating programmatic snapshot queries:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class SessionChatState(MessagesState):
    session_rating: int

# 2. Checkpointer
checkpointer = MemorySaver()

# 3. Node Definitions
def chatbot_node(state: SessionChatState) -> dict:
    last_msg = state["messages"][-1].content
    response = AIMessage(content=f"Received: '{last_msg}'. How may I assist you further?")
    return {"messages": [response], "session_rating": 5}

# 4. Graph Assembly
builder = StateGraph(SessionChatState)
builder.add_node("bot", chatbot_node)
builder.add_edge(START, "bot")
builder.add_edge("bot", END)

app = builder.compile(checkpointer=checkpointer)

# 5. Multi-Turn Invocations with thread_id
cfg = {"configurable": {"thread_id": "session_881"}}

# Turn 1
app.invoke({"messages": [HumanMessage(content="Hello!")]}, config=cfg)

# Turn 2
app.invoke({"messages": [HumanMessage(content="What is my status?")]}, config=cfg)

# 6. Snapshot Inspection
snapshot = app.get_state(cfg)
print("Latest Checkpoint ID:", snapshot.config["configurable"]["checkpoint_id"])
print("Total Messages in Thread:", len(snapshot.values["messages"]))
print("Session Rating:", snapshot.values["session_rating"])
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Thread Snapshot & Checkpoint Lifecycle Inspector", divider="rainbow")

st.markdown("""
Simulate superstep executions across independent threads.
Observe how checkpoints are committed and inspect values, next nodes, and checkpoint IDs in real time.
""")

if "persist_engine" not in st.session_state:
    st.session_state.persist_engine = PersistenceSimulatorEngine()

p_eng: PersistenceSimulatorEngine = st.session_state.persist_engine

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Superstep Execution")
    active_thread = st.selectbox("Select Thread ID:", ["thread_alpha", "thread_beta", "thread_gamma"])
    executing_node = st.selectbox("Executing Node:", ["agent", "tools", "reviewer"])
    msg_input = st.text_input("State Delta Message Content:", value="User requested account summary")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        commit_btn = st.button("🚀 Commit Superstep", type="primary")
    with col_b2:
        reset_btn = st.button("🧹 Clear Thread History")

    if reset_btn:
        p_eng.threads[active_thread] = []
        st.rerun()

with wb_c2:
    st.subheader("📋 Checkpointer Snapshot Timeline")
    if commit_btn:
        new_snap = p_eng.run_superstep(
            active_thread,
            executing_node,
            {"messages": f"[{executing_node.upper()}]: {msg_input}"}
        )
        st.success(f"Committed Checkpoint [{new_snap['checkpoint_id']}]!")

    snapshots = p_eng.get_thread_snapshots(active_thread)
    st.markdown(f"**Total Checkpoints in `{active_thread}`:** `{len(snapshots)}`")

    if snapshots:
        for snap in reversed(snapshots):
            st.markdown(f"**Step {snap['step']}** `Checkpoint ID: {snap['checkpoint_id']}`", unsafe_allow_html=True)
            st.write(f"- Executed Node: `{snap['node']}`")
            st.write(f"- Next Node(s): `{snap['next']}`")
            st.json(snap["values"])
            st.divider()
    else:
        st.info("No checkpoints committed for this thread yet. Click 'Commit Superstep' to start.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the fundamental difference between an execution superstep and a checkpoint?", expanded=False):
    st.markdown("""
    **Answer**:
    A superstep is an execution cycle where active nodes run concurrently.
    A checkpoint is the persistent snapshot committed to disk/database at the completion of that superstep,
    recording the reconciled state values, monotonic channel versions, and next nodes.
    """)

with st.expander("Q2: Why must every persistent invocation specify a thread_id?", expanded=False):
    st.markdown("""
    **Answer**:
    `thread_id` is the primary partitioning key in checkpointer storage.
    Without `thread_id`, the checkpointer cannot determine which conversation history to load,
    rendering stateful multi-turn interactions impossible.
    """)

with st.expander("Q3: What data structures does get_state() return?", expanded=False):
    st.markdown("""
    **Answer**:
    `get_state()` returns a `StateSnapshot` object containing:
    - `values`: The channel dictionary.
    - `next`: Nodes scheduled for the next step.
    - `config`: Thread and checkpoint ID keys.
    - `metadata`: Step number and timestamp details.
    """)

with st.expander("Q4: Can a single thread be executed concurrently by two different workers?", expanded=False):
    st.markdown("""
    **Answer**:
    No. Checkpointers enforce optimistic locking or database row locks on `thread_id` to prevent concurrent write collisions.
    Concurrent requests on the same thread should be queued.
    """)

with st.expander("Q5: What is Time Travel in LangGraph and how does checkpointing enable it?", expanded=False):
    st.markdown("""
    **Answer**:
    Because checkpointers store every step immutably, developers can pass a historical `checkpoint_id` in `config`
    to load that exact historical state snapshot and fork a new timeline from that point.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing thread_id in Production API Calls")
    st.markdown("""
    *Mechanism*: Invoking `app.invoke(input)` without passing `{"configurable": {"thread_id": "..."}}`.  
    *Consequence*: The checkpointer does not persist state; graph behaves as an ephemeral single-turn script.  
    *Fix*: Always supply a unique `thread_id` for every user session.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Deploying MemorySaver in Scaled Production Services")
    st.markdown("""
    *Mechanism*: Using `MemorySaver` in a multi-pod Kubernetes deployment behind a load balancer.  
    *Consequence*: User requests hit different pods with separate RAM stores, causing immediate amnesia.  
    *Fix*: Use centralized database checkpointers (`PostgresSaver`).
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Unindexed Checkpoint Database Tables")
    st.markdown("""
    *Mechanism*: Failing to index `thread_id` and `checkpoint_id` in relational storage.  
    *Consequence*: State lookups degrade from 5ms to multiple seconds as table size grows.  
    *Fix*: Use official migration scripts via `PostgresSaver.setup()`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Storing Non-Serializable Objects in State")
    st.markdown("""
    *Mechanism*: Placing socket connections or database client instances directly into state channels.  
    *Consequence*: Checkpointer serialization fails with `TypeError`.  
    *Fix*: Keep state limited to JSON-serializable primitives and message classes.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Checkpoint**\\nAn immutable snapshot of graph channel values saved at every superstep boundary.")
with col_fc2:
    st.info("**thread_id**\\nThe unique partition key indexing checkpoints for a specific conversation session.")
with col_fc3:
    st.info("**StateSnapshot**\\nThe object returned by get_state() exposing values, next nodes, and metadata.")
with col_fc4:
    st.info("**Time Travel**\\nThe ability to replay or fork execution from any historical checkpoint snapshot.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know why persistence is mandatory for stateful multi-turn agents.", value=True)
c2 = st.checkbox("I understand what data is stored inside a checkpoint snapshot.", value=True)
c3 = st.checkbox("I know how thread_id isolates conversations across multiple users.", value=True)
c4 = st.checkbox("I can inspect historical thread snapshots using get_state().", value=True)
c5 = st.checkbox("I understand why MemorySaver cannot be used in scaled multi-container environments.", value=True)
c6 = st.checkbox("I know how the 4-phase checkpoint lifecycle (Read, Execute, Reconcile, Commit) works.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 14.1 'What is Persistence' with full architectural and practical comprehension!")
