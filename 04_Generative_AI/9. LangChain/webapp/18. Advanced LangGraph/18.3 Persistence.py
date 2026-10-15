import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class StoredCheckpointRecord:
    """Represents an atomic serialized checkpoint state snapshot."""
    def __init__(
        self,
        checkpoint_id: str,
        superstep: int,
        node_name: str,
        state_data: Dict[str, Any],
        parent_id: Optional[str]
    ):
        self.checkpoint_id = checkpoint_id
        self.superstep = superstep
        self.node_name = node_name
        self.state_data = state_data
        self.parent_id = parent_id

class MockPersistenceStudio:
    """
    Simulates LangGraph thread persistence, superstep checkpointing,
    fault-tolerant resumption, and time-travel rewinding.
    """
    def __init__(self):
        self.threads: Dict[str, List[StoredCheckpointRecord]] = {
            "thread-alpha-101": [
                StoredCheckpointRecord(
                    "chk-001", 1, "START",
                    {"messages": ["User: Run DB migration"], "status": "init", "step": 1},
                    None
                ),
                StoredCheckpointRecord(
                    "chk-002", 2, "planner_node",
                    {"messages": ["User: Run DB migration", "Planner: Generated 3 schema changes"], "status": "planned", "step": 2},
                    "chk-001"
                ),
                StoredCheckpointRecord(
                    "chk-003", 3, "human_approval_gate",
                    {"messages": ["User: Run DB migration", "Planner: Generated 3 schema changes", "Gate: Paused for DBA approval"], "status": "paused", "step": 3},
                    "chk-002"
                )
            ]
        }

    def resume_thread(self, thread_id: str, approve_action: bool) -> Dict[str, Any]:
        history = self.threads.get(thread_id, [])
        if not history:
            return {"error": "Thread not found"}

        latest = history[-1]
        if latest.node_name != "human_approval_gate":
            return {"status": "Thread already completed or not paused"}

        if approve_action:
            new_chk = StoredCheckpointRecord(
                f"chk-00{len(history)+1}", len(history)+1, "migration_executor",
                {
                    "messages": latest.state_data["messages"] + ["DBA: Approved", "Executor: Schema migration applied successfully."],
                    "status": "completed",
                    "step": len(history)+1
                },
                latest.checkpoint_id
            )
            history.append(new_chk)
            return {"status": "Resumed & Completed", "latest_checkpoint": new_chk}
        else:
            new_chk = StoredCheckpointRecord(
                f"chk-00{len(history)+1}", len(history)+1, "rollback_node",
                {
                    "messages": latest.state_data["messages"] + ["DBA: Rejected", "Executor: Migration aborted. Zero changes made."],
                    "status": "aborted",
                    "step": len(history)+1
                },
                latest.checkpoint_id
            )
            history.append(new_chk)
            return {"status": "Resumed & Aborted", "latest_checkpoint": new_chk}

    def time_travel_rewind(self, thread_id: str, target_checkpoint_id: str) -> Dict[str, Any]:
        history = self.threads.get(thread_id, [])
        for rec in history:
            if rec.checkpoint_id == target_checkpoint_id:
                # Fork thread state from this historical checkpoint
                forked_chk = StoredCheckpointRecord(
                    f"chk-fork-{rec.superstep}", rec.superstep, f"forked_from_{rec.node_name}",
                    dict(rec.state_data),
                    rec.checkpoint_id
                )
                return {
                    "restored_checkpoint": rec,
                    "forked_state": forked_chk
                }
        return {"error": "Checkpoint ID not found"}


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="18.3 Persistence — LangChain Course",
    page_icon="💾",
    layout="wide"
)

st.title("18.3 Persistence")
st.caption("Deep-Dive into LangGraph Checkpointers: Thread Isolation, State Persistence, Crash Recovery, and Time-Travel Debugging")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: Checkpointing, Threads & Time-Travel Engine", divider="rainbow")
st.markdown("""
In real-world enterprise deployments, agents cannot run purely in ephemeral RAM. Network sockets disconnect, 
containers restart, and human stakeholders require days to review privileged actions.

**LangGraph Persistence** serializes graph state at every superstep using **Checkpointers**. This architecture unlocks:
- **Thread Isolation**: Multiple concurrent users maintain isolated state partitions.
- **Crash Recovery**: Failed workflows resume from the exact last saved superstep.
- **Time Travel**: Developers can rewind execution to any prior checkpoint, inspect state, and fork alternative outcomes.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph PersistenceArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_execution {
            label = "Step 1: Graph Execution Supersteps";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            step1 [label="Superstep 1: Intake Node\\nInput State -> Updates", fillcolor="#0F172A", color="#38BDF8"];
            step2 [label="Superstep 2: Planner Node\\nGenerates plan", fillcolor="#0F172A", color="#38BDF8"];
            step3 [label="Superstep 3: Approval Breakpoint\\nExecution halts for human DBA", fillcolor="#78350F", color="#F59E0B"];
        }

        subgraph cluster_database {
            label = "Step 2: Persistent Checkpointer (PostgresSaver / SqliteSaver)";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            
            chk_table [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#0369A1" colspan="4"><b>Checkpoints Table (partition by thread_id)</b></td></tr>
                <tr><td bgcolor="#1E293B"><b>thread_id</b></td><td bgcolor="#1E293B"><b>checkpoint_id</b></td><td bgcolor="#1E293B"><b>parent_id</b></td><td bgcolor="#1E293B"><b>channel_values</b></td></tr>
                <tr><td bgcolor="#1E293B">thread-101</td><td bgcolor="#1E293B">chk-001</td><td bgcolor="#1E293B">null</td><td bgcolor="#1E293B">{'step': 1}</td></tr>
                <tr><td bgcolor="#1E293B">thread-101</td><td bgcolor="#1E293B">chk-002</td><td bgcolor="#1E293B">chk-001</td><td bgcolor="#1E293B">{'step': 2}</td></tr>
                <tr><td bgcolor="#1E293B">thread-101</td><td bgcolor="#1E293B">chk-003</td><td bgcolor="#1E293B">chk-002</td><td bgcolor="#1E293B">{'step': 3}</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_timetravel {
            label = "Step 3: Time-Travel & Resumption";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            rewind_node [label="Time-Travel Inspection\\napp.get_state_history(thread_id)\\nRewind to chk-002 and fork alternative plan", fillcolor="#064E3B", color="#10B981"];
            resume_node [label="Resume from Breakpoint\\napp.invoke(None, config)\\nResumes execution from chk-003", fillcolor="#022C22", color="#059669"];
        }

        step1 -> chk_table [label="commit snapshot 1", style="dotted"];
        step1 -> step2;
        step2 -> chk_table [label="commit snapshot 2", style="dotted"];
        step2 -> step3;
        step3 -> chk_table [label="commit snapshot 3", style="dotted"];

        chk_table -> rewind_node [label="load historical chk-002"];
        chk_table -> resume_node [label="load latest chk-003"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: CHECKPOINTS ---
st.header("2. Anatomy of a Checkpoint", divider="rainbow")
st.markdown("""
A **Checkpoint** is an immutable, point-in-time snapshot of the entire graph state saved at the conclusion of a superstep.

### Key Checkpoint Attributes:
- **`thread_id`**: The partition key scoping all checkpoints belonging to a particular user session or task.
- **`checkpoint_id`**: A unique UUID identifying this exact superstep snapshot.
- **`parent_checkpoint_id`**: Pointer to the immediately preceding snapshot, forming an immutable directed lineage tree.
- **`channel_values`**: The serialized dictionary containing all state channels at this superstep.
- **`versions_seen`**: Monotonic counter mapping channel modifications.
""")

st.divider()

# --- SECTION 3: THREADS ---
st.header("3. Threads: Session Isolation & Partitioning", divider="rainbow")
st.markdown("""
LangGraph uses **Threads** to isolate state across independent conversations and tasks. 
Every invocation requires passing a `configurable` dictionary specifying the target `thread_id`:
""")

st.code("""
# Thread configuration
config_user_a = {"configurable": {"thread_id": "session_user_alice_42"}}
config_user_b = {"configurable": {"thread_id": "session_user_bob_99"}}

# User Alice's message is persisted strictly to Alice's thread
app.invoke({"messages": [HumanMessage(content="Hello")]}, config=config_user_a)

# Bob's graph has an isolated state with zero access to Alice's messages
app.invoke({"messages": [HumanMessage(content="Hi")]}, config=config_user_b)
""", language="python")

st.divider()

# --- SECTION 4: STATE PERSISTENCE ---
st.header("4. State Persistence Backends", divider="rainbow")
st.markdown("""
LangGraph provides pluggable checkpointer backends tailored for different environments:
""")

backend_df = pd.DataFrame({
    "Checkpointer Class": ["MemorySaver", "SqliteSaver", "PostgresSaver"],
    "Storage Location": ["In-Memory (Python RAM dictionary)", "Local Disk (SQLite database file)", "Production PostgreSQL Server"],
    "Durability": ["Ephemeral: Lost when process exits", "Persistent: Survives process restarts on single host", "Enterprise Grade: High-availability, clustering, multi-region"],
    "Best For": ["Unit tests, local prototyping", "Desktop tools, embedded CLI apps", "Production microservices, Kubernetes clusters, SaaS APIs"]
})
st.dataframe(backend_df, use_container_width=True)

st.divider()

# --- SECTION 5: RESUME EXECUTION ---
st.header("5. Resume Execution from Breakpoints", divider="rainbow")
st.markdown("""
When a graph hits an interrupt (configured via `interrupt_before` or `interrupt_after`), execution pauses cleanly 
and commits the state. 

To **resume execution**, invoke the graph with **`None` as input** while providing the identical `thread_id` config. 
LangGraph loads the latest checkpoint from disk and continues execution!
""")

st.code("""
# 1. Initial execution halts before 'delete_database'
app.invoke({"query": "Drop legacy tables"}, config=config)

# 2. Human reviews action out-of-band and approves...

# 3. Resume from breakpoint: pass None as input!
app.invoke(None, config=config)
""", language="python")

st.divider()

# --- SECTION 6: FAULT TOLERANCE ---
st.header("6. Fault Tolerance & Automatic Recovery", divider="rainbow")
st.markdown("""
In distributed cloud architectures, node workers and pods crash unexpectedly due to OOM errors, pod evictions, 
or network disconnects.

With checkpointing enabled:
1. When a worker pod dies during superstep 4, supersteps 1, 2, and 3 remain securely committed in PostgreSQL.
2. A new replacement worker pod re-attaches to the thread ID.
3. LangGraph reads the latest valid checkpoint and resumes from the start of superstep 4 without re-running 
   supersteps 1-3, saving compute time and third-party API costs.
""")

st.divider()

# --- SECTION 7: TIME TRAVEL ---
st.header("7. Time Travel: State History Inspection & Rewinding", divider="rainbow")
st.markdown("""
**Time Travel** is one of LangGraph's most extraordinary superpowers. Developers and operators can inspect every 
prior checkpoint in a thread's history and rewind execution to an earlier state.
""")

col_tt1, col_tt2 = st.columns(2)
with col_tt1:
    st.subheader("Inspecting State Lineage")
    st.code("""
# List all historical checkpoints for a thread
for state_snapshot in app.get_state_history(config):
    print("Checkpoint ID:", state_snapshot.config["configurable"]["checkpoint_id"])
    print("Node that ran:", state_snapshot.metadata["source"])
    print("State content:", state_snapshot.values)
    """, language="python")

with col_tt2:
    st.subheader("Rewinding & Forking State")
    st.code("""
# Target an earlier checkpoint ID
rewind_config = {
    "configurable": {
        "thread_id": "session_user_alice_42",
        "checkpoint_id": "historical-checkpoint-uuid-002"
    }
}

# Resume from the past to explore an alternate branch!
app.invoke(
    {"messages": [HumanMessage(content="Let's try a different approach")]},
    config=rewind_config
)
    """, language="python")

st.divider()

# --- SECTION 8: PERSISTENT CONVERSATIONS ---
st.header("8. Persistent Multi-Turn Conversations", divider="rainbow")
st.markdown("""
Combining `add_messages` with a persistent checkpointer gives conversational agents **indefinite memory**. 
A user can chat with an agent on Monday, close their browser, return on Friday, and the agent retains full 
conversational context without needing external memory injection code.
""")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise PostgresSaver Checkpointing Engine", divider="rainbow")
st.markdown("""
Below is the production code to configure LangGraph with a persistent PostgreSQL checkpointer in an enterprise environment.
""")

st.code("""
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# 1. State Definition
class ProductionChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 2. Node Function
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot_node(state: ProductionChatState):
    return {"messages": [model.invoke(state["messages"])]}

# 3. Graph Assembly
builder = StateGraph(ProductionChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 4. Production PostgreSQL Connection Pool & Checkpointer
DB_URI = "postgresql://postgres:secret@db.internal:5432/langgraph_db"

def initialize_production_graph():
    # Setup persistent connection pool
    connection_pool = ConnectionPool(conninfo=DB_URI, max_size=20)
    checkpointer = PostgresSaver(connection_pool)
    checkpointer.setup()  # Creates checkpoints tables on first startup
    
    # Compile graph with persistent Postgres checkpointer
    app = builder.compile(checkpointer=checkpointer)
    return app

# Example Invocation:
# app = initialize_production_graph()
# config = {"configurable": {"thread_id": "tenant-99_user-412"}}
# response = app.invoke({"messages": [HumanMessage(content="Hello!")]}, config=config)
""", language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Persistence & Time-Travel Simulator", divider="rainbow")
st.markdown("""
Inspect thread checkpoints stored in our simulated checkpointer. Resume a paused thread across an approval gate, 
or use **Time Travel** to rewind execution to a historical superstep and inspect the exact state.
""")

# Instantiate module-scope studio
pers_studio = MockPersistenceStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    thread_select = st.selectbox("Select Active Thread:", ["thread-alpha-101"])
    current_history = pers_studio.threads[thread_select]

    st.markdown(f"**Current Checkpoints in Thread ({len(current_history)})**:")
    for chk in current_history:
        st.write(f"- `{chk.checkpoint_id}` (Superstep {chk.superstep}): Node `[{chk.node_name}]`")

    st.markdown("### Action Controls")
    resume_approval_toggle = st.toggle("DBA Approval Action (True = Approve, False = Abort)", value=True)
    resume_btn = st.button("🚀 Resume Paused Thread Execution", type="primary", use_container_width=True)

    st.markdown("### Time-Travel Controls")
    target_chk_id = st.selectbox("Select Historical Checkpoint to Rewind:", [c.checkpoint_id for c in current_history])
    rewind_btn = st.button("⏪ Rewind & Fork State (Time-Travel)", use_container_width=True)

with wb_col2:
    st.subheader("Checkpoint Telemetry & Time-Travel Trace")

    if resume_btn:
        res_out = pers_studio.resume_thread(thread_select, resume_approval_toggle)
        st.success(f"Execution Status: {res_out['status']}")
        if "latest_checkpoint" in res_out:
            st.json(res_out["latest_checkpoint"].state_data)

    elif rewind_btn:
        rewind_out = pers_studio.time_travel_rewind(thread_select, target_chk_id)
        st.info(f"Rewound to Checkpoint: `{target_chk_id}`")
        st.write("**Restored Historical State Snapshot**:")
        st.json(rewind_out["restored_checkpoint"].state_data)
        st.caption(f"**Parent Checkpoint Lineage**: `{rewind_out['restored_checkpoint'].parent_id}`")

    else:
        st.write("Select an action to inspect persistence checkpoints.")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does LangGraph track checkpoint lineage during time-travel branching?",
     "Each checkpoint stores a `checkpoint_id`, a `parent_checkpoint_id`, and a `thread_id`. When an agent executes normally, each new checkpoint points to the previous one in a linear chain. When a developer rewinds to an earlier checkpoint and invokes new actions, LangGraph creates a new checkpoint whose parent_id points back to the historical checkpoint, creating a branching DAG tree analogous to a Git commit tree."),
    ("Why should teams use `PostgresSaver` instead of `SqliteSaver` in production Kubernetes environments?",
     "Kubernetes pods are ephemeral: if a pod crashes or auto-scales horizontally, local SQLite files stored in the container filesystem are wiped or inaccessible to other replica pods. `PostgresSaver` connects to a centralized, managed PostgreSQL cluster (e.g. AWS RDS or Cloud SQL), guaranteeing that all API pods share the exact same checkpoint state across horizontal scale-out."),
    ("What is the difference between `interrupt_before` and `interrupt_after`?",
     "`interrupt_before=['node_a']` pauses execution immediately before `node_a` runs, allowing humans to inspect or edit the inputs to `node_a`. `interrupt_after=['node_a']` pauses execution immediately after `node_a` completes, allowing humans to inspect, validate, or reject the output produced by `node_a` before any downstream nodes consume it."),
    ("How does `app.invoke(None, config)` work under the hood when resuming an interrupted graph?",
     "Passing `None` signals to LangGraph that no new user input is being introduced. LangGraph queries the checkpointer using `thread_id`, retrieves the latest checkpoint snapshot, inspects the next scheduled nodes from the graph topology, and resumes execution seamlessly from the exact breakpoint without re-executing completed nodes."),
    ("What are the storage costs and retention considerations for LangGraph checkpoints in high-traffic applications?",
     "Because checkpointers record state snapshots at every superstep, a 10-step agent conversation with large message contexts generates 10 complete state snapshots. In high-traffic SaaS systems, this causes database table bloat. Production systems must implement checkpoint retention policies (e.g. TTL pruning, soft-deleting intermediate checkpoints while retaining the final checkpoint).")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Non-Picklable or Non-JSON-Serializable Objects in State. Checkpointers serialize state to disk. If your state contains unpicklable Python objects (such as open file handles, threading locks, or active database connections), the checkpointer will crash with a SerializationError.")
    st.error("**Trap 2 (Breaking Error)**: Reusing the Same `thread_id` for Different Users. If two distinct end-users share the same `thread_id`, their conversations will collide in the checkpointer, leaking user messages across tenant boundaries. Always generate cryptographically secure, unique thread IDs.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Forgetting Database Indexing on `thread_id` and `checkpoint_id`. In PostgresSaver, querying checkpoints across millions of rows without composite B-tree indexes on `(thread_id, checkpoint_id)` causes slow full-table scans that throttle agent response times.")
    st.warning("**Trap 4 (Quality Warning)**: Editing Checkpoints Without Updating Child Pointers. Modifying historical checkpoints directly via raw SQL queries rather than using `app.update_state()` corrupts checkpoint parent lineage trees, breaking time-travel navigation.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Checkpoint"):
        st.markdown("**Immutable snapshot of all state channels saved per superstep with parent lineage pointers.**")
    with st.expander("🎴 Flashcard 2: thread_id"):
        st.markdown("**Primary partition key scoping persistent state and conversation history per user session.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: PostgresSaver"):
        st.markdown("**Enterprise production checkpointer storing state in PostgreSQL for high availability.**")
    with st.expander("🎴 Flashcard 4: Time Travel"):
        st.markdown("**Rewinding thread execution to historical checkpoints to inspect state or fork alternate paths.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the anatomy of a Checkpoint: checkpoint_id, thread_id, and parent lineage.", value=True)
chk2 = st.checkbox("I know why thread_id partitions state across multi-tenant sessions.", value=True)
chk3 = st.checkbox("I can differentiate between MemorySaver, SqliteSaver, and PostgresSaver checkpointers.", value=True)
chk4 = st.checkbox("I understand how to resume interrupted graphs by invoking with None and thread config.", value=True)
chk5 = st.checkbox("I can explain how time travel rewinds state and forks historical execution branches.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 persistence technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 18.3 Persistence!")
