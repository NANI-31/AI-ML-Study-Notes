import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="36.2 Advanced State Management - LangChain Curriculum",
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
    .state-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-check {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AdvancedStateManagerEngine:
    """
    Simulates LangGraph state channels, reducer mechanics (add_messages, custom merge),
    checkpoint DAG history, and time-travel rollbacks.
    """
    def __init__(self):
        self.checkpoints = [
            {"step": 1, "node": "START", "messages_count": 1, "active_key": "raw_input", "checkpoint_id": "chk_001_init"},
            {"step": 2, "node": "classifier", "messages_count": 1, "active_key": "intent='billing'", "checkpoint_id": "chk_002_class"},
            {"step": 3, "node": "tool_runner", "messages_count": 2, "active_key": "tool_result=OK", "checkpoint_id": "chk_003_tool"},
            {"step": 4, "node": "synthesizer", "messages_count": 3, "active_key": "output_ready", "checkpoint_id": "chk_004_done"}
        ]

    def simulate_reducer(self, current_list: List[str], incoming_delta: List[str], deduplicate: bool = True) -> List[str]:
        if not deduplicate:
            return current_list + incoming_delta
        # Deduplicate while preserving order
        seen = set(current_list)
        result = list(current_list)
        for item in incoming_delta:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def get_checkpoint(self, step: int) -> Dict[str, Any]:
        for c in self.checkpoints:
            if c["step"] == step:
                return c
        return self.checkpoints[-1]

# Instantiate engine
state_engine = AdvancedStateManagerEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Cyclic State Graph Transitions, Channel Reducers, & Thread Checkpointing", divider="rainbow")

st.markdown("""
In stateful generative AI workflows, state is not a passive global dictionary. 
In modern LangGraph, state is composed of **type-safe channels** with explicit **reducer functions** 
that govern how node outputs update state. After every node transition, a **durable checkpointer** 
records an immutable snapshot to disk, enabling auditability, crash fault-tolerance, and time-travel debugging.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph State_Management_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_state_def {
            label = "Type-Safe State Schema & Reducer Channel Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            schema_def [label="AgentState (TypedDict / Pydantic)\\n(Defines typed channels & validation rules)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            channel_messages [label="Channel: messages\\n(Annotated with add_messages reducer)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            channel_custom [label="Channel: aggregate_metrics\\n(Custom operator.add / merge function)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_execution {
            label = "Node Execution & Delta Return";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            node_worker [label="Graph Node Function\\n(Reads current state, returns only key delta dict)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            reducer_runtime [label="Channel Reducer Application\\n(Applies state[key] = reducer(state[key], delta[key]))", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_checkpointer {
            label = "Durable Checkpoint Persistence (PostgreSQL)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            checkpointer_engine [label="PostgresSaver Checkpointer\\n(Serializes state snapshot + parent checkpoint pointer)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            thread_table [label="Database Thread Ledger\\n(Indexed by thread_id + checkpoint_ns + step)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        schema_def -> channel_messages;
        schema_def -> channel_custom;

        channel_messages -> node_worker;
        channel_custom -> node_worker;

        node_worker -> reducer_runtime [label="returns {'messages': [AIMessage]}"];
        reducer_runtime -> checkpointer_engine [label="commits validated state"];

        checkpointer_engine -> thread_table;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Channel Reducer Invariant:</strong> Nodes in a StateGraph do not mutate state in place. 
    Instead, nodes return a partial delta dictionary. The runtime channels pass this delta through their registered 
    reducer function (such as <code>add_messages</code>) to produce the next immutable state snapshot.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: CUSTOM STATE
# ==============================================================================
st.header("2. Custom State: TypedDict Schemas and Pydantic Invariant Validation", divider="rainbow")

st.markdown("""
In LangGraph, state is explicitly defined as a Python class—typically a `TypedDict` or Pydantic model. 
This provides full static type-checking and code autocompletion.
""")

st.code('''from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class EnterpriseAgentState(TypedDict):
    # Channel with automatic list concatenation & ID-based message deduplication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Overwrite channel (last write wins)
    user_tenant_id: str
    is_authenticated: bool
    retry_count: int
    
    # Optional scratchpad for sub-nodes
    search_citations: Optional[List[dict]]
''', language="python")

# ==============================================================================
# SECTION 3: STATE REDUCERS
# ==============================================================================
st.header("3. State Reducers: Annotated Channels and add_messages Mechanics", divider="rainbow")

st.markdown("""
By default, if a channel is defined without an `Annotated` reducer, **last write wins**: 
any node returning that key completely overwrites the previous value. 
**Reducers** define how incoming deltas combine with existing state.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Default Behavior (Overwrite)
    ```python
    class SimpleState(TypedDict):
        counter: int
    ```
    - If Node A returns `{"counter": 5}`, the counter becomes 5.
    - If Node B returns `{"counter": 1}`, the counter is overwritten to 1.
    """)

with col2:
    st.markdown("""
    ### Custom Reducer (`operator.add`)
    ```python
    import operator
    class AccumulatedState(TypedDict):
        counter: Annotated[int, operator.add]
    ```
    - If Node A returns `{"counter": 5}`, counter is 5.
    - If Node B returns `{"counter": 1}`, counter becomes `5 + 1 = 6`!
    """)

st.markdown("""
### The `add_messages` Reducer
The built-in `add_messages` reducer handles chat messages intelligently:
1. **Append**: New messages are appended to the existing conversation list.
2. **Update by ID**: If an incoming message has an ID matching an existing message, it **replaces** that specific message in place rather than duplicating it.
3. **Delete by ID**: Passing a `RemoveMessage(id="msg_123")` removes that message from state.
""")

# ==============================================================================
# SECTION 4: PERSISTENT STATE
# ==============================================================================
st.header("4. Persistent State: In-Memory vs PostgreSQL / SQLite Checkpointing", divider="rainbow")

st.markdown("""
Production agents require persistence across process restarts. 
LangGraph decouples graph logic from storage through the checkpointer interface:
""")

check_df = pd.DataFrame([
    {"Checkpointer": "MemorySaver", "Durability": "Process RAM only", "Performance": "Microsecond latency", "Production Use": "Prohibited (lost on pod restart)"},
    {"Checkpointer": "SqliteSaver", "Durability": "Local disk file", "Performance": "Fast local I/O", "Production Use": "Edge devices & single-instance daemons"},
    {"Checkpointer": "PostgresSaver", "Durability": "Relational database", "Performance": "Scalable pool (~5ms)", "Production Use": "Enterprise Production Standard"}
])
st.dataframe(check_df, use_container_width=True)

# ==============================================================================
# SECTION 5: THREAD MANAGEMENT
# ==============================================================================
st.header("5. Thread Management: Multi-Tenant Partitioning and State Forking", divider="rainbow")

st.markdown("""
A single compiled LangGraph instance serves millions of users concurrently by partitioning state into **Threads**. 
A thread is referenced by passing a `configurable` dictionary during invocation:
""")

st.code('''# Invoking the graph on Thread A
config_user_a = {"configurable": {"thread_id": "session_user_alice_001"}}
# graph.invoke({"messages": [("user", "Hello!")]}, config=config_user_a)

# Invoking the graph on Thread B (completely isolated state)
config_user_b = {"configurable": {"thread_id": "session_user_bob_002"}}
# graph.invoke({"messages": [("user", "What is my balance?")]}, config=config_user_b)
''', language="python")

# ==============================================================================
# SECTION 6: CHECKPOINTS
# ==============================================================================
st.header("6. Checkpoints: Step-Level Snapshots and Time-Travel Replays", divider="rainbow")

st.markdown("""
Every time a graph node finishes execution, LangGraph creates a **checkpoint**. 
Each checkpoint stores:
1. The full state dictionary at that moment.
2. The `checkpoint_id` (a monotonically increasing hash or timestamp).
3. The `parent_checkpoint_id` pointing to the previous step.
4. The next pending node or tool call to execute.
""")

st.markdown("""
### Time-Travel Debugging
Because checkpoints form a Directed Acyclic Graph (DAG) of history, developers can 
**fork or replay execution from any past checkpoint** (`graph.update_state(config, {"messages": [...]})`), 
testing alternate prompts or fixing erroneous tool outputs retroactively.
""")

# ==============================================================================
# SECTION 7: STATE RECOVERY
# ==============================================================================
st.header("7. State Recovery: Resuming Interrupted Workflows and Crash Fault-Tolerance", divider="rainbow")

st.markdown("""
If an infrastructure node experiences a crash or Kubernetes evicts a pod during an agent run:
""")

st.markdown("""
1. The pod terminates mid-execution.
2. Kubernetes restarts the container or schedules a new pod.
3. The client reconnects using the original `thread_id`.
4. The server loads the **most recent durable checkpoint** committed to PostgreSQL before the crash.
5. The agent resumes execution seamlessly from the failed step boundary with **zero data loss**.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full State Management Pipeline with Custom Reducers", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**state management engine with channel reducers and persistent snapshot history**.
""")

st.code('''from typing import Dict, Any, List

class ProductionChannelStateManager:
    def __init__(self):
        # Database ledger of checkpoints: thread_id -> list of snapshots
        self.checkpoints_db: Dict[str, List[dict]] = {}

    def apply_reducer(self, current_list: List[str], delta_list: List[str]) -> List[str]:
        """Custom message reducer: appends items with ID-based deduplication."""
        seen = set(current_list)
        merged = list(current_list)
        for item in delta_list:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

    def commit_checkpoint(self, thread_id: str, step_name: str, new_messages: List[str]) -> dict:
        history = self.checkpoints_db.setdefault(thread_id, [])
        
        # Load last state
        last_state = history[-1]["state"] if history else {"messages": []}
        
        # Apply reducer
        updated_messages = self.apply_reducer(last_state["messages"], new_messages)
        new_state = {"messages": updated_messages}
        
        # Create immutable checkpoint record
        checkpoint_entry = {
            "step_index": len(history) + 1,
            "step_name": step_name,
            "state": new_state,
            "checkpoint_id": f"chk_{thread_id}_{len(history) + 1}"
        }
        history.append(checkpoint_entry)
        return checkpoint_entry

    def rollback_to_step(self, thread_id: str, step_index: int) -> dict:
        """Time-travel rollback: restores state to a designated step index."""
        history = self.checkpoints_db.get(thread_id, [])
        target = next((c for c in history if c["step_index"] == step_index), None)
        if not target:
            raise ValueError("Checkpoint step not found")
        return target

state_mgr = ProductionChannelStateManager()
# Turn 1
c1 = state_mgr.commit_checkpoint("th_user_404", "ingress", ["User: Need billing status"])
# Turn 2
c2 = state_mgr.commit_checkpoint("th_user_404", "tool_runner", ["Tool: Invoice #991 is PAID"])

print("Current Checkpoint State:", c2["state"])
# Rollback to Turn 1
c_restored = state_mgr.rollback_to_step("th_user_404", 1)
print("Rolled Back State:", c_restored["state"])
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Custom Channel Reducer Simulator & Time-Travel Stepper", divider="rainbow")

st.markdown("""
Experiment with custom channel reducer merge behaviors and step through 
checkpoint history snapshots to inspect time-travel state transitions.
""")

wb_view = st.selectbox(
    "Select State Management Tool:",
    ["1. Custom Channel Reducer Sandbox", "2. Checkpoint Time-Travel Step Inspector", "3. Thread Partitioning Simulator"]
)

if wb_view == "1. Custom Channel Reducer Sandbox":
    st.subheader("Channel Reducer Sandbox")
    st.markdown("Current Channel State: `['msg_1: Hello', 'msg_2: What is my plan?']`")

    incoming_msg = st.text_input("Incoming Delta Message to Append:", "msg_3: Your plan is Enterprise.")
    dedup_toggle = st.toggle("Enable ID-Based Deduplication", value=True)

    base_list = ["msg_1: Hello", "msg_2: What is my plan?"]
    reduced_list = state_engine.simulate_reducer(base_list, [incoming_msg], dedup_toggle)

    st.markdown("#### State Post-Reducer Application:")
    st.code(str(reduced_list), language="python")
    st.success(f"Channel successfully updated. Total messages in state: {len(reduced_list)}")

elif wb_view == "2. Checkpoint Time-Travel Step Inspector":
    st.subheader("Checkpoint Time-Travel Step Inspector")
    step_choice = st.slider("Step Through Historical Checkpoints:", 1, 4, 3, 1)

    chk_data = state_engine.get_checkpoint(step_choice)
    st.markdown(f"**Viewing Checkpoint ID:** `{chk_data['checkpoint_id']}`")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Graph Node Executed", chk_data["node"])
    with m2:
        st.metric("Messages in State", chk_data["messages_count"])

    st.info(f"**Active Channel Value:** `{chk_data['active_key']}`")
    st.caption("Time-traveling allows restoring and replaying execution from any historical step.")

else:
    st.subheader("Multi-Tenant Thread Isolation Simulator")
    st.markdown("Thread State Partitioning in PostgreSQL:")
    st.info("""
    - `thread_alice`: State = {messages: 4, auth: True, tenant: 'corp_alpha'}
    - `thread_bob`:   State = {messages: 2, auth: False, tenant: 'corp_beta'}
    - `thread_carol`: State = {messages: 8, auth: True, tenant: 'corp_alpha'}
    """)
    st.success("All threads completely isolated. Execution across threads is concurrency-safe and zero-leak.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is a channel reducer in LangGraph, and why is it necessary?"):
    st.markdown("""
    **Answer:**  
    A channel reducer defines the aggregation mathematical function applied whenever a graph node returns an update 
    to a state channel. Without reducers, returning a key simply overwrites the previous value. Reducers (like `operator.add` 
    or `add_messages`) allow appending lists, merging dictionaries, or performing conflict resolution across parallel nodes.
    """)

with st.expander("Q2: How does the add_messages reducer handle message updates and deletions?"):
    st.markdown("""
    **Answer:**  
    `add_messages` inspects each incoming message's unique `id`. If no matching ID exists, it appends the message. 
    If an existing message shares the same ID, it replaces the existing message in place. If a `RemoveMessage(id=...)` 
    is received, it deletes that message from the list, enabling selective context pruning without resetting conversation state.
    """)

with st.expander("Q3: Why is MemorySaver strictly unsuitable for production multi-worker deployments?"):
    st.markdown("""
    **Answer:**  
    `MemorySaver` stores state in the local Python process heap memory. In production, requests are load-balanced across 
    multiple container instances. If Turn 1 hits Worker A and Turn 2 hits Worker B, Worker B has no access to Worker A's 
    memory, causing state loss. Furthermore, pod restarts destroy all active threads. Production requires `PostgresSaver`.
    """)

with st.expander("Q4: How does thread configuration partition state across multiple users?"):
    st.markdown("""
    **Answer:**  
    LangGraph indexes all checkpoints by `(thread_id, checkpoint_ns, checkpoint_id)`. By passing 
    `{"configurable": {"thread_id": "usr_998"}}` in the run config, the graph runtime ensures all reads and writes 
    are scoped strictly to that partition, preventing state leaks between users.
    """)

with st.expander("Q5: What is time-travel debugging, and how do checkpoint DAGs enable it?"):
    st.markdown("""
    **Answer:**  
    Because checkpointers store every intermediate step as an immutable snapshot pointing to its parent, the history 
    forms a navigable DAG. Developers can retrieve any past checkpoint, inspect the exact prompt or state, modify state keys, 
    and branch execution forward from that point in time, enabling root-cause debugging of agent failures.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Mutating State In-Place Inside Graph Nodes**  
Executing `state["messages"].append(new_msg)` directly inside a node modifies mutable Python objects in place, 
bypassing channel reducers and preventing checkpointers from detecting state updates. Always return a new dictionary delta.
""")

st.error("""
**Trap 2: Storing Unserializable Objects in Agent State**  
Putting active database connections, thread locks, or open file handles inside state crashes the checkpointer, 
as state must be JSON/Pickle serializable for database storage.
""")

st.warning("""
**Trap 3: Omitting add_messages on Message Channels**  
Declaring `messages: List[BaseMessage]` without `Annotated[..., add_messages]` causes every node to overwrite 
the entire message list with its own single return value, wiping conversation history!
""")

st.warning("""
**Trap 4: Reusing Static thread_id Values Across Different User Sessions**  
Hardcoding `thread_id="default"` causes all user requests to merge into a single shared conversation thread, 
leaking private user messages across sessions.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Channel Reducer</h4>
        <p>A pure function governing how incoming node output deltas combine with existing state channel values.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>add_messages</h4>
        <p>A specialized reducer that appends, updates by ID, or deletes messages in agent conversation history.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Thread Isolation</h4>
        <p>Partitioning state checkpoints by unique thread identifiers to guarantee multi-tenant execution safety.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Time-Travel Debugging</h4>
        <p>Navigating historical checkpoint snapshots to inspect, modify, and replay past execution steps.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to design custom state schemas using TypedDict and Annotated channels.", value=True)
chk_2 = st.checkbox("I understand the mechanics of the add_messages reducer, including ID-based updates and deletions.", value=True)
chk_3 = st.checkbox("I know why PostgresSaver is required for multi-worker production deployments.", value=True)
chk_4 = st.checkbox("I can manage multi-tenant state isolation using configurable thread_id identifiers.", value=True)
chk_5 = st.checkbox("I understand how checkpoints record step-level snapshots to support time-travel replays.", value=True)
chk_6 = st.checkbox("I know how durable checkpointers enable fault-tolerant state recovery following pod crashes.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced State Management competencies verified! Ready for Advanced Orchestration.")
