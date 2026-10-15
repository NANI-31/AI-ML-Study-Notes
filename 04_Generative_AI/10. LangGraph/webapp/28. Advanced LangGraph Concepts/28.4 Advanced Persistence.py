import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="28.4 Advanced Persistence - LangGraph Curriculum",
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
    .badge-persist {
        background-color: #3B0764;
        color: #E9D5FF;
        border: 1px solid #A855F7;
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
class AdvancedPersistenceSimulator:
    """
    Simulates checkpoint namespaces, branching DAG history,
    and state forking recovery mechanics.
    """
    def __init__(self):
        self.checkpoints = [
            {"id": "chk_001", "parent": None, "ns": "root", "step": 1, "node": "intake", "state": {"stage": "init"}},
            {"id": "chk_002", "parent": "chk_001", "ns": "root", "step": 2, "node": "agent", "state": {"stage": "planning"}},
            {"id": "chk_003", "parent": "chk_002", "ns": "root", "step": 3, "node": "agent", "state": {"stage": "faulty_plan"}},
            # Forked branch from chk_002
            {"id": "chk_fork_004", "parent": "chk_002", "ns": "root", "step": 3, "node": "agent", "state": {"stage": "corrected_plan"}},
            {"id": "chk_fork_005", "parent": "chk_fork_004", "ns": "root", "step": 4, "node": "executor", "state": {"stage": "success"}}
        ]

    def get_history(self) -> List[Dict[str, Any]]:
        return self.checkpoints

    def fork_checkpoint(self, base_id: str, patch_data: Dict[str, Any]) -> str:
        new_id = f"chk_branch_{len(self.checkpoints)+1}"
        new_entry = {
            "id": new_id,
            "parent": base_id,
            "ns": "root",
            "step": 3,
            "node": "time_travel_update",
            "state": patch_data
        }
        self.checkpoints.append(new_entry)
        return new_id

pers_sim = AdvancedPersistenceSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Checkpoint Lineage Graphs, Namespaced Isolation, and Long-Running Durability", divider="rainbow")

st.markdown("""
Persistence is the structural backbone of LangGraph. Rather than maintaining state as a simple linear log, 
LangGraph's checkpointer models persistence as a **Directed Acyclic Graph (DAG) of immutable state snapshots**. 

Every checkpoint references its `parent_checkpoint_id` and belongs to a hierarchical `checkpoint_ns` (namespace). 
This architectural model enables isolated subgraph persistence, time-travel branching without history destruction, 
fault recovery from arbitrary points in time, and workflows that can pause for days awaiting human approval.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedPersistenceBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_namespace {
            label="1. Checkpoint Namespace Hierarchy";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            root_ns [label="Root Namespace: ''\\nParent Orchestrator Snapshots", fillcolor="#0F172A"];
            sub_ns1 [label="Child Namespace: ('researcher')\\nSub-Agent 1 Snapshots", fillcolor="#1E293B"];
            sub_ns2 [label="Child Namespace: ('writer')\\nSub-Agent 2 Snapshots", fillcolor="#1E293B"];
        }

        subgraph cluster_dag {
            label="2. Checkpoint Lineage & Branching DAG";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chk_root [label="Checkpoint 1 (Root)\\nstep 0", fillcolor="#0F172A"];
            chk_step1 [label="Checkpoint 2\\nstep 1", fillcolor="#1E293B"];
            chk_fault [label="Checkpoint 3 (Error State)\\nstep 2", fillcolor="#831843", fontcolor="#FBCFE8"];
            chk_fork [label="Forked Checkpoint 3B\\nupdate_state(chk_002)\\nstep 2 (Corrected)", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            chk_resolved [label="Checkpoint 4B\\nstep 3 (Success)", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_storage {
            label="3. Durable Persistent Store (PostgresSaver)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chk_table [shape=plaintext, label=<<table border="1" cellborder="1" cellspacing="0" bgcolor="#0F172A" color="#334155">
                <tr><td bgcolor="#1E293B"><b>thread_id</b></td><td bgcolor="#1E293B"><b>checkpoint_ns</b></td><td bgcolor="#1E293B"><b>checkpoint_id</b></td><td bgcolor="#1E293B"><b>parent_id</b></td></tr>
                <tr><td>thread_401</td><td>root</td><td>chk_001</td><td>null</td></tr>
                <tr><td>thread_401</td><td>root</td><td>chk_002</td><td>chk_001</td></tr>
                <tr><td>thread_401</td><td>('researcher')</td><td>chk_sub_01</td><td>null</td></tr>
            </table>>];
        }

        root_ns -> sub_ns1;
        root_ns -> sub_ns2;

        chk_root -> chk_step1;
        chk_step1 -> chk_fault [label="Original Path"];
        chk_step1 -> chk_fork [label="Time Travel Fork"];
        chk_fork -> chk_resolved;

        chk_step1 -> chk_table [label="Atomic Commit"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Checkpoint Namespaces
# ==============================================================================
st.header("2. Checkpoint Namespaces: Hierarchical Tuple Keys, Subgraph Scoping, and Cross-Namespace Queries", divider="rainbow")

st.markdown("""
A **Checkpoint Namespace** (`checkpoint_ns`) is a tuple of strings that partitions state persistence within a thread.

### Why Namespaces Matter:
- **Subgraph Isolation**: When a parent graph runs a nested subgraph, the subgraph's intermediate states are stored 
  under `("parent_node", "child_node")`, completely preventing state key collisions with the parent.
- **Selective Traversal**: Inspection APIs allow querying `get_state(config, subgraphs=True)` to traverse all child namespaces 
  or filter by a specific namespace tuple.
""")

st.code('''# Specifying a checkpoint namespace in RunnableConfig
config = {
    "configurable": {
        "thread_id": "session-401",
        "checkpoint_ns": ("research_team", "web_crawler")
    }
}

# Queries state specific to that sub-agent namespace
snapshot = app.get_state(config)
print("Crawler Channel Values:", snapshot.values)
''', language="python")

# ==============================================================================
# SECTION 3: State History
# ==============================================================================
st.header("3. State History: Traversal Algorithms, Reverse Chronological Iterators, and Audit Logging", divider="rainbow")

st.markdown("""
`graph.get_state_history(config)` yields an iterator of `StateSnapshot` objects in reverse chronological order 
(newest to oldest), walking back the tree of checkpoints linked via `parent_checkpoint_id`.

### Use Cases for State History:
- **Audit Verification**: Proving what reasoning path the agent took and which tools were called.
- **Replay Debugging**: Stepping backward through execution to identify the exact superstep where hallucinations occurred.
- **Branch Detection**: Identifying divergent branches created by time-travel forks.
""")

# ==============================================================================
# SECTION 4: Forking State
# ==============================================================================
st.header("4. Forking State: Creating Non-Destructive Branches via update_state() and Parent Linkage", divider="rainbow")

st.markdown("""
In LangGraph, past checkpoints are strictly immutable. When an engineer calls `update_state(config, values)` 
pointing to an older `checkpoint_id`, LangGraph does **not** overwrite the past. 

Instead, it creates a **new checkpoint** whose `parent_checkpoint_id` points to the historical snapshot, 
creating an explicit branch in the DAG.
""")

with st.expander("🔍 View State Forking & Branching DAG Flow", expanded=True):
    fork_dag_dot = """
    digraph ForkDAG {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        c1 [label="Checkpoint 1\\nInitial Input", fillcolor="#0F172A"];
        c2 [label="Checkpoint 2\\nPlanner Node", fillcolor="#1E293B"];
        c3_bad [label="Checkpoint 3A (Flawed)\\nBad Tool Arguments", fillcolor="#831843", fontcolor="#FBCFE8"];
        c3_fork [label="Checkpoint 3B (Forked)\\nCorrected Tool Args", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        c4_good [label="Checkpoint 4B\\nValid Output", fillcolor="#064E3B", fontcolor="#A7F3D0"];

        c1 -> c2;
        c2 -> c3_bad [label="Original Execution"];
        c2 -> c3_fork [label="update_state(chk_2)"];
        c3_fork -> c4_good [label="invoke(None)"];
    }
    """
    st.graphviz_chart(fork_dag_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: Time Travel
# ==============================================================================
st.header("5. Time Travel: Rewinding Execution DAGs, State Inspection, and Branch Modification", divider="rainbow")

st.markdown("""
**Time Travel** allows developers and supervisors to:
1. Rewind to any historical snapshot using `get_state({"configurable": {"thread_id": tid, "checkpoint_id": chk_id}})`.
2. Inspect channel values and pending writes at that instant.
3. Patch state channels using `update_state(..., as_node="target_node")`.
4. Resume execution along the new branch using `invoke(None, new_config)`.
""")

# ==============================================================================
# SECTION 6: Recovery
# ==============================================================================
st.header("6. Recovery: Checkpoint Resurrection, Clearing Pending Writes, and Re-Invoking Faulted Threads", divider="rainbow")

st.markdown("""
When an unhandled exception or server restart crashes an active run, the thread remains safe:
- **Resurrection**: Simply invoke the thread again with `app.invoke(None, config)`. 
  LangGraph loads the last committed superstep and resumes execution.
- **Clearing Pending Writes**: If a crashed node left corrupted pending writes, calling `update_state(config, {})` 
  creates a clean checkpoint barrier, clearing poisoned tasks.
""")

# ==============================================================================
# SECTION 7: Long-Running Execution
# ==============================================================================
st.header("7. Long-Running Execution: Workflows Spanning Days/Weeks, Human Approval Waits, and Heartbeat Resumption", divider="rainbow")

st.markdown("""
Because checkpoints persist permanently in PostgreSQL, agent workflows are not constrained by HTTP timeouts:
- **Long-Term Pauses**: A financial approval graph can call `interrupt()` and wait 14 days for CFO approval. 
  Zero RAM or CPU is consumed during the pause.
- **Heartbeat Resumption**: When the CFO approves via web portal, the API server retrieves the thread from PostgreSQL 
  and continues the workflow instantly.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Enterprise Namespaced Checkpoint & Time-Travel Recovery Harness", divider="rainbow")

st.markdown("""
The following complete module implements an enterprise persistence utility for navigating checkpoint trees, 
forking historical branches, and recovering interrupted threads.
""")

st.code('''from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

class EnterprisePersistenceManager:
    """
    Production utility for checkpoint tree traversal,
    time-travel branching, and state recovery.
    """
    def __init__(self, compiled_graph):
        self.graph = compiled_graph

    def get_thread_snapshots(self, thread_id: str) -> List[dict]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshots = []
        for s in self.graph.get_state_history(config):
            cfg = s.config.get("configurable", {})
            snapshots.append({
                "checkpoint_id": cfg.get("checkpoint_id"),
                "parent_id": s.parent_config.get("configurable", {}).get("checkpoint_id") if s.parent_config else None,
                "step": s.metadata.get("step"),
                "next_tasks": list(s.next),
                "values": s.values
            })
        return snapshots

    def rewind_and_fork(
        self,
        thread_id: str,
        target_checkpoint_id: str,
        state_patch: Dict[str, Any],
        as_node: str
    ) -> Dict[str, Any]:
        fork_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": target_checkpoint_id
            }
        }
        # Fork state creating child branch
        new_config = self.graph.update_state(fork_config, values=state_patch, as_node=as_node)
        # Continue execution along forked timeline
        return self.graph.invoke(None, new_config)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Checkpoint DAG Branching Sandbox & Recovery Simulator", divider="rainbow")

st.markdown("""
Inspect historical checkpoint DAG lineages, simulate time-travel rollbacks, and generate non-destructive state branches.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Time-Travel Branching Controls")
    history = pers_sim.get_history()
    chk_options = [c["id"] for c in history]
    
    target_chk = st.selectbox("Select Checkpoint to Fork From", chk_options, index=1)
    patch_val = st.text_input("State Value to Inject", value="corrected_financial_plan")
    
    if st.button("Execute Time-Travel Fork", type="primary"):
        new_id = pers_sim.fork_checkpoint(target_chk, {"stage": patch_val})
        st.success(f"Forked new checkpoint branch: `{new_id}` (Parent: `{target_chk}`)")
        st.rerun()

with col_wb2:
    st.subheader("Checkpoint Lineage DAG Table")
    df_history = pd.DataFrame(history)
    st.dataframe(df_history, use_container_width=True)
    st.info("Notice that original history remains immutable; new branches extend the DAG without modifying past snapshots.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What are checkpoint namespaces and why are they formatted as tuples?"):
    st.markdown("""
    **Answer:**  
    `checkpoint_ns` partitions checkpointer storage within a thread. Tuples represent the hierarchical path of nested subgraphs 
    (e.g., `("parent", "research_subgraph", "crawler")`). This hierarchical tuple structure allows subgraphs to maintain 
    isolated state and permits recursive queries across all child namespaces.
    """)

with st.expander("Q2: How does update_state() create a fork instead of overwriting historical records?"):
    st.markdown("""
    **Answer:**  
    LangGraph checkpointers treat checkpoints as strictly append-only. When `update_state(config, values)` targets an older checkpoint ID, 
    it writes a brand new checkpoint record whose `parent_checkpoint_id` is set to that older ID, creating a new branch in the tree 
    while preserving the original branch completely intact.
    """)

with st.expander("Q3: What role does the as_node parameter play during time-travel forking?"):
    st.markdown("""
    **Answer:**  
    `as_node` tells the graph runtime which node generated the state update. This dictates:  
    1. Which conditional edges or outgoing edges are evaluated next.  
    2. Which node appears in `snapshot.next` for subsequent invocation.
    """)

with st.expander("Q4: How do long-running agents survive database checkpointer connection loss?"):
    st.markdown("""
    **Answer:**  
    `PostgresSaver` uses connection pooling with automatic reconnection. If the database experiences a brief restart, 
    the pool reconnects. If a superstep write fails, the node retries; if the process dies, state remains safe at the last committed checkpoint.
    """)

with st.expander("Q5: What is the difference between an interrupt pause and an unhandled process failure?"):
    st.markdown("""
    **Answer:**  
    - **Interrupt Pause**: Clean, deliberate suspension. The node calls `interrupt()`, pending writes are recorded, and the thread waits for `Command(resume=...)`.  
    - **Unhandled Failure**: Abrupt crash mid-step. The uncommitted superstep is discarded, and the thread can be safely resurrected from the prior committed checkpoint.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Passing Mutable In-Place Dictionaries to update_state()**  
Passing a modified in-place dictionary to `update_state()` without shallow copying can cause reference mutations 
that leak between checkpoints in in-memory test suites.
""")

st.error("""
**Trap 2: Forgetting as_node During Time-Travel Forks**  
Invoking `update_state()` without `as_node` can leave `snapshot.next` empty or defaulted to START, causing the subsequent 
`invoke(None, config)` to exit immediately without executing downstream nodes.
""")

st.warning("""
**Trap 3: Unbounded Checkpoint History Traversal on Production Threads**  
Calling `list(graph.get_state_history(config))` on threads with thousands of steps loads massive JSON histories into RAM. 
Always use pagination or limit generators.
""")

st.warning("""
**Trap 4: Checkpoint Namespace Mismatch in Subgraph Queries**  
Attempting to query subgraph state without specifying `subgraphs=True` or the exact `checkpoint_ns` tuple returns 
only the parent graph's channels.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>checkpoint_ns</h4>
        <p>A tuple-based namespace key partitioning checkpointer snapshots between parent graphs and nested subgraphs.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>State Forking</h4>
        <p>Branching execution from an earlier checkpoint ID to create an alternative timeline without modifying past history.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Time-Travel Debugging</h4>
        <p>The ability to inspect, rewind, patch, and resume agent workflows from any historical checkpoint snapshot.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Durable Long-Running Workflows</h4>
        <p>Workflows that pause for days or weeks awaiting human approval, consuming zero compute while persisted in PostgreSQL.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand how checkpoint namespaces isolate parent and subgraph persistence.", value=True)
chk_2 = st.checkbox("I know how to traverse state history using get_state_history().", value=True)
chk_3 = st.checkbox("I can implement time-travel branching and state forking with update_state(as_node=...).", value=True)
chk_4 = st.checkbox("I understand how to recover crashed agent threads from the last committed checkpoint.", value=True)
chk_5 = st.checkbox("I know how to design durable workflows that pause across days or weeks for human approval.", value=True)
chk_6 = st.checkbox("I understand why checkpoints are strictly append-only and immutable DAG nodes.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced persistence and time-travel engineering mastery confirmed! Chapters 26, 27, and 28 complete!")
