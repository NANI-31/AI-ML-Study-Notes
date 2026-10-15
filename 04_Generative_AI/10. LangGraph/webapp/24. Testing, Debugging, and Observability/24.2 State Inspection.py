import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="24.2 State Inspection - LangGraph Curriculum",
    page_icon="🔍",
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
    .badge-checkpoint {
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
class StateInspectionSimulator:
    """
    Simulates checkpoint inspection, historical snapshot iteration,
    and time-travel branching in LangGraph.
    """
    def __init__(self):
        self.threads: Dict[str, List[Dict[str, Any]]] = {
            "thread-alpha": [
                {
                    "checkpoint_id": "1ef9a01-chk01",
                    "parent_id": None,
                    "step": 0,
                    "node": "__start__",
                    "values": {"messages": ["User: Refactor data pipeline"], "status": "initialized", "plan": []},
                    "next": ("planner",),
                    "pending_writes": []
                },
                {
                    "checkpoint_id": "1ef9a01-chk02",
                    "parent_id": "1ef9a01-chk01",
                    "step": 1,
                    "node": "planner",
                    "values": {"messages": ["User: Refactor data pipeline", "Planner: Split into 3 stages"], "status": "planned", "plan": ["Extract", "Clean", "Load"]},
                    "next": ("executor",),
                    "pending_writes": []
                },
                {
                    "checkpoint_id": "1ef9a01-chk03",
                    "parent_id": "1ef9a01-chk02",
                    "step": 2,
                    "node": "executor",
                    "values": {"messages": ["User: Refactor data pipeline", "Planner: Split into 3 stages", "Executor: Extracted records"], "status": "executing", "plan": ["Clean", "Load"]},
                    "next": ("evaluator",),
                    "pending_writes": []
                },
                {
                    "checkpoint_id": "1ef9a01-chk04",
                    "parent_id": "1ef9a01-chk03",
                    "step": 3,
                    "node": "evaluator",
                    "values": {"messages": ["User: Refactor data pipeline", "Planner: Split into 3 stages", "Executor: Extracted records", "Evaluator: Warning - schema mismatch"], "status": "needs_correction", "plan": ["Clean", "Load"]},
                    "next": ("human_approval",),
                    "pending_writes": [("human_approval", "request_feedback", {"risk": "schema_change"})]
                }
            ],
            "thread-beta": [
                {
                    "checkpoint_id": "1ef9b02-chk01",
                    "parent_id": None,
                    "step": 0,
                    "node": "__start__",
                    "values": {"messages": ["User: Summarize Q3 earnings"], "status": "initialized"},
                    "next": ("summarizer",),
                    "pending_writes": []
                },
                {
                    "checkpoint_id": "1ef9b02-chk02",
                    "parent_id": "1ef9b02-chk01",
                    "step": 1,
                    "node": "summarizer",
                    "values": {"messages": ["User: Summarize Q3 earnings", "Bot: Revenue rose 14% YoY"], "status": "completed"},
                    "next": (),
                    "pending_writes": []
                }
            ]
        }

    def get_state(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        history = self.threads.get(thread_id, [])
        if not history:
            return None
        if checkpoint_id:
            for item in history:
                if item["checkpoint_id"] == checkpoint_id:
                    return item
            return None
        return history[-1]

    def get_state_history(self, thread_id: str) -> List[Dict[str, Any]]:
        return list(reversed(self.threads.get(thread_id, [])))

    def time_travel_fork(self, thread_id: str, target_checkpoint_id: str, state_update: Dict[str, Any]) -> str:
        history = self.threads.get(thread_id, [])
        target_idx = None
        for i, chk in enumerate(history):
            if chk["checkpoint_id"] == target_checkpoint_id:
                target_idx = i
                break
        if target_idx is None:
            return "Target checkpoint not found"

        base_chk = history[target_idx]
        new_values = dict(base_chk["values"])
        new_values.update(state_update)
        new_chk_id = f"{base_chk['checkpoint_id']}-fork{len(history)+1}"
        new_snapshot = {
            "checkpoint_id": new_chk_id,
            "parent_id": base_chk["checkpoint_id"],
            "step": base_chk["step"] + 1,
            "node": "time_travel_update",
            "values": new_values,
            "next": ("executor",),
            "pending_writes": []
        }
        self.threads[thread_id].append(new_snapshot)
        return new_chk_id

simulator = StateInspectionSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: State Inspection, Checkpoint Trees, and Time-Travel Mechanics", divider="rainbow")

st.markdown("""
LangGraph's checkpointing architecture records the complete immutable state snapshot at each superstep barrier. 
By persisting state transitions into a persistent checkpointer (such as `MemorySaver`, `SqliteSaver`, or `PostgresSaver`), 
developers gain programmatic read and write access to the execution timeline. 
Inspection APIs (`get_state` and `get_state_history`) allow auditing current and past node values, viewing scheduled next 
tasks, diagnosing pending channel writes, and executing deterministic time-travel rollbacks.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StateInspectionBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_runtime {
            label="1. LangGraph Execution Runtime";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            node_step0 [label="Superstep 0: __start__\\nInitial Input State", fillcolor="#0F172A"];
            node_step1 [label="Superstep 1: Planner Node\\nChannel Reducers Applied", fillcolor="#1E293B"];
            node_step2 [label="Superstep 2: Executor Node\\nTool Outputs & Artifacts", fillcolor="#1E293B"];
            node_step3 [label="Superstep 3: Interrupt / Pause\\nWaiting for External Input", fillcolor="#312E81"];
        }

        subgraph cluster_checkpointer {
            label="2. Checkpoint Storage Engine (Postgres / SQLite)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chk_table [shape=plaintext, label=<<table border="1" cellborder="1" cellspacing="0" bgcolor="#0F172A" color="#334155">
                <tr><td bgcolor="#1E293B"><b>thread_id</b></td><td bgcolor="#1E293B"><b>checkpoint_id</b></td><td bgcolor="#1E293B"><b>parent_id</b></td><td bgcolor="#1E293B"><b>next_nodes</b></td></tr>
                <tr><td>thread-alpha</td><td>chk_001</td><td>null</td><td>planner</td></tr>
                <tr><td>thread-alpha</td><td>chk_002</td><td>chk_001</td><td>executor</td></tr>
                <tr><td>thread-alpha</td><td>chk_003</td><td>chk_002</td><td>evaluator</td></tr>
            </table>>];
        }

        subgraph cluster_apis {
            label="3. State Inspection & Time-Travel APIs";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            api_get_state [label="graph.get_state(config)\\nReads latest or specific snapshot\\nReturns: StateSnapshot", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            api_history [label="graph.get_state_history(config)\\nIterates past checkpoint DAG\\nReturns: Iterator[StateSnapshot]", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            api_update [label="graph.update_state(config, values)\\nForks or modifies state at checkpoint\\nGenerates new child checkpoint", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        node_step0 -> node_step1 -> node_step2 -> node_step3;
        node_step1 -> chk_table [label="Persists Step 1"];
        node_step2 -> chk_table [label="Persists Step 2"];
        node_step3 -> chk_table [label="Persists Step 3"];

        chk_table -> api_get_state [label="Point Query"];
        chk_table -> api_history [label="Timeline Scan"];
        api_update -> chk_table [label="Forks New Checkpoint"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: get_state()
# ==============================================================================
st.header("2. get_state(): Querying Active Snapshots and Scheduled Next Steps", divider="rainbow")

st.markdown("""
The `graph.get_state(config)` method retrieves a `StateSnapshot` object representing the current state of a thread. 
The configuration dictionary requires a `thread_id` and can optionally include a specific `checkpoint_id`.

### StateSnapshot Structure:
- **`values`**: Current channel dictionary containing state keys (such as `messages`, `plan`, `context`).
- **`next`**: Tuple of node names scheduled to execute in the next superstep. If empty `()`, the graph has terminated or reached `END`.
- **`config`**: The exact runnable configuration (including `thread_id` and current `checkpoint_id`).
- **`metadata`**: Step index, write provenance, source nodes, and timing metrics.
- **`created_at`**: ISO timestamp when the checkpoint was saved.
- **`parent_config`**: Reference to the antecedent checkpoint that produced this snapshot.
""")

st.code('''from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Instantiate graph with persistence checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-prod-401"}}

# Run one step or interrupt
app.invoke({"query": "Migrate schema"}, config)

# Query current snapshot
current_snapshot = app.get_state(config)

print("Current Values:", current_snapshot.values)
print("Next Nodes to Run:", current_snapshot.next)
print("Current Checkpoint ID:", current_snapshot.config["configurable"]["checkpoint_id"])
''', language="python")

# ==============================================================================
# SECTION 3: get_state_history()
# ==============================================================================
st.header("3. get_state_history(): Iterating Historical State Snapshots Over Threads", divider="rainbow")

st.markdown("""
`graph.get_state_history(config)` returns an iterator of `StateSnapshot` objects in reverse chronological order 
(from newest checkpoint to oldest).

### Key Iteration Patterns:
1. **Auditing State Evolution**: Inspecting how channel reducers (such as message list append or dictionary merging) accumulated values across steps.
2. **Replay Validation**: Comparing intermediate reasoning steps before and after model temperature or prompt updates.
3. **Filtering by Node Execution**: Finding the exact checkpoint generated immediately after a specific tool or agent node finished.
""")

st.code('''# Iterate through historical checkpoints in reverse order
history_generator = app.get_state_history(config)

for idx, snapshot in enumerate(history_generator):
    chk_id = snapshot.config["configurable"].get("checkpoint_id")
    step_num = snapshot.metadata.get("step", "N/A")
    next_tasks = snapshot.next
    print(f"Step {step_num} | Checkpoint: {chk_id} | Scheduled Next: {next_tasks}")
    print(f"  Messages Count: {len(snapshot.values.get('messages', []))}")
''', language="python")

# ==============================================================================
# SECTION 4: Checkpoint Inspection
# ==============================================================================
st.header("4. Checkpoint Inspection: Parsing Checkpoint IDs, Parent IDs, and Pending Writes", divider="rainbow")

st.markdown("""
Checkpoints form a directed acyclic tree. Every checkpoint references its `parent_checkpoint_id`. 
In addition to committed channel values, inspectable checkpoints contain **pending writes**—tasks that were scheduled or written 
by nodes before an interrupt occurred.
""")

with st.expander("🔍 View Checkpoint Tree & Pending Writes Architecture", expanded=True):
    chk_tree_dot = """
    digraph CheckpointTree {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        chk1 [label="Checkpoint 1 (Root)\\nchk_root_001\\nparent: null\\nnode: __start__", fillcolor="#0F172A"];
        chk2 [label="Checkpoint 2\\nchk_agent_002\\nparent: chk_root_001\\nnode: agent", fillcolor="#1E293B"];
        chk3 [label="Checkpoint 3 (Interrupt)\\nchk_human_003\\nparent: chk_agent_002\\nPending Write: ToolRequest", fillcolor="#312E81"];
        chk4_prod [label="Branch A (Standard Resume)\\nchk_approved_004\\nparent: chk_human_003", fillcolor="#064E3B"];
        chk4_fork [label="Branch B (Time-Travel Fork)\\nchk_forked_004\\nparent: chk_agent_002 (Rollback)", fillcolor="#831843"];

        chk1 -> chk2 [label="step 1"];
        chk2 -> chk3 [label="step 2"];
        chk3 -> chk4_prod [label="resume()"];
        chk2 -> chk4_fork [label="update_state(chk_002)"];
    }
    """
    st.graphviz_chart(chk_tree_dot, use_container_width=True)

st.markdown("""
### Parsing Pending Writes
When execution interrupts before a node completes its side effects, `snapshot.tasks` displays pending work:
- **`task.id`**: Unique identifier of the scheduled step.
- **`task.name`**: Target node identifier.
- **`task.interrupts`**: Interruption payloads passed to `interrupt()`.
""")

# ==============================================================================
# SECTION 5: Thread Inspection
# ==============================================================================
st.header("5. Thread Inspection: Multi-User Session Isolation and Thread Concurrency", divider="rainbow")

st.markdown("""
Threads isolate execution contexts. Two concurrent users sending queries to the same compiled graph operate in strictly 
separated state partitions as long as their `thread_id` configurations differ.

### Multi-Tenant Isolation Guarantees:
- **Zero Cross-Talk**: Checkpointer operations filter rows by `WHERE thread_id = :thread_id`.
- **Independent Checkpoint Branches**: Rolling back or forking a checkpoint on `thread-001` has zero side effects on `thread-002`.
- **Session Lifecycles**: Ephemeral checkpointers clean up threads upon garbage collection, whereas PostgreSQL checkpointers support indexed partition purging.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    #### Thread Partitioning Scheme
    ```
    PostgreSQL Checkpoints Table:
    +------------+---------------+----------+
    | thread_id  | checkpoint_id | values   |
    +------------+---------------+----------+
    | tenant_A   | chk_101       | {...}    |
    | tenant_A   | chk_102       | {...}    |
    | tenant_B   | chk_201       | {...}    |
    +------------+---------------+----------+
    ```
    """)
with col2:
    st.markdown("""
    #### Concurrent Lock Handling
    In LangGraph Cloud and Postgres checkpointers:
    - Thread-level row locking prevents conflicting simultaneous writes.
    - If Node A and Node B attempt to write to the same thread simultaneously, the checkpointer serializes supersteps deterministically.
    """)

# ==============================================================================
# SECTION 6: Subgraph State Inspection
# ==============================================================================
st.header("6. Subgraph State Inspection: Recursively Inspecting Child Namespaces", divider="rainbow")

st.markdown("""
When building hierarchical multi-agent architectures, parent graphs delegate tasks to compiled subgraphs. 
By default, `graph.get_state(config)` inspects only the parent graph's top-level channels.

To inspect nested subgraph states, LangGraph provides the `subgraphs=True` argument:
- `graph.get_state(config, subgraphs=True)`: Returns the parent snapshot and all active child subgraph snapshots.
- Subgraph snapshots include a `checkpoint_ns` (checkpoint namespace) tuple detailing the hierarchy: e.g., `("parent", "researcher_subgraph:uuid")`.
""")

st.code('''# Inspect parent state and all active child subgraphs
parent_snapshot = app.get_state(config, subgraphs=True)

# If child subgraphs are paused or executing:
for snapshot in app.get_state_history(config, subgraphs=True):
    ns = snapshot.config["configurable"].get("checkpoint_ns", "root")
    print(f"Namespace: {ns} | Step: {snapshot.metadata.get('step')} | Node: {snapshot.next}")
''', language="python")

# ==============================================================================
# SECTION 7: Time-Travel Debugging
# ==============================================================================
st.header("7. Time-Travel Debugging: Forking Historical Checkpoints and State Modification via update_state()", divider="rainbow")

st.markdown("""
**Time-travel debugging** is the ability to rewind an agent to an earlier checkpoint, modify its state variables, 
and branch execution down a new path.

### The Time-Travel Workflow:
1. **Locate Target Checkpoint**: Query `get_state_history()` to locate the step immediately before a hallucination or failure.
2. **Construct Configuration**: Create a config specifying both `thread_id` and the historical `checkpoint_id`.
3. **Execute `update_state(config, values, as_node=...)`**:
   - Creates a **new child checkpoint** branching off the historical checkpoint.
   - Updates channel values according to the state reducer rules.
   - Attributes the write to the specified node (`as_node`), determining which node executes next.
4. **Resume Execution**: Call `app.invoke(None, config)` to continue execution from the newly branched checkpoint.
""")

st.code('''# Step 1: Select historical checkpoint
target_chk_id = "1ef9a01-chk02"  # Checkpoint before planner failed
fork_config = {
    "configurable": {
        "thread_id": "thread-alpha",
        "checkpoint_id": target_chk_id
    }
}

# Step 2: Inject corrected state as if generated by the planner
corrected_state = {
    "plan": ["Extract valid schema", "Clean null records", "Load to warehouse"],
    "status": "corrected_by_developer"
}

# update_state creates a new child checkpoint branching from chk02
new_config = app.update_state(
    fork_config,
    values=corrected_state,
    as_node="planner"
)

# Step 3: Resume execution from new branch
new_result = app.invoke(None, new_config)
''', language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full State Inspection & Time-Travel Debugging Pipeline", divider="rainbow")

st.markdown("""
Below is a self-contained, enterprise-grade inspection and recovery module implementing checkpoint history retrieval, 
health verification, and time-travel rollbacks.
""")

st.code('''from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class CheckpointAuditRecord:
    checkpoint_id: str
    parent_id: Optional[str]
    step_index: int
    scheduled_nodes: List[str]
    messages_count: int
    has_pending_writes: bool
    status: str

class ProductionStateInspector:
    """
    Enterprise utility for inspecting LangGraph checkpointers,
    evaluating state anomalies, and generating recovery forks.
    """
    def __init__(self, compiled_graph):
        self.graph = compiled_graph

    def audit_thread(self, thread_id: str) -> List[CheckpointAuditRecord]:
        config = {"configurable": {"thread_id": thread_id}}
        records = []
        for snapshot in self.graph.get_state_history(config):
            cfg = snapshot.config.get("configurable", {})
            meta = snapshot.metadata or {}
            
            records.append(CheckpointAuditRecord(
                checkpoint_id=cfg.get("checkpoint_id", "unknown"),
                parent_id=snapshot.parent_config.get("configurable", {}).get("checkpoint_id") if snapshot.parent_config else None,
                step_index=meta.get("step", -1),
                scheduled_nodes=list(snapshot.next),
                messages_count=len(snapshot.values.get("messages", [])),
                has_pending_writes=bool(snapshot.tasks),
                status=snapshot.values.get("status", "unknown")
            ))
        return records

    def rollback_and_fork(
        self,
        thread_id: str,
        target_checkpoint_id: str,
        injected_values: Dict[str, Any],
        as_node: str
    ) -> Dict[str, Any]:
        fork_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": target_checkpoint_id
            }
        }
        # Fork state creating child checkpoint
        new_config = self.graph.update_state(fork_config, values=injected_values, as_node=as_node)
        # Continue execution along forked timeline
        return self.graph.invoke(None, new_config)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: State History Visualizer & Time-Travel Branching Sandbox", divider="rainbow")

st.markdown("""
Experiment with real checkpoint inspection and time-travel rollbacks on simulated threads.
""")

col_wb1, col_wb2 = st.columns([1, 2])

with col_wb1:
    st.subheader("Inspection Controls")
    selected_thread = st.selectbox(
        "Select Active Thread ID",
        ["thread-alpha", "thread-beta"],
        index=0
    )
    
    inspect_mode = st.radio(
        "Inspection Mode",
        ["Latest State (get_state)", "Full History (get_state_history)", "Time-Travel Fork"]
    )

with col_wb2:
    st.subheader("Checkpoint Inspector Results")
    
    if inspect_mode == "Latest State (get_state)":
        snapshot = simulator.get_state(selected_thread)
        if snapshot:
            st.markdown(f"**Thread ID:** `{selected_thread}` | **Active Checkpoint:** `{snapshot['checkpoint_id']}`")
            st.json({
                "checkpoint_id": snapshot["checkpoint_id"],
                "parent_id": snapshot["parent_id"],
                "step": snapshot["step"],
                "scheduled_next": list(snapshot["next"]),
                "channel_values": snapshot["values"],
                "pending_writes": snapshot["pending_writes"]
            })
        else:
            st.warning("No state found for selected thread.")
            
    elif inspect_mode == "Full History (get_state_history)":
        history = simulator.get_state_history(selected_thread)
        st.markdown(f"**Found {len(history)} Historical Checkpoints** (Reverse Chronological Order)")
        
        hist_table = []
        for chk in history:
            hist_table.append({
                "Checkpoint ID": chk["checkpoint_id"],
                "Parent ID": chk["parent_id"] or "ROOT",
                "Step": chk["step"],
                "Node": chk["node"],
                "Scheduled Next": ", ".join(chk["next"]) if chk["next"] else "END",
                "Status": chk["values"].get("status", "N/A")
            })
        st.dataframe(pd.DataFrame(hist_table), use_container_width=True)

    else:
        st.markdown("#### Time-Travel Forking Sandbox")
        history = simulator.threads.get(selected_thread, [])
        checkpoint_options = [chk["checkpoint_id"] for chk in history]
        
        target_checkpoint = st.selectbox("Select Target Checkpoint to Fork From", checkpoint_options)
        target_snapshot = simulator.get_state(selected_thread, target_checkpoint)
        
        if target_snapshot:
            st.info(f"Target Snapshot State at step {target_snapshot['step']}: {target_snapshot['values'].get('status')}")
            
            injected_patch = st.text_input(
                "Inject State Patch (Key=Value)",
                value="status=forked_and_corrected"
            )
            
            if st.button("Execute Time-Travel Fork", type="primary"):
                k, v = injected_patch.split("=", 1)
                fork_id = simulator.time_travel_fork(selected_thread, target_checkpoint, {k.strip(): v.strip()})
                st.success(f"Successfully created forked checkpoint: `{fork_id}`")
                st.rerun()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What is the exact difference between get_state() and get_state_history()?"):
    st.markdown("""
    **Answer:**  
    `get_state(config)` performs a point query returning a single `StateSnapshot` representing the current (or specifically requested) 
    state of a thread. In contrast, `get_state_history(config)` returns a generator that iterates through all past snapshots in reverse 
    chronological order, walking back the tree of checkpoints linked via `parent_checkpoint_id`.
    """)

with st.expander("Q2: How does update_state() create a fork instead of overwriting existing checkpoints?"):
    st.markdown("""
    **Answer:**  
    Checkpoints in LangGraph are append-only and strictly immutable. When `update_state(config, values)` is called with a configuration pointing 
    to an older `checkpoint_id`, LangGraph creates a **new checkpoint row** whose `parent_checkpoint_id` is set to that older checkpoint. 
    The original history remains untouched, creating an explicit tree branch in the persistence store.
    """)

with st.expander("Q3: What role does the as_node argument play in update_state()?"):
    st.markdown("""
    **Answer:**  
    `as_node` tells the graph runtime which node is responsible for generating the state write. This is critical for two reasons:
    1. **Routing**: Conditional edges or direct edges originating from `as_node` determine which node will be scheduled in `snapshot.next`.
    2. **Channel Reducers**: If specific channels have reducers scoped to node writers, attributing the write to `as_node` ensures proper reducer resolution.
    """)

with st.expander("Q4: What are pending writes and how are they inspected?"):
    st.markdown("""
    **Answer:**  
    Pending writes are state updates or tool calls produced by a node that have not yet been committed to channels because the superstep was interrupted 
    (e.g., via `interrupt()`). They are inspected via `snapshot.tasks`, which lists each pending task, its scheduled node name, and any interrupt payloads.
    """)

with st.expander("Q5: How does LangGraph handle thread isolation in a multi-tenant PostgreSQL database?"):
    st.markdown("""
    **Answer:**  
    `PostgresSaver` indexes all checkpoint and write tables with compound primary keys on `(thread_id, checkpoint_id)`. All queries enforce 
    strict parameter filtering by `thread_id`. Simultaneous requests to distinct thread IDs execute with row-level concurrency without cross-thread lock contention.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Passing Mutable References to update_state()**  
Modifying a dictionary in-place and passing it into `update_state()` without copying can cause memory-leak side effects if using `MemorySaver`. 
Always pass fresh dictionary payloads or copies to guarantee checkpointer snapshot immutability.
""")

st.error("""
**Trap 2: Ignoring Next Tasks After Time-Travel Forks**  
When invoking `update_state(config, values)` without specifying `as_node`, LangGraph may default the scheduled next step to `__start__` 
or leave `snapshot.next` undefined, causing `app.invoke(None, config)` to immediately exit without executing downstream logic.
""")

st.warning("""
**Trap 3: Memory Exhaustion During Unbounded get_state_history() Traversal**  
Calling `list(graph.get_state_history(config))` on long-running threads with thousands of steps loads all historical snapshots into RAM simultaneously. 
Always use pagination or generator stream consumers in production monitoring pipelines.
""")

st.warning("""
**Trap 4: Subgraph Namespace Mismatch During Nested State Inspection**  
Attempting to inspect a nested subgraph using the root `thread_id` without specifying `subgraphs=True` returns only parent state. 
Subgraphs store checkpoints in partitioned namespaces that require explicit recursive traversal.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>StateSnapshot</h4>
        <p>An immutable object capturing channel values, scheduled next nodes, runnable config, step index, and parent checkpoint pointers.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Time-Travel Branching</h4>
        <p>Rewinding to an earlier checkpoint ID and calling update_state() to fork execution without modifying past history.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Pending Writes (tasks)</h4>
        <p>Buffered state transitions or interrupt payloads generated by nodes prior to an execution pause or breakpoint barrier.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>subgraphs=True</h4>
        <p>Argument passed to inspection methods to recursively retrieve state snapshots from nested child graph namespaces.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the components of a StateSnapshot (values, next, config, metadata, parent_config).", value=True)
chk_2 = st.checkbox("I know how to iterate past states using get_state_history() in reverse chronological order.", value=True)
chk_3 = st.checkbox("I can inspect pending writes and interrupt payloads via snapshot.tasks.", value=True)
chk_4 = st.checkbox("I understand how thread_id guarantees state isolation in multi-tenant persistence layers.", value=True)
chk_5 = st.checkbox("I know how to inspect nested subgraphs using subgraphs=True and checkpoint namespaces.", value=True)
chk_6 = st.checkbox("I can implement time-travel debugging by branching from historical checkpoints with update_state(as_node=...).", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("State inspection and time-travel debugging mastery confirmed! Ready for LangSmith observability.")
