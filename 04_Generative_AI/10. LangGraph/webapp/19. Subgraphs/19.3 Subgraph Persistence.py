import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="19.3 Subgraph Persistence - LangGraph Curriculum",
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
    .badge-ns {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-pause {
        background-color: #382B14;
        color: #FBBF24;
        border: 1px solid #D97706;
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
class SubgraphPersistenceSimulator:
    """
    Simulates subgraph persistence lifecycles in LangGraph:
    - checkpoint_ns (namespace) segmentation
    - Subgraph human-in-the-loop interrupts
    - Per-thread vs Stateless vs Per-invocation persistence modes
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.thread_checkpoints = {}
        self.active_interrupt = None

    def execute_subgraph_with_interrupt(
        self,
        thread_id: str,
        operation_amount: float,
        require_human_approval: bool
    ) -> Dict[str, Any]:
        # Step 1: Parent starts
        parent_ckpt_id = f"ckpt_parent_{thread_id}_01"
        self.thread_checkpoints[("", parent_ckpt_id)] = {
            "thread_id": thread_id,
            "ns": "",
            "state": {"order_amount": operation_amount, "status": "DISPATCHED"}
        }

        # Step 2: Child starts under nested namespace
        child_ns = "payment_subgraph"
        child_ckpt_id = f"ckpt_child_{thread_id}_01"

        if require_human_approval and operation_amount > 1000.0:
            # Subgraph interrupts!
            self.thread_checkpoints[(child_ns, child_ckpt_id)] = {
                "thread_id": thread_id,
                "ns": child_ns,
                "state": {"pending_amount": operation_amount, "approved": False},
                "interrupt": {
                    "question": f"High value transfer of ${operation_amount:,.2f} USD detected in subgraph. Approve?",
                    "resumable": True
                }
            }
            self.active_interrupt = (thread_id, child_ns, child_ckpt_id)
            return {
                "status": "INTERRUPTED",
                "namespace": child_ns,
                "message": f"Execution halted inside child subgraph namespace '{child_ns}'! Human review required.",
                "pending_amount": operation_amount
            }
        else:
            # Child completes immediately
            self.thread_checkpoints[(child_ns, child_ckpt_id)] = {
                "thread_id": thread_id,
                "ns": child_ns,
                "state": {"pending_amount": operation_amount, "approved": True, "tx_hash": "0x4a9b..."},
                "interrupt": None
            }
            return {
                "status": "COMPLETED",
                "namespace": child_ns,
                "message": f"Child subgraph executed and persisted without interrupts. Amount: ${operation_amount:,.2f} USD.",
                "pending_amount": operation_amount
            }

    def resume_subgraph_interrupt(self, approve: bool) -> Dict[str, Any]:
        if not self.active_interrupt:
            return {"status": "NO_ACTIVE_INTERRUPT", "message": "No active subgraph interrupt to resume."}
        
        thread_id, child_ns, child_ckpt_id = self.active_interrupt
        self.active_interrupt = None
        
        resumed_ckpt_id = f"ckpt_child_{thread_id}_02"
        self.thread_checkpoints[(child_ns, resumed_ckpt_id)] = {
            "thread_id": thread_id,
            "ns": child_ns,
            "state": {"approved": approve, "status": "APPROVED" if approve else "REJECTED"}
        }
        return {
            "status": "RESUMED",
            "decision": "APPROVED" if approve else "REJECTED",
            "message": f"Subgraph namespace '{child_ns}' resumed with human decision: {'APPROVED' if approve else 'REJECTED'}."
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Subgraph Persistence Lifecycles, Namespaces, and Checkpointing Modes", divider="rainbow")

st.markdown("""
When working with hierarchical multi-agent systems, managing persistence is complex:
- If a child graph compiles with its own checkpointer, how are checkpoints stored?
- If the child graph triggers an `interrupt()` for human approval, does the parent wait or crash?
- How does PostgreSQL differentiate checkpoints belonging to the parent from checkpoints belonging to child nodes?

LangGraph solves this with **Checkpoint Namespaces (`checkpoint_ns`)**.
Every checkpointer record in the database is indexed not only by `thread_id` and `checkpoint_id`, but also by a **hierarchical namespace path** (e.g. `""` for root parent, `"research_team"` for child, `"research_team:sql_agent"` for grandchild).

The architectural blueprint below illustrates the three persistence modes and checkpoint namespace indexing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SubgraphPersistenceTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_parent {
            label="1. Root Parent Graph (thread_id='session_101', checkpoint_ns='')";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            p_node_1 [label="Parent Node: Dispatches Subgraph\\nCheckpoint ID: 1efb-88a2-0001", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            p_node_2 [label="Parent Node: Final Aggregator\\nCheckpoint ID: 1efb-88a2-0005", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_child_persistence {
            label="2. Child Subgraph Execution (checkpoint_ns='payment_subgraph')";
            color="#4C1D95";
            fillcolor="#2E1065";
            style="filled,rounded";
            c_step_1 [label="Child Superstep 1: Validate Card\\nCheckpoint ID: 1efb-88a2-0002", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            c_interrupt [label="interrupt() Triggered\\nHuman Approval Pending!\\nCheckpoint ID: 1efb-88a2-0003", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            c_step_2 [label="Child Superstep 2: Execute Transfer\\nCheckpoint ID: 1efb-88a2-0004", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_database {
            label="3. Relational Checkpoint Store (PostgreSQL / SQLite)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            db_table [label="Checkpoints Table Index:\\n(thread_id, checkpoint_ns, checkpoint_id)\\nGuarantees Zero Collision Between Graphs", fillcolor="#18181B", fontcolor="#E4E4E7", color="#71717A"];
        }

        p_node_1 -> c_step_1 [label="Invokes Subgraph"];
        c_step_1 -> c_interrupt [label="Amount > $1,000"];
        c_interrupt -> c_step_2 [label="Human Calls Command(resume=True)"];
        c_step_2 -> p_node_2 [label="Subgraph Finishes & Returns"];

        p_node_1 -> db_table [label="Writes ns=''"];
        c_step_1 -> db_table [label="Writes ns='payment_subgraph'"];
        c_interrupt -> db_table [label="Persists Pause State"];
        c_step_2 -> db_table [label="Writes ns='payment_subgraph'"];
        p_node_2 -> db_table [label="Writes ns=''"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: PER-INVOCATION PERSISTENCE
# ==============================================================================
st.header("2. Per-Invocation Persistence: Ephemeral In-Memory State and Isolated Task Runs", divider="rainbow")

st.markdown("""
In **Per-Invocation Persistence**, the child subgraph uses an ephemeral in-memory checkpointer (or none at all) tied strictly to the lifecycle of a single parent run.
""")

st.markdown("""
- **Mechanism**: The child graph compiles without a persistent checkpointer, or with `MemorySaver()`.
- **Lifecycle**: While the child is running its internal cycles, it checkpoints in RAM. Once the child graph reaches `END`, its internal history is cleared.
- **Use Case**: One-off calculations, web searches, or document summarization where only the final deliverable matters, not the intermediate steps.
""")

# ==============================================================================
# SECTION 3: PER-THREAD PERSISTENCE
# ==============================================================================
st.header("3. Per-Thread Persistence: Long-Term Memory and Continuous Multi-Turn Subgraphs", divider="rainbow")

st.markdown("""
In **Per-Thread Persistence**, the child subgraph shares the exact same database checkpointer as the parent graph.
""")

st.markdown("""
- **Mechanism**: The parent compiles with a persistent checkpointer (e.g., `PostgresSaver`). When the child is invoked, the execution configuration (`thread_id`) is forwarded.
- **Namespace Separation**: LangGraph automatically prefixes all child checkpoints with the child's node name (`checkpoint_ns`).
- **Resilience**: If the container crashes while executing the child subgraph, the entire thread can be resumed exactly from the child's last superstep!
""")

st.code('''# Forwarding thread checkpointer to child subgraph
def invoke_persistent_subgraph_node(state: State, config: RunnableConfig) -> dict:
    # Forwarding config ensures child writes to the same PostgreSQL store under child ns
    child_result = child_app.invoke(
        {"query": state["query"]},
        config=config # Mandatory for Per-Thread Persistence!
    )
    return {"result": child_result["output"]}''', language="python")

# ==============================================================================
# SECTION 4: STATELESS SUBGRAPHS
# ==============================================================================
st.header("4. Stateless Subgraphs: Pure Functional Micro-Chains and Zero Checkpointing Overhead", divider="rainbow")

st.markdown("""
A **Stateless Subgraph** is compiled completely without a checkpointer: `child_app = child_builder.compile()`.
""")

st.markdown("""
- **Zero I/O Overhead**: No database queries, inserts, or serialization costs are incurred during the child's execution.
- **Pure Function Semantics**: Operates like an ordinary Python function. Given input X, returns output Y.
- **When to Use**: High-throughput micro-reasoning loops (e.g., parsing a list of 100 regex patterns or evaluating small JSON schemas) where saving database checkpoints on every micro-step would create severe database bottlenecks.
""")

# ==============================================================================
# SECTION 5: SUBGRAPH CHECKPOINTS
# ==============================================================================
st.header("5. Subgraph Checkpoints: Hierarchical Namespaces (checkpoint_ns) in PostgreSQL", divider="rainbow")

st.markdown("""
In relational checkpointer databases like PostgreSQL, the primary composite key for the `checkpoints` table is:
`(thread_id, checkpoint_ns, checkpoint_id)`
""")

st.markdown("""
- **Parent Checkpoints**: Stored with `checkpoint_ns = ""` (empty string).
- **First-Level Child**: Stored with `checkpoint_ns = "research_subgraph"`.
- **Nested Grandchild**: Stored with `checkpoint_ns = "research_subgraph:code_reviewer"`.

Because of this composite key, state snapshots from the child never overwrite parent state, and developers can inspect the complete timeline of either the parent or any child independently!
""")

# ==============================================================================
# SECTION 6: SUBGRAPH INTERRUPTS
# ==============================================================================
st.header("6. Subgraph Interrupts: Pausing Nested Graphs for Human-in-the-Loop Approvals", divider="rainbow")

st.markdown("""
What happens when a node inside a nested child subgraph calls `interrupt()`?
""")

st.markdown("""
1. **Child Halts**: The child node calling `interrupt()` suspends immediately.
2. **Parent Suspends**: Because the child is running synchronously or cooperatively within the parent superstep, the parent execution halts at that exact step.
3. **Namespace Preserved**: The checkpoint records the pending interrupt under the child's `checkpoint_ns`.
4. **Resumption via Command**: To resume, the client calls `app.invoke(Command(resume=payload), config=config)`. LangGraph routes the resumption command down into the nested child namespace!
""")

st.code('''from langgraph.types import interrupt, Command

def high_risk_child_node(state: ChildState) -> dict:
    if state["transfer_amount"] > 1000:
        # Pauses inside child subgraph!
        human_decision = interrupt({
            "action": "WIRE_TRANSFER",
            "amount": state["transfer_amount"],
            "recipient": state["recipient"]
        })
        if not human_decision.get("approved"):
            raise PermissionError("Human rejected transfer!")
    return {"status": "SUCCESS"}''', language="python")

# ==============================================================================
# SECTION 7: SUBGRAPH STATE INSPECTION
# ==============================================================================
st.header("7. Subgraph State Inspection: Inspecting Deep Thread Snapshots and Time Travel", divider="rainbow")

st.markdown("""
To inspect the state of a paused or completed child subgraph, use the `subgraphs=True` parameter on `app.get_state()`:
""")

st.code('''# Inspecting state across all nested subgraph namespaces
snapshot = app.get_state(config, subgraphs=True)

# snapshot.tasks contains tasks across all levels
for task in snapshot.tasks:
    print(f"Task Name: {task.name}")
    print(f"Namespace: {task.name}")
    print(f"Pending Interrupts: {task.interrupts}")''', language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Tier Persistent Subgraph with Human Interrupts", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a parent graph with a persistent child subgraph featuring human approval interrupts.
""")

st.code('''from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# 1. Child Subgraph with HITL Interrupt
class ChildPaymentState(TypedDict):
    amount: float
    vendor: str
    approved: bool
    confirmation_id: Optional[str]

def audit_payment_node(state: ChildPaymentState) -> dict:
    # Trigger interrupt if amount exceeds threshold
    if state["amount"] > 500.0:
        user_approval = interrupt({
            "message": f"Approve payment of ${state['amount']:.2f} to {state['vendor']}?"
        })
        return {"approved": bool(user_approval)}
    return {"approved": True}

def execute_payment_node(state: ChildPaymentState) -> dict:
    if not state.get("approved"):
        return {"confirmation_id": "REJECTED_BY_ADMIN"}
    return {"confirmation_id": f"PAY-TX-SUCCESS-{int(state['amount'])}"}

child_builder = StateGraph(ChildPaymentState)
child_builder.add_node("audit_payment", audit_payment_node)
child_builder.add_node("execute_payment", execute_payment_node)
child_builder.add_edge(START, "audit_payment")
child_builder.add_edge("audit_payment", "execute_payment")
child_builder.add_edge("execute_payment", END)

# 2. Parent Graph with Checkpointer
class ParentWorkflowState(TypedDict):
    invoice_id: str
    amount: float
    vendor: str
    final_status: str

def initiate_node(state: ParentWorkflowState) -> dict:
    return {"final_status": "IN_PROGRESS"}

parent_builder = StateGraph(ParentWorkflowState)
parent_builder.add_node("initiate", initiate_node)
# Add compiled child directly with shared checkpointer!
parent_builder.add_node("payment_subgraph", child_builder.compile())

parent_builder.add_edge(START, "initiate")
parent_builder.add_edge("initiate", "payment_subgraph")
parent_builder.add_edge("payment_subgraph", END)

checkpointer = MemorySaver()
app = parent_builder.compile(checkpointer=checkpointer)

print("Parent and Persistent Child Graph with HITL Interrupt Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Subgraph Checkpoint Namespace & Interrupt Simulator", divider="rainbow")

st.markdown("""
Simulate how interrupts pause execution inside a nested subgraph namespace and observe how resumes resolve.
""")

sim = SubgraphPersistenceSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("💳 Simulation 1: High-Value Subgraph Transfer")
    th_id = st.text_input("Thread ID", value="thread_tx_501")
    transfer_amount = st.number_input("Transfer Amount (USD)", min_value=10.0, max_value=50000.0, value=2500.0, step=100.0)
    enable_hitl = st.toggle("Enable Subgraph Human Approval Guardrail (> $1,000)", value=True)

    if st.button("⚡ Dispatch Workflow"):
        res = sim.execute_subgraph_with_interrupt(
            thread_id=th_id,
            operation_amount=transfer_amount,
            require_human_approval=enable_hitl
        )
        st.session_state["sub_persist_res"] = res

    if "sub_persist_res" in st.session_state and st.session_state["sub_persist_res"]["status"] == "INTERRUPTED":
        st.warning("⚠️ **PAUSED INSIDE CHILD SUBGRAPH**: Human Approval Required!")
        c_app, c_rej = st.columns(2)
        with c_app:
            if st.button("✅ Approve Transfer"):
                resume_res = sim.resume_subgraph_interrupt(True)
                st.session_state["sub_persist_res"] = resume_res
        with c_rej:
            if st.button("❌ Reject Transfer"):
                resume_res = sim.resume_subgraph_interrupt(False)
                st.session_state["sub_persist_res"] = resume_res

with col_sim2:
    st.subheader("🗄️ Checkpoint Namespace Storage Explorer")
    if "sub_persist_res" in st.session_state:
        res = st.session_state["sub_persist_res"]
        st.info(f"Status: **{res['status']}** | {res['message']}")
        
        st.markdown("#### Database Composite Index Records:")
        records = []
        for (ns, ckpt_id), data in sim.thread_checkpoints.items():
            records.append({
                "thread_id": data["thread_id"],
                "checkpoint_ns": f"'{ns}' (Root)" if ns == "" else f"'{ns}' (Child)",
                "checkpoint_id": ckpt_id,
                "has_interrupt": bool(data.get("interrupt"))
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: How does LangGraph distinguish between checkpoints of a parent graph and its child subgraphs?", expanded=False):
    st.markdown("""
    **Answer**:
    Using the `checkpoint_ns` (checkpoint namespace) field in the checkpointer database index `(thread_id, checkpoint_ns, checkpoint_id)`.
    Parent checkpoints have an empty namespace `""`, while child subgraph checkpoints are prefixed with the child's node name (e.g. `"payment_subgraph"`).
    """)

with st.expander("Q2: What happens when a child subgraph triggers an interrupt() call?", expanded=False):
    st.markdown("""
    **Answer**:
    The child node execution pauses, which causes the parent graph's active superstep to pause at that node.
    The checkpointer records the pause state with the child's pending interrupt metadata.
    The client resumes execution by sending `app.invoke(Command(resume=...), config=config)`.
    """)

with st.expander("Q3: When should a subgraph be designed as stateless rather than persistent?", expanded=False):
    st.markdown("""
    **Answer**:
    When the subgraph performs high-frequency, idempotent transformations (e.g., JSON schema validation, regex extraction, small mathematical calculations) where writing checkpoints to PostgreSQL on every step would create unnecessary database I/O latency.
    """)

with st.expander("Q4: How do you inspect the state of a child subgraph using get_state()?", expanded=False):
    st.markdown("""
    **Answer**:
    By calling `app.get_state(config, subgraphs=True)`.
    This returns state snapshots across the root graph and all nested child namespaces, exposing pending tasks and interrupts.
    """)

with st.expander("Q5: What is the difference between Per-Invocation and Per-Thread subgraph persistence?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Per-Invocation Persistence** stores intermediate state in ephemeral RAM for the duration of a single run and discards it when the child completes.
    - **Per-Thread Persistence** writes intermediate state to the durable database checkpointer indexed by `checkpoint_ns`, allowing long-term conversation history and crash recovery across restarts.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Invoking Child with a Different Thread ID")
    st.markdown("""
    *Mechanism*: Manually overriding `thread_id` when invoking a child subgraph inside a persistent parent workflow.  
    *Consequence*: Child checkpoints detach from the parent thread, breaking state inspection and automated recovery.  
    *Fix*: Always forward the parent's `config` directly to child invocations.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Trying to Resume a Subgraph Interrupt with Bare State Input")
    st.markdown("""
    *Mechanism*: Calling `app.invoke({"approved": True}, config=config)` on a thread paused inside a subgraph interrupt.  
    *Consequence*: Overwrites or conflicts with input channels rather than delivering the approval to the paused `interrupt()` statement.  
    *Fix*: Always resume via `Command(resume=True)`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Database Connection Exhaustion from Concurrent Subgraph Checkpoints")
    st.markdown("""
    *Mechanism*: Running 20 parallel subgraphs on a map-reduce superstep, each attempting to write checkpoints to PostgreSQL simultaneously.  
    *Consequence*: Connection pool saturation and database locks.  
    *Fix*: Configure database connection pooling appropriately or use stateless subgraphs for fan-out tasks.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Time Travel Warning): Forking Parent State Without Forking Subgraph Namespaces")
    st.markdown("""
    *Mechanism*: Updating parent state via `app.update_state()` without specifying `as_node` or namespace when modifying child state.  
    *Consequence*: Mutates parent channels instead of child channels.  
    *Fix*: Use `as_node="subgraph_node"` when updating nested states.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**checkpoint_ns**\\nThe database namespace prefix separating parent checkpoints from child subgraph checkpoints.")
with col_fc2:
    st.info("**Stateless Subgraph**\\nA compiled graph without a checkpointer, operating as a zero-overhead pure function.")
with col_fc3:
    st.info("**Nested Interrupt**\\nAn interrupt() call originating inside a child subgraph, pausing the entire hierarchy.")
with col_fc4:
    st.info("**Command(resume=...)**\\nThe mechanism used to deliver human decisions directly to paused subgraph interrupts.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the role of checkpoint_ns in composite database keys.", value=True)
c2 = st.checkbox("I know the difference between per-invocation, per-thread, and stateless persistence.", value=True)
c3 = st.checkbox("I know what happens when interrupt() is called from inside a child subgraph.", value=True)
c4 = st.checkbox("I know how to resume a nested subgraph interrupt using Command(resume=...).", value=True)
c5 = st.checkbox("I understand how to inspect nested subgraph states using get_state(subgraphs=True).", value=True)
c6 = st.checkbox("I know why stateless subgraphs are preferred for high-throughput map-reduce nodes.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 19.3 'Subgraph Persistence' with full architectural and practical comprehension!")
