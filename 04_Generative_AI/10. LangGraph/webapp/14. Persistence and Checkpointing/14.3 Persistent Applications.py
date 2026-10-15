import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="14.3 Persistent Applications - LangGraph Curriculum",
    page_icon="🛡️",
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
    .badge-durable {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-crash {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
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
class DurableWorkflowSimulatorEngine:
    """
    Simulates durable workflow patterns:
    1. Resuming after server crash
    2. Human approval pauses
    3. Multi-day long-running state continuity
    """
    def __init__(self):
        self.workflow_state = "IDLE"
        self.step_history = []

    def simulate_workflow_step(self, step_action: str) -> Dict[str, Any]:
        if step_action == "START_JOB":
            self.workflow_state = "RUNNING_PHASE_1"
            self.step_history = [
                {"step": 1, "node": "data_ingest", "status": "COMMITTED", "detail": "100k records staged in checkpoint"}
            ]
        elif step_action == "CRASH_SERVER":
            self.workflow_state = "CRASHED"
            self.step_history.append(
                {"step": 2, "node": "data_transform", "status": "FAILED", "detail": "SIGKILL received (Container terminated)"}
            )
        elif step_action == "RESUME_JOB":
            self.workflow_state = "RECOVERED"
            self.step_history.append(
                {"step": 2, "node": "data_transform", "status": "COMMITTED", "detail": "Recovered from Checkpoint 1; re-executed step 2 cleanly"}
            )
            self.step_history.append(
                {"step": 3, "node": "human_gate", "status": "PAUSED", "detail": "Waiting for supervisor approval via interrupt()"}
            )
        elif step_action == "APPROVE_HUMAN":
            self.workflow_state = "COMPLETED"
            self.step_history.append(
                {"step": 4, "node": "commit_publish", "status": "COMMITTED", "detail": "Supervisor signed off; published to production"}
            )

        return {
            "current_state": self.workflow_state,
            "steps": self.step_history
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Durable Execution and Fault-Tolerant Application Topologies", divider="rainbow")

st.markdown("""
A **Persistent Application** leverages checkpointers to provide **Durable Execution**:
a paradigm where code execution state is treated as a persistent database record.
If a virtual machine restarts, a network cable is unplugged, or an API crashes, execution does not restart from the beginning.
The runtime re-hydrates the exact state from the latest checkpoint and resumes seamlessly.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    durable_blueprint_dot = """
    digraph DurableExecution {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        STEP_1 [label="Node 1: Extract Data\\n(Committed to Checkpoint 1)", fillcolor="#064E3B", color="#34D399"];
        STEP_2 [label="Node 2: Transform Data\\n(Committed to Checkpoint 2)", fillcolor="#064E3B", color="#34D399"];

        subgraph cluster_crash {
            label = "Simulated Hardware Failure";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#DC2626";
            fontcolor = "#F87171";

            CRASH_EVENT [label="Node 3: Compute Embeddings\\n🚨 SERVER CRASH (OOM / Pod Eviction)", fillcolor="#3B1A1A", color="#F87171"];
        }

        subgraph cluster_recovery {
            label = "Automated State Machine Recovery";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#38BDF8";
            fontcolor = "#38BDF8";

            RELOAD_CKPT [label="Worker Reboot: Load Checkpoint 2\\nReads last good state from Postgres", fillcolor="#2E1065", color="#C084FC"];
            RETRY_NODE3 [label="Re-execute Node 3 Cleanly\\n(Zero data loss / No restart from step 1)", fillcolor="#0F172A", color="#38BDF8"];
        }

        HUMAN_APPROVAL [label="Node 4: Human Sign-Off Gate\\ninterrupt() pauses for 48 hours", fillcolor="#431407", color="#FB923C"];
        PUBLISH_FINAL [label="Node 5: Publish Results\\n(Workflow Completed)", fillcolor="#064E3B", color="#34D399"];

        STEP_1 -> STEP_2;
        STEP_2 -> CRASH_EVENT;
        CRASH_EVENT -> RELOAD_CKPT [label="On Worker Respawn", style=dashed, color="#F87171"];
        RELOAD_CKPT -> RETRY_NODE3;
        RETRY_NODE3 -> HUMAN_APPROVAL;
        HUMAN_APPROVAL -> PUBLISH_FINAL [label="Command(resume=True)"];
    }
    """
    st.graphviz_chart(durable_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Durable Execution Principle:</b> In traditional software, state is lost when the process terminates.
In LangGraph durable execution, the process is merely a transient compute worker; the checkpoint database
is the source of truth.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: RESUMING FAILED WORKFLOWS
# ==============================================================================
st.header("2. Resuming Failed Workflows: Node Crash Recovery and State Reconciliation", divider="rainbow")

st.markdown("""
When a worker crashes during execution, resuming the workflow requires only invoking the graph
with the same `thread_id` and passing `None` as the input:
""")

st.code("""
# Resuming a stalled or crashed workflow:
config = {"configurable": {"thread_id": "batch_migration_thread_10"}}

# Passing None commands LangGraph to resume execution from the pending 'next' nodes
# recorded in the latest checkpoint:
resumed_result = app.invoke(None, config=config)
""", language="python")

# ==============================================================================
# SECTION 3: CONVERSATION PERSISTENCE
# ==============================================================================
st.header("3. Conversation Persistence: Continuous Multi-Session Chat Experiences", divider="rainbow")

st.markdown("""
Persistent applications maintain continuous context across days and weeks.
Because state is tied to `thread_id`, a customer service conversation can pause, the user can close their browser,
and resume three days later with full conversational fidelity.
""")

# ==============================================================================
# SECTION 4: HUMAN APPROVAL WORKFLOWS
# ==============================================================================
st.header("4. Human Approval Workflows: Stateful Pausing and Asynchronous Resumption", divider="rainbow")

st.markdown("""
Long-running agents often require human review before executing irreversible actions.
Using `interrupt()`, the graph pauses execution and persists the state snapshot:
""")

st.code("""
from langgraph.types import interrupt, Command

def approval_gate_node(state: MessagesState) -> dict:
    # 1. Pause execution and persist state snapshot
    decision = interrupt({
        "action": "Execute Wire Transfer",
        "amount_usd": 15000.0,
        "recipient": "Vendor Corp"
    })
    
    # 2. Resumes here when human submits approval!
    if decision.get("approved"):
        return {"messages": [AIMessage(content="Transfer approved by supervisor.")]}
    else:
        return {"messages": [AIMessage(content="Transfer rejected.")]}

# To resume the graph externally:
app.invoke(Command(resume={"approved": True}), config=config)
""", language="python")

# ==============================================================================
# SECTION 5: LONG-RUNNING AGENTS
# ==============================================================================
st.header("5. Long-Running Agents: Multi-Day Background Research and Batch Jobs", divider="rainbow")

st.markdown("""
Long-running agents handle asynchronous tasks taking hours or days (e.g. web scraping 5,000 pages,
compiling market analysis, or awaiting external webhooks).
Because state is checkpointed at every step, infrastructure autoscaling can safely spin pods down
during idle waits and spin them up when events arrive.
""")

# ==============================================================================
# SECTION 6: FAULT TOLERANCE
# ==============================================================================
st.header("6. Fault Tolerance: Network Partition Resilience and Zero-Loss Guarantees", divider="rainbow")

st.markdown("""
Fault-tolerant persistent applications combine:
1. **Idempotent Nodes**: Ensuring repeated node execution produces identical outcomes.
2. **Transaction Isolation**: Storing intermediate writes in `checkpoint_writes` before superstep commit.
3. **Dead-Letter Resumption**: Rerouting permanently failed nodes to alert channels.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise Durable Workflow Engine with Human Approval", divider="rainbow")

st.markdown("""
Below is the complete, runnable implementation of a durable workflow featuring
crash resumption, state inspection, and human approval interrupts:
""")

st.code("""
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. State Definition
class DurableDeploymentState(MessagesState):
    deployment_id: str
    stage: str
    approved: bool

# 2. Nodes
def prepare_deployment_node(state: DurableDeploymentState) -> dict:
    return {
        "stage": "STAGED",
        "messages": [AIMessage(content=f"Deployment {state['deployment_id']} built and passed unit tests.")]
    }

def human_gate_node(state: DurableDeploymentState) -> dict:
    # Interrupts and awaits human sign-off
    human_response = interrupt({"request": f"Approve release of {state['deployment_id']} to production?"})
    return {
        "approved": human_response.get("approved", False),
        "stage": "APPROVED" if human_response.get("approved") else "REJECTED"
    }

def release_node(state: DurableDeploymentState) -> dict:
    if state["approved"]:
        return {"messages": [AIMessage(content="Release deployed to production successfully!")]}
    return {"messages": [AIMessage(content="Release cancelled by supervisor.")]}

# 3. Assemble Graph
builder = StateGraph(DurableDeploymentState)
builder.add_node("prepare", prepare_deployment_node)
builder.add_node("gate", human_gate_node)
builder.add_node("release", release_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "gate")
builder.add_edge("gate", "release")
builder.add_edge("release", END)

checkpointer = MemorySaver()
deployment_app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Crash Recovery & Human Approval Simulation Lab", divider="rainbow")

st.markdown("""
Test durable execution scenarios: start a multi-step job, simulate a server crash,
resume execution from the last checkpoint, and complete a human approval gate.
""")

if "durable_eng" not in st.session_state:
    st.session_state.durable_eng = DurableWorkflowSimulatorEngine()

d_eng: DurableWorkflowSimulatorEngine = st.session_state.durable_eng

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Lifecycle Actions")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("1. Start Multi-Step Job"):
            d_eng.simulate_workflow_step("START_JOB")
            st.rerun()
    with col_a2:
        if st.button("2. 🔥 Simulate Server Crash"):
            d_eng.simulate_workflow_step("CRASH_SERVER")
            st.rerun()

    col_a3, col_a4 = st.columns(2)
    with col_a3:
        if st.button("3. 🔄 Resume from Checkpoint"):
            d_eng.simulate_workflow_step("RESUME_JOB")
            st.rerun()
    with col_a4:
        if st.button("4. ✅ Sign Off Human Approval"):
            d_eng.simulate_workflow_step("APPROVE_HUMAN")
            st.rerun()

    if st.button("Reset Simulation Lab"):
        d_eng.workflow_state = "IDLE"
        d_eng.step_history = []
        st.rerun()

with wb_col2:
    st.subheader("📋 Durable State Machine Telemetry")
    status_badge = "badge-crash" if d_eng.workflow_state == "CRASHED" else "badge-durable"
    st.markdown(f"**Current State**: <span class='{status_badge}'>{d_eng.workflow_state}</span>", unsafe_allow_html=True)
    
    st.markdown("#### Chronological Execution Log")
    if d_eng.step_history:
        for item in d_eng.step_history:
            st.markdown(f"**Step {item['step']} — Node: `{item['node']}`** `[{item['status']}]`")
            st.write(f"- Detail: {item['detail']}")
            st.divider()
    else:
        st.info("Workflow is IDLE. Click '1. Start Multi-Step Job' to begin.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: How do you resume a paused or crashed LangGraph workflow without re-running earlier nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    Invoke the graph with `app.invoke(None, config=config)`.
    By passing `None` as input alongside the `thread_id` config, LangGraph inspects the latest checkpoint's `next` field
    and resumes execution strictly from the pending nodes, preserving all previous state.
    """)

with st.expander("Q2: What is the purpose of Command(resume=...) in human approval workflows?", expanded=False):
    st.markdown("""
    **Answer**:
    When a graph is paused via `interrupt()`, it awaits external input.
    `Command(resume=payload)` supplies that external payload directly to the paused node function,
    allowing execution to resume immediately from that point.
    """)

with st.expander("Q3: Why must nodes in durable workflows be designed to be idempotent?", expanded=False):
    st.markdown("""
    **Answer**:
    If a worker crashes halfway through a node or during network transmission, the recovery engine may re-run that node.
    If the node performs side-effects (e.g. charging a credit card or sending an email) without idempotency keys,
    the customer may be billed twice upon resumption.
    """)

with st.expander("Q4: Can a workflow paused with interrupt() remain in storage for multiple days?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Because the checkpoint is written to database storage (Postgres), the Python process can shut down completely.
    Days or weeks later, an HTTP webhook can invoke `Command(resume=...)` to pick up execution.
    """)

with st.expander("Q5: What is the difference between durable execution and basic database logging?", expanded=False):
    st.markdown("""
    **Answer**:
    Database logging merely records audit logs.
    Durable execution is an active control flow system: it serializes the entire state machine, manages pending branches,
    and programmatically re-instantiates execution graphs from checkpoints.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Non-Idempotent Side-Effects During Workflow Resumption")
    st.markdown("""
    *Mechanism*: Executing external REST mutations inside a node without idempotency keys.  
    *Consequence*: If the server crashes and resumes from checkpoint, the mutation executes multiple times.  
    *Fix*: Always pass unique transaction or idempotency UUIDs to external APIs.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing Input When Resuming Paused Interrupts")
    st.markdown("""
    *Mechanism*: Calling `app.invoke(None, config=config)` on a graph paused by `interrupt()`.  
    *Consequence*: Graph raises an error because `interrupt()` requires resumption payload.  
    *Fix*: Always use `app.invoke(Command(resume=data), config=config)`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Retaining Stalled Checkpoints Indefinitely")
    st.markdown("""
    *Mechanism*: Human approval requests that are abandoned by users remaining in database forever.  
    *Consequence*: Accumulates thousands of orphaned active threads.  
    *Fix*: Implement TTL timeouts that automatically cancel or expire stalled approval threads.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Attempting Interrupts Without Checkpointers")
    st.markdown("""
    *Mechanism*: Using `interrupt()` on a graph compiled without a checkpointer.  
    *Consequence*: Immediate runtime failure (`ValueError: interrupt requires checkpointer`).  
    *Fix*: Always provide a checkpointer during compilation.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Durable Execution**\\nA paradigm where process state is persisted in databases, allowing zero-loss crash recovery.")
with col_fc2:
    st.info("**Command(resume=...)**\\nThe invocation primitive used to resume graphs paused by interrupt() calls.")
with col_fc3:
    st.info("**Idempotency**\\nEnsuring a node operation produces the exact same outcome even if re-executed after a crash.")
with col_fc4:
    st.info("**app.invoke(None)**\\nThe pattern used to resume a paused or crashed graph from its pending next nodes.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how durable execution allows workflows to resume seamlessly after crashes.", value=True)
c2 = st.checkbox("I understand how to resume workflows using app.invoke(None, config=config).", value=True)
c3 = st.checkbox("I can implement human approval gates with interrupt() and Command(resume=...).", value=True)
c4 = st.checkbox("I understand why idempotency is critical for nodes that perform side-effects.", value=True)
c5 = st.checkbox("I know how long-running agents can span multiple days across container restarts.", value=True)
c6 = st.checkbox("I understand the difference between active durable execution and passive audit logging.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 14.3 'Persistent Applications' with full architectural and practical comprehension!")
