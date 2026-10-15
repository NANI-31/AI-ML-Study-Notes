import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="16.1 HITL Fundamentals - LangGraph Curriculum",
    page_icon="🧑‍💼",
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
    .badge-hitl {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-reject {
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
class HITLSimulatorEngine:
    """
    Simulates the core HITL interactions:
    Approval, Rejection, State Editing, and Feedback Ingestion.
    """
    def __init__(self):
        self.pending_actions = {
            "tx_104": {
                "action": "Wire Transfer",
                "recipient": "Supplier Corp",
                "original_amount_usd": 18500.0,
                "risk_score": "HIGH",
                "status": "AWAITING_REVIEW"
            }
        }

    def process_human_decision(
        self,
        tx_id: str,
        decision: str,
        modified_amount: Optional[float] = None,
        feedback_text: Optional[str] = None
    ) -> Dict[str, Any]:
        tx = self.pending_actions.get(tx_id, {})
        final_amount = modified_amount if modified_amount is not None else tx.get("original_amount_usd", 0.0)

        if decision == "APPROVE":
            status = "APPROVED"
            detail = f"Supervisor approved action: Dispatched {final_amount:.2f} USD to {tx.get('recipient')}."
        elif decision == "REJECT":
            status = "REJECTED"
            detail = f"Supervisor rejected action. Transaction aborted cleanly. Reason: {feedback_text or 'Policy violation'}."
        else: # EDIT
            status = "MODIFIED_AND_APPROVED"
            detail = f"Supervisor edited amount from {tx.get('original_amount_usd')} to {final_amount:.2f} USD and approved execution."

        return {
            "tx_id": tx_id,
            "decision": decision,
            "status": status,
            "execution_detail": detail,
            "feedback_logged": feedback_text or "None"
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Human-in-the-Loop (HITL) Triad and Governance Model", divider="rainbow")

st.markdown("""
**Human-in-the-Loop (HITL)** is the design paradigm that places human oversight at critical junction points in an agent's workflow.
In LangGraph, HITL is natively integrated into the state graph:
the graph pauses execution, saves a persistent checkpoint snapshot, emits an approval payload to an external interface,
and awaits human intervention (**Approval**, **Rejection**, or **State Editing**) before proceeding.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    hitl_blueprint_dot = """
    digraph HITLArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        AGENT_PLAN [label="Node: Agent Plans Action\\nEmits tool_calls: [execute_wire_transfer(amount=18500)]", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_checkpoint_gate {
            label = "HITL Execution Gate (interrupt)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            PAUSE_SNAPSHOT [label="interrupt(payload)\\n1. Execution paused\\n2. Checkpoint committed to Postgres\\n3. Webhook paged to Supervisor UI", fillcolor="#2E1065", color="#C084FC"];
            HUMAN_ACTOR [label="Human Supervisor Review\\nEvaluates parameters in web portal", shape=ellipse, fillcolor="#431407", color="#FB923C"];
            RESUME_COMMAND [label="Command(resume=decision)\\nInjects approval, edit, or rejection", fillcolor="#064E3B", color="#34D399"];

            PAUSE_SNAPSHOT -> HUMAN_ACTOR [label="Presents UI"];
            HUMAN_ACTOR -> RESUME_COMMAND [label="Submits decision"];
        }

        subgraph cluster_routes {
            label = "Post-Approval Execution Routing";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            EXECUTE_TOOL [label="Approved: Execute Tool\\nWire transfer committed to bank API", fillcolor="#064E3B", color="#34D399"];
            ABORT_PATH [label="Rejected: Abort & Notify\\nRollback transaction & notify user", fillcolor="#3B1A1A", color="#F87171"];
        }

        AGENT_PLAN -> PAUSE_SNAPSHOT;
        RESUME_COMMAND -> EXECUTE_TOOL [label="decision == 'approve'"];
        RESUME_COMMAND -> ABORT_PATH [label="decision == 'reject'"];
    }
    """
    st.graphviz_chart(hitl_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Governance Guarantee:</b> High-risk side-effects (financial transactions, deleting infrastructure, publishing content)
never execute autonomously. The agent prepares the action, but only a verified human can authorize the release.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: WHY HUMAN-IN-THE-LOOP?
# ==============================================================================
st.header("2. Why Human-in-the-Loop?: Safety Guardrails, High-Liability Operations & Alignment", divider="rainbow")

st.markdown("""
Even frontier models suffer from hallucinations, prompt injections, and edge-case ambiguity.
HITL provides:
1. **Liability Mitigation**: Regulatory frameworks (e.g. EU AI Act) mandate human oversight on credit, medical, and legal decisions.
2. **Adversarial Defense**: Prevents malicious prompt injections from forcing the agent to execute unauthorized wire transfers.
3. **Continuous Alignment**: Human corrections provide high-value feedback loops for model fine-tuning.
""")

# ==============================================================================
# SECTION 3: HUMAN APPROVAL
# ==============================================================================
st.header("3. Human Approval: Binary Gatekeeper Patterns and Sign-Off Checkpoints", divider="rainbow")

st.markdown("""
In a binary approval pattern, the human answers a simple Yes/No question:
""")

st.code("""
from langgraph.types import interrupt

def gatekeeper_node(state: MessagesState) -> dict:
    # Pauses and passes payload to human
    approved = interrupt("Do you authorize transferring 18,500 USD to Supplier Corp?")
    
    if approved:
        return {"status": "AUTHORIZED"}
    return {"status": "DENIED"}
""", language="python")

# ==============================================================================
# SECTION 4: HUMAN REJECTION
# ==============================================================================
st.header("4. Human Rejection: Cancellation Handling, Fallback Routing, and Abort Paths", divider="rainbow")

st.markdown("""
When a human rejects an action, the graph must handle the cancellation gracefully:
- Log the supervisor's rejection rationale into the message history.
- Notify the end-user politely without crashing.
- Suggest alternative actions.
""")

# ==============================================================================
# SECTION 5: HUMAN EDITING
# ==============================================================================
st.header("5. Human Editing: Modifying State and Tool Arguments Before Action Execution", divider="rainbow")

st.markdown("""
Rather than rejecting an entire request, supervisors often want to **edit tool arguments in-flight**:
(e.g., reducing a requested refund from 500 USD to 150 USD).
LangGraph enables the human to supply modified parameters directly through `Command(resume=...)`:
""")

st.code("""
# Resuming with edited parameters:
app.invoke(
    Command(resume={"action": "APPROVE", "edited_amount": 150.0}),
    config=config
)
""", language="python")

# ==============================================================================
# SECTION 6: HUMAN FEEDBACK
# ==============================================================================
st.header("6. Human Feedback: Incorporating Guidance and Clarification into Graph History", divider="rainbow")

st.markdown("""
In conversational workflows, the human can inject guidance or clarifying instructions:
*"Focus exclusively on European market regulations for this report."*
The feedback is appended as a `HumanMessage` into the active state channel.
""")

# ==============================================================================
# SECTION 7: HUMAN INTERVENTION
# ==============================================================================
st.header("7. Human Intervention: Out-of-Band State Overrides and Mid-Flight Corrections", divider="rainbow")

st.markdown("""
LangGraph allows administrators to update the state of a paused graph at any time using `update_state()`:
""")

st.code("""
# Manually overriding graph state from outside the node:
app.update_state(
    config,
    {"risk_level": "CRITICAL", "messages": [HumanMessage(content="Supervisor escalated ticket.")]}
)
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Action HITL Supervisory Workflow", divider="rainbow")

st.markdown("""
Below is the complete implementation of an enterprise HITL approval workflow:
""")

st.code("""
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. State Definition
class HITLApprovalState(MessagesState):
    amount_usd: float
    supervisor_decision: str

# 2. Nodes
def plan_action_node(state: HITLApprovalState) -> dict:
    return {
        "amount_usd": 18500.0,
        "messages": [AIMessage(content="Planned wire transfer of 18,500 USD. Submitting to supervisor.")]
    }

def human_approval_node(state: HITLApprovalState) -> dict:
    # Pause graph and await human decision
    review = interrupt({
        "prompt": "Authorize Wire Transfer",
        "amount": state["amount_usd"]
    })
    return {
        "supervisor_decision": review.get("decision", "REJECT"),
        "amount_usd": review.get("edited_amount", state["amount_usd"])
    }

def execute_transfer_node(state: HITLApprovalState) -> dict:
    if state["supervisor_decision"] == "APPROVE":
        return {"messages": [AIMessage(content=f"SUCCESS: Transferred {state['amount_usd']:.2f} USD.")]}
    return {"messages": [AIMessage(content="ABORTED: Transfer rejected by supervisor.")]}

# 3. Assemble Graph
builder = StateGraph(HITLApprovalState)
builder.add_node("plan", plan_action_node)
builder.add_node("approval", human_approval_node)
builder.add_node("execute", execute_transfer_node)

builder.add_edge(START, "plan")
builder.add_edge("plan", "approval")
builder.add_edge("approval", "execute")
builder.add_edge("execute", END)

checkpointer = MemorySaver()
hitl_app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Human Approval, Rejection & Editing Simulation Console", divider="rainbow")

st.markdown("""
Experience the supervisor review console. Inspect an in-flight financial transaction,
choose whether to **Approve**, **Reject**, or **Edit** the transaction amount, and observe execution.
""")

hitl_eng = HITLSimulatorEngine()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("🧑‍💼 Supervisor Decision Gate")
    target_tx = "tx_104"
    tx_meta = hitl_eng.pending_actions[target_tx]
    
    st.info(f"**Transaction ID**: `{target_tx}`\\n- **Recipient**: {tx_meta['recipient']}\\n- **Original Amount**: {tx_meta['original_amount_usd']:.2f} USD\\n- **Risk Score**: {tx_meta['risk_score']}")

    chosen_decision = st.radio("Supervisor Action:", ["APPROVE", "REJECT", "EDIT_AND_APPROVE"])
    
    edit_val = None
    feedback = None
    if chosen_decision == "EDIT_AND_APPROVE":
        edit_val = st.number_input("Override Approved Amount (USD):", value=12000.0, step=500.0)
    elif chosen_decision == "REJECT":
        feedback = st.text_input("Rejection Rationale:", value="Amount exceeds quarterly supplier cap.")

    submit_btn = st.button("🚀 Commit Human Decision", type="primary")

with wb_c2:
    st.subheader("📋 Execution Feedback")
    if submit_btn:
        res = hitl_eng.process_human_decision(
            target_tx,
            "APPROVE" if chosen_decision == "APPROVE" else ("REJECT" if chosen_decision == "REJECT" else "EDIT"),
            modified_amount=edit_val,
            feedback_text=feedback
        )
        
        badge = "badge-reject" if res["decision"] == "REJECT" else "badge-hitl"
        st.markdown(f"### Status: <span class='{badge}'>{res['status']}</span>", unsafe_allow_html=True)
        st.write(f"- Detail: {res['execution_detail']}")
        st.write(f"- Feedback Logged: `{res['feedback_logged']}`")
    else:
        st.info("Select an action and click 'Commit Human Decision' to simulate supervisor sign-off.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: How does interrupt() differ from standard Python input()?", expanded=False):
    st.markdown("""
    **Answer**:
    `input()` blocks the local thread and holds system resources in memory.
    `interrupt()` serializes the state to the checkpointer database, exits the thread cleanly, and frees all memory.
    The process can restart or shut down while waiting for external human input hours or days later.
    """)

with st.expander("Q2: What is the exact syntax for resuming an interrupted graph?", expanded=False):
    st.markdown("""
    **Answer**:
    `app.invoke(Command(resume=payload), config=config)`.
    The `payload` passed to `Command(resume=...)` is directly returned by the paused `interrupt()` expression inside the node.
    """)

with st.expander("Q3: Can a human edit messages or variables in state without writing a custom node?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! By calling `app.update_state(config, {"messages": [edited_message]}, as_node="node_name")`.
    This forks or updates the checkpoint snapshot directly from external administrative tools.
    """)

with st.expander("Q4: Why does interrupt() require a persistent checkpointer?", expanded=False):
    st.markdown("""
    **Answer**:
    Because execution is paused asynchronously. Without a checkpointer, there is no storage medium
    to preserve the state snapshot while waiting for human review.
    """)

with st.expander("Q5: What happens if a human approval request is abandoned and never answered?", expanded=False):
    st.markdown("""
    **Answer**:
    The thread remains safely paused in the checkpointer database at the interrupt state.
    In production, a background cron job monitors idle interrupted threads and expires them after a defined TTL.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting to Pass Command(resume=...) on Interrupted Threads")
    st.markdown("""
    *Mechanism*: Calling `app.invoke(user_input, config=config)` on a thread paused at an `interrupt()`.  
    *Consequence*: Graph raises `GraphInterruptError: Cannot invoke thread awaiting interrupt resumption`.  
    *Fix*: Always resume with `Command(resume=data)`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing Schema Validation on Human Input Payloads")
    st.markdown("""
    *Mechanism*: Assuming human input from external web forms will always match the expected dictionary schema.  
    *Consequence*: Downstream nodes crash with `KeyError` or `TypeError`.  
    *Fix*: Validate human resumption payloads with Pydantic before updating state.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Using Interrupts for Trivial Actions")
    st.markdown("""
    *Mechanism*: Pausing the graph to request approval for harmless read-only queries.  
    *Consequence*: Destroys user experience and creates review fatigue for supervisors.  
    *Fix*: Gate only high-liability write/delete operations behind interrupts.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Silent Interrupt Pauses")
    st.markdown("""
    *Mechanism*: Triggering `interrupt()` without sending an alert (email/Slack) to the supervisor.  
    *Consequence*: Workflow sits idle in the database indefinitely.  
    *Fix*: Integrate automated notification webhooks into the interrupt handler.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**HITL**\\nHuman-in-the-Loop: Architecture embedding human oversight at critical decision boundaries.")
with col_fc2:
    st.info("**interrupt()**\\nThe LangGraph function that pauses execution and emits an approval payload.")
with col_fc3:
    st.info("**Command(resume=...)**\\nThe invocation primitive used to resume graphs paused by interrupt().")
with col_fc4:
    st.info("**update_state()**\\nMethod allowing external administrative tools to mutate state in persistent checkpoints.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know why HITL is mandatory for high-liability and irreversible agent actions.", value=True)
c2 = st.checkbox("I understand how interrupt() pauses execution without consuming memory.", value=True)
c3 = st.checkbox("I know how to resume an interrupted graph using Command(resume=...).", value=True)
c4 = st.checkbox("I can implement human editing of tool arguments in-flight.", value=True)
c5 = st.checkbox("I understand how update_state() enables out-of-band state modifications.", value=True)
c6 = st.checkbox("I know how to prevent stalled threads using TTL expiration policies.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 16.1 'HITL Fundamentals' with full architectural and practical comprehension!")
