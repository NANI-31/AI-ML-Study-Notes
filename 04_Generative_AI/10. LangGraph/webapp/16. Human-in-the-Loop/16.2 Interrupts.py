import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="16.2 Interrupts - LangGraph Curriculum",
    page_icon="⏸️",
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
    .badge-pause {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-resume {
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
class InterruptLifecycleEngine:
    """
    Simulates the exact interrupt() and Command(resume=...) cycle.
    """
    def __init__(self):
        self.paused_state = None

    def trigger_interrupt(self, action_description: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.paused_state = {
            "status": "INTERRUPTED",
            "action": action_description,
            "payload": payload,
            "checkpoint_id": "1ef9921_paused_snap"
        }
        return self.paused_state

    def resume_interrupt(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.paused_state:
            return {"status": "ERROR", "message": "No active interrupt pending."}

        res = {
            "status": "RESUMED_AND_COMPLETED",
            "action_executed": self.paused_state["action"],
            "decision_injected": resume_data,
            "result_message": f"Execution successfully resumed with decision: {resume_data.get('decision')}."
        }
        self.paused_state = None
        return res


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The interrupt() State Machine Lifecycle", divider="rainbow")

st.markdown("""
In LangGraph v0.2+, **`interrupt()`** replaces legacy static breakpoint configurations.
`interrupt()` can be called **anywhere inside any node function**.
When invoked, it:
1. Emits an **interrupt payload** (the review data intended for external callers).
2. Suspends node execution and writes an atomic state snapshot to the checkpointer.
3. Exits the graph execution without error.
4. Resumes seamlessly when the host application invokes `Command(resume=payload)`.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    interrupt_blueprint_dot = """
    digraph InterruptStateLifecycle {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        START_NODE [label="Graph Invocation: app.invoke(input, config)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_node_inside {
            label = "Node Execution Function: def billing_node(state)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            PRE_INTERRUPT [label="1. Prepare Action Data\\namount = state['amount']", fillcolor="#0F172A", color="#38BDF8"];
            CALL_INTERRUPT [label="2. decision = interrupt({'amount': amount})\\n🚨 Graph Execution Suspended!", fillcolor="#3B1A1A", color="#F87171"];
            POST_INTERRUPT [label="4. Resumes with decision value!\\nif decision['approved']: execute_charge()", fillcolor="#064E3B", color="#34D399"];

            PRE_INTERRUPT -> CALL_INTERRUPT;
            CALL_INTERRUPT -> POST_INTERRUPT [label="On Command(resume=...)", style=dashed, color="#34D399"];
        }

        subgraph cluster_external {
            label = "External Host Application / Client UI";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            INSPECT_SNAPSHOT [label="Client inspects state: state = app.get_state(config)\\nReads payload from state.tasks[0].interrupts[0].value", fillcolor="#2E1065", color="#C084FC"];
            SUBMIT_RESUME [label="3. Client calls: app.invoke(\\nCommand(resume={'approved': True}), config=config)", fillcolor="#142E1F", color="#34D399"];

            INSPECT_SNAPSHOT -> SUBMIT_RESUME [label="Supervisor clicks Approve"];
        }

        START_NODE -> PRE_INTERRUPT;
        CALL_INTERRUPT -> INSPECT_SNAPSHOT [label="Emits payload to client"];
        SUBMIT_RESUME -> POST_INTERRUPT [label="Injected back into node"];
    }
    """
    st.graphviz_chart(interrupt_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Core Innovation:</b> Legacy frameworks required hardcoding breakpoints at fixed node boundaries.
<code>interrupt()</code> allows dynamic, conditional interrupts deep within complex conditional logic
(e.g., interrupt only if <code>amount > 10,000</code> or <code>confidence < 0.7</code>).
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: INTERRUPT()
# ==============================================================================
st.header("2. interrupt(): The Native LangGraph Pausing Primitive", divider="rainbow")

st.markdown("""
`interrupt()` is imported directly from `langgraph.types`:
""")

st.code("""
from langgraph.types import interrupt

def security_gate_node(state: MessagesState) -> dict:
    user_request = state["messages"][-1].content
    
    # Conditional interrupt triggered only for high-risk operations
    if "drop table" in user_request.lower() or "delete" in user_request.lower():
        # Execution pauses here!
        human_approval = interrupt({
            "warning": "High risk database mutation requested",
            "query": user_request
        })
        
        # When resumed, human_approval holds the payload passed to Command(resume=...)
        if human_approval.get("allowed"):
            return {"messages": [AIMessage(content="Database operation authorized and executed.")]}
        else:
            return {"messages": [AIMessage(content="Operation cancelled by administrator.")]}
            
    return {"messages": [AIMessage(content="Standard query executed.")]}
""", language="python")

# ==============================================================================
# SECTION 3: INTERRUPT PAYLOADS
# ==============================================================================
st.header("3. Interrupt Payloads: Structuring Data for External Reviewers", divider="rainbow")

st.markdown("""
The argument passed to `interrupt(payload)` is serialized and presented to external systems.
Best practices for structuring payloads:
1. **Action Intent**: What action is the agent attempting to execute?
2. **Parameters**: The exact values and arguments (e.g. transfer amount, recipient).
3. **Risk Rationale**: Why did the agent trigger an interrupt?
""")

payload_df = pd.DataFrame([
    {
        "Field": "action",
        "Type": "str",
        "Example": "'database_schema_drop'"
    },
    {
        "Field": "parameters",
        "Type": "Dict[str, Any]",
        "Example": "{'target_table': 'customer_pii', 'cascade': True}"
    },
    {
        "Field": "risk_assessment",
        "Type": "str",
        "Example": "'Irreversible schema change; requires Level 2 DBA approval.'"
    }
])
st.table(payload_df)

# ==============================================================================
# SECTION 4: INTERRUPT STATE
# ==============================================================================
st.header("4. Interrupt State: Inspecting the Paused Checkpoint via tasks[].interrupts", divider="rainbow")

st.markdown("""
When a graph is interrupted, you can inspect the pending interrupt via `get_state()`:
""")

st.code("""
state = app.get_state(config)

# State snapshot details
print("Is graph paused?", len(state.tasks) > 0 and len(state.tasks[0].interrupts) > 0)

# Inspect payload
pending_interrupt = state.tasks[0].interrupts[0]
print("Interrupt Payload:", pending_interrupt.value)
# Output: {'warning': 'High risk database mutation...', 'query': '...'}
""", language="python")

# ==============================================================================
# SECTION 5: RESUMING EXECUTION
# ==============================================================================
st.header("5. Resuming Execution: The Command Dispatch Lifecycle", divider="rainbow")

st.markdown("""
Resuming an interrupted graph requires dispatching a `Command` object:
""")

st.code("""
from langgraph.types import Command

# Resuming the graph with the reviewer's authorization
app.invoke(
    Command(resume={"allowed": True, "approved_by": "dba_admin_jane"}),
    config=config
)
""", language="python")

# ==============================================================================
# SECTION 6: COMMAND(RESUME=...)
# ==============================================================================
st.header("6. Command(resume=...): Injecting External Input Back into Nodes", divider="rainbow")

st.markdown("""
How does data flow back into the node?
The argument passed to `Command(resume=VALUE)` becomes the **direct return value** of the
`interrupt()` call inside the node function:
""")

st.code("""
# Inside node:
decision = interrupt(...)  # Evaluates to whatever was passed to resume!
# If caller executed: Command(resume={"approved": False})
# Then: decision == {"approved": False}
""", language="python")

# ==============================================================================
# SECTION 7: INTERRUPT HANDLING
# ==============================================================================
st.header("7. Interrupt Handling: Error Trapping, Timeouts, and Re-Interrupt Cycles", divider="rainbow")

st.markdown("""
Advanced interrupt workflows support:
- **Multiple Interrupts in One Node**: An agent can call `interrupt()` multiple times sequentially.
- **Validation Re-Interrupts**: If human input is invalid, the node can re-interrupt with an error message:
""")

st.code("""
while True:
    input_data = interrupt("Please enter your 6-digit confirmation code:")
    if len(str(input_data)) == 6 and str(input_data).isdigit():
        break # Valid!
    # Re-loops and interrupts again if invalid
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Complete Interrupt-Resume Pipeline", divider="rainbow")

st.markdown("""
Below is the complete implementation of a production interrupt and resume workflow:
""")

st.code("""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. State Definition
class WireTransferState(MessagesState):
    amount: float
    confirmed: bool

# 2. Node with Dynamic Interrupt
def transfer_node(state: WireTransferState) -> dict:
    amt = state["amount"]
    
    # Interrupt only if transfer exceeds 10,000 USD threshold
    if amt > 10000.0:
        approval_payload = {
            "title": "Wire Transfer Authorization Required",
            "amount_usd": amt,
            "threshold_exceeded": True
        }
        human_decision = interrupt(approval_payload)
        
        if human_decision.get("approved"):
            return {
                "confirmed": True,
                "messages": [AIMessage(content=f"Authorized: {amt:.2f} USD transferred successfully.")]
            }
        else:
            return {
                "confirmed": False,
                "messages": [AIMessage(content=f"Rejected: Transfer of {amt:.2f} USD denied.")]
            }
            
    # Under threshold: execute directly
    return {
        "confirmed": True,
        "messages": [AIMessage(content=f"Directly executed: {amt:.2f} USD transferred.")]
    }

# 3. Assemble Graph
builder = StateGraph(WireTransferState)
builder.add_node("transfer", transfer_node)
builder.add_edge(START, "transfer")
builder.add_edge("transfer", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Interrupt Payload & Command Resume Simulator", divider="rainbow")

st.markdown("""
Step through the `interrupt()` lifecycle: trigger a high-risk operation, inspect the emitted
interrupt payload, and supply a `Command(resume=...)` payload to resume execution.
""")

engine = InterruptLifecycleEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Simulation Trigger")
    action_type = st.selectbox(
        "Select Operation to Trigger:",
        ["Drop Production Table (customers_v1)", "Transfer 50,000 USD to Supplier Corp", "Grant Root Admin Access to dev_user"]
    )
    
    if st.button("1. Trigger Action (Calls interrupt())", type="primary"):
        engine.trigger_interrupt(action_type, {"requested_by": "developer_alex", "severity": "CRITICAL"})
        st.session_state.pending_interrupt = engine.paused_state

    st.divider()
    st.subheader("Resume Interrupted State")
    admin_decision = st.selectbox("Administrator Decision:", ["APPROVED", "REJECTED"])
    
    if st.button("2. Resume Execution (Command(resume=...))"):
        if "pending_interrupt" in st.session_state and st.session_state.pending_interrupt:
            engine.paused_state = st.session_state.pending_interrupt
            res = engine.resume_interrupt({"decision": admin_decision, "reviewer": "chief_security_officer"})
            st.session_state.resume_result = res
            st.session_state.pending_interrupt = None
        else:
            st.warning("No active interrupt pending! Click 'Trigger Action' first.")

with wb_col2:
    st.subheader("📋 Interrupt Telemetry View")
    if "pending_interrupt" in st.session_state and st.session_state.pending_interrupt:
        st.markdown("### Status: <span class='badge-pause'>INTERRUPTED / PAUSED</span>", unsafe_allow_html=True)
        st.warning("Graph execution suspended. Checkpoint saved.")
        st.json(st.session_state.pending_interrupt)
    elif "resume_result" in st.session_state and st.session_state.resume_result:
        st.markdown("### Status: <span class='badge-resume'>RESUMED & COMPLETED</span>", unsafe_allow_html=True)
        st.success(st.session_state.resume_result["result_message"])
        st.json(st.session_state.resume_result)
    else:
        st.info("Trigger an action on the left to observe the interrupt payload.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the primary difference between interrupt() in LangGraph v0.2+ and legacy breakpoints?", expanded=False):
    st.markdown("""
    **Answer**:
    - Legacy breakpoints (`interrupt_before=["node"]`): Static boundaries configured at graph compile time. They paused execution before or after an entire node without passing dynamic payloads.
    - `interrupt()` (v0.2+): A native Python function callable dynamically *anywhere* inside a node. It emits structured payloads to reviewers and receives dynamic resumption values directly via `Command(resume=...)`.
    """)

with st.expander("Q2: Where is the value returned by interrupt() stored while the graph is paused?", expanded=False):
    st.markdown("""
    **Answer**:
    The emitted payload is stored in the checkpointer database within the active task record.
    External applications retrieve it via `state.tasks[0].interrupts[0].value`.
    """)

with st.expander("Q3: Can a node contain multiple interrupt() calls?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! A node can execute step 1, call `interrupt()` for approval, execute step 2, and call `interrupt()` again
    for a second confirmation. Each resumption advances execution to the next interrupt or node completion.
    """)

with st.expander("Q4: What happens if an external caller invokes a paused graph without using Command(resume=...)?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph raises an error informing the caller that the thread is currently suspended at an interrupt and
    can only be resumed by passing a `Command(resume=...)` object.
    """)

with st.expander("Q5: Does interrupt() consume computing resources or hold open database connections while paused?", expanded=False):
    st.markdown("""
    **Answer**:
    No! The node execution frame is terminated, database connections are returned to the pool, and memory is freed.
    The graph remains dormant in storage until explicitly resumed.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Calling interrupt() Without a Checkpointer Attached")
    st.markdown("""
    *Mechanism*: Using `interrupt()` on a graph compiled without `checkpointer=...`.  
    *Consequence*: Graph crashes immediately with `ValueError: interrupt() requires a checkpointer`.  
    *Fix*: Always pass a checkpointer (e.g. `MemorySaver` or `PostgresSaver`) to `compile()`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Passing Non-Serializable Objects to interrupt()")
    st.markdown("""
    *Mechanism*: Passing complex custom class instances or database cursors to `interrupt(payload)`.  
    *Consequence*: Serialization fails when the checkpointer attempts to persist the interrupt.  
    *Fix*: Pass only JSON-compliant dictionaries and primitives.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Missing Timeout Monitoring on Interrupted Threads")
    st.markdown("""
    *Mechanism*: Allowing interrupted threads to remain suspended indefinitely without expiration.  
    *Consequence*: Thousands of abandoned sessions accumulate in PostgreSQL.  
    *Fix*: Implement an automated background task to cancel or expire abandoned approvals after 48 hours.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Side-Effects Before interrupt()")
    st.markdown("""
    *Mechanism*: Sending an email or modifying a database row *before* calling `interrupt()` in the same node.  
    *Consequence*: If the supervisor later rejects the action, the side-effect has already occurred.  
    *Fix*: Always place destructive side-effects *after* the `interrupt()` call has returned approval.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**interrupt()**\\nThe dynamic primitive that suspends graph execution and passes review payloads to callers.")
with col_fc2:
    st.info("**Command(resume=...)**\\nThe invocation wrapper used to resume an interrupted graph with reviewer inputs.")
with col_fc3:
    st.info("**state.tasks[0].interrupts**\\nThe state snapshot path where pending interrupt payloads are inspected.")
with col_fc4:
    st.info("**Dynamic Breakpoint**\\nA pausing gate triggered conditionally based on runtime state variables.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to call interrupt() inside any LangGraph node function.", value=True)
c2 = st.checkbox("I understand how the argument to interrupt() is serialized to storage.", value=True)
c3 = st.checkbox("I know how to inspect pending interrupts via state.tasks[0].interrupts.", value=True)
c4 = st.checkbox("I can resume execution using app.invoke(Command(resume=payload)).", value=True)
c5 = st.checkbox("I understand why side-effects must always be placed AFTER interrupt() calls.", value=True)
c6 = st.checkbox("I know how to implement multiple sequential interrupts in a single node.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 16.2 'Interrupts' with full architectural and practical comprehension!")
