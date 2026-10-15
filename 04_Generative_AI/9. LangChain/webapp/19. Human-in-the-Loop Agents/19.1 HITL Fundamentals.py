import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.1 HITL Fundamentals - LangChain Architecture",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards, badges, and layout
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #065F46;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #78350F;
        color: #FBBF24;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-fail {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockHITLReviewSimulator:
    """
    Deterministic simulator representing a LangGraph Human-in-the-Loop review lifecycle.
    Demonstrates interruption at critical boundaries, state persistence, approval,
    rejection, and manual state payload editing before node execution resumption.
    """
    def __init__(self):
        self.threads: Dict[str, Dict[str, Any]] = {
            "thread-alpha-101": {
                "thread_id": "thread-alpha-101",
                "action": "execute_wire_transfer",
                "parameters": {
                    "recipient": "Vendor-Global-Corp",
                    "account_iban": "DE89370400440532013000",
                    "amount_usd": 245000.00,
                    "urgency": "High",
                    "memo": "Annual software server license payment"
                },
                "status": "INTERRUPTED",
                "checkpoint_id": "chk_7721a_pending",
                "next_node": "payment_gateway_dispatch",
                "history": ["agent_reasoning", "risk_evaluation_gate"]
            },
            "thread-beta-202": {
                "thread_id": "thread-beta-202",
                "action": "drop_customer_table_partition",
                "parameters": {
                    "table_name": "customer_analytics_2023",
                    "cluster_region": "us-east-1",
                    "backup_verified": True,
                    "target_retention_days": 90
                },
                "status": "INTERRUPTED",
                "checkpoint_id": "chk_9942b_pending",
                "next_node": "database_admin_tool",
                "history": ["query_analyzer", "retention_policy_check"]
            }
        }

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return self.threads.get(thread_id)

    def process_action(self, thread_id: str, decision: str, edited_params: Optional[Dict[str, Any]] = None, feedback: str = "") -> Dict[str, Any]:
        thread = self.threads.get(thread_id)
        if not thread:
            return {"status": "ERROR", "message": "Thread identifier not found."}

        if decision == "APPROVE":
            thread["status"] = "APPROVED_EXECUTED"
            thread["checkpoint_id"] = f"chk_{thread_id[-3:]}_completed"
            thread["history"].append(thread["next_node"])
            return {
                "status": "SUCCESS",
                "outcome": "Action approved by human reviewer. Graph resumed execution to completion.",
                "executed_node": thread["next_node"],
                "final_parameters": thread["parameters"],
                "thread_id": thread_id
            }
        elif decision == "REJECT":
            thread["status"] = "REJECTED_ABORTED"
            thread["checkpoint_id"] = f"chk_{thread_id[-3:]}_halted"
            thread["history"].append("abort_handler")
            return {
                "status": "REJECTED",
                "outcome": f"Action rejected. Execution halted safely without invoking '{thread['next_node']}'.",
                "reviewer_feedback": feedback,
                "thread_id": thread_id
            }
        elif decision == "EDIT_AND_APPROVE":
            if edited_params:
                thread["parameters"] = edited_params
            thread["status"] = "MODIFIED_EXECUTED"
            thread["checkpoint_id"] = f"chk_{thread_id[-3:]}_modified"
            thread["history"].append("human_override_state_injection")
            thread["history"].append(thread["next_node"])
            return {
                "status": "SUCCESS_MODIFIED",
                "outcome": "State successfully patched with human edits. Graph resumed with updated payload.",
                "executed_node": thread["next_node"],
                "modified_parameters": thread["parameters"],
                "thread_id": thread_id
            }
        return {"status": "UNKNOWN", "message": "Unrecognized decision branch."}

# Main Application Title & Objective
st.title("🛡️ 19.1 Human-in-the-Loop (HITL) Fundamentals")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Human-in-the-Loop (HITL) establishes an authoritative supervision barrier between autonomous agent reasoning and 
real-world, high-consequence side effects. In LangGraph, HITL is natively achieved through state checkpoints, persistent threads, 
and deterministic execution interrupts (`interrupt_before` / `interrupt_after` or dynamic `interrupt()` calls).
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Human-in-the-Loop Lifecycle & State Breakpoints", divider="rainbow")
st.markdown("""
Unlike traditional software loops that block thread memory while waiting for synchronous user input, modern agent architectures 
treat human review as a **durable asynchronous state pause**. When an agent approaches a sensitive action, the graph writes a 
checkpoint to persistent storage, halts execution, and yields control back to the host system. When the human reviewer responds, 
the graph reloads the exact snapshot and seamlessly continues or rewrites history.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = graphviz.Digraph(comment="HITL Architecture")
    blueprint_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    blueprint_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    blueprint_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    blueprint_dot.node("input", "User Request / Trigger\n[Thread Assigned: thread_id=T-404]", fillcolor="#0F172A", shape="ellipse")
    blueprint_dot.node("agent_node", "Agent Reasoning Node\n[LLM Formulates Action & Arguments]", fillcolor="#1E293B")
    blueprint_dot.node("eval_gate", "Risk Assessment / Tool Guard\n[Is Action Classified as Sensitive?]", shape="diamond", fillcolor="#334155")
    blueprint_dot.node("checkpoint", "Checkpointer Serialization\n[State Persisted to SQLite/PostgreSQL]", fillcolor="#0E7490")
    blueprint_dot.node("interrupt_node", "EXECUTION INTERRUPT\n[Graph Halts | Yields to Application API]", fillcolor="#B45309", fontcolor="#FEF3C7")
    blueprint_dot.node("human_ui", "Human Reviewer Interface\n[Review Parameters, Diff, Impact]", fillcolor="#4338CA")
    blueprint_dot.node("action_branch", "Reviewer Action Branch\n[Approve / Reject / Edit State]", shape="diamond", fillcolor="#312E81")
    blueprint_dot.node("tool_exec", "High-Impact Tool Execution\n[Wire Transfer / DB Drop / Email Send]", fillcolor="#047857")
    blueprint_dot.node("abort_node", "Aborted Workflow Node\n[Notify User & Log Audit Trail]", fillcolor="#881337")
    blueprint_dot.node("end_node", "Graph Workflow Complete\n[End State Persisted]", fillcolor="#0F172A", shape="ellipse")

    blueprint_dot.edge("input", "agent_node")
    blueprint_dot.edge("agent_node", "eval_gate")
    blueprint_dot.edge("eval_gate", "tool_exec", label=" Low Risk (Auto-Execute)")
    blueprint_dot.edge("eval_gate", "checkpoint", label=" High Risk Action")
    blueprint_dot.edge("checkpoint", "interrupt_node")
    blueprint_dot.edge("interrupt_node", "human_ui", label=" Notification Webhook")
    blueprint_dot.edge("human_ui", "action_branch")
    blueprint_dot.edge("action_branch", "tool_exec", label=" Approved / Edited Payload")
    blueprint_dot.edge("action_branch", "abort_node", label=" Rejected / Disapproved")
    blueprint_dot.edge("tool_exec", "end_node")
    blueprint_dot.edge("abort_node", "end_node")

    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: WHAT IS HUMAN-IN-THE-LOOP? ---
st.header("2. What is Human-in-the-Loop?", divider="rainbow")
st.markdown("""
**Human-in-the-Loop (HITL)** is an architectural paradigm where algorithmic autonomous decision-making integrates human 
oversight, validation, or steering at predefined checkpoints. In AI systems, it bridges the gap between pure autonomy 
(which carries hallucination and liability risks) and pure manual labor (which lacks scalability).
""")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Autonomous Execution vs. HITL Oversight")
    st.markdown("""
    * **Purely Autonomous Agents**:
      * LLM determines tool calls and arguments and invokes them immediately.
      * Irreversible real-world operations execute with zero sanity verification.
      * Unbounded compounding errors: A subtle reasoning hallucination can cascade into financial or data loss.
    * **HITL-Governed Agents**:
      * Agent acts as a proactive **preparer** of actions rather than an uncontrolled **executor**.
      * Identifies sensitive intent and yields control at defined deterministic boundary gates.
      * Maintains an immutable audit trail of who verified what payload, when, and under what conditions.
    """)
with col_b:
    st.subheader("HITL System Characteristics")
    st.markdown("""
    | Attribute | Fully Autonomous | Human-in-the-Loop |
    | :--- | :--- | :--- |
    | **Latency** | Milliseconds to seconds | Asynchronous (minutes to hours) |
    | **Process State** | Ephemeral memory thread | Durable persisted checkpoint |
    | **Accountability** | Model developer / enterprise liability | Explicit verified human sign-off |
    | **Error Recovery** | Automated retries or catastrophic crash | Human intervention and live state patching |
    | **Regulatory Compliance** | Non-compliant for high-risk AI (EU AI Act) | Compliant with Article 14 Human Oversight |
    """)

# --- SECTION 3: WHY HUMAN APPROVAL IS NEEDED ---
st.header("3. Why Human Approval is Needed", divider="rainbow")
st.markdown("""
Autonomous models are probabilistic pattern engines that lack situational empathy, legal accountability, and physical permanence awareness. 
Real-world systems operate under the **Principle of Asymmetric Risk**: while the upside of automated speed is incremental, the downside of an 
unchecked hallucinated mutation can be catastrophic.
""")

col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.markdown("""
    <div class="metric-card">
        <h4>1. Irreversible Side Effects</h4>
        <p>Operations such as debiting a bank account, transmitting legal notices, deleting customer databases, or rebooting production infrastructure cannot be undone via simple code rollback.</p>
        <span class="badge-fail">High Consequence</span>
    </div>
    """, unsafe_allow_html=True)
with col_c2:
    st.markdown("""
    <div class="metric-card">
        <h4>2. Hallucinated Arguments</h4>
        <p>A model may correctly select the tool <code>refund_customer</code> but accidentally hallucinate an extra zero (refunding 10,000 USD instead of 1,000 USD) due to tokenization or attention drift.</p>
        <span class="badge-warn">Payload Risk</span>
    </div>
    """, unsafe_allow_html=True)
with col_c3:
    st.markdown("""
    <div class="metric-card">
        <h4>3. Regulatory & Legal Mandates</h4>
        <p>Frameworks including GDPR Article 22, the EU AI Act High-Risk systems classification, and SOC2 Type II require human oversight for automated profiling and significant financial/legal decisions.</p>
        <span class="badge-pass">Compliance Shield</span>
    </div>
    """, unsafe_allow_html=True)

# In-section diagram: Risk Spectrum
with st.expander("🔍 View Action Risk Spectrum & Approval Tiers", expanded=False):
    risk_dot = graphviz.Digraph(comment="Risk Spectrum")
    risk_dot.attr(bgcolor="#0B1120", rankdir="TB")
    risk_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    risk_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    risk_dot.node("tier1", "Tier 1: Read-Only Actions\n[Vector Search, Schema Read, Web Lookup]\nPOLICY: Full Autonomy (0 Interrupts)", fillcolor="#065F46")
    risk_dot.node("tier2", "Tier 2: Reversible Modifications\n[Create Draft Ticket, Update Cache, Send Slack Bot Ping]\nPOLICY: Automated with Async Notification", fillcolor="#1E3A8A")
    risk_dot.node("tier3", "Tier 3: High-Impact Operations\n[Customer Email, Financial Transaction, DB Mutation]\nPOLICY: Mandatory Synchronous/Async Human Review Gate", fillcolor="#78350F")
    risk_dot.node("tier4", "Tier 4: Destructive Operations\n[Drop Table, Terminate VM, Revoke API Keys]\nPOLICY: Dual-Officer Human Sign-off + Audit Trail", fillcolor="#7F1D1D")

    risk_dot.edge("tier1", "tier2", label=" Escalating Risk")
    risk_dot.edge("tier2", "tier3")
    risk_dot.edge("tier3", "tier4")
    st.graphviz_chart(risk_dot, use_container_width=True)

# --- SECTION 4: AGENT INTERRUPTION ---
st.header("4. Agent Interruption Mechanisms", divider="rainbow")
st.markdown("""
In LangGraph, an **interruption** is a first-class control flow primitive. Unlike standard synchronous code that calls `input()` 
and blocks CPU threads, LangGraph interrupts are powered by the **Checkpointer Engine**.
""")

st.markdown("""
### Interruption Topologies
1. **Static Graph Compilation Interrupts**:
   - `interrupt_before=["tools"]`: The graph automatically halts right before executing the node named `tools`. State is frozen as outputted by the reasoning node.
   - `interrupt_after=["tools"]`: The node completes its run, but before transitioning to the next node, execution freezes to allow human inspection of the tool output.
2. **Dynamic In-Node Interrupts (`interrupt()` API in LangGraph v0.2+)**:
   - Nodes can conditionally call `interrupt(value)` based on runtime parameter evaluation (e.g., if `amount > 10000`).
   - The value passed to `interrupt()` is surfaced directly to the human reviewer as the review prompt.
""")

col_int1, col_int2 = st.columns(2)
with col_int1:
    st.markdown("**How Graph Memory Freezes During Interruption:**")
    st.markdown("""
    * Node output is staged into channel reducers.
    * The state dictionary is serialized and written to the configured `Checkpointer` (e.g., `MemorySaver`, `SqliteSaver`, `PostgresSaver`).
    * The graph execution cycle exits cleanly, returning a status indicating `tasks=[...]` are awaiting input.
    * The server thread is completely released back to the event loop.
    """)
with col_int2:
    st.markdown("**Checkpointer State Snapshot at Interrupt:**")
    st.code("""
# Inspection of graph state during pause:
state = graph.get_state(config={"configurable": {"thread_id": "T-101"}})

print(state.values)    # Complete messages & variables
print(state.next)      # Tuple of next nodes: ('tools',)
print(state.tasks)     # Pending tasks awaiting resolution
    """, language="python")

# --- SECTION 5: HUMAN APPROVAL ---
st.header("5. Human Approval Protocol", divider="rainbow")
st.markdown("""
When a human reviewer evaluates an interrupted task and decides that the agent proposed safe and accurate actions, 
the **Approval Protocol** is engaged. The system issues a resume command without altering the state values.
""")

st.markdown("""
### Resuming Without Modification
To approve and allow the paused node to proceed as planned:
```python
# The human reviewer approves the pending tool call:
# Resuming with None tells LangGraph to continue from the saved checkpoint
config = {"configurable": {"thread_id": "thread-alpha-101"}}

for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```
* **Event Sequence on Approval**:
  1. The checkpointer loads the checkpoint associated with `thread-alpha-101`.
  2. The scheduler identifies `next=("tools",)`.
  3. The `tools` node executes with the exact parameters formulated by the agent.
  4. Execution advances through subsequent edges until reaching `END` or another interrupt.
""")

# --- SECTION 6: HUMAN REJECTION ---
st.header("6. Human Rejection Handling", divider="rainbow")
st.markdown("""
When the human reviewer determines that an action is dangerous, erroneous, or out of scope, the action must be **rejected**. 
Simply killing the process is an anti-pattern because the conversational agent remains unaware of why the action did not occur.
""")

col_rej1, col_rej2 = st.columns(2)
with col_rej1:
    st.subheader("The Structured Rejection Pattern")
    st.markdown("""
    In production architectures, rejecting a tool call requires updating the graph state to inject a synthetic **ToolMessage** 
    or user feedback message signaling that the action was vetoed. This allows the model to apologize, choose an alternative route, 
    or ask the user for clarification.
    """)
    st.markdown("""
    * **Step 1**: Prevent the pending tool node from executing.
    * **Step 2**: Inject an error/rejection message matching the tool call ID.
    * **Step 3**: Re-route the graph back to the agent reasoning node.
    """)
with col_rej2:
    st.subheader("Rejection State Update Example")
    st.code("""
from langchain_core.messages import ToolMessage

# Create rejection response tied to tool_call_id
rejection_msg = ToolMessage(
    tool_call_id="call_9938a",
    content="ACTION REJECTED BY ADMIN: Account balance validation failed. Do not retry.",
    status="error"
)

# Update state as if the tool failed, rerouting back to agent
graph.update_state(
    config={"configurable": {"thread_id": "T-101"}},
    values={"messages": [rejection_msg]},
    as_node="tools"
)
# Resume agent so it reads the rejection
graph.stream(None, config)
    """, language="python")

# --- SECTION 7: HUMAN EDITING ---
st.header("7. Human State Editing & Correction", divider="rainbow")
st.markdown("""
Frequently, an agent formulates an action that is **mostly correct**, but contains a minor syntax error, incorrect parameter, 
or uncalibrated threshold. Forcing full rejection wastes tokens and degrades user experience. 
LangGraph empowers human reviewers to **edit graph state in-place** before resuming execution.
""")

st.markdown("""
### In-Place Parameter Patching via `update_state`
Using `graph.update_state()`, the human reviewer overwrites specific fields in the current state checkpoint.
""")

st.code("""
# 1. Fetch the interrupted state
current_state = graph.get_state(config)
pending_tool_call = current_state.values["messages"][-1].tool_calls[0]

# 2. Human reviewer modifies arguments (e.g., reducing refund amount from 5,000 USD to 500 USD)
modified_tool_call = pending_tool_call.copy()
modified_tool_call["args"]["amount"] = 500.00

# 3. Overwrite the last message with the corrected tool call
new_message = current_state.values["messages"][-1]
new_message.tool_calls = [modified_tool_call]

# 4. Commit updated state into the thread checkpoint
graph.update_state(
    config,
    values={"messages": [new_message]}
)

# 5. Resume execution with safe parameters
graph.stream(None, config)
""", language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: LangGraph HITL Review Gate with Interrupt & Resumption", divider="rainbow")
st.markdown("""
The following complete production pattern implements a typed `StateGraph` equipped with `SqliteSaver` persistence, 
static and dynamic interrupt conditions, human state modification, and safe dispatching.
""")

st.code('''
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Define Typed State Schema with message reducer
class AgentHITLState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_risk_action: Optional[dict]
    audit_status: str

# 2. Define Business Logic Nodes
def planner_node(state: AgentHITLState) -> dict:
    last_msg = state["messages"][-1].content
    if "transfer" in last_msg.lower():
        # Agent plans high-risk wire transfer
        return {
            "pending_risk_action": {
                "action": "wire_transfer",
                "recipient": "Global-Tech-Vendor",
                "amount": 75000.00
            },
            "audit_status": "PENDING_HUMAN_REVIEW"
        }
    return {
        "messages": [AIMessage(content="General query processed safely.")],
        "audit_status": "COMPLETED"
    }

def sensitive_tool_execution(state: AgentHITLState) -> dict:
    action = state["pending_risk_action"]
    result_msg = f"EXECUTED: {action['action']} to {action['recipient']} of {action['amount']:,.2f} USD"
    return {
        "messages": [AIMessage(content=result_msg)],
        "audit_status": "EXECUTED_CONFIRMED",
        "pending_risk_action": None
    }

# 3. Router Edge Logic
def route_risk(state: AgentHITLState) -> Literal["sensitive_tool", "END"]:
    if state.get("pending_risk_action"):
        return "sensitive_tool"
    return END

# 4. Assemble Graph with Interrupt Gate
builder = StateGraph(AgentHITLState)
builder.add_node("planner", planner_node)
builder.add_node("sensitive_tool", sensitive_tool_execution)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", route_risk, {"sensitive_tool": "sensitive_tool", END: END})
builder.add_edge("sensitive_tool", END)

# Memory Checkpointer is required for thread-based interrupts
memory = MemorySaver()

# Compile with interrupt_before targeting the sensitive node
app = builder.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_tool"]
)
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: HITL Approval & State Editing Simulator", divider="rainbow")
st.markdown("""
Experiment with real-time HITL state interruption. Select a suspended agent thread, review the proposed action payload, 
and choose whether to **Approve**, **Reject**, or **Edit Parameters** before resuming graph execution.
""")

# Instantiate Module Scope Simulator
if "hitl_sim" not in st.session_state:
    st.session_state.hitl_sim = MockHITLReviewSimulator()

sim = st.session_state.hitl_sim

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("1. Interrupted Threads in Checkpoint Store")
    thread_selection = st.selectbox(
        "Select Pending Thread:",
        options=list(sim.threads.keys()),
        index=0
    )
    thread_data = sim.get_thread(thread_selection)

    st.markdown(f"**Thread ID**: `{thread_data['thread_id']}`")
    st.markdown(f"**Current Status**: `{thread_data['status']}`")
    st.markdown(f"**Next Node to Run**: `{thread_data['next_node']}`")
    st.markdown(f"**Checkpoint ID**: `{thread_data['checkpoint_id']}`")
    
    st.markdown("#### Proposed Action Payload:")
    st.json(thread_data["parameters"])

with col_wb2:
    st.subheader("2. Human Reviewer Decision Gate")
    action_decision = st.radio(
        "Choose Review Action:",
        options=["Approve As-Is", "Edit Payload & Approve", "Reject Action"],
        index=0
    )

    edited_params = None
    feedback_reason = ""

    if action_decision == "Edit Payload & Approve":
        st.markdown("**Live State Payload Editor:**")
        if "amount_usd" in thread_data["parameters"]:
            new_amount = st.number_input(
                "Adjust Amount (USD):", 
                min_value=1.0, 
                max_value=1000000.0, 
                value=float(thread_data["parameters"]["amount_usd"]) / 10.0,
                step=500.0
            )
            new_memo = st.text_input("Reviewer Authorization Memo:", value="Reviewed and discounted by Finance Officer.")
            edited_params = dict(thread_data["parameters"])
            edited_params["amount_usd"] = new_amount
            edited_params["memo"] = new_memo
        elif "table_name" in thread_data["parameters"]:
            new_retention = st.number_input("Safe Retention Days:", min_value=30, max_value=365, value=180)
            edited_params = dict(thread_data["parameters"])
            edited_params["target_retention_days"] = new_retention

    elif action_decision == "Reject Action":
        feedback_reason = st.text_area(
            "Rejection Rationale (Injected into Agent State):",
            value="Action rejected due to compliance risk. Unverified recipient identity."
        )

    if st.button("🚀 Commit Decision & Resume Graph Execution", use_container_width=True):
        if action_decision == "Approve As-Is":
            result = sim.process_action(thread_selection, "APPROVE")
            st.success(result["outcome"])
            st.json(result)
        elif action_decision == "Edit Payload & Approve":
            result = sim.process_action(thread_selection, "EDIT_AND_APPROVE", edited_params=edited_params)
            st.success(result["outcome"])
            st.json(result)
        elif action_decision == "Reject Action":
            result = sim.process_action(thread_selection, "REJECT", feedback=feedback_reason)
            st.error(result["outcome"])
            st.json(result)

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How does LangGraph handle human approval asynchronously across web workers without memory leaks?"):
    st.markdown("""
    LangGraph achieves zero memory retention during waiting periods by decoupling process execution from state persistence. 
    When an interrupt condition is reached, the state snapshot is flushed to an external `Checkpointer` (e.g., PostgreSQL or Redis) 
    keyed by `thread_id`. The executing process terminates cleanly. When the approval webhook arrives, any web worker in the cluster 
    can reload the thread state by `thread_id` and resume execution immediately.
    """)

with st.expander("Q2: What is the difference between `interrupt_before` and dynamic in-node `interrupt()` in LangGraph?"):
    st.markdown("""
    * **`interrupt_before`**: Configured statically at graph compile time. Pauses execution before the specified node begins, 
      regardless of runtime parameter values.
    * **`interrupt()`**: Invoked dynamically inside node code. Allows conditional interruption based on dynamic runtime logic 
      (e.g., pausing only if transfer amount exceeds 10,000 USD, while auto-approving sub-10,000 amounts).
    """)

with st.expander("Q3: When a human edits state using `update_state()`, how does LangGraph reconcile the graph history?"):
    st.markdown("""
    `update_state()` creates a new checkpoint that forks from or extends the current thread's history. It applies the updates 
    through the channel reducers defined on the state schema. If updating messages with an `add_messages` reducer, existing 
    messages can be overwritten by matching their unique message `id`.
    """)

with st.expander("Q4: Why is returning a generic error better than silently aborting an agent when an action is rejected?"):
    st.markdown("""
    If a graph is aborted silently, the LLM reasoning context is destroyed, leaving the user with an unexplained pause. 
    By injecting a structured rejection message (`ToolMessage(status='error', content='...')`), the LLM receives semantic feedback, 
    allowing it to explain the rejection gracefully to the user and request revised instructions.
    """)

with st.expander("Q5: What are the primary performance and consistency trade-offs of using `MemorySaver` vs `PostgresSaver` in HITL?"):
    st.markdown("""
    `MemorySaver` stores thread states in an in-memory dictionary. If the process restarts while waiting for a human review 
    (which can take hours or days), all pending threads are permanently lost. `PostgresSaver` guarantees ACID durability and enables 
    horizontal multi-pod cluster scaling, but introduces small database I/O serialization latency per step.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Production Trap 1 (Fatal Error): Synchronous Blocking in Node Code**  
    Calling `time.sleep()` or waiting on a Python `threading.Event()` inside a node locks the web worker thread, leads to socket timeouts, 
    and fails instantly in serverless or auto-scaling architectures. Always use native checkpointer-backed graph interrupts.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Tool Call ID Mismatch in Human Injections**  
    When rejecting or replacing an action, failing to include the exact `tool_call_id` in the replacement message violates the OpenAI / Anthropic 
    tool calling specification, causing subsequent model invocations to throw `400 Invalid Tool Call ID` API errors.
    """)
with col_trap2:
    st.error("""
    **Production Trap 3 (Fatal Error): Using In-Memory Checkpointers in Production HITL**  
    Deploying `MemorySaver` to a containerized production cluster means any pod recycling, deployment rollout, or OOM event permanently 
    destroys pending review states, leaving human reviewers with dangling, unresolvable approval buttons.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Unbounded Thread Checkpoint Accumulation**  
    Persisting every micro-step in durable storage indefinitely without retention or compaction policies will bloat PostgreSQL table sizes 
    and degrade checkpoint query latency. Always establish TTL or archive policies for completed threads.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 1: Thread Isolation</h4>
        <p><strong>Concept:</strong> <code>configurable.thread_id</code></p>
        <p><strong>Definition:</strong> The unique partition key in a LangGraph checkpointer that isolates one user's paused state and conversation history from all other concurrent executions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 2: Static Interrupts</h4>
        <p><strong>Concept:</strong> <code>interrupt_before / interrupt_after</code></p>
        <p><strong>Definition:</strong> Compile-time graph parameters that instruct the Pregel execution engine to halt before or after designated node names.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc2:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 3: State Rewriting</h4>
        <p><strong>Concept:</strong> <code>graph.update_state()</code></p>
        <p><strong>Definition:</strong> The API method that allows human reviewers to patch variables or overwrite message payloads in a paused thread checkpoint before continuing.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 4: Asynchronous Resumption</h4>
        <p><strong>Concept:</strong> <code>graph.stream(None, config)</code></p>
        <p><strong>Definition:</strong> Passing <code>None</code> as the input payload reloads the latest checkpoint for the specified thread and triggers the next scheduled node.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of Human-in-the-Loop fundamentals and LangGraph state interruption patterns:")

chk1 = st.checkbox("1. I understand the difference between blocking synchronous code and durable checkpointer-based graph interrupts.")
chk2 = st.checkbox("2. I know how to configure `interrupt_before` and `interrupt_after` during `graph.compile()`.")
chk3 = st.checkbox("3. I understand how `graph.stream(None, config)` reloads an interrupted thread and resumes execution.")
chk4 = st.checkbox("4. I know how to reject an action gracefully by injecting an error ToolMessage to avoid corrupting LLM conversation history.")
chk5 = st.checkbox("5. I can use `graph.update_state()` to allow human reviewers to edit payload parameters before execution.")
chk6 = st.checkbox("6. I understand why durable checkpointers (`PostgresSaver`) are mandatory for production HITL systems.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Human-in-the-Loop Fundamentals in LangChain & LangGraph.")
