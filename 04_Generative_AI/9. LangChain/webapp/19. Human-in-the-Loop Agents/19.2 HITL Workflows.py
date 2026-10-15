import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.2 HITL Workflows - LangChain Architecture",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and status badges
st.markdown("""
<style>
    .workflow-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pre {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-post {
        background-color: #065F46;
        color: #6EE7B7;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-dual {
        background-color: #701A75;
        color: #F472B6;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockHITLWorkflowsEngine:
    """
    Simulation engine showcasing multi-pattern human approval workflows:
    1. Pre-execution authorization (wire transfer, DB drop, mass email)
    2. Post-execution quality auditing (draft response critique, data sanity check)
    3. Multi-tier dual-control validation for enterprise governance.
    """
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {
            "OP-FIN-701": {
                "id": "OP-FIN-701",
                "category": "Financial Operations",
                "gate_type": "PRE_EXECUTION",
                "title": "Corporate Wire Transfer",
                "payload": {
                    "beneficiary": "Nordic Data Solutions AB",
                    "iban": "SE4550000000058392019942",
                    "amount_usd": 120000.00,
                    "currency": "USD",
                    "compliance_code": "INV-2026-X88"
                },
                "status": "AWAITING_APPROVAL",
                "risk_score": 92
            },
            "OP-COMM-404": {
                "id": "OP-COMM-404",
                "category": "Email & Customer Outbound",
                "gate_type": "PRE_EXECUTION",
                "title": "Mass Security Breach Notification",
                "payload": {
                    "recipients_count": 48200,
                    "sender_alias": "security-alert@enterprise.com",
                    "subject": "Mandatory Password Reset Required",
                    "content_preview": "We have detected unusual activity on account clusters. Please reset your credentials."
                },
                "status": "AWAITING_APPROVAL",
                "risk_score": 85
            },
            "OP-DB-912": {
                "id": "OP-DB-912",
                "category": "Database Modification",
                "gate_type": "PRE_EXECUTION",
                "title": "Database Schema Truncate",
                "payload": {
                    "target_database": "prod_warehouse_eu",
                    "target_table": "raw_telemetry_events_q1",
                    "command": "TRUNCATE TABLE raw_telemetry_events_q1;",
                    "replica_synced": True
                },
                "status": "AWAITING_APPROVAL",
                "risk_score": 98
            },
            "OP-AUDIT-109": {
                "id": "OP-AUDIT-109",
                "category": "Post-Execution Quality Audit",
                "gate_type": "POST_EXECUTION",
                "title": "AI Legal Contract Summary",
                "payload": {
                    "contract_id": "MSA-ACME-2026-09",
                    "summary_generated": "The vendor assumes unlimited liability for data corruption with net-30 terms.",
                    "confidence_score": 0.88,
                    "execution_completed": True
                },
                "status": "AWAITING_VERIFICATION",
                "risk_score": 64
            }
        }

    def evaluate_operation(self, op_id: str, decision: str, notes: str = "") -> Dict[str, Any]:
        op = self.operations.get(op_id)
        if not op:
            return {"status": "ERROR", "message": "Operation ID not found."}

        if decision == "AUTHORIZE":
            op["status"] = "AUTHORIZED_DISPATCHED" if op["gate_type"] == "PRE_EXECUTION" else "VERIFIED_ACCEPTED"
            return {
                "status": "SUCCESS",
                "decision": "AUTHORIZE",
                "message": f"Operation '{op['title']}' officially cleared and committed.",
                "auditor_notes": notes or "Auto-authorized without additional comments.",
                "current_status": op["status"]
            }
        elif decision == "REJECT":
            op["status"] = "REJECTED_BLOCKED" if op["gate_type"] == "PRE_EXECUTION" else "DISCREDITED_FLAGGED"
            return {
                "status": "REJECTED",
                "decision": "REJECT",
                "message": f"Operation '{op['title']}' vetoed. Side effects prevented.",
                "auditor_notes": notes or "Rejected due to policy violation.",
                "current_status": op["status"]
            }
        return {"status": "UNKNOWN", "message": "Unknown action."}

# Main Application Title & Objective
st.title("⚖️ 19.2 Human-in-the-Loop (HITL) Workflows")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Production AI agents operate across diverse operational risk boundaries. Real-world HITL requires specialized workflow topologies: 
**Pre-execution approval gates** for irreversible side effects, **Post-execution review gates** for sensitive content verification, 
and **Dual-control governance** for high-stakes financial, database, and customer-facing interactions.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise HITL Workflow Matrix & Gate Typologies", divider="rainbow")
st.markdown("""
The architecture below delineates how an incoming agent tool proposal is dynamically triaged into either a **Pre-Execution Gate** 
(interrupted before side-effect dispatch) or a **Post-Execution Gate** (executed in sandbox or draft state, then validated prior to release).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    bp_dot = graphviz.Digraph(comment="HITL Workflows")
    bp_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    bp_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    bp_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    bp_dot.node("plan_node", "Agent Formulation Node\n[Generates Structured Tool Call]", fillcolor="#0F172A", shape="ellipse")
    bp_dot.node("classifier", "Action Sensitivity Classifier\n[Evaluates Risk, Scope & Policy Tier]", shape="diamond", fillcolor="#334155")
    
    # Subgraph for Pre-Execution
    bp_dot.node("pre_gate", "PRE-EXECUTION REVIEW GATE\n[interrupt_before=['sensitive_tool']]\nState Frozen at Checkpoint", fillcolor="#1E3A8A")
    bp_dot.node("pre_human", "Reviewer Action\n[Authorize or Cancel]", shape="diamond", fillcolor="#312E81")
    bp_dot.node("live_tool", "Live Production Dispatch\n[Bank Gateway / SQL Execution / Mass Mail]", fillcolor="#065F46")
    bp_dot.node("pre_abort", "Abort & Error Injection\n[ToolMessage(status='vetoed')]", fillcolor="#7F1D1D")

    # Subgraph for Post-Execution
    bp_dot.node("draft_tool", "Sandbox / Draft Tool Node\n[Generate Staged Output / Email Draft]", fillcolor="#0E7490")
    bp_dot.node("post_gate", "POST-EXECUTION AUDIT GATE\n[interrupt_after=['draft_tool']]\nInspect Generated Output", fillcolor="#701A75")
    bp_dot.node("post_human", "Human Quality Gate\n[Approve Release or Request Revision]", shape="diamond", fillcolor="#831843")
    bp_dot.node("release_node", "Production Output Released\n[Publish Article / Send Customer Email]", fillcolor="#047857")

    bp_dot.node("end_flow", "Workflow Finished\n[Audit Trail Recorded]", fillcolor="#0F172A", shape="ellipse")

    bp_dot.edge("plan_node", "classifier")
    bp_dot.edge("classifier", "pre_gate", label=" Irreversible Side Effects\n(Wire, DB Drop, Delete)")
    bp_dot.edge("classifier", "draft_tool", label=" Content Generation / Draft\n(Contract, Report, Email Draft)")

    bp_dot.edge("pre_gate", "pre_human")
    bp_dot.edge("pre_human", "live_tool", label=" Authorized")
    bp_dot.edge("pre_human", "pre_abort", label=" Rejected")
    bp_dot.edge("live_tool", "end_flow")
    bp_dot.edge("pre_abort", "end_flow")

    bp_dot.edge("draft_tool", "post_gate")
    bp_dot.edge("post_gate", "post_human")
    bp_dot.edge("post_human", "release_node", label=" Quality Approved")
    bp_dot.edge("post_human", "plan_node", label=" Revision Feedback")
    bp_dot.edge("release_node", "end_flow")

    st.graphviz_chart(bp_dot, use_container_width=True)

# --- SECTION 2: APPROVAL BEFORE TOOL EXECUTION ---
st.header("2. Approval Before Tool Execution", divider="rainbow")
st.markdown("""
**Pre-execution approval** is the foundational guardrail for preventing catastrophic, irreversible mutations. 
In LangGraph, this is implemented by configuring `interrupt_before=["tool_node_name"]` or issuing dynamic in-node interrupts.
""")

col_pre1, col_pre2 = st.columns(2)
with col_pre1:
    st.subheader("Core Mechanism")
    st.markdown("""
    * **Execution Freeze**: The reasoning agent outputs an `AIMessage` containing `tool_calls=[{name: 'execute_transfer', args: {...}}]`.
    * **Before Invoking**: The graph halts prior to entering the tool node.
    * **Human Visibility**: The reviewer inspects the planned tool name, arguments, and conversational context without the tool ever running.
    * **State Mutation**: If approved, the tool runs with identical arguments. If modified via `graph.update_state()`, the tool receives revised parameters.
    """)
with col_pre2:
    st.subheader("LangGraph Implementation Pattern")
    st.code("""
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(StateSchema)
builder.add_node("agent", call_model)
builder.add_node("tools", execute_tools)

builder.add_edge("agent", "tools")
builder.add_edge("tools", "agent")

# Halts immediately before 'tools' node executes
app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"]
)
    """, language="python")

# --- SECTION 3: APPROVAL AFTER TOOL EXECUTION ---
st.header("3. Approval After Tool Execution", divider="rainbow")
st.markdown("""
Certain tools are read-only or staged, but their **raw output** must be reviewed by a human before that data is synthesized into 
a customer-facing response, committed to a legal document, or forwarded downstream. This is **Post-execution review**.
""")

col_post1, col_post2 = st.columns(2)
with col_post1:
    st.subheader("When to Use Post-Execution Gates")
    st.markdown("""
    * **Sandbox Synthesis**: Tool generated a drafted legal response, and an attorney must verify citations before client delivery.
    * **Data Validation**: A tool fetched sensitive corporate metrics; a compliance officer verifies redaction before the agent continues.
    * **Human Rating & Reinforcement**: Reviewer rates output quality, injecting human feedback directly into the prompt context for the next turn.
    """)
with col_post2:
    st.subheader("LangGraph Configuration")
    st.code("""
# Halts immediately AFTER the staging node finishes,
# allowing the reviewer to inspect state['draft_payload']
app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["generate_contract_draft"]
)

# Reviewer inspects output:
state = app.get_state(config)
draft = state.values["draft_payload"]
# Continue to final delivery node:
app.stream(None, config)
    """, language="python")

# --- SECTION 4: SENSITIVE OPERATIONS ---
st.header("4. Sensitive Operations Governance", divider="rainbow")
st.markdown("""
Enterprises define **Sensitive Operations** as actions that cross trust boundaries, access personally identifiable information (PII), 
or affect operational availability. A dynamic policy router inspects tool calls and routes them according to classification tiers.
""")

col_sen1, col_sen2 = st.columns(2)
with col_sen1:
    st.subheader("Classification Matrix")
    st.markdown("""
    | Operational Scope | Sensitivity Level | Governance Requirement |
    | :--- | :--- | :--- |
    | Public Data Search | Low | 100% Autonomous |
    | User Profile Update | Medium | Single-Reviewer Pre-Approval |
    | Production Credentials | Critical | Multi-Factor Authentication Sign-off |
    | Cross-Border Data Transfer | High | Legal Compliance Clearance |
    """)
with col_sen2:
    st.subheader("Dynamic Sensitivity Router")
    st.code("""
def route_sensitivity(state: AgentState):
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return "END"
    
    for tool_call in last_msg.tool_calls:
        if tool_call["name"] in SENSITIVE_TOOL_ALLOWLIST:
            return "pre_approval_gate"
    
    return "standard_tools"
    """, language="python")

# --- SECTION 5: FINANCIAL OPERATIONS ---
st.header("5. Financial Operations & Dual-Control Gates", divider="rainbow")
st.markdown("""
Financial actions (wire transfers, payment settlements, invoicing, and refunds) are subject to stringent accounting controls. 
In high-value financial agents, systems implement the **Four-Eyes Principle (Dual-Control Gate)**, requiring two independent human 
officers to sign off before high-dollar disbursements occur.
""")

col_fin1, col_fin2 = st.columns(2)
with col_fin1:
    st.subheader("Financial Risk Tiers")
    st.markdown("""
    * **Tier A (Under 500 USD)**: Auto-cleared with asynchronous notification to manager.
    * **Tier B (500 to 25,000 USD)**: Single-officer synchronous authorization gate.
    * **Tier C (Over 25,000 USD)**: Dual-officer sign-off (Finance Officer + Department VP).
    * **Tier D (Foreign Currency / Unverified IBAN)**: Mandatory identity verification freeze.
    """)
with col_fin2:
    st.subheader("Dual-Control State Schema")
    st.code("""
class FinancialState(TypedDict):
    transfer_details: dict
    officer_1_approval: Optional[str]
    officer_2_approval: Optional[str]
    compliance_hash: str

def check_dual_clearance(state: FinancialState):
    amount = state["transfer_details"]["amount"]
    if amount > 25000:
        if not (state.get("officer_1_approval") and state.get("officer_2_approval")):
            return "await_second_officer"
    return "execute_disbursement"
    """, language="python")

# In-section diagram: Dual-Control Financial Gate
with st.expander("🔍 View Dual-Control Financial Gate Workflow", expanded=False):
    fin_dot = graphviz.Digraph(comment="Financial Dual Gate")
    fin_dot.attr(bgcolor="#0B1120", rankdir="TB")
    fin_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    fin_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    fin_dot.node("req", "Payment Tool Request\n[Amount: 120,000 USD]", fillcolor="#0F172A")
    fin_dot.node("gate1", "Officer 1 Review (Finance Manager)\n[Signed: officer_1=APPROVED]", fillcolor="#1E3A8A")
    fin_dot.node("gate2", "Officer 2 Review (VP Operations)\n[Signed: officer_2=APPROVED]", fillcolor="#701A75")
    fin_dot.node("disburse", "Core Banking API Dispatch\n[Disbursement Committed]", fillcolor="#065F46")

    fin_dot.edge("req", "gate1", label=" Interrupt 1")
    fin_dot.edge("gate1", "gate2", label=" Checkpoint Update & Interrupt 2")
    fin_dot.edge("gate2", "disburse", label=" Dual Signatures Verified")

    st.graphviz_chart(fin_dot, use_container_width=True)

# --- SECTION 6: EMAIL APPROVAL ---
st.header("6. Email & Outbound Communication Approval", divider="rainbow")
st.markdown("""
Autonomous outbound communication carries substantial brand, legal, and operational liabilities. 
An agent drafting customer emails, compliance warnings, or marketing broadcasts must submit drafted text for human curation.
""")

col_mail1, col_mail2 = st.columns(2)
with col_mail1:
    st.subheader("Email Guardrail Architecture")
    st.markdown("""
    * **Drafting vs. Sending Separation**: Never give the agent a unified `send_email` tool. Instead, provide `draft_email` and `dispatch_email` as distinct nodes.
    * **Subject & Body Staging**: Staged in graph state under `draft_outbound`.
    * **Recipient Audit**: Automated regex verification checks for sensitive internal domains or unintended BCC lists.
    * **Reviewer Interface**: Human reviewer can tweak phrasing, remove hallucinated commitments, and approve dispatch.
    """)
with col_mail2:
    st.subheader("Separation of Powers Code Pattern")
    st.code("""
# Node 1: Draft Creation (Agent Autonomy)
def draft_email_node(state):
    return {"draft": {"to": "client@corp.com", "body": "Your refund is approved."}}

# Gate: Interrupt before dispatch node
builder.add_node("draft_email", draft_email_node)
builder.add_node("dispatch_email", actual_smtp_dispatch)
builder.add_edge("draft_email", "dispatch_email")

app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["dispatch_email"]
)
    """, language="python")

# --- SECTION 7: DATABASE MODIFICATION APPROVAL ---
st.header("7. Database Modification & Destructive Action Approval", divider="rainbow")
st.markdown("""
Database tools executed by AI agents present significant risk of data corruption, unintended deletions (`DROP TABLE`, `TRUNCATE`), 
or massive unindexed scans that degrade cluster performance. HITL gates enforce SQL inspection before connection execution.
""")

col_db1, col_db2 = st.columns(2)
with col_db1:
    st.subheader("SQL Safety Classifications")
    st.markdown("""
    * **Read-Only Queries (`SELECT`)**:
      * Auto-approved if syntax parser verifies absence of data modification keywords (`UPDATE`, `DELETE`, `INSERT`).
      * Read query timeouts and limit caps (e.g., `LIMIT 100`) enforced automatically.
    * **Data Manipulation (`INSERT`, `UPDATE`)**:
      * Requires single-reviewer approval showing estimated impacted row count (`EXPLAIN`).
    * **Schema Definition / Truncation (`DROP`, `ALTER`, `TRUNCATE`)**:
      * Mandatory DBA sign-off with verified snapshot backup confirmation.
    """)
with col_db2:
    st.subheader("Dry-Run SQL Validation Gate")
    st.code("""
def database_review_gate(state):
    sql = state["proposed_query"]
    is_destructive = any(kw in sql.upper() for kw in ["DROP", "DELETE", "TRUNCATE"])
    
    if is_destructive:
        # Halt execution and require DBA sign-off
        return interrupt({
            "prompt": "Destructive SQL operation detected.",
            "query": sql,
            "estimated_impact": "Full Table Deletion"
        })
    return execute_query(sql)
    """, language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Multi-Tiered Dual-Gate Approval Engine", divider="rainbow")
st.markdown("""
The following complete production script illustrates a comprehensive enterprise approval engine incorporating pre-execution 
sensitivity classification, financial dual-control validation, and audit recording using LangGraph primitives.
""")

st.code('''
from typing import Annotated, TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class EnterpriseActionState(TypedDict):
    action_type: str        # "FINANCIAL", "COMMUNICATION", "DATABASE"
    payload: dict
    primary_approval: bool
    secondary_approval: bool
    rejection_reason: Optional[str]
    audit_log: list[str]

def risk_classifier_node(state: EnterpriseActionState) -> dict:
    action = state["action_type"]
    return {
        "audit_log": [f"Risk classifier evaluated action: {action}"]
    }

def primary_approval_gate(state: EnterpriseActionState) -> dict:
    # This node is interrupted before execution via interrupt_before
    return {
        "primary_approval": True,
        "audit_log": state["audit_log"] + ["Primary approval registered."]
    }

def secondary_approval_gate(state: EnterpriseActionState) -> dict:
    # Triggered only for high-value actions exceeding dual-control threshold
    return {
        "secondary_approval": True,
        "audit_log": state["audit_log"] + ["Secondary VP approval registered."]
    }

def execute_action_node(state: EnterpriseActionState) -> dict:
    return {
        "audit_log": state["audit_log"] + [f"Action executed successfully: {state['payload']}"]
    }

def route_next_gate(state: EnterpriseActionState) -> Literal["secondary_approval_gate", "execute_action_node"]:
    if state["action_type"] == "FINANCIAL" and state["payload"].get("amount", 0) > 50000:
        if not state.get("secondary_approval"):
            return "secondary_approval_gate"
    return "execute_action_node"

# Assemble Graph
workflow = StateGraph(EnterpriseActionState)
workflow.add_node("classifier", risk_classifier_node)
workflow.add_node("primary_approval_gate", primary_approval_gate)
workflow.add_node("secondary_approval_gate", secondary_approval_gate)
workflow.add_node("execute_action", execute_action_node)

workflow.add_edge(START, "classifier")
workflow.add_edge("classifier", "primary_approval_gate")
workflow.add_conditional_edges("primary_approval_gate", route_next_gate)
workflow.add_edge("secondary_approval_gate", "execute_action")
workflow.add_edge("execute_action", END)

# MemorySaver checkpointer enables thread interruptions
engine = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["primary_approval_gate", "secondary_approval_gate"]
)
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: High-Risk Action Guardrails Simulator", divider="rainbow")
st.markdown("""
Interact with simulated enterprise workflows awaiting review across **Financial**, **Outbound Communication**, 
**Database**, and **Post-Execution Audit** domains.
""")

# Instantiate Module Scope Simulator
if "workflows_sim" not in st.session_state:
    st.session_state.workflows_sim = MockHITLWorkflowsEngine()

engine = st.session_state.workflows_sim

col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.subheader("1. Pending Governance Queue")
    selected_op_id = st.selectbox(
        "Select Pending Operation:",
        options=list(engine.operations.keys()),
        format_func=lambda x: f"{x} - {engine.operations[x]['title']} ({engine.operations[x]['category']})"
    )
    current_op = engine.operations[selected_op_id]

    st.markdown(f"**Operation ID**: `{current_op['id']}`")
    st.markdown(f"**Category**: `{current_op['category']}`")
    st.markdown(f"**Gate Type**: `{current_op['gate_type']}`")
    st.markdown(f"**Status**: `{current_op['status']}`")
    st.markdown(f"**Risk Score**: `{current_op['risk_score']} / 100`")

    st.markdown("#### Staged Operation Payload:")
    st.json(current_op["payload"])

with col_w2:
    st.subheader("2. Reviewer Authorization Action")
    decision_type = st.radio(
        "Authorizer Action:",
        options=["Authorize Operation", "Veto / Reject Operation"],
        index=0
    )

    review_comment = st.text_input(
        "Audit Trail Notes / Compliance Justification:",
        value="Compliance checks verified against standard operating procedures."
    )

    if st.button("⚖️ Commit Reviewer Sign-Off", use_container_width=True):
        act_key = "AUTHORIZE" if decision_type == "Authorize Operation" else "REJECT"
        res = engine.evaluate_operation(selected_op_id, act_key, review_comment)
        if res["status"] == "SUCCESS":
            st.success(res["message"])
        else:
            st.error(res["message"])
        st.json(res)

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: What are the architectural differences between Pre-Execution and Post-Execution approval gates?"):
    st.markdown("""
    * **Pre-Execution Gate**: Halts execution *before* a tool node runs (`interrupt_before`). Essential for irreversible operations 
      (wire transfer, deleting records) where preventing the initial side effect is critical.
    * **Post-Execution Gate**: Allows a tool to execute (often in a sandbox or draft mode), halting *after* completion (`interrupt_after`). 
      Used when the generated artifact (contract draft, summary, translation) requires human curation before public distribution.
    """)

with st.expander("Q2: How do you prevent an AI agent from executing an unapproved SQL query while waiting for DBA authorization?"):
    st.markdown("""
    By enforcing strict separation between query formulation and execution. The agent writes SQL text into graph state. 
    The execution node has an `interrupt_before` gate. The database credentials exist *only* within the execution node, 
    making it architecturally impossible for the agent to bypass the gate and query the database directly.
    """)

with st.expander("Q3: How do you implement the Four-Eyes Principle (Dual-Control) in LangGraph?"):
    st.markdown("""
    By chaining two distinct approval nodes with checkpointers. Node 1 registers Officer 1's authorization into state 
    and triggers an interrupt. Upon resumption by Officer 2, the conditional router checks if Officer 2 is distinct from 
    Officer 1. If valid, Node 2 records the second sign-off and routes to the execution node.
    """)

with st.expander("Q4: Why should email drafting tools and email sending tools be separated into distinct graph nodes?"):
    st.markdown("""
    Combining drafting and sending into a single tool allows hallucinated arguments to reach live recipients with zero opportunity 
    for intervention. Separating drafting into a state mutation and sending into a gated execution node guarantees that drafts can 
    be inspected, revised, and approved prior to SMTP transport.
    """)

with st.expander("Q5: What happens if an interrupted thread in LangGraph receives a new user message before human review is completed?"):
    st.markdown("""
    In default configurations, injecting a new user message forks the conversation from the current checkpoint or overwrites 
    pending tasks depending on the channel reducer policy. In production, systems implement concurrency locks on threads 
    with pending reviews to prevent user messages from desynchronizing in-flight approvals.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_pt1, col_pt2 = st.columns(2)
with col_pt1:
    st.error("""
    **Production Trap 1 (Fatal Error): Single-Tool Draft-and-Send Anti-Pattern**  
    Bundling draft creation and network transmission into a single autonomous function call (`send_email(body=...)`) removes the 
    architectural boundary required for pre-execution inspection, exposing the organization to immediate email spoofing or data leakage.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Missing Read-Only SQL Sanitization**  
    Assuming that queries starting with `SELECT` are always safe. Attackers can leverage stacked queries (e.g., `SELECT 1; DROP TABLE users;`) 
    or write-side subqueries. Always enforce AST-level SQL validation before classifying an operation as read-only.
    """)
with col_pt2:
    st.error("""
    **Production Trap 3 (Fatal Error): Authorization Token Leakage in Graph State**  
    Storing sensitive administrator authorization passwords or private signing keys directly inside `StateGraph` dictionary values. 
    These state dictionaries are persisted to database checkpointers in plain text. Always store review authorization claims as signed JWT references.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Reviewer Fatigue Leading to Blind Approvals**  
    Requiring human approval for every trivial operation creates reviewer fatigue, causing operators to blindly approve dangerous requests. 
    Calibrate risk thresholds carefully so only genuinely sensitive actions trigger interrupts.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 1: Pre-Execution Gate</h4>
        <p><strong>Concept:</strong> <code>interrupt_before</code></p>
        <p><strong>Definition:</strong> Halting execution prior to entering a tool node to inspect and authorize irreversible actions before side effects occur.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 2: Post-Execution Gate</h4>
        <p><strong>Concept:</strong> <code>interrupt_after</code></p>
        <p><strong>Definition:</strong> Halting execution after node execution finishes to allow human review of generated content or artifacts prior to downstream delivery.</p>
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 3: Four-Eyes Principle</h4>
        <p><strong>Concept:</strong> Dual-Control Authorization</p>
        <p><strong>Definition:</strong> A compliance control requiring two independent human operators to sign off on high-value actions (e.g. transfers exceeding 50,000 USD) before execution.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 4: Separation of Powers</h4>
        <p><strong>Concept:</strong> Draft vs. Dispatch Split</p>
        <p><strong>Definition:</strong> Splitting dangerous workflows into a staging node (generates payload) and a dispatch node (executes side effect) separated by an approval gate.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of enterprise HITL workflow architectures:")

c1 = st.checkbox("1. I understand when to implement pre-execution approval versus post-execution quality gates.")
c2 = st.checkbox("2. I know how to configure `interrupt_before` and `interrupt_after` for tool nodes in LangGraph.")
c3 = st.checkbox("3. I can design a dynamic risk classifier node to route low-risk vs. high-risk actions into separate branches.")
c4 = st.checkbox("4. I understand how to implement the Four-Eyes Principle (Dual-Control) for high-value financial actions.")
c5 = st.checkbox("5. I understand the importance of separating drafting from sending in outbound email and messaging workflows.")
c6 = st.checkbox("6. I know how to protect production databases by gating schema modification and truncate commands behind DBA sign-off.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Human-in-the-Loop Workflows in LangChain & LangGraph.")
