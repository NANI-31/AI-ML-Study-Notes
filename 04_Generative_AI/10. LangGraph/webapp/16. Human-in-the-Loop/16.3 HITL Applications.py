import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="16.3 HITL Applications - LangGraph Curriculum",
    page_icon="🏢",
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
    .badge-app {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-auth {
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
class HITLApplicationMatrixEngine:
    """
    Simulates the 6 primary real-world enterprise HITL application domains:
    1. Financial Approval
    2. Email Approval
    3. Tool Approval
    4. Content Moderation
    5. High-Risk Operations
    6. Human Escalation
    """
    def __init__(self):
        self.domains = {
            "Financial Approval": {
                "risk_tier": "CRITICAL",
                "trigger_condition": "Transaction amount > 10,000 USD",
                "default_action": "Execute wire transfer via SWIFT API",
                "human_gate": "Requires dual-sign-off by Finance Director"
            },
            "Email Approval": {
                "risk_tier": "HIGH",
                "trigger_condition": "Outbound blast to > 500 enterprise customers",
                "default_action": "Send broadcast message via SendGrid",
                "human_gate": "Requires copy review by Communications Lead"
            },
            "Tool Approval": {
                "risk_tier": "MEDIUM-HIGH",
                "trigger_condition": "Invoking SQL mutation (DELETE / UPDATE / DROP)",
                "default_action": "Execute write query against PostgreSQL cluster",
                "human_gate": "Requires query plan sign-off by DBA"
            },
            "Content Moderation": {
                "risk_tier": "HIGH",
                "trigger_condition": "LLM output flagged for sensitive compliance keywords",
                "default_action": "Publish knowledge base article to public portal",
                "human_gate": "Requires legal / trust & safety audit"
            },
            "High-Risk Operations": {
                "risk_tier": "CRITICAL",
                "trigger_condition": "Kubernetes cluster restart or VPC security group update",
                "default_action": "Apply Terraform infrastructure modification",
                "human_gate": "Requires DevOps on-call confirmation"
            },
            "Human Escalation": {
                "risk_tier": "VARIABLE",
                "trigger_condition": "Customer sentiment analysis detects anger or legal threats",
                "default_action": "Continue automated self-service chat",
                "human_gate": "Transfers conversation to live human agent"
            }
        }

    def simulate_gate_decision(self, domain: str, decision: str, edited_data: Optional[str]) -> Dict[str, Any]:
        info = self.domains[domain]
        if decision == "APPROVE":
            outcome = "Authorized: Action dispatched to production infrastructure."
            status = "APPROVED"
        elif decision == "REJECT":
            outcome = "Cancelled: Action aborted cleanly with zero side-effects."
            status = "REJECTED"
        else:
            outcome = f"Modified: Action updated with payload: '{edited_data}' and dispatched."
            status = "MODIFIED_AND_EXECUTED"

        return {
            "domain": domain,
            "risk_tier": info["risk_tier"],
            "decision": decision,
            "status": status,
            "outcome": outcome
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise HITL Applications and Governance Matrix", divider="rainbow")

st.markdown("""
Human-in-the-Loop is not a theoretical pattern: it is the **foundational safety architecture**
powering production enterprise AI deployments.
From financial compliance to email marketing, database administration, and customer service escalation,
embedding `interrupt()` gates ensures that AI agents operate safely within organizational risk tolerances.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    hitl_apps_blueprint_dot = """
    digraph HITLApplicationsBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        AGENT_DISPATCH [label="Autonomous Agent Proposes Action\\n(Analyzes task & formulates action payload)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_matrix {
            label = "Enterprise Governance Matrix & Gates";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            GATE_FINANCE [label="1. Financial Gate\\namount > 10,000 USD\\ninterrupt({'type': 'finance'})", fillcolor="#431407", color="#FB923C"];
            GATE_EMAIL [label="2. Outbound Comms Gate\\nrecipients > 500\\ninterrupt({'type': 'email'})", fillcolor="#2E1065", color="#C084FC"];
            GATE_DB [label="3. Destructive Tool Gate\\nSQL write / drop table\\ninterrupt({'type': 'db_ops'})", fillcolor="#3B1A1A", color="#F87171"];
            GATE_MODERATION [label="4. Content Safety Gate\\nBrand compliance audit\\ninterrupt({'type': 'moderation'})", fillcolor="#0F172A", color="#38BDF8"];
            GATE_INFRA [label="5. Infra DevOps Gate\\nTerraform / K8s mutation\\ninterrupt({'type': 'devops'})", fillcolor="#450A0A", color="#F87171"];
            GATE_ESCALATE [label="6. Customer Escalation\\nSentiment anger / legal threat\\ninterrupt({'type': 'escalate'})", fillcolor="#1E1E38", color="#8B5CF6"];
        }

        POSTGRES_CHECKPOINT [label="Committed to PostgresSaver\\nExecution suspended awaiting sign-off", fillcolor="#0F172A", color="#34D399"];

        AGENT_DISPATCH -> GATE_FINANCE;
        AGENT_DISPATCH -> GATE_EMAIL;
        AGENT_DISPATCH -> GATE_DB;
        AGENT_DISPATCH -> GATE_MODERATION;
        AGENT_DISPATCH -> GATE_INFRA;
        AGENT_DISPATCH -> GATE_ESCALATE;

        GATE_FINANCE -> POSTGRES_CHECKPOINT;
        GATE_EMAIL -> POSTGRES_CHECKPOINT;
        GATE_DB -> POSTGRES_CHECKPOINT;
        GATE_MODERATION -> POSTGRES_CHECKPOINT;
        GATE_INFRA -> POSTGRES_CHECKPOINT;
        GATE_ESCALATE -> POSTGRES_CHECKPOINT;
    }
    """
    st.graphviz_chart(hitl_apps_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Production Takeaway:</b> Enterprises do not reject AI agents because of accuracy:
they reject them because of liability. Implementing verifiable HITL gates satisfies internal audit
and legal requirements, unlocking deployment across mission-critical domains.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: FINANCIAL APPROVAL
# ==============================================================================
st.header("2. Financial Approval: Large Transaction Gates and Dual-Sign-Off Protocols", divider="rainbow")

st.markdown("""
Financial workflows enforce hard dollar thresholds. Any transaction exceeding the limit
automatically interrupts for human sign-off:
""")

st.code("""
def wire_transfer_gate(state: FinancialState) -> dict:
    if state["transfer_amount_usd"] > 10000.0:
        approval = interrupt({
            "action": "wire_transfer",
            "amount": state["transfer_amount_usd"],
            "recipient": state["destination_account"]
        })
        if not approval.get("authorized"):
            return {"status": "ABORTED_BY_SUPERVISOR"}
            
    execute_swift_transfer(state["transfer_amount_usd"])
    return {"status": "COMMITTED"}
""", language="python")

# ==============================================================================
# SECTION 3: EMAIL APPROVAL
# ==============================================================================
st.header("3. Email Approval: Human Review and In-Flight Modification of Outbound Comms", divider="rainbow")

st.markdown("""
Customer-facing communications drafted by LLMs should be reviewed before sending.
The human can read the draft, edit typos, and approve the email in-flight:
""")

st.code("""
def email_review_node(state: MessagesState) -> dict:
    draft = state["drafted_email"]
    
    # Reviewer can edit the draft content in the web UI
    user_decision = interrupt({"draft_subject": draft["subject"], "draft_body": draft["body"]})
    
    # Send the approved or edited version
    final_body = user_decision.get("edited_body", draft["body"])
    sendgrid_client.send_email(to=draft["to"], body=final_body)
    return {"status": "SENT"}
""", language="python")

# ==============================================================================
# SECTION 4: TOOL APPROVAL
# ==============================================================================
st.header("4. Tool Approval: Pre-Execution Authorization Gates on Destructive Functions", divider="rainbow")

st.markdown("""
Rather than placing interrupts around the entire graph, you can place approval gates
directly around specific **destructive tools**:
""")

st.code("""
@tool
def execute_sql_mutation(query: str) -> str:
    \"\"\"Execute write or delete SQL query.\"\"\"
    # Gate within the tool itself!
    confirmed = interrupt({"sql_query": query, "risk": "DATABASE_MUTATION"})
    if confirmed.get("allow"):
        return run_sql(query)
    return "Error: Database mutation rejected by DBA."
""", language="python")

# ==============================================================================
# SECTION 5: CONTENT MODERATION
# ==============================================================================
st.header("5. Content Moderation: Safety Scrubbing and Brand Voice Verification", divider="rainbow")

st.markdown("""
Before articles, FAQs, or press releases generated by an AI agent are published to public portals,
a content moderation gate presents the text to brand compliance reviewers for verification.
""")

# ==============================================================================
# SECTION 6: HIGH-RISK OPERATIONS
# ==============================================================================
st.header("6. High-Risk Operations: Infrastructure Mutators, Database Drops, and Credential Grants", divider="rainbow")

st.markdown("""
Operations like restarting Kubernetes clusters, flushing Redis production caches, or granting IAM permissions
must mandate human review to prevent catastrophic operational downtime.
""")

# ==============================================================================
# SECTION 7: HUMAN ESCALATION
# ==============================================================================
st.header("7. Human Escalation: Automated Paging and Supervisor Handoffs", divider="rainbow")

st.markdown("""
In customer service bots, sentiment analysis models monitor conversation turns.
If anger, frustration, or legal threats are detected, the graph interrupts and pages a human agent:
""")

st.code("""
def customer_triage_node(state: MessagesState) -> dict:
    sentiment = analyze_sentiment(state["messages"][-1].content)
    
    if sentiment == "HOSTILE" or "lawyer" in state["messages"][-1].content.lower():
        # Interrupt and notify human representative
        human_agent_reply = interrupt({
            "escalation_reason": "Customer expressed intense frustration",
            "full_history": state["messages"]
        })
        return {"messages": [AIMessage(content=f"[Human Agent]: {human_agent_reply['reply']}")]}
        
    return {"messages": [model.invoke(state["messages"])]}
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Multi-Domain Enterprise Approval Portal", divider="rainbow")

st.markdown("""
Below is the complete production implementation of an enterprise governance portal
managing multi-domain approval gates:
""")

st.code("""
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. State Definition
class EnterpriseGovernanceState(MessagesState):
    operation_domain: Literal["FINANCE", "EMAIL", "INFRASTRUCTURE"]
    payload: dict
    authorized: bool

# 2. Gate Node
def governance_gate_node(state: EnterpriseGovernanceState) -> dict:
    op = state["operation_domain"]
    data = state["payload"]
    
    # Generic interrupt structured for enterprise portal
    decision = interrupt({
        "domain": op,
        "details": data,
        "severity": "CRITICAL" if op in ["FINANCE", "INFRASTRUCTURE"] else "HIGH"
    })
    
    return {
        "authorized": decision.get("approved", False),
        "messages": [AIMessage(content=f"Governance review completed. Decision: {decision.get('approved')}")]
    }

def execution_node(state: EnterpriseGovernanceState) -> dict:
    if state["authorized"]:
        return {"messages": [AIMessage(content="Action successfully committed to enterprise backend.")]}
    return {"messages": [AIMessage(content="Action cancelled per governance policy.")]}

# 3. Assemble Graph
builder = StateGraph(EnterpriseGovernanceState)
builder.add_node("gate", governance_gate_node)
builder.add_node("execute", execution_node)

builder.add_edge(START, "gate")
builder.add_edge("gate", "execute")
builder.add_edge("execute", END)

checkpointer = MemorySaver()
governance_app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Enterprise HITL Operations Console & Approval Sandbox", divider="rainbow")

st.markdown("""
Test how approval gates behave across different enterprise domains.
Select a domain, choose a supervisor action, and inspect the governance outcome.
""")

matrix_engine = HITLApplicationMatrixEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Domain Selection")
    chosen_domain = st.selectbox(
        "Select Enterprise Application Domain:",
        list(matrix_engine.domains.keys())
    )
    
    domain_info = matrix_engine.domains[chosen_domain]
    st.info(f"**Risk Tier**: `{domain_info['risk_tier']}`\\n- **Trigger Condition**: {domain_info['trigger_condition']}\\n- **Default Action**: {domain_info['default_action']}\\n- **Human Gate Requirement**: {domain_info['human_gate']}")

    supervisor_choice = st.radio("Supervisor Review Action:", ["APPROVE", "REJECT", "EDIT_AND_APPROVE"])
    edit_str = None
    if supervisor_choice == "EDIT_AND_APPROVE":
        edit_str = st.text_input("Modified Action Payload:", value="Target reduced to 5,000 USD / Single Region")

    commit_action_btn = st.button("🚀 Commit Governance Decision", type="primary")

with wb_col2:
    st.subheader("📋 Governance Audit Telemetry")
    if commit_action_btn:
        res = matrix_engine.simulate_gate_decision(chosen_domain, supervisor_choice, edit_str)
        badge = "badge-auth" if "APPROVE" in res["decision"] else "badge-app"
        
        st.markdown(f"### Status: <span class='{badge}'>{res['status']}</span>", unsafe_allow_html=True)
        st.write(f"- **Domain Evaluated**: `{res['domain']}`")
        st.write(f"- **Risk Level**: `{res['risk_tier']}`")
        st.success(res["outcome"])
    else:
        st.info("Select a domain on the left and submit a decision to inspect the audit log.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What are the 6 primary real-world application domains for HITL in enterprise AI?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Financial Approvals**: Wiring funds or executing refund disbursements.
    2. **Email & Comms**: Reviewing marketing broadcasts or customer dispute letters.
    3. **Tool Approvals**: Authorizing write/delete operations on production databases.
    4. **Content Moderation**: Ensuring public knowledge articles conform to safety standards.
    5. **Infrastructure DevOps**: Authorizing Kubernetes cluster scaling or Terraform mutations.
    6. **Customer Escalation**: Handoffs from chatbot to human representative upon detecting anger or legal threats.
    """)

with st.expander("Q2: How can an interrupt gate be embedded directly inside a tool function?", expanded=False):
    st.markdown("""
    **Answer**:
    By calling `interrupt()` directly inside the tool callable decorated with `@tool`.
    When the tool is executed by `ToolNode`, it hits the `interrupt()`, suspends graph execution, and passes the tool's
    arguments to the reviewer.
    """)

with st.expander("Q3: How do you implement dual-sign-off in LangGraph financial applications?", expanded=False):
    st.markdown("""
    **Answer**:
    By chaining two sequential `interrupt()` calls or verifying in the node state that two distinct supervisor
    user IDs have submitted approvals in `Command(resume=...)` before executing the transfer.
    """)

with st.expander("Q4: What role does sentiment analysis play in human escalation workflows?", expanded=False):
    st.markdown("""
    **Answer**:
    Sentiment models evaluate incoming user turns for hostility, frustration, or legal terminology.
    When a threshold is breached, the graph branches to an escalation node that pauses execution and pages a live human.
    """)

with st.expander("Q5: Why is audit logging critical for HITL governance?", expanded=False):
    st.markdown("""
    **Answer**:
    Enterprise compliance requires proving *who* authorized an action, *when* it was signed off, and *what* data was presented.
    Checkpointers preserve the exact immutable state snapshot and approval payload for regulatory compliance.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Bypassing Approval Gates via Direct Tool Invocation")
    st.markdown("""
    *Mechanism*: Binding destructive tools to the model without gatekeeper nodes or embedded interrupts.  
    *Consequence*: Prompt injection attacks trick the model into executing unauthorized deletions autonomously.  
    *Fix*: Always gate destructive tools behind `interrupt()` checks.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing Role-Based Access Control on Review Endpoints")
    st.markdown("""
    *Mechanism*: Allowing any authenticated user to resume high-value financial interrupts.  
    *Consequence*: Privilege escalation vulnerabilities.  
    *Fix*: Verify supervisor IAM permissions before calling `Command(resume=...)`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Reviewer Fatigue from Over-Gating")
    st.markdown("""
    *Mechanism*: Forcing human approval on trivial, low-risk requests (e.g. transfers < 10 USD).  
    *Consequence*: Supervisors approve requests blindly without reading due to alert fatigue.  
    *Fix*: Implement dynamic thresholds (e.g. gate only when amount > 5,000 USD).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Missing SLA Escalation on Unanswered Approvals")
    st.markdown("""
    *Mechanism*: An approval sits in queue for 24 hours while the assigned supervisor is out of office.  
    *Consequence*: Customer SLA deadlines are breached.  
    *Fix*: Implement timer cron jobs that re-route pending approvals to backup managers.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Threshold Gating**\\nTriggering approval interrupts only when numeric metrics (cost, risk, recipient count) exceed limits.")
with col_fc2:
    st.info("**In-Flight Editing**\\nThe ability for human reviewers to adjust tool arguments before approving execution.")
with col_fc3:
    st.info("**Sentiment Escalation**\\nAutomatically transferring users from automated agents to humans upon detecting distress.")
with col_fc4:
    st.info("**Dual-Sign-Off**\\nRequiring multiple independent human authorizations for high-liability transactions.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the 6 primary real-world enterprise HITL application domains.", value=True)
c2 = st.checkbox("I understand how to implement threshold-based gating for financial transactions.", value=True)
c3 = st.checkbox("I can embed interrupt() calls directly inside tool functions.", value=True)
c4 = st.checkbox("I know how to implement in-flight editing of email drafts before sending.", value=True)
c5 = st.checkbox("I understand how to design sentiment-triggered customer escalation workflows.", value=True)
c6 = st.checkbox("I know how to prevent reviewer fatigue by tuning approval thresholds.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 16.3 'HITL Applications' with full architectural and practical comprehension!")
