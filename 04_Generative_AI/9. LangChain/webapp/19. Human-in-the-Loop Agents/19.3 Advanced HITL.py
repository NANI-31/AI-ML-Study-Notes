import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.3 Advanced HITL - LangChain Architecture",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and escalation levels
st.markdown("""
<style>
    .hitl-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-l1 {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-l2 {
        background-color: #78350F;
        color: #FDE68A;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-l3 {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockAdvancedHITLEngine:
    """
    Enterprise simulator for Advanced Human-in-the-Loop workflows.
    Demonstrates dynamic interrupts, persistent checkpointer rehydration,
    multi-level escalation (L1 Analyst -> L2 Lead -> L3 Executive),
    human feedback conditioning, and cryptographic-style audit trail logging.
    """
    def __init__(self):
        self.incidents: Dict[str, Dict[str, Any]] = {
            "INC-8801": {
                "id": "INC-8801",
                "title": "Vendor Bank Account Modification",
                "tier": "Level 1 (Ops Analyst)",
                "sla_seconds": 300,
                "elapsed_seconds": 380,  # Breached SLA, ready for escalation
                "status": "ESCALATED_TO_L2",
                "payload": {
                    "vendor_name": "Apex Global Cloud Ltd",
                    "new_iban": "GB29BARC20041539201940",
                    "old_iban": "GB99MIDL40051511223344",
                    "change_reason": "Corporate restructuring"
                },
                "feedback_history": ["Initial automated validation: Flagged foreign bank routing."],
                "audit_trail": [
                    {"step": "T0: Created", "actor": "agent_worker_pod_3", "action": "interrupt_triggered"},
                    {"step": "T1: SLA Breached", "actor": "sla_watcher_cron", "action": "escalated_to_L2"}
                ]
            },
            "INC-9942": {
                "id": "INC-9942",
                "title": "Autonomous Infrastructure Scale-Down",
                "tier": "Level 2 (Engineering Lead)",
                "sla_seconds": 600,
                "elapsed_seconds": 120,
                "status": "AWAITING_REVIEW",
                "payload": {
                    "cluster_id": "k8s-prod-eu-west-1",
                    "nodes_to_drain": 24,
                    "estimated_savings_monthly_usd": 18500.00,
                    "target_idle_metric": "CPU < 15% for 48h"
                },
                "feedback_history": ["Metrics verified against Prometheus telemetry."],
                "audit_trail": [
                    {"step": "T0: Created", "actor": "infra_agent_01", "action": "interrupt_triggered"}
                ]
            }
        }

    def escalate_incident(self, inc_id: str, to_tier: str, reason: str) -> Dict[str, Any]:
        inc = self.incidents.get(inc_id)
        if not inc:
            return {"status": "ERROR", "message": "Incident not found."}
        inc["tier"] = to_tier
        inc["status"] = f"ESCALATED_TO_{to_tier.split()[0].upper()}"
        inc["audit_trail"].append({
            "step": f"Escalated to {to_tier}",
            "actor": "system_escalation_router",
            "action": f"Reason: {reason}"
        })
        return {
            "status": "SUCCESS",
            "message": f"Incident {inc_id} successfully escalated to {to_tier}.",
            "incident": inc
        }

    def resolve_with_feedback(self, inc_id: str, reviewer: str, feedback: str, action: str) -> Dict[str, Any]:
        inc = self.incidents.get(inc_id)
        if not inc:
            return {"status": "ERROR", "message": "Incident not found."}
        inc["status"] = "RESOLVED_COMMITTED" if action == "APPROVE" else "VETOED_ABORTED"
        inc["feedback_history"].append(f"{reviewer}: {feedback}")
        inc["audit_trail"].append({
            "step": f"Resolution: {action}",
            "actor": reviewer,
            "action": f"Decision: {action} | Feedback: {feedback}"
        })
        return {
            "status": "SUCCESS",
            "message": f"Incident {inc_id} resolved with status {inc['status']}.",
            "incident": inc
        }

# Main Application Title & Objective
st.title("🛡️ 19.3 Advanced Human-in-the-Loop (HITL)")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Advanced Human-in-the-Loop systems move beyond basic stop-and-wait triggers into **durable distributed state management**, 
**dynamic in-node interrupts**, **multi-level escalation hierarchies with SLA timers**, **in-context human feedback loops**, 
and **immutable compliance audit trails**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Advanced HITL Escalation Matrix & Immutable Audit Engine", divider="rainbow")
st.markdown("""
The architecture below illustrates the complete lifecycle of a mission-critical operation handled by an enterprise HITL engine: 
from in-node dynamic interruption and checkpointer hydration, to multi-tier escalation, reviewer feedback injection, 
and append-only audit trail logging.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    adv_dot = graphviz.Digraph(comment="Advanced HITL")
    adv_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    adv_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    adv_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    adv_dot.node("agent_node", "Agent Reasoning Node\n[Dynamic interrupt() Check]", fillcolor="#0F172A", shape="ellipse")
    adv_dot.node("ckpt_store", "Durable Checkpointer\n[PostgreSQL / Redis Multi-Pod Store]", fillcolor="#0E7490")
    adv_dot.node("sla_watcher", "SLA Watcher & Timer\n[Tracks Unresolved Interrupt Duration]", shape="diamond", fillcolor="#334155")
    
    # Escalation Tiers
    adv_dot.node("tier_l1", "Level 1: Operations Analyst\n[Standard Review Queue - SLA: 5m]", fillcolor="#1E3A8A")
    adv_dot.node("tier_l2", "Level 2: Engineering / Finance Lead\n[Escalated Queue - SLA: 15m]", fillcolor="#78350F")
    adv_dot.node("tier_l3", "Level 3: Executive VP Sign-off\n[High Consequence Tier - SLA: 1h]", fillcolor="#7F1D1D")

    # Feedback and Resumption
    adv_dot.node("feedback_gate", "Reviewer Feedback Injection\n[State Mutation: update_state()]", fillcolor="#4338CA")
    adv_dot.node("audit_log", "Immutable Audit Engine\n[Cryptographic SHA-256 Event Log]", fillcolor="#047857")
    adv_dot.node("resume_node", "Graph Resumption Node\n[app.stream(None, config)]", fillcolor="#065F46", shape="ellipse")

    adv_dot.edge("agent_node", "ckpt_store", label=" State Persisted")
    adv_dot.edge("ckpt_store", "sla_watcher")
    adv_dot.edge("sla_watcher", "tier_l1", label=" Normal Route")
    adv_dot.edge("sla_watcher", "tier_l2", label=" SLA Timeout / Escalation")
    adv_dot.edge("tier_l2", "tier_l3", label=" Critical Override")

    adv_dot.edge("tier_l1", "feedback_gate", label=" L1 Resolution")
    adv_dot.edge("tier_l2", "feedback_gate", label=" L2 Resolution")
    adv_dot.edge("tier_l3", "feedback_gate", label=" L3 Resolution")

    adv_dot.edge("feedback_gate", "audit_log", label=" Record Decision")
    adv_dot.edge("audit_log", "resume_node", label=" Resume Execution")

    st.graphviz_chart(adv_dot, use_container_width=True)

# --- SECTION 2: INTERRUPTS ---
st.header("2. Dynamic In-Node Interrupts & Pregel Breakpoints", divider="rainbow")
st.markdown("""
While compile-time static interrupts (`interrupt_before=["tools"]`) halt unconditionally, advanced systems require **Dynamic Interrupts**. 
LangGraph provides the native `interrupt()` function, allowing any node to pause execution conditionally based on dynamic runtime data.
""")

col_int1, col_int2 = st.columns(2)
with col_int1:
    st.subheader("Dynamic `interrupt()` Capabilities")
    st.markdown("""
    * **In-Flight Data Inspection**: Examine exact variable values, user tier, or computed risk scores before deciding to halt.
    * **Bidirectional Value Exchange**: Calling `res = interrupt(prompt_data)` pauses execution. When resumed, `res` receives the human reviewer's input directly!
    * **Zero Compile-Time Static Binding**: Nodes determine on-the-fly whether human intervention is required, avoiding unnecessary pauses for routine actions.
    """)
with col_int2:
    st.subheader("Dynamic Interrupt Example")
    st.code("""
from langgraph.types import interrupt

def execute_financial_wire(state: dict) -> dict:
    amount = state["amount"]
    
    # Only pause if amount exceeds threshold
    if amount > 10000:
        # Halt execution and yield payload to reviewer
        reviewer_decision = interrupt({
            "action": "wire_transfer",
            "amount": amount,
            "recipient": state["recipient"]
        })
        
        # When resumed, reviewer_decision contains human response
        if reviewer_decision.get("approved") is False:
            return {"status": "VETOED_BY_HUMAN"}
            
    return {"status": "DISPATCHED_TO_GATEWAY"}
    """, language="python")

# --- SECTION 3: RESUMING EXECUTION ---
st.header("3. Deterministic Resumption Protocols & Event Streaming", divider="rainbow")
st.markdown("""
Resuming an interrupted graph requires explicit identification of the execution thread and state version. 
In distributed systems, the worker that resumes the graph is rarely the same pod that paused it.
""")

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.subheader("Resumption Protocols")
    st.markdown("""
    1. **Resuming Dynamic Interrupts with `Command(resume=...)`**:
       * LangGraph v0.2+ introduces `Command(resume=value)` to inject human responses directly into the waiting `interrupt()` call.
    2. **Resuming Static Interrupts with `stream(None, config)`**:
       * Passing `None` triggers Pregel to reload the latest checkpoint for `thread_id` and execute the pending node.
    3. **Event Stream Continuity**:
       * Resumption streams events seamlessly as if execution were never paused, maintaining continuous telemetry for connected frontends.
    """)
with col_res2:
    st.subheader("Command Resumption Code")
    st.code("""
from langgraph.types import Command

# Provide reviewer response to the interrupted node
human_response = {
    "approved": True,
    "reviewer": "sarah.chen@enterprise.com",
    "notes": "Verified invoices match PO-9921"
}

# Resume the graph by passing Command into stream()
config = {"configurable": {"thread_id": "thread-fin-99"}}

for event in graph.stream(Command(resume=human_response), config):
    print("Resumed Node Event:", event)
    """, language="python")

# --- SECTION 4: PERSISTENT STATE ---
st.header("4. Persistent State Architecture Across Distributed Pods", divider="rainbow")
st.markdown("""
Production HITL requires **stateless web servers backed by durable distributed checkpointers**. 
Human reviews take minutes, hours, or days. Storing thread state in RAM (`MemorySaver`) guarantees data loss during deployments or pod restarts.
""")

st.markdown("""
### Enterprise Checkpointer Topology: PostgreSQL vs. Redis
| Architecture Metric | MemorySaver (Dev Only) | RedisSaver (Fast Cache) | PostgresSaver (Production Standard) |
| :--- | :--- | :--- | :--- |
| **Durability** | Ephemeral (Lost on restart) | High (with AOF persistence) | ACID Guaranteed (WAL backed) |
| **Horizontal Pod Scaling** | Not supported (Single node) | Yes (Shared cluster) | Yes (Connection pooled cluster) |
| **History & Time Travel** | In-memory history | In-memory with TTL | Queryable SQL checkpoint table |
| **Cold State Recovery** | Impossible | Supported | High performance indexing by thread_id |
""")

st.code("""
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Enterprise PostgreSQL Checkpointer with connection pooling
DB_URI = "postgresql://agent_user:secure_pwd@db.internal:5432/agent_state"

with ConnectionPool(conninfo=DB_URI, max_size=20) as pool:
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # Automatically initializes checkpoint tables
    
    # Graph is compiled with durable PostgreSQL persistence
    app = builder.compile(checkpointer=checkpointer)
""", language="python")

# --- SECTION 5: HUMAN FEEDBACK ---
st.header("5. Human Feedback Injection & In-Context Reinforcement", divider="rainbow")
st.markdown("""
HITL is not merely a binary pass/fail gate; it is a **continuous feedback engine**. When an operator reviews an agent's 
intermediate thoughts or drafted responses, qualitative feedback can be injected back into the prompt context to guide self-correction.
""")

col_fb1, col_fb2 = st.columns(2)
with col_fb1:
    st.subheader("Feedback Feedback Topologies")
    st.markdown("""
    * **Prompt Conditioning**: Injects reviewer critique into the conversation history as a system or human message (e.g., *"Critique: The tone is too informal; revise using formal enterprise legal diction"*).
    * **Parameter Fine-Tuning**: Rewrites specific structured arguments before execution.
    * **Model Self-Correction**: Forces the model to reflect on the feedback and output an updated trajectory without restarting the conversation from scratch.
    """)
with col_fb2:
    st.subheader("Feedback Injection Pattern")
    st.code("""
# Operator submits qualitative feedback:
feedback = "Please remove technical jargon and emphasize warranty terms."

# Inject feedback as HumanMessage directed to the agent
graph.update_state(
    config={"configurable": {"thread_id": "T-404"}},
    values={"messages": [HumanMessage(content=f"OPERATOR FEEDBACK: {feedback}")]}
)

# Resume graph so agent refines its response
for chunk in graph.stream(None, config):
    print(chunk)
    """, language="python")

# --- SECTION 6: ESCALATION WORKFLOWS ---
st.header("6. Escalation Workflows & SLA Timers", divider="rainbow")
st.markdown("""
In enterprise operations, human reviewers may be unavailable, leading to bottlenecked workflows. 
Advanced HITL architectures incorporate **SLA Timers and Escalation Matrices** to route unattended reviews up the management hierarchy.
""")

col_esc1, col_esc2 = st.columns(2)
with col_esc1:
    st.subheader("Escalation Tier Matrix")
    st.markdown("""
    * **Tier 1 (Frontline Operations)**: Target SLA = 5 minutes. Initial triage and verification of routine sensitive operations.
    * **Tier 2 (Engineering / Finance Lead)**: Target SLA = 15 minutes. Automatically escalated if Tier 1 times out or if anomaly score exceeds 75%.
    * **Tier 3 (VP / Chief Security Officer)**: Target SLA = 1 hour. High-consequence destructive actions or dual-veto appeals.
    * **Fallback Policy**: If all tiers time out, fail-safe rule applies: **Default to Abort / Deny** to guarantee security.
    """)
with col_esc2:
    st.subheader("SLA Watchdog Architecture")
    st.markdown("""
    An independent background cron worker (e.g., Celery / Temporal / AWS SQS) queries the checkpointer for threads where:
    ```sql
    status = 'INTERRUPTED' 
    AND now() - checkpoint_timestamp > interval '5 minutes'
    ```
    Upon match, the watchdog triggers an escalation event, updating the thread state and notifying the next tier.
    """)

# In-section diagram: Escalation Hierarchy
with st.expander("🔍 View Enterprise Escalation Matrix & SLA Timers Flow", expanded=False):
    esc_dot = graphviz.Digraph(comment="Escalation Matrix")
    esc_dot.attr(bgcolor="#0B1120", rankdir="TB")
    esc_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    esc_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    esc_dot.node("int_point", "Interrupt Reached\n[Thread Assigned: T-505]", fillcolor="#0F172A")
    esc_dot.node("l1", "Level 1: Analyst Queue\n[SLA: 5 Minutes]", fillcolor="#1E3A8A")
    esc_dot.node("l2", "Level 2: Team Lead Queue\n[SLA: 15 Minutes]", fillcolor="#78350F")
    esc_dot.node("l3", "Level 3: Executive Tier\n[SLA: 60 Minutes]", fillcolor="#7F1D1D")
    esc_dot.node("fail_safe", "Fail-Safe Default: AUTO-VETO\n[Security Guarantee: Zero Unchecked Mutations]", fillcolor="#881337")

    esc_dot.edge("int_point", "l1")
    esc_dot.edge("l1", "l2", label=" SLA Timeout (> 5m)")
    esc_dot.edge("l2", "l3", label=" SLA Timeout (> 15m)")
    esc_dot.edge("l3", "fail_safe", label=" SLA Timeout (> 60m)")

    st.graphviz_chart(esc_dot, use_container_width=True)

# --- SECTION 7: AUDIT TRAILS ---
st.header("7. Cryptographic & Immutable Audit Trails for Compliance", divider="rainbow")
st.markdown("""
Regulated industries (financial services, healthcare, defense) require an **unalterable record of all autonomous decisions and human interventions**. 
An immutable audit trail records every state transition, reviewer signature, IP address, and payload hash.
""")

col_aud1, col_aud2 = st.columns(2)
with col_aud1:
    st.subheader("Audit Event Schema")
    st.markdown("""
    Each audit log entry must record:
    * `timestamp_utc`: Nanosecond-precision ISO timestamp.
    * `thread_id`: Unique conversation identifier.
    * `checkpoint_hash`: SHA-256 hash of the exact state dictionary at the moment of review.
    * `reviewer_identity`: Verified SSO user email / public key signature.
    * `decision`: `APPROVE`, `REJECT`, or `MODIFY`.
    * `diff_payload`: Exact JSON diff between agent proposal and human execution.
    """)
with col_aud2:
    st.subheader("Append-Only Audit Logger")
    st.code("""
import hashlib, json, time

def log_audit_event(thread_id, state_snapshot, reviewer, decision):
    state_bytes = json.dumps(state_snapshot, sort_keys=True).encode()
    state_hash = hashlib.sha256(state_bytes).hexdigest()
    
    event = {
        "timestamp": time.time(),
        "thread_id": thread_id,
        "state_hash": state_hash,
        "reviewer": reviewer,
        "decision": decision
    }
    # Append to WORM (Write Once Read Many) compliant storage
    audit_db.append(event)
    """, language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Escalation Matrix with Immutable Audit Trail & Feedback Loop", divider="rainbow")
st.markdown("""
The following complete production pattern implements an enterprise-grade `StateGraph` featuring dynamic `interrupt()`, 
multi-tier escalation routing, and immutable audit event generation.
""")

st.code('''
import time, hashlib, json
from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# 1. Define Advanced HITL State Schema
class AdvancedHITLState(TypedDict):
    task_id: str
    action_type: str
    payload: dict
    current_tier: str       # "L1", "L2", "L3"
    sla_start_epoch: float
    audit_events: list[dict]
    resolution: Optional[str]

# 2. Define High-Security Node with Dynamic Interrupt
def gated_action_node(state: AdvancedHITLState) -> dict:
    # Build payload presented to reviewer
    review_context = {
        "task_id": state["task_id"],
        "action": state["action_type"],
        "payload": state["payload"],
        "assigned_tier": state["current_tier"]
    }
    
    # Dynamic Interrupt yields context and waits for human Command(resume=...)
    human_verdict = interrupt(review_context)
    
    # Hash state for immutable audit logging
    state_digest = hashlib.sha256(json.dumps(state["payload"], sort_keys=True).encode()).hexdigest()
    audit_record = {
        "timestamp": time.time(),
        "tier": state["current_tier"],
        "reviewer": human_verdict.get("reviewer", "anonymous"),
        "decision": human_verdict.get("decision", "DENY"),
        "state_hash": state_digest,
        "feedback": human_verdict.get("feedback", "")
    }
    
    if human_verdict.get("decision") == "APPROVE":
        return {
            "resolution": "EXECUTED",
            "audit_events": state["audit_events"] + [audit_record]
        }
    else:
        return {
            "resolution": "VETOED",
            "audit_events": state["audit_events"] + [audit_record]
        }

# 3. Graph Assembly
builder = StateGraph(AdvancedHITLState)
builder.add_node("gated_action", gated_action_node)
builder.add_edge(START, "gated_action")
builder.add_edge("gated_action", END)

# MemorySaver checkpointer for thread persistence
audit_app = builder.compile(checkpointer=MemorySaver())
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Multi-Level Escalation & Audit Trail Sandbox", divider="rainbow")
st.markdown("""
Experiment with multi-tier incident escalation, reviewer feedback injection, and real-time cryptographic audit trail generation.
""")

# Instantiate Module Scope Simulator
if "adv_hitl_sim" not in st.session_state:
    st.session_state.adv_hitl_sim = MockAdvancedHITLEngine()

sim_adv = st.session_state.adv_hitl_sim

col_s1, col_s2 = st.columns([1, 1])

with col_s1:
    st.subheader("1. Active Incident Queue")
    inc_select = st.selectbox(
        "Select Pending Incident:",
        options=list(sim_adv.incidents.keys()),
        format_func=lambda x: f"{x} - {sim_adv.incidents[x]['title']} ({sim_adv.incidents[x]['tier']})"
    )
    inc_data = sim_adv.incidents[inc_select]

    st.markdown(f"**Incident ID**: `{inc_data['id']}`")
    st.markdown(f"**Current Escalation Tier**: `{inc_data['tier']}`")
    st.markdown(f"**Status**: `{inc_data['status']}`")
    st.markdown(f"**SLA Target**: `{inc_data['sla_seconds']} seconds` | **Elapsed**: `{inc_data['elapsed_seconds']} seconds`")

    st.markdown("#### Incident Payload:")
    st.json(inc_data["payload"])

    st.markdown("#### Feedback History:")
    for fb in inc_data["feedback_history"]:
        st.info(fb)

with col_s2:
    st.subheader("2. Reviewer Action & Escalation Controls")
    
    tab_action = st.radio(
        "Choose Operator Mode:",
        options=["Submit Resolution (Approve / Reject)", "Escalate Incident to Higher Tier"],
        index=0
    )

    if tab_action == "Submit Resolution (Approve / Reject)":
        reviewer_name = st.text_input("Reviewer Identity (Email / SSO):", value="sec_lead_sarah@enterprise.com")
        feedback_notes = st.text_area("Reviewer Feedback & Directives:", value="Verified credentials against vendor records.")
        decision_choice = st.selectbox("Decision:", options=["APPROVE", "REJECT"])

        if st.button("🚀 Commit Resolution to Audit Trail", use_container_width=True):
            r = sim_adv.resolve_with_feedback(inc_select, reviewer_name, feedback_notes, decision_choice)
            st.success(r["message"])

    elif tab_action == "Escalate Incident to Higher Tier":
        target_tier = st.selectbox(
            "Escalate To Tier:",
            options=["Level 2 (Engineering Lead)", "Level 3 (Executive VP Sign-off)"]
        )
        esc_reason = st.text_input("Escalation Reason:", value="SLA window threshold exceeded. Requires senior sign-off.")

        if st.button("⚠️ Escalate Incident Now", use_container_width=True):
            r = sim_adv.escalate_incident(inc_select, target_tier, esc_reason)
            st.warning(r["message"])

st.markdown("### 📋 Cryptographic-Style Audit Trail")
audit_df = pd.DataFrame(inc_data["audit_trail"])
st.dataframe(audit_df, use_container_width=True)

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How does LangGraph's dynamic `interrupt()` differ from static `interrupt_before` in production?"):
    st.markdown("""
    `interrupt_before` halts unconditionally at a specific node boundary defined at graph compilation. 
    In contrast, `interrupt()` is called inside node code, allowing conditional pauses based on runtime parameter evaluation 
    (e.g., only halting if transfer amount exceeds 50,000 USD). Furthermore, `interrupt()` supports bidirectional communication: 
    passing data out to the reviewer and returning human input directly to the resumed node via `Command(resume=...)`.
    """)

with st.expander("Q2: Why is PostgreSQL connection pooling critical when deploying HITL checkpointers to Kubernetes?"):
    st.markdown("""
    In Kubernetes, multiple agent pods scale up and down horizontally while processing hundreds of concurrent conversations. 
    Without a connection pooler (such as PgBouncer or `psycopg_pool.ConnectionPool`), each pod creates separate database connections, 
    quickly exhausting PostgreSQL `max_connections` and causing catastrophic 500 errors during traffic spikes.
    """)

with st.expander("Q3: How do you design an automated SLA escalation mechanism for interrupted threads?"):
    st.markdown("""
    By pairing a background scheduler (e.g. Celery beat, AWS EventBridge, Temporal) with the persistent checkpointer. 
    The scheduler queries checkpoints for threads with `status = 'INTERRUPTED'` and elapsed time exceeding SLA thresholds. 
    When an SLA breach is detected, the scheduler updates the thread state (`graph.update_state()`) to assign the task 
    to a higher escalation tier and triggers manager notification alerts.
    """)

with st.expander("Q4: What makes an audit trail legally defensible in autonomous AI operations?"):
    st.markdown("""
    A legally defensible audit trail requires:
    1. **Immutability**: Stored in append-only WORM (Write Once Read Many) storage.
    2. **Cryptographic Integrity**: SHA-256 hashes of the exact state dictionary before and after human intervention.
    3. **Authenticated Identity**: Strong attribution via enterprise SSO or digital signatures rather than self-reported strings.
    4. **Nanosecond Timestamps**: Standardized UTC timestamps showing exact latency intervals.
    """)

with st.expander("Q5: What is the fail-safe default rule in HITL, and why is it mandatory for enterprise security?"):
    st.markdown("""
    The fail-safe rule states: **In the event of an unresolvable timeout, network failure, or ambiguous reviewer input, the system must default to DENY/ABORT**. 
    Defaulting to execute risks catastrophic side effects, while defaulting to abort ensures that the system fails in a secure, known state.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_trp1, col_trp2 = st.columns(2)
with col_trp1:
    st.error("""
    **Production Trap 1 (Fatal Error): Fail-Open Default on Timeout**  
    Configuring an agent to automatically execute a sensitive action if the human reviewer does not respond within the SLA window. 
    This creates an immediate attack vector where malicious actors simply delay operator responses to force autonomous execution. Always fail-closed.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Mutating Checkpoint History Directly in SQL**  
    Manually editing rows in the checkpointer's SQL database table to 'fix' a stuck state bypasses channel reducers and version hashes, 
    corrupting the thread's checkpoint lineage and breaking future time-travel operations.
    """)
with col_trp2:
    st.error("""
    **Production Trap 3 (Fatal Error): Plaintext Secret Storage in Checkpoint Snapshots**  
    Allowing API tokens, passwords, or customer PII to be serialized into the checkpoint payload. 
    Because checkpoints persist indefinitely for audit and time-travel debugging, unredacted secrets in state dictionaries create persistent compliance breaches.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Phantom Resumptions from Race Conditions**  
    Allowing two reviewers to submit conflicting decisions simultaneously on the same paused thread. 
    Always use optimistic locking (`checkpoint_id` matching) when invoking `Command(resume=...)` to ensure only the first reviewer commits.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_card1, col_card2 = st.columns(2)
with col_card1:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 1: Dynamic Interrupts</h4>
        <p><strong>Concept:</strong> <code>interrupt(payload)</code></p>
        <p><strong>Definition:</strong> In-node function call that suspends graph execution on-the-fly and surfaces structured context to human reviewers.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 2: Resumption with Command</h4>
        <p><strong>Concept:</strong> <code>Command(resume=value)</code></p>
        <p><strong>Definition:</strong> The LangGraph primitive used to feed human operator decisions and payload edits directly into a waiting <code>interrupt()</code> call.</p>
    </div>
    """, unsafe_allow_html=True)
with col_card2:
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 3: Fail-Closed Security</h4>
        <p><strong>Concept:</strong> Default-Deny Policy</p>
        <p><strong>Definition:</strong> The foundational security principle that if human approval times out or encounters errors, the sensitive action is automatically aborted.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Flashcard 4: Immutable Audit Trail</h4>
        <p><strong>Concept:</strong> Cryptographic State Logging</p>
        <p><strong>Definition:</strong> An append-only log recording SHA-256 state hashes, reviewer identities, timestamps, and exact diffs for regulatory compliance.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of Advanced Human-in-the-Loop architectures and enterprise resilience patterns:")

chk_a1 = st.checkbox("1. I understand how to implement dynamic conditional interrupts using LangGraph's native `interrupt()` function.")
chk_a2 = st.checkbox("2. I know how to resume an interrupted graph using `Command(resume=...)` to pass human reviewer data into waiting nodes.")
chk_a3 = st.checkbox("3. I understand why durable checkpointers (`PostgresSaver`) with connection pooling are required in multi-pod environments.")
chk_a4 = st.checkbox("4. I know how to inject qualitative human feedback into graph state to trigger model reflection and self-correction.")
chk_a5 = st.checkbox("5. I can design a multi-tier escalation hierarchy with SLA watchdog timers and fail-closed security guarantees.")
chk_a6 = st.checkbox("6. I understand how to build an immutable, legally defensible audit trail using cryptographic state hashing.")

if chk_a1 and chk_a2 and chk_a3 and chk_a4 and chk_a5 and chk_a6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Advanced Human-in-the-Loop in LangChain & LangGraph.")
