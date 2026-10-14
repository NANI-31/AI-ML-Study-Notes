import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="36.3 Production Mistakes - LangSmith Master Curriculum",
    page_icon="🚨",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ProductionMistakeAuditorEngine:
    """
    Production Readiness & LLMOps Vulnerability Auditor for LangSmith.
    Evaluates enterprise infrastructure across the 5 deadly production anti-patterns:
    1. No Monitoring (Flying blind with zero telemetry or alerting)
    2. No Regression Testing (Shipping prompt/code changes without golden baselines)
    3. No Cost Tracking (Uncapped token burn, no per-tenant cost attribution)
    4. No Failure Analysis (Swallowed exceptions, ignored error traces, no root-cause triage)
    5. No Quality Monitoring (Zero online grounding checks, ignoring customer feedback)
    """
    def __init__(self):
        self.audit_history: List[Dict[str, Any]] = []

    def audit_production_posture(
        self,
        has_monitoring: bool,
        has_regression_testing: bool,
        has_cost_tracking: bool,
        has_failure_analysis: bool,
        has_quality_monitoring: bool,
        daily_query_volume: int = 50000
    ) -> Dict[str, Any]:
        """
        Computes the operational risk score, financial vulnerability, and readiness level
        based on active LLMOps governance practices.
        """
        vulnerabilities = []
        mitigations = []

        # 1. Monitoring Pillar
        if not has_monitoring:
            vulnerabilities.append({
                "pillar": "Real-Time Monitoring",
                "risk_level": "CRITICAL",
                "impact": "Silent outages and severe customer attrition without operational visibility.",
                "remedy": "Enable LangSmith distributed tracing; establish p95 latency and error rate alerts."
            })

        # 2. Regression Testing Pillar
        if not has_regression_testing:
            vulnerabilities.append({
                "pillar": "Regression Testing",
                "risk_level": "CRITICAL",
                "impact": "Prompt tweaks silently break critical customer flows and capabilities.",
                "remedy": "Implement automated pre-merge evaluation against an immutable 500-example golden benchmark."
            })

        # 3. Cost Tracking Pillar
        if not has_cost_tracking:
            unmonitored_exposure = round((daily_query_volume * 800 * 30 / 1_000_000) * 10.0, 2)
            vulnerabilities.append({
                "pillar": "Cost Tracking",
                "risk_level": "HIGH",
                "impact": f"Uncapped token spend risk; estimated unmonitored monthly exposure: {unmonitored_exposure:,.2f} USD.",
                "remedy": "Track input/output tokens per run; group by tenant_id; configure hard budget cap tripwires."
            })

        # 4. Failure Analysis Pillar
        if not has_failure_analysis:
            vulnerabilities.append({
                "pillar": "Failure Analysis",
                "risk_level": "HIGH",
                "impact": "Swallowed exceptions and silent tool failures degrade agent capabilities unnoticed.",
                "remedy": "Query LangSmith for status='error'; triage stack traces in weekly error review boards."
            })

        # 5. Quality Monitoring Pillar
        if not has_quality_monitoring:
            vulnerabilities.append({
                "pillar": "Quality Monitoring",
                "risk_level": "HIGH",
                "impact": "Undetected vendor model drift, hallucination spikes, and decaying user satisfaction.",
                "remedy": "Stream 5% stratified telemetry to online evaluation judges; capture thumbs up/down feedback."
            })

        # Risk Score Calculation (0 = Safe, 100 = Catastrophic)
        risk_penalty_map = {
            "has_monitoring": 30 if not has_monitoring else 0,
            "has_regression_testing": 25 if not has_regression_testing else 0,
            "has_cost_tracking": 15 if not has_cost_tracking else 0,
            "has_failure_analysis": 15 if not has_failure_analysis else 0,
            "has_quality_monitoring": 15 if not has_quality_monitoring else 0
        }
        total_risk_score = sum(risk_penalty_map.values())

        if total_risk_score == 0:
            readiness_tier = "ENTERPRISE_GRADE (Fully Guarded)"
            status = "APPROVED"
        elif total_risk_score <= 30:
            readiness_tier = "MODERATE_RISK (Operational Gaps)"
            status = "WARNING"
        else:
            readiness_tier = "EXTREME_VULNERABILITY (Disaster Prone)"
            status = "CRITICAL_HAZARD"

        report = {
            "total_risk_score": total_risk_score,
            "readiness_tier": readiness_tier,
            "status": status,
            "vulnerabilities": vulnerabilities,
            "checklist": {
                "Real-Time Monitoring": "ENABLED" if has_monitoring else "DISABLED",
                "Regression Testing": "ENABLED" if has_regression_testing else "DISABLED",
                "Cost Tracking": "ENABLED" if has_cost_tracking else "DISABLED",
                "Failure Analysis": "ENABLED" if has_failure_analysis else "DISABLED",
                "Quality Monitoring": "ENABLED" if has_quality_monitoring else "DISABLED"
            }
        }
        self.audit_history.append(report)
        return report

# Initialize engine in session state
if "prod_mistakes_engine" not in st.session_state:
    st.session_state.prod_mistakes_engine = ProductionMistakeAuditorEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("🚨 Chapter 36.3: Common LangSmith Production Mistakes")
st.caption("Unmonitored Traffic, Unchecked Prompt Regressions, Runaway Cloud Spend, Masked Tool Errors, and Silent Quality Drift")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production LLM Operational Failures and Governance Anti-Patterns", divider="rainbow")

st.markdown("""
Deploying generative AI applications to production without comprehensive LLMOps governance is an operational hazard. 
Unlike classical microservices where a stable binary remains predictable until the next deployment, LLM applications 
interact with non-deterministic cloud models, dynamic external knowledge bases, and variable user context lengths. 
Operating without observability, regression testing, cost controls, failure triage, and quality surveillance 
guarantees enterprise-level incidents.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ProductionMistakesArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_mistakes {
            label = "1. The 5 Deadly Production Anti-Patterns";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F43F5E";
            fontsize = 12;

            M1 [label="Anti-Pattern 1: No Monitoring\\n(Flying Blind: Zero distributed tracing)", fillcolor="#7C2D12", color="#EA580C"];
            M2 [label="Anti-Pattern 2: No Regression Testing\\n(Merging prompt changes without golden baselines)", fillcolor="#7C2D12", color="#EA580C"];
            M3 [label="Anti-Pattern 3: No Cost Tracking\\n(Uncapped token spend / surprise cloud invoices)", fillcolor="#7C2D12", color="#EA580C"];
            M4 [label="Anti-Pattern 4: No Failure Analysis\\n(Swallowing tool exceptions / ignoring error spans)", fillcolor="#7C2D12", color="#EA580C"];
            M5 [label="Anti-Pattern 5: No Quality Monitoring\\n(Silent drift from vendor model weight updates)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_consequences {
            label = "2. Production Blast Radius";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            C_BLIND [label="Silent Outages & Churn\\n(Users abandon broken app; engineers unaware)", fillcolor="#881337", color="#F43F5E"];
            C_REG [label="Accidental Capability Drops\\n(Minor prompt tweak breaks 25% of core user flows)", fillcolor="#881337", color="#F43F5E"];
            C_BILL [label="Runaway Cloud Spend\\n(Recursive loops burn thousands of USD in hours)", fillcolor="#881337", color="#F43F5E"];
            C_TOOL [label="Degraded Agent Loops\\n(Tools fail silently; agent outputs hallucinations)", fillcolor="#881337", color="#F43F5E"];
            C_DRIFT [label="Quality Decay\\n(Hallucinations spike without triggering 500 errors)", fillcolor="#881337", color="#F43F5E"];
        }

        subgraph cluster_governance {
            label = "3. Enterprise LLMOps Governance Pillars";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            G_TRACE [label="End-to-End Tracing Telemetry\\n(LangSmith production projects with p95 SLAs)", fillcolor="#065F46", color="#059669"];
            G_GATE [label="Automated Regression Gates\\n(500-question golden dataset CI block on delta < 0)", fillcolor="#065F46", color="#059669"];
            G_COST [label="Token Accounting & Budget Tripwires\\n(Per-tenant unit economics + automatic rate caps)", fillcolor="#065F46", color="#059669"];
            G_TRIAGE [label="Weekly Error Triage Sprints\\n(Query status='error' traces + root-cause fixes)", fillcolor="#065F46", color="#059669"];
            G_SURV [label="Online Quality Surveillance\\n(5% telemetry sampling + user thumbs up/down)", fillcolor="#065F46", color="#059669"];
        }

        M1 -> C_BLIND [label="Leads to"];
        M2 -> C_REG [label="Leads to"];
        M3 -> C_BILL [label="Leads to"];
        M4 -> C_TOOL [label="Leads to"];
        M5 -> C_DRIFT [label="Leads to"];

        C_BLIND -> G_TRACE [label="Cured By"];
        C_REG -> G_GATE [label="Cured By"];
        C_BILL -> G_COST [label="Cured By"];
        C_TOOL -> G_TRIAGE [label="Cured By"];
        C_DRIFT -> G_SURV [label="Cured By"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Production Governance
1. **Never Deploy Prompts Directly to Production Without Automated Regression Testing**: A change as simple as adding *"Please be concise"* can cause a system prompt to omit crucial reasoning steps, breaking downstream JSON formatting and cascading into production outages.
2. **Never Treat Token Usage as an Uncapped Utility**: Uncapped API keys combined with autonomous agents or retry loops will inevitably encounter infinite loops or runaway recursion. Hard billing quotas and per-session token limits are non-negotiable.
3. **HTTP 200 OK Does Not Mean the Application Succeeded**: In traditional software, `200 OK` indicates health. In generative AI, `200 OK` can conceal complete factual fabrications, toxic outputs, or ungrounded responses. Monitoring quality is as important as monitoring HTTP status codes.
""")

st.divider()

# ==============================================================================
# SECTION 2: NO MONITORING
# ==============================================================================
st.header("2. No Monitoring (Flying Blind, Silent Customer Attrition and Undetected Outages)", divider="rainbow")

st.markdown("""
Running an LLM application without distributed tracing is like flying an aircraft with the cockpit instruments turned off. 
When customer complaints arrive, engineering has no visibility into what the model generated, which tools were called, 
or where latency was lost.
""")

st.markdown("""
### The Symptoms of "Flying Blind"
* **Zero Telemetry on Latency Waterfalls**: An API endpoint suddenly takes 9 seconds to respond. Without tracing, developers cannot tell whether the delay originated in the database retriever, the embedding API, or a slow LLM generation.
* **Invisible Outages**: When a third-party model endpoint suffers degraded performance or regional throttling, users experience failures while infrastructure monitors (CPU, memory) report healthy 10% utilization.
* **Silent Customer Attrition**: Frustrated users leave the platform without filing bug reports, resulting in declining engagement metrics with zero diagnostic traces to explain why.
""")

st.code('''
# ANTI-PATTERN: Completely unmonitored production endpoint
from fastapi import FastAPI
from my_rag import rag_chain

app = FastAPI()

@app.post("/chat")
def chat_endpoint(query: str):
    # Flying blind! Zero tracing, zero latency breakdown, zero error visibility!
    return {"answer": rag_chain.invoke(query)}

# PRODUCTION REMEDY: Fully instrumented with LangSmith & metadata
from langsmith import traceable

@app.post("/chat")
@traceable(
    name="customer_chat_endpoint",
    run_type="chain",
    tags=["env:production", "channel:api"]
)
def chat_endpoint_monitored(query: str, tenant_id: str, session_id: str):
    # Full distributed trace captured automatically in LangSmith
    return {"answer": rag_chain.invoke(query)}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production services run LangSmith tracing across 100% of production traffic. 
Alerts notify on-call engineers via Slack and PagerDuty if error rates exceed 1.0% or if p95 latency breaches 2500ms.
""")

st.divider()

# ==============================================================================
# SECTION 3: NO REGRESSION TESTING
# ==============================================================================
st.header("3. No Regression Testing (Breaking Existing Workflows with Prompt Updates and Accidental Capabilities Drop)", divider="rainbow")

st.markdown("""
In traditional software, modifying an established function requires unit tests to verify that existing behavior remains intact. 
In generative AI, developers frequently edit system prompts directly to fix a single bug reported by a user, 
without realizing that the prompt change breaks 20 other previously working customer workflows.
""")

st.markdown("""
### The Butterfly Effect of Prompt Tweaks
* **Fixing Bug A Breaks Workflow B**: Adding an instruction to *"Always format answers in markdown tables"* fixes a reporting query but completely breaks a mobile client expecting plain-text strings.
* **Accidental Loss of Invariants**: Shortening a system prompt to reduce token costs accidentally deletes a negative constraint (e.g., *"Never mention competitors"*), exposing the company to reputational damage.
* **The Solution**: An automated CI/CD Regression Gate. Before any prompt or model update can be merged, it must run against an immutable 500-question Golden Dataset in LangSmith. If candidate accuracy drops compared to the baseline (`Delta < 0.0`), the pull request is blocked.
""")

regression_impact_df = pd.DataFrame([
    {"Prompt Change": "Add: 'Always summarize briefly'", "Intended Effect": "Lower token costs", "Unintended Regression": "Model omits crucial steps in multi-step technical guides", "Detected By": "Automated Golden Benchmark Gate"},
    {"Prompt Change": "Switch to faster model endpoint", "Intended Effect": "Reduce latency by 40%", "Unintended Regression": "JSON tool calling output format compliance drops from 99% to 88%", "Detected By": "Automated Schema Validator Suite"},
    {"Prompt Change": "Edit few-shot examples", "Intended Effect": "Improve tone politeness", "Unintended Regression": "Model begins hallucinating facts on out-of-domain questions", "Detected By": "Negative Edge-Case Refusal Suite"}
])
st.dataframe(regression_impact_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Git pull requests trigger an automated GitHub Action that runs `langsmith.evaluate()`. 
The action blocks merging unless candidate scores match or exceed the production baseline across all evaluated metrics.
""")

st.divider()

# ==============================================================================
# SECTION 4: NO COST TRACKING
# ==============================================================================
st.header("4. No Cost Tracking (Surprise Cloud Invoices, Infinite Loop Token Burns and Uncapped API Quotas)", divider="rainbow")

st.markdown("""
LLM cloud APIs operate on a metered, pay-per-token model. Deploying an application without real-time token tracking 
and budget controls is an invitation to surprise invoices totaling tens of thousands of dollars.
""")

st.markdown("""
### The 3 Financial Disaster Scenarios
1. **The Autonomous Agent Infinite Loop**: An agent gets stuck trying to satisfy a tool condition, executing 50 consecutive tool calls with an expanding context window. A single runaway request burns 500,000 tokens ($5.00 USD).
2. **The Prompt Bloat Creep**: Over months of iteration, developers keep adding context and instructions to the system prompt, growing it from 500 tokens to 8,000 tokens per request. At 100,000 daily requests, annual spend increases by over $70,000 USD without anyone noticing.
3. **No Multi-Tenant Attribution**: An enterprise customer on a flat-rate tier uses 100x more tokens than their subscription covers, turning what appeared to be a profitable account into a massive financial loss.
""")

st.code('''
# LangSmith SDK Token Accounting & Circuit Breaker
def enforce_token_budget_guard(run_tree):
    """
    Inspects token consumption on live LangSmith traces and trips circuit breakers.
    """
    total_tokens = run_tree.total_tokens
    if total_tokens > 25_000:
        # Trip emergency circuit breaker
        raise RuntimeError(f"Request token burn ({total_tokens}) exceeded safety ceiling (25,000 tokens). Terminated.")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Finance and engineering teams configure hard budget alerts in LangSmith. 
If daily spend reaches 80% of the allocated threshold, warnings are dispatched; at 100%, non-essential traffic is automatically 
routed to smaller, cost-effective models (e.g., GPT-4o-mini).
""")

st.divider()

# ==============================================================================
# SECTION 5: NO FAILURE ANALYSIS
# ==============================================================================
st.header("5. No Failure Analysis (Swallowing Exceptions, Masking Tool Crashes and Ignoring Error Spans)", divider="rainbow")

st.markdown("""
In many production codebases, developers wrap tool calls in generic `try-except` blocks:
```python
try:
    result = execute_sql_query(query)
except Exception:
    result = "Error occurred"
```
While this prevents the HTTP server from crashing, it blinds the LLM to the root cause and blinds engineering 
to systemic database failures.
""")

st.markdown("""
### The Consequences of Masked Failures
* **Hallucination Triggered by Swallowed Errors**: When a tool fails and returns a generic string, the LLM often fabricates an answer to satisfy the user, turning a simple database timeout into a dangerous factual hallucination.
* **Unobserved Third-Party API Degradation**: If an external weather API or CRM endpoint fails 15% of the time, developers never notice because the top-level application returns `200 OK`.
* **Zero Post-Mortem Traceability**: Without capturing the exact exception type, stack trace, and input payload in LangSmith, engineers cannot reproduce or fix systemic failures.
""")

failure_comparison_df = pd.DataFrame([
    {"Practice": "Swallowing Exceptions (Anti-Pattern)", "Tool Crash Behavior": "Returns 'Error' string; LLM hallucinates", "LangSmith Status": "Marked 'success' (Green)", "Engineering Visibility": "Zero (Completely invisible)"},
    {"Practice": "Structured Error Spans (Best Practice)", "Tool Crash Behavior": "Returns explicit error context for retry", "LangSmith Status": "Marked 'error' (Red) with full stack trace", "Engineering Visibility": "Immediate (Filtered via status='error')"}
])
st.dataframe(failure_comparison_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Teams conduct weekly "Error Triage Sprints." In LangSmith, they filter for `status: error`, group by exception type, 
and resolve the top 3 most frequent failure modes in each sprint.
""")

st.divider()

# ==============================================================================
# SECTION 6: NO QUALITY MONITORING
# ==============================================================================
st.header("6. No Quality Monitoring (Undetected Model Weight Drift, Stale Ground Truths and Parametric Hallucination Surges)", divider="rainbow")

st.markdown("""
Software that is completely functional at 9:00 AM can begin generating lower-quality outputs by 2:00 PM without a single code commit. 
This occurs because foundation model providers periodically update model weights, vector database knowledge bases go stale, 
and user inquiry patterns drift. Without continuous quality monitoring, degradation goes completely undetected.
""")

st.markdown("""
### The 3 Sources of Silent Quality Drift
1. **Silent Model Provider Weight Updates**: Cloud providers update models behind static endpoints (e.g., `gpt-4o`). Subtle changes in output formatting or refusal behaviors can degrade production pipelines overnight.
2. **Knowledge Base Staleness**: RAG retrieval quality decays when documentation is updated in the real world but vector embeddings remain stale.
3. **User Intent Shift**: Marketing campaigns bring new user demographics with distinct query styles that the system prompt was never calibrated to handle.
""")

st.code('''
from langsmith import Client

client = Client()

# Continuous online quality surveillance loop
def run_online_quality_surveillance():
    """
    Periodically samples 5% of production traces and runs automated faithfulness evaluation.
    """
    recent_runs = client.list_runs(
        project_name="prod-customer-service",
        execution_order=1,
        limit=100
    )
    # Evaluate context faithfulness asynchronously
    for run in recent_runs:
        evaluate_faithfulness_async(run.id, run.inputs, run.outputs)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production architectures stream a 5% stratified sample of production traces through online evaluation daemons in LangSmith. 
If weekly average faithfulness drops below 0.88, an automated quality alert is dispatched to the engineering team.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: LLMOps Production Readiness and Vulnerability Auditor (ProductionMistakeAuditorEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the Production Readiness Auditor. 
It assesses infrastructure postures across the 5 core pillars, calculates risk scores, identifies vulnerabilities, 
and outputs actionable remediation plans.
""")

st.code('''
class ProductionMistakeAuditorEngine:
    """
    Production Readiness & LLMOps Vulnerability Auditor for LangSmith.
    Evaluates enterprise infrastructure across the 5 deadly production anti-patterns:
    1. No Monitoring (Flying blind with zero telemetry or alerting)
    2. No Regression Testing (Shipping prompt/code changes without golden baselines)
    3. No Cost Tracking (Uncapped token burn, no per-tenant cost attribution)
    4. No Failure Analysis (Swallowed exceptions, ignored error traces, no root-cause triage)
    5. No Quality Monitoring (Zero online grounding checks, ignoring customer feedback)
    """
    def __init__(self):
        self.audit_history: List[Dict[str, Any]] = []

    def audit_production_posture(
        self,
        has_monitoring: bool,
        has_regression_testing: bool,
        has_cost_tracking: bool,
        has_failure_analysis: bool,
        has_quality_monitoring: bool,
        daily_query_volume: int = 50000
    ) -> Dict[str, Any]:
        """
        Computes the operational risk score, financial vulnerability, and readiness level
        based on active LLMOps governance practices.
        """
        vulnerabilities = []

        if not has_monitoring:
            vulnerabilities.append({
                "pillar": "Real-Time Monitoring",
                "risk_level": "CRITICAL",
                "impact": "Silent outages and severe customer attrition without operational visibility.",
                "remedy": "Enable LangSmith distributed tracing; establish p95 latency and error rate alerts."
            })

        if not has_regression_testing:
            vulnerabilities.append({
                "pillar": "Regression Testing",
                "risk_level": "CRITICAL",
                "impact": "Prompt tweaks silently break critical customer flows and capabilities.",
                "remedy": "Implement automated pre-merge evaluation against an immutable 500-example golden benchmark."
            })

        if not has_cost_tracking:
            unmonitored_exposure = round((daily_query_volume * 800 * 30 / 1_000_000) * 10.0, 2)
            vulnerabilities.append({
                "pillar": "Cost Tracking",
                "risk_level": "HIGH",
                "impact": f"Uncapped token spend risk; estimated unmonitored monthly exposure: {unmonitored_exposure:,.2f} USD.",
                "remedy": "Track input/output tokens per run; group by tenant_id; configure hard budget cap tripwires."
            })

        if not has_failure_analysis:
            vulnerabilities.append({
                "pillar": "Failure Analysis",
                "risk_level": "HIGH",
                "impact": "Swallowed exceptions and silent tool failures degrade agent capabilities unnoticed.",
                "remedy": "Query LangSmith for status='error'; triage stack traces in weekly error review boards."
            })

        if not has_quality_monitoring:
            vulnerabilities.append({
                "pillar": "Quality Monitoring",
                "risk_level": "HIGH",
                "impact": "Undetected vendor model drift, hallucination spikes, and decaying user satisfaction.",
                "remedy": "Stream 5% stratified telemetry to online evaluation judges; capture thumbs up/down feedback."
            })

        risk_penalty_map = {
            "has_monitoring": 30 if not has_monitoring else 0,
            "has_regression_testing": 25 if not has_regression_testing else 0,
            "has_cost_tracking": 15 if not has_cost_tracking else 0,
            "has_failure_analysis": 15 if not has_failure_analysis else 0,
            "has_quality_monitoring": 15 if not has_quality_monitoring else 0
        }
        total_risk_score = sum(risk_penalty_map.values())

        if total_risk_score == 0:
            readiness_tier = "ENTERPRISE_GRADE (Fully Guarded)"
            status = "APPROVED"
        elif total_risk_score <= 30:
            readiness_tier = "MODERATE_RISK (Operational Gaps)"
            status = "WARNING"
        else:
            readiness_tier = "EXTREME_VULNERABILITY (Disaster Prone)"
            status = "CRITICAL_HAZARD"

        return {
            "total_risk_score": total_risk_score,
            "readiness_tier": readiness_tier,
            "status": status,
            "vulnerabilities": vulnerabilities,
            "checklist": {
                "Real-Time Monitoring": "ENABLED" if has_monitoring else "DISABLED",
                "Regression Testing": "ENABLED" if has_regression_testing else "DISABLED",
                "Cost Tracking": "ENABLED" if has_cost_tracking else "DISABLED",
                "Failure Analysis": "ENABLED" if has_failure_analysis else "DISABLED",
                "Quality Monitoring": "ENABLED" if has_quality_monitoring else "DISABLED"
            }
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Production Readiness Health Checker and Risk Matrix Sandbox", divider="rainbow")

st.markdown("""
Audit your LLMOps infrastructure configuration across the 5 production pillars. 
Toggle active governance capabilities to observe how missing practices escalate operational risk scores.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### LLMOps Governance Pillars")
    t_mon = st.toggle("1. Distributed Real-Time Monitoring", value=True, help="Captures full traces with p95 latency alerts.")
    t_reg = st.toggle("2. Automated CI Regression Testing", value=True, help="Enforces pre-merge golden dataset evaluation.")
    t_cost = st.toggle("3. Token Economics & Cost Tracking", value=True, help="Monitors input/output tokens and budget caps.")
    t_fail = st.toggle("4. Failure Analysis & Exception Triage", value=True, help="Surfaces error spans and tool exceptions.")
    t_qual = st.toggle("5. Continuous Online Quality Monitoring", value=True, help="Surveils context faithfulness and user feedback.")
    vol_slider = st.slider("Daily Production Query Volume", min_value=1000, max_value=200000, value=50000, step=5000)

with wb_col2:
    st.markdown("### Production Readiness Audit & Vulnerability Assessment")
    audit_report = st.session_state.prod_mistakes_engine.audit_production_posture(
        has_monitoring=t_mon,
        has_regression_testing=t_reg,
        has_cost_tracking=t_cost,
        has_failure_analysis=t_fail,
        has_quality_monitoring=t_qual,
        daily_query_volume=vol_slider
    )

    if audit_report["status"] == "APPROVED":
        st.success(f"**POSTURE: {audit_report['readiness_tier']}** (Risk Score: {audit_report['total_risk_score']}/100)")
    elif audit_report["status"] == "WARNING":
        st.warning(f"**POSTURE: {audit_report['readiness_tier']}** (Risk Score: {audit_report['total_risk_score']}/100)")
    else:
        st.error(f"**POSTURE: {audit_report['readiness_tier']}** (Risk Score: {audit_report['total_risk_score']}/100)")

    # Score Gauge
    m_a1, m_a2 = st.columns(2)
    with m_a1:
        st.metric("Total Operational Risk Score", f"{audit_report['total_risk_score']} / 100", delta_color="inverse")
    with m_a2:
        st.metric("Active Vulnerabilities Count", f"{len(audit_report['vulnerabilities'])}")

    # Vulnerability Table
    if audit_report["vulnerabilities"]:
        st.markdown("#### Active Production Vulnerabilities")
        v_df = pd.DataFrame(audit_report["vulnerabilities"])
        st.dataframe(v_df, use_container_width=True)
    else:
        st.markdown("All 5 enterprise LLMOps governance pillars active. System is production-ready.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

q_exp6 = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with q_exp6:
    st.markdown("""
    **Q1: Why is treating HTTP 200 OK as a sign of health dangerous for generative AI applications?**  
    *Answer:* In traditional software architectures, a `200 OK` HTTP status indicates that the request succeeded. In generative AI, a model endpoint can return `200 OK` while outputting complete factual hallucinations, toxic responses, or empty text. Monitoring generative systems requires dual-layer observability: traditional infrastructure metrics (HTTP error codes, CPU, memory) combined with semantic output quality monitoring (context faithfulness, user feedback, refusal rates).
    """)

    st.markdown("""
    **Q2: How does automated regression testing prevent the 'butterfly effect' in prompt engineering?**  
    *Answer:* Prompts are complex natural language instruction spaces. A minor change intended to fix a specific bug (e.g., adding *"Please respond briefly"*) can inadvertently cause the model to drop necessary steps, ignore formatting rules, or hallucinate on out-of-domain queries. Automated regression testing executes the modified prompt against an immutable golden benchmark dataset in LangSmith before code merge, ensuring that fixing one edge case does not break existing capabilities.
    """)

    st.markdown("""
    **Q3: What mechanisms prevent runaway cloud costs from autonomous multi-agent systems?**  
    *Answer:* 1) Setting explicit recursion limits (`recursion_limit=15`) in LangGraph StateGraph engines; 2) Enforcing per-request token caps (e.g., terminating runs that exceed 25,000 total tokens); 3) Implementing circuit breakers that trip if hourly organization spend exceeds allocated thresholds; and 4) Tagging all spans with `tenant_id` to attribute costs and throttle abusive users.
    """)

    st.markdown("""
    **Q4: Why must exceptions in tool-calling workflows never be silently swallowed?**  
    *Answer:* When tool execution crashes (e.g., SQL syntax error or CRM API timeout) and the code silently catches the error, returning a generic string like *"Error occurred"*, the downstream LLM often attempts to answer the user by fabricating facts out of training data. Furthermore, swallowing exceptions prevents LangSmith from flagging the run as `status: error`, blinding engineers to systemic tool degradations.
    """)

    st.markdown("""
    **Q5: How do you detect silent model degradation caused by third-party model provider weight updates?**  
    *Answer:* Cloud providers periodically update weights behind static endpoints (e.g., `gpt-4o`). Continuous evaluation daemons run scheduled nightly benchmarks against a frozen golden dataset with pinned parameters (`temperature=0`). If benchmark scores drop by more than 2 standard deviations while repository code has had zero commits, an automated alert flags vendor model drift, triggering failover to pinned model snapshots.
    """)

st.markdown("### Production Traps and Anti-Patterns")

tc_b1, tc_b2 = st.columns(2)

with tc_b1:
    st.error("""
    **Trap 1: Merging Prompts Directly to Production**
    * **Failure**: Modifying system prompts in production without running benchmark evaluations.
    * **Consequence**: The prompt change breaks 20% of core customer workflows, requiring an emergency rollback during peak hours.
    * **Remedy**: Make pre-merge evaluation on a 500-question Golden Dataset a mandatory branch protection rule in GitHub/GitLab.
    """)

    st.error("""
    **Trap 2: Uncapped API Keys with Autonomous Agents**
    * **Failure**: Deploying autonomous agents with uncapped credit cards and no recursion limits.
    * **Consequence**: An agent caught in a tool execution loop burns thousands of dollars overnight before someone notices the invoice.
    * **Remedy**: Set hard budget limits in LLM provider consoles and enforce strict `recursion_limit` in agent graphs.
    """)

with tc_b2:
    st.warning("""
    **Trap 3: Silently Swallowing Tool Exceptions**
    * **Failure**: Wrapping tool calls in `except: pass` to keep the HTTP server returning 200 OK.
    * **Consequence**: LLMs fabricate answers when tools fail, and engineering cannot find the errors in LangSmith.
    * **Remedy**: Allow exceptions to be recorded as structured error spans in LangSmith with full stack traces.
    """)

    st.warning("""
    **Trap 4: Zero Feedback Collection**
    * **Failure**: Neglecting to add thumbs up/down feedback widgets to the user interface.
    * **Consequence**: Engineering has no ground-truth visibility into customer satisfaction and cannot identify failing production queries.
    * **Remedy**: Capture user ratings and route negative traces directly into LangSmith annotation queues for weekly triage.
    """)

st.markdown("### Core Technical Flashcards")

fc_d1, fc_d2 = st.columns(2)

with fc_d1:
    with st.container(border=True):
        st.markdown("**Front**: Why is HTTP 200 OK an insufficient health metric for LLMs?")
        st.markdown("**Back**: LLMs can return HTTP 200 OK while generating complete factual hallucinations, toxic responses, or empty text.")

    with fc_d1:
        with st.container(border=True):
            st.markdown("**Front**: What is the primary purpose of a pre-merge Regression Gate?")
            st.markdown("**Back**: To automatically evaluate proposed prompt changes against an immutable golden dataset, blocking merges that degrade performance.")

with fc_d2:
    with st.container(border=True):
        st.markdown("**Front**: How do swallowed exceptions trigger model hallucinations?")
        st.markdown("**Back**: When a tool fails silently and returns a generic error string, the model attempts to guess the answer, inventing plausible-sounding false facts.")

    with fc_d2:
        with st.container(border=True):
            st.markdown("**Front**: How do you detect third-party model provider weight updates?")
            st.markdown("**Back**: By running scheduled nightly evaluations on a frozen golden dataset; sudden score drops on zero code commits indicate vendor model drift.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of avoiding Common LangSmith Production Mistakes:")

pm_chk1 = st.checkbox("I understand why distributed tracing is mandatory to prevent flying blind in production.", value=True)
pm_chk2 = st.checkbox("I know how to enforce pre-merge regression testing against an immutable golden benchmark dataset.", value=True)
pm_chk3 = st.checkbox("I can implement token accounting, per-tenant cost attribution, and budget circuit breakers.", value=True)
pm_chk4 = st.checkbox("I understand the danger of swallowed tool exceptions and know how to capture structured error spans.", value=True)
pm_chk5 = st.checkbox("I know how to monitor online quality drift caused by silent vendor model weight updates.", value=True)
pm_chk6 = st.checkbox("I can conduct a comprehensive production readiness audit across the 5 LLMOps governance pillars.", value=True)

if pm_chk1 and pm_chk2 and pm_chk3 and pm_chk4 and pm_chk5 and pm_chk6:
    st.success("Mastery Certified: You have mastered avoiding common production mistakes, establishing regression gates, and hardening LLMOps architectures in LangSmith.")
