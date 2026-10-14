import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.3 Production Questions - LangSmith Curriculum",
    page_icon="📡",
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
    .badge-prod {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-alert {
        background-color: #450A0A;
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class ProductionTelemetryEngine:
    """
    Simulates production LLM runtime telemetry:
    Latency percentiles, token usage distributions, cost models,
    trace diagnostics, online RAG feedback, and agent failure modes.
    """
    def __init__(self):
        self.pricing_catalog = {
            "gpt-4o": {"prompt_per_1m": 2.50, "completion_per_1m": 10.00},
            "gpt-4o-mini": {"prompt_per_1m": 0.15, "completion_per_1m": 0.60},
            "claude-3-5-sonnet": {"prompt_per_1m": 3.00, "completion_per_1m": 15.00}
        }
        
        self.sample_traces = [
            {
                "trace_id": "tr-9812-a",
                "session_id": "sess-user-101",
                "app": "support-agent",
                "status": "SUCCESS",
                "ttft_ms": 380,
                "total_latency_ms": 1420,
                "prompt_tokens": 850,
                "completion_tokens": 120,
                "cost_usd": 0.0033,
                "feedback": "thumbs_up",
                "tool_calls": 2,
                "error": None
            },
            {
                "trace_id": "tr-9813-b",
                "session_id": "sess-user-102",
                "app": "rag-summarizer",
                "status": "ERROR",
                "ttft_ms": 420,
                "total_latency_ms": 4890,
                "prompt_tokens": 3200,
                "completion_tokens": 15,
                "cost_usd": 0.0081,
                "feedback": "thumbs_down",
                "tool_calls": 1,
                "error": "VectorDBTimeout: Qdrant query timed out after 4000ms"
            },
            {
                "trace_id": "tr-9814-c",
                "session_id": "sess-user-103",
                "app": "sql-agent",
                "status": "ERROR",
                "ttft_ms": 290,
                "total_latency_ms": 12400,
                "prompt_tokens": 7400,
                "completion_tokens": 680,
                "cost_usd": 0.0253,
                "feedback": None,
                "tool_calls": 8,
                "error": "GraphRecursionError: Recursion limit of 8 reached without termination"
            },
            {
                "trace_id": "tr-9815-d",
                "session_id": "sess-user-104",
                "app": "support-agent",
                "status": "SUCCESS",
                "ttft_ms": 310,
                "total_latency_ms": 980,
                "prompt_tokens": 420,
                "completion_tokens": 85,
                "cost_usd": 0.0019,
                "feedback": "thumbs_up",
                "tool_calls": 1,
                "error": None
            }
        ]

    def compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rate = self.pricing_catalog.get(model, self.pricing_catalog["gpt-4o"])
        cost = (prompt_tokens / 1_000_000) * rate["prompt_per_1m"] + (completion_tokens / 1_000_000) * rate["completion_per_1m"]
        return round(cost, 6)

    def filter_traces(self, status: str, min_latency: int) -> List[Dict[str, Any]]:
        results = []
        for t in self.sample_traces:
            if status != "ALL" and t["status"] != status:
                continue
            if t["total_latency_ms"] < min_latency:
                continue
            results.append(t)
        return results

# Instantiate engine
prod_engine = ProductionTelemetryEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production Telemetry, Real-Time Observability, & SLI/SLO Alerting", divider="rainbow")

st.markdown("""
Deploying generative AI to production requires a fundamental shift from offline evaluation to **real-time operational observability**.
In production, LLMs are non-deterministic, third-party network APIs that can experience unpredictable latency spikes, rate limits, 
silent context drift, and runaway agent recursion loops. LangSmith serves as the control plane for production telemetry, 
streaming run events asynchronously to provide actionable Service Level Indicators (SLIs) and automated alerting.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Production_Telemetry_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_app {
            label = "Production Application Runtime (Web Server / FastApi / Streamlit)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            user_req [label="User Traffic\\n(REST / WebSocket)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            app_code [label="App Execution & Agent Graph\\n(@traceable / LangChain / LangGraph)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            background_worker [label="Non-Blocking Background Telemetry Queue\\n(Batch Exporter Thread)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_langsmith {
            label = "LangSmith Observability Platform Control Plane";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            trace_ingest [label="Trace Ingestion API\\n(Validates Spans, Projects, Metadata)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            metrics_engine [label="Real-Time Metrics Engine\\n(Aggregates TTFT, Latency, Tokens, Costs)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            feedback_service [label="Feedback & Annotation Queue\\n(Captures Thumbs Up/Down, User Corrections)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_alerts {
            label = "Monitoring, Dashboards & Automated Alerting";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            sli_dashboard [label="Live Operational Dashboards\\n(p50/p95/p99 Latency, Error Rate %)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            alert_rules [label="Automated PagerDuty / Slack Webhooks\\n(Runaway Cost Alerts, High Error Spikes)", fillcolor="#7F1D1D", fontcolor="#F87171", color="#EF4444"];
            golden_backfill [label="Active Learning Curation\\n(Exports Bad Production Runs to Datasets)", fillcolor="#0F172A", fontcolor="#A78BFA"];
        }

        user_req -> app_code;
        app_code -> background_worker [label="queues trace events"];
        background_worker -> trace_ingest [label="async batch post (gzip)"];
        trace_ingest -> metrics_engine;
        app_code -> feedback_service [label="client.create_feedback()"];
        metrics_engine -> sli_dashboard;
        metrics_engine -> alert_rules [label="exceeds SLO threshold"];
        feedback_service -> golden_backfill [label="flags negative traces"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Zero-Impact Telemetry Guarantee:</strong> In production, LangSmith trace collection executes in a dedicated 
    background daemon thread pool. Network calls to LangSmith are non-blocking and batched, ensuring that telemetry 
    never adds network overhead or latency to the user-facing request critical path.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: HOW DO YOU MONITOR LLM LATENCY?
# ==============================================================================
st.header("2. How to Monitor LLM Latency: Time to First Token (TTFT), Component Latency, and Percentiles", divider="rainbow")

st.markdown("""
Monitoring LLM latency requires measuring two distinct phases: **Time to First Token (TTFT)** 
and **Tokens Per Second (TPS) / Completion Latency**. Average latency is a misleading metric in distributed 
AI systems because queueing delays and long-context documents create severe right-skewed tail latency.
""")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    ### Critical Latency SLIs
    - **Time to First Token (TTFT)**:
      The duration between sending the prompt and receiving the first streamed token byte. 
      Determines perceived responsiveness in user-facing chats. Typical target: `< 500ms`.
    - **Total Duration**:
      The complete clock time from request dispatch to final span closure. Includes retrieval, prompt formatting, 
      token generation, and guardrail verification.
    - **Percentile Distributions (p50, p95, p99)**:
      - *p50 (Median)*: Expected performance for average user requests.
      - *p95*: Performance experienced by the 95th percentile (identifies slow retrievers or heavy prompts).
      - *p99*: Worst-case edge scenarios (identifies network timeouts, API throttling, or recursion loops).
    """)

with c2:
    st.markdown("""
    ### Root Causes of Latency Degradation
    1. **Context Bloat**: When prompts grow from 1,000 to 32,000 tokens due to chat history or large document retrieval, prefill time scales quadratically or linearly.
    2. **Serial Tool Execution**: Agents that call tools in serial chains instead of parallel tool calls (`tool_choice="auto"` with multiple functions) multiply round-trip delays.
    3. **Provider Rate Limiting (HTTP 429)**: Providers queue requests or back off exponentially during peak traffic hours.
    4. **Vector Database Cold Starts**: Disk-backed vector indices or unindexed filter queries causing multi-second retrieval bottlenecks.
    """)

# ==============================================================================
# SECTION 3: HOW DO YOU TRACK TOKEN USAGE?
# ==============================================================================
st.header("3. How to Track Token Usage: Prompt vs Completion Spans, Context Growth, and Token Spike Detection", divider="rainbow")

st.markdown("""
Tokens are the core unit of capacity, cost, and latency in generative AI. 
LangSmith automatically extracts token counts from provider completion responses and aggregates them 
hierarchically across the trace tree.
""")

token_cols = st.columns(3)
with token_cols[0]:
    st.markdown("""
    <div class="metric-card">
        <h4>Prompt Tokens</h4>
        <span class="badge-prod">INPUT VOLUME</span>
        <p>System instructions, few-shot examples, retrieved document chunks, and chat conversation history.</p>
        <p style="color: #94A3B8; font-size: 12px;">Dominates RAG pipelines; cheaper per token.</p>
    </div>
    """, unsafe_allow_html=True)

with token_cols[1]:
    st.markdown("""
    <div class="metric-card">
        <h4>Completion Tokens</h4>
        <span class="badge-prod">OUTPUT GENERATION</span>
        <p>Tokens synthesized by the LLM during autoregressive decoding.</p>
        <p style="color: #94A3B8; font-size: 12px;">3x to 4x more expensive per token; dictates streaming duration.</p>
    </div>
    """, unsafe_allow_html=True)

with token_cols[2]:
    st.markdown("""
    <div class="metric-card">
        <h4>Total Trace Tokens</h4>
        <span class="badge-prod">HIERARCHICAL SUM</span>
        <p>The sum of all tokens across root runs, sub-chains, tool planning nodes, and guardrail classifiers.</p>
        <p style="color: #94A3B8; font-size: 12px;">Essential for per-request cost attribution.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
### Detecting Token Inflation and runaway Context Growth
- **Unbounded Chat History**: If a conversational application appends every user and assistant message without a sliding window (`k=10`) or summarization pass, token usage grows linearly with every user turn.
- **Recursive Agent Loops**: An agent stuck in a tool invocation loop will resend the growing trajectory repeatedly, causing token usage to explode exponentially.
- **Over-Retrieval in RAG**: Retrieving `top_k=20` chunks of 1,000 tokens injects 20,000 tokens into every prompt, 80% of which are irrelevant noise.
""")

# ==============================================================================
# SECTION 4: HOW DO YOU MONITOR LLM COSTS?
# ==============================================================================
st.header("4. How to Monitor LLM Costs: Per-Tenant Cost Attribution, Model Price Mapping, and Spend Anomaly Guards", divider="rainbow")

st.markdown("""
Unlike traditional software where compute cost is mostly fixed, **LLM operational costs are variable and query-dependent**. 
A single malicious or accidental query can cost dollars if not guarded by strict quotas and cost attribution models.
""")

st.markdown("""
### Enterprise Cost Management Architecture in LangSmith
1. **Dynamic Model Pricing Catalogs**:
   LangSmith maintains internal pricing dictionaries mapping model IDs (e.g., `gpt-4o-2024-08-06`, `claude-3-5-sonnet`) 
   to prompt and completion token unit costs, computing exact dollar costs per run automatically.
2. **Multi-Tenant Cost Attribution**:
   By tagging runs with metadata: `metadata={"tenant_id": "cust-842", "tier": "enterprise", "feature": "billing-rag"}`, 
   engineering teams can group, filter, and allocate exact infrastructure costs to specific customers or product features.
3. **Spend Anomaly Detection**:
   Establish automated alerts when daily spend exceeds a projected budget, or when single-run costs exceed a threshold (e.g., `cost > $0.15`).
""")

cost_dot = """
digraph CostAttribution {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4, fontname="Helvetica"];
    node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10];

    run_span [label="LLM Run Telemetry\\nModel: gpt-4o | Prompt: 4,000 tokens | Comp: 600 tokens", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    pricing_map [label="LangSmith Pricing Catalog\\nPrompt: $2.50/M | Comp: $10.00/M", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    cost_calc [label="Calculated Run Cost: $0.0160 USD", fillcolor="#1E1B4B", fontcolor="#A78BFA"];

    subgraph cluster_breakdown {
        label = "Metadata Dimensions (Cost Attribution)";
        style = "filled,rounded";
        fillcolor = "#1E293B";
        color = "#64748B";
        fontcolor = "#F8FAFC";

        dim_tenant [label="tenant_id: acme-corp", fillcolor="#0F172A", fontcolor="#34D399"];
        dim_feature [label="feature: doc-search", fillcolor="#0F172A", fontcolor="#38BDF8"];
        dim_env [label="environment: production", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    }

    run_span -> cost_calc;
    pricing_map -> cost_calc;
    cost_calc -> dim_tenant;
    cost_calc -> dim_feature;
    cost_calc -> dim_env;
}
"""
st.graphviz_chart(cost_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: HOW DO YOU DEBUG PRODUCTION TRACES?
# ==============================================================================
st.header("5. How to Debug Production Traces: Filtering by Status, Inspecting Inputs/Outputs, and Root Cause Analysis", divider="rainbow")

st.markdown("""
When a production LLM request fails or gives a bizarre response, traditional logs (e.g., stdout print statements) 
fail to provide context because they lack parent-child hierarchy, prompts are truncated, and token metadata is missing. 
LangSmith provides full-fidelity trace inspection.
""")

st.markdown("""
### 4-Step Production Trace Debugging Workflow
1. **Filter by Error Status & Latency Spikes**:
   In the LangSmith trace browser, apply filter queries: `and(eq(status, "error"), gte(duration, 5.0))`. This surfaces runs that crashed after prolonged execution.
2. **Inspect the Execution Tree (Root vs Child Spans)**:
   Identify precisely which node in the DAG failed. Did the root chain fail, or did a leaf node like a VectorDB retriever or Python sandbox tool crash?
3. **Inspect Exact Prompt Inputs and Provider Error Payloads**:
   View the raw, fully-interpolated prompt string sent to the model (including system messages, tool definitions, and few-shot examples) and the raw provider response code (e.g., `400 Bad Request: Context window exceeded`).
4. **Reproduce in LangSmith Playground**:
   Click **Open in Playground** directly from the failed production trace. Modify temperature, change prompt parameters, or test alternative models without redeploying code.
""")

# ==============================================================================
# SECTION 6: HOW DO YOU MONITOR RAG QUALITY?
# ==============================================================================
st.header("6. How to Monitor RAG Quality: Online Feedback Collection, User Signals, and Drift Detection", divider="rainbow")

st.markdown("""
Offline evaluation on test datasets only measures expected scenarios. In production, real users ask novel questions, 
documents in the knowledge base get updated or corrupted, and model providers silently update weights. 
Monitoring RAG quality requires **continuous online feedback collection**.
""")

st.markdown("""
### Production RAG Quality Monitoring Signals
- **Explicit User Feedback (Thumbs Up / Thumbs Down / Star Ratings)**:
  Directly log user interactions against the LangSmith `run_id` via `client.create_feedback()`.
- **Implicit User Signals (Copy-to-Clipboard, Dwell Time, Regenerate Clicks)**:
  When a user immediately hits "Regenerate Answer" or reformulates their query, flag the preceding run with negative implicit feedback.
- **Automated Online Evaluators (Sampling Shadow Evaluators)**:
  Configure LangSmith rules to automatically sample 5% of production runs and evaluate them in the background using an LLM-as-a-judge for groundedness and safety.
- **Retriever Drift Detection**:
  Monitor the average cosine similarity of retrieved documents over time. A sudden drop indicates that user queries are diverging from the indexed corpus topics.
""")

# ==============================================================================
# SECTION 7: HOW DO YOU MONITOR AGENT RELIABILITY?
# ==============================================================================
st.header("7. How to Monitor Agent Reliability: Tool Failure Rates, Recursion Loop Aborts, and State Invariant Violations", divider="rainbow")

st.markdown("""
Agents are autonomous systems that make decisions in cyclic feedback loops. 
Unlike simple prompt chains, agents fail in distinctive operational modes that require specialized telemetry.
""")

st.markdown("""
### Core Reliability Metrics for Production Agents
1. **Tool Invocation Failure Rate**:
   Percentage of tool calls that throw runtime exceptions (e.g., API 500, JSON Schema validation failure, SQL syntax error).
2. **Recursion Limit Trip Rate**:
   In LangGraph, hitting the maximum step limit (e.g., `recursion_limit=25`) means the agent entered an unrecoverable infinite loop. 
   Tracking recursion trips is a primary indicator of planning failure.
3. **Tool Hallucination Rate**:
   The LLM attempts to call a function name that does not exist in its defined tool catalog.
4. **State Schema Invariant Violations**:
   The agent node outputs state updates that violate Pydantic validation schemas or drop required session keys.
""")

agent_trap_df = pd.DataFrame([
    {"Agent Failure Mode": "Infinite Ping-Pong Loop", "Symptom": "Agent alternates between two tools without making forward progress.", "Telemetry Signature": "Step count hits maximum recursion limit with repeating tool sequence in trace tree.", "Remedy": "Add loop-detection reducer in LangGraph state or enforce max_iterations per tool."},
    {"Agent Failure Mode": "Invalid Tool Schema Generation", "Symptom": "LLM passes strings where integers are required or hallucinates missing parameters.", "Telemetry Signature": "ValidationError in tool node span with HTTP 400 from tool executor.", "Remedy": "Use Pydantic V2 with strict type validation and few-shot tool calling prompts."},
    {"Agent Failure Mode": "Silent Tool Failure Swallowing", "Symptom": "Tool returns error string, but agent ignores error and synthesizes false answer.", "Telemetry Signature": "Tool node outputs error string, but next LLM span has high confidence answer.", "Remedy": "Raise explicit exceptions in tool logic or route tool errors to dedicated error recovery node."}
])
st.dataframe(agent_trap_df, use_container_width=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Telemetry Monitoring Pipeline with Custom Metrics and Feedback Logging", divider="rainbow")

st.markdown("""
Below is the production-grade Python implementation demonstrating how to trace a multi-component 
application with custom metadata, log latency metrics, record execution tags, and attach user feedback scores.
""")

st.code('''import time
from typing import Dict, Any
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

# Initialize LangSmith client
# In production, ensure LANGSMITH_TRACING="true" and LANGSMITH_PROJECT="production-telemetry"
client = Client()

@traceable(
    run_type="retriever",
    name="production_vector_retriever",
    tags=["vector-db", "qdrant", "prod"]
)
def retrieve_support_docs(query: str) -> list[str]:
    """
    Traced retriever span: records latency and retrieved context.
    """
    time.sleep(0.08) # Simulates network retrieval
    return [
        "SLA Tier 1: Hardware replacements are dispatched within 24 business hours.",
        "SLA Tier 2: Priority ticketing with 4-hour initial response time."
    ]

@traceable(
    run_type="chain",
    name="enterprise_support_pipeline",
    tags=["customer-support", "tier-1"]
)
def customer_support_pipeline(user_query: str, tenant_id: str) -> Dict[str, Any]:
    """
    Root traced pipeline with custom metadata injection for cost and tenant attribution.
    """
    # Access current active run tree to attach dynamic runtime metadata
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.metadata["tenant_id"] = tenant_id
        run_tree.metadata["environment"] = "production"
        run_tree.metadata["app_version"] = "v2.4.1"

    # Step 1: Retrieval
    docs = retrieve_support_docs(user_query)

    # Step 2: Generation simulation
    time.sleep(0.25)
    answer = f"Based on SLA guidelines: {docs[0]}"

    return {
        "answer": answer,
        "retrieved_count": len(docs),
        "run_id": str(run_tree.id) if run_tree else "local-span"
    }

# Simulating production request lifecycle & feedback ingestion
response = customer_support_pipeline(
    user_query="When will my replacement hardware ship?",
    tenant_id="tenant-enterprise-901"
)

print(f"Generated Response: {response['answer']}")
print(f"Associated LangSmith Run ID: {response['run_id']}")

# Log User Feedback (Thumbs Up / Thumbs Down) asynchronously
if response["run_id"] != "local-span":
    client.create_feedback(
        run_id=response["run_id"],
        key="user_satisfaction",
        score=1.0, # 1.0 = positive thumbs_up, 0.0 = negative thumbs_down
        comment="User clicked thumbs up in support portal web UI.",
        source_info={"interface": "web-widget", "user_role": "admin"}
    )
    print("Feedback successfully attached to production run!")
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Production Observability Console, Trace Inspector, & Alert Simulator", divider="rainbow")

st.markdown("""
Use the interactive dashboard below to filter live production trace streams, 
inspect latency anomalies, view token cost attributions, and simulate automated alerts.
""")

wb_view = st.selectbox(
    "Select Observability Tool:",
    ["1. Production Trace Browser & Root Cause Inspector", "2. Dynamic Token Cost Calculator", "3. SLI/SLO Alert Threshold Simulator"]
)

if wb_view == "1. Production Trace Browser & Root Cause Inspector":
    st.subheader("Live Production Trace Stream")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        status_filter = st.selectbox("Filter by Run Status:", ["ALL", "SUCCESS", "ERROR"])
    with f_col2:
        min_lat = st.slider("Minimum Latency Filter (ms):", 0, 15000, 500, 250)

    filtered_traces = prod_engine.filter_traces(status_filter, min_lat)
    st.markdown(f"**Found {len(filtered_traces)} matching production traces:**")

    trace_df = pd.DataFrame(filtered_traces)
    st.dataframe(trace_df[["trace_id", "app", "status", "total_latency_ms", "prompt_tokens", "completion_tokens", "cost_usd", "feedback"]], use_container_width=True)

    if filtered_traces:
        selected_tid = st.selectbox("Select Trace ID to Inspect Details:", [t["trace_id"] for t in filtered_traces])
        detail = next(t for t in filtered_traces if t["trace_id"] == selected_tid)

        st.markdown(f"#### Trace Inspection: `{detail['trace_id']}`")
        if detail["status"] == "ERROR":
            st.error(f"**Failure Error Payload:** {detail['error']}")
        else:
            st.success("Run completed successfully with zero span errors.")

        st.info(f"**Time to First Token (TTFT):** {detail['ttft_ms']}ms | **Total Duration:** {detail['total_latency_ms']}ms | **Tool Calls:** {detail['tool_calls']}")

elif wb_view == "2. Dynamic Token Cost Calculator":
    st.subheader("Operational Cost Modeler")
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        sel_model = st.selectbox("Model Checkpoint:", ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"])
    with c_m2:
        in_prompt = st.number_input("Average Prompt Tokens:", min_value=100, max_value=128000, value=2500, step=500)
    with c_m3:
        in_comp = st.number_input("Average Completion Tokens:", min_value=10, max_value=8192, value=350, step=50)

    reqs_per_day = st.slider("Expected Daily Query Volume:", 1000, 1000000, 50000, 5000)

    cost_per_query = prod_engine.compute_cost(sel_model, in_prompt, in_comp)
    daily_cost = round(cost_per_query * reqs_per_day, 2)
    monthly_cost = round(daily_cost * 30, 2)

    met1, met2, met3 = st.columns(3)
    with met1:
        st.metric("Cost Per Single Query", f"${cost_per_query}")
    with met2:
        st.metric("Projected Daily Spend", f"${daily_cost}")
    with met3:
        st.metric("Projected Monthly Spend", f"${monthly_cost}")

else:
    st.subheader("SLI/SLO Alert Rule Simulator")
    st.markdown("Configure threshold rules to simulate real-time PagerDuty / Slack alerts on telemetry drift:")

    alert_lat = st.slider("p95 Latency SLO Target (ms):", 500, 5000, 2000, 250)
    alert_err = st.slider("Error Rate SLO Target (%):", 0.1, 10.0, 2.0, 0.5)

    sim_p95 = 2650 # Simulated current telemetry
    sim_err = 3.5  # Simulated current error rate

    st.markdown(f"**Simulated Production SLIs:** Current p95 = `{sim_p95}ms` | Current Error Rate = `{sim_err}%`")

    if sim_p95 > alert_lat:
        st.error(f"ALERT FIRED: p95 Latency ({sim_p95}ms) exceeds SLO threshold ({alert_lat}ms)! Triggering auto-scaling rule.")
    else:
        st.success(f"Latency SLO Healthy: Current p95 ({sim_p95}ms) within allowable limit.")

    if sim_err > alert_err:
        st.error(f"ALERT FIRED: Production error rate ({sim_err}%) breaches threshold ({alert_err}%)! Routing webhook to Slack #incident-ai.")
    else:
        st.success("Error Rate SLO Healthy.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why is average latency an ineffective metric for monitoring production LLMs, and what should be used instead?"):
    st.markdown("""
    **Answer:**  
    LLM latency exhibits heavy right-skewed tail distributions due to varying prompt lengths, retrieval chunk counts, 
    and model provider network queues. A low average latency can hide catastrophic 10-second delays experienced by 
    users with long queries. Teams should monitor percentile distributions (p50, p90, p95, and p99) alongside 
    Time to First Token (TTFT) for streaming responsiveness.
    """)

with st.expander("Q2: How does LangSmith ensure that telemetry tracing does not add latency to user requests in production?"):
    st.markdown("""
    **Answer:**  
    LangSmith uses an asynchronous, non-blocking background queue architecture. The `@traceable` decorator 
    records spans in local memory and hands them off to a background thread pool (Batch Exporter). 
    Events are batched, compressed with gzip, and posted asynchronously to the LangSmith ingestion API. 
    If the network drops or LangSmith times out, the user request completes without failure or latency impact.
    """)

with st.expander("Q3: How do you attribute LLM infrastructure costs to specific enterprise tenants in a multi-tenant SaaS application?"):
    st.markdown("""
    **Answer:**  
    When invoking chains or agents, inject tenant identifiers into the run tree metadata 
    (e.g., `run_tree.metadata["tenant_id"] = "cust_123"`). LangSmith indexes this metadata, allowing engineers 
    to query aggregated token counts and dollar spends grouped by tenant in the dashboard or via the Python SDK.
    """)

with st.expander("Q4: What telemetry signals indicate that an autonomous agent is trapped in an infinite loop?"):
    st.markdown("""
    **Answer:**  
    1. Hitting the maximum recursion limit (e.g., LangGraph `GraphRecursionError`).
    2. Repeated sequences of identical tool calls with identical arguments in the trace span tree.
    3. Monotonically increasing total latency and prompt token count per step without generating final answer text.
    4. Rapid spike in run cost for a single session.
    """)

with st.expander("Q5: How do you establish an active learning loop between production failures and golden evaluation datasets?"):
    st.markdown("""
    **Answer:**  
    When users submit negative feedback (thumbs down) or when automated guardrails flag an error, 
    a background worker captures the failed trace via the LangSmith SDK and adds the input/output pair 
    to a dedicated 'Edge-Cases & Failures' dataset. Engineers can then reproduce, add golden ground truths, 
    and run regression evaluations before shipping a fix.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Synchronous Tracing or Blocking Logging in the User Request Path**  
Using blocking HTTP calls to report telemetry directly inside the user request loop causes user-facing latency 
to spike whenever the observability backend experiences network lag. Always use asynchronous batch workers.
""")

st.error("""
**Trap 2: Failure to Implement Hard Recursion Limits and Cost Circuit Breakers**  
Deploying an agent without a strict recursion limit (e.g., maximum 10 steps) allows buggy prompts to loop 
hundreds of times, exhausting API budgets and racking up massive cloud bills in minutes.
""")

st.warning("""
**Trap 3: Neglecting Per-Tenant Rate Limiting and Quota Enforcements**  
Allowing unmetered access allows a single runaway tenant script to consume global provider rate limits (HTTP 429), 
causing denial of service for all other users.
""")

st.warning("""
**Trap 4: Discarding Negative Production Traces Without Golden Dataset Ingestion**  
Simply resolving production error tickets without persisting the offending inputs into regression test suites 
guarantees that the same regression will recur in future releases.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Time to First Token (TTFT)</h4>
        <p>The duration from prompt dispatch to the first streaming token arriving at the client, measuring perceived responsiveness.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Batch Exporter</h4>
        <p>An asynchronous background worker thread that pools and posts trace spans without delaying the application critical path.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Cost Attribution</h4>
        <p>The practice of tagging runs with tenant, feature, and environment metadata to allocate variable LLM token spend accurately.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Active Learning Loop</h4>
        <p>The workflow of filtering negative production feedback traces and backfilling them into versioned golden evaluation datasets.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can differentiate Time to First Token (TTFT) from total duration and explain why p95/p99 percentiles matter.", value=True)
chk_2 = st.checkbox("I know how to track prompt, completion, and total tokens across hierarchical trace trees in LangSmith.", value=True)
chk_3 = st.checkbox("I can implement multi-tenant cost attribution using metadata tagging and dynamic pricing catalogs.", value=True)
chk_4 = st.checkbox("I understand how to debug production failures using trace filtering, input/output inspection, and the Playground.", value=True)
chk_5 = st.checkbox("I can design online feedback collection pipelines capturing explicit and implicit user satisfaction signals.", value=True)
chk_6 = st.checkbox("I can detect agent reliability hazards including recursion limit trips, tool call failures, and state schema violations.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Production observability interview topics and engineering standards verified! Ready for LangSmith Coding Problems.")
