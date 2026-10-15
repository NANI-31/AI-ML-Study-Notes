import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.3 Monitoring - LangChain Master Curriculum",
    page_icon="📡",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ProductionMonitoringDashboardEngine:
    """
    Enterprise Monitoring and SLA Surveillance Engine for LangChain applications.
    Simulates real-time surveillance across the 7 production observability vectors:
    1. Production Traces (Span tree generation, parent-child run linkages)
    2. Error Rates (HTTP 429 rate limits, 500 crashes, parse timeouts)
    3. Latency Metrics (p50, p90, p99 percentiles, TTFT profiling)
    4. Token Usage (Input vs output token velocity, context saturation)
    5. Model Costs (Real-time expenditure tracking against tenant budgets)
    6. Tool Failures (API timeouts, tool rejection ratios, circuit breaker status)
    7. User Feedback (Implicit click-throughs, explicit thumbs up/down ratings)
    """
    def __init__(self, tenant_id: str = "enterprise-tenant-alpha"):
        self.tenant_id = tenant_id

    def generate_simulated_telemetry(
        self,
        request_count: int = 500,
        primary_model: str = "gpt-4o",
        tool_failure_rate_pct: float = 5.0,
        sla_p95_threshold_ms: float = 2500.0,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Simulates an active stream of production traces and computes golden operational KPIs.
        """
        rng = np.random.default_rng(random_seed)

        # 1. Cost model pricing (per 1M tokens)
        pricing_table = {
            "gpt-4o": {"in": 2.50, "out": 10.00},
            "gpt-4o-mini": {"in": 0.15, "out": 0.60},
            "claude-3-5-sonnet": {"in": 3.00, "out": 15.00}
        }
        rates = pricing_table.get(primary_model, {"in": 2.50, "out": 10.00})

        # 2. Synthetic token counts
        input_tokens = rng.integers(300, 1800, size=request_count)
        output_tokens = rng.integers(80, 500, size=request_count)
        total_input_tokens = int(np.sum(input_tokens))
        total_output_tokens = int(np.sum(output_tokens))
        total_tokens = total_input_tokens + total_output_tokens

        # 3. Cost calculation in USD
        total_cost_usd = round(
            (total_input_tokens / 1_000_000.0) * rates["in"] +
            (total_output_tokens / 1_000_000.0) * rates["out"],
            4
        )

        # 4. Latency distribution (log-normal distribution)
        base_latencies = rng.lognormal(mean=7.0, sigma=0.45, size=request_count)  # in ms
        p50_latency = float(np.percentile(base_latencies, 50))
        p90_latency = float(np.percentile(base_latencies, 90))
        p95_latency = float(np.percentile(base_latencies, 95))
        p99_latency = float(np.percentile(base_latencies, 99))

        # 5. Tool execution & error rates
        tool_invocations = rng.integers(1, 4, size=request_count)
        total_tool_calls = int(np.sum(tool_invocations))
        simulated_failures = rng.uniform(0, 100, size=total_tool_calls) < tool_failure_rate_pct
        failed_tool_calls = int(np.sum(simulated_failures))
        tool_error_rate = round((failed_tool_calls / max(1, total_tool_calls)) * 100.0, 2)

        # General pipeline errors (rate limits, network, tool crashes)
        pipeline_errors = rng.uniform(0, 100, size=request_count) < (tool_failure_rate_pct * 0.6 + 1.2)
        total_pipeline_errors = int(np.sum(pipeline_errors))
        overall_error_rate = round((total_pipeline_errors / request_count) * 100.0, 2)

        # 6. User feedback ratings (1 to 5 stars or thumbs up/down)
        feedback_given_mask = rng.uniform(0, 1, size=request_count) < 0.35
        feedback_count = int(np.sum(feedback_given_mask))
        if feedback_count > 0:
            # Positive correlation with latency and tool health
            base_satisfaction = 0.88 - (tool_failure_rate_pct * 0.015)
            thumbs_up_flags = rng.uniform(0, 1, size=feedback_count) < max(0.4, base_satisfaction)
            thumbs_up_count = int(np.sum(thumbs_up_flags))
            positive_feedback_pct = round((thumbs_up_count / feedback_count) * 100.0, 1)
        else:
            feedback_count = 0
            thumbs_up_count = 0
            positive_feedback_pct = 92.0

        # SLA compliance check
        p95_violation = p95_latency > sla_p95_threshold_ms

        # Generate sample trace table
        sample_size = min(10, request_count)
        traces = []
        for i in range(sample_size):
            err_flag = bool(pipeline_errors[i])
            tool_err = bool(failed_tool_calls > 0 and (i % 3 == 0) and tool_failure_rate_pct > 0)
            status_str = "FAILED" if (err_flag or tool_err) else "OK"
            traces.append({
                "Trace ID": f"trc-{i+1001:04d}",
                "Model": primary_model,
                "Prompt Tokens": int(input_tokens[i]),
                "Completion Tokens": int(output_tokens[i]),
                "Latency (ms)": round(float(base_latencies[i]), 1),
                "Tool Calls": int(tool_invocations[i]),
                "Status": status_str,
                "User Feedback": "👍 Positive" if (i % 2 == 0) else ("👎 Negative" if err_flag else "None")
            })

        return {
            "request_count": request_count,
            "primary_model": primary_model,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": total_cost_usd,
            "p50_latency_ms": round(p50_latency, 1),
            "p90_latency_ms": round(p90_latency, 1),
            "p95_latency_ms": round(p95_latency, 1),
            "p99_latency_ms": round(p99_latency, 1),
            "sla_p95_threshold_ms": sla_p95_threshold_ms,
            "p95_violation": p95_violation,
            "total_tool_calls": total_tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "tool_error_rate_pct": tool_error_rate,
            "total_pipeline_errors": total_pipeline_errors,
            "overall_error_rate_pct": overall_error_rate,
            "feedback_count": feedback_count,
            "positive_feedback_pct": positive_feedback_pct,
            "sample_traces": traces
        }


# ==============================================================================
# HEADER BLOCK
# ==============================================================================
st.title("23.3 Monitoring: Production Surveillance and Real-Time Telemetry Cockpits")
st.caption("LangChain v0.3 Enterprise Production Operations • Distributed Traces, Percentile Latency, Token Velocity, Cost Attribution, Tool Resilience, and User Feedback")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production LLM Monitoring and Telemetry Surveillance Architecture", divider="rainbow")

st.markdown("""
Production LLM applications are non-deterministic, distributed systems that connect cloud models, 
vector databases, API endpoints, and autonomous agents. Traditional server monitoring (CPU, RAM, HTTP status codes) 
is entirely blind to LLM degradation modes such as context bloat, prompt drift, token billing explosions, 
silent tool hallucinations, and degraded completion relevance.

An enterprise LangChain surveillance engine implements a multi-tier telemetry pipeline that captures every prompt, 
tool call, intermediate JSON payload, and token stream into structured distributed traces, continuously computing 
real-time percentiles and alerting on SLA violations.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MonitoringBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=11, fontcolor="#F8FAFC", style="filled", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_app_tier {
            label = "Tier 1: Application Ingestion and Runtime Traces";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            app_gateway [label="Production Application Gateway\\n(FastAPI / Streamlit / LangServe)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            lcel_chain [label="LCEL Pipeline / StateGraph Runtime\\n(Traced Runnables with Callbacks)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            callback_dispatcher [label="LangChain BaseCallbackHandler\\n(on_chain_start, on_tool_end, on_llm_new_token)", shape=ellipse, fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_telemetry_bus {
            label = "Tier 2: Asynchronous Telemetry Ingestion Bus";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            telemetry_queue [label="High-Throughput Buffer\\n(Kafka / Redis Stream / Async Queue)", shape=box, fillcolor="#B45309", color="#F59E0B"];
            span_processor [label="Distributed Trace Assembler\\n(Parent-Child Spans & Token Counters)", shape=box, fillcolor="#B45309", color="#F59E0B"];
        }

        subgraph cluster_analytics_tier {
            label = "Tier 3: Aggregation, Telemetry Stores & Surveillance Engines";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            trace_db [label="Trace & Span Store\\n(LangSmith / OpenTelemetry / ClickHouse)", shape=cylinder, fillcolor="#4C1D95", color="#8B5CF6"];
            metric_store [label="Time-Series Metric Store\\n(Prometheus / Datadog / CloudWatch)", shape=cylinder, fillcolor="#4C1D95", color="#8B5CF6"];
            feedback_store [label="Feedback & Annotation Store\\n(User Thumbs Up/Down & Annotations)", shape=cylinder, fillcolor="#4C1D95", color="#8B5CF6"];
        }

        subgraph cluster_observability_tier {
            label = "Tier 4: Live Cockpit, Budget Governors & Alarm Triggers";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            cockpit_dashboard [label="Real-Time Observability Cockpit\\n(p50/p90/p99 Latency, Error Spikes, Costs)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
            alert_manager [label="SLA Alarm Manager & Circuit Breaker\\n(PagerDuty, Slack, Auto-Fallback Tripwires)", shape=octagon, fillcolor="#991B1B", color="#EF4444"];
        }

        app_gateway -> lcel_chain [label="User Request"];
        lcel_chain -> callback_dispatcher [label="Execution Events"];
        callback_dispatcher -> telemetry_queue [label="Non-Blocking Async Flush"];
        telemetry_queue -> span_processor [label="Event Batches"];
        span_processor -> trace_db [label="Complete Trace Trees"];
        span_processor -> metric_store [label="Aggregated Metrics"];
        app_gateway -> feedback_store [label="User Reactions"];
        trace_db -> cockpit_dashboard [label="Trace Drilldown"];
        metric_store -> cockpit_dashboard [label="KPI Streaming"];
        feedback_store -> cockpit_dashboard [label="Quality Ratings"];
        metric_store -> alert_manager [label="p95 > SLA or Rate Limit"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: PRODUCTION TRACES
# ==============================================================================
st.header("2. Production Traces (Distributed Span Aggregation, End-to-End Run Trees and Trace Retention)", divider="rainbow")

st.markdown("""
In distributed systems, a single trace captures the complete lifecycle of a request as it traverses multiple services. 
In LangChain, a **Run Tree** represents an invocation tree composed of nested **Spans**:

1. **Root Span**: The overarching client invocation (e.g., `POST /api/chat`), recording total request wall-clock time and user identity.
2. **Chain Spans**: Intermediate LCEL transformations, prompt construction, memory formatting, and reranking.
3. **Model Spans**: Concrete LLM invocations recording prompt tokens, completion tokens, time-to-first-token (TTFT), and temperature.
4. **Tool Spans**: Database queries, web searches, calculator runs, and external API requests with raw JSON inputs and outputs.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise environments, tracing must never block user-facing inference threads. Handlers serialize trace events 
into background thread worker pools or asynchronous queues (such as Redis or Kafka), buffering payloads before flushing 
in batches to OpenTelemetry collectors, LangSmith, or ClickHouse. Trace retention policies often store 100% of error 
and negative-feedback traces, while downsampling successful traces to 5% to 10% to curb storage bills.
""")

with st.expander("🔬 Trace Hierarchy Visualization"):
    trace_tree_dot = """
    digraph TraceTree {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=10, fontcolor="#F8FAFC", style="filled", penwidth=1.2];
        edge [fontname="Arial", fontsize=9, fontcolor="#94A3B8", color="#38BDF8"];

        root [label="Root Run: POST /v1/chat (Total: 1840ms)\\nstatus=200, tenant=enterprise-alpha", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
        retriever [label="Child Span 1: VectorRetriever (120ms)\\ntop_k=4, score_threshold=0.75", shape=box, fillcolor="#047857", color="#10B981"];
        prompt [label="Child Span 2: ChatPromptTemplate (5ms)\\ntemplate_format=f-string", shape=box, fillcolor="#047857", color="#10B981"];
        llm_call [label="Child Span 3: ChatOpenAI gpt-4o (1650ms)\\nTTFT=320ms, tokens=740, cost=0.0034 USD", shape=box, fillcolor="#B45309", color="#F59E0B"];
        tool_call [label="Grandchild Span 3.1: PythonREPLTool (65ms)\\ncode='sum([x**2 for x in data])'", shape=ellipse, fillcolor="#4C1D95", color="#8B5CF6"];
        parser [label="Child Span 4: StrOutputParser (2ms)\\noutput_type=str", shape=box, fillcolor="#047857", color="#10B981"];

        root -> retriever;
        root -> prompt;
        root -> llm_call;
        llm_call -> tool_call;
        root -> parser;
    }
    """
    st.graphviz_chart(trace_tree_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: ERROR RATES
# ==============================================================================
st.header("3. Error Rates (Exception Categorization, HTTP 429 Alerts and Failure Spikes)", divider="rainbow")

st.markdown("""
LLM applications fail differently than standard CRUD applications. Error tracking must decompose failure rates into 
granular categories to isolate provider outages from application bugs:

- **Rate Limit Errors (HTTP 429)**: Provider quota exhaustion (Tokens Per Minute - TPM, or Requests Per Minute - RPM). Indicates traffic surges or uneven batch scheduling.
- **Provider Outages (HTTP 502/503/504)**: OpenAI, Anthropic, or Azure infrastructure dropouts requiring automatic model failover.
- **Context Length Exceeded (HTTP 400)**: Prompt plus max completion tokens exceeds model window limits (e.g., 128k tokens for GPT-4o).
- **Tool Execution Crashes**: Schema mismatches, database query timeouts, or uncaught tool code exceptions.
- **JSON Parsing Errors**: The model outputs conversational markdown (`Here is the JSON: ...`) instead of raw JSON schema, breaking downstream parsers.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production monitoring setups define automated alert rules on Prometheus or Datadog:
- **Warning Alarm**: Overall error rate exceeds 2.0% over a 5-minute rolling window.
- **Critical Alarm**: HTTP 429 errors exceed 5.0% or 503 errors exceed 1.0%, triggering an immediate PagerDuty page and tripping circuit breakers to secondary providers.
""")

st.divider()

# ==============================================================================
# SECTION 4: LATENCY
# ==============================================================================
st.header("4. Latency (Percentile Profiling: p50/p90/p99, TTFT vs Inter-Token Latency and Degradation Alarms)", divider="rainbow")

st.markdown("""
Averages are notoriously deceptive in Generative AI. A mean latency of 1.2 seconds can easily mask a 99th percentile (p99) 
of 12.5 seconds caused by severe prompt queuing, large context prefill, or multi-step tool loops. 

Engineers monitor three core latency dimensions:
1. **Time-to-First-Token (TTFT)**: The duration from request dispatch until the first token arrives at the client browser. Measures prompt processing speed and server queue time.
2. **Inter-Token Latency (ITL)**: The generation time per subsequent token (typically 20ms to 50ms per token). Measures decoding speed.
3. **Total Round-Trip Latency (e2e)**: The complete lifecycle duration, evaluated at **p50** (median user experience), **p90** (tail experience), and **p99** (worst-case SLA bound).
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production web apps, TTFT determines perceived responsiveness. If TTFT exceeds 1500ms, users perceive the system 
as hung and abandon sessions. Production SLAs guarantee `p95 TTFT <= 1200ms` and `p95 Total <= 3500ms`. When p95 exceeds 
these boundaries, automated autoscaling triggers or model routing switches to lightweight models like `gpt-4o-mini`.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOKEN USAGE
# ==============================================================================
st.header("5. Token Usage (Input vs Output Token Velocity, Prompt Bloat and Context Length Auditing)", divider="rainbow")

st.markdown("""
Monitoring token consumption is essential for both operational reliability and cost control. 
Engineers distinguish sharply between **Input (Prompt) Tokens** and **Output (Completion) Tokens**:

- **Input Token Velocity**: Tracks how quickly prompts expand. In conversational bots, unmanaged chat histories cause context bloat, where turn 20 consumes 10x the tokens of turn 1 without adding new information.
- **Output Token Velocity**: Measures generative volume. Runaway loops or verbose model instructions lead to high generation times and token waste.
- **Context Saturation Ratio**: Calculated as `current_tokens / model_context_limit`. When this ratio exceeds 80%, compression algorithms (such as recursive summarization) must be triggered immediately.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production pipelines track token consumption per tenant, per endpoint, and per model version. 
Sudden spikes in average input tokens alert engineers to retrieval query leakage or document chunk duplication 
in vector ingestion pipelines.
""")

st.divider()

# ==============================================================================
# SECTION 6: MODEL COSTS
# ==============================================================================
st.header("6. Model Costs (Real-Time Budget Consumption, Pricing Tiers and Tenant Unit Economics)", divider="rainbow")

st.markdown("""
Without active surveillance, generative AI features can exhaust monthly infrastructure budgets in days. 
Because LLM providers charge asymmetrically (completion tokens are typically 3x to 5x more expensive than prompt tokens), 
real-time financial monitoring is mandatory:

- **Per-Tenant Unit Economics**: Attributing every cent to specific enterprise customers or internal teams.
- **Cost Per Successful Interaction**: Normalizing expenditure against business value (e.g., cost per customer ticket resolved).
- **Dynamic Cost Governors**: Automated rate limit throttles or degradation triggers that prevent runaway agent loops from running up thousands of dollars.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern AI platforms attach pricing metadata to model descriptors. As traces flush, cost aggregators compute 
usage in USD in real time. If a tenant consumes 90% of their monthly allocated budget, the system automatically 
degrades model routing from premier models (GPT-4o) to cost-effective alternatives (GPT-4o-mini).
""")

st.divider()

# ==============================================================================
# SECTION 7: TOOL FAILURES
# ==============================================================================
st.header("7. Tool Failures (External API Crash Tracking, Timeout Ratios and Flaky Tool Circuit Breakers)", divider="rainbow")

st.markdown("""
Autonomous agents rely heavily on tools: database connectors, internal REST APIs, vector search indices, and web search engines. 
When tools degrade, the entire agent pipeline risks collapse:

- **Flaky Tool Syndrome**: A third-party API that times out 15% of the time causes multi-step agents to spin out into repeated retry loops.
- **Schema Rejection**: Tools receiving invalid arguments formatted by the LLM (e.g., string passed where integer expected).
- **Circuit Breaker Protection**: When a tool's failure rate exceeds a critical threshold (e.g., 20% over 50 calls), the circuit breaker opens, bypassing the flaky tool and returning a graceful degradation message directly to the model.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Tool metrics are tracked on dedicated dashboards displaying invocation volume, success percentages, average execution time, 
and error distributions. Flaky tools are automatically sandboxed with strict execution timeouts (e.g., max 3000ms).
""")

st.divider()

# ==============================================================================
# SECTION 8: USER FEEDBACK
# ==============================================================================
st.header("8. User Feedback (Live Thumbs Up/Down Stream, Explicit Ratings and Quality Flywheels)", divider="rainbow")

st.markdown("""
System metrics (latency, errors) measure operational health, but only **User Feedback** measures actual semantic quality:

1. **Explicit Feedback**: Thumbs up (👍) / Thumbs down (👎), star ratings (1 to 5), and freeform user correction comments.
2. **Implicit Feedback**: Copy-to-clipboard actions, regenerated responses, conversation abandonments, and downstream click-throughs.
3. **The Quality Flywheel**: Every response marked with a negative rating (👎) is tagged with its trace ID and automatically routed into an evaluation dataset for offline regression benchmarking and prompt tuning.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise AI products embed feedback widgets directly into chat interfaces. Clicking thumbs-down sends an HTTP event 
linked to the trace ID, annotating the trace in LangSmith or internal databases. High-value enterprise customers 
often trigger automated slack notifications to AI engineering teams whenever a 1-star review is submitted.
""")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Production Monitoring Engine (ProductionMonitoringDashboardEngine)", divider="rainbow")

st.markdown("""
The following complete Python implementation demonstrates `ProductionMonitoringDashboardEngine`, an enterprise 
surveillance class that aggregates traces, tracks percentiles, audits tool errors, and evaluates SLA compliance.
""")

st.code(r'''
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ProductionMonitoringDashboardEngine:
    """
    Enterprise Monitoring and SLA Surveillance Engine for LangChain applications.
    """
    def __init__(self, tenant_id: str = "enterprise-tenant-alpha"):
        self.tenant_id = tenant_id

    def generate_simulated_telemetry(
        self,
        request_count: int = 500,
        primary_model: str = "gpt-4o",
        tool_failure_rate_pct: float = 5.0,
        sla_p95_threshold_ms: float = 2500.0,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(random_seed)

        pricing_table = {
            "gpt-4o": {"in": 2.50, "out": 10.00},
            "gpt-4o-mini": {"in": 0.15, "out": 0.60},
            "claude-3-5-sonnet": {"in": 3.00, "out": 15.00}
        }
        rates = pricing_table.get(primary_model, {"in": 2.50, "out": 10.00})

        input_tokens = rng.integers(300, 1800, size=request_count)
        output_tokens = rng.integers(80, 500, size=request_count)
        total_input_tokens = int(np.sum(input_tokens))
        total_output_tokens = int(np.sum(output_tokens))

        total_cost_usd = round(
            (total_input_tokens / 1_000_000.0) * rates["in"] +
            (total_output_tokens / 1_000_000.0) * rates["out"],
            4
        )

        base_latencies = rng.lognormal(mean=7.0, sigma=0.45, size=request_count)
        p50 = float(np.percentile(base_latencies, 50))
        p95 = float(np.percentile(base_latencies, 95))
        p99 = float(np.percentile(base_latencies, 99))

        tool_calls = rng.integers(1, 4, size=request_count)
        total_tool_calls = int(np.sum(tool_calls))
        failed_tool_calls = int(np.sum(rng.uniform(0, 100, size=total_tool_calls) < tool_failure_rate_pct))

        return {
            "request_count": request_count,
            "total_cost_usd": total_cost_usd,
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "p99_latency_ms": round(p99, 1),
            "p95_violation": p95 > sla_p95_threshold_ms,
            "tool_error_rate_pct": round((failed_tool_calls / max(1, total_tool_calls)) * 100.0, 2)
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Production Observability Cockpit and SLA Alarm Sandbox", divider="rainbow")

st.markdown("""
Tune live traffic volume, model tiers, tool instability, and SLA deadlines below to simulate 
how the production surveillance dashboard responds to traffic spikes and system degradation.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    traffic_volume = st.slider("Simulated Request Volume", min_value=100, max_value=2500, value=750, step=50)
    selected_model = st.selectbox("Primary Model Tier", options=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"], index=0)
with col_wb2:
    tool_failure_rate = st.slider("Simulated Tool Failure Rate (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
    sla_threshold = st.slider("Target p95 Latency SLA Threshold (ms)", min_value=1000.0, max_value=3500.0, value=2000.0, step=100.0)

# Instantiate Module-level engine
engine = ProductionMonitoringDashboardEngine()
telemetry_data = engine.generate_simulated_telemetry(
    request_count=traffic_volume,
    primary_model=selected_model,
    tool_failure_rate_pct=tool_failure_rate,
    sla_p95_threshold_ms=sla_threshold
)

st.subheader("📊 Live Telemetry Golden Signals")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Total Requests", f"{telemetry_data['request_count']:,}")
m_col2.metric("Total Tokens", f"{telemetry_data['total_tokens']:,}")
m_col3.metric("Total Cost (USD)", f"${telemetry_data['total_cost_usd']:.4f}")
m_col4.metric("User Approval (👍)", f"{telemetry_data['positive_feedback_pct']}%")

m_col5, m_col6, m_col7, m_col8 = st.columns(4)
m_col5.metric("p50 Latency", f"{telemetry_data['p50_latency_ms']} ms")
m_col6.metric("p95 Latency", f"{telemetry_data['p95_latency_ms']} ms", delta=f"{round(telemetry_data['p95_latency_ms'] - sla_threshold, 1)} ms vs SLA", delta_color="inverse")
m_col7.metric("Pipeline Error Rate", f"{telemetry_data['overall_error_rate_pct']}%")
m_col8.metric("Tool Error Rate", f"{telemetry_data['tool_error_rate_pct']}%")

# SLA Alarm Banners
if telemetry_data["p95_violation"]:
    st.error(f"🚨 CRITICAL SLA VIOLATION: Observed p95 latency ({telemetry_data['p95_latency_ms']} ms) exceeds SLA threshold ({sla_threshold} ms). Tripping autoscaling alert!")
else:
    st.success(f"✅ SLA COMPLIANT: Observed p95 latency ({telemetry_data['p95_latency_ms']} ms) is well within the {sla_threshold} ms target.")

if telemetry_data["tool_error_rate_pct"] > 15.0:
    st.warning(f"⚠️ TOOL INSTABILITY WARNING: Tool error rate is {telemetry_data['tool_error_rate_pct']}%. Recommend opening circuit breaker for degraded external endpoints.")

st.subheader("📋 Live Production Trace Stream (Sample)")
trace_df = pd.DataFrame(telemetry_data["sample_traces"])
st.dataframe(trace_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("11. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions on LLM Monitoring")

with st.expander("Q1: Why is monitoring average latency completely inadequate for production LLM systems?"):
    st.markdown("""
    **Answer**: Average latency hides the long-tail distribution typical of generative AI. 
    While 80% of short queries may complete in 800ms, multi-step agent queries and massive 50,000-token context prefill jobs 
    can take 15 to 30 seconds. Relying on mean latency masks these severe bottlenecks, which only become visible when tracking 
    **p95** and **p99** percentiles.
    """)

with st.expander("Q2: How do you differentiate between prompt prefill latency and autoregressive decoding latency?"):
    st.markdown("""
    **Answer**: Prompt prefill latency corresponds to **Time-to-First-Token (TTFT)**, which scales primarily with the size 
    of the input prompt and vector context. Autoregressive decoding latency corresponds to **Inter-Token Latency (ITL)**, 
    which scales with the number of generated completion tokens and GPU memory bandwidth. Measuring them separately allows 
    engineers to determine whether slow responses are caused by bloated prompts or overly verbose generation.
    """)

with st.expander("Q3: What telemetry strategy prevents streaming LLM calls from slowing down user responses?"):
    st.markdown("""
    **Answer**: Handlers must emit telemetry events **asynchronously** into non-blocking thread queues (or background worker loops) 
    rather than making synchronous HTTP calls inside callback hooks. If a callback synchronously posts trace data to an external 
    endpoint on every token chunk, network round-trips will severely bottleneck token streaming to the user.
    """)

with st.expander("Q4: How do you calculate per-tenant LLM unit economics in a shared multi-tenant cluster?"):
    st.markdown("""
    **Answer**: Every trace must be injected with tenant metadata (`tenant_id`, `department`, `app_name`). 
    As spans complete, prompt and completion tokens are tagged with that tenant ID and multiplied by the model's exact pricing tiers. 
    Aggregating these token costs over billing intervals provides precise per-tenant unit economics and cost attribution.
    """)

with st.expander("Q5: What is a Flaky Tool Circuit Breaker and how does it protect LLM agent workflows?"):
    st.markdown("""
    **Answer**: A circuit breaker tracks tool call failure rates over a sliding execution window. If failures exceed a set 
    threshold (e.g., 25% errors over 20 calls), the breaker 'trips' into an open state. Subsequent requests immediately fail fast 
    or fall back to cached responses without calling the broken external service, preventing the agent from spinning out in 
    prolonged, costly retry loops.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("**Trap 1 (Synchronous Telemetry Emission)**: Making blocking REST calls to logging backends inside token streaming callbacks. This degrades generation throughput and destroys user streaming experience.")
    st.warning("**Trap 2 (Unsampled Trace Retention)**: Storing 100% of full-text prompt and completion payloads forever. At scale, LLM trace databases quickly consume gigabytes of storage, incurring huge infrastructure costs.")
with col_t2:
    st.error("**Trap 3 (Blindness to Tool Timeouts)**: Catching tool timeouts without incrementing error metrics. The agent treats the empty response as normal data, silently hallucinating answers while monitoring shows 100% success.")
    st.warning("**Trap 4 (Ignoring User Feedback Flywheels)**: Collecting thumbs-up/down ratings without linking them to exact trace IDs. Disconnected ratings prevent engineers from debugging and reproducing the failure in offline evaluations.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 3 core latency metrics for streaming LLM applications?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Time-to-First-Token (TTFT), 2. Inter-Token Latency (ITL), 3. Total End-to-End Latency.")

    with st.container(border=True):
        st.markdown("**Front**: Why are completion tokens priced higher than prompt tokens?")
        st.caption("Flip for answer:")
        st.info("**Back**: Prompt tokens are computed in parallel across GPU cores during prefill; completion tokens must be generated sequentially one token at a time.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of downsampling traces in high-volume production?")
        st.caption("Flip for answer:")
        st.info("**Back**: To control storage costs and telemetry ingestion load by storing 100% of errors and low ratings, but only 5% to 10% of routine successful requests.")

    with st.container(border=True):
        st.markdown("**Front**: What is the Golden Signal triad for LLM agent health?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. p95 End-to-End Latency, 2. Tool Failure / Circuit Breaker Rate, 3. Net User Satisfaction / Feedback Ratio.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Production Monitoring in LangChain:")

c1 = st.checkbox("I understand the multi-tier architecture required for distributed LLM tracing and span aggregation.", value=True)
c2 = st.checkbox("I know why tracking p95/p99 percentiles is critical compared to deceiving average latency metrics.", value=True)
c3 = st.checkbox("I understand the distinction between prompt prefill latency (TTFT) and autoregressive decoding latency.", value=True)
c4 = st.checkbox("I can calculate dynamic token costs and attribute real-time unit economics to individual tenants.", value=True)
c5 = st.checkbox("I know how to monitor tool crash ratios and implement circuit breakers to prevent runaway agent retry loops.", value=True)
c6 = st.checkbox("I know how to link explicit user feedback (thumbs up/down) to exact trace IDs to build evaluation datasets.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Production Monitoring, real-time telemetry surveillance, and SLA engineering in LangChain.")
