import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.2 Logging - LangChain Master Curriculum",
    page_icon="📝",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class TelemetryLoggingEngine:
    """
    Enterprise Structured Telemetry Logger Engine for LangChain applications.
    Simulates structured JSON logging across the 6 core dimensions:
    1. Application Logging (Structured JSON, tenant context, correlation IDs)
    2. Debug Logging (Prompt payload formatting, parser intermediate states)
    3. Error Logging (Exception categorization, tool crashes, stack traces)
    4. Token Logging (Prompt tokens vs Completion tokens)
    5. Cost Logging (Dynamic model pricing tiers and tenant chargeback)
    6. Latency Logging (TTFT, model prefill, decoding speed)
    """
    def __init__(self, service_name: str = "customer-rag-service"):
        self.service_name = service_name
        self.log_stream: List[Dict[str, Any]] = []

    def emit_telemetry_event(
        self,
        query: str,
        model_name: str = "gpt-4o",
        log_level: str = "INFO",
        prompt_tokens: int = 450,
        completion_tokens: int = 150,
        retrieval_ms: float = 120.0,
        ttft_ms: float = 310.0,
        total_latency_ms: float = 1250.0,
        simulate_error: bool = False
    ) -> Dict[str, Any]:
        """
        Emits a structured JSON telemetry log event adhering to enterprise schema standards.
        """
        # Dynamic Cost Calculation (USD)
        # GPT-4o: 2.50 USD / 1M input, 10.00 USD / 1M output
        # GPT-4o-mini: 0.15 USD / 1M input, 0.60 USD / 1M output
        if "mini" in model_name:
            in_rate, out_rate = 0.15, 0.60
        else:
            in_rate, out_rate = 2.50, 10.00

        input_cost = (prompt_tokens / 1_000_000.0) * in_rate
        output_cost = (completion_tokens / 1_000_000.0) * out_rate
        total_cost_usd = round(input_cost + output_cost, 5)

        event_status = "ERROR" if simulate_error else "SUCCESS"
        error_details = {
            "exception_type": "ToolExecutionTimeout" if simulate_error else None,
            "error_code": "504_GATEWAY_TIMEOUT" if simulate_error else None,
            "message": "CRM database connection pool timed out after 5000ms" if simulate_error else None
        }

        log_record = {
            "timestamp": "2026-10-13T21:14:00Z",
            "service": self.service_name,
            "log_level": "ERROR" if simulate_error else log_level,
            "correlation_id": "corr-8f12a-992b",
            "tenant_id": "tenant_acme_corp_819",
            "model_metadata": {
                "model_name": model_name,
                "temperature": 0.0,
                "framework": "LangChain v0.3 LCEL"
            },
            "token_telemetry": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "cost_telemetry": {
                "input_cost_usd": round(input_cost, 5),
                "output_cost_usd": round(output_cost, 5),
                "total_cost_usd": total_cost_usd
            },
            "latency_telemetry": {
                "retrieval_ms": retrieval_ms,
                "time_to_first_token_ms": ttft_ms,
                "total_latency_ms": total_latency_ms
            },
            "execution_status": event_status,
            "error_details": error_details
        }
        self.log_stream.append(log_record)
        return log_record

# Initialize engine in session state
if "telemetry_logging_engine" not in st.session_state:
    st.session_state.telemetry_logging_engine = TelemetryLoggingEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("23.2 Logging in LangChain")
st.caption("Chapter 23: Callbacks, Logging, and Monitoring | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In generative AI engineering, traditional plaintext log messages (`print("Done!")`) are completely inadequate. 
LLM applications require **Structured Telemetry Logging**: emitting machine-readable JSON payloads containing 
token breakdowns, per-request financial cost calculations, percentile latencies (TTFT vs generation time), 
trace correlation IDs, and categorized exception traces.

In this lesson, you will master:
1. Application logging: structured JSON logging, tenant context, and distributed trace IDs
2. Debug logging: inspecting prompt template variable binding and intermediate LCEL states
3. Error logging: capturing stack traces and eliminating silent failure traps
4. Token logging: isolating prompt tokens vs completion tokens to prevent context bloat
5. Cost and latency logging: computing real-time unit economics and waterfall timing metrics
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Telemetry Logging and Observability Pipeline Architecture", divider="rainbow")

st.markdown("""
A production logging pipeline decouples telemetry collection from application business logic. 
Callback handlers emit structured JSON events to standard output (stdout), where log forwarders 
(FluentBit, Vector, or AWS CloudWatch) ship them to analytical data stores (Datadog, OpenSearch, Snowflake).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LoggingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_app {
            label = "1. LangChain LCEL Application Layer";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            CHAIN [label="LCEL Pipeline Execution\\n(prompt | model | parser)", fillcolor="#1E3A8A", color="#3B82F6"];
            HANDLER [label="StructuredTelemetryHandler\\n(Extends BaseCallbackHandler)", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_payload {
            label = "2. Structured JSON Telemetry Record";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F59E0B";
            fontsize = 12;

            LOG_JSON [label="JSON Log Event\\n{'trace_id': '8f12a', 'tokens': 600, 'cost_usd': 0.0026, 'ttft_ms': 310}", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_forwarding {
            label = "3. Enterprise Log Ingestion & Forwarding";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            STDOUT [label="stdout Stream\\n(12-Factor App Standard)", fillcolor="#334155", color="#64748B"];
            AGENT [label="Log Forwarder (Vector / FluentBit)\\n(Batches & Routes to Cloud Sinks)", fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_analytics {
            label = "4. Analytics & Billing Sinks";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            DOG [label="Datadog / CloudWatch\\n(p95 Latency & Error Alarms)", fillcolor="#065F46", color="#059669"];
            SNOW [label="Snowflake / BigQuery\\n(Per-Tenant Cost Accounting)", fillcolor="#065F46", color="#059669"];
        }

        CHAIN -> HANDLER [label="Triggers Lifecycle Hooks"];
        HANDLER -> LOG_JSON [label="Synthesizes Payload"];
        LOG_JSON -> STDOUT [label="Emits JSON Line"];
        STDOUT -> AGENT [label="Scrapes Container Logs"];
        AGENT -> DOG [label="Metrics Stream"];
        AGENT -> SNOW [label="Billing Events"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Logging Rigor
1. **Never Emit Unstructured String Messages**: Logging `print(f"Tokens: {tokens}")` creates unsearchable noise. Logs must always be serialized as structured single-line JSON objects with consistent key names across all microservices.
2. **Every Log Line Must Include a Correlation Trace ID**: Without a shared `correlation_id` across log records, diagnosing an incident that spanned a frontend gateway, retriever service, and LLM call is virtually impossible.
3. **Log Tokens and Costs at the Source**: Cloud LLM API bills arrive as aggregate monthly totals. To calculate customer unit economics or detect runaway loops, token usage and dollar costs must be logged on every individual request.
""")

st.divider()

# ==============================================================================
# SECTION 2: APPLICATION LOGGING
# ==============================================================================
st.header("2. Application Logging (Structured JSON Formats, Contextual Metadata and Trace IDs)", divider="rainbow")

st.markdown("""
Application logging records high-level operational events: request submission, authentication, 
tenant context, and workflow completion. In multi-tenant enterprise architectures, every log event 
must carry contextual metadata.
""")

st.markdown("""
### Mandatory Application Log Fields
* **`timestamp`**: ISO-8601 UTC timestamp (`2026-10-13T21:14:00Z`).
* **`correlation_id`**: Unique request UUID propagated across HTTP headers (`X-Correlation-ID`).
* **`tenant_id`**: Customer account identifier for multi-tenant data partitioning.
* **`service_name`**: Component name (`customer-rag-backend`).
* **`log_level`**: Standard severity (`INFO`, `WARN`, `ERROR`).
""")

st.code('''
import json
from langchain_core.callbacks import BaseCallbackHandler

class StructuredJSONLoggingHandler(BaseCallbackHandler):
    def __init__(self, service_name: str, correlation_id: str, tenant_id: str):
        self.service_name = service_name
        self.correlation_id = correlation_id
        self.tenant_id = tenant_id

    def on_chain_end(self, outputs: dict, **kwargs):
        log_entry = {
            "service": self.service_name,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "event": "CHAIN_EXECUTION_COMPLETED",
            "status": "SUCCESS"
        }
        # Emits pure JSON string to stdout
        print(json.dumps(log_entry))
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Log forwarders like Vector parse the JSON string, extract `tenant_id` as an indexed facet, 
and allow engineering teams to filter millions of logs by customer in milliseconds.
""")

st.divider()

# ==============================================================================
# SECTION 3: DEBUG LOGGING
# ==============================================================================
st.header("3. Debug Logging (Verbose Model Prefills, Raw Prompt Payloads and Intermediate Chains)", divider="rainbow")

st.markdown("""
When developing complex prompt chains, unexpected outputs often stem from subtle template formatting errors 
or variable binding bugs. Debug logging records the exact rendered prompt strings before they are dispatched to the model.
""")

st.markdown("""
### What Belongs in Debug Logging
* **Full Rendered Prompts**: The complete system instructions and user message with all variables interpolated.
* **Retrieved Chunk Text**: The exact document snippets passed into the context window.
* **Parser Intermediate States**: The raw string output from the LLM before JSON parsing or Pydantic validation occurs.
* **Environment Gate**: Debug logging must be disabled in production (`LOG_LEVEL=INFO`) to prevent log bloat and protect sensitive user data.
""")

st.code('''
# Inspecting intermediate prompt payloads via LangChain debug mode
from langchain.globals import set_debug

# Enables global verbose trace logging to console (for local development ONLY)
set_debug(True)

# LCEL Pipeline will now print every prompt formatting step, raw tokens, and parser conversions
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production, teams use dynamic log-level switching. If a tenant reports a reproducible bug, 
engineers enable `DEBUG` logging specifically for that single tenant's requests without impacting global cluster volume.
""")

st.divider()

# ==============================================================================
# SECTION 4: ERROR LOGGING
# ==============================================================================
st.header("4. Error Logging (Exception Categorization, Stack Traces and Swallowed Failure Traps)", divider="rainbow")

st.markdown("""
LLM applications fail across multiple distributed layers: provider rate limits (`HTTP 429`), 
output parser JSON syntax errors, vector database timeouts, and schema validation crashes. 
Error logging must categorize exceptions and capture full stack traces.
""")

st.markdown("""
### The Swallowed Error Trap
A dangerous anti-pattern in generative AI is silently catching errors to keep the application running:
```python
try:
    return crm_tool.invoke(query)
except Exception:
    return "Error occurred" # SWALLOWED! LLM will now hallucinate!
```
When exceptions are swallowed, the LLM hallucinates an answer, the application logs `200 OK`, 
and engineering remains completely blind to a systemic database outage.
""")

error_taxonomy_df = pd.DataFrame([
    {"Error Category": "Provider Rate Limit (HTTP 429)", "Root Cause": "Exceeded tokens-per-minute (TPM) quota", "Action": "Log as WARN; retry with exponential backoff and jitter"},
    {"Error Category": "OutputParserException", "Root Cause": "Model generated invalid JSON or conversational filler", "Action": "Log as ERROR; trigger automatic retry prompt with schema repair"},
    {"Error Category": "ToolExecutionTimeout (HTTP 504)", "Root Cause": "Internal SQL database connection timed out", "Action": "Log as CRITICAL; trip circuit breaker and alert on-call"},
    {"Error Category": "ContextLengthExceeded", "Root Cause": "Prompt plus retrieval chunks exceeded model context window", "Action": "Log as ERROR; enable contextual compression chunking"}
])
st.dataframe(error_taxonomy_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Callback handlers listen to `on_chain_error` and `on_tool_error`, capturing the full Python exception object 
and emitting structured alert payloads directly to Sentry or PagerDuty.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOKEN LOGGING
# ==============================================================================
st.header("5. Token Logging (Prompt Tokens, Completion Tokens and Context Inflation Auditing)", divider="rainbow")

st.markdown("""
Tokens are the fundamental currency of generative AI. 
Logging only total token count is insufficient; applications must log **Prompt Tokens** (input) and **Completion Tokens** (output) separately.
""")

st.markdown("""
### Why Separate Token Accounting is Mandatory
* **Asymmetric Pricing**: Completion tokens cost 3x to 4x more than prompt tokens across major model providers (e.g., GPT-4o costs 2.50 USD / 1M input tokens vs 10.00 USD / 1M output tokens).
* **Context Inflation Auditing**: Tracking prompt tokens across release sprints reveals "prompt creep"—where developers gradually add context until prompt costs double unnoticed.
* **Verbosity Spikes**: Tracking completion tokens reveals when prompt changes make the model excessively verbose, inflating latency and bills.
""")

st.code('''
# LangChain Token Usage Callback Pattern
class TokenAuditLoggingHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        print(f"[TOKEN AUDIT] Input: {prompt_tokens} | Output: {completion_tokens} | Total: {total_tokens}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Token counts logged in JSON format are ingested into Prometheus and graphed on Grafana dashboards, 
showing hourly token consumption velocity and alerting when usage exceeds historical norms.
""")

st.divider()

# ==============================================================================
# SECTION 6: COST LOGGING
# ==============================================================================
st.header("6. Cost Logging (Dynamic Pricing Calculators, Tenant Attribution and Budget Caps)", divider="rainbow")

st.markdown("""
Cost logging translates raw token counts into real dollars and cents at the point of request execution. 
Logging cost on every request enables accurate financial forecasting and customer chargeback models.
""")

st.markdown("""
### Dynamic Pricing Formulas
* `Input Cost = (Prompt Tokens / 1,000,000) * Input Rate per 1M`
* `Output Cost = (Completion Tokens / 1,000,000) * Output Rate per 1M`
* `Total Cost = Input Cost + Output Cost`
""")

cost_formula_table = pd.DataFrame([
    {"Model Name": "GPT-4o (Standard)", "Input Rate / 1M": "2.50 USD", "Output Rate / 1M": "10.00 USD", "Sample Call (500 in / 150 out)": "0.00275 USD"},
    {"Model Name": "GPT-4o-mini", "Input Rate / 1M": "0.15 USD", "Output Rate / 1M": "0.60 USD", "Sample Call (500 in / 150 out)": "0.00016 USD"},
    {"Model Name": "Claude 3.5 Sonnet", "Input Rate / 1M": "3.00 USD", "Output Rate / 1M": "15.00 USD", "Sample Call (500 in / 150 out)": "0.00375 USD"}
])
st.dataframe(cost_formula_table, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Finance teams query data warehouses for `SUM(total_cost_usd) GROUP BY tenant_id`. 
This allows SaaS providers to identify unprofitable enterprise accounts and enforce fair-use billing policies.
""")

st.divider()

# ==============================================================================
# SECTION 7: LATENCY LOGGING
# ==============================================================================
st.header("7. Latency Logging (Waterfall Breakdown: TTFT, Token Generation Speed and I/O Overhead)", divider="rainbow")

st.markdown("""
A single request latency metric (e.g., *"total request took 2.4 seconds"*) tells engineers nothing about where time was spent. 
Production latency logging breaks total execution time into its constituent waterfall stages:
""")

st.markdown("""
### The Latency Breakdown Breakdown
1. **Retrieval Latency**: Vector database embedding calculation and k-NN search time (`Target < 150ms`).
2. **Time-to-First-Token (TTFT)**: Time elapsed until the first token streams from the LLM provider (`Target < 400ms`).
3. **Inter-Token Latency (Decoding Time)**: Time spent sequentially generating subsequent tokens (`Target 30-60 tokens/sec`).
4. **Output Parsing Latency**: Pydantic schema deserialization time (`Target < 10ms`).
""")

st.code('''
import time

class LatencyWaterfallLogger(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.t_start = time.time()
        self.ttft_recorded = False

    def on_llm_new_token(self, token, **kwargs):
        if not self.ttft_recorded:
            self.ttft_ms = (time.time() - self.t_start) * 1000
            self.ttft_recorded = True

    def on_chain_end(self, outputs, **kwargs):
        total_ms = (time.time() - self.t_start) * 1000
        print(f"[LATENCY] TTFT: {self.ttft_ms:.1f}ms | Total: {total_ms:.1f}ms")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
APM tools (Datadog, OpenTelemetry) visualize these log timestamps as interactive distributed trace spans, 
allowing on-call engineers to spot database query slowdowns instantly.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Structured Telemetry Logger Engine (TelemetryLoggingEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the Telemetry Logger Engine. 
It captures tokens, calculates dynamic model pricing, records multi-phase latency metrics, 
and serializes structured JSON telemetry records.
""")

st.code('''
class TelemetryLoggingEngine:
    """
    Enterprise Structured Telemetry Logger Engine for LangChain applications.
    Simulates structured JSON logging across the 6 core dimensions:
    1. Application Logging (Structured JSON, tenant context, correlation IDs)
    2. Debug Logging (Prompt payload formatting, parser intermediate states)
    3. Error Logging (Exception categorization, tool crashes, stack traces)
    4. Token Logging (Prompt tokens vs Completion tokens)
    5. Cost Logging (Dynamic model pricing tiers and tenant chargeback)
    6. Latency Logging (TTFT, model prefill, decoding speed)
    """
    def __init__(self, service_name: str = "customer-rag-service"):
        self.service_name = service_name
        self.log_stream: List[Dict[str, Any]] = []

    def emit_telemetry_event(
        self,
        query: str,
        model_name: str = "gpt-4o",
        log_level: str = "INFO",
        prompt_tokens: int = 450,
        completion_tokens: int = 150,
        retrieval_ms: float = 120.0,
        ttft_ms: float = 310.0,
        total_latency_ms: float = 1250.0,
        simulate_error: bool = False
    ) -> Dict[str, Any]:
        """
        Emits a structured JSON telemetry log event adhering to enterprise schema standards.
        """
        # Dynamic Cost Calculation (USD)
        if "mini" in model_name:
            in_rate, out_rate = 0.15, 0.60
        else:
            in_rate, out_rate = 2.50, 10.00

        input_cost = (prompt_tokens / 1_000_000.0) * in_rate
        output_cost = (completion_tokens / 1_000_000.0) * out_rate
        total_cost_usd = round(input_cost + output_cost, 5)

        event_status = "ERROR" if simulate_error else "SUCCESS"
        error_details = {
            "exception_type": "ToolExecutionTimeout" if simulate_error else None,
            "error_code": "504_GATEWAY_TIMEOUT" if simulate_error else None,
            "message": "CRM database connection pool timed out after 5000ms" if simulate_error else None
        }

        log_record = {
            "timestamp": "2026-10-13T21:14:00Z",
            "service": self.service_name,
            "log_level": "ERROR" if simulate_error else log_level,
            "correlation_id": "corr-8f12a-992b",
            "tenant_id": "tenant_acme_corp_819",
            "model_metadata": {
                "model_name": model_name,
                "temperature": 0.0,
                "framework": "LangChain v0.3 LCEL"
            },
            "token_telemetry": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "cost_telemetry": {
                "input_cost_usd": round(input_cost, 5),
                "output_cost_usd": round(output_cost, 5),
                "total_cost_usd": total_cost_usd
            },
            "latency_telemetry": {
                "retrieval_ms": retrieval_ms,
                "time_to_first_token_ms": ttft_ms,
                "total_latency_ms": total_latency_ms
            },
            "execution_status": event_status,
            "error_details": error_details
        }
        self.log_stream.append(log_record)
        return log_record
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Telemetry Logger Cockpit and Multi-Metric Cost Sandbox", divider="rainbow")

st.markdown("""
Simulate production request executions and inspect the resulting structured JSON telemetry payload. 
Experiment with model pricing tiers, token volumes, latency watermarks, and simulated tool errors.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Telemetry Emission Controls")
    model_choice = st.selectbox("LLM Provider Endpoint", ["gpt-4o", "gpt-4o-mini"], index=0)
    p_tokens = st.slider("Prompt Tokens (Input)", min_value=100, max_value=4000, value=650, step=50)
    c_tokens = st.slider("Completion Tokens (Output)", min_value=50, max_value=1500, value=220, step=10)
    ret_time = st.slider("Vector Retrieval Latency (ms)", min_value=20, max_value=500, value=110, step=10)
    ttft_time = st.slider("Time-to-First-Token (ms)", min_value=100, max_value=1500, value=340, step=20)
    sim_err_toggle = st.toggle("Simulate Upstream Tool Crash (504 Timeout)", value=False)

with wb_col2:
    st.markdown("### Emitted Structured JSON Telemetry Event")
    log_event = st.session_state.telemetry_logging_engine.emit_telemetry_event(
        query="Customer inquiry regarding invoice 8821",
        model_name=model_choice,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        retrieval_ms=float(ret_time),
        ttft_ms=float(ttft_time),
        total_latency_ms=float(ret_time + ttft_time + (c_tokens * 8)),
        simulate_error=sim_err_toggle
    )

    if log_event["execution_status"] == "SUCCESS":
        st.success(f"**STATUS: 200 OK ({log_event['execution_status']})** — Log Record emitted to stdout.")
    else:
        st.error(f"**STATUS: {log_event['error_details']['error_code']}** — Captured structured error span.")

    # High-level metric badges
    c_m = log_event["cost_telemetry"]
    l_m = log_event["latency_telemetry"]
    t_m = log_event["token_telemetry"]

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Total Tokens", f"{t_m['total_tokens']:,}")
    with b2:
        st.metric("Call Cost", f"{c_m['total_cost_usd']:.5f} USD")
    with b3:
        st.metric("TTFT Latency", f"{l_m['time_to_first_token_ms']:.0f} ms")

    # Raw JSON Output Viewer
    st.markdown("#### Serialized JSON Log Payload")
    st.json(log_event)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

with st.expander("Q1: Why is structured JSON logging mandatory over plaintext logging in production LLMOps?"):
    st.markdown("""
    **Answer**: Plaintext logs (`logger.info("Done generating")`) cannot be efficiently parsed, aggregated, or queried 
    by log analytical backends (OpenSearch, Datadog). Structured JSON logs serialize every metric as a typed key-value pair 
    (`"prompt_tokens": 450`, `"cost_usd": 0.0025`, `"tenant_id": "cust_82"`). This enables sub-second faceted queries, 
    automated alerting on p95 latency spikes, and exact per-tenant cost attribution.
    """)

with st.expander("Q2: Why must prompt tokens and completion tokens be logged separately?"):
    st.markdown("""
    **Answer**: Prompt tokens and completion tokens have asymmetric pricing: major providers price completion tokens 3x-4x 
    higher than prompt tokens because completion tokens require sequential autoregressive decoding, whereas prompt tokens 
    are processed in parallel during prefill. Furthermore, separate tracking allows engineers to differentiate between 
    context inflation (prompt bloat) and model verbosity spikes.
    """)

with st.expander("Q3: What is the 'Swallowed Exception Trap' in LLM error handling?"):
    st.markdown("""
    **Answer**: The swallowed exception trap occurs when code catches a tool or database crash and returns a generic string 
    (`"Tool failed"`) to prevent an HTTP 500 error. The downstream LLM observes the error string and attempts to guess the answer, 
    hallucinating plausible facts out of training data. The application returns HTTP 200 OK, completely hiding the critical tool failure.
    """)

with st.expander("Q4: What is the role of a correlation trace ID in distributed LLM logging?"):
    st.markdown("""
    **Answer**: A correlation trace ID is a unique UUID passed across microservices via HTTP headers (`X-Correlation-ID`). 
    In modern AI applications, a single user request traverses an API gateway, a vector retrieval service, and an LLM engine. 
    Logging the same correlation ID on every log line allows engineers to reconstruct the complete end-to-end execution path 
    during an incident post-mortem.
    """)

with st.expander("Q5: How do you log Time-to-First-Token (TTFT) when using streaming chat models?"):
    st.markdown("""
    **Answer**: TTFT is captured using the `on_llm_new_token` callback. When the chain starts, a high-resolution timestamp 
    is recorded. When the callback fires for the very first token chunk, the elapsed duration is computed (`t_first - t_start`) 
    and recorded as TTFT before setting a boolean flag to ignore subsequent tokens.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_lt1, col_lt2 = st.columns(2)
with col_lt1:
    st.error("**Trap 1 (Logging Raw PII to Unencrypted Log Streams)**: Dumping unredacted customer prompts into stdout logs. If prompts contain credit cards, SSNs, or emails, plain logs violate GDPR, HIPAA, and SOC-2 compliance. Always scrub PII before log emission.")
    st.warning("**Trap 2 (Debug Logging Enabled in Production)**: Leaving `set_debug(True)` on in production clusters. Printing massive 10,000-token prompt strings to container stdout logs causes disk exhaustion and inflates cloud logging bills.")
with col_lt2:
    st.error("**Trap 3 (Single Total Latency Metric)**: Only logging total request latency. If a request takes 5 seconds, engineers cannot tell whether the delay was in the vector database, model prefill, or token generation.")
    st.warning("**Trap 4 (Missing Log Schema Versioning)**: Modifying JSON log key names arbitrarily (`tokens` to `total_tokens`). Downstream ingestion pipelines and billing dashboards break when log field schemas change without deprecation notices.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the 12-Factor App standard for application logging?")
        st.caption("Flip for answer:")
        st.info("**Back**: Applications should never route or manage log files directly; they should emit structured event streams to stdout, letting log forwarders handle routing.")

    with st.container(border=True):
        st.markdown("**Front**: Why are completion tokens priced higher than prompt tokens?")
        st.caption("Flip for answer:")
        st.info("**Back**: Prompt tokens are processed in parallel during prefill; completion tokens must be generated sequentially one by one (autoregressively), consuming more GPU compute.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What are the 4 key stages of LLM latency logging?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Retrieval Latency, 2. Time-to-First-Token (TTFT), 3. Inter-Token Generation Speed, 4. Output Parsing Latency.")

    with st.container(border=True):
        st.markdown("**Front**: What is the primary operational consequence of swallowing tool exceptions?")
        st.caption("Flip for answer:")
        st.info("**Back**: The LLM fabricates answers from memory, transforming an infrastructure timeout into a silent, customer-facing factual hallucination.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Telemetry Logging in LangChain:")

c1 = st.checkbox("I understand why structured JSON logging is required over plaintext print statements.", value=True)
c2 = st.checkbox("I know how to inject correlation trace IDs and tenant metadata into every log record.", value=True)
c3 = st.checkbox("I understand the danger of swallowing exceptions and know how to capture structured error logs.", value=True)
c4 = st.checkbox("I can log prompt tokens and completion tokens separately to track asymmetric model costs.", value=True)
c5 = st.checkbox("I know how to calculate real-time call costs dynamically using model pricing tiers.", value=True)
c6 = st.checkbox("I can break down total request latency into retrieval time, TTFT, and generation speed.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Telemetry Logging, structured JSON observability, and cost-latency profiling in LangChain.")
