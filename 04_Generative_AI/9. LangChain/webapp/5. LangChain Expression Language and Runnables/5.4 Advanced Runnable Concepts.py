import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.4 Advanced Runnable Concepts - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("5.4 Advanced Runnable Concepts: Configuration, Observability & Fault Tolerance")
st.caption("Chapter 5: LangChain Expression Language and Runnables | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In mission-critical enterprise environments, an AI pipeline cannot simply be a happy-path chain. 
It must be **fault-tolerant**, **fully observable**, **rate-limit resilient**, and capable of 
**zero-downtime failover**.

Through advanced Runnable extensions in LangChain v0.3+, developers harden LCEL pipelines using 
**`RunnableConfig` context propagation**, **custom lifecycle callbacks**, **threadpool concurrency throttles**, 
**automated exponential retries (`with_retry`)**, and **seamless secondary provider fallbacks (`with_fallbacks`)**.

In this lesson, you will master:
- **The `RunnableConfig` Protocol**: Propagating metadata, session IDs, and tags across deeply nested DAGs.
- **Enterprise Observability**: Hooking custom event callbacks to track token consumption and intermediate states.
- **Concurrency Governance**: Throttling parallel worker pools with `max_concurrency` to prevent provider rate limits.
- **Fault-Tolerant Retries**: Configuring exponential backoff policies with jitter for transient errors.
- **Automated Multi-Cloud Failover**: Building zero-downtime resilient pipelines via `.with_fallbacks()`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockFaultToleranceEngine:
    """Simulates retries, fallback switching, and callback telemetry in module scope."""
    def __init__(self, primary_failure_rate: float, max_retries: int, enable_fallback: bool):
        self.primary_failure_rate = primary_failure_rate
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

    def execute_resilient_pipeline(self, query: str) -> Dict[str, Any]:
        logs = []
        logs.append(f"[START] Initiating execution for query: '{query[:30]}...'")
        
        # Primary Model Simulation
        primary_success = np.random.random() > self.primary_failure_rate
        retries_used = 0
        
        if not primary_success:
            logs.append("[WARNING] Primary Model (OpenAI GPT-4o) returned HTTP 500 Internal Server Error.")
            while retries_used < self.max_retries:
                retries_used += 1
                logs.append(f"[RETRY {retries_used}] Retrying Primary Model with exponential backoff...")
                if np.random.random() > self.primary_failure_rate:
                    primary_success = True
                    logs.append("[SUCCESS] Primary Model recovered on retry attempt.")
                    break

        if primary_success:
            return {
                "status": "SUCCESS_PRIMARY",
                "resolved_by": "Primary Model (OpenAI GPT-4o)",
                "retries_used": retries_used,
                "fallback_triggered": False,
                "latency_ms": 320 + (retries_used * 250),
                "execution_logs": logs,
                "output": f"Analysis complete for '{query}'. Synthesized via Primary Engine."
            }
        
        # Fallback Triggered
        if self.enable_fallback:
            logs.append("[FAILOVER] Primary Model exhausted retry budget. Triggering Fallback Model (Anthropic Claude 3.5 Sonnet)...")
            logs.append("[SUCCESS] Fallback Model returned HTTP 200 OK.")
            return {
                "status": "SUCCESS_FALLBACK",
                "resolved_by": "Fallback Model (Anthropic Claude 3.5 Sonnet)",
                "retries_used": retries_used,
                "fallback_triggered": True,
                "latency_ms": 320 + (retries_used * 250) + 420,
                "execution_logs": logs,
                "output": f"Analysis complete for '{query}'. Synthesized via Fallback Engine after primary outage."
            }
        else:
            logs.append("[FATAL] Primary Model failed and no fallback was configured. Pipeline crashed with APIConnectionError.")
            return {
                "status": "FAILED",
                "resolved_by": "None (Pipeline Crashed)",
                "retries_used": retries_used,
                "fallback_triggered": False,
                "latency_ms": 320 + (retries_used * 250),
                "execution_logs": logs,
                "output": "Error: Pipeline execution halted due to unhandled provider outage."
            }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Advanced Runnable Configuration, Observability & Fault-Tolerance Lifecycle", divider="rainbow")

st.markdown("""
The diagram below illustrates how configuration, callbacks, retries, and fallbacks wrap 
around an LCEL pipeline to ensure enterprise reliability and observability.
""")

dot_blueprint = """
digraph AdvancedRunnableTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_config {
        label="1. Configuration & Context Injection (RunnableConfig)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_config_obj [label="RunnableConfig Dictionary\n- tags: ['prod', 'audit']\n- metadata: {'user_id': '104'}\n- callbacks: [TracerHandler()]\n- max_concurrency: 4", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_resilience {
        label="2. Fault-Tolerance Envelope";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_primary [label="Primary LCEL Pipeline (.with_retry)\n(OpenAI GPT-4o Chain)\nAuto-retries on 429 / 500 errors", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_fallback [label="Secondary Fallback Pipeline (.with_fallbacks)\n(Anthropic Claude 3.5 Sonnet Chain)\nActivated on primary exhaustion", fillcolor="#1E293B", fontcolor="#F59E0B"];

        node_primary -> node_fallback [label="intercepts exception\ntransparent failover", style="dashed", color="#EF4444"];
    }

    subgraph cluster_observability {
        label="3. Distributed Telemetry & Callback Hooks";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_tracer [label="LangSmith / Custom Callbacks\non_chain_start, on_llm_end, on_chain_error\nPropagates latency, tokens, cost", fillcolor="#1E293B", fontcolor="#34D399"];
    }

    node_config_obj -> node_primary [label="passed to .invoke()", color="#38BDF8"];
    node_primary -> node_tracer [label="emits events", color="#34D399"];
    node_fallback -> node_tracer [label="emits events", color="#34D399"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Advanced Runnables encapsulate the core business logic inside 
> a **fault-tolerant resilience envelope**. Notice how `RunnableConfig` injects context downward 
> while the callback system extracts telemetry upward, and `.with_fallbacks()` protects user space 
> from upstream infrastructure outages.
""")

# --- SECTION 2: RUNNABLE CONFIGURATION ---
st.header("2. Runnable Configuration: The RunnableConfig Specification & Thread-Local Propagation", divider="rainbow")

st.markdown("""
The `RunnableConfig` object is a standardized typed dictionary passed into any Runnable execution method. 
LangChain automatically propagates this dictionary across all parent and child Runnables in the execution DAG.

### Canonical `RunnableConfig` Keys
- **`tags`** (`List[str]`): Categorical labels used for filtering traces in LangSmith.
- **`metadata`** (`Dict[str, Any]`): Arbitrary key-value metadata attached to the execution span.
- **`callbacks`** (`List[BaseCallbackHandler]`): Custom callback handlers listening to lifecycle events.
- **`max_concurrency`** (`int`): Maximum number of parallel threads allowed during `.batch()` and `RunnableParallel`.
- **`recursion_limit`** (`int`): Maximum execution depth for cyclic agents (preventing infinite loops).
""")

st.code('''from langchain_core.runnables import RunnableConfig

# Defining production execution config
config: RunnableConfig = {
    "tags": ["cloud-audit", "release-2026.1"],
    "metadata": {
        "tenant_id": "acme-corp",
        "user_session": "sess-8921",
        "environment": "production"
    },
    "max_concurrency": 4,
    "recursion_limit": 25
}

# Invocations automatically forward this config down the entire DAG:
# response = chain.invoke({"query": "Scan VPC"}, config=config)
''', language="python")

# --- SECTION 3: RUNTIME CONFIGURATION ---
st.header("3. Runtime Configuration: Per-Request Dynamic Overrides & Parameter Swapping", divider="rainbow")

st.markdown("""
Beyond metadata, `RunnableConfig` supports **configurable fields**, allowing applications to dynamically 
swap model checkpoints, temperatures, or prompt parameters per request without rebuilding the pipeline DAG.
""")

st.code('''from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

# Declare configurable chat model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).configurable_fields(
    temperature=ConfigurableField(id="runtime_temp", name="Temperature Override")
)

# Calling with dynamic runtime temperature override:
res = model.invoke(
    "Generate code",
    config={"configurable": {"runtime_temp": 0.0}}
)
''', language="python")

# --- SECTION 4: CALLBACKS ---
st.header("4. Callbacks: Custom Event Handlers, Token Counting & Streaming Interception", divider="rainbow")

st.markdown("""
Callbacks allow developers to hook into the internal lifecycle of any Runnable. 
By subclassing **`BaseCallbackHandler`**, you can listen to events such as:
- `on_chain_start`: Triggered when an LCEL sequence begins execution.
- `on_llm_start`: Triggered when the Chat Model begins sending API requests.
- `on_llm_new_token`: Triggered every time a new streaming token chunk is generated.
- `on_llm_end`: Triggered when generation completes (exposing token counts).
- `on_chain_error`: Triggered if any component throws an exception.
""")

st.code('''from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List

class AuditLoggingCallbackHandler(BaseCallbackHandler):
    """Custom callback logging execution events to corporate telemetry."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"[AUDIT] Dispatched request to model: {serialized.get('name', 'ChatModel')}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        tokens = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        print(f"[AUDIT] Model completed. Token consumption: {tokens}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"[ALERT] Pipeline execution failed: {str(error)}")

# Passing callback handler into chain invocation:
# handler = AuditLoggingCallbackHandler()
# chain.invoke({"input": "Test"}, config={"callbacks": [handler]})
''', language="python")

# --- SECTION 5: METADATA ---
st.header("5. Metadata: Session IDs, User Tracking & Observability Trace Attributes", divider="rainbow")

st.markdown("""
Metadata attached in `RunnableConfig` is preserved across all distributed spans. 
In observability platforms like **LangSmith**, this metadata enables instant filtering 
by user ID, organizational department, or prompt version.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Audit {target}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Invocations with rich metadata attributes
result = chain.invoke(
    {"target": "s3://finance-archive"},
    config={
        "metadata": {
            "department": "SecOps",
            "compliance_framework": "PCI-DSS",
            "audit_run_id": "audit-2026-03-15-uuid4"
        }
    }
)
''', language="python")

# --- SECTION 6: TAGS ---
st.header("6. Tags: Categorical Tagging, Production Filtering & LangSmith Organization", divider="rainbow")

st.markdown("""
Tags are simple string labels attached to Runnables. They can be declared statically on component definition 
or dynamically during runtime invocation.

In production, tags allow developers to partition telemetry:
- Distinguish `"test"` runs from `"production"` traffic.
- Isolate experimental A/B testing prompt variants (`"prompt-v2-experiment"`).
- Filter traces generated by background automated workers (`"background-worker"`).
""")

st.code('''# Static Tagging on Component Creation:
model = ChatOpenAI(model="gpt-4o", tags=["frontier-model", "production"])

# Dynamic Tagging on Invocation:
# response = chain.invoke({"input": "Hello"}, config={"tags": ["ab-test-cohort-b"]})
''', language="python")

# --- SECTION 7: CONCURRENCY ---
st.header("7. Concurrency: Controlling Thread Pools, Rate Limits & max_concurrency Budgets", divider="rainbow")

st.markdown("""
When processing batches via `.batch()` or evaluating parallel branches via `RunnableParallel`, 
uncontrolled concurrency will trigger provider rate limit errors (HTTP 429).

Setting **`max_concurrency`** inside `RunnableConfig` bounds the internal `ThreadPoolExecutor` 
to a safe number of concurrent requests.
""")

st.code('''from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
items = [{"doc": f"Document {i}"} for i in range(100)]

# Enforce a maximum of 4 concurrent worker threads to protect API rate limits:
# results = chain.batch(items, config={"max_concurrency": 4})
''', language="python")

# --- SECTION 8: ERROR HANDLING ---
st.header("8. Error Handling: Intercepting Failures, Exception Handling Runnables & Graceful Degradation", divider="rainbow")

st.markdown("""
In enterprise microservices, an unhandled exception inside an LLM pipeline should not crash the entire web server. 
LangChain provides **`.with_exception_handling()`** and custom error boundary Runnables to return graceful fallback payloads.
""")

st.code('''from langchain_core.runnables import RunnableLambda

def safe_math_evaluator(inputs: dict) -> str:
    try:
        val = int(inputs["number"])
        return f"Calculated: {100 / val}"
    except ZeroDivisionError:
        return "ERROR: Cannot divide by zero. Defaulting to safe fallback: 0.0"

# Wrapping in safe Runnable
resilient_math = RunnableLambda(safe_math_evaluator)
print(resilient_math.invoke({"number": "0"}))
''', language="python")

# --- SECTION 9: RETRIES ---
st.header("9. Retries: Automated Retries (with_retry), Exponential Backoff & Jitter Policies", divider="rainbow")

st.markdown("""
Frontier APIs frequently throw transient errors (network drops, HTTP 429, HTTP 503). 
Rather than writing manual `try/except` loops, LangChain provides **`.with_retry()`**.

`.with_retry()` automatically re-executes the Runnable using **exponential backoff with jitter**, 
intercepting specified exception classes before failing.
""")

st.code('''from langchain_openai import ChatOpenAI
import httpx

base_model = ChatOpenAI(model="gpt-4o", timeout=10.0)

# Wrap model with automated retry policy
resilient_model = base_model.with_retry(
    stop_after_attempt=3,                     # Retry up to 3 times
    wait_exponential_jitter=True,             # Randomized exponential backoff
    retry_if_exception_type=(httpx.TimeoutException, httpx.HTTPStatusError)
)

# Invocations automatically retry on transient failures:
# res = resilient_model.invoke("Audit cluster health")
''', language="python")

# --- SECTION 10: FALLBACKS ---
st.header("10. Fallbacks: Resilience Architecture, Multi-Provider Failover & .with_fallbacks()", divider="rainbow")

st.markdown("""
Retries solve transient network hiccups, but they cannot save your application if the primary provider 
experiences a prolonged multi-hour outage.

**`.with_fallbacks()`** allows you to attach a list of secondary and tertiary fallback Runnables. 
If the primary pipeline raises an unrecoverable exception, LangChain automatically intercepts it 
and transparently reroutes the input to the secondary provider with zero user downtime.
""")

dot_fallback = """
digraph FallbackRouting {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_in [label="Incoming Request\n{'query': 'Analyze enterprise architecture'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_primary [label="Primary Model Pipeline\nOpenAI ChatOpenAI (gpt-4o)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_error [label="Outage / Exception Raised\n(HTTP 500 / 429 / Timeout)", fillcolor="#1E293B", fontcolor="#EF4444"];
    node_fallback [label="Fallback Model Pipeline (.with_fallbacks)\nAnthropic ChatAnthropic (claude-3-5-sonnet)", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_success [label="Successful Response Emitted\nZero application downtime", fillcolor="#1E293B", fontcolor="#34D399"];

    node_in -> node_primary;
    node_primary -> node_error [label="fails on outage", color="#EF4444"];
    node_error -> node_fallback [label="with_fallbacks intercept", color="#F59E0B"];
    node_fallback -> node_success [color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: Zero-Downtime Multi-Provider Fallback Lifecycle", expanded=True):
    st.graphviz_chart(dot_fallback, use_container_width=True)

st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableWithFallbacks

# Primary model: OpenAI GPT-4o
primary_model = ChatOpenAI(model="gpt-4o", timeout=10.0, max_retries=1)

# Secondary fallback: Anthropic Claude 3.5 Sonnet
secondary_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=12.0)

# Resilient composite chain
resilient_chain: RunnableWithFallbacks = primary_model.with_fallbacks([secondary_model])

# If OpenAI fails, execution instantly switches to Claude:
response = resilient_chain.invoke("Summarize cloud security architecture.")
print(response.content)
''', language="python")

# --- SECTION 11: PRODUCTION IMPLEMENTATION ---
st.header("11. Production Implementation: Enterprise Fault-Tolerant Observability Pipeline", divider="rainbow")

st.markdown("""
Below is an enterprise-grade resilient pipeline combining `RunnableConfig` tagging, 
custom event callback logging, automated retries with exponential backoff, and multi-cloud fallback routing.
""")

st.code('''from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableWithFallbacks, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class ProductionTelemetryHandler(BaseCallbackHandler):
    """Logs pipeline execution telemetry for SOC2 audit compliance."""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass  # Hook into distributed tracing

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass  # Alert on-call SRE pager

class ResilientEnterpriseService:
    """Production service with multi-cloud failover, retries, and observability telemetry."""

    def __init__(self, openai_key: str, anthropic_key: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise cloud security compliance auditor."),
            ("human", "{query}")
        ])
        parser = StrOutputParser()

        # Primary Model (OpenAI) with retries
        primary_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=10.0,
            api_key=openai_key
        ).with_retry(stop_after_attempt=2)

        # Fallback Model (Anthropic Claude)
        fallback_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            timeout=12.0,
            api_key=anthropic_key
        )

        # Resilient Model with Fallback
        resilient_model = primary_llm.with_fallbacks([fallback_llm])

        # Master LCEL Pipeline
        self.pipeline = prompt | resilient_model | parser
        self.telemetry_handler = ProductionTelemetryHandler()

    def execute_audit(self, query_text: str, tenant_id: str) -> str:
        # Construct runtime configuration
        config: RunnableConfig = {
            "tags": ["production", "security-audit"],
            "metadata": {"tenant_id": tenant_id},
            "callbacks": [self.telemetry_handler]
        }
        return self.pipeline.invoke({"query": query_text}, config=config)
''', language="python")

# --- SECTION 12: INTERACTIVE WORKBENCH ---
st.header("12. Interactive Workbench: Interactive Fault-Tolerance & Runtime Configuration Simulator", divider="rainbow")

st.markdown("""
Test the fault-tolerance and retry mechanisms in the interactive simulator below. 
Adjust the **Primary Model Failure Probability**, configure **Max Retries**, enable or disable **Fallbacks**, 
and observe how the resilient pipeline recovers from simulated provider outages.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    sim_failure_rate = st.slider("Primary Model Outage / Failure Rate", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    retries_setting = st.slider("Max Retries (.with_retry)", min_value=0, max_value=3, value=2, step=1)
    enable_fallback_setting = st.toggle("Enable Fallback Model (.with_fallbacks)", value=True)

with col_wb2:
    test_query_wb = st.text_input("Test Query", value="Audit multi-region RDS database replica lag.")
    st.info("""
    **Resilience Mechanics:**
    1. If Primary fails, pipeline attempts up to `max_retries` with exponential backoff.
    2. If retries exhaust, pipeline switches seamlessly to Secondary Fallback if enabled.
    3. If fallback is disabled, pipeline crashes with an unhandled exception.
    """)

if st.button("🚀 Execute Resilient Pipeline Simulation", type="primary"):
    engine = MockFaultToleranceEngine(
        primary_failure_rate=sim_failure_rate,
        max_retries=retries_setting,
        enable_fallback=enable_fallback_setting
    )
    result = engine.execute_resilient_pipeline(test_query_wb)
    
    st.subheader("📊 Execution Diagnostics & Recovery Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", result["status"])
    m2.metric("Resolved By", result["resolved_by"].split(" (")[0])
    m3.metric("Retries Used", result["retries_used"])
    m4.metric("Total Latency", f"{result['latency_ms']} ms")

    st.write("### Execution Lifecycle Audit Logs")
    for log_entry in result["execution_logs"]:
        if "[START]" in log_entry:
            st.caption(f"🔵 {log_entry}")
        elif "[WARNING]" in log_entry:
            st.warning(log_entry)
        elif "[RETRY" in log_entry:
            st.info(f"🔄 {log_entry}")
        elif "[FAILOVER]" in log_entry:
            st.warning(f"🔀 {log_entry}")
        elif "[SUCCESS]" in log_entry:
            st.success(f"✅ {log_entry}")
        elif "[FATAL]" in log_entry:
            st.error(f"❌ {log_entry}")

    st.write("### Pipeline Response Output")
    st.code(result["output"], language="markdown")

# --- SECTION 13: TECHNICAL INTERVIEW PREPARATION ---
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does .with_fallbacks() differ from .with_retry() in production architecture?",
     ".with_retry() re-executes the exact same Runnable component using exponential backoff to recover from transient glitches (like temporary network drops or brief 429 rate limits). .with_fallbacks() handles catastrophic, unrecoverable failures (like prolonged cloud outages or permanent 500 errors) by transparently switching the entire execution to an alternative provider or simpler heuristic model."),
    ("What is the purpose of the RunnableConfig object and how is it propagated?",
     "RunnableConfig is a standardized dictionary passed as an optional second argument to execution methods (.invoke, .batch, .stream). It carries thread-local context such as metadata, tags, custom callbacks, and max_concurrency limits. LangChain automatically propagates this dictionary across all child Runnables throughout the entire DAG without manual forwarding."),
    ("How do custom Callbacks in LangChain enable enterprise compliance and auditing?",
     "By subclassing BaseCallbackHandler, developers can attach hooks (on_llm_start, on_llm_end, on_chain_error) that intercept raw prompts, model responses, token counts, and errors in real-time. This allows companies to stream audit logs to security SIEM platforms (like Datadog or Splunk) and enforce DLP filters before returning answers to users."),
    ("Why should developers bound max_concurrency when executing .batch() or RunnableParallel?",
     "Without max_concurrency, LangChain attempts to spawn threads for every single item in the batch. For large batches (e.g., 500 items), this creates 500 simultaneous outbound HTTP requests, triggering severe rate limit throttling (HTTP 429) from API providers and exhausting system socket file descriptors."),
    ("What are ConfigurableFields and how do they enable multi-tenant customization?",
     "ConfigurableFields expose specific parameters of a Runnable (such as temperature, model name, or prompt text) to be overridden on a per-request basis via the `config={'configurable': {...}}` dictionary. This allows a single deployed pipeline to serve multiple tenants with custom configurations without rebuilding the DAG.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Cascading Latency from Unbounded Fallback Timeouts. If your primary and secondary models both have 60-second timeouts, a cascading outage forces users to wait over 2 minutes before seeing an error. Always set aggressive timeouts (8–12 seconds) when chaining fallbacks.")
    st.error("**Trap 2 (Breaking Error)**: Modifying State Inside Callbacks. Performing heavy blocking operations or mutating application state inside `BaseCallbackHandler` slows down the entire pipeline. Always keep callbacks lightweight and non-blocking.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Silent Retry Loops Multiplying Token Bills. Misconfiguring `.with_retry(stop_after_attempt=10)` on an endpoint that returns 400 Bad Request wastes money retrying an unfixable syntax error. Only retry transient 429, 500, and 503 errors.")
    st.warning("**Trap 4 (Observability Warning)**: High-Cardinality Metadata Tags. Attaching unique millisecond timestamps as `tags` instead of `metadata` pollutes LangSmith dashboards. Use `tags` for categorical grouping and `metadata` for unique IDs.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Fallback Attachment"):
        st.markdown("**primary.with_fallbacks([secondary]) creates a zero-downtime resilient failover chain.**")
    with st.expander("🎴 Flashcard 2: Transient Retry Decorator"):
        st.markdown("**model.with_retry(stop_after_attempt=3) executes exponential backoff with jitter.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Context Propagation"):
        st.markdown("**RunnableConfig carries tags, metadata, callbacks, and max_concurrency across the DAG.**")
    with st.expander("🎴 Flashcard 4: Lifecycle Hook Base Class"):
        st.markdown("**BaseCallbackHandler provides hooks like on_llm_start and on_llm_end for telemetry.**")

st.divider()

# --- SECTION 14: SELF-ASSESSMENT CHECKLIST ---
st.header("14. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the components of the RunnableConfig specification.", value=True)
chk2 = st.checkbox("I can implement dynamic runtime parameter overrides via configurable_fields.", value=True)
chk3 = st.checkbox("I know how to intercept execution lifecycle events using custom Callbacks.", value=True)
chk4 = st.checkbox("I can bound worker concurrency in .batch() using max_concurrency.", value=True)
chk5 = st.checkbox("I can configure automated retries (.with_retry) and zero-downtime fallbacks (.with_fallbacks).", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 5.4 Advanced Runnable Concepts. You have completed Chapter 5: LangChain Expression Language and Runnables!")
