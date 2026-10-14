import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.1 Introduction to LangSmith - LangChain Architecture",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling telemetry cards and metrics
st.markdown("""
<style>
    .telemetry-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-status-ok {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-status-err {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockLangSmithTelemetrySimulator:
    """
    Simulation engine showcasing LangSmith observability telemetry,
    tracing runs, error attribution, and cost tracking.
    """
    def __init__(self):
        self.runs = [
            {
                "run_id": "run-001-rag-query",
                "name": "CustomerSupportRAG",
                "run_type": "chain",
                "status": "SUCCESS",
                "latency_ms": 840,
                "prompt_tokens": 520,
                "completion_tokens": 140,
                "cost_usd": 0.0032,
                "error": None
            },
            {
                "run_id": "run-002-sql-agent",
                "name": "SQLDataAnalyst",
                "run_type": "agent",
                "status": "SUCCESS",
                "latency_ms": 2350,
                "prompt_tokens": 1820,
                "completion_tokens": 310,
                "cost_usd": 0.0124,
                "error": None
            },
            {
                "run_id": "run-003-tool-failure",
                "name": "WeatherQueryChain",
                "run_type": "tool",
                "status": "ERROR",
                "latency_ms": 120,
                "prompt_tokens": 150,
                "completion_tokens": 0,
                "cost_usd": 0.0004,
                "error": "APIConnectionError: Endpoint timeout on external weather gateway."
            },
            {
                "run_id": "run-004-summarizer",
                "name": "ExecutiveSummaryModel",
                "run_type": "llm",
                "status": "SUCCESS",
                "latency_ms": 1120,
                "prompt_tokens": 3400,
                "completion_tokens": 280,
                "cost_usd": 0.0195,
                "error": None
            }
        ]

    def get_summary_metrics(self) -> Dict[str, Any]:
        total_runs = len(self.runs)
        successes = sum(1 for r in self.runs if r["status"] == "SUCCESS")
        total_cost = sum(r["cost_usd"] for r in self.runs)
        avg_latency = sum(r["latency_ms"] for r in self.runs) / total_runs
        return {
            "total_runs": total_runs,
            "success_rate": f"{(successes / total_runs) * 100:.1f}%",
            "total_cost_usd": f"{total_cost:.4f} USD",
            "avg_latency_ms": f"{avg_latency:.1f} ms"
        }

# Main Application Title & Objective
st.title("🔬 21.1 Introduction to LangSmith")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
LangSmith is the enterprise observability, debugging, evaluation, and monitoring platform designed specifically for 
non-deterministic LLM applications and agentic graphs. In production, building an AI system without observability is operating 
blindly: LangSmith captures full execution lineage, token consumption, latency waterfalls, and failure diagnostics with zero code changes.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The LangSmith Observability & Telemetry Pipeline", divider="rainbow")
st.markdown("""
The architecture below delineates how execution traces are automatically extracted from LangChain Runnables and LangGraph agents. 
Using an asynchronous background worker pool, telemetry packets are serialized and dispatched out-of-band to LangSmith's 
high-throughput ingestion platform without adding network latency to user-facing inference requests.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    smith_dot = graphviz.Digraph(comment="LangSmith Architecture")
    smith_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    smith_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    smith_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    smith_dot.node("app_layer", "Application Code / Agent Graph\n[LangChain / LangGraph Node Execution]", fillcolor="#0F172A", shape="ellipse")
    smith_dot.node("tracer_hook", "LangChain Native Tracer Hook\n[LANGCHAIN_TRACING_V2=true]", fillcolor="#1E3A8A")
    smith_dot.node("batch_queue", "Async In-Memory Batching Queue\n[Zero-Overhead Local Ring Buffer]", fillcolor="#0E7490")
    smith_dot.node("bg_worker", "Background Dispatch Worker\n[Flushes Telemetry via Async HTTP POST]", fillcolor="#334155")
    
    # LangSmith Platform
    smith_dot.node("ingest_api", "LangSmith Ingestion Gateway\n[High-Throughput Global Ingestion API]", fillcolor="#701A75")
    smith_dot.node("trace_db", "Trace Storage & Indexing Engine\n[Distributed Run Tree Hierarchies]", fillcolor="#4338CA")
    smith_dot.node("dashboard", "LangSmith Web Console\n[Trace Waterfall / Debugger / Monitoring / Evals]", fillcolor="#047857", shape="ellipse")

    smith_dot.edge("app_layer", "tracer_hook", label=" Execution Events")
    smith_dot.edge("tracer_hook", "batch_queue", label=" Zero Latency Hook")
    smith_dot.edge("batch_queue", "bg_worker")
    smith_dot.edge("bg_worker", "ingest_api", label=" Async Batch POST")
    smith_dot.edge("ingest_api", "trace_db")
    smith_dot.edge("trace_db", "dashboard", label=" Real-Time Telemetry")

    st.graphviz_chart(smith_dot, use_container_width=True)

# --- SECTION 2: WHAT IS LANGSMITH? ---
st.header("2. What is LangSmith?", divider="rainbow")
st.markdown("""
**LangSmith** is a unified DevOps and observability platform built by the creators of LangChain. It manages the entire 
lifecycle of LLM applications: from local development and prompt engineering to production tracing, regression testing, and quality monitoring.
""")

col_ls1, col_ls2 = st.columns(2)
with col_ls1:
    st.subheader("Core Capabilities of LangSmith")
    st.markdown("""
    * **Traceability**: Visualizes the nested hierarchy of prompts, LLM calls, tool executions, and retrieval operations.
    * **Debugging**: Inspects exact input/output payloads, system prompt hydration, and error stack traces.
    * **Evaluation**: Runs automated benchmark datasets against new model versions to detect accuracy regressions.
    * **Monitoring**: Tracks token consumption, operational costs, latency P95/P99, and error rates across production fleets.
    """)
with col_ls2:
    st.subheader("Traditional APM vs. LangSmith")
    st.markdown("""
    | Operational Metric | Traditional APM (Datadog/NewRelic) | LangSmith LLM Observability |
    | :--- | :--- | :--- |
    | **Primary Focus** | Server CPU, RAM, SQL query duration | Model prompts, token metrics, hallucinations |
    | **Payload Inspection** | Truncated strings / JSON metrics | Complete multi-turn conversation histories |
    | **Cost Tracking** | Infrastructure host billings | Per-token, per-model, per-user cost allocation |
    | **Evaluation** | HTTP 200/500 status codes | Semantic correctness, RAG faithfulness, groundedness |
    """)

# --- SECTION 3: WHY OBSERVABILITY MATTERS ---
st.header("3. Why Observability Matters in Non-Deterministic Systems", divider="rainbow")
st.markdown("""
Unlike traditional deterministic software where identical inputs yield identical outputs, LLM systems are probabilistic. 
Bugs do not always throw exceptions (`HTTP 500`); they often manifest as subtle degradations: **hallucinated tool arguments**, 
**context window overflows**, **excessive latency spikes**, or **silent format drift**.
""")

col_obs1, col_obs2, col_obs3 = st.columns(3)
with col_obs1:
    st.markdown("""
    <div class="telemetry-card">
        <h4>1. Silent Logical Failures</h4>
        <p>An agent might return a polite answer that is factually incorrect due to a retriever failing silently and returning 0 documents.</p>
        <span class="badge-status-err">Invisible Without Traces</span>
    </div>
    """, unsafe_allow_html=True)
with col_obs2:
    st.markdown("""
    <div class="telemetry-card">
        <h4>2. Financial Leakage</h4>
        <p>A loop in a ReAct agent can burn 100,000 tokens across 15 recursive tool calls in seconds. Without live cost alerts, budgets vanish rapidly.</p>
        <span class="badge-status-err">Cost Danger</span>
    </div>
    """, unsafe_allow_html=True)
with col_obs3:
    st.markdown("""
    <div class="telemetry-card">
        <h4>3. Latency Attribution</h4>
        <p>When an agent takes 8 seconds to respond, was the delay caused by LLM queueing, vector search, or an external API gateway?</p>
        <span class="badge-status-ok">Actionable Root Cause</span>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 4: LANGCHAIN + LANGSMITH ---
st.header("4. LangChain + LangSmith Native Integration", divider="rainbow")
st.markdown("""
The most powerful feature of LangSmith is its **zero-code native instrumentation**. 
Because LangChain and LangGraph have tracing hooks embedded into their core Runnable abstractions, enabling LangSmith 
requires setting four environment variables without modifying a single line of business logic.
""")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.subheader("Environment Configuration")
    st.markdown("""
    Set these environment variables before running your application:
    ```bash
    export LANGCHAIN_TRACING_V2="true"
    export LANGCHAIN_API_KEY="lsv2_pt_..."
    export LANGCHAIN_PROJECT="production-customer-support"
    export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    ```
    * `LANGCHAIN_TRACING_V2`: Enables the modern v2 tracing dispatcher.
    * `LANGCHAIN_PROJECT`: Groups traces into logical project workspaces.
    """)
with col_cfg2:
    st.subheader("Automatic Hook Binding")
    st.code("""
# Zero tracing code needed in application scripts:
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

chain = prompt | model

# This invocation is automatically captured and uploaded to LangSmith!
response = chain.invoke({"text": "Enterprise observability is critical."})
    """, language="python")

# --- SECTION 5: TRACING ---
st.header("5. Tracing Capabilities & Run Trees", divider="rainbow")
st.markdown("""
In LangSmith, an execution trace is modeled as a **Hierarchical Run Tree**. 
A top-level agent run acts as the root node, containing child runs for internal prompt templates, LLM calls, tool dispatches, 
and vector store retrievals.
""")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.subheader("Run Tree Hierarchical Levels")
    st.markdown("""
    * **Root Run (`chain` / `agent`)**: The entry-point user invocation.
    * **Child Node (`llm`)**: Records prompt tokens, completion tokens, model parameters, and TTFT.
    * **Child Node (`tool`)**: Records tool function name, serialized JSON arguments, and return payload.
    * **Child Node (`retriever`)**: Records query string, embedding similarity scores, and retrieved document chunks.
    """)
with col_tr2:
    st.subheader("Run Object Metadata Structure")
    st.code("""
{
  "id": "run-9942a",
  "name": "ChatOpenAI",
  "run_type": "llm",
  "parent_run_id": "run-001a",
  "inputs": {"prompts": ["Translate to French..."]},
  "outputs": {"generations": [{"text": "L'observabilité..."}]},
  "extra": {
    "metadata": {"user_id": "usr_882", "tier": "enterprise"},
    "tags": ["prod", "v2-release"]
  }
}
    """, language="python")

# --- SECTION 6: DEBUGGING ---
st.header("6. Root-Cause Debugging Workflows", divider="rainbow")
st.markdown("""
When an agent fails in production, recreating the exact error locally can be nearly impossible due to dynamic retrieved contexts 
and probabilistic variations. LangSmith provides dedicated tools to inspect, isolate, and debug failures directly from production traces.
""")

col_dbg1, col_dbg2 = st.columns(2)
with col_dbg1:
    st.subheader("Debugging Features in Console")
    st.markdown("""
    * **Exact Prompt Reconstruction**: See the exact rendered prompt sent to the LLM, including injected system instructions and context.
    * **Open in Playground**: One-click export from a failed production trace directly into the interactive LangSmith Playground to tweak prompts.
    * **Error Stack Attribution**: Drill down through nested runs to find the exact component that raised an exception.
    """)
with col_dbg2:
    st.subheader("Debugging Exception Inspection")
    st.code("""
# LangSmith captures complete exception context:
try:
    result = agent.invoke(user_query)
except Exception as e:
    # Trace is flagged as status="ERROR"
    # Exception type, message, and full stack trace
    # are permanently indexed in the LangSmith console
    pass
    """, language="python")

# --- SECTION 7: MONITORING ---
st.header("7. Production Telemetry & Latency Monitoring", divider="rainbow")
st.markdown("""
LangSmith monitoring aggregates telemetry across thousands of production runs into real-time dashboards 
tracking **P50/P90/P99 latency**, **token consumption**, **monetary expenditure**, and **user feedback scores**.
""")

st.markdown("""
### Core Monitoring Metrics
| Metric | Diagnostic Value | Alerting Threshold |
| :--- | :--- | :--- |
| **Error Rate** | Tracks API timeouts, schema parsing failures, and rate limits | Alert if > 2.0% in 5-minute window |
| **P99 Latency** | Identifies extreme latency outliers affecting user experience | Alert if P99 > 12.0 seconds |
| **Token Burn Rate** | Monitors total input and output token consumption per hour | Alert on anomalous cost spikes (> 50 USD/hr) |
| **Feedback Score** | User thumbs-up / thumbs-down ratings submitted via client UI | Alert if negative feedback exceeds 15% |
""")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Zero-Overhead LangSmith Client Configuration & Tracing Context Manager", divider="rainbow")
st.markdown("""
The following complete production pattern illustrates how to configure a robust LangSmith tracing client programmatically, 
using custom tags, metadata annotation, and async flush handling.
""")

st.code('''
from typing import Dict, Any
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

# 1. Initialize LangSmith Client
smith_client = Client()

# 2. Production Service with Custom Tracing Context
def execute_observed_workflow(user_query: str, customer_id: str, plan_tier: str) -> Dict[str, Any]:
    """
    Executes a LangChain workflow with explicit project tagging,
    customer metadata annotation, and error tracking.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise financial assistant. Answer accurately."),
        ("human", "{query}")
    ])
    chain = prompt | model

    # Programmatic tracing context with metadata and tags
    with tracing_v2_enabled(
        project_name="enterprise-fintech-prod",
        tags=["production", plan_tier, "financial_advisory"],
        client=smith_client
    ):
        # Pass runtime metadata attached to the root trace
        config = {
            "metadata": {
                "customer_id": customer_id,
                "plan_tier": plan_tier,
                "region": "us-east-1"
            }
        }
        
        response = chain.invoke({"query": user_query}, config=config)
        return {
            "content": response.content,
            "status": "COMPLETED"
        }
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Observability Telemetry & Run Lifecycle Dashboard", divider="rainbow")
st.markdown("""
Inspect real-time telemetry from simulated production runs. Examine how LangSmith aggregates **Latency**, 
**Token Usage**, and **Cost Metrics** while isolating errors.
""")

# Instantiate Module Scope Simulator
if "smith_sim" not in st.session_state:
    st.session_state.smith_sim = MockLangSmithTelemetrySimulator()

sim_sm = st.session_state.smith_sim

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
summary = sim_sm.get_summary_metrics()

with col_m1:
    st.metric("Total Observed Runs", summary["total_runs"])
with col_m2:
    st.metric("Success Rate", summary["success_rate"])
with col_m3:
    st.metric("Total Incurred Cost", summary["total_cost_usd"])
with col_m4:
    st.metric("Average Run Latency", summary["avg_latency_ms"])

st.markdown("### 📋 Production Run Telemetry Stream")
runs_df = pd.DataFrame(sim_sm.runs)
st.dataframe(runs_df[["run_id", "name", "run_type", "status", "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd"]], use_container_width=True)

st.subheader("Drill-Down Run Inspection")
selected_run_id = st.selectbox("Select Run ID to Inspect:", options=[r["run_id"] for r in sim_sm.runs])
run_details = next(r for r in sim_sm.runs if r["run_id"] == selected_run_id)

st.json(run_details)

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How does LangSmith achieve zero latency overhead on user-facing requests during tracing?"):
    st.markdown("""
    LangSmith avoids blocking the request thread by using an asynchronous in-memory background worker. 
    When an execution event fires, the telemetry payload is enqueued onto an in-memory queue and the method returns immediately. 
    A separate background thread flushes batches of traces over HTTPS to the LangSmith ingestion API asynchronously.
    """)

with st.expander("Q2: What is the purpose of `LANGCHAIN_PROJECT` in production deployments?"):
    st.markdown("""
    `LANGCHAIN_PROJECT` groups traces into distinct, isolated environments (e.g. `dev`, `staging`, `prod-us-east`). 
    Without project segmentation, development testing and automated CI/CD runs contaminate production metrics, 
    distorting latency benchmarks and cost tracking dashboards.
    """)

with st.expander("Q3: How does LangSmith differentiate between an `llm`, `chain`, and `tool` run type?"):
    st.markdown("""
    LangSmith inspects the component's internal Runnable hierarchy:
    * `llm`: Subclasses of `BaseLanguageModel` / `BaseChatModel` (records token counts and prompts).
    * `chain`: Composed Runnables (`RunnableSequence`, `StateGraph`) managing orchestrations.
    * `tool`: Instances of `BaseTool` executing discrete functions with typed input/output arguments.
    """)

with st.expander("Q4: Can LangSmith trace applications built without LangChain (e.g., raw OpenAI SDK calls)?"):
    st.markdown("""
    Yes. LangSmith provides the `@traceable` decorator from the `langsmith` SDK. 
    Wrapping any arbitrary Python function with `@traceable` captures inputs, outputs, exceptions, and execution duration 
    into a LangSmith run tree without requiring any LangChain imports.
    """)

with st.expander("Q5: What privacy controls does LangSmith provide for sensitive PII in traces?"):
    st.markdown("""
    LangSmith provides multiple privacy guardrails:
    1. Client-side hiding of inputs/outputs via `hide_inputs=True` and `hide_outputs=True`.
    2. Enterprise Self-Hosted / On-Premise deployments where trace data never leaves the customer VPC.
    3. Custom callback masking that redacts credit card numbers or HIPAA data before dispatch.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_ls_t1, col_ls_t2 = st.columns(2)
with col_ls_t1:
    st.error("""
    **Production Trap 1 (Fatal Error): Committing `LANGCHAIN_API_KEY` to Version Control**  
    Hardcoding LangSmith API credentials directly in application source code. 
    Attackers scanning GitHub can steal the API key to read production customer conversation logs containing sensitive corporate secrets.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Contaminating Production Metrics with Test Traffic**  
    Running automated pytest integration suites against the `production` LangSmith project. 
    Synthetic test failures skew production SLA metrics and trigger false-alarm on-call alerts.
    """)
with col_ls_t2:
    st.error("""
    **Production Trap 3 (Fatal Error): Unbounded In-Memory Telemetry Buffering**  
    If the LangSmith ingestion API becomes unreachable during a network partition, an unconfigured client queue can accumulate 
    millions of trace dictionaries in memory, eventually triggering an Out-of-Memory (OOM) crash of the production container.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Omitting User Attribution in Metadata**  
    Failing to attach `user_id` or `tenant_id` to trace metadata. 
    When an enterprise customer reports an isolated bug, searching through 500,000 anonymous traces to find their specific conversation is virtually impossible.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc_s1, col_fc_s2 = st.columns(2)
with col_fc_s1:
    st.markdown("""
    <div class="telemetry-card">
        <h4>Flashcard 1: LANGCHAIN_TRACING_V2</h4>
        <p><strong>Config:</strong> Environment Flag</p>
        <p><strong>Definition:</strong> Setting <code>LANGCHAIN_TRACING_V2="true"</code> globally enables zero-code asynchronous trace ingestion to LangSmith.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="telemetry-card">
        <h4>Flashcard 2: Run Tree</h4>
        <p><strong>Hierarchy:</strong> Execution Lineage</p>
        <p><strong>Definition:</strong> The nested tree structure recording parent chains, child LLM calls, and leaf tool executions within a single user transaction.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc_s2:
    st.markdown("""
    <div class="telemetry-card">
        <h4>Flashcard 3: @traceable Decorator</h4>
        <p><strong>Decorator:</strong> Universal SDK Tracer</p>
        <p><strong>Definition:</strong> The LangSmith decorator that instrument any arbitrary Python function without requiring LangChain primitives.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="telemetry-card">
        <h4>Flashcard 4: Project Segmentation</h4>
        <p><strong>Variable:</strong> <code>LANGCHAIN_PROJECT</code></p>
        <p><strong>Definition:</strong> The environment parameter used to isolate dev, staging, and production traces into separate analytical dashboards.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of LangSmith observability and telemetry architectures:")

ch_sm1 = st.checkbox("1. I understand what LangSmith is and why observability is essential in non-deterministic LLM applications.")
ch_sm2 = st.checkbox("2. I know how to configure `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`.")
ch_sm3 = st.checkbox("3. I understand how hierarchical run trees capture parent chains, child models, and tool invocations.")
ch_sm4 = st.checkbox("4. I know how LangSmith enables zero-overhead asynchronous background trace dispatching.")
ch_sm5 = st.checkbox("5. I understand how to inspect exact prompt payloads and debug production failures in the LangSmith console.")
ch_sm6 = st.checkbox("6. I understand how to use `tracing_v2_enabled()` to annotate production traces with custom metadata and tags.")

if ch_sm1 and ch_sm2 and ch_sm3 and ch_sm4 and ch_sm5 and ch_sm6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Introduction to LangSmith.")
