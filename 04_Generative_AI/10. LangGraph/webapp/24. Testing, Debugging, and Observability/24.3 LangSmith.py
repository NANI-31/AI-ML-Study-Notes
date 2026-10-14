import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="24.3 LangSmith - LangGraph Curriculum",
    page_icon="🔍",
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
    .badge-span {
        background-color: #172554;
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE TELEMETRY SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class LangSmithTelemetrySimulator:
    """
    Simulates production LangSmith trace hierarchies, token accounting,
    latency tracking, and error diagnostics for LangGraph workloads.
    """
    def __init__(self):
        self.mock_runs = [
            {
                "run_id": "run-8fa29-01",
                "name": "CustomerSupportGraph",
                "run_type": "chain",
                "status": "success",
                "latency_ms": 1420,
                "prompt_tokens": 850,
                "completion_tokens": 210,
                "total_tokens": 1060,
                "spans": [
                    {"name": "START -> router", "type": "node", "latency_ms": 12, "tokens": 0, "status": "ok"},
                    {"name": "router (LLM Classifier)", "type": "llm", "latency_ms": 380, "tokens": 240, "status": "ok"},
                    {"name": "billing_agent (Planner)", "type": "node", "latency_ms": 15, "tokens": 0, "status": "ok"},
                    {"name": "billing_llm (ChatOpenAI)", "type": "llm", "latency_ms": 820, "tokens": 780, "status": "ok"},
                    {"name": "stripe_invoice_tool", "type": "tool", "latency_ms": 190, "tokens": 0, "status": "ok"},
                    {"name": "END", "type": "node", "latency_ms": 3, "tokens": 0, "status": "ok"}
                ]
            },
            {
                "run_id": "run-8fa29-02",
                "name": "CodeReviewAgent",
                "run_type": "chain",
                "status": "error",
                "latency_ms": 2840,
                "prompt_tokens": 1450,
                "completion_tokens": 420,
                "total_tokens": 1870,
                "error_message": "RateLimitError: TPM limit reached on gpt-4o",
                "spans": [
                    {"name": "START -> linter", "type": "node", "latency_ms": 22, "tokens": 0, "status": "ok"},
                    {"name": "ast_parser_tool", "type": "tool", "latency_ms": 140, "tokens": 0, "status": "ok"},
                    {"name": "critic_agent (LLM)", "type": "llm", "latency_ms": 2678, "tokens": 1870, "status": "error"}
                ]
            },
            {
                "run_id": "run-8fa29-03",
                "name": "ResearchSynthesisGraph",
                "run_type": "chain",
                "status": "success",
                "latency_ms": 3850,
                "prompt_tokens": 3200,
                "completion_tokens": 980,
                "total_tokens": 4180,
                "spans": [
                    {"name": "query_expander", "type": "node", "latency_ms": 410, "tokens": 450, "status": "ok"},
                    {"name": "retriever_node (Parallel)", "type": "chain", "latency_ms": 620, "tokens": 0, "status": "ok"},
                    {"name": "vector_search_1", "type": "tool", "latency_ms": 280, "tokens": 0, "status": "ok"},
                    {"name": "vector_search_2", "type": "tool", "latency_ms": 310, "tokens": 0, "status": "ok"},
                    {"name": "synthesizer_llm", "type": "llm", "latency_ms": 2820, "tokens": 3730, "status": "ok"}
                ]
            }
        ]

    def get_runs(self) -> List[Dict[str, Any]]:
        return self.mock_runs

    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        for run in self.mock_runs:
            if run["run_id"] == run_id:
                return run
        return None

telemetry_sim = LangSmithTelemetrySimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Observability Pipeline, Tracing Spans, and LangSmith Telemetry", divider="rainbow")

st.markdown("""
LangSmith provides enterprise-grade observability, continuous evaluation, and production monitoring for LangGraph agents. 
When enabled, the LangChain OpenTelemetry-compatible tracing harness instruments every graph invocation, superstep barrier, 
node execution, tool call, and individual LLM inference span. Telemetry streams asynchronously to the LangSmith backend 
without impeding real-time runtime throughput.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangSmithBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_app {
            label="1. LangGraph Execution Engine";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            app_invoke [label="app.invoke(inputs, config)\\nGenerates Root Run Span", fillcolor="#0F172A"];
            superstep [label="Superstep Execution Barrier\\nDispatches Node Spans", fillcolor="#1E293B"];
            llm_call [label="LLM Invocation Span\\nCaptures Prompts & Completions", fillcolor="#312E81"];
            tool_call [label="Tool Invocation Span\\nCaptures Arguments & Tool Results", fillcolor="#064E3B"];
        }

        subgraph cluster_client {
            label="2. Background Trace Exporter";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            exporter [label="Async Batch Queue\\nNon-blocking Worker Thread\\nBuffers Telemetry Events", fillcolor="#1E293B"];
        }

        subgraph cluster_cloud {
            label="3. LangSmith Observability Platform";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            trace_db [label="Trace & Run Database\\nIndexed by Project, Run ID & Thread ID", fillcolor="#0F172A"];
            metrics_engine [label="Metrics & Analytics Engine\\nToken Usage, Latency p95, Error Rates", fillcolor="#1E293B"];
            dag_visualizer [label="Interactive Graph Visualizer\\nStep Replay & Prompt Regression Testing", fillcolor="#047857"];
        }

        app_invoke -> superstep;
        superstep -> llm_call;
        superstep -> tool_call;

        app_invoke -> exporter [label="Emits Span Start/End"];
        llm_call -> exporter [label="Tokens & Latency"];
        tool_call -> exporter [label="I/O Payloads"];

        exporter -> trace_db [label="HTTPS / TLS Ingestion"];
        trace_db -> metrics_engine;
        trace_db -> dag_visualizer;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: LangSmith Introduction
# ==============================================================================
st.header("2. LangSmith Introduction: Managed Platform Architecture and Observability Contract", divider="rainbow")

st.markdown("""
LangSmith is a centralized platform designed to bridge local agent development and enterprise production operations. 
It addresses the non-deterministic nature of large language model graphs by transforming black-box agent loops into fully auditable, 
reproducible trace trees.

### Core Observability Pillars:
1. **Zero-Code Instrumentation**: Native integration via environment variables without requiring intrusive code modifications.
2. **Deterministic Replayability**: Any run trace can be inspected, extracted into a dataset, and replayed against new prompts or models.
3. **Multi-Tenant Tagging**: Traces can be enriched with metadata (such as `user_id`, `tenant_id`, `environment="production"`), 
   enabling fine-grained cost allocation and security auditing.
""")

# ==============================================================================
# SECTION 3: Tracing
# ==============================================================================
st.header("3. Tracing: Automatic and Explicit Instrumentations across Graphs and Chains", divider="rainbow")

st.markdown("""
Enabling tracing in any Python application utilizing LangGraph requires configuring environment variables. 
The underlying `langchain-core` callbacks automatically detect the credentials and initialize background telemetry streaming.

### Standard Configuration:
```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT="prod-langgraph-agent"
```

### Passing Metadata and Tags in RunnableConfig:
To associate telemetry with specific production users and business transactions, inject tags and metadata into the invocation configuration:
""")

st.code('''config = {
    "configurable": {
        "thread_id": "session-1092"
    },
    "tags": ["tenant-enterprise", "feature-q3-summary"],
    "metadata": {
        "user_id": "usr_99824",
        "org_id": "org_healthcare_corp",
        "app_version": "v2.4.1"
    }
}

# All child nodes, LLMs, and tools automatically inherit these tags and metadata
response = app.invoke({"query": "Generate compliance report"}, config)
''', language="python")

# ==============================================================================
# SECTION 4: Run Traces
# ==============================================================================
st.header("4. Run Traces: Tree-Structured Spans, Supersteps, Node Inputs, and Outputs", divider="rainbow")

st.markdown("""
In LangSmith, a **Run** represents a single unit of execution. In a LangGraph application, runs are structured 
as hierarchical trees reflecting the graph's superstep boundaries and node dispatch.
""")

with st.expander("🔍 View Hierarchical Run Trace Tree Architecture", expanded=True):
    trace_tree_dot = """
    digraph RunTraceTree {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        root [label="Root Run (LangGraph Chain)\\nName: SupportAgent | Latency: 1.84s | Status: OK", fillcolor="#0F172A"];
        node_agent [label="Node Run: agent\\nType: node | Latency: 1.12s", fillcolor="#1E293B"];
        llm_agent [label="LLM Run: ChatAnthropic\\nModel: claude-3-5-sonnet | Latency: 1.08s\\nTokens: 1,240 (Prompt: 980, Comp: 260)", fillcolor="#312E81"];
        node_tools [label="Node Run: tools\\nType: node | Latency: 0.70s", fillcolor="#1E293B"];
        tool_sql [label="Tool Run: sql_query_runner\\nQuery: SELECT revenue FROM financials...\\nLatency: 0.68s", fillcolor="#064E3B"];

        root -> node_agent [label="superstep 1"];
        node_agent -> llm_agent [label="child span"];
        root -> node_tools [label="superstep 2"];
        node_tools -> tool_sql [label="child span"];
    }
    """
    st.graphviz_chart(trace_tree_dot, use_container_width=True)

st.markdown("""
### Run Trace Components:
- **`id`**: Unique UUID assigned to the run span.
- **`run_type`**: Discriminator such as `chain`, `llm`, `tool`, or `retriever`.
- **`inputs`**: Exact dictionary payload passed into the node or model.
- **`outputs`**: Exact state delta or completion generated by the node.
- **`child_runs`**: List of nested sub-spans executed during this run.
""")

# ==============================================================================
# SECTION 5: Token Usage
# ==============================================================================
st.header("5. Token Usage: Monitoring Prompt, Completion, and Total Token Overhead", divider="rainbow")

st.markdown("""
LangSmith automatically aggregates token metrics across all LLM inference spans inside a graph execution. 
Token consumption is categorized into:
- **Prompt Tokens**: Input tokens including system prompts, conversation history, and tool schemas.
- **Completion Tokens**: Output tokens generated by the language model.
- **Total Tokens**: Sum of prompt and completion counts.

### Managing Token Bloat in LangGraph:
1. **ReAct Loop Accumulation**: Long loops re-send full conversation histories on every iteration. 
   A 5-step loop can easily consume 10x the tokens of a single invocation.
2. **State Compression**: LangSmith telemetry highlights exactly which node introduces excessive context, 
   enabling targeted summarization or filtering via channel reducers.
""")

token_df = pd.DataFrame({
    "Node / LLM Component": ["Query Classifier", "Researcher Agent (Turn 1)", "Researcher Agent (Turn 2)", "Synthesizer"],
    "Prompt Tokens": [180, 1250, 2400, 3100],
    "Completion Tokens": [45, 320, 410, 890],
    "Total Tokens": [225, 1570, 2810, 3990],
    "Relative Cost Share (%)": [3.2, 22.8, 40.7, 33.3]
})
st.dataframe(token_df, use_container_width=True)

# ==============================================================================
# SECTION 6: Latency
# ==============================================================================
st.header("6. Latency: Span Duration Profiling, Bottleneck Detection, and TTFT Analysis", divider="rainbow")

st.markdown("""
Latency in multi-agent graphs is governed by sequential supersteps, model inference time, and tool I/O bottlenecks. 
LangSmith provides granular timing breakdowns to isolate performance degradation.

### Key Latency Metrics:
- **Execution Duration**: Total wall-clock time from run initiation to termination.
- **Time to First Token (TTFT)**: Latency before streaming starts on LLM calls.
- **Tool Overhead vs Model Overhead**: Ratio of time spent waiting for external databases/APIs vs awaiting LLM completions.
- **p50, p90, and p99 Latencies**: Aggregated service level metrics across production workloads.
""")

# ==============================================================================
# SECTION 7: Errors
# ==============================================================================
st.header("7. Errors: Exception Capture, Stack Trace Aggregation, and Retry Failures", divider="rainbow")

st.markdown("""
When a node throws an unhandled exception or when an API provider returns a 429/500 error, LangSmith flags the corresponding 
run span as `status: error`.

### Diagnosing Production Failures:
1. **Stack Trace Aggregation**: The exact traceback is preserved alongside the inputs that triggered the failure.
2. **Context Preservation**: Unlike standard Python error logs, LangSmith retains the full preceding graph state, 
   enabling immediate local reproduction via time-travel debugging.
3. **Filter by Error Type**: Production dashboards enable filtering runs by exception class (e.g., `OutputParserException`, 
   `RateLimitError`, `TimeoutError`).
""")

# ==============================================================================
# SECTION 8: Graph Visualization
# ==============================================================================
st.header("8. Graph Visualization: Interactive DAG Topologies and Run Replay in LangSmith", divider="rainbow")

st.markdown("""
LangSmith features an integrated topological graph visualizer that maps runtime traces directly onto the compiled `StateGraph` blueprint.

### Visualizer Features:
- **Path Highlighting**: Active execution pathways are rendered in real time, showing branching decisions taken by conditional edges.
- **Step Scrubbing**: Developers can scrub through supersteps, observing how state evolves node-by-node.
- **Playground Integration**: A single click transfers historical inputs from an error span into the interactive LangSmith Playground 
  for prompt experimentation and model comparison.
""")

# ==============================================================================
# SECTION 9: Agent Debugging
# ==============================================================================
st.header("9. Agent Debugging: Diagnosing ReAct Loops, Hallucinations, and Tool Failures", divider="rainbow")

st.markdown("""
Debugging autonomous agents presents unique challenges due to dynamic routing and multi-turn loops. 
LangSmith equips engineers with specialized workflows to resolve complex agent issues:

1. **Tool Invocation Auditing**: Verifying whether the model hallucinated tool arguments or passed invalid JSON.
2. **Infinite Loop Detection**: Identifying cycles where an agent calls the same tool repeatedly without making progress.
3. **Guardrail Violations**: Flagging when node outputs violate output parsing schemas or content safety policies.
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full LangSmith Tracing, Telemetry, and Custom Metadata Pipeline", divider="rainbow")

st.markdown("""
The following module demonstrates production-ready tracing configuration with custom span metadata, runtime tags, 
and OpenTelemetry-compatible telemetry reporting.
""")

st.code('''from typing import Dict, Any, Optional
from langchain_core.tracers.context import collect_runs
from langgraph.graph import StateGraph, START, END

class MonitoredLangGraphPipeline:
    """
    Production wrapper for LangGraph compilation and execution
    with automatic LangSmith trace tagging and metric extraction.
    """
    def __init__(self, compiled_graph, project_name: str = "production-agents"):
        self.graph = compiled_graph
        self.project_name = project_name

    def execute_with_telemetry(
        self,
        inputs: Dict[str, Any],
        thread_id: str,
        user_id: str,
        environment: str = "production"
    ) -> Dict[str, Any]:
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "tags": [environment, f"user:{user_id}"],
            "metadata": {
                "project": self.project_name,
                "environment": environment,
                "user_id": user_id,
                "execution_mode": "monitored"
            }
        }

        # Execute graph within LangSmith run collector
        with collect_runs() as cb:
            result = self.graph.invoke(inputs, config)
            # Retrieve root run trace
            root_run = cb.traced_runs[0] if cb.traced_runs else None
            
        telemetry_summary = {
            "run_id": str(root_run.id) if root_run else None,
            "latency_ms": getattr(root_run, "total_duration", 0),
            "total_tokens": getattr(root_run, "total_tokens", 0),
            "status": getattr(root_run, "status", "unknown")
        }
        
        return {
            "output": result,
            "telemetry": telemetry_summary
        }
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Observability Telemetry Dashboard & Run Trace Inspector", divider="rainbow")

st.markdown("""
Explore simulated production traces, examine span hierarchies, and diagnose token and latency bottlenecks.
""")

runs = telemetry_sim.get_runs()
run_options = [f"{r['name']} ({r['run_id']}) - {r['status'].upper()}" for r in runs]

selected_idx = st.selectbox(
    "Select Telemetry Run Trace",
    range(len(run_options)),
    format_func=lambda i: run_options[i]
)
selected_run = runs[selected_idx]

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Total Latency", f"{selected_run['latency_ms']} ms")
with col_kpi2:
    st.metric("Total Tokens", f"{selected_run['total_tokens']:,}")
with col_kpi3:
    st.metric("Prompt Tokens", f"{selected_run['prompt_tokens']:,}")
with col_kpi4:
    status_label = "SUCCESS" if selected_run["status"] == "success" else "FAILED"
    st.metric("Run Status", status_label)

st.markdown("### Execution Span Breakdown")
spans_data = selected_run["spans"]
span_table = []
for s in spans_data:
    span_table.append({
        "Span Name": s["name"],
        "Span Type": s["type"].upper(),
        "Duration (ms)": s["latency_ms"],
        "Tokens": s["tokens"],
        "Status": s["status"].upper()
    })
st.dataframe(pd.DataFrame(span_table), use_container_width=True)

if selected_run["status"] == "error":
    st.error(f"Error Diagnostic: {selected_run.get('error_message')}")
else:
    st.success("All graph execution spans completed within nominal latency thresholds.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How does LangSmith tracing impact real-time execution performance in high-throughput systems?"):
    st.markdown("""
    **Answer:**  
    LangSmith tracing uses an asynchronous, background worker queue to batch and stream telemetry over HTTPS. 
    Node execution and LLM streaming are not blocked on trace submission. The latency overhead on graph execution 
    is minimal (typically under 2-5 milliseconds per superstep).
    """)

with st.expander("Q2: How do you trace nested subgraphs and preserve thread continuity in LangSmith?"):
    st.markdown("""
    **Answer:**  
    LangGraph automatically propagates the parent run ID and execution context to compiled subgraphs. When subgraphs 
    execute, LangSmith models them as child spans under the parent graph's superstep node, indexing them under the same 
    root `run_id` and retaining full thread continuity.
    """)

with st.expander("Q3: What is the recommended strategy for masking PII/sensitive data before sending traces to LangSmith?"):
    st.markdown("""
    **Answer:**  
    Developers can implement custom `RunTracer` callbacks or client-side masking hooks (`hide_inputs` and `hide_outputs` 
    in tracer configs) to redact PII (such as SSNs, passwords, or medical records) before traces leave the application boundary.
    """)

with st.expander("Q4: How does LangSmith calculate token counts for streaming LLM calls?"):
    st.markdown("""
    **Answer:**  
    For streaming responses where token usage is not explicitly included in stream chunks by the model provider, 
    LangSmith applies model-specific tokenizers (e.g., `tiktoken` for OpenAI models) to count input prompts and assembled 
    output chunks, ensuring accurate billing and observability.
    """)

with st.expander("Q5: What is the difference between tags and metadata in LangSmith run configurations?"):
    st.markdown("""
    **Answer:**  
    - **Tags**: A list of string labels (e.g., `["production", "gpt-4o", "tenant-vip"]`) designed for rapid indexing, filtering, and dashboard segmentation.  
    - **Metadata**: A key-value dictionary (e.g., `{"user_id": "123", "org_id": "456", "version": "1.0"}`) designed for detailed structured analysis and cost attribution.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Hardcoding LangSmith API Keys in Source Code**  
Never hardcode `LANGSMITH_API_KEY` into application files or version control. Always inject credentials through secret managers 
or secure container environment variables.
""")

st.error("""
**Trap 2: Trace Explosion from High-Frequency Polling Graphs**  
Running graphs with high-frequency background loops or polling intervals without disabling tracing or applying sampling 
can generate millions of trivial spans, causing rate limits and unexpected telemetry costs.
""")

st.warning("""
**Trap 3: Forgetting Project Names in Multi-Environment Deployments**  
Failing to specify `LANGSMITH_PROJECT` routes staging, development, and production runs into the default project workspace, 
contaminating production evaluation benchmarks with test runs.
""")

st.warning("""
**Trap 4: Incomplete Trace Visibility Due to Synchronous Blocking Calls**  
Calling synchronous LLM endpoints inside async graph nodes can block the event loop and delay telemetry flush cycles, 
leading to dropped spans if processes terminate abruptly.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Run Span Hierarchy</h4>
        <p>A nested tree structure where root graph runs encapsulate supersteps, node runs, tool invocations, and individual LLM calls.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Token Attribution</h4>
        <p>Tracking prompt and completion token consumption per node to pinpoint reasoning loops and optimize agent context windows.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>collect_runs()</h4>
        <p>A context manager from langchain_core used to synchronously capture active run traces for automated testing and evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Asynchronous Tracing</h4>
        <p>Non-blocking telemetry export ensuring zero throughput penalty on user-facing agent response streaming.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand how to enable and configure LangSmith via environment variables.", value=True)
chk_2 = st.checkbox("I know how to structure run trace hierarchies across supersteps and nodes.", value=True)
chk_3 = st.checkbox("I can monitor and audit prompt, completion, and total token usage across agent loops.", value=True)
chk_4 = st.checkbox("I understand how to identify latency bottlenecks and analyze Time to First Token (TTFT).", value=True)
chk_5 = st.checkbox("I know how to capture, diagnose, and debug errors and exceptions in LangSmith.", value=True)
chk_6 = st.checkbox("I can enrich traces with custom metadata, tags, and thread IDs for production auditing.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangSmith observability and production tracing mastery confirmed! Ready for testing methodologies.")
