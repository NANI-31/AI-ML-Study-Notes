import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="36.1 Tracing Mistakes - LangSmith Master Curriculum",
    page_icon="⚠️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class TracingMistakeDiagnosticEngine:
    """
    Diagnostic Triage & Hygiene Linter Engine for LangSmith Tracing Mistakes.
    Detects and auto-remediates the 5 major production tracing anti-patterns:
    1. Missing Traces (Broken callback propagation, uninstrumented async threads)
    2. Incorrect Projects (Default project sprawl, staging bleeding into prod)
    3. Missing Metadata (Anonymous spans, missing tenant/session/version keys)
    4. Poor Trace Organization (Flat hierarchy sprawl, missing parent-child links)
    5. Excessive Tracing (Tracing tight in-memory loops, payload bloat)
    """
    def __init__(self):
        self.diagnostic_history: List[Dict[str, Any]] = []

    def diagnose_trace_submission(
        self,
        project_name: str,
        has_parent_run_id: bool,
        metadata_keys: List[str],
        span_count: int,
        is_async_thread_boundary: bool = False,
        callbacks_passed: bool = True
    ) -> Dict[str, Any]:
        """
        Analyzes a simulated trace emission pattern for operational anti-patterns and suggests fixes.
        """
        detected_mistakes = []
        remediations = []
        severity = "HEALTHY"

        # Check 1: Missing Traces / Broken Callback Propagation
        if is_async_thread_boundary and not callbacks_passed:
            detected_mistakes.append({
                "mistake": "Missing Traces (Broken Thread Propagation)",
                "category": "Data Loss",
                "severity": "CRITICAL",
                "details": "Callbacks/contextvars not passed across async worker thread boundary. Child spans dropped."
            })
            remediations.append("Pass RunnableConfig or active RunTree explicitly across thread boundaries.")
            severity = "CRITICAL"

        # Check 2: Incorrect Projects / Default Sprawl
        if project_name.lower() in ["default", "my-project", "test"]:
            detected_mistakes.append({
                "mistake": "Incorrect Project (Default Project Sprawl)",
                "category": "Namespace Pollution",
                "severity": "HIGH",
                "details": f"Traces routed to generic '{project_name}' project. Pollutes production SLAs and ruins cost attribution."
            })
            remediations.append("Set explicit environment prefix (e.g., 'prod-customer-rag' or 'dev-sandbox').")
            if severity != "CRITICAL":
                severity = "HIGH"

        # Check 3: Missing Metadata / Anonymous Spans
        required_keys = ["tenant_id", "session_id", "app_version"]
        missing_required = [k for k in required_keys if k not in metadata_keys]
        if missing_required:
            detected_mistakes.append({
                "mistake": "Missing Mandatory Metadata",
                "category": "Observability Gap",
                "severity": "MEDIUM",
                "details": f"Spans lack essential context keys: {missing_required}. Cannot filter by customer or debug specific sessions."
            })
            remediations.append(f"Inject mandatory metadata keys via middleware: {missing_required}.")
            if severity == "HEALTHY":
                severity = "MEDIUM"

        # Check 4: Poor Trace Organization / Flat Hierarchy
        if not has_parent_run_id and span_count > 10:
            detected_mistakes.append({
                "mistake": "Poor Trace Organization (Flat Hierarchy Sprawl)",
                "category": "Structural Chaos",
                "severity": "MEDIUM",
                "details": f"{span_count} child operations emitted as disjoint root runs rather than a nested trace tree."
            })
            remediations.append("Wrap multi-step workflows in a root `@traceable(run_type='chain')` to form a unified tree.")
            if severity == "HEALTHY":
                severity = "MEDIUM"

        # Check 5: Excessive Tracing / Payload Bloat
        if span_count > 500:
            detected_mistakes.append({
                "mistake": "Excessive Tracing (Loop Overhead & Throttling)",
                "category": "Performance Degradation",
                "severity": "HIGH",
                "details": f"Generated {span_count} spans for a single request. Likely instrumenting tight loops or trivial string helpers."
            })
            remediations.append("Remove `@traceable` from internal loops; restrict tracing to network/IO boundaries.")
            if severity != "CRITICAL":
                severity = "HIGH"

        report = {
            "status": "APPROVED" if len(detected_mistakes) == 0 else "REJECTED_ANTIPATTERNS_FOUND",
            "overall_severity": severity,
            "mistakes_detected": detected_mistakes,
            "remediation_actions": remediations,
            "metrics": {
                "project_name": project_name,
                "span_count": span_count,
                "metadata_key_count": len(metadata_keys),
                "is_nested_tree": has_parent_run_id
            }
        }
        self.diagnostic_history.append(report)
        return report

# Initialize engine in session state
if "tracing_mistakes_engine" not in st.session_state:
    st.session_state.tracing_mistakes_engine = TracingMistakeDiagnosticEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("⚠️ Chapter 36.1: Common LangSmith Tracing Mistakes")
st.caption("Broken Callbacks, Project Pollution, Anonymous Spans, Flat Hierarchy Sprawl, and Excessive Tracing Overhead")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Common Tracing Anti-Patterns and Diagnostic Triage Architecture", divider="rainbow")

st.markdown("""
LangSmith provides seamless tracing out of the box, but subtle architectural misconceptions can undermine 
observability. Common mistakes range from broken context propagation dropping child spans, to project name collisions 
polluting production analytics, to missing metadata leaving traces anonymous and unsearchable. 
Understanding these anti-patterns is essential to building resilient observability pipelines.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph TracingMistakesArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_mistakes {
            label = "1. The 5 Major Tracing Anti-Patterns";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F43F5E";
            fontsize = 12;

            M1 [label="Anti-Pattern 1: Missing Traces\\n(Broken callbacks in async thread pools)", fillcolor="#7C2D12", color="#EA580C"];
            M2 [label="Anti-Pattern 2: Incorrect Projects\\n('default' project sprawl / staging in prod)", fillcolor="#7C2D12", color="#EA580C"];
            M3 [label="Anti-Pattern 3: Missing Metadata\\n(Zero tenant_id / session_id; anonymous)", fillcolor="#7C2D12", color="#EA580C"];
            M4 [label="Anti-Pattern 4: Poor Organization\\n(Flat disjoint spans / orphan children)", fillcolor="#7C2D12", color="#EA580C"];
            M5 [label="Anti-Pattern 5: Excessive Tracing\\n(Tracing tight loops / 1,000s spans per req)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_impact {
            label = "2. Production Operational Impact";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            I_BLIND [label="Observability Blackouts\\n(Zero traces for background worker failures)", fillcolor="#881337", color="#F43F5E"];
            I_POLL [label="Metric Corruption\\n(Dev experiments skewing prod latency SLAs)", fillcolor="#881337", color="#F43F5E"];
            I_TRIAGE [label="Unsearchable Production Incidents\\n(Cannot find customer trace from ticket)", fillcolor="#881337", color="#F43F5E"];
            I_OVER [label="Network & Rate Limit Throttling\\n(LangSmith SDK 429 drop / memory bloat)", fillcolor="#881337", color="#F43F5E"];
        }

        subgraph cluster_remediation {
            label = "3. Enterprise Remediation Architecture";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            R_CONFIG [label="Explicit RunnableConfig & ContextVars\\n(Pass config={'callbacks': ...} to threads)", fillcolor="#065F46", color="#059669"];
            R_ENV [label="Environment-Prefixed Projects\\n(LANGCHAIN_PROJECT=prod-customer-service)", fillcolor="#065F46", color="#059669"];
            R_SCHEM [label="Enforced Metadata Middleware\\n(Inject tenant_id, session_id, app_version)", fillcolor="#065F46", color="#059669"];
            R_TREE [label="Unified Root Run Hierarchies\\n(@traceable(run_type='chain') wrapper)", fillcolor="#065F46", color="#059669"];
            R_BOUND [label="Restricted Span Boundaries\\n(Only instrument LLM, Retriever, Tool boundaries)", fillcolor="#065F46", color="#059669"];
        }

        M1 -> I_BLIND [label="Causes"];
        M2 -> I_POLL [label="Causes"];
        M3 -> I_TRIAGE [label="Causes"];
        M4 -> I_TRIAGE [label="Causes"];
        M5 -> I_OVER [label="Causes"];

        I_BLIND -> R_CONFIG [label="Resolved By"];
        I_POLL -> R_ENV [label="Resolved By"];
        I_TRIAGE -> R_SCHEM [label="Resolved By"];
        I_TRIAGE -> R_TREE [label="Resolved By"];
        I_OVER -> R_BOUND [label="Resolved By"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Tracing Failure Modes
1. **Never Assume Async Threads Inherit Tracing Context Automatically**: In Python, asynchronous tasks (e.g., `asyncio.to_thread` or thread pools) lose thread-local `ContextVar` state unless explicitly propagated. This causes downstream tool calls to vanish from LangSmith traces.
2. **The 'default' Project is an Operational Hazard**: Leaving `LANGCHAIN_PROJECT` unset defaults to `default`. In teams with multiple engineers, test runs instantly pollute production projects, skewing token costs and latency SLAs.
3. **An Anonymous Trace is Nearly Useless During an Incident**: If an enterprise customer reports an erroneous response, searching 500,000 daily traces without `tenant_id` or `session_id` metadata is searching for a needle in a haystack.
""")

st.divider()

# ==============================================================================
# SECTION 2: MISSING TRACES
# ==============================================================================
st.header("2. Missing Traces (Broken Callback Propagation, ContextVar Thread Loss and Uninstrumented Lambdas)", divider="rainbow")

st.markdown("""
One of the most frustrating experiences in LLM engineering is looking at a LangSmith trace and seeing only the top-level 
chain, with all child tool invocations, retriever calls, or sub-agent steps missing. 
This occurs when the tracing context fails to propagate across execution boundaries.
""")

st.markdown("""
### The 3 Common Root Causes of Dropped Traces
* **Thread Pool Boundary Loss**: In Python, `concurrent.futures.ThreadPoolExecutor` runs in a distinct OS thread that does NOT inherit Python's `contextvars` by default. LangSmith's `@traceable` relies on `ContextVar` to link child spans to parent runs.
* **Missing `config` Parameter in LangChain Runnables**: When invoking a sub-chain or custom function inside a `RunnableLambda`, forgetting to pass the `config` argument (`sub_chain.invoke(inputs, config=config)`) severs the callback chain.
* **Uninstrumented Raw SDK Calls**: Calling `openai.OpenAI()` or `anthropic.Anthropic()` directly without the `wrap_openai()` patch or without `@traceable` creates an invisible gap in the trace waterfall.
""")

st.code('''
# ANTI-PATTERN: Context severed across thread pool
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable

@traceable(name="tool_execution")
def run_tool_worker(item):
    # This child span loses its parent context! Appears as an orphan root run!
    return external_api_call(item)

# PRODUCTION REMEDY: Explicitly propagate RunTree or RunnableConfig
from langsmith.run_trees import RunTree

@traceable(name="parent_pipeline")
def safe_parallel_execution(items):
    # Capture active run context
    parent_run = RunTree.get_current()
    
    with ThreadPoolExecutor() as executor:
        # Pass parent run to worker threads
        futures = [executor.submit(run_tool_worker_with_context, item, parent_run) for item in items]
        return [f.result() for f in futures]
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern LCEL and LangGraph, stateful execution graphs automatically manage context propagation across nodes. 
However, whenever custom Python functions or thread pools are used inside nodes, engineers must ensure the `RunnableConfig` 
is explicitly passed downstream: `my_tool.invoke(args, config=config)`.
""")

st.divider()

# ==============================================================================
# SECTION 3: INCORRECT PROJECTS
# ==============================================================================
st.header("3. Incorrect Projects (Project Name Pollution, Staging Bleeding into Prod and Default Project Sprawl)", divider="rainbow")

st.markdown("""
LangSmith projects are top-level organizational containers designed to isolate environments, compliance boundaries, 
and application domains. Routing traces to the wrong project causes metric pollution, misleading cost reports, 
and compromised access control.
""")

st.markdown("""
### The Dangers of Project Pollution
1. **The 'default' Project Sprawl**: Leaving `LANGCHAIN_PROJECT` unset silently dumps all traces into `default`. Developer laptop tests, automated CI test runs, and live user queries intermingle, corrupting all latency percentiles.
2. **Staging Bleeding into Production**: Developers accidentally using production API keys in local development, or forgetting to set `LANGCHAIN_PROJECT=dev-*`, injects synthetic test queries into production analytics.
3. **Retention Policy Mismatch**: Production projects typically require 90-day retention for audit compliance, whereas development sandboxes only need 7-day retention. Mixing them inflates storage costs.
""")

project_mistake_df = pd.DataFrame([
    {"Scenario": "Unset Environment Variable", "Configured Project": "default", "Severity": "HIGH", "Consequence": "Zero environment isolation; corrupts production latency percentiles"},
    {"Scenario": "Hardcoded Project in Code", "Configured Project": "prod-customer-service", "Severity": "CRITICAL", "Consequence": "Local unit tests and staging QA runs pollute production project"},
    {"Scenario": "Correct Enterprise Config", "Configured Project": "prod-customer-service-rag", "Severity": "COMPLIANT", "Consequence": "Strict environment isolation with automated retention policies"}
])
st.dataframe(project_mistake_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Never hardcode project names in source code. Inject `LANGCHAIN_PROJECT` at runtime through environment manifests:
Kubernetes ConfigMaps for production (`prod-app-name`) and `.env` files for local development (`dev-username-sandbox`).
""")

st.divider()

# ==============================================================================
# SECTION 4: MISSING METADATA
# ==============================================================================
st.header("4. Missing Metadata (Anonymous Spans, Inability to Filter by User/Tenant and Unreproducible Errors)", divider="rainbow")

st.markdown("""
A trace without metadata is an anonymous black box. In production, knowing *what* happened is only half the battle; 
you must know *who* it happened to, in *what* session, on *what* software version, and under *what* tenant configuration.
""")

st.markdown("""
### The Operational Cost of Missing Metadata
* **Inability to Triage Customer Tickets**: When customer support receives an escalation ("User Alice received an ungrounded response at 2:15 PM"), support engineers cannot query by `metadata.user_id = "Alice"`.
* **Zero Cost Allocation by Tenant**: SaaS companies cannot bill enterprise customers for their exact token usage without a `tenant_id` metadata tag.
* **Unreproducible Incidents**: Without recording `app_version`, `prompt_commit`, and active feature flags, engineers cannot reproduce subtle production bugs that occurred under specific configurations.
""")

st.code('''
# ANTI-PATTERN: Completely anonymous root span
@traceable(name="chat_handler")
def handle_chat_bad(query: str):
    return pipeline.invoke(query)

# PRODUCTION REMEDY: Fully contextualized metadata injection
@traceable(
    name="chat_handler",
    metadata={
        "tenant_id": request.headers.get("X-Tenant-ID"),
        "user_id": current_user.id,
        "session_id": session.id,
        "app_version": "v2.4.1",
        "git_commit": "7e81b2a",
        "feature_flag_reranker": True
    }
)
def handle_chat_good(query: str):
    return pipeline.invoke(query)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise web frameworks (FastAPI, Django) use middleware to automatically inject HTTP request headers 
(e.g., `X-Tenant-ID`, `X-Session-ID`) into the root LangSmith tracing context, ensuring zero traces are emitted without identity context.
""")

st.divider()

# ==============================================================================
# SECTION 5: POOR TRACE ORGANIZATION
# ==============================================================================
st.header("5. Poor Trace Organization (Flat Hierarchy Sprawl, Orphan Spans and Missing Parent-Child Links)", divider="rainbow")

st.markdown("""
LangSmith's primary strength is visualizing multi-step execution graphs: a user query triggers an agent, 
which calls a retriever, which calls an embedding model, which triggers a tool, which synthesizes a response. 
Poor trace organization shatters this tree into a flat, disconnected list of hundreds of disjoint spans.
""")

st.markdown("""
### The Anatomy of Flat Hierarchy Sprawl
* **Orphan Spans**: When child tools are decorated with `@traceable` but the orchestrating function is NOT decorated, each tool execution appears as an isolated top-level root run in LangSmith.
* **Shattered Latency Waterfalls**: Instead of seeing a single 3.2-second root trace with clear child breakdowns, engineers see 12 disconnected spans scattered across the project list with no parent link.
* **Loss of Cost Attribution**: Token usage from disconnected child calls cannot be rolled up to calculate total request cost.
""")

org_diff_df = pd.DataFrame([
    {"Tracing Structure": "Flat Hierarchy Sprawl (Anti-Pattern)", "Root Runs per Request": "10-15 Disjoint Spans", "Gantt Waterfall": "Broken / Fragmented", "Total Cost Rollup": "Impossible (Calculated individually)"},
    {"Tracing Structure": "Unified Nested Tree (Best Practice)", "Root Runs per Request": "Exactly 1 Root Run", "Gantt Waterfall": "Complete visual timeline", "Total Cost Rollup": "Automatic (Summed to root run)"}
])
st.dataframe(org_diff_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Always define a single top-level orchestrator decorated with `@traceable(run_type="chain")`. 
All downstream retrievers, tools, and models will automatically nest beneath this root run, creating a unified trace tree.
""")

st.divider()

# ==============================================================================
# SECTION 6: EXCESSIVE TRACING
# ==============================================================================
st.header("6. Excessive Tracing (Unbounded Overhead, High Latency in Tight Loops and Ingestion Rate Limit Throttling)", divider="rainbow")

st.markdown("""
While missing traces is dangerous, excessive tracing introduces significant performance and stability risks. 
Decorating high-frequency internal loops with `@traceable` generates thousands of spans per request, causing client-side 
memory bloat, thread contention, and LangSmith API rate-limit throttling (`HTTP 429`).
""")

st.markdown("""
### The Traps of Over-Instrumentation
1. **Instrumenting In-Memory Tight Loops**: Wrapping a token parser or regex tokenizer executed 10,000 times in a loop. Tracing overhead (span object allocation, timestamps, queue dispatch) dwarfs the execution time of the function.
2. **Ingestion Rate Limit Throttling**: Emitting 2,000 spans per second triggers client-side queue overflow or HTTP 429 throttling from the LangSmith ingestion endpoint, causing traces to be dropped.
3. **Massive Payloads (Context Bloat)**: Passing massive multi-megabyte binary payloads (images, raw PDF byte strings) through traced inputs and outputs causes slow serialization and ballooning storage costs.
""")

st.code('''
# SEVERE ANTI-PATTERN: Tracing inside a tight loop
@traceable(name="tokenize_word")
def tokenize_word(word: str):
    # Generates 5,000 spans for a single paragraph! Extreme overhead!
    return word.lower().strip()

# PRODUCTION REMEDY: Instrument at the batch boundary
@traceable(name="preprocess_document_batch", run_type="chain")
def preprocess_document_batch(document_text: str):
    # Traced once per document; internal loop runs in pure memory
    tokens = [w.lower().strip() for w in document_text.split()]
    return tokens
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production guidelines dictate: *Only trace I/O boundaries.* 
Never trace functions that execute entirely in CPU memory without external network calls, file system access, or LLM inference.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Tracing Anti-Pattern Detector and Hygiene Linter (TracingMistakeDiagnosticEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the diagnostic linter. 
It analyzes simulated span submissions, flags missing context, catches project naming sprawl, identifies broken thread 
propagation, and recommends automated remediation actions.
""")

st.code('''
class TracingMistakeDiagnosticEngine:
    """
    Diagnostic Triage & Hygiene Linter Engine for LangSmith Tracing Mistakes.
    Detects and auto-remediates the 5 major production tracing anti-patterns:
    1. Missing Traces (Broken callback propagation, uninstrumented async threads)
    2. Incorrect Projects (Default project sprawl, staging bleeding into prod)
    3. Missing Metadata (Anonymous spans, missing tenant/session/version keys)
    4. Poor Trace Organization (Flat hierarchy sprawl, missing parent-child links)
    5. Excessive Tracing (Tracing tight in-memory loops, payload bloat)
    """
    def __init__(self):
        self.diagnostic_history: List[Dict[str, Any]] = []

    def diagnose_trace_submission(
        self,
        project_name: str,
        has_parent_run_id: bool,
        metadata_keys: List[str],
        span_count: int,
        is_async_thread_boundary: bool = False,
        callbacks_passed: bool = True
    ) -> Dict[str, Any]:
        """
        Analyzes a simulated trace emission pattern for operational anti-patterns and suggests fixes.
        """
        detected_mistakes = []
        remediations = []
        severity = "HEALTHY"

        # Check 1: Missing Traces / Broken Callback Propagation
        if is_async_thread_boundary and not callbacks_passed:
            detected_mistakes.append({
                "mistake": "Missing Traces (Broken Thread Propagation)",
                "category": "Data Loss",
                "severity": "CRITICAL",
                "details": "Callbacks/contextvars not passed across async worker thread boundary. Child spans dropped."
            })
            remediations.append("Pass RunnableConfig or active RunTree explicitly across thread boundaries.")
            severity = "CRITICAL"

        # Check 2: Incorrect Projects / Default Sprawl
        if project_name.lower() in ["default", "my-project", "test"]:
            detected_mistakes.append({
                "mistake": "Incorrect Project (Default Project Sprawl)",
                "category": "Namespace Pollution",
                "severity": "HIGH",
                "details": f"Traces routed to generic '{project_name}' project. Pollutes production SLAs and ruins cost attribution."
            })
            remediations.append("Set explicit environment prefix (e.g., 'prod-customer-rag' or 'dev-sandbox').")
            if severity != "CRITICAL":
                severity = "HIGH"

        # Check 3: Missing Metadata / Anonymous Spans
        required_keys = ["tenant_id", "session_id", "app_version"]
        missing_required = [k for k in required_keys if k not in metadata_keys]
        if missing_required:
            detected_mistakes.append({
                "mistake": "Missing Mandatory Metadata",
                "category": "Observability Gap",
                "severity": "MEDIUM",
                "details": f"Spans lack essential context keys: {missing_required}. Cannot filter by customer or debug specific sessions."
            })
            remediations.append(f"Inject mandatory metadata keys via middleware: {missing_required}.")
            if severity == "HEALTHY":
                severity = "MEDIUM"

        # Check 4: Poor Trace Organization / Flat Hierarchy
        if not has_parent_run_id and span_count > 10:
            detected_mistakes.append({
                "mistake": "Poor Trace Organization (Flat Hierarchy Sprawl)",
                "category": "Structural Chaos",
                "severity": "MEDIUM",
                "details": f"{span_count} child operations emitted as disjoint root runs rather than a nested trace tree."
            })
            remediations.append("Wrap multi-step workflows in a root `@traceable(run_type='chain')` to form a unified tree.")
            if severity == "HEALTHY":
                severity = "MEDIUM"

        # Check 5: Excessive Tracing / Payload Bloat
        if span_count > 500:
            detected_mistakes.append({
                "mistake": "Excessive Tracing (Loop Overhead & Throttling)",
                "category": "Performance Degradation",
                "severity": "HIGH",
                "details": f"Generated {span_count} spans for a single request. Likely instrumenting tight loops or trivial string helpers."
            })
            remediations.append("Remove `@traceable` from internal loops; restrict tracing to network/IO boundaries.")
            if severity != "CRITICAL":
                severity = "HIGH"

        return {
            "status": "APPROVED" if len(detected_mistakes) == 0 else "REJECTED_ANTIPATTERNS_FOUND",
            "overall_severity": severity,
            "mistakes_detected": detected_mistakes,
            "remediation_actions": remediations,
            "metrics": {
                "project_name": project_name,
                "span_count": span_count,
                "metadata_key_count": len(metadata_keys),
                "is_nested_tree": has_parent_run_id
            }
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Tracing Anti-Pattern Scanner and Auto-Remediation Sandbox", divider="rainbow")

st.markdown("""
Test how the Tracing Anti-Pattern Scanner inspects trace telemetry configurations. 
Simulate broken callbacks, default project sprawl, anonymous metadata, and excessive span loops to verify detection.
""")

wb_c1, wb_c2 = st.columns([1, 2])

with wb_c1:
    st.markdown("### Trace Submission Parameters")
    target_project = st.selectbox(
        "Project Target Name",
        ["prod-support-copilot", "default", "my-project", "dev-sandbox-alice"],
        index=0
    )
    thread_boundary_toggle = st.toggle(
        "Dispatches Across Async Thread Pool",
        value=False,
        help="Simulates worker threads executed via ThreadPoolExecutor."
    )
    callbacks_propagated = st.toggle(
        "Pass Callbacks / ContextVars to Thread",
        value=True,
        help="Whether active RunTree or RunnableConfig is passed into thread."
    )
    has_parent_link = st.toggle(
        "Has Parent Run ID (Nested Tree)",
        value=True,
        help="Unchecking simulates flat disjoint hierarchy where children have no parent."
    )
    meta_keys_preset = st.selectbox(
        "Metadata Payload Preset",
        [
            "Full Enterprise Keys (tenant_id, session_id, app_version)",
            "Partial Context (session_id only)",
            "Empty Metadata (Anonymous Spans)"
        ],
        index=0
    )
    simulated_span_count = st.slider(
        "Spans Generated per Single Request",
        min_value=2,
        max_value=1000,
        value=8,
        step=50,
        help="Values > 500 simulate excessive tracing in tight loops."
    )

with wb_c2:
    st.markdown("### Live Linter Diagnostic Inspection Report")

    # Map metadata selection
    if "Full" in meta_keys_preset:
        active_meta = ["tenant_id", "session_id", "app_version", "git_commit"]
    elif "Partial" in meta_keys_preset:
        active_meta = ["session_id"]
    else:
        active_meta = []

    diag_report = st.session_state.tracing_mistakes_engine.diagnose_trace_submission(
        project_name=target_project,
        has_parent_run_id=has_parent_link,
        metadata_keys=active_meta,
        span_count=simulated_span_count,
        is_async_thread_boundary=thread_boundary_toggle,
        callbacks_passed=callbacks_propagated
    )

    if diag_report["status"] == "APPROVED":
        st.success(f"**STATUS: {diag_report['status']}** — Tracing configuration conforms to production standards.")
    else:
        st.error(f"**STATUS: {diag_report['status']} (Severity: {diag_report['overall_severity']})** — Anti-patterns detected!")

    # Detected Mistakes Table
    if diag_report["mistakes_detected"]:
        st.markdown("#### Detected Anti-Patterns")
        mistakes_df = pd.DataFrame(diag_report["mistakes_detected"])
        st.dataframe(mistakes_df, use_container_width=True)

        st.markdown("#### Automated Remediation Recommendations")
        for rem in diag_report["remediation_actions"]:
            st.info(f"💡 **Fix:** {rem}")
    else:
        st.markdown("All spans properly nested with full metadata and clean environment isolation.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

q_exp4 = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with q_exp4:
    st.markdown("""
    **Q1: Why do child spans disappear from LangSmith traces when executing in Python ThreadPoolExecutor?**  
    *Answer:* Python's `ThreadPoolExecutor` spawns separate operating system threads that do not automatically inherit the active thread's `contextvars` context. Because LangSmith's `@traceable` decorator relies on `ContextVar` to identify the active parent run ID, functions executed in worker threads fail to find a parent run and are either dropped or emitted as disjoint orphan root traces. To fix this, explicitly pass the active `RunTree` or `RunnableConfig` into worker threads.
    """)

    st.markdown("""
    **Q2: What are the consequences of letting applications default to the 'default' LangSmith project?**  
    *Answer:* The `default` project becomes an unmanageable catch-all where local developer laptop runs, automated CI/CD canary tests, and live production customer requests intermingle. This corrupts production latency percentiles (p95/p99), makes error alerting unreliable, breaks tenant cost accounting, and creates security risks by exposing customer data in shared development views.
    """)

    st.markdown("""
    **Q3: How does 'flat hierarchy sprawl' impact the debugging experience in LangSmith?**  
    *Answer:* When multi-step workflows lack an orchestrating parent run, each child retriever, LLM call, and tool appears as an isolated top-level trace in LangSmith. Instead of an intuitive waterfall showing a single 3-second workflow with child timings, engineers see dozens of fragmented spans with no visual link, making it impossible to determine which user query triggered which tool call.
    """)

    st.markdown("""
    **Q4: What causes excessive tracing overhead, and what is the rule of thumb for span scoping?**  
    *Answer:* Excessive tracing occurs when developers place `@traceable` on high-frequency in-memory functions (e.g., tokenizers, regex cleaners, or string splits executed inside tight loops). This introduces Python runtime overhead, thread contention in the background trace dispatcher, and triggers HTTP 429 throttling from LangSmith. Rule of thumb: *Only trace I/O boundaries* (LLM calls, database queries, external API tools, and top-level chain entry points).
    """)

    st.markdown("""
    **Q5: What minimum set of metadata keys must be attached to every production trace?**  
    *Answer:* At minimum: 1) `tenant_id` (for multi-tenant cost allocation and data isolation); 2) `session_id` (for reconstructing multi-turn conversations); 3) `app_version` / `git_commit` (for version lineage and regression tracking); and 4) `environment` (`prod`, `staging`, `canary`).
    """)

st.markdown("### Production Traps and Anti-Patterns")

tc_col1, tc_col2 = st.columns(2)

with tc_col1:
    st.error("""
    **Trap 1: Hardcoding Project Names in Code**
    * **Failure**: Setting `os.environ["LANGCHAIN_PROJECT"] = "prod-app"` directly inside Python application code.
    * **Consequence**: When the code runs in local development or CI unit tests, it writes synthetic test traces into the production project container.
    * **Remedy**: Always configure project names via external deployment manifests (Kubernetes ConfigMaps, `.env`).
    """)

    st.error("""
    **Trap 2: Tracing In-Memory String Tokenizers**
    * **Failure**: Decorating a word-level token cleaning function with `@traceable`.
    * **Consequence**: A single 500-word paragraph generates 500 individual spans, crashing performance and filling LangSmith with noise.
    * **Remedy**: Decorate the high-level batch preprocessing function, keeping the internal loop in native Python.
    """)

with tc_col2:
    st.warning("""
    **Trap 3: Missing Context in LangChain RunnableLambda**
    * **Failure**: Using `RunnableLambda(my_func)` where `my_func` calls another chain without accepting `config`.
    * **Consequence**: The child chain's callbacks are severed, dropping all downstream model spans from the trace tree.
    * **Remedy**: Always accept `config` in custom runnables and pass it downstream: `sub_chain.invoke(input, config=config)`.
    """)

    st.warning("""
    **Trap 4: Anonymous Traces during Security Incidents**
    * **Failure**: Emitting traces without capturing user or tenant identifiers.
    * **Consequence**: When security requests an audit of an adversarial prompt injection incident, engineers cannot identify which user account was responsible.
    * **Remedy**: Enforce an API gateway middleware that automatically binds authenticated user IDs to trace metadata.
    """)

st.markdown("### Core Technical Flashcards")

fc_b1, fc_b2 = st.columns(2)

with fc_b1:
    with st.container(border=True):
        st.markdown("**Front**: Why does `ThreadPoolExecutor` lose LangSmith tracing context?")
        st.markdown("**Back**: Worker threads do not inherit parent `contextvars` by default; child spans fail to find the active parent run ID and become orphans.")

    with fc_b1:
        with st.container(border=True):
            st.markdown("**Front**: What is the primary operational consequence of using the 'default' project?")
            st.markdown("**Back**: Test scripts, CI runs, and production traffic collide, invalidating latency SLAs and corrupting token cost attribution.")

with fc_b2:
    with st.container(border=True):
        st.markdown("**Front**: What is the 'I/O Boundary Rule' of production tracing?")
        st.markdown("**Back**: Only trace operations that cross network or disk boundaries (LLMs, retrievers, tools); never trace tight in-memory loops.")

    with fc_b2:
        with st.container(border=True):
            st.markdown("**Front**: How do you avoid 'flat hierarchy sprawl' in multi-step workflows?")
            st.markdown("**Back**: Wrap the orchestrating function in a top-level `@traceable(run_type='chain')` so all child operations nest under a single root run.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of avoiding Common LangSmith Tracing Mistakes:")

tm_chk1 = st.checkbox("I know how to preserve tracing context across async thread boundaries using explicit RunTree propagation.", value=True)
tm_chk2 = st.checkbox("I understand why the 'default' project is forbidden in production and how to enforce environment prefixes.", value=True)
tm_chk3 = st.checkbox("I can identify and inject the mandatory metadata keys (tenant_id, session_id, app_version) to prevent anonymous traces.", value=True)
tm_chk4 = st.checkbox("I know how to structure multi-step workflows into unified nested trees to avoid flat hierarchy sprawl.", value=True)
tm_chk5 = st.checkbox("I understand the I/O Boundary Rule to prevent excessive tracing overhead in tight loops.", value=True)
tm_chk6 = st.checkbox("I can implement an automated diagnostic linter to detect tracing anti-patterns prior to deployment.", value=True)

if tm_chk1 and tm_chk2 and tm_chk3 and tm_chk4 and tm_chk5 and tm_chk6:
    st.success("Mastery Certified: You have mastered diagnosing and eliminating common LangSmith tracing mistakes, callback drops, and project sprawl.")
