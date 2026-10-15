import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.2 Runnable Tracing - LangSmith Master Curriculum",
    page_icon="⚡",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class RunnableTracingEngine:
    """
    Production LCEL Runnable Tracing & Orchestration Engine.
    Emulates LangSmith trace collection across RunnableSequence (linear chains),
    RunnableParallel (concurrent scatter-gather), RunnableBranch (conditional routing),
    and RunnableConfig (runtime metadata and tags injection).
    """
    def __init__(self, project_name: str = "lcel-runnable-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 700

    def execute_lcel_workflow(
        self,
        input_query: str,
        execution_mode: str = "parallel",
        intent_category: str = "technical",
        config_tags: Optional[List[str]] = None,
        config_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a simulated LCEL workflow and captures detailed LangSmith trace spans.
        """
        self._span_counter += 1
        trace_id = f"trace_lcel_{self._span_counter}"
        root_run_id = f"run_root_{self._span_counter}"

        config_tags = config_tags or ["production", "v2.4"]
        config_metadata = config_metadata or {"tenant_id": "ORG-ALPHA", "environment": "prod"}

        child_spans = []
        total_latency = 0.0

        if execution_mode == "sequential":
            # 1. Linear RunnableSequence: Step 1 -> Step 2 -> Step 3
            step1_latency = 15.0
            step2_latency = 220.0
            step3_latency = 8.0
            total_latency = step1_latency + step2_latency + step3_latency

            child_spans = [
                {
                    "run_id": f"span_seq_1_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableStep1_InputValidator",
                    "run_type": "chain",
                    "latency_ms": step1_latency,
                    "inputs": {"raw_query": input_query},
                    "outputs": {"clean_query": input_query.strip().lower()}
                },
                {
                    "run_id": f"span_seq_2_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableStep2_LLMInference",
                    "run_type": "llm",
                    "latency_ms": step2_latency,
                    "inputs": {"prompt": f"Process query: {input_query}"},
                    "outputs": {"generation": f"Sequential output for '{input_query}'"}
                },
                {
                    "run_id": f"span_seq_3_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableStep3_OutputParser",
                    "run_type": "parser",
                    "latency_ms": step3_latency,
                    "inputs": {"raw_generation": f"Sequential output for '{input_query}'"},
                    "outputs": {"parsed_result": f"Parsed: {input_query}"}
                }
            ]
            final_output = f"Sequential pipeline executed in {round(total_latency, 1)}ms."

        elif execution_mode == "parallel":
            # 2. RunnableParallel: Fan-out Branch A (Vector Search) and Branch B (Web Search) concurrently
            branch_a_latency = 45.0
            branch_b_latency = 68.0
            # Total latency is bounded by slowest branch + gather overhead
            total_latency = max(branch_a_latency, branch_b_latency) + 4.0

            child_spans = [
                {
                    "run_id": f"span_par_a_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableParallel_BranchA_VectorDB",
                    "run_type": "retriever",
                    "latency_ms": branch_a_latency,
                    "inputs": {"query": input_query},
                    "outputs": {"vector_hits": ["Doc A1: Internal architectural specification."]}
                },
                {
                    "run_id": f"span_par_b_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableParallel_BranchB_WebSearch",
                    "run_type": "tool",
                    "latency_ms": branch_b_latency,
                    "inputs": {"query": input_query},
                    "outputs": {"web_hits": ["Web B1: Public API release announcement."]}
                }
            ]
            final_output = f"Parallel fan-out completed in {round(total_latency, 1)}ms (max branch: {max(branch_a_latency, branch_b_latency)}ms)."

        else:
            # 3. RunnableBranch: Dynamic routing based on intent classification
            classifier_latency = 22.0
            if intent_category == "technical":
                target_route = "TechnicalDocumentationChain"
                route_latency = 180.0
            else:
                target_route = "BillingFaqChain"
                route_latency = 95.0

            total_latency = classifier_latency + route_latency

            child_spans = [
                {
                    "run_id": f"span_branch_class_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": "RunnableBranch_Classifier",
                    "run_type": "chain",
                    "latency_ms": classifier_latency,
                    "inputs": {"text": input_query},
                    "outputs": {"selected_route": target_route}
                },
                {
                    "run_id": f"span_branch_exec_{self._span_counter}",
                    "parent_run_id": root_run_id,
                    "name": target_route,
                    "run_type": "chain",
                    "latency_ms": route_latency,
                    "inputs": {"query": input_query, "route": target_route},
                    "outputs": {"response": f"Routed to specialized handler '{target_route}'."}
                }
            ]
            final_output = f"Conditionally routed to '{target_route}' in {round(total_latency, 1)}ms."

        trace_record = {
            "trace_id": trace_id,
            "project_name": self.project_name,
            "root_run": {
                "run_id": root_run_id,
                "name": f"LCEL_{execution_mode.capitalize()}Orchestrator",
                "run_type": "chain",
                "inputs": {"query": input_query, "mode": execution_mode},
                "outputs": {"result": final_output},
                "total_latency_ms": round(total_latency, 2),
                "tags": config_tags,
                "metadata": config_metadata
            },
            "child_spans": child_spans
        }
        self.active_traces.append(trace_record)
        return trace_record


# --- HERO SECTION ---
st.title("23.2 Runnable Tracing")
st.caption("Chapter 23: LangSmith with LangChain | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
LangChain Expression Language (LCEL) replaced monolithic legacy chains with composable, type-safe primitives known as **Runnables**.
Whether composing linear sequences (`RunnableSequence`), concurrent scatter-gather branches (`RunnableParallel`),
or dynamic conditional routers (`RunnableBranch`), tracing the exact data flow across pipe boundaries is essential for performance engineering.
**Runnable Tracing** in LangSmith provides deep runtime inspection:
1. **The Unified Runnable Protocol**: How `invoke`, `batch`, and `stream` map transparently to LangSmith distributed runs.
2. **RunnableSequence Decomposition**: Tracking inter-stage state mutations across linear multi-pipe workflows.
3. **RunnableParallel Profiling**: Measuring concurrent branch fan-out, barrier synchronization, and resource contention.
4. **RunnableBranch Observability**: Visualizing dynamic condition routing decisions and circuit-breaker fallbacks.
5. **RunnableConfig Injection**: Dynamically propagating runtime metadata, tenant tags, and trace parameters without polluting business logic.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LCEL Runnable Orchestration & Asynchronous Trace Collector", divider="rainbow")

st.markdown("""
The architecture below illustrates the three fundamental execution topologies of LCEL:
**Linear Sequences**, **Parallel Fan-Out**, and **Conditional Branches**.
LangSmith's non-blocking callback handler wraps each runnable primitive into correlated execution spans.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    lcel_blueprint_dot = """
    digraph LCELOrchestration {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_ingress_query {
            label="1. Ingress Request with RunnableConfig";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            UserReq [label="User Query Ingress\nconfig={'metadata': {'tenant': 'ACME'}, 'tags': ['prod']}", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_topology_types {
            label="2. LCEL Core Composition Topologies";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            subgraph cluster_seq {
                label="Pattern A: RunnableSequence (A | B | C)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#10B981";
                fontcolor="#34D399";
                Seq1 [label="PromptTemplate (2ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Seq2 [label="ChatModel (210ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Seq3 [label="StrOutputParser (1ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Seq1 -> Seq2 -> Seq3;
            }

            subgraph cluster_par {
                label="Pattern B: RunnableParallel (Concurrent Fan-Out)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#F59E0B";
                fontcolor="#FCD34D";
                ParSplit [label="RunnableParallel Splitter", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
                ParA [label="Branch A: VectorDB (45ms)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
                ParB [label="Branch B: WebSearch (68ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
                ParJoin [label="Barrier Join: max(45, 68) = 68ms", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
                ParSplit -> ParA -> ParJoin;
                ParSplit -> ParB -> ParJoin;
            }

            subgraph cluster_branch {
                label="Pattern C: RunnableBranch (Conditional Routing)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#EC4899";
                fontcolor="#F472B6";
                Classifier [label="Intent Router (22ms)", shape=diamond, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
                TechChain [label="Technical Chain (180ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
                BillChain [label="Billing FAQ Chain (95ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
                Classifier -> TechChain [label="if 'tech'"];
                Classifier -> BillChain [label="if 'billing'"];
            }
        }

        subgraph cluster_telemetry_hub {
            label="3. LangSmith Centralized Trace Stream";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#A855F7";
            fontcolor="#D8B4FE";
            fontsize=11;

            Collector [label="Async LangChainTracer\nInherits Config Metadata & Tags\nZero Pipeline Blocking", shape=component, fillcolor="#581C87", fontcolor="#F3E8FF", color="#C084FC"];
        }

        UserReq -> Seq1;
        UserReq -> ParSplit;
        UserReq -> Classifier;

        Seq3 -> Collector [style=dotted, color="#C084FC"];
        ParJoin -> Collector [style=dotted, color="#C084FC"];
        TechChain -> Collector [style=dotted, color="#C084FC"];
        BillChain -> Collector [style=dotted, color="#C084FC"];
    }
    """
    st.graphviz_chart(lcel_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: RUNNABLE TRACING
# ==============================================================================
st.header("2. Runnable Tracing (The Runnable Interface, Native Callbacks & Non-Blocking Tracing)", divider="rainbow")

st.markdown("""
The foundation of LCEL observability is that **every Runnable natively accepts callback managers**:
```python
class Runnable:
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        # Executes callbacks: on_chain_start, on_chain_end, on_chain_error
```
When LangSmith is enabled via `LANGSMITH_TRACING=true`, LangChain registers an internal `LangChainTracer` instance in the root execution context.
Every call to `.invoke()`, `.batch()`, or `.stream()` automatically dispatches telemetry events without requiring decorators on every function.
""")

col_rt1, col_rt2 = st.columns([1, 1])

with col_rt1:
    st.markdown("### The 4 Core Runnable Execution Primitives:")
    st.markdown("""
    * **`invoke(input)`**: Synchronous execution returning single output. Captures 1 span from start to finish.
    * **`ainvoke(input)`**: Asynchronous coroutine execution over `asyncio`. Yields non-blocking event loop execution.
    * **`batch([inputs])`**: Concurrently evaluates a list of inputs using an internal thread pool executor. Captures child runs grouped under a batch parent.
    * **`stream(input)`**: Yields generator chunks incrementally over the wire. LangSmith measures **Time-To-First-Token (TTFT)** and total streaming duration.
    """)

with col_rt2:
    st.markdown("""
    ### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
    In LangChain v0.3+, tracing a custom Python function inside an LCEL pipeline is achieved via `RunnableLambda`:
    ```python
    from langchain_core.runnables import RunnableLambda

    def custom_cleaning_step(text: str) -> str:
        return text.strip().replace("\n", " ")

    # Automatically traced as a distinct run span in LangSmith
    cleaner = RunnableLambda(custom_cleaning_step)
    chain = cleaner | prompt | model
    ```
    LangSmith names the span `custom_cleaning_step` and captures its exact input string and sanitized output.
    """)

st.divider()

# ==============================================================================
# SECTION 3: RUNNABLE SEQUENCES
# ==============================================================================
st.header("3. Runnable Sequences (Linear Chains, Sequential Data Passing & Inter-Stage Payloads)", divider="rainbow")

st.markdown("""
A **RunnableSequence** is formed whenever Runnables are composed with the pipe operator:
`sequence = step_1 | step_2 | step_3`
The output of `step_1` is fed directly as the input to `step_2`.

In production debugging, a common failure is **Silent Data Corruption Between Stages**:
for example, `step_1` outputs a dictionary with key `{"context_text": "..."}`, but `step_2` expects `{"context": "..."}`.
In LangSmith, you can click on each individual step in the trace tree to inspect its exact input/output boundary,
instantly identifying where the schema mismatch occurred.
""")

col_seq1, col_seq2 = st.columns([1, 1])

with col_seq1:
    st.markdown("### RunnableSequence Latency Accumulation:")
    st.markdown("""
    Because RunnableSequences execute sequentially, total latency is the strict sum of all constituent steps:
    `T_total = t_step1 + t_step2 + t_step3 + overhead`
    If a 3-step sequence takes 1,850ms, inspecting the LangSmith run tree immediately pinpoints which stage contributed the delay
    (e.g., 20ms in prompt formatting, 1,780ms in LLM inference, 50ms in output parsing).
    """)

with col_seq2:
    st.markdown("### Sequential Data Passing Trace Lineage")
    seq_flow_df = pd.DataFrame([
        {"Step": "Step 1: Input Validator", "Input": "{'raw': '  SSN: 000-11-2222  '}", "Output": "{'clean': '000-11-2222'}", "Latency": "15ms"},
        {"Step": "Step 2: LLM Masker", "Input": "{'clean': '000-11-2222'}", "Output": "{'masked': '[REDACTED_SSN]'}", "Latency": "220ms"},
        {"Step": "Step 3: Output Parser", "Input": "{'masked': '[REDACTED_SSN]'}", "Output": "'[REDACTED_SSN]'", "Latency": "8ms"}
    ])
    st.dataframe(seq_flow_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: RUNNABLE PARALLEL
# ==============================================================================
st.header("4. Runnable Parallel (Concurrent Fan-Out, Scatter-Gather Latency & Resource Contention)", divider="rainbow")

st.markdown("""
`RunnableParallel` (or mapping dictionaries in LCEL) executes multiple independent runnables concurrently on separate threads:
```python
map_chain = RunnableParallel({
    "context": retriever,
    "summary": summary_model,
    "user_profile": fetch_user_profile
})
```
In LangSmith, `RunnableParallel` creates a **Scatter-Gather Run Tree**:
all child spans share the exact same start timestamp.
The total duration of the parallel step is bounded by the **single slowest branch**:
`T_parallel = max(t_branch_1, t_branch_2, t_branch_3) + thread_overhead`.
""")

col_rp1, col_rp2 = st.columns([1, 1])

with col_rp1:
    st.markdown("### Parallel Concurrency Metrics in LangSmith:")
    st.markdown("""
    * **Scatter-Gather Synchronization**: Inspecting the waterfall timeline confirms that Branch A (45ms) and Branch B (68ms) executed concurrently rather than serially.
    * **Thread Pool Saturation**: If 20 users invoke a 4-branch parallel chain simultaneously, an 8-thread worker pool becomes saturated, causing branch start timestamps to skew.
    * **Early Termination & Exceptions**: If Branch A throws a network exception, LangSmith marks Branch A red while tracking whether Branch B completed or was cancelled.
    """)

with col_rp2:
    st.markdown("### Parallel Execution Timeline")
    dot_par_timeline = """
    digraph ParallelTimeline {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Start [label="T=0ms: RunnableParallel Dispatched", shape=ellipse, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        B1 [label="Branch 1: Dense Vector Search (45ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        B2 [label="Branch 2: Web Serper API (68ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        B3 [label="Branch 3: SQL Customer Profile (18ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Barrier [label="T=72ms: Barrier Join Complete\nDuration = max(45, 68, 18) + 4ms", shape=ellipse, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        Start -> B1;
        Start -> B2;
        Start -> B3;
        B1 -> Barrier;
        B2 -> Barrier;
        B3 -> Barrier;
    }
    """
    st.graphviz_chart(dot_par_timeline, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: RUNNABLE BRANCHES
# ==============================================================================
st.header("5. Runnable Branches (Dynamic Routing, Conditional Execution Spans & Fallback Circuits)", divider="rainbow")

st.markdown("""
`RunnableBranch` (or `.with_fallbacks()`) provides conditional routing in LCEL:
evaluating a predicate condition and directing input to specialized downstream chains.
```python
branch = RunnableBranch(
    (lambda x: "code" in x["query"], code_specialist_chain),
    (lambda x: "legal" in x["query"], legal_specialist_chain),
    general_support_chain  # default fallback
)
```
In LangSmith, tracing a `RunnableBranch` captures:
1. The **Classifier / Predicate Run**: The exact boolean evaluation or classification label.
2. The **Selected Target Route**: Verifies whether the query was routed to the intended specialist or inadvertently fell back to the default handler.
""")

col_rb1, col_rb2 = st.columns([1, 1])

with col_rb1:
    st.markdown("### Common Branch Routing Bugs Caught in LangSmith:")
    st.markdown("""
    * **Classifier Flaw**: An ambiguous question (*"What is the legal code for corporate refunds?"*) triggering the wrong specialist branch.
    * **Silent Default Fallback Overload**: 95% of queries triggering the default fallback branch because the predicate condition syntax had a subtle bug.
    * **Fallback Latency Cascades**: When `primary_chain.with_fallbacks([backup_chain])` triggers, the trace displays both the primary failure span (e.g., HTTP 504) and the successful backup span, explaining why total latency spiked from 300ms to 2,800ms.
    """)

with col_rb2:
    st.markdown("### Conditional Branch Execution Hierarchy")
    dot_branch_hier = """
    digraph BranchHierarchy {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        BranchRoot [label="Parent: RunnableBranch Router\n(run_type: chain)", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        Classifier [label="Child 1: IntentClassifier\nEvaluates: 'technical' (22ms)", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        TechRoute [label="Child 2: TechnicalDocumentationChain\n(Executed Route: 180ms)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        BypassedRoute [label="Bypassed: BillingFaqChain\n(0ms - Not Executed)", shape=box, fillcolor="#1E293B", fontcolor="#94A3B8", style="dotted"];

        BranchRoot -> Classifier;
        Classifier -> TechRoute [label="Condition Match", color="#10B981"];
        Classifier -> BypassedRoute [label="Unmatched", style=dotted, color="#94A3B8"];
    }
    """
    st.graphviz_chart(dot_branch_hier, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: RUNNABLE CONFIGURATION
# ==============================================================================
st.header("6. Runnable Configuration (Runtime Configs, Dynamic Metadata, Tags & Callbacks Injection)", divider="rainbow")

st.markdown("""
Every Runnable invocation accepts a `RunnableConfig` dictionary:
```python
config = {
    "tags": ["tenant_acme", "priority_support", "v2.4"],
    "metadata": {
        "user_id": "usr_99812",
        "conversation_id": "conv_4410",
        "model_temperature": 0.2
    },
    "callbacks": [custom_tracer],
    "recursion_limit": 25,
    "run_name": "CustomInferencePipeline"
}
chain.invoke(query, config=config)
```
In LangSmith, `RunnableConfig` parameters are automatically inherited by **every child span** down the entire execution tree,
enabling powerful faceted filtering and multi-tenant audit querying in the LangSmith dashboard.
""")

col_rc1, col_rc2 = st.columns([1, 1])

with col_rc1:
    st.markdown("### Best Practices for RunnableConfig in Production:")
    st.markdown("""
    * **Faceted Search with Tags**: Use `tags` for broad categorical filters (e.g., `["production", "canary", "mobile_client"]`).
    * **Audit Attribution with Metadata**: Use `metadata` for high-cardinality keys (e.g., `user_id`, `tenant_id`, `session_id`).
    * **Custom Run Names**: Set `run_name="CustomerSupportV2"` to override generic `RunnableSequence` names in the trace explorer for clean readability.
    * **Dynamic Callbacks**: Inject custom LangSmith client instances dynamically per request for multi-workspace isolation.
    """)

with col_rc2:
    st.markdown("### RunnableConfig Ingestion Hierarchy")
    dot_config = """
    digraph ConfigInheritance {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Config [label="RunnableConfig Payload\ntags: ['prod', 'v2.4']\nmetadata: {'tenant': 'ACME'}", shape=component, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        RootSpan [label="Root Chain Span\nInherits: tags + metadata", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        Child1 [label="Child Span 1 (Prompt)\nInherits: tags + metadata", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Child2 [label="Child Span 2 (Model)\nInherits: tags + metadata", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];

        Config -> RootSpan -> Child1;
        RootSpan -> Child2;
    }
    """
    st.graphviz_chart(dot_config, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise Runnable Tracing Engine (RunnableTracingEngine)", divider="rainbow")

st.markdown("""
Below is the verified production engine `RunnableTracingEngine`.
It models execution across `RunnableSequence`, `RunnableParallel`, and `RunnableBranch`,
capturing child spans, runtime latency calculations, and `RunnableConfig` inheritance in pure Python.
""")

st.code('''
class RunnableTracingEngine:
    """
    Production LCEL Runnable Tracing & Orchestration Engine.
    Emulates LangSmith trace collection across RunnableSequence,
    RunnableParallel, RunnableBranch, and RunnableConfig.
    """
    def __init__(self, project_name: str = "lcel-runnable-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 700

    def execute_lcel_workflow(
        self,
        input_query: str,
        execution_mode: str = "parallel",
        intent_category: str = "technical",
        config_tags: Optional[List[str]] = None,
        config_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self._span_counter += 1
        trace_id = f"trace_lcel_{self._span_counter}"
        root_run_id = f"run_root_{self._span_counter}"
        config_tags = config_tags or ["production", "v2.4"]
        config_metadata = config_metadata or {"tenant_id": "ORG-ALPHA", "environment": "prod"}

        if execution_mode == "sequential":
            step1_latency, step2_latency, step3_latency = 15.0, 220.0, 8.0
            total_latency = step1_latency + step2_latency + step3_latency
            child_spans = [
                {"name": "RunnableStep1_InputValidator", "run_type": "chain", "latency_ms": step1_latency},
                {"name": "RunnableStep2_LLMInference", "run_type": "llm", "latency_ms": step2_latency},
                {"name": "RunnableStep3_OutputParser", "run_type": "parser", "latency_ms": step3_latency}
            ]
        elif execution_mode == "parallel":
            branch_a, branch_b = 45.0, 68.0
            total_latency = max(branch_a, branch_b) + 4.0
            child_spans = [
                {"name": "RunnableParallel_BranchA_VectorDB", "run_type": "retriever", "latency_ms": branch_a},
                {"name": "RunnableParallel_BranchB_WebSearch", "run_type": "tool", "latency_ms": branch_b}
            ]
        else:
            classifier_latency = 22.0
            target_route = "TechnicalDocumentationChain" if intent_category == "technical" else "BillingFaqChain"
            route_latency = 180.0 if intent_category == "technical" else 95.0
            total_latency = classifier_latency + route_latency
            child_spans = [
                {"name": "RunnableBranch_Classifier", "run_type": "chain", "latency_ms": classifier_latency},
                {"name": target_route, "run_type": "chain", "latency_ms": route_latency}
            ]

        trace_record = {
            "trace_id": trace_id,
            "root_run": {
                "run_id": root_run_id,
                "name": f"LCEL_{execution_mode.capitalize()}Orchestrator",
                "total_latency_ms": round(total_latency, 2),
                "tags": config_tags,
                "metadata": config_metadata
            },
            "child_spans": child_spans
        }
        self.active_traces.append(trace_record)
        return trace_record
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
runnable_engine_inst = RunnableTracingEngine()

# Test Sequential Mode
seq_test = runnable_engine_inst.execute_lcel_workflow("Query A", execution_mode="sequential")
assert len(seq_test["child_spans"]) == 3, "Sequential pipeline must have 3 child spans"
assert seq_test["root_run"]["total_latency_ms"] == 243.0, "Sequential latency must sum stages (15 + 220 + 8 = 243)"

# Test Parallel Mode
par_test = runnable_engine_inst.execute_lcel_workflow("Query B", execution_mode="parallel")
assert len(par_test["child_spans"]) == 2, "Parallel pipeline must have 2 child spans"
assert par_test["root_run"]["total_latency_ms"] == 72.0, "Parallel latency must bound to max(45, 68) + 4 = 72"

# Test Branch Mode
branch_test = runnable_engine_inst.execute_lcel_workflow("Query C", execution_mode="branch", intent_category="billing")
assert branch_test["child_spans"][1]["name"] == "BillingFaqChain", "Must route to BillingFaqChain"

st.success("Automated Verification Passed: RunnableTracingEngine successfully verified across Sequential, Parallel, and Branch execution modes.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: LCEL Runnable Execution Simulator & Parallel Fan-Out Profiler", divider="rainbow")

st.markdown("""
Experiment with LCEL execution topologies in real time.
Select execution modes, configure runtime tags and tenant metadata, and inspect how LangSmith organizes the resulting run spans.
""")

wb_r1, wb_r2 = st.columns([1, 1])

with wb_r1:
    st.markdown("#### LCEL Execution Topology Configuration")
    opt_exec_mode = st.selectbox("Select Execution Topology", ["parallel", "sequential", "branch"])
    opt_intent = st.selectbox("Simulated Query Intent (for Branch mode)", ["technical", "billing"])
    input_run_q = st.text_input("Input Query String", value="How to deploy multi-region clusters with zero downtime?")

with wb_r2:
    st.markdown("#### RunnableConfig Parameters")
    in_tenant = st.text_input("Tenant ID Metadata", value="TENANT-ACME-CORP")
    in_tags_raw = st.text_input("Comma-Separated Tags", value="production, canaries, us-east-1")

tags_list = [t.strip() for t in in_tags_raw.split(",") if t.strip()]
meta_dict = {"tenant_id": in_tenant, "engine": "LCEL_v0.3"}

live_lcel_trace = runnable_engine_inst.execute_lcel_workflow(
    input_query=input_run_q,
    execution_mode=opt_exec_mode,
    intent_category=opt_intent,
    config_tags=tags_list,
    config_metadata=meta_dict
)

st.markdown("### LangSmith Run Tree Hierarchy")

m_t1, m_t2, m_t3 = st.columns(3)
m_t1.metric("Total Execution Duration", f"{live_lcel_trace['root_run']['total_latency_ms']} ms")
m_t2.metric("Child Spans Executed", f"{len(live_lcel_trace['child_spans'])} Spans")
m_t3.metric("Inherited Tags Count", f"{len(live_lcel_trace['root_run']['tags'])} Tags")

lcel_span_table = [
    {
        "Span Name": live_lcel_trace["root_run"]["name"],
        "Execution Stage": "ROOT (Coordinator)",
        "Latency (ms)": live_lcel_trace["root_run"]["total_latency_ms"],
        "Inherited Metadata": str(live_lcel_trace["root_run"]["metadata"])
    }
]
for s in live_lcel_trace["child_spans"]:
    lcel_span_table.append({
        "Span Name": s["name"],
        "Execution Stage": s.get("run_type", "child_step"),
        "Latency (ms)": s["latency_ms"],
        "Inherited Metadata": "Inherited from Root Config"
    })

st.dataframe(pd.DataFrame(lcel_span_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: How does RunnableParallel execution differ from RunnableSequence in latency profiling?**"):
    st.markdown("""
    **Answer**:
    * In a **RunnableSequence**, components execute sequentially. Total latency is strictly additive: `sum(t_i)`.
    * In a **RunnableParallel**, all branches execute concurrently on separate threads. Total latency is bounded by the single slowest branch: `max(t_i) + overhead`.
    * In LangSmith, RunnableParallel child spans share identical start timestamps, visually forming a horizontal multi-branch fan-out in the trace waterfall view.
    """)

with st.expander("**Q2: How does RunnableConfig facilitate multi-tenant tracing in enterprise B2B SaaS?**"):
    st.markdown("""
    **Answer**:
    `RunnableConfig` decouples session context from function signatures.
    Instead of adding `tenant_id` parameters to every internal function, you pass `config={"metadata": {"tenant_id": "ORG_88"}}` to the top-level `chain.invoke()`.
    LangSmith propagates this metadata down the entire call stack.
    Engineers can then query the LangSmith API: `client.list_runs(metadata={"tenant_id": "ORG_88"})` to isolate traces for a specific customer during incident triage.
    """)

with st.expander("**Q3: What happens when one branch in a RunnableParallel throws an exception?**"):
    st.markdown("""
    **Answer**:
    By default in Python's thread pool executors, if one branch raises an exception, the entire parallel execution fails and bubbles up the exception.
    In LangSmith, the failing branch span is highlighted in red with its exact exception traceback, while sibling spans indicate whether they completed or were cancelled.
    To prevent single-branch failures from breaking the whole pipeline, engineers use `.with_fallbacks()` or `return_exceptions=True` patterns.
    """)

with st.expander("**Q4: What is the difference between Runnable.stream() and Runnable.astream_events() in LangSmith?**"):
    st.markdown("""
    **Answer**:
    * `stream()` yields output tokens from the final runnable in the chain as they are generated.
    * `astream_events()` emits fine-grained, real-time Abstract Syntax Tree (AST) events from *all* intermediate runnables (e.g., when a tool starts, when a retriever finishes, when a token is emitted).
    * LangSmith uses this streaming event interface to update live traces in real time before the total chain execution finishes.
    """)

with st.expander("**Q5: How do you trace custom Python functions in an LCEL pipeline without breaking runnable compatibility?**"):
    st.markdown("""
    **Answer**:
    Wrap the custom Python function in `RunnableLambda`:
    ```python
    from langchain_core.runnables import RunnableLambda

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_retriever = retriever | RunnableLambda(format_docs)
    ```
    This automatically instruments the function with LangChain callbacks, capturing inputs, outputs, and execution duration as a distinct span in LangSmith.
    """)

st.markdown("### Production Traps & Failure Modes")

rtr_col1, rtr_col2 = st.columns(2)

with rtr_col1:
    st.error("""
    **Trap 1: Thread Pool Exhaustion in High-Concurrency RunnableParallel**
    * **Failure**: Using `RunnableParallel` with 8 branches per request on a server handling 100 concurrent requests without sizing the Python `ThreadPoolExecutor`.
    * **Consequence**: The thread pool runs out of worker threads, causing parallel branches to block and queue, turning 50ms parallel lookups into 2,000ms latency spikes.
    * **Remedy**: Size thread pools appropriately or switch to native asynchronous coroutines (`ainvoke` with `asyncio.gather`).
    """)

    st.error("""
    **Trap 2: Losing Context Metadata Across Custom Threads**
    * **Failure**: Spawning raw `threading.Thread` instances inside a runnable instead of using LangChain's native thread pool abstractions.
    * **Consequence**: The Python `contextvars` context is severed, causing child spans to lose their `parent_run_id` and creating fragmented orphan traces.
    * **Remedy**: Always use `RunnableParallel` or `asyncio` which preserve `contextvars` propagation natively.
    """)

with rtr_col2:
    st.warning("""
    **Trap 3: Unhandled Route Fallbacks in RunnableBranch**
    * **Failure**: Creating a `RunnableBranch` where incoming edge cases match none of the explicit conditions and no default branch is provided.
    * **Consequence**: The branch returns `None` or raises an unhandled routing exception, breaking downstream prompt components.
    * **Remedy**: Always specify a robust default fallback runnable at the end of every `RunnableBranch` declaration.
    """)

    st.warning("""
    **Trap 4: Missing Tag Cleanup in Reused Config Dictionaries**
    * **Failure**: Reusing a mutable `config` dictionary across multiple requests without clearing request-specific tags.
    * **Consequence**: Tags from previous user sessions bleed into subsequent user traces, corrupting audit analytics in LangSmith.
    * **Remedy**: Construct fresh, immutable `RunnableConfig` dictionaries for every invocation.
    """)

st.markdown("### Core Technical Flashcards")

fcr_c1, fcr_c2 = st.columns(2)

with fcr_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the formula for RunnableParallel execution latency?")
        st.markdown("**Back**: `T_parallel = max(t_branch_1, t_branch_2, ...) + thread_overhead`. Total execution time is bounded by the single slowest branch rather than the sum of branches.")

    with fcr_c2:
        st.markdown("**Front**: How does RunnableConfig enable multi-tenant trace filtering in LangSmith?")
        st.markdown("**Back**: By attaching metadata (e.g., `{'tenant_id': 'ACME'}`) to the root config, automatically propagating the attributes down to all child spans for faceted dashboard querying.")

with fcr_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 3 core composition primitives in LCEL?")
        st.markdown("**Back**: 1. `RunnableSequence` (Linear sequential pipes), 2. `RunnableParallel` (Concurrent scatter-gather branches), and 3. `RunnableBranch` (Conditional dynamic routing).")

    with fcr_c2:
        st.markdown("**Front**: How do you wrap arbitrary Python functions for LCEL tracing?")
        st.markdown("**Back**: Using `RunnableLambda(my_func)`, which integrates standard Python functions into the runnable protocol and automatically instruments them in LangSmith.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LCEL Runnable Tracing principles:")

rc_chk1 = st.checkbox("I understand the unified Runnable interface (invoke, ainvoke, batch, stream) and its native tracing.", value=True)
rc_chk2 = st.checkbox("I can profile linear RunnableSequence latency and isolate inter-stage state corruption in LangSmith.", value=True)
rc_chk3 = st.checkbox("I understand how RunnableParallel executes concurrent fan-out and how slowest branches govern latency.", value=True)
rc_chk4 = st.checkbox("I can trace dynamic routing decisions and fallback execution in RunnableBranch architectures.", value=True)
rc_chk5 = st.checkbox("I know how to inject tags and metadata via RunnableConfig for multi-tenant trace filtering.", value=True)
rc_chk6 = st.checkbox("I can wrap custom Python functions in RunnableLambda to preserve contextvars and trace lineage.", value=True)

if rc_chk1 and rc_chk2 and rc_chk3 and rc_chk4 and rc_chk5 and rc_chk6:
    st.success("Mastery Certified: You have mastered LCEL runnable tracing, parallel fan-out profiling, and runtime configuration management.")
