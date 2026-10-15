import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.1 Frequently Asked Questions - LangSmith Curriculum",
    page_icon="❓",
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
    .badge-faq {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class LangSmithFAQConceptEngine:
    """
    Simulates core LangSmith architectural entities:
    Runs, Traces, Projects, Datasets, Experiments, and Evaluators.
    """
    def __init__(self):
        self.entities = {
            "Run": {
                "definition": "The atomic execution span representing a single invocation (LLM, Chain, Tool, Retriever).",
                "key_attributes": ["run_id", "name", "inputs", "outputs", "duration_ms", "tokens", "status"],
                "example": "ChatOpenAI invocation taking 1,120ms and generating 140 completion tokens."
            },
            "Trace": {
                "definition": "A directed acyclic graph (DAG) or tree of nested runs from a single root invocation.",
                "key_attributes": ["trace_id", "root_run", "child_runs", "total_latency", "total_tokens"],
                "example": "Full RAG pipeline encompassing Query Rewrite -> Vector Search -> Synthesizer."
            },
            "Project": {
                "definition": "A tenant-scoped namespace organizing traces by environment, team, or application.",
                "key_attributes": ["project_name", "retention_policy", "run_count", "metadata_filters"],
                "example": "production-customer-support or staging-agent-v2."
            },
            "Dataset": {
                "definition": "A versioned collection of input-output examples used for benchmarking and regression testing.",
                "key_attributes": ["dataset_id", "example_count", "splits", "schema_type"],
                "example": "golden-rag-financial-benchmark with 150 verified Q&A pairs."
            },
            "Experiment": {
                "definition": "A recorded evaluation run testing a specific code/prompt version against a dataset.",
                "key_attributes": ["experiment_id", "dataset_ref", "prompt_commit", "evaluation_scores"],
                "example": "prompt-v3-eval-gpt4o scoring 94% on correctness."
            },
            "Evaluator": {
                "definition": "A programmatic, heuristic, or LLM-as-a-judge function that scores run outputs.",
                "key_attributes": ["evaluator_type", "criteria_rubric", "score_range", "feedback_key"],
                "example": "Factual accuracy evaluator grading answers between 0.0 and 1.0."
            }
        }

    def get_entity_details(self, entity_name: str) -> Dict[str, Any]:
        return self.entities.get(entity_name, {})

faq_engine = LangSmithFAQConceptEngine()

# --- HEADER SECTION ---
st.title("37.1 Frequently Asked Questions: Core LangSmith Architecture")
st.caption("Chapter 37: LangSmith Interview Topics | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
Master the high-frequency foundational questions asked in senior GenAI and LLMOps engineering interviews. 
This module explores the core primitives of the LangSmith platform—differentiating runtime execution from observability, 
explaining run tree hierarchies, and defining the contracts governing datasets, experiments, and evaluators.

In this lesson, you will master:
1. The architectural definition and enterprise necessity of LangSmith.
2. The core differences separating LangChain, LangGraph, and LangSmith.
3. The anatomy of distributed traces, runs, and span hierarchies.
4. Project organization, environment tagging, and multi-tenant scoping.
5. Golden datasets, automated experiments, and programmatic evaluators.
""")
st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise LLM Observability & Evaluation Platform Ecosystem", divider="rainbow")

st.markdown("""
The architecture below illustrates how LangSmith functions as the centralized observability and evaluation control plane 
for modern LLM applications. Whether using LangChain, LangGraph, or raw OpenAI/Anthropic SDKs, applications stream background 
telemetry events into LangSmith to build hierarchical run traces, collect human feedback, and trigger automated regression evaluations.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangSmithEcosystemBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_apps {
            label="1. Application Runtime Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            app_chain [label="LangChain Pipelines (LCEL)\\nChains, Prompts & Parsers", fillcolor="#0F172A"];
            app_graph [label="LangGraph State Machines\\nCyclic Multi-Agent Graphs", fillcolor="#0F172A"];
            app_raw [label="Custom Python / SDK Invocations\\n@traceable Functions & LiteLLM", fillcolor="#0F172A"];
        }

        subgraph cluster_telemetry {
            label="2. Background Asynchronous Telemetry Ingestion";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            collector [label="Async Worker Queue (HTTPS Ingestion)\\nZero-blocking Nonce / Trace Exporter", fillcolor="#312E81"];
            run_db [label="Distributed Run Database\\nPartitioned by Project & Thread ID", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_features {
            label="3. LangSmith Core Platform Capabilities";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            feat_trace [label="Tracing & Run Trees\\nWaterfall Latencies & Token Counts", fillcolor="#1E293B"];
            feat_hub [label="Prompt Hub\\nVersioned Prompts & Playground", fillcolor="#1E293B"];
            feat_eval [label="Evaluation & Datasets\\nGolden Benchmarks & LLM-as-a-Judge", fillcolor="#831843", fontcolor="#FBCFE8"];
            feat_monitor [label="Production Monitoring\\nError Rates, Drifts & Cost Tracking", fillcolor="#1E293B"];
        }

        app_chain -> collector [label="Emits Spans"];
        app_graph -> collector [label="Emits Superstep Events"];
        app_raw -> collector [label="@traceable Calls"];

        collector -> run_db;
        run_db -> feat_trace;
        run_db -> feat_hub;
        run_db -> feat_eval;
        run_db -> feat_monitor;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: What is LangSmith?
# ==============================================================================
st.header("2. What is LangSmith?: Enterprise LLM Observability, Tracing, and Quality Control", divider="rainbow")

st.markdown("""
**LangSmith** is an enterprise-grade platform for debugging, testing, evaluating, and monitoring Large Language Model (LLM) 
applications. While traditional software engineering relies on APMs (like Datadog or New Relic) to track HTTP latencies and CPU loads, 
LLM applications exhibit non-deterministic behaviors, complex prompt template hierarchies, multi-turn tool loops, 
and token-based cost profiles that traditional APMs cannot inspect.

### Core Value Propositions:
- **Visibility**: Turn black-box model invocations into transparent execution trees.
- **Evaluation**: Programmatically grade model outputs against versioned ground truth datasets.
- **Continuous Improvement**: Collect user feedback in production to continually augment golden evaluation datasets.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern enterprise stacks, LangSmith is deployed as the foundational observability layer alongside CI/CD pipelines. 
Instead of logging model prompts to plain-text log files, every inference is enriched with tenant IDs, cost metrics, 
and user session identifiers, enabling instant regression detection prior to production deployment.
""")

# ==============================================================================
# SECTION 3: LangSmith vs LangChain
# ==============================================================================
st.header("3. LangSmith vs LangChain: Runtime Framework vs Telemetry and Evaluation Platform", divider="rainbow")

st.markdown("""
A common interview question explores the architectural boundary between LangChain and LangSmith:

| Dimension | LangChain | LangSmith |
| :--- | :--- | :--- |
| **Primary Role** | Orchestration & Framework | Observability & Evaluation Platform |
| **Execution Layer** | Runs inside your application runtime (Python / TypeScript) | Cloud-hosted SaaS or VPC-deployed control plane |
| **Core Primitives** | Prompts, LLMs, Output Parsers, Runnables (LCEL) | Runs, Traces, Datasets, Experiments, Feedback |
| **Dependency** | Can be used without LangSmith | Can monitor any code (including raw OpenAI or Anthropic SDKs) |
""")

# ==============================================================================
# SECTION 4: LangSmith vs LangGraph
# ==============================================================================
st.header("4. LangSmith vs LangGraph: Cyclic Agent State Machine vs Graph-Level Execution Tracing", divider="rainbow")

st.markdown("""
Interviewers frequently ask candidates to articulate how LangSmith integrates with LangGraph:

- **LangGraph**: An orchestration runtime for building cyclic, stateful, multi-agent systems using state graphs, 
  channel reducers, human-in-the-loop interrupts, and durable checkpointers.
- **LangSmith**: The observability platform that captures each LangGraph superstep, rendering the active graph topology, 
  tracking state updates across nodes, and measuring intermediate token expenditures.
""")

# ==============================================================================
# SECTION 5: What is Tracing?
# ==============================================================================
st.header("5. What is Tracing?: OpenTelemetry-Compatible Event Collection and Span Instrumentation", divider="rainbow")

st.markdown("""
**Tracing** is the automated capture and correlation of events that occur throughout an application's execution lifecycle. 
In LangSmith:
- Every function call, prompt formatting step, vector database retrieval, and LLM completion is recorded as a **Span**.
- Spans are assigned timestamps, durations, inputs, outputs, error traces, and parent-child linkage.
- Tracing operates asynchronously in background threads, introducing negligible overhead (under 3ms) to the user request path.
""")

# ==============================================================================
# SECTION 6: What is a Run?
# ==============================================================================
st.header("6. What is a Run?: Fundamental Unit of Execution (Inputs, Outputs, Metadata, Latency)", divider="rainbow")

st.markdown("""
In LangSmith, a **Run** represents a single, discrete invocation of a component.

### Anatomy of a Run:
- **`id`**: Unique UUID assigned to the execution.
- **`run_type`**: Discriminator categorizing the component (`llm`, `chain`, `tool`, `retriever`, `prompt`).
- **`inputs`**: Exact serialized dictionary passed to the component.
- **`outputs`**: Exact serialized dictionary or string returned by the component.
- **`extra`**: Telemetry metadata including model parameters (temperature, max_tokens) and token usage.
- **`parent_run_id`**: Pointer to the antecedent run that spawned this span.
""")

# ==============================================================================
# SECTION 7: What is a Trace?
# ==============================================================================
st.header("7. What is a Trace?: Tree-Structured Directed Acyclic Graph of Nested Execution Spans", divider="rainbow")

st.markdown("""
A **Trace** is the complete hierarchical collection of runs initiated by a single user request. 
Traces start with a single **Root Run** (typically a Chain or Graph execution) and branch downward into child spans:
""")

with st.expander("🔍 View Trace Tree Span Breakdown Flow", expanded=True):
    trace_dot = """
    digraph TraceSpanTree {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        root [label="Root Run (Chain: CustomerSupport)\\nTotal Latency: 1.45s | Total Tokens: 820", fillcolor="#0F172A"];
        span_query [label="Child Run 1 (PromptTemplate)\\nLatency: 2ms", fillcolor="#1E293B"];
        span_embed [label="Child Run 2 (Embeddings)\\nLatency: 85ms", fillcolor="#1E293B"];
        span_retriever [label="Child Run 3 (VectorStore: Qdrant)\\nLatency: 42ms | Chunks: 4", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        span_llm [label="Child Run 4 (ChatModel: gpt-4o)\\nLatency: 1.32s | Tokens: 820", fillcolor="#312E81"];

        root -> span_query;
        root -> span_embed;
        span_embed -> span_retriever;
        root -> span_llm;
    }
    """
    st.graphviz_chart(trace_dot, use_container_width=True)

# ==============================================================================
# SECTION 8: What is a Project?
# ==============================================================================
st.header("8. What is a Project?: Tenant-Scoped Namespace for Organizing Production and Development Runs", divider="rainbow")

st.markdown("""
A **Project** in LangSmith is a logical namespace grouping related traces. 
Projects allow engineering teams to isolate data environments and configure custom data retention rules:

- **`staging-agent`**: Captures integration tests and development runs.
- **`prod-customer-service`**: Receives live production user traffic.
- **Cost & Rate Segregation**: Projects allow filtering usage dashboards to calculate precise departmental spend.
""")

# ==============================================================================
# SECTION 9: What is a Dataset?
# ==============================================================================
st.header("9. What is a Dataset?: Versioned Ground Truth Collections of Examples for Regression Testing", divider="rainbow")

st.markdown("""
A **Dataset** is a collection of curated examples used for automated evaluation and regression testing. 
Each dataset example consists of:
- **`inputs`**: The query or state passed to the application.
- **`outputs`**: The expected ground-truth completion, reference document IDs, or target tool call parameters.
- **`metadata`**: Tags, creation timestamps, and split labels (e.g., `train`, `validation`, `test`).
""")

# ==============================================================================
# SECTION 10: What is an Experiment?
# ==============================================================================
st.header("10. What is an Experiment?: Evaluating Application Versions Against Target Datasets", divider="rainbow")

st.markdown("""
An **Experiment** in LangSmith represents a programmatic evaluation run. 
When you test a prompt update or new model checkpoint against a dataset, LangSmith:
1. Executes the candidate application across every dataset example.
2. Ingests all outputs into a new experiment bucket.
3. Invokes configured evaluators to score each output.
4. Renders side-by-side comparative benchmark diffs against baseline experiments.
""")

# ==============================================================================
# SECTION 11: What is an Evaluator?
# ==============================================================================
st.header("11. What is an Evaluator?: Programmatic, Heuristic, and LLM-as-a-Judge Scoring Functions", divider="rainbow")

st.markdown("""
An **Evaluator** is a function that measures the quality of a run's output against target criteria.

### The Three Tiers of Evaluators:
1. **Deterministic Evaluators**: Exact match, regex assertions, JSON schema validation, Levenshtein distance.
2. **Heuristic Evaluators**: Reading grade level, length limits, sentiment polarity, forbidden word filters.
3. **LLM-as-a-Judge**: Foundation models prompted with explicit grading rubrics to evaluate qualitative dimensions 
   (groundedness, factual consistency, tone, completeness).
""")

# ==============================================================================
# SECTION 12: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("12. Production Implementation: Full LangSmith Client Setup, Tracing Decorators, and Dataset Creation", divider="rainbow")

st.markdown("""
The following complete script demonstrates the end-to-end LangSmith SDK workflow: 
enabling tracing, utilizing the `@traceable` decorator on non-LangChain Python code, creating a dataset, and querying runs.
""")

st.code('''from langsmith import Client, traceable
from typing import Dict, Any
import datetime

# 1. Initialize Client
client = Client()

# 2. Instrument custom Python code with @traceable
@traceable(
    run_type="chain",
    name="FinancialAnalysisPipeline",
    tags=["finance", "q3-audit"],
    metadata={"environment": "production", "version": "v2.1"}
)
def run_financial_pipeline(ticker: str, fiscal_year: int) -> Dict[str, Any]:
    # Simulated execution
    extracted_data = fetch_sec_filing(ticker, fiscal_year)
    analysis = generate_insights(extracted_data)
    return {"ticker": ticker, "summary": analysis, "status": "approved"}

@traceable(run_type="tool", name="SECFilingFetcher")
def fetch_sec_filing(ticker: str, year: int) -> str:
    return f"SEC 10-K for {ticker} fiscal year {year}: Revenue rose 14%."

@traceable(run_type="llm", name="InsightGenerator")
def generate_insights(text: str) -> str:
    return f"Insights synthesized from text: High financial resilience."

# 3. Create a Golden Benchmark Dataset
def create_golden_dataset():
    dataset_name = f"Finance-Benchmark-{datetime.date.today()}"
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Verified Q3 financial queries for regression evaluation."
    )
    
    # Seed dataset with verified ground truth
    client.create_examples(
        inputs=[{"ticker": "NVDA", "year": 2024}, {"ticker": "AAPL", "year": 2024}],
        outputs=[{"status": "approved"}, {"status": "approved"}],
        dataset_id=dataset.id
    )
    print(f"Created dataset: {dataset_name} (ID: {dataset.id})")
''', language="python")

# ==============================================================================
# SECTION 13: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("13. Interactive Workbench: Core Observability Concept Explorer & Run Hierarchy Inspector", divider="rainbow")

st.markdown("""
Select any core LangSmith concept to inspect its architectural definition, key system attributes, and real-world example.
""")

selected_concept = st.selectbox(
    "Select LangSmith Architectural Entity to Inspect",
    ["Run", "Trace", "Project", "Dataset", "Experiment", "Evaluator"]
)

details = faq_engine.get_entity_details(selected_concept)

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown(f"### {selected_concept} Architecture")
    st.info(details.get("definition", ""))
    st.markdown("**Key System Attributes:**")
    for attr in details.get("key_attributes", []):
        st.markdown(f"- `{attr}`")

with c2:
    st.markdown("### Real-World Production Example")
    st.success(details.get("example", ""))
    
    st.markdown("#### Interview Tip")
    if selected_concept in ["Run", "Trace"]:
        st.write("Always emphasize that a Trace is the root container DAG, while a Run is an individual span within the tree.")
    elif selected_concept in ["Dataset", "Experiment"]:
        st.write("Clarify that Datasets represent static ground truth, while Experiments represent application runs tested against datasets.")
    else:
        st.write("Highlight that evaluators bridge deterministic regex checks with qualitative LLM-as-a-judge rubrics.")

# ==============================================================================
# SECTION 14: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What is the exact relationship between a Run and a Trace in LangSmith?"):
    st.markdown("""
    **Answer:**  
    A **Run** is the fundamental atomic span representing a single execution step (such as an LLM call, prompt formatting, or tool execution). 
    A **Trace** is the complete directed acyclic graph (DAG) or tree of all nested runs originating from a single root invocation. 
    Every child run references its `parent_run_id`, while all runs in the tree share the same root `trace_id`.
    """)

with st.expander("Q2: Can LangSmith trace applications built without LangChain?"):
    st.markdown("""
    **Answer:**  
    Yes. LangSmith is framework-agnostic. Non-LangChain applications can be fully traced using the `@traceable` decorator 
    from the `langsmith` Python/TypeScript SDK, the `wrap_openai()` client wrapper, or standard OpenTelemetry trace exporters.
    """)

with st.expander("Q3: How does LangSmith handle performance overhead during real-time streaming?"):
    st.markdown("""
    **Answer:**  
    LangSmith uses an asynchronous, background worker queue to batch and stream telemetry payloads over HTTPS. 
    Node execution and LLM token streaming are never blocked on telemetry delivery. Overhead is typically under 2-3 milliseconds.
    """)

with st.expander("Q4: What is the primary difference between online and offline evaluations in LangSmith?"):
    st.markdown("""
    **Answer:**  
    - **Offline Evaluation**: Pre-deployment testing where a pipeline is executed against a static golden dataset using `Client.evaluate()`.  
    - **Online Evaluation**: Real-time production monitoring where evaluators run asynchronously on live user traffic traces, 
      tagging runs with automated quality scores or user feedback (`thumbs_up` / `thumbs_down`).
    """)

with st.expander("Q5: How are LangSmith Projects used to separate development, staging, and production environments?"):
    st.markdown("""
    **Answer:**  
    By setting the `LANGSMITH_PROJECT` environment variable (e.g., `prod-agent`, `staging-agent`), traces are routed to 
    isolated workspace namespaces. This prevents staging and unit test runs from polluting production latency percentiles, 
    token analytics, and evaluation benchmarks.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Hardcoding LANGSMITH_API_KEY into Version Control**  
Hardcoding API keys into repository code exposes enterprise trace data and proprietary prompts. Always inject credentials 
via secure environment variables or secret vaults.
""")

st.error("""
**Trap 2: Logging Unredacted PII in Input/Output Spans**  
Sending credit card numbers, passwords, or HIPAA-protected health data to LangSmith violates compliance regulations. 
Implement client-side masking hooks (`hide_inputs` / `hide_outputs`) before telemetry export.
""")

st.warning("""
**Trap 3: Forgetting Project Scoping Leading to Default Project Pollution**  
Failing to specify `LANGSMITH_PROJECT` sends all traces to the `default` project bucket, mixing local developer experiments 
with production workloads.
""")

st.warning("""
**Trap 4: Treating LLM-as-a-Judge Scores as Absolute Truth Without Calibration**  
Relying solely on an LLM judge without validating its agreement rate against human annotators can mask model drift and false positives.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Run vs Trace</h4>
        <p>A Run is an atomic execution span; a Trace is the complete hierarchical DAG of runs forming an end-to-end request lifecycle.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>@traceable Decorator</h4>
        <p>A LangSmith SDK decorator that instruments standard Python functions with automatic run creation, latency timing, and input/output capture.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Golden Dataset</h4>
        <p>A curated, version-controlled collection of input-output benchmark pairs used for automated regression testing and CI/CD quality gates.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Client.evaluate()</h4>
        <p>The programmatic LangSmith API method for benchmarking an application pipeline against a target dataset with custom evaluators.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 15: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("15. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can articulate the exact architectural differences between LangChain, LangGraph, and LangSmith.", value=True)
chk_2 = st.checkbox("I understand the hierarchy connecting Root Runs, Child Spans, and complete Traces.", value=True)
chk_3 = st.checkbox("I know how to instrument non-LangChain Python functions using the @traceable decorator.", value=True)
chk_4 = st.checkbox("I can explain how Projects partition environments and isolate production telemetry.", value=True)
chk_5 = st.checkbox("I understand the role of Golden Datasets and Experiments in regression testing.", value=True)
chk_6 = st.checkbox("I can categorize evaluators into deterministic, heuristic, and LLM-as-a-judge tiers.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangSmith core architectural concepts and interview fundamentals verified! Ready for LLM Evaluation questions.")
