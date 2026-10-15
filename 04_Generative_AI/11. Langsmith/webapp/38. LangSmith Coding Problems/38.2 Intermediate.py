import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="38.2 Intermediate Coding Problems - LangSmith Curriculum",
    page_icon="⚡",
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
    .badge-code {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-pass {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class LangSmithIntermediateEngine:
    """
    Simulates intermediate LangSmith workflows:
    Custom evaluators, prompt benchmarking, RAG evaluation,
    LangGraph state tracing, token tracking, and failure analysis.
    """
    def __init__(self):
        self.prompt_experiments = {
            "Prompt V1 (Direct Zero-Shot)": {
                "accuracy": 0.78,
                "latency_p95_ms": 780,
                "avg_tokens": 420,
                "cost_per_1k": 0.0014,
                "failure_rate": 0.04
            },
            "Prompt V2 (Few-Shot with Structured JSON)": {
                "accuracy": 0.94,
                "latency_p95_ms": 1150,
                "avg_tokens": 890,
                "cost_per_1k": 0.0031,
                "failure_rate": 0.005
            }
        }
        
        self.failed_runs_db = [
            {
                "run_id": "run-err-501",
                "node_name": "sql_query_generator",
                "error_type": "SyntaxError",
                "error_message": "Unexpected token LIMIT inside subquery block",
                "retry_count": 2,
                "recovered": False
            },
            {
                "run_id": "run-err-502",
                "node_name": "retriever_node",
                "error_type": "ConnectionTimeout",
                "error_message": "ChromaDB connection timed out after 3000ms",
                "retry_count": 1,
                "recovered": True
            },
            {
                "run_id": "run-err-503",
                "node_name": "output_guardrail",
                "error_type": "ValidationError",
                "error_message": "JSON schema violation: 'sentiment_score' missing",
                "retry_count": 0,
                "recovered": False
            }
        ]

    def evaluate_custom_similarity(self, predicted: str, expected: str) -> Dict[str, Any]:
        pred_words = set(predicted.lower().split())
        exp_words = set(expected.lower().split())
        overlap = len(pred_words.intersection(exp_words)) / max(len(exp_words), 1)
        passed = overlap >= 0.70
        return {
            "overlap_score": round(overlap, 3),
            "passed": passed,
            "status": "PASS" if passed else "FAIL"
        }

    def simulate_langgraph_step_trace(self, node: str, state_input: str) -> Dict[str, Any]:
        return {
            "node_name": node,
            "state_snapshot_keys": ["messages", "current_step", "intermediate_steps"],
            "step_latency_ms": np.random.randint(120, 480),
            "tokens_generated": np.random.randint(25, 95),
            "next_edge": "agent" if node == "tools" else "tools" if "search" in state_input.lower() else "__end__"
        }

# Instantiate engine
intermediate_engine = LangSmithIntermediateEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Evaluation Execution Engines, LangGraph Tracing, and Root-Cause Diagnostics", divider="rainbow")

st.markdown("""
At the intermediate level, LangSmith transitions from simple logging into an **interactive diagnostic and validation engine**. 
This blueprint illustrates the lifecycle of executing custom evaluation functions, inspecting complex cyclical 
LangGraph state machines, aggregating token expenditures, and performing root-cause analysis on failing traces.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Intermediate_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_graph {
            label = "LangGraph Cyclic State Machine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            agent_node [label="Agent Reasoning Node\\n(State: messages, plan)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            tool_node [label="Tool Execution Node\\n(State: tool_outputs)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            cond_edge [label="Conditional Routing Edge\\n(should_continue?)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_telemetry {
            label = "LangSmith Tracing & Diagnostics Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            trace_stream [label="Hierarchical Trace DAG\\n(Node Spans, State Snapshots)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            token_agg [label="Token & Cost Aggregator\\n(Prompt vs Completion breakdown)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            error_audit [label="Failed Run Analyzer\\n(Stack Trace, Retry Diagnostics)", fillcolor="#7F1D1D", fontcolor="#F87171", color="#EF4444"];
        }

        subgraph cluster_eval {
            label = "Programmatic Evaluation Framework";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            custom_eval [label="Custom Evaluator Suite\\n(Deterministic, Embedding, Rubric)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            eval_harness [label="client.evaluate() Harness\\n(Parallel Dataset Ingestion)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            ab_compare [label="Prompt A/B Comparison\\n(Win Rate, Latency, Cost Deltas)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        agent_node -> cond_edge;
        cond_edge -> tool_node [label="calls tool"];
        tool_node -> agent_node [label="returns state update"];
        agent_node -> trace_stream [label="emits node span"];
        tool_node -> trace_stream [label="emits tool span"];

        trace_stream -> token_agg;
        trace_stream -> error_audit;

        eval_harness -> custom_eval [label="applies evaluators"];
        custom_eval -> ab_compare [label="experiment metrics"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Key Intermediate Concept:</strong> LangSmith natively recognizes LangGraph state machines. 
    Every node invocation becomes a child span, and the complete state dictionary before and after execution 
    is recorded in the run's metadata, making cyclic debugging transparent.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: BUILD A CUSTOM EVALUATOR
# ==============================================================================
st.header("2. Build a Custom Evaluator: Deterministic and RunEvaluator Protocols", divider="rainbow")

st.markdown("""
In LangSmith, a **custom evaluator** is a function that inspects a completed `Run` alongside its corresponding 
dataset `Example` and returns a structured feedback dictionary.
""")

st.markdown("""
### The Evaluation Protocol Contract
A standard LangSmith evaluator function accepts `(run, example)` and returns:
- `key`: The metric identifier string (e.g., `"exact_match"`, `"toxicity"`, `"json_valid"`).
- `score`: A numeric grade (float between `0.0` and `1.0` or integer `0`/`1`).
- `comment` (optional): Human-readable explanation of why the score was assigned.
""")

st.code('''from typing import Dict, Any
from langsmith.schemas import Run, Example

def evaluate_json_schema(run: Run, example: Example) -> Dict[str, Any]:
    """
    Custom evaluator: verifies that output is valid JSON and contains required keys.
    """
    import json
    prediction = run.outputs.get("output", "")
    
    try:
        parsed = json.loads(prediction)
        has_required_keys = "summary" in parsed and "confidence" in parsed
        score = 1.0 if has_required_keys else 0.5
        comment = "Valid JSON with all required keys." if score == 1.0 else "Valid JSON but missing keys."
    except Exception as e:
        score = 0.0
        comment = f"JSON parse error: {str(e)}"
        
    return {
        "key": "json_schema_valid",
        "score": score,
        "comment": comment
    }
''', language="python")

# ==============================================================================
# SECTION 3: EVALUATE A PROMPT
# ==============================================================================
st.header("3. Evaluate a Prompt: Dataset Benchmarking and Running `evaluate()`", divider="rainbow")

st.markdown("""
To evaluate a prompt, wrap your prompt template and model invocation in a callable target function, 
then pass it to LangSmith's programmatic `evaluate()` runner.
""")

st.code('''from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define prompt application under test
def prompt_v1_pipeline(inputs: dict) -> dict:
    topic = inputs["topic"]
    # Simulated model call with Prompt V1
    prompt = f"Explain {topic} in one concise sentence."
    # response = llm.invoke(prompt)
    return {"output": f"{topic} is a distributed tracing platform for LLMs."}

# Run programmatic evaluation against benchmark dataset
experiment_results = evaluate(
    prompt_v1_pipeline,
    data="educational-concepts-benchmark",
    evaluators=[evaluate_json_schema],
    experiment_prefix="prompt-v1-baseline",
    max_concurrency=2
)
print("Evaluation Experiment Name:", experiment_results.experiment_name)
''', language="python")

# ==============================================================================
# SECTION 4: COMPARE TWO PROMPTS
# ==============================================================================
st.header("4. Compare Two Prompts: Side-by-Side Experiments and Win-Rate Calculation", divider="rainbow")

st.markdown("""
Comparing two prompts requires executing both candidates against the identical dataset under 
identical conditions, then performing a differential metric analysis.
""")

st.code('''# Evaluating Prompt Candidate A
exp_a = evaluate(
    prompt_v1_pipeline,
    data="educational-concepts-benchmark",
    evaluators=[evaluate_json_schema],
    experiment_prefix="prompt-v1-zero-shot"
)

# Evaluating Prompt Candidate B (Few-Shot CoT)
def prompt_v2_pipeline(inputs: dict) -> dict:
    topic = inputs["topic"]
    prompt = f"System: Always return JSON with 'summary' and 'confidence'. Topic: {topic}"
    return {"output": f'{{"summary": "{topic} is an LLM observability tool.", "confidence": 0.98}}'}

exp_b = evaluate(
    prompt_v2_pipeline,
    data="educational-concepts-benchmark",
    evaluators=[evaluate_json_schema],
    experiment_prefix="prompt-v2-few-shot"
)

# Compare Experiment Metrics via Client API
metrics_a = client.get_test_run(experiment_name=exp_a.experiment_name)
metrics_b = client.get_test_run(experiment_name=exp_b.experiment_name)

print("Candidate A JSON Score:", metrics_a.feedback_stats.get("json_schema_valid"))
print("Candidate B JSON Score:", metrics_b.feedback_stats.get("json_schema_valid"))
''', language="python")

# ==============================================================================
# SECTION 5: EVALUATE A RAG PIPELINE
# ==============================================================================
st.header("5. Evaluate a RAG Pipeline: Context Relevance, Groundedness, and Correctness", divider="rainbow")

st.markdown("""
A production RAG pipeline requires composite evaluators that audit the retriever and generator separately.
""")

st.code('''def evaluate_rag_retrieval(run: Run, example: Example) -> Dict[str, Any]:
    """Audits whether retriever fetched the expected document ID."""
    retrieved_docs = run.outputs.get("retrieved_docs", [])
    expected_doc_id = example.outputs.get("target_doc_id")
    
    found = any(doc.get("id") == expected_doc_id for doc in retrieved_docs)
    return {
        "key": "retrieval_hit_rate",
        "score": 1.0 if found else 0.0,
        "comment": f"Target document {expected_doc_id} found in top_k." if found else "Missing target document."
    }

def evaluate_rag_faithfulness(run: Run, example: Example) -> Dict[str, Any]:
    """Audits if answer contains claims not in context."""
    answer = run.outputs.get("answer", "").lower()
    context = " ".join([d.get("text", "") for d in run.outputs.get("retrieved_docs", [])]).lower()
    
    # Heuristic: checks keyword containment
    words = [w for w in answer.split() if len(w) > 4]
    grounded_words = sum(1 for w in words if w in context)
    ratio = grounded_words / max(len(words), 1)
    
    return {
        "key": "faithfulness",
        "score": round(ratio, 2)
    }
''', language="python")

# ==============================================================================
# SECTION 6: TRACE A LANGGRAPH WORKFLOW
# ==============================================================================
st.header("6. Trace a LangGraph Workflow: Instrumenting Nodes, Edges, and Checkpointers", divider="rainbow")

st.markdown("""
LangGraph workflows natively emit hierarchical spans for every node execution, state reducer update, 
and checkpointer snapshot when LangSmith tracing is active.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langsmith import traceable

class AgentState(TypedDict):
    messages: List[str]
    current_node: str

def planner_node(state: AgentState) -> dict:
    """Node 1: Generates plan."""
    return {"messages": state["messages"] + ["Plan: Retrieve vector context"], "current_node": "planner"}

def executor_node(state: AgentState) -> dict:
    """Node 2: Executes plan."""
    return {"messages": state["messages"] + ["Execution: Retrieved 3 chunks"], "current_node": "executor"}

def router_edge(state: AgentState) -> str:
    """Conditional Edge routing logic."""
    if len(state["messages"]) >= 3:
        return END
    return "executor"

# Build and Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", router_edge)
workflow.add_edge("executor", END)

app = workflow.compile()

# When executed, LangSmith captures 'planner', 'router_edge', and 'executor' as distinct spans!
final_state = app.invoke({"messages": ["User query: How does LangGraph trace?"], "current_node": "start"})
print("LangGraph final state:", final_state)
''', language="python")

# ==============================================================================
# SECTION 7: TRACK TOKEN USAGE
# ==============================================================================
st.header("7. Track Token Usage: Prompt vs Completion Breakdown and Aggregates", divider="rainbow")

st.markdown("""
Tracking token usage allows engineering teams to identify cost leaks, context explosions, and token spikes.
""")

st.code('''from langsmith import Client

client = Client()

def audit_project_token_usage(project_name: str):
    """Fetches runs and calculates aggregate token expenditures."""
    runs = client.list_runs(
        project_name=project_name,
        is_root=True,
        execution_order=1
    )
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for r in runs:
        if r.prompt_tokens:
            total_prompt_tokens += r.prompt_tokens
        if r.completion_tokens:
            total_completion_tokens += r.completion_tokens
            
    print(f"Project: {project_name}")
    print(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Combined Tokens:   {(total_prompt_tokens + total_completion_tokens):,}")

# audit_project_token_usage("marketing-copywriter-dev")
''', language="python")

# ==============================================================================
# SECTION 8: ANALYZE FAILED RUNS
# ==============================================================================
st.header("8. Analyze Failed Runs: Exception Filtering, Stack Traces, and Root Causes", divider="rainbow")

st.markdown("""
Production failures require targeted querying to isolate whether errors were due to third-party rate limits, 
vector database timeouts, or agent parsing crashes.
""")

st.code('''from langsmith import Client

client = Client()

def analyze_recent_failures(project_name: str, limit: int = 10):
    """Filters project runs for errors and displays root causes."""
    error_runs = client.list_runs(
        project_name=project_name,
        error="true",  # Filter only runs with exceptions
        limit=limit
    )
    
    print(f"--- FAILED RUNS AUDIT: {project_name} ---")
    for r in error_runs:
        print(f"Run ID:    {r.id}")
        print(f"Run Name:  {r.name}")
        print(f"Latency:   {r.total_tokens} tokens | {round((r.end_time - r.start_time).total_seconds(), 2)}s")
        print(f"Error Msg: {r.error}")
        print("-" * 50)

# analyze_recent_failures("customer-support-qa")
''', language="python")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Comprehensive Intermediate Evaluator & LangGraph Diagnostics Suite", divider="rainbow")

st.markdown("""
Below is the full, executable Python suite combining custom evaluators, programmatic evaluation, 
and token auditing into a production-ready testing module.
""")

st.code('''import os
from typing import Dict, Any
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

client = Client()

# 1. Custom Evaluator: Strict Accuracy and Brevity
def evaluate_exact_grounding(run, example) -> Dict[str, Any]:
    output = run.outputs.get("output", "")
    target = example.outputs.get("target", "")
    
    # Semantic token overlap
    out_tokens = set(output.lower().split())
    target_tokens = set(target.lower().split())
    overlap = len(out_tokens.intersection(target_tokens)) / max(len(target_tokens), 1)
    
    return {
        "key": "semantic_overlap",
        "score": round(overlap, 3),
        "comment": f"Token overlap: {round(overlap * 100, 1)}%"
    }

# 2. Target Traced Function
@traceable(run_type="chain", name="faq_extractor_chain")
def faq_extractor(inputs: Dict[str, Any]) -> Dict[str, Any]:
    text = inputs.get("text", "")
    # Simulates entity extraction
    return {
        "output": f"Extracted entities for: {text[:30]}"
    }

# 3. Execution Pipeline
DATASET_NAME = "intermediate-audit-ds"

if not client.has_dataset(dataset_name=DATASET_NAME):
    ds = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        inputs=[{"text": "LangGraph manages stateful cyclic agents."}],
        outputs=[{"target": "stateful cyclic agents"}],
        dataset_id=ds.id
    )

results = evaluate(
    faq_extractor,
    data=DATASET_NAME,
    evaluators=[evaluate_exact_grounding],
    experiment_prefix="intermediate-eval-run"
)
print(f"Experiment finished: {results.experiment_name}")
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Custom Evaluator & Prompt Comparison Sandbox", divider="rainbow")

st.markdown("""
Use the workbench below to test custom evaluator scoring logic, compare prompt candidate metrics, 
and inspect simulated LangGraph node traces.
""")

wb_tab = st.selectbox(
    "Select Interactive Tool:",
    ["1. Custom Token Overlap Evaluator Sandbox", "2. Prompt A/B Benchmark Comparator", "3. LangGraph Trace Inspector"]
)

if wb_tab == "1. Custom Token Overlap Evaluator Sandbox":
    st.subheader("Test Custom Evaluator Scoring")
    pred_str = st.text_input("Predicted Output:", "LangGraph provides stateful cyclic agent orchestration.")
    exp_str = st.text_input("Expected Ground Truth:", "Stateful cyclic agent orchestration.")

    if st.button("Calculate Evaluator Score"):
        eval_res = intermediate_engine.evaluate_custom_similarity(pred_str, exp_str)
        st.markdown(f"**Overlap Score:** `{eval_res['overlap_score']}`")
        if eval_res["passed"]:
            st.success(f"Status: {eval_res['status']} (Threshold >= 0.70 reached)")
        else:
            st.error(f"Status: {eval_res['status']} (Below required threshold)")

elif wb_tab == "2. Prompt A/B Benchmark Comparator":
    st.subheader("Prompt A/B Side-by-Side Metrics")
    comp_df = pd.DataFrame(intermediate_engine.prompt_experiments).T
    st.dataframe(comp_df, use_container_width=True)
    st.info("Prompt V2 demonstrates +16% accuracy improvement with 0.5% failure rate, justifying the higher token count.")

else:
    st.subheader("LangGraph Node Execution Tracer")
    sel_node = st.selectbox("Select Graph Node to Simulate:", ["planner", "tools", "executor"])
    user_state = st.text_input("Current State Message:", "User asks for search query")

    if st.button("Simulate Node Run Span"):
        step_trace = intermediate_engine.simulate_langgraph_step_trace(sel_node, user_state)
        st.markdown(f"**Node Name:** `{step_trace['node_name']}`")
        st.markdown(f"**Execution Latency:** `{step_trace['step_latency_ms']}ms` | **Tokens Generated:** `{step_trace['tokens_generated']}`")
        st.markdown(f"**State Snapshot Keys:** `{step_trace['state_snapshot_keys']}`")
        st.success(f"**Conditional Routing Edge:** `{step_trace['next_edge']}`")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do you structure a custom evaluator function in LangSmith to return multi-metric feedback?"):
    st.markdown("""
    **Answer:**  
    A custom evaluator can return either a single feedback dictionary or a list of dictionaries. 
    To return multi-metric feedback from a single pass, return a list such as:
    `[{"key": "accuracy", "score": 0.95}, {"key": "latency_compliance", "score": 1.0}]`.
    LangSmith unpacks and logs each key as an independent metric.
    """)

with st.expander("Q2: How does LangSmith capture intermediate state changes in cyclic LangGraph agents?"):
    st.markdown("""
    **Answer:**  
    LangGraph natively passes execution callbacks into every compiled node. 
    At each step, LangGraph creates a child run span in LangSmith containing the input state before node execution, 
    the delta returned by the node, and the post-reduction state dictionary stored in run metadata.
    """)

with st.expander("Q3: Why should prompt comparisons always use versioned dataset splits rather than production random samples?"):
    st.markdown("""
    **Answer:**  
    Production traffic fluctuates in difficulty, query length, and domain distribution. 
    Comparing Prompt A on Monday against Prompt B on Tuesday introduces external variance. 
    A fixed, versioned benchmark dataset ensures both prompts are evaluated on the exact same distribution.
    """)

with st.expander("Q4: How do you query LangSmith programmatically to extract all runs that exceeded a specific latency threshold?"):
    st.markdown("""
    **Answer:**  
    Use `client.list_runs(project_name=..., filter='gte(duration, 5.0)')`. 
    The LangSmith filter language supports comparative operators (`gte`, `lte`, `eq`, `contains`) across duration, 
    error status, run type, and custom metadata keys.
    """)

with st.expander("Q5: What is the primary operational danger of ignoring token usage in recursive agent workflows?"):
    st.markdown("""
    **Answer:**  
    If an agent loops recursively while appending full conversation history and tool outputs to its state, 
    prompt token volume grows quadratically with each step (O(N^2)). This quickly leads to context window 
    exhaustion, latency degradation, and massive cost spikes.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Evaluating Without Specifying max_concurrency in client.evaluate()**  
Running evaluation over large datasets with default or unbounded concurrency can trigger HTTP 429 rate limits 
from model providers, causing benchmark runs to fail midway. Always set `max_concurrency=4` or similar.
""")

st.error("""
**Trap 2: Swallowing Exceptions in Custom Evaluators**  
If a custom evaluator contains a bug and crashes, failing to log the error can silently mark test runs 
as un-scored without alerting the engineer.
""")

st.warning("""
**Trap 3: Over-Optimizing Prompts for Benchmark Datasets (Benchmark Overfitting)**  
Tuning a prompt specifically to pass 20 hand-crafted edge cases without validating against general queries 
often degrades general reasoning capability.
""")

st.warning("""
**Trap 4: Forgetting to Track Completion vs Prompt Tokens Separately**  
Reporting only total tokens conceals whether cost increases are due to bloated input contexts or overly verbose model outputs.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>RunEvaluator</h4>
        <p>A protocol contract accepting a Run and Example to output structured scores, keys, and rationales.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>max_concurrency</h4>
        <p>An evaluate() parameter controlling the number of parallel test runs dispatched to avoid model provider rate limits.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>StateGraph Span</h4>
        <p>A child trace span generated for each LangGraph node transition capturing state input, delta, and post-reduction state.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>list_runs(filter=...)</h4>
        <p>The programmatic SDK method for querying traces filtered by error status, latency duration, or custom metadata tags.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can implement custom evaluators following the (run, example) -> dict contract.", value=True)
chk_2 = st.checkbox("I know how to execute programmatic prompt evaluations using client.evaluate().", value=True)
chk_3 = st.checkbox("I can perform side-by-side prompt comparisons and calculate win rates across experiments.", value=True)
chk_4 = st.checkbox("I understand how to evaluate RAG pipelines across context retrieval and answer faithfulness.", value=True)
chk_5 = st.checkbox("I know how LangSmith traces LangGraph state machines and captures node transitions.", value=True)
chk_6 = st.checkbox("I can filter and diagnose failed runs using list_runs() and inspect exception stack traces.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Intermediate LangSmith coding competencies verified! Ready for Advanced Coding Problems.")
