import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.3 End-to-End Testing - LangSmith Master Curriculum",
    page_icon="🎯",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class EndToEndTestingEngine:
    """
    Production End-to-End (E2E) Testing & Regression Comparison Engine.
    Simulates complete user request lifecycles and automated regression comparison:
    1. Realistic user query ingestion with context
    2. Complete multi-step pipeline execution
    3. Final output synthesis and formatting validation
    4. Multi-metric evaluation (Accuracy, Grounding, Formatting, Latency)
    5. Baseline vs Candidate regression comparison with gating verdicts
    """
    def __init__(self, project_name: str = "e2e-regression-testing"):
        self.project_name = project_name
        self.experiments: List[Dict[str, Any]] = []

    def run_e2e_regression_comparison(
        self,
        test_suite_size: int = 5,
        candidate_model_variant: str = "v2_optimized_prompt",
        strict_latency_gate_ms: float = 2500.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation comparing Baseline (v1) against Candidate (v2).
        """
        # Benchmark questions
        queries = [
            {"id": "q1", "text": "What are our GDPR retention obligations for marketing cookies?", "ground_truth": "Marketing cookies require explicit consent and cannot be retained beyond 12 months without renewal."},
            {"id": "q2", "text": "Calculate prorated refund for annual subscription cancelled on day 120 of 365 (365 USD total).", "ground_truth": "Refund amount is 245.00 USD (1.00 USD/day for remaining 245 days)."},
            {"id": "q3", "text": "How do I configure OAuth2 PKCE flow in our React frontend?", "ground_truth": "Generate a code verifier and code challenge using SHA-256, then include code_challenge in /authorize."},
            {"id": "q4", "text": "Summarize corporate remote work travel policy in 3 bullet points.", "ground_truth": "Three bullet points covering per diem, approval thresholds, and booking tool requirements."},
            {"id": "q5", "text": "Explain difference between SOC-2 Type I and Type II audits.", "ground_truth": "Type I tests controls at a point in time; Type II tests operating effectiveness over 6-12 months."}
        ][:test_suite_size]

        baseline_runs = []
        candidate_runs = []

        for q in queries:
            # Baseline execution simulation (v1)
            b_lat = 1850.0 + float(len(q["text"]) * 5)
            b_acc = 0.88
            b_grd = 0.85
            b_fmt = 0.90
            baseline_runs.append({
                "id": q["id"], "query": q["text"],
                "latency_ms": b_lat, "accuracy": b_acc,
                "grounding": b_grd, "formatting": b_fmt,
                "composite": (b_acc * 0.4) + (b_grd * 0.4) + (b_fmt * 0.2)
            })

            # Candidate execution simulation (v2)
            if candidate_model_variant == "v2_optimized_prompt":
                c_lat = 1420.0 + float(len(q["text"]) * 4) # Faster & better
                c_acc = 0.94
                c_grd = 0.92
                c_fmt = 0.95
            else: # v2_regressed
                c_lat = 2750.0 + float(len(q["text"]) * 7) # Slower & worse
                c_acc = 0.79
                c_grd = 0.75
                c_fmt = 0.85

            candidate_runs.append({
                "id": q["id"], "query": q["text"],
                "latency_ms": c_lat, "accuracy": c_acc,
                "grounding": c_grd, "formatting": c_fmt,
                "composite": (c_acc * 0.4) + (c_grd * 0.4) + (c_fmt * 0.2)
            })

        # Aggregate metrics
        avg_base_acc = float(np.mean([r["accuracy"] for r in baseline_runs]))
        avg_cand_acc = float(np.mean([r["accuracy"] for r in candidate_runs]))
        delta_acc = avg_cand_acc - avg_base_acc

        avg_base_grd = float(np.mean([r["grounding"] for r in baseline_runs]))
        avg_cand_grd = float(np.mean([r["grounding"] for r in candidate_runs]))
        delta_grd = avg_cand_grd - avg_base_grd

        avg_base_lat = float(np.mean([r["latency_ms"] for r in baseline_runs]))
        avg_cand_lat = float(np.mean([r["latency_ms"] for r in candidate_runs]))
        delta_lat = avg_cand_lat - avg_base_lat

        # Gating check
        passed_accuracy = delta_acc >= -0.01 # No accuracy regression
        passed_grounding = delta_grd >= -0.01 # No grounding regression
        passed_latency = avg_cand_lat <= strict_latency_gate_ms

        all_passed = passed_accuracy and passed_grounding and passed_latency

        result = {
            "candidate_variant": candidate_model_variant,
            "test_suite_size": len(queries),
            "passed_release_gate": all_passed,
            "summary": {
                "Accuracy": {"baseline": avg_base_acc, "candidate": avg_cand_acc, "delta": delta_acc, "passed": passed_accuracy},
                "Grounding": {"baseline": avg_base_grd, "candidate": avg_cand_grd, "delta": delta_grd, "passed": passed_grounding},
                "Latency (ms)": {"baseline": avg_base_lat, "candidate": avg_cand_lat, "delta": delta_lat, "passed": passed_latency}
            },
            "detailed_candidate_runs": candidate_runs,
            "detailed_baseline_runs": baseline_runs
        }

        self.experiments.append(result)
        return result


# Instantiate engine
e2e_engine = EndToEndTestingEngine()

# --- HEADER SECTION ---
st.title("🎯 Chapter 29.3: End-to-End Testing & Regression Comparison")
st.markdown("""
Welcome to the production guide on **End-to-End (E2E) Testing and Regression Analysis** in **LangSmith**.
While unit tests examine isolated components and integration tests verify service boundaries,
**End-to-End Testing** evaluates the entire user journey from raw query to final delivered artifact.
Crucially, production E2E testing is not just about computing raw scores—it is about **Comparative Regression Analysis**:
quantifying the exact delta between the current production baseline and a proposed candidate release
to guarantee that no quality, grounding, or latency regressions reach users.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Full Lifecycle End-to-End Testing & Regression Comparison Architecture", divider="rainbow")

st.markdown("""
The architecture below illustrates the LangSmith E2E evaluation and regression comparison pipeline.
A Golden Benchmark dataset is evaluated concurrently across both the Production Baseline and Candidate Release.
Automated evaluators grade accuracy, grounding, formatting, and latency, feeding a differential scorecard
that enforces automated release gating.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph E2ETesting {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_dataset {
            label="1. Golden Benchmark Evaluation Dataset";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            bench_ds [label="LangSmith Benchmark Dataset\nN Curated Examples\n{input_query, ground_truth, metadata}", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_executions {
            label="2. Parallel Execution Branches";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            base_run [label="Production Baseline (v1.0)\nStable Active Production Chain\nReference Latency & Accuracy", fillcolor="#1E293B", color="#38BDF8"];
            cand_run [label="Candidate Release (v2.0)\nProposed Prompt / Model Update\nTarget for Promotion", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_evaluators {
            label="3. Automated Multi-Metric Evaluation";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            ev_acc [label="Semantic Accuracy Evaluator\nRubric-based LLM Judge", fillcolor="#1E293B", color="#38BDF8"];
            ev_grd [label="Contextual Grounding Evaluator\nFactual Faithfulness check", fillcolor="#1E293B", color="#38BDF8"];
            ev_lat [label="Latency & Token Evaluator\nExecution Duration & Cost", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_comparison {
            label="4. Differential Regression Analysis & Gating";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            delta_engine [label="LangSmith Delta Comparator\nDelta = Candidate - Baseline\nWin / Tie / Loss Statistical Breakdown", fillcolor="#0F172A", color="#10B981"];
            gate_pass [label="Release Approved: PROMOTED\nDelta Accuracy >= 0.0 & Latency within SLA", fillcolor="#064E3B", color="#10B981"];
            gate_block [label="Release Blocked: REGRESSION DETECTED\nDelta Accuracy < 0.0 or Latency Breached", fillcolor="#7F1D1D", color="#EF4444"];
        }

        bench_ds -> base_run [label="Feed Inputs"];
        bench_ds -> cand_run [label="Feed Inputs"];

        base_run -> ev_acc;
        base_run -> ev_grd;
        base_run -> ev_lat;

        cand_run -> ev_acc;
        cand_run -> ev_grd;
        cand_run -> ev_lat;

        ev_acc -> delta_engine;
        ev_grd -> delta_engine;
        ev_lat -> delta_engine;

        delta_engine -> gate_pass [label="No Regressions", color="#10B981"];
        delta_engine -> gate_block [label="Regression Found", color="#EF4444"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of End-to-End Testing
The following table summarizes the operational workflow for each syllabus topic:
""")

e2e_topics_df = pd.DataFrame([
    {
        "Pillar": "1. User Query",
        "Role in E2E": "Realistic user prompt ingestion with multi-turn chat history context",
        "Verification Metric": "Input sanitization, persona fidelity, token budget adherence"
    },
    {
        "Pillar": "2. Complete Execution",
        "Role in E2E": "Unbroken invocation of the full graph/chain including live retrieval and tools",
        "Verification Metric": "Total turnaround latency (p95 <= 3,000ms), zero unhandled exceptions"
    },
    {
        "Pillar": "3. Final Response",
        "Role in E2E": "Synthesized text, structured JSON, or cited document deliverables",
        "Verification Metric": "Constraint compliance (e.g., bullet counts, negative words), format validity"
    },
    {
        "Pillar": "4. Evaluation",
        "Role in E2E": "Automated scoring across semantic accuracy, grounding, and tone",
        "Verification Metric": "LLM-as-a-judge rubric score, Ragas faithfulness score (0.0 to 1.0)"
    },
    {
        "Pillar": "5. Regression Comparison",
        "Role in E2E": "Direct statistical comparison against current production baseline",
        "Verification Metric": "Delta Score (Candidate - Baseline >= 0.0), Win/Tie/Loss ratio"
    }
])
st.dataframe(e2e_topics_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: USER QUERY
# ==============================================================================
st.header("2. User Query (Realistic Persona Ingestion, Semantic Normalization & Context Seeding)", divider="rainbow")

st.markdown("""
The entry point of an E2E test is the user query.
Unlike unit tests that test arbitrary isolated strings, E2E queries represent complete user states:
including session identifiers, user tenant metadata, previous conversation turns, and injected user personas.
""")

st.markdown("### Structuring E2E Query Payloads")
st.markdown("""
* **Conversational Context**: Passing multi-turn history (`chat_history: [HumanMessage, AIMessage]`) to test whether the system retains context across turns.
* **Metadata Tagging**: Attaching tenant credentials and authorization scopes to ensure security filters execute under realistic user permissions.
* **Persona Normalization**: Testing across varied phrasing styles: terse queries ("refund policy?"), verbose explanations, and queries with minor typographical errors.
""")

st.code('''
# Example E2E test case payload in LangSmith
e2e_input = {
    "query": "Can I get an invoice reprint for order #98214?",
    "chat_history": [
        ("human", "I made a purchase on January 15th."),
        ("ai", "I can help with that purchase. What is your order number?")
    ],
    "user_metadata": {"tier": "enterprise", "tenant_id": "cust_alpha"}
}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith:
* Datasets store inputs as structured JSON objects mirroring production API schemas.
* When running `evaluate()`, LangSmith unpacks these dictionaries into the pipeline's invocation parameters automatically.
""")

st.divider()

# ==============================================================================
# SECTION 3: COMPLETE EXECUTION
# ==============================================================================
st.header("3. Complete Execution (Multi-Turn Graph Traversal, Tool Call Execution & Latency Profiling)", divider="rainbow")

st.markdown("""
In E2E testing, the application executes as a black box:
the test harness provides the input query and waits for the entire pipeline (retrievers, tools,
chat models, parsers, and checkpointers) to complete.
""")

st.markdown("### What Complete Execution Validates")
st.markdown("""
1. **End-to-End Latency SLA**: Verifying that the combined execution time of all internal steps satisfies the user experience SLA (e.g., p95 <= 3,000ms).
2. **Token Economy**: Measuring the cumulative tokens consumed across all intermediate model calls to calculate cost-per-query.
3. **Trace Completeness**: Confirming that LangSmith captures an unbroken trace hierarchy linking the root user request down to every intermediate tool span.
""")

st.code('''
from langsmith import evaluate

# Running end-to-end evaluation with LangSmith SDK
experiment_results = evaluate(
    target=production_agent_app, # Full compiled LangGraph or LCEL chain
    data="enterprise-golden-benchmark-v1",
    evaluators=[accuracy_evaluator, grounding_evaluator],
    experiment_prefix="candidate-v2.1",
    metadata={"git_commit": "a4f82b1", "environment": "staging"}
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In automated release pipelines:
* LangSmith automatically displays the full execution tree for every benchmark item.
* If a candidate model takes 4 intermediate reasoning steps instead of 2, engineers can inspect the trace to see where the agent experienced indecision.
""")

st.divider()

# ==============================================================================
# SECTION 4: FINAL RESPONSE
# ==============================================================================
st.header("4. Final Response (Synthesis Coherence, Constraint Compliance & Formatting Invariants)", divider="rainbow")

st.markdown("""
The final response is the deliverable returned to the client.
E2E testing audits the final output for structural, lexical, and semantic invariants:
* **Constraint Compliance**: Ensuring the response obeys length constraints, forbidden word lists, and formatting directives (e.g., Markdown vs JSON).
* **Factual Completeness**: Verifying that all core elements of the ground-truth answer are present.
* **Tone & Brand Safety**: Auditing that the completion maintains professional, helpful, and non-defensive language.
""")

st.code('''
def assert_final_response_invariants(response: str, expected_keywords: list[str]) -> bool:
    """
    Validates structural and keyword constraints on the final response.
    """
    # 1. Non-empty check
    if not response or len(response.strip()) < 20:
        return False
        
    # 2. Check for required domain terms
    for kw in expected_keywords:
        if kw.lower() not in response.lower():
            return False
            
    # 3. Check for absence of hallucinated disclaimers
    if "as an ai language model" in response.lower():
        return False
        
    return True
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production evaluation:
* Automated programmatic evaluators check hard invariants (JSON schema, character limits) first.
* LLM-as-a-judge evaluators then evaluate nuanced qualities (clarity, coherence, completeness).
""")

st.divider()

# ==============================================================================
# SECTION 5: EVALUATION
# ==============================================================================
st.header("5. Evaluation (Multi-Metric LLM-as-a-Judge, Groundedness & Semantic Faithfulness Scoring)", divider="rainbow")

st.markdown("""
Evaluating end-to-end performance requires combining multiple complementary metrics.
Relying on a single metric (e.g., ROUGE or exact string match) fails in GenAI because models can phrase
accurate answers in countless valid ways.
""")

st.markdown("### The Three Core E2E Evaluation Metrics")
st.markdown("""
1. **Semantic Accuracy (Correctness)**: Does the generated answer communicate the same factual conclusions as the reference ground truth?
2. **Contextual Grounding (Faithfulness)**: Are all statements in the final response backed by retrieved documentation stored in intermediate spans?
3. **Task Completion (Fulfillment)**: Did the agent fulfill all multi-part sub-questions posed by the user?
""")

st.code('''
from langsmith.schemas import Run, Example

def semantic_correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Custom LangSmith LLM-as-a-judge evaluator for semantic accuracy.
    """
    prediction = run.outputs.get("output", "")
    ground_truth = example.outputs.get("ground_truth", "")
    
    score = llm_judge_score(prediction=prediction, reference=ground_truth)
    return {"key": "semantic_correctness", "score": score}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith:
* Evaluator results are recorded directly on the run records.
* The web UI provides aggregate metric charts displaying mean scores, score distributions, and standard deviations across the entire evaluation dataset.
""")

st.divider()

# ==============================================================================
# SECTION 6: REGRESSION COMPARISON
# ==============================================================================
st.header("6. Regression Comparison (Baseline vs Candidate Delta Analysis & Release Gating)", divider="rainbow")

st.markdown("""
The crowning achievement of E2E evaluation is **Regression Comparison**.
In production, a team never asks: *"Is the model good?"*
They ask: *"Is the proposed candidate release strictly better than or equal to what is currently serving production users?"*
""")

st.markdown("### The Delta Formulation")
st.markdown("""
For every metric (Accuracy, Grounding, Latency):
`Delta = Candidate_Score - Baseline_Score`

* **Delta Accuracy >= 0.0**: Candidate matches or improves on baseline accuracy.
* **Delta Grounding >= 0.0**: Candidate maintains or improves factual grounding.
* **Delta Latency <= 0.0**: Candidate is faster than or equal to baseline latency.
* **Win / Tie / Loss Ratio**: On how many specific benchmark examples did the candidate beat, tie, or lose to the baseline?
""")

st.code('''
# Automated CI/CD release gate asserting on regression deltas
def assert_no_regressions(candidate_summary, baseline_summary):
    delta_accuracy = candidate_summary["accuracy"] - baseline_summary["accuracy"]
    delta_latency = candidate_summary["latency_ms"] - baseline_summary["latency_ms"]
    
    # Release Gate Contracts
    assert delta_accuracy >= -0.01, f"Accuracy regressed by {delta_accuracy * 100:.1f}%! Release blocked."
    assert delta_latency <= 200.0, f"Latency increased by {delta_latency:.1f}ms! Release blocked."
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
The LangSmith web application features an **Experiment Comparison View**:
* Side-by-side tables display Candidate vs Baseline outputs for every single benchmark example.
* Examples where the candidate scored lower than the baseline are highlighted in bright red as **Regressions**, allowing instant debugging before deployment.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Evaluation & Regression Gating Engine (EndToEndTestingEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `EndToEndTestingEngine` models parallel evaluation across baseline and candidate versions,
computes metric deltas, and enforces automated release gating contracts.
""")

st.code('''
class EndToEndTestingEngine:
    """
    Production End-to-End (E2E) Testing & Regression Comparison Engine.
    Simulates complete user request lifecycles and automated regression comparison:
    1. Realistic user query ingestion with context
    2. Complete multi-step pipeline execution
    3. Final output synthesis and formatting validation
    4. Multi-metric evaluation (Accuracy, Grounding, Formatting, Latency)
    5. Baseline vs Candidate regression comparison with gating verdicts
    """
    def __init__(self, project_name: str = "e2e-regression-testing"):
        self.project_name = project_name
        self.experiments: List[Dict[str, Any]] = []

    def run_e2e_regression_comparison(...) -> Dict[str, Any]:
        """Executes end-to-end evaluation comparing Baseline (v1) against Candidate (v2)."""
        ...
''', language="python")

st.info("The E2E engine is instantiated in module scope and powers the interactive regression comparator below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: E2E Pipeline Simulator & Model Regression Comparator", divider="rainbow")

st.markdown("""
Simulate an end-to-end regression benchmark comparing the Production Baseline against a Candidate Release.
Inspect the Delta scorecards, examine per-query wins and regressions, and observe the automated deployment gating decision.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("E2E Regression Controls")

    candidate_variant = st.selectbox(
        "Select Candidate Release Variant:",
        options=["v2_optimized_prompt", "v2_regressed"],
        format_func=lambda v: "1. Candidate v2 (Optimized Prompt: Higher Acc, Lower Latency)" if v == "v2_optimized_prompt" else "2. Candidate v2 (Regressed Model: Lower Acc, Higher Latency)"
    )

    suite_size = st.slider(
        "Evaluation Benchmark Size (Queries):",
        min_value=3,
        max_value=5,
        value=5
    )

    latency_sla = st.slider(
        "Maximum Allowed p95 Latency SLA (ms):",
        min_value=1500.0,
        max_value=3500.0,
        value=2500.0,
        step=100.0
    )

    run_e2e_btn = st.button("🎯 Run E2E Regression Benchmark", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Regression Comparison Scorecard")

    if run_e2e_btn or len(e2e_engine.experiments) == 0:
        exp_res = e2e_engine.run_e2e_regression_comparison(
            test_suite_size=suite_size,
            candidate_model_variant=candidate_variant,
            strict_latency_gate_ms=latency_sla
        )
    else:
        exp_res = e2e_engine.experiments[-1]

    # Top Level Gate Metrics
    m1, m2, m3 = st.columns(3)
    status_label = "PROMOTED (RELEASE APPROVED)" if exp_res["passed_release_gate"] else "BLOCKED (REGRESSION CAUGHT)"
    m1.metric("CI/CD Release Decision", status_label)
    m2.metric("Benchmark Queries", exp_res["test_suite_size"])
    m3.metric("Candidate Variant", exp_res["candidate_variant"][:20])

    if exp_res["passed_release_gate"]:
        st.success("🎉 **Deployment Approved**: Candidate demonstrates zero accuracy regression and satisfies latency SLA.")
    else:
        st.error("🚨 **Deployment Blocked**: Candidate regressed compared to production baseline! Review delta analysis below.")

    # Metric Delta Table
    st.markdown("#### Differential Regression Analysis (Candidate vs Baseline)")
    summary_rows = []
    for metric_name, m_data in exp_res["summary"].items():
        delta_val = m_data["delta"]
        if "Latency" in metric_name:
            delta_str = f"{delta_val:+.1f} ms"
            status_txt = "FASTER" if delta_val <= 0 else "SLOWER"
        else:
            delta_str = f"{delta_val * 100:+.1f} %"
            status_txt = "IMPROVED" if delta_val >= 0 else "REGRESSED"

        summary_rows.append({
            "Metric": metric_name,
            "Baseline (v1)": f"{m_data['baseline'] * 100:.1f}%" if "Latency" not in metric_name else f"{m_data['baseline']:.1f} ms",
            "Candidate (v2)": f"{m_data['candidate'] * 100:.1f}%" if "Latency" not in metric_name else f"{m_data['candidate']:.1f} ms",
            "Delta (v2 - v1)": delta_str,
            "Verdict": "PASSED" if m_data["passed"] else "FAILED"
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Detailed Per-Query Scorecard
    st.markdown("#### Per-Query Performance Comparison")
    detail_data = []
    for i, c_run in enumerate(exp_res["detailed_candidate_runs"]):
        b_run = exp_res["detailed_baseline_runs"][i]
        diff_score = c_run["composite"] - b_run["composite"]
        detail_data.append({
            "Query ID": c_run["id"],
            "Query Preview": c_run["query"][:35] + "...",
            "Baseline Score": f"{b_run['composite'] * 100:.1f}%",
            "Candidate Score": f"{c_run['composite'] * 100:.1f}%",
            "Score Delta": f"{diff_score * 100:+.1f}%",
            "Outcome": "WIN" if diff_score > 0 else ("TIE" if diff_score == 0 else "LOSS")
        })
    st.dataframe(pd.DataFrame(detail_data), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the primary difference between Integration Testing and End-to-End Testing?"):
    st.markdown("""
    **Answer:**
    * **Integration Testing**: Verifies that specific connected components (e.g., prompt to model, retriever to vector store) communicate across service boundaries correctly.
    * **End-to-End Testing**: Evaluates the complete system as a black box from the end user's perspective. It ingests realistic user queries, executes the entire pipeline, and evaluates final response accuracy, grounding, constraint compliance, and latency against a Golden Benchmark.
    """)

with st.expander("Q2: Why is Delta Analysis more valuable than absolute scoring in release gating?"):
    st.markdown("""
    **Answer:**
    Absolute scores can fluctuate based on benchmark difficulty or evaluator prompt calibration.
    Delta Analysis (`Delta = Candidate - Baseline`) controls for evaluator subjectivity by comparing the candidate directly against the current production baseline on the exact same benchmark queries.
    This guarantees that promotions are granted only if the candidate release demonstrably matches or exceeds the baseline.
    """)

with st.expander("Q3: How do you identify regression in a candidate model that has higher average accuracy?"):
    st.markdown("""
    **Answer:**
    A candidate model can improve its global average accuracy (e.g., from 85% to 88%) while simultaneously experiencing catastrophic regressions on specific query slices (e.g., scoring 0% on a critical compliance task where the baseline scored 100%).
    In LangSmith:
    1. Inspect the **Loss Ratio** in the experiment comparison view.
    2. Filter specifically for examples where `candidate_score < baseline_score`.
    3. Review the regressed examples to ensure the model did not sacrifice critical safety or core features for minor gains elsewhere.
    """)

with st.expander("Q4: What role does Contextual Grounding play in E2E RAG evaluation?"):
    st.markdown("""
    **Answer:**
    Contextual Grounding (Faithfulness) evaluates whether every claim in the model's final response can be directly inferred from the documents retrieved during that specific run.
    Without grounding evaluation, a model might generate a factually true answer based on its training weights while ignoring retrieved internal documentation, masking broken retrieval pipelines.
    """)

with st.expander("Q5: How do you prevent E2E evaluation suites from consuming excessive token budgets?"):
    st.markdown("""
    **Answer:**
    1. **Two-Tier Benchmarking**: Run a compact canary regression suite (30-50 examples) on pull requests; reserve the full 500-example E2E suite for nightly runs or release candidates.
    2. **Cost-Effective LLM Judges**: Use calibrated smaller models (e.g., `gpt-4o-mini`) for routine evaluations, reserving frontier models (`gpt-4o`) for final release audits.
    3. **Caching**: Cache retrieved document contexts across runs so that prompt variations do not re-trigger expensive vector search and embedding computations.
    """)

st.markdown("### Production Traps & Failure Modes")

e2e_tr_col1, e2e_tr_col2 = st.columns(2)

with e2e_tr_col1:
    st.error("""
    **Trap 1: Gating Exclusively on Global Average Accuracy**
    * **Failure**: Approving a candidate release because its overall benchmark score increased from 86% to 89%.
    * **Consequence**: The candidate broke 5 critical billing calculation queries that users rely on daily, causing billing errors in production.
    * **Remedy**: Always inspect the individual regression loss list; enforce a zero-regression contract on critical tier queries.
    """)

    st.error("""
    **Trap 2: Neglecting Latency Regressions During Model Upgrades**
    * **Failure**: Upgrading to a larger model that achieves +2% higher accuracy without tracking latency impact.
    * **Consequence**: End-to-end response time jumps from 1.5s to 5.8s, causing users to abandon sessions and leading to client timeouts.
    * **Remedy**: Include strict latency SLAs (e.g., p95 <= 3,000ms) alongside accuracy metrics in automated release gating.
    """)

with e2e_tr_col2:
    st.warning("""
    **Trap 3: Evaluating Against Un-Versioned Baselines**
    * **Failure**: Comparing candidate runs against a historical baseline recorded under different evaluation prompts.
    * **Consequence**: Inconsistent evaluator grading corrupts delta comparisons, creating false positives or masking true regressions.
    * **Remedy**: Re-evaluate the baseline concurrently or use fixed, immutable evaluation rubric versions in LangSmith.
    """)

    st.warning("""
    **Trap 4: Evaluating Without Contextual Grounding Checks**
    * **Failure**: Evaluating final responses only for superficial plausibility without comparing against retrieved state chunks.
    * **Consequence**: Hallucinated answers slip past evaluation into production because they sound confident and well-written.
    * **Remedy**: Always include an automated Grounding/Faithfulness evaluator in production RAG E2E test suites.
    """)

st.markdown("### Core Technical Flashcards")

fce2e_c1, fce2e_c2 = st.columns(2)

with fce2e_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the mathematical definition of a Regression Delta in E2E evaluation?")
        st.markdown("**Back**: `Delta = Candidate_Score - Baseline_Score`. A positive delta indicates improvement; a negative delta signals regression.")

    with fce2e_c2:
        st.markdown("**Front**: Why must E2E testing evaluate Contextual Grounding in addition to Accuracy?")
        st.markdown("**Back**: To ensure the model derived its answer strictly from retrieved knowledge rather than hallucinating from pre-trained parametric memory.")

with fce2e_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is a Win / Tie / Loss ratio in LangSmith experiment comparison?")
        st.markdown("**Back**: The count of benchmark queries where the candidate performed better than (Win), identical to (Tie), or worse than (Loss) the production baseline.")

    with fce2e_c2:
        st.markdown("**Front**: Why is black-box E2E testing critical alongside white-box unit testing?")
        st.markdown("**Back**: Because even if all individual units pass, emergent multi-step agent behaviors and compounding prompt interactions can only be measured at the system boundary.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of End-to-End testing and regression gating:")

e2e_chk1 = st.checkbox("I understand the end-to-end evaluation lifecycle from user query to differential regression analysis.", value=True)
e2e_chk2 = st.checkbox("I can structure realistic user payloads with conversational chat history and metadata context.", value=True)
e2e_chk3 = st.checkbox("I know how to profile complete execution latency, token burn, and trace hierarchies in LangSmith.", value=True)
e2e_chk4 = st.checkbox("I can implement multi-metric evaluation combining semantic accuracy, grounding, and formatting.", value=True)
e2e_chk5 = st.checkbox("I know how to compute regression deltas (Candidate - Baseline) and analyze Win/Tie/Loss ratios.", value=True)
e2e_chk6 = st.checkbox("I can enforce automated CI/CD release gating contracts that prevent regressions from reaching production.", value=True)

if e2e_chk1 and e2e_chk2 and e2e_chk3 and e2e_chk4 and e2e_chk5 and e2e_chk6:
    st.success("Mastery Certified: You have mastered End-to-End testing, multi-metric evaluation, and differential regression gating in LangSmith.")
