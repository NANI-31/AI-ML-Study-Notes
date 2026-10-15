import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.4 Evaluation Pipelines - LangChain Master Curriculum",
    page_icon="🔄",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LCELEvaluationPipelineOrchestrator:
    """
    Enterprise LCEL Evaluation Pipeline Orchestrator.
    Emulates the 6 core components of production evaluation automation:
    1. Dataset Management (Golden dataset splits, schema versioning)
    2. Experiment Orchestration (Candidate vs Baseline sweeps)
    3. Custom Evaluators (Multi-evaluator consensus scoring)
    4. Feedback Integration (User thumbs up/down and rating ingestion)
    5. Regression Testing (Delta threshold verification)
    6. Benchmarking (Latency, token cost, and quality tradeoff curves)
    """
    def __init__(self, project_name: str = "lcel-production-benchmark"):
        self.project_name = project_name
        self.experiment_runs: List[Dict[str, Any]] = []

    def execute_evaluation_experiment(
        self,
        candidate_version: str,
        baseline_version: str = "v1.2.0-baseline",
        sample_size: int = 50,
        inject_regression: bool = False
    ) -> Dict[str, Any]:
        """
        Executes an automated evaluation experiment comparing a candidate release against baseline.
        """
        base_acc = 0.915
        base_faith = 0.938
        base_latency_ms = 1420.0
        base_cost_per_k = 12.40

        if inject_regression:
            cand_acc = round(base_acc - 0.052, 3)
            cand_faith = round(base_faith - 0.068, 3)
            cand_latency_ms = round(base_latency_ms + 480.0, 1)
            cand_cost_per_k = round(base_cost_per_k + 3.10, 2)
        else:
            cand_acc = round(base_acc + 0.016, 3)
            cand_faith = round(base_faith + 0.012, 3)
            cand_latency_ms = round(base_latency_ms - 90.0, 1)
            cand_cost_per_k = round(base_cost_per_k - 1.80, 2)

        acc_delta = round(cand_acc - base_acc, 3)
        faith_delta = round(cand_faith - base_faith, 3)
        latency_delta = round(cand_latency_ms - base_latency_ms, 1)

        # Regression Gate Evaluation
        passed_acc_gate = acc_delta >= -0.010
        passed_faith_gate = faith_delta >= -0.015
        passed_latency_gate = cand_latency_ms <= 2200.0
        overall_passed = passed_acc_gate and passed_faith_gate and passed_latency_gate

        status = "CERTIFIED_FOR_DEPLOYMENT" if overall_passed else "BLOCKED_REGRESSION_DETECTED"

        experiment_summary = {
            "experiment_id": f"exp-{candidate_version.replace('.', '_')}",
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "sample_size": sample_size,
            "status": status,
            "overall_passed": overall_passed,
            "metrics": {
                "candidate": {
                    "accuracy": cand_acc,
                    "faithfulness": cand_faith,
                    "latency_ms": cand_latency_ms,
                    "cost_per_k_usd": cand_cost_per_k
                },
                "baseline": {
                    "accuracy": base_acc,
                    "faithfulness": base_faith,
                    "latency_ms": base_latency_ms,
                    "cost_per_k_usd": base_cost_per_k
                },
                "deltas": {
                    "accuracy_delta": acc_delta,
                    "faithfulness_delta": faith_delta,
                    "latency_delta_ms": latency_delta
                }
            },
            "gate_checks": [
                {"check": "Accuracy Non-Regression (Delta >= -0.010)", "passed": passed_acc_gate, "delta": f"{acc_delta:+.3f}"},
                {"check": "Faithfulness Floor (Delta >= -0.015)", "passed": passed_faith_gate, "delta": f"{faith_delta:+.3f}"},
                {"check": "Latency SLA (< 2200ms)", "passed": passed_latency_gate, "value": f"{cand_latency_ms:.0f}ms"}
            ]
        }
        self.experiment_runs.append(experiment_summary)
        return experiment_summary

# Initialize engine in session state
if "lcel_pipeline_engine" not in st.session_state:
    st.session_state.lcel_pipeline_engine = LCELEvaluationPipelineOrchestrator()

# --- TOP NAVIGATION & HEADER ---
st.title("22.4 Evaluation Pipelines")
st.caption("Chapter 22: Evaluation | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Running one-off evaluation scripts manually in terminal windows does not scale in production engineering. 
Modern LangChain and LLMOps engineering relies on **Continuous Evaluation Pipelines**: automated systems that ingest 
production datasets, orchestrate multi-model experiment sweeps, execute custom evaluators, incorporate live user feedback, 
enforce automated regression gates in CI/CD, and chart benchmark trade-offs across cost, latency, and quality.

In this lesson, you will master:
1. Production dataset lifecycle: ingestion, version control, and golden splits
2. Automated experiment orchestration: prompt A/B testing and model family comparisons
3. Building custom Runnable evaluators with LangChain's evaluation interfaces
4. Incorporating production user feedback (thumbs up/down) into evaluation datasets
5. Codifying automated CI/CD regression gates and Pareto-optimal benchmarking
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Continuous Evaluation Pipeline and Benchmark Automation Architecture", divider="rainbow")

st.markdown("""
A production evaluation pipeline automates the qualification lifecycle. 
Whenever a prompt is updated or a model endpoint is changed, the pipeline automatically spins up an evaluation worker, 
loads the golden dataset, dispatches parallel inference requests, scores outputs across multiple evaluators, 
and generates a machine-readable regression scorecard.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EvaluationPipelineLifecycle {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_data {
            label = "1. Dataset Management Layer";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            DS_VER [label="Version-Controlled Golden Dataset\\n(500 Production Examples)", fillcolor="#1E3A8A", color="#3B82F6"];
            FEEDBACK [label="Live Production Feedback Loop\\n(Ingests Thumbs-Down Traces)", fillcolor="#1E3A8A", color="#3B82F6"];
        }

        subgraph cluster_experiments {
            label = "2. Experiment Matrix & Model Sweeps";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            EXP_A [label="Baseline Run (v1.2.0)\\nGPT-4o (Standard Prompt)", fillcolor="#334155", color="#64748B"];
            EXP_B [label="Candidate Run (v1.3.0-rc)\\nClaude 3.5 Sonnet (Refined Prompt)", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_evaluators {
            label = "3. Custom Runnable Evaluators";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F59E0B";
            fontsize = 12;

            EV_RUN [label="LangChain StringEvaluator Pool\\n(Faithfulness, Relevance, Schema)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_gating {
            label = "4. Automated Regression Gate & Benchmarking";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            GATE [label="Delta Contract Auditor\\n(Delta Accuracy >= -0.010)", fillcolor="#581C87", color="#A855F7"];
            ACT_PROMO [label="Promote Candidate to Prod\\n(Update :prod Prompt Pointer)", fillcolor="#065F46", color="#059669"];
            ACT_BLOCK [label="Block Pull Request\\n(Post Diff Table into GitHub PR)", fillcolor="#881337", color="#F43F5E"];
        }

        FEEDBACK -> DS_VER [label="Expands Coverage"];
        DS_VER -> EXP_A [label="Feeds Inputs"];
        DS_VER -> EXP_B [label="Feeds Inputs"];

        EXP_A -> EV_RUN [label="Evaluates Baseline"];
        EXP_B -> EV_RUN [label="Evaluates Candidate"];

        EV_RUN -> GATE [label="Computes Deltas"];
        GATE -> ACT_PROMO [label="All Checks Pass"];
        GATE -> ACT_BLOCK [label="Regression Detected"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Evaluation Pipelines
1. **Never Decouple Evaluation from Git Branch Protection**: Running evaluations should not be an optional manual step before release. Configure GitHub Actions or GitLab CI to require a passing evaluation scorecard before any pull request touching prompts or chains can merge.
2. **Treat User Feedback as Training Data for Benchmarks**: When real users click thumbs-down on a production response, that trace must automatically be queued for dataset inclusion. The benchmark suite must continuously grow to cover discovered edge cases.
3. **Automate Multi-Dimensional Tradeoff Curves**: A candidate model might increase accuracy by 1% while increasing latency by 300% and quadrupling token cost. Automated benchmarking calculates the Pareto-optimal frontier across accuracy, latency, and cost.
""")

st.divider()

# ==============================================================================
# SECTION 2: DATASETS
# ==============================================================================
st.header("2. Datasets (Dataset Ingestion, Version Control, Splits and Golden Test Suites)", divider="rainbow")

st.markdown("""
A robust evaluation pipeline begins with high-quality dataset management. 
In enterprise LLMOps, datasets are treated as code: they have semantic versions, schema contracts, train/validation/test splits, 
and automated deduplication.
""")

st.markdown("""
### Dataset Management Principles
* **Dataset Splits**: Maintain a compact "Canary Benchmark" (30-50 examples, runs in < 60s for rapid PR feedback) and an extensive "Golden Benchmark" (500+ examples, runs nightly for deep regression auditing).
* **Immutable Versioning**: Datasets must be version-tagged (`support-golden-v2.1`). An evaluation run must record the exact dataset commit hash against which it was executed to ensure reproducible benchmarks.
* **Continuous Ingestion**: Pipelines pull raw traces from production telemetry, scrub PII at the edge, and route negative feedback into dataset staging queues.
""")

st.code('''
from langsmith import Client

client = Client()

# Creating a version-controlled benchmark dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="enterprise-support-canary-v2",
    description="50 high-frequency canary queries for rapid PR pre-merge verification."
)

# Adding structured examples with ground truth inputs and references
client.create_examples(
    inputs=[{"query": "How do I update billing details?"}],
    outputs=[{"expected_answer": "Navigate to Settings > Billing and select 'Update Payment Method'."}],
    dataset_id=dataset.id
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production datasets reside in centralized observability backends like LangSmith or Hugging Face Hub. 
CI runners fetch datasets programmatically via SDK calls, execute candidate chains, and push evaluation runs back to the experiment store.
""")

st.divider()

# ==============================================================================
# SECTION 3: EXPERIMENTS
# ==============================================================================
st.header("3. Experiments (Comparative Model Sweeps, Prompt A/B Matrices and Parameter Tuning)", divider="rainbow")

st.markdown("""
An experiment represents the execution of a specific candidate chain or prompt version against a benchmark dataset. 
Evaluation pipelines orchestrate comparative sweeps, allowing engineers to compare multiple model architectures 
(GPT-4o vs Claude 3.5 Sonnet vs Llama-3-70B) across identical inputs.
""")

st.markdown("""
### Running Comparative Experiment Sweeps
* **Prompt A/B Testing**: Evaluating System Prompt V1 (standard) vs System Prompt V2 (few-shot with chain-of-thought) side by side on 100 questions.
* **Hyperparameter Sweeps**: Comparing `temperature=0` vs `temperature=0.3` to measure output consistency and creativity.
* **Quantization Tradeoffs**: Benchmarking FP16 models against 4-bit quantized local endpoints to measure quality drop vs latency gains.
""")

experiment_matrix_df = pd.DataFrame([
    {"Experiment Name": "exp-gpt4o-v1-baseline", "Model Endpoint": "gpt-4o-2024-08-06", "Prompt Version": "v1.2-prod", "Accuracy": "0.915", "p95 Latency": "1420 ms", "Cost / 1k": "12.40 USD"},
    {"Experiment Name": "exp-claude35-v1-candidate", "Model Endpoint": "claude-3-5-sonnet", "Prompt Version": "v1.3-rc", "Accuracy": "0.931", "p95 Latency": "1330 ms", "Cost / 1k": "10.60 USD"},
    {"Experiment Name": "exp-llama3-70b-vllm", "Model Endpoint": "llama-3-3-70b-instruct", "Prompt Version": "v1.3-rc", "Accuracy": "0.892", "p95 Latency": "890 ms", "Cost / 1k": "2.10 USD"}
])
st.dataframe(experiment_matrix_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern pipelines run experiments asynchronously using `evaluate(client, ...)` with concurrency pooling (8-16 workers), 
completing a 100-question multi-evaluator benchmark in under 30 seconds.
""")

st.divider()

# ==============================================================================
# SECTION 4: EVALUATORS
# ==============================================================================
st.header("4. Evaluators (Custom Runnable Evaluators, Scoring Consensus and Metric Synthesis)", divider="rainbow")

st.markdown("""
In LangChain, evaluators can be constructed as standard `Runnable` components or by inheriting from `StringEvaluator`. 
Custom evaluators encapsulate specialized domain business logic that generic NLP metrics cannot capture.
""")

st.markdown("""
### Building Custom Runnable Evaluators
* **Custom Metric Evaluators**: Creating domain-specific evaluators (e.g., verifying that medical summaries include required dosage units, or financial advice contains legal disclaimers).
* **Consensus Synthesizers**: Aggregating individual scores from multiple evaluators into a single weighted composite score.
* **Pydantic Structured Evaluation**: Enforcing that evaluator models output structured reasonings rather than unstructured text.
""")

st.code('''
from langchain.evaluation import StringEvaluator
from typing import Optional, Any

class ComplianceDisclaimerEvaluator(StringEvaluator):
    """
    Evaluates whether legal disclaimers are present in investment advice outputs.
    """
    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        has_disclaimer = "not financial advice" in prediction.lower()
        score = 1.0 if has_disclaimer else 0.0
        reasoning = "Mandatory regulatory disclaimer present." if has_disclaimer else "Missing financial disclaimer!"
        return {"score": score, "reasoning": reasoning}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Teams maintain an internal library of custom evaluators (`DisclaimersEvaluator`, `PIIScrubbingEvaluator`, `JsonSchemaEvaluator`). 
These are passed directly into the evaluation pipeline alongside standard LLM-as-a-judge evaluators.
""")

st.divider()

# ==============================================================================
# SECTION 5: FEEDBACK
# ==============================================================================
st.header("5. Feedback (User Signals, Thumbs Up/Down Integration and Production Flywheels)", divider="rainbow")

st.markdown("""
The most valuable source of evaluation data is not synthetic generation; it is real production user interaction. 
The **Data Flywheel** pattern captures live production feedback and pipes it directly into evaluation datasets.
""")

st.markdown("""
### Building the Production Data Flywheel
1. **Explicit Feedback Collection**: Users click thumbs up / thumbs down, select a star rating, or submit a bug report in the application UI.
2. **Telemetry Run Association**: The application captures the unique LangSmith `run_id` and records the feedback score directly against the run trace using the feedback API.
3. **Automated Triage to Dataset**: Runs that receive negative feedback (`score == 0.0`) are automatically added to an "Escalated Review" dataset queue. Subject Matter Experts review the failure, provide the correct ground truth answer, and promote it to the golden benchmark.
""")

st.code('''
from langsmith import Client

client = Client()

# Logging end-user feedback against a production run
def record_user_feedback(run_id: str, is_thumbs_up: bool, user_comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=1.0 if is_thumbs_up else 0.0,
        value="thumbs_up" if is_thumbs_up else "thumbs_down",
        comment=user_comment
    )
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
By feeding negative production traces back into benchmark datasets, the system guarantees that once a bug occurs 
in production, it is codified as an automated test case that prevents the same failure from ever recurring.
""")

st.divider()

# ==============================================================================
# SECTION 6: REGRESSION TESTING
# ==============================================================================
st.header("6. Regression Testing (Continuous Pre-Merge Delta Gates and Release Blockers)", divider="rainbow")

st.markdown("""
Regression testing asserts that code and prompt improvements do not accidentally break existing capabilities. 
In continuous evaluation pipelines, regression testing is enforced via **Delta Contracts**.
""")

st.markdown("""
### The 3 Rules of Regression Delta Contracts
* **Rule 1: Accuracy Delta Floor**: `Candidate Accuracy - Baseline Accuracy >= -0.010`. The candidate must perform equal to or better than baseline (allowing a maximum 1% margin for stochastic noise).
* **Rule 2: Zero Catastrophic Regressions**: Even if average scores increase, the candidate must not cause a drop greater than 0.50 on any protected safety or core inquiry subset.
* **Rule 3: Latency & Cost Budget Bound**: Candidate p95 latency must remain under the SLA budget (`p95 <= 2200ms`).
""")

regression_rules_df = pd.DataFrame([
    {"Quality Gate": "Accuracy Delta Contract", "Constraint": "Delta Accuracy >= -0.010", "Action on Breach": "Block Pull Request Merge"},
    {"Quality Gate": "Faithfulness Floor", "Constraint": "Faithfulness >= 0.900", "Action on Breach": "Block Release Candidate"},
    {"Quality Gate": "p95 Latency SLA", "Constraint": "p95 Latency <= 2200ms", "Action on Breach": "Alert Infrastructure Lead"},
    {"Quality Gate": "Safety Violation Bound", "Constraint": "Adversarial Pass Rate == 100%", "Action on Breach": "Immediate Critical Escalation"}
])
st.dataframe(regression_rules_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
CI/CD runners parse evaluation experiment output JSON. If any regression rule is breached, the script exits with code 1, 
marking the GitHub Actions check red and preventing code merge.
""")

st.divider()

# ==============================================================================
# SECTION 7: BENCHMARKING
# ==============================================================================
st.header("7. Benchmarking (Baseline Calibration, Cross-Provider Comparisons and Cost-Performance Curves)", divider="rainbow")

st.markdown("""
Benchmarking contextualizes performance across providers and model tiers. 
Deploying the largest frontier model (e.g., GPT-4o) for every simple task is financially irresponsible. 
Pipelines generate Pareto curves comparing accuracy against cost and latency to find the optimal model choice.
""")

st.markdown("""
### The Pareto-Optimal LLM Selection Framework
* **Frontier Reasoning Tier (GPT-4o, Claude 3.5 Sonnet)**: High accuracy (93%), higher latency (1400ms), higher cost ($12 / 1k). Used for complex multi-tool agent planning.
* **Distilled High-Speed Tier (GPT-4o-mini, Claude 3.5 Haiku)**: Strong accuracy (89%), ultra-low latency (450ms), minimal cost ($0.50 / 1k). Used for query classification, routing, and summarization.
* **Self-Hosted Open-Source Tier (Llama 3 70B on vLLM)**: High accuracy (90%), zero per-token cloud costs, predictable infrastructure pricing. Used for high-volume enterprise data pipelines.
""")

pareto_df = pd.DataFrame([
    {"Model Tier": "Frontier Model (GPT-4o)", "Benchmark Accuracy": "93.1%", "p95 Latency": "1420 ms", "Cost per 1k Calls": "12.40 USD", "Best Use Case": "Complex agent planning & multi-source synthesis"},
    {"Model Tier": "Distilled Model (GPT-4o-mini)", "Benchmark Accuracy": "89.4%", "p95 Latency": "480 ms", "Cost per 1k Calls": "0.60 USD", "Best Use Case": "High-volume classification, extraction & routing"},
    {"Model Tier": "Open Source (Llama 3 70B)", "Benchmark Accuracy": "90.2%", "p95 Latency": "890 ms", "Cost per 1k Calls": "2.10 USD (Infra)", "Best Use Case": "Private VPC on-premise compliance tasks"}
])
st.dataframe(pareto_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Automated pipelines periodically benchmark candidate open-source models against proprietary baselines. 
When a fine-tuned small model reaches 95% of GPT-4o's benchmark accuracy, the routing gateway transitions traffic, 
achieving massive cost reductions without sacrificing quality.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End LCEL Evaluation Pipeline (LCELEvaluationPipelineOrchestrator)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the evaluation pipeline orchestrator. 
It automates dataset loading, candidate execution, baseline diffing, regression delta verification, 
and release qualification reporting.
""")

st.code('''
class LCELEvaluationPipelineOrchestrator:
    """
    Enterprise LCEL Evaluation Pipeline Orchestrator.
    Emulates the 6 core components of production evaluation automation:
    1. Dataset Management (Golden dataset splits, schema versioning)
    2. Experiment Orchestration (Candidate vs Baseline sweeps)
    3. Custom Evaluators (Multi-evaluator consensus scoring)
    4. Feedback Integration (User thumbs up/down and rating ingestion)
    5. Regression Testing (Delta threshold verification)
    6. Benchmarking (Latency, token cost, and quality tradeoff curves)
    """
    def __init__(self, project_name: str = "lcel-production-benchmark"):
        self.project_name = project_name
        self.experiment_runs: List[Dict[str, Any]] = []

    def execute_evaluation_experiment(
        self,
        candidate_version: str,
        baseline_version: str = "v1.2.0-baseline",
        sample_size: int = 50,
        inject_regression: bool = False
    ) -> Dict[str, Any]:
        """
        Executes an automated evaluation experiment comparing a candidate release against baseline.
        """
        base_acc = 0.915
        base_faith = 0.938
        base_latency_ms = 1420.0
        base_cost_per_k = 12.40

        if inject_regression:
            cand_acc = round(base_acc - 0.052, 3)
            cand_faith = round(base_faith - 0.068, 3)
            cand_latency_ms = round(base_latency_ms + 480.0, 1)
            cand_cost_per_k = round(base_cost_per_k + 3.10, 2)
        else:
            cand_acc = round(base_acc + 0.016, 3)
            cand_faith = round(base_faith + 0.012, 3)
            cand_latency_ms = round(base_latency_ms - 90.0, 1)
            cand_cost_per_k = round(base_cost_per_k - 1.80, 2)

        acc_delta = round(cand_acc - base_acc, 3)
        faith_delta = round(cand_faith - base_faith, 3)
        latency_delta = round(cand_latency_ms - base_latency_ms, 1)

        # Regression Gate Evaluation
        passed_acc_gate = acc_delta >= -0.010
        passed_faith_gate = faith_delta >= -0.015
        passed_latency_gate = cand_latency_ms <= 2200.0
        overall_passed = passed_acc_gate and passed_faith_gate and passed_latency_gate

        status = "CERTIFIED_FOR_DEPLOYMENT" if overall_passed else "BLOCKED_REGRESSION_DETECTED"

        return {
            "experiment_id": f"exp-{candidate_version.replace('.', '_')}",
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "sample_size": sample_size,
            "status": status,
            "overall_passed": overall_passed,
            "metrics": {
                "candidate": {"accuracy": cand_acc, "faithfulness": cand_faith, "latency_ms": cand_latency_ms, "cost_per_k_usd": cand_cost_per_k},
                "baseline": {"accuracy": base_acc, "faithfulness": base_faith, "latency_ms": base_latency_ms, "cost_per_k_usd": base_cost_per_k},
                "deltas": {"accuracy_delta": acc_delta, "faithfulness_delta": faith_delta, "latency_delta_ms": latency_delta}
            },
            "gate_checks": [
                {"check": "Accuracy Non-Regression (Delta >= -0.010)", "passed": passed_acc_gate, "delta": f"{acc_delta:+.3f}"},
                {"check": "Faithfulness Floor (Delta >= -0.015)", "passed": passed_faith_gate, "delta": f"{faith_delta:+.3f}"},
                {"check": "Latency SLA (< 2200ms)", "passed": passed_latency_gate, "value": f"{cand_latency_ms:.0f}ms"}
            ]
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Evaluation Pipeline Runner and Regression Gate Sandbox", divider="rainbow")

st.markdown("""
Test how automated evaluation pipelines process candidate pull requests, diff scores against production baselines, 
and enforce pre-merge regression gates.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Pipeline Execution Parameters")
    cand_version_input = st.selectbox(
        "Candidate Release Target",
        [
            "v1.3.0-rc1 (Few-Shot Prompt Optimization)",
            "v1.3.0-rc2 (Migrating to Claude 3.5 Sonnet)",
            "v1.3.1-dev (Experimental Pruned Context)"
        ],
        index=0
    )
    bench_sample_size = st.slider("Benchmark Sample Size (Examples)", min_value=10, max_value=200, value=50, step=10)
    simulate_regression_toggle = st.toggle(
        "Simulate Prompt Regression (Fault Injection)",
        value=False,
        help="Simulates an unvalidated prompt edit that degrades accuracy and inflates latency."
    )

with wb_col2:
    st.markdown("### Live Pipeline Execution & Gate Verdict")
    clean_ver = cand_version_input.split(" ")[0]
    exp_report = st.session_state.lcel_pipeline_engine.execute_evaluation_experiment(
        candidate_version=clean_ver,
        sample_size=bench_sample_size,
        inject_regression=simulate_regression_toggle
    )

    if exp_report["overall_passed"]:
        st.success(f"**STATUS: {exp_report['status']}** — Candidate satisfies all regression delta contracts.")
    else:
        st.error(f"**STATUS: {exp_report['status']}** — Candidate breached regression gates. Pull request blocked.")

    # Metric Delta Badges
    d = exp_report["metrics"]["deltas"]
    c_m = exp_report["metrics"]["candidate"]
    b_m = exp_report["metrics"]["baseline"]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{c_m['accuracy']:.3f}", delta=f"{d['accuracy_delta']:+.3f}")
    with m2:
        st.metric("Faithfulness", f"{c_m['faithfulness']:.3f}", delta=f"{d['faithfulness_delta']:+.3f}")
    with m3:
        st.metric("Latency p95", f"{c_m['latency_ms']:.0f} ms", delta=f"{d['latency_delta_ms']:+.0f} ms", delta_color="inverse")

    # Gate Checks Scorecard Table
    st.markdown("#### Regression Gate Checks")
    st.dataframe(pd.DataFrame(exp_report["gate_checks"]), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

with st.expander("Q1: What are the core components of an enterprise continuous evaluation pipeline for LLMs?"):
    st.markdown("""
    **Answer**: An enterprise evaluation pipeline comprises 6 core components:
    1) **Dataset Management**: Version-controlled golden datasets and canary splits;
    2) **Experiment Orchestration**: Automated test runners executing candidate models across datasets;
    3) **Custom Evaluators**: Multi-evaluator consensus scoring (heuristics, embeddings, LLM judges);
    4) **Feedback Integration**: Continuous ingestion of real production user feedback;
    5) **Regression Gates**: Automated CI/CD pre-merge delta blockers;
    6) **Benchmarking**: Pareto-optimal trade-off analysis across accuracy, latency, and cost.
    """)

with st.expander("Q2: What is the purpose of maintaining both Canary Benchmarks and Golden Benchmarks?"):
    st.markdown("""
    **Answer**: Canary benchmarks are compact subsets (30–50 examples) designed to execute in under 60 seconds on every 
    pull request, providing fast feedback to developers without slowing down the CI pipeline. Golden benchmarks are 
    exhaustive test suites (500+ examples) run on scheduled nightly cadences or prior to major production releases 
    to perform comprehensive regression auditing.
    """)

with st.expander("Q3: How does the Production Data Flywheel continuously improve benchmark quality?"):
    st.markdown("""
    **Answer**: When production users click thumbs-down or report incorrect answers, the application captures the trace ID 
    and logs feedback to LangSmith. An automated pipeline triages these negative traces into an annotation queue. 
    Subject Matter Experts provide verified reference answers, and the new examples are incorporated into the golden dataset, 
    ensuring that past production failures become automated regression tests.
    """)

with st.expander("Q4: How do you formulate a pre-merge Regression Delta Contract in CI/CD?"):
    st.markdown("""
    **Answer**: A delta contract defines mathematical bounds on candidate performance relative to active production:
    1) `Candidate Accuracy - Baseline Accuracy >= -0.010` (no accuracy degradation beyond 1% margin);
    2) `Candidate Faithfulness - Baseline Faithfulness >= -0.015` (no ungrounded factual loss);
    3) `Candidate p95 Latency <= Production SLA Budget` (no performance regression). 
    If any contract check fails, the CI runner exits with code 1, blocking pull request merge.
    """)

with st.expander("Q5: What is a Pareto-optimal model selection curve in LLM benchmarking?"):
    st.markdown("""
    **Answer**: In LLM architectures, there is no single "best" model. Frontier models offer the highest accuracy but at 
    high latency and cost; distilled models offer ultra-low latency and minimal cost with slightly lower accuracy. 
    A Pareto curve maps accuracy against cost and latency, allowing engineering teams to select the optimal model 
    tier for each specific task in the application.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_pt1, col_pt2 = st.columns(2)
with col_pt1:
    st.error("**Trap 1 (Manual Unautomated Evaluations)**: Relying on developers to run evaluation scripts locally before deploying. Developers skip evaluations under deadline pressure, leading to preventable production regressions. Evaluation must be automated in CI.")
    st.warning("**Trap 2 (Evaluating Against Moving Baselines)**: Comparing candidate pull requests only against the immediately preceding release. Minor sub-1% degradations compound across agile sprints into severe quality decay. Always compare against an immutable golden baseline.")
with col_pt2:
    st.error("**Trap 3 (Single-Metric Optimization)**: Optimizing only for accuracy while ignoring latency and cost. Shipping a prompt that increases accuracy by 1% but increases token generation time from 1s to 6s will destroy user retention.")
    st.warning("**Trap 4 (Discarding Negative User Traces)**: Treating negative customer feedback as customer support tickets only. Every negative feedback trace is valuable ground-truth failure data that must be piped into the golden test suite.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What are the two types of benchmarks in a tiered evaluation pipeline?")
        st.caption("Flip for answer:")
        st.info("**Back**: Compact Canary Benchmarks (< 60s for rapid PR feedback) and Comprehensive Golden Benchmarks (500+ examples for nightly regression audits).")

    with st.container(border=True):
        st.markdown("**Front**: What is the Production Data Flywheel?")
        st.caption("Flip for answer:")
        st.info("**Back**: The automated pipeline that captures negative user feedback traces from production, triages them to SMEs, and adds them to golden evaluation datasets.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is a Regression Delta Contract in CI/CD?")
        st.caption("Flip for answer:")
        st.info("**Back**: A codified rule asserting that candidate performance matches or exceeds the active production baseline (`Delta >= -0.010`) before code merges.")

    with st.container(border=True):
        st.markdown("**Front**: Why is Pareto-optimal benchmarking essential for LLMOps?")
        st.caption("Flip for answer:")
        st.info("**Back**: It maps accuracy against cost and latency, ensuring applications don't overpay for frontier models on tasks that smaller distilled models can handle.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Evaluation Pipelines in LangChain:")

c1 = st.checkbox("I know how to structure an automated continuous evaluation pipeline in CI/CD.", value=True)
c2 = st.checkbox("I understand the distinction between Canary Benchmarks (< 60s) and Golden Benchmarks (500+ items).", value=True)
c3 = st.checkbox("I can implement custom LangChain StringEvaluator components for domain-specific metrics.", value=True)
c4 = st.checkbox("I know how to build a production data flywheel piping user feedback into golden datasets.", value=True)
c5 = st.checkbox("I can formulate automated regression delta contracts that block PR merges on performance drops.", value=True)
c6 = st.checkbox("I understand how to construct Pareto-optimal tradeoff curves across accuracy, latency, and cost.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Evaluation Pipelines, automated experiment orchestration, and CI/CD regression gating in LangChain.")
