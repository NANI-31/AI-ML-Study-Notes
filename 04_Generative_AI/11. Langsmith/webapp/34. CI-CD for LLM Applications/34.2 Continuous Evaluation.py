import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="34.2 Continuous Evaluation - LangSmith Master Curriculum",
    page_icon="🧪",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ContinuousEvaluationOrchestratorEngine:
    """
    Enterprise Continuous Evaluation Orchestrator for LLM Applications.
    Emulates scheduled recurring benchmark evaluation, statistical quality gating,
    experiment diffing, automated failure detection, and canary model validation.
    """
    def __init__(self, project_name: str = "prod-customer-rag-continuous-eval"):
        self.project_name = project_name
        self.evaluation_history: List[Dict[str, Any]] = []

    def run_continuous_evaluation(
        self,
        candidate_version: str,
        baseline_version: str = "v2.4.0-prod",
        min_accuracy_threshold: float = 0.88,
        min_faithfulness_threshold: float = 0.90,
        max_hallucination_rate: float = 0.05,
        max_latency_p95_ms: float = 2400.0,
        inject_failure_drift: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a scheduled continuous evaluation run comparing a candidate model/prompt
        artifact against the production golden baseline across synthetic and production traffic samples.
        """
        # Baseline reference metrics
        base_acc = 0.912
        base_faith = 0.935
        base_halluc = 0.038
        base_p95 = 1850.0

        # Candidate metrics generation
        if inject_failure_drift:
            cand_acc = round(base_acc - 0.065, 3)
            cand_faith = round(base_faith - 0.082, 3)
            cand_halluc = round(base_halluc + 0.045, 3)
            cand_p95 = round(base_p95 + 720.0, 1)
        else:
            cand_acc = round(base_acc + 0.018, 3)
            cand_faith = round(base_faith + 0.012, 3)
            cand_halluc = round(max(0.015, base_halluc - 0.012), 3)
            cand_p95 = round(base_p95 - 90.0, 1)

        # Delta calculations
        acc_delta = round(cand_acc - base_acc, 3)
        faith_delta = round(cand_faith - base_faith, 3)
        halluc_delta = round(cand_halluc - base_halluc, 3)
        latency_delta = round(cand_p95 - base_p95, 1)

        # SLA Checks
        checks = [
            {
                "metric": "Accuracy SLA",
                "target": f">= {min_accuracy_threshold:.2f}",
                "candidate": f"{cand_acc:.3f}",
                "baseline": f"{base_acc:.3f}",
                "delta": f"{acc_delta:+.3f}",
                "passed": cand_acc >= min_accuracy_threshold and acc_delta >= -0.01
            },
            {
                "metric": "Faithfulness Floor",
                "target": f">= {min_faithfulness_threshold:.2f}",
                "candidate": f"{cand_faith:.3f}",
                "baseline": f"{base_faith:.3f}",
                "delta": f"{faith_delta:+.3f}",
                "passed": cand_faith >= min_faithfulness_threshold and faith_delta >= -0.015
            },
            {
                "metric": "Hallucination Bound",
                "target": f"<= {max_hallucination_rate:.2f}",
                "candidate": f"{cand_halluc:.3f}",
                "baseline": f"{base_halluc:.3f}",
                "delta": f"{halluc_delta:+.3f}",
                "passed": cand_halluc <= max_hallucination_rate and halluc_delta <= 0.01
            },
            {
                "metric": "Latency p95 Budget",
                "target": f"<= {max_latency_p95_ms:.0f}ms",
                "candidate": f"{cand_p95:.1f}ms",
                "baseline": f"{base_p95:.1f}ms",
                "delta": f"{latency_delta:+.1f}ms",
                "passed": cand_p95 <= max_latency_p95_ms and latency_delta <= 250.0
            }
        ]

        all_passed = all(c["passed"] for c in checks)
        status = "CERTIFIED" if all_passed else "REJECTED_QUALITY_GATED"

        report = {
            "evaluation_id": f"eval-sched-{candidate_version.replace('.', '_')}",
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "dataset_name": "golden-benchmark-rag-500",
            "sample_size": 500,
            "status": status,
            "overall_passed": all_passed,
            "checks": checks,
            "candidate_metrics": {
                "accuracy": cand_acc,
                "faithfulness": cand_faith,
                "hallucination_rate": cand_halluc,
                "p95_latency_ms": cand_p95
            },
            "baseline_metrics": {
                "accuracy": base_acc,
                "faithfulness": base_faith,
                "hallucination_rate": base_halluc,
                "p95_latency_ms": base_p95
            }
        }
        self.evaluation_history.append(report)
        return report

# Initialize engine in session state
if "cont_eval_engine" not in st.session_state:
    st.session_state.cont_eval_engine = ContinuousEvaluationOrchestratorEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("🧪 Chapter 34.2: Continuous Evaluation in LangSmith")
st.caption("Automated Scheduled Benchmarking, Statistical Quality Gating, Win/Loss Experiment Diffing, and Canary Model Certification")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Continuous Evaluation Pipeline and Quality Gating Architecture", divider="rainbow")

st.markdown("""
Continuous Evaluation (CE) is the disciplined operational practice of continuously running automated evaluation suites 
against LLM applications at scheduled cadence or upon trigger events. While Continuous Integration validates pull requests 
prior to merge, Continuous Evaluation perpetually assesses production telemetry, shadow traffic, and golden test suites 
to detect distribution shifts, model provider weight updates, and retrieval degradations before users experience failures.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ContinuousEvaluationPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_triggers {
            label = "1. Evaluation Triggers and Ingestion Layer";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            T1 [label="Scheduled Cron Job\\n(Nightly / Hourly Trigger)", fillcolor="#334155", color="#64748B"];
            T2 [label="Production Telemetry Sampling\\n(LangSmith Stream 5% Stratified)", fillcolor="#334155", color="#64748B"];
            T3 [label="Model Provider Alert\\n(Vendor Checkpoint Shift)", fillcolor="#334155", color="#64748B"];
        }

        subgraph cluster_datasets {
            label = "2. Benchmark Dataset Management";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            D1 [label="Golden Benchmark Suite\\n(500 Verified Multi-Turn Examples)", fillcolor="#1E3A8A", color="#3B82F6"];
            D2 [label="Edge-Case Adversarial Suite\\n(Prompt Injection and Boundary Traps)", fillcolor="#1E3A8A", color="#3B82F6"];
            D3 [label="Curated Production Failure Queue\\n(Triage from Human Dislikes)", fillcolor="#1E3A8A", color="#3B82F6"];
        }

        subgraph cluster_execution {
            label = "3. Asynchronous Evaluation Runner Pool";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            E1 [label="langsmith.evaluate() Daemon\\n(Configured Concurrency = 16)", fillcolor="#047857", color="#10B981"];
            E2 [label="LLM-as-a-Judge Cluster\\n(Faithfulness and Relevance)", fillcolor="#047857", color="#10B981"];
            E3 [label="Deterministic Heuristics Engine\\n(Exact Match, Latency, Regex)", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_gating {
            label = "4. Statistical Diffing and Quality Gate Floor";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            G1 [label="Experiment Result Diffing\\n(Candidate vs Active Baseline)", fillcolor="#7C2D12", color="#EA580C"];
            G2 [label="SLA Threshold Enforcer\\n(Accuracy >= 0.88, Faithfulness >= 0.90)", fillcolor="#7C2D12", color="#EA580C"];
            G3 [label="Automated Failure Detector\\n(Z-Score Drift, Anomaly Trigger)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_action {
            label = "5. Automated Governance and Routing";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            A1 [label="Pass: Challenger Model Certified\\n(Promoted to Canary / Prod Tag)", fillcolor="#065F46", color="#059669"];
            A2 [label="Fail: Deployment Blocked\\n(Incident Alerted to Slack and PagerDuty)", fillcolor="#881337", color="#F43F5E"];
        }

        T1 -> D1 [label="Triggers Evaluation"];
        T2 -> D3 [label="Streams Samples"];
        T3 -> D2 [label="Triggers Guard Verification"];

        D1 -> E1 [label="Batches Data"];
        D2 -> E1 [label="Injects Edge Cases"];
        D3 -> E1 [label="Tests Regressions"];

        E1 -> E2 [label="Dispatches Prompts"];
        E1 -> E3 [label="Executes Assertions"];

        E2 -> G1 [label="Calculates Scores"];
        E3 -> G1 [label="Returns Latencies"];

        G1 -> G2 [label="Computes Deltas"];
        G2 -> G3 [label="Verifies Bounds"];

        G3 -> A1 [label="All SLAs Met"];
        G3 -> A2 [label="Breached Floor"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Continuous Evaluation
1. **Never Rely on Static One-Off Benchmarks**: LLM applications change dynamically as third-party model providers silently roll out weights, embedding models update, and knowledge documents evolve. Scheduled continuous evaluation is the only guarantee of invariant output quality.
2. **Decouple Evaluation Cadence from Release Cadence**: Evaluating solely upon code merge leaves the system vulnerable between releases. Run scheduled nightly sweeps against frozen golden datasets to guarantee long-term stability.
3. **Dual-Gate Quality Contracts**: Every candidate evaluation must satisfy two distinct conditions: an absolute quality floor (e.g., faithfulness >= 0.90) and a relative non-regression constraint (`delta_faithfulness >= -0.015`).
""")

st.divider()

# ==============================================================================
# SECTION 2: RUNNING EVALUATION DATASETS
# ==============================================================================
st.header("2. Running Evaluation Datasets (Scheduled Batches, Concurrency Pools and Asynchronous Runners)", divider="rainbow")

st.markdown("""
Executing evaluation datasets at enterprise scale requires rigorous orchestration of concurrency, rate-limiting, and error retry semantics. 
Running a 1,000-example golden test suite sequentially against an LLM judge would take hours; executing all concurrently would trigger API rate limits (`429 Too Many Requests`).
""")

st.markdown("""
### Architectural Mechanics of Enterprise Evaluation Runners
* **Asynchronous Concurrency Throttling**: The LangSmith `evaluate()` execution runner accepts a `max_concurrency` parameter (typically set to 8–16 workers). This distributes requests across a controlled thread pool while adhering to token-per-minute (TPM) and requests-per-minute (RPM) limits.
* **Deterministic Environment Isolation**: Evaluation runners must pin all execution parameters: `temperature=0`, seed numbers when supported, and static vector store snapshots to guarantee that score variance stems from code or model weights rather than sampling randomness.
* **Resilient Checkpointing**: If an evaluation run is interrupted by network failure at example 450 of 500, LangSmith persists already evaluated rows, allowing runs to resume without re-running completed inferences.
""")

st.code('''
from langsmith import evaluate, Client
from my_rag_app import rag_application_target
from my_evaluators import faithfulness_evaluator, relevance_evaluator

client = Client()

# Asynchronous scheduled evaluation runner with concurrency controls
results = evaluate(
    rag_application_target,
    data="golden-production-benchmark-v3",
    evaluators=[faithfulness_evaluator, relevance_evaluator],
    experiment_prefix="scheduled-nightly-eval",
    max_concurrency=12,
    metadata={
        "environment": "production-canary",
        "cadence": "nightly-cron",
        "runner_commit": "git-sha-7e81b2a",
        "retriever_top_k": 5
    }
)

print(f"Evaluation complete. Run results URL: {results.experiment_name}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern cloud architectures (AWS ECS, Kubernetes CronJobs, or GitHub Actions Workflows), evaluation jobs run in headless worker containers scheduled via Cron. 
The container authenticates with LangSmith via `LANGCHAIN_API_KEY`, executes `evaluate()`, captures the resulting experiment URL, and exports machine-readable summary metrics as JSON artifacts for the deployment orchestrator.
""")

st.divider()

# ==============================================================================
# SECTION 3: COMPARING EXPERIMENT RESULTS
# ==============================================================================
st.header("3. Comparing Experiment Results (Baseline vs Candidate Diffing, Win/Loss Matrix and Delta Tracking)", divider="rainbow")

st.markdown("""
Evaluating a candidate in isolation provides raw scores, but tells engineering teams nothing about relative quality. 
A candidate scoring 0.89 accuracy might look acceptable until compared to the production baseline running at 0.93. 
LangSmith provides programmatic experiment comparison to calculate exact delta matrices and item-by-item win/loss/tie tallies.
""")

st.markdown("""
### The Three Dimensions of Experiment Diffing
1. **Global Aggregate Delta (Delta)**: The arithmetic difference between candidate aggregate mean and baseline aggregate mean across each metric.
2. **Item-Level Win/Loss Distribution**: Aggregates can hide regressions. A candidate might improve 100 questions by 0.05 points while severely breaking 20 questions by dropping 0.80 points, yielding a positive mean but introducing unacceptable catastrophic failures.
3. **Statistical Significance Test**: Verifying whether differences are statistically meaningful or merely stochastic noise using paired t-tests or Wilcoxon signed-rank tests across score distributions.
""")

diff_data = pd.DataFrame([
    {"Evaluation Metric": "Answer Accuracy", "Baseline (v2.4.0)": "0.912", "Candidate (v2.5.0-rc1)": "0.930", "Delta": "+0.018", "Win / Tie / Loss": "112 / 368 / 20", "Status": "IMPROVED"},
    {"Evaluation Metric": "Context Faithfulness", "Baseline (v2.4.0)": "0.935", "Candidate (v2.5.0-rc1)": "0.947", "Delta": "+0.012", "Win / Tie / Loss": "84 / 401 / 15", "Status": "IMPROVED"},
    {"Evaluation Metric": "Hallucination Rate", "Baseline (v2.4.0)": "0.038", "Candidate (v2.5.0-rc1)": "0.026", "Delta": "-0.012", "Win / Tie / Loss": "19 / 476 / 5", "Status": "IMPROVED"},
    {"Evaluation Metric": "p95 Latency (ms)", "Baseline (v2.4.0)": "1850 ms", "Candidate (v2.5.0-rc1)": "1760 ms", "Delta": "-90 ms", "Win / Tie / Loss": "230 / 180 / 90", "Status": "FASTER"},
    {"Evaluation Metric": "Cost per 1k Calls", "Baseline (v2.4.0)": "14.20 USD", "Candidate (v2.5.0-rc1)": "11.80 USD", "Delta": "-2.40 USD", "Win / Tie / Loss": "N/A", "Status": "SAVINGS"}
])
st.dataframe(diff_data, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Top-tier AI engineering teams require candidates to provide a net positive Win/Loss ratio (`Wins > 3 * Losses`). 
If a candidate has even 5 losses where the output is judged catastrophic (e.g., toxic hallucination or security disclosure), the release is blocked regardless of superior average performance across the remaining 495 examples.
""")

st.divider()

# ==============================================================================
# SECTION 4: QUALITY THRESHOLDS
# ==============================================================================
st.header("4. Quality Thresholds (Minimum Accuracy SLAs, Faithfulness Floors and Negative Metric Bounds)", divider="rainbow")

st.markdown("""
Quality thresholds in continuous evaluation act as automated circuit breakers. Without codified thresholds, 
engineering teams fall victim to subjective judgment calls, often shipping degraded models under deadline pressure. 
Enterprise quality gating establishes immutable, machine-enforceable rules that define what can run in production.
""")

st.markdown("""
### The Triad of Threshold Rules
* **Absolute Metric Floors**: The non-negotiable minimum score a candidate must reach regardless of baseline scores (e.g., `faithfulness >= 0.90` and `guardrail_safety >= 0.999`).
* **Relative Regression Bounds**: The maximum acceptable negative delta compared to active production (e.g., `delta_accuracy >= -0.010`). Even if absolute accuracy is 0.92, a candidate dropping from 0.94 is gated.
* **Negative Metric Ceilings**: Metrics where lower is better have hard upper limits (e.g., `hallucination_rate <= 0.040` and `p95_latency <= 2200ms`).
""")

st.code('''
class QualityThresholdContract:
    """
    Codified Quality Contract for Automated Evaluation Gating.
    """
    MIN_ACCURACY_FLOOR = 0.880
    MIN_FAITHFULNESS_FLOOR = 0.900
    MAX_HALLUCINATION_CEILING = 0.050
    MAX_LATENCY_P95_MS = 2400.0
    MAX_ALLOWED_REGRESSION_DELTA = -0.010

    @classmethod
    def evaluate_compliance(cls, candidate_metrics: dict, baseline_metrics: dict) -> Tuple[bool, List[str]]:
        violations = []

        # 1. Floor checks
        if candidate_metrics["accuracy"] < cls.MIN_ACCURACY_FLOOR:
            violations.append(f"Accuracy {candidate_metrics['accuracy']} breached floor {cls.MIN_ACCURACY_FLOOR}")
        if candidate_metrics["faithfulness"] < cls.MIN_FAITHFULNESS_FLOOR:
            violations.append(f"Faithfulness {candidate_metrics['faithfulness']} breached floor {cls.MIN_FAITHFULNESS_FLOOR}")

        # 2. Ceiling checks
        if candidate_metrics["hallucination_rate"] > cls.MAX_HALLUCINATION_CEILING:
            violations.append(f"Hallucination rate {candidate_metrics['hallucination_rate']} breached ceiling {cls.MAX_HALLUCINATION_CEILING}")
        if candidate_metrics["p95_latency_ms"] > cls.MAX_LATENCY_P95_MS:
            violations.append(f"Latency p95 {candidate_metrics['p95_latency_ms']}ms breached budget {cls.MAX_LATENCY_P95_MS}ms")

        # 3. Relative delta checks
        acc_delta = candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
        if acc_delta < cls.MAX_ALLOWED_REGRESSION_DELTA:
            violations.append(f"Accuracy delta {acc_delta:+.3f} exceeds allowed regression delta {cls.MAX_ALLOWED_REGRESSION_DELTA}")

        return (len(violations) == 0, violations)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production pipelines, these threshold violations trigger exit code 1 in the CI/CD runner. 
GitHub Actions or GitLab CI marks the job failed, blocks pull request merging, and posts a Markdown summary table into the PR discussion.
""")

st.divider()

# ==============================================================================
# SECTION 5: AUTOMATED FAILURE DETECTION
# ==============================================================================
st.header("5. Automated Failure Detection (Statistical Anomaly Triggers, Degradation Alerts and Notification Webhooks)", divider="rainbow")

st.markdown("""
Silent degradation is the primary mode of failure in production LLM applications. Unlike traditional software crashes 
that generate `500 Internal Server Error` HTTP status codes, LLM degradation manifests as subtly ungrounded responses, 
verbosity changes, or declining relevance while still returning `200 OK`. Automated failure detection continuously 
analyzes telemetry streams to trigger immediate alerts.
""")

st.markdown("""
### Core Detection Mechanisms
1. **Online Statistical Drift Detection**: Calculating moving 24-hour rolling averages of LLM judge scores and triggering alerts when the mean drops by more than 2 standard deviations (Z <= -2.0) from the 30-day baseline.
2. **Deterministic Constraint Alarms**: Immediate critical alerts fired on single-event occurrences of zero-tolerance failures, such as PII leakage or jailbreak bypass.
3. **Webhook Dispatch Architecture**: Forwarding structured alert payloads containing the triggering trace ID, score delta, and candidate metadata to Slack, PagerDuty, or Datadog for immediate on-call engineering triage.
""")

alert_event_df = pd.DataFrame([
    {"Trigger": "Nightly Eval Accuracy Drop", "Detection Mechanism": "Z-Score Rolling Mean", "Threshold": "Delta < -0.03", "Severity": "P1 High", "Action": "Block Release & Alert Slack #llm-ops"},
    {"Trigger": "Online Faithfulness Floor Breach", "Detection Mechanism": "Stratified Telemetry 5%", "Threshold": "Score < 0.85 (1hr)", "Severity": "P0 Critical", "Action": "Rollback to Prior Model Tag & Page On-Call"},
    {"Trigger": "Vendor Weight Update Drift", "Detection Mechanism": "Canary Suite Differential", "Threshold": "KL Divergence > 0.15", "Severity": "P2 Medium", "Action": "Open Jira Ticket for Prompt Calibration"},
    {"Trigger": "p95 Latency Spike", "Detection Mechanism": "Rolling Percentile Monitor", "Threshold": "p95 > 2800ms", "Severity": "P1 High", "Action": "Trigger VectorDB Index Cache Re-warm"}
])
st.dataframe(alert_event_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern LLMOps stacks pair LangSmith with enterprise observability platforms. LangSmith webhooks broadcast 
evaluation run summaries to AWS EventBridge or Webhook endpoints, which format rich incident cards for Slack with direct 
one-click links to the failing LangSmith trace view.
""")

st.divider()

# ==============================================================================
# SECTION 6: MODEL VALIDATION
# ==============================================================================
st.header("6. Model Validation (A/B Testing Sign-Off, Canary Evaluation and Challenger Model Certification)", divider="rainbow")

st.markdown("""
Promoting a newly fine-tuned model or updated system prompt to 100% of production traffic based purely on offline 
benchmarks carries residual risk. Continuous Evaluation closes this loop through multi-stage Model Validation: 
offline qualification, shadow evaluation, canary traffic routing, and online A/B testing sign-off.
""")

st.markdown("""
### The 4-Stage Progressive Promotion Funnel
* **Stage A: Offline Golden Suite Certification**: The candidate model must pass the 500-question golden benchmark suite with zero quality threshold violations.
* **Stage B: Shadow Traffic Evaluation**: The candidate receives 100% of live production prompts in parallel with production (responses are discarded to avoid impacting users). LangSmith traces both runs, and online evaluators compare their outputs asynchronously.
* **Stage C: Canary Evaluation (5% - 10%)**: The candidate handles real user traffic for 5% of sessions. Live user feedback (thumbs up/down, copy rates, session durations) is monitored in real-time for 24 hours.
* **Stage D: Production Promotion & Tagging**: Once canary quality meets or exceeds the baseline across all metrics, the LangSmith Prompt Hub or Model Gateway tag is updated from `staging` to `prod`.
""")

cert_stages = pd.DataFrame([
    {"Validation Stage": "1. Offline Golden Gate", "Traffic Volume": "0% (Synthetic/Curated)", "Duration": "15 Minutes", "Passing Gate": "Accuracy >= 0.88, Faithfulness >= 0.90", "Status": "PASS"},
    {"Validation Stage": "2. Shadow Ingestion", "Traffic Volume": "100% Mirrored (Async)", "Duration": "12 Hours", "Passing Gate": "Zero Crashes, Shadow Delta >= 0.00", "Status": "PASS"},
    {"Validation Stage": "3. Canary Traffic", "Traffic Volume": "5% Live Users", "Duration": "24 Hours", "Passing Gate": "User Thumbs-Down <= 2.5%, p95 <= 2000ms", "Status": "ACTIVE"},
    {"Validation Stage": "4. Full Promotion", "Traffic Volume": "100% Primary", "Duration": "Indefinite", "Passing Gate": "Unanimous Stakeholder Sign-Off", "Status": "QUEUED"}
])
st.dataframe(cert_stages, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Leading teams use LiteLLM Proxy or Portkey in front of their LLMs configured with dynamic routing rules. 
The routing rules read model tags maintained in LangSmith. When an automated continuous evaluation run passes all 4 stages, 
it triggers a GitOps PR or API call that updates the production routing pointer without restarting application servers.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Continuous Evaluation Orchestration Daemon (ContinuousEvaluationOrchestratorEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the continuous evaluation orchestrator. 
It defines benchmark evaluation execution, delta metric calculations against baselines, multi-rule quality threshold auditing, 
and automated deployment certification reporting.
""")

st.code('''
class ContinuousEvaluationOrchestratorEngine:
    """
    Enterprise Continuous Evaluation Orchestrator for LLM Applications.
    Emulates scheduled recurring benchmark evaluation, statistical quality gating,
    experiment diffing, automated failure detection, and canary model validation.
    """
    def __init__(self, project_name: str = "prod-customer-rag-continuous-eval"):
        self.project_name = project_name
        self.evaluation_history: List[Dict[str, Any]] = []

    def run_continuous_evaluation(
        self,
        candidate_version: str,
        baseline_version: str = "v2.4.0-prod",
        min_accuracy_threshold: float = 0.88,
        min_faithfulness_threshold: float = 0.90,
        max_hallucination_rate: float = 0.05,
        max_latency_p95_ms: float = 2400.0,
        inject_failure_drift: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a scheduled continuous evaluation run comparing a candidate model/prompt
        artifact against the production golden baseline across synthetic and production traffic samples.
        """
        # Baseline reference metrics
        base_acc = 0.912
        base_faith = 0.935
        base_halluc = 0.038
        base_p95 = 1850.0

        # Candidate metrics generation
        if inject_failure_drift:
            cand_acc = round(base_acc - 0.065, 3)
            cand_faith = round(base_faith - 0.082, 3)
            cand_halluc = round(base_halluc + 0.045, 3)
            cand_p95 = round(base_p95 + 720.0, 1)
        else:
            cand_acc = round(base_acc + 0.018, 3)
            cand_faith = round(base_faith + 0.012, 3)
            cand_halluc = round(max(0.015, base_halluc - 0.012), 3)
            cand_p95 = round(base_p95 - 90.0, 1)

        # Delta calculations
        acc_delta = round(cand_acc - base_acc, 3)
        faith_delta = round(cand_faith - base_faith, 3)
        halluc_delta = round(cand_halluc - base_halluc, 3)
        latency_delta = round(cand_p95 - base_p95, 1)

        # SLA Checks
        checks = [
            {
                "metric": "Accuracy SLA",
                "target": f">= {min_accuracy_threshold:.2f}",
                "candidate": f"{cand_acc:.3f}",
                "baseline": f"{base_acc:.3f}",
                "delta": f"{acc_delta:+.3f}",
                "passed": cand_acc >= min_accuracy_threshold and acc_delta >= -0.01
            },
            {
                "metric": "Faithfulness Floor",
                "target": f">= {min_faithfulness_threshold:.2f}",
                "candidate": f"{cand_faith:.3f}",
                "baseline": f"{base_faith:.3f}",
                "delta": f"{faith_delta:+.3f}",
                "passed": cand_faith >= min_faithfulness_threshold and faith_delta >= -0.015
            },
            {
                "metric": "Hallucination Bound",
                "target": f"<= {max_hallucination_rate:.2f}",
                "candidate": f"{cand_halluc:.3f}",
                "baseline": f"{base_halluc:.3f}",
                "delta": f"{halluc_delta:+.3f}",
                "passed": cand_halluc <= max_hallucination_rate and halluc_delta <= 0.01
            },
            {
                "metric": "Latency p95 Budget",
                "target": f"<= {max_latency_p95_ms:.0f}ms",
                "candidate": f"{cand_p95:.1f}ms",
                "baseline": f"{base_p95:.1f}ms",
                "delta": f"{latency_delta:+.1f}ms",
                "passed": cand_p95 <= max_latency_p95_ms and latency_delta <= 250.0
            }
        ]

        all_passed = all(c["passed"] for c in checks)
        status = "CERTIFIED" if all_passed else "REJECTED_QUALITY_GATED"

        report = {
            "evaluation_id": f"eval-sched-{candidate_version.replace('.', '_')}",
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "dataset_name": "golden-benchmark-rag-500",
            "sample_size": 500,
            "status": status,
            "overall_passed": all_passed,
            "checks": checks,
            "candidate_metrics": {
                "accuracy": cand_acc,
                "faithfulness": cand_faith,
                "hallucination_rate": cand_halluc,
                "p95_latency_ms": cand_p95
            },
            "baseline_metrics": {
                "accuracy": base_acc,
                "faithfulness": base_faith,
                "hallucination_rate": base_halluc,
                "p95_latency_ms": base_p95
            }
        }
        return report
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Automated Evaluation Matrix Sweep and Quality Threshold Simulator", divider="rainbow")

st.markdown("""
Test how continuous evaluation orchestrators process scheduled benchmark runs, verify candidate improvements against baselines, 
and block degraded releases when quality thresholds are breached.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Evaluation Parameters")
    cand_ver_input = st.selectbox(
        "Candidate Release Version",
        ["v2.5.0-rc1 (Optimized System Prompt)", "v2.5.0-rc2 (Quantized 8-bit LLM)", "v2.6.0-dev (Experimental Fine-Tune)"],
        index=0
    )
    inject_drift_toggle = st.toggle(
        "Simulate Unintended Distribution Drift / Regression",
        value=False,
        help="Simulates model weight change or bad prompt injection causing accuracy loss and latency inflation."
    )
    acc_sla_slider = st.slider(
        "Minimum Accuracy SLA Floor",
        min_value=0.80,
        max_value=0.95,
        value=0.88,
        step=0.01
    )
    faith_sla_slider = st.slider(
        "Minimum Faithfulness Floor",
        min_value=0.80,
        max_value=0.98,
        value=0.90,
        step=0.01
    )
    halluc_max_slider = st.slider(
        "Max Hallucination Rate Ceiling",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.005
    )
    latency_budget_slider = st.slider(
        "Max p95 Latency Budget (ms)",
        min_value=1500,
        max_value=3500,
        value=2400,
        step=100
    )

with wb_col2:
    st.markdown("### Live Continuous Evaluation Pipeline Execution")
    clean_version = cand_ver_input.split(" ")[0]
    eval_result = st.session_state.cont_eval_engine.run_continuous_evaluation(
        candidate_version=clean_version,
        min_accuracy_threshold=acc_sla_slider,
        min_faithfulness_threshold=faith_sla_slider,
        max_hallucination_rate=halluc_max_slider,
        max_latency_p95_ms=float(latency_budget_slider),
        inject_failure_drift=inject_drift_toggle
    )

    if eval_result["overall_passed"]:
        st.success(f"**STATUS: {eval_result['status']}** — Candidate meets all enterprise quality floors and non-regression contracts.")
    else:
        st.error(f"**STATUS: {eval_result['status']}** — Candidate breached quality thresholds. Automated release blocked.")

    # Quality Gate Scorecard
    st.markdown("#### Quality Gate Scorecard vs Production Baseline (`v2.4.0-prod`)")
    scorecard_df = pd.DataFrame(eval_result["checks"])
    st.dataframe(scorecard_df, use_container_width=True)

    # Metric Comparison Badges
    cand_m = eval_result["candidate_metrics"]
    base_m = eval_result["baseline_metrics"]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            "Accuracy",
            f"{cand_m['accuracy']:.3f}",
            delta=f"{cand_m['accuracy'] - base_m['accuracy']:+.3f}"
        )
    with m_col2:
        st.metric(
            "Faithfulness",
            f"{cand_m['faithfulness']:.3f}",
            delta=f"{cand_m['faithfulness'] - base_m['faithfulness']:+.3f}"
        )
    with m_col3:
        st.metric(
            "Hallucination Rate",
            f"{cand_m['hallucination_rate']:.3f}",
            delta=f"{cand_m['hallucination_rate'] - base_m['hallucination_rate']:+.3f}",
            delta_color="inverse"
        )
    with m_col4:
        st.metric(
            "p95 Latency",
            f"{cand_m['p95_latency_ms']:.0f} ms",
            delta=f"{cand_m['p95_latency_ms'] - base_m['p95_latency_ms']:+.0f} ms",
            delta_color="inverse"
        )

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

qa1, qa2, qa3, qa4, qa5 = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True), None, None, None, None

with qa1:
    st.markdown("""
    **Q1: What is the fundamental difference between Continuous Integration (CI) and Continuous Evaluation (CE) for LLMs?**  
    *Answer:* Continuous Integration runs fast, deterministic tests (prompt validation, mock unit tests, compact canary evaluations) during pull requests before code merges. Continuous Evaluation runs perpetually in production on scheduled intervals or telemetry streams, evaluating real-world distribution drift, vendor model updates, and comprehensive golden datasets at scale. CI guards the repository codebase; CE guards live production quality and model behavior.
    """)

    st.markdown("""
    **Q2: Why is evaluating average score improvement across a benchmark dataset insufficient to certify an LLM release?**  
    *Answer:* A candidate model can increase average accuracy by improving 100 easy questions by 0.05 points while simultaneously experiencing catastrophic regressions on 10 critical security or edge-case prompts by dropping 0.90 points. Average metrics obscure severe tail risk. Enterprise continuous evaluation requires an item-level Win/Loss matrix and hard zero-regression constraints on critical safety and business invariant subsets.
    """)

    st.markdown("""
    **Q3: How do you prevent API rate limit throttling (HTTP 429) during high-throughput continuous evaluation runs?**  
    *Answer:* Configure concurrency pooling via the `max_concurrency` parameter in `langsmith.evaluate()`, implement exponential backoff with jitter on HTTP clients, distribute evaluations across multiple API keys or Azure OpenAI regional deployments, and use local deterministic heuristics (regex, exact match, embedding cosine distance) wherever full LLM-as-a-judge calls are unnecessary.
    """)

    st.markdown("""
    **Q4: How should an enterprise continuous evaluation system handle silent model updates from third-party LLM providers?**  
    *Answer:* Third-party providers periodically update model weights behind static endpoints (e.g., `gpt-4o`). Continuous evaluation daemons run scheduled nightly runs against an immutable, frozen golden dataset. If scores drop by more than 2 standard deviations while your code and prompts have had zero commits, an automated alert flags a vendor distribution shift, triggering automatic rollback to pinned snapshot endpoints (e.g., `gpt-4o-2024-08-06`).
    """)

    st.markdown("""
    **Q5: What are the three essential components of an automated quality threshold contract?**  
    *Answer:* 1) Absolute quality floors (e.g., faithfulness >= 0.90 regardless of history); 2) Relative regression bounds (e.g., delta accuracy >= -0.010 compared to active production); and 3) Negative metric ceilings (e.g., hallucination rate <= 0.040 and p95 latency <= 2000ms).
    """)

st.markdown("### Production Traps and Anti-Patterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: The Averaging Illusion (Hiding Tail Catastrophes)**
    * **Failure**: Approving a model release because global accuracy rose from 0.89 to 0.91 without reviewing the Win/Loss matrix.
    * **Consequence**: The new model completely breaks 15 mission-critical customer account balance lookups, leading to severe production incidents despite higher average scores.
    * **Remedy**: Enforce an item-level regression gate asserting zero catastrophic drops on protected golden test subsets.
    """)

    st.error("""
    **Trap 2: Evaluating with Stochastic Temperatures in Automated Pipelines**
    * **Failure**: Running scheduled nightly benchmarks with `temperature=0.7`.
    * **Consequence**: Natural sampling variance triggers false-positive quality alerts, causing on-call fatigue and team distrust in automated gates.
    * **Remedy**: Always set `temperature=0` in automated benchmark evaluation runners to isolate actual code and model changes.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Unbounded Evaluation Costs from Large LLM Judges**
    * **Failure**: Using GPT-4o as the judge for every run on a 5,000-example dataset run every hour.
    * **Consequence**: Monthly evaluation bills exceed production application operating costs, burning thousands of dollars unnecessarily.
    * **Remedy**: Run large LLM judges on stratified sample subsets (10%) nightly, while using smaller distilled models (e.g., GPT-4o-mini, Claude 3.5 Haiku) or programmatic heuristics for high-frequency sweeps.
    """)

    st.warning("""
    **Trap 4: Missing Baseline Version Pinning**
    * **Failure**: Comparing a candidate against an undefined or moving baseline rather than an immutable production release tag.
    * **Consequence**: Inconsistent delta calculations where developers compare candidates to each other rather than the deployed customer-facing standard.
    * **Remedy**: Always explicitly designate an immutable production golden experiment ID as the baseline reference in continuous evaluation.
    """)

st.markdown("### Core Technical Flashcards")

fc_col1, fc_col2 = st.columns(2)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: What are the two types of gates in a dual-gate continuous evaluation contract?")
        st.markdown("**Back**: An Absolute Floor (minimum acceptable performance, e.g., faithfulness >= 0.90) and a Relative Non-Regression Delta (`Delta >= -0.010` vs active production).")

    with st.container(border=True):
        st.markdown("**Front**: How does Shadow Traffic Evaluation reduce deployment risk?")
        st.markdown("**Back**: It mirrors 100% of live production prompts to the candidate asynchronously, evaluating responses in LangSmith without returning candidate outputs to users.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the role of `max_concurrency` in `langsmith.evaluate()`?")
        st.markdown("**Back**: It throttles asynchronous worker threads to maximize evaluation throughput without breaching LLM provider Rate Limits (RPM / TPM).")

    with st.container(border=True):
        st.markdown("**Front**: How does Continuous Evaluation detect silent LLM provider weight updates?")
        st.markdown("**Back**: By running scheduled nightly evaluations against a frozen golden dataset; sudden score drops on zero code changes indicate vendor checkpoint drift.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Continuous Evaluation for LLM applications:")

ce_chk1 = st.checkbox("I know how to structure an automated scheduled Continuous Evaluation pipeline in LangSmith.", value=True)
ce_chk2 = st.checkbox("I understand the difference between Continuous Integration (pre-merge) and Continuous Evaluation (post-merge / scheduled).", value=True)
ce_chk3 = st.checkbox("I can implement an experiment diffing matrix that calculates item-level Win/Loss/Tie distributions.", value=True)
ce_chk4 = st.checkbox("I know how to codify dual-gate quality threshold contracts combining absolute floors and relative regression bounds.", value=True)
ce_chk5 = st.checkbox("I understand how to detect silent vendor model drift using rolling statistical anomaly detection.", value=True)
ce_chk6 = st.checkbox("I know how to execute progressive model validation across offline, shadow, canary, and production stages.", value=True)

if ce_chk1 and ce_chk2 and ce_chk3 and ce_chk4 and ce_chk5 and ce_chk6:
    st.success("Mastery Certified: You have mastered Continuous Evaluation, automated quality gating, experiment diffing, and canary validation in LangSmith.")
