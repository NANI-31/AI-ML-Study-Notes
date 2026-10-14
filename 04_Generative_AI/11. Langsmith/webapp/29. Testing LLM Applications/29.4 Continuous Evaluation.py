import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.4 Continuous Evaluation - LangSmith Master Curriculum",
    page_icon="🔄",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ContinuousEvaluationEngine:
    """
    Production Continuous Evaluation & Online Quality Monitoring Daemon.
    Emulates the full LLMOps continuous evaluation lifecycle:
    1. Automated evaluation pipelines triggered via CI/CD
    2. Scheduled cron benchmark runs (Nightly Golden Benchmark)
    3. Online production evaluation daemons (real-time sampling & scoring)
    4. Quality monitoring, drift detection, and automated alerting (Slack/PagerDuty)
    """
    def __init__(self, project_name: str = "continuous-evaluation-daemon"):
        self.project_name = project_name
        self.evaluation_events: List[Dict[str, Any]] = []
        self._event_counter = 200

    def trigger_evaluation_cycle(
        self,
        evaluation_mode: str = "scheduled_nightly",
        sample_rate_pct: int = 10,
        quality_sla_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Executes a continuous evaluation run and checks quality against SLA thresholds.
        """
        self._event_counter += 1
        event_id = f"eval_evt_{self._event_counter}"

        # Simulated telemetry stream
        if evaluation_mode == "scheduled_nightly":
            total_samples = 100
            mean_accuracy = 0.92
            mean_faithfulness = 0.89
            latency_p95 = 1420.0
            trigger_reason = "Nightly Cron (0 2 * * *)"
        elif evaluation_mode == "ci_cd_pr_merge":
            total_samples = 25
            mean_accuracy = 0.94
            mean_faithfulness = 0.95
            latency_p95 = 1150.0
            trigger_reason = "GitHub Actions PR #841 Merge"
        else: # online_production_stream
            total_samples = int(1000 * (sample_rate_pct / 100.0))
            mean_accuracy = 0.81 # Dropping due to real user drift
            mean_faithfulness = 0.78
            latency_p95 = 2100.0
            trigger_reason = f"Online Ingress Daemon ({sample_rate_pct}% Sampling)"

        composite_quality = (mean_accuracy * 0.6) + (mean_faithfulness * 0.4)
        sla_breached = composite_quality < quality_sla_threshold

        event_record = {
            "event_id": event_id,
            "mode": evaluation_mode,
            "trigger_reason": trigger_reason,
            "samples_evaluated": total_samples,
            "mean_accuracy": mean_accuracy,
            "mean_faithfulness": mean_faithfulness,
            "composite_quality": round(composite_quality, 3),
            "latency_p95_ms": latency_p95,
            "sla_threshold": quality_sla_threshold,
            "sla_breached": sla_breached,
            "alert_dispatched": "PagerDuty P2 Alert: Quality score breached SLA threshold" if sla_breached else "None (Nominal)"
        }

        self.evaluation_events.append(event_record)
        return event_record


# Instantiate engine
continuous_eval_engine = ContinuousEvaluationEngine()

# --- HEADER SECTION ---
st.title("🔄 Chapter 29.4: Continuous Evaluation & Production Monitoring")
st.markdown("""
Welcome to the production guide on **Continuous Evaluation** in **LangSmith**.
Evaluating an AI application before deployment is only half the battle.
Once a model is live in production, user behavior drifts, third-party data changes,
and upstream foundation models undergo subtle silent updates.
Continuous evaluation transforms testing from a one-off pre-release checkpoint into an ongoing,
automated engineering loop: executing scheduled benchmark suites, sampling live production traces,
running asynchronous background evaluators, and triggering automated alerts the instant quality drifts.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Continuous Evaluation & Production Quality Feedback Loop", divider="rainbow")

st.markdown("""
The architecture below illustrates the dual-loop continuous evaluation system.
The **Offline Loop** runs scheduled nightly benchmarks and CI/CD pull request gates against Golden Datasets.
The **Online Loop** samples live production traces, runs asynchronous LLM-as-a-judge evaluators in the background,
and feeds quality metrics to Datadog/Slack while recycling edge-case failures back into the dataset.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ContinuousEval {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_offline {
            label="1. Offline Continuous Evaluation Loop (CI/CD & Cron)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            cron_trigger [label="Scheduled Cron Job\nNightly at 02:00 UTC", fillcolor="#1E293B", color="#38BDF8"];
            golden_ds [label="Golden Benchmark Dataset\nVersioned SemVer (v2.1)", fillcolor="#1E293B", color="#38BDF8"];
            offline_runner [label="LangSmith evaluate()\nFull regression sweep", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_online {
            label="2. Online Production Evaluation Loop (Live Streams)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            prod_traffic [label="Live Production Traffic\n10,000 requests / day", fillcolor="#1E293B", color="#38BDF8"];
            sampler [label="Probabilistic Sampler\nSample Rate: 5-10% of traffic", fillcolor="#1E293B", color="#38BDF8"];
            async_judge [label="Async Background Evaluator\nLightweight model evaluates faithfulness & toxicity", fillcolor="#0F172A", color="#F59E0B"];
        }

        subgraph cluster_telemetry {
            label="3. Telemetry Aggregation & Drift Detection";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_dashboard [label="LangSmith Production Dashboards\nRolling 7-day Quality Trend, p95 Latency", fillcolor="#0F172A", color="#10B981"];
            alert_sentry [label="Automated Sentry / PagerDuty\nAlert if Quality < 0.85 or Drift > 10%", fillcolor="#7F1D1D", color="#EF4444"];
        }

        subgraph cluster_feedback {
            label="4. Continuous Improvement Feedback Loop";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            curation_queue [label="Annotation & Curation Queue\nLow-scoring traces converted to new benchmark examples", fillcolor="#1E293B", color="#38BDF8"];
        }

        cron_trigger -> offline_runner;
        golden_ds -> offline_runner;
        offline_runner -> ls_dashboard;

        prod_traffic -> sampler;
        sampler -> async_judge [label="Sampled Traces"];
        async_judge -> ls_dashboard [label="Live Scores"];

        ls_dashboard -> alert_sentry [label="SLA Breach"];
        async_judge -> curation_queue [label="Score < 0.60"];
        curation_queue -> golden_ds [label="Enrich Golden Set"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of Continuous Evaluation
The following matrix details the operational mechanics for each syllabus component:
""")

ce_topics_df = pd.DataFrame([
    {
        "Pillar": "1. Evaluation Pipelines",
        "Execution Cadence": "Triggered via Python SDK or CI/CD runner",
        "Operational Role": "Orchestrates test executions across datasets, evaluators, and reporting hooks",
        "LangSmith Tooling": "client.evaluate() with thread pool parallelism"
    },
    {
        "Pillar": "2. Automated Experiments",
        "Execution Cadence": "On every pull request merge or model parameter change",
        "Operational Role": "Compares candidate git branch against main branch baseline automatically",
        "LangSmith Tooling": "GitHub Actions integration + LangSmith experiment diffing"
    },
    {
        "Pillar": "3. Scheduled Evaluations",
        "Execution Cadence": "Nightly / Weekly cron schedule",
        "Operational Role": "Guarantees no temporal performance drift occurs as external APIs and databases evolve",
        "LangSmith Tooling": "Scheduled cron tasks running automated regression suites"
    },
    {
        "Pillar": "4. Production Evaluation",
        "Execution Cadence": "Continuous real-time background stream",
        "Operational Role": "Samples live user interactions and grades response quality without adding user latency",
        "LangSmith Tooling": "Online evaluators attached directly to LangSmith projects"
    },
    {
        "Pillar": "5. Quality Monitoring",
        "Execution Cadence": "Persistent monitoring & threshold alerting",
        "Operational Role": "Tracks rolling SLA metrics, cost burn, and triggers Slack/PagerDuty incident alerts",
        "LangSmith Tooling": "LangSmith Monitoring rules, webhook dispatches, and dashboard charts"
    }
])
st.dataframe(ce_topics_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: EVALUATION PIPELINES
# ==============================================================================
st.header("2. Evaluation Pipelines (LangSmith evaluate() Orchestration, Parallelism & Asynchronous Workers)", divider="rainbow")

st.markdown("""
An evaluation pipeline is an automated script that coordinates:
1. Loading an active benchmark dataset from LangSmith.
2. Invoking the application pipeline across dataset inputs using worker thread pools.
3. Running programmatic and LLM-as-a-judge evaluators across all generated outputs.
4. Aggregating scores and logging the complete experiment to LangSmith.
""")

st.code('''
from langsmith import evaluate
from langsmith.evaluation import LangChainStringEvaluator

# 1. Define custom evaluator
faithfulness_eval = LangChainStringEvaluator(
    "faithfulness",
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "context": run.outputs["context"],
        "input": example.inputs["query"]
    }
)

# 2. Run evaluation pipeline with concurrency
results = evaluate(
    target=my_agent_app,
    data="enterprise-golden-v2",
    evaluators=[faithfulness_eval],
    experiment_prefix="nightly-pipeline",
    max_concurrency=4 # Parallel async worker threads
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production LLMOps:
* `evaluate()` uses asynchronous worker pools (`max_concurrency=4` to `10`) to process 200 benchmark examples in under 60 seconds.
* The pipeline automatically retries transient network errors, ensuring test suites complete without operator intervention.
""")

st.divider()

# ==============================================================================
# SECTION 3: AUTOMATED EXPERIMENTS
# ==============================================================================
st.header("3. Automated Experiments (CI/CD Webhook Triggers, Matrix Sweeps & Branch Benchmarks)", divider="rainbow")

st.markdown("""
Automated experiments embed evaluation directly into the developer workflow.
Whenever a developer opens a pull request that modifies a prompt template, adjusts temperature,
or updates a retriever Top-K setting, the CI pipeline triggers an automated LangSmith experiment.
""")

st.markdown("### Matrix Sweeps in Automated Testing")
st.markdown("""
* **Hyperparameter Sweeps**: Testing multiple temperatures (`0.0`, `0.3`, `0.7`) across the same benchmark to find the optimal trade-off between creativity and grounding.
* **Model Sweeps**: Evaluating the exact same prompt across `gpt-4o`, `claude-3-5-sonnet`, and `gemini-1.5-pro` to determine performance-per-dollar.
* **PR Comment Bot**: CI bots post an automated markdown comment on the pull request showing the Delta scorecard and release recommendation.
""")

st.code('''
# CI/CD script running automated experiment on pull request
import os
from langsmith import Client

client = Client()
pr_number = os.environ.get("PR_NUMBER", "dev")

# Run experiment tagged with PR metadata
results = client.evaluate(
    target=app_candidate,
    data="canary-regression-v1",
    experiment_prefix=f"pr-{pr_number}-benchmark",
    metadata={"pr_number": pr_number, "branch": os.environ.get("GIT_BRANCH")}
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In GitHub Actions / GitLab CI:
* If the automated experiment detects a statistically significant regression on the canary benchmark, the merge button is locked and an alert is posted to the team's Slack channel.
""")

st.divider()

# ==============================================================================
# SECTION 4: SCHEDULED EVALUATIONS
# ==============================================================================
st.header("4. Scheduled Evaluations (Cron Schedules, Nightly Golden Benchmarks & Temporal Drift Tracking)", divider="rainbow")

st.markdown("""
Code does not have to change for an LLM application to break.
External dependencies (vector databases, third-party APIs, and cloud LLM provider weights) evolve continuously.
Scheduled evaluations run full regression suites on fixed cron schedules (e.g., nightly at 02:00 UTC)
to detect temporal degradation before users encounter issues.
""")

st.markdown("### What Scheduled Evaluations Detect")
st.markdown("""
1. **Silent Model Upgrades**: Cloud providers periodically update model weights or steerability. Scheduled benchmarks catch when a model family's output behavior changes.
2. **Database Decay**: Detecting when newly indexed documents dilute search relevance in the vector store.
3. **API Deprecations**: Catching changes in external tool APIs that cause schema parsing errors.
""")

st.code('''
# crontab definition for nightly evaluation
# 0 2 * * * /usr/local/bin/python /app/scripts/run_nightly_benchmark.py >> /var/log/eval.log 2>&1
''')

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith:
* Nightly benchmark runs are aggregated on a historical line chart.
* Teams view a 90-day trajectory of accuracy, latency, and cost, confirming that system performance is stable or improving over time.
""")

st.divider()

# ==============================================================================
# SECTION 5: PRODUCTION EVALUATION
# ==============================================================================
st.header("5. Production Evaluation (Online Evaluator Daemons, Real-Time Run Scoring & Sampling Strategies)", divider="rainbow")

st.markdown("""
Production evaluation (Online Evaluation) grades real live user requests as they happen.
Unlike pre-deployment benchmarks that use fixed test queries, online evaluation samples real customer traffic
and executes background evaluators to score live performance.
""")

st.markdown("### Online Evaluation Architecture")
st.markdown("""
* **Zero User Latency**: Online evaluators run **asynchronously** in background daemon threads or serverless workers (AWS Lambda / Google Cloud Functions). The user receives their response immediately without waiting for evaluation.
* **Probabilistic Sampling**: Evaluating 100% of production traffic with LLM judges is cost-prohibitive. Systems sample 5-10% of requests, providing statistically significant quality trends at manageable cost.
* **Automated Feedback Attachment**: Evaluator scores are logged directly to the live trace via `client.create_feedback()`.
""")

st.code('''
from langsmith import Client

client = Client()

def async_online_evaluator_daemon(run_id: str, query: str, response: str, context: str):
    """
    Background worker that runs asynchronously on sampled production runs.
    """
    # 1. Run lightweight LLM judge
    grounding_score = calculate_grounding(response, context)
    
    # 2. Attach feedback to live trace in LangSmith
    client.create_feedback(
        run_id=run_id,
        key="online_grounding",
        score=grounding_score,
        comment="Scored by async background evaluator worker"
    )
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith provides native **Online Evaluators**:
* Engineers configure online evaluator rules directly in the LangSmith project settings.
* Whenever a production run completes, LangSmith automatically queues the run for evaluation based on the configured sampling rate, without requiring custom worker infrastructure.
""")

st.divider()

# ==============================================================================
# SECTION 6: QUALITY MONITORING
# ==============================================================================
st.header("6. Quality Monitoring (SLA Thresholds, PagerDuty Alerts & Feedback Drift Analytics)", divider="rainbow")

st.markdown("""
Quality monitoring aggregates online and offline evaluation scores into real-time operational telemetry.
When quality degrades below established SLAs, monitoring systems dispatch instant alerts
to on-call engineers.
""")

st.markdown("### The Three Essential Alert Rules")
st.markdown("""
1. **Quality SLA Breach**: Alert if the 1-hour rolling average of `online_grounding` drops below 0.85.
2. **Error-Rate Spike**: Alert if unhandled exceptions exceed 2% of total production requests over a 15-minute window.
3. **Negative User Feedback Surge**: Alert if the thumbs-down ratio increases by more than 50% compared to the 7-day rolling average.
""")

st.code('''
# Webhook notification payload dispatched by LangSmith Monitor
alert_payload = {
    "alert_name": "Production Grounding SLA Breached",
    "project": "prod-customer-agent",
    "metric": "online_grounding",
    "current_value": 0.78,
    "sla_threshold": 0.85,
    "severity": "P2_WARNING",
    "dashboard_url": "https://smith.langchain.com/o/org/projects/p/prod-customer-agent"
}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise SRE operations:
* LangSmith Monitoring rules connect directly to PagerDuty and OpsGenie.
* When an alert triggers, on-call engineers can click the alert link to immediately open a filtered view of the low-scoring production traces that triggered the threshold breach.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Continuous Evaluation Orchestration Daemon (ContinuousEvaluationEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `ContinuousEvaluationEngine` models scheduled cron sweeps, CI/CD PR triggers,
and online production sampling daemons, asserting quality SLA contracts and dispatching alerts.
""")

st.code('''
class ContinuousEvaluationEngine:
    """
    Production Continuous Evaluation & Online Quality Monitoring Daemon.
    Emulates the full LLMOps continuous evaluation lifecycle:
    1. Automated evaluation pipelines triggered via CI/CD
    2. Scheduled cron benchmark runs (Nightly Golden Benchmark)
    3. Online production evaluation daemons (real-time sampling & scoring)
    4. Quality monitoring, drift detection, and automated alerting (Slack/PagerDuty)
    """
    def __init__(self, project_name: str = "continuous-evaluation-daemon"):
        self.project_name = project_name
        self.evaluation_events: List[Dict[str, Any]] = []
        self._event_counter = 200

    def trigger_evaluation_cycle(...) -> Dict[str, Any]:
        """Executes a continuous evaluation run and checks quality against SLA thresholds."""
        ...
''', language="python")

st.info("The continuous evaluation engine is instantiated in module scope and powers the interactive monitoring workbench below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Continuous Evaluation Simulator & Online Quality Daemon", divider="rainbow")

st.markdown("""
Simulate continuous evaluation cycles across Scheduled Nightly benchmarks, CI/CD PR merges,
and Online Production Traffic sampling.
Observe how quality scores update in real time and see when SLA breaches trigger automated PagerDuty alerts.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Continuous Evaluation Controls")

    eval_mode = st.selectbox(
        "Evaluation Trigger Mode:",
        options=["scheduled_nightly", "ci_cd_pr_merge", "online_production_stream"],
        format_func=lambda m: {
            "scheduled_nightly": "1. Scheduled Nightly Cron Benchmark (02:00 UTC)",
            "ci_cd_pr_merge": "2. CI/CD PR Merge Gate (GitHub Actions)",
            "online_production_stream": "3. Online Production Sampling Daemon (Live Traffic)"
        }[m]
    )

    sample_pct = st.slider(
        "Traffic Sampling Rate (%):",
        min_value=1,
        max_value=25,
        value=10,
        help="Percentage of production traffic routed to background evaluators"
    )

    sla_target = st.slider(
        "Quality SLA Target Threshold:",
        min_value=0.75,
        max_value=0.95,
        value=0.85,
        step=0.01,
        help="Minimum composite quality score before triggering on-call alerts"
    )

    run_cycle_btn = st.button("🔄 Trigger Continuous Evaluation Cycle", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Continuous Evaluation Scorecard")

    if run_cycle_btn or len(continuous_eval_engine.evaluation_events) == 0:
        evt_res = continuous_eval_engine.trigger_evaluation_cycle(
            evaluation_mode=eval_mode,
            sample_rate_pct=sample_pct,
            quality_sla_threshold=sla_target
        )
    else:
        evt_res = continuous_eval_engine.evaluation_events[-1]

    # Metrics Overview
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Composite Quality", f"{evt_res['composite_quality'] * 100:.1f} %")
    m2.metric("Samples Graded", evt_res["samples_evaluated"])
    m3.metric("p95 Latency", f"{evt_res['latency_p95_ms']:.0f} ms")
    sla_status = "BREACHED (ALERT)" if evt_res["sla_breached"] else "NOMINAL (OK)"
    m4.metric("SLA Status", sla_status)

    if evt_res["sla_breached"]:
        st.error(f"🚨 **Quality SLA Breached**: {evt_res['alert_dispatched']}")
    else:
        st.success("✅ **Quality Nominal**: Evaluation metrics satisfy all production SLA contracts.")

    # Details Breakdown
    with st.container(border=True):
        st.markdown(f"**Event ID**: `{evt_res['event_id']}` | **Trigger**: {evt_res['trigger_reason']}")
        st.markdown(f"*Semantic Accuracy*: **{evt_res['mean_accuracy'] * 100:.1f}%**")
        st.markdown(f"*Contextual Faithfulness*: **{evt_res['mean_faithfulness'] * 100:.1f}%**")
        st.markdown(f"*SLA Threshold*: **{evt_res['sla_threshold'] * 100:.1f}%**")

    # History Table
    st.markdown("#### Historical Continuous Evaluation Stream")
    hist_df = pd.DataFrame(continuous_eval_engine.evaluation_events)
    st.dataframe(hist_df[["event_id", "mode", "samples_evaluated", "composite_quality", "latency_p95_ms", "sla_breached"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the architectural difference between Pre-Deployment Benchmarking and Continuous Evaluation?"):
    st.markdown("""
    **Answer:**
    * **Pre-Deployment Benchmarking**: A static gate run during CI/CD against a fixed Golden Dataset before code is promoted. It confirms that the new release satisfies initial quality bars.
    * **Continuous Evaluation**: An ongoing operational loop that monitors live production traffic, samples real user interactions, runs background evaluators, and executes scheduled benchmarks. It detects emergent model drift, data drift, and database decay that occur over days and weeks in production.
    """)

with st.expander("Q2: How do you implement Online Evaluation without adding latency to user requests?"):
    st.markdown("""
    **Answer:**
    Online evaluation is decoupled from the user request path:
    1. The application processes the user query and streams the response immediately.
    2. The trace is logged to LangSmith asynchronously via background thread workers.
    3. LangSmith's online evaluator daemon or a background serverless function picks up sampled traces from the stream.
    4. Evaluator scores are calculated asynchronously and attached back to the trace via `client.create_feedback()`.
    The end user experiences zero added latency.
    """)

with st.expander("Q3: Why is 5-10% probabilistic sampling sufficient for online production monitoring?"):
    st.markdown("""
    **Answer:**
    Evaluating 100% of production traffic using LLM judges would double or triple overall API spending.
    Probabilistic sampling (5-10%) provides:
    * **Statistical Validity**: For an application handling 10,000 requests/day, a 10% sample yields 1,000 daily evaluations—more than enough statistical power to detect quality drift with high confidence intervals.
    * **Cost Control**: Caps evaluation expenditures at a predictable, fractional budget.
    """)

with st.expander("Q4: What causes 'Model Drift' in an application where the codebase has not been modified?"):
    st.markdown("""
    **Answer:**
    1. **Upstream Foundation Model Updates**: Cloud providers (OpenAI, Anthropic) periodically deploy subtle weight updates, system prompt adjustments, or safety fine-tuning that alter output formatting or reasoning styles.
    2. **Vector Database Content Shifts**: As new documents are ingested into production vector stores, retrieval candidate distributions change, occasionally diluting top-ranked context.
    3. **User Population Drift**: As the product scales, user personas diversify, introducing new vocabularies and intent patterns not represented in the initial training data.
    """)

with st.expander("Q5: How does the Continuous Evaluation loop enrich the Golden Benchmark dataset over time?"):
    st.markdown("""
    **Answer:**
    When online evaluators detect a production trace with a low quality score (< 0.60) or negative user feedback:
    1. The trace is automatically routed into a LangSmith Annotation Queue.
    2. A domain SME reviews the failure, identifies the root cause, and writes the corrected ground-truth answer.
    3. The verified example is appended to the Golden Dataset.
    4. This permanently immunizes future models against that specific failure mode.
    """)

st.markdown("### Production Traps & Failure Modes")

ce_tr_col1, ce_tr_col2 = st.columns(2)

with ce_tr_col1:
    st.error("""
    **Trap 1: Evaluating 100% of Production Traffic with Frontier LLM Judges**
    * **Failure**: Triggering an asynchronous GPT-4o evaluation on every single live user request.
    * **Consequence**: Evaluation costs exceed the operational cost of serving users, bankrupting the GenAI budget.
    * **Remedy**: Implement probabilistic sampling (5-10%) and use fast, cost-effective models (e.g., `gpt-4o-mini`) for routine online evaluation.
    """)

    st.error("""
    **Trap 2: Synchronous Evaluation in the User Request Path**
    * **Failure**: Running the evaluation judge inside the API request handler before returning the response to the user.
    * **Consequence**: User response latency doubles from 1.5s to 3.5s, severely degrading the customer experience.
    * **Remedy**: Always evaluate asynchronously in background worker daemons or LangSmith online evaluator queues.
    """)

with ce_tr_col2:
    st.warning("""
    **Trap 3: Alert Fatigue from Overly Sensitive Thresholds**
    * **Failure**: Setting PagerDuty alerts to trigger whenever a single individual production trace receives a low score.
    * **Consequence**: On-call engineers receive dozens of false alarms daily from unusual or out-of-scope user inputs, leading to alert fatigue.
    * **Remedy**: Base alerts on rolling statistical aggregations (e.g., 1-hour rolling average < 0.85 across 50+ samples).
    """)

    st.warning("""
    **Trap 4: Neglecting the Dataset Recycling Feedback Loop**
    * **Failure**: Monitoring production quality scores without feeding failed traces back into the Golden Benchmark.
    * **Consequence**: The team observes that quality is dropping, but lacks the structured test cases needed to fix the regression in CI/CD.
    * **Remedy**: Build an automated bridge routing low-scoring traces directly into the LangSmith Annotation Queue.
    """)

st.markdown("### Core Technical Flashcards")

fcce_c1, fcce_c2 = st.columns(2)

with fcce_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary operational goal of Continuous Evaluation?")
        st.markdown("**Back**: To detect real-time quality drift, model behavior shifts, and data decay in live production traffic before users experience systemic outages.")

    with fcce_c2:
        st.markdown("**Front**: How is evaluation feedback recorded on live traces in LangSmith?")
        st.markdown("**Back**: Using `client.create_feedback(run_id=..., key=..., score=...)`, attaching structured quality metrics and commentary directly to the production run record.")

with fcce_c1:
    with st.container(border=True):
        st.markdown("**Front**: Why should online evaluation be executed asynchronously?")
        st.markdown("**Back**: To ensure zero latency impact on live users; the user receives their response immediately while background daemons score the interaction.")

    with fcce_c2:
        st.markdown("**Front**: What is the relationship between Continuous Evaluation and Golden Datasets?")
        st.markdown("**Back**: Low-scoring production traces identified by continuous evaluation are curated by SMEs and recycled into Golden Datasets, creating a self-improving quality loop.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of continuous evaluation and quality monitoring:")

ce_chk1 = st.checkbox("I understand the dual-loop architecture of offline regression sweeps and online stream monitoring.", value=True)
ce_chk2 = st.checkbox("I can configure automated LangSmith evaluation pipelines with asynchronous worker concurrency.", value=True)
ce_chk3 = st.checkbox("I know how to schedule nightly cron regression benchmarks to detect silent model and data drift.", value=True)
ce_chk4 = st.checkbox("I can implement asynchronous online evaluation daemons using probabilistic sampling (5-10%).", value=True)
ce_chk5 = st.checkbox("I know how to set up rolling quality SLA monitoring rules and dispatch automated PagerDuty alerts.", value=True)
ce_chk6 = st.checkbox("I understand how to close the feedback loop by recycling production failure traces into Golden Datasets.", value=True)

if ce_chk1 and ce_chk2 and ce_chk3 and ce_chk4 and ce_chk5 and ce_chk6:
    st.success("Mastery Certified: You have mastered Continuous Evaluation, online quality monitoring, and automated drift alerting in LangSmith.")
