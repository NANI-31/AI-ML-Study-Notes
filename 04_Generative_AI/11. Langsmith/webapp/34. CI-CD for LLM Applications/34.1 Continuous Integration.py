import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="34.1 Continuous Integration - LangSmith Master Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LLMContinuousIntegrationEngine:
    """
    Production Continuous Integration (CI) Engine for LLM Applications.
    Emulates the 5 sequential stages of a pull-request CI pipeline in LangSmith:
    1. Prompt Tests (Variable binding, double-escaping, length bounds)
    2. Unit Tests (Mocked LLM chains, Pydantic parser schemas, tool errors)
    3. Dataset Tests (Schema conformance, duplicate checks, null ground-truths)
    4. Evaluation Tests (Canary benchmark sweeps, metric thresholding)
    5. Regression Tests (Delta comparison against production baseline)
    """
    def __init__(self, project_name: str = "llm-continuous-integration"):
        self.project_name = project_name
        self.pipeline_runs: List[Dict[str, Any]] = []

    def execute_ci_pipeline(
        self,
        pr_branch: str,
        commit_sha: str = "c7a81e9",
        inject_regression: bool = False
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end CI pipeline pass across all five testing stages.
        """
        stages = []
        overall_passed = True
        blocking_stage = None

        # Stage 1: Prompt Tests
        stage1_pass = True
        stages.append({
            "stage_num": 1,
            "stage_name": "Prompt Tests",
            "tests_run": 14,
            "tests_passed": 14,
            "status": "PASSED",
            "latency_ms": 12.0,
            "details": "Verified 14 prompt templates; zero unbound variables; double-escaping valid."
        })

        # Stage 2: Unit Tests
        stage2_pass = True
        stages.append({
            "stage_num": 2,
            "stage_name": "Component Unit Tests",
            "tests_run": 32,
            "tests_passed": 32,
            "status": "PASSED",
            "latency_ms": 85.0,
            "details": "Mocks executed; Pydantic parsers validated; tool schema contracts 100% compliant."
        })

        # Stage 3: Dataset Tests
        stage3_pass = True
        stages.append({
            "stage_num": 3,
            "stage_name": "Dataset Schema Tests",
            "tests_run": 8,
            "tests_passed": 8,
            "status": "PASSED",
            "latency_ms": 34.0,
            "details": "Golden benchmark verified; zero null ground-truths; cosine duplicate check clean."
        })

        # Stage 4: Evaluation Tests
        stage4_pass = not inject_regression
        if stage4_pass:
            stages.append({
                "stage_num": 4,
                "stage_name": "Evaluation Tests (Canary Suite)",
                "tests_run": 25,
                "tests_passed": 24,
                "status": "PASSED",
                "latency_ms": 1450.0,
                "details": "Canary score 96.0% (threshold >= 90.0%); latency p95 within 1800ms SLA."
            })
        else:
            stages.append({
                "stage_num": 4,
                "stage_name": "Evaluation Tests (Canary Suite)",
                "tests_run": 25,
                "tests_passed": 18,
                "status": "FAILED",
                "latency_ms": 2850.0,
                "details": "Canary score 72.0% fell below 90.0% threshold! Latency p95 breached SLA."
            })
            overall_passed = False
            blocking_stage = "Stage 4: Evaluation Tests"

        # Stage 5: Regression Tests
        if overall_passed:
            stages.append({
                "stage_num": 5,
                "stage_name": "Regression Delta Gate",
                "tests_run": 25,
                "tests_passed": 25,
                "status": "PASSED",
                "latency_ms": 420.0,
                "details": "Delta accuracy +2.4% vs baseline; zero critical task regressions."
            })
        else:
            stages.append({
                "stage_num": 5,
                "stage_name": "Regression Delta Gate",
                "tests_run": 0,
                "tests_passed": 0,
                "status": "SKIPPED",
                "latency_ms": 0.0,
                "details": "Skipped due to upstream failure in Stage 4."
            })

        total_latency = sum(s["latency_ms"] for s in stages)

        pipeline_record = {
            "branch": pr_branch,
            "commit_sha": commit_sha,
            "overall_status": "APPROVED (GREEN)" if overall_passed else "BLOCKED (RED)",
            "passed": overall_passed,
            "blocking_stage": blocking_stage,
            "total_latency_ms": round(total_latency, 1),
            "stages": stages
        }

        self.pipeline_runs.append(pipeline_record)
        return pipeline_record


# Instantiate engine
ci_engine = LLMContinuousIntegrationEngine()

# --- HEADER SECTION ---
st.title("🏗️ Chapter 34.1: Continuous Integration for LLM Applications")
st.markdown("""
Welcome to the production guide on **Continuous Integration (CI)** for GenAI applications managed through **LangSmith**.
In modern software engineering, CI guarantees that every pull request satisfies rigorous quality standards before merge.
For LLM applications, code integration is fundamentally multi-layered:
pull requests modify not only Python logic, but also prompt templates, vector search parameters,
retriever schemas, and foundation model versions.
This module covers how to construct an automated five-stage CI pipeline in GitHub Actions and LangSmith
to catch prompt formatting bugs, unit regressions, dataset corruption, and accuracy drops before merging code.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Continuous Integration Pipeline for GenAI Applications", divider="rainbow")

st.markdown("""
The architecture below outlines the automated GitHub Actions CI pipeline integrated with LangSmith.
Every pull request triggers a five-stage verification pipeline: fast static prompt tests, mocked unit tests,
dataset schema validations, a lightweight canary evaluation benchmark, and differential regression gating.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ContinuousIntegration {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_git_trigger {
            label="1. Ingress Trigger (GitHub PR Event)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            git_pr [label="Pull Request Opened / Synchronized\nBranch: 'feature/prompt-optimization'\nCommit: c7a81e9", fillcolor="#1E293B", color="#38BDF8"];
            gha_runner [label="GitHub Actions Runner\nTriggers CI Workflow: .github/workflows/llm-ci.yml", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_ci_stages {
            label="2. Five Sequential CI Testing Stages";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            s1_prompt [label="Stage 1: Prompt Tests (Local, < 1s)\nValidates variable bindings, double escaping {{..}}", fillcolor="#1E293B", color="#38BDF8"];
            s2_unit [label="Stage 2: Unit Tests (Local Mocks, < 3s)\nFakeListChatModel, Pydantic parser tests", fillcolor="#1E293B", color="#38BDF8"];
            s3_dataset [label="Stage 3: Dataset Tests (< 2s)\nSchema validation, null checks, deduplication", fillcolor="#1E293B", color="#38BDF8"];
            s4_eval [label="Stage 4: Evaluation Tests (< 60s)\nCanary benchmark sweep (25 examples in LangSmith)", fillcolor="#0F172A", color="#F59E0B"];
            s5_regr [label="Stage 5: Regression Delta Gate (< 10s)\nAssert Delta Acc >= 0.0 & Latency within SLA", fillcolor="#0F172A", color="#A855F7"];
        }

        subgraph cluster_verdict {
            label="3. CI Gating Decision & PR Feedback";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            pr_comment [label="LangSmith PR Bot Comment\nPosts Delta Scorecard & Latency Profile to PR", fillcolor="#064E3B", color="#10B981"];
            merge_lock [label="Merge Protection Status\nGreen: Merge Allowed | Red: Merge Blocked", fillcolor="#064E3B", color="#10B981"];
        }

        git_pr -> gha_runner;
        gha_runner -> s1_prompt;
        s1_prompt -> s2_unit [label="Prompts Valid"];
        s2_unit -> s3_dataset [label="Units Green"];
        s3_dataset -> s4_eval [label="Datasets Sound"];
        s4_eval -> s5_regr [label="Canary Pass"];
        s5_regr -> pr_comment [label="Zero Regression"];
        pr_comment -> merge_lock;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Stages of LLM Continuous Integration
The following matrix summarizes the scope, execution runtime, and gating criteria for each stage:
""")

ci_stages_df = pd.DataFrame([
    {
        "CI Stage": "1. Prompt Tests",
        "Target Surface": "Prompt templates, system message formatting, variables",
        "Execution Environment": "Local pytest runner (zero LLM calls, < 1s)",
        "Failure Signal": "KeyError on missing variable, broken JSON double-escaping"
    },
    {
        "CI Stage": "2. Unit Tests",
        "Target Surface": "Custom runnables, output parsers, tool logic, retrievers",
        "Execution Environment": "pytest with FakeListChatModel & mocked databases (< 3s)",
        "Failure Signal": "Pydantic validation failure, tool exception crash, broken pipe"
    },
    {
        "CI Stage": "3. Dataset Tests",
        "Target Surface": "Golden benchmark datasets, test splits, metadata schemas",
        "Execution Environment": "Automated dataset validator script (< 2s)",
        "Failure Signal": "Missing ground-truth outputs, duplicate inputs, invalid tags"
    },
    {
        "CI Stage": "4. Evaluation Tests",
        "Target Surface": "End-to-end canary benchmark sweep on candidate branch",
        "Execution Environment": "LangSmith evaluate() against canary dataset (< 60s)",
        "Failure Signal": "Canary pass rate < 90.0%, p95 latency > 2,000ms"
    },
    {
        "CI Stage": "5. Regression Tests",
        "Target Surface": "Differential comparison between candidate PR and main branch",
        "Execution Environment": "LangSmith experiment diff comparison (< 10s)",
        "Failure Signal": "Negative accuracy delta (< 0.0), regression on critical slices"
    }
])
st.dataframe(ci_stages_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: PROMPT TESTS
# ==============================================================================
st.header("2. Prompt Tests (Variable Assertion, Syntax Validation & Formatting Dry-Runs)", divider="rainbow")

st.markdown("""
Prompt tests are the fastest, most foundational layer of GenAI CI.
A broken prompt variable breaks the entire application at runtime with an immediate `KeyError`.
Prompt tests run statically in pytest without invoking external APIs, verifying syntax and variables in milliseconds.
""")

st.markdown("### Core Assertions in Prompt Testing")
st.markdown("""
* **Variable Binding Invariants**: Ensuring every variable referenced in `{var}` matches the expected input dictionary keys.
* **Partial Binding Validation**: Verifying that static defaults bound via `.partial()` are preserved.
* **Escaped JSON Braces**: Ensuring system prompts containing JSON schema examples correctly use double curly braces (`{{` and `}}`).
* **Character Budget Constraints**: Ensuring formatted prompts do not exceed pre-configured length limits under maximum expected user inputs.
""")

st.code('''
import pytest
from langchain_core.prompts import ChatPromptTemplate

def test_system_prompt_variable_integrity():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for {organization}. Output JSON format: {{{{id: int}}}}"),
        ("human", "{user_query}")
    ])
    
    # Assert expected input variables
    assert set(prompt.input_variables) == {"organization", "user_query"}
    
    # Dry-run format execution
    formatted = prompt.format_messages(organization="Acme", user_query="Hello")
    assert len(formatted) == 2
    assert "{id: int}" in formatted[0].content # Verifies proper double-escaping
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production CI pipelines:
* Every prompt file in the repository is audited by a test loop that verifies no prompt can be merged with unbound variables.
* Tests run in under 50 milliseconds during local pre-commit checks.
""")

st.divider()

# ==============================================================================
# SECTION 3: UNIT TESTS
# ==============================================================================
st.header("3. Unit Tests (Isolated Mocking, Parser Robustness & Tool Schema Contracts)", divider="rainbow")

st.markdown("""
Unit tests verify the isolated execution logic of application components:
output parsers, custom runnable lambdas, and tool definitions.
Unit tests rely on **100% mocked external dependencies**: using `FakeListChatModel` instead of live LLMs
and mocked data fixtures instead of live databases.
""")

st.markdown("### Essential Unit Testing Fixtures")
st.markdown("""
1. **Parser Edge Cases**: Testing `JsonOutputParser` against markdown fences (` ```json `), conversational preambles, and malformed closing brackets.
2. **Tool Argument Schemas**: Asserting that calling tools with missing or mistyped arguments raises `pydantic.ValidationError`.
3. **Runnable Data Transformations**: Testing custom filter functions, query normalization, and context trimming.
""")

st.code('''
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class AnalysisOutput(BaseModel):
    summary: str = Field(...)
    score: float = Field(...)

def test_full_pipeline_with_mocked_llm():
    parser = JsonOutputParser(pydantic_object=AnalysisOutput)
    # Mock LLM returns deterministic JSON string
    fake_llm = FakeListChatModel(responses=['{"summary": "Test OK", "score": 0.95}'])
    
    chain = fake_llm | parser
    result = chain.invoke("Analyze quarterly results")
    
    assert result["summary"] == "Test OK"
    assert result["score"] == 0.95
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern developer workflows:
* Unit test suites containing 50+ component tests execute in under 3 seconds in CI.
* Tests cost zero dollars, preventing developers from incurring cloud bills during routine development.
""")

st.divider()

# ==============================================================================
# SECTION 4: DATASET TESTS
# ==============================================================================
st.header("4. Dataset Tests (Schema Integrity, Duplicate Pruning & Missing Label Auditing)", divider="rainbow")

st.markdown("""
In evaluation-driven development, benchmark datasets are code.
If a developer updates an evaluation dataset in Git or LangSmith, CI must verify that the dataset
itself is sound before using it to grade candidate models.
""")

st.markdown("### Dataset Health Invariants")
st.markdown("""
* **Non-Null Ground Truths**: Ensuring no test case contains empty or `None` expected outputs.
* **Input Schema Completeness**: Confirming all required keys (e.g., `query`, `chat_history`) are present in every example.
* **Exact Duplicate Rejection**: Ensuring no identical queries were accidentally duplicated in the dataset.
* **Metadata Category Stratification**: Verifying that mandatory test slices (e.g., safety, boundary, basic) are present.
""")

st.code('''
def test_golden_dataset_integrity(dataset_examples: list[dict]):
    seen_queries = set()
    
    for ex in dataset_examples:
        # 1. Assert input query exists
        query = ex["inputs"].get("query", "").strip()
        assert len(query) > 0, "Empty query detected in dataset!"
        
        # 2. Assert no duplicate queries
        assert query.lower() not in seen_queries, f"Duplicate query detected: {query}"
        seen_queries.add(query.lower())
        
        # 3. Assert ground truth exists
        ground_truth = ex["outputs"].get("ground_truth", "")
        assert len(ground_truth) > 0, f"Missing ground-truth for query: {query}"
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise LLMOps:
* Changes to evaluation datasets are managed via Pull Requests in Git or version-tagged in LangSmith.
* Dataset CI tests run automatically whenever an example is added or edited.
""")

st.divider()

# ==============================================================================
# SECTION 5: EVALUATION TESTS
# ==============================================================================
st.header("5. Evaluation Tests (Canary Experiment Runs, LLM-as-a-Judge Calibration & Score Thresholds)", divider="rainbow")

st.markdown("""
Evaluation tests bridge static code verification and real model execution.
In Stage 4 of CI, the candidate application is invoked against a **Canary Benchmark Dataset**
(typically 25-50 high-leverage examples) running against fast models (e.g., `gpt-4o-mini`).
""")

st.markdown("### Canary Benchmark Execution")
st.markdown("""
* **Automated LangSmith Experiment**: The CI runner calls `langsmith.evaluate()`, creating a distinct experiment run tagged with the Git PR number.
* **Multi-Metric Scoring**: Evaluators score accuracy, grounding, and formatting.
* **Hard Pass-Rate Gate**: The pipeline asserts that the canary pass rate meets or exceeds the minimum acceptance threshold (e.g., pass rate >= 90.0%).
""")

st.code('''
from langsmith import Client

client = Client()

def run_canary_evaluation_ci_gate(app_candidate, pr_number: str):
    # Run evaluation against canary dataset
    results = client.evaluate(
        app_candidate,
        data="canary-benchmark-v1",
        evaluators=[accuracy_evaluator, grounding_evaluator],
        experiment_prefix=f"pr-{pr_number}-canary"
    )
    
    # Assert minimum acceptable score
    summary = results.to_pandas()
    pass_rate = summary["feedback.accuracy"].mean()
    assert pass_rate >= 0.90, f"Canary evaluation failed! Pass rate {pass_rate*100:.1f}% below 90% threshold."
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In GitHub Actions:
* The canary evaluation runs in 45-60 seconds using parallel worker threads.
* If the canary pass rate breaches the 90% threshold, the CI build fails immediately, preventing broken code from reaching staging.
""")

st.divider()

# ==============================================================================
# SECTION 6: REGRESSION TESTS
# ==============================================================================
st.header("6. Regression Tests (Differential Delta Gates, Loss-Rate Ceilings & Pre-Merge Blocking)", divider="rainbow")

st.markdown("""
The final stage of CI is the **Regression Delta Gate**.
Even if a model passes the canary score threshold (e.g., achieving 92%), it must not regress
on capabilities that the production baseline already mastered.
""")

st.markdown("### Differential Gating Rules")
st.markdown("""
1. **Delta Accuracy Contract**: `Delta = Candidate_Accuracy - Baseline_Accuracy >= 0.0`. The candidate must match or improve upon the baseline.
2. **Loss-Rate Ceiling**: The percentage of benchmark examples where the candidate performed worse than the baseline must not exceed 5%.
3. **Zero-Tolerance Safety Gate**: Zero regressions permitted on safety, security, or compliance probes.
""")

st.code('''
def assert_regression_delta_contract(candidate_metrics: dict, baseline_metrics: dict):
    delta_accuracy = candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
    delta_latency = candidate_metrics["latency_ms"] - baseline_metrics["latency_ms"]
    
    # Contract 1: No accuracy regression
    assert delta_accuracy >= -0.01, f"Accuracy regressed by {delta_accuracy*100:.1f}%! Merge blocked."
    
    # Contract 2: No latency explosion
    assert delta_latency <= 250.0, f"Latency regressed by +{delta_latency:.1f}ms! Merge blocked."
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise CI bots:
* A bot automatically posts a differential scorecard comment on the GitHub pull request showing the Delta comparison.
* The branch protection rule locks the merge button until the regression gate turns green.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: CI Pipeline Execution & Gating Engine (LLMContinuousIntegrationEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `LLMContinuousIntegrationEngine` models a complete five-stage pull-request CI pipeline,
executing prompt tests, unit tests, dataset audits, canary sweeps, and regression gates.
""")

st.code('''
class LLMContinuousIntegrationEngine:
    """
    Production Continuous Integration (CI) Engine for LLM Applications.
    Emulates the 5 sequential stages of a pull-request CI pipeline in LangSmith:
    1. Prompt Tests (Variable binding, double-escaping, length bounds)
    2. Unit Tests (Mocked LLM chains, Pydantic parser schemas, tool errors)
    3. Dataset Tests (Schema conformance, duplicate checks, null ground-truths)
    4. Evaluation Tests (Canary benchmark sweeps, metric thresholding)
    5. Regression Tests (Delta comparison against production baseline)
    """
    def __init__(self, project_name: str = "llm-continuous-integration"):
        self.project_name = project_name
        self.pipeline_runs: List[Dict[str, Any]] = []

    def execute_ci_pipeline(
        self,
        pr_branch: str,
        commit_sha: str = "c7a81e9",
        inject_regression: bool = False
    ) -> Dict[str, Any]:
        """Executes an end-to-end CI pipeline pass across all five testing stages."""
        ...
''', language="python")

st.info("The CI engine is instantiated in module scope and powers the interactive pipeline simulator below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Pre-Merge CI Test Runner & Gating Simulator", divider="rainbow")

st.markdown("""
Simulate executing a five-stage GitHub Actions pull request CI pipeline.
Toggle between healthy code changes and regressed pull requests, observe stage-by-stage execution,
and see how automated regression gates protect production from broken merges.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("CI Pipeline Execution Controls")

    pr_branch_name = st.text_input("Pull Request Branch Name:", value="feature/optimized-system-prompt")
    commit_hash = st.text_input("Commit SHA:", value="c7a81e9")

    simulate_fault = st.toggle(
        "Inject Model Accuracy & Latency Regression",
        value=False,
        help="Simulates a prompt change that drops canary accuracy below 90% and exceeds latency SLA"
    )

    run_ci_btn = st.button("🏗️ Trigger Pull Request CI Pipeline", use_container_width=True)

with wb_col2:
    st.subheader("GitHub Actions CI Pipeline Report")

    if run_ci_btn or len(ci_engine.pipeline_runs) == 0:
        ci_res = ci_engine.execute_ci_pipeline(
            pr_branch=pr_branch_name,
            commit_sha=commit_hash,
            inject_regression=simulate_fault
        )
    else:
        ci_res = ci_engine.pipeline_runs[-1]

    # Metrics Overview
    m1, m2, m3 = st.columns(3)
    status_label = "MERGE APPROVED (GREEN)" if ci_res["passed"] else "MERGE BLOCKED (RED)"
    m1.metric("CI Pipeline Status", status_label)
    m2.metric("Total CI Latency", f"{ci_res['total_latency_ms']:.1f} ms")
    m3.metric("Commit SHA", ci_res["commit_sha"])

    if ci_res["passed"]:
        st.success("🎉 **All CI Stages Passed**: PR satisfies prompt, unit, dataset, canary, and regression contracts.")
    else:
        st.error(f"🚨 **CI Pipeline Blocked at {ci_res['blocking_stage']}**: Pull request cannot be merged.")

    # Stage-by-Stage Table
    st.markdown("#### Sequential CI Stage Execution Waterfall")
    stages_df = pd.DataFrame(ci_res["stages"])
    st.dataframe(stages_df[["stage_num", "stage_name", "status", "tests_run", "latency_ms", "details"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does an LLM Continuous Integration pipeline differ from traditional software CI?"):
    st.markdown("""
    **Answer:**
    Traditional CI validates deterministic code syntax, unit tests, and build artifacts.
    LLM CI adds three non-deterministic, AI-specific validation layers:
    1. **Prompt Template Tests**: Validating variable bindings and character budgets.
    2. **Dataset Tests**: Enforcing schema invariants and preventing benchmark contamination.
    3. **Evaluation & Regression Gates**: Running canary model sweeps and asserting that the candidate release matches or improves upon baseline accuracy, grounding, and latency metrics.
    """)

with st.expander("Q2: Why should Stage 1 and Stage 2 of LLM CI run without calling live foundation models?"):
    st.markdown("""
    **Answer:**
    1. **Speed**: Static prompt tests and mocked unit tests execute in milliseconds, providing rapid local feedback (< 3s total runtime).
    2. **Cost**: Running live frontier models on every Git commit burns thousands of API dollars per day across active engineering teams.
    3. **Reliability**: Eliminates external network flakiness and cloud rate limits from blocking developer pull requests.
    Live models are reserved for Stage 4 canary sweeps.
    """)

with st.expander("Q3: What is a Canary Benchmark Dataset in CI pipelines?"):
    st.markdown("""
    **Answer:**
    A Canary Benchmark is a compact, high-leverage subset of the full evaluation suite (typically 25-50 examples):
    * Specially curated to include prompt-sensitive edge cases, negative constraint tests, and critical core tasks.
    * Designed to execute in 45-60 seconds in CI, providing high statistical confidence without stalling developer velocity.
    """)

with st.expander("Q4: What is a Regression Delta Contract, and why is it essential for pre-merge gating?"):
    st.markdown("""
    **Answer:**
    A Regression Delta Contract asserts that `Candidate_Score - Baseline_Score >= 0.0`.
    Even if a candidate achieves an impressive absolute score (e.g., 93%), if it breaks critical features that the baseline handled correctly, it fails the delta contract.
    This guarantees that the production system monotonically improves or maintains quality across releases.
    """)

with st.expander("Q5: How do CI bots integrate LangSmith evaluation results into GitHub pull requests?"):
    st.markdown("""
    **Answer:**
    When `langsmith.evaluate()` finishes in the GitHub Actions runner:
    1. The runner extracts summary metrics and differential deltas.
    2. A GitHub API call posts an automated markdown comment on the pull request containing a summary scorecard, delta table, and direct link to the LangSmith experiment trace.
    3. If the delta contract fails, the bot sets the GitHub commit status to `failure`, blocking the merge button.
    """)

st.markdown("### Production Traps & Failure Modes")

ci_tr_col1, ci_tr_col2 = st.columns(2)

with ci_tr_col1:
    st.error("""
    **Trap 1: Running Full 2,000-Example Benchmarks on Every Pull Request**
    * **Failure**: Triggering the entire production benchmark suite on every Git commit.
    * **Consequence**: CI runs take 45+ minutes and burn hundreds of dollars per PR, stalling developer velocity and leading teams to disable CI checks.
    * **Remedy**: Use a compact 25-50 example Canary Suite for PR CI; run the full 2,000-example suite nightly.
    """)

    st.error("""
    **Trap 2: Ignoring Variable Binding Unit Tests**
    * **Failure**: Merging prompt changes without automated prompt variable unit tests.
    * **Consequence**: A typo in a template variable (`{user_qury}`) slips into production, causing runtime `KeyError` crashes on 100% of user requests.
    * **Remedy**: Make Stage 1 prompt unit tests mandatory; assert that all template variables match expected inputs.
    """)

with ci_tr_col2:
    st.warning("""
    **Trap 3: Flaky Assertions from Non-Zero Model Temperatures**
    * **Failure**: Running canary evaluation tests with `temperature=0.7`.
    * **Consequence**: Stochastic completions cause PR tests to randomly pass or fail on identical code commits, destroying team trust in CI.
    * **Remedy**: Always enforce `temperature=0` in automated CI evaluation pipelines.
    """)

    st.warning("""
    **Trap 4: Missing Dataset Schema Tests**
    * **Failure**: Modifying benchmark datasets in Git without automated schema validation.
    * **Consequence**: A pull request adds examples with missing ground-truth outputs, crashing automated evaluators in subsequent CI stages.
    * **Remedy**: Enforce Stage 3 dataset tests that validate non-null outputs and deduplicate inputs before running evaluations.
    """)

st.markdown("### Core Technical Flashcards")

fcci_c1, fcci_c2 = st.columns(2)

with fcci_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 5 sequential stages of an LLM CI pipeline?")
        st.markdown("**Back**: 1. Prompt Tests, 2. Unit Tests, 3. Dataset Tests, 4. Evaluation Tests (Canary), and 5. Regression Tests (Delta Gate).")

    with fcci_c2:
        st.markdown("**Front**: What is the runtime SLA for Stages 1, 2, and 3 in local CI?")
        st.markdown("**Back**: Under 5 seconds combined, because all prompts, parsers, and datasets are evaluated locally with zero external network or model calls.")

with fcci_c1:
    with st.container(border=True):
        st.markdown("**Front**: What does a Regression Delta Gate enforce before code merge?")
        st.markdown("**Back**: It asserts that the proposed release performs equal to or better than the active production baseline (`Delta >= 0.0`), preventing quality regressions.")

    with fcci_c2:
        st.markdown("**Front**: How does `FakeListChatModel` facilitate zero-cost CI unit testing?")
        st.markdown("**Back**: It returns predetermined string responses sequentially from an in-memory list, verifying chain piping and parsing logic without incurring API latency or costs.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Continuous Integration for LLM applications:")

ci_chk1 = st.checkbox("I know how to structure an automated 5-stage CI pipeline for LLM applications.", value=True)
ci_chk2 = st.checkbox("I can write static prompt unit tests verifying variable bindings and double-escaped braces.", value=True)
ci_chk3 = st.checkbox("I understand how to use `FakeListChatModel` to test component pipelines locally in milliseconds.", value=True)
ci_chk4 = st.checkbox("I know how to audit dataset schemas and prune duplicates before running evaluations.", value=True)
ci_chk5 = st.checkbox("I can configure compact Canary Benchmarks (< 60s) for rapid pull request verification.", value=True)
ci_chk6 = st.checkbox("I understand how to enforce pre-merge branch protection using Regression Delta Contracts.", value=True)

if ci_chk1 and ci_chk2 and ci_chk3 and ci_chk4 and ci_chk5 and ci_chk6:
    st.success("Mastery Certified: You have mastered Continuous Integration for LLM applications, automated canary testing, and regression gating in LangSmith.")
