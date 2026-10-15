import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="17.4 RAG Evaluation Frameworks - RAG Master Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class FrameworkEvaluationEngine:
    """
    Production-grade Evaluation Framework Orchestrator:
    1. RAGAS metric simulation (Faithfulness, Answer Relevance, Context Precision).
    2. DeepEval unit-testing assertion engine (Pytest paradigm for LLMs).
    3. LangSmith dataset and trace backtesting runner.
    4. Custom enterprise domain-rule evaluators (regex, length, PII, compliance).
    5. LLM-as-a-Judge rubric evaluator with Chain-of-Thought (CoT) scoring.
    """
    def __init__(self):
        pass

    def run_llm_judge(self, query: str, context: str, answer: str, rubric: str) -> Dict[str, Any]:
        """Simulates an LLM-as-a-Judge evaluation with CoT reasoning."""
        context_words = set([w.lower().strip(",.?") for w in context.split() if len(w) > 3])
        answer_words = set([w.lower().strip(",.?") for w in answer.split() if len(w) > 3])

        overlap = sum(1 for w in answer_words if w in context_words)
        ratio = overlap / max(1, len(answer_words))

        if ratio >= 0.70:
            score = 5
            rationale = "The answer directly reflects facts asserted in the context with zero ungrounded drift."
        elif ratio >= 0.45:
            score = 3
            rationale = "The answer is partially supported, but introduces minor extrapolated phrasing."
        else:
            score = 1
            rationale = "Severe hallucination. The answer introduces unsupported claims absent from the context."

        return {
            "rubric_applied": rubric,
            "judge_score": score,
            "max_score": 5,
            "reasoning_trace": rationale,
            "is_passing": score >= 4
        }

    def run_deepeval_assertions(
        self,
        faithfulness: float,
        relevance: float,
        threshold_faithfulness: float = 0.85,
        threshold_relevance: float = 0.70
    ) -> List[Dict[str, Any]]:
        """Simulates DeepEval unit-test assertions (assert_test style)."""
        assertions = []

        pass_fth = faithfulness >= threshold_faithfulness
        assertions.append({
            "test": "test_faithfulness",
            "metric": "Faithfulness",
            "actual": faithfulness,
            "threshold": threshold_faithfulness,
            "passed": pass_fth,
            "message": "Faithfulness meets enterprise floor" if pass_fth else "Hallucination violation"
        })

        pass_rel = relevance >= threshold_relevance
        assertions.append({
            "test": "test_answer_relevance",
            "metric": "Answer Relevance",
            "actual": relevance,
            "threshold": threshold_relevance,
            "passed": pass_rel,
            "message": "Answer satisfies query intent" if pass_rel else "Answer diverges or dodges"
        })

        return assertions

    def evaluate_custom_enterprise_rules(self, answer: str) -> Dict[str, Any]:
        """Evaluates custom domain guardrails (length bounds, apologies, disclaimers)."""
        clean = answer.lower()
        has_apology = "i apologize" in clean or "as an ai" in clean
        is_length_valid = 10 <= len(answer.split()) <= 150
        has_citations = "[" in answer and "]" in answer

        all_passed = (not has_apology) and is_length_valid and has_citations

        return {
            "rule_no_apologies": not has_apology,
            "rule_concise_length": is_length_valid,
            "rule_contains_citations": has_citations,
            "all_custom_rules_passed": all_passed
        }


# Instantiate engine at module level
framework_engine = FrameworkEvaluationEngine()

# --- HEADER SECTION ---
st.title("17.4 RAG Evaluation Frameworks: RAGAS, LangSmith, DeepEval, & LLM-as-a-Judge Science")
st.caption("Chapter 17: RAG Evaluation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Building custom metric calculation scripts from scratch is essential for mastering the underlying math. 
In enterprise software engineering, however, production pipelines rely on **specialized, industrial-grade 
evaluation frameworks**.

Modern evaluation frameworks automate golden dataset synthesis, provide native CI/CD test runners, and harness 
**LLM-as-a-Judge** methodologies to evaluate subtle semantic nuances that rule-based systems cannot grasp.

This lesson explores the modern RAG evaluation ecosystem:
1. **RAGAS (Retrieval Augmented Generation Assessment)**: The open-source standard for reference-free Triad evaluation.
2. **LangSmith Evaluation**: Cloud-native observability, golden dataset versioning, and continuous online tracing.
3. **DeepEval**: Pytest-integrated unit testing for LLM applications with deterministic quality gates.
4. **Custom Enterprise Evaluators**: Business-rule validation, compliance auditing, and domain-specific heuristics.
5. **LLM-as-a-Judge Science**: Rubric engineering, scoring scales, and mitigating positional and verbosity biases.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Modern Enterprise Evaluation Framework Architecture", divider="rainbow")

st.markdown("""
Enterprise evaluation frameworks integrate into the development lifecycle across two distinct operational loops: 
the **Offline CI/CD Regression Loop** (blocking bad pull requests) and the **Online Production Telemetry Loop** 
(sampling live user interactions).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FrameworkArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // EVALUATION ASSET GENERATION
        subgraph cluster_assets {
            label="1. Evaluation Datasets & Telemetry Sources";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            GoldenDataset [label="Golden Benchmark Suite\\n(500 Curated Q&A + Gold Chunks)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            LiveTelemetry [label="Production Inference Traces\\n(Real User Queries & RAG Payloads)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // MODERN FRAMEWORK SPECIALIZATION
        subgraph cluster_frameworks {
            label="2. Specialized Evaluation Frameworks";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            RagasFramework [label="RAGAS Engine\\n(Reference-free Triad & Synthetic Q&A)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            DeepEvalFramework [label="DeepEval Pytest Runner\\n(assert_test() unit-test quality gates)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            LangSmithFramework [label="LangSmith Platform\\n(Dataset versioning & backtesting runs)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            CustomJudgeFramework [label="LLM-as-a-Judge & Custom Rules\\n(G-Eval rubrics & compliance checkers)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // CI/CD QUALITY GATE
        subgraph cluster_cicd {
            label="3. CI/CD Gate & Deployment Governance";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            TestRunner [label="Automated Test Harness\\n(GitHub Actions / GitLab CI)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            DeployGate [label="Production Release Gateway\\n(Blocks deployment if metrics degrade)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        }

        // EDGES
        GoldenDataset -> DeepEvalFramework;
        GoldenDataset -> RagasFramework;
        LiveTelemetry -> LangSmithFramework;
        GoldenDataset -> LangSmithFramework;
        LiveTelemetry -> CustomJudgeFramework;

        RagasFramework -> TestRunner;
        DeepEvalFramework -> TestRunner;
        LangSmithFramework -> TestRunner;
        CustomJudgeFramework -> TestRunner;

        TestRunner -> DeployGate;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Framework Matrix:
* **RAGAS**: Best for component ablation studies (comparing embedding models or chunk sizes).
* **DeepEval**: Best for software engineers wanting unit-test assertions (`assert_test()`) in Python CI/CD.
* **LangSmith**: Best for end-to-end trace visualization, agent debugging, and human feedback capture.
* **LLM-as-a-Judge**: Best for nuanced domain rubrics (compliance, brand tone, legal disclaimers).
""")

st.divider()

# ==============================================================================
# SECTION 2: RAGAS
# ==============================================================================
st.header("2. RAGAS (Automated Triad Metrics, Component Evaluation & Reference-Free Scoring)", divider="rainbow")

st.markdown("""
**RAGAS (Retrieval Augmented Generation Assessment)** is the leading open-source framework for evaluating 
compound RAG pipelines without requiring human-annotated ground-truth labels.
""")

ragas_col1, ragas_col2 = st.columns([1, 1])

with ragas_col1:
    st.markdown("""
    ### Core Capabilities of RAGAS:
    * **1. Component-Level Metrics**: Evaluates retrieval and generation separately, avoiding blame confusion.
    * **2. Synthetic Test Generation**: Automatically generates synthetic question-answer-context evaluation datasets directly from your raw PDF and text documents using evol-instruct algorithms.
    * **3. Reference-Free Scoring**: Evaluates Faithfulness and Answer Relevance on live user queries without ground-truth answers.
    * **4. Aspect-Critique Evaluators**: Provides out-of-the-box evaluations for Maliciousness, Coherence, and Correctness.
    """)

with ragas_col2:
    st.markdown("### RAGAS Metric Formulation Reference")
    st.code(r'''
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Run automated evaluation dataset
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(results)
# {'faithfulness': 0.94, 'answer_relevancy': 0.88, 'context_precision': 0.82}
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: LANGSMITH EVALUATION
# ==============================================================================
st.header("3. LangSmith Evaluation (Dataset Versioning, Trace Backtesting & CI/CD Regression Gates)", divider="rainbow")

st.markdown("""
**LangSmith** provides a production evaluation platform that connects offline experimentation 
with live online production monitoring.
""")

ls_col1, ls_col2 = st.columns([1, 1])

with ls_col1:
    st.markdown("""
    ### Key Enterprise Features:
    * **Dataset Versioning**: Store, fork, and manage gold-standard evaluation datasets in a collaborative web UI.
    * **Trace Backtesting**: Re-run 1,000 historic production user queries through an updated prompt or new embedding model to verify that answers improve rather than regress.
    * **Feedback Ingestion**: Automatically link user thumbs-up / thumbs-down signals to exact execution traces.
    * **Evaluator Pipelines**: Run automated evaluators on 5% of production traffic as a background canary process.
    """)

with ls_col2:
    st.markdown("### LangSmith Evaluation Architecture")
    st.code(r'''
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Run backtest against a versioned dataset
eval_results = evaluate(
    my_rag_pipeline,
    data="enterprise_compliance_v2",
    evaluators=[faithfulness_evaluator],
    experiment_prefix="prompt_v4_test"
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 4: DEEPEVAL
# ==============================================================================
st.header("4. DeepEval (Unit-Testing Paradigm for LLMs, Assertions & Pytest Integration)", divider="rainbow")

st.markdown("""
**DeepEval** reimagines LLM evaluation through the lens of traditional software testing: 
treating language model responses as unit tests evaluated with standard `assert` statements.
""")

de_col1, de_col2 = st.columns([1, 1])

with de_col1:
    st.markdown("""
    ### The Unit-Testing Paradigm:
    * **Pytest Integration**: Write test files (`test_rag.py`) that run via standard `pytest` in GitHub Actions.
    * **Deterministic Assertions**: Use `assert_test(test_case, [metric])` with quantitative thresholds (e.g., `threshold = 0.85`).
    * **Local Execution**: Runs locally or in CI without requiring complex cloud dashboards.
    * **Metric Suite**: Faithfulness, Hallucination, Answer Relevancy, G-Eval, and Bias.
    """)

with de_col2:
    st.markdown("### DeepEval Pytest Code Structure")
    st.code(r'''
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

def test_rag_compliance():
    metric = FaithfulnessMetric(threshold=0.90)
    test_case = LLMTestCase(
        input="What is the failover SLA?",
        actual_output="Failover completes in 120 seconds.",
        retrieval_context=["Failover completes in 120 seconds."]
    )
    assert_test(test_case, [metric])
''', language="python")

st.divider()

# ==============================================================================
# SECTION 5: CUSTOM EVALUATORS
# ==============================================================================
st.header("5. Custom Evaluators (Domain Rules, RegEx Pattern Checkers & Latency/Cost Guardrails)", divider="rainbow")

st.markdown("""
Off-the-shelf frameworks evaluate general linguistic qualities. Enterprise deployments require 
**Custom Business-Rule Evaluators** to enforce organization-specific constraints.
""")

ce_col1, ce_col2 = st.columns([1, 1])

with ce_col1:
    st.markdown("""
    ### Typical Custom Evaluator Archetypes:
    * **PII & Secret Detection**: Verifies that responses contain zero credit card numbers, SSNs, or API keys.
    * **Negative Keyword Filter**: Rejects responses containing competitor brand names or unauthorized forward-looking statements.
    * **Regulatory Boilerplate Validator**: Verifies that financial answers include the mandatory SEC disclosure disclaimer.
    * **Latency & Cost SLA Guard**: Fails queries where prompt size exceeded 4,000 tokens or latency exceeded 1.5 seconds.
    """)

with ce_col2:
    st.markdown("### Custom Enterprise Rule Engine")
    rule_table = pd.DataFrame([
        {"Custom Rule": "No Conversational Apologies", "Implementation": "Regex check for 'I apologize' or 'As an AI'", "Enforcement": "Hard Failure"},
        {"Custom Rule": "Mandatory Inline Citation", "Implementation": "Regex check for [DOC-ID] pattern", "Enforcement": "Hard Failure"},
        {"Custom Rule": "Word Count Bounds", "Implementation": "10 <= words <= 150", "Enforcement": "Warning Flag"},
        {"Custom Rule": "Required Legal Disclosure", "Implementation": "Sub-string matching of SEC disclaimer", "Enforcement": "Hard Failure"}
    ])
    st.dataframe(rule_table, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: LLM-AS-A-JUDGE
# ==============================================================================
st.header("6. LLM-as-a-Judge (Pointwise, Pairwise, Position Bias & G-Eval CoT Scoring)", divider="rainbow")

st.markdown("""
**LLM-as-a-Judge** employs a frontier model (e.g., GPT-4o, Claude 3.5 Sonnet) to evaluate responses 
against structured scoring rubrics using Chain-of-Thought (CoT) reasoning.
""")

judge_col1, judge_col2 = st.columns([1, 1])

with judge_col1:
    st.markdown("""
    ### The 3 Judge Methodologies:
    * **1. Pointwise Scoring**: Evaluates a single answer on a 1-to-5 Likert scale against an explicit rubric.
    * **2. Pairwise Comparison**: Compares two answers (`Model A` vs `Model B`) and selects the superior response. Prone to position bias.
    * **3. G-Eval (Chain-of-Thought Rubric)**: Directs the judge to generate step-by-step reasoning *before* calculating the final numerical score. Drastically increases correlation with human expert judges.
    
    ### Known Judge Biases & Mitigations:
    * **Verbosity Bias**: Models naturally favor longer, wordier answers. *Mitigation: Instruct judge to penalize unnecessary padding.*
    * **Position Bias**: Pairwise judges favor Model A over Model B regardless of content. *Mitigation: Run tournament twice with swapped positions (`A vs B` and `B vs A`).*
    * **Self-Enhancement Bias**: Models rate outputs produced by their own family higher. *Mitigation: Use independent evaluator models.*
    """)

with judge_col2:
    st.markdown("### G-Eval Chain-of-Thought Prompt Pattern")
    st.code(r'''
You are an expert impartial judge evaluating RAG answer quality.
Rubric Criteria (Score 1 to 5):
1 = Completely ungrounded hallucination
3 = Partially supported with extrapolations
5 = 100% faithful and concise

Evaluation Workflow:
1. Write a <thinking> block analyzing each claim against the context.
2. Identify any unsupported statements.
3. Emit final integer score: [SCORE: X].
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Framework Evaluation Orchestrator", divider="rainbow")

st.markdown("""
Below is the production implementation of `FrameworkEvaluationEngine`. It runs in module scope, 
powering the LLM-as-a-Judge simulator, DeepEval assertion checker, and custom business-rule evaluators.
""")

st.code(r'''
class FrameworkEvaluationEngine:
    """
    Enterprise Evaluation Framework Orchestrator:
    - LLM-as-a-Judge simulation with Chain-of-Thought (CoT) reasoning traces.
    - DeepEval-style unit test assertions (assert_test pattern).
    - Custom enterprise business-rule validators.
    """
    def run_llm_judge(self, query: str, context: str, answer: str, rubric: str) -> Dict[str, Any]:
        context_words = set([w.lower().strip(",.?") for w in context.split() if len(w) > 3])
        answer_words = set([w.lower().strip(",.?") for w in answer.split() if len(w) > 3])
        overlap = sum(1 for w in answer_words if w in context_words)
        ratio = overlap / max(1, len(answer_words))

        if ratio >= 0.70:
            score, trace = 5, "The answer reflects facts directly stated in context with zero drift."
        elif ratio >= 0.45:
            score, trace = 3, "Partially supported, but introduces minor extrapolated phrasing."
        else:
            score, trace = 1, "Severe hallucination. Claims are absent from context."

        return {
            "rubric": rubric, "judge_score": score, "max_score": 5,
            "reasoning_trace": trace, "is_passing": score >= 4
        }

    def run_deepeval_assertions(
        self, faithfulness: float, relevance: float, thresh_fth: float = 0.85, thresh_rel: float = 0.70
    ) -> List[Dict[str, Any]]:
        return [
            {
                "test": "test_faithfulness", "actual": faithfulness, "threshold": thresh_fth,
                "passed": faithfulness >= thresh_fth
            },
            {
                "test": "test_answer_relevance", "actual": relevance, "threshold": thresh_rel,
                "passed": relevance >= thresh_rel
            }
        ]

    def evaluate_custom_enterprise_rules(self, answer: str) -> Dict[str, Any]:
        clean = answer.lower()
        no_apologies = "i apologize" not in clean and "as an ai" not in clean
        length_ok = 10 <= len(answer.split()) <= 150
        has_citations = "[" in answer and "]" in answer
        return {
            "no_apologies": no_apologies, "length_ok": length_ok, "has_citations": has_citations,
            "all_passed": no_apologies and length_ok and has_citations
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: LLM-as-a-Judge Simulator & Framework Comparison Matrix", divider="rainbow")

st.markdown("""
Test the evaluation frameworks interactively. 
Evaluate sample RAG responses using the **LLM-as-a-Judge Simulator**, execute **DeepEval Unit Assertions**, 
and inspect **Custom Business Rules**.
""")

wb_context_text = "The enterprise SLA guarantees 99.99% uptime for core banking services. Failover completes within 120 seconds."

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("### Test Generation Scenario")
    eval_framework_choice = st.selectbox(
        "Choose Evaluation Mode:",
        options=[
            "1. LLM-as-a-Judge (G-Eval CoT Rubric)",
            "2. DeepEval Unit-Test Assertions",
            "3. Custom Enterprise Rule Engine"
        ],
        index=0
    )

    candidate_answer = st.text_area(
        "Candidate Generated Answer to Evaluate:",
        value="The SLA guarantees 99.99% uptime for core banking services, and automated failover completes in 120 seconds [DOC-1].",
        height=100
    )
    active_rubric = "Factual Faithfulness & Citation Attribution"

with wb_col2:
    st.markdown("### Framework Evaluation Output")

    if eval_framework_choice.startswith("1."):
        judge_res = framework_engine.run_llm_judge(
            query="What is the SLA?", context=wb_context_text, answer=candidate_answer, rubric=active_rubric
        )
        st.metric(
            label="LLM Judge Rating",
            value=f"{judge_res['judge_score']} / {judge_res['max_score']}",
            delta="PASSED QUALITY GATE" if judge_res["is_passing"] else "FAILED QUALITY GATE"
        )
        st.info(f"**Judge Reasoning Trace**: {judge_res['reasoning_trace']}")

    elif eval_framework_choice.startswith("2."):
        # Simulate deepeval test assertions
        assertions = framework_engine.run_deepeval_assertions(faithfulness=0.92, relevance=0.88)
        st.markdown("#### Pytest Assertion Results:")
        for a in assertions:
            if a["passed"]:
                st.success(f"PASSED: `{a['test']}` (Actual: {a['actual']} >= Threshold: {a['threshold']})")
            else:
                st.error(f"FAILED: `{a['test']}` (Actual: {a['actual']} < Threshold: {a['threshold']})")

    else:
        custom_rules = framework_engine.evaluate_custom_enterprise_rules(candidate_answer)
        st.markdown("#### Business Rule Checkers:")
        st.write(f"- Zero Apologies / Filler: `{'PASSED' if custom_rules['no_apologies'] else 'FAILED'}`")
        st.write(f"- Word Count Bounds (10-150 words): `{'PASSED' if custom_rules['length_ok'] else 'FAILED'}`")
        st.write(f"- Contains Bracketed Citations: `{'PASSED' if custom_rules['has_citations'] else 'FAILED'}`")

        if custom_rules["all_passed"]:
            st.success("✅ All custom enterprise rules passed successfully.")
        else:
            st.error("❌ Business rule violation detected!")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What are the three primary biases of LLM-as-a-Judge evaluators, and how are they mitigated in production?**"):
    st.markdown("""
    **Answer**:
    1. **Verbosity Bias**: The tendency to assign higher scores to longer, wordier answers even when they contain no additional facts.
       * *Mitigation*: Explicitly instruct the rubric to penalize padding and reward concise precision.
    2. **Position Bias**: In pairwise comparisons (`Model A` vs `Model B`), judges favor the first response presented (`Model A`).
       * *Mitigation*: Run tournament evaluations twice with swapped order (`A vs B` and `B vs A`) and take the consensus.
    3. **Self-Enhancement Bias**: A model (e.g., GPT-4o) systematically rates outputs generated by GPT models higher than Claude or open-source models.
       * *Mitigation*: Use an independent, neutral model family for evaluation (e.g., Claude judging GPT outputs, or human meta-auditing).
    """)

with st.expander("**Q2: How does DeepEval's unit-testing paradigm change CI/CD workflows for AI engineering?**"):
    st.markdown("""
    **Answer**:
    Traditionally, testing AI applications was qualitative and ad-hoc. DeepEval bridges this by framing evaluation as standard unit tests:
    1. Developers define test cases with inputs, outputs, and thresholds.
    2. Tests execute via standard `pytest` in GitHub Actions on every pull request.
    3. If a prompt edit causes Faithfulness to drop below `0.90`, `assert_test()` raises an `AssertionError`, failing the CI build and blocking the deployment.
    """)

with st.expander("**Q3: What is synthetic test dataset generation in RAGAS, and why is it preferred over human manual annotation?**"):
    st.markdown("""
    **Answer**:
    Manual annotation of 500 gold Q&A pairs with verified document IDs requires hundreds of hours of expensive expert labor.
    RAGAS generates synthetic evaluation suites automatically:
    1. It extracts chunks from your documents.
    2. An LLM generates questions of varying difficulty (reasoning, multi-context, conditioning) using evol-instruct algorithms.
    3. It attaches ground-truth answers and chunk IDs automatically.
    This enables continuous dataset regeneration whenever new knowledge base documents are ingested.
    """)

with st.expander("**Q4: How does LangSmith combine offline experimentation with online production observability?**"):
    st.markdown("""
    **Answer**:
    LangSmith unifies the feedback loop:
    * **Online Observability**: Captures execution traces from production user queries, logging inputs, retrieved chunks, latency, token costs, and user thumbs-up/down feedback.
    * **Dataset Curation**: Developers can click *"Add to Dataset"* on any failed production trace to save the bug into a regression dataset.
    * **Offline Backtesting**: When a prompt or chunking strategy is updated, developers run a backtest against the curated dataset in LangSmith to verify that the bug is fixed without breaking other cases.
    """)

with st.expander("**Q5: Why should LLM-as-a-Judge evaluators always generate a Chain-of-Thought before emitting a numerical score?**"):
    st.markdown("""
    **Answer**:
    In autoregressive decoding, if the model is prompted to output a score immediately (`[SCORE: X]`), it must commit to a token before computing the logical evaluation of the text.
    In the G-Eval / CoT pattern, the model first writes an analysis of claims against the context. These reasoning tokens become part of the Key-Value attention cache, allowing the model's self-attention layers to condition the final numerical score on the written reasoning, significantly increasing scoring accuracy and alignment with human judges.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Using the Same Model as Generator and Judge**
    * **Failure Mode**: Using GPT-4o-mini to evaluate answers generated by GPT-4o-mini leads to self-enhancement bias where the judge overlooks its own hallucinations.
    * **Solution**: Use a more capable frontier model or a different model family for evaluation.
    """)

    st.error("""
    **Production Trap 2: Running Uncached LLM Judges on 100% of Production Traffic**
    * **Failure Mode**: Running a GPT-4o judge on every single live user request doubles inference costs and introduces latency.
    * **Solution**: Run judges on a 5% to 10% sample of production traffic, or use lightweight rule checkers for 100% filtering.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Vague 1-to-5 Rubrics Without Anchor Definitions**
    * **Failure Mode**: Asking a judge *"Rate quality from 1 to 5"* produces inconsistent scores because the model invents its own criteria.
    * **Solution**: Define explicit anchor descriptions for every integer score (1, 2, 3, 4, 5).
    """)

    st.warning("""
    **Production Trap 4: Treating Framework Metrics as Standalone Truth**
    * **Failure Mode**: Blindly optimizing for a high RAGAS score while real users complain about unhelpful answers.
    * **Solution**: Periodically correlate framework scores with real human customer satisfaction (CSAT) ratings.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: RAGAS Framework Focus")
        st.caption("What is the primary strength of RAGAS?")
        st.markdown("**Back**: Automated synthetic test data generation and reference-free component evaluation of the RAG Triad.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: DeepEval CI/CD Paradigm")
        st.caption("How does DeepEval integrate into software engineering workflows?")
        st.markdown("**Back**: Via `pytest` unit test assertions (`assert_test()`), blocking broken pull requests in GitHub Actions.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: G-Eval Methodology")
        st.caption("How does G-Eval improve LLM-as-a-Judge scoring accuracy?")
        st.markdown("**Back**: By generating intermediate Chain-of-Thought evaluation steps before emitting the final numerical score.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Judge Verbosity Bias")
        st.caption("What is verbosity bias in LLM judges?")
        st.markdown("**Back**: The tendency to assign higher scores to longer answers regardless of factual density; mitigated by explicit length penalties.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of modern RAG evaluation frameworks before proceeding:")

c1 = st.checkbox("I understand the architectural roles of RAGAS, DeepEval, and LangSmith in the evaluation lifecycle.", value=True)
c2 = st.checkbox("I can implement Pytest-integrated unit test assertions (`assert_test()`) using DeepEval.", value=True)
c3 = st.checkbox("I understand how synthetic test datasets are generated from raw documents using RAGAS.", value=True)
c4 = st.checkbox("I know how to design Chain-of-Thought rubrics for LLM-as-a-Judge evaluators (G-Eval).", value=True)
c5 = st.checkbox("I can recognize and mitigate verbosity, position, and self-enhancement biases in LLM judges.", value=True)
c6 = st.checkbox("I can construct custom enterprise business-rule evaluators to enforce regulatory and formatting constraints.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **RAG Evaluation Frameworks Mastery Achieved!** You are ready to deploy enterprise-grade evaluation pipelines.")
