import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="36.2 Evaluation Mistakes - LangSmith Master Curriculum",
    page_icon="🚫",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class EvaluationMistakeDiagnosticEngine:
    """
    Diagnostic & Calibration Engine for LLM Evaluation Mistakes in LangSmith.
    Identifies and models the 5 primary benchmark anti-patterns:
    1. Bad Datasets (Contamination, synthetic monoculture, zero negative edge cases)
    2. Weak Criteria (Vague subjective rubrics without concrete operational boundaries)
    3. LLM Judge Over-Reliance (Verbosity bias, self-enhancement bias, position bias)
    4. Ignoring Human Evaluation (No human annotation queues, blind trust in automated scores)
    5. Wrong Metrics (Using BLEU/Perplexity for RAG instead of Faithfulness and Retrieval Precision)
    """
    def __init__(self):
        self.simulation_records: List[Dict[str, Any]] = []

    def simulate_evaluation_bias(
        self,
        candidate_length_words: int,
        is_self_model_family: bool,
        dataset_synthetic_percent: int,
        metric_choice: str,
        rubric_clarity: str
    ) -> Dict[str, Any]:
        """
        Simulates the score distortions introduced by evaluation flaws and anti-patterns.
        """
        # True Underlying Grounded Quality (Objective ground truth = 0.75)
        true_quality = 0.75

        # Bias 1: Verbosity / Length Inflation Bias
        # LLM judges systematically award higher scores to wordier responses regardless of factual substance
        length_inflation = min(0.18, max(-0.10, (candidate_length_words - 50) * 0.0008))

        # Bias 2: Self-Enhancement / Model Family Bias
        family_bias = 0.08 if is_self_model_family else 0.0

        # Bias 3: Weak Rubric Variance
        rubric_penalty = 0.0
        if rubric_clarity == "Vague ('Rate 1-5')":
            rubric_variance_noise = 0.12  # High stochastic variance
            rubric_penalty = -0.05
        elif rubric_clarity == "Moderate":
            rubric_variance_noise = 0.04
        else:  # Strict Binary Multi-Criterion Rubric
            rubric_variance_noise = 0.01

        # Bias 4: Metric Mismatch Distortion
        metric_distortion = 0.0
        if metric_choice == "BLEU / ROUGE (Surface N-Gram)":
            # Penalizes semantically correct rephrasings severely
            metric_distortion = -0.22
        elif metric_choice == "Perplexity":
            # Uncorrelated with factual grounding
            metric_distortion = -0.15

        # Computed Distorted Score vs True Ground Truth
        distorted_score = round(
            min(1.0, max(0.0, true_quality + length_inflation + family_bias + metric_distortion + rubric_variance_noise)),
            3
        )
        measurement_error = round(distorted_score - true_quality, 3)

        # Flag detected anti-patterns
        flaws_detected = []
        if dataset_synthetic_percent > 85:
            flaws_detected.append("Synthetic Monoculture: Benchmark is > 85% synthetic data; lacks production distribution complexity.")
        if candidate_length_words > 200 and length_inflation > 0.10:
            flaws_detected.append("Verbosity Bias: Candidate received inflated score primarily due to word count rather than accuracy.")
        if is_self_model_family:
            flaws_detected.append("Self-Enhancement Bias: Judge and candidate belong to same model family, introducing positive distortion.")
        if "BLEU" in metric_choice or "Perplexity" in metric_choice:
            flaws_detected.append("Wrong Metric: Surface n-gram or perplexity metric misrepresents semantic RAG retrieval and faithfulness.")
        if rubric_clarity.startswith("Vague"):
            flaws_detected.append("Weak Rubric: Subjective grading rubric causes uninterpretable scores with high inter-run variance.")

        report = {
            "true_quality_score": true_quality,
            "distorted_measured_score": distorted_score,
            "measurement_error": measurement_error,
            "flaws_detected": flaws_detected,
            "bias_breakdown": {
                "verbosity_inflation": round(length_inflation, 3),
                "family_bias": round(family_bias, 3),
                "metric_distortion": round(metric_distortion, 3),
                "rubric_variance": round(rubric_variance_noise, 3)
            },
            "status": "CALIBRATED" if abs(measurement_error) <= 0.05 else "BIASED_MEASUREMENT"
        }
        self.simulation_records.append(report)
        return report

# Initialize engine in session state
if "eval_mistakes_engine" not in st.session_state:
    st.session_state.eval_mistakes_engine = EvaluationMistakeDiagnosticEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("🚫 Chapter 36.2: Common LangSmith Evaluation Mistakes")
st.caption("Contaminated Datasets, Vague Rubrics, LLM Judge Biases, Neglected Human Oversight, and Flawed Metric Choices")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Evaluation Anti-Patterns and Flawed Benchmark Diagnostics", divider="rainbow")

st.markdown("""
In AI engineering, a flawed evaluation system is worse than having no evaluation system at all. 
While having no evaluation forces teams to proceed cautiously with manual checks, a flawed evaluation pipeline provides 
*false mathematical confidence*—certifying broken, hallucinating, or unsafe models while rejecting superior candidates. 
Diagnosing and eliminating benchmark anti-patterns is a critical LLMOps skill.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EvaluationAntiPatterns {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_anti_patterns {
            label = "1. The 5 Core Evaluation Anti-Patterns";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F43F5E";
            fontsize = 12;

            A1 [label="Anti-Pattern 1: Bad Datasets\\n(100% Synthetic / Zero Missing Context Edge Cases)", fillcolor="#7C2D12", color="#EA580C"];
            A2 [label="Anti-Pattern 2: Weak Criteria\\n(Vague Rubric: 'Grade 1-5' without clear bounds)", fillcolor="#7C2D12", color="#EA580C"];
            A3 [label="Anti-Pattern 3: Judge Over-Reliance\\n(Uncalibrated LLM Judge: Verbosity & Self-Bias)", fillcolor="#7C2D12", color="#EA580C"];
            A4 [label="Anti-Pattern 4: Ignoring Humans\\n(Zero SME Review / No LangSmith Annotation Queues)", fillcolor="#7C2D12", color="#EA580C"];
            A5 [label="Anti-Pattern 5: Wrong Metrics\\n(Using BLEU/Perplexity for RAG Factuality)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_distortions {
            label = "2. Systematic Measurement Distortions";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            D_ECHO [label="Synthetic Echo Chamber\\n(High scores on test; instant crash on real users)", fillcolor="#881337", color="#F43F5E"];
            D_VAR [label="Severe Inter-Run Variance\\n(Identical runs swinging +/- 20% due to loose rubrics)", fillcolor="#881337", color="#F43F5E"];
            D_LEN [label="Length Hallucination Inflation\\n(Wordy, ungrounded responses rated higher)", fillcolor="#881337", color="#F43F5E"];
            D_MIS [label="Surface Lexical Penalties\\n(Valid synonym rephrasings marked as failures)", fillcolor="#881337", color="#F43F5E"];
        }

        subgraph cluster_solutions {
            label = "3. Enterprise Remediation Standards";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            S_DATA [label="Stratified Production Sampling\\n(70% real traffic, 20% edge/refusal, 10% adversarial)", fillcolor="#065F46", color="#059669"];
            S_RUB [label="Multi-Criterion Binary Rubrics\\n(Strict binary assertions for faithfulness & tone)", fillcolor="#065F46", color="#059669"];
            S_TRI [label="Multi-Evaluator Triangulation\\n(Code heuristics + embeddings + calibrated judge)", fillcolor="#065F46", color="#059669"];
            S_HUM [label="Confidence-Gated Annotation Queues\\n(Route ambiguous scores to SME review)", fillcolor="#065F46", color="#059669"];
            S_MET [label="Domain-Aligned RAG Metrics\\n(Context Precision, Faithfulness, Answer Relevance)", fillcolor="#065F46", color="#059669"];
        }

        A1 -> D_ECHO [label="Produces"];
        A2 -> D_VAR [label="Produces"];
        A3 -> D_LEN [label="Produces"];
        A5 -> D_MIS [label="Produces"];

        D_ECHO -> S_DATA [label="Resolved By"];
        D_VAR -> S_RUB [label="Resolved By"];
        D_LEN -> S_TRI [label="Resolved By"];
        A4 -> S_HUM [label="Resolved By"];
        D_MIS -> S_MET [label="Resolved By"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Evaluation Integrity
1. **Never Trust Benchmarks Composed Exclusively of Synthetic Data**: When an LLM generates synthetic test cases and another LLM evaluates them, they create a synthetic echo chamber. Real user queries have typos, colloquial phrasing, missing context, and incomplete thoughts that synthetic generators fail to model.
2. **Verbosity Bias is Pervasive in LLM Judges**: Uncalibrated LLM judges routinely award higher ratings to longer, more articulate answers even when they contain fabricated facts, while penalizing concise, 100% accurate responses.
3. **BLEU and Perplexity Do Not Measure Factuality**: Using machine translation metrics (BLEU/ROUGE) or language modeling loss (Perplexity) to evaluate RAG systems penalizes valid paraphrasing and completely ignores whether an answer is grounded in the retrieved context.
""")

st.divider()

# ==============================================================================
# SECTION 2: BAD EVALUATION DATASETS
# ==============================================================================
st.header("2. Bad Evaluation Datasets (Contaminated Benchmarks, Severe Duplication and Synthetic Monoculture)", divider="rainbow")

st.markdown("""
The most frequent cause of flawed evaluations is the dataset itself. 
A benchmark that does not reflect real-world user intent distributions leads to false confidence during development 
and immediate failure in production.
""")

st.markdown("""
### The 3 Pathology Patterns in Benchmark Datasets
* **1. Synthetic Monoculture (The Echo Chamber)**: Generating 1,000 test cases using `gpt-4o` prompts. The generated questions are unnaturally clean, grammatically perfect, and follow predictable patterns that do not represent production traffic.
* **2. Data Contamination (Train/Test Leakage)**: Including examples in the evaluation set that were present in the model's fine-tuning corpus or embedding store, resulting in artificially high memorization scores.
* **3. Severe Duplication & Zero Hard Negatives**: Having 200 variations of *"How do I reset my password?"* while having zero unanswerable questions, zero adversarial prompt injections, and zero edge cases testing missing context.
""")

dataset_mistake_df = pd.DataFrame([
    {"Dataset Pathology": "100% Synthetic Generation", "Real-World Flaw": "Missing colloquial phrasing, typos, and multi-intent queries", "Fix": "Seed 70% of dataset from anonymized production LangSmith traces"},
    {"Dataset Pathology": "All-Positive Happy Path", "Real-World Flaw": "System learns to always answer, causing hallucinations on missing data", "Fix": "Inject 20% unanswerable queries requiring polite refusal"},
    {"Dataset Pathology": "Massive Redundant Clustering", "Real-World Flaw": "Artificially skews scores toward a handful of repetitive FAQs", "Fix": "Run semantic deduplication (cosine similarity < 0.85) during dataset curation"}
])
st.dataframe(dataset_mistake_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Teams use LangSmith's dataset management tools to run automated deduplication. 
Before adding a candidate query to a golden benchmark, the system computes embeddings and rejects candidates with `cosine_similarity > 0.90` to existing examples, ensuring benchmark diversity.
""")

st.divider()

# ==============================================================================
# SECTION 3: WEAK EVALUATION CRITERIA
# ==============================================================================
st.header("3. Weak Evaluation Criteria (Vague Rubrics, Subjective Prompt Grading and Lack of Measurable Invariants)", divider="rainbow")

st.markdown("""
Asking an LLM judge: *"Is this answer good? Rate from 1 to 5"* produces garbage evaluation metrics. 
Different runs will rate the same response a 3, 4, or 5 depending on token sampling variance, temperature, 
and prompt framing. Production evaluation demands explicit, multi-criterion operational rubrics.
""")

st.markdown("""
### Anatomy of a Weak vs Strong Evaluation Rubric
* **Weak Rubric (Subjective & Ambiguous)**:
  ```text
  Rate the assistant's answer from 1 to 5 based on how helpful and accurate it is.
  ```
* **Strong Rubric (Objective, Binary Criteria)**:
  ```text
  Evaluate the candidate response against the retrieved context based strictly on:
  1. Faithfulness (1 or 0): Does the response contain ANY factual claim not directly supported by the context?
  2. Completeness (1 or 0): Does the response address all specific sub-questions asked?
  3. Non-Hallucination (1 or 0): If context is empty, did it refuse?
  Score = 1.0 only if Faithfulness == 1 AND Completeness == 1. Otherwise 0.0.
  ```
""")

st.code('''
# Production-Grade Multi-Criterion Binary Rubric in LangSmith
FAITHFULNESS_JUDGE_PROMPT = """You are a strict compliance auditor evaluating an AI assistant's response.
Retrieved Context:
{context}

User Query:
{query}

Candidate Response:
{candidate_response}

Instructions:
Evaluate whether every claim made in the candidate response is directly substantiated by the retrieved context.
Do NOT reward articulate phrasing, politeness, or length.
If the response mentions ANY external fact not present in the context, score FAITHFULNESS as 0.

Output your verdict as a JSON object:
{{
  "claims_evaluated": ["claim 1", "claim 2"],
  "unsupported_claims": [],
  "faithfulness_score": 1,
  "rationale": "All claims are strictly grounded in context."
}}
"""
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Top teams enforce Pydantic output schemas on LLM judges. The judge cannot return a free-text integer; 
it must output a structured list of individual claims, label each claim as supported/unsupported, and compute 
the final score deterministically from the ratio of supported claims.
""")

st.divider()

# ==============================================================================
# SECTION 4: OVER-RELYING ON LLM JUDGES
# ==============================================================================
st.header("4. Over-Relying on LLM Judges (Self-Enhancement Bias, Positional Bias and Length Hallucination Inflation)", divider="rainbow")

st.markdown("""
LLM-as-a-judge is a powerful evaluation tool, but treating it as infallible truth is a dangerous mistake. 
Academic and industrial research has documented three systematic biases in LLM judges:
""")

st.markdown("""
### The 3 Core Biases of LLM Judges
1. **Verbosity Bias (Length Inflation)**: Evaluator models systematically assign higher quality ratings to longer, multi-paragraph responses. A concise 2-sentence response containing the exact correct answer often scores lower than a 300-word response that repeats the query and adds ungrounded filler.
2. **Self-Enhancement Bias**: When GPT-4 evaluates outputs from GPT-4, Claude 3.5, and Llama 3, it statistically favors its own model family's stylistic choices and formatting preferences by 5–10%.
3. **Position Bias in Pairwise Grading**: In pairwise A/B prompt evaluation, LLM judges show a marked preference for whichever response is presented first (Option A) in the prompt, regardless of content quality.
""")

judge_bias_table = pd.DataFrame([
    {"Judge Bias": "Verbosity Bias", "Mechanism": "Model equates length with thoroughness", "Remedy": "Strip formatting; instruct judge to penalize unnecessary length; evaluate conciseness explicitly"},
    {"Judge Bias": "Self-Enhancement Bias", "Mechanism": "Favors own token distribution style", "Remedy": "Use cross-model judging (e.g., Claude judging GPT, GPT judging Claude) or take consensus average"},
    {"Judge Bias": "Pairwise Position Bias", "Mechanism": "Attention decay over long prompts favors Option A", "Remedy": "Run evaluation twice with swapped positions (A/B and B/A); score as a tie if outcomes conflict"}
])
st.dataframe(judge_bias_table, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production pipelines run position-swapped pairwise evaluations. Candidate A and Candidate B are evaluated as `[A, B]` 
and then immediately re-evaluated as `[B, A]`. If the judge selects A in the first run and B in the second run, 
the result is flagged as positional noise and discounted.
""")

st.divider()

# ==============================================================================
# SECTION 5: IGNORING HUMAN EVALUATION
# ==============================================================================
st.header("5. Ignoring Human Evaluation (Disconnected from User Sentiment, Blind Trust in Automated Scores and Zero Adjudication)", divider="rainbow")

st.markdown("""
A common trap is assuming that because automated evaluation scores are high (e.g., 0.94 faithfulness), 
users must be satisfied. In reality, automated judges can easily optimize for metrics that drift away from user satisfaction.
""")

st.markdown("""
### The Cost of Neglecting Human Review
* **Blind Spots on User Frustration**: An automated judge might award 1.0 for a technically faithful response that is completely unhelpful, overly pedantic, or condescending in tone.
* **Evaluator Drift**: Over time, automated evaluators drift out of calibration with evolving business requirements unless periodically audited against human judgment.
* **Unsupervised Ground Truth Curation**: Building evaluation datasets using purely automated labels introduces systematic errors that propagate through all subsequent releases.
""")

st.code('''
from langsmith import Client

client = Client()

# Route disputed evaluation traces to LangSmith Annotation Queue
def triage_trace_to_human_queue(run_id: str, automated_score: float, reason: str):
    client.create_annotation_queue_run(
        queue_name="weekly-sme-calibration-queue",
        run_id=run_id,
        metadata={
            "automated_score": automated_score,
            "escalation_reason": reason,
            "requires_sme_adjudication": True
        }
    )
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams hold weekly "eval calibration sessions." Domain experts and prompt engineers review 50 randomly sampled 
runs from LangSmith Annotation Queues, comparing human scores against automated judge scores to calculate 
Cohen's Kappa (inter-rater agreement). If agreement drops below 0.80, the automated judge prompt is recalibrated.
""")

st.divider()

# ==============================================================================
# SECTION 6: MEASURING THE WRONG METRIC
# ==============================================================================
st.header("6. Measuring the Wrong Metric (Perplexity / BLEU for RAG, Over-Indexing on Formatting vs Grounding)", divider="rainbow")

st.markdown("""
Using traditional NLP metrics to evaluate generative AI applications is one of the most common rookie mistakes. 
Metrics developed for machine translation (BLEU, ROUGE) or language modeling (Perplexity) fail to evaluate the qualities 
that actually determine whether a RAG or agentic application succeeds in production.
""")

st.markdown("""
### Why Legacy Metrics Fail for Generative AI
1. **BLEU / ROUGE (Surface N-Gram Overlap)**: Penalizes perfectly valid, semantically identical responses simply because the model used synonyms or reordered sentences. An answer stating *"The refund window is 30 days"* receives a terrible BLEU score against a reference of *"Customers may request refunds within one month"*, despite being 100% accurate.
2. **Perplexity (Language Modeling Loss)**: Measures how well the model predicts the next token on a general text corpus. A model can have low perplexity while confidently generating complete factual hallucinations.
3. **The Correct Metrics for RAG**:
   * **Context Precision**: What proportion of retrieved chunks are relevant?
   * **Context Recall**: Were all facts necessary to answer the question retrieved?
   * **Faithfulness**: Are all claims in the answer supported by the retrieved context?
   * **Answer Relevance**: Does the generated answer directly address the user's inquiry?
""")

metric_fit_df = pd.DataFrame([
    {"Application Domain": "RAG Factuality & Grounding", "Flawed Metric (Avoid)": "BLEU / ROUGE / Perplexity", "Correct Metric (Use)": "Context Faithfulness (LLM Judge Rubric)", "Why": "Checks factual support, not superficial n-gram overlap"},
    {"Application Domain": "Retrieval Quality", "Flawed Metric (Avoid)": "Raw Cosine Distance", "Correct Metric (Use)": "MRR, NDCG@k, Context Recall", "Why": "Measures whether ground-truth documents appear in top ranks"},
    {"Application Domain": "Structured JSON Extraction", "Flawed Metric (Avoid)": "Exact String Match", "Correct Metric (Use)": "Pydantic Schema Validation + Field Semantic F1", "Why": "Validates schema types and semantic fields, ignoring whitespace"},
    {"Application Domain": "Agent Tool Use", "Flawed Metric (Avoid)": "Final Output Accuracy", "Correct Metric (Use)": "Tool Call Precision & Trajectory Efficiency", "Why": "Detects unnecessary, redundant, or incorrect tool invocations"}
])
st.dataframe(metric_fit_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern production teams use the RAG Triad framework (Ragas / TruLens / LangSmith Evaluators):
Retrieval Relevance, Context Faithfulness, and Answer Relevance. Surface string metrics (BLEU) are strictly reserved 
for deterministic localization tasks (e.g., verifying exact Spanish-to-French translations).
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Evaluation Bias and Flaw Diagnostic Engine (EvaluationMistakeDiagnosticEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the Evaluation Bias Diagnostic Engine. 
It simulates how common evaluation flaws (verbosity bias, self-enhancement bias, weak rubrics, metric mismatches) 
distort measurements away from true underlying model quality.
""")

st.code('''
class EvaluationMistakeDiagnosticEngine:
    """
    Diagnostic & Calibration Engine for LLM Evaluation Mistakes in LangSmith.
    Identifies and models the 5 primary benchmark anti-patterns:
    1. Bad Datasets (Contamination, synthetic monoculture, zero negative edge cases)
    2. Weak Criteria (Vague subjective rubrics without concrete operational boundaries)
    3. LLM Judge Over-Reliance (Verbosity bias, self-enhancement bias, position bias)
    4. Ignoring Human Evaluation (No human annotation queues, blind trust in automated scores)
    5. Wrong Metrics (Using BLEU/Perplexity for RAG instead of Faithfulness and Retrieval Precision)
    """
    def __init__(self):
        self.simulation_records: List[Dict[str, Any]] = []

    def simulate_evaluation_bias(
        self,
        candidate_length_words: int,
        is_self_model_family: bool,
        dataset_synthetic_percent: int,
        metric_choice: str,
        rubric_clarity: str
    ) -> Dict[str, Any]:
        """
        Simulates the score distortions introduced by evaluation flaws and anti-patterns.
        """
        # True Underlying Grounded Quality (Objective ground truth = 0.75)
        true_quality = 0.75

        # Bias 1: Verbosity / Length Inflation Bias
        length_inflation = min(0.18, max(-0.10, (candidate_length_words - 50) * 0.0008))

        # Bias 2: Self-Enhancement / Model Family Bias
        family_bias = 0.08 if is_self_model_family else 0.0

        # Bias 3: Weak Rubric Variance
        rubric_penalty = 0.0
        if rubric_clarity == "Vague ('Rate 1-5')":
            rubric_variance_noise = 0.12
            rubric_penalty = -0.05
        elif rubric_clarity == "Moderate":
            rubric_variance_noise = 0.04
        else:  # Strict Binary Multi-Criterion Rubric
            rubric_variance_noise = 0.01

        # Bias 4: Metric Mismatch Distortion
        metric_distortion = 0.0
        if metric_choice == "BLEU / ROUGE (Surface N-Gram)":
            metric_distortion = -0.22
        elif metric_choice == "Perplexity":
            metric_distortion = -0.15

        # Computed Distorted Score vs True Ground Truth
        distorted_score = round(
            min(1.0, max(0.0, true_quality + length_inflation + family_bias + metric_distortion + rubric_variance_noise)),
            3
        )
        measurement_error = round(distorted_score - true_quality, 3)

        flaws_detected = []
        if dataset_synthetic_percent > 85:
            flaws_detected.append("Synthetic Monoculture: Benchmark is > 85% synthetic data; lacks production complexity.")
        if candidate_length_words > 200 and length_inflation > 0.10:
            flaws_detected.append("Verbosity Bias: Candidate received inflated score primarily due to word count.")
        if is_self_model_family:
            flaws_detected.append("Self-Enhancement Bias: Judge and candidate belong to same model family.")
        if "BLEU" in metric_choice or "Perplexity" in metric_choice:
            flaws_detected.append("Wrong Metric: Surface n-gram metric misrepresents semantic RAG factuality.")
        if rubric_clarity.startswith("Vague"):
            flaws_detected.append("Weak Rubric: Subjective grading rubric causes uninterpretable scores with high variance.")

        return {
            "true_quality_score": true_quality,
            "distorted_measured_score": distorted_score,
            "measurement_error": measurement_error,
            "flaws_detected": flaws_detected,
            "bias_breakdown": {
                "verbosity_inflation": round(length_inflation, 3),
                "family_bias": round(family_bias, 3),
                "metric_distortion": round(metric_distortion, 3),
                "rubric_variance": round(rubric_variance_noise, 3)
            },
            "status": "CALIBRATED" if abs(measurement_error) <= 0.05 else "BIASED_MEASUREMENT"
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Evaluation Flaw Simulator and Metric Bias Calibrator", divider="rainbow")

st.markdown("""
Experiment with evaluation configuration flaws: simulate verbosity bias, self-enhancement bias, 
weak rubrics, and metric mismatches to see how measured benchmark scores diverge from true underlying quality.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Evaluation Configuration Controls")
    cand_len = st.slider(
        "Candidate Response Length (Words)",
        min_value=20,
        max_value=350,
        value=220,
        step=10,
        help="Higher lengths trigger verbosity bias in uncalibrated LLM judges."
    )
    same_family_toggle = st.toggle(
        "Same Model Family (Judge & Candidate)",
        value=True,
        help="Simulates GPT-4 evaluating GPT-4, triggering self-enhancement bias."
    )
    synth_pct = st.slider(
        "Dataset Synthetic Content (%)",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
        help="Values > 85% create a synthetic monoculture echo chamber."
    )
    metric_select = st.selectbox(
        "Primary Evaluation Metric",
        [
            "Context Faithfulness (Multi-Criterion LLM Judge)",
            "BLEU / ROUGE (Surface N-Gram)",
            "Perplexity"
        ],
        index=0
    )
    rubric_select = st.selectbox(
        "Rubric Formulation Quality",
        [
            "Strict Multi-Criterion Binary Rubric",
            "Moderate Rubric",
            "Vague ('Rate 1-5')"
        ],
        index=0
    )

with wb_col2:
    st.markdown("### Live Benchmark Distortion & Measurement Error Report")
    bias_report = st.session_state.eval_mistakes_engine.simulate_evaluation_bias(
        candidate_length_words=cand_len,
        is_self_model_family=same_family_toggle,
        dataset_synthetic_percent=synth_pct,
        metric_choice=metric_select,
        rubric_clarity=rubric_select
    )

    if bias_report["status"] == "CALIBRATED":
        st.success(f"**STATUS: {bias_report['status']}** — Measurement error ({bias_report['measurement_error']:+.3f}) is within acceptable calibration bounds.")
    else:
        st.error(f"**STATUS: {bias_report['status']}** — Measurement error ({bias_report['measurement_error']:+.3f}) severely distorts benchmark reality!")

    # Metric Comparison
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("True Model Quality", f"{bias_report['true_quality_score']:.2f}")
    with c2:
        st.metric("Measured Benchmark Score", f"{bias_report['distorted_measured_score']:.3f}")
    with c3:
        st.metric(
            "Measurement Distortion",
            f"{bias_report['measurement_error']:+.3f}",
            delta_color="inverse"
        )

    # Detected Flaws Breakdown
    if bias_report["flaws_detected"]:
        st.markdown("#### Active Evaluation Anti-Patterns")
        for flaw in bias_report["flaws_detected"]:
            st.warning(f"⚠️ {flaw}")

    # Bias Breakdown Grid
    st.markdown("#### Cumulative Bias Decomposition")
    bb = bias_report["bias_breakdown"]
    decomp_df = pd.DataFrame([
        {"Bias Dimension": "Verbosity / Length Inflation", "Distortion Delta": f"{bb['verbosity_inflation']:+.3f}", "Impact": "Inflates wordy responses"},
        {"Bias Dimension": "Self-Enhancement Bias", "Distortion Delta": f"{bb['family_bias']:+.3f}", "Impact": "Favors same-family phrasing"},
        {"Bias Dimension": "Metric Mismatch Distortion", "Distortion Delta": f"{bb['metric_distortion']:+.3f}", "Impact": "Penalizes valid paraphrasing"},
        {"Bias Dimension": "Rubric Noise / Variance", "Distortion Delta": f"{bb['rubric_variance']:+.3f}", "Impact": "Stochastic scoring inconsistency"}
    ])
    st.dataframe(decomp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

q_exp5 = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with q_exp5:
    st.markdown("""
    **Q1: Why is BLEU or ROUGE an inappropriate metric for evaluating RAG answer factuality?**  
    *Answer:* BLEU and ROUGE measure exact surface-level n-gram lexical overlap against a reference text. In generative AI, there are dozens of semantically identical ways to phrase an answer using different vocabulary and sentence structures. BLEU severely penalizes correct paraphrasing while ignoring whether an answer contains fabricated claims. Factuality in RAG requires semantic entailment evaluation (Context Faithfulness) via calibrated LLM judges or natural language inference (NLI) models.
    """)

    st.markdown("""
    **Q2: What is Verbosity Bias in LLM judges, and how do you mitigate it?**  
    *Answer:* Verbosity bias is the systematic tendency of LLM judges to assign higher scores to longer, more elaborate responses, even when the extra length consists of unhelpful filler or subtle hallucinations. Mitigation strategies include: 1) Instructing the judge prompt to explicitly penalize unnecessary length; 2) Evaluating length-normalized conciseness; 3) Standardizing candidate outputs by summarizing them before evaluation; and 4) Using concise, claim-by-claim binary extraction rubrics.
    """)

    st.markdown("""
    **Q3: How does a synthetic monoculture dataset degrade model evaluation reliability?**  
    *Answer:* A dataset generated purely by prompting an LLM (e.g., GPT-4o) inherits the stylistic and cognitive assumptions of that generator. Synthetic queries are unnaturally articulate, grammatically complete, and free of typos, missing the colloquial, messy, and adversarial reality of human users. Models that achieve 98% pass rates on synthetic benchmarks frequently fail in production when real users ask ambiguous, incomplete, or out-of-scope questions.
    """)

    st.markdown("""
    **Q4: How do you design an evaluation rubric that minimizes judge variance across runs?**  
    *Answer:* Replace subjective Likert scales (1 to 5) with unambiguous binary checklists. For each criterion (e.g., faithfulness, question answering completeness, tone), provide concrete operational definitions: *"Does claim X appear in context Y? (1 or 0)"*. Enforce structured JSON outputs where the judge must extract individual claims before outputting a score, and pin `temperature=0` on the judge model.
    """)

    st.markdown("""
    **Q5: What role do LangSmith Annotation Queues play in calibrating automated evaluators?**  
    *Answer:* Annotation queues bridge the gap between automated scoring and human ground truth. Human Subject Matter Experts (SMEs) review a sampled queue of ambiguous or disputed traces, providing gold-standard scores. Measuring inter-rater reliability (Cohen's Kappa) between human annotators and automated LLM judges allows teams to detect judge drift and refine evaluation prompts continuously.
    """)

st.markdown("### Production Traps and Anti-Patterns")

tc_a1, tc_a2 = st.columns(2)

with tc_a1:
    st.error("""
    **Trap 1: The 'Rate 1 to 5' Subjective Rubric**
    * **Failure**: Prompting an LLM judge: *"Rate the helpfulness of this response from 1 to 5."*
    * **Consequence**: The judge outputs arbitrary numbers with massive variance between runs, making score comparisons across releases meaningless.
    * **Remedy**: Use multi-criterion binary checklists with structured claim extraction and explicit scoring rules.
    """)

    st.error("""
    **Trap 2: Self-Judging Model Architectures**
    * **Failure**: Using GPT-4o as the sole judge to evaluate GPT-4o fine-tunes against Claude 3.5 competitors.
    * **Consequence**: Self-enhancement bias awards 5-10% higher scores to GPT-4o outputs simply because the judge recognizes its own style.
    * **Remedy**: Use cross-model judge ensembles (e.g., combining Claude and GPT judges) or objective programmatic heuristics.
    """)

with tc_a2:
    st.warning("""
    **Trap 3: Zero Human Review Calibration**
    * **Failure**: Running automated evaluations for months without human domain expert verification.
    * **Consequence**: Automated scores remain high while real paying customers complain of declining answer quality, leading to churn.
    * **Remedy**: Hold regular human review calibration sessions using LangSmith Annotation Queues.
    """)

    st.warning("""
    **Trap 4: Evaluating Without Unanswerable Edge Cases**
    * **Failure**: Only testing queries where the ground truth is present in the database.
    * **Consequence**: The system is never penalized for hallucinating on missing context, hiding critical production failure modes.
    * **Remedy**: Include 15-20% unanswerable queries in the golden benchmark requiring polite refusals.
    """)

st.markdown("### Core Technical Flashcards")

fc_c1, fc_c2 = st.columns(2)

with fc_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is Verbosity Bias in LLM-as-a-judge evaluations?")
        st.markdown("**Back**: The systematic tendency of LLM judges to award higher scores to wordier, longer answers regardless of factual accuracy.")

    with fc_c1:
        with st.container(border=True):
            st.markdown("**Front**: Why is BLEU ineffective for evaluating RAG applications?")
            st.markdown("**Back**: It measures exact lexical n-gram overlap, penalizing valid paraphrasing while ignoring whether the answer is grounded in retrieved facts.")

with fc_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is the 'Synthetic Monoculture' benchmark flaw?")
        st.markdown("**Back**: Evaluating models exclusively on LLM-generated test cases that lack the typos, slang, and ambiguity of real production traffic.")

    with fc_c2:
        with st.container(border=True):
            st.markdown("**Front**: How do binary rubrics improve LLM judge consistency?")
            st.markdown("**Back**: By replacing subjective 1-5 scales with explicit binary checks (e.g., 'Is every claim supported? 1 or 0'), drastically reducing score variance.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of avoiding Common LangSmith Evaluation Mistakes:")

em_chk1 = st.checkbox("I know how to identify and eliminate synthetic monoculture in evaluation benchmark datasets.", value=True)
em_chk2 = st.checkbox("I can replace vague 1-5 Likert scales with objective, multi-criterion binary rubrics.", value=True)
em_chk3 = st.checkbox("I understand the mechanisms of verbosity bias, self-enhancement bias, and positional bias in LLM judges.", value=True)
em_chk4 = st.checkbox("I know how to implement human-in-the-loop review calibration using LangSmith Annotation Queues.", value=True)
em_chk5 = st.checkbox("I understand why BLEU and Perplexity fail for RAG and know which domain-aligned metrics to use instead.", value=True)
em_chk6 = st.checkbox("I can implement an automated evaluation bias diagnostic simulator to calibrate benchmark measurements.", value=True)

if em_chk1 and em_chk2 and em_chk3 and em_chk4 and em_chk5 and em_chk6:
    st.success("Mastery Certified: You have mastered diagnosing and eliminating common LangSmith evaluation mistakes, judge biases, and benchmark flaws.")
