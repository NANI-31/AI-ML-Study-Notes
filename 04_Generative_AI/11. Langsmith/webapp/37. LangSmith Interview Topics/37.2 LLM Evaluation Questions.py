import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.2 LLM Evaluation Questions - LangSmith Curriculum",
    page_icon="⚖️",
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
    .eval-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .score-badge {
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
# MODULE-SCOPE SIMULATOR & EVALUATION WORKBENCH ENGINE
# ==============================================================================
class LLMEvaluationEngine:
    """
    Simulates production LLM evaluation harnesses, RAG triad metrics,
    agent trajectory tracking, and prompt A/B testing.
    """
    def __init__(self):
        self.rag_benchmarks = [
            {
                "query": "What is the timeout threshold for LangSmith batch export?",
                "retrieved_context": "LangSmith batch export operations timeout after 300 seconds if the data payload exceeds 500MB.",
                "generated_answer": "The timeout threshold for LangSmith batch export is 300 seconds.",
                "ground_truth": "LangSmith batch export times out after 300 seconds for payloads over 500MB.",
                "context_relevance": 0.95,
                "groundedness": 1.00,
                "answer_relevance": 0.98,
                "hallucination_detected": False
            },
            {
                "query": "How do you configure SQLite checkpointer in LangGraph?",
                "retrieved_context": "PostgreSQL checkpointer requires connection pooling via psycopg. LangGraph supports durable persistence.",
                "generated_answer": "You configure SQLite checkpointer by passing a connection string to SqliteSaver.from_conn_string(':memory:').",
                "ground_truth": "Use SqliteSaver.from_conn_string('checkpoints.db') to persist graph state.",
                "context_relevance": 0.30,
                "groundedness": 0.20,
                "answer_relevance": 0.85,
                "hallucination_detected": True
            },
            {
                "query": "Can LangSmith traces be filtered by tags in the web UI?",
                "retrieved_context": "LangSmith web console allows filtering traces by run name, latency, tags, and feedback scores.",
                "generated_answer": "Yes, you can filter traces by tags, run names, latency, and feedback scores directly in the UI.",
                "ground_truth": "Yes, tags are first-class filter criteria in the LangSmith run browser.",
                "context_relevance": 0.92,
                "groundedness": 0.99,
                "answer_relevance": 0.96,
                "hallucination_detected": False
            }
        ]

        self.prompt_ab_tests = [
            {"candidate": "Prompt A (Concise Zero-Shot)", "win_rate": 0.44, "avg_latency_ms": 620, "token_cost_usd": 0.0012, "faithfulness": 0.88},
            {"candidate": "Prompt B (Few-Shot CoT with Rubric)", "win_rate": 0.56, "avg_latency_ms": 1180, "token_cost_usd": 0.0028, "faithfulness": 0.97}
        ]

    def evaluate_rag_sample(self, context_rel: float, grounded: float, answer_rel: float) -> Dict[str, Any]:
        harmonic_mean = 3 / (1/max(context_rel, 0.001) + 1/max(grounded, 0.001) + 1/max(answer_rel, 0.001))
        passed = (context_rel >= 0.70) and (grounded >= 0.80) and (answer_rel >= 0.75)
        return {
            "composite_score": round(harmonic_mean, 3),
            "passed": passed,
            "status": "PASSED" if passed else "FAILED_THRESHOLD"
        }

    def simulate_judge_scoring(self, answer: str, context: str, criteria: str) -> Dict[str, Any]:
        length_words = len(answer.split())
        words_in_context = sum(1 for w in answer.split() if w.lower() in context.lower())
        overlap_ratio = words_in_context / max(length_words, 1)

        faithfulness = min(round(overlap_ratio * 1.2, 2), 1.0)
        coherence = 0.92 if length_words > 5 else 0.45

        return {
            "criteria": criteria,
            "faithfulness_score": faithfulness,
            "coherence_score": coherence,
            "rationale": f"The model response contains {length_words} words with an observed lexical overlap of {round(overlap_ratio * 100, 1)} percent against context.",
            "verdict": "ACCEPTED" if faithfulness >= 0.75 else "REJECTED_HALLUCINATION"
        }

# Instantiate workbench simulator
eval_engine = LLMEvaluationEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: End-to-End LLM Evaluation, RAG Triad, & Agent Trajectory Auditing", divider="rainbow")

st.markdown("""
In enterprise generative AI applications, **LLM evaluation is an engineering discipline**, not an ad-hoc eyeball test.
LangSmith sits at the center of the evaluation lifecycle, transforming raw model outputs into structured, repeatable, 
and statistically rigorous quality benchmarks across automated LLM-as-a-judge pipelines, RAG triad assessments, 
agent trajectory audits, and CI/CD regression suites.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LLM_Evaluation_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_dataset {
            label = "Golden Benchmark Dataset Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            golden_ds [label="Versioned Golden Dataset\\n(Inputs, Ground Truths, Metadata)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            example_splits [label="Dataset Splits\\n(train, val, test, edge-cases)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_target {
            label = "Target Application Execution Under Test";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            app_runner [label="Target Application Pipeline\\n(RAG, LangGraph Agent, Prompt Chain)", fillcolor="#1E1B4B", fontcolor="#A78BFA", color="#818CF8"];
            trace_collector [label="LangSmith Tracing Engine\\n(Captures Inputs, Intermediates, Outputs)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_evaluators {
            label = "Multi-Tier Evaluator Harness";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            eval_deterministic [label="Tier 1: Deterministic Evaluators\\n(Exact Match, Regex, JSON Schema, BLEU/ROUGE)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            eval_rag_triad [label="Tier 2: RAG Triad Evaluators\\n(Context Relevance, Groundedness, Answer Relevance)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            eval_judge [label="Tier 3: LLM-as-a-Judge\\n(Rubric Scoring, Pairwise Preference, CoT Rationale)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            eval_agent [label="Tier 4: Agent Trajectory Evaluator\\n(Tool Call Order, Tool Arguments, State Invariants)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
        }

        subgraph cluster_results {
            label = "LangSmith Experiments & CI/CD Regression Gates";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            experiment_run [label="LangSmith Experiment Record\\n(Aggregated Metrics, Feedback Scores, Spans)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            ci_gate [label="CI/CD Quality Gate\\n(Pass/Fail Threshold Check, PR Blocking)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        }

        golden_ds -> example_splits [label="slices"];
        example_splits -> app_runner [label="drives benchmark runs"];
        app_runner -> trace_collector [label="records execution DAG"];
        trace_collector -> eval_deterministic [label="run inputs/outputs"];
        trace_collector -> eval_rag_triad [label="contexts + output"];
        trace_collector -> eval_judge [label="prompt + response"];
        trace_collector -> eval_agent [label="tool trace sequence"];

        eval_deterministic -> experiment_run [label="scores (0.0 - 1.0)"];
        eval_rag_triad -> experiment_run [label="triad vectors"];
        eval_judge -> experiment_run [label="rubric grades"];
        eval_agent -> experiment_run [label="trajectory accuracy"];

        experiment_run -> ci_gate [label="evaluates delta against baseline"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Key Architectural Takeaway:</strong> In LangSmith, evaluation is decoupled from production code. 
    A test harness takes a versioned dataset, feeds inputs through your target application, traces all intermediate spans, 
    and applies a battery of evaluators in parallel. Results are saved as an immutable <strong>Experiment</strong> linked to the dataset.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: WHAT IS LLM-AS-A-JUDGE?
# ==============================================================================
st.header("2. What is LLM-as-a-Judge?: Rubric-Based Scoring, Pairwise Comparisons, and Bias Mitigation", divider="rainbow")

st.markdown("""
**LLM-as-a-judge** is an evaluation methodology where a capable language model (e.g., GPT-4o, Claude 3.5 Sonnet) 
is used to evaluate the outputs of other models. Traditional NLP metrics such as BLEU, ROUGE, and METEOR measure exact 
token overlaps and fail to capture semantic nuances, logical coherence, and factual accuracy.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Primary Evaluation Modes
    1. **Single-Answer Rubric Grading**:
       The judge receives the input query, generated answer, and reference criteria. It assigns a score on a defined scale (e.g., 1–5 or 0.0–1.0) along with a step-by-step chain-of-thought rationale.
    2. **Pairwise Comparison (A/B Arena)**:
       The judge evaluates two candidate model outputs side-by-side for the same prompt, selecting which is superior or declaring a tie.
    3. **Reference-Guided vs Reference-Free**:
       - *Reference-Guided*: Compares the answer against an authoritative ground truth.
       - *Reference-Free*: Evaluates internal coherence, tone, formatting, and safety without a ground truth.
    """)

with col2:
    st.markdown("""
    ### Inherent Biases and Engineering Mitigations
    - **Position Bias**: Pairwise judges tend to favor the first option presented.
      *Mitigation*: Evaluate twice with swapped order (`(A, B)` and `(B, A)`), averaging results or requiring concordance.
    - **Verbosity Bias**: Judges naturally favor longer, more elaborate responses even when low in substance.
      *Mitigation*: Explicitly penalize verbosity in the evaluation prompt rubric and normalize by length.
    - **Self-Enhancement Bias**: Models often prefer answers generated by themselves or their family.
      *Mitigation*: Use cross-model evaluation (e.g., evaluate Claude with GPT-4o, and GPT-4o with Claude).
    """)

# Graphviz for LLM Judge Flow
judge_flow_dot = """
digraph JudgeFlow {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4, fontname="Helvetica"];
    node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10];

    subgraph cluster_inputs {
        label = "Judge Prompt Composition";
        style = "filled,rounded";
        fillcolor = "#1E293B";
        color = "#64748B";
        fontcolor = "#F8FAFC";

        user_query [label="Original User Query", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        candidate_ans [label="Candidate LLM Response", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        rubric_spec [label="Evaluation Rubric & Examples", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    }

    judge_llm [label="Judge LLM (e.g., GPT-4o / Claude 3.5)\\nForces Chain-of-Thought Reasoning First", fillcolor="#1E1B4B", fontcolor="#A78BFA", color="#818CF8"];
    structured_output [label="Structured Scorecard\\n{score: float, reasoning: str, verdict: bool}", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];

    user_query -> judge_llm;
    candidate_ans -> judge_llm;
    rubric_spec -> judge_llm;
    judge_llm -> structured_output [label="JSON Mode / Tool Call"];
}
"""
st.graphviz_chart(judge_flow_dot, use_container_width=True)

# ==============================================================================
# SECTION 3: HOW DO YOU EVALUATE RAG?
# ==============================================================================
st.header("3. How to Evaluate RAG: The RAG Triad (Context Relevance, Groundedness, Answer Relevance)", divider="rainbow")

st.markdown("""
Evaluating Retrieval-Augmented Generation requires breaking the system into its two foundational sub-systems:
the **Retriever** and the **Generator**. Evaluating only the final answer masks whether errors originated from 
irrelevant document retrieval or generator hallucination.
""")

st.markdown("""
### The RAG Triad Framework
The RAG Triad decomposes RAG evaluation into three orthogonal, non-overlapping dimensions:
""")

triad_cols = st.columns(3)
with triad_cols[0]:
    st.markdown("""
    <div class="metric-card">
        <h4>1. Context Relevance</h4>
        <span class="eval-badge">RETRIEVER METRIC</span>
        <p style="margin-top: 8px;"><strong>Question:</strong> Did the retriever fetch chunks relevant to the user query?</p>
        <p><strong>Formula:</strong> Relevant Chunks / Total Retrieved Chunks</p>
        <p style="color: #94A3B8; font-size: 12px;">Prevents noise injection and context window pollution.</p>
    </div>
    """, unsafe_allow_html=True)

with triad_cols[1]:
    st.markdown("""
    <div class="metric-card">
        <h4>2. Groundedness (Faithfulness)</h4>
        <span class="eval-badge">GENERATOR METRIC</span>
        <p style="margin-top: 8px;"><strong>Question:</strong> Is the answer strictly derived from retrieved context?</p>
        <p><strong>Formula:</strong> Verified Claims in Context / Total Claims</p>
        <p style="color: #94A3B8; font-size: 12px;">Detects factual hallucinations and ungrounded extrapolation.</p>
    </div>
    """, unsafe_allow_html=True)

with triad_cols[2]:
    st.markdown("""
    <div class="metric-card">
        <h4>3. Answer Relevance</h4>
        <span class="eval-badge">END-TO-END METRIC</span>
        <p style="margin-top: 8px;"><strong>Question:</strong> Does the answer directly address the user's inquiry?</p>
        <p><strong>Formula:</strong> Semantic similarity between query and generated answer</p>
        <p style="color: #94A3B8; font-size: 12px;">Catches evasion, irrelevant truthful statements, and topic drift.</p>
    </div>
    """, unsafe_allow_html=True)

rag_table_df = pd.DataFrame([
    {"Failure Mode": "Low Context Relevance, High Groundedness", "Symptom": "Model correctly answers based on retrieved docs, but retrieved docs are wrong.", "Root Cause": "Vector similarity threshold too low, poor chunking, or bad embeddings.", "Fix": "Tune chunk sizes, use hybrid search (BM25 + Dense) with Reciprocal Rank Fusion."},
    {"Failure Mode": "High Context Relevance, Low Groundedness", "Symptom": "Retrieved docs contain exact facts, but model hallucinates different numbers.", "Root Cause": "LLM ignored context or over-relied on parametric memory.", "Fix": "Tighten system prompt: 'Answer ONLY based on context. Say I do not know if missing.'"},
    {"Failure Mode": "High Groundedness, Low Answer Relevance", "Symptom": "Model outputs true statements from context, but completely ignores user's question.", "Root Cause": "Prompt fails to emphasize user intent or generates tangential facts.", "Fix": "Refine query reformulation and prompt instruction hierarchy."}
])
st.dataframe(rag_table_df, use_container_width=True)

# ==============================================================================
# SECTION 4: HOW DO YOU EVALUATE AGENTS?
# ==============================================================================
st.header("4. How to Evaluate Agents: Trajectory Tracking, Tool Selection Accuracy, and Goal Completion", divider="rainbow")

st.markdown("""
Unlike static single-turn chains, **AI Agents are dynamic state machines** that reason, plan, invoke tools, 
observe environment feedback, and iterate over multiple cycles. Evaluating an agent by only checking its final text response 
is inadequate because an agent might reach the right answer through an invalid or expensive path.
""")

st.markdown("""
### Dimensions of Agent Trajectory Evaluation
1. **Tool Selection Accuracy**: Did the agent invoke the correct tool for the specific sub-task? (e.g., calling `calculate_tax` instead of doing mental math).
2. **Tool Argument Validity**: Were the arguments passed to the tool schema-compliant and semantically accurate?
3. **Trajectory Efficiency (Step Count)**: Did the agent take the optimal path, or did it wander into redundant tool calls and cyclic loops?
4. **State Invariant Integrity**: Across LangGraph state transitions, did the agent maintain required schema invariants without dropping state keys?
5. **Goal Completion**: Did the agent successfully terminate with a valid resolution, or did it hit recursion limits and crash?
""")

agent_flow_dot = """
digraph AgentTrajectory {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4, fontname="Helvetica"];
    node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10];

    subgraph cluster_trajectory {
        label = "Trace Execution DAG Under Audit";
        style = "filled,rounded";
        fillcolor = "#1E293B";
        color = "#64748B";
        fontcolor = "#F8FAFC";

        step1 [label="Step 1: Plan & Tool Selection\\nCheck: Correct Tool Chosen?", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        step2 [label="Step 2: Tool Invocation\\nCheck: Valid Schema & Arguments?", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        step3 [label="Step 3: State Reduction\\nCheck: State Reducer Integrity?", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        step4 [label="Step 4: Goal Termination\\nCheck: Complete or Recursion Limit?", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    }

    step1 -> step2 [label="correct tool call"];
    step2 -> step3 [label="observation returned"];
    step3 -> step4 [label="final state reached"];
}
"""
st.graphviz_chart(agent_flow_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: HOW DO YOU MEASURE HALLUCINATION?
# ==============================================================================
st.header("5. How to Measure Hallucination: Factual Consistency Scoring and Source Verification", divider="rainbow")

st.markdown("""
Hallucination occurs when an LLM generates statements that are unfaithful to provided source documents 
(extrinsic hallucination) or factually contradictory to reality (intrinsic hallucination).
In LangSmith pipelines, measuring hallucination requires objective, repeatable metrics.
""")

st.markdown("""
### Three Core Methodologies for Hallucination Detection
1. **Natural Language Inference (NLI) Entailment**:
   Break the generated answer into atomic propositional claims (claim_1, claim_2, ..., claim_n). For each claim, use an NLI model (e.g., RoBERTa-large-MNLI) or an LLM judge to verify if the retrieved context **entails**, **contradicts**, or is **neutral** to the claim.
   Faithfulness Score = Sum of Entailed Claims / Total Claims Generated
2. **Back-Translation & Self-Consistency Verification**:
   Prompt the LLM to generate the questions that would yield its own generated claims. If the synthesized question deviates significantly from the original user prompt or context facts, a hallucination is flagged.
3. **Automated Source Attribution & Citation Matching**:
   Force the model to provide precise inline citations (e.g., [Doc 2, para 4]). The evaluator programmatically checks whether the cited span actually contains the factual assertion.
""")

# ==============================================================================
# SECTION 6: HOW DO YOU PERFORM REGRESSION TESTING?
# ==============================================================================
st.header("6. How to Perform Regression Testing: Versioned Golden Datasets and CI/CD Quality Gates", divider="rainbow")

st.markdown("""
In traditional software, regression tests verify that new code commits do not break existing unit tests. 
In LLM systems, code commits, prompt revisions, model upgrades, and retriever adjustments can silently degrade 
answer quality. **LangSmith regression testing automates quality verification in CI/CD pipelines.**
""")

st.markdown("""
### Enterprise Regression Testing Architecture
1. **Curate the Golden Benchmark**: Maintain a dataset of 100–500 validated input-output pairs covering typical queries, edge cases, adversarial prompts, and complex multi-hop queries.
2. **Execute Baseline Run**: Run the production pipeline against the dataset and store the experiment run in LangSmith as the reference baseline.
3. **Pull Request Quality Gate (CI Trigger)**:
   - On every pull request touching prompts or agent code, GitHub Actions executes `client.evaluate()`.
   - The candidate run produces composite scores for accuracy, latency, and cost.
4. **Delta Comparison & Automated Blocking**:
   The CI runner queries LangSmith API to compare the candidate experiment against the baseline:
   - If `candidate.accuracy < baseline.accuracy - 0.02` (accuracy drop > 2%): **BLOCK MERGE**.
   - If `candidate.latency_p95 > baseline.latency_p95 * 1.25` (latency increase > 25%): **WARN**.
   - If `candidate.cost > baseline.cost * 1.50` (cost inflation > 50%): **BLOCK MERGE**.
""")

ci_dot = """
digraph CI_Regression {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4, fontname="Helvetica"];
    node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10];

    pr_commit [label="PR Commit\\n(Prompt or Agent Code Change)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    ci_runner [label="GitHub Actions / CI Runner\\nRuns test_eval_regression.py", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
    langsmith_eval [label="LangSmith Evaluation Service\\nExecutes Candidate Experiment", fillcolor="#0F172A", fontcolor="#38BDF8"];
    delta_check [label="Delta Threshold Analyzer\\nCompares Candidate vs Baseline", fillcolor="#0F172A", fontcolor="#F8FAFC"];

    pr_gate_pass [label="PASS: Merge Approved\\n(No quality or cost regression)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
    pr_gate_fail [label="FAIL: PR Blocked\\n(Accuracy -3.4% below tolerance)", fillcolor="#7F1D1D", fontcolor="#F87171", color="#EF4444"];

    pr_commit -> ci_runner;
    ci_runner -> langsmith_eval;
    langsmith_eval -> delta_check;
    delta_check -> pr_gate_pass [label="within tolerance"];
    delta_check -> pr_gate_fail [label="regression detected"];
}
"""
st.graphviz_chart(ci_dot, use_container_width=True)

# ==============================================================================
# SECTION 7: HOW DO YOU COMPARE TWO PROMPTS?
# ==============================================================================
st.header("7. How to Compare Two Prompts: A/B Testing, Side-by-Side Experiments, and Statistical Superiority", divider="rainbow")

st.markdown("""
Prompt engineering must move beyond intuition. When optimizing a prompt for accuracy, token efficiency, 
or formatting fidelity, you must prove **statistical superiority** using side-by-side LangSmith experiments.
""")

st.markdown("""
### Steps for Scientific Prompt Comparison
1. **Hold Constant All Confounders**: Use the identical model checkpoint (e.g., `gpt-4o-2024-08-06`), temperature (`0.0`), top_p, and test dataset examples. The only independent variable is the prompt text.
2. **Execute Parallel Experiments**:
   - Run Experiment A with `prompt_v1`.
   - Run Experiment B with `prompt_v2`.
3. **Multi-Objective Comparison Matrix**:
   Evaluate on three competing dimensions:
   - *Quality Score* (Accuracy, Groundedness, Format Compliance)
   - *Latency* (Time to first token, total duration)
   - *Cost* (Prompt tokens, completion tokens)
4. **Statistical Significance Testing**:
   For sample sizes N > 50, perform a paired Wilcoxon signed-rank test or McNemar test. Do not declare Prompt B better if the delta is within random stochastic noise.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Automated RAG and Prompt Evaluation Pipeline with LangSmith Client", divider="rainbow")

st.markdown("""
Below is the complete, self-contained production Python script demonstrating how to use the modern 
`langsmith.Client` to create an evaluation dataset, define deterministic and LLM-as-a-judge evaluators, 
and execute `client.evaluate()` with quality threshold assertion.
""")

st.code('''import os
from typing import Dict, Any
from langsmith import Client
from langsmith.evaluation import evaluate, LangSmithValue

# Initialize LangSmith Client
# Requires: export LANGSMITH_API_KEY="lsv2_pt_..."
client = Client()

# 1. Define or Ensure Benchmark Golden Dataset
DATASET_NAME = "production-rag-quality-v1"

if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden evaluation dataset for customer support RAG pipeline."
    )
    # Seed benchmark examples
    client.create_examples(
        inputs=[
            {"question": "What is the return window for enterprise hardware?"},
            {"question": "How do I rotate an expired API key?"}
        ],
        outputs=[
            {"expected_answer": "Enterprise hardware has a 90-day return policy."},
            {"expected_answer": "Navigate to Settings > Security > API Keys and select Rotate Key."}
        ],
        dataset_id=dataset.id
    )

# 2. Define Target Application Pipeline Under Test
def target_rag_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates production RAG pipeline invocation.
    Retrieves context and synthesizes response.
    """
    question = inputs["question"]
    if "return window" in question.lower():
        return {
            "answer": "Enterprise hardware has a 90-day return policy from the delivery date.",
            "retrieved_context": ["Hardware SLA section 4.2: 90-day return policy applies to enterprise tiers."]
        }
    return {
        "answer": "Go to Settings > Security and click Rotate Key to generate a new secret.",
        "retrieved_context": ["Security guide: API key rotation is available under Settings > Security."]
    }

# 3. Define Custom Evaluators
def evaluate_exact_grounding(run, example) -> Dict[str, Any]:
    """
    Deterministic evaluator: verifies key claims exist in context.
    """
    answer = run.outputs.get("answer", "")
    contexts = run.outputs.get("retrieved_context", [])
    context_text = " ".join(contexts)

    # Simplified token overlap heuristic
    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(context_text.lower().split())
    overlap = len(ans_tokens.intersection(ctx_tokens)) / max(len(ans_tokens), 1)

    return {
        "key": "groundedness_score",
        "score": round(overlap, 2),
        "comment": f"Context overlap ratio: {round(overlap, 2)}"
    }

def evaluate_answer_accuracy(run, example) -> Dict[str, Any]:
    """
    Evaluates response against golden expected output.
    """
    prediction = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("expected_answer", "").lower()

    # Exact semantic match or inclusion
    score = 1.0 if ("90-day" in prediction and "90-day" in expected) or ("rotate" in prediction and "rotate" in expected) else 0.0
    return {
        "key": "correctness",
        "score": score
    }

# 4. Execute Programmatic LangSmith Evaluation
results = evaluate(
    target_rag_application,
    data=DATASET_NAME,
    evaluators=[evaluate_exact_grounding, evaluate_answer_accuracy],
    experiment_prefix="ci-rag-audit",
    max_concurrency=4
)

print(f"Evaluation completed successfully! LangSmith Experiment Name: {results.experiment_name}")
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: LLM-as-a-Judge Rubric Simulator & Evaluation Scorecard Sandbox", divider="rainbow")

st.markdown("""
Experiment with the simulator below to inspect how LLM-as-a-Judge evaluators assess 
different candidate responses against retrieved context and evaluation rubrics.
""")

wb_mode = st.selectbox(
    "Select Evaluation Workbench Tool:",
    ["1. RAG Triad Evaluator Simulator", "2. LLM-as-a-Judge Custom Rubric Tester", "3. Prompt A/B Test Comparator"]
)

if wb_mode == "1. RAG Triad Evaluator Simulator":
    st.subheader("RAG Triad Multi-Metric Scorecard")
    sample_idx = st.selectbox("Select Benchmark Test Case:", [0, 1, 2], format_func=lambda x: eval_engine.rag_benchmarks[x]["query"])
    active_sample = eval_engine.rag_benchmarks[sample_idx]

    st.markdown(f"**User Query:** `{active_sample['query']}`")
    st.info(f"**Retrieved Context:** {active_sample['retrieved_context']}")
    st.success(f"**Generated Answer:** {active_sample['generated_answer']}")
    st.markdown(f"**Ground Truth:** {active_sample['ground_truth']}")

    st.markdown("#### Adjust Measured Metrics:")
    c1, c2, c3 = st.columns(3)
    with c1:
        c_rel = st.slider("Context Relevance:", 0.0, 1.0, active_sample["context_relevance"], 0.05)
    with c2:
        c_gnd = st.slider("Groundedness (Faithfulness):", 0.0, 1.0, active_sample["groundedness"], 0.05)
    with c3:
        c_ans = st.slider("Answer Relevance:", 0.0, 1.0, active_sample["answer_relevance"], 0.05)

    eval_result = eval_engine.evaluate_rag_sample(c_rel, c_gnd, c_ans)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Composite Harmonic Score", f"{eval_result['composite_score']}")
    with res_col2:
        status_color = "green" if eval_result["passed"] else "red"
        st.markdown(f"**Evaluation Status:** <span style='color:{status_color}; font-weight:bold; font-size:18px;'>{eval_result['status']}</span>", unsafe_allow_html=True)

elif wb_mode == "2. LLM-as-a-Judge Custom Rubric Tester":
    st.subheader("Rubric Scoring & Rationale Generation")
    test_context = st.text_area("Source Context:", "LangSmith allows exporting traces up to 500MB via bulk export within 300 seconds.")
    test_answer = st.text_area("Candidate Model Answer:", "You can export up to 500MB of traces in under 300 seconds via LangSmith bulk export.")
    test_criteria = st.selectbox("Evaluation Criteria:", ["Faithfulness & Factual Grounding", "Completeness & Detail", "Safety & Compliance"])

    if st.button("Run Judge Evaluation"):
        judge_res = eval_engine.simulate_judge_scoring(test_answer, test_context, test_criteria)
        st.markdown(f"**Judge Verdict:** `{judge_res['verdict']}`")
        st.markdown(f"**Faithfulness Score:** `{judge_res['faithfulness_score']}` | **Coherence:** `{judge_res['coherence_score']}`")
        st.info(f"**Judge Rationale:** {judge_res['rationale']}")

else:
    st.subheader("Prompt A/B Side-by-Side Experiment Comparison")
    ab_df = pd.DataFrame(eval_engine.prompt_ab_tests)
    st.dataframe(ab_df, use_container_width=True)
    st.caption("Prompt B achieved +12% higher win-rate and +9% faithfulness at the cost of +560ms latency and higher token expenditure.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why is testing on raw string exact-match inadequate for production LLM evaluations?"):
    st.markdown("""
    **Answer:**  
    LLM outputs exhibit stochastic variance and semantic diversity. A model can produce completely correct answers 
    using different vocabulary, sentence structures, or formatting that fail string equality tests (`predicted == expected`). 
    Production systems require semantic embedding similarity, deterministic schema parsing, or LLM-as-a-judge rubric grading.
    """)

with st.expander("Q2: How does the RAG Triad isolate whether a failure originated in retrieval or generation?"):
    st.markdown("""
    **Answer:**  
    The RAG Triad decouples evaluation:
    - **Context Relevance** tests only the query-to-context relationship. If low, retrieval failed.
    - **Groundedness** tests only the context-to-answer relationship. If low despite high context relevance, the generator hallucinated.
    - **Answer Relevance** tests whether the final answer fulfilled the query.
    This pinpointing enables targeted optimization rather than blind prompt adjustments.
    """)

with st.expander("Q3: How do you mitigate position and verbosity bias when building LLM-as-a-judge evaluators?"):
    st.markdown("""
    **Answer:**  
    1. **Position Bias**: Run pairwise comparisons twice by swapping option positions (`[A, B]` and `[B, A]`), requiring unanimous agreement.
    2. **Verbosity Bias**: Include explicit brevity instructions in the judge prompt rubric, penalize unneeded verbosity, or enforce length normalization before scoring.
    3. **Chain of Thought**: Force the judge to output step-by-step reasoning before providing the final numeric grade.
    """)

with st.expander("Q4: What makes agent evaluation fundamentally harder than standard chain evaluation?"):
    st.markdown("""
    **Answer:**  
    Chains are linear DAGs with deterministic input/output pairs. Agents are cyclic, stateful systems with non-deterministic 
    multi-step trajectories. An agent can call tools multiple times, backtrack, recover from exceptions, or enter infinite loops. 
    Agent evaluation requires checking tool choice accuracy, parameter schemas, loop termination, step efficiency, and state invariants.
    """)

with st.expander("Q5: How do you implement automated PR quality gates with LangSmith in a CI/CD pipeline?"):
    st.markdown("""
    **Answer:**  
    In CI (e.g., GitHub Actions), trigger an evaluation script that runs candidate code against a versioned LangSmith dataset. 
    Use the LangSmith Client to fetch baseline experiment metrics, compute the score deltas, and fail the CI build 
    (exit code 1) if key metrics regress beyond defined tolerance bands (e.g., accuracy drops by > 2% or cost rises > 30%).
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Evaluating Without Frozen Reference Datasets (Moving Target Fallacy)**  
Evaluating against unversioned live production queries makes it impossible to compare model changes fairly. 
Always evaluate against an immutable, version-controlled Golden Dataset.
""")

st.error("""
**Trap 2: Circular Dependency: Using the Same Model Family as Judge and Candidate**  
Using GPT-4o-mini to evaluate GPT-4o-mini produces high self-preference bias and misses systemic blind spots. 
Use a strictly higher-tier model (e.g., GPT-4o or Claude 3.5 Sonnet) or cross-provider judges.
""")

st.warning("""
**Trap 3: Relying Solely on End-to-End Metrics and Ignoring Intermediate Spans**  
Checking only the final response hides inefficient 10-step agent loops or near-hallucinated retrieval failures. 
Trace and audit all intermediate spans.
""")

st.warning("""
**Trap 4: Evaluating on Unrepresentative Synthetic Data Only**  
Synthetic datasets generated by LLMs often lack realistic user typos, ambiguous grammar, and adversarial phrasing. 
Continuously backfill golden datasets with anonymized production failure traces.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>RAG Triad</h4>
        <p>A 3-part evaluation framework measuring Context Relevance, Groundedness (Faithfulness), and Answer Relevance to pinpoint RAG failures.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Pairwise Comparison</h4>
        <p>An evaluation method where an LLM judge evaluates two model responses side-by-side to determine which is superior, with swapped order mitigation.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Agent Trajectory</h4>
        <p>The sequence of thoughts, tool selections, tool arguments, and environment observations taken by an agent to solve a goal.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Regression Threshold</h4>
        <p>A mathematical limit in CI/CD pipelines defining the allowable performance degradation before a pull request is blocked.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can define the three pillars of the RAG Triad and explain how they isolate retrieval vs generation bugs.", value=True)
chk_2 = st.checkbox("I understand the mechanisms and mitigations for position, verbosity, and self-enhancement biases in LLM judges.", value=True)
chk_3 = st.checkbox("I know how to evaluate multi-step agent trajectories for tool accuracy, efficiency, and state invariants.", value=True)
chk_4 = st.checkbox("I can articulate how to measure factual hallucination using propositional claim decomposition and NLI entailment.", value=True)
chk_5 = st.checkbox("I understand how to design automated CI/CD regression gates using versioned golden datasets and delta thresholds.", value=True)
chk_6 = st.checkbox("I can implement a scientific A/B prompt experiment holding model checkpoints, temperature, and test datasets constant.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LLM Evaluation interview topics and architectural competencies verified! Ready for Production Questions.")
