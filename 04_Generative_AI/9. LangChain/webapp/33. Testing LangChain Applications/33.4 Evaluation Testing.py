import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="33.4 Evaluation Testing - Quantitative Benchmarking & RAG Triad",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for evaluation telemetry
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .score-badge {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class EvaluationBenchmarkingSimulator:
    """
    Simulates quantitative evaluation pipelines:
    RAG Triad scoring (Context Relevance, Faithfulness, Answer Relevance),
    LLM-as-a-judge rubrics, agent trajectory efficiency, and dataset benchmarking.
    """
    def __init__(self):
        self.benchmark_runs = {
            "Enterprise Policy RAG System": {
                "dataset_size": 250,
                "context_relevance": 0.92,
                "faithfulness": 0.96,
                "answer_relevance": 0.89,
                "agent_step_efficiency": 0.94,
                "human_satisfaction_rating": 4.6,
                "summary": "High factual grounding. Low retrieval dilution across 250 golden test items."
            },
            "Financial SQL Agent": {
                "dataset_size": 180,
                "context_relevance": 0.84,
                "faithfulness": 0.98,
                "answer_relevance": 0.91,
                "agent_step_efficiency": 0.82,
                "human_satisfaction_rating": 4.3,
                "summary": "100% SQL syntax pass rate. Slight step inflation due to exploratory schema calls."
            },
            "Customer Support Multi-Agent": {
                "dataset_size": 320,
                "context_relevance": 0.78,
                "faithfulness": 0.88,
                "answer_relevance": 0.85,
                "agent_step_efficiency": 0.75,
                "human_satisfaction_rating": 3.9,
                "summary": "Occasional repetitive routing steps observed in ticket escalation flow."
            }
        }

    def evaluate_benchmark(self, system_name: str) -> Dict[str, Any]:
        return self.benchmark_runs.get(system_name, self.benchmark_runs["Enterprise Policy RAG System"])


# --- LESSON HEADER ---
st.title("33.4 Evaluation Testing: Quantitative Benchmarking & RAG Triad")
st.markdown("""
Evaluation testing moves beyond binary pass/fail unit assertions to continuous, quantitative quality scoring.
Using automated frameworks (such as Ragas and LangSmith), teams measure system performance across the
**RAG Triad** (Context Relevance, Faithfulness, and Answer Relevance), evaluate multi-step agent trajectories,
deploy LLM-as-a-judge scoring rubrics, and calibrate metrics against human domain expert feedback.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Quantitative Evaluation & Benchmarking", divider="rainbow")
st.markdown("""
Modern evaluation pipelines run asynchronously against curated test datasets.
Queries are dispatched through the LangChain system, capturing retrieved contexts, tool steps, and final completions.
Evaluators (LLM-as-a-judge, semantic embedding models, and deterministic heuristics) compute multi-dimensional scores
that feed directly into continuous deployment quality gates.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EvaluationTesting_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_dataset {
            label = "Curated Evaluation Datasets";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            GoldenDataset [label="Golden Test Dataset\\n(Questions, Ground Truth Answers, Reference Contexts)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_system_under_test {
            label = "LangChain / LangGraph System Execution";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            SystemPipeline [label="LangChain Pipeline / Agent\\n(Executes Retrieval -> Reasoning -> Completion)", fillcolor="#1E293B", color="#60A5FA"];
            TelemetryTrace [label="Execution Telemetry & Trace\\n(Input, Retrieved Chunks, Tool Sequence, Output)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_eval_engine {
            label = "Multi-Dimensional Evaluation Engine (Ragas / LangSmith)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ContextRelEval [label="1. Context Relevance\\n(Is retrieved context concise and relevant to query?)", fillcolor="#0369A1", color="#38BDF8"];
            FaithfulnessEval [label="2. Groundedness / Faithfulness\\n(Is the answer derived strictly from retrieved context?)", fillcolor="#0369A1", color="#38BDF8"];
            AnswerRelEval [label="3. Answer Relevance\\n(Does the answer directly address the original prompt?)", fillcolor="#0369A1", color="#38BDF8"];
            TrajectoryEval [label="4. Agent Trajectory Efficiency\\n(Did agent take optimal path with minimal tool calls?)", fillcolor="#0369A1", color="#38BDF8"];
            JudgeModel [label="LLM-as-a-Judge Evaluator\\n(Rubric-guided scoring with reasoning rationale)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_governance {
            label = "Evaluation Dashboard & Quality Gates";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            QualityDashboard [label="Enterprise Benchmark Scorecard\\n(Pass threshold: Faithfulness >= 0.90, Quality Gates)", fillcolor="#065F46", color="#10B981"];
        }

        GoldenDataset -> SystemPipeline [label="Batch Evaluation Run"];
        SystemPipeline -> TelemetryTrace [label="Capture Execution Traces"];
        TelemetryTrace -> ContextRelEval [label="Evaluate Retrieval Chunks"];
        TelemetryTrace -> FaithfulnessEval [label="Evaluate Claim Grounding"];
        TelemetryTrace -> AnswerRelEval [label="Evaluate Final Answer"];
        TelemetryTrace -> TrajectoryEval [label="Evaluate Step Trajectory"];

        JudgeModel -> ContextRelEval [label="Score Rubric"];
        JudgeModel -> FaithfulnessEval [label="Score Rubric"];
        JudgeModel -> AnswerRelEval [label="Score Rubric"];

        ContextRelEval -> QualityDashboard [label="Metric Aggregation"];
        FaithfulnessEval -> QualityDashboard [label="Metric Aggregation"];
        AnswerRelEval -> QualityDashboard [label="Metric Aggregation"];
        TrajectoryEval -> QualityDashboard [label="Metric Aggregation"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: DATASET CREATION ---
st.header("2. Golden Dataset Creation & Curation: Synthetic Generation and Human Curation", divider="rainbow")
st.markdown("""
A high-quality evaluation pipeline is impossible without an authoritative golden dataset.
Enterprise datasets combine synthetic question-context-answer generation with human-expert curation,
ensuring comprehensive coverage across standard inquiries, edge cases, and adversarial questions.
""")

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    st.markdown("### Golden Dataset Schema")
    st.markdown("""
    - **`question`**: The real-world user inquiry.
    - **`ground_truth`**: Human-verified authoritative answer.
    - **`reference_contexts`**: Specific document passages that contain the ground truth facts.
    - **`metadata`**: Domain category, difficulty level, and source document identifiers.
    """)
with col_ds2:
    st.code("""
# Synthetic Dataset Generation with LangChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

synth_prompt = PromptTemplate.from_template(\"\"\"
You are a test dataset generator. Given the following document text,
generate 3 diverse user questions that can be answered strictly from this text,
along with the authoritative ground-truth answer for each.

Document:
{document_text}

Output format: JSON list of objects with 'question' and 'ground_truth' fields.
\"\"\")

synth_chain = synth_prompt | ChatOpenAI(model="gpt-4o")
    """, language="python")

# --- SECTION 3: AUTOMATED EVALUATION ---
st.header("3. Automated Evaluation & LLM-as-a-Judge: Criteria Rubrics and Scoring", divider="rainbow")
st.markdown("""
Automated evaluation uses a capable judge model (e.g. GPT-4o) guided by detailed scoring rubrics.
The judge analyzes the system's generated output against defined criteria (such as correctness, tone, or safety),
outputting a numerical score alongside an explicit chain-of-thought justification.
""")

st.code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class JudgeEvaluation(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1 to 5 quality score")
    reasoning: str = Field(..., description="Justification explaining the score based on the rubric")

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are an impartial judge evaluating AI responses.
Score 5: Completely accurate, fully grounded, perfectly answers the query.
Score 3: Partially correct or includes extraneous details.
Score 1: Hallucinates facts or fails to address the user inquiry.
Always provide explicit reasoning before deciding your score.\"\"\"),
    ("human", \"\"\"[Query]: {query}
[Context]: {context}
[AI Completion]: {completion}

Evaluate the response:\"\"\")
])

judge_chain = judge_prompt | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(JudgeEvaluation)
""", language="python")

# --- SECTION 4: HUMAN EVALUATION ---
st.header("4. Human Evaluation & Calibration: Feedback Loops and Likert Scales", divider="rainbow")
st.markdown("""
Automated metrics must be continuously calibrated against human domain experts.
Human-in-the-loop evaluation collects ratings on 5-point Likert scales (or thumbs up/down feedback),
computing Inter-Annotator Agreement (Cohen's Kappa) to identify ambiguous test cases and refine judge rubrics.
""")

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.markdown("### Human Calibration Protocol")
    st.markdown("""
    - **Blind Evaluation**: Reviewers grade model completions without knowing whether it was produced by the new candidate or baseline model.
    - **Cohen's Kappa / Fleiss' Kappa**: Measure agreement among human annotators to ensure guidelines are unambiguous.
    - **Judge Alignment**: Correlate LLM-as-a-judge scores with human scores (target Pearson correlation >= 0.80).
    """)
with col_h2:
    st.code("""
# Telemetry Feedback Capture in LangSmith
from langsmith import Client

client = Client()

def log_user_feedback(run_id: str, score: float, comment: str):
    \"\"\"Log explicit end-user or domain-expert feedback to LangSmith run.\"\"\"
    client.create_feedback(
        run_id=run_id,
        key="human_quality_score",
        score=score,  # e.g., 1.0 for positive, 0.0 for negative
        comment=comment
    )
    """, language="python")

# --- SECTION 5: AGENT TRAJECTORY TESTING ---
st.header("5. Agent Trajectory & Multi-Step Reasoning Evaluation: Step Efficiency and Tool Paths", divider="rainbow")
st.markdown("""
Evaluating multi-step agents requires assessing not only the final answer, but also the journey taken to reach it.
An agent that invokes 15 redundant tools before answering a question is inefficient and expensive.
Trajectory evaluation scores **Tool Selection Accuracy**, **Argument Precision**, and **Step Minimality**.
""")

st.code("""
def evaluate_agent_trajectory(executed_steps: list, optimal_steps: list) -> dict:
    \"\"\"
    Computes trajectory efficiency score based on optimal tool execution path.
    \"\"\"
    executed_tools = [step["tool"] for step in executed_steps]
    optimal_tools = [step["tool"] for step in optimal_steps]
    
    # Check if all necessary tools were called
    tools_recalled = set(optimal_tools).issubset(set(executed_tools))
    
    # Compute step inflation ratio
    redundancy_ratio = len(executed_steps) / max(len(optimal_steps), 1)
    efficiency_score = max(0.0, min(1.0, 1.0 - (redundancy_ratio - 1.0) * 0.2))
    
    return {
        "tools_recalled": tools_recalled,
        "step_count": len(executed_steps),
        "optimal_step_count": len(optimal_steps),
        "trajectory_efficiency": round(efficiency_score, 2)
    }
""", language="python")

# --- SECTION 6: RAG EVALUATION ---
st.header("6. RAG Triad & Multi-Dimensional Evaluation: Context Relevance, Faithfulness, and Answer Relevance", divider="rainbow")
st.markdown("""
The **RAG Triad** is the industry standard framework for diagnosing RAG pipeline failures:
1. **Context Relevance**: Did the retriever fetch chunks relevant to the user's question, without noise?
2. **Faithfulness (Groundedness)**: Can every single claim in the completion be derived directly from the retrieved context?
3. **Answer Relevance**: Does the final answer directly address the user's inquiry?
""")

col_rag1, col_rag2 = st.columns(2)
with col_rag1:
    st.markdown("### The Three Pillars of RAG")
    st.markdown("""
    - **Low Context Relevance**: Retriever failure. Top-k contains noisy, unrelated documents that dilute model attention.
    - **Low Faithfulness**: Generator failure. The model hallucinated facts not present in the retrieved chunks.
    - **Low Answer Relevance**: Alignment failure. The model produced a factual response, but answered a different question.
    """)
with col_rag2:
    st.code("""
# Ragas Metric Evaluation Paradigm
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# Construct evaluation dataset
eval_dataset = Dataset.from_dict({
    "question": ["What is our remote work policy?"],
    "contexts": [["Employees are allowed up to 3 remote days per week."]],
    "answer": ["Our policy permits up to 3 days of remote work each week."],
    "ground_truth": ["3 days remote work per week."]
})

# Run multi-dimensional RAG evaluation
results = evaluate(
    eval_dataset,
    metrics=[context_precision, faithfulness, answer_relevancy]
)
print(results)
    """, language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: End-to-End Enterprise Evaluation Pipeline with LangSmith", divider="rainbow")
st.markdown("""
Below is an automated evaluation harness using the LangSmith evaluation framework to test candidate chains
against a persistent dataset, generating quantitative scorecards for CI deployment gates.
""")

st.code("""
# eval_pipeline.py
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

client = Client()

# 1. Define candidate application chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions accurately using company context: {context}"),
    ("human", "{question}")
])
candidate_chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Define custom evaluator function
def evaluate_factual_accuracy(run, example):
    prediction = run.outputs["text"]
    reference = example.outputs["ground_truth"]
    
    # Scoring criteria
    is_match = reference.lower() in prediction.lower()
    return {"key": "exact_fact_match", "score": 1.0 if is_match else 0.0}

# 3. Execute automated evaluation across LangSmith dataset
experiment_results = evaluate(
    lambda inputs: candidate_chain.invoke(inputs),
    data="enterprise-qa-benchmark-v1",
    evaluators=[evaluate_factual_accuracy],
    experiment_prefix="gpt-4o-mini-candidate"
)
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Evaluation Metrics & Trajectory Benchmarking Sandbox", divider="rainbow")
st.markdown("""
Inspect quantitative evaluation metrics across enterprise benchmark datasets.
Review RAG Triad scores, trajectory efficiencies, and human expert calibration.
""")

bench_sim = EvaluationBenchmarkingSimulator()

col_sel, col_scores = st.columns([1, 1])

with col_sel:
    st.subheader("⚖️ Target System Selection")
    chosen_system = st.selectbox(
        "Choose Evaluation Benchmark Run:",
        list(bench_sim.benchmark_runs.keys())
    )
    run_eval_btn = st.button("🚀 Load Quantitative Evaluation Scorecard", type="primary")

with col_scores:
    st.subheader("📊 RAG Triad & System Metrics")
    if run_eval_btn:
        res = bench_sim.evaluate_benchmark(chosen_system)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Context Relevance", f"{res['context_relevance'] * 100:.1f}%")
            st.metric("Faithfulness (Groundedness)", f"{res['faithfulness'] * 100:.1f}%")
        with col_m2:
            st.metric("Answer Relevance", f"{res['answer_relevance'] * 100:.1f}%")
            st.metric("Trajectory Efficiency", f"{res['agent_step_efficiency'] * 100:.1f}%")
            
        st.markdown(
            f"<div class='metric-card'>"
            f"<b>Dataset Size:</b> {res['dataset_size']} Golden QA Items<br>"
            f"<b>Human Expert CSAT:</b> <span class='score-badge'>{res['human_satisfaction_rating']} / 5.0</span><br><br>"
            f"<b>Executive Summary:</b> {res['summary']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["faithfulness"] >= 0.90 and res["context_relevance"] >= 0.80:
            st.success("Quality Gate Status: PASSED (System meets enterprise deployment standards)")
        else:
            st.warning("Quality Gate Status: REVIEW NEEDED (One or more metrics below strict production threshold)")
    else:
        st.info("Select a system and generate the evaluation scorecard.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What are the three metrics comprising the 'RAG Triad' and what does each measure?"):
    st.markdown("""
    **Answer**:
    1. **Context Relevance**: Measures how much of the retrieved context is actually necessary and relevant to answer the question, penalizing retriever noise.
    2. **Faithfulness (Groundedness)**: Measures whether every statement in the generated answer can be mathematically or logically deduced from the retrieved context, detecting hallucinations.
    3. **Answer Relevance**: Measures whether the final response directly addresses the user's intent without drifting off-topic.
    """)

with st.expander("Q2: How do you prevent judge bias when using LLM-as-a-judge?"):
    st.markdown("""
    **Answer**: LLM judges suffer from position bias (favoring the first option in pairwise evaluations), verbosity bias (favoring longer, verbose answers),
    and self-enhancement bias (favoring responses from their own model family). Mitigations include: swapping position orders and averaging scores,
    enforcing length-normalized rubrics, requiring explicit chain-of-thought rationale before the score, and calibrating scores against human ratings.
    """)

with st.expander("Q3: How is agent trajectory efficiency calculated?"):
    st.markdown("""
    **Answer**: Trajectory efficiency compares the actual sequence of tools called by the agent against the minimal, optimal path identified in
    a golden test case. If an agent calls 8 tools when the query only requires 2, its efficiency score is heavily penalized, highlighting loops,
    redundant tool executions, and inefficient reasoning.
    """)

with st.expander("Q4: Why should automated evaluation metrics be calibrated against human ratings?"):
    st.markdown("""
    **Answer**: Without human calibration, automated metrics risk measuring proxies that diverge from business value. Calibrating involves
    having domain experts grade a sample (e.g. 100 responses) and computing statistical correlation (Pearson or Spearman rank correlation)
    between the automated metric and human scores. A metric with low correlation (< 0.70) must have its prompt and rubric revised.
    """)

with st.expander("Q5: What is the role of synthetic dataset generation in RAG evaluation?"):
    st.markdown("""
    **Answer**: Manual curation of hundreds of diverse QA pairs is slow and expensive. Synthetic generation utilizes capable models to extract
    passages, generate plausible user questions, and create reference answers across thousands of documents. Human experts then audit a subset
    (10-20%) for quality, yielding comprehensive benchmark coverage in a fraction of the time.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Evaluating Only Answer Quality in RAG)**: Grading only the final answer while ignoring retrieved contexts. If the retriever is poor, the LLM may answer from its pre-trained memory rather than private company documents.")
    st.warning("**Trap 2 (Verbosity Bias in LLM Judges)**: Using an uncalibrated LLM judge that awards high scores to long, wordy answers that say very little. Always enforce conciseness criteria in judge rubrics.")
with col_trap2:
    st.error("**Trap 3 (Single-Metric Optimization)**: Optimizing purely for Faithfulness. An agent that answers 'I don't know' to every question achieves 100% Faithfulness, but 0% utility. Always evaluate the full triad.")
    st.warning("**Trap 4 (Static Benchmark Datasets)**: Never updating the evaluation dataset as production usage evolves. The system maintains high benchmark scores while failing on new production query patterns.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What does Faithfulness measure in RAG evaluation?")
        st.caption("Flip for answer:")
        st.info("**Back**: Whether every factual claim in the answer is grounded in the retrieved context (absence of hallucinations).")
    with st.container(border=True):
        st.markdown("**Front**: What does Context Relevance measure?")
        st.caption("Flip for answer:")
        st.info("**Back**: How concise and relevant the retrieved document chunks are relative to the user query.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How do you counter position bias in LLM-as-a-judge comparisons?")
        st.caption("Flip for answer:")
        st.info("**Back**: Swap the presentation order of the candidate answers and average the resulting scores across runs.")
    with st.container(border=True):
        st.markdown("**Front**: What metric tracks agent reasoning path quality?")
        st.caption("Flip for answer:")
        st.info("**Back**: Trajectory efficiency (measuring tool selection accuracy and minimal step count against optimal path).")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the components of the RAG Triad: Context Relevance, Faithfulness, and Answer Relevance.", value=True)
c2 = st.checkbox("I can build golden evaluation datasets combining synthetic generation with expert curation.", value=True)
c3 = st.checkbox("I know how to implement LLM-as-a-judge scoring with structured rationale and rubrics.", value=True)
c4 = st.checkbox("I understand how to mitigate judge biases including position, verbosity, and self-enhancement.", value=True)
c5 = st.checkbox("I can evaluate multi-step agent trajectory efficiency and tool path minimality.", value=True)
c6 = st.checkbox("I know how to establish automated CI quality gates using evaluation frameworks like LangSmith and Ragas.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 33: Testing LangChain Applications.")
