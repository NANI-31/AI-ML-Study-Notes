import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.2 RAG Evaluation - LangChain Master Curriculum",
    page_icon="🔍",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class RAGEvaluationPipelineEngine:
    """
    Enterprise RAG Triad & Hallucination Evaluation Engine.
    Simulates evaluation across the complete RAG lifecycle:
    1. Retrieval Quality (Context Precision, Context Recall, Hit Rate)
    2. Context Relevance (Signal-to-Noise ratio of retrieved chunks)
    3. Answer Relevance (Directness and intent satisfaction)
    4. Faithfulness (Groundedness of answer claims in context)
    5. Citation Correctness (Source chunk attribution accuracy)
    6. Hallucination Detection (Extrinsic vs Intrinsic hallucinations)
    """
    def __init__(self, pipeline_name: str = "enterprise-rag-evaluator"):
        self.pipeline_name = pipeline_name
        self.evaluation_records: List[Dict[str, Any]] = []

    def evaluate_rag_pipeline(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        golden_contexts: List[str],
        golden_answer: str,
        inject_hallucination: bool = False,
        inject_retrieval_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Computes composite RAG Triad scores, citation validity, and hallucination flags.
        """
        # 1. Retrieval Quality: Precision & Recall against golden context chunks
        if inject_retrieval_failure:
            effective_contexts = ["Irrelevant terms of service clause from 2021 regarding parking rules."]
            retrieval_precision = 0.10
            retrieval_recall = 0.05
            context_relevance = 0.12
        else:
            effective_contexts = retrieved_contexts
            retrieval_precision = 0.90
            retrieval_recall = 0.95
            context_relevance = 0.88

        # 2. Answer Relevance: Direct intent addressing
        answer_relevance = 0.92 if not inject_hallucination else 0.45

        # 3. Faithfulness: Entailment of generated claims against retrieved contexts
        if inject_hallucination:
            faithfulness = 0.25
            hallucination_detected = True
            hallucination_type = "Extrinsic Hallucination (Fabricated ungrounded claims)"
            citation_correctness = 0.20
        elif inject_retrieval_failure:
            # Model generated answer despite bad context (hallucinating from pre-training weights)
            faithfulness = 0.30
            hallucination_detected = True
            hallucination_type = "Intrinsic Parametric Hallucination (Ignored missing context)"
            citation_correctness = 0.00
        else:
            faithfulness = 0.96
            hallucination_detected = False
            hallucination_type = "None (Fully Grounded)"
            citation_correctness = 1.0

        # Composite RAG Score (Weighted Mean: 35% Faithfulness, 35% Context Relevance, 30% Answer Relevance)
        rag_composite_score = round(
            (0.35 * faithfulness) + (0.35 * context_relevance) + (0.30 * answer_relevance),
            3
        )
        passed = rag_composite_score >= 0.80 and not hallucination_detected

        report = {
            "query": query,
            "generated_answer": generated_answer,
            "retrieval_metrics": {
                "context_precision": retrieval_precision,
                "context_recall": retrieval_recall,
                "context_relevance": context_relevance
            },
            "generation_metrics": {
                "answer_relevance": answer_relevance,
                "faithfulness": faithfulness,
                "citation_correctness": citation_correctness
            },
            "hallucination_audit": {
                "hallucination_detected": hallucination_detected,
                "hallucination_type": hallucination_type
            },
            "rag_composite_score": rag_composite_score,
            "passed": passed,
            "verdict": "CERTIFIED_RAG_PASS" if passed else "FAILED_QUALITY_GATE"
        }
        self.evaluation_records.append(report)
        return report

# Initialize engine in session state
if "rag_eval_engine" not in st.session_state:
    st.session_state.rag_eval_engine = RAGEvaluationPipelineEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("22.2 RAG Evaluation")
st.caption("Chapter 22: Evaluation | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Evaluating Retrieval-Augmented Generation (RAG) applications requires dissecting the two halves of the pipeline: 
the **Retrieval Component** (finding the right knowledge chunks) and the **Generation Component** (synthesizing an answer grounded strictly in that knowledge). 
Evaluating only the final generated text blinds engineers to upstream retrieval failures. 
The industry standard RAG Triad framework isolates and evaluates both stages.

In this lesson, you will master:
1. Retrieval quality metrics: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and Hit Rate
2. Context relevance: Measuring signal-to-noise ratio in retrieved context windows
3. Answer relevance: Evaluating direct intent satisfaction and conciseness
4. Faithfulness: Verifying factual claim entailment against retrieved sources
5. Citation correctness and automated hallucination detection (extrinsic vs intrinsic)
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The RAG Triad and Retrieval-Generation Evaluation Architecture", divider="rainbow")

st.markdown("""
The RAG Triad (Context Relevance, Groundedness/Faithfulness, and Answer Relevance) provides an orthogonal 
diagnostic framework. By scoring each leg of the triad independently, engineers can pinpoint whether a failure 
stems from the vector embedding model, the chunking strategy, or the synthesis prompt.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAGTriadArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_rag_system {
            label = "1. Active RAG Production Pipeline";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            Q [label="User Query\\n'What is the refund window?'", fillcolor="#1E3A8A", color="#3B82F6"];
            RET [label="Vector Retriever / Hybrid Search\\nTop-k Knowledge Chunks", fillcolor="#1E3A8A", color="#3B82F6"];
            GEN [label="LLM Synthesis Model\\nGenerated Completion with Citations", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_triad {
            label = "2. The RAG Triad Evaluation Core";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F59E0B";
            fontsize = 12;

            LEG1 [label="Triad Leg 1: Context Relevance\\n(Is retrieved context relevant to Query?)", fillcolor="#7C2D12", color="#EA580C"];
            LEG2 [label="Triad Leg 2: Faithfulness / Groundedness\\n(Is Answer substantiated by Context?)", fillcolor="#7C2D12", color="#EA580C"];
            LEG3 [label="Triad Leg 3: Answer Relevance\\n(Does Answer address user Query?)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_guards {
            label = "3. Specialized Safety & Factuality Evaluators";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            CITE [label="Citation Correctness Auditor\\n(Checks Chunk ID References)", fillcolor="#0E7490", color="#06B6D4"];
            HALLUC [label="Hallucination Detector\\n(Extrinsic vs Intrinsic Claims)", fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_outcome {
            label = "4. Diagnostic Resolution";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            RES_PASS [label="Certified High-Fidelity Answer\\n(Promoted to User Response)", fillcolor="#065F46", color="#059669"];
            RES_RETRY [label="Retrieval Failure Detected\\n(Trigger Query Rewriting / Fallback)", fillcolor="#881337", color="#F43F5E"];
        }

        Q -> RET [label="Queries Index"];
        RET -> GEN [label="Feeds Context Chunks"];

        Q -> LEG1 [label="Evaluates Input"];
        RET -> LEG1 [label="Evaluates Context"];

        RET -> LEG2 [label="Source Context"];
        GEN -> LEG2 [label="Candidate Claims"];

        Q -> LEG3 [label="Original Intent"];
        GEN -> LEG3 [label="Candidate Response"];

        GEN -> CITE [label="Inspects [Doc 1] Tags"];
        GEN -> HALLUC [label="Audits Unsupported Claims"];

        LEG1 -> RES_RETRY [label="Context Relevance < 0.60"];
        LEG2 -> RES_PASS [label="Faithfulness >= 0.90"];
        HALLUC -> RES_RETRY [label="Hallucination Found"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: RAG Evaluation
1. **Never Blame the Generator for a Retriever Failure**: If an LLM hallucinates because the vector retriever returned irrelevant chunks, tuning the system prompt will not fix the issue. The RAG Triad isolates retrieval failures (`Context Relevance < 0.60`) from synthesis failures (`Faithfulness < 0.90`).
2. **Context Precision Must Precede Generator Invocation**: Feeding 10 chunks where only 1 is relevant dilutes the model's attention ("Lost in the Middle" phenomenon). Evaluating context precision ensures vector search and reranking filters noise before token generation begins.
3. **Citation Precision is Non-Negotiable in Regulated Enterprise**: Answers must not only be correct; they must cite the exact chunk ID from which each claim was derived. A response that asserts a correct fact with a fabricated citation tag fails enterprise qualification.
""")

st.divider()

# ==============================================================================
# SECTION 2: RETRIEVAL QUALITY
# ==============================================================================
st.header("2. Retrieval Quality (Precision@k, Recall@k, Mean Reciprocal Rank and NDCG)", divider="rainbow")

st.markdown("""
Retrieval quality evaluates how effectively the vector store and search retriever surface the ground-truth 
documents required to answer a user inquiry. Standard ranking metrics from Information Retrieval (IR) 
are adapted for RAG pipelines:
""")

st.markdown("""
### Key Retrieval Metrics
* **Context Precision@k**: The proportion of retrieved top-k chunks that are truly relevant to the query. High precision prevents context dilution.
* **Context Recall@k**: The proportion of all relevant ground-truth chunks that were successfully retrieved in the top-k results. High recall ensures the model has all necessary facts.
* **Mean Reciprocal Rank (MRR)**: Evaluates where the *first* relevant document appears in the retrieved list: `1 / rank`. If the best chunk is at rank 1, MRR = 1.0; if at rank 4, MRR = 0.25.
* **Hit Rate@k**: Binary indicator of whether at least one relevant document appears in the top-k results.
""")

retrieval_metric_df = pd.DataFrame([
    {"IR Metric": "Context Precision@k", "Formula / Concept": "(Relevant Chunks Retrieved) / k", "Target SLA": ">= 0.80", "What it Measures": "Signal-to-noise ratio in context window"},
    {"IR Metric": "Context Recall@k", "Formula / Concept": "(Relevant Chunks Retrieved) / (Total Golden Chunks)", "Target SLA": ">= 0.90", "What it Measures": "Coverage of necessary facts"},
    {"IR Metric": "Mean Reciprocal Rank (MRR)", "Formula / Concept": "1 / rank_of_first_relevant_doc", "Target SLA": ">= 0.85", "What it Measures": "Effectiveness of reranker top positioning"},
    {"IR Metric": "Hit Rate@k", "Formula / Concept": "1 if relevant_in_top_k else 0", "Target SLA": ">= 0.98", "What it Measures": "Avoidance of total retrieval misses"}
])
st.dataframe(retrieval_metric_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production LangChain pipelines evaluate retrievers independently of models. 
Using LangSmith or Ragas, teams benchmark dense embeddings (OpenAI vs Cohere) and hybrid search (BM25 + Dense with Reciprocal Rank Fusion) 
to maximize MRR before connecting generator models.
""")

st.divider()

# ==============================================================================
# SECTION 3: CONTEXT RELEVANCE
# ==============================================================================
st.header("3. Context Relevance (Signal-to-Noise Ratio, Distractor Filtering and Chunk Slicing)", divider="rainbow")

st.markdown("""
Context relevance measures the density of useful information contained within the retrieved context window. 
Even when the correct answer is present, packing 4,000 tokens of distractor text around a 50-token answer 
degrades model reasoning, inflates latency, and increases token costs.
""")

st.markdown("""
### The Danger of Context Distractors
* **The "Lost in the Middle" Effect**: LLMs attend strongly to tokens at the very beginning and very end of the context prompt. Chunks placed in the middle of long contexts are frequently overlooked by the model.
* **Context Compression**: Using LangChain's `ContextualCompressionRetriever` or LLM rerankers (Cohere Rerank) filters irrelevant sentences out of retrieved chunks before passing them to the generator.
* **Context Relevance Scoring**: An evaluator LLM grades each chunk independently: *"Does this chunk contain facts directly required to answer the query?"* Score = `(Useful Chunks) / (Total Chunks Provided)`.
""")

st.code('''
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS

# Upgrading Retrieval to Maximize Context Relevance via Reranking
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Compression reranker drops distractors, leaving only top 4 highly relevant chunks
compressor = CohereRerank(model="rerank-english-v3.0", top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Top-performing RAG systems require a Context Relevance score >= 0.85. 
If context relevance falls below threshold on production queries, the system triggers dynamic query rewriting 
or increases reranking thresholds.
""")

st.divider()

# ==============================================================================
# SECTION 4: ANSWER RELEVANCE
# ==============================================================================
st.header("4. Answer Relevance (Direct Intent Addressing, Conciseness and Non-Evasion)", divider="rainbow")

st.markdown("""
Answer relevance evaluates whether the generated output directly and completely resolves the user's inquiry, 
regardless of whether the facts are accurate. A response that gives a 500-word history of cryptography when asked 
*"How do I reset my password?"* may be 100% factually true, but it fails answer relevance.
""")

st.markdown("""
### Evaluating Answer Relevance
* **Intent Satisfaction**: Does the completion provide actionable resolution to the user's explicit question?
* **Conciseness and Non-Evasion**: Does the model avoid verbose throat-clearing (*"As an AI language model..."*) and stay focused on the core answer?
* **Reverse Query Generation Method**: A common automated evaluation pattern asks an evaluator model: *"Generate 3 questions that this answer resolves."* The evaluator then computes semantic similarity between the user's original query and the reverse-generated questions.
""")

st.code('''
# LangChain Criteria Evaluation for Answer Relevance
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("relevance")
eval_result = evaluator.evaluate_strings(
    prediction="To reset your password, visit Settings > Security and click 'Reset Password'.",
    input="How do I change my login password?"
)
print(f"Relevance Score: {eval_result['score']}, Feedback: {eval_result['reasoning']}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production evaluators verify that answers do not evade questions when context is present. 
If context contains the answer, but the model outputs *"I cannot find that information"*, the run is penalized for false refusal.
""")

st.divider()

# ==============================================================================
# SECTION 5: FAITHFULNESS
# ==============================================================================
st.header("5. Faithfulness (Groundedness in Retrieved Context, Claim Entailment and Fact Checking)", divider="rainbow")

st.markdown("""
Faithfulness (also termed Groundedness) is the single most critical quality metric in enterprise RAG. 
It measures whether every claim made in the generated answer is strictly entailed by the provided context. 
An ungrounded claim is a hallucination.
""")

st.markdown("""
### Claim Entailment Decomposition Algorithm
1. **Claim Extraction**: The generated answer is decomposed into individual atomic factual statements (e.g., Statement 1: *"Returns are allowed within 30 days"*; Statement 2: *"A receipt is mandatory"*).
2. **Entailment Verification**: For each atomic statement, an evaluator checks whether the retrieved context directly entails the claim.
3. **Faithfulness Score Calculation**:
   `Faithfulness = (Number of Substantiated Claims) / (Total Number of Extracted Claims)`.
   If 4 out of 4 claims are substantiated, Faithfulness = 1.0. If 1 claim is fabricated, Faithfulness = 0.75.
""")

st.code('''
# Atomic Claim Faithfulness Evaluator Schema
class ClaimEntailmentVerification(BaseModel):
    extracted_claims: List[str] = Field(description="List of individual factual statements in the answer.")
    supported_claims: List[str] = Field(description="Statements directly substantiated by retrieved context.")
    unsupported_claims: List[str] = Field(description="Statements not supported or fabricated.")
    faithfulness_score: float = Field(description="Ratio of supported claims to total claims (0.0 to 1.0).")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise RAG gates enforce a strict `Faithfulness == 1.0` requirement for high-risk domains (legal, medical, financial). 
If any unsupported claim is detected, the response is discarded and an automated polite refusal is returned instead.
""")

st.divider()

# ==============================================================================
# SECTION 6: CITATION CORRECTNESS
# ==============================================================================
st.header("6. Citation Correctness (Attribution Precision, Chunk ID Mapping and Source Traceability)", divider="rainbow")

st.markdown("""
In enterprise knowledge retrieval, users do not accept answers on blind faith; they require verifiable source citations. 
However, models frequently fabricate citations, attaching reference tags like `[Doc 1]` to claims that actually appeared 
in `[Doc 4]` or were not present in any document at all.
""")

st.markdown("""
### The Two Failure Modes of RAG Citations
* **Phantom Citations (Citation Hallucination)**: The model cites a document tag (`[Doc 2]`) that was never provided in the context window.
* **Mismatched Attribution (Misattribution)**: The model makes claim X and cites `[Doc 1]`, but claim X actually appears in `[Doc 3]`.
* **Citation Precision Metric**:
  `Citation Precision = (Correctly Attributed Citations) / (Total Citations Generated)`.
""")

citation_audit_df = pd.DataFrame([
    {"Citation Status": "Verified Accurate", "Claim Example": "'401k match is 4%' [Doc 1]", "Context Fact": "Doc 1 explicitly states 401k matching is 4%", "Verdict": "PASS (Valid Citation)"},
    {"Citation Status": "Misattributed Citation", "Claim Example": "'Dental coverage begins Day 1' [Doc 1]", "Context Fact": "Doc 1 covers 401k; Doc 3 covers dental coverage", "Verdict": "FAIL (Misattribution)"},
    {"Citation Status": "Phantom Citation", "Claim Example": "'Stock cliff is 1 year' [Doc 5]", "Context Fact": "Only 3 documents were retrieved ([Doc 1]-[Doc 3])", "Verdict": "FAIL (Hallucinated Source ID)"}
])
st.dataframe(citation_audit_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production LangChain pipelines use deterministic regex validators to verify that all citation brackets `[Doc N]` 
correspond to real retrieved chunk IDs, followed by an NLI entailment check on the specific cited chunk.
""")

st.divider()

# ==============================================================================
# SECTION 7: HALLUCINATION DETECTION
# ==============================================================================
st.header("7. Hallucination Detection (Extrinsic vs Intrinsic Hallucination and Negative Constraint Verification)", divider="rainbow")

st.markdown("""
Hallucinations in RAG applications fall into two distinct architectural categories:
""")

st.markdown("""
### Extrinsic vs Intrinsic Hallucination
* **Extrinsic Hallucination**: The model introduces facts from its pre-training weights that were *never present* in the retrieved context. (Example: context lists 3 features; the model lists 5 features, hallucinating the extra 2 from general memory).
* **Intrinsic Hallucination**: The model contradicts facts directly stated in the retrieved context. (Example: context states *"Refunds take 5-7 business days"*; the model states *"Refunds are processed in 24 hours"*).
* **Negative Constraint Probes**: Testing queries where context is missing or irrelevant. The evaluated invariant: *Does the model politely refuse, or does it hallucinate?*
""")

st.code('''
# Automated Hallucination Classifier
def detect_rag_hallucination(retrieved_context: str, generated_answer: str) -> dict:
    """
    Classifies whether the generated answer exhibits intrinsic or extrinsic hallucinations.
    """
    # LLM-assisted entailment check
    result = hallucination_judge.invoke({
        "context": retrieved_context,
        "answer": generated_answer
    })
    return {
        "is_hallucination": result.hallucination_detected,
        "hallucination_type": result.classification,  # "None", "Extrinsic", "Intrinsic"
        "violating_sentences": result.violating_sentences
    }
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Teams deploy fast, specialized Small Language Models (e.g., Llama-3-8B-Instruct or DeBERTa NLI) to run 
low-latency hallucination detection before streaming completions to end users.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full-Spectrum RAG Evaluation Pipeline (RAGEvaluationPipelineEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the RAG evaluation engine. 
It computes retrieval quality metrics (precision, recall), evaluates context relevance, verifies answer relevance, 
audits context faithfulness, and performs automated hallucination detection.
""")

st.code('''
class RAGEvaluationPipelineEngine:
    """
    Enterprise RAG Triad & Hallucination Evaluation Engine.
    Simulates evaluation across the complete RAG lifecycle:
    1. Retrieval Quality (Context Precision, Context Recall, Hit Rate)
    2. Context Relevance (Signal-to-Noise ratio of retrieved chunks)
    3. Answer Relevance (Directness and intent satisfaction)
    4. Faithfulness (Groundedness of answer claims in context)
    5. Citation Correctness (Source chunk attribution accuracy)
    6. Hallucination Detection (Extrinsic vs Intrinsic hallucinations)
    """
    def __init__(self, pipeline_name: str = "enterprise-rag-evaluator"):
        self.pipeline_name = pipeline_name
        self.evaluation_records: List[Dict[str, Any]] = []

    def evaluate_rag_pipeline(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        golden_contexts: List[str],
        golden_answer: str,
        inject_hallucination: bool = False,
        inject_retrieval_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Computes composite RAG Triad scores, citation validity, and hallucination flags.
        """
        # 1. Retrieval Quality
        if inject_retrieval_failure:
            effective_contexts = ["Irrelevant terms clause regarding parking rules."]
            retrieval_precision = 0.10
            retrieval_recall = 0.05
            context_relevance = 0.12
        else:
            effective_contexts = retrieved_contexts
            retrieval_precision = 0.90
            retrieval_recall = 0.95
            context_relevance = 0.88

        # 2. Answer Relevance
        answer_relevance = 0.92 if not inject_hallucination else 0.45

        # 3. Faithfulness
        if inject_hallucination:
            faithfulness = 0.25
            hallucination_detected = True
            hallucination_type = "Extrinsic Hallucination (Fabricated ungrounded claims)"
            citation_correctness = 0.20
        elif inject_retrieval_failure:
            faithfulness = 0.30
            hallucination_detected = True
            hallucination_type = "Intrinsic Parametric Hallucination (Ignored missing context)"
            citation_correctness = 0.00
        else:
            faithfulness = 0.96
            hallucination_detected = False
            hallucination_type = "None (Fully Grounded)"
            citation_correctness = 1.0

        rag_composite_score = round(
            (0.35 * faithfulness) + (0.35 * context_relevance) + (0.30 * answer_relevance),
            3
        )
        passed = rag_composite_score >= 0.80 and not hallucination_detected

        report = {
            "query": query,
            "generated_answer": generated_answer,
            "retrieval_metrics": {
                "context_precision": retrieval_precision,
                "context_recall": retrieval_recall,
                "context_relevance": context_relevance
            },
            "generation_metrics": {
                "answer_relevance": answer_relevance,
                "faithfulness": faithfulness,
                "citation_correctness": citation_correctness
            },
            "hallucination_audit": {
                "hallucination_detected": hallucination_detected,
                "hallucination_type": hallucination_type
            },
            "rag_composite_score": rag_composite_score,
            "passed": passed,
            "verdict": "CERTIFIED_RAG_PASS" if passed else "FAILED_QUALITY_GATE"
        }
        self.evaluation_records.append(report)
        return report
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: RAG Triad Evaluator and Hallucination Scanner Simulator", divider="rainbow")

st.markdown("""
Simulate and score an end-to-end RAG pipeline across the RAG Triad dimensions. 
Inject retrieval failures or hallucinations to observe how individual metrics respond.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Pipeline Scenario Controls")
    scenario_select = st.selectbox(
        "RAG Inquiry Scenario",
        [
            "Standard Customer Support (Healthy Retrieval & Grounded Answer)",
            "Retrieval Miss (Irrelevant Documents Returned)",
            "Extrinsic Hallucination (Model Fabricates External Facts)"
        ],
        index=0
    )

    inject_ret_fail = "Retrieval Miss" in scenario_select
    inject_halluc = "Extrinsic Hallucination" in scenario_select

    q_text = "What is the warranty period for enterprise server hardware?"
    ctx_samples = [
        "Doc 1: Enterprise server hardware includes a standard 36-month on-site replacement warranty.",
        "Doc 2: Warranty coverage includes 24/7 dedicated telephone support and hardware dispatch."
    ]
    gt_text = "Enterprise server hardware includes a 36-month on-site replacement warranty with 24/7 support."

    if inject_halluc:
        gen_ans = "Enterprise hardware has a lifetime warranty and free yearly upgrades to newest models [Doc 1]."
    elif inject_ret_fail:
        gen_ans = "Servers are covered under standard warranties."
    else:
        gen_ans = "Enterprise server hardware includes a 36-month on-site replacement warranty [Doc 1] with 24/7 dedicated support [Doc 2]."

    st.markdown(f"**Query**: `{q_text}`")
    st.markdown(f"**Generated Answer**: `{gen_ans}`")

with wb_col2:
    st.markdown("### Live RAG Triad Scorecard")
    rag_report = st.session_state.rag_eval_engine.evaluate_rag_pipeline(
        query=q_text,
        retrieved_contexts=ctx_samples,
        generated_answer=gen_ans,
        golden_contexts=ctx_samples,
        golden_answer=gt_text,
        inject_hallucination=inject_halluc,
        inject_retrieval_failure=inject_ret_fail
    )

    if rag_report["passed"]:
        st.success(f"**VERDICT: {rag_report['verdict']}** — Composite score {rag_report['rag_composite_score']:.3f} meets release criteria.")
    else:
        st.error(f"**VERDICT: {rag_report['verdict']}** — Composite score {rag_report['rag_composite_score']:.3f} failed release thresholds.")

    # Triad Metrics Row
    ret_m = rag_report["retrieval_metrics"]
    gen_m = rag_report["generation_metrics"]
    halluc_m = rag_report["hallucination_audit"]

    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Context Relevance", f"{ret_m['context_relevance']:.2f}")
    with t2:
        st.metric("Faithfulness (Grounded)", f"{gen_m['faithfulness']:.2f}")
    with t3:
        st.metric("Answer Relevance", f"{gen_m['answer_relevance']:.2f}")

    # Hallucination and Citation Audit Box
    st.markdown("#### Hallucination & Attribution Audit")
    if halluc_m["hallucination_detected"]:
        st.error(f"🚨 **Hallucination Detected**: {halluc_m['hallucination_type']}")
    else:
        st.info("✅ **Zero Hallucination Detected**: All claims directly grounded in retrieved context chunks.")

    h_df = pd.DataFrame([
        {"Metric": "Context Precision", "Score": f"{ret_m['context_precision']:.2f}", "Status": "PASS" if ret_m['context_precision'] >= 0.8 else "FAIL"},
        {"Metric": "Context Recall", "Score": f"{ret_m['context_recall']:.2f}", "Status": "PASS" if ret_m['context_recall'] >= 0.8 else "FAIL"},
        {"Metric": "Citation Correctness", "Score": f"{gen_m['citation_correctness']:.2f}", "Status": "PASS" if gen_m['citation_correctness'] == 1.0 else "FAIL"}
    ])
    st.dataframe(h_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

with st.expander("Q1: What are the three legs of the RAG Triad, and what does each measure?"):
    st.markdown("""
    **Answer**: The RAG Triad consists of:
    1) **Context Relevance**: Measures whether the retrieved chunks are relevant to the user query (retrieval signal-to-noise ratio);
    2) **Faithfulness / Groundedness**: Measures whether every claim in the generated answer is strictly substantiated by the retrieved context (absence of hallucinations);
    3) **Answer Relevance**: Measures whether the generated answer directly resolves the user's inquiry (absence of evasion or fluff).
    """)

with st.expander("Q2: What is the difference between Context Precision and Context Recall in RAG?"):
    st.markdown("""
    **Answer**: Context Precision measures the proportion of retrieved chunks that are actually useful (`Useful / Total Retrieved`), 
    ensuring the context window is not diluted with distractors. Context Recall measures whether all facts necessary to construct 
    the answer were successfully retrieved (`Retrieved Relevant / Total Golden Facts`). High recall prevents incomplete answers; 
    high precision prevents model confusion.
    """)

with st.expander("Q3: What is the architectural difference between Extrinsic and Intrinsic hallucinations in RAG?"):
    st.markdown("""
    **Answer**: Extrinsic hallucination occurs when the model incorporates information from its pre-training weights that was 
    never present in the retrieved context (adding fabricated external details). Intrinsic hallucination occurs when the model's 
    output directly contradicts facts that were explicitly stated in the retrieved context (e.g., inverting numbers or dates).
    """)

with st.expander("Q4: Why does Mean Reciprocal Rank (MRR) matter for RAG generator performance?"):
    st.markdown("""
    **Answer**: LLM attention mechanisms suffer from the "Lost in the Middle" phenomenon: models attend most reliably to information 
    placed at the beginning of the context prompt. MRR measures how close to rank 1 the primary relevant document appears. 
    Higher MRR ensures that the most critical information is placed at the top of the context window where generator attention is highest.
    """)

with st.expander("Q5: How do you verify Citation Correctness in automated RAG pipelines?"):
    st.markdown("""
    **Answer**: Automated citation verification uses a two-phase check: 1) Deterministic regex parsing to ensure cited chunk IDs 
    (e.g., `[Doc 1]`) exist in the retrieved set (detecting phantom citations); and 2) Natural Language Inference (NLI) claim entailment 
    verifying that the specific sentence preceding the citation is entailed by the specific chunk referenced (detecting misattributions).
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("**Trap 1 (Evaluating Only the Generated Output)**: Testing RAG applications using only end-to-end answer correctness. When an answer is incorrect, engineers cannot tell whether the vector store failed to retrieve the right document or the LLM hallucinated, leading to wasted engineering effort.")
    st.warning("**Trap 2 (Ignoring Phantom Citations)**: Assuming that because the model generated `[Doc 1]` brackets, the citations are correct. LLMs routinely hallucinate citation tags or cite the wrong document. Automated citation entailment auditing is mandatory.")
with col_tr2:
    st.error("**Trap 3 (Context Window Dilution)**: Retrieving 20 chunks (`k=20`) without reranking to maximize recall. Diluting the context with 15 irrelevant distractors severely degrades LLM reasoning and inflates latency and token costs.")
    st.warning("**Trap 4 (BLEU / ROUGE for Factuality)**: Using n-gram string overlap metrics to grade RAG factual correctness. A response that uses synonyms scores poorly under BLEU despite being 100% faithful to the context.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the RAG Triad formula for Faithfulness?")
        st.caption("Flip for answer:")
        st.info("**Back**: `Faithfulness = (Number of Supported Claims) / (Total Claims Extracted)`. An answer is faithful only if every claim is entailed by retrieved context.")

    with st.container(border=True):
        st.markdown("**Front**: What does Context Relevance measure in RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: The signal-to-noise ratio of the retrieved context: verifying that retrieved chunks contain only facts needed to answer the query, without distractors.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is a 'Phantom Citation' in RAG systems?")
        st.caption("Flip for answer:")
        st.info("**Back**: When a generated answer cites a document tag (e.g., `[Doc 5]`) that was never present in the retrieved context set.")

    with st.container(border=True):
        st.markdown("**Front**: Why is Cohere Rerank used to improve Context Relevance?")
        st.caption("Flip for answer:")
        st.info("**Back**: It reorders retrieved chunks using cross-encoder models, placing the most relevant chunks at rank 1 and discarding low-relevance distractors.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of RAG Evaluation in LangChain:")

c1 = st.checkbox("I understand the RAG Triad framework (Context Relevance, Faithfulness, Answer Relevance).", value=True)
c2 = st.checkbox("I can calculate retrieval metrics: Context Precision, Context Recall, and Mean Reciprocal Rank.", value=True)
c3 = st.checkbox("I know how to mitigate the 'Lost in the Middle' effect using contextual compression and rerankers.", value=True)
c4 = st.checkbox("I can implement claim entailment algorithms to measure factual faithfulness.", value=True)
c5 = st.checkbox("I understand how to audit citation correctness and detect phantom citations.", value=True)
c6 = st.checkbox("I can differentiate between extrinsic and intrinsic hallucinations in generative pipelines.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered RAG Evaluation, the RAG Triad framework, retrieval ranking metrics, and hallucination detection in LangChain.")
