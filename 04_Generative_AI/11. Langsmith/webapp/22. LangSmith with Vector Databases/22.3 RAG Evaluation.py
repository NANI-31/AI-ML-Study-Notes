import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.3 RAG Evaluation - LangSmith Master Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class RAGEvaluationPipelineEngine:
    """
    Automated RAG Triad & End-to-End Evaluation Engine.
    Emulates LangSmith custom evaluators across Context Recall,
    Context Precision, Faithfulness (Grounding), Answer Relevance, and Experiment Tracking.
    """
    def __init__(self, experiment_name: str = "rag-triad-benchmark-v1"):
        self.experiment_name = experiment_name
        self.evaluation_results: List[Dict[str, Any]] = []

    def evaluate_rag_run(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str,
        expected_doc_ids: List[str],
        retrieved_doc_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Computes the complete RAG Triad metrics + End-to-End accuracy:
        1. Context Recall: Ratio of expected doc IDs captured in retrieval.
        2. Context Precision: Rank-weighted precision of relevant chunks.
        3. Faithfulness: Lexical grounding of answer claims in retrieved contexts.
        4. Answer Relevance: Semantic entity overlap between query and generated answer.
        5. Composite End-to-End Score: Weighted harmonic mean of Triad components.
        """
        # 1. Context Recall
        if not expected_doc_ids:
            context_recall = 1.0
        else:
            hits = sum(1 for doc_id in expected_doc_ids if doc_id in retrieved_doc_ids)
            context_recall = round(hits / len(expected_doc_ids), 2)

        # 2. Context Precision (Rank-weighted)
        precisions = []
        running_hits = 0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in expected_doc_ids:
                running_hits += 1
                precisions.append(running_hits / rank)
        context_precision = round(sum(precisions) / max(1, len(precisions)), 2) if precisions else 0.0

        # 3. Faithfulness (Grounding: answer words vs combined context)
        combined_ctx = " ".join(retrieved_contexts).lower()
        ans_tokens = set(generated_answer.lower().split())
        stopwords = {"the", "a", "an", "is", "are", "in", "on", "at", "to", "for", "of", "and", "or", "it", "with"}
        salient_ans = ans_tokens - stopwords

        if not salient_ans:
            faithfulness = 1.0
        else:
            grounded_count = sum(1 for token in salient_ans if token in combined_ctx)
            faithfulness = round(grounded_count / len(salient_ans), 2)

        # 4. Answer Relevance (query keywords addressed in generated answer)
        q_tokens = set(query.lower().split()) - stopwords
        if not q_tokens:
            answer_relevance = 1.0
        else:
            addressed_count = sum(1 for token in q_tokens if token in generated_answer.lower())
            answer_relevance = round(addressed_count / len(q_tokens), 2)

        # 5. Composite Score
        composite_score = round(
            (context_recall * 0.25) +
            (context_precision * 0.20) +
            (faithfulness * 0.35) +
            (answer_relevance * 0.20),
            2
        )

        status = "PASSED" if composite_score >= 0.75 else ("MARGINAL" if composite_score >= 0.50 else "FAILED")

        eval_record = {
            "query": query,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "composite_score": composite_score,
            "status": status,
            "langsmith_feedback_keys": {
                "rag_recall": context_recall,
                "rag_precision": context_precision,
                "rag_faithfulness": faithfulness,
                "rag_relevance": answer_relevance
            }
        }
        self.evaluation_results.append(eval_record)
        return eval_record


# --- HERO SECTION ---
st.title("22.3 RAG Evaluation")
st.caption("Chapter 22: LangSmith with Vector Databases | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
Building a working RAG prototype requires only a few lines of code; **proving that a RAG system answers accurately and reliably across 10,000 edge cases requires rigorous evaluation**.
Deploying prompt or chunking modifications without an evaluation harness is blind engineering: a change that improves one question often silently breaks twenty others.
**RAG Evaluation** in LangSmith operationalizes the industry-standard **RAG Triad** framework:
1. **Retrieval Evaluation (Context Recall)**: Verifying that vector search retrieved all essential factual evidence from the corpus.
2. **Context Evaluation (Context Precision)**: Ensuring high-relevance chunks appear at the top ranks without irrelevant noise dilution.
3. **Generation Evaluation (Faithfulness & Hallucination)**: Verifying that every claim in the generated answer is strictly grounded in retrieved text.
4. **End-to-End Evaluation (Answer Relevance & Task Completion)**: Measuring whether the synthesized response directly answers the user's inquiry.
5. **Continuous Regression Testing**: Running automated evaluation suites against versioned golden datasets (`Client.evaluate(...)`) on every pull request.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The RAG Triad & LangSmith Evaluation Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates the automated RAG evaluation workflow.
A **Golden Dataset** containing reference Q&A pairs triggers the RAG pipeline.
LangSmith executes **Custom Evaluators** to measure the **RAG Triad**, streaming feedback metrics into central comparison dashboards.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    rag_eval_blueprint = """
    digraph RAGEvaluationPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_dataset {
            label="1. Golden Evaluation Dataset Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            GoldenData [label="LangSmith Versioned Dataset\nExample: {query, ground_truth, expected_docs}", shape=cylinder, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_rag_execution {
            label="2. Instrumented RAG Pipeline Under Test";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            Retriever [label="Vector Search Retriever\nReturns Top-K Chunks", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            Generator [label="Frontier LLM Generator\nSynthesizes Candidate Answer", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_rag_triad {
            label="3. The RAG Triad Automated Evaluator Suite";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            ContextRecall [label="Evaluator 1: Context Recall\n(Retrieved Docs vs Expected Docs)", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
            ContextPrecision [label="Evaluator 2: Context Precision\n(Signal-to-Noise Rank Weight)", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
            Faithfulness [label="Evaluator 3: Faithfulness / Grounding\n(Answer Claims vs Retrieved Context)", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
            AnswerRelevance [label="Evaluator 4: Answer Relevance\n(Generated Answer vs User Query)", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
        }

        subgraph cluster_experiment_results {
            label="4. LangSmith Experiment Comparison & Regression Dashboard";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#EC4899";
            fontcolor="#F472B6";
            fontsize=11;

            Scorecard [label="LangSmith Experiment Scorecard\nFeedback Metrics Logged: client.create_feedback()", shape=component, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            CI_Gate [label="CI/CD Quality Gate (GitHub Actions)\nPass: Composite Score >= 0.75 | Fail: Block PR", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        GoldenData -> Retriever [label="Test Query"];
        Retriever -> Generator [label="Retrieved Contexts"];
        Generator -> Faithfulness [label="Candidate Answer"];

        Retriever -> ContextRecall [label="Retrieved Chunks"];
        Retriever -> ContextPrecision;
        Retriever -> Faithfulness [label="Context"];
        Generator -> AnswerRelevance;

        ContextRecall -> Scorecard;
        ContextPrecision -> Scorecard;
        Faithfulness -> Scorecard;
        AnswerRelevance -> Scorecard;

        Scorecard -> CI_Gate [label="Automated Regression Verdict"];
    }
    """
    st.graphviz_chart(rag_eval_blueprint, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: RETRIEVAL EVALUATION
# ==============================================================================
st.header("2. Retrieval Evaluation (Context Recall, Hit Rate@K, MRR & Golden Query-Doc Pairs)", divider="rainbow")

st.markdown("""
**Retrieval Evaluation** measures how effectively the vector search subsystem locates the necessary factual needles in the haystack.
If the retrieval layer misses the key document, downstream generation is doomed to hallucinate or reply *"I do not know"*.

Primary Retrieval Metrics:
1. **Context Recall**: The proportion of reference ground-truth evidence that was successfully retrieved into the candidate pool.
2. **Hit Rate @ K**: Binary indicator of whether at least one correct document was present in the Top-K results.
3. **Mean Reciprocal Rank (MRR)**: Measures the reciprocal rank of the first relevant chunk (`1 / rank`). If the first relevant chunk is rank 1, MRR is 1.0; if rank 2, MRR is 0.5.
""")

col_re1, col_re2 = st.columns([1, 1])

with col_re1:
    st.markdown("### Golden Dataset Schema for Retrieval Testing:")
    st.code('''
# LangSmith Dataset Example Schema
{
  "dataset_name": "enterprise-rag-golden-v1",
  "examples": [
    {
      "inputs": {"question": "What is the maximum allowed pod density per worker node?"},
      "outputs": {
        "ground_truth": "The maximum pod density is 110 pods per worker node.",
        "expected_doc_ids": ["doc_k8s_limits_04"]
      }
    }
  ]
}
''', language="json")

with col_re2:
    st.markdown("### Context Recall Diagnostic Logic")
    st.markdown("""
    In LangSmith, a Custom Evaluator compares the retrieved run outputs against the dataset reference:
    ```python
    def evaluate_context_recall(run, example):
        retrieved_ids = [doc.metadata["id"] for doc in run.outputs["documents"]]
        expected_ids = example.outputs["expected_doc_ids"]
        hits = sum(1 for doc_id in expected_ids if doc_id in retrieved_ids)
        recall = hits / len(expected_ids) if expected_ids else 1.0
        return {"key": "context_recall", "score": recall}
    ```
    If Context Recall drops after an index migration, engineers know immediately that chunking parameters or distance metrics degraded search coverage.
    """)

st.divider()

# ==============================================================================
# SECTION 3: CONTEXT EVALUATION
# ==============================================================================
st.header("3. Context Evaluation (Context Precision, Noise Ratio & Extractive Information Density)", divider="rainbow")

st.markdown("""
While Context Recall ensures that the correct evidence is *present*, **Context Precision** measures whether the correct evidence is *ranked at the top* rather than buried under irrelevant noise chunks.

Passing 10 chunks where only the 9th chunk is relevant creates an unfavorable signal-to-noise ratio.
The LLM generator wastes attention compute reading 8 irrelevant chunks, inflating token burn and risking context dilution.
""")

col_ce1, col_ce2 = st.columns([1, 1])

with col_ce1:
    st.markdown("### Context Precision vs Context Recall:")
    prec_comp_df = pd.DataFrame([
        {"Metric": "Context Recall", "Core Question": "Did we retrieve the answer document?", "Impact of Failure": "Immediate hallucination or 'I do not know'"},
        {"Metric": "Context Precision", "Core Question": "Is the answer ranked at the top of the prompt?", "Impact of Failure": "Lost-in-the-Middle attention sag and token waste"},
        {"Metric": "Noise Ratio", "Core Question": "What percentage of retrieved tokens are irrelevant?", "Impact of Failure": "High TTFT latency and elevated API costs"}
    ])
    st.dataframe(prec_comp_df, use_container_width=True)

with col_ce2:
    st.markdown("### Signal-to-Noise Ranking Topology")
    dot_prec = """
    digraph ContextPrecisionFlow {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        PoorRank [label="Low Precision (Rank 5 of 5 Relevant)\n[Noise, Noise, Noise, Noise, ANSWER]\nPrecision Score: 0.20", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
        HighRank [label="High Precision (Rank 1 of 5 Relevant)\n[ANSWER, Noise, Noise, Noise, Noise]\nPrecision Score: 1.00", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        PoorRank -> HighRank [label="Add Cross-Encoder Reranker", color="#38BDF8"];
    }
    """
    st.graphviz_chart(dot_prec, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: GENERATION EVALUATION
# ==============================================================================
st.header("4. Generation Evaluation (Faithfulness, Answer Relevance & Hallucination Scoring)", divider="rainbow")

st.markdown("""
Generation evaluation isolates the performance of the language model generator given the retrieved context.
It answers two critical questions in the RAG Triad:
1. **Faithfulness (Grounding)**: Is every factual assertion made in the generated answer directly supported by the retrieved context? If the answer contains claims absent from the context, it is flagged as a hallucination.
2. **Answer Relevance**: Does the generated answer directly address the user's inquiry, or did the model produce verbose tangential rambling?
""")

col_ge1, col_ge2 = st.columns([1, 1])

with col_ge1:
    st.markdown("### Evaluating Faithfulness via LLM-as-a-Judge:")
    st.markdown("""
    In LangSmith, Faithfulness can be evaluated programmatically or via a rubric-based LLM Judge:
    * **Step 1: Claim Extraction**: Break the generated answer into individual standalone atomic claims.
    * **Step 2: Context Entailment**: For each claim, evaluate if the retrieved context text logically entails the claim (`Yes / No`).
    * **Step 3: Faithfulness Score**:
      `Score = Entailed_Claims / Total_Claims`.
    * A score of `1.0` indicates perfect factual grounding; a score below `0.70` indicates partial hallucination.
    """)

with col_ge2:
    st.markdown("### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This")
    st.markdown("""
    Using `langsmith`'s prebuilt or custom evaluators:
    ```python
    from langsmith.evaluation import evaluate

    def faithfulness_evaluator(run, example):
        context = run.outputs["retrieved_context"]
        answer = run.outputs["answer"]
        # Fast NLI cross-encoder or structured rubric
        score = nli_model.predict(premise=context, hypothesis=answer)
        return {"key": "faithfulness", "score": score}

    # Execute full evaluation across golden dataset
    results = evaluate(
        rag_pipeline,
        data="golden-eval-v1",
        evaluators=[faithfulness_evaluator],
        experiment_prefix="prompt-v2-experiment"
    )
    ```
    """)

st.divider()

# ==============================================================================
# SECTION 5: END-TO-END EVALUATION
# ==============================================================================
st.header("5. End-to-End Evaluation (Semantic Equivalence, Task Completion & Regression Testing)", divider="rainbow")

st.markdown("""
End-to-end evaluation assesses whether the entire RAG pipeline—from raw query ingress to final synthesized output—successfully satisfies user expectations.
In enterprise CI/CD workflows, every prompt change or model version upgrade triggers an automated **Regression Test**:
comparing current experiment scores against a baseline run in LangSmith.
""")

col_ee1, col_ee2 = st.columns([1, 1])

with col_ee1:
    st.markdown("### Continuous Evaluation CI/CD Pipeline:")
    st.markdown("""
    1. A developer creates a Pull Request updating the RAG prompt template.
    2. GitHub Actions executes `langsmith.evaluation.evaluate()` against the 200-question Golden Dataset.
    3. LangSmith computes the Delta vs Production Baseline:
       * `Context Recall: +0.02 (Improved)`
       * `Faithfulness: -0.15 (REGRESSION DETECTED!)`
    4. The CI pipeline fails the build, blocking the PR from merging and preventing production degradation.
    """)

with col_ee2:
    st.markdown("### LangSmith Experiment Comparison View")
    exp_compare_df = pd.DataFrame([
        {"Experiment": "prod-baseline-v1", "Recall": 0.88, "Precision": 0.74, "Faithfulness": 0.94, "Composite": 0.86, "Status": "BASELINE"},
        {"Experiment": "prompt-exp-v2 (PR #402)", "Recall": 0.88, "Precision": 0.74, "Faithfulness": 0.72, "Composite": 0.77, "Status": "BLOCKED (-18% Faithfulness)"},
        {"Experiment": "rerank-exp-v3 (PR #408)", "Recall": 0.96, "Precision": 0.91, "Faithfulness": 0.95, "Composite": 0.94, "Status": "PASSED (+8% Composite)"}
    ])
    st.dataframe(exp_compare_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("6. Production Implementation: Automated RAG Triad Evaluator (RAGEvaluationPipelineEngine)", divider="rainbow")

st.markdown("""
Below is the complete production evaluation engine `RAGEvaluationPipelineEngine`.
It implements rank-weighted context precision, context recall against golden doc IDs,
faithfulness grounding checks, and composite score benchmarking in pure Python.
""")

st.code('''
class RAGEvaluationPipelineEngine:
    """
    Automated RAG Triad & End-to-End Evaluation Engine.
    Emulates LangSmith custom evaluators across Context Recall,
    Context Precision, Faithfulness (Grounding), and Answer Relevance.
    """
    def __init__(self, experiment_name: str = "rag-triad-benchmark-v1"):
        self.experiment_name = experiment_name
        self.evaluation_results: List[Dict[str, Any]] = []

    def evaluate_rag_run(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str,
        expected_doc_ids: List[str],
        retrieved_doc_ids: List[str]
    ) -> Dict[str, Any]:
        # 1. Context Recall
        hits = sum(1 for doc_id in expected_doc_ids if doc_id in retrieved_doc_ids)
        context_recall = round(hits / max(1, len(expected_doc_ids)), 2)

        # 2. Context Precision
        precisions = []
        running_hits = 0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in expected_doc_ids:
                running_hits += 1
                precisions.append(running_hits / rank)
        context_precision = round(sum(precisions) / max(1, len(precisions)), 2) if precisions else 0.0

        # 3. Faithfulness (Grounding in context)
        combined_ctx = " ".join(retrieved_contexts).lower()
        ans_tokens = set(generated_answer.lower().split())
        stopwords = {"the", "a", "an", "is", "are", "in", "on", "at", "to", "for", "of", "and", "or"}
        salient_ans = ans_tokens - stopwords
        grounded_count = sum(1 for token in salient_ans if token in combined_ctx)
        faithfulness = round(grounded_count / max(1, len(salient_ans)), 2)

        # 4. Answer Relevance
        q_tokens = set(query.lower().split()) - stopwords
        addressed_count = sum(1 for token in q_tokens if token in generated_answer.lower())
        answer_relevance = round(addressed_count / max(1, len(q_tokens)), 2)

        # 5. Composite Score
        composite_score = round(
            (context_recall * 0.25) +
            (context_precision * 0.20) +
            (faithfulness * 0.35) +
            (answer_relevance * 0.20),
            2
        )
        status = "PASSED" if composite_score >= 0.75 else ("MARGINAL" if composite_score >= 0.50 else "FAILED")

        return {
            "context_recall": context_recall,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "composite_score": composite_score,
            "status": status
        }
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
rag_eval_inst = RAGEvaluationPipelineEngine()

eval_test_res = rag_eval_inst.evaluate_rag_run(
    query="What is the maximum worker pod density?",
    retrieved_contexts=["Kubernetes worker node limits support up to 110 pods per node."],
    generated_answer="The maximum worker pod density is 110 pods per node.",
    ground_truth_answer="Worker nodes support 110 pods.",
    expected_doc_ids=["doc_k8s_limits_04"],
    retrieved_doc_ids=["doc_k8s_limits_04", "doc_storage_01"]
)

assert eval_test_res["context_recall"] == 1.0, "Expected doc_id was retrieved; recall must be 1.0"
assert eval_test_res["context_precision"] == 1.0, "Expected doc_id was at Rank 1; precision must be 1.0"
assert eval_test_res["faithfulness"] >= 0.80, "Generated answer is grounded in context; faithfulness must be high"
assert eval_test_res["status"] == "PASSED", "High-accuracy run must PASS"

st.success("Automated Verification Passed: RAGEvaluationPipelineEngine successfully verified across Context Recall, Precision, Faithfulness, and Composite Scoring.")

st.divider()

# ==============================================================================
# SECTION 7: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("7. Interactive Workbench: Live RAG Triad Evaluation Suite & Experiment Benchmark Simulator", divider="rainbow")

st.markdown("""
Benchmark RAG performance across custom inputs.
Adjust the generated answer, test ungrounded hallucinations, simulate rank inversion, and observe real-time RAG Triad scorecard feedback.
""")

wb_e1, wb_e2 = st.columns([1, 1])

with wb_e1:
    st.markdown("#### Test Query & Ground Truth Definition")
    in_eval_query = st.text_input("User Test Query", value="What is the enterprise cloud backup retention window?")
    in_ground_truth = st.text_input("Golden Ground-Truth Answer", value="Enterprise backups are retained for 90 calendar days.")
    in_expected_docs = st.text_input("Expected Ground-Truth Doc IDs (comma-separated)", value="doc_backup_policy_01")

with wb_e2:
    st.markdown("#### Retrieved Context & Generated Response")
    in_retrieved_ctx = st.text_area(
        "Retrieved Context Chunks (Text):",
        value="Cloud Disaster Recovery Policy: Automated daily backups are archived to cold storage and retained for 90 calendar days."
    )
    in_retrieved_ids = st.text_input("Retrieved Doc IDs in Rank Order (comma-separated)", value="doc_backup_policy_01, doc_faq_02")
    in_candidate_ans = st.text_area(
        "Candidate LLM Generated Answer:",
        value="Enterprise cloud backups are retained for 90 calendar days in cold storage."
    )

exp_docs_list = [d.strip() for d in in_expected_docs.split(",") if d.strip()]
ret_docs_list = [d.strip() for d in in_retrieved_ids.split(",") if d.strip()]

live_eval_out = rag_eval_inst.evaluate_rag_run(
    query=in_eval_query,
    retrieved_contexts=[in_retrieved_ctx],
    generated_answer=in_candidate_ans,
    ground_truth_answer=in_ground_truth,
    expected_doc_ids=exp_docs_list,
    retrieved_doc_ids=ret_docs_list
)

st.markdown("### LangSmith Experiment Scorecard Results")

em_1, em_2, em_3, em_4, em_5 = st.columns(5)
em_1.metric("Context Recall", f"{live_eval_out['context_recall'] * 100}%")
em_2.metric("Context Precision", f"{live_eval_out['context_precision'] * 100}%")
em_3.metric("Faithfulness (Grounding)", f"{live_eval_out['faithfulness'] * 100}%")
em_4.metric("Answer Relevance", f"{live_eval_out['answer_relevance'] * 100}%")
em_5.metric("Composite Verdict", live_eval_out["status"], delta=f"Score: {live_eval_out['composite_score']}")

st.markdown("#### Scorecard Metric Evaluation Breakdown")
scorecard_table = [
    {"Evaluation Metric": "Context Recall", "Score": live_eval_out["context_recall"], "Target Threshold": ">= 0.80", "Status": "PASS" if live_eval_out["context_recall"] >= 0.80 else "FAIL"},
    {"Evaluation Metric": "Context Precision", "Score": live_eval_out["context_precision"], "Target Threshold": ">= 0.70", "Status": "PASS" if live_eval_out["context_precision"] >= 0.70 else "FAIL"},
    {"Evaluation Metric": "Faithfulness (Grounding)", "Score": live_eval_out["faithfulness"], "Target Threshold": ">= 0.85", "Status": "PASS" if live_eval_out["faithfulness"] >= 0.85 else "FAIL"},
    {"Evaluation Metric": "Answer Relevance", "Score": live_eval_out["answer_relevance"], "Target Threshold": ">= 0.75", "Status": "PASS" if live_eval_out["answer_relevance"] >= 0.75 else "FAIL"},
    {"Evaluation Metric": "Overall Composite", "Score": live_eval_out["composite_score"], "Target Threshold": ">= 0.75", "Status": live_eval_out["status"]}
]
st.dataframe(pd.DataFrame(scorecard_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What are the three pillars of the RAG Triad, and which pipeline components does each isolate?**"):
    st.markdown("""
    **Answer**:
    The RAG Triad isolates the three foundational interfaces of RAG:
    1. **Context Relevance / Precision** (Query -> Context): Measures whether the retrieved documents are topically focused without noise. Isolates the retriever and reranker.
    2. **Groundedness / Faithfulness** (Context -> Answer): Measures whether the generated answer is strictly supported by the retrieved context. Isolates the LLM generator for hallucinations.
    3. **Answer Relevance** (Query -> Answer): Measures whether the generated response directly answers the user's initial question. Isolates instruction following and prompt alignment.
    """)

with st.expander("**Q2: How does LangSmith's client.evaluate() integrate into continuous CI/CD pipelines?**"):
    st.markdown("""
    **Answer**:
    In a modern GitOps workflow:
    1. Every pull request modifying prompt templates, chunkers, or models triggers a GitHub Action workflow.
    2. The workflow executes `client.evaluate(rag_chain, data="golden-test-dataset-v2", evaluators=[...])`.
    3. LangSmith runs the evaluation, computes average benchmark scores, and compares them against the production baseline tag (`#prod_baseline`).
    4. If any core metric (e.g., Faithfulness) drops by more than 5%, the script exits with non-zero status, blocking the pull request automatically.
    """)

with st.expander("**Q3: What is the difference between Context Recall and Context Precision in RAG evaluation?**"):
    st.markdown("""
    **Answer**:
    * **Context Recall** measures *completeness*: did the retriever return all documents necessary to synthesize the complete answer? (Binary presence: hit vs miss).
    * **Context Precision** measures *ranking quality*: are the relevant documents positioned at the top ranks (Rank 1 and 2) or buried at the bottom (Rank 9 and 10)?
    * A system can have 100% Context Recall (all facts retrieved) but very poor Context Precision (facts buried under 8 noise chunks), causing the LLM to hallucinate due to the Lost-in-the-Middle attention deficit.
    """)

with st.expander("**Q4: Why is LLM-as-a-Judge preferred over classical metrics like BLEU or ROUGE for RAG answer evaluation?**"):
    st.markdown("""
    **Answer**:
    BLEU and ROUGE rely on strict n-gram lexical overlap.
    If the ground-truth is *"The policy duration is 90 days"* and the LLM responds *"Coverage extends for approximately three months"*, BLEU/ROUGE assign a near-zero score despite the fact that the answer is semantically 100% correct.
    LLM-as-a-Judge utilizes language models prompted with structured rubrics, accurately assessing semantic equivalence, factual entailment, and tone regardless of lexical variations.
    """)

with st.expander("**Q5: How do you build a high-quality Golden Dataset for LangSmith evaluation without manual annotation of thousands of rows?**"):
    st.markdown("""
    **Answer**:
    * **Synthetic Dataset Generation (RAGAS / LangSmith)**:
      1. Pass document chunks to a powerful model (e.g., Claude 3.5 Sonnet / GPT-4o) with prompts instructed to generate 3 diverse questions: a direct factoid, a multi-hop reasoning question, and a conditional question.
      2. The model generates the synthetic question, extracts the ground-truth answer, and tags the exact chunk ID.
      3. Filter generated pairs using automated consistency checks, followed by human spot-checking of a 10% sample.
    """)

st.markdown("### Production Traps & Failure Modes")

etr_col1, etr_col2 = st.columns(2)

with etr_col1:
    st.error("""
    **Trap 1: Evaluating Only End-to-End Answers (Skipping Retrieval Evaluation)**
    * **Failure**: Measuring only final answer correctness without logging context recall or context precision.
    * **Consequence**: When an answer is incorrect, engineers waste days tweaking system prompts when the root problem was actually a broken vector retriever that never retrieved the document.
    * **Remedy**: Always evaluate the retrieval stage independently using Context Recall and Context Precision.
    """)

    st.error("""
    **Trap 2: Subjective Manual 'Eyeball' Testing of Prompts**
    * **Failure**: Tweaking prompt templates and manually testing 2-3 sample questions in a web UI before deploying to production.
    * **Consequence**: The modification fixes the 2 test queries but silently breaks 25% of existing user queries (regression blindness).
    * **Remedy**: Never deploy prompt changes without running automated evaluation against a versioned Golden Dataset in LangSmith.
    """)

with etr_col2:
    st.warning("""
    **Trap 3: Over-Relying on Uncalibrated LLM Judges**
    * **Failure**: Using a single zero-shot prompt with an LLM judge without validating inter-rater agreement against human annotators.
    * **Consequence**: The judge exhibits self-enhancement bias (favoring answers generated by its own model family) or assigns inconsistent scores.
    * **Remedy**: Provide few-shot examples and explicit multi-point rubrics in the judge prompt, calibrating against a human-scored gold standard.
    """)

    st.warning("""
    **Trap 4: Stale Evaluation Datasets Diverging from Production Query Traffic**
    * **Failure**: Evaluating against a synthetic dataset created 6 months ago while production users ask about new products.
    * **Consequence**: Benchmark scores look pristine (98%), but user satisfaction plummets because test questions no longer reflect real user intent.
    * **Remedy**: Continuously sample real user production traces (via LangSmith annotation queues) to refresh and expand the Golden Dataset.
    """)

st.markdown("### Core Technical Flashcards")

fce_c1, fce_c2 = st.columns(2)

with fce_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the RAG Triad framework?")
        st.markdown("**Back**: An evaluation paradigm assessing the 3 core relationships in RAG: 1. Context Precision (Query to Context), 2. Faithfulness/Grounding (Context to Answer), and 3. Answer Relevance (Query to Answer).")

    with st.container(border=True):
        st.markdown("**Front**: What does Faithfulness measure in RAG evaluation?")
        st.markdown("**Back**: The extent to which all factual claims in the generated answer are grounded in the retrieved context chunks, identifying model hallucinations.")

with fce_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Context Precision rank-weighted?")
        st.markdown("**Back**: Because relevant chunks positioned at top ranks (Rank 1-2) are far more useful to the LLM generator than relevant chunks buried under irrelevant noise at lower ranks.")

    with st.container(border=True):
        st.markdown("**Front**: What is the role of Golden Datasets in LangSmith?")
        st.markdown("**Back**: Immutable reference datasets containing validated input queries, expected document IDs, and reference ground-truth answers, serving as the benchmark standard for automated regression testing.")

st.divider()

# ==============================================================================
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of RAG Evaluation & LangSmith Benchmarking principles:")

e_chk1 = st.checkbox("I can define and calculate the 3 pillars of the RAG Triad (Recall, Precision, Faithfulness).", value=True)
e_chk2 = st.checkbox("I understand why Context Recall and Context Precision must be evaluated independently of generation.", value=True)
e_chk3 = st.checkbox("I know how to implement custom evaluators and log feedback metrics via client.create_feedback().", value=True)
e_chk4 = st.checkbox("I understand how to automate regression testing in CI/CD pipelines to block degraded prompt pull requests.", value=True)
e_chk5 = st.checkbox("I can differentiate between lexical metrics (BLEU/ROUGE) and semantic LLM-as-a-judge evaluators.", value=True)
e_chk6 = st.checkbox("I know how to continuously curate and update versioned Golden Datasets using production trace sampling.", value=True)

if e_chk1 and e_chk2 and e_chk3 and e_chk4 and e_chk5 and e_chk6:
    st.success("Mastery Certified: You have mastered RAG evaluation, RAG Triad metric calculation, and automated LangSmith experiment benchmarking.")
