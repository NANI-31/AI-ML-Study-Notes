import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Set

st.set_page_config(
    page_title="20.1 Retrieval Evaluation - Embeddings Curriculum",
    page_icon="📊",
    layout="wide"
)

# Title & Overview
st.title("20.1 Retrieval Evaluation: Precision@K, Recall@K, MRR, NDCG, & Hit Rate")
st.caption("Chapter 20: Embedding Evaluation | Domain: 3. Vector Embeddings")
st.markdown("""
In machine learning, models are evaluated with classification accuracy or F1 scores.
In semantic search and vector retrieval, however, simple accuracy is meaningless: users do not care if a relevant document was retrieved at rank 500—they only care if it was in the **top 3 or top 5 results**.

To scientifically benchmark and compare embedding models, Information Retrieval (IR) engineers use a standardized suite of ranking metrics:
* **Precision@K**: What proportion of the top $K$ retrieved documents are relevant?
* **Recall@K**: What proportion of all known relevant documents were captured in the top $K$?
* **Hit Rate (Hit@K)**: Did the top $K$ results contain at least one correct answer?
* **Mean Reciprocal Rank (MRR)**: How high up in the ranking was the very first relevant document?
* **Normalized Discounted Cumulative Gain (NDCG@K)**: Does the model rank the most relevant documents at the very top, penalizing relevant items that appear lower down?

This lesson explores the mathematical formulas, implementation mechanics, and trade-offs across these foundational retrieval metrics.
""")

st.divider()

# Section 1: The Core IR Metric Suite
st.header("1. Mathematical Foundations of Information Retrieval Metrics", divider="rainbow")

col_ir1, col_ir2 = st.columns([1, 1])

with col_ir1:
    st.markdown("""
    ### The 5 Essential Ranking Metrics
    * **1. Precision@K**:
      * Formula: `Precision@K = Count(Relevant docs in top K) / K`
      * Measures signal-to-noise ratio in prompt context. Low precision means prompt is flooded with distractors.
    * **2. Recall@K**:
      * Formula: `Recall@K = Count(Relevant docs in top K) / Total Relevant docs in corpus`
      * Measures coverage. Low recall means critical facts were omitted.
    * **3. Hit Rate@K (Hit@K)**:
      * Formula: `1 if Count(Relevant docs in top K) >= 1 else 0`
      * Common in single-answer question-answering systems (e.g. customer support FAQs).
    * **4. Mean Reciprocal Rank (MRR)**:
      * Formula: `RR = 1 / Rank of first relevant document`
      * `MRR = Average(RR across all test queries)`
      * If first hit is at Rank 1, RR = 1.0. At Rank 2, RR = 0.5. At Rank 5, RR = 0.2.
    * **5. NDCG@K (Normalized Discounted Cumulative Gain)**:
      * Evaluates graded relevance (e.g. 3 = Perfect, 2 = Good, 1 = Minor, 0 = Irrelevant).
      * Applies logarithmic position discounting: `DCG@K = sum((2^rel_i - 1) / log2(i + 1))`
      * `NDCG@K = DCG@K / IDCG@K` (normalized against ideal perfect sorting).
    """)

with col_ir2:
    st.markdown("### Ranking Sensitivity & Position Discounting")
    rank_dot = """
    digraph RankingSensitivity {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_model_a {
            label="Model A: High NDCG & MRR"; style="dashed"; color="#10B981";
            A1 [label="Rank 1: Relevant Doc\\n(Score: 1 / log2(2) = 1.0)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            A2 [label="Rank 2: Distractor", fillcolor="#F3F4F6"];
            A3 [label="Rank 3: Distractor", fillcolor="#F3F4F6"];
            MRR_A [label="MRR = 1.00 | NDCG@3 = 1.00", fillcolor="#A7F3D0", style="filled,bold"];
        }

        subgraph cluster_model_b {
            label="Model B: Low NDCG & MRR"; style="dashed"; color="#EF4444";
            B1 [label="Rank 1: Distractor", fillcolor="#F3F4F6"];
            B2 [label="Rank 2: Distractor", fillcolor="#F3F4F6"];
            B3 [label="Rank 3: Relevant Doc\\n(Score: 1 / log2(4) = 0.50)", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
            MRR_B [label="MRR = 0.33 | NDCG@3 = 0.46", fillcolor="#FCA5A5", style="filled,bold"];
        }

        QueryIn [label="Identical Query & Identical Recall (1/1)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        QueryIn -> A1;
        QueryIn -> B1;
        A1 -> MRR_A;
        B3 -> MRR_B;
    }
    """
    st.graphviz_chart(rank_dot, use_container_width=False)

st.markdown("### Comprehensive IR Metric Comparison Matrix")
st.markdown("""
| Metric | Range | Position Sensitive? | Graded Relevance? | Best Evaluated For |
| :--- | :--- | :--- | :--- | :--- |
| **Precision@K** | [0.0, 1.0] | No (treats all top $K$ equally) | Binary (0 or 1) | Prompt token budget efficiency. |
| **Recall@K** | [0.0, 1.0] | No (treats all top $K$ equally) | Binary (0 or 1) | Multi-part research questions. |
| **Hit Rate@K** | [0, 1] | No (binary pass/fail) | Binary (0 or 1) | Single-needle FAQ lookup. |
| **MRR** | [0.0, 1.0] | **Yes (1 / rank)** | Binary (First hit only) | Search engines, conversational assistants. |
| **NDCG@K** | [0.0, 1.0] | **Yes (Logarithmic discount)** | **Yes (0, 1, 2, 3)** | E-commerce, multi-level relevance ranking. |
""")

st.divider()

# Section 2: Production Code: Pure-Python IR Metrics Engine
st.header("2. Production Code: Pure-Python Information Retrieval Metrics Engine", divider="rainbow")

st.markdown("""
The production utility below implements all 5 core IR metrics in pure Python without external library dependencies: computing Precision, Recall, MRR, NDCG, and Hit Rate over ranked document lists.
""")

st.code(r'''
import math
from typing import List, Set, Dict, Any

class RetrievalMetricsEngine:
    """
    Computes foundational Information Retrieval (IR) metrics:
    Precision@K, Recall@K, HitRate@K, MRR, and NDCG@K.
    """
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
        return relevant_in_k / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
        return relevant_in_k / float(len(ground_truth_ids))

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in ground_truth_ids for doc_id in top_k) else 0.0

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in ground_truth_ids:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculates NDCG@K supporting graded relevance (0, 1, 2, 3).
        """
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0

        # 1. Compute Discounted Cumulative Gain (DCG@K)
        dcg = 0.0
        for i, doc_id in enumerate(top_k, 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        # 2. Compute Ideal DCG (IDCG@K)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, 1):
            idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
''', language="python")

st.divider()

# Section 3: Interactive IR Benchmark Arena
st.header("3. Interactive Workbench: Retrieval Ranking Simulator & Benchmark Arena", divider="rainbow")

st.markdown("""
Experiment with different ranking outcomes for a simulated retrieval query.
See how changing the rank order of relevant documents impacts **MRR**, **NDCG**, **Precision**, and **Recall**.
""")

class InteractiveIREngine:
    @staticmethod
    def calc_metrics(retrieved: List[str], relevant_set: Set[str], graded_dict: Dict[str, float], k: int):
        # Precision@K
        hits_k = sum(1 for d in retrieved[:k] if d in relevant_set)
        p_k = hits_k / k
        # Recall@K
        r_k = hits_k / len(relevant_set) if relevant_set else 0.0
        # Hit Rate@K
        hr_k = 1.0 if hits_k > 0 else 0.0
        # MRR
        rr = 0.0
        for idx, d in enumerate(retrieved, 1):
            if d in relevant_set:
                rr = 1.0 / idx
                break
        # NDCG@K
        dcg = sum([(2**graded_dict.get(d, 0.0) - 1.0) / np.log2(i + 1) for i, d in enumerate(retrieved[:k], 1)])
        ideal_rels = sorted(graded_dict.values(), reverse=True)[:k]
        idcg = sum([(2**rel - 1.0) / np.log2(i + 1) for i, rel in enumerate(ideal_rels, 1)])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return p_k, r_k, hr_k, rr, ndcg

ir_engine = InteractiveIREngine()

# Benchmark scenario definition
benchmark_query = "What is the policy for bereavement leave?"
ground_truth_map = {
    "doc_01": 3.0, # Perfect match (Bereavement Policy)
    "doc_02": 2.0, # Good match (General Compassionate Leave)
    "doc_03": 0.0, # Distractor (Annual Vacation)
    "doc_04": 0.0, # Distractor (Health Insurance)
    "doc_05": 0.0, # Distractor (401k Retirement)
    "doc_06": 0.0  # Distractor (Remote Work Protocol)
}
relevant_keys = {"doc_01", "doc_02"}

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("1. Benchmark Query & Ground Truth")
    st.markdown(f"**Query:** *{benchmark_query}*")
    st.markdown("**Ground-Truth Relevant Documents (Total: 2):**")
    st.write("• `doc_01`: Explicit Bereavement Policy (Relevance Grade: **3**)")
    st.write("• `doc_02`: Compassionate Leave SOP (Relevance Grade: **2**)")

    st.subheader("2. Select Model Ranking System")
    system_choice = st.radio(
        "Choose Ranking Profile to Benchmark:",
        [
            "System A (Ideal SOTA Retriever: Docs at Rank 1 & 2)",
            "System B (Suboptimal Retriever: First Hit at Rank 3, Second at Rank 5)",
            "System C (Degraded Retriever: Only 1 Hit at Rank 6)"
        ],
        index=0
    )

    k_eval = st.slider("Evaluation Cutoff ($K$):", min_value=1, max_value=6, value=3)

    if "System A" in system_choice:
        simulated_ranked_list = ["doc_01", "doc_02", "doc_03", "doc_04", "doc_05", "doc_06"]
    elif "System B" in system_choice:
        simulated_ranked_list = ["doc_03", "doc_04", "doc_01", "doc_05", "doc_02", "doc_06"]
    else:
        simulated_ranked_list = ["doc_03", "doc_04", "doc_05", "doc_06", "doc_01", "doc_02"]

with col_sim2:
    st.subheader(f"3. Retrieval Performance Dashboard (K = {k_eval})")

    p_val, r_val, hr_val, mrr_val, ndcg_val = ir_engine.calc_metrics(
        simulated_ranked_list,
        relevant_keys,
        ground_truth_map,
        k_eval
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Precision@{k_eval}", f"{p_val:.2f}", help="Relevant docs / K")
    m2.metric(f"Recall@{k_eval}", f"{r_val:.2f}", help="Relevant retrieved / Total relevant (2)")
    m3.metric(f"Hit Rate@{k_eval}", f"{hr_val:.1f}", help="1 if at least one relevant doc retrieved")

    m4, m5, m6 = st.columns(3)
    m4.metric("MRR", f"{mrr_val:.3f}", help="Reciprocal rank of first relevant doc")
    m5.metric(f"NDCG@{k_eval}", f"{ndcg_val:.3f}", help="Normalized Discounted Cumulative Gain")
    first_rank = int(1.0 / mrr_val) if mrr_val > 0 else "None"
    m6.metric("First Hit Rank", f"Rank {first_rank}" if mrr_val > 0 else "Missed")

    st.markdown("#### Retrieved Rank Position Breakdown")
    breakdown_rows = []
    for rank, did in enumerate(simulated_ranked_list[:k_eval], 1):
        rel = ground_truth_map.get(did, 0.0)
        is_rel = did in relevant_keys
        breakdown_rows.append({
            "Rank": f"#{rank}",
            "Doc ID": did,
            "Is Relevant?": "✅ Yes" if is_rel else "❌ Distractor",
            "Grade": rel,
            "Discount 1/log2(i+1)": f"{1.0 / np.log2(rank + 1):.3f}"
        })
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Evaluation Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is MRR (Mean Reciprocal Rank) often preferred over Precision@K for search engines and virtual assistants?**
    * **Answer**: In web search or conversational Q&A, users rarely inspect multiple documents; they click or read the **very first relevant result**.
    * If Model A puts the correct answer at Rank 1, its Reciprocal Rank is `1 / 1 = 1.0`.
    * If Model B puts the answer at Rank 4, its Reciprocal Rank drops to `1 / 4 = 0.25`.
    MRR directly reflects user satisfaction by heavily rewarding models that surface the correct answer at the very top of the list.

    **Q2: What is the key advantage of NDCG over binary metrics like Recall@K?**
    * **Answer**: Binary metrics treat relevance as a black-and-white 0 or 1. In reality, some documents are *perfect authoritative answers* (Grade 3), others are *partially helpful* (Grade 1), and others are *irrelevant* (Grade 0).
    **NDCG** handles graded relevance and incorporates **positional discounting**, rewarding algorithms that place Grade 3 documents higher than Grade 1 documents.

    **Q3: What does a high Recall@K combined with a low Precision@K indicate in a RAG system?**
    * **Answer**: It means that the retriever successfully captured the necessary information (the needle), but brought along a large amount of irrelevant distractor noise (the haystack).
    * **Impact**: Increases LLM prompt latency, consumes token budget, and raises the risk of the **Lost-in-the-Middle** hallucination effect.
    * **Fix**: Decrease $K$, implement a Cross-Encoder Reranker, or apply dynamic similarity thresholding.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Evaluating Only at Large $K$ ($K=50$)**:
      * *Issue*: Reporting Recall@50 as 95% and declaring the retrieval system solved.
      * *Consequence*: You cannot pass 50 chunks into an LLM without catastrophic prompt cost, latency, and hallucination.
      * *Fix*: Always benchmark at realistic LLM prompt cutoffs: $K=3$, $K=5$, and $K=10$.

    * **Trap 2: Ignoring Tie-Breaking in Graded NDCG Calculation**:
      * *Issue*: Comparing models when multiple documents share identical cosine similarity scores.
      * *Consequence*: Arbitrary sorting order causes NDCG to fluctuate wildly on identical benchmark runs.
      * *Fix*: Implement deterministic secondary sorting (e.g. by document ID) for tie-breaks.

    * **Trap 3: Unbalanced Benchmark Query Sets**:
      * *Issue*: Evaluating a model on 100 queries where 90 queries test common questions and only 10 test difficult edge cases.
      * *Consequence*: High overall MRR masks catastrophic failure on critical edge cases.
      * *Fix*: Segment evaluation across distinct difficulty tiers (Easy, Medium, Hard/Adversarial).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Precision@K**
      * *Definition*: The percentage of the top $K$ retrieved documents that are actually relevant to the query.
    * **Flashcard 2: Recall@K**
      * *Definition*: The percentage of all known relevant documents in the entire corpus that were successfully retrieved in the top $K$.
    * **Flashcard 3: Mean Reciprocal Rank (MRR)**
      * *Definition*: The average across all test queries of the reciprocal rank `1 / rank` of the first relevant document.
    * **Flashcard 4: NDCG (Normalized Discounted Cumulative Gain)**
      * *Definition*: A position-sensitive metric evaluating graded relevance normalized against the theoretically ideal sorting order.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Retrieval Evaluation:

- [ ] I can calculate Precision@K, Recall@K, and Hit Rate@K by hand.
- [ ] I understand the mathematical formulation and logarithmic discounting of NDCG.
- [ ] I know why MRR is ideal for single-turn Q&A and search engines.
- [ ] I can diagnose RAG issues when Recall is high but Precision is low.
- [ ] I understand why retrieval metrics must be evaluated at small values of $K$ ($K=3, 5$).
""")
