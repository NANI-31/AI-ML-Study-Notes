import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="9.3 Similarity Thresholds - Embeddings Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("9.3 Similarity Thresholds: Quality Gating, Precision-Recall Tuning, & Dynamic Cutoffs")
st.caption("Chapter 9: Semantic Similarity | Domain: 3. Vector Embeddings")
st.markdown("""
In naive vector search, systems rely strictly on **Top-K Retrieval**: returning the top 3 or top 5 nearest neighbors regardless of their actual similarity score.
This creates a critical vulnerability in enterprise RAG and search:
* If an employee asks: *"What is the enterprise policy on time travel?"*
* Top-K search will still faithfully retrieve 3 documents—perhaps a vacation policy, an engineering road map, and a science fiction blog post—simply because they were the closest vectors in the database!
* Passing these low-similarity distractors into an LLM causes catastrophic **hallucinations and false claims**.

A **Similarity Threshold (tau)** serves as a strict mathematical quality gate:
`Retrieved Documents = { doc | Cosine_Similarity(query, doc) >= tau }`

Selecting the optimal threshold requires balancing **Precision** (eliminating irrelevant noise) and **Recall** (retaining valid documents with divergent wording).

This lesson explores score tiering, empirical threshold calibration, and dynamic score-drop filtering.
""")

st.divider()

# Section 1: The Top-K Fallacy & Threshold Quality Gating
st.header("1. The Top-K Fallacy & Threshold Quality Gating", divider="rainbow")

col_th1, col_th2 = st.columns([1, 1])

with col_th1:
    st.markdown("""
    ### Why Top-K Retrieval Requires Threshold Gating:
    * **The Irrelevant Query Trap**:
      * Top-K answers the question: *"Which K documents are closest?"*
      * A Similarity Threshold answers the question: *"Are any of these documents actually relevant?"*
      * Without a threshold gate, out-of-domain queries inject garbage context into generative LLMs.
    * **Empirical Score Tiers for Dense Embeddings**:
      * **Tier 1 (tau >= 0.85) — High Confidence**:
        * Direct paraphrases, verbatim entity matches, or concise factual definitions.
        * Safe for automated zero-shot actions and direct answer extraction.
      * **Tier 2 (0.72 <= tau < 0.85) — Moderate Relevance**:
        * Relevant passages, complementary explanations, or related policies.
        * Excellent candidate pool for cross-encoder reranking.
      * **Tier 3 (tau < 0.72) — Noise & Topic Drift**:
        * Generic domain overlap, coincidental vocabulary, or unrelated chatter.
        * Should be filtered out to prevent prompt pollution.
    """)

with col_th2:
    st.markdown("### Gating Architecture: Top-K vs. Threshold Filter")
    th_dot = """
    digraph ThresholdGating {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        UserQuery [label="User Query\\n'Out-of-domain question'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        VectorSearch [label="Vector DB Top-K Search (K=3)\\nDoc 1 (Score: 0.54)\\nDoc 2 (Score: 0.48)\\nDoc 3 (Score: 0.32)", fillcolor="#FEF3C7"];
        
        ThresholdGate [label="Threshold Gate (tau = 0.75)\\nFilters all docs with score < 0.75", shape=diamond, fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];
        
        FilteredOut [label="Safe Fallback Response\\n'No relevant documents found'\\nZero hallucinations!", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        
        PromptPollution [label="Without Gate: Naive LLM Injection\\nProduces massive hallucinations!", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];

        UserQuery -> VectorSearch -> ThresholdGate;
        ThresholdGate -> FilteredOut [label="All scores < 0.75"];
        VectorSearch -> PromptPollution [label="If ungated", style=dashed, color="#EF4444"];
    }
    """
    st.graphviz_chart(th_dot, use_container_width=False)

st.markdown("### Threshold Tradeoff Matrix")
st.markdown("""
| Threshold Level (tau) | Precision (Noise Elimination) | Recall (Coverage) | Risk of Hallucination | Primary Deployment Scenario |
| :--- | :--- | :--- | :--- | :--- |
| **Very High (tau >= 0.88)** | **Near 100%** | Low (drops paraphrases) | Extremely Low | Automated financial/medical compliance |
| **Balanced (tau = 0.75 - 0.82)** | **High (~85%)** | **High (~85%)** | Low | **Standard Enterprise RAG Pipelines** |
| **Permissive (tau = 0.65 - 0.72)** | Moderate (~65%) | Near 100% | Moderate | Pre-retrieval candidate generation for Rerankers |
| **No Threshold (tau = 0.00)** | Very Low | 100% | **Severe** | Naive baseline (NOT recommended for production) |
""")

st.divider()

# Section 2: Production Code: Dynamic Threshold Filter Engine
st.header("2. Production Implementation: Dynamic Threshold Filtering Engine", divider="rainbow")

st.markdown("""
The production utility below implements both **Static Threshold Gating** and **Dynamic Score-Drop Filtering** (retaining documents only if their score is within a relative margin of the top match).
""")

st.code(r'''
from typing import List, Dict, Any, Optional

class ThresholdFilterEngine:
    """
    Production-grade similarity gating utility.
    Applies absolute threshold gates and dynamic relative margin cutoffs.
    """
    def __init__(self, static_threshold: float = 0.75, dynamic_margin: float = 0.12):
        self.static_threshold = static_threshold
        self.dynamic_margin = dynamic_margin

    def filter_results(
        self, 
        candidates: List[Dict[str, Any]], 
        use_dynamic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Filters candidate retrieval hits based on similarity thresholds.
        candidates: list of dicts with 'doc_id', 'text', 'score' sorted descending.
        """
        if not candidates:
            return []

        # 1. Apply absolute minimum threshold gate
        gated = [c for c in candidates if c["score"] >= self.static_threshold]
        if not gated:
            return []

        # 2. Optional: Apply dynamic relative margin filter
        # If top hit is 0.94 and margin is 0.12, cutoff is 0.94 - 0.12 = 0.82
        if use_dynamic and len(gated) > 1:
            top_score = gated[0]["score"]
            cutoff = top_score - self.dynamic_margin
            gated = [c for c in gated if c["score"] >= cutoff]

        return gated
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveThresholdSimulator:
    def __init__(self):
        # Ground truth candidate documents for a sample query:
        # Query: "What is the company policy on remote work and home office stipends?"
        self.corpus_pool = [
            {
                "id": "DOC_001",
                "text": "Remote Work Policy: Full-time employees are eligible for hybrid scheduling with a $500 annual home office ergonomic stipend.",
                "score": 0.92,
                "is_actually_relevant": True
            },
            {
                "id": "DOC_002",
                "text": "Home Office Expense Guidelines: Internet reimbursements of up to $75/month are processed via expense portal.",
                "score": 0.86,
                "is_actually_relevant": True
            },
            {
                "id": "DOC_003",
                "text": "Office Ergonomics: Desk chairs and standing converters are available for headquarters employees.",
                "score": 0.77,
                "is_actually_relevant": True
            },
            {
                "id": "DOC_004",
                "text": "Vacation and PTO Policy: Employees accrue 15 days of personal paid time off annually.",
                "score": 0.69,
                "is_actually_relevant": False
            },
            {
                "id": "DOC_005",
                "text": "Company Travel Expenses: Airfare and hotel accommodations must be booked through corporate travel portal.",
                "score": 0.61,
                "is_actually_relevant": False
            },
            {
                "id": "DOC_006",
                "text": "Server Infrastructure: Remote SSH access to production clusters requires hardware YubiKey MFA.",
                "score": 0.52,
                "is_actually_relevant": False
            }
        ]

    def evaluate_threshold(self, tau: float) -> Dict[str, Any]:
        retrieved = [doc for doc in self.corpus_pool if doc["score"] >= tau]
        
        tp = sum(1 for d in retrieved if d["is_actually_relevant"])
        fp = sum(1 for d in retrieved if not d["is_actually_relevant"])
        fn = sum(1 for d in self.corpus_pool if d["is_actually_relevant"] and d not in retrieved)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

        return {
            "retrieved_count": len(retrieved),
            "retrieved_docs": retrieved,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

st.divider()

# Section 3: Interactive Workbench: Precision-Recall Threshold Playground
st.header("3. Interactive Workbench: Precision-Recall Threshold Playground", divider="rainbow")

st.markdown("""
Adjust the **Similarity Threshold (tau)** slider below to observe how the quality gate filters candidate documents for the query:
*"What is the company policy on remote work and home office stipends?"*
Watch how **Precision**, **Recall**, and **F1-Score** react dynamically as you slide between permissive (low tau) and strict (high tau) cutoffs.
""")

th_sim = InteractiveThresholdSimulator()

col_th_l, col_th_r = st.columns([1, 1])

with col_th_l:
    st.subheader("1. Configure Quality Gate")
    tau_val = st.slider("Similarity Threshold (tau):", min_value=0.40, max_value=0.95, value=0.75, step=0.01)
    
    st.markdown("#### Candidate Pool (Ranked by Cosine Score)")
    df_pool = pd.DataFrame([
        {
            "Doc ID": d["id"],
            "Score": d["score"],
            "Ground Truth": "Relevant" if d["is_actually_relevant"] else "Irrelevant Distractor",
            "Snippet": d["text"][:60] + "..."
        }
        for d in th_sim.corpus_pool
    ])
    st.dataframe(df_pool, use_container_width=True)

with col_th_r:
    st.subheader("2. Quality Gate Evaluation")
    eval_res = th_sim.evaluate_threshold(tau_val)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{eval_res['precision']*100:.1f}%")
    m2.metric("Recall", f"{eval_res['recall']*100:.1f}%")
    m3.metric("F1-Score", f"{eval_res['f1']:.3f}")

    st.metric("Documents Passing Gate", f"{eval_res['retrieved_count']} / {len(th_sim.corpus_pool)}")

    st.markdown("#### Retrieved Documents Passing Gate:")
    if eval_res["retrieved_count"] == 0:
        st.warning("⚠️ **Zero Documents Passed**: Threshold is too strict! Fallback handler triggered ('No relevant knowledge base records found').")
    else:
        df_ret = pd.DataFrame([
            {
                "Doc ID": d["id"],
                "Cosine Score": d["score"],
                "Status": "✅ True Positive" if d["is_actually_relevant"] else "❌ False Positive Noise",
                "Text": d["text"]
            }
            for d in eval_res["retrieved_docs"]
        ])
        st.dataframe(
            df_ret.style.format({"Cosine Score": "{:.2f}"}).background_gradient(subset=["Cosine Score"], cmap="Blues"),
            use_container_width=True
        )

    if eval_res['precision'] == 1.0 and eval_res['recall'] == 1.0:
        st.success("🌟 **Optimal Threshold!** 100% Precision and 100% Recall: all relevant documents captured with zero distractor noise.")
    elif eval_res['precision'] < 0.70:
        st.error(f"🚨 **Prompt Pollution Warning**: {eval_res['fp']} irrelevant distractor documents leaked into the prompt (Precision = {eval_res['precision']*100:.1f}%).")
    elif eval_res['recall'] < 0.70:
        st.warning(f"📉 **Low Recall Warning**: {eval_res['fn']} relevant documents were filtered out by the strict threshold (Recall = {eval_res['recall']*100:.1f}%).")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Threshold Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is relying solely on Top-K retrieval without a similarity threshold dangerous for production RAG?**
    * **Answer**:
      * Top-K retrieval guarantees returning K documents regardless of relevance.
      * When a user submits an adversarial prompt, out-of-domain query, or completely irrelevant question, the vector database still returns the K highest-scoring items.
      * Even if these items have low similarity (e.g. 0.40), the LLM prompt is populated with them.
      * The LLM attempts to synthesize an answer from the irrelevant context, leading to hallucinations, user confusion, and security leakage.
      * A similarity threshold acts as an essential circuit breaker: if no document exceeds the threshold, the system returns a safe fallback message (*"No relevant policy found"*).

    **Q2: How do you tune the optimal similarity threshold for a proprietary enterprise dataset?**
    * **Answer**:
      * 1. **Curate an Evaluation Benchmark**: Build a validation set of 200-500 realistic user queries labeled with ground truth relevant document IDs.
      * 2. **Include Negative / Out-of-Scope Queries**: Add 50-100 queries for which NO relevant documents exist in the corpus.
      * 3. **Sweep Threshold Values**: Grid search tau from 0.50 to 0.95 in increments of 0.01.
      * 4. **Calculate F1-Score / Precision@K**: Plot the Precision-Recall curve and select the tau value that maximizes F1-score (or prioritizes precision if hallucination tolerance is zero).

    **Q3: What is Dynamic Relative Thresholding (or Margin Filtering)?**
    * **Answer**:
      * Fixed static thresholds can fail when queries have varying baseline similarities.
      * **Dynamic Relative Filtering**:
        1. Find the top-scoring document ($s_{\\text{max}}$).
        2. Set a relative margin $\\delta$ (e.g. 0.10).
        3. Keep candidate documents only if $s_i \\ge \\max(\\tau_{\\text{static}}, s_{\\text{max}} - \\delta)$.
      * If Doc 1 scores 0.95 and Doc 2 scores 0.72 (a massive 0.23 drop), Doc 2 is dropped even if it passes a static 0.70 threshold, preventing second-rate distractors from polluting context.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Hardcoding Thresholds Across Different Embedding Models**:
      * *Issue*: Tuning tau = 0.82 for `all-MiniLM-L6-v2`, then switching to `text-embedding-3-small` and keeping tau = 0.82.
      * *Consequence*: Different models have different score distributions! A 0.82 in MiniLM might correspond to 0.65 in OpenAI embeddings, filtering out all valid documents.
      * *Fix*: Re-calibrate thresholds whenever updating embedding model architectures.

    * **Trap 2: Ignoring Domain Specificity**:
      * *Issue*: Setting a single global threshold (e.g. 0.80) across both general FAQs and technical legal contracts.
      * *Consequence*: Legal jargon queries with exact phrasing hit 0.90+, while colloquial user questions hit 0.72, creating an uneven user experience.
      * *Fix*: Segment thresholds by corpus category or use dynamic relative margin filtering.

    * **Trap 3: Not Providing a Fallback Route when Gated to Zero Results**:
      * *Issue*: Threshold filters all documents, and code throws an unhandled `IndexError: list index out of range` on `results[0]`.
      * *Consequence*: 500 Internal Server Error in the application.
      * *Fix*: Implement an explicit check: if `len(filtered_docs) == 0`, route to a polite fallback LLM response or web search fallback.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Similarity Threshold (tau)**
      * *Definition*: A minimum cosine similarity score required for a retrieved document to be accepted into downstream processing.
    * **Flashcard 2: Top-K Fallacy**
      * *Definition*: The erroneous assumption that returning the top K nearest neighbors guarantees relevance, ignoring out-of-domain queries.
    * **Flashcard 3: Dynamic Margin Filtering**
      * *Definition*: Filtering documents whose similarity score drops significantly below the top-scoring document's score.
    * **Flashcard 4: Hallucination Circuit Breaker**
      * *Definition*: Using a similarity threshold to trigger safe fallback responses when no relevant knowledge base records exist.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Similarity Thresholds:

- [ ] I can articulate why top-K retrieval without a threshold gate causes LLM hallucinations.
- [ ] I understand the Precision-Recall tradeoff when adjusting similarity threshold values.
- [ ] I know how to benchmark and tune thresholds using F1-score optimization on validation sets.
- [ ] I understand Dynamic Relative Margin filtering and how it eliminates second-rate distractors.
- [ ] I know why similarity thresholds must be re-calibrated whenever switching embedding models.
""")
