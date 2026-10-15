import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="17.2 Retrieval Metrics - RAG Master Curriculum",
    page_icon="🎯",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class RetrievalMetricsEngine:
    """
    Production-grade Information Retrieval (IR) Evaluation Engine:
    1. Precision & Precision@K (Purity and noise suppression).
    2. Recall & Recall@K (Coverage of ground-truth gold documents).
    3. Hit Rate@K (Binary success indicator).
    4. Mean Reciprocal Rank (MRR) calculation.
    5. Discounted Cumulative Gain (DCG) and Normalized DCG (NDCG) with graded relevance.
    """
    def __init__(self):
        pass

    def precision_at_k(self, retrieved: List[str], relevant_gold: set, k: int) -> float:
        """Computes Precision@K: count of relevant docs in top-k / k."""
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant_gold)
        return float(hits / k)

    def recall_at_k(self, retrieved: List[str], relevant_gold: set, k: int) -> float:
        """Computes Recall@K: count of relevant docs in top-k / total gold docs."""
        if not relevant_gold or k <= 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant_gold)
        return float(hits / len(relevant_gold))

    def hit_rate_at_k(self, retrieved: List[str], relevant_gold: set, k: int) -> float:
        """Computes Hit Rate@K: 1.0 if any relevant doc is in top-k, else 0.0."""
        top_k = retrieved[:k]
        return 1.0 if any(doc in relevant_gold for doc in top_k) else 0.0

    def reciprocal_rank(self, retrieved: List[str], relevant_gold: set) -> float:
        """Computes Reciprocal Rank (RR): 1 / rank of first relevant doc (1-indexed)."""
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant_gold:
                return float(1.0 / rank)
        return 0.0

    def dcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Computes Discounted Cumulative Gain at K:
        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
        """
        top_k = retrieved[:k]
        dcg = 0.0
        for i, doc in enumerate(top_k, start=1):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / np.log2(i + 1)
        return float(dcg)

    def ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Computes Normalized Discounted Cumulative Gain at K:
        NDCG@K = DCG@K / IDCG@K
        """
        actual_dcg = self.dcg_at_k(retrieved, relevance_scores, k)
        # Ideal DCG: sorted in descending order of true relevance
        ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
        ideal_dcg = self.dcg_at_k(ideal_order, relevance_scores, k)
        return float(actual_dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0

    def evaluate_retrieval_run(
        self,
        retrieved_ranked: List[str],
        gold_set: set,
        graded_scores: Dict[str, float],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Runs a complete evaluation suite across multiple K cutoffs."""
        metrics_by_k = []
        for k in k_values:
            metrics_by_k.append({
                "K Cutoff": f"Top-{k}",
                "Precision@K": round(self.precision_at_k(retrieved_ranked, gold_set, k), 3),
                "Recall@K": round(self.recall_at_k(retrieved_ranked, gold_set, k), 3),
                "Hit Rate@K": round(self.hit_rate_at_k(retrieved_ranked, gold_set, k), 3),
                "NDCG@K": round(self.ndcg_at_k(retrieved_ranked, graded_scores, k), 3)
            })

        mrr = self.reciprocal_rank(retrieved_ranked, gold_set)
        return {
            "metrics_table": metrics_by_k,
            "mrr": round(mrr, 3),
            "first_hit_rank": int(1.0 / mrr) if mrr > 0.0 else None
        }


# Instantiate engine at module level
ir_engine = RetrievalMetricsEngine()

# --- HEADER SECTION ---
st.title("17.2 Retrieval Metrics: Precision, Recall, Hit Rate, MRR, & NDCG Ranking Science")
st.caption("Chapter 17: RAG Evaluation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In Retrieval-Augmented Generation, the retrieval engine acts as the foundational knowledge bottleneck. 
If your retriever fails to surface gold passages in the Top-K candidates, **no amount of prompt engineering 
or LLM reasoning can salvage the answer**.

Information Retrieval (IR) metrics provide the quantitative science for measuring search quality. 
Unlike subjective qualitative reviews, IR metrics provide deterministic, automated benchmarks across seven foundational dimensions:
1. **Precision**: How much retrieved text is relevant signal vs irrelevant noise?
2. **Recall**: Did the retriever capture all factual evidence required for multi-hop synthesis?
3. **Hit Rate**: Did the candidate pool contain at least one valid answer-bearing passage?
4. **Mean Reciprocal Rank (MRR)**: How high up in the ranking does the first relevant document appear?
5. **Normalized Discounted Cumulative Gain (NDCG)**: Are documents prioritized by their degree of relevance?
6. **Recall@K & Precision@K**: How do coverage and context clutter scale across varying cutoff thresholds?
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: End-to-End Retrieval Evaluation Pipeline", divider="rainbow")

st.markdown("""
The retrieval evaluation pipeline compares an ordered list of retriever hits against human-annotated gold 
documents and graded relevance judgements across multiple Top-K cutoff boundaries.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RetrievalEvalPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // INPUT RETRIEVAL ASSETS
        subgraph cluster_inputs {
            label="1. Retrieval Candidates & Golden References";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            RankedHits [label="Ranked Retrieval Candidates\\n(Doc 1, Doc 2, ..., Doc N)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            GoldJudgments [label="Ground-Truth Gold Standards\\n(Binary Relevance & Graded Scores 0-3)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // METRICS COMPUTATION SUITE
        subgraph cluster_metrics {
            label="2. Quantitative IR Metrics Computation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            PrecRecallComp [label="Precision@K & Recall@K\\n(Signal Purity vs Completeness)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            HitRateComp [label="Hit Rate@K\\n(Binary Success at Cutoff K)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            MrrComp [label="Mean Reciprocal Rank (MRR)\\n(1 / First Relevant Rank)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            NdcgComp [label="NDCG@K\\n(Logarithmic Discounted Cumulative Gain)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // AGGREGATION & REPORTING
        subgraph cluster_output {
            label="3. Diagnostic Telemetry & Gating";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            MetricScorecard [label="Cutoff Sensitivity Scorecard\\n(Top-1, Top-3, Top-5, Top-10)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            GateDecider [label="CI/CD Quality Gate\\n(Passes if Hit Rate@5 >= 0.95 and MRR >= 0.70)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        }

        // EDGES
        RankedHits -> PrecRecallComp;
        GoldJudgments -> PrecRecallComp;
        RankedHits -> HitRateComp;
        GoldJudgments -> HitRateComp;
        RankedHits -> MrrComp;
        GoldJudgments -> MrrComp;
        RankedHits -> NdcgComp;
        GoldJudgments -> NdcgComp;

        PrecRecallComp -> MetricScorecard;
        HitRateComp -> MetricScorecard;
        MrrComp -> MetricScorecard;
        NdcgComp -> MetricScorecard;

        MetricScorecard -> GateDecider;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Metric Relationships:
* **Binary vs Graded**: Hit Rate, Precision, Recall, and MRR treat relevance as binary (`1` or `0`). NDCG evaluates graded relevance (`0 = Irrelevant`, `1 = Marginal`, `2 = Relevant`, `3 = Perfect`).
* **Unranked vs Ranked**: Precision and Recall treat all items in Top-K equally regardless of order. MRR and NDCG reward placing top results at rank 1.
""")

st.divider()

# ==============================================================================
# SECTION 2: PRECISION
# ==============================================================================
st.header("2. Precision (Information Signal, False Positive Cost & Purity)", divider="rainbow")

st.markdown("""
**Precision** measures the proportion of retrieved documents that are truly relevant:
`Precision = Relevant_Retrieved / Total_Retrieved`.
""")

prec_col1, prec_col2 = st.columns([1, 1])

with prec_col1:
    st.markdown("""
    ### Why Precision Dictates RAG Costs:
    * **Context Window Clutter**: If a system retrieves 10 documents, but only 2 are relevant, Precision is `0.20`. The remaining 8 documents are **distractors**.
    * **Financial Cost**: Irrelevant chunks consume input tokens, increasing LLM API inference costs.
    * **Attention Degradation**: Low precision triggers the Lost-in-the-Middle attention slump, increasing hallucination rates by up to 3x.
    """)

with prec_col2:
    st.markdown("### Precision Tradeoff Matrix")
    st.info("""
    **Rule of Thumb**: For factual RAG workflows, aim for **Precision@5 >= 0.60**. 
    If precision drops below 0.40, activate cross-encoder reranking or tighter similarity score thresholds.
    """)

st.divider()

# ==============================================================================
# SECTION 3: RECALL
# ==============================================================================
st.header("3. Recall (Exhaustiveness, False Negative Risk & Information Coverage)", divider="rainbow")

st.markdown("""
**Recall** measures the proportion of all relevant documents in the corpus that were successfully retrieved:
`Recall = Relevant_Retrieved / Total_Relevant_Gold`.
""")

rec_col1, rec_col2 = st.columns([1, 1])

with rec_col1:
    st.markdown("""
    ### The High Cost of Recall Gaps:
    * **The Retrieval Ceiling**: If a question requires synthesizing 3 distinct documents (multi-hop) and the retriever surfaces only 2, the maximum achievable recall is `0.67`.
    * **Generator Starvation**: The LLM cannot reason over facts it was never given. A recall miss forces either a refusal or an extrapolative guess.
    * **The Precision-Recall Tradeoff**: Increasing Top-K from 5 to 20 increases Recall (capturing more gold passages), but decreases Precision (injecting more distractors).
    """)

with rec_col2:
    st.markdown("### Precision vs Recall Frontier Across Top-K Cutoffs")
    k_vals = np.array([1, 2, 3, 5, 10, 20])
    sim_prec = np.array([0.90, 0.85, 0.72, 0.58, 0.35, 0.20])
    sim_rec = np.array([0.35, 0.55, 0.70, 0.88, 0.95, 0.99])
    pr_df = pd.DataFrame({
        "Top-K": k_vals,
        "Precision": sim_prec,
        "Recall": sim_rec
    }).set_index("Top-K")
    st.line_chart(pr_df)
    st.caption("As K increases, Recall rises toward 1.0 while Precision degrades sharply.")

st.divider()

# ==============================================================================
# SECTION 4: HIT RATE
# ==============================================================================
st.header("4. Hit Rate (Binary Retrieval Success & Top-K Coverage)", divider="rainbow")

st.markdown("""
**Hit Rate@K** (also called Success@K) evaluates whether *at least one* relevant document was present in the Top-K results:
`Hit_Rate@K = 1.0 if (Count(Relevant in Top-K) >= 1) else 0.0`.
""")

hr_col1, hr_col2 = st.columns([1, 1])

with hr_col1:
    st.markdown("""
    ### When to Use Hit Rate:
    * **Factoid Q&A Applications**: For customer support or simple lookups where finding a single authoritative passage suffices to answer the inquiry.
    * **Search Viability Threshold**: If Hit Rate@10 is below 0.90, the embedding model or vector indexing is fundamentally misaligned with user search vocabulary.
    * **Baseline Benchmark**: The easiest metric to track in production telemetry without needing exhaustive multi-document annotations.
    """)

with hr_col2:
    st.markdown("### Hit Rate Targets by Domain")
    hit_targets = pd.DataFrame([
        {"Enterprise Tier": "Customer Support FAQ", "Recommended Metric": "Hit Rate@3 >= 0.92", "Priority": "High Top-3 Precision"},
        {"Enterprise Tier": "Internal Enterprise Search", "Recommended Metric": "Hit Rate@5 >= 0.95", "Priority": "Balanced Precision & Recall"},
        {"Enterprise Tier": "Legal & Compliance Discovery", "Recommended Metric": "Hit Rate@10 >= 0.99", "Priority": "Zero False Negatives"}
    ])
    st.dataframe(hit_targets, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 5: MRR (MEAN RECIPROCAL RANK)
# ==============================================================================
st.header("5. Mean Reciprocal Rank - MRR (First-Item Rank Position & Top-1 Efficiency)", divider="rainbow")

st.markdown("""
**Mean Reciprocal Rank (MRR)** evaluates how high up in the ranking the **first** relevant document appears. 
For a single query: `Reciprocal_Rank = 1 / Rank_of_First_Relevant_Doc`. 
Averaged across `N` queries: `MRR = (1 / N) * Sum(RR_i)`.
""")

mrr_col1, mrr_col2 = st.columns([1, 1])

with mrr_col1:
    st.markdown("""
    ### Reciprocal Rank Scoring Mechanics:
    * First relevant document at **Rank 1**: `RR = 1 / 1 = 1.0` (Perfect)
    * First relevant document at **Rank 2**: `RR = 1 / 2 = 0.5`
    * First relevant document at **Rank 3**: `RR = 1 / 3 = 0.33`
    * First relevant document at **Rank 10**: `RR = 1 / 10 = 0.10`
    * Zero relevant documents in Top-K: `RR = 0.0`
    
    ### Importance for Lost-in-the-Middle Mitigation:
    High MRR (>0.75) guarantees that gold passages land in the **Primacy Zone** (prompt start), 
    preventing them from sinking into the attention dead-zone in the middle of long prompts.
    """)

with mrr_col2:
    st.markdown("### Reciprocal Rank Penalty Curve")
    ranks = np.arange(1, 11)
    rr_vals = 1.0 / ranks
    rr_chart = pd.DataFrame({
        "Rank of 1st Gold Doc": ranks,
        "Reciprocal Rank Score": rr_vals
    }).set_index("Rank of 1st Gold Doc")
    st.bar_chart(rr_chart)
    st.caption("MRR drops precipitously from 1.0 to 0.33 by Rank 3.")

st.divider()

# ==============================================================================
# SECTION 6: NDCG (NORMALIZED DISCOUNTED CUMULATIVE GAIN)
# ==============================================================================
st.header("6. Normalized Discounted Cumulative Gain - NDCG (Graded Relevance & Logarithmic Discounting)", divider="rainbow")

st.markdown("""
Unlike binary metrics, **NDCG** accommodates **graded relevance judgments** 
(e.g., `3 = Perfect Answer`, `2 = Highly Relevant`, `1 = Marginally Relevant`, `0 = Irrelevant`).
""")

ndcg_col1, ndcg_col2 = st.columns([1, 1])

with ndcg_col1:
    st.markdown("""
    ### The 3 Steps of NDCG:
    1. **Cumulative Gain (CG@K)**: Sum of relevance scores: `CG@K = Sum_{i=1}^K rel_i`. (Ignores ranking order).
    2. **Discounted Cumulative Gain (DCG@K)**: Penalizes documents appearing lower in the rank:
       `DCG@K = Sum_{i=1}^K (rel_i / log2(i + 1))`
    3. **Normalized DCG (NDCG@K)**: Divides DCG@K by the Ideal DCG (IDCG@K) achieved by sorting documents in perfect descending relevance:
       `NDCG@K = DCG@K / IDCG@K`
    * Result is strictly bounded between `0.0` and `1.0`.
    """)

with ndcg_col2:
    st.markdown("### Graded Relevance Scale Definition")
    graded_scale = pd.DataFrame([
        {"Score": "Grade 3: Perfect", "Meaning": "Directly and authoritatively answers the query completely."},
        {"Score": "Grade 2: Highly Relevant", "Meaning": "Answers core components of the query; requires slight secondary synthesis."},
        {"Score": "Grade 1: Marginal", "Meaning": "Discusses related entities; provides background but lacks direct answer."},
        {"Score": "Grade 0: Irrelevant", "Meaning": "Completely off-topic or contains misleading information."}
    ])
    st.dataframe(graded_scale, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: RECALL@K
# ==============================================================================
st.header("7. Recall@K (Cutoff Retrieval Coverage & K-Sensitivity Analysis)", divider="rainbow")

st.markdown("""
**Recall@K** measures coverage evaluated specifically at rank cutoff `K`. 
Evaluating Recall across `K in [1, 3, 5, 10, 20]` establishes the **Retrieval Saturation Curve**.
""")

rk_col1, rk_col2 = st.columns([1, 1])

with rk_col1:
    st.markdown("""
    ### Determining Optimal K for RAG:
    * **Under-Sizing K (K=2)**: Results in high precision, but drops Recall@K on multi-hop questions where 4 supporting passages are needed.
    * **Over-Sizing K (K=25)**: Reaches 99% Recall, but inflates latency and token budget, pushing gold facts into the attention valley.
    * **Elbow Analysis**: Plotting Recall@K reveals the point of diminishing returns—typically `K = 5` or `K = 7` in enterprise corpora.
    """)

with rk_col2:
    st.markdown("### Typical Recall Saturation Curve")
    k_steps = [1, 2, 3, 5, 8, 10, 15]
    recall_curve = [0.42, 0.62, 0.78, 0.89, 0.94, 0.96, 0.98]
    st.line_chart(pd.DataFrame({"Recall@K": recall_curve}, index=k_steps))
    st.caption("Recall flattens significantly after K=5; expanding beyond K=8 provides minimal coverage gain at high token cost.")

st.divider()

# ==============================================================================
# SECTION 8: PRECISION@K
# ==============================================================================
st.header("8. Precision@K (Top-K Purity, Context Window Clutter & Cost Impact)", divider="rainbow")

st.markdown("""
**Precision@K** measures the purity of the top `K` passages passed to the prompt:
`Precision@K = Count(Relevant in Top-K) / K`.
""")

pk_col1, pk_col2 = st.columns([1, 1])

with pk_col1:
    st.markdown("""
    ### Precision@K Degradation Dynamics:
    * Because corpora contain a finite number of gold documents (often 1 or 2), Precision@K mathematically drops as `K` exceeds the true number of relevant documents.
    * **Precision@1**: Tests whether the Top-1 result is accurate (critical for search engines and direct answer cards).
    * **Precision@5**: Balances evidence gathering against prompt clutter.
    """)

with pk_col2:
    st.markdown("### Production Retrieval Target Benchmarks")
    target_table = pd.DataFrame([
        {"Metric": "Hit Rate@5", "Acceptable Threshold": ">= 0.90", "Production Champion": ">= 0.96"},
        {"Metric": "MRR", "Acceptable Threshold": ">= 0.65", "Production Champion": ">= 0.82"},
        {"Metric": "NDCG@5", "Acceptable Threshold": ">= 0.70", "Production Champion": ">= 0.88"},
        {"Metric": "Recall@5", "Acceptable Threshold": ">= 0.80", "Production Champion": ">= 0.92"}
    ])
    st.dataframe(target_table, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Comprehensive Retrieval Metrics Suite", divider="rainbow")

st.markdown("""
Below is the production implementation of `RetrievalMetricsEngine`. It computes all seven retrieval metrics 
in module scope, supporting binary and graded relevance evaluations across custom K cutoffs.
""")

st.code(r'''
class RetrievalMetricsEngine:
    """
    Enterprise Information Retrieval (IR) Evaluation Engine:
    - Precision@K & Recall@K
    - Hit Rate@K
    - Mean Reciprocal Rank (MRR)
    - Graded DCG@K and Normalized DCG@K (NDCG)
    """
    def precision_at_k(self, retrieved: List[str], gold: set, k: int) -> float:
        if k <= 0: return 0.0
        hits = sum(1 for d in retrieved[:k] if d in gold)
        return float(hits / k)

    def recall_at_k(self, retrieved: List[str], gold: set, k: int) -> float:
        if not gold or k <= 0: return 0.0
        hits = sum(1 for d in retrieved[:k] if d in gold)
        return float(hits / len(gold))

    def hit_rate_at_k(self, retrieved: List[str], gold: set, k: int) -> float:
        return 1.0 if any(d in gold for d in retrieved[:k]) else 0.0

    def reciprocal_rank(self, retrieved: List[str], gold: set) -> float:
        for r, d in enumerate(retrieved, start=1):
            if d in gold:
                return float(1.0 / r)
        return 0.0

    def dcg_at_k(self, retrieved: List[str], scores: Dict[str, float], k: int) -> float:
        dcg = 0.0
        for i, d in enumerate(retrieved[:k], start=1):
            rel = scores.get(d, 0.0)
            dcg += rel / np.log2(i + 1)
        return float(dcg)

    def ndcg_at_k(self, retrieved: List[str], scores: Dict[str, float], k: int) -> float:
        actual = self.dcg_at_k(retrieved, scores, k)
        ideal_order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        ideal = self.dcg_at_k(ideal_order, scores, k)
        return float(actual / ideal) if ideal > 0.0 else 0.0
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Live Top-K Benchmark Calculator & Metric Comparison", divider="rainbow")

st.markdown("""
Experiment with ranking permutations in real time. 
Drag candidate document IDs to change retrieval order and inspect how **Precision@K**, **Recall@K**, **MRR**, 
and **NDCG@K** respond to ranking shifts.
""")

test_corpus_scores = {
    "DOC-101": 3.0,  # Perfect gold
    "DOC-202": 2.0,  # Highly relevant
    "DOC-303": 0.0,  # Irrelevant distractor
    "DOC-404": 1.0,  # Marginally relevant
    "DOC-505": 0.0,  # Irrelevant distractor
    "DOC-606": 0.0   # Irrelevant distractor
}
binary_gold_docs = {"DOC-101", "DOC-202"}

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("### Candidate Ranking Configuration")
    ranking_preset = st.selectbox(
        "Choose Simulated Retriever Ranking:",
        options=[
            "Optimal Ranking (DOC-101 at Rank 1, DOC-202 at Rank 2)",
            "Delayed First Hit (DOC-303 at Rank 1, DOC-101 at Rank 2)",
            "Buried Gold Facts (Distractors at Ranks 1-3, DOC-101 at Rank 4)",
            "Complete Retrieval Miss (Zero Gold Documents in Top-6)"
        ],
        index=0
    )

    if ranking_preset.startswith("Optimal"):
        sim_retrieved = ["DOC-101", "DOC-202", "DOC-404", "DOC-303", "DOC-505", "DOC-606"]
    elif ranking_preset.startswith("Delayed"):
        sim_retrieved = ["DOC-303", "DOC-101", "DOC-202", "DOC-404", "DOC-505", "DOC-606"]
    elif ranking_preset.startswith("Buried"):
        sim_retrieved = ["DOC-303", "DOC-505", "DOC-606", "DOC-101", "DOC-404", "DOC-202"]
    else:
        sim_retrieved = ["DOC-303", "DOC-505", "DOC-606", "DOC-999", "DOC-888", "DOC-777"]

    st.write("**Current Evaluated Retrieval Order**:")
    for rank, doc in enumerate(sim_retrieved, start=1):
        rel = test_corpus_scores.get(doc, 0.0)
        st.write(f"Rank {rank}: `{doc}` (True Graded Relevance: {rel})")

with wb_col2:
    st.markdown("### Live Evaluator Output Scorecard")
    eval_suite = ir_engine.evaluate_retrieval_run(
        retrieved_ranked=sim_retrieved,
        gold_set=binary_gold_docs,
        graded_scores=test_corpus_scores,
        k_values=[1, 2, 3, 5]
    )

    m1, m2 = st.columns(2)
    m1.metric("Mean Reciprocal Rank (MRR)", f"{eval_suite['mrr']:.3f}")
    first_rank_label = f"Rank {eval_suite['first_hit_rank']}" if eval_suite['first_hit_rank'] else "No Hits"
    m2.metric("First Relevant Document", first_rank_label)

    st.markdown("#### Cutoff Sensitivity Table Across Top-K:")
    st.dataframe(pd.DataFrame(eval_suite["metrics_table"]), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the mathematical and operational difference between Recall@K and Precision@K in RAG?**"):
    st.markdown("""
    **Answer**:
    * **Precision@K = Hits in Top-K / K**: Measures the density of signal vs noise in the context window. High Precision@K keeps prompt tokens focused and reduces LLM distraction.
    * **Recall@K = Hits in Top-K / Total Gold Documents**: Measures information completeness. High Recall@K guarantees that the model has all necessary facts for synthesis.
    * **Tradeoff**: As K increases, Recall@K monotonically increases (or stays constant), while Precision@K typically degrades.
    """)

with st.expander("**Q2: Why is NDCG superior to Precision and Recall for evaluating cross-encoder rerankers?**"):
    st.markdown("""
    **Answer**:
    Precision and Recall treat relevance as binary (`0` or `1`) and treat all positions within Top-K equally.
    In contrast:
    1. **Graded Relevance**: NDCG recognizes that a 10-page runbook (Grade 3) is far more valuable than an informal Slack comment (Grade 1).
    2. **Positional Discounting**: NDCG divides scores by `log2(rank + 1)`, severely penalizing the retriever if a Grade 3 document is placed at rank 5 instead of rank 1. This directly addresses the Lost-in-the-Middle attention bias.
    """)

with st.expander("**Q3: What does a high Hit Rate@5 combined with a low MRR indicate about a retrieval pipeline?**"):
    st.markdown("""
    **Answer**:
    * **Diagnosis**: High Hit Rate@5 (e.g., 0.95) means the retriever successfully places a gold document within the top 5 candidates.
    * **The Problem**: Low MRR (e.g., 0.25 to 0.33) indicates that the gold document consistently lands at rank 3, 4, or 5 rather than rank 1.
    * **Root Cause**: The dense retriever or bi-encoder has high broad recall, but lacks fine-grained lexical and cross-attention precision.
    * **Remedy**: Deploy a cross-encoder reranker (e.g., BGE-Reranker-Large or Cohere Rerank) to push gold documents from rank 5 up to rank 1.
    """)

with st.expander("**Q4: How does chunk size affect Precision@K and Recall@K?**"):
    st.markdown("""
    **Answer**:
    * **Small Chunks (e.g., 100 tokens)**: Yield high Precision@K (chunks contain very little noise), but lower Recall@K (facts spanning multiple sentences get split across chunks).
    * **Large Chunks (e.g., 1,000 tokens)**: Yield high Recall@K (fewer broken facts), but poor Precision@K (each chunk contains 80% irrelevant text, saturating the prompt context).
    """)

with st.expander("**Q5: Why is evaluating MRR alone insufficient for multi-hop RAG queries?**"):
    st.markdown("""
    **Answer**:
    MRR only evaluates the rank of the **first** relevant document. Once a single gold document is found, MRR completely ignores all subsequent documents.
    In multi-hop queries, answering the question requires 2 or 3 distinct facts across different documents. A pipeline that retrieves Gold Document 1 at rank 1, but misses Gold Document 2 entirely, achieves a perfect MRR of `1.0`, while the LLM will fail to answer the question due to low Recall@K.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Evaluating MRR on Multi-Hop Queries**
    * **Failure Mode**: Reporting MRR = 0.92 and assuming the retriever is ready for production, unaware that secondary bridge documents are missing.
    * **Solution**: Mandate Recall@K and NDCG@K alongside MRR for multi-hop workloads.
    """)

    st.error("""
    **Production Trap 2: Blindly Increasing K to 'Fix' Recall**
    * **Failure Mode**: Increasing Top-K from 5 to 25 to improve recall, causing LLM inference costs to quadruple and inducing middle-attention hallucinations.
    * **Solution**: Improve embedding model fine-tuning or hybrid BM25 search rather than expanding K.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Evaluating Only at K=10**
    * **Failure Mode**: Reporting metrics only at K=10 hides whether documents are clustered at rank 1 or rank 9.
    * **Solution**: Always report metrics across multiple cutoffs (K in [1, 3, 5, 10]).
    """)

    st.warning("""
    **Production Trap 4: Treating Binary Relevance as Graded Truth**
    * **Failure Mode**: Evaluating complex domain search using binary 0/1 labels masks subtle quality regressions between high-authority policies and informal chatter.
    * **Solution**: Annotate golden datasets with 4-tier graded relevance (0 to 3) and evaluate with NDCG.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: MRR Formula")
        st.caption("How is Mean Reciprocal Rank calculated?")
        st.markdown("**Back**: Inverse of the rank of the first relevant document: `1 / rank_first_hit`, averaged across queries.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: NDCG Discount Factor")
        st.caption("How does DCG mathematically discount lower ranks?")
        st.markdown("**Back**: Divides the graded relevance score by `log2(rank + 1)`, penalizing relevant items placed at lower positions.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Hit Rate@K vs Recall@K")
        st.caption("What is the core distinction between Hit Rate and Recall?")
        st.markdown("**Back**: Hit Rate is binary (at least one hit in Top-K); Recall measures the fraction of all gold documents retrieved.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Optimal Top-K Saturation")
        st.caption("How to select the production K cutoff?")
        st.markdown("**Back**: Perform elbow analysis where Recall@K flattens, balancing information coverage against prompt token inflation.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of retrieval evaluation science before proceeding:")

c1 = st.checkbox("I understand the mathematical definitions and tradeoffs of Precision@K, Recall@K, and Hit Rate@K.", value=True)
c2 = st.checkbox("I can explain why MRR is ideal for factoid lookup but insufficient for multi-hop synthesis.", value=True)
c3 = st.checkbox("I know how Discounted Cumulative Gain (DCG) logarithmic discounting rewards placing top documents at Rank 1.", value=True)
c4 = st.checkbox("I can calculate Normalized Discounted Cumulative Gain (NDCG) using graded relevance scales.", value=True)
c5 = st.checkbox("I understand why blindly expanding K to maximize recall degrades LLM inference latency and precision.", value=True)
c6 = st.checkbox("I know how to construct cutoff sensitivity scorecards across K in [1, 3, 5, 10] for CI/CD regression gates.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **Retrieval Metrics Mastery Achieved!** You are ready to benchmark and optimize enterprise vector search pipelines.")
