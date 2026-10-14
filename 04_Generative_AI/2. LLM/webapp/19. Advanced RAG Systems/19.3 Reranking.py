import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.3 Reranking - Advanced RAG",
    page_icon="⚖️",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class RerankingEvaluationSuite:
    """
    Self-contained implementation of two-stage retrieval and evaluation:
    - Simulated Cross-Encoder all-to-all cross-attention scoring
    - Stage 1 (Bi-Encoder) to Stage 2 (Cross-Encoder) reranking funnel
    - Exact calculation of Hit Rate@K, MRR@K, and NDCG@K metrics
    """
    
    CORPUS = [
        {
            "id": "DOC-A",
            "title": "PostgreSQL Replica Lag Troubleshooting",
            "text": "To resolve streaming replica lag, verify wal_keep_size settings and ensure network bandwidth between primary and standby nodes is at least 10 Gbps.",
            "bi_score": 0.82,
            "cross_score": 0.96, # True answer for replica lag
            "ground_truth_rel": 3
        },
        {
            "id": "DOC-B",
            "title": "PostgreSQL Installation on Ubuntu",
            "text": "Install postgresql using apt-get update && apt-get install postgresql-16. Configure default port 5432.",
            "bi_score": 0.88, # High keyword/vector overlap, but irrelevant to lag
            "cross_score": 0.22,
            "ground_truth_rel": 0
        },
        {
            "id": "DOC-C",
            "title": "Network Bandwidth Optimization Guide",
            "text": "Optimizing switch buffers and enabling jumbo frames (MTU 9000) prevents packet drops in high-throughput datacenter clusters.",
            "bi_score": 0.75,
            "cross_score": 0.65, # Partially relevant
            "ground_truth_rel": 1
        },
        {
            "id": "DOC-D",
            "title": "Kubernetes StatefulSet Storage Config",
            "text": "Provision persistent volumes with ReadWriteOnce access modes for single-pod database replicas.",
            "bi_score": 0.79,
            "cross_score": 0.30,
            "ground_truth_rel": 0
        },
        {
            "id": "DOC-E",
            "title": "Write-Ahead Log (WAL) Retention Policy",
            "text": "Configuring wal_keep_size determines the minimum volume of past log segments retained in pg_wal to avoid disconnects.",
            "bi_score": 0.70,
            "cross_score": 0.88, # Highly relevant secondary evidence
            "ground_truth_rel": 2
        }
    ]

    @classmethod
    def execute_two_stage_rerank(cls, top_k1: int = 5, top_k2: int = 3, threshold: float = 0.0) -> Tuple[List[Dict], List[Dict]]:
        """Simulates Stage 1 Bi-Encoder retrieval followed by Stage 2 Cross-Encoder reranking."""
        # Stage 1: Sort by Bi-Encoder score
        stage1 = sorted(cls.CORPUS, key=lambda x: x["bi_score"], reverse=True)[:top_k1]
        
        # Stage 2: Rerank the stage 1 candidates by Cross-Encoder score
        stage2 = sorted(stage1, key=lambda x: x["cross_score"], reverse=True)
        stage2_filtered = [doc for doc in stage2 if doc["cross_score"] >= threshold][:top_k2]
        
        return stage1, stage2_filtered

    @staticmethod
    def calculate_metrics(ranked_docs: List[Dict], k: int = 3) -> Dict[str, float]:
        """Calculates Hit Rate@K, MRR@K, and NDCG@K."""
        top_k_docs = ranked_docs[:k]
        
        # 1. Hit Rate @ K (at least one relevant doc with rel > 0)
        hit = 1.0 if any(d["ground_truth_rel"] > 0 for d in top_k_docs) else 0.0
        
        # 2. MRR @ K (rank of first document with highest relevance == 3)
        mrr = 0.0
        for rank, d in enumerate(top_k_docs, 1):
            if d["ground_truth_rel"] == 3:
                mrr = 1.0 / rank
                break
                
        # 3. NDCG @ K
        dcg = 0.0
        for rank, d in enumerate(top_k_docs, 1):
            rel = d["ground_truth_rel"]
            dcg += (2**rel - 1) / np.log2(rank + 1)
            
        # Ideal DCG (sorted ideal relevance: [3, 2, 1, 0, 0])
        ideal_rels = sorted([d["ground_truth_rel"] for d in RerankingEvaluationSuite.CORPUS], reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels, 1):
            idcg += (2**rel - 1) / np.log2(rank + 1)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "hit_rate": hit,
            "mrr": mrr,
            "ndcg": ndcg
        }


# --- HEADER & INTRODUCTION ---
st.title("19.3 Reranking: Cross-Encoders, Two-Stage Retrieval, & Evaluation Metrics")
st.caption("Chapter 19: Advanced RAG Systems | Domain: 2. Large Language Models")

st.markdown("""
In information retrieval engineering, **speed and precision exist in direct architectural tension**.

**Bi-Encoders (Vector Embeddings)** are blindingly fast because they encode queries and documents independently: 
millions of vectors can be searched in sub-10-millisecond response times using HNSW graphs. 
However, compressing an entire 500-word paragraph into a single 1536-dimensional vector inevitably loses fine-grained nuances, 
negations, and subtle relational dependencies.

**Cross-Encoders**, by contrast, feed the query and document together through full bidirectional transformer attention layers, 
evaluating every single query token against every document token. While far too computationally expensive to scan 1,000,000 documents, 
Cross-Encoders are the ultimate **Rerankers**: scoring the top-50 coarse candidates retrieved in Stage 1 and surfacing 
the top-3 gold-standard passages for prompt generation.

This lesson explores the Two-Stage Retrieve-and-Rerank funnel, Cross-Encoder scoring mechanics, and standard retrieval evaluation metrics 
(MRR, Hit Rate, and NDCG).
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Two-Stage Retrieve-and-Rerank Funnel", divider="rainbow")

st.markdown("""
How modern enterprise RAG platforms orchestrate coarse candidate generation followed by high-precision reranking:
""")

rerank_blueprint_dot = """
digraph RerankFunnel {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_stage1 {
        label="Stage 1: Coarse Candidate Generator (High Recall)";
        color="#1E293B";
        fontcolor="#38BDF8";
        Corpus [label="Enterprise Corpus\\n(1,000,000 Documents)", fillcolor="#1E293B"];
        BiEncoder [label="Bi-Encoder HNSW + BM25\\nIndependent Vector Dot Product\\nLatency: < 5 ms", fillcolor="#0F766E"];
        Top50 [label="Top-50 Candidates\\nHigh Recall (95%)\\nModerate Precision (60%)", fillcolor="#0369A1"];
    }

    subgraph cluster_stage2 {
        label="Stage 2: Fine Cross-Encoder Reranker (High Precision)";
        color="#1E293B";
        fontcolor="#38BDF8";
        CrossEncoder [label="Cross-Encoder Model\\n(e.g., BGE-Reranker-Large)\\nFull All-to-All Attention\\nLatency: ~25 ms", fillcolor="#6D28D9"];
        SigmoidCalib [label="Relevance Calibration\\nSigmoid Probability Scoring\\nThreshold Clamping (tau >= 0.70)", fillcolor="#4338CA"];
        Top5Gold [label="Top-3 to Top-5 Gold Chunks\\nHigh Precision (98%)\\nClean Context for LLM", fillcolor="#059669"];
    }

    Corpus -> BiEncoder -> Top50;
    Top50 -> CrossEncoder -> SigmoidCalib -> Top5Gold;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(rerank_blueprint_dot, use_container_width=True)

st.markdown("""
### Retrieval Architecture Trade-Offs Matrix
""")

rerank_matrix = [
    {
        "Architecture": "Bi-Encoder (First Stage)",
        "Input Data Flow": "Query and Document encoded separately: Embed(Q) and Embed(D)",
        "Token Interaction": "Zero cross-attention; simple dot product or cosine similarity",
        "Search Speed": "Millions of docs in < 10 ms via ANN HNSW index",
        "Semantic Precision": "Moderate (loses subtle negations, exact relations)"
    },
    {
        "Architecture": "Cross-Encoder (Second Stage)",
        "Input Data Flow": "Concatenated into single sequence: [CLS] Query [SEP] Document [SEP]",
        "Token Interaction": "Full all-to-all bidirectional cross-attention across all 24 layers",
        "Search Speed": "Scans 50-100 candidates in ~20-30 ms on GPU",
        "Semantic Precision": "State-of-the-Art (detects deep relational nuances)"
    },
    {
        "Architecture": "Late Interaction (ColBERTv2)",
        "Input Data Flow": "Token-level vector matrices for query and document",
        "Token Interaction": "MaxSim: sum of max dot products between query and document token vectors",
        "Search Speed": "Scans 10,000 candidates in ~15 ms with inverted centroid index",
        "Semantic Precision": "Near-Cross-Encoder accuracy with near-Bi-Encoder speed"
    },
    {
        "Architecture": "Generative LLM Reranker (RankGPT)",
        "Input Data Flow": "Passes 20 documents directly into LLM prompt; asks LLM to rank",
        "Token Interaction": "Autoregressive generative attention",
        "Search Speed": "Slow (200-500 ms per query)",
        "Semantic Precision": "Highest (utilizes full LLM reasoning capabilities)"
    }
]
st.dataframe(pd.DataFrame(rerank_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - CROSS-ENCODER RERANKERS ---
st.header("2. Cross-Encoder Rerankers: Full All-to-All Attention & Token Interaction", divider="rainbow")

st.markdown("""
Why does a **Cross-Encoder** achieve vastly superior semantic accuracy compared to a Bi-Encoder?

In a Bi-Encoder, the query and document never interact inside the neural network: each is independently compressed 
into a single vector. 

In a **Cross-Encoder**, the query and document are concatenated into a single combined input sequence:

```text
Input Tensor = [CLS] Query Tokens [SEP] Document Tokens [SEP]
```
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("All-to-All Bidirectional Cross-Attention")
    st.markdown("""
    When passed through the transformer:
    - Every token in the query computes self-attention against **every single token in the document** across all 24 layers!
    - The attention heads can directly evaluate grammatical dependencies:
      - Does the negation word *"not"* in the document apply to the action in the query?
      - Does the word *"bank"* in the query align with *"deposit"* or *"river"* in the document?
    - The final `[CLS]` token output is projected through a single linear layer to produce an unnormalized scalar relevance logit.
    """)

with col2:
    st.info("""
    **Leading Production Cross-Encoder Models**:
    - **BGE-Reranker-Large / BGE-Reranker-v2-m3**: Open-weight state-of-the-art multilingual reranker.
    - **Cohere Rerank v3**: Commercial cloud API supporting long-context document reranking.
    - **Jina Reranker v2**: Compact 8k-token context cross-encoder with low GPU memory footprint.
    - **ms-marco-MiniLM-L6-v2**: Ultra-fast CPU-compatible cross-encoder (~5ms on CPU).
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - RELEVANCE SCORING ---
st.header("3. Relevance Scoring: Logit Calibration, Sigmoid Probabilities, & Thresholding", divider="rainbow")

st.markdown("""
A Cross-Encoder model outputs a raw classification logit `z in (-inf, +inf)`. 
To use this score effectively in production, the raw logit is mapped to a normalized probability 
using the **Sigmoid function**:

```text
Relevance_Probability = Sigmoid(z) = 1.0 / ( 1.0 + exp(-z) )  in [0.0, 1.0]
```

### Calibrating Production Relevance Thresholds
Unlike Cosine similarity (where scores cluster tightly between 0.65 and 0.85), 
Cross-Encoder calibrated probabilities exhibit **sharp discriminatory separation**:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - **Score >= 0.80 (Direct Answer)**: The document explicitly and directly answers the user's question. Safe to pass to the LLM.
    - **Score 0.50 - 0.79 (Background Context)**: The document is on the same topic, but may not contain the exact factual answer.
    - **Score < 0.40 (Irrelevant Distractor)**: The document shares keywords, but has zero factual relevance. Must be pruned!
    """)
with col2:
    st.warning("""
    **The Negative Threshold Filter**:
    Enforcing `threshold >= 0.70` after reranking acts as an automated hallucination barrier: 
    if all 50 candidates score below 0.70, the system immediately returns *"No verified answer found"* 
    without spending GPU compute on the final LLM synthesis step!
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - TOP-K RERANKING ---
st.header("4. Top-K Reranking: Latency-Quality Trade-Offs & Candidate Funnel Sizing", divider="rainbow")

st.markdown("""
Designing the two-stage funnel requires balancing Stage 1 candidate volume (K1) against Stage 2 reranking latency (K2):
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Golden Funnel Ratio (K1 = 50 -> K2 = 5)")
    st.markdown("""
    - **Stage 1 (K1 = 50)**: Vector search and BM25 retrieve the top 50 candidates in `< 5ms`. 
      At K1 = 50, recall is typically over 96% (the true answer is almost certainly present in the pool).
    - **Stage 2 (K2 = 5)**: The Cross-Encoder scores all 50 pairs in a single batched GPU forward pass (~20ms). 
      It sorts the list and extracts only the top 3 to 5 gold-standard chunks.
    - **The Latency Paradox**: Even though reranking adds 20ms, it **reduces total end-to-end RAG latency** 
      because the LLM only ingests 5 chunks (1,500 tokens) instead of 50 chunks (15,000 tokens), 
      saving hundreds of milliseconds of prompt prefill time!
    """)

with col2:
    st.subheader("Batched Cross-Encoder Inference")
    st.markdown("""
    Never execute Cross-Encoder scoring in a sequential Python loop (`for doc in docs: score(q, doc)`). 
    Always batch candidate pairs together:
    ```python
    pairs = [[query, doc.text] for doc in stage1_candidates]
    # Executed as a single batched tensor on GPU Tensor Cores
    scores = reranker_model.predict(pairs, batch_size=64)
    ```
    GPU parallelization processes 64 pairs in virtually the same wall-clock time as a single pair!
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - RETRIEVAL EVALUATION ---
st.header("5. Retrieval Evaluation: MRR, Hit Rate (Recall@K), NDCG, & MAP Metrics", divider="rainbow")

st.markdown("""
To scientifically validate whether a reranker improves retrieval, information retrieval engineers 
track three standardized mathematical metrics across benchmark test suites (like MS MARCO or internal golden sets):
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 1. Hit Rate @ K (Recall@K)
    ```text
    HitRate@K = (1 / |Q|) * Sum [ Indicator(Hit in Top-K) ]
    ```
    - Evaluates whether **at least one** relevant document appears anywhere within the top-K returned results.
    - A binary metric: 1.0 if found, 0.0 otherwise.
    - Simple and intuitive, but completely ignores where in the top-K list the hit appeared.
    """)

with col2:
    st.markdown("""
    ### 2. Mean Reciprocal Rank (MRR@K)
    ```text
    MRR@K = (1 / |Q|) * Sum [ 1.0 / Rank_first ]
    ```
    - Evaluates the **exact rank position** of the first relevant document:
      - Hit at Rank 1: Score = `1/1 = 1.0`
      - Hit at Rank 2: Score = `1/2 = 0.5`
      - Hit at Rank 3: Score = `1/3 = 0.33`
    - Heavily rewards systems that elevate the correct answer to the very top slot!
    """)

with col3:
    st.markdown("""
    ### 3. NDCG @ K (Normalized DCG)
    ```text
    DCG@K = Sum [ (2^rel_i - 1) / log2(i + 1) ]
    NDCG@K = DCG@K / IDCG@K
    ```
    - Evaluates **graded relevance** (e.g. 3 = Perfect, 2 = Good, 1 = Minor, 0 = Irrelevant).
    - Applies a logarithmic penalty as rank descends.
    - Divided by Ideal DCG (IDCG) so score ranges cleanly from `0.0` to `1.0`.
    - The primary North Star metric of production search engines!
    """)

st.divider()

# --- SECTION 6: PRODUCTION CODE ---
st.header("6. Production Implementation: Cross-Encoder Simulator & Retrieval Metrics Suite in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements a production two-stage retrieval pipeline 
and evaluates Hit Rate@K, MRR@K, and NDCG@K:
""")

code_block = '''import numpy as np
from typing import List, Dict, Any

class TwoStageRetrieverAndEvaluator:
    """
    Self-contained production implementation of two-stage retrieval
    and information retrieval evaluation metrics (Hit Rate, MRR, NDCG).
    """
    
    @staticmethod
    def rerank_candidates(
        stage1_candidates: List[Dict[str, Any]],
        cross_encoder_scores: List[float],
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Applies Cross-Encoder scores, filters by threshold, and ranks top-K."""
        reranked = []
        for doc, score in zip(stage1_candidates, cross_encoder_scores):
            if score >= threshold:
                item = dict(doc)
                item["rerank_score"] = float(score)
                reranked.append(item)
                
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def evaluate_retrieval(ranked_docs: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
        """
        Calculates Hit Rate@K, MRR@K, and NDCG@K given ground-truth relevance labels.
        Assumes each document in ranked_docs has a 'relevance' key (integer 0 to 3).
        """
        top_k_docs = ranked_docs[:k]
        
        # 1. Hit Rate @ K
        hit = 1.0 if any(d.get("relevance", 0) > 0 for d in top_k_docs) else 0.0
        
        # 2. MRR @ K (rank of first hit with relevance >= 2)
        mrr = 0.0
        for rank, d in enumerate(top_k_docs, 1):
            if d.get("relevance", 0) >= 2:
                mrr = 1.0 / rank
                break
                
        # 3. NDCG @ K
        dcg = 0.0
        for rank, d in enumerate(top_k_docs, 1):
            rel = d.get("relevance", 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)
            
        # Ideal DCG calculation
        all_rels = sorted([d.get("relevance", 0) for d in ranked_docs], reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(all_rels, 1):
            idcg += (2**rel - 1) / np.log2(rank + 1)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "hit_rate": float(hit),
            "mrr": float(mrr),
            "ndcg": float(ndcg)
        }
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 7: WORKBENCH ---
st.header("7. Interactive Workbench: Two-Stage Funnel Simulator & Metric Calculator", divider="rainbow")

st.markdown("""
Simulate how adding a **Cross-Encoder Reranker** repairs Bi-Encoder ranking errors and dramatically boosts 
**MRR** and **NDCG** metrics live:
""")

wb_col1, wb_col2, wb_col3 = st.columns(3)
with wb_col1:
    k1_val = st.slider("Stage 1 Candidates (K1)", min_value=3, max_value=5, value=5, step=1)
with wb_col2:
    k2_val = st.slider("Stage 2 Reranked Final Chunks (K2)", min_value=1, max_value=5, value=3, step=1)
with wb_col3:
    threshold_input = st.slider("Reranker Score Threshold", min_value=0.0, max_value=0.90, value=0.40, step=0.05)

# Run simulation
stage1_res, stage2_res = RerankingEvaluationSuite.execute_two_stage_rerank(top_k1=k1_val, top_k2=k2_val, threshold=threshold_input)

# Calculate metrics
metrics_stage1 = RerankingEvaluationSuite.calculate_metrics(stage1_res, k=k2_val)
metrics_stage2 = RerankingEvaluationSuite.calculate_metrics(stage2_res, k=k2_val)

st.subheader("Retrieval Quality Benchmark: Before vs After Reranking")
m1, m2, m3 = st.columns(3)
m1.metric("Hit Rate @ K", f"{metrics_stage2['hit_rate']:.2f}", delta=f"{metrics_stage2['hit_rate'] - metrics_stage1['hit_rate']:.2f} vs Stage 1")
m2.metric("Mean Reciprocal Rank (MRR)", f"{metrics_stage2['mrr']:.3f}", delta=f"{metrics_stage2['mrr'] - metrics_stage1['mrr']:.3f} vs Stage 1")
m3.metric("NDCG @ K", f"{metrics_stage2['ndcg']:.3f}", delta=f"{metrics_stage2['ndcg'] - metrics_stage1['ndcg']:.3f} vs Stage 1")

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.markdown("### Stage 1: Bi-Encoder Coarse Ranks")
    st.caption("Notice how DOC-B (installation guide) fooled the Bi-Encoder with high keyword overlap, pushing the true answer (DOC-A) down!")
    t1 = []
    for rank, doc in enumerate(stage1_res, 1):
        t1.append({
            "Rank": f"#{rank}",
            "ID": doc["id"],
            "Bi-Encoder Score": f"{doc['bi_score']:.2f}",
            "Ground Truth Rel": doc["ground_truth_rel"]
        })
    st.dataframe(pd.DataFrame(t1), use_container_width=True)

with res_col2:
    st.markdown("### Stage 2: Cross-Encoder Reranked Ranks")
    st.caption("The Cross-Encoder evaluates full cross-attention: it exposes DOC-B as irrelevant and elevates DOC-A (replica lag fix) to Rank #1!")
    t2 = []
    for rank, doc in enumerate(stage2_res, 1):
        t2.append({
            "Rank": f"#{rank}",
            "ID": doc["id"],
            "Cross-Encoder Score": f"{doc['cross_score']:.2f}",
            "Ground Truth Rel": doc["ground_truth_rel"]
        })
    st.dataframe(pd.DataFrame(t2), use_container_width=True)

st.divider()

# --- SECTION 8: INTERVIEW PREP ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why can't we use Cross-Encoders as the first-stage retriever across millions of documents?",
        "a": "A Cross-Encoder requires concatenating the query and document into a single sequence [CLS] Query [SEP] Doc [SEP] and passing them together through all 24 layers of bidirectional self-attention. Because attention is computed across the joint sequence, documents cannot be pre-embedded offline: every query requires executing a full transformer forward pass against every document in the database. For 1,000,000 documents, this would take hundreds of seconds and millions of GPU FLOPs per query. Bi-Encoders precompute vectors offline, allowing fast dot products in sub-10ms."
    },
    {
        "q": "What is the fundamental difference between MRR and NDCG in retrieval evaluation?",
        "a": "Mean Reciprocal Rank (MRR) evaluates binary relevance and focuses exclusively on the rank of the single highest relevant document: MRR = 1 / Rank_first. It completely ignores what documents appear after the first hit. Normalized Discounted Cumulative Gain (NDCG) evaluates graded relevance (e.g. 3 = Perfect, 2 = Partial, 0 = Irrelevant) across the entire returned list. It rewards systems for ordering all returned documents in descending order of utility using a logarithmic rank discount."
    },
    {
        "q": "How does ColBERT (Late Interaction) achieve near-Cross-Encoder accuracy with near-Bi-Encoder speed?",
        "a": "ColBERT (Khattab & Zaharia) avoids collapsing an entire document into a single vector. Instead, it precomputes token-level vectors for every token in the document offline. At query time, it embeds the query tokens and computes 'Late Interaction' using a MaxSim operator: for every query token, it finds the maximum dot product across all document tokens, and sums the maximums. This captures fine-grained token-level cross-matching without requiring full all-to-all cross-attention layers, running in ~15ms."
    },
    {
        "q": "What is the optimal funnel ratio between Stage 1 candidates (K1) and Stage 2 candidates (K2)?",
        "a": "The industry standard ratio is K1 in [30, 100] (typically K1 = 50) and K2 in [3, 5]. At K1 = 50, candidate recall is over 95%. Scoring 50 pairs in a single batched GPU forward pass takes approximately 20-25ms on an NVIDIA A10G/L4 GPU. Keeping K2 = 5 ensures the final context window injected into the LLM contains only high-relevance evidence (1,500 tokens), preventing prompt bloat and reducing generation prefill latency."
    },
    {
        "q": "Why does adding a Cross-Encoder reranker often reduce overall end-to-end RAG system latency?",
        "a": "While the Cross-Encoder forward pass adds ~25ms of retrieval compute, it filters out 45 noisy, irrelevant candidates. Passing only the top 5 gold-standard chunks (1,500 tokens) to the downstream LLM instead of 50 chunks (15,000 tokens) slashes LLM prompt prefill time by hundreds of milliseconds. In high-concurrency serving, smaller prompt tokens translate directly to faster TTFT and lower hosting costs."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Starving the Reranker with Too Few Stage 1 Candidates (Low K1)**
    Configuring Stage 1 to retrieve only 5 candidates (`K1 = 5`). A Cross-Encoder can only reorder documents that are actually 
    present in the candidate pool! If the true answer was ranked #8 by the Bi-Encoder, a `K1 = 5` funnel completely misses it. 
    Always size `K1 >= 30`.
    """)
    st.warning("""
    **Trap 3: Sequential CPU Loop Invocations of Cross-Encoder Scoring**
    Invoking `cross_encoder.predict([query, doc])` in a Python `for` loop over 50 documents on CPU. 
    Each pair incurs separate overhead, taking over 800ms! Always batch all 50 pairs into a single 2D tensor on GPU.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Discarding the Threshold Cutoff and Flooding the Prompt**
    Omitting a minimum Cross-Encoder probability threshold. On queries where zero relevant documents exist in the enterprise database, 
    the reranker will still return the top-3 least-bad documents (even with scores of 0.15), forcing the LLM to hallucinate from irrelevant text. 
    Enforce `threshold >= 0.70`.
    """)
    st.warning("""
    **Trap 4: Evaluating Retrieval Quality on Hit Rate Alone Without NDCG**
    Tracking only Hit Rate @ 5. A system that places the correct document at Rank #5 has the exact same Hit Rate (1.0) 
    as a system that places it at Rank #1. However, the Rank #5 system suffers from Lost-in-the-Middle degradation. 
    Always optimize for MRR and NDCG.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Cross-Encoder"):
        st.markdown("**How does Cross-Encoder attend?**\\n\\n`[CLS] Q [SEP] D [SEP]`. Full all-to-all cross-attention across all transformer layers.")
with fc2:
    with st.popover("Flashcard 2: Funnel Ratio"):
        st.markdown("**Standard Two-Stage Funnel?**\\n\\n`K1 = 50` candidates from vector/BM25 -> `K2 = 3 to 5` gold chunks passed to LLM.")
with fc3:
    with st.popover("Flashcard 3: MRR Metric"):
        st.markdown("**What is MRR?**\\n\\n`1 / Rank_first`. Measures the reciprocal rank of the first relevant hit.")
with fc4:
    with st.popover("Flashcard 4: Late Interaction"):
        st.markdown("**What is ColBERT MaxSim?**\\n\\nSums maximum dot products between token-level vector matrices. High speed + token interaction.")

st.divider()

# --- SECTION 9: SELF-ASSESSMENT ---
st.header("9. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why Bi-Encoders are required for Stage 1 candidate generation and why Cross-Encoders are too slow to scan full databases.",
    "I can explain the architectural difference between independent vector dot products and all-to-all cross-attention.",
    "I know how to calibrate raw Cross-Encoder logits via the Sigmoid function and apply score thresholding.",
    "I understand why a two-stage funnel (K1=50 -> K2=5) often reduces total end-to-end RAG latency.",
    "I can mathematically define and calculate Hit Rate@K, MRR@K, and NDCG@K.",
    "I understand Late Interaction architectures (ColBERTv2) and how MaxSim bridges the speed-precision divide."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of Cross-Encoder reranking, two-stage retrieval funnels, and retrieval evaluation metrics.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
