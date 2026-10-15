import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="23.2 Hybrid Retrieval - Embeddings Curriculum",
    page_icon="🔀",
    layout="wide"
)

# Title & Overview
st.title("23.2 Hybrid Retrieval: Dense, Sparse, Score Fusion, & Reciprocal Rank Fusion (RRF)")
st.caption("Chapter 23: Hybrid Search | Domain: 3. Vector Embeddings")
st.markdown("""
In hybrid search, querying a database executes **two independent retrievers in parallel**:
1. **Dense Retriever**: Evaluates dense semantic embeddings (e.g. OpenAI `text-embedding-3`, BGE) over an HNSW vector index.
2. **Sparse Retriever**: Evaluates lexical tokens (e.g. BM25, SPLADE) over an inverted index.

Each retriever produces a ranked list of candidate documents with completely incompatible score distributions:
* Dense similarity scores reside neatly between **-1.0 and 1.0** (or 0.0 and 1.0 for normalized cosine).
* BM25 scores are unbounded positive numbers ranging from **0.0 to 35.0+** depending on corpus size and term rarity.

How do you combine these two disparate candidate lists into a single, unified ranking?
* **Linear Score Fusion (Convex Blending)**: Normalizing scores and applying a weighted average: `alpha * Dense + (1 - alpha) * Sparse`.
* **Reciprocal Rank Fusion (RRF)**: Discarding raw scores entirely and merging candidates strictly based on their **rank positions**: `sum(1 / (k + rank))`.

This lesson explores the mathematics and operational engineering of hybrid fusion algorithms.
""")

st.divider()

# Section 1: Hybrid Architecture & Score Fusion Taxonomy
st.header("1. Dual-Retriever Architecture & Fusion Mathematics", divider="rainbow")

col_fus1, col_fus2 = col_fus1, col_fus2 = st.columns([1, 1])

with col_fus1:
    st.markdown("""
    ### Linear Score Fusion vs. Reciprocal Rank Fusion (RRF)
    * **1. Linear Convex Score Fusion**:
      * Formula: `Score = alpha * Norm(DenseScore) + (1 - alpha) * Norm(SparseScore)`
      * Requires normalizing BM25 scores (MinMax scaling or Sigmoid).
      * **Drawback**: Extremely brittle. If an unusual query produces an extreme BM25 outlier score (e.g. 45.0), the normalized scale distorts all other candidates.
    * **2. Reciprocal Rank Fusion (RRF)**:
      * Formula: `RRF Score(d) = sum(1 / (k + rank_i(d)))` across all retrievers $i$.
      * **Scale-Invariant**: Completely ignores raw score values; operates strictly on ordinal rank positions (1st, 2nd, 3rd).
      * **The Constant $k=60$**: Proven empirically (Cormack et al., 2009) to dampen the advantage of high-ranking outliers while rewarding documents that appear in the top results of *both* retrievers!
    """)

with col_fus2:
    st.markdown("### Parallel Dual-Retriever Pipeline")
    fuse_dot = """
    digraph HybridPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="User Query\\n'AWS Lambda timeout 504'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_dense {
            label="Dense Stream"; style="dashed"; color="#10B981";
            D_Embed [label="Bi-Encoder Vectorizer", fillcolor="#D1FAE5"];
            D_HNSW [label="HNSW Index Search\\nTop 20 Dense Candidates", fillcolor="#A7F3D0"];
            D_Embed -> D_HNSW;
        }

        subgraph cluster_sparse {
            label="Sparse Stream"; style="dashed"; color="#3B82F6";
            S_Invert [label="BM25 Inverted Index\\nTop 20 Sparse Candidates", fillcolor="#93C5FD"];
        }

        RRFNode [label="Reciprocal Rank Fusion (RRF)\\nRRF(d) = 1/(60+Rank_D) + 1/(60+Rank_S)", fillcolor="#FEF08A", color="#D97706", style="filled,bold"];
        FinalTopK [label="Consolidated Top-K Context\\nPassed to LLM Prompt", fillcolor="#10B981", color="#059669", style="filled,bold"];

        UserQ -> D_Embed;
        UserQ -> S_Invert;
        D_HNSW -> RRFNode;
        S_Invert -> RRFNode;
        RRFNode -> FinalTopK;
    }
    """
    st.graphviz_chart(fuse_dot, use_container_width=False)

st.markdown("### Linear Fusion vs. Reciprocal Rank Fusion (RRF)")
st.markdown("""
| Feature | Linear Score Fusion ($\alpha$) | Reciprocal Rank Fusion (RRF) |
| :--- | :--- | :--- |
| **Input Requirements** | Raw numerical scores (Cosine & BM25) | Ordinal rank positions only ($1, 2, 3, ...$) |
| **Score Normalization** | **Mandatory (MinMax or Sigmoid)** | **Zero normalization required** |
| **Sensitivity to Outliers**| High (Extreme BM25 scores distort rankings) | **Zero (Completely scale-invariant)** |
| **Hyperparameter Tuning**| Requires tuning $\alpha$ per dataset | Universal default $k=60$ works across all domains |
| **Industry Adoption** | Qdrant, Pinecone hybrid | Azure AI Search, Elasticsearch, Cohere Rerank |
""")

st.divider()

# Section 2: Production Code: Hybrid Dual-Engine Fusion
st.header("2. Production Code: Hybrid Dual-Engine with RRF & Linear Blending", divider="rainbow")

st.markdown("""
The production utility below demonstrates both Reciprocal Rank Fusion (RRF) and Normalized Linear Score Fusion across simulated Dense and Sparse candidate lists.
""")

st.code(r'''
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class HybridFusionRetriever:
    """
    Combines ranked candidate lists from Dense and Sparse retrievers
    using Reciprocal Rank Fusion (RRF) and Linear Convex Blending.
    """
    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
        k_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score: sum(1 / (k + rank)) across both candidate lists.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Process Dense Rankings
        for rank, item in enumerate(dense_results, 1):
            doc_id = item["id"]
            doc_store[doc_id] = item
            dense_ranks[doc_id] = rank
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)

        # 2. Process Sparse Rankings
        for rank, item in enumerate(sparse_results, 1):
            doc_id = item["id"]
            doc_store[doc_id] = item
            sparse_ranks[doc_id] = rank
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)

        # 3. Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        final_ranked = []
        for doc_id in sorted_ids[:top_k]:
            doc_data = doc_store[doc_id].copy()
            doc_data["rrf_score"] = rrf_scores[doc_id]
            doc_data["dense_rank"] = dense_ranks.get(doc_id, None)
            doc_data["sparse_rank"] = sparse_ranks.get(doc_id, None)
            final_ranked.append(doc_data)

        return final_ranked

    @staticmethod
    def linear_score_fusion(
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Linear combination: alpha * norm_dense + (1 - alpha) * norm_sparse
        """
        # Extract and normalize sparse scores to [0, 1]
        sparse_scores = [s[1] for s in sparse_results]
        s_min, s_max = min(sparse_scores, default=0.0), max(sparse_scores, default=1.0)
        s_range = max(1e-6, s_max - s_min)

        combined_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict[str, Any]] = {}

        for doc, score in dense_results:
            doc_id = doc["id"]
            doc_store[doc_id] = doc
            combined_scores[doc_id] += alpha * max(0.0, score)

        for doc, score in sparse_results:
            doc_id = doc["id"]
            doc_store[doc_id] = doc
            norm_sparse = (score - s_min) / s_range
            combined_scores[doc_id] += (1.0 - alpha) * norm_sparse

        sorted_ids = sorted(combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True)
        
        output = []
        for doc_id in sorted_ids[:top_k]:
            d = doc_store[doc_id].copy()
            d["blended_score"] = combined_scores[doc_id]
            output.append(d)
        return output
''', language="python")

st.divider()

# Section 3: Interactive Hybrid Fusion Arena
st.header("3. Interactive Workbench: Live Fusion Algorithm Arena", divider="rainbow")

st.markdown("""
Observe how **Reciprocal Rank Fusion (RRF)** dynamically promotes documents that appear in the top results of *both* retrievers over documents that only appear in one.
""")

# Sample candidates with different dense vs sparse ranks
candidates = [
    {"id": "doc_01", "title": "AWS Lambda Architecture & Timeout Configuration", "dense_rank": 3, "sparse_rank": 2, "text": "Configuring execution timeout limits and dead-letter queues in serverless AWS Lambda."},
    {"id": "doc_02", "title": "Nginx Ingress Error 504 Troubleshooting Log", "dense_rank": 12, "sparse_rank": 1, "text": "Upstream timeout error code 504 on reverse proxy gateways in Kubernetes."},
    {"id": "doc_03", "title": "Serverless Microservice Cold-Start Mitigation", "dense_rank": 1, "sparse_rank": 15, "text": "Provisioned concurrency eliminates initialization latency in Python microservices."},
    {"id": "doc_04", "title": "API Gateway Request Quota Management", "dense_rank": 2, "sparse_rank": 4, "text": "Throttling and rate-limiting incoming client HTTP calls with token bucket algorithms."},
    {"id": "doc_05", "title": "General Cloud Computing SLA Guarantees", "dense_rank": 20, "sparse_rank": 18, "text": "Standard monthly service level availability definitions and billing adjustments."}
]

col_f_cfg, col_f_view = st.columns([1, 1])

with col_f_cfg:
    st.subheader("1. Fusion Algorithm Parameters")
    algorithm_choice = st.radio(
        "Select Score Merging Method:",
        ["Reciprocal Rank Fusion (RRF)", "Linear Score Weighting (Alpha Blending)"],
        index=0
    )

    if algorithm_choice == "Reciprocal Rank Fusion (RRF)":
        k_val = st.slider("RRF Smoothing Constant ($k$):", min_value=1, max_value=100, value=60, help="Standard industry default is 60 (Cormack et al.).")
        st.markdown(f"**Active Formula:** `RRF(d) = 1 / ({k_val} + DenseRank) + 1 / ({k_val} + SparseRank)`")
    else:
        alpha_val = st.slider(
            "Dense Weight ($\\alpha$):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0 = Pure Dense Semantic. 0.0 = Pure Sparse BM25. 0.5 = Equal Blend."
        )
        st.markdown(f"**Active Formula:** `Score = {alpha_val:.2f} * DenseScore + {1.0 - alpha_val:.2f} * SparseScore`")

    k_display = st.slider("Results to Display ($K$):", min_value=1, max_value=5, value=4)

with col_f_view:
    st.subheader(f"2. Unified Final Ranking ({algorithm_choice.split('(')[0]})")

    table_data = []
    for c in candidates:
        d_rank = c["dense_rank"]
        s_rank = c["sparse_rank"]

        if algorithm_choice == "Reciprocal Rank Fusion (RRF)":
            score = (1.0 / (k_val + d_rank)) + (1.0 / (k_val + s_rank))
            score_formatted = f"{score:.5f}"
        else:
            # Simulate normalized scores from ranks
            norm_dense = max(0.0, 1.0 - (d_rank - 1) / 20.0)
            norm_sparse = max(0.0, 1.0 - (s_rank - 1) / 20.0)
            score = alpha_val * norm_dense + (1.0 - alpha_val) * norm_sparse
            score_formatted = f"{score:.3f}"

        table_data.append({
            "doc_id": c["id"],
            "title": c["title"],
            "dense_rank": f"#{d_rank}",
            "sparse_rank": f"#{s_rank}",
            "final_score": score,
            "score_str": score_formatted,
            "text": c["text"]
        })

    table_data.sort(key=lambda x: x["final_score"], reverse=True)

    st.markdown("#### Merged & Re-Ranked Candidate Table")
    df_ranked = pd.DataFrame([
        {
            "Rank": f"#{i+1}",
            "Doc ID": row["doc_id"],
            "Title": row["title"],
            "Dense Rank": row["dense_rank"],
            "Sparse Rank": row["sparse_rank"],
            "Combined Score": row["score_str"]
        }
        for i, row in enumerate(table_data[:k_display])
    ])
    st.dataframe(df_ranked, use_container_width=True)

    st.info("💡 **Notice the Fusion Dynamics**: `doc_04` and `doc_01` rank in the top positions because they achieved strong scores across **both** retrievers, beating documents that ranked #1 in only one retriever!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Fusion Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Reciprocal Rank Fusion (RRF) preferred over linear weighted score averaging in enterprise search?**
    * **Answer**:
      1. **Scale Invariance**: Dense vectors produce cosine scores bounded in `[-1, 1]`, while BM25 produces unbounded positive numbers `[0, 30+]`. Normalizing BM25 across arbitrary queries requires complex tuning and fails on outliers. RRF uses ordinal rank integers ($1, 2, 3$), making it 100% scale-invariant.
      2. **Robust Default ($k=60$)**: Decades of empirical Information Retrieval research show that RRF with $k=60$ performs within 1% of the theoretical best-tuned weighted model on almost every benchmark without any hyperparameter tuning.
      3. **Natural Redundancy Promotion**: A document that appears at Rank 3 in Dense and Rank 4 in BM25 will consistently defeat a document that is Rank 1 in Dense but completely absent from BM25.

    **Q2: What is the purpose of the smoothing constant $k$ (typically 60) in the RRF formula `1 / (k + rank)`?**
    * **Answer**: Without $k$ (i.e. `1 / rank`), Rank 1 receives a massive score of $1.0$, while Rank 2 receives $0.5$ (a 50% drop), heavily over-biasing the algorithm toward whichever document got first place in a single retriever.
    Adding $k=60$ flattens the curve:
    * Rank 1 receives `1 / 61 = 0.01639`.
    * Rank 2 receives `1 / 62 = 0.01612` (only a 1.6% drop).
    This ensures that appearing in the top 5 of *both* retrievers (`0.016 + 0.015 = 0.031`) easily beats appearing at Rank 1 in only one retriever (`0.016 + 0 = 0.016`).

    **Q3: What is SPLADE (Sparse Lexical and Expansion Model) and how does it relate to hybrid search?**
    * **Answer**: SPLADE is a neural network model that predicts **learned sparse vector representations**.
    Unlike BM25 which matches exact input words, SPLADE expands the vocabulary with semantically related words directly into the inverted index!
    It produces high-dimensional sparse vectors that can be stored in standard inverted indexes (Lucene / Elasticsearch) while incorporating neural semantic expansion.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Passing Insufficient Candidate Depths to RRF**:
      * *Issue*: Fetching only top $K=3$ from Dense and top $K=3$ from BM25 before computing RRF.
      * *Consequence*: If the best overall document was at Rank 4 in both retrievers, it is completely absent from the fusion pool.
      * *Fix*: Fetch at least 20 to 50 candidates from each retriever before merging into final Top-5.

    * **Trap 2: Normalizing BM25 with Hardcoded Global Maximums**:
      * *Issue*: Dividing BM25 scores by a hardcoded 25.0 assuming BM25 never exceeds 25.
      * *Consequence*: On long documents with rare acronyms, BM25 can reach 40.0, overflowing the normalized scale ($> 1.0$) and skewing linear weights.
      * *Fix*: Always use query-specific dynamic MinMax scaling or switch to RRF.

    * **Trap 3: Linear Score Weighting Overfitting to a Single Test Query**:
      * *Issue*: Tuning $\\alpha = 0.85$ on a single query and deploying it to production.
      * *Consequence*: When users search for exact product codes or error numbers, the 85% dense weighting causes the system to miss exact keyword matches.
      * *Fix*: Validate fusion parameters across a heterogeneous query benchmark.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Reciprocal Rank Fusion (RRF)**
      * *Definition*: A scale-invariant rank aggregation algorithm merging candidates from multiple retrievers using `1 / (k + rank)`.
    * **Flashcard 2: The RRF Smoothing Constant ($k=60$)**
      * *Definition*: The empirical parameter dampening top-rank advantages to reward documents retrieved in the top results of multiple search engines.
    * **Flashcard 3: Linear Score Fusion (Convex Blending)**
      * *Definition*: Combining normalized dense cosine similarity and sparse BM25 scores via `alpha * Dense + (1 - alpha) * Sparse`.
    * **Flashcard 4: SPLADE**
      * *Definition*: A neural sparse representation model that predicts learned term expansions directly into inverted index vocabularies.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Hybrid Retrieval:

- [ ] I can articulate the architectural difference between Linear Score Fusion and RRF.
- [ ] I understand why RRF is scale-invariant and requires zero score normalization.
- [ ] I know why $k=60$ in RRF promotes documents appearing across multiple retrievers.
- [ ] I understand why candidate depth ($K=20-50$) must be retrieved prior to RRF merging.
- [ ] I can implement both RRF and Linear convex blending in production code.
""")
