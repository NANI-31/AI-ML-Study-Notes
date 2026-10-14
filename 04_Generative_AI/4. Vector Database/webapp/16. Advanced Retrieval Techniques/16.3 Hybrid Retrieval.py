import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="16.3 Hybrid Retrieval - Vector DB Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("16.3 Hybrid Retrieval: Combining BM25 Sparse Search, Dense Vectors & Reciprocal Rank Fusion (RRF)")
st.caption("Chapter 16: Advanced Retrieval Techniques | Domain: 4. Vector Database")
st.markdown("""
In the real world, neither pure dense vector search nor pure keyword search is sufficient on its own:
* **Dense Vector Search** excels at understanding **semantic intent, synonyms, and natural language concepts**, 
  but notoriously fails on exact part numbers, error codes, legal statute citations, and specific acronyms.
* **Sparse Keyword Search (BM25)** excels at **exact lexical matching, rare proper nouns, and unique identifier lookups**, 
  but is completely blind to synonyms and paraphrasing (the *Vocabulary Mismatch problem*).

**Hybrid Retrieval** unites both paradigms: querying a dense vector database and an inverted keyword index simultaneously, 
then merging their rankings using **Reciprocal Rank Fusion (RRF)**.

In this lesson, we analyze the strengths and failure modes of Sparse vs. Dense retrieval, master the mathematics 
and invariants of Reciprocal Rank Fusion (RRF), implement a production Python hybrid retriever, and experiment 
with an interactive Live Hybrid Search & RRF Rank Visualizer.
""")

st.divider()

# Section 1: The Complementary Dual-Engine Paradigm
st.header("1. The Complementary Dual-Engine Paradigm", divider="rainbow")

col_dual1, col_dual2 = col_dual1, col_dual2 = st.columns([1.1, 0.9])

with col_dual1:
    st.markdown("""
    ### Dense vs. Sparse: The Yin and Yang of Information Retrieval
    
    | Dimension | Dense Vector Retrieval (ChromaDB / HNSW) | Sparse Keyword Retrieval (BM25 / Inverted Index) |
    | :--- | :--- | :--- |
    | **Core Mechanism** | Bi-encoder neural embeddings in continuous metric space. | Term Frequency / Inverse Document Frequency (TF-IDF). |
    | **Superpower** | Conceptual semantics, synonyms (*"fast car"* matches *"rapid automobile"*). | Exact lexical token matches, SKU codes, function signatures. |
    | **Fatal Weakness** | Misses exact identifiers (e.g. `ERR_SSL_PROTOCOL_ERROR` or `CVE-2024-38077`). | Zero synonym understanding (*"slow network"* misses *"high latency"*). |
    | **Data Structure** | Multi-layer proximity graph (HNSW). | Inverted post lists (Posting lists with skip pointers). |
    | **Production Role** | High-recall semantic foundation. | High-precision exact keyword filter. |
    """)

with col_dual2:
    st.markdown("### The Hybrid Retrieval Pipeline Architecture")
    hyb_dot = """
    digraph HybridPipeline {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Query [label="User Query\\n'Fix error ERR_SSL_PROTOCOL_ERROR in Envoy'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_dual {
            label="Parallel Search Engines"; style="dashed"; color="#059669";
            Dense [label="Dense Vector Engine (ChromaDB)\\nTop-20 Semantic Neighbors", fillcolor="#D1FAE5", color="#059669"];
            Sparse [label="Sparse Keyword Engine (BM25)\\nTop-20 Lexical Keyword Hits", fillcolor="#FEF3C7", color="#D97706"];
        }

        RRF [label="Reciprocal Rank Fusion (RRF)\\nRRF_Score = sum( 1 / (60 + Rank) )", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        TopK [label="Final Grounded Context Chunks\\n(Highest combined rank)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Query -> Dense;
        Query -> Sparse;
        Dense -> RRF [label="Dense Ranks 1..20"];
        Sparse -> RRF [label="Sparse Ranks 1..20"];
        RRF -> TopK;
    }
    """
    st.graphviz_chart(hyb_dot, use_container_width=False)

st.divider()

# Section 2: Merging Search Results: Linear Combination vs RRF
st.header("2. Merging Search Results: Linear Combination vs RRF", divider="rainbow")

col_merge1, col_merge2 = col_merge1, col_merge2 = st.columns([1, 1])

with col_merge1:
    st.markdown("""
    ### Why Linear Score Combination is Fragile
    A naive approach to hybrid retrieval tries to combine scores linearly:
    ```text
    Final_Score = alpha * Dense_Score + (1 - alpha) * Sparse_Score
    ```
    * **The Normalization Disaster**:
      * Dense Cosine Similarity is strictly bounded in `[0.0, 1.0]`.
      * BM25 scores are unbounded (`[0.0, +infinity)`), depending on document length and corpus frequency.
      * Normalizing BM25 using min-max scaling varies wildly on every single query depending on the top hit!
      * In practice, the system requires endless manual tuning of `alpha` that breaks across different queries.
    """)

with col_merge2:
    st.markdown("""
    ### The Elegance of Reciprocal Rank Fusion (RRF)
    Published by Cormack, Clarke, and Buettcher (SIGIR 2009), **Reciprocal Rank Fusion (RRF)** discards arbitrary raw scores entirely, operating strictly on **ordinal ranks**:
    
    ```text
    RRF_Score(d) = sum_{m in Models} [ 1.0 / ( k_const + Rank_m(d) ) ]
    ```
    
    Where:
    * **`Rank_m(d)`**: The 1-based rank position of document `d` in model `m` (1st, 2nd, 3rd...).
    * **`k_const`**: A smoothing constant (standard industry default: **`60`**).
    * **Key Advantage**: Zero score calibration required! A document appearing in the top ranks of *both* lists receives a massive boost, while documents strong in only one list are gracefully blended.
    """)

st.divider()

# Section 3: Building a Production Hybrid Retriever in Python
st.header("3. Building a Production Hybrid Retriever in Python", divider="rainbow")

st.markdown("""
Below is a complete, production-grade implementation of a **Hybrid Retriever with Reciprocal Rank Fusion**:
""")

st.code("""from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

class HybridRetriever:
    \"\"\"Unites ChromaDB dense search with BM25 sparse search via RRF.\"\"\"
    
    def __init__(self, chroma_collection, documents: List[str], metadatas: List[Dict]):
        self.collection = chroma_collection
        self.documents = documents
        self.metadatas = metadatas
        # Tokenize corpus for BM25
        self.corpus_tokens = [doc.lower().split() for doc in documents]

    def _mock_bm25_search(self, query: str, top_n: int = 20) -> List[int]:
        \"\"\"Scores documents based on keyword frequency match.\"\"\"
        q_tokens = set(query.lower().split())
        scores = []
        for idx, doc_toks in enumerate(self.corpus_tokens):
            match_count = sum(1 for t in doc_toks if t in q_tokens)
            scores.append((match_count, idx))
        # Sort descending by match count
        scores.sort(key=lambda x: x[0], reverse=True)
        return [idx for count, idx in scores[:top_n] if count > 0]

    def search_hybrid(self, query_text: str, k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # 1. Execute Dense Search in ChromaDB
        dense_results = self.collection.query(query_texts=[query_text], n_results=20)
        dense_ids = dense_results["ids"][0]
        
        # 2. Execute Sparse Keyword Search
        sparse_indices = self._mock_bm25_search(query_text, top_n=20)
        sparse_ids = [f"doc_{idx}" for idx in sparse_indices]

        # 3. Compute Reciprocal Rank Fusion (RRF) Scores
        rrf_scores = defaultdict(float)

        # Process Dense Ranks
        for rank_1_based, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank_1_based)

        # Process Sparse Ranks
        for rank_1_based, doc_id in enumerate(sparse_ids, start=1):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank_1_based)

        # 4. Sort documents by final RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [{"id": doc_id, "rrf_score": score} for doc_id, score in sorted_docs]
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Hybrid Search & RRF Rank Visualizer
st.header("4. Interactive Workbench: Live Hybrid Search & RRF Rank Visualizer", divider="rainbow")

st.markdown("""
Test how Reciprocal Rank Fusion combines sparse and dense rankings. Select a scenario containing 
an exact error code and observe how RRF guarantees both keyword precision and semantic depth.
""")

col_scen_in, col_scen_cfg = st.columns([1.1, 0.9])

with col_scen_in:
    st.subheader("1. Test Search Scenario")
    scenario_sel = st.selectbox(
        "Choose Query Scenario:",
        [
            "Exact Error Code: 'How to fix ERR_ENVOY_TLS_082 failure'",
            "Conceptual Synonym: 'How to speed up sluggish database connections'",
            "Mixed Technical Inquiry: 'PostgreSQL WAL recovery window under SOC2'"
        ]
    )

with col_scen_cfg:
    st.subheader("2. RRF Parameters")
    rrf_k_const = st.slider("RRF Constant (k_const):", min_value=10, max_value=100, value=60, step=5)
    final_k = st.slider("Final Merged Top-K:", min_value=2, max_value=4, value=3)

# Mock Candidates & Ranks
if "ERR_ENVOY" in scenario_sel:
    MOCK_RANKS = [
        {"id": "DOC-ERR-082", "title": "Runbook: Resolving ERR_ENVOY_TLS_082 Handshake Mismatch", "dense_rank": 8, "sparse_rank": 1},
        {"id": "DOC-TLS-OVERVIEW", "title": "Envoy Ingress TLS Protocol Overview", "dense_rank": 1, "sparse_rank": 5},
        {"id": "DOC-CERT-ROTATION", "title": "Automated SPIRE Certificate Renewal", "dense_rank": 2, "sparse_rank": 9},
        {"id": "DOC-GENERIC-LOGS", "title": "Error Logging & Ingress Tracing Guidelines", "dense_rank": 3, "sparse_rank": 2}
    ]
elif "sluggish" in scenario_sel:
    MOCK_RANKS = [
        {"id": "DOC-LATENCY-MIT", "title": "Mitigating Latency Spikes in Connection Pools", "dense_rank": 1, "sparse_rank": 8},
        {"id": "DOC-PG-TUNING", "title": "PostgreSQL Max Connection & Buffer Tuning", "dense_rank": 2, "sparse_rank": 7},
        {"id": "DOC-SLOW-QUERY", "title": "Sluggish Query Diagnostic Runbook", "dense_rank": 6, "sparse_rank": 1},
        {"id": "DOC-REDIS-CACHE", "title": "Redis Read-Replica Caching Architecture", "dense_rank": 3, "sparse_rank": 9}
    ]
else:
    MOCK_RANKS = [
        {"id": "DOC-SOC2-SLA", "title": "SOC2 Infrastructure Backup & PITR Mandates", "dense_rank": 1, "sparse_rank": 1},
        {"id": "DOC-PG-WAL", "title": "PostgreSQL Continuous WAL Archiving to S3", "dense_rank": 2, "sparse_rank": 2},
        {"id": "DOC-DISASTER-REC", "title": "Disaster Recovery RTO and RPO Targets", "dense_rank": 3, "sparse_rank": 6},
        {"id": "DOC-AUDIT-LOG", "title": "Compliance Log Retention Timelines", "dense_rank": 5, "sparse_rank": 3}
    ]

# Calculate RRF Scores
for doc in MOCK_RANKS:
    score_dense = 1.0 / (rrf_k_const + doc["dense_rank"])
    score_sparse = 1.0 / (rrf_k_const + doc["sparse_rank"])
    doc["score_dense"] = score_dense
    doc["score_sparse"] = score_sparse
    doc["rrf_score"] = score_dense + score_sparse

sorted_rrf = sorted(MOCK_RANKS, key=lambda x: x["rrf_score"], reverse=True)

st.divider()

col_table_res, col_table_analysis = st.columns([1.2, 0.8])

with col_table_res:
    st.subheader("Reciprocal Rank Fusion Ranking Table")
    
    df_rrf = pd.DataFrame([
        {
            "Final Rank": f"#{idx+1}",
            "Document ID": d["id"],
            "Document Title": d["title"],
            "Dense Rank": f"#{d['dense_rank']}",
            "Sparse Rank": f"#{d['sparse_rank']}",
            "RRF Score": f"{d['rrf_score']:.5f}"
        }
        for idx, d in enumerate(sorted_rrf)
    ])
    st.dataframe(df_rrf, use_container_width=True, hide_index=True)

with col_table_analysis:
    st.subheader("Fusion Analysis & Winner")
    winner = sorted_rrf[0]
    st.success(f"🏆 **Top Result**: **{winner['id']}**")
    st.markdown(f"""
    * **Title**: {winner['title']}
    * **Dense Contribution**: `1 / ({rrf_k_const} + {winner['dense_rank']}) = {winner['score_dense']:.5f}`
    * **Sparse Contribution**: `1 / ({rrf_k_const} + {winner['sparse_rank']}) = {winner['score_sparse']:.5f}`
    * **Total RRF Score**: **`{winner['rrf_score']:.5f}`**
    """)
    
    if "ERR_ENVOY" in scenario_sel:
        st.info("💡 Notice: Even though Dense search ranked the exact runbook at #8 (due to lack of vocabulary for random error IDs), BM25 ranked it at #1. RRF combined both signals to elevate it to the top!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is Reciprocal Rank Fusion (RRF) preferred over linear score normalization in production hybrid search?"):
    st.markdown("""
    **Answer**:
    1. **Scale Incompatibility**: Dense cosine similarity is bounded in `[-1.0, 1.0]`, while BM25 keyword scores are unbounded positive numbers (`0.0` to `+inf`) based on term frequency and document length.
    2. **Min-Max Fragility**: Attempting to normalize BM25 with `(score - min) / (max - min)` creates erratic score distributions that fluctuate wildly depending on whether a query contains 1 or 5 search terms.
    3. **Zero Calibration**: RRF evaluates **ordinal positions (ranks)** rather than raw scores: `1.0 / (60 + rank)`. It requires zero parameter tuning, remains completely stable across document collection updates, and mathematically boosts items that perform well across both engines.
    """)

with st.expander("Q2: What is the purpose of the smoothing constant `k = 60` in the RRF formula?"):
    st.markdown("""
    **Answer**:
    The constant `k = 60` was empirically established in the original 2009 Cormack et al. research paper:
    * It mitigates the impact of **rank volatility** at the very top of the list.
    * Without `k` (i.e. `1 / rank`), Rank 1 gets a score of `1.0`, while Rank 2 gets `0.5` (a 50% drop!). This would cause Rank 1 in one system to completely overpower Rank 2 and Rank 3 in another.
    * With `k = 60`:
      * Rank 1 score: `1 / 61 = 0.01639`
      * Rank 2 score: `1 / 62 = 0.01612` (only a 1.6% drop).
    * This allows documents that achieve good ranks (e.g. Rank 3 in both lists) to outscore a document that ranked #1 in one list but was completely absent in the other.
    """)

with st.expander("Q3: When does Dense Vector search fail completely, and how does Sparse BM25 save the retrieval?"):
    st.markdown("""
    **Answer**:
    Dense Vector search fails in **Out-Of-Vocabulary (OOV) and Exact Identifier Scenarios**:
    * **Serial Numbers & Product SKUs**: `HW-GPU-H100-SXM5-80GB`.
    * **Error Codes & Exception Names**: `NullPointerException`, `ERR_CONNECTION_REFUSED_0x82`.
    * **Specific Legal Statute Citations**: `GDPR Article 33 Paragraph 1`.
    * **Variable & Function Identifiers**: `def authenticate_jwt_claims()`.
    Embedding model tokenizers split these exact terms into fragmented sub-word tokens, producing generic, diluted vectors. Sparse BM25 treats them as exact lexical tokens, finding the precise document in milliseconds.
    """)

with st.expander("Q4: Can BM25 and Vector Search be run in parallel, and how does that affect query latency?"):
    st.markdown("""
    **Answer**:
    **Yes**. Because BM25 and vector search operate on independent indices:
    * In production, the query service dispatches asynchronous requests concurrently:
      ```python
      dense_future = thread_pool.submit(vector_store.query, query)
      sparse_future = thread_pool.submit(bm25_index.search, query)
      dense_res, sparse_res = dense_future.result(), sparse_future.result()
      ```
    * Total retrieval latency equals `max(Dense_Latency, Sparse_Latency) + RRF_Merge_Latency`.
    * Since BM25 takes ~1ms and HNSW takes ~2ms, parallel hybrid search executes in **under 3 milliseconds total**.
    """)

with st.expander("Q5: What are Learned Sparse Embeddings (e.g. SPLADE) and how do they fit into hybrid retrieval?"):
    st.markdown("""
    **Answer**:
    * Traditional BM25 uses static lexical tokens without semantic expansion.
    * **SPLADE (Sparse Lexical and Expansion Model)**: A neural model based on BERT that generates **sparse vectors in vocabulary space** (30,000 dimensions, where most values are 0).
    * SPLADE automatically performs **term expansion** (e.g., when encoding *"sluggish database"*, it activates vocabulary weights for *"slow"*, *"latency"*, and *"PostgreSQL"*).
    * It can be stored directly inside inverted indices, providing the semantic expansion of dense vectors with the exact-match precision of inverted indices.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Naive Linear Score Combination Without Normalization**  
    Adding `0.9 * Cosine + 0.1 * BM25` where BM25 score is 42.0 causes the BM25 term to completely dominate the result (4.2 vs 0.9), rendering vector search useless. Use RRF.
    """)
    st.warning("""
    **Trap 2: Setting `k_const` Too Low in RRF (e.g. `k = 1`)**  
    Setting `k_const = 1` makes the score distribution extremely steep, penalizing Rank 2 documents unfairly. Stick to standard `k_const = 60`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Sequential Serial Querying**  
    Running BM25 first, waiting for it to finish, and then running Vector Search doubles query latency. Always execute both queries asynchronously in parallel.
    """)
    st.warning("""
    **Trap 4: Discarding Metadata Filters in BM25**  
    Applying metadata filtering to the vector database but forgetting to apply the same filter to the BM25 engine allows unauthorized tenant documents to contaminate the merged ranking.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Hybrid Retrieval?"):
        st.markdown("""
        **Dense + Sparse**:  
        Combines semantic vector search (ChromaDB) with lexical keyword search (BM25) to achieve both high recall and exact precision.
        """)
    with st.expander("🗂️ Flashcard: What is the RRF formula?"):
        st.markdown("""
        **RRF Equation**:  
        `Score = sum( 1 / (60 + Rank) )`. Merges search rankings using ordinal positions with zero score calibration.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why is k=60 used in RRF?"):
        st.markdown("""
        **Rank Smoothing**:  
        Prevents Rank 1 from disproportionately overpowering Rank 2 and Rank 3, rewarding documents that rank well across both engines.
        """)
    with st.expander("🗂️ Flashcard: What are Learned Sparse Embeddings?"):
        st.markdown("""
        **SPLADE**:  
        Neural models that generate sparse vocabulary-space vectors with automated term expansion for inverted indices.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why dense vector search and sparse BM25 search are complementary.", value=True)
chk2 = st.checkbox("I know why linear score combination fails and why Reciprocal Rank Fusion (RRF) is the gold standard.", value=True)
chk3 = st.checkbox("I can calculate RRF scores with `k_const = 60` across multiple ranking lists.", value=True)
chk4 = st.checkbox("I can implement an asynchronous parallel hybrid retriever in Python.", value=True)
chk5 = st.checkbox("I understand the role of learned sparse embeddings like SPLADE in modern retrieval.", value=True)
chk6 = st.checkbox("I can architect a resilient hybrid search pipeline that handles exact error codes and semantic intent.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Hybrid Retrieval and are ready for Chapter 16.4: Query Transformation!")
