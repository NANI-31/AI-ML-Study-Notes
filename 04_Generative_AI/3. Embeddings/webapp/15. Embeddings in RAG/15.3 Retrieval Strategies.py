import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Set

st.set_page_config(
    page_title="15.3 Retrieval Strategies - Embeddings Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("15.3 Advanced Retrieval Strategies: Top-K, Thresholding, Hybrid Search, & Multi-Query")
st.caption("Chapter 15: Embeddings in RAG | Domain: 3. Vector Embeddings")
st.markdown("""
Naive RAG systems rely exclusively on basic Top-K vector retrieval: embed the query, find the $K$ closest vectors, and pass them to the LLM.
In enterprise production, **pure Top-K retrieval frequently fails**:
* **The Noise Flood**: When a user asks an out-of-domain question, Top-K still returns $K$ chunks, filling the prompt with misleading garbage.
* **The Synonym / Out-of-Vocabulary Blindspot**: Dense embeddings can struggle with exact acronyms, error codes, part numbers, or rare technical terms where BM25 keyword search excels.
* **The Query Phrasing Trap**: A user's specific phrasing might match poorly against document headings.

To achieve production-grade 99%+ accuracy, modern architectures deploy **Advanced Retrieval Strategies**:
* **Top-K Retrieval**: Fixed-depth nearest neighbor ranking.
* **Similarity Thresholding**: Dynamic score cutoffs to reject irrelevant noise.
* **Metadata Filtering**: Partitioning search by tenant, time, security tags, or categories.
* **Hybrid Search**: Fusing Dense vector search with Sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF).
* **Query Expansion & HyDE**: Expanding queries or generating hypothetical document embeddings.
* **Multi-Query Retrieval**: Decomposing ambiguous prompts into multiple diverse search queries.
""")

st.divider()

# Section 1: Taxonomy & Architecture of Advanced Retrieval Strategies
st.header("1. Taxonomy & Architecture of Retrieval Strategies", divider="rainbow")

col_strat1, col_strat2 = st.columns([1, 1])

with col_strat1:
    st.markdown("""
    ### The 6 Core Retrieval Paradigms
    * **1. Top-K Retrieval**:
      * Retrieves exactly $K$ nearest neighbors. Fast and simple, but vulnerable to false positives when no relevant content exists.
    * **2. Similarity Thresholding**:
      * Discards any candidate chunk whose similarity score is below $\\tau$ (e.g., $< 0.70$). Adapts dynamically from 0 to $K$ chunks.
    * **3. Metadata Filtering**:
      * Enforces hard constraints (`department == 'engineering' AND year >= 2024`) before traversing or ranking vector graphs.
    * **4. Hybrid Search (Dense + Sparse Fusion)**:
      * Combines Dense embeddings (captures conceptual meaning) with Sparse BM25 / SPLADE (captures exact codes, IDs, and jargon).
      * Merged via **Reciprocal Rank Fusion (RRF)**:
        `RRF Score = 1 / (60 + Rank_Dense) + 1 / (60 + Rank_Sparse)`
    * **5. Query Expansion & HyDE**:
      * Generates hypothetical answers or synonym expansions before embedding to bridge the query-passage vocabulary gap.
    * **6. Multi-Query Retrieval**:
      * Prompts an LLM to generate 3-5 query reformulations, queries the vector database concurrently for all variations, and deduplicates the merged candidate set.
    """)

with col_strat2:
    st.markdown("### Hybrid & Multi-Query Retrieval Architecture")
    strat_dot = """
    digraph RetrievalStrategies {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="User Query\\n'Error 504 on Nginx ingress'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_hybrid {
            label="Hybrid Search Engines"; style="dashed"; color="#8B5CF6";
            DenseRet [label="Dense Vector Search\\n(Semantic meaning)", fillcolor="#EDE9FE"];
            SparseRet [label="Sparse BM25 Search\\n(Exact token matches)", fillcolor="#DDD6FE"];
            RRFNode [label="Reciprocal Rank Fusion\\nRRF Score Fusion", fillcolor="#C4B5FD", color="#7C3AED", style="filled,bold"];
        }

        subgraph cluster_multi {
            label="Multi-Query Expansion"; style="dashed"; color="#F59E0B";
            MQGen [label="LLM Multi-Query Generator\\nGenerates 3 variations", fillcolor="#FEF3C7"];
        }

        FilterNode [label="Score Threshold & Metadata Filter\\nDrop scores < 0.65", fillcolor="#D1FAE5", color="#10B981"];
        FinalK [label="Final Re-Ranked Chunks\\nPassed to Prompt Context", fillcolor="#10B981", style="filled,bold"];

        UserQ -> DenseRet [label="Vectorize"];
        UserQ -> SparseRet [label="Tokenize"];
        UserQ -> MQGen [label="Expand variations"];
        DenseRet -> RRFNode;
        SparseRet -> RRFNode;
        RRFNode -> FilterNode;
        FilterNode -> FinalK;
    }
    """
    st.graphviz_chart(strat_dot, use_container_width=False)

st.markdown("### Strategy Comparison Matrix")
st.markdown("""
| Retrieval Strategy | Latency Overhead | Handles Jargon / Part IDs | Rejects Irrelevant Queries | Resolves Ambiguous Phrasing | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-K Retrieval** | Lowest (Baseline) | Poor | Poor (Forces $K$ items) | Poor | Quick prototyping, narrow corpora. |
| **Similarity Threshold** | Negligible | Poor | **Excellent** (Returns 0 if irrelevant) | Poor | Q&A systems with strict hallucination limits. |
| **Metadata Filtering** | Low | N/A (Categorical) | **Excellent** | Neutral | Multi-tenant apps, role-based access control. |
| **Hybrid Search (RRF)** | Low (+5-15ms) | **Outstanding** | Good | Moderate | Enterprise search, technical manuals, bug logs. |
| **Multi-Query** | Moderate (+200-400ms) | Moderate | Moderate | **Outstanding** | Complex user prompts, exploratory search. |
""")

st.divider()

# Section 2: Production Code: Hybrid Search & RRF Engine
st.header("2. Production Code: Multi-Strategy Engine with Reciprocal Rank Fusion", divider="rainbow")

st.markdown("""
The production utility below implements a complete Hybrid Search engine combining Dense Cosine Similarity and Sparse Lexical BM25 ranking via Reciprocal Rank Fusion (RRF), along with dynamic threshold filtering.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

class HybridRetrievalEngine:
    """
    Implements Hybrid Dense + Sparse BM25 retrieval fused with
    Reciprocal Rank Fusion (RRF) and dynamic similarity thresholding.
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        self.corpus: List[Dict[str, Any]] = []

    def mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            doc["vector"] = self.mock_embed(doc["text"])
            doc["tokens"] = set(doc["text"].lower().split())
            self.corpus.append(doc)

    def dense_search(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = self.mock_embed(query)
        scored = []
        for doc in self.corpus:
            sim = float(np.dot(doc["vector"], q_vec))
            scored.append((doc, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def sparse_search(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            # Overlap token score simulating BM25 lexical match
            overlap = len(doc["tokens"].intersection(q_tokens))
            score = float(overlap) / max(1, len(q_tokens))
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def hybrid_rrf_search(self, query: str, top_k: int = 3, rrf_k: int = 60, min_threshold: float = 0.0) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(query)
        sparse_results = self.sparse_search(query)

        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        # 1. Accumulate Dense RRF scores
        for rank, (doc, sim) in enumerate(dense_results, 1):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Accumulate Sparse RRF scores
        for rank, (doc, lex_score) in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort by combined RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if score >= min_threshold:
                results.append({"doc": doc_lookup[doc_id], "rrf_score": score})

        return results
''', language="python")

st.divider()

# Section 3: Interactive Retrieval Strategy Arena
st.header("3. Interactive Workbench: Retrieval Strategy Comparison Arena", divider="rainbow")

st.markdown("""
Compare how different retrieval algorithms respond to the same query on an enterprise technical corpus.
""")

class StrategyArenaEngine:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

arena = StrategyArenaEngine()

corpus_items = [
    {"id": "doc_01", "text": "Kubernetes Ingress controllers route external HTTP traffic to internal cluster services.", "category": "devops", "year": 2024},
    {"id": "doc_02", "text": "Error 504 Gateway Timeout occurs when an upstream gateway fails to respond within proxy_read_timeout in Nginx.", "category": "troubleshooting", "year": 2023},
    {"id": "doc_03", "text": "Multi-head attention projects queries, keys, and values into parallel subspaces for sequence modeling.", "category": "ai", "year": 2024},
    {"id": "doc_04", "text": "PostgreSQL error code 28P01 indicates invalid password authentication during database connection establishment.", "category": "troubleshooting", "year": 2022},
    {"id": "doc_05", "text": "LoRA fine-tuning freezes base LLM weights and updates low-rank parameter adapter matrices to conserve VRAM.", "category": "ai", "year": 2024},
    {"id": "doc_06", "text": "AWS S3 403 Forbidden errors are typically caused by missing IAM permissions or mismatched bucket policies.", "category": "troubleshooting", "year": 2023},
    {"id": "doc_07", "text": "Vector databases accelerate similarity lookups using Hierarchical Navigable Small World (HNSW) graph indexing.", "category": "ai", "year": 2024},
    {"id": "doc_08", "text": "Redis cache eviction algorithms include Least Recently Used (LRU) and Least Frequently Used (LFU).", "category": "database", "year": 2023}
]

col_in, col_strat = st.columns([1, 1])

with col_in:
    st.subheader("1. Query & Corpus")
    sample_queries = [
        "Error 504 proxy_read_timeout Nginx",
        "How do transformers compute attention?",
        "PostgreSQL password authentication error 28P01",
        "What is the capital of France?" # Intentional out-of-domain query
    ]
    query_choice = st.selectbox("Select Preset Query or Enter Custom:", sample_queries)
    custom_q = st.text_input("Active Query:", value=query_choice)

    st.subheader("2. Select Retrieval Strategy")
    strat_type = st.radio(
        "Strategy Mode:",
        ["1. Pure Top-K Vector Search", "2. Similarity Threshold Filter", "3. Hybrid Search (Dense + Lexical)", "4. Multi-Query Reformulation"],
        index=2
    )

    k_val = st.slider("Max Candidates ($K$):", min_value=1, max_value=6, value=3)

    if strat_type == "2. Similarity Threshold Filter":
        threshold_val = st.slider("Cosine Similarity Threshold Cutoff:", min_value=0.0, max_value=0.8, value=0.35, step=0.05)
    elif strat_type == "3. Hybrid Search (Dense + Lexical)":
        sparse_weight = st.slider("Sparse Lexical Weight (BM25 Emphasis):", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

with col_strat:
    st.subheader(f"3. Retrieval Results: {strat_type}")
    
    q_vec = arena.embed(custom_q)
    q_tokens = set(custom_q.lower().split())

    scored_records = []
    for item in corpus_items:
        # Dense similarity
        c_vec = arena.embed(item["text"])
        dense_sim = float(np.dot(c_vec, q_vec))
        
        # Sparse lexical score
        c_tokens = set(item["text"].lower().split())
        token_overlap = len(c_tokens.intersection(q_tokens))
        sparse_score = float(token_overlap) / max(1, len(q_tokens))
        
        scored_records.append({
            "id": item["id"],
            "text": item["text"],
            "category": item["category"],
            "dense_sim": dense_sim,
            "sparse_score": sparse_score
        })

    # Strategy 1: Top-K Vector
    if "1. Pure Top-K" in strat_type:
        scored_records.sort(key=lambda x: x["dense_sim"], reverse=True)
        final_list = scored_records[:k_val]
        st.caption("Standard nearest neighbors ranked strictly by vector dot product.")
        for r in final_list:
            with st.expander(f"{r['id']} | Dense Cosine: {r['dense_sim']:.4f}", expanded=True):
                st.markdown(f"> {r['text']}")

    # Strategy 2: Threshold Filter
    elif "2. Similarity Threshold" in strat_type:
        scored_records.sort(key=lambda x: x["dense_sim"], reverse=True)
        filtered_list = [r for r in scored_records if r["dense_sim"] >= threshold_val][:k_val]
        st.caption(f"Filtered to retain candidates with Cosine Similarity >= {threshold_val:.2f}.")
        if len(filtered_list) == 0:
            st.error("Zero candidates met the similarity threshold! Out-of-domain noise was successfully rejected.")
        else:
            for r in filtered_list:
                with st.expander(f"{r['id']} | Dense Cosine: {r['dense_sim']:.4f}", expanded=True):
                    st.markdown(f"> {r['text']}")

    # Strategy 3: Hybrid Search
    elif "3. Hybrid Search" in strat_type:
        # Calculate RRF or weighted combination
        for r in scored_records:
            r["hybrid_score"] = (1.0 - sparse_weight) * r["dense_sim"] + sparse_weight * r["sparse_score"]
        scored_records.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_list = scored_records[:k_val]
        st.caption(f"Fusing Dense Similarity and Lexical Token Matches (Sparse Weight: {sparse_weight:.1f}).")
        for r in final_list:
            with st.expander(f"{r['id']} | Hybrid Score: {r['hybrid_score']:.4f} (Dense: {r['dense_sim']:.2f}, Lexical: {r['sparse_score']:.2f})", expanded=True):
                st.markdown(f"> {r['text']}")

    # Strategy 4: Multi-Query
    else:
        st.caption("Simulating LLM generation of 3 distinct query variations:")
        variations = [
            custom_q,
            f"Explain technical root cause of {custom_q}",
            f"Step by step fix for {custom_q}"
        ]
        st.write(f"1. `{variations[0]}`\n2. `{variations[1]}`\n3. `{variations[2]}`")
        
        # Merge candidate pools across variations
        merged_ids = set()
        multi_results = []
        for v in variations:
            v_vec = arena.embed(v)
            for item in corpus_items:
                c_vec = arena.embed(item["text"])
                sim = float(np.dot(c_vec, v_vec))
                if sim > 0.15 and item["id"] not in merged_ids:
                    merged_ids.add(item["id"])
                    multi_results.append((item, sim))
        
        multi_results.sort(key=lambda x: x[1], reverse=True)
        st.write(f"Retrieved **{len(multi_results)}** unique candidate documents across all 3 query variants:")
        for doc, score in multi_results[:k_val]:
            with st.expander(f"{doc['id']} | Score: {score:.4f}", expanded=True):
                st.markdown(f"> {doc['text']}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Strategy Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does Reciprocal Rank Fusion (RRF) combine disparate score scales from Dense and Sparse retrievers?**
    * **Answer**: Dense models produce cosine scores between -1.0 and 1.0, while BM25 produces unbounded positive floats (e.g. 0.0 to 35.0). Direct linear score addition is impossible without manual normalization.
    Instead, **Reciprocal Rank Fusion (RRF)** works entirely on **rank positions**:
    `RRF Score = sum(1 / (k + rank_i))` where $k$ is a constant (typically 60).
    Because RRF relies strictly on order rankings rather than raw numerical scores, it provides stable, scale-invariant fusion across completely different retrieval algorithms.

    **Q2: What is Hypothetical Document Embeddings (HyDE) and when should it be used?**
    * **Answer**: In HyDE, an LLM is prompted to generate a *hypothetical answer* to the user's query (e.g. *"Write a hypothetical documentation paragraph answering: How do I rotate API keys in AWS?"*). Even if the hypothetical text contains factual errors, its vocabulary and stylistic embedding are closer to candidate document passages in latent space than the short raw question was. It is ideal for asymmetric queries where the user query is very short or conceptually abstract.

    **Q3: When is pure semantic vector search strictly inferior to BM25 keyword search?**
    * **Answer**: Pure dense vector search degrades when queries contain exact alphanumeric identifiers, rare error codes (e.g. `0xC0000005`, `CVE-2024-3094`), part numbers, product SKUs, or exact variable names. Dense tokenizers often split rare identifiers into fragmented subwords, diluting semantic coordinates, whereas inverted indices (BM25) match exact strings with 100% precision.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Applying Naive Score Thresholding Across Different Embedding Models**:
      * *Issue*: Hardcoding a threshold of 0.75 across different embedding models.
      * *Consequence*: Model A may produce cosine scores clustered in [0.80, 0.95], while Model B produces scores in [0.40, 0.70]. A hardcoded 0.75 cutoff returns all documents in Model A and zero documents in Model B.
      * *Fix*: Calibrate threshold values empirically on a validation benchmark for each specific embedding model.

    * **Trap 2: High Latency Cascades in Multi-Query Retrieval**:
      * *Issue*: Calling an LLM synchronously to generate 5 query reformulations before triggering retrieval.
      * *Consequence*: Adds 400-800ms of LLM generation latency before retrieval even begins.
      * *Fix*: Use fast speculative models (e.g., small 7B models or local fast inference) or generate variations asynchronously.

    * **Trap 3: Linear Score Blending without Normalization**:
      * *Issue*: Calculating `Score = Cosine_Score + BM25_Score`.
      * *Consequence*: BM25 scores (range 0 to 20+) completely drown out cosine scores (range 0 to 1), reducing hybrid search to pure keyword search.
      * *Fix*: Always use rank-based fusion (RRF) or normalize scores using MinMax / Sigmoid scaling.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Reciprocal Rank Fusion (RRF)**
      * *Definition*: A robust rank-aggregation algorithm that combines candidate lists from multiple retrievers using `1 / (60 + rank)` without requiring score normalization.
    * **Flashcard 2: HyDE (Hypothetical Document Embeddings)**
      * *Definition*: A query transformation technique that uses an LLM to generate a synthetic answer document, which is then embedded to retrieve real documents.
    * **Flashcard 3: Dynamic Similarity Threshold**
      * *Definition*: A cutoff rule that discards retrieved chunks whose similarity score falls below a confidence threshold to prevent hallucination on out-of-domain queries.
    * **Flashcard 4: Multi-Query Retrieval**
      * *Definition*: Generating multiple alternative phrasings of a user question to query the vector index from diverse semantic angles, followed by deduplication.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Advanced Retrieval Strategies:

- [ ] I can articulate the limitations of naive Top-K retrieval in production.
- [ ] I know how dynamic similarity thresholding prevents out-of-domain hallucinations.
- [ ] I understand how Reciprocal Rank Fusion (RRF) combines dense and sparse search rankings.
- [ ] I can explain the mechanics and benefits of HyDE and Multi-Query expansion.
- [ ] I know when BM25 outperforms dense vector embeddings.
""")
