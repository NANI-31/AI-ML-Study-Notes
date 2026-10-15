import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="34.2 RAG Performance - LangChain Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .rag-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-rank {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class RAGPerformanceEngine:
    """
    Simulates high-throughput RAG search, Reciprocal Rank Fusion (RRF),
    Cross-Encoder reranking, and HNSW index tuning.
    """
    def __init__(self):
        self.raw_documents = [
            {"id": "doc_1", "text": "LangChain v0.3 introduces streamlined partner packages and universal Runnable interfaces.", "dense_score": 0.88, "bm25_score": 0.40, "rerank_score": 0.95},
            {"id": "doc_2", "text": "Vector database indexes utilize HNSW graphs with efSearch and M parameters to control precision.", "dense_score": 0.84, "bm25_score": 0.92, "rerank_score": 0.91},
            {"id": "doc_3", "text": "Embedding models map unstructured tokens into high-dimensional geometric coordinate manifolds.", "dense_score": 0.79, "bm25_score": 0.35, "rerank_score": 0.65},
            {"id": "doc_4", "text": "Cross-encoder rerankers jointly compute token cross-attention between queries and documents.", "dense_score": 0.72, "bm25_score": 0.75, "rerank_score": 0.88}
        ]

    def compute_rrf_rank(self, k: int = 60) -> List[Dict[str, Any]]:
        # Sort by dense
        dense_sorted = sorted(self.raw_documents, key=lambda x: x["dense_score"], reverse=True)
        # Sort by bm25
        bm25_sorted = sorted(self.raw_documents, key=lambda x: x["bm25_score"], reverse=True)

        rrf_scores = {}
        for rank, doc in enumerate(dense_sorted, 1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (k + rank))

        for rank, doc in enumerate(bm25_sorted, 1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (k + rank))

        results = []
        for doc in self.raw_documents:
            results.append({
                "id": doc["id"],
                "text": doc["text"][:50] + "...",
                "dense_score": doc["dense_score"],
                "bm25_score": doc["bm25_score"],
                "rrf_score": round(rrf_scores[doc["id"]], 5),
                "rerank_score": doc["rerank_score"]
            })
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results

    def estimate_hnsw_perf(self, m: int, ef_search: int) -> Dict[str, Any]:
        # Theoretical model of HNSW trade-offs
        recall = min(round(0.85 + (ef_search / 200.0) * 0.14, 3), 0.999)
        qps = max(int(2500 - (ef_search * 8.5) - (m * 12)), 250)
        build_time_multiplier = round(1.0 + (m / 16.0) * 0.5, 2)
        return {
            "m": m,
            "ef_search": ef_search,
            "recall_at_10": recall,
            "estimated_qps": qps,
            "index_build_mult": build_time_multiplier
        }

# Instantiate engine
rag_engine = RAGPerformanceEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced RAG Ingestion, HNSW Indexing, & Cross-Encoder Reranking", divider="rainbow")

st.markdown("""
In enterprise retrieval-augmented generation (RAG), simple dense vector search is insufficient. 
High-performance RAG combines **hybrid retrieval** (dense vectors + sparse BM25 keyword search), 
**Reciprocal Rank Fusion (RRF)**, **metadata pre-filtering**, and a secondary **Cross-Encoder reranking pass** 
to achieve 98%+ retrieval precision while maintaining millisecond search latencies.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAG_Performance_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingestion {
            label = "Optimized Chunking & Vectorization Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            source_docs [label="Raw Documents\\n(PDFs, Markdown, Web pages)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            semantic_splitter [label="Semantic Chunking Engine\\n(500-token chunks, 50-token overlap, metadata headers)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            dense_embedder [label="Dense Embedding Model (BGE / OpenAI)\\n(Matryoshka 512-dim truncation)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            sparse_indexer [label="Sparse Keyword Indexer\\n(BM25 vocabulary inverted index)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_vectordb {
            label = "Vector Database Indexing (Qdrant / Pinecone / Chroma)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            hnsw_index [label="HNSW Graph Index (M=32, efConstruction=128)\\n(Sub-millisecond approximate nearest neighbor search)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            metadata_filter [label="Metadata Pre-Filter\\n(Filter: tenant_id == 'prod_01' AND year >= 2024)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_search {
            label = "Two-Stage Retrieval & Reranking Pipeline";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            query_in [label="Incoming User Query", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            hybrid_retriever [label="Hybrid Search & Reciprocal Rank Fusion (RRF)\\n(Fetches top 50 dense + top 50 sparse candidates)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            cross_encoder [label="Cross-Encoder Reranker (Cohere / BGE-Reranker)\\n(Deep cross-attention rescores top 50 -> Top 5)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            llm_synthesizer [label="LLM Synthesis Context Window\\n(Passes only pristine top-5 relevant chunks)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        source_docs -> semantic_splitter;
        semantic_splitter -> dense_embedder;
        semantic_splitter -> sparse_indexer;

        dense_embedder -> hnsw_index;
        sparse_indexer -> hnsw_index;

        query_in -> metadata_filter;
        metadata_filter -> hybrid_retriever;
        hnsw_index -> hybrid_retriever;

        hybrid_retriever -> cross_encoder [label="Top 50 candidates"];
        cross_encoder -> llm_synthesizer [label="Top 5 curated chunks"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>The Two-Stage Retrieval Paradigm:</strong> Bi-encoders (vector embeddings) are ultra-fast for candidate generation 
    (filtering 1,000,000 documents to 50 in 10ms). Cross-encoders are computationally intensive but execute full cross-attention 
    between query and document, ensuring the final 5 chunks fed to the LLM have maximum semantic relevance.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: CHUNK OPTIMIZATION
# ==============================================================================
st.header("2. Chunk Optimization: Chunk Size vs Overlap Tuning and Parent-Child Splitting", divider="rainbow")

st.markdown("""
Chunking strategy is the single largest determinant of RAG performance. 
Arbitrary character-based chunking without regard for document structure breaks semantic coherence.
""")

st.markdown("""
### The Chunking Trade-Off Continuum
- **Small Chunks (100–250 tokens)**: High vector retrieval precision because the embedding vector represents a focused concept. However, when passed to the LLM, individual small chunks lack surrounding explanatory context.
- **Large Chunks (800–1,500 tokens)**: Abundant context for the generator, but the dense embedding vector becomes diffuse and loses specificity, causing the retriever to miss precise factual matches.
- **The Modern Standard: Parent-Child Splitting (`ParentDocumentRetriever`)**: Store small chunks (150 tokens) in the vector store for index retrieval. When a small chunk matches, retrieve its enclosing **parent chunk** (800 tokens) or full section from docstore to feed into the LLM context window.
""")

# ==============================================================================
# SECTION 3: EMBEDDING OPTIMIZATION
# ==============================================================================
st.header("3. Embedding Optimization: Matryoshka Dimensionality Reduction and Batch Vectorization", divider="rainbow")

st.markdown("""
Generating embeddings in production presents two bottlenecks: **RAM/storage explosion** in vector databases 
and **network latency** during ingestion.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Matryoshka Embeddings (MRL)
    Modern embedding models (such as `text-embedding-3-large` or `nomic-embed-text`) are trained such that the first D dimensions capture the vast majority of semantic variance.
    - **Original**: 3,072 dimensions (12 KB per vector).
    - **Truncated (MRL)**: 512 dimensions (2 KB per vector).
    - **Impact**: 83% reduction in RAM and search latency with less than 1.5% loss in Mean Reciprocal Rank (MRR).
    """)

with col2:
    st.markdown("""
    ### Batch Vectorization
    Never vectorize documents one-by-one in a synchronous loop.
    - Always pass batches of 64 to 256 chunks to `embed_documents(batch)`.
    - Batching minimizes HTTP handshake overhead and saturates GPU tensor cores during local embedding model inference.
    """)

# ==============================================================================
# SECTION 4: RETRIEVAL OPTIMIZATION
# ==============================================================================
st.header("4. Retrieval Optimization: Hybrid Dense + Sparse BM25 Search and Reciprocal Rank Fusion", divider="rainbow")

st.markdown("""
Dense vector search excels at conceptual and semantic matching ('automobile malfunctions' matches 'car breakdown'). 
However, dense search frequently fails on exact keyword searches, acronyms, product SKUs, and function names ('error code 0x80070005').
""")

st.markdown("""
### The Solution: Hybrid Search with Reciprocal Rank Fusion (RRF)
Hybrid search executes both dense vector search and sparse BM25 search in parallel, 
then combines candidate lists using Reciprocal Rank Fusion:
""")

st.markdown("""
RRF Score Formula:  
`RRF_score(doc) = sum(1 / (k + rank_i(doc)))` where `k` is a constant (typically 60) that dampens the influence of top ranks.
""")

st.code('''from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain v0.3 introduces clean partner packages."),
    Document(page_content="Error code 0x80070005 indicates an access denied permission fault.")
]

# 1. Sparse BM25 Keyword Retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

# 2. Dense Vector Retriever (assumes vectorstore initialized)
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Hybrid Ensemble with Reciprocal Rank Fusion (RRF)
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5]
# )
''', language="python")

# ==============================================================================
# SECTION 5: RERANKING
# ==============================================================================
st.header("5. Reranking: Cross-Encoder Rerankers vs Bi-Encoder Retrieval", divider="rainbow")

st.markdown("""
In the two-stage retrieval pattern, the first stage retrieves a broad set of candidates (e.g., top 50) 
at high speed. The second stage applies a **Cross-Encoder Reranker** (such as Cohere Rerank or BGE-Reranker-Large) 
to re-score and re-order the candidates.
""")

rerank_comp_df = pd.DataFrame([
    {"Architecture": "Bi-Encoder (Embedding Models)", "How It Works": "Embeds query and document independently into vectors; computes dot product.", "Speed": "Ultra-fast (< 5ms over millions of vectors)", "Semantic Nuance": "Moderate (loses word-level cross-interaction)"},
    {"Architecture": "Cross-Encoder (Rerankers)", "How It Works": "Passes query and document concatenated into full multi-layer transformer cross-attention.", "Speed": "Slower (~40ms over 50 documents)", "Semantic Nuance": "State-of-the-art (evaluates exact relationship between query and context)"}
])
st.dataframe(rerank_comp_df, use_container_width=True)

# ==============================================================================
# SECTION 6: METADATA FILTERING
# ==============================================================================
st.header("6. Metadata Filtering: Pre-Filtering vs Post-Filtering in Vector Stores", divider="rainbow")

st.markdown("""
In multi-tenant enterprise systems, users should only retrieve documents from their own organization or permission group. 
How metadata filters are executed dramatically impacts both speed and recall.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("""
    ### Post-Filtering (Antipattern)
    1. Search HNSW index to find top 50 nearest vectors globally.
    2. Filter out documents where `tenant_id != current_tenant`.
    - **Failure Mode**: If only 5% of documents belong to `current_tenant`, the top 50 global results may contain 0 results matching the filter!
    """)

with col_m2:
    st.markdown("""
    ### Pre-Filtering (Production Standard)
    1. Modern engines (Qdrant, Pinecone) traverse the HNSW graph while dynamically restricting candidate evaluation to vectors satisfying the payload filter.
    - **Guaranteed Recall**: 100% of returned top k documents satisfy the required security filter.
    """)

# ==============================================================================
# SECTION 7: VECTOR INDEXING
# ==============================================================================
st.header("7. Vector Indexing: HNSW Graph Parameters (M, efConstruction, efSearch) and Quantization", divider="rainbow")

st.markdown("""
The **Hierarchical Navigable Small World (HNSW)** graph is the gold standard for vector indexing. 
Tuning its hyperparameters dictates the exact trade-off between search throughput (Queries Per Second) and Recall@K.
""")

hnsw_params_df = pd.DataFrame([
    {"Parameter": "M (Connections per Node)", "Typical Range": "16 to 64", "Role": "Defines the number of bidirectional links per graph node. Higher M improves recall for high-dimensional vectors at the expense of memory consumption."},
    {"Parameter": "efConstruction (Build Search Depth)", "Typical Range": "64 to 256", "Role": "Controls graph exploration depth during index creation. Higher values produce higher-quality graphs but increase indexing time."},
    {"Parameter": "efSearch (Query Search Depth)", "Typical Range": "32 to 128", "Role": "Dynamic query-time parameter. Increasing efSearch improves Recall@10 at the expense of search latency."}
])
st.dataframe(hnsw_params_df, use_container_width=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full High-Performance RAG Pipeline with Hybrid Search and Reranking", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating a complete 
**high-performance two-stage retrieval pipeline with Reciprocal Rank Fusion**.
""")

st.code('''from typing import List, Dict, Any

class ProductionHybridRerankPipeline:
    def __init__(self):
        # Sample document collection with metadata
        self.documents = [
            {"id": "doc_1", "text": "LangChain v0.3 separates langchain-core from partner packages.", "tenant": "org_a"},
            {"id": "doc_2", "text": "Error 0x80070005: Permission denied during vector index write.", "tenant": "org_a"},
            {"id": "doc_3", "text": "HNSW parameters M and efSearch dictate vector retrieval latency.", "tenant": "org_b"},
            {"id": "doc_4", "text": "Cross-encoders compute deep cross-attention for reranking.", "tenant": "org_a"}
        ]

    def dense_search(self, query: str, tenant_id: str) -> List[Dict[str, Any]]:
        # Simulates dense vector similarity with metadata pre-filtering
        filtered = [d for d in self.documents if d["tenant"] == tenant_id]
        return filtered[:3]

    def sparse_search(self, query: str, tenant_id: str) -> List[Dict[str, Any]]:
        # Simulates BM25 keyword matching with metadata pre-filtering
        q_words = set(query.lower().split())
        filtered = [d for d in self.documents if d["tenant"] == tenant_id]
        # Sort by keyword match
        filtered.sort(key=lambda d: len(q_words.intersection(set(d["text"].lower().split()))), reverse=True)
        return filtered[:3]

    def reciprocal_rank_fusion(self, dense_results: List[dict], sparse_results: List[dict], k: int = 60) -> List[dict]:
        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_results, 1):
            doc_map[doc["id"]] = doc
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (k + rank))

        for rank, doc in enumerate(sparse_results, 1):
            doc_map[doc["id"]] = doc
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (k + rank))

        ranked_docs = []
        for doc_id, score in rrf_scores.items():
            item = doc_map[doc_id].copy()
            item["rrf_score"] = round(score, 5)
            ranked_docs.append(item)

        ranked_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
        return ranked_docs

    def execute_rag_pipeline(self, query: str, tenant_id: str) -> List[dict]:
        # Step 1: Parallel Hybrid Retrieval with Metadata Pre-Filtering
        dense_candidates = self.dense_search(query, tenant_id)
        sparse_candidates = self.sparse_search(query, tenant_id)

        # Step 2: Reciprocal Rank Fusion
        fused_candidates = self.reciprocal_rank_fusion(dense_candidates, sparse_candidates)
        return fused_candidates

pipeline = ProductionHybridRerankPipeline()
results = pipeline.execute_rag_pipeline("Error 0x80070005 permissions", tenant_id="org_a")
print("Top Fused Candidates:", results)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Chunk Size vs Recall Sandbox & HNSW Parameter Simulator", divider="rainbow")

st.markdown("""
Experiment with Reciprocal Rank Fusion re-scoring, explore HNSW graph parameter trade-offs, 
and observe how two-stage reranking elevates relevant documents to the top ranks.
""")

wb_view = st.selectbox(
    "Select RAG Optimization Tool:",
    ["1. Hybrid Search & Reciprocal Rank Fusion (RRF) Inspector", "2. HNSW Index Hyperparameter Simulator", "3. Two-Stage Reranking Rank Re-Orderer"]
)

if wb_view == "1. Hybrid Search & Reciprocal Rank Fusion (RRF) Inspector":
    st.subheader("Reciprocal Rank Fusion (RRF) Scoring Matrix")
    k_constant = st.slider("RRF Dampening Constant (k):", 10, 100, 60, 5)

    rrf_data = rag_engine.compute_rrf_rank(k_constant)
    st.dataframe(pd.DataFrame(rrf_data), use_container_width=True)
    st.info("Notice how RRF reconciles differences: documents that score well across BOTH dense and sparse retrieval achieve the highest fused rank.")

elif wb_view == "2. HNSW Index Hyperparameter Simulator":
    st.subheader("HNSW Graph Performance Simulator")
    c1, c2 = st.columns(2)
    with c1:
        param_m = st.slider("M (Bidirectional Connections):", 8, 64, 32, 8)
    with c2:
        param_ef = st.slider("efSearch (Search Depth):", 16, 256, 64, 16)

    hnsw_stats = rag_engine.estimate_hnsw_perf(param_m, param_ef)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Estimated Recall@10", f"{hnsw_stats['recall_at_10'] * 100:.1f}%")
    with m2:
        st.metric("Search Throughput (QPS)", f"{hnsw_stats['estimated_qps']} QPS")
    with m3:
        st.metric("Index Build Memory Multiplier", f"{hnsw_stats['index_build_mult']}x")

else:
    st.subheader("Two-Stage Reranking Candidate Elevator")
    st.markdown("Observe how the Cross-Encoder elevates 'doc_1' to Rank 1 based on deep cross-attention, despite lower initial keyword match:")

    comp_df = pd.DataFrame(rag_engine.raw_documents)
    st.dataframe(comp_df, use_container_width=True)
    st.success("Cross-encoder rerank score (0.95) correctly identifies doc_1 as the most relevant answer to technical LangChain inquiries.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the fundamental difference between Bi-Encoders and Cross-Encoders in modern RAG?"):
    st.markdown("""
    **Answer:**  
    - **Bi-encoders** embed queries and documents separately into vectors. Similarity is computed via dot product, making search ultra-fast over millions of documents via vector indexes.
    - **Cross-encoders** concatenate the query and document into a single transformer input, computing full all-to-all cross-attention across all token pairs. They are computationally expensive but vastly more accurate, making them ideal as a second-stage reranker over 30 to 50 candidates.
    """)

with st.expander("Q2: How does Reciprocal Rank Fusion (RRF) combine ranking lists without normalizing raw distance scores?"):
    st.markdown("""
    **Answer:**  
    Different retrieval algorithms produce non-comparable scores (e.g., BM25 scores can range from 0 to 40+, while cosine similarities range from 0 to 1). RRF bypasses score calibration by using only the ordinal ranks: `score = sum(1 / (k + rank))`. This makes fusion robust against differing score distributions.
    """)

with st.expander("Q3: Why is metadata pre-filtering strongly preferred over post-filtering in production vector stores?"):
    st.markdown("""
    **Answer:**  
    Post-filtering executes global approximate nearest neighbor search first, then discards non-matching records. If the metadata predicate matches only a small fraction of the database (e.g., a specific tenant's files), post-filtering often returns empty or truncated result sets. Pre-filtering restricts graph traversal strictly to matching nodes.
    """)

with st.expander("Q4: How does Matryoshka Representation Learning (MRL) optimize vector storage without re-training?"):
    st.markdown("""
    **Answer:**  
    MRL-trained models structure embedding vectors such that the leading dimensions contain the most informative representations. An application can slice off the trailing dimensions (e.g., truncating from 3,072 to 512 dimensions) to save 83% of storage and RAM with minimal loss in retrieval accuracy.
    """)

with st.expander("Q5: What is the ParentDocumentRetriever pattern, and what problem does it solve?"):
    st.markdown("""
    **Answer:**  
    It resolves the conflict between chunk size for retrieval vs chunk size for synthesis. Small chunks (150 tokens) are indexed in the vector store for precise semantic matching, but when retrieved, the retriever returns the larger parent document (1,000 tokens) from docstore, ensuring the LLM receives complete context.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Relying Exclusively on Dense Vector Search for Part Numbers and SKUs**  
Dense embeddings map alphanumeric identifiers (e.g., 'ERR-404-X') into diffuse semantic vectors, resulting in poor retrieval. 
Always implement hybrid search with BM25 for technical and catalog data.
""")

st.error("""
**Trap 2: Passing All Retrieved Candidates (Top 50) Directly to the LLM**  
Dumping dozens of chunks into prompt context induces 'lost in the middle' syndrome, skyrockets token cost, 
and degrades generation precision. Rerank down to the top 3 to 5 chunks.
""")

st.warning("""
**Trap 3: Using Excessive efSearch in HNSW During High-Traffic Spikes**  
Setting `efSearch=512` delivers marginal recall improvements (< 1%) while cutting QPS by 70%, 
causing request queue backups during traffic surges.
""")

st.warning("""
**Trap 4: Forgetting to Re-index When Upgrading Embedding Models**  
Updating an embedding model without rebuilding the vector database index corrupts similarity search, 
as vectors from different models occupy incompatible coordinate spaces.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Reciprocal Rank Fusion (RRF)</h4>
        <p>An algorithm combining disparate ranking lists into a unified score using reciprocal rank positions rather than calibrated scores.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Cross-Encoder Reranker</h4>
        <p>A transformer model computing full cross-attention between a query and document candidate to produce precise relevance scores.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>ParentDocumentRetriever</h4>
        <p>A LangChain retrieval architecture indexing fine-grained child chunks while retrieving comprehensive parent chunks for generation.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>HNSW Graph</h4>
        <p>Hierarchical Navigable Small World index enabling sub-millisecond approximate nearest neighbor vector search.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to tune chunk size and overlap using the ParentDocumentRetriever pattern.", value=True)
chk_2 = st.checkbox("I understand Matryoshka Representation Learning (MRL) and embedding dimensionality reduction.", value=True)
chk_3 = st.checkbox("I can implement hybrid retrieval combining dense vector search and sparse BM25 with RRF.", value=True)
chk_4 = st.checkbox("I understand why two-stage Cross-Encoder reranking dramatically improves retrieval precision.", value=True)
chk_5 = st.checkbox("I know the difference between metadata pre-filtering and post-filtering in vector databases.", value=True)
chk_6 = st.checkbox("I understand how HNSW parameters (M, efConstruction, efSearch) control the QPS vs Recall trade-off.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("RAG Performance optimization competencies verified! Ready for Agent Performance.")
