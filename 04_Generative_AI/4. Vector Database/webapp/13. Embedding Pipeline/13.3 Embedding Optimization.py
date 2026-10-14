import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import hashlib
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="13.3 Embedding Optimization - Vector DB Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("13.3 Embedding Optimization: Batching, Caching, Normalization & Cost Reduction")
st.caption("Chapter 13: Embedding Pipeline | Domain: 4. Vector Database")
st.markdown("""
In enterprise AI deployments, generating embeddings across tens of millions of documents can quickly become a 
**massive financial and computational bottleneck**.

Consider ingesting 10,000,000 document chunks of 500 tokens each (5 billion tokens total):
* If processed naively (calling embedding APIs one chunk at a time with full dimensions and no caching), 
  the process takes **weeks of runtime, thousands of dollars in API bills**, and exhausts server RAM.
* By applying **Batching, Persistent Content Caching, L2 Unit Normalization, and Matryoshka Dimension Slicing**, 
  you can **cut financial costs by 70% to 90% and reduce ingestion runtime from days to hours**.

In this lesson, we explore the five pillars of embedding optimization, examine why vector normalization enables 
ultra-fast SIMD dot products, build an enterprise Cached Batch Embedder in Python, and test an interactive Cost & Optimization Calculator.
""")

st.divider()

# Section 1: The Five Pillars of Embedding Optimization
st.header("1. The Five Pillars of Embedding Optimization", divider="rainbow")

col_pil1, col_pil2 = col_pil1, col_pil2 = st.columns([1.1, 0.9])

with col_pil1:
    st.markdown("""
    ### The 5 Optimization Pillars
    
    1. **Batch Vectorization (Bulk Inference)**:
       * Amortizes network round-trips, TLS handshakes, and GPU kernel launch overheads by transmitting 100 to 1,000 chunks per request.
       * Accelerates throughput by **15x to 50x** compared to serial single-string calls.
    
    2. **Content Hash Caching (Never Pay Twice)**:
       * Compute a SHA-256 hash of each cleaned text chunk: `key = sha256(text)`.
       * Store previously computed embeddings in Redis or SQLite. When re-ingesting documents, unchanged paragraphs return instantly from cache at **$0 cost**.
    
    3. **L2 Unit Normalization**:
       * Scale all vectors to unit length (`||v|| = 1.0`).
       * Transforms expensive Euclidean distance and Cosine math into a **single hardware dot product**, drastically speeding up HNSW graph traversal.
    
    4. **Matryoshka Dimension Slicing (MRL)**:
       * Truncate embedding dimensions (e.g. from 1,536 to 512 dimensions) on supported models (OpenAI `text-embedding-3`, BGE-M3), slashing RAM and compute by 66% with <2% recall loss.
    
    5. **Tiered Model Selection**:
       * Use lightweight, open-weight local models (e.g. `all-MiniLM-L6-v2` or `bge-small-en-v1.5` on ONNX/CPU) for high-volume internal logs, reserving paid API models for customer-facing RAG.
    """)

with col_pil2:
    st.markdown("### Optimization Architectural Blueprint")
    opt_dot = """
    digraph OptimizationFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ChunksIn [label="100,000 Chunks to Ingest", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        CacheCheck [label="1. SHA-256 Cache Filter\\n(Redis / SQLite)", fillcolor="#FEF3C7", color="#D97706"];

        CachedHits [label="Cached Chunks (60%)\\nInstant Recall ($0 cost)", fillcolor="#D1FAE5", color="#059669"];
        UncachedMisses [label="Uncached Chunks (40%)\\nRequires Inference", fillcolor="#FEE2E2", color="#DC2626"];

        BatchEngine [label="2. Dynamic Batcher\\n(Batches of 250 chunks)", fillcolor="#FEF3C7"];
        MRLTruncate [label="3. Matryoshka Slicing\\n(1536 -> 512 dims)", fillcolor="#E0E7FF"];
        L2Norm [label="4. L2 Normalization\\nv = v / ||v||", fillcolor="#C7D2FE"];

        Store [label="ChromaDB Collection\\n66% Less RAM | Sub-2ms Search", fillcolor="#A7F3D0", style="filled,bold"];

        ChunksIn -> CacheCheck;
        CacheCheck -> CachedHits [label="60k hits"];
        CacheCheck -> UncachedMisses [label="40k misses"];
        UncachedMisses -> BatchEngine;
        BatchEngine -> MRLTruncate;
        MRLTruncate -> L2Norm;
        CachedHits -> Store;
        L2Norm -> Store;
    }
    """
    st.graphviz_chart(opt_dot, use_container_width=False)

st.divider()

# Section 2: The Mathematics of Vector Normalization
st.header("2. The Mathematics of Vector Normalization", divider="rainbow")

col_norm1, col_norm2 = col_norm1, col_norm2 = st.columns([1, 1])

with col_norm1:
    st.markdown("""
    ### Why Unit Normalization Accelerates Vector Search
    The standard formula for Cosine Similarity between vector `A` and vector `B` requires computing:
    
    ```text
    Cosine_Similarity(A, B) = Dot_Product(A, B) / ( Norm(A) * Norm(B) )
    ```
    
    This requires:
    1. Computing `Dot_Product(A, B) = sum(A_i * B_i)`
    2. Computing `Norm(A) = sqrt(sum(A_i^2))`
    3. Computing `Norm(B) = sqrt(sum(B_i^2))`
    4. Two square roots and a division per neighbor during HNSW traversal.
    
    ### The Normalization Shortcut
    If every vector is **L2-normalized at ingestion time** such that `Norm(v) = 1.0`:
    
    ```text
    Norm(A) = 1.0  and  Norm(B) = 1.0
    Therefore: Cosine_Similarity(A, B) = Dot_Product(A, B)
    ```
    
    * **Impact**: The database discards square roots and divisions entirely! Distance evaluation collapses to a single hardware-accelerated **Inner Product (dot product)** instruction (`_mm256_fmadd_ps`), accelerating HNSW graph search by **3x to 5x**.
    """)

with col_norm2:
    st.markdown("""
    ### Python Implementation of L2 Normalization
    ```python
    import numpy as np

    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        \"\"\"Scales vector to unit Euclidean length (L2 norm = 1.0).\"\"\"
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    # Example:
    v = np.array([3.0, 4.0], dtype=np.float32)
    # Norm is sqrt(3^2 + 4^2) = 5.0
    v_norm = normalize_l2(v)  # [0.6, 0.8]
    # New norm: sqrt(0.6^2 + 0.8^2) = 1.0
    ```
    
    * **ChromaDB Integration**: When a collection is initialized with `metadata={"hnsw:space": "cosine"}`, Chroma automatically applies unit normalization before indexing.
    """)

st.divider()

# Section 3: High-Throughput Cached Batch Embedder Pattern
st.header("3. High-Throughput Cached Batch Embedder Pattern", divider="rainbow")

st.markdown("""
Below is the production Python architecture for an **Enterprise Cached Embedder** that combines SQLite persistent caching, 
bulk batching, and Matryoshka dimension truncation:
""")

st.code("""import hashlib
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseCachedEmbedder:
    \"\"\"High-performance embedder with content hash caching and dimension slicing.\"\"\"
    
    def __init__(self, db_path: str = "./embedding_cache.sqlite", target_dim: int = 512):
        self.target_dim = target_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute(\"\"\"
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    vector_blob BLOB,
                    dim INTEGER,
                    created_epoch INTEGER
                )
            \"\"\")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def embed_chunks(self, texts: List[str], client, batch_size: int = 100) -> List[List[float]]:
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        # 1. Check local persistent cache
        for idx, text in enumerate(texts):
            h = self._hash_text(text)
            cursor = self.conn.execute("SELECT vector_blob FROM embedding_cache WHERE content_hash = ?", (h,))
            row = cursor.fetchone()
            if row:
                # Cache Hit: Unpack float32 numpy blob
                vec = np.frombuffer(row[0], dtype=np.float32)
                results[idx] = vec.tolist()
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        print(f"Cache Hits: {len(texts) - len(missing_texts)} | Cache Misses: {len(missing_texts)}")

        # 2. Batch vectorization for cache misses
        for i in range(0, len(missing_texts), batch_size):
            b_texts = missing_texts[i : i + batch_size]
            b_indices = missing_indices[i : i + batch_size]

            # Call embedding model API
            response = client.embeddings.create(input=b_texts, model="text-embedding-3-small")
            
            with self.conn:
                for j, item in enumerate(response.data):
                    raw_vec = np.array(item.embedding, dtype=np.float32)
                    
                    # Apply Matryoshka Slicing if requested
                    if self.target_dim and self.target_dim < len(raw_vec):
                        raw_vec = raw_vec[:self.target_dim]
                        # Renormalize to unit length
                        raw_vec = raw_vec / np.linalg.norm(raw_vec)

                    orig_idx = b_indices[j]
                    results[orig_idx] = raw_vec.tolist()

                    # Save to persistent cache
                    h = self._hash_text(b_texts[j])
                    self.conn.execute(
                        "INSERT OR IGNORE INTO embedding_cache VALUES (?, ?, ?, strftime('%s', 'now'))",
                        (h, raw_vec.tobytes(), len(raw_vec))
                    )

        return results
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Cost & Ingestion ROI Calculator
st.header("4. Interactive Workbench: Cost & Ingestion ROI Calculator", divider="rainbow")

st.markdown("""
Model the real-world operational savings of applying **Batching, Persistent Caching, and Dimension Slicing** 
on your organization's vector dataset.
""")

col_calc_p1, col_calc_p2 = col_calc_p1, col_calc_p2 = st.columns([1, 1])

with col_calc_p1:
    st.subheader("1. Ingestion Volume & Pricing")
    
    total_docs = st.select_slider(
        "Total Document Chunks:",
        options=[50_000, 250_000, 1_000_000, 5_000_000, 20_000_000],
        value=1_000_000,
        format_func=lambda x: f"{x:,} chunks"
    )
    
    avg_tokens = st.slider("Average Tokens per Chunk:", min_value=100, max_value=1000, value=500, step=50)
    
    model_choice = st.selectbox(
        "Embedding Model Tier:",
        [
            "OpenAI text-embedding-3-small ($0.02 / 1M tokens)",
            "OpenAI text-embedding-3-large ($0.13 / 1M tokens)",
            "Cohere embed-english-v3.0 ($0.10 / 1M tokens)",
            "Self-Hosted Local BGE-large ($0.00 API + GPU Cloud VM)"
        ]
    )
    
    if "3-small" in model_choice:
        cost_per_m = 0.02
        default_dim = 1536
    elif "3-large" in model_choice:
        cost_per_m = 0.13
        default_dim = 3072
    elif "Cohere" in model_choice:
        cost_per_m = 0.10
        default_dim = 1024
    else:
        cost_per_m = 0.00
        default_dim = 1024

with col_calc_p2:
    st.subheader("2. Optimization Levers")
    
    cache_hit_pct = st.slider("Expected Cache Hit Rate (% unchanged content):", min_value=0, max_value=90, value=60, step=5)
    batch_size_choice = st.select_slider("Ingestion Batch Size:", options=[1, 10, 50, 100, 500, 1000], value=250)
    mrl_slice = st.selectbox("Matryoshka Dimension Slicing:", ["Full Dimensions (100% RAM)", "512 Dimensions (66% RAM Savings)", "256 Dimensions (83% RAM Savings)"])

# Financial and Performance Math
total_tokens = total_docs * avg_tokens
total_tokens_millions = total_tokens / 1_000_000

# Base cost (no cache)
base_api_cost = total_tokens_millions * cost_per_m
# Optimized cost (after cache hits)
tokens_to_embed = total_tokens * (1.0 - (cache_hit_pct / 100.0))
opt_api_cost = (tokens_to_embed / 1_000_000) * cost_per_m

# Runtime Estimation
# Single calls: ~120ms each; Batches: ~200ms per batch of N
if batch_size_choice == 1:
    req_count = total_docs * (1.0 - (cache_hit_pct / 100.0))
    time_sec = req_count * 0.12
else:
    batches = int(np.ceil((total_docs * (1.0 - (cache_hit_pct / 100.0))) / batch_size_choice))
    time_sec = batches * 0.25

hours_est = time_sec / 3600.0

# RAM footprint calculation
if "512" in mrl_slice:
    active_dim = 512
elif "256" in mrl_slice:
    active_dim = 256
else:
    active_dim = default_dim

base_ram_gb = (total_docs * default_dim * 4 * 1.3) / (1024**3)
opt_ram_gb = (total_docs * active_dim * 4 * 1.3) / (1024**3)

st.divider()

col_roi_metrics, col_roi_summary = st.columns([1.1, 0.9])

with col_roi_metrics:
    st.subheader("Optimization Financial & Resource Impact")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="API Ingestion Cost",
            value=f"${opt_api_cost:,.2f}",
            delta=f"-${base_api_cost - opt_api_cost:,.2f} Saved" if base_api_cost > 0 else "Self-Hosted"
        )
        st.metric(
            label="Vector RAM Allocation",
            value=f"{opt_ram_gb:.2f} GB",
            delta=f"-{((base_ram_gb - opt_ram_gb) / base_ram_gb) * 100:.1f}% RAM"
        )
    with col_m2:
        st.metric(
            label="Estimated Ingestion Duration",
            value=f"{hours_est:.1f} Hours",
            delta=f"{batch_size_choice}x Batching Speedup"
        )
        st.metric(
            label="Tokens Bypassed via Cache",
            value=f"{total_tokens * (cache_hit_pct / 100.0) / 1e6:,.1f} M Tokens",
            help="Tokens served instantly from SQLite hash cache."
        )

with col_roi_summary:
    st.subheader("Executive Architecture Summary")
    st.markdown(f"""
    * **Corpus Scale**: `{total_docs:,} chunks` ({total_tokens_millions:,.0f} Million tokens).
    * **Cache Efficiency**: `{cache_hit_pct}% hit rate` eliminates `{total_docs * (cache_hit_pct / 100.0):,}` redundant API calls.
    * **Batching Multiplier**: Using batches of `{batch_size_choice}` reduces HTTP handshake overhead by **{batch_size_choice}x**.
    * **Memory Dimension Reduction**: Slicing to `{active_dim} dims` cuts vector server RAM from **{base_ram_gb:.1f} GB down to {opt_ram_gb:.1f} GB**.
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why does vector normalization allow replacing Euclidean distance with Dot Product?"):
    st.markdown("""
    **Answer**:
    The squared Euclidean distance between two vectors `A` and `B` expands algebraically as:  
    `||A - B||^2 = ||A||^2 + ||B||^2 - 2 * <A, B>`.
    
    If both vectors are **L2-normalized to unit length** (`||A|| = 1.0` and `||B|| = 1.0`):  
    `||A - B||^2 = 1.0 + 1.0 - 2 * <A, B> = 2.0 - 2 * <A, B>`.
    
    * Notice that the squared Euclidean distance is a strictly monotonic linear function of the negative dot product.
    * Minimizing Euclidean distance is mathematically identical to **maximizing the Dot Product**.
    * Therefore, the database can discard Euclidean square roots and subtractions entirely, ranking vectors strictly by their inner product!
    """)

with st.expander("Q2: How do you design an embedding cache that invalidates entries when document content changes?"):
    st.markdown("""
    **Answer**:
    Use **Content-Addressable Cryptographic Hashing**:
    1. Rather than keying the cache by arbitrary IDs (`chunk_12`), key the cache by the **SHA-256 hash of the sanitized text string**:  
       `cache_key = sha256(cleaned_chunk_text)`.
    2. If the author updates the document, the new text produces a different SHA-256 hash. The cache lookup misses and triggers a fresh embedding call.
    3. The old embedding key remains unreferenced and can be pruned by an LRU eviction policy. This guarantees that stale embeddings can never contaminate retrieval.
    """)

with st.expander("Q3: What are the risks of using excessively large batch sizes (e.g. 5,000 chunks) when calling cloud embedding APIs?"):
    st.markdown("""
    **Answer**:
    1. **Payload Size & Timeout**: 5,000 chunks of 500 tokens equals 2.5 million tokens. The HTTP request body will exceed 10 MB, causing gateway timeouts (`HTTP 504 Gateway Timeout`) or payload size rejections (`HTTP 413 Payload Too Large`).
    2. **All-or-Nothing Failure**: If a single chunk in a 5,000-chunk batch triggers a content moderation filter or encoding error, the entire batch of 5,000 chunks fails, forcing complex retry partitioning.
    3. **Optimal Size**: Keep batches between **100 and 500 chunks** (~50k to 100k tokens).
    """)

with st.expander("Q4: How does Matryoshka Representation Learning (MRL) differ from Principal Component Analysis (PCA)?"):
    st.markdown("""
    **Answer**:
    * **PCA**: A post-hoc linear dimensionality reduction technique. PCA calculates orthogonal eigenvectors of the covariance matrix. While it reduces dimensions, it discards non-linear semantic manifolds and requires an extra matrix multiplication projection step for every incoming query.
    * **MRL**: Embedded directly into the transformer's loss function during neural network pre-training. The model is penalized for retrieval errors at prefix dimensions `[64, 128, 256, 512, 1024]`. Truncating an MRL vector requires zero projection math—you simply slice the Python array (`vec[:512]`) and re-normalize.
    """)

with st.expander("Q5: In a multi-tenant SaaS application, should the embedding cache be shared across tenants or partitioned per tenant?"):
    st.markdown("""
    **Answer**:
    The embedding cache can safely be **globally shared across tenants**:
    * Embedding generation is purely a mathematical mapping from text to high-dimensional space: `f(text) -> vector`.
    * If Tenant A and Tenant B both ingest the public Apache 2.0 license text, the resulting vector embedding is mathematically identical.
    * Caching by `sha256(text)` does not disclose tenant identity, access permissions, or proprietary metadata (metadata is stored in ChromaDB, not in the embedding cache). Sharing the cache maximizes deduplication savings across the entire customer base.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Forgetting to Re-Normalize After Slicing MRL Vectors**  
    Truncating an MRL vector from 1,536 to 512 dimensions drops its L2 norm below 1.0. Forgetting to re-normalize (`v = v / norm(v)`) breaks cosine similarity math in Chroma. Always re-normalize after slicing.
    """)
    st.warning("""
    **Trap 2: Caching Un-Normalized Raw Text**  
    Hashing text with trailing whitespace or un-stripped HTML causes minor formatting differences to miss the cache. Always apply the full cleaning pipeline *before* computing the SHA-256 cache key.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Serial Single-String Embedding Ingestion**  
    Calling `embeddings.create(input="single string")` in a `for` loop turns a 1-hour ingestion job into a 30-hour bottleneck due to HTTP TLS handshake overhead. Always ingest in batches.
    """)
    st.warning("""
    **Trap 4: Mixing Dot Product Collections with Unnormalized Vectors**  
    Using `hnsw:space: "ip"` on vectors that have not been L2-normalized causes search results to favor long, high-magnitude vectors rather than semantically similar ones.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: How does L2 normalization speed up search?"):
        st.markdown("""
        **Dot Product Speedup**:  
        When `||v|| = 1.0`, Cosine Similarity equals Inner Product, replacing square roots and divisions with a single hardware dot product.
        """)
    with st.expander("🗂️ Flashcard: How to key an embedding cache?"):
        st.markdown("""
        **Content Hashing**:  
        `key = sha256(cleaned_text)`. Automatically invalidates cache when content changes and enables safe multi-tenant sharing.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Matryoshka Slicing?"):
        st.markdown("""
        **MRL Truncation**:  
        Models trained so prefix dimensions carry core information, allowing slicing (e.g. 1536 -> 512) to cut RAM by 66% with <2% recall loss.
        """)
    with st.expander("🗂️ Flashcard: What is the recommended batch size?"):
        st.markdown("""
        **Batch Recommendation**:  
        100 to 500 chunks per batch, balancing network handshake amortization with API payload limits.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know the 5 pillars of embedding optimization: Batching, Caching, Normalization, MRL, and Model Selection.", value=True)
chk2 = st.checkbox("I understand why L2 normalization reduces Cosine Similarity to a hardware dot product.", value=True)
chk3 = st.checkbox("I can implement a high-throughput persistent embedding cache using SQLite and SHA-256.", value=True)
chk4 = st.checkbox("I know how to safely slice Matryoshka embeddings and re-normalize them to unit length.", value=True)
chk5 = st.checkbox("I can calculate financial ROI and runtime speedups for enterprise ingestion pipelines.", value=True)
chk6 = st.checkbox("I can design a cost-effective, rate-limited batch embedding architecture for production scale.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completed Chapter 13: Embedding Pipeline and are ready for Chapter 14: Retrieval-Augmented Generation (RAG) with Vector Databases!")
