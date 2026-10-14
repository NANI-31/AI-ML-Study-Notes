import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
import hashlib
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="21.1 Performance Optimization - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("21.1 Performance Optimization: Batching, Async Pipelines, Caching, & Parallel Workers")
st.caption("Chapter 21: Embedding Optimization | Domain: 3. Vector Embeddings")
st.markdown("""
In prototype scripts, generating embeddings one by one (`for doc in docs: embed(doc)`) is simple and easy to debug.
However, in an enterprise setting ingesting **millions of documents**, naive sequential embedding causes catastrophic bottlenecks:
* **Network Roundtrip Penalty**: Ingesting 100,000 documents sequentially over an API with 50ms latency takes **over 83 hours**!
* **GPU Compute Starvation**: GPUs thrive on massive parallel matrix multiplications. Passing one 50-token sentence at a time utilizes less than 2% of tensor core capacity.
* **Redundant API Re-computation**: Re-running an indexing pipeline re-embeds identical, unchanged documents, burning CPU hours and API budgets.

By applying **Performance Optimization Levers**—GPU Tensor Batching, Asynchronous I/O, Cryptographic Hash Caching, and Parallel Worker Pools—you can accelerate ingestion throughput by **50x to 200x**!

This lesson explores how to design high-throughput, low-latency embedding ingestion architectures.
""")

st.divider()

# Section 1: The High-Throughput Ingestion Architecture
st.header("1. The 5 Levers of High-Throughput Vector Ingestion", divider="rainbow")

col_perf1, col_perf2 = col_perf1, col_perf2 = st.columns([1, 1])

with col_perf1:
    st.markdown("""
    ### The 5 Performance Levers
    * **1. Dynamic Tensor Batching**:
      * Groups 32, 64, or 128 chunks into a single forward pass.
      * Maximizes GPU Tensor Core memory bandwidth and saturates cloud REST payloads (e.g. OpenAI accepts up to 2048 texts per call).
    * **2. Asynchronous I/O (`asyncio` / `aiohttp`)**:
      * Dispatches concurrent HTTP/gRPC requests non-blockingly while waiting for network responses, hiding transmission latency.
    * **3. Cryptographic Hash Caching (`CacheBackedEmbeddings`)**:
      * Computes SHA-256 / MurmurHash on normalized text chunks. If the vector exists in Redis or local disk cache, retrieval is instant ($< 0.5\\text{ms}$) with zero GPU cost.
    * **4. Deduplication & Reusing Embeddings**:
      * Skips re-embedding identical boilerplates, headers, disclaimers, and unchanged document versions.
    * **5. Multi-Worker Parallel Processing (`ProcessPoolExecutor`)**:
      * Offloads CPU-intensive tokenization, regex cleaning, and JSON serialization across all CPU cores.
    """)

with col_perf2:
    st.markdown("### High-Throughput Ingestion Architecture")
    ing_dot = """
    digraph HighThroughputIngestion {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawDocs [label="Raw Unstructured Corpus\\n(100,000 Chunks)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_cache {
            label="Layer 1: Deduplication & Cache"; style="dashed"; color="#10B981";
            HashGen [label="SHA-256 Chunk Hasher", fillcolor="#D1FAE5"];
            RedisCache [label="Redis / KV Store\\nCache Hit (0ms GPU)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        subgraph cluster_batch {
            label="Layer 2: Batching & Async Dispatch"; style="dashed"; color="#8B5CF6";
            BatchAgg [label="Dynamic Batch Aggregator\\n(Batch Size = 64)", fillcolor="#EDE9FE"];
            AsyncPool [label="Async Worker Pool\\n(Concurrent HTTP/GPU forward)", fillcolor="#DDD6FE", color="#7C3AED", style="filled,bold"];
        }

        VDB [label="Vector Store Bulk Insert\\nHNSW Disk Commit", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        RawDocs -> HashGen;
        HashGen -> RedisCache [label="Check hash"];
        RedisCache -> VDB [label="Cache hit: reuse vector"];
        HashGen -> BatchAgg [label="Cache miss"];
        BatchAgg -> AsyncPool [label="Batched tensor"];
        AsyncPool -> VDB [label="Write new vectors & update cache"];
    }
    """
    st.graphviz_chart(ing_dot, use_container_width=False)

st.markdown("### Ingestion Strategy Benchmark Comparison (100,000 Documents)")
st.markdown("""
| Ingestion Architecture | Concurrency | GPU Tensor Utilization | Throughput (docs/sec) | Time for 100k Chunks |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Sequential (1 by 1)** | 1 (Blocking) | < 3% | ~20 docs/sec | **83.3 Minutes** |
| **Batched Synchronous (Batch=64)** | 1 (Blocking) | ~65% | ~350 docs/sec | **4.7 Minutes** |
| **Async Concurrent Batched** | 8 concurrent batches | **~95%** | **~1,800 docs/sec** | **55 Seconds (90x faster!)** |
| **Async Batched + 50% Cache Hit** | 8 concurrent + Cache | **~95%** | **~3,400 docs/sec** | **29 Seconds (170x faster!)** |
""")

st.divider()

# Section 2: Production Code: High-Throughput Async Batching Engine
st.header("2. Production Code: High-Throughput Engine with SHA-256 Caching", divider="rainbow")

st.markdown("""
The production utility below demonstrates an end-to-end high-throughput embedding engine featuring SHA-256 content hashing, in-memory cache reuse, dynamic batch aggregation, and throughput telemetry.
""")

st.code(r'''
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple

class HighThroughputEmbeddingEngine:
    """
    Optimizes embedding ingestion throughput via:
    - SHA-256 content hash caching (zero redundant computation)
    - Dynamic batch aggregation (maximizes tensor parallelism)
    - High-throughput execution telemetry
    """
    def __init__(self, embedding_dim: int = 64, batch_size: int = 32):
        self.dim = embedding_dim
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}

    def compute_chunk_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def mock_gpu_batch_embed(self, texts: List[str]) -> np.ndarray:
        """Simulates parallel GPU matrix multiplication for a batch."""
        vectors = []
        for t in texts:
            np.random.seed(abs(hash(t)) % (2**31 - 1))
            v = np.random.randn(self.dim).astype(np.float32)
            vectors.append(v / np.maximum(np.linalg.norm(v), 1e-12))
        return np.array(vectors)

    def ingest_corpus(self, documents: List[str]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        total_docs = len(documents)
        cache_hits = 0
        cache_misses = 0

        final_embeddings: List[np.ndarray] = [None] * total_docs
        missed_indices: List[int] = []
        missed_texts: List[str] = []

        # Step 1: Check SHA-256 Cache
        for idx, doc in enumerate(documents):
            h = self.compute_chunk_hash(doc)
            if h in self.cache:
                final_embeddings[idx] = self.cache[h]
                cache_hits += 1
            else:
                missed_indices.append(idx)
                missed_texts.append(doc)
                cache_misses += 1

        # Step 2: Dynamic Batch Processing for Missed Chunks
        for b_start in range(0, len(missed_texts), self.batch_size):
            batch_chunk_texts = missed_texts[b_start : b_start + self.batch_size]
            batch_chunk_indices = missed_indices[b_start : b_start + self.batch_size]

            # Forward pass on GPU
            batch_vectors = self.mock_gpu_batch_embed(batch_chunk_texts)

            # Store in output and cache
            for i, vec in enumerate(batch_vectors):
                orig_idx = batch_chunk_indices[i]
                final_embeddings[orig_idx] = vec
                h = self.compute_chunk_hash(batch_chunk_texts[i])
                self.cache[h] = vec

        elapsed_sec = time.perf_counter() - t_start
        throughput = total_docs / max(1e-6, elapsed_sec)

        return {
            "total_documents": total_docs,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / float(total_docs) if total_docs > 0 else 0.0,
            "elapsed_seconds": elapsed_sec,
            "throughput_docs_per_sec": throughput
        }
''', language="python")

st.divider()

# Section 3: Interactive Ingestion Throughput & Latency Studio
st.header("3. Interactive Workbench: Ingestion Throughput & Latency Studio", divider="rainbow")

st.markdown("""
Simulate how different optimization strategies affect the total processing time and throughput of an enterprise ingestion pipeline.
""")

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("1. Ingestion Pipeline Parameters")
    corpus_size = st.slider("Corpus Document Count (Passages):", min_value=100, max_value=5000, value=1000, step=100)
    sim_batch_size = st.select_slider("Batch Size:", options=[1, 8, 16, 32, 64, 128], value=32)
    cache_hit_ratio = st.slider("Cache Hit Rate (% of Duplicate Chunks):", min_value=0, max_value=80, value=30, step=5)
    use_async = st.checkbox("Enable Async Concurrent Dispatch (4 concurrent workers)", value=True)

with col_sim2:
    st.subheader("2. Simulated Benchmark Telemetry")

    # Simulation math:
    # Per-request network/IO overhead: 20ms
    # Per-token/chunk compute overhead: 1.5ms
    net_overhead_per_batch = 0.020 # 20 ms
    compute_per_doc = 0.0015 # 1.5 ms

    misses = int(corpus_size * (1.0 - cache_hit_ratio / 100.0))
    hits = corpus_size - misses

    # Cache hit time: instant lookup (0.05ms)
    time_cache = hits * 0.00005

    # Compute time for misses
    num_batches = int(np.ceil(misses / float(sim_batch_size)))
    time_compute = num_batches * net_overhead_per_batch + misses * compute_per_doc

    if use_async:
        time_compute = time_compute / 3.5 # 4 workers with ~3.5x speedup

    total_time_optimized = time_cache + time_compute
    throughput_optimized = corpus_size / max(1e-4, total_time_optimized)

    # Baseline: sequential, no cache, batch size = 1
    total_time_baseline = corpus_size * (net_overhead_per_batch + compute_per_doc)
    throughput_baseline = corpus_size / total_time_baseline
    speedup = total_time_baseline / total_time_optimized

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimized Elapsed Time", f"{total_time_optimized:.2f}s", delta=f"-{total_time_baseline - total_time_optimized:.1f}s")
    m2.metric("Throughput", f"{throughput_optimized:.0f} docs/sec", delta=f"{speedup:.1f}x Speedup")
    m3.metric("Cache Hit Savings", f"{hits} Chunks ({cache_hit_ratio}%)")

    st.markdown("#### Comparative Architecture Breakdown")
    arch_comparison = [
        {"Mode": "Baseline (Sequential 1-by-1)", "Batches": corpus_size, "Time": f"{total_time_baseline:.2f}s", "Throughput": f"{throughput_baseline:.0f} docs/s", "Speedup": "1.0x"},
        {"Mode": f"Batched (Batch={sim_batch_size})", "Batches": int(np.ceil(corpus_size / sim_batch_size)), "Time": f"{(int(np.ceil(corpus_size / sim_batch_size))*net_overhead_per_batch + corpus_size*compute_per_doc):.2f}s", "Throughput": f"{corpus_size / (int(np.ceil(corpus_size / sim_batch_size))*net_overhead_per_batch + corpus_size*compute_per_doc):.0f} docs/s", "Speedup": f"{total_time_baseline / (int(np.ceil(corpus_size / sim_batch_size))*net_overhead_per_batch + corpus_size*compute_per_doc):.1f}x"},
        {"Mode": "Optimized (Batch + Async + Cache)", "Batches": num_batches, "Time": f"{total_time_optimized:.2f}s", "Throughput": f"{throughput_optimized:.0f} docs/s", "Speedup": f"{speedup:.1f}x"}
    ]
    st.dataframe(pd.DataFrame(arch_comparison), use_container_width=True)

    st.success(f"🚀 Pipeline achieved **{speedup:.1f}x acceleration** over naive sequential ingestion!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Optimization Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do you determine the optimal GPU batch size for local embedding generation without crashing with Out-Of-Memory (OOM)?**
    * **Answer**:
      1. GPU memory consumed by a forward pass scales as: `Memory ~ ModelWeights + BatchSize * MaxSeqLength * HiddenDim * LayerCount * FloatBytes`.
      2. Begin with an exponential search: start at `batch_size = 8`, doubling to 16, 32, 64, 128 while profiling memory via `torch.cuda.max_memory_allocated()`.
      3. Implement **Dynamic Batching by Token Count**: rather than grouping fixed chunk counts, aggregate batches by *total token count* (e.g. cap batch at 8,192 tokens total). This prevents OOM spikes caused by occasional extra-long passages.

    **Q2: How does Cryptographic Hash Caching (`CacheBackedEmbeddings`) work in distributed RAG architectures?**
    * **Answer**:
      * Each text chunk is stripped of leading/trailing whitespace and lowercased.
      * A deterministic hash is computed: `Key = ModelName + ":" + SHA256(ChunkText)`.
      * The worker queries a centralized key-value store (Redis / DynamoDB):
        * If found: returns the stored vector bytes immediately (0ms GPU time).
        * If missing: queues the chunk for GPU embedding, then writes the result to Redis with a TTL.
      * This ensures that re-indexing jobs or multi-worker pipelines never compute the same embedding twice across the entire enterprise.

    **Q3: When does Asynchronous I/O fail to improve embedding throughput?**
    * **Answer**: Async I/O optimizes **I/O-bound** operations (e.g. waiting for remote OpenAI or Cohere REST APIs over the internet).
    * If you are running embeddings **locally on your own GPU**, the bottleneck is **Compute-bound** (CUDA matrix multiplications). Adding thousands of async coroutines on a saturated GPU provides zero speedup and can actually introduce CPU context-switching overhead. Local GPU pipelines require optimal batch sizing and multiprocessing tokenization workers rather than async REST loops.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Hashing Text Without Normalization**:
      * *Issue*: Hashing raw text with inconsistent trailing whitespace (`"Hello "` vs `"Hello"`).
      * *Consequence*: The cache treats identical passages as different keys, causing a 0% cache hit rate.
      * *Fix*: Always `.strip()` and normalize Unicode strings before computing the cryptographic hash.

    * **Trap 2: Blocking the Event Loop with Synchronous Tokenization in Async Code**:
      * *Issue*: Calling heavy CPU tokenizers inside an `async def` function without `asyncio.to_thread()`.
      * *Consequence*: Blocks the single Python event loop, turning concurrent async requests into blocking sequential operations.
      * *Fix*: Offload CPU tokenization to a `ProcessPoolExecutor` or worker thread.

    * **Trap 3: Oversized Batches Causing Cloud API Timeouts**:
      * *Issue*: Sending 5,000 documents in a single HTTP POST request to an embedding endpoint.
      * *Consequence*: The API gateway drops the connection after 30 seconds with a 504 Gateway Timeout.
      * *Fix*: Cap batch sizes between 64 and 256 items and implement exponential backoff retry wrappers.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: GPU Tensor Batching**
      * *Definition*: Grouping multiple text sequences into a single matrix tensor to maximize GPU parallel compute utilization.
    * **Flashcard 2: CacheBackedEmbeddings**
      * *Definition*: An idempotent caching architecture that maps text content hashes to pre-computed vectors to prevent redundant compute.
    * **Flashcard 3: Dynamic Token Batching**
      * *Definition*: Sizing ingestion batches by cumulative token count rather than passage count to prevent CUDA Out-Of-Memory errors.
    * **Flashcard 4: I/O-Bound vs. Compute-Bound Scaling**
      * *Definition*: Scaling cloud API ingestion via async concurrency (I/O-bound) vs. scaling local GPU ingestion via tensor batching (Compute-bound).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Performance Optimization:

- [ ] I can articulate the 5 levers of high-throughput vector ingestion.
- [ ] I understand how GPU tensor core batching accelerates forward pass throughput.
- [ ] I know how to implement SHA-256 content hash caching to prevent redundant API fees.
- [ ] I can distinguish between I/O-bound (cloud APIs) and compute-bound (local GPU) bottlenecks.
- [ ] I know how to prevent CUDA OOM errors using dynamic token batching.
""")
