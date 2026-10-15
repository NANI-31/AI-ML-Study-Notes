import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="7.2 Adding Embeddings - Vector DB Curriculum",
    page_icon="🧬",
    layout="wide"
)

# Title & Overview
st.title("7.2 Adding Embeddings: Manual Ingestion, Auto-Embeddings & Model Integrations")
st.caption("Chapter 7: Adding Data to a Vector Database | Domain: 4. Vector Database")
st.markdown("""
While ChromaDB provides automatic embedding helpers for rapid development, enterprise production architectures 
almost universally transition to **Pre-Computed (Manual) Embedding Ingestion**.

Why? Because in large-scale systems, embedding generation is computationally intensive (requiring GPU clusters, 
batch tensor parallelization, or external rate-limited API calls), while vector database insertion is I/O-intensive 
(writing to WAL and building HNSW graph indices). 

Decoupling embedding calculation from database storage allows you to scale **vector compute workers** independently 
from your **database storage cluster**.

In this lesson, we explore both ingestion paradigms:
1. **Manually Generated Embeddings** (`collection.add(embeddings=...)`)
2. **Automatic In-Database Embeddings** (`collection.add(documents=...)`)
3. The Chroma **`EmbeddingFunction` Protocol**
4. Enterprise Model Integrations (OpenAI, HuggingFace, FastEmbed, and Self-Hosted Endpoints)
""")

st.divider()

# Section 1: Manual vs. Automatic Ingestion Architectures
st.header("1. Manual vs. Automatic Embedding Ingestion", divider="rainbow")

col_mode1, col_mode2 = col_mode1, col_mode2 = st.columns([1, 1])

with col_mode1:
    st.markdown("""
    ### Mode 1: Manually Generated Embeddings (Production Standard)
    * **How it works**:
      1. Upstream GPU workers (e.g. Ray, Celery, or AWS Batch) compute embeddings in parallel:
         `vectors = model.encode(texts, batch_size=256)`.
      2. The pre-computed arrays are passed directly into the database:
         ```python
         collection.add(
             embeddings=vectors,
             documents=texts,
             metadatas=metadata_list,
             ids=id_list
         )
         ```
    * **Advantages**:
      * **Compute Decoupling**: Database CPU/RAM is never starved by heavy neural network inference.
      * **Caching & Resilience**: Embeddings can be persisted to Parquet/S3; if the database cluster restarts, you don't have to re-pay API fees!
      * **Supports Custom Modalities**: Essential for multimodal images, audio, or custom fine-tuned models.
    """)

with col_mode2:
    st.markdown("""
    ### Mode 2: Automatic In-Database Embeddings (Developer Convenience)
    * **How it works**:
      1. You pass only text: `collection.add(documents=texts, ids=ids)`.
      2. ChromaDB's registered embedding function vectorizes the text on-the-fly inside the client process.
    * **Advantages**:
      * Zero boilerplate: perfect for rapid experimentation, prototypes, and local scripts.
    * **Limitations**:
      * Tightly couples database ingestion to embedding latency.
      * Ingestion slows to a crawl if the client process is bottlenecked by CPU inference.
    """)

st.markdown("### Architecture Comparison Flowchart")
embed_arch_dot = """
digraph EmbeddingIngestionModes {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    RawText [label="Raw Text Documents\\n(1,000,000 Chunks)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_mode_manual {
        label="Mode 1: Decoupled Manual Ingestion (Enterprise)"; style="dashed"; color="#059669";
        GPUCluster [label="Distributed GPU Workers (Ray)\\nBatch encode: 10k docs/sec", fillcolor="#D1FAE5"];
        ParquetCache [label="Intermediate Storage (S3 / Parquet)\\nPersistent Embedding Cache", fillcolor="#A7F3D0"];
        VDBManual [label="Vector DB (Pure Storage & HNSW)\\ncollection.add(embeddings=vecs)", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
    }

    subgraph cluster_mode_auto {
        label="Mode 2: Embedded Auto-Vectorization"; style="dashed"; color="#DC2626";
        ClientProcess [label="Single Client Process\\nRuns ONNX / API synchronously", fillcolor="#FEE2E2"];
        VDBAuto [label="Vector DB Storage\\ncollection.add(documents=texts)", fillcolor="#FECACA", color="#DC2626", style="filled,bold"];
    }

    RawText -> GPUCluster -> ParquetCache -> VDBManual;
    RawText -> ClientProcess -> VDBAuto;
}
"""
st.graphviz_chart(embed_arch_dot, use_container_width=False)

st.divider()

# Section 2: Deep Dive into the Chroma `EmbeddingFunction` Protocol
st.header("2. The Chroma `EmbeddingFunction` Protocol Dissected", divider="rainbow")

st.markdown("""
ChromaDB defines a lightweight, clean protocol for all embedding functions. Any class implementing 
a `__call__` method accepting a list of strings (`Documents`) and returning a list of float arrays (`Embeddings`) 
can be bound to a collection:
""")

col_proto1, col_proto2 = col_proto1, col_proto2 = st.columns([1, 1])

with col_proto1:
    st.markdown("""
    ### The Official Protocol Definition
    ```python
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class MyCustomEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            \"\"\"
            input: List[str] (texts to embed)
            returns: List[List[float]] or np.ndarray (dense vectors)
            \"\"\"
            # 1. Tokenize & Infer
            vectors = my_model.encode(input)
            
            # 2. Return as list of floats
            return vectors.tolist()
    ```
    * **Thread Safety**: The `__call__` method should be re-entrant and thread-safe.
    * **Batching**: Chroma passes documents in batches (typically matching the ingestion batch size).
    """)

with col_proto2:
    st.markdown("""
    ### Built-in Commercial & Open-Source Functions
    Chroma bundles pre-built integrations in `chromadb.utils.embedding_functions`:
    * **`OpenAIEmbeddingFunction`**:
      * Models: `text-embedding-3-small`, `text-embedding-3-large`.
      * Supports Matryoshka dimension truncation (`dimensions=512`).
    * **`SentenceTransformerEmbeddingFunction`**:
      * Direct PyTorch integration with HuggingFace models.
    * **`FastEmbedEmbeddingFunction`**:
      * CPU-optimized ONNX runtime with zero PyTorch dependency.
    * **`CohereEmbeddingFunction`**:
      * Enterprise multilingual models (`embed-multilingual-v3.0`).
    """)

st.divider()

# Section 3: Interactive Ingestion Mode Simulator & Cost Estimator
st.header("3. Interactive Ingestion Mode Simulator & Cost Estimator", divider="rainbow")

st.markdown("""
Compare ingestion throughput, compute resource utilization, and API costs across 
**Auto-Embedding**, **Decoupled GPU Workers**, and **Cloud APIs**:
""")

col_calc_in, col_calc_out = st.columns([1, 1])

with col_calc_in:
    st.subheader("Workload & Model Configuration")
    sim_doc_count = st.select_slider(
        "Corpus Volume (Document Chunks)",
        options=[1000, 10000, 50000, 250000, 1000000, 5000000],
        value=50000,
        format_func=lambda x: f"{x:,} chunks"
    )
    sim_avg_tokens = st.slider("Average Tokens per Chunk", 100, 1000, 400, 50)
    sim_model_choice = st.selectbox(
        "Embedding Model Architecture",
        options=[
            "OpenAI text-embedding-3-small ($0.02 / 1M tokens)",
            "OpenAI text-embedding-3-large ($0.13 / 1M tokens)",
            "Local Open-Weights (BGE-M3 / FastEmbed on CPU)",
            "Dedicated GPU Cluster (NVIDIA A10G / Ray)"
        ]
    )

with col_calc_out:
    st.subheader("Performance & Financial Projections")
    
    total_tokens = sim_doc_count * sim_avg_tokens
    
    if "3-small" in sim_model_choice:
        api_cost = (total_tokens / 1_000_000) * 0.02
        ingest_time_min = (sim_doc_count / 150) / 60.0 # ~150 docs/sec API rate limit
        recommended_mode = "**Mode 1: Decoupled Manual Ingestion** with retry backoff queue."
    elif "3-large" in sim_model_choice:
        api_cost = (total_tokens / 1_000_000) * 0.13
        ingest_time_min = (sim_doc_count / 120) / 60.0
        recommended_mode = "**Mode 1: Decoupled Manual Ingestion** (cache vectors to S3 first!)."
    elif "Local Open-Weights" in sim_model_choice:
        api_cost = 0.0 # Free
        ingest_time_min = (sim_doc_count / 250) / 60.0 # ~250 docs/sec on 8-core CPU
        recommended_mode = "**Mode 2: Auto-Embedding via FastEmbed** (Simple & free)."
    else: # GPU Cluster
        api_cost = 15.0 # ~$1.00/hr cloud GPU
        ingest_time_min = (sim_doc_count / 4000) / 60.0 # ~4,000 docs/sec on GPU
        recommended_mode = "**Mode 1: Distributed Ray GPU Pipeline** -> Batch Parquet to Chroma."
        
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.metric("Total Tokens Processed", f"{total_tokens:,.0f}", f"{total_tokens/1e6:.2f}M Tokens")
        st.metric("Embedding API / Compute Cost", f"${api_cost:.2f}")
    with c_p2:
        st.metric("Estimated Ingestion Duration", f"{ingest_time_min:.1f} minutes", f"Model: {sim_model_choice.split(' ')[0]}")
        st.info(f"**Recommended Architecture:** {recommended_mode}")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Manual Vector Ingestion & Custom Embedding Protocol", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates:
1. Writing a custom **Retry-Aware Embedding Function** conforming to Chroma's protocol.
2. Ingesting **Pre-Computed Manual Vectors** directly into a collection:
""")

code_embed_demo = '''import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import numpy as np

# 1. Custom Embedding Function conforming to Chroma protocol
class CustomMockEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function generating normalized vectors."""
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            np.random.seed(abs(hash(doc)) % (2**32))
            v = np.random.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(v)
            unit_v = (v / (norm if norm > 0 else 1.0)).tolist()
            embeddings.append(unit_v)
        return embeddings

client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

# Demonstration A: Auto-Embedding using Custom Function
print("=== Demonstration A: Auto-Embedding Mode ===")
custom_ef = CustomMockEmbeddingFunction(dimension=8)
col_auto = client.create_collection(
    name="auto_embedded_collection",
    embedding_function=custom_ef,
    metadata={"hnsw:space": "cosine"}
)

col_auto.add(
    documents=["Kubernetes ingress routing with NGINX.", "PostgreSQL connection pooling."],
    ids=["doc_auto_1", "doc_auto_2"]
)
print(f"Auto-Embedded Collection Count: {col_auto.count()}")

# Demonstration B: Pre-Computed Manual Vector Ingestion (Enterprise Standard)
print("\\n=== Demonstration B: Pre-Computed Manual Vector Ingestion ===")
# Here, we do NOT pass an embedding function to the collection
col_manual = client.create_collection(
    name="manual_vector_collection",
    metadata={"hnsw:space": "cosine"}
)

# Simulated pre-computed vectors from external GPU / Ray pipeline
np.random.seed(42)
precomputed_vectors = [
    np.random.randn(8).tolist(),
    np.random.randn(8).tolist()
]

# Pass embeddings directly!
col_manual.add(
    embeddings=precomputed_vectors,
    documents=["Pre-computed document 1 text", "Pre-computed document 2 text"],
    ids=["doc_man_1", "doc_man_2"],
    metadatas=[{"source": "gpu_pipeline"}, {"source": "gpu_pipeline"}]
)
print(f"Manual Vector Collection Count: {col_manual.count()}")

# Querying manual collection using pre-computed query vector
query_vec = np.random.randn(8).tolist()
hits = col_manual.query(query_embeddings=[query_vec], n_results=1)
print(f"Manual Query Top Hit ID: {hits['ids'][0][0]}")
'''

st.code(code_embed_demo, language="python")

if st.button("🚀 Execute Embedding Ingestion Demonstrations", type="primary"):
    exec_scope = {}
    exec(code_embed_demo, globals(), exec_scope)
    
    cnt_a = exec_scope["col_auto"].count()
    cnt_b = exec_scope["col_manual"].count()
    man_hit = exec_scope["hits"]["ids"][0][0]
    
    st.success("Both Auto-Embedding and Manual Pre-Computed Ingestions executed successfully!")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Auto-Embedded Collection Count:** `{cnt_a}`")
    with c2:
        st.markdown(f"**Manual Vector Collection Count:** `{cnt_b}` (Top Hit: `{man_hit}`)")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. In enterprise architecture, why should vector generation be decoupled from database insertion?"""
a1 = """* **Resource Asymmetry**:
  * Vector generation is GPU-bound, high-latency, and tensor-parallel.
  * Vector database insertion is I/O-bound, memory-page-bound, and graph-index-bound.
  * Running them in a single synchronous process causes massive resource starvation and locks up connection pools.
* **Cost & Fault Tolerance**:
  * If a database crash occurs during ingestion of 100,000 documents, a tightly coupled script must re-run all embedding API calls (wasting thousands of dollars!).
  * In a decoupled architecture, vectors are computed by distributed GPU workers (Ray / Celery) and cached to Parquet files on Amazon S3. Database insertion is a fast, repeatable I/O operation."""

q2 = """2. What happens if you pass `embeddings` to a collection that also has an `embedding_function` defined?"""
a2 = """* ChromaDB prioritizes the explicitly supplied **`embeddings` parameter**.
* If you pass both `embeddings` and `documents`, Chroma will index the provided vector arrays directly and bypass executing the `embedding_function`.
* The `documents` strings are simply stored in SQLite as the text payload for future hydration.
* This allows you to store raw text alongside custom pre-computed vectors without triggering duplicate model inference."""

q3 = """3. How does Matryoshka Representation Learning (MRL) work when passing manual embeddings into ChromaDB?"""
a3 = """* Models like OpenAI `text-embedding-3-small` and `text-embedding-3-large` support MRL.
* You can generate a 1536-dimensional vector, slice the first 512 dimensions (`vector[:512]`), re-normalize it to unit length (`v / np.linalg.norm(v)`), and insert it into a collection configured for `dimension=512`.
* This cuts Chroma's memory consumption and HNSW graph size by **66.6%** while maintaining 98%+ of the original retrieval recall."""

q4 = """4. What are the key performance differences between SentenceTransformers and FastEmbed in local deployments?"""
a4 = """* **SentenceTransformers**:
  * Built on full PyTorch and HuggingFace Transformers.
  * Massive dependency footprint (PyTorch wheel is ~800 MB).
  * High memory overhead and slower cold-start times.
* **FastEmbed (Qdrant library)**:
  * Built on lightweight **ONNX Runtime CPU**.
  * Zero PyTorch dependency: install package is under 20 MB!
  * Optimized with Intel AVX-512 and ARM NEON quantized execution, achieving 2x-4x higher CPU inference throughput with negligible RAM overhead."""

q5 = """5. Why should embeddings always be stored as 32-bit floats (`float32`) rather than 64-bit floats (`float64`)?"""
a5 = """* Python NumPy arrays default to `float64` (8 bytes per number).
* High-dimensional embedding spaces do not require 64-bit precision; deep learning models are trained in FP16 or FP32.
* Storing 1536 dimensions in `float64` consumes **12.28 KB per vector** instead of **6.14 KB in float32**—doubling your RAM usage, doubling disk I/O, and disabling hardware SIMD 32-bit vectorization registers.
* Always cast vectors to `np.float32` prior to insertion."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Embedding Ingestion Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Accidental Float64 RAM Inflation Trap**  
> Passing raw Python lists of high-precision floats into Chroma without casting to `float32` can double 
> the in-memory array footprint. Always ensure embeddings are structured as `np.float32` arrays before 
> serializing to database collections.

> [!WARNING]
> **Trap 2: Model Version Drift Across Decoupled Worker Nodes**  
> In distributed GPU ingestion pipelines, Worker Node A runs `bge-base-en-v1.5` while Worker Node B has an older 
> cached image running `bge-base-en-v1.0`. The two models generate incompatible coordinate spaces! 
> Always record the exact model hash and version string inside the collection metadata to catch worker version drift.

> [!WARNING]
> **Trap 3: Synchronous API Calls in Single-Threaded Ingestion Loops**  
> Calling `openai.embeddings.create()` one document chunk at a time in a `for` loop achieves less than 5 docs/second 
> due to sequential network latency. Always batch requests into chunks of 100-500 documents and execute calls asynchronously.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: When to Use Manual Embeddings?"):
        st.markdown("""
        **Manual Embeddings**:  
        Mandatory in enterprise pipelines to decouple heavy GPU inference from database storage and allow vector caching in S3.
        """)
    with st.expander("🗂️ Flashcard: What is FastEmbed?"):
        st.markdown("""
        **FastEmbed**:  
        A lightweight ONNX Runtime CPU embedding library with zero PyTorch dependency, achieving high CPU throughput.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What Protocol Does Chroma Use?"):
        st.markdown("""
        **`EmbeddingFunction` Protocol**:  
        Requires implementing a `__call__(input: Documents) -> Embeddings` method returning a list of float arrays.
        """)
    with st.expander("🗂️ Flashcard: Why Cast to `float32`?"):
        st.markdown("""
        **Float32 Precision**:  
        Avoids doubling RAM usage and matches modern hardware SIMD vector registers (AVX-512).
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to add pre-computed vectors manually using `collection.add(embeddings=...)`.", value=True)
chk2 = st.checkbox("I understand why enterprise pipelines decouple vector generation from database ingestion.")
chk3 = st.checkbox("I know how to author a custom embedding function conforming to the Chroma protocol.")
chk4 = st.checkbox("I understand the performance advantages of FastEmbed (ONNX) over SentenceTransformers (PyTorch).")
chk5 = st.checkbox("I know why embeddings should always be formatted as `float32` rather than `float64`.")
chk6 = st.checkbox("I can implement a working dual-mode embedding ingestion script in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Adding Embeddings and are ready for Chapter 7.3: Metadata!")
