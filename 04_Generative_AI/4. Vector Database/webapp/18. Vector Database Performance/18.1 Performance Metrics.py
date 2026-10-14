import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="18.1 Performance Metrics - Vector DB Curriculum",
    page_icon="⏱️",
    layout="wide"
)

# Title & Overview
st.title("18.1 Vector Database Performance Metrics: Latency, Indexing, Throughput, Recall & Memory Footprint")
st.caption("Chapter 18: Vector Database Performance | Domain: 4. Vector Database")
st.markdown("""
Evaluating a vector database requires fundamentally different metrics than benchmarking a relational SQL database. 
Traditional databases prioritize ACID transaction throughput (TPS) and B-Tree index scan speeds. 

In vector databases, performance is governed by **The Fundamental Performance Triangle**:
a perpetual engineering trade-off between **Search Latency**, **Retrieval Recall (Accuracy)**, and **RAM Footprint**.

To architect, tune, and monitor production vector systems (like ChromaDB, Milvus, or Qdrant), engineers must track 
six primary telemetry metrics:
1. **Query Latency (p50, p95, p99)**: The millisecond round-trip time of vector similarity search.
2. **Indexing Time & Throughput**: Vectors ingested per second during batch construction.
3. **Throughput (QPS)**: Concurrency ceiling before query queue saturation.
4. **Recall@K**: Percentage of true nearest neighbors returned relative to exact brute-force search.
5. **Precision@K & MRR**: Information retrieval ranking fidelity.
6. **Memory Footprint**: RAM consumed per million vectors across vectors, graph edges, and metadata.

In this lesson, we explore these metrics, analyze the memory sizing formulas for high-dimensional vectors, 
and experiment with an interactive Vector DB Performance & Sizing Calculator.
""")

st.divider()

# Section 1: The Fundamental Performance Triangle
st.header("1. The Fundamental Performance Triangle", divider="rainbow")

col_tri1, col_tri2 = col_tri1, col_tri2 = st.columns([1.1, 0.9])

with col_tri1:
    st.markdown("""
    ### The Three Inescapable Trade-Offs
    
    * **1. High Recall (Accuracy > 98%)**:
      * Requires exploring deeper graph paths (`efSearch >= 128`) or retaining high-precision float32 coordinates.
      * *Cost*: Higher CPU compute, higher p99 latency, and massive RAM consumption.
    
    * **2. Ultra-Low Latency (< 2 ms)**:
      * Requires shallow graph exploration (`efSearch = 16 - 32`) or small connectivity `M`.
      * *Cost*: Recall drops from 98% down to 82%, missing true nearest neighbors.
    
    * **3. Lean Memory Footprint (Quantization)**:
      * Compresses vectors by 4x to 16x using Scalar Quantization (SQ8) or Product Quantization (PQ).
      * *Cost*: Vector coordinates are approximated, introducing distance distortion that degrades precision.
    """)

with col_tri2:
    st.markdown("### The Trade-Off Triangle")
    tri_dot = """
    digraph PerformanceTriangle {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Recall [label="1. High Recall (Accuracy)\\nExact nearest neighbors\\nZero missed documents", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        Latency [label="2. Ultra-Low Latency\\np99 < 3ms sub-second response\\nHigh QPS throughput", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Memory [label="3. Lean Memory (RAM)\\nAffordable hardware costs\\nCompressed quantization", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        Recall -> Latency [label="Trade-off: Higher efSearch increases latency", dir=both, color="#DC2626"];
        Latency -> Memory [label="Trade-off: In-memory graphs consume RAM", dir=both, color="#DC2626"];
        Memory -> Recall [label="Trade-off: Quantization degrades recall", dir=both, color="#DC2626"];
    }
    """
    st.graphviz_chart(tri_dot, use_container_width=False)

st.divider()

# Section 2: Deep-Dive into the Six Core Metrics
st.header("2. Deep-Dive into the Six Core Metrics", divider="rainbow")

col_met1, col_met2 = col_met1, col_met2 = st.columns([1, 1])

with col_met1:
    st.markdown("""
    ### 1. Query Latency Breakdown
    End-to-end query latency `T_total` consists of 4 distinct phases:
    ```text
    T_total = T_network + T_embedding + T_hnsw_traverse + T_hydration
    ```
    * `T_embedding`: Bi-encoder inference (~15ms on OpenAI API, ~4ms local ONNX).
    * `T_hnsw_traverse`: In-memory graph navigation (**1.2 - 2.5 ms**).
    * `T_hydration`: Fetching raw text and JSON metadata from disk SQLite (**0.5 - 2.0 ms**).
    * **SLA Standard**: Production vector search p99 latency should remain strictly **under 15 ms** (excluding external LLM calls).

    ### 2. Ingestion & Indexing Throughput
    * Measured in **vectors ingested per second (VPS)**.
    * Real-time single inserts (`collection.add` with 1 vector) achieve ~50 VPS due to SQLite transaction commit overhead.
    * Batched inserts (`batch_size = 500 - 1000`) achieve **2,000 to 8,000 VPS** on modern NVMe drives.
    """)

with col_met2:
    st.markdown("""
    ### 3. Throughput (Queries Per Second - QPS)
    * The maximum volume of concurrent search inquiries a cluster can serve without request queuing.
    * Because HNSW traversal is CPU-bound, QPS scales linearly with available CPU cores:
      ```text
      Max_QPS = (Number_of_CPU_Cores * 1000) / Average_Latency_ms
      ```

    ### 4. Recall@K (Accuracy vs Ground Truth)
    * Measures the fraction of true nearest neighbors returned:
      ```text
      Recall@K = |Retrieved_K intersect GroundTruth_K| / K
      ```
    * Evaluated against exact brute-force Flat L2 search (Flat Index).
    * Target SLA: **Recall@10 >= 95%**.

    ### 5. RAM Footprint Calculation Formula
    For `N` vectors of dimension `D` with HNSW connectivity `M`:
    ```text
    RAM_bytes = N * ( (D * 4 bytes) + (M * 2 * 8 bytes) + Overhead )
    ```
    """)

st.divider()

# Section 3: Exact RAM Footprint Sizing Formulas
st.header("3. Exact RAM Footprint Sizing Formulas", divider="rainbow")

st.markdown("""
Calculating memory capacity beforehand prevents production out-of-memory (OOM) crashes. 
Review the byte allocation breakdown:
""")

col_calc_a, col_calc_b = col_calc_a, col_calc_b = st.columns([1, 1])

with col_calc_a:
    st.markdown("""
    ### Uncompressed Float32 Sizing Example
    * **Dataset**: 1,000,000 vectors (`N = 1,000,000`)
    * **Dimension**: 1,536 (`D = 1,536`, OpenAI text-embedding-3-small)
    * **HNSW Connectivity**: `M = 16`
    
    1. **Raw Vector Coordinates**:
       * `1,000,000 * 1,536 * 4 bytes = 6.14 GB`
    2. **HNSW Graph Edges**:
       * Each vector has `2 * M = 32` neighbor pointers (8 bytes per pointer).
       * `1,000,000 * 32 * 8 bytes = 0.26 GB`
    3. **Metadata & Inverted Index Cache**:
       * Typically `~1.5 GB` for document text and SQLite cache.
    * **Total RAM Required**: **~8.0 GB RAM per 1M vectors**.
    """)

with col_calc_b:
    st.markdown("""
    ### Quantized Sizing Comparison (SQ8 vs PQ)
    
    | Storage Format | Precision | Bytes / Vector (1536 dim) | RAM for 1M Vectors | Recall Retention |
    | :--- | :---: | :---: | :---: | :---: |
    | **Uncompressed Float32** | 32-bit float | **6,144 bytes** | **~8.0 GB** | **100% (Baseline)** |
    | **Scalar Quantization (SQ8)** | 8-bit int | **1,536 bytes** | **~2.5 GB (-68%)** | **~98.5%** |
    | **Product Quantization (PQ)** | 8-bit centroid | **384 bytes** | **~0.9 GB (-88%)** | **~92.0%** |
    | **Binary Quantization (BQ)** | 1-bit sign | **192 bytes** | **~0.4 GB (-95%)** | **~85.0%** |
    """)

st.divider()

# Section 4: Interactive Workbench: Vector DB Performance & Sizing Calculator
st.header("4. Interactive Workbench: Vector DB Performance & Sizing Calculator", divider="rainbow")

st.markdown("""
Input your production dataset size, vector dimensions, and HNSW parameters to simulate 
exact RAM footprints, throughput capacity, and estimated p99 search latencies.
""")

col_calc_in1, col_calc_in2 = col_calc_in1, col_calc_in2 = st.columns(2)

with col_calc_in1:
    st.subheader("1. Corpus & Embedding Parameters")
    n_vectors_mil = st.slider("Total Number of Vectors (Millions):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    dim_choice = st.selectbox(
        "Vector Dimensionality (D):",
        [
            "1,536 (OpenAI text-embedding-3-small)",
            "3,072 (OpenAI text-embedding-3-large)",
            "1,024 (BAAI BGE-large-en-v1.5)",
            "384 (all-MiniLM-L6-v2)"
        ]
    )
    d_val = int(dim_choice.split()[0].replace(",", ""))

with col_calc_in2:
    st.subheader("2. HNSW & Hardware Parameters")
    hnsw_m = st.selectbox("HNSW Edge Connectivity (M):", [16, 32, 64], index=0)
    hnsw_ef = st.slider("Search Beam Width (`efSearch`):", min_value=16, max_value=256, value=64, step=16)
    cpu_cores = st.slider("Available Server CPU Cores:", min_value=2, max_value=32, value=8, step=2)

# Calculations
n_exact = n_vectors_mil * 1_000_000
raw_vec_bytes = n_exact * d_val * 4
hnsw_edge_bytes = n_exact * (hnsw_m * 2) * 8
meta_bytes = n_exact * 1200  # ~1.2 KB text/meta per chunk
total_ram_bytes = raw_vec_bytes + hnsw_edge_bytes + meta_bytes
total_ram_gb = total_ram_bytes / (1024 ** 3)

# Quantized alternatives
sq8_ram_gb = ((n_exact * d_val * 1) + hnsw_edge_bytes + (meta_bytes * 0.5)) / (1024 ** 3)
pq_ram_gb = ((n_exact * (d_val // 4) * 1) + hnsw_edge_bytes + (meta_bytes * 0.5)) / (1024 ** 3)

# Latency and throughput estimation
base_lat_ms = 0.5 + (np.log2(n_vectors_mil + 1) * 0.3) + (hnsw_ef * 0.015)
qps_capacity = int((cpu_cores * 1000) / base_lat_ms)

# Recall estimation
if hnsw_ef >= 128:
    recall_est = 98.6
elif hnsw_ef >= 64:
    recall_est = 96.2
else:
    recall_est = 89.4

st.divider()

col_res_hw, col_res_perf = st.columns([1.1, 0.9])

with col_res_hw:
    st.subheader("RAM Capacity & Sizing Projections")
    
    df_sizes = pd.DataFrame([
        {"Format": "Full Precision (Float32)", "RAM Required": f"{total_ram_gb:.2f} GB", "Expected Recall": f"{recall_est:.1f}%"},
        {"Format": "Scalar Quantized (SQ8)", "RAM Required": f"{sq8_ram_gb:.2f} GB", "Expected Recall": f"{recall_est - 1.2:.1f}%"},
        {"Format": "Product Quantized (PQ)", "RAM Required": f"{pq_ram_gb:.2f} GB", "Expected Recall": f"{recall_est - 5.5:.1f}%"}
    ])
    st.dataframe(df_sizes, use_container_width=True, hide_index=True)
    
    st.caption(f"Raw vectors consume {raw_vec_bytes / (1024**3):.2f} GB | Graph edges consume {hnsw_edge_bytes / (1024**3):.2f} GB.")

with col_res_perf:
    st.subheader("Performance & Throughput Capacity")
    st.metric(label="Estimated p99 Search Latency", value=f"{base_lat_ms:.2f} ms")
    st.metric(label="Throughput Capacity (Max QPS)", value=f"{qps_capacity:,} QPS", delta=f"{cpu_cores} Parallel Cores")
    
    if total_ram_gb > 16.0:
        st.warning(f"⚠️ Recommendation: Full precision requires {total_ram_gb:.1f} GB RAM. Consider deploying on a 32GB RAM node or enabling SQ8 quantization.")
    else:
        st.success(f"✅ Sizing Healthy: Comfortably fits in a standard 16GB RAM instance.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the difference between Recall@K and Precision@K in vector database benchmarking?"):
    st.markdown("""
    **Answer**:
    * **Recall@K**: Evaluates whether the Approximate Nearest Neighbor (ANN) index found the actual mathematical nearest neighbors.
      `Recall@K = |ANN_Results intersect Ground_Truth_Exact_Flat_Results| / K`.
      If brute-force search returns vectors `[A, B, C]` and HNSW returns `[A, B, D]`, `Recall@3 = 2/3 = 66.7%`.
    * **Precision@K**: Evaluates information retrieval utility against labeled human ground truth (e.g. out of 5 retrieved documents, how many actually answer the user query).
    * In vector database systems, **Recall@K is the gold-standard system benchmark** because it measures algorithmic index fidelity independently of human labeling.
    """)

with st.expander("Q2: How do you measure true Ground Truth to calculate Recall@K for an HNSW index?"):
    st.markdown("""
    **Answer**:
    1. Extract a representative test set of 1,000 queries.
    2. Run an **exhaustive brute-force Flat L2/Cosine search** over the entire unindexed dataset (evaluating every single vector). Because brute-force evaluates every vector, its recall is by definition **100.0%**.
    3. Save the exact top-K document IDs as the `Ground_Truth` baseline.
    4. Execute the same 1,000 queries against the HNSW index with target `efSearch`.
    5. Compare the sets of retrieved IDs for each query and compute the average overlap ratio.
    """)

with st.expander("Q3: Why does query throughput (QPS) plateau even when CPU utilization reaches 100%?"):
    st.markdown("""
    **Answer**:
    HNSW graph traversal is **memory-bandwidth and CPU cache-bound**:
    * Navigating graph pointers involves jumping between non-contiguous memory addresses (pointer chasing).
    * When multiple CPU threads hammer the memory bus simultaneously, L1/L2/L3 cache misses skyrocket, causing CPU cores to stall waiting for RAM memory bus lines.
    * **Remedies**:
      1. Pin worker threads to specific NUMA nodes (using `numactl`).
      2. Quantize vectors (SQ8) to fit more coordinates into the L3 CPU cache.
      3. Scale horizontally across read replicas rather than overloading a single socket.
    """)

with st.expander("Q4: What causes p99 latency spikes in embedded vector databases like ChromaDB?"):
    st.markdown("""
    **Answer**:
    1. **SQLite Disk I/O Contention**: While HNSW traversal is in RAM, hydrating the raw document text and metadata reads from SQLite. Concurrent writes or checkpointing (`WAL checkpoint`) locks the SQLite database, causing tail latency spikes.
    2. **Garbage Collection (GC) Pauses**: In Python/Java runtimes, allocating millions of temporary candidate objects triggers major GC stop-the-world pauses.
    3. **Cold OS Page Cache**: If the index files exceed available RAM, Linux flushes pages to disk, causing synchronous NVMe read stalls.
    """)

with st.expander("Q5: How does Scalar Quantization (SQ8) cut RAM by 75% with negligible recall loss?"):
    st.markdown("""
    **Answer**:
    Standard embeddings use 32-bit single-precision floats (`4 bytes` per dimension).
    * **Scalar Quantization (SQ8)** computes the global minimum and maximum float values across each vector component and maps them to an 8-bit unsigned integer `[0, 255]` (`1 byte` per dimension):
      `uint8_val = int( ((float_val - min_val) / (max_val - min_val)) * 255 )`.
    * This immediately cuts coordinate storage from 4 bytes to 1 byte (a **75% reduction**).
    * Because neural network embeddings are naturally noise-tolerant, the distance approximation error is tiny, preserving **98%+ of original Recall@10**.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Benchmarking Without Warming the OS Page Cache**  
    Running latency benchmarks immediately upon starting the database reports slow 250ms disk read latencies. Run 500 warmup queries first to ensure index pages reside in RAM.
    """)
    st.warning("""
    **Trap 2: Ignoring HNSW Graph Edge Overhead in RAM Calculations**  
    Calculating RAM as `N * D * 4` forgets that HNSW edges add another 25-30% memory overhead. Account for pointer connectivity `M`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Single-Threaded Index Ingestion**  
    Ingesting 1,000,000 vectors one-by-one synchronously takes 8 hours. Use batch upserts (`batch_size = 500`) with worker thread pools to complete ingestion in minutes.
    """)
    st.warning("""
    **Trap 4: Tuning `efSearch` Too High in Production**  
    Setting `efSearch = 500` to squeeze out an extra 0.5% recall triples search latency from 2ms to 6ms, cutting server QPS capacity by 60%. Keep `efSearch` between 32 and 96.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the Performance Triangle?"):
        st.markdown("""
        **The 3 Trade-Offs**:  
        Search Latency vs. Retrieval Recall (Accuracy) vs. RAM Footprint.
        """)
    with st.expander("🗂️ Flashcard: How is Recall@K measured?"):
        st.markdown("""
        **Ground Truth Overlap**:  
        Fraction of ANN results matching exact brute-force Flat L2/Cosine search.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How much RAM does 1M 1536-dim vectors need?"):
        st.markdown("""
        **~8.0 GB RAM**:  
        6.14 GB raw float32 vectors + 0.26 GB HNSW graph + ~1.5 GB metadata & cache.
        """)
    with st.expander("🗂️ Flashcard: What is the benefit of SQ8?"):
        st.markdown("""
        **4x Compression**:  
        Converts 32-bit floats to 8-bit ints, slashing RAM by ~70% with >98% recall retention.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the Vector Database Performance Triangle: Latency vs Recall vs Memory.", value=True)
chk2 = st.checkbox("I know how to calculate exact RAM capacity requirements for high-dimensional vector datasets.", value=True)
chk3 = st.checkbox("I understand the components of end-to-end query latency (embedding + HNSW + hydration).", value=True)
chk4 = st.checkbox("I know how to measure Recall@K against brute-force Flat L2 ground truth baselines.", value=True)
chk5 = st.checkbox("I understand how Scalar Quantization (SQ8) reduces memory footprints by 70%.", value=True)
chk6 = st.checkbox("I can calculate the QPS throughput ceiling based on CPU core count and average search latency.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Performance Metrics and are ready for Chapter 18.2: Vector Database Optimization!")
