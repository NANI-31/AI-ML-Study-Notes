import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="5.1 Vector Databases - ChromaDB Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Title & Overview
st.title("5.1 Vector Databases: Vector Storage, Indexing Structures & Nearest-Neighbor Foundations")
st.caption("Chapter 5: Vector Database Fundamentals | Domain: 5. ChromaDB")
st.markdown("""
In classical computer science, data structures are optimized for 1-dimensional ordering: 
integers, strings, and timestamps can be sorted linearly from lowest to highest. 
Relational databases leverage this property to build B-Trees, enabling logarithmic `O(log N)` point queries 
and range scans.

However, in modern AI applications, embeddings reside in **384 to 3,072 dimensional space**. 
High-dimensional vectors cannot be linearly sorted: there is no natural definition of whether 
`[0.4, -0.2, 0.8]` is "greater than" or "less than" `[-0.1, 0.9, 0.3]`.

A **Vector Database** is a specialized engine engineered specifically to solve this challenge:
* **Dense Vector Storage**: Compact memory layouts for millions of 32-bit floating-point arrays.
* **Non-Linear Indexing Structures**: Multi-layer geometric graphs (HNSW) and Voronoi cell partitions (IVF) that replace linear sorting.
* **Nearest-Neighbor Retrieval**: Calculating geometric proximity across multi-dimensional space in sub-10 millisecond latencies.

In this lesson, we explore the mechanical foundations of vector storage, compare indexing algorithms, 
dissect Nearest-Neighbor search mechanics without LaTeX math equations, and test an interactive 
Vector Storage & ANN Indexing Lab.
""")

st.divider()

# Section 1: Anatomy of a Vector Database Engine
st.header("1. Anatomy of a Vector Database Engine", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### Why Relational Databases Break at Scale
    * **The Flat Scan Bottleneck**: Without a vector index, finding the closest match to an incoming query requires a brute-force Euclidean distance calculation against **every single record in the database** (Time complexity: `O(N * D)`).
    * Across 10,000,000 documents with 1,536 dimensions, a single query requires **15.36 billion floating-point operations** (taking 4–10 seconds per search!).
    * **The Vector Database Solution**:
      * Replaces brute-force scans with **Approximate Nearest Neighbor (ANN)** indexing.
      * Traverses pre-computed geometric graphs (HNSW) in `O(log N)` logarithmic time, evaluating only ~200 candidate vectors instead of 10 million.
      * Delivers sub-15ms search latency with 98%+ recall accuracy!
    """)

with col_anat2:
    st.markdown("### Vector Engine Subsystem Topology")
    eng_dot = """
    digraph VectorSubsystems {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="User Query Vector\\n[0.12, -0.45, 0.88, ...]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_engine {
            label="Vector Database Subsystems"; style="dashed"; color="#1E40AF";
            HNSWIndex [label="Indexing Subsystem (HNSW / IVF)\\nGeometric Graph / Cluster Centroids\\nNarrows search to candidate subset", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            StorageEngine [label="Storage Subsystem\\nMemory-Mapped Floats / Quantization\\nFast SIMD Distance Calculations", fillcolor="#FEF3C7", color="#D97706"];
            MetadataStore [label="Metadata Filter Engine\\nSQLite / Inverted Index Filter", fillcolor="#EDE9FE", color="#7C3AED"];
        }

        TopK [label="Top-K Ranked Results\\nIDs, Distances, Hydrated Text", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InQuery -> HNSWIndex [label="Traverse Graph"];
        HNSWIndex -> StorageEngine [label="Fetch Candidate Vectors"];
        HNSWIndex -> MetadataStore [label="Apply Predicate Bitmask"];
        StorageEngine -> TopK;
    }
    """
    st.graphviz_chart(eng_dot, use_container_width=False)

st.divider()

# Section 2: Indexing Architectures: Flat, IVF & HNSW
st.header("2. Indexing Architectures: Flat, IVF & HNSW", divider="rainbow")

st.markdown("""
To balance search latency, index build time, and memory consumption, vector databases 
employ different indexing algorithms:
""")

INDEX_COMPARISON = [
    {
        "Index Algorithm": "Flat (Brute-Force / Exhaustive)",
        "Search Complexity": "O(N * D) - Linear Scan",
        "Recall Accuracy": "100% (Exact Ground Truth)",
        "Memory Overhead": "Zero index overhead (raw vectors only)",
        "Best Application": "< 10,000 vectors; benchmarking ground truth."
    },
    {
        "Index Algorithm": "IVF (Inverted File Index)",
        "Search Complexity": "O(Nprobe * (N / K) * D)",
        "Recall Accuracy": "90% - 97%",
        "Memory Overhead": "Low (~10% extra memory for centroids)",
        "Best Application": "Partitioning space into K Voronoi clusters; good for billion-scale disk storage."
    },
    {
        "Index Algorithm": "HNSW (Hierarchical Navigable Small World)",
        "Search Complexity": "O(log N) - Logarithmic Graph Traversal",
        "Recall Accuracy": "95% - 99.8%",
        "Memory Overhead": "High (~100% extra RAM for graph edges)",
        "Best Application": "In-memory real-time search (< 20ms); ChromaDB's core engine."
    }
]

st.dataframe(pd.DataFrame(INDEX_COMPARISON), use_container_width=True, hide_index=True)

st.divider()

# Section 3: Clean Python Implementation of Flat vs Graph Search Simulation
st.header("3. Clean Python Implementation of Flat vs Graph Search Simulation", divider="rainbow")

st.code("""import numpy as np
import time

# 1. Generate synthetic dataset of 100,000 384-dimensional vectors
num_vectors = 100000
dimensions = 384
np.random.seed(42)

dataset = np.random.randn(num_vectors, dimensions).astype(np.float32)
# Normalize to unit length
dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)

# Generate a query vector
query = np.random.randn(dimensions).astype(np.float32)
query /= np.linalg.norm(query)

# --- 2. Flat Brute-Force Search: Computes dot product against all 100,000 rows ---
t0 = time.perf_counter()
similarities = np.dot(dataset, query)
top_k_indices = np.argsort(similarities)[-5:][::-1]
flat_time_ms = (time.perf_counter() - t0) * 1000

print(f"Flat Brute-Force Search Time: {flat_time_ms:.2f} ms")
print(f"Top-1 Index: {top_k_indices[0]} (Sim: {similarities[top_k_indices[0]]:.4f})")

# In ChromaDB, HNSW reduces the candidate evaluation pool from 100,000 to ~200,
# achieving sub-1ms search times in C++!
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Vector Storage & Indexing Lab
st.header("4. Interactive Workbench: Vector Storage & Indexing Lab", divider="rainbow")

st.markdown("""
Simulate scaling a vector database from 10,000 to 10,000,000 vectors. 
Evaluate how dimensionality, index type, and concurrency impact RAM consumption, search latency, and infrastructure cost.
""")

col_sim_ds, col_sim_idx = col_sim_ds, col_sim_idx = st.columns(2)

with col_sim_ds:
    st.subheader("1. Dataset Dimensions")
    total_vectors = st.select_slider(
        "Total Indexed Vectors:",
        options=[10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
        value=500000
    )
    dim_choice = st.selectbox("Vector Dimensionality:", [384, 768, 1536, 3072], index=2)

with col_sim_idx:
    st.subheader("2. Index Architecture")
    idx_choice = st.radio(
        "Choose Index Type:",
        ["Flat (Brute-Force Scan)", "IVF (Clustered Inverted Index)", "HNSW (Navigable Small World Graph)"]
    )

# Calculations
raw_bytes = total_vectors * dim_choice * 4  # 4 bytes per float32
raw_ram_gb = raw_bytes / (1024**3)

if "Flat" in idx_choice:
    total_ram_gb = raw_ram_gb * 1.05
    search_latency_ms = (total_vectors * dim_choice) / 50000000.0  # Approx CPU FLOPs
    recall_pct = "100.0% (Exact)"
elif "IVF" in idx_choice:
    total_ram_gb = raw_ram_gb * 1.20
    search_latency_ms = max(1.5, ((total_vectors * 0.05) * dim_choice) / 50000000.0)
    recall_pct = "94.5% - 97.0%"
else:  # HNSW
    total_ram_gb = raw_ram_gb * 1.85  # HNSW edge pointers
    search_latency_ms = max(0.8, np.log2(total_vectors) * 0.4)
    recall_pct = "98.5% - 99.8%"

st.divider()

col_met1, col_met2, col_met3 = st.columns(3)
col_met1.metric(label="Total RAM Footprint", value=f"{total_ram_gb:.2f} GB RAM")
col_met2.metric(label="Estimated Single-Query Latency", value=f"{search_latency_ms:.2f} ms")
col_met3.metric(label="Search Recall Accuracy", value=recall_pct)

if "HNSW" in idx_choice:
    st.success(f"🏆 **ChromaDB Production Configuration**: HNSW delivers {search_latency_ms:.2f}ms query latency with {recall_pct} recall, making it ideal for interactive conversational AI.")
elif "Flat" in idx_choice and total_vectors >= 500000:
    st.error(f"🔴 **CRITICAL LATENCY BOTTLENECK**: Flat scans take {search_latency_ms:.2f}ms per query! Unsuitable for interactive user applications.")
else:
    st.info("💡 **IVF Configuration**: Provides balanced memory usage for large cold-storage datasets.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is Approximate Nearest Neighbor (ANN) preferred over Exact k-NN in production?"):
    st.markdown("""
    **Answer**:
    * **Exact k-NN** requires computing distances against every vector in the collection (Linear scan `O(N)`). At 10 million vectors, this takes seconds per query and cannot scale to concurrent traffic.
    * **Approximate Nearest Neighbor (ANN)** trades a negligible amount of accuracy (typically 0.5%–2% recall loss) for **100x–10,000x faster search speeds** (`O(log N)` logarithmic time).
    * For LLM context retrieval, a 98% recall rate is virtually indistinguishable from 100%, but sub-15ms response latency is critical for user experience.
    """)

with st.expander("Q2: How does the HNSW graph achieve logarithmic search time without checking all nodes?"):
    st.markdown("""
    **Answer**:
    * HNSW is inspired by the **Skip List** data structure, extended into multi-dimensional geometric graphs.
    * It maintains multiple layers of graphs:
      * **Top Layers (Sparse)**: Contain long-range "expressway" links between distant clusters. Search begins here, taking giant hops across space.
      * **Bottom Layer (Dense, Layer 0)**: Contains short-range links connecting immediate nearest neighbors.
    * The search algorithm greedily hops through the sparse layers to rapidly find the vicinity of the query, then descends into Layer 0 to fine-tune the exact Top-K nearest neighbors.
    """)

with st.expander("Q3: What is Product Quantization (PQ), and how does it reduce memory?"):
    st.markdown("""
    **Answer**:
    * **Product Quantization (PQ)** is a lossy vector compression technique:
      1. Splits a high-dimensional vector (e.g. 1536 floats = 6,144 bytes) into 16 smaller sub-vectors (96 floats each).
      2. Clusters each sub-vector space using k-means to create 256 cluster centroids.
      3. Replaces each 96-float sub-vector with a single 1-byte integer code pointing to its closest centroid.
    * **Result**: Reduces a 6,144-byte vector down to just **16 bytes**—an astonishing **97% reduction in RAM footprint**!
    """)

with st.expander("Q4: What is the difference between Inverted File Index (IVF) and HNSW?"):
    st.markdown("""
    **Answer**:
    * **IVF (Clustering-Based)**: Divides vector space into K Voronoi cells using k-means clustering. At query time, it identifies the closest cluster centroids and only searches the vectors inside those specific cells. (Lower RAM overhead, slower search).
    * **HNSW (Graph-Based)**: Connects vectors in a navigable multi-layer web. (Higher RAM overhead for storing edge links, but 5x–10x faster search latency and higher recall).
    """)

with st.expander("Q5: What are the primary bottlenecks in scaling vector databases to billions of vectors?"):
    st.markdown("""
    **Answer**:
    1. **RAM Sizing**: 1 billion 1536-dim vectors require 6 TB of raw RAM for floats alone, plus 5+ TB for HNSW graph edges.
    2. **Network Sharding**: Splitting vector indices across distributed worker nodes requires scatter-gather queries, where network latency dominates.
    3. **Index Build Time**: Constructing HNSW graphs on 1 billion nodes requires massive distributed batch compute jobs lasting hours or days.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Running Flat Brute-Force Scans in Production Web APIs**  
    Relying on flat cosine scans works in 1,000-doc test notebooks but grinds to a halt under 100 concurrent production users. Always use HNSW indices.
    """)
    st.warning("""
    **Trap 2: Under-Provisioning RAM for HNSW Collections**  
    HNSW graph pointers consume an additional 50%–100% of memory beyond raw vector floats. Sizing servers strictly for raw float bytes leads to OOM crashes.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Over-Tuning Search Parameters for 100% Recall**  
    Setting `efSearch` excessively high to chase the last 0.2% recall multiplies query latency by 5x with zero perceptible difference in LLM answer quality.
    """)
    st.warning("""
    **Trap 4: Forgetting SIMD CPU Instruction Support in Docker**  
    Deploying vector DB containers without compiling for AVX2 / AVX-512 forces the engine to run unvectorized scalar distance math, dropping search speed 8x.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is ANN search?"):
        st.markdown("""
        **Approximate Nearest Neighbor**:  
        Trades <2% recall for 100x–10,000x faster search speeds in logarithmic O(log N) time.
        """)
    with st.expander("🗂️ Flashcard: How does HNSW work?"):
        st.markdown("""
        **Multi-Layer Skip Graph**:  
        Hops through sparse expressways before refining nearest neighbors in dense base layer.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Product Quantization?"):
        st.markdown("""
        **Vector Compression**:  
        Compresses vectors into 1-byte centroid codes, shrinking RAM footprint by up to 97%.
        """)
    with st.expander("🗂️ Flashcard: Why do B-Trees fail on vectors?"):
        st.markdown("""
        **No 1D Sorting**:  
        High-dimensional points cannot be linearly sorted from lowest to highest.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why high-dimensional vectors cannot be indexed using standard relational B-Trees.", value=True)
chk2 = st.checkbox("I understand the trade-offs between Flat scans, IVF clustering, and HNSW graph traversal.", value=True)
chk3 = st.checkbox("I know how HNSW's multi-layer skip graph achieves logarithmic O(log N) search latency.", value=True)
chk4 = st.checkbox("I understand how Product Quantization compresses vector dimensions to save RAM.", value=True)
chk5 = st.checkbox("I know how to size server RAM accounting for both raw vector floats and HNSW graph edges.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 5.2: Vector Similarity Metrics in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 5.1: Vector Databases and are ready for Chapter 5.2: Vector Similarity!")
