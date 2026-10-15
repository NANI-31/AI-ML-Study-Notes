import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.1 What is an Index - Vector DB Curriculum",
    page_icon="🗂️",
    layout="wide"
)

# Title & Overview
st.title("10.1 What is a Vector Index: Architecture, Scaling & Construction")
st.caption("Chapter 10: Vector Indexing | Domain: 4. Vector Database")
st.markdown("""
In relational databases, querying an unindexed table causes a **Full Table Scan**—inspecting all rows sequentially. 
While a sequential scan on a few thousand integers takes a few milliseconds, in vector databases, **an unindexed search requires calculating 
high-dimensional floating-point dot products or Euclidean distances across every single stored vector**.

For a corpus of 10,000,000 vectors with 1,536 dimensions (e.g., OpenAI `text-embedding-3-small`), a single unindexed query requires 
performing **15.36 billion floating-point multiplications**. At scale, this exhaustive search is completely unusable for real-time applications.

A **Vector Index** is a specialized data structure that pre-organizes high-dimensional vectors in space—using graphs, trees, or clustered centroids—to 
prune away the vast majority of the search space, transforming brute-force linear scaling `O(N * D)` into sub-linear logarithmic scaling `O(log N * D)`.

In this lesson, we explore why indexing is mandatory, compare exhaustive flat scanning against indexed search, examine index construction 
mechanics and storage trade-offs, and experiment with an interactive latency and complexity scaling simulator.
""")

st.divider()

# Section 1: The Curse of Dimensionality & Exhaustive Search Collapse
st.header("1. The Curse of Dimensionality & Exhaustive Search Collapse", divider="rainbow")

col_dim1, col_dim2 = col_dim1, col_dim2 = st.columns([1.1, 0.9])

with col_dim1:
    st.markdown("""
    ### Search Without Indexing (Flat / Brute-Force Scan)
    When a database stores raw vectors without an index (often called `IndexFlatL2` or `IndexFlatIP` in FAISS):
    1. Every incoming query vector `Q` must be compared against **every stored vector `V_i` in the collection** (`i = 1 .. N`).
    2. For dimension `D`, computing Euclidean distance requires:
       * `D` subtractions
       * `D` squarings
       * `D - 1` additions
       * Computational cost: **~ 3 * D floating point operations (FLOPs)** per vector comparison.
    3. **Total Work**: `3 * N * D` FLOPs per query.

    ### Why Scale Breaks Exhaustive Scans
    * **10,000 Vectors (1,536-dim)**: ~46 million operations. Can run on a modern CPU in ~8-15 milliseconds.
    * **1,000,000 Vectors (1,536-dim)**: ~4.6 billion operations. Takes ~400-800 milliseconds on a multi-core CPU.
    * **50,000,000 Vectors (1,536-dim)**: ~230 billion operations. Latency exceeds 15-30 seconds per query, crashing under even 2 concurrent requests!
    """)

with col_dim2:
    st.markdown("### Flat Scan vs Indexed Traversal")
    comp_dot = """
    digraph IndexComparison {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_flat {
            label="Flat Search: O(N * D)\\n100% Vectors Checked"; style="dashed"; color="#EF4444";
            QueryF [label="Query Vector", fillcolor="#FEE2E2", color="#DC2626"];
            V1 [label="V1 (calc)", fillcolor="#F3F4F6"];
            V2 [label="V2 (calc)", fillcolor="#F3F4F6"];
            V3 [label="V3 (calc)", fillcolor="#F3F4F6"];
            Vn [label="V_10M (calc)", fillcolor="#F3F4F6"];
            QueryF -> V1;
            QueryF -> V2;
            QueryF -> V3;
            QueryF -> Vn;
        }

        subgraph cluster_indexed {
            label="Indexed Search: O(log N * D)\\n< 1% Vectors Checked"; style="dashed"; color="#10B981";
            QueryI [label="Query Vector", fillcolor="#D1FAE5", color="#059669"];
            Layer2 [label="Top Entry Node", fillcolor="#EFF6FF", color="#3B82F6"];
            Layer1 [label="Cluster Centroid", fillcolor="#FEF3C7", color="#D97706"];
            Target [label="Target Nearest Nodes\\n(Pruned Candidates)", fillcolor="#A7F3D0", color="#047857", style="filled,bold"];
            QueryI -> Layer2;
            Layer2 -> Layer1;
            Layer1 -> Target;
        }
    }
    """
    st.graphviz_chart(comp_dot, use_container_width=False)

st.divider()

# Section 2: Search With Indexing: Geometric Partitioning & Graph Routing
st.header("2. Search With Indexing: Geometric Partitioning & Graph Routing", divider="rainbow")

col_part1, col_part2 = st.columns([1, 1])

with col_part1:
    st.markdown("""
    ### How Vector Indexing Works
    Rather than inspecting every vector sequentially, a vector index partitions the high-dimensional space or builds 
    topological highway links between points.
    
    The three primary architectural paradigms are:
    
    1. **Graph-Based Indexing (e.g., HNSW)**:
       * Builds multi-layer proximity graphs where vectors are connected by edges to their nearest neighbors.
       * Top layers act as long-distance express highways (skip lists).
       * Query navigates greedily down through layers to the closest cluster in `O(log N)` hops.
    2. **Cluster-Based Indexing (e.g., IVF / Voronoi Partitioning)**:
       * Uses k-means clustering to divide vector space into `k` Voronoi cells.
       * Query only inspects vectors located inside the closest `nprobe` centroids.
    3. **Compression & Quantization (e.g., Product Quantization - PQ)**:
       * Breaks high-dimensional vectors into sub-vectors and replaces them with 1-byte codebook centroids.
       * Accelerates distance computations using precomputed lookup tables in CPU L1/L2 cache.
    """)

with col_part2:
    st.markdown("""
    ### The Fundamental Trade-Off Triangle
    Vector indexing introduces the foundational engineering trade-off of modern search systems:
    
    ```text
                 Recall (Search Accuracy)
                         /\\
                        /  \\
                       /    \\
                      /      \\
     Latency (Query Speed) --- Index Build Time & RAM
    ```
    
    * **Exact Search (Flat)**: 100% Recall, Terrible Latency, Low Memory (raw vectors only).
    * **Graph Index (HNSW)**: 95-99% Recall, Ultra-Low Latency (~2-5ms), High Memory (graph edges add 20-50% RAM).
    * **Quantized Index (IVF-PQ)**: 85-95% Recall, Low Latency, Minimal Memory (85-95% RAM reduction).
    """)

st.divider()

# Section 3: Index Construction, Updates & Storage Architecture
st.header("3. Index Construction, Updates & Storage Architecture", divider="rainbow")

col_con1, col_con2 = col_con1, col_con2 = st.columns([1, 1])

with col_con1:
    st.markdown("""
    ### Index Construction (Build Phase)
    Building an index is computationally expensive:
    * **Clustering (IVF)**: Requires training a k-means model over hundreds of thousands of sample vectors to establish cluster centroids.
    * **Graph Construction (HNSW)**: When adding vector `V`, the algorithm must search the existing graph to find its `M` nearest neighbors and insert bidirectional edges.
    * **Build Complexity**: Inserting `N` items into HNSW takes `O(N * log N * D * efConstruction)` operations.
    * **Batch vs Streaming**: Bulk ingestion is 5x-10x faster than single-item streaming insertions because graph locks and memory reallocations are amortized.
    """)

with col_con2:
    st.markdown("""
    ### Index Storage: In-Memory vs Disk Persistence
    Where does the index live?
    1. **In-Memory (RAM / CPU Cache)**:
       * The fastest approach. HNSW graph pointers and float vectors are resident in RAM.
       * ChromaDB default for fast lookups.
       * **Cost**: 10 million 1536-dim vectors require ~60-80 GB of high-speed RAM.
    2. **Memory-Mapped Files (mmap)**:
       * The index is persisted on an NVMe SSD and mapped into the operating system virtual memory address space.
       * The OS automatically pages frequently accessed nodes into RAM page cache.
       * Reduces dedicated process RAM while maintaining sub-10ms latency.
    3. **Disk-Native Storage (e.g., DiskANN)**:
       * Compresses vectors in RAM and keeps large graph adjacency lists on NVMe SSDs, reading edges via asynchronous I/O (`io_uring`).
    """)

st.divider()

# Section 4: Interactive Workbench: Search Complexity & Latency Simulator
st.header("4. Interactive Workbench: Search Complexity & Latency Simulator", divider="rainbow")

st.markdown("""
Simulate and compare computational complexity, floating-point operations (FLOPs), estimated query latency, 
and RAM footprint between **Unindexed Flat Scan** and **Indexed HNSW** across dataset sizes.
""")

col_sim_cfg, col_sim_tech = st.columns([1, 1])

with col_sim_cfg:
    st.subheader("1. Corpus Parameters")
    dataset_size = st.select_slider(
        "Total Vector Count (N):",
        options=[1_000, 10_000, 50_000, 250_000, 1_000_000, 5_000_000, 20_000_000],
        value=250_000,
        format_func=lambda x: f"{x:,} vectors"
    )
    
    vector_dims = st.selectbox(
        "Vector Dimensions (D):",
        [384, 768, 1536, 3072],
        index=2,
        help="384 (MiniLM), 768 (BGE/MPNet), 1536 (OpenAI text-embedding-3-small), 3072 (OpenAI 3-large)"
    )

with col_sim_tech:
    st.subheader("2. HNSW Index Tuning")
    hnsw_m = st.slider("Max Edges per Node (M):", min_value=8, max_value=64, value=16, step=8)
    hnsw_ef = st.slider("Search Depth (efSearch):", min_value=10, max_value=200, value=64, step=10)

# Calculations
# Flat: N * D * 3 FLOPs
flat_flops = dataset_size * vector_dims * 3
# Estimated latency: assume modern CPU core does 15 GFLOPs/sec
est_flat_ms = (flat_flops / (15 * 10**9)) * 1000

# HNSW: ~ log2(N) * M * efSearch * D * 3 FLOPs
hnsw_hops = int(np.log2(max(10, dataset_size)))
hnsw_comps = hnsw_hops * hnsw_ef
hnsw_flops = hnsw_comps * vector_dims * 3
est_hnsw_ms = max(0.5, (hnsw_flops / (15 * 10**9)) * 1000)

# Memory footprint
# Raw vectors: N * D * 4 bytes (float32)
raw_vec_bytes = dataset_size * vector_dims * 4
raw_vec_mb = raw_vec_bytes / (1024 * 1024)

# HNSW graph overhead: N * M * 8 bytes (pointers) + metadata
graph_overhead_mb = (dataset_size * hnsw_m * 8) / (1024 * 1024)
total_hnsw_mb = raw_vec_mb + graph_overhead_mb

st.divider()

col_comp_tbl, col_comp_metrics = st.columns([1.2, 0.8])

with col_comp_tbl:
    st.subheader("Head-to-Head Comparison")
    
    speedup = est_flat_ms / est_hnsw_ms
    
    comp_data = [
        {
            "Metric": "Algorithmic Complexity",
            "Unindexed Flat Scan": "O(N * D)",
            "Indexed HNSW Search": "O(log N * D * efSearch)"
        },
        {
            "Metric": "Vectors Evaluated per Query",
            "Unindexed Flat Scan": f"{dataset_size:,} (100.0%)",
            "Indexed HNSW Search": f"{hnsw_comps:,} ({((hnsw_comps / dataset_size) * 100):.3f}%)"
        },
        {
            "Metric": "Total Query FLOPs",
            "Unindexed Flat Scan": f"{flat_flops / 1e6:,.1f} MFLOPs",
            "Indexed HNSW Search": f"{hnsw_flops / 1e6:,.2f} MFLOPs"
        },
        {
            "Metric": "Estimated Query Latency (1 CPU core)",
            "Unindexed Flat Scan": f"{est_flat_ms:.2f} ms",
            "Indexed HNSW Search": f"{est_hnsw_ms:.2f} ms"
        },
        {
            "Metric": "Estimated RAM Footprint",
            "Unindexed Flat Scan": f"{raw_vec_mb:,.1f} MB (Vectors only)",
            "Indexed HNSW Search": f"{total_hnsw_mb:,.1f} MB (Vectors + Graph)"
        },
        {
            "Metric": "Recall (Accuracy)",
            "Unindexed Flat Scan": "100.0% (Exact Ground Truth)",
            "Indexed HNSW Search": "96.5% - 99.2% (Approximate)"
        }
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

with col_comp_metrics:
    st.subheader("Performance Highlights")
    st.metric(
        label="Speedup Factor",
        value=f"{speedup:,.0f}x Faster",
        delta=f"-{est_flat_ms - est_hnsw_ms:.1f} ms query time"
    )
    st.metric(
        label="Search Space Pruned",
        value=f"{100.0 - (hnsw_comps / dataset_size * 100):.2f}%",
        delta="Skipped without distance calc"
    )
    st.metric(
        label="RAM Graph Overhead",
        value=f"+{(graph_overhead_mb / raw_vec_mb * 100):.1f}% RAM",
        help="Additional memory allocated to store HNSW graph adjacency pointers."
    )

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why cannot traditional B-Tree or Hash indexes be used for high-dimensional vector search?"):
    st.markdown("""
    **Answer**:
    Traditional relational indices rely on **1-dimensional orderability**:
    1. **B-Trees** require total order (`<`, `=`, `>`). In a 1,536-dimensional space, there is no natural linear ordering of vectors (Vector A is not strictly "less than" Vector B).
    2. **Hash Tables** require exact equality matching. Even if two vectors differ by `0.00001` in a single dimension, a cryptographic hash produces completely different buckets, destroying semantic proximity.
    3. Spatial trees like R-Trees and KD-Trees suffer from the **Curse of Dimensionality**: when dimensions exceed ~10-20, spatial partitioning fails because almost all points become equidistant from the bounding box, causing the query to visit every tree branch and degrading to `O(N)` exhaustive search.
    """)

with st.expander("Q2: What is the primary difference between exact nearest neighbor search and approximate nearest neighbor (ANN) search?"):
    st.markdown("""
    **Answer**:
    * **Exact Nearest Neighbor (ENN)** guarantees 100% recall by computing distance between the query vector and every vector in the dataset. Complexity is strictly `O(N * D)`.
    * **Approximate Nearest Neighbor (ANN)** trades a tiny fraction of accuracy (typically achieving 95-99% recall) in exchange for exponential speedups (`O(log N)` or sub-linear `O(sqrt(N))`). ANN navigates graph shortcuts or cluster centroids, pruning away 99%+ of points.
    """)

with st.expander("Q3: What causes high memory consumption during HNSW index construction?"):
    st.markdown("""
    **Answer**:
    HNSW index construction requires allocating:
    1. **Raw Vector Buffer**: `N * D * 4 bytes` (float32).
    2. **Graph Edge Adjacency Lists**: Each node stores up to `M` 64-bit neighbor pointers on Layer 0, plus additional pointers across higher layers (`~N * M * 8 bytes`).
    3. **Construction State & Heaps**: During parallel insertion, each worker thread maintains a candidate priority queue of size `efConstruction`. For `M=32` and `efConstruction=200`, graph overhead typically adds **25% to 50% additional RAM** on top of raw vector storage.
    """)

with st.expander("Q4: Can an existing HNSW index be updated dynamically when new vectors arrive?"):
    st.markdown("""
    **Answer**:
    Yes, HNSW natively supports dynamic streaming insertions. 
    When a new vector `V_new` is inserted, the engine traverses the current graph to find its nearest neighbors at each layer and establishes bidirectional edges up to degree `M`. 
    However, **deletions are complex**: deleting a node leaves orphan graph edges or requires running graph repair algorithms. ChromaDB marks nodes as tombstones in SQLite/HNSW and prunes them during background compaction.
    """)

with st.expander("Q5: What is the difference between memory-mapped (mmap) index storage and in-memory storage?"):
    st.markdown("""
    **Answer**:
    * **In-Memory**: The index is loaded directly into heap RAM (`malloc`). Guarantees zero page-fault latency, but server restarts require re-reading the entire index into RAM, and dataset size is hard-capped by physical RAM.
    * **Memory-Mapped (mmap)**: The operating system maps the on-disk index file into the virtual address space. Pages are loaded lazily on demand. If system memory is tight, inactive graph pages are evicted without losing data. Allows querying indices larger than physical RAM if hosted on fast NVMe SSDs.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Deploying Unindexed Flat Scans to Production**  
    Prototypes tested on 500 documents run fast on flat scans. In production with 500,000 documents, query latency jumps from 2ms to 450ms, causing backend API gateway timeouts. Always benchmark with realistic corpus volumes.
    """)
    st.warning("""
    **Trap 2: Ignoring RAM Sizing for Graph Pointers**  
    Calculating RAM requirements based solely on `N * D * 4 bytes` ignores HNSW graph overhead. For 10M 1536-dim vectors, vectors require ~60GB, but the graph requires an extra ~20GB!
    """)

with col_trap2:
    st.error("""
    **Trap 3: Single-Item Streaming Ingestion at Scale**  
    Calling `collection.add()` one document at a time in a loop triggers index graph rebalancing and disk flush overhead for every record. Always ingest in batches of 500 to 2,000 vectors.
    """)
    st.warning("""
    **Trap 4: Over-indexing Small Datasets (< 1,000 vectors)**  
    For tiny collections (< 1,000 vectors), building complex indices adds unnecessary overhead. A raw flat matrix multiplication in NumPy or PyTorch is often faster due to SIMD cache locality.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the complexity of Flat Scan?"):
        st.markdown("""
        **Flat Scan Complexity**:  
        `O(N * D)`. Every vector must be compared against the query.
        """)
    with st.expander("🗂️ Flashcard: What is the complexity of HNSW?"):
        st.markdown("""
        **HNSW Complexity**:  
        `O(log N * D * efSearch)`. Prunes >99% of vector calculations.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why do B-Trees fail on vectors?"):
        st.markdown("""
        **No 1D Order**:  
        High-dimensional vectors have no natural scalar order (`<` or `>`), and hash tables break on floating-point distance.
        """)
    with st.expander("🗂️ Flashcard: What is the primary trade-off of ANN?"):
        st.markdown("""
        **ANN Trade-off**:  
        Sacrifices 1-3% recall (search accuracy) in exchange for 100x-1000x faster query latency.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why flat exhaustive vector search collapses at production scale.", value=True)
chk2 = st.checkbox("I know why B-Trees, Hash Tables, and KD-Trees fail for high-dimensional embeddings.", value=True)
chk3 = st.checkbox("I can explain the three primary index architectures: Graph (HNSW), Cluster (IVF), and Quantized (PQ).", value=True)
chk4 = st.checkbox("I understand the trade-off triangle between Recall, Latency, and Memory Footprint.", value=True)
chk5 = st.checkbox("I know how to calculate raw vector RAM and HNSW graph adjacency pointer overhead.", value=True)
chk6 = st.checkbox("I can select the appropriate vector indexing strategy based on corpus size and latency SLAs.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered What is an Index and are ready for Chapter 10.2: Approximate Nearest Neighbor (ANN)!")
