import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class IndexingTrilemmaSimulator:
    """
    Simulates the vector indexing trilemma: trading off Recall, Query Latency,
    Memory Footprint, and Build Time across different index paradigms.
    """
    PARADIGMS = {
        "Flat (Brute Force Exact)": {
            "recall": 100.0,
            "latency_ms": 28.5,
            "ram_mb_per_100k": 614.4,  # 100k * 1536 * 4 bytes
            "build_time_sec": 0.05,
            "scalability": "Poor (Linear degradation)",
            "complexity": "O(N * d)"
        },
        "Inverted File Index (IVF-Flat)": {
            "recall": 91.5,
            "latency_ms": 3.8,
            "ram_mb_per_100k": 625.0,  # Vectors + inverted list pointers
            "build_time_sec": 4.2,
            "scalability": "Good (Cluster partition)",
            "complexity": "O((nprobe/k_centroids) * N * d)"
        },
        "Hierarchical Navigable Small World (HNSW)": {
            "recall": 98.2,
            "latency_ms": 1.4,
            "ram_mb_per_100k": 1150.0, # Vectors + graph edge lists (1.8x)
            "build_time_sec": 18.5,
            "scalability": "High throughput, High RAM",
            "complexity": "O(log N * d)"
        },
        "Product Quantization (IVF-PQ)": {
            "recall": 84.0,
            "latency_ms": 2.1,
            "ram_mb_per_100k": 96.0,   # Compressed 8-bit codes (85% RAM reduction)
            "build_time_sec": 8.0,
            "scalability": "High density, Moderate recall",
            "complexity": "O(M * lookup)"
        }
    }

    @classmethod
    def simulate_benchmark(cls, vector_count: int, dimension: int = 1536) -> List[Dict[str, Any]]:
        results = []
        scale = vector_count / 100000.0
        
        for name, spec in cls.PARADIGMS.items():
            ram_mb = round(spec["ram_mb_per_100k"] * scale * (dimension / 1536.0), 1)
            # Scaling latency realistically
            if "Flat" in name and "IVF" not in name:
                latency = round(spec["latency_ms"] * scale, 2)
            else:
                latency = round(spec["latency_ms"] * (1.0 + 0.15 * np.log10(max(1.0, scale))), 2)

            qps = int(1000.0 / max(0.1, latency))
            build_time = round(spec["build_time_sec"] * scale, 1)

            results.append({
                "Index Strategy": name,
                "Recall (%)": spec["recall"],
                "Latency (p95)": f"{latency} ms",
                "Est. Throughput": f"{qps} QPS",
                "RAM Footprint": f"{ram_mb} MB",
                "Build Time": f"{build_time} s",
                "Complexity": spec["complexity"]
            })
        return results


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="9.1 Indexing Concepts - RAG Master Curriculum",
    page_icon="🧭",
    layout="wide"
)

st.title("9.1 Vector Indexing Concepts: Exact vs. Approximate Search, Index Construction, and the Vector Trilemma")
st.caption("Chapter 9: Vector Indexing and Search | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
In relational databases, creating an index on a scalar column (like a B-Tree on `user_id`) produces 
**zero loss in recall** while accelerating lookups from O(N) to O(log N). 

In high-dimensional vector spaces, the **Curse of Dimensionality** prevents exact sub-linear searches. 
To achieve sub-10ms query speeds across millions of dense vectors, systems engineers must abandon exact 
exhaustive scans and employ **Approximate Nearest Neighbor (ANN)** indexing.

This introduces the central engineering tradeoff of vector retrieval: **The Vector Indexing Trilemma**, 
balancing **Search Recall**, **Query Latency**, and **Memory / Hardware Cost**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: High-Dimensional Vector Indexing Paradigms", divider="rainbow")
st.markdown("""
The architecture below outlines how raw vectors are processed through index construction algorithms 
and mapped into distinct search data structures.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    index_bp_dot = """
    digraph IndexingTaxonomy {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_raw {
            label = "1. Raw Coordinate Vectors";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_vecs [label="Uncompressed Embeddings\\n(Float32 / 1536 Dimensions)", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_strategies {
            label = "2. Indexing Paradigms & Transformations";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            flat_idx [label="Flat (Exhaustive Scan)\\nZero transformation, raw array scan", fillcolor="#334155", color="#EF4444"];
            cluster_idx [label="Space Partitioning (IVF)\\nK-Means Voronoi Centroids", fillcolor="#334155", color="#F59E0B"];
            graph_idx [label="Navigable Graphs (HNSW)\\nMulti-layer skip graphs & proximity edges", fillcolor="#334155", color="#10B981"];
            quant_idx [label="Quantization (PQ / SQ)\\nVector compression into byte codes", fillcolor="#334155", color="#8B5CF6"];
        }

        subgraph cluster_search {
            label = "3. Query Execution & Candidate Selection";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_in [label="Incoming Query Vector", fillcolor="#0F172A", color="#EC4899"];
            eval_node [label="Pruned Candidate Exploration\\nGraph Walk / Centroid Probe / Lookup Table", fillcolor="#0F172A", color="#38BDF8"];
            topk_res [label="Top-K Ranked Results\\n(Recall vs. Latency Tradeoff)", fillcolor="#0F172A", color="#10B981"];

            q_in -> eval_node;
            eval_node -> topk_res;
        }

        raw_vecs -> flat_idx [label="Direct Memory Mapping"];
        raw_vecs -> cluster_idx [label="K-Means Clustering"];
        raw_vecs -> graph_idx [label="Greedy Edge Insertion"];
        raw_vecs -> quant_idx [label="Subspace Codebook Training"];

        flat_idx -> eval_node [label="100% Recall / O(N) Scan"];
        cluster_idx -> eval_node [label="Probe Closest Centroids"];
        graph_idx -> eval_node [label="Downhill Graph Traversal"];
        quant_idx -> eval_node [label="Asymmetric Distance Table"];
    }
    """
    st.graphviz_chart(index_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG pipelines, **Flat exact search** is almost never used once collections exceed 50,000 vectors. 
Engines default to **HNSW (Hierarchical Navigable Small World)** for latency-critical, high-recall workloads (e.g., Pinecone, Qdrant, Chroma). 
However, as corpora scale beyond 50 million vectors, uncompressed HNSW consumes prohibitive amounts of RAM. 
Modern enterprise deployments combine **HNSW with Scalar Quantization (SQ8)** or **IVF-PQ** to slash RAM consumption by 75%-85% 
while preserving >95% retrieval recall.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHAT IS VECTOR INDEXING?
# ==============================================================================

st.header("2. What is Vector Indexing?", divider="rainbow")
st.markdown("""
**Vector Indexing** is the computational process of constructing spatial, clustering, or graph-based data structures 
over high-dimensional embedding vectors to accelerate nearest-neighbor retrieval.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why Scalar Indexing Fails on Vectors")
    st.markdown("""
    * **The Nature of 1D Keys**:
      Scalar indexes (B-Trees) sort elements along a single dimensional axis (x1 < x2 < x3). 
      A query for `id = 1042` prunes half of the search tree at every binary branch.
    * **The Curse of High Dimensions**:
      In 1,536-dimensional space, all pairwise distances tend to converge toward a narrow distribution. 
      Points are approximately equidistant from one another in the hypersphere.
    * **Spatial Partitioning Breakdown**:
      Classical spatial partitioning trees (KD-Trees, R-Trees) break down once dimensions exceed 15-20, 
      degrading to worst-case O(N) linear scans.
    """)

with col2:
    st.subheader("The Role of Approximate Nearest Neighbors (ANN)")
    st.markdown("""
    Vector indexing solves this bottleneck by redefining the objective:
    * Instead of guaranteeing **100% exact mathematically closest neighbors**,
    * ANN algorithms find vectors that are **statistically close with high probability (95% - 99% recall)**,
    * Reducing search complexity from exhaustive O(N * d) down to sub-linear O(log N * d) or O(1).
    """)

st.divider()

# ==============================================================================
# SECTION 3: EXACT SEARCH
# ==============================================================================

st.header("3. Exact Search: Flat k-NN & Brute-Force Baselines", divider="rainbow")
st.markdown("""
**Exact Search** (also called Flat Indexing or Brute-Force k-NN) calculates the geometric distance between the query vector 
and every single stored vector in the database without skipping any candidate.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Exact Search Mechanics")
    st.markdown("""
    * **Computational Complexity**:
      O(N * d) where N is the number of documents and d is vector dimensionality.
    * **Zero Index Build Time**:
      Requires no indexing phase, no clustering centroids, and no edge graphs. Insertion is O(1).
    * **Golden Benchmark Standard**:
      Because exact search guarantees 100% recall, it serves as the ground truth benchmark against which all ANN algorithms are evaluated.
    """)

with col2:
    st.subheader("When Exact Search is Justified")
    st.markdown("""
    1. **Small Datasets (< 50,000 vectors)**:
       On modern CPUs using AVX-512 SIMD vector instructions, a linear scan across 20,000 vectors takes under 4 ms, making complex ANN indices unnecessary.
    2. **High-Accuracy Audits**:
       Legal or scientific discovery tasks where missing a single relevant precedent is unacceptable.
    3. **Highly Dynamic Streams**:
       Workloads with thousands of writes per second where building graph indices would saturate CPU capacity.
    """)

st.divider()

# ==============================================================================
# SECTION 4: APPROXIMATE NEAREST NEIGHBOR SEARCH
# ==============================================================================

st.header("4. Approximate Nearest Neighbor (ANN) Search", divider="rainbow")
st.markdown("""
**Approximate Nearest Neighbor (ANN)** algorithms trade a minute fraction of mathematical precision 
for orders-of-magnitude faster retrieval speeds.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Measuring ANN Quality: Recall@K")
    st.markdown("""
    The primary metric evaluating an ANN index is **Recall@K**:
    
    `Recall@K = (True Exact Neighbors found by ANN) / K`
    
    * If exact search returns top-5 IDs `[A, B, C, D, E]`,
    * And the ANN algorithm returns `[A, B, C, D, X]`,
    * The ANN index achieved **80% Recall@5** (4 out of 5 exact matches).
    * In production RAG systems, target Recall@K is typically **95% to 99%**.
    """)

with col2:
    st.subheader("ANN Traversal Paradigms")
    st.markdown("""
    * **Clustering & Voronoi Cells (IVF)**:
      Groups vectors into clusters. Query evaluates only points within the nearest centroids.
    * **Proximity Graphs (HNSW)**:
      Organizes vectors into multi-layer navigable networks. Traversal navigates downhill toward nearest nodes.
    * **Locality-Sensitive Hashing (LSH)**:
      Hashes vectors such that close points collide in the same hash buckets.
    * **Quantization & Codebooks (PQ)**:
      Compresses vectors into compact byte codes and uses precalculated distance lookup tables.
    """)

st.divider()

# ==============================================================================
# SECTION 5: INDEX CONSTRUCTION
# ==============================================================================

st.header("5. Index Construction & Build Latency", divider="rainbow")
st.markdown("""
Index construction is the upfront computational investment required to organize vectors into searchable structures.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Build Complexity Across Paradigms")
    st.markdown("""
    * **Flat Index**:
      Build time is **zero**. Vectors are stored in a contiguous memory array.
    * **IVF (Inverted File)**:
      Requires a **training phase** using k-means clustering to establish centroid locations (O(N * k * d * iterations)). Moderately fast build time.
    * **HNSW (Graph-Based)**:
      Requires inserting vectors one-by-one into the graph, searching for nearest neighbors at each layer, and updating bidirectional edges (O(N log N)). Highly CPU-intensive build phase.
    * **Product Quantization (PQ)**:
      Requires training subspace codebooks via k-means, followed by quantizing all dataset vectors into compressed byte arrays.
    """)

with col2:
    st.subheader("Build Time vs. Query Throughput Tradeoff")
    build_df = pd.DataFrame([
        {"Index Type", "Build Time (1M Vectors)", "RAM Overhead", "Insertion Cost"},
        {"Flat (No Index)", "Instant (< 1 sec)", "0% (Raw vectors only)", "O(1) Append"},
        {"IVF-Flat", "15 - 30 seconds", "~2% (Centroid vectors)", "Fast (Assign to cluster)"},
        {"HNSW", "2 - 5 minutes", "50% - 100% (Graph edge lists)", "Slow (Graph walk + re-link)"},
        {"IVF-PQ", "45 - 90 seconds", "-80% (Massive reduction)", "Moderate (Quantize + cluster)"}
    ])
    st.dataframe(build_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: SEARCH SPEED
# ==============================================================================

st.header("6. Search Speed & Query Throughput (QPS vs. Recall)", divider="rainbow")
st.markdown("""
In production systems, search latency is directly linked to queries-per-second (QPS) capacity and server hardware cost.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Latency Mechanics")
    st.markdown("""
    * **Memory Bound vs. Compute Bound**:
      * Flat exact search is **Compute & Memory-Bandwidth Bound**: it must stream hundreds of megabytes of coordinates from RAM into CPU registers for every query.
      * HNSW is **Pointer & Latency Bound**: it hops through graph pointers in RAM, evaluating only a tiny fraction (< 1%) of all vectors.
    * **The Speedup Factor**:
      On a 1,000,000 vector index, HNSW achieves a **20x to 100x query speedup** compared to a flat scan (e.g., 1.5 ms vs. 50 ms).
    """)

with col2:
    st.subheader("The Recall vs. Latency Curve")
    st.markdown("""
    Query speed in ANN indexes is tunable at search time:
    * In HNSW, increasing `efSearch` explores more candidate branches, increasing Recall from 92% to 99%, but doubles search latency.
    * In IVF, increasing `nprobe` searches more neighboring Voronoi centroids, approaching exact search latency as `nprobe` approaches total clusters.
    """)

st.divider()

# ==============================================================================
# SECTION 7: MEMORY REQUIREMENTS
# ==============================================================================

st.header("7. Memory Requirements & RAM Footprint: The Dimensionality Penalty", divider="rainbow")
st.markdown("""
The memory footprint of vector storage dictates the infrastructure bill of enterprise RAG deployments.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Calculating Raw Vector RAM")
    st.markdown("""
    For a collection of N vectors with dimension d stored as Float32 (4 bytes per float):
    
    `Raw RAM = N * d * 4 bytes`
    
    * **100,000 vectors** at 1536-dim: `100,000 * 1536 * 4 = 614.4 MB`
    * **1,000,000 vectors** at 1536-dim: `6.14 GB`
    * **10,000,000 vectors** at 1536-dim: **61.4 GB**
    * **100,000,000 vectors** at 1536-dim: **614.4 GB**
    """)

with col2:
    st.subheader("Index Memory Multipliers")
    st.markdown("""
    * **HNSW Overhead**:
      Each node in HNSW maintains M bidirectional links per layer. 
      For M = 32, edge lists add an extra **40% to 100% memory overhead**, 
      turning a 61.4 GB raw vector dataset into **90 GB to 120 GB of RAM**.
    * **Quantization Savings**:
      * **Scalar Quantization (SQ8)**: Compresses Float32 to Int8, reducing vector memory by **75%** (1.53 GB per 1M vectors).
      * **Product Quantization (PQ)**: Compresses 1536 floats into 64 or 128 bytes, slashing memory by **85% - 95%**.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("8. Production Implementation: Exact vs. Approximate Partition Index Benchmark Engine", divider="rainbow")
st.markdown("""
Below is an enterprise benchmarking implementation comparing exact flat scanning 
against an inverted cluster partition index to evaluate search latency and Recall@K.
""")

st.code('''import time
import numpy as np
from typing import List, Dict, Any, Tuple

class IndexBenchmarkEngine:
    """
    Benchmarks Exact Flat k-NN vs. Cluster Partitioned (IVF) search
    to measure empirical recall and query latency.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.centroids: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.inverted_lists: Dict[int, List[int]] = {}

    def fit(self, data: np.ndarray, num_clusters: int = 16) -> None:
        """Trains inverted cluster partitions using simplified k-means."""
        self.vectors = data / np.linalg.norm(data, axis=1, keepdims=True)
        N = self.vectors.shape[0]
        
        # Pick random initial centroids
        indices = np.random.choice(N, size=num_clusters, replace=False)
        self.centroids = self.vectors[indices].copy()
        
        # Assign vectors to nearest centroid
        sims = np.dot(self.vectors, self.centroids.T)
        cluster_assignments = np.argmax(sims, axis=1)
        
        self.inverted_lists = {c: [] for c in range(num_clusters)}
        for idx, cluster_id in enumerate(cluster_assignments):
            self.inverted_lists[cluster_id].append(idx)

    def search_exact(self, query_vec: np.ndarray, top_k: int = 5) -> Tuple[List[int], float]:
        """Performs exhaustive brute-force exact scan across all vectors."""
        qv = query_vec / np.linalg.norm(query_vec)
        start = time.perf_counter()
        scores = np.dot(self.vectors, qv)
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return top_indices, elapsed_ms

    def search_approximate(self, query_vec: np.ndarray, top_k: int = 5, nprobe: int = 2) -> Tuple[List[int], float]:
        """Performs partitioned search probing only the top 'nprobe' closest clusters."""
        qv = query_vec / np.linalg.norm(query_vec)
        start = time.perf_counter()
        
        # 1. Find nearest centroids
        centroid_scores = np.dot(self.centroids, qv)
        best_clusters = np.argsort(centroid_scores)[::-1][:nprobe]
        
        # 2. Score only candidate vectors in those clusters
        candidate_ids = []
        for c in best_clusters:
            candidate_ids.extend(self.inverted_lists[c])
            
        if not candidate_ids:
            return [], (time.perf_counter() - start) * 1000.0
            
        sub_vectors = self.vectors[candidate_ids]
        sub_scores = np.dot(sub_vectors, qv)
        sorted_sub = np.argsort(sub_scores)[::-1][:top_k]
        top_indices = [candidate_ids[i] for i in sorted_sub]
        
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return top_indices, elapsed_ms
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("9. Interactive Workbench: The Vector Indexing Trilemma Simulator", divider="rainbow")
st.markdown("""
Explore how dataset scale and vector dimensionality affect Recall, Latency, RAM consumption, and Build Time 
across the four fundamental vector indexing paradigms.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Corpus Scale Parameters")
    corpus_size_val = st.select_slider(
        "Corpus Size (Vectors)",
        options=[10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
        value=500000
    )

    vector_dim_val = st.selectbox(
        "Vector Dimensionality",
        [384, 768, 1536, 3072],
        index=2
    )

    st.markdown("""
    **The Vector Trilemma Tradeoff**:
    * **Flat**: 100% Recall, but latency collapses as scale increases.
    * **HNSW**: Sub-2ms latency and 98% Recall, but requires massive RAM.
    * **IVF-PQ**: Slashes RAM by 85%, but loses fine-grained distance resolution.
    """)

with col2:
    st.subheader("Comparative Benchmark Metrics")
    benchmark_data = IndexingTrilemmaSimulator.simulate_benchmark(corpus_size_val, vector_dim_val)
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    st.info(f"At {corpus_size_val:,} vectors ({vector_dim_val} dims), HNSW achieves sub-2ms query times but requires ~{float(benchmark_data[2]['RAM Footprint'].replace(' MB', '')) / 1024.0:.1f} GB of RAM.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: What is the Vector Indexing Trilemma and how does it influence production database choices?"):
    st.markdown("""
    **Answer**:
    The Vector Indexing Trilemma states that a system cannot simultaneously maximize **Search Recall**, **Query Throughput (Low Latency)**, 
    and **Memory Efficiency (Low RAM Cost)**. 
    * To achieve maximum recall and low latency, you must pay high memory overhead (e.g., uncompressed HNSW).
    * To minimize memory consumption and maintain low latency, you must sacrifice search recall (e.g., Product Quantization).
    * To guarantee 100% exact recall with low memory, you must accept high query latency (e.g., Flat brute-force scan).
    Production architectures select index parameters based on where their application sits on this tradeoff triangle.
    """)

with st.expander("Q2: Why does the Curse of Dimensionality break classical spatial indexes like KD-Trees and R-Trees?"):
    st.markdown("""
    **Answer**:
    KD-Trees and R-Trees partition metric space by hyperplanes along coordinate axes. In low dimensions (2D or 3D), 
    each plane cleanly halves the search space. In high-dimensional spaces (e.g., 1,536 dimensions), 
    the volume of the hypercube grows exponentially (2^d), and data points become distributed sparsely across the hypersphere surface. 
    Bounding boxes overlap almost completely, forcing the tree search to backtrack into virtually every branch, 
    degrading search performance to O(N) brute-force complexity with high tree-pointer overhead.
    """)

with st.expander("Q3: How is Recall@K calculated when benchmarking an Approximate Nearest Neighbor index?"):
    st.markdown("""
    **Answer**:
    Recall@K measures the fraction of true nearest neighbors returned by the ANN index relative to the ground-truth exact scan:
    `Recall@K = |ANN_Results ∩ Exact_Results| / K`.
    For example, if an exact brute-force scan returns the top-10 true closest document IDs, and the ANN index returns 9 of those 10 IDs, 
    the index achieved Recall@10 of 0.90 (90%). In production RAG systems, benchmarks require testing across thousands of representative queries.
    """)

with st.expander("Q4: When is a Flat (Brute Force) index the correct engineering choice in production?"):
    st.markdown("""
    **Answer**:
    A Flat index is the optimal choice when:
    1. Dataset size is small (< 50,000 vectors), where modern AVX-512 SIMD vector instructions can execute brute-force scans in under 4 ms.
    2. Write throughput is high with continuous insertions, where paying the CPU cost of HNSW graph updates or IVF retraining would bottleneck ingestion.
    3. Exact 100% recall is strictly required (e.g., legal or scientific audits).
    4. Hardware memory is limited, as Flat indices have 0% index overhead.
    """)

with st.expander("Q5: What is the difference between compute-bound and memory-bandwidth-bound vector searches?"):
    st.markdown("""
    **Answer**:
    * **Compute-Bound**: The bottleneck is the rate at which floating-point multiply-accumulate (FMAC) operations execute on the CPU/GPU.
    * **Memory-Bandwidth-Bound**: In exact brute-force scans on large datasets, the processor is idle waiting for gigabytes of raw vector coordinates to stream from DDR RAM across the memory bus into L3 cache. ANN indices like HNSW avoid this memory bandwidth bottleneck by traversing sparse pointer graphs, touching only a fraction of coordinates.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying Uncompressed HNSW on Cloud VMs Without Budgeting for Graph Overhead**  
    Calculating RAM solely as `N * d * 4 bytes` without factoring in HNSW edge lists causes server OOM crashes. 
    HNSW graph links add 50% to 100% extra memory overhead on top of raw vector coordinates.
    """)
    st.error("""
    **Trap 2: Benchmarking Recall@K on Synthetically Uniform Vectors**  
    Testing ANN recall on randomly generated Gaussian vectors gives misleading results. 
    Real text embeddings cluster tightly on non-linear semantic manifolds, which can alter HNSW and IVF recall characteristics significantly.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Using Flat Indexes in High-QPS Production Chatbots**  
    Running exact flat search on 200,000 vectors works fine for a single user, but under 50 concurrent chatbot queries, 
    CPU usage hits 100% and latency spikes to seconds. Always transition to ANN before scaling traffic.
    """)
    st.warning("""
    **Trap 4: Over-Optimizing for 99.9% Recall at the Expense of 5x Latency**  
    Tuning `efSearch` or `nprobe` to push Recall from 97% to 99.9% often requires a 5x increase in query latency. 
    In RAG systems, downstream LLM synthesis is resilient to minor rank differences; 95%-97% recall is usually optimal.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Vector Indexing Trilemma")
        st.caption("What are the 3 axes?")
        st.markdown("**Back**: Search Recall, Query Latency, and Memory Footprint / Hardware Cost. Optimizing any two compromises the third.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Recall@K Formula")
        st.caption("How is it measured?")
        st.markdown("**Back**: `|ANN_Results ∩ Exact_Results| / K`. Measures the proportion of true mathematical closest neighbors captured by the ANN algorithm.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Flat Index Advantage")
        st.caption("When to use it?")
        st.markdown("**Back**: Small datasets (<50k vectors) with high write throughput. Zero build time, zero memory overhead, and 100% exact recall.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: HNSW Memory Cost")
        st.caption("Why is HNSW RAM-heavy?")
        st.markdown("**Back**: Maintains multi-layer bidirectional graph edges for every vector, adding 50%-100% memory overhead over raw Float32 coordinates.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Indexing Concepts:")

c1 = st.checkbox("I understand why the Curse of Dimensionality breaks relational B-Tree and KD-Tree indexes.", value=True)
c2 = st.checkbox("I understand the Vector Indexing Trilemma: Recall vs. Latency vs. Memory Footprint.", value=True)
c3 = st.checkbox("I know how to calculate Recall@K against an exact brute-force ground truth benchmark.", value=True)
c4 = st.checkbox("I can calculate the exact RAM footprint for Float32 vectors and estimate HNSW graph overhead.", value=True)
c5 = st.checkbox("I know when Flat exact search is appropriate versus when ANN indexes are mandatory.", value=True)
c6 = st.checkbox("I can implement a benchmark engine in Python comparing exact vs. partitioned nearest neighbor retrieval.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Indexing Concepts and understand the core tradeoffs governing high-dimensional search engines.")
