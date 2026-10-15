import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.3 Feature Comparison - Vector DB Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("4.3 Comprehensive Feature Comparison: Relational, Search Engine, Document & Vector DBs")
st.caption("Chapter 4: Vector Database vs Traditional Databases | Domain: 4. Vector Database")
st.markdown("""
Modern software architectures rarely rely on a single database. Enterprise stacks employ specialized databases 
tuned for distinct access patterns: **Relational Databases (PostgreSQL, MySQL)** for transactions, 
**Inverted Search Engines (Elasticsearch, OpenSearch)** for text search, **Document Stores (MongoDB)** for flexible JSON trees, 
and **Vector Databases (Qdrant, Milvus, Pinecone)** for semantic AI retrieval.

In this lesson, we conduct an exhaustive architectural feature comparison across the 6 core pillars of database systems:
1. **Data Model & Schemas**
2. **Indexing Algorithms**
3. **Query Languages & APIs**
4. **Query Execution Pipelines**
5. **Horizontal & Vertical Scalability**
6. **Latency SLAs & Hardware Utilization**
""")

st.divider()

# Section 1: The Master 6-Dimensional Feature Matrix
st.header("1. Master Architectural Comparison Across Database Classes", divider="rainbow")

st.markdown("""
The table below contrasts the 4 primary database paradigms powering modern software:
""")

comp_matrix_df = pd.DataFrame({
    "System Pillar": [
        "1. Data Model",
        "2. Primary Index",
        "3. Query Language",
        "4. Execution Engine",
        "5. Scalability Model",
        "6. Typical Latency SLA"
    ],
    "Relational RDBMS (PostgreSQL)": [
        "Rigid tables, rows, columns, foreign keys, ACID schemas",
        "B+ Trees, B-Trees, Hash, GiST/GIN",
        "Declarative SQL (`SELECT ... JOIN`)",
        "Cost-Based Optimizer, Nested Loops, Hash Joins, MVCC",
        "Primarily vertical; read replicas; complex sharding",
        "< 1 ms (PK point lookup) / 10-50 ms (Complex JOINs)"
    ],
    "Search Engine (Elasticsearch)": [
        "Semi-structured JSON documents with typed mappings",
        "Inverted Index, FST, BKD Trees (Lucene)",
        "Lucene syntax / Elasticsearch Query DSL (JSON)",
        "Token dictionary lookup, BM25 scoring, WAND pruning",
        "Horizontal shard distribution with replicas",
        "10 - 40 ms (Full-text keyword queries)"
    ],
    "Document DB (MongoDB)": [
        "Hierarchical BSON documents with dynamic fields",
        "B-Trees on document fields, compound indexes",
        "MQL (MongoDB Query Language) / Aggregation Pipelines",
        "B-Tree traversal, document heap scans, oplog replication",
        "Horizontal sharding via shard keys and mongos routers",
        "1 - 5 ms (Indexed document queries)"
    ],
    "Vector Database (Qdrant / Milvus)": [
        "High-dimensional dense vectors + attached JSON payloads",
        "HNSW Graphs, IVF-PQ Centroids, ScaNN, DiskANN",
        "Native Python/REST/gRPC SDKs, Vector SQL (`<=>`)",
        "SIMD AVX-512 distance kernels, Bitset filter masking, Top-K heap",
        "Cloud-native decoupled storage/compute, segment sharding",
        "2 - 8 ms (Million-vector ANN retrieval)"
    ]
})
st.dataframe(comp_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Deep Dive into Indexing, Query Languages & Execution
st.header("2. Architectural Deep Dive: Indexing, Language & Execution", divider="rainbow")

col_dd1, col_dd2 = col_dd1, col_dd2 = st.columns([1, 1])

with col_dd1:
    st.markdown("""
    ### 1. Indexing Architectures Compared
    * **B+ Tree (RDBMS)**:
      * Hierarchical page trees ordering scalar values along a 1D axis.
      * Delivers deterministic `O(log N)` point lookups and sequential range scans. Completely collapses on high-dimensional vectors.
    * **Inverted Index (Search Engines)**:
      * Maps discrete tokens to sorted arrays of matching document IDs (`term -> [doc_3, doc_12, doc_99]`).
      * Optimized for token intersection and BM25 term frequency math.
    * **HNSW & IVF-PQ (Vector Databases)**:
      * **HNSW**: Multi-layer proximity graph routing queries through hierarchical highway links down to dense local neighborhoods.
      * **IVF-PQ**: Clusters space into Voronoi cells and compresses vectors into 8-bit quantized centroid codes.
    """)

with col_dd2:
    st.markdown("""
    ### 2. Query Languages & Execution Pipelines
    * **SQL vs. Vector SDKs**:
      * *SQL*: Declarative: you state *what* you want (`SELECT * WHERE balance > 100`), the query planner decides *how* to execute it.
      * *Vector Search*: Imperative/Functional: client sends query vector `q`, target collection, `limit=k`, and optional metadata filters.
    * **Hardware Execution Engines**:
      * *RDBMS*: CPU-bound, memory-page cache limited, heavily optimized for disk page buffer management (8KB/16KB blocks).
      * *Vector DB*: Heavily accelerated by hardware vector extensions: **SIMD AVX-512**, **ARM NEON**, and **NVIDIA CUDA Tensor Cores** calculating billions of dot products per second.
    """)

st.markdown("### Execution Pipeline Architecture Diagram")
pipe_dot = """
digraph PipelineComparison {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    subgraph cluster_rdbms {
        label="RDBMS Execution Pipeline"; style="dashed"; color="#2563EB";
        SQLIn [label="SQL String", fillcolor="#EFF6FF"];
        CBO [label="Cost-Based Optimizer", fillcolor="#DBEAFE"];
        BTreeScan [label="B+ Tree Page Traversal", fillcolor="#BFDBFE"];
        BufferHit [label="Buffer Pool (8KB Pages)", fillcolor="#93C5FD"];
        SQLOut [label="Exact Tuples", fillcolor="#60A5FA", color="#2563EB", style="filled,bold"];
        SQLIn -> CBO -> BTreeScan -> BufferHit -> SQLOut;
    }

    subgraph cluster_vdb {
        label="Vector DB Execution Pipeline"; style="dashed"; color="#059669";
        VectorIn [label="Query Vector q", fillcolor="#ECFDF5"];
        PlanFilter [label="Bitset Pre-Filter Plan", fillcolor="#D1FAE5"];
        HNSWWalk [label="HNSW Graph Hop Router", fillcolor="#A7F3D0"];
        SIMDKernel [label="SIMD AVX-512 FMA Kernel", fillcolor="#6EE7B7"];
        VDBOut [label="Top-K Hydrated Payloads", fillcolor="#34D399", color="#059669", style="filled,bold"];
        VectorIn -> PlanFilter -> HNSWWalk -> SIMDKernel -> VDBOut;
    }
}
"""
st.graphviz_chart(pipe_dot, use_container_width=False)

st.divider()

# Section 3: Scalability Models & Latency SLAs
st.header("3. Scalability Models & Latency Profiles", divider="rainbow")

col_scale1, col_scale2 = col_scale1, col_scale2 = st.columns([1, 1])

with col_scale1:
    st.markdown("""
    ### Scalability Paradigms
    * **Relational Databases (Vertical Heavy)**:
      * Designed when hardware was single-server. Scaling writes typically requires upgrading CPU/RAM (vertical scale).
      * Sharding across multiple nodes breaks cross-shard foreign keys and ACID transactions, requiring complex orchestration (Vitess, Citus).
    * **Vector Databases (Cloud-Native & Disaggregated)**:
      * Built for modern Kubernetes environments with decoupled storage and compute:
        * **Stateless Query Nodes**: Scale horizontally to handle high QPS.
        * **Index Nodes**: Heavy CPU/GPU nodes building HNSW graphs in the background.
        * **Storage Nodes**: Columnar segment storage with persistence to cloud object storage (Amazon S3 / Google Cloud Storage).
    """)

with col_scale2:
    st.markdown("""
    ### Latency SLA Profiles Across Workloads
    * **RDBMS Point Lookup (`WHERE id = 42`)**:
      * **0.2 to 0.8 ms** (sub-millisecond in-memory B-Tree hit).
    * **Vector DB ANN Top-10 Retrieval**:
      * **2.0 to 6.0 ms** (sub-10ms across 10 million vectors with HNSW).
    * **Elasticsearch BM25 Search**:
      * **15.0 to 35.0 ms** (inverted list scoring and merging).
    * **Multi-Table Relational 4-Way JOIN**:
      * **25.0 to 120.0 ms** (multi-table hash join, disk spillover if working memory exceeded).
    """)

st.divider()

# Section 4: Interactive Feature Matrix & Workload Benchmark Profiler
st.header("4. Interactive Workload & Architecture Profiler", divider="rainbow")

st.markdown("""
Select your enterprise workload to inspect which database technology dominates each system pillar, 
along with estimated latency SLAs and infrastructure recommendations:
""")

workload_choice = st.selectbox(
    "Choose Enterprise Workload Scenario",
    options=[
        "Scenario A: GenAI RAG Application (10M doc chunks, semantic QA, LLM context)",
        "Scenario B: Core Financial Banking Ledger (Double-entry accounting, strict ACID)",
        "Scenario C: E-Commerce Product Catalog (Filters by price/brand + visual/semantic search)",
        "Scenario D: High-Volume Log Analytics (500k log events/sec, full-text debugging)"
    ]
)

col_prof1, col_prof2 = col_prof1, col_prof2 = st.columns([1, 1])

with col_prof1:
    st.subheader("Workload Characteristics")
    if "Scenario A: GenAI" in workload_choice:
        st.info("""
        * **Data Nature**: Unstructured text chunks, code, PDFs.
        * **Query Type**: Semantic similarity (`"How do I fix pod eviction?"`).
        * **Primary SLA**: Sub-10ms Top-K retrieval.
        * **Optimal Solution**: **Dedicated Vector Database (Qdrant / Milvus)**.
        """)
    elif "Scenario B: Core Financial" in workload_choice:
        st.info("""
        * **Data Nature**: Structured accounts, balances, transactions.
        * **Query Type**: Point updates, balance transfers, reconciliation.
        * **Primary SLA**: Strict ACID Serializability, Zero data loss.
        * **Optimal Solution**: **Relational Database (PostgreSQL / CockroachDB)**.
        """)
    elif "Scenario C: E-Commerce" in workload_choice:
        st.info("""
        * **Data Nature**: Structured SKU metadata + Unstructured product images & reviews.
        * **Query Type**: Hybrid (Scalar filters `price < 50` + Vector similarity).
        * **Primary SLA**: Sub-15ms filtered retrieval.
        * **Optimal Solution**: **Hybrid Vector DB or PostgreSQL with `pgvector`**.
        """)
    else:
        st.info("""
        * **Data Nature**: Semi-structured log events, timestamps, stack traces.
        * **Query Type**: Full-text substring search, error code matching, aggregations.
        * **Primary SLA**: High write throughput, fast inverted index aggregation.
        * **Optimal Solution**: **Search Engine (Elasticsearch / OpenSearch / ClickHouse)**.
        """)

with col_prof2:
    st.subheader("Architectural Fitness Scorecard")
    if "Scenario A: GenAI" in workload_choice:
        scores = {"Vector DB (Qdrant/Milvus)": 98, "Elasticsearch": 72, "PostgreSQL (pgvector)": 84, "Standard RDBMS": 20}
    elif "Scenario B: Core Financial" in workload_choice:
        scores = {"Standard RDBMS (PostgreSQL)": 99, "Document DB (MongoDB)": 65, "Elasticsearch": 30, "Vector DB": 10}
    elif "Scenario C: E-Commerce" in workload_choice:
        scores = {"Hybrid Vector DB": 92, "PostgreSQL (pgvector)": 89, "Elasticsearch": 80, "Document DB": 70}
    else:
        scores = {"Elasticsearch / OpenSearch": 96, "ClickHouse": 94, "PostgreSQL": 55, "Vector DB": 40}
        
    for name, sc in scores.items():
        st.markdown(f"**{name}** — Score: `{sc}/100`")
        st.progress(sc / 100)

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Multi-Engine Query Latency Benchmark Simulator", divider="rainbow")

st.markdown("""
The following complete Python program benchmarks the computational mechanics of 
**B-Tree Key Lookup**, **Inverted Keyword Match**, and **Vector SIMD Dot Product Search**:
""")

code_bench = '''import numpy as np
import time
import bisect

class MultiEngineBenchmark:
    """Benchmarks the operational mechanics of RDBMS B-Tree, Inverted Search, and Vector SIMD."""
    def __init__(self, n_items: int = 100000, dim: int = 128):
        self.n_items = n_items
        self.dim = dim
        
        # 1. RDBMS B-Tree Index (Sorted 1D keys)
        self.b_tree_keys = list(range(n_items))
        
        # 2. Inverted Index (term -> matching doc ids)
        self.inverted_index = {
            "error": list(range(0, n_items, 5)), # 20,000 matches
            "database": list(range(0, n_items, 10)) # 10,000 matches
        }
        
        # 3. Vector DB Matrix (N, d)
        np.random.seed(42)
        raw_mat = np.random.randn(n_items, dim).astype(np.float32)
        norms = np.linalg.norm(raw_mat, axis=1, keepdims=True)
        self.vector_matrix = raw_mat / norms

    def bench_btree_lookup(self, target_id: int) -> float:
        """Simulates RDBMS B+ Tree O(log N) point lookup."""
        t0 = time.perf_counter()
        idx = bisect.bisect_left(self.b_tree_keys, target_id)
        found = self.b_tree_keys[idx] if idx < len(self.b_tree_keys) else None
        return (time.perf_counter() - t0) * 1000.0 # ms

    def bench_inverted_intersection(self, term1: str, term2: str) -> float:
        """Simulates Search Engine Inverted Index posting list intersection."""
        t0 = time.perf_counter()
        set_a = set(self.inverted_index[term1])
        set_b = set(self.inverted_index[term2])
        common = set_a & set_b
        return (time.perf_counter() - t0) * 1000.0 # ms

    def bench_vector_simd_scan(self, query_vec: np.ndarray, top_k: int = 5) -> float:
        """Simulates Vector DB SIMD Dot Product Scan (100k items)."""
        t0 = time.perf_counter()
        scores = np.dot(self.vector_matrix, query_vec) # SIMD dot product
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        return (time.perf_counter() - t0) * 1000.0 # ms

# Execute Benchmark
runner = MultiEngineBenchmark(n_items=50000, dim=64)
q_vec = np.random.randn(64).astype(np.float32)
q_vec /= np.linalg.norm(q_vec)

lat_btree = runner.bench_btree_lookup(target_id=24150)
lat_inverted = runner.bench_inverted_intersection("error", "database")
lat_vector = runner.bench_vector_simd_scan(q_vec, top_k=5)

print(f"1. RDBMS B-Tree Point Lookup Latency: {lat_btree:.4f} ms")
print(f"2. Search Inverted Posting Merge Latency: {lat_inverted:.4f} ms")
print(f"3. Vector Matrix Dot Product Scan Latency: {lat_vector:.4f} ms")
'''

st.code(code_bench, language="python")

if st.button("🚀 Execute Engine Benchmark Simulator", type="primary"):
    exec_scope = {}
    exec(code_bench, globals(), exec_scope)
    
    l_b = exec_scope["lat_btree"]
    l_i = exec_scope["lat_inverted"]
    l_v = exec_scope["lat_vector"]
    
    st.success("Benchmark completed successfully!")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("RDBMS B-Tree Point Lookup", f"{l_b:.4f} ms", "O(log N) In-Memory")
    with c2:
        st.metric("Search Inverted Posting Merge", f"{l_i:.4f} ms", "Set Intersection")
    with c3:
        st.metric("Vector Matrix Dot Product (50k)", f"{l_v:.4f} ms", "SIMD AVX Matrix-Vector")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. In enterprise system design, why wouldn't you simply use a relational database with `pgvector` for all vector search workloads?"""
a1 = """* **When `pgvector` is Ideal**:
  * You already use PostgreSQL.
  * Your collection size is under 5-10 million vectors.
  * You need strict ACID transactions and relational JOINs between business tables and embeddings in a single database.
* **When a Dedicated Vector DB (Qdrant, Milvus) is Required**:
  * **Scale (> 20M-100M+ vectors)**: HNSW graphs require immense RAM (e.g. 50-150 GB). PostgreSQL's shared buffer pool and VACUUM garbage collection are not engineered for continuous in-memory graph mutations.
  * **Decoupled Architecture**: In high-load RAG systems, vector search queries consume 100% of CPU SIMD registers. Running this on your primary transactional database degrades business OLTP transactions.
  * **Advanced Filtering**: Dedicated vector engines feature single-stage bitset graph filtering and distributed horizontal sharding out of the box."""

q2 = """2. How does the indexing structure of an Inverted Index (Lucene) contrast with an HNSW graph?"""
a2 = """* **Inverted Index (Lucene / Elasticsearch)**:
  * Maps discrete tokens to sorted arrays of matching document IDs (`token -> [id1, id2, id3]`).
  * Optimized for discrete boolean intersections (`A AND B`) and BM25 term frequency math.
  * Incapable of semantic generalization: if the query token is missing from the dictionary, score is 0.0.
* **HNSW (Vector DB)**:
  * A multi-layer geometric proximity graph where nodes represent vectors and edges connect spatial neighbors.
  * Upper layers have sparse highway links; lower layers have dense local links.
  * Navigates continuous coordinate space in `O(log N)` time based on geometric angle rather than token dictionaries."""

q3 = """3. Why is horizontal scaling fundamentally more challenging in an RDBMS than in a Vector Database?"""
a3 = """* **RDBMS Complexity**:
  * Enforcing relational foreign keys, unique constraints, and ACID transactions across multiple network nodes requires distributed commit protocols (Two-Phase Commit - 2PC) and distributed locking.
  * Multi-table JOINs across distributed shards force expensive network data shuffling, bottlenecking throughput.
* **Vector DB Simplicity**:
  * Vector databases do not have foreign key constraints or multi-table relational joins.
  * Data is partitioned into self-contained immutable segments (each with its own independent HNSW index).
  * Sharded searches operate via a simple **scatter-gather** pattern: the query is dispatched to all shards in parallel, each shard returns its local Top-K, and the coordinator merges them into the global Top-K."""

q4 = """4. What are the query latency expectations across RDBMS point lookups, Elasticsearch text searches, and Vector DB ANN retrievals?"""
a4 = """* **RDBMS Point Lookup (`WHERE id = 42`)**: Sub-millisecond (0.1 - 0.8 ms). B-Tree depth is typically 3-4 levels; leaf page is immediately fetched from buffer cache.
* **Vector DB ANN Retrieval (Top-10)**: Single-digit milliseconds (2.0 - 8.0 ms). Traverses ~500 graph nodes using SIMD hardware dot products.
* **Elasticsearch BM25 Search**: Moderate latency (10 - 35 ms). Decodes posting lists, calculates BM25 term frequencies, and aggregates results across cluster shards."""

q5 = """5. How does memory management differ between traditional databases (Buffer Pool) and Vector Databases?"""
a5 = """* **RDBMS (Buffer Pool)**:
  * Divides memory into fixed-size pages (e.g. 8KB). Pages are loaded from disk on demand and evicted via LRU policies.
  * Data does not need to fit entirely in RAM to be queried.
* **Vector Databases (In-Memory Graphs)**:
  * Fast HNSW graph traversal requires bidirectional graph edge pointer jumps in nanoseconds.
  * Swapping HNSW graph nodes to and from disk causes catastrophic thrashing, increasing query latency by 50x-100x!
  * Consequently, vector databases require that the HNSW index reside directly in RAM (or use specialized memory-mapped SSD structures like DiskANN)."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Feature Comparison Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The "PostgreSQL Can Do Everything" Resource Starvation Trap**  
> Running high-traffic RAG vector queries (`pgvector`) on the exact same PostgreSQL database instance that 
> processes user checkouts and billing transactions causes CPU spikes that stall transactional connection pools. 
> Keep high-QPS vector workloads on a dedicated read-replica or a standalone vector database.

> [!WARNING]
> **Trap 2: Ignoring RAM Sizing for HNSW Indexes**  
> Because relational databases run fine with 16 GB of RAM on 200 GB tables (using page swapping), developers 
> assume vector databases can do the same. When an HNSW graph exceeds physical RAM, disk thrashing slows queries 
> from 4ms to 600ms! Size your server RAM to hold all vector arrays plus 30-50% graph index overhead.

> [!WARNING]
> **Trap 3: Using Elasticsearch for Pure Dense Semantic Search at Scale**  
> While Elasticsearch supports dense vectors, its underlying JVM heap management, garbage collection pauses, 
> and Lucene segment merges are less memory-efficient for dense vectors than specialized native C++/Rust vector engines 
> (Milvus, Qdrant). Use Elasticsearch when BM25 keyword matching is your primary driver.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is the Main Index in an RDBMS vs Vector DB?"):
        st.markdown("""
        **Index Contrast**:  
        RDBMS uses B+ Trees for 1D ordered scalars; Vector DB uses HNSW proximity graphs and IVF-PQ clusters for high-dimensional geometric spaces.
        """)
    with st.expander("🗂️ Flashcard: Why is Vector DB Horizontal Scaling Easier?"):
        st.markdown("""
        **Horizontal Scaling**:  
        Vector DBs lack cross-table relational foreign keys and JOINs, allowing straightforward scatter-gather query execution across sharded segments.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What Hardware Accelerates Vector DBs?"):
        st.markdown("""
        **Hardware SIMD Acceleration**:  
        AVX-512, ARM NEON, and GPU CUDA Tensor Cores execute fused multiply-accumulate (FMA) dot products in single hardware clock cycles.
        """)
    with st.expander("🗂️ Flashcard: When Use `pgvector` vs Dedicated Vector DB?"):
        st.markdown("""
        **Architecture Boundary**:  
        Use `pgvector` for < 10M vectors with tight relational joins; use dedicated Vector DB for > 20M vectors, ultra-high QPS, or decoupled GenAI pipelines.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can contrast the data models of RDBMS, Search Engines, Document DBs, and Vector DBs.", value=True)
chk2 = st.checkbox("I understand why HNSW proximity graphs replace B+ Trees for vector search.")
chk3 = st.checkbox("I know why vector databases scale horizontally via scatter-gather without foreign key locks.")
chk4 = st.checkbox("I understand the typical latency SLA profiles across database paradigms.")
chk5 = st.checkbox("I can articulate the exact architectural trade-offs between `pgvector` and dedicated vector engines.")
chk6 = st.checkbox("I know why HNSW indexes require dedicated RAM residency to prevent disk thrashing.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered the Feature Comparison and are ready for Chapter 4.4: When to Use Which!")
