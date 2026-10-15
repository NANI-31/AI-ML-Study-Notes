import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="22.3 Search Optimization - Embeddings Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("22.3 Vector Search Optimization: Hyperparameter Tuning, Pareto Curves, & Distributed Sharding")
st.caption("Chapter 22: Advanced Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
Deploying a vector database with default out-of-the-box parameters often results in either:
* **Over-provisioning**: Burning excessive RAM and CPU compute for marginal 0.1% recall gains.
* **Under-provisioning**: Suffering from unacceptable 80% recall drops that ruin RAG applications.

To achieve production stability, search engineers must master **Vector Search Parameter Optimization**:
* **HNSW Parameters**: `M` (edge density), `efConstruction` (build quality), and `efSearch` (runtime query budget).
* **IVF Parameters**: `nlist` (Voronoi centroid count) and `nprobe` (runtime cells probed).
* **Runtime Dynamic Tuning**: Adjusting `efSearch` on a per-query basis without rebuilding the index (e.g. low budget for fast autocomplete, high budget for deep legal compliance research).
* **Large-Scale Distributed Sharding**: Horizontally partitioning billions of vectors across multi-node clusters using MapReduce scatter-gather query aggregation.

This lesson explores how to systematically calibrate index parameters to optimize queries per second (QPS) and recall.
""")

st.divider()

# Section 1: Parameter Tuning Mechanics & Distributed Sharding
st.header("1. HNSW & IVF Parameter Mechanics & Distributed Sharding", divider="rainbow")

col_opt1, col_opt2 = col_opt1, col_opt2 = st.columns([1, 1])

with col_opt1:
    st.markdown("""
    ### The Critical Hyperparameters
    * **1. `M` (HNSW Max Connection Edges per Node)**:
      * Controls graph connectivity (typically 16, 32, or 64).
      * Higher $M$ improves recall and graph traversal stability on high-dimensional data, but increases RAM footprint linearly and slows down index construction.
    * **2. `efConstruction` (HNSW Build Exploration Depth)**:
      * The size of the dynamic candidate list evaluated while building the graph (typically 64 to 200).
      * Has **zero impact on runtime query speed**; only affects index build duration and graph quality.
    * **3. `efSearch` (HNSW Runtime Query Budget)**:
      * The size of the dynamic priority queue evaluated during live user search (typically 32 to 128).
      * **Fully tunable at query time!** Allows client applications to dynamically choose between speed and accuracy.
    * **4. `nlist` & `nprobe` (IVF Centroids & Probes)**:
      * `nlist`: Total K-Means centroids (rule of thumb: $4 \\times \\sqrt{N}$).
      * `nprobe`: Number of nearest Voronoi cells searched per query (typically 8 to 32).
    """)

with col_opt2:
    st.markdown("### Distributed Sharded Cluster Architecture")
    shard_dot = """
    digraph DistributedSearch {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="Incoming User Query\\n'Enterprise Cloud Compliance'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        QRouter [label="Distributed Query Router / Coordinator\\n(Scatter-Gather Engine)", fillcolor="#FEF08A", style="filled,bold"];

        subgraph cluster_cluster {
            label="Horizontal Index Shards"; style="dashed"; color="#8B5CF6";
            S1 [label="Shard Node 1\\n(HNSW 3.3M Vectors)", fillcolor="#EDE9FE"];
            S2 [label="Shard Node 2\\n(HNSW 3.3M Vectors)", fillcolor="#EDE9FE"];
            S3 [label="Shard Node 3\\n(HNSW 3.3M Vectors)", fillcolor="#EDE9FE"];
        }

        Merger [label="Top-K Aggregator & Deduplication\\nGlobal Top-K Reranking", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserQ -> QRouter;
        QRouter -> S1 [label="Scatter query"];
        QRouter -> S2 [label="Scatter query"];
        QRouter -> S3 [label="Scatter query"];
        S1 -> Merger [label="Local Top-10"];
        S2 -> Merger [label="Local Top-10"];
        S3 -> Merger [label="Local Top-10"];
    }
    """
    st.graphviz_chart(shard_dot, use_container_width=False)

st.markdown("### Parameter Tuning Reference Guide")
st.markdown("""
| Parameter | Index Family | Tunable at Runtime? | Recommended Default | Increasing Value Causes | Decreasing Value Causes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`M`** | HNSW | **No (Build time)** | 16 or 32 | Higher recall; +50% RAM; 2x slower build. | Lower RAM; potential graph disconnection. |
| **`efConstruction`**| HNSW | **No (Build time)** | 128 | Higher quality graph; slower index build. | Faster build; lower baseline graph recall. |
| **`efSearch`** | HNSW | **YES (Per Query!)** | 64 | **Higher Recall (+1-3%); higher query latency.** | **Lower Latency (< 2ms); lower recall.** |
| **`nlist`** | IVF | **No (Build time)** | $4 \\times \\sqrt{N}$ | Smaller partitions; faster search per cell. | Giant partitions; slower linear scans per cell. |
| **`nprobe`** | IVF | **YES (Per Query!)** | 16 | **Higher Recall; linear latency increase.** | **Lower Latency; risk of missing neighbors.** |
""")

st.divider()

# Section 2: Production Code: Automated Parameter Sweep Engine
st.header("2. Production Code: Parameter Grid Search & Pareto Tuner", divider="rainbow")

st.markdown("""
The production utility below demonstrates an automated grid tuner that benchmarks an index over varying `efSearch` values, recording QPS, latency, and Recall@10 to discover the optimal Pareto operating point.
""")

st.code(r'''
import time
import numpy as np
from typing import List, Dict, Any, Tuple

class HNSWParameterTuner:
    """
    Automates empirical grid searches over runtime search parameters
    to discover the optimal Recall vs. Latency Pareto curve.
    """
    def __init__(self, ground_truth_top_k: List[List[int]]):
        self.ground_truth = ground_truth_top_k

    def simulate_query_execution(
        self,
        ef_search: int,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Simulates HNSW search with a specific efSearch parameter.
        Calculates empirical recall against ground-truth and latency.
        """
        t0 = time.perf_counter()
        
        # In real benchmark: runs index.search(queries, k, efSearch=ef_search)
        # Latency model: base graph hops + ef_search candidate queue evaluations
        simulated_latency_ms = 0.45 + (ef_search * 0.035)
        
        # Recall model: asymptotic curve approaching 99.8%
        simulated_recall = min(99.8, (1.0 - np.exp(-ef_search / 22.0)) * 100.0)
        
        # Calculate Queries Per Second (QPS)
        qps = 1000.0 / max(0.01, simulated_latency_ms)

        return {
            "ef_search": ef_search,
            "latency_ms": simulated_latency_ms,
            "qps": qps,
            "recall_at_10": simulated_recall
        }

    def run_pareto_sweep(self, ef_search_candidates: List[int]) -> List[Dict[str, float]]:
        results = []
        for ef in ef_search_candidates:
            metrics = self.simulate_query_execution(ef_search=ef)
            results.append(metrics)
        return results
''', language="python")

st.divider()

# Section 3: Interactive HNSW & IVF Parameter Studio
st.header("3. Interactive Workbench: Live Hyperparameter Tuning Studio", divider="rainbow")

st.markdown("""
Adjust the HNSW and IVF tuning parameters interactively.
Observe the immediate impact on **RAM consumption**, **Index Build Duration**, **QPS Throughput**, and **Recall@10**.
""")

col_tune1, col_tune2 = col_tune1, col_tune2 = st.columns([1, 1])

with col_tune1:
    st.subheader("1. HNSW Hyperparameters")
    param_m = st.select_slider(
        "Max Bidirectional Edges (`M`):",
        options=[8, 16, 32, 64],
        value=32,
        help="Higher M increases graph connectivity and recall, but consumes more RAM."
    )

    param_ef_construction = st.slider(
        "Build Candidate Pool (`efConstruction`):",
        min_value=32,
        max_value=256,
        value=128,
        step=32,
        help="Higher values improve graph quality during indexing at the cost of build time."
    )

    param_ef_search = st.slider(
        "Runtime Search Budget (`efSearch`):",
        min_value=16,
        max_value=256,
        value=64,
        step=16,
        help="Tunable dynamically per query. Higher values increase recall at the cost of latency."
    )

    st.subheader("2. Dataset Baseline Scale")
    scale_dataset = st.selectbox(
        "Benchmark Dataset Scale:",
        ["100,000 Vectors (Prototype)", "1,000,000 Vectors (Production)", "10,000,000 Vectors (Heavy Enterprise)"],
        index=1
    )
    n_vecs = 100000 if "100,000" in scale_dataset else (1000000 if "1,000,000" in scale_dataset else 10000000)

with col_tune2:
    st.subheader("2. Real-Time Telemetry & Pareto Zone")

    # Analytical physics model
    # RAM: raw float32 (1536 dim) = 6144 bytes. Graph edges = M * 4 bytes * 2 directions
    graph_edge_bytes_per_vec = param_m * 8 * 1.5 # ~1.5 layers
    total_bytes_per_vec = 6144 + graph_edge_bytes_per_vec
    total_index_ram_gb = (n_vecs * total_bytes_per_vec) / (1024.0**3)

    # Build time: scales with N * efConstruction * log(N)
    build_seconds = (n_vecs * param_ef_construction * np.log2(n_vecs)) / 8000000.0
    build_time_str = f"{build_seconds / 60.0:.1f} min" if build_seconds >= 60 else f"{build_seconds:.1f} sec"

    # Query latency: scales with log(N) + efSearch
    query_latency_ms = 0.50 + (np.log2(n_vecs) * 0.05) + (param_ef_search * 0.025)
    qps = 1000.0 / query_latency_ms

    # Empirical recall model
    recall_est = min(99.9, (1.0 - np.exp(-param_ef_search / (param_m * 0.85))) * 100.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Query Latency", f"{query_latency_ms:.2f} ms")
    m2.metric("Throughput (QPS)", f"{qps:.0f} req/s")
    m3.metric("Recall@10", f"{recall_est:.1f}%")

    m4, m5 = st.columns(2)
    m4.metric("Total RAM Required", f"{total_index_ram_gb:.2f} GB")
    m5.metric("Index Build Time", build_time_str)

    # Generate Pareto sweep curve for this M setting
    ef_vals = [16, 32, 64, 96, 128, 192, 256]
    lats = [0.50 + (np.log2(n_vecs) * 0.05) + (e * 0.025) for e in ef_vals]
    qps_vals = [1000.0 / l for l in lats]
    rec_vals = [min(99.9, (1.0 - np.exp(-e / (param_m * 0.85))) * 100.0) for e in ef_vals]

    df_sweep = pd.DataFrame({
        "efSearch": ef_vals,
        "Latency (ms)": [round(x, 2) for x in lats],
        "QPS": [int(x) for x in qps_vals],
        "Recall@10 (%)": [round(x, 1) for x in rec_vals]
    })
    
    st.markdown("#### Parameter Sweep Table (`efSearch` Curve)")
    st.dataframe(df_sweep, use_container_width=True)

    if recall_est >= 98.0:
        st.success(f"🌟 **Optimal Pareto Tier**: Delivering **{recall_est:.1f}% Recall** at **{qps:.0f} QPS** per node!")
    else:
        st.warning(f"⚠️ Recall is {recall_est:.1f}%. Consider increasing `efSearch` to 64 or 96 for enterprise accuracy.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Tuning Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is `efSearch` dynamically tunable per query in HNSW, while `M` and `efConstruction` are immutable?**
    * **Answer**:
      * `M` and `efConstruction` dictate the **physical topology of the graph**: the maximum number of edges per node and how thoroughly edges were wired during insertion. Modifying them requires physically re-allocating memory and re-wiring graph pointers across the entire dataset.
      * In contrast, `efSearch` is simply the **size of the candidate priority queue** maintained in memory during a single query traversal. Increasing `efSearch` simply tells the greedy search algorithm to inspect more neighbors before halting, making it 100% dynamic per API call.

    **Q2: How does distributed sharding execute Top-K nearest neighbor aggregation across 10 cluster nodes?**
    * **Answer**: Distributed vector databases (Qdrant, Milvus) use a **Scatter-Gather Architecture**:
      1. **Scatter**: The Query Coordinator broadcasts the query vector and requested $K$ (e.g. $K=10$) to all 10 shard nodes simultaneously.
      2. **Local Search**: Each shard node searches its local HNSW index and returns its **local Top-10 candidates** along with distance scores and document IDs.
      3. **Gather & Merge**: The Coordinator receives $10 \\times 10 = 100$ candidate records, sorts them into a single global priority queue, takes the **global Top-10**, and hydrates the final documents from storage.

    **Q3: What role does CPU SIMD instruction vectorization (AVX-512) play in vector search optimization?**
    * **Answer**: Computing dot products involves millions of repeated floating-point multiply-accumulate operations. Without vectorization, the CPU processes one float at a time.
    With **AVX-512**, modern CPUs process sixteen 32-bit floats in a single clock cycle across 512-bit registers. Compiling vector libraries (FAISS, HNSWlib) with AVX-512 provides a **4x to 8x throughput speedup** over non-vectorized code.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Hardcoding `efSearch` to Extreme Values in Production**:
      * *Issue*: Setting `efSearch = 500` to squeeze out an extra 0.2% recall.
      * *Consequence*: Query latency surges from 3ms to 40ms, cutting server QPS capacity by 90% and causing timeouts under peak load.
      * *Fix*: Standardize `efSearch` between 32 and 96, where the Pareto curve exhibits maximum efficiency.

    * **Trap 2: Ignoring Network Serialization Latency in Distributed Shards**:
      * *Issue*: Requesting $K=1,000$ from 20 shards in a distributed cluster.
      * *Consequence*: Shards transmit 20,000 candidate vectors over internal network links, causing coordinator network saturation and serialization bottlenecks.
      * *Fix*: Only return document IDs and float scores from shard workers; hydrate full text payloads only for the final global top $K$.

    * **Trap 3: Inadequate `efConstruction` Leading to Graph Disconnection**:
      * *Issue*: Setting `efConstruction = 16` to accelerate index creation.
      * *Consequence*: The graph contains isolated subgraphs that are unreachable from the upper skip-list entry points, permanently capping recall at 70%.
      * *Fix*: Maintain `efConstruction >= 2 * M`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `efSearch`**
      * *Definition*: The dynamic candidate queue size evaluated during HNSW query traversal, tunable at runtime to balance latency and recall.
    * **Flashcard 2: `M` (Max Connections)**
      * *Definition*: The maximum number of bidirectional graph links allocated per node in HNSW, determining graph connectivity and RAM consumption.
    * **Flashcard 3: Scatter-Gather Aggregation**
      * *Definition*: The distributed pattern where a coordinator queries multiple index shards concurrently and merges their local Top-K results into a global Top-K.
    * **Flashcard 4: AVX-512 SIMD Vectorization**
      * *Definition*: CPU hardware instructions that process sixteen 32-bit floats in parallel per clock cycle, accelerating vector distance calculations by 4x-8x.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Search Optimization:

- [ ] I can articulate the roles of `M`, `efConstruction`, and `efSearch` in HNSW.
- [ ] I know why `efSearch` can be tuned dynamically per query while `M` is fixed.
- [ ] I can explain how distributed clusters execute Scatter-Gather Top-K aggregation.
- [ ] I understand how to discover the optimal Pareto operating zone for QPS vs Recall.
- [ ] I understand the throughput impact of AVX-512 SIMD vectorization.
""")
