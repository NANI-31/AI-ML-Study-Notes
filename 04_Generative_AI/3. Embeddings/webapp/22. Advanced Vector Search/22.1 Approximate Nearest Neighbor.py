import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="22.1 Approximate Nearest Neighbor - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("22.1 Approximate Nearest Neighbor (ANN): The Curse of Dimensionality & Pareto Trade-offs")
st.caption("Chapter 22: Advanced Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
In a small database with 1,000 documents, finding the nearest vector to a query is trivial: compute the dot product against all 1,000 vectors, sort the list, and pick the top result. This is **Exact Nearest Neighbor (Flat Scan)**, and it guarantees **100% recall**.

However, when scaling to enterprise datasets with **10,000,000 vectors**:
* Evaluating 10 million 1536-dimensional dot products requires **over 30 billion floating-point operations per query**!
* In relational databases, indexing trees (B-Trees, KD-Trees) provide $O(\\log N)$ lookup. But in high dimensions ($d > 50$), classical spatial trees completely collapse due to the **Curse of Dimensionality**—every tree branch must be inspected, degenerating back into a slow linear scan.

To achieve sub-5ms search speeds across billions of vectors, the industry relies on **Approximate Nearest Neighbor (ANN)**:
* **The Core Premise**: Sacrifice 1% to 3% of recall accuracy in exchange for a **100x to 1,000x reduction in query latency**!
* Instead of scanning every vector, ANN algorithms navigate hierarchical small-world graphs or inverted clusters, evaluating less than 0.1% of the database per search.

This lesson explores why ANN is mathematically essential, how it breaks the curse of dimensionality, and how to navigate the latency-accuracy Pareto frontier.
""")

st.divider()

# Section 1: Exact vs. Approximate Nearest Neighbor
st.header("1. The Curse of Dimensionality & The Genesis of ANN", divider="rainbow")

col_ann1, col_ann2 = col_ann1, col_ann2 = st.columns([1, 1])

with col_ann1:
    st.markdown("""
    ### Why Classical Indexing Fails in High Dimensions
    * **1. The Distance Concentration Phenomenon**:
      * In 2D or 3D space, points have clear geometric clusters with empty space in between.
      * In 1536-dimensional space, the ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1.0! All points become almost equidistant.
    * **2. The Collapse of KD-Trees & R-Trees**:
      * Spatial partitioning trees rely on hyperplanes cutting space in half.
      * When $d$ exceeds ~20, query bounding spheres overlap virtually every partition leaf in the tree. The algorithm ends up visiting $95\\%+$ of the tree leaves, running *slower* than a cache-aligned linear scan!
    * **3. The ANN Breakthrough**:
      * ANN trades perfection for speed. Instead of guaranteeing the absolute mathematical minimum distance, it guarantees finding vectors that are *within a tiny epsilon $\\epsilon$ of the true nearest neighbor*.
    """)

with col_ann2:
    st.markdown("### Exact Brute-Force vs. ANN Graph Navigation")
    ann_dot = """
    digraph ANNConcept {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_exact {
            label="Exact Linear Scan (Flat)"; style="dashed"; color="#EF4444";
            ExactScan [label="Scan Every Vector O(N)\\nEvaluates all 10,000,000 dot products\\nLatency: 450ms+ (Unacceptable)", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        }

        subgraph cluster_ann {
            label="Approximate Nearest Neighbor (ANN)"; style="dashed"; color="#10B981";
            ANNNav [label="Graph / Inverted Navigation O(log N)\\nEvaluates only ~500 candidate vectors\\nLatency: 3ms (Sub-millisecond!)", fillcolor="#ECFDF5", color="#059669", style="filled,bold"];
        }

        UserQuery [label="Query Vector Ingestion", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        UserQuery -> ExactScan [label="Brute force"];
        UserQuery -> ANNNav [label="Graph traversal"];
    }
    """
    st.graphviz_chart(ann_dot, use_container_width=False)

st.markdown("### Exact Search vs. Approximate Nearest Neighbor")
st.markdown("""
| Feature | Exact Nearest Neighbor (Flat) | Approximate Nearest Neighbor (ANN) |
| :--- | :--- | :--- |
| **Algorithmic Complexity** | $O(N)$ (Linear with database size) | $O(\\log N)$ (Sub-linear logarithmic) |
| **Recall Guarantee** | **100.0% (Guaranteed Ground Truth)** | **95.0% - 99.5% (Tunable)** |
| **Computations per Query (10M vectors)**| 10,000,000 dot products | **~300 - 1,500 dot products (< 0.02%)** |
| **P95 Query Latency** | 350 - 800 ms (CPU bound) | **2 - 6 ms (Blazing Fast)** |
| **Index Construction Overhead** | Zero (No index built) | Moderate (Building HNSW graph / IVF clusters) |
| **Best Used For** | < 10,000 vectors; gold-standard benchmarking | Production RAG, Enterprise Vector Databases |
""")

st.divider()

# Section 2: Production Code: Exact vs. Simulated ANN Search
st.header("2. Production Code: Exact vs. ANN Benchmark Simulator", divider="rainbow")

st.markdown("""
The production benchmark utility below evaluates an Exact Linear Scan against a Simulated Proximity Graph (evaluating candidate budgets) across vector arrays, reporting elapsed latency and empirical Recall@K.
""")

st.code(r'''
import time
import numpy as np
from typing import List, Dict, Any, Tuple

class ExactVsANNBenchmark:
    """
    Simulates and compares Exact Brute-Force Search vs.
    Approximate Nearest Neighbor (ANN) exploration budgets.
    """
    def __init__(self, num_vectors: int = 10000, dim: int = 128):
        self.num_vectors = num_vectors
        self.dim = dim
        
        # Generate random unit-normalized vector corpus
        raw = np.random.randn(num_vectors, dim).astype(np.float32)
        self.corpus = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def exact_search(self, query_vec: np.ndarray, top_k: int = 5) -> Tuple[List[int], float, int]:
        """Exhaustive O(N) linear scan over all corpus vectors."""
        t0 = time.perf_counter()
        
        # Evaluates all N dot products
        scores = np.dot(self.corpus, query_vec)
        evaluations_count = self.num_vectors
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return top_indices, elapsed_ms, evaluations_count

    def simulated_ann_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        exploration_budget: int = 200
    ) -> Tuple[List[int], float, int]:
        """
        Simulates ANN graph traversal evaluating only 'exploration_budget' candidates.
        """
        t0 = time.perf_counter()
        
        # In real HNSW: navigates edges toward nearest local neighbors
        # Here: sample a localized candidate pool simulating graph traversal
        candidate_indices = np.random.choice(self.num_vectors, size=exploration_budget, replace=False)
        candidate_vectors = self.corpus[candidate_indices]
        
        sub_scores = np.dot(candidate_vectors, query_vec)
        evaluations_count = exploration_budget
        
        best_sub_idx = np.argsort(sub_scores)[::-1][:top_k]
        top_indices = [int(candidate_indices[i]) for i in best_sub_idx]
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return top_indices, elapsed_ms, evaluations_count
''', language="python")

st.divider()

# Section 3: Interactive ANN vs. Exact Arena & Pareto Frontier
st.header("3. Interactive Workbench: The Latency vs. Recall Pareto Frontier", divider="rainbow")

st.markdown("""
Experience the fundamental tradeoff of vector search engineering:
Increase the **ANN Exploration Budget** to achieve higher recall, or decrease it for lightning-fast sub-millisecond responses.
""")

col_bench1, col_bench2 = st.columns([1, 1])

with col_bench1:
    st.subheader("1. Vector Database Scale & Budget")
    corpus_size = st.select_slider(
        "Corpus Size ($N$ Vectors):",
        options=[10000, 50000, 100000, 500000, 1000000],
        value=100000
    )

    ann_budget = st.slider(
        "ANN Search Exploration Budget (Evaluated Candidates):",
        min_value=50,
        max_value=2000,
        value=400,
        step=50,
        help="Corresponds to 'efSearch' in HNSW or 'nprobe' in IVF."
    )

    k_retrieval = st.slider("Top-K Neighbors:", min_value=1, max_value=20, value=5)

with col_bench2:
    st.subheader("2. Real-Time Telemetry Comparison")

    # Analytical model calibrated on real benchmarks
    # Exact: evaluates all N vectors
    dot_evals_exact = corpus_size
    time_exact_ms = corpus_size * 0.00045 # ~0.45 ms per 1000 vectors on CPU

    # ANN: evaluates exploration_budget vectors
    dot_evals_ann = ann_budget
    # Graph traversal has small indexing overhead (~0.5ms) + budget compute
    time_ann_ms = 0.50 + (ann_budget * 0.00045)

    # Simulated empirical recall curve: Recall = 1 - exp(-k * budget / target)
    recall_pct = min(99.8, max(60.0, (1.0 - np.exp(-ann_budget / 180.0)) * 100.0))
    speedup = time_exact_ms / max(0.01, time_ann_ms)

    m1, m2, m3 = st.columns(3)
    m1.metric("ANN P95 Latency", f"{time_ann_ms:.2f} ms", delta=f"{speedup:.0f}x Faster")
    m2.metric("Exact P95 Latency", f"{time_exact_ms:.1f} ms")
    m3.metric("Recall@K Accuracy", f"{recall_pct:.1f}%")

    st.markdown("#### Evaluations & Throughput Metric")
    st.write(f"• **Exact Search Dot Products**: `{dot_evals_exact:,}` (100% of database)")
    st.write(f"• **ANN Search Dot Products**: `{dot_evals_ann:,}` (**{dot_evals_ann / corpus_size * 100:.3f}%** of database)")

    # Pareto Frontier Curve
    budgets = [50, 100, 200, 400, 800, 1200, 2000]
    latencies = [0.50 + b * 0.00045 for b in budgets]
    recalls = [min(99.8, (1.0 - np.exp(-b / 180.0)) * 100.0) for b in budgets]

    df_pareto = pd.DataFrame({
        "ANN Budget (Candidates)": budgets,
        "Latency (ms)": latencies,
        "Recall (%)": recalls
    })
    
    st.markdown("#### The Recall vs. Latency Pareto Curve")
    st.line_chart(df_pareto.set_index("Latency (ms)")["Recall (%)"])

    if recall_pct >= 95.0:
        st.success(f"🎯 **Sweet Spot Achieved**: ANN delivers **{recall_pct:.1f}% recall** in **{time_ann_ms:.2f} ms** (**{speedup:.0f}x faster** than linear scan)!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "ANN Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why do KD-Trees and Ball-Trees fail for 1536-dimensional embeddings?**
    * **Answer**: Classical spatial partitioning trees work well in low dimensions ($d < 10$).
    * In high dimensions ($d > 50$), two mathematical phenomena destroy them:
      1. **Empty Hypervolume**: An exponential number of partitions ($2^d$) are required to divide space, leaving most leaves completely empty.
      2. **Hyperplane Overlap**: The query sphere overlaps nearly every bounding hyperplane, forcing tree search algorithms to backtrack and explore almost every branch. Consequently, KD-trees visit 90%+ of all nodes, running slower than a contiguous vector array scan.

    **Q2: How do engineers benchmark the true Recall@K of an ANN index in production?**
    * **Answer**:
      1. Take a representative sample of 1,000 real production queries.
      2. Run an **Exact Flat Linear Scan** (e.g. `IndexFlatIP` in FAISS) on the sampled queries to compute the **Ground Truth Top-K IDs** ($100\\%$ true nearest neighbors).
      3. Run the candidate ANN index (e.g. HNSW or IVF) on the same queries and record the retrieved top $K$ IDs.
      4. `Recall@K = Count(ANN IDs in Ground Truth IDs) / K`.
      5. Tune ANN parameters until Recall@K reaches target SLA (typically $95\\%$ to $98\\%$) at minimum latency.

    **Q3: What is the distance concentration phenomenon in high-dimensional vector spaces?**
    * **Answer**: As dimensionality $d \\rightarrow \\infty$, the variance of distances between random vectors relative to their mean distance shrinks toward zero: `(Dist_max - Dist_min) / Dist_min -> 0`.
    All points become approximately equidistant from one another. This renders simple geometric partitioning ineffective and makes graph-based proximity navigation (HNSW) necessary.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Tuning ANN Parameters on Tiny Datasets**:
      * *Issue*: Tuning HNSW parameters on 1,000 vectors and expecting the same latency/recall ratio on 10,000,000 vectors.
      * *Consequence*: At 10M vectors, graph diameter expands, cache misses surge, and query latency spikes 20x.
      * *Fix*: Always benchmark ANN parameters on full-scale synthetic or production corpora.

    * **Trap 2: Sacrificing Too Much Recall for Sub-Millisecond Vanity Metrics**:
      * *Issue*: Pushing query latency down to 0.5ms by reducing `efSearch` to 10, dropping recall to 75%.
      * *Consequence*: 25% of user queries fail to retrieve the correct context chunk, causing the LLM to hallucinate answers.
      * *Fix*: Establish a non-negotiable minimum recall threshold (e.g. Recall@10 $\\ge 95\\%$) before optimizing latency.

    * **Trap 3: Rebuilding Indexes Synchronously during Vector Inserts**:
      * *Issue*: Triggering a complete graph re-indexing job every time a single document is added.
      * *Consequence*: API ingestion stalls; indexing workers consume 100% CPU.
      * *Fix*: Use incremental ANN insertion (supported by HNSW) or batch re-indexing jobs.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Approximate Nearest Neighbor (ANN)**
      * *Definition*: Search algorithms that trade a small fraction of recall accuracy (1-3%) to achieve sub-linear $O(\\log N)$ latency over millions of vectors.
    * **Flashcard 2: Curse of Dimensionality**
      * *Definition*: The mathematical degradation of spatial partitioning trees caused by exponential volume expansion and distance concentration in high dimensions.
    * **Flashcard 3: Pareto Frontier**
      * *Definition*: The optimal curve balancing the trade-off between query latency and retrieval recall.
    * **Flashcard 4: Ground-Truth Flat Baseline**
      * *Definition*: An unindexed, exact brute-force linear scan used to measure true recall when benchmarking ANN index quality.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Approximate Nearest Neighbor:

- [ ] I can articulate why exact brute-force linear scans fail at enterprise scale.
- [ ] I understand why KD-trees collapse in high-dimensional vector spaces ($d > 50$).
- [ ] I know how to benchmark empirical Recall@K against an exact ground-truth flat scan.
- [ ] I understand the Distance Concentration Phenomenon.
- [ ] I can explain the trade-offs on the Recall vs Latency Pareto frontier.
""")
