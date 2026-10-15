import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.2 Approximate Nearest Neighbor - Vector DB Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("10.2 Approximate Nearest Neighbor (ANN): Theory, Recall@K & The Pareto Frontier")
st.caption("Chapter 10: Vector Indexing | Domain: 4. Vector Database")
st.markdown("""
In classical computer science, algorithms are evaluated on strict deterministic correctness: searching an array must return the exact element, 
and sorting must achieve 100% order fidelity.

In modern AI engineering, **Exact Nearest Neighbor (ENN)** is computationally intractable at web scale. 
To deliver sub-10 millisecond search across millions of embedding vectors, vector databases rely on **Approximate Nearest Neighbor (ANN)** algorithms.

ANN embraces a pragmatic mathematical compromise: **sacrifice 1% to 5% of retrieval accuracy in exchange for a 100x to 1,000x acceleration in query speed**.

In this lesson, we examine the formal foundations of ANN, explore how **Recall@K** measures approximation quality, analyze the 
**Speed vs. Accuracy Pareto Frontier**, understand why semantic search naturally tolerates approximation errors, and experiment with an 
interactive Pareto benchmark simulator.
""")

st.divider()

# Section 1: Exact Nearest Neighbor vs Approximate Nearest Neighbor
st.header("1. Exact Nearest Neighbor (ENN) vs Approximate Nearest Neighbor (ANN)", divider="rainbow")

col_def1, col_def2 = col_def1, col_def2 = st.columns([1.1, 0.9])

with col_def1:
    st.markdown("""
    ### Exact Nearest Neighbor (ENN / Flat Search)
    * **Mathematical Objective**: Given query vector `Q` and dataset `V = {v_1, v_2, ..., v_N}`, find the top `K` vectors that strictly minimize distance `D(Q, v_i)`.
    * **Guarantee**: 100% mathematical precision. The returned list is the absolute **Ground Truth**.
    * **Bottleneck**: Complexity is strictly `O(N * D)`. As `N` scales into millions, latency violates real-time SLAs (>500ms).

    ### Approximate Nearest Neighbor (ANN)
    * **Mathematical Objective**: Find a set of `K` candidate vectors whose distances to `Q` are **sufficiently close** to the true nearest neighbors with high probability.
    * **Heuristic Strategy**: Space partitioning (Voronoi cells), graph skip lists (HNSW), or localized hashing (LSH) navigate toward the highest-density neighborhood of `Q` without evaluating more than 0.1% to 1% of points.
    * **Result**: Latency drops from hundreds of milliseconds to 2-5 milliseconds, with a 95% to 99% probability of including the true top-K neighbors.
    """)

with col_def2:
    st.markdown("### The Trade-off Frontier Concept")
    pareto_dot = """
    digraph TradeoffConcept {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Exact [label="Exact k-NN (Flat)\\nRecall: 100%\\nLatency: 450 ms\\nFLOPs: 1.5 Billion", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        AnnFast [label="Aggressive ANN (Low efSearch)\\nRecall: 92%\\nLatency: 1.2 ms\\nFLOPs: 3 Million", fillcolor="#FEF3C7", color="#D97706"];

        AnnBalanced [label="Balanced ANN (Production Sweet Spot)\\nRecall: 98%\\nLatency: 3.8 ms\\nFLOPs: 8 Million", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        AnnAccurate [label="High-Recall ANN (Deep efSearch)\\nRecall: 99.5%\\nLatency: 14.5 ms\\nFLOPs: 25 Million", fillcolor="#E0E7FF", color="#3B82F6"];

        Exact -> AnnFast [label="Prune 99.8%\\nCalculations", style="dashed"];
        AnnFast -> AnnBalanced [label="Increase Search Depth"];
        AnnBalanced -> AnnAccurate [label="Diminishing Returns"];
    }
    """
    st.graphviz_chart(pareto_dot, use_container_width=False)

st.divider()

# Section 2: Measuring ANN Quality: Recall@K and NDCG
st.header("2. Measuring ANN Quality: Recall@K & NDCG", divider="rainbow")

col_met1, col_met2 = col_met1, col_met2 = st.columns([1, 1])

with col_met1:
    st.markdown("""
    ### The Primary Benchmark Metric: Recall@K
    To evaluate whether an ANN index is accurate enough for production, engineers compute **Recall@K against an Exact Flat baseline**:

    ```text
    Recall@K = (Number of true Top-K items retrieved by ANN) / K
    ```

    #### Concrete Example (K = 5):
    * **Ground Truth (Flat Scan)**: `["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]`
    * **ANN Output (HNSW)**: `["doc_A", "doc_B", "doc_C", "doc_E", "doc_Z"]`
    * **Overlap**: 4 out of 5 items match (`doc_A`, `doc_B`, `doc_C`, `doc_E`).
    * **Recall@5**: `4 / 5 = 0.80 (80%)`.

    ### Normalized Discounted Cumulative Gain (NDCG@K)
    While Recall@K treats all positions equally, **NDCG@K** rewards returning the absolute best item at **Rank 1**:
    * Missing the true #1 item incurs a heavy penalty.
    * Missing the true #10 item incurs a minor penalty.
    * Crucial for user-facing search and recommendations where only the top 1-3 items are viewed.
    """)

with col_met2:
    st.markdown("""
    ### Why Generative AI Tolerates Approximation
    In traditional database operations (e.g., retrieving an account balance), an approximate result is unacceptable. 
    However, in **Generative AI and RAG architectures**, ANN approximation is completely harmless:

    1. **Semantic Redundancy**: If a user asks *"How to setup Redis replication?"*, documents ranked #1, #2, #3, and #4 typically contain the exact same setup instructions phrased slightly differently. Missing the #1 doc and returning #2 delivers the exact same answer to the LLM.
    2. **Embedding Model Margin of Error**: Even state-of-the-art embedding models have semantic variance. The mathematical difference between distance 0.142 and 0.145 is often noise rather than meaningful relevance divergence.
    3. **LLM Context Synthesis**: RAG prompts inject 3 to 7 context chunks. The generator synthesizes across all retrieved passages, rendering minor rank permutations inconsequential.
    """)

st.divider()

# Section 3: The Latency vs Recall Pareto Frontier
st.header("3. The Latency vs Recall Pareto Frontier", divider="rainbow")

st.markdown("""
The **Pareto Frontier** represents the optimal trade-off boundary between query throughput (Queries Per Second - QPS) and Recall@K. 
In vector databases, you adjust your position on this frontier using **search depth parameters**:
* **HNSW**: `efSearch` (size of the dynamic candidate exploration list during traversal).
* **IVF**: `nprobe` (number of Voronoi centroid clusters inspected per query).
""")

col_pf1, col_pf2 = col_pf1, col_pf2 = st.columns([1, 1])

with col_pf1:
    st.markdown("""
    #### The Law of Diminishing Returns in ANN
    * Moving from **80% to 95% Recall** is cheap: requires only a small increase in `efSearch` (+20% latency).
    * Moving from **95% to 98% Recall** is moderate: requires doubling `efSearch` (+100% latency).
    * Moving from **98% to 99.8% Recall** is extremely expensive: requires quadrupling `efSearch`, approaching near-flat scan latency!
    * **Industry Rule of Thumb**: Most production enterprise RAG systems optimize for **95% to 98% Recall@10**.
    """)

with col_pf2:
    st.markdown("""
    #### Production Parameter Tuning Guide
    | Target SLA | Recommended Recall@10 | HNSW `efSearch` | IVF `nprobe` | Typical Latency |
    | :--- | :--- | :--- | :--- | :--- |
    | **Ultra-High QPS** (Autocomplete) | ~90% - 92% | 16 - 32 | 4 - 8 | 0.8 - 1.5 ms |
    | **Standard RAG** (Enterprise Chat) | ~96% - 98% | 64 - 128 | 16 - 32 | 2.5 - 5.0 ms |
    | **High-Precision** (Legal / Medical) | ~99.0% - 99.5% | 200 - 400 | 64 - 128 | 12 - 25 ms |
    """)

st.divider()

# Section 4: Interactive Workbench: Recall@K & Pareto Frontier Simulator
st.header("4. Interactive Workbench: Recall@K & Pareto Frontier Simulator", divider="rainbow")

st.markdown("""
Tune dataset scale and search parameters to visualize the live Pareto trade-off curve between Recall@10 and Query Latency.
""")

col_sim_c1, col_sim_c2 = col_sim_c1, col_sim_c2 = st.columns([1, 1])

with col_sim_c1:
    st.subheader("1. Index & Scale Configuration")
    sim_n = st.select_slider(
        "Corpus Size (N vectors):",
        options=[10_000, 50_000, 250_000, 1_000_000, 5_000_000],
        value=250_000,
        format_func=lambda x: f"{x:,} vectors"
    )
    
    sim_k = st.slider("Top-K to Retrieve:", min_value=1, max_value=20, value=10)

with col_sim_c2:
    st.subheader("2. Search Depth Tuning")
    sim_ef = st.slider(
        "Active Search Depth (HNSW efSearch):",
        min_value=8,
        max_value=256,
        value=64,
        step=8,
        help="Higher efSearch increases recall but evaluates more graph nodes, raising latency."
    )

# Compute Pareto Curve Data Points
ef_test_points = [8, 16, 32, 64, 96, 128, 192, 256, 384]

curve_rows = []
for ef in ef_test_points:
    # Model recall as a logistic function of efSearch relative to K and N
    # Typical empirical curve: Recall = 1 - exp(- (ef / (k * 1.8)))
    raw_recall = 1.0 - np.exp(- (ef / (sim_k * 1.5)))
    recall_val = min(0.999, max(0.60, raw_recall))
    
    # Latency model: base hop latency + ef search linear expansion
    hops = np.log2(sim_n)
    nodes_visited = int(hops * ef * 0.45)
    lat_ms = 0.4 + (nodes_visited * 0.0008)
    qps = 1000.0 / lat_ms
    
    curve_rows.append({
        "efSearch": ef,
        "Recall@K (%)": round(recall_val * 100, 2),
        "Latency (ms)": round(lat_ms, 2),
        "QPS (Req/sec)": int(qps),
        "Nodes Visited": nodes_visited
    })

df_curve = pd.DataFrame(curve_rows)

# Identify current selected point
curr_recall = min(0.999, max(0.60, 1.0 - np.exp(- (sim_ef / (sim_k * 1.5)))))
curr_nodes = int(np.log2(sim_n) * sim_ef * 0.45)
curr_lat = 0.4 + (curr_nodes * 0.0008)
curr_qps = 1000.0 / curr_lat

# Exact scan baseline for comparison
exact_comps = sim_n
exact_lat = max(15.0, (sim_n / 10000) * 1.8)

st.divider()

col_out_chart, col_out_stats = st.columns([1.3, 0.7])

with col_out_chart:
    st.subheader("The Empirical Pareto Frontier Curve")
    st.line_chart(
        data=df_curve.set_index("Latency (ms)")[["Recall@K (%)"]],
        use_container_width=True
    )
    st.caption("Notice the steep initial gain transitioning into a flat plateau where doubling latency yields < 1% recall improvement.")

with col_out_stats:
    st.subheader("Active Configuration Metrics")
    st.metric(
        label="Simulated Recall@10",
        value=f"{curr_recall * 100:.2f}%",
        delta=f"efSearch = {sim_ef}"
    )
    st.metric(
        label="Query Latency",
        value=f"{curr_lat:.2f} ms",
        delta=f"{exact_lat / curr_lat:.0f}x faster than Exact Flat"
    )
    st.metric(
        label="Throughput (1 Core)",
        value=f"{int(curr_qps):,} QPS",
        help="Estimated single-threaded queries per second."
    )

st.subheader("Pareto Parameter Trade-Off Table")
st.dataframe(
    df_curve.style.background_gradient(subset=["Recall@K (%)"], cmap="Greens").background_gradient(subset=["Latency (ms)"], cmap="Reds"),
    use_container_width=True,
    hide_index=True
)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: How do you measure Recall@K of an ANN vector database in an automated production CI/CD pipeline?"):
    st.markdown("""
    **Answer**:
    1. **Sample Query Set**: Generate a golden benchmark evaluation suite of 1,000 representative query vectors.
    2. **Ground Truth Generation**: Run exact flat scans (`IndexFlatL2` in FAISS or exhaustive matrix multiplication) to compute the true mathematical top-K IDs for all 1,000 queries. Store this ground truth mapping.
    3. **ANN Benchmark Execution**: Query the candidate ANN index (e.g. ChromaDB HNSW) with the same 1,000 vectors and record retrieved IDs and latency.
    4. **Recall Calculation**: For each query, compute `|ANN_IDs ∩ True_IDs| / K`. Average across all 1,000 queries.
    5. **Gatekeeper Alert**: Fail CI/CD deployment if Recall@10 drops below 95% or p99 latency exceeds 10ms.
    """)

with st.expander("Q2: Why does increasing `efSearch` improve Recall@K at query time without rebuilding the index?"):
    st.markdown("""
    **Answer**:
    The HNSW graph structure (nodes, layers, and edges) is static once built. 
    However, when searching the graph:
    * The algorithm maintains a priority queue of candidate nodes to visit.
    * The parameter **`efSearch` defines the capacity of this candidate queue**.
    * A larger queue (`efSearch = 128`) allows the search algorithm to explore more alternate branching paths and avoid getting trapped in local minima (dead ends), leading to higher recall.
    * Because `efSearch` only governs exploration breadth during traversal, it can be dynamically tuned per-query without touching the stored index!
    """)

with st.expander("Q3: When would an engineering team deliberately choose 90% Recall over 99% Recall?"):
    st.markdown("""
    **Answer**:
    In ultra-high throughput, latency-critical applications such as:
    1. **Typeahead / Semantic Autocomplete**: The user is typing keystrokes; responses must return in under 2ms. A 90% recall is completely acceptable because the user's intent is evolving with every keystroke.
    2. **Large-Scale E-Commerce Retrieval (Candidate Generation)**: The vector index retrieves an initial pool of 200 candidates to pass to an expensive cross-encoder reranker. Getting 90% of the true top-200 is sufficient because the reranker re-evaluates all candidates anyway.
    3. **Infrastructure Cost Reduction**: Achieving 90% recall might allow handling 5,000 QPS on 4 servers, whereas 99% recall might require 24 servers to handle the same QPS load.
    """)

with st.expander("Q4: What is the difference between Recall@K and Precision@K in the context of ANN vector search?"):
    st.markdown("""
    **Answer**:
    In classical Information Retrieval, Precision is `Relevant / Retrieved` and Recall is `Relevant_Retrieved / Total_Relevant_in_Corpus`.
    
    In ANN evaluation, the definition is adapted:
    * **Recall@K** measures: *"What fraction of the true top-K nearest neighbors did the ANN search successfully retrieve?"*
    * Because the query always retrieves exactly `K` candidates, **Recall@K and Precision@K are mathematically identical**:  
      `Precision@K = Overlap / K = Recall@K`.  
    Therefore, the industry standard terminology strictly uses **Recall@K**.
    """)

with st.expander("Q5: What is the relationship between the number of dataset vectors (N) and the required `efSearch` to maintain 98% recall?"):
    st.markdown("""
    **Answer**:
    In HNSW, the diameter of the graph and the number of layers scale logarithmically with `N` (`Layers ~ log(N)`). 
    To maintain a constant recall target (e.g. 98%) as the dataset grows from 10,000 to 10,000,000 vectors:
    * The required `efSearch` does **not** scale linearly; it scales sub-linearly (`~ log(N)`).
    * If `efSearch=32` achieves 98% recall on 10k vectors, typically `efSearch=64` or `efSearch=96` is sufficient to maintain 98% recall on 10 million vectors.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Over-Optimizing for 99.9% Recall in RAG**  
    Spending weeks tuning indices to reach 99.9% recall increases server hardware costs by 5x with zero noticeable difference in LLM generation quality. Aim for 95%-98% Recall@10.
    """)
    st.warning("""
    **Trap 2: Benchmarking Recall on Synthetic Random Vectors**  
    Uniform random vectors do not exhibit the clustered manifold geometry of real embeddings. Benchmarks run on synthetic Gaussian noise will severely misestimate real-world HNSW recall. Always benchmark on actual embedding model outputs.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Setting `efSearch < K`**  
    If you request `n_results = 20` but set `efSearch = 10`, Chroma/HNSW cannot return 20 candidates without degrading quality or raising errors. Always enforce `efSearch >= K`.
    """)
    st.warning("""
    **Trap 4: Ignoring p99 Latency Spikes**  
    While mean query latency may be 3ms, queries landing in sparse peripheral regions of the vector space can explore the entire graph, spiking p99 latency to 80ms. Always monitor p99 and p99.9 latency SLAs.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Recall@K?"):
        st.markdown("""
        **Recall@K**:  
        Fraction of true Ground Truth nearest neighbors found by ANN: `|ANN_K ∩ Flat_K| / K`.
        """)
    with st.expander("🗂️ Flashcard: What is the Pareto Frontier in ANN?"):
        st.markdown("""
        **Pareto Frontier**:  
        The curve representing the optimal trade-off between query latency (speed) and Recall@K (accuracy).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why can RAG tolerate ANN error?"):
        st.markdown("""
        **Semantic Redundancy**:  
        Top candidate chunks often contain identical facts, and embedding models have natural variance.
        """)
    with st.expander("🗂️ Flashcard: Can `efSearch` be changed without rebuilding?"):
        st.markdown("""
        **Dynamic Tuning**:  
        Yes. `efSearch` only controls search traversal queue size at query time; the index is unchanged.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the mathematical difference between Exact Nearest Neighbor and ANN.", value=True)
chk2 = st.checkbox("I can calculate Recall@K against an Exact Flat baseline.", value=True)
chk3 = st.checkbox("I understand why the Latency vs Recall Pareto frontier exhibits diminishing returns.", value=True)
chk4 = st.checkbox("I know why Generative AI and RAG architectures tolerate minor ANN approximation errors.", value=True)
chk5 = st.checkbox("I know how to tune `efSearch` dynamically to balance throughput (QPS) and recall.", value=True)
chk6 = st.checkbox("I can establish automated CI/CD benchmark gates to monitor ANN recall regression.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Approximate Nearest Neighbor and are ready for Chapter 10.3: Popular Index Algorithms (HNSW, IVF, PQ, ScaNN, Annoy)!")
