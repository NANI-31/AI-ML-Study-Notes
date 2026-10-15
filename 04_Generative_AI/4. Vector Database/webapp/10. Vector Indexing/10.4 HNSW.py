import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.4 HNSW Deep Dive - Vector DB Curriculum",
    page_icon="🕸️",
    layout="wide"
)

# Title & Overview
st.title("10.4 HNSW Deep Dive: Multi-Layer Architecture, M, efConstruction & efSearch")
st.caption("Chapter 10: Vector Indexing | Domain: 4. Vector Database")
st.markdown("""
**Hierarchical Navigable Small World (HNSW)** is the undisputed gold standard vector indexing algorithm of the generative AI era. 
It powers the search engines of **ChromaDB, Pinecone, Weaviate, Qdrant, and Milvus**.

HNSW achieves sub-5 millisecond query latency with 98%+ recall across millions of vectors by fusing two foundational computer science concepts:
1. **Skip Lists**: Hierarchical multi-layer linked structures that enable `O(log N)` search by skipping over vast stretches of space at upper layers.
2. **Navigable Small World (NSW) Graphs**: Networks where any two nodes can reach each other in a small number of hops (the "six degrees of separation" property).

In this lesson, we dissect HNSW's multi-layer graph architecture, master the three critical tuning hyperparameters 
(**`M`**, **`efConstruction`**, **`efSearch`**), understand how ChromaDB configures HNSW, and experiment with an interactive 
multi-layer graph traversal simulator.
""")

st.divider()

# Section 1: The Multi-Layer Graph Architecture
st.header("1. The Multi-Layer Graph Architecture", divider="rainbow")

col_hnsw1, col_hnsw2 = col_hnsw1, col_hnsw2 = st.columns([1.1, 0.9])

with col_hnsw1:
    st.markdown("""
    ### How Multi-Layer Skip Graphs Work
    In traditional 1D skip lists, elements appear in higher layers with decaying probability `p = 1 / 2`. 
    HNSW adapts this to high-dimensional metric spaces:

    * **Layer 0 (Base Layer)**: Contains **100% of all vectors in the database**. Every node is connected to its local neighbors with up to `2 * M` bidirectional edges.
    * **Layer 1 (Arterial Layer)**: Contains a random sample of vectors (~`1 / M` of the dataset). Links span medium distances across thematic clusters.
    * **Layer 2 (Express Highway)**: Contains very few vectors (~`1 / M^2` of the dataset). Links span vast distances across the entire vector space.
    * **Top Entry Point**: Traversal always begins at the single global entry node residing on the highest populated layer.

    ### The Greedy Routing Search Algorithm
    1. **Top-Down Zoom**: The search starts at the entry node on `Layer_Max`.
    2. **Local Greedy Step**: Evaluates distance to all neighbors of the current node. If a neighbor is closer to query `Q`, hop to it.
    3. **Layer Drop**: When no neighbor is closer to `Q` than the current node (a local minimum on that layer), the algorithm drops down to the same node on `Layer - 1`.
    4. **Repeat**: Proceeds downward until it hits **Layer 0**.
    5. **Layer 0 Exploration**: Performs a bounded priority-queue search of depth **`efSearch`** to collect the top-K nearest neighbors.
    """)

with col_hnsw2:
    st.markdown("### HNSW Multi-Layer Skip Graph Anatomy")
    hnsw_dot = """
    digraph HNSWAnatomy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_l2 {
            label="Layer 2: Express Highway (Few nodes, Long hops)"; style="dashed"; color="#DC2626";
            E2 [label="Entry Node", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
            N2 [label="Node X", fillcolor="#F3F4F6"];
            E2 -> N2 [label="Long distance hop", color="#DC2626"];
        }

        subgraph cluster_l1 {
            label="Layer 1: Regional Roads (Medium density)"; style="dashed"; color="#D97706";
            E1 [label="Entry Node", fillcolor="#FEF3C7"];
            N1A [label="Node A", fillcolor="#F3F4F6"];
            N1B [label="Node B", fillcolor="#F3F4F6"];
            E1 -> N1A;
            N1A -> N1B;
        }

        subgraph cluster_l0 {
            label="Layer 0: Local Neighborhoods (All N vectors, Dense links)"; style="dashed"; color="#059669";
            E0 [label="Entry Node", fillcolor="#D1FAE5"];
            N0A [label="Vector 1", fillcolor="#E0E7FF"];
            N0B [label="Vector 2", fillcolor="#E0E7FF"];
            N0C [label="Vector 3", fillcolor="#E0E7FF"];
            N0D [label="Target Nearest Neighbor", fillcolor="#A7F3D0", color="#047857", style="filled,bold"];
            E0 -> N0A;
            N0A -> N0B;
            N0B -> N0C;
            N0C -> N0D [label="efSearch Beam", color="#059669", style="bold"];
        }

        E2 -> E1 [style="dotted", label="Drop layer"];
        N1B -> N0C [style="dotted", label="Drop layer"];
    }
    """
    st.graphviz_chart(hnsw_dot, use_container_width=False)

st.divider()

# Section 2: The Three Foundational Tuning Hyperparameters
st.header("2. The Three Foundational Tuning Hyperparameters", divider="rainbow")

st.markdown("""
HNSW behavior is governed by three primary knobs. Understanding when to tune each parameter is vital for production engineering:
""")

col_param1, col_param2, col_param3 = st.columns(3)

with col_param1:
    st.markdown("""
    #### 1. `M` (Max Connections)
    * **Definition**: The maximum number of bidirectional edges connecting each node on layers > 0 (Layer 0 allows up to `2 * M`).
    * **Typical Values**: `16` to `64` (default in ChromaDB: **16**).
    * **Impact on Index**:
      * Higher `M` improves graph connectivity and recall for complex high-dimensional datasets.
      * Higher `M` **linearly increases RAM footprint** (more pointer storage) and slows down insertion speed.
    * **Rule**: Use `M=16` for standard text embeddings (384-768 dim); use `M=32` to `64` for high dimensions (1536+ dim) or extreme recall demands.
    """)

with col_param2:
    st.markdown("""
    #### 2. `efConstruction` (Build Depth)
    * **Definition**: The size of the dynamic candidate priority queue evaluated during **index construction** (when adding vectors).
    * **Typical Values**: `100` to `400` (default in ChromaDB: **100**).
    * **Impact on Index**:
      * Higher `efConstruction` searches more deeply to discover the true `M` best neighbors when wiring new vectors into the graph.
      * Increases index build time and ingestion CPU load.
      * **Has ZERO impact on search query latency** or query RAM!
    * **Rule**: Increase to `200-400` for offline bulk datasets to maximize graph quality.
    """)

with col_param3:
    st.markdown("""
    #### 3. `efSearch` (Query Depth)
    * **Definition**: The size of the dynamic candidate priority queue maintained during **query traversal** at Layer 0.
    * **Typical Values**: `32` to `256` (default in ChromaDB: **100**).
    * **Impact on Index**:
      * Directly controls the **Pareto trade-off between Recall and Query Latency**.
      * Can be adjusted **dynamically per query** without rebuilding or reloading the index.
      * **Has ZERO impact on RAM** or index build time!
    * **Rule**: Set `efSearch >= K`. For real-time autocomplete use `32`; for precision legal/medical RAG use `128-256`.
    """)

st.divider()

# Section 3: ChromaDB HNSW Configuration & Tuning
st.header("3. ChromaDB HNSW Configuration & Tuning", divider="rainbow")

col_cfg1, col_cfg2 = col_cfg1, col_cfg2 = st.columns([1, 1])

with col_cfg1:
    st.markdown("""
    ### Configuring HNSW in ChromaDB
    In ChromaDB, HNSW parameters are passed during **collection creation** inside the `metadata` dictionary:
    
    ```python
    import chromadb

    client = chromadb.PersistentClient(path="./my_chroma_db")

    # Create collection with custom tuned HNSW hyperparameters
    collection = client.create_collection(
        name="production_enterprise_knowledge",
        metadata={
            # 1. Distance metric space
            "hnsw:space": "cosine",  # "cosine" | "l2" | "ip"
            
            # 2. Max edges per node (RAM vs Recall)
            "hnsw:M": 32,            # Default: 16
            
            # 3. Construction search depth (Build time vs Quality)
            "hnsw:construction_ef": 200,  # Default: 100
            
            # 4. Query search depth (Latency vs Recall)
            "hnsw:search_ef": 128,   # Default: 100
            
            # 5. Number of threads during build
            "hnsw:num_threads": 8
        }
    )
    ```
    """)

with col_cfg2:
    st.markdown("""
    ### Memory Footprint Formula for HNSW
    To size Kubernetes pods or EC2 instances accurately for HNSW collections, use the following engineering formula:
    
    ```text
    Total_RAM_Bytes = N * [ (D * 4) + (M * 2 * 8) + Overhead ]
    ```
    
    Where:
    * `N`: Total vector count.
    * `D`: Embedding dimensions (e.g. 1536).
    * `D * 4`: Bytes for raw float32 vector components.
    * `M * 2 * 8`: 64-bit edge pointer storage across layers (Layer 0 holds `2 * M` pointers of 8 bytes).
    * `Overhead`: ~64 bytes per node for metadata, locks, and SQLite IDs.
    
    #### Sizing Example (1,000,000 vectors, 1536-dim, M=32):
    * Raw Vectors: `1M * 1536 * 4 bytes = 6.14 GB`
    * Graph Edges: `1M * (32 * 2 * 8) = 0.51 GB`
    * Overhead & Buffers: `~1.2 GB`
    * **Recommended Provisioned RAM**: **~8 GB to 10 GB**.
    """)

st.divider()

# Section 4: Interactive Workbench: HNSW Multi-Layer Traversal & Tuner
st.header("4. Interactive Workbench: HNSW Traversal & Hyperparameter Tuner", divider="rainbow")

st.markdown("""
Adjust `M`, `efConstruction`, and `efSearch` to observe their live mathematical impact on RAM footprint, 
build duration, Recall@10, and single-query latency.
""")

col_wb_knobs, col_wb_sim = st.columns([1, 1.2])

with col_wb_knobs:
    st.subheader("1. Hyperparameter Knobs")
    
    ui_m = st.select_slider(
        "Max Connections (M):",
        options=[8, 16, 24, 32, 48, 64],
        value=32,
        help="Higher M improves graph connectivity for high-dimensional vectors at the cost of more pointer RAM."
    )
    
    ui_ef_c = st.slider(
        "Build Depth (efConstruction):",
        min_value=50,
        max_value=400,
        value=200,
        step=25,
        help="Higher values find better neighbors during insertion; slows build time but has 0 query latency impact."
    )
    
    ui_ef_s = st.slider(
        "Query Depth (efSearch):",
        min_value=16,
        max_value=300,
        value=100,
        step=16,
        help="Governs search depth at query time. Adjusts the Recall vs Latency trade-off."
    )
    
    ui_dataset_n = st.select_slider(
        "Corpus Size (N Vectors):",
        options=[50_000, 200_000, 1_000_000, 5_000_000],
        value=1_000_000,
        format_func=lambda x: f"{x:,} vectors"
    )

with col_wb_sim:
    st.subheader("2. Simulated Performance Metrics")
    
    # Mathematical Modeling
    # RAM calculation
    d = 1536
    raw_vec_bytes = ui_dataset_n * d * 4
    graph_bytes = ui_dataset_n * (ui_m * 2 * 8)
    overhead_bytes = ui_dataset_n * 72
    total_ram_gb = (raw_vec_bytes + graph_bytes + overhead_bytes) / (1024**3)
    
    # Recall modeling: logistic curve based on efSearch and M
    base_recall = 1.0 - (0.35 / (1.0 + (ui_ef_s / 40.0)))
    m_boost = min(0.04, (ui_m - 8) * 0.0007)
    ef_c_boost = min(0.03, (ui_ef_c - 50) * 0.00008)
    sim_recall = min(0.998, base_recall + m_boost + ef_c_boost)
    
    # Latency modeling: log2(N) layers + efSearch hops
    layers = int(np.log2(ui_dataset_n) / np.log2(max(2, ui_m)))
    hops = layers + (ui_ef_s * 0.4)
    sim_lat_ms = 0.5 + (hops * 0.03)
    sim_qps = 1000.0 / sim_lat_ms
    
    # Build time relative multiplier
    build_multiplier = (ui_ef_c / 100.0) * (ui_m / 16.0)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="Estimated Recall@10",
            value=f"{sim_recall * 100:.2f}%",
            delta="High Accuracy" if sim_recall > 0.97 else "Fast Mode"
        )
        st.metric(
            label="Query Latency (p95)",
            value=f"{sim_lat_ms:.2f} ms",
            delta=f"~{int(sim_qps):,} QPS per core"
        )
    with col_m2:
        st.metric(
            label="Total RAM Footprint",
            value=f"{total_ram_gb:.2f} GB",
            delta=f"Graph edges: {(graph_bytes / (1024**3)):.2f} GB"
        )
        st.metric(
            label="Relative Build Time",
            value=f"{build_multiplier:.2f}x Baseline",
            help="Relative to baseline M=16, efConstruction=100."
        )

st.subheader("3. Generated ChromaDB Collection Configuration")
st.code(f"""import chromadb

client = chromadb.PersistentClient(path="./chroma_store")

collection = client.create_collection(
    name="production_knowledge_base",
    metadata={{
        "hnsw:space": "cosine",
        "hnsw:M": {ui_m},
        "hnsw:construction_ef": {ui_ef_c},
        "hnsw:search_ef": {ui_ef_s}
    }}
)
# Collection initialized with {ui_dataset_n:,} target capacity.
# Estimated RAM allocation: {total_ram_gb:.2f} GB
""", language="python")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the heuristic neighbor selection algorithm in HNSW, and why does it outperform simple nearest neighbor selection?"):
    st.markdown("""
    **Answer**:
    When a new vector is inserted into HNSW, it must select up to `M` neighbors from a candidate set.
    * **Naive Selection**: Simply pick the `M` closest vectors.  
      *Problem*: If 20 candidate vectors belong to the same dense cluster, they will all connect to each other, forming a tight clique with zero external connectivity. The graph becomes fragmented and search gets trapped in local minima.
    * **HNSW Heuristic Selection (Shrink Rule)**:  
      Picks neighbor `e` if it is close to `v`, but **discards** `e` if `e` is already closer to an already-selected neighbor than to `v`.  
      *Benefit*: Forces edges to branch out into diverse directions, maximizing angular diversity and maintaining the Small World navigable highway property.
    """)

with st.expander("Q2: Why does `efConstruction` have zero effect on query-time search latency?"):
    st.markdown("""
    **Answer**:
    `efConstruction` is strictly an **index-building hyperparameter**. 
    It defines how deep the graph exploration queue extends when inserting new vectors to locate the optimal `M` neighbors. 
    Once the index build is complete, the graph structure (its nodes and edges) is finalized. 
    At query time, the search algorithm explores the graph using **`efSearch`**, completely independent of `efConstruction`.
    """)

with st.expander("Q3: What happens if `efSearch` is set smaller than `n_results` (K)?"):
    st.markdown("""
    **Answer**:
    `efSearch` represents the size of the dynamic candidate priority queue maintained during Layer 0 search. 
    If `efSearch < K` (e.g., requesting top-20 results with `efSearch=10`), the algorithm cannot even track 20 candidates simultaneously. 
    In ChromaDB and hnswlib, setting `efSearch < K` either triggers an automatic internal clamp (`efSearch = max(efSearch, K)`) or degrades recall severely. **Rule**: Always ensure `efSearch >= K`, typically `efSearch >= 2 * K`.
    """)

with st.expander("Q4: How does HNSW handle node deletions in ChromaDB?"):
    st.markdown("""
    **Answer**:
    Deleting a node from an HNSW graph is mathematically challenging because removing a node breaks bidirectional edges and can partition the graph. 
    * In ChromaDB, deletions are handled via **Tombstoning**: the deleted vector ID is flagged as inactive in SQLite metadata and added to an exclusion bitset. During HNSW graph traversal, tombstoned nodes can still be hopped through as transit bridges, but are excluded from query results.
    * Periodically, or when the tombstone ratio exceeds a threshold (e.g., 20-30%), ChromaDB triggers a background compaction that rebuilds the graph segment to reclaim RAM.
    """)

with st.expander("Q5: Why is Layer 0 in HNSW allowed to have `2 * M` connections instead of `M`?"):
    st.markdown("""
    **Answer**:
    Higher layers (Layer 1, Layer 2) have very low node density and serve exclusively as coarse express highways, where degree `M` is sufficient to navigate between distant clusters. 
    However, **Layer 0 contains all `N` nodes** and requires dense local connectivity to ensure high recall during the final neighborhood exploration. Allowing `2 * M` edges on Layer 0 provides the necessary clustering density without noticeably slowing down top-down greedy routing.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Setting `M` Excessively High (e.g., M=128)**  
    Increasing `M` beyond 64 provides virtually zero additional recall on text embeddings while doubling pointer RAM consumption and causing severe cache misses in CPU L3 cache. Maintain `M` between 16 and 48.
    """)
    st.warning("""
    **Trap 2: Modifying `hnsw:M` After Collection Creation**  
    `M` is an immutable structural parameter fixed during graph construction. In ChromaDB, changing `hnsw:M` requires creating a new collection and re-ingesting documents. Only `hnsw:search_ef` can be changed dynamically.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Single-Threaded Index Ingestion**  
    Inserting 1,000,000 vectors on a single CPU thread with `efConstruction=200` can take over 4 hours. Always configure `hnsw:num_threads` to match available CPU cores and ingest in batches.
    """)
    st.warning("""
    **Trap 4: Starving the System of RAM Headroom**  
    Deploying an HNSW service on a server with barely enough RAM to hold vectors causes Linux Out-Of-Memory (OOM) kills during dynamic insertions when temporary neighbor heaps are allocated. Maintain a 30% RAM buffer.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is M in HNSW?"):
        st.markdown("""
        **Parameter M**:  
        Maximum number of bidirectional edges per node (Layer 0 allows `2*M`). Controls graph connectivity and RAM.
        """)
    with st.expander("🗂️ Flashcard: What is efConstruction?"):
        st.markdown("""
        **efConstruction**:  
        Candidate queue size during index building. Governs build time and graph quality; zero query latency impact.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is efSearch?"):
        st.markdown("""
        **efSearch**:  
        Candidate queue size during query search at Layer 0. Tunes the Recall vs Latency trade-off per query.
        """)
    with st.expander("🗂️ Flashcard: How are deletions handled in HNSW?"):
        st.markdown("""
        **Tombstoning**:  
        Nodes are marked as deleted in an exclusion bitset and pruned during background index compaction.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand how HNSW combines Skip Lists with Navigable Small World graphs.", value=True)
chk2 = st.checkbox("I know the exact roles of M, efConstruction, and efSearch.", value=True)
chk3 = st.checkbox("I know how to configure custom HNSW hyperparameters in ChromaDB collection metadata.", value=True)
chk4 = st.checkbox("I can calculate the RAM footprint of an HNSW index including raw vectors and graph edges.", value=True)
chk5 = st.checkbox("I understand the heuristic neighbor selection shrink rule and why it prevents cliques.", value=True)
chk6 = st.checkbox("I can tune HNSW parameters to achieve >98% Recall@10 with sub-5ms p95 latency.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completed Chapter 10: Vector Indexing & Algorithms and are ready for Chapter 11: Vector Database Storage!")
