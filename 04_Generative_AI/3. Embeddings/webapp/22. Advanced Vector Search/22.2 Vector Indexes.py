import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="22.2 Vector Indexes - Embeddings Curriculum",
    page_icon="🗺️",
    layout="wide"
)

# Title & Overview
st.title("22.2 Vector Indexes: HNSW, IVF, Product Quantization (PQ), & Flat Baselines")
st.caption("Chapter 22: Advanced Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
A vector database is not a monolith—its retrieval engine is powered by an underlying **Vector Index**.
Choosing the wrong vector index for your dataset can result in out-of-memory crashes, hours-long indexing build freezes, or sluggish query responses.

There are four primary families of vector indexes used in production systems (FAISS, Qdrant, Chroma, Milvus):
1. **Flat Index**: Unindexed brute-force scan. 100% recall, zero build time, but scales linearly $O(N)$.
2. **Inverted File Index (IVF)**: Clusters space into Voronoi cells via K-Means. Fast build time and low RAM, but requires hyperparameter tuning (`nlist`, `nprobe`).
3. **Hierarchical Navigable Small World (HNSW)**: Multi-layer graph skip-list. State-of-the-art query speed and recall, but consumes 1.5x - 2.5x more RAM for graph edges.
4. **Product Quantization (PQ)**: Vector compression decomposing vectors into sub-vectors and codebooks. Enables holding billions of vectors in memory.

This lesson explores how each index family works internally, their trade-offs, and how to configure them for production scale.
""")

st.divider()

# Section 1: The 4 Vector Index Architectures
st.header("1. Core Architectures: Flat, IVF, HNSW, & Product Quantization", divider="rainbow")

col_idx1, col_idx2 = col_idx1, col_idx2 = st.columns([1, 1])

with col_idx1:
    st.markdown("""
    ### The 4 Index Paradigms
    * **1. Flat Index (`IndexFlatIP` / `IndexFlatL2`)**:
      * Stores raw contiguous vectors without indexing.
      * Serves as the **ground-truth baseline** for accuracy benchmarking.
    * **2. Inverted File Index (`IVF`)**:
      * Clusters the vector corpus into $K$ Voronoi cells (`nlist`).
      * At query time, only the closest $C$ centroids (`nprobe`) are inspected.
      * Fast to build, low RAM, but requires a training phase on representative data.
    * **3. Hierarchical Navigable Small World (`HNSW`)**:
      * Constructs a multi-layer graph inspired by probabilistic skip-lists.
      * **Top Layers**: Sparse long-range "highway" edges for fast coarse navigation across the corpus.
      * **Bottom Layer**: Dense local neighborhood graph for fine-grained nearest neighbor convergence.
      * Delivers the highest recall at lowest latency across all ANN benchmarks!
    * **4. Product Quantization (`PQ`)**:
      * Slices vectors into $M$ sub-vectors and quantizes each into byte codebook centroids.
      * Frequently combined with IVF (**`IVF-PQ`**) to search billion-scale datasets on modest hardware.
    """)

with col_idx2:
    st.markdown("### HNSW Skip-List vs. IVF Voronoi Partitioning")
    struct_dot = """
    digraph IndexStructures {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_hnsw {
            label="HNSW: Multi-Layer Skip Graph"; style="dashed"; color="#10B981";
            L2 [label="Layer 2: Expressway Highway\\n(Sparse long-range links)", fillcolor="#D1FAE5"];
            L1 [label="Layer 1: Regional Neighborhood\\n(Medium density links)", fillcolor="#A7F3D0"];
            L0 [label="Layer 0: Dense Base Graph\\n(Contains all N vectors)", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
            L2 -> L1 -> L0 [label="Greedy descent"];
        }

        subgraph cluster_ivf {
            label="IVF: Voronoi Partitioning"; style="dashed"; color="#3B82F6";
            Centroids [label="K-Means Centroids\\n(nlist Voronoi centers)", fillcolor="#DBEAFE"];
            InvertedLists [label="Inverted Posting Lists\\nOnly probe nprobe closest cells", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
            Centroids -> InvertedLists [label="Assign vectors to centroids"];
        }
    }
    """
    st.graphviz_chart(struct_dot, use_container_width=False)

st.markdown("### Master Index Specification Matrix")
st.markdown("""
| Index Type | Index Build Time | RAM Footprint | P95 Latency (1M) | Recall@10 Range | Best Production Fit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flat** | **Zero (Instant)** | Low (Raw vectors only) | ~150 - 400 ms | **100.0% (Exact)** | < 20,000 vectors; gold-standard benchmarking. |
| **IVF-Flat** | Fast (K-Means train) | Low (Raw vectors + posting) | ~8 - 25 ms | 92.0% - 97.0% | Moderate datasets (50k - 2M) with limited RAM. |
| **HNSW** | Slow (Graph construction) | **High (Raw + Graph links: +50-100%)** | **~2 - 5 ms (Fastest)** | **97.0% - 99.5%** | **Default Enterprise RAG (Highest quality).** |
| **IVF-PQ** | Moderate (PQ train) | **Ultra-Low (8x - 16x compressed)** | ~5 - 12 ms | 85.0% - 94.0% | Billion-scale search on constrained RAM budgets. |
""")

st.divider()

# Section 2: Production Code: Index Factory & Configuration Builder
st.header("2. Production Code: Index Factory & FAISS Configuration Generator", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to configure index parameters programmatically, calculating estimated RAM usage and generating standard FAISS / Vector DB index strings.
""")

st.code(r'''
from typing import Dict, Any

class VectorIndexConfigurator:
    """
    Generates index configuration strings and calculates
    theoretical memory and build-time metrics for FAISS / Milvus.
    """
    @staticmethod
    def recommend_index(
        num_vectors: int,
        dim: int,
        available_ram_gb: float
    ) -> Dict[str, Any]:
        # Raw Float32 vector size in GB
        raw_vector_ram_gb = (num_vectors * dim * 4.0) / (1024.0 ** 3)
        
        # Rule 1: Small corpus -> Flat index
        if num_vectors < 25000:
            return {
                "index_type": "Flat",
                "faiss_string": "IndexFlatIP",
                "estimated_ram_gb": raw_vector_ram_gb,
                "expected_recall": "100%",
                "rationale": "Small corpus size: brute-force exact scan runs in sub-10ms with zero index build overhead."
            }

        # Rule 2: High RAM budget -> HNSW (Best speed & recall)
        hnsw_ram_gb = raw_vector_ram_gb * 1.8 # ~80% extra RAM for edge links
        if hnsw_ram_gb <= available_ram_gb * 0.75: # Keep under 75% system RAM
            return {
                "index_type": "HNSW",
                "faiss_string": "HNSW32",
                "search_params": {"efSearch": 64, "M": 32, "efConstruction": 128},
                "estimated_ram_gb": hnsw_ram_gb,
                "expected_recall": "98.5% - 99.5%",
                "rationale": "Sufficient RAM available: HNSW delivers industry-best sub-5ms query latency and peak recall."
            }

        # Rule 3: Moderate RAM budget -> IVF-Flat
        ivf_nlist = int(4 * (num_vectors ** 0.5)) # Rule of thumb: 4 * sqrt(N)
        ivf_ram_gb = raw_vector_ram_gb * 1.15
        if ivf_ram_gb <= available_ram_gb * 0.80:
            return {
                "index_type": "IVF-Flat",
                "faiss_string": f"IVF{ivf_nlist},Flat",
                "search_params": {"nlist": ivf_nlist, "nprobe": 16},
                "estimated_ram_gb": ivf_ram_gb,
                "expected_recall": "94.0% - 97.0%",
                "rationale": "HNSW exceeds RAM: IVF-Flat reduces memory overhead while maintaining sub-20ms search."
            }

        # Rule 4: Constrained RAM / Massive Scale -> IVF-PQ
        pq_bytes = 64 # Compress to 64 bytes per vector
        ivf_pq_ram_gb = (num_vectors * pq_bytes) / (1024.0 ** 3)
        return {
            "index_type": "IVF-PQ",
            "faiss_string": f"IVF{ivf_nlist},PQ{pq_bytes}",
            "search_params": {"nlist": ivf_nlist, "nprobe": 32},
            "estimated_ram_gb": ivf_pq_ram_gb,
            "expected_recall": "88.0% - 93.0%",
            "rationale": "Memory wall reached: Product Quantization compresses vectors by 10x-15x to prevent OOM crashes."
        }
''', language="python")

st.divider()

# Section 3: Interactive Vector Index Selector Studio
st.header("3. Interactive Workbench: The Automated Index Selector Studio", divider="rainbow")

st.markdown("""
Input your production requirements (vector count, dimensions, available server RAM) and let the engine calculate the **optimal index architecture**, RAM footprint, and build time.
""")

col_in1, col_in2 = st.columns([1, 1])

with col_in1:
    st.subheader("1. Production Workload Specifications")
    target_vectors = st.number_input(
        "Corpus Vector Count ($N$):",
        min_value=5000,
        max_value=50000000,
        value=1000000,
        step=50000
    )

    target_dim = st.selectbox(
        "Vector Dimensions ($D$):",
        [384, 768, 1536, 3072],
        index=2
    )

    server_ram = st.slider(
        "Available Server RAM (GB):",
        min_value=2,
        max_value=128,
        value=16,
        step=2
    )

with col_in2:
    st.subheader("2. Recommended Vector Index Architecture")

    raw_ram = (target_vectors * target_dim * 4.0) / (1024.0**3)
    hnsw_ram = raw_ram * 1.85
    ivf_ram = raw_ram * 1.15
    pq_ram = (target_vectors * 64.0) / (1024.0**3)

    # Decision tree
    if target_vectors < 20000:
        rec_type = "Flat Index (Brute Force)"
        rec_faiss = "IndexFlatIP"
        rec_ram = raw_ram
        rec_recall = "100.0% (Exact)"
        rec_lat = "< 5 ms"
        rec_status = "Optimal"
        explanation = "Corpus fits comfortably in cache; brute force exact scan has 0s build time."
    elif hnsw_ram <= server_ram * 0.70:
        rec_type = "HNSW (Hierarchical Navigable Small World)"
        rec_faiss = "HNSW32"
        rec_ram = hnsw_ram
        rec_recall = "98.5% - 99.5%"
        rec_lat = "2 - 5 ms"
        rec_status = "Top Tier Performance"
        explanation = "Server RAM is ample. HNSW guarantees blazing sub-5ms latency and peak recall."
    elif ivf_ram <= server_ram * 0.80:
        rec_type = "IVF-Flat (Inverted File Index)"
        nlist = int(4 * (target_vectors**0.5))
        rec_faiss = f"IVF{nlist},Flat"
        rec_ram = ivf_ram
        rec_recall = "93.0% - 96.0%"
        rec_lat = "10 - 25 ms"
        rec_status = "Memory Balanced"
        explanation = "HNSW would exhaust RAM. IVF-Flat partitions vectors into Voronoi cells to fit in memory."
    else:
        rec_type = "IVF-PQ (Inverted File + Product Quantization)"
        nlist = int(4 * (target_vectors**0.5))
        rec_faiss = f"IVF{nlist},PQ64"
        rec_ram = pq_ram
        rec_recall = "88.0% - 93.0%"
        rec_lat = "6 - 15 ms"
        rec_status = "Compressed Heavy Scale"
        explanation = "Massive dataset! PQ vector compression allows holding millions of vectors in modest RAM."

    st.success(f"Recommended Index: **{rec_type}**")
    st.caption(f"FAISS Pipeline String: `{rec_faiss}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Index RAM", f"{rec_ram:.2f} GB", delta=f"{server_ram - rec_ram:.1f} GB Free")
    m2.metric("Expected Recall", rec_recall)
    m3.metric("Expected Latency", rec_lat)

    st.markdown("#### Index Comparison Across All 4 Families")
    df_compare = pd.DataFrame([
        {"Index Family": "Flat (Brute Force)", "RAM Required": f"{raw_ram:.2f} GB", "Query Latency": f"~{target_vectors * 0.00045:.1f} ms", "Recall": "100%", "Build Time": "Instant (0s)"},
        {"Index Family": "IVF-Flat", "RAM Required": f"{ivf_ram:.2f} GB", "Query Latency": "~12 - 25 ms", "Recall": "94%", "Build Time": "Fast (~1 min)"},
        {"Index Family": "HNSW", "RAM Required": f"{hnsw_ram:.2f} GB", "Query Latency": "~2 - 5 ms", "Recall": "99%", "Build Time": "Slow (~15 min)"},
        {"Index Family": "IVF-PQ", "RAM Required": f"{pq_ram:.2f} GB", "Query Latency": "~8 - 15 ms", "Recall": "90%", "Build Time": "Moderate (~5 min)"}
    ])
    st.dataframe(df_compare, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Index Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does HNSW require up to 80% more RAM than the raw vectors themselves?**
    * **Answer**: In addition to storing the raw vector coordinates at Layer 0, HNSW maintains a dense bidirectional graph structure.
    * For each node, it stores a list of up to $M$ integer neighbor pointers (e.g. $M=32$ or $M=64$ edges per node).
    * Furthermore, nodes elevated into upper layers (Layer 1, Layer 2) maintain additional skip-list edge lists. Storing millions of graph edges and node lists consumes tens of gigabytes of secondary pointer memory in RAM.

    **Q2: What is the parameter `nprobe` in an IVF index and how does it balance recall vs. latency?**
    * **Answer**: In an IVF index, space is partitioned into `nlist` Voronoi cells (e.g. `nlist = 1024`).
    * When a query arrives, it finds the single closest Voronoi centroid. If `nprobe = 1`, it only searches vectors inside that single cell (blazing fast, but risks missing neighbors across cell borders).
    * Increasing `nprobe` (e.g. `nprobe = 16` or `32`) searches the top 32 closest Voronoi cells. Higher `nprobe` increases recall toward 98%+ at the cost of linearly inspecting more candidates.

    **Q3: When should an enterprise choose IVF-PQ over HNSW?**
    * **Answer**:
      1. When dataset size exceeds **50 million to 1 billion vectors**. Storing 1 billion 1536-dim vectors in HNSW requires over **10 Terabytes of RAM** (astronomically expensive).
      2. In contrast, IVF-PQ compresses each vector to 64 bytes, allowing 1 billion vectors to fit in ~64 GB of RAM on a single modest cloud server.
      3. Acceptable tradeoff: An application where 90% recall is acceptable (e.g. broad e-commerce candidate generation) rather than mission-critical legal compliance.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Building an IVF Index Without Representative Training Data**:
      * *Issue*: Training IVF K-Means centroids on a tiny 500-vector dataset and then inserting 1,000,000 vectors.
      * *Consequence*: The Voronoi centroids are heavily biased, resulting in highly skewed posting lists and terrible recall.
      * *Fix*: Train IVF centroids on at least 100,000 to 250,000 representative vectors.

    * **Trap 2: Setting `M` (Max Edges) Too High in HNSW**:
      * *Issue*: Setting $M=128$ in HNSW to maximize recall.
      * *Consequence*: Index build time surges by 5x and graph memory overhead doubles, while Recall@10 only improves from 99.1% to 99.4%.
      * *Fix*: Standardize $M=16$ or $M=32$ for optimal balance.

    * **Trap 3: Forgetting That IVF Indexes Require Re-training on Heavy Domain Drift**:
      * *Issue*: Corpus topics shift from financial news to medical papers over time without updating IVF centroids.
      * *Consequence*: New medical vectors are shoehorned into distant financial centroids, degrading retrieval accuracy.
      * *Fix*: Periodically re-cluster and rebuild IVF indexes when corpus distribution changes.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Hierarchical Navigable Small World (HNSW)**
      * *Definition*: A multi-layer proximity graph index inspired by skip-lists providing the fastest query latency and highest recall in modern vector databases.
    * **Flashcard 2: Inverted File Index (IVF)**
      * *Definition*: An index that partitions space into Voronoi cells via K-Means clustering, restricting search to `nprobe` candidate cells.
    * **Flashcard 3: Product Quantization (PQ)**
      * *Definition*: Slicing high-dimensional vectors into smaller sub-vectors and quantizing them to codebook byte indices for massive memory compression.
    * **Flashcard 4: Flat Index**
      * *Definition*: An unindexed, exact brute-force vector array guaranteeing 100% ground-truth recall with zero build time.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Vector Indexes:

- [ ] I can articulate the architectural mechanics of Flat, IVF, HNSW, and PQ indexes.
- [ ] I understand how the multi-layer skip graph functions in HNSW.
- [ ] I know how `nlist` and `nprobe` govern speed and recall in IVF indexes.
- [ ] I can explain why HNSW consumes 1.5x - 2.5x more RAM than raw vectors.
- [ ] I can select the optimal index architecture based on vector count, dimensions, and RAM budget.
""")
