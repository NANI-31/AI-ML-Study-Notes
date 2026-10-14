import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class AlgorithmTraversalSimulator:
    """
    Simulates graph hops in HNSW and cluster probing in IVF to visualize
    how different indexing algorithms navigate high-dimensional spaces.
    """
    @classmethod
    def simulate_hnsw_walk(cls, layers: int = 3, ef_search: int = 16) -> List[Dict[str, Any]]:
        hops = []
        current_node = "Entry_Node_L2"
        for layer in range(layers - 1, -1, -1):
            hop_count = max(2, int(ef_search / (layer + 1)))
            hops.append({
                "Layer": f"Layer {layer}",
                "Starting Node": current_node,
                "Nodes Evaluated": hop_count,
                "Mechanism": "Long-range skip link" if layer > 0 else "Dense local neighborhood",
                "Best Node Reached": f"Node_L{layer}_Target"
            })
            current_node = f"Node_L{layer}_Target"
        return hops

    @classmethod
    def simulate_ivf_probe(cls, nlist: int = 100, nprobe: int = 4, total_vectors: int = 100000) -> Dict[str, Any]:
        vecs_per_cell = int(total_vectors / nlist)
        total_scanned = vecs_per_cell * nprobe
        scan_percentage = round((total_scanned / total_vectors) * 100.0, 2)
        return {
            "Total Centroids (nlist)": nlist,
            "Probed Centroids (nprobe)": nprobe,
            "Average Vectors per Cell": vecs_per_cell,
            "Vectors Evaluated": f"{total_scanned:,}",
            "Pruning Efficiency": f"{100.0 - scan_percentage}% pruned",
            "Estimated Recall": f"{min(99.0, 75.0 + (nprobe / nlist) * 200.0):.1f}%"
        }


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="9.2 Index Algorithms - RAG Master Curriculum",
    page_icon="⚙️",
    layout="wide"
)

st.title("9.2 Index Algorithms: HNSW, IVF, Product Quantization, Flat Baselines, and Hybrid ANN Architectures")
st.caption("Chapter 9: Vector Indexing and Search | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Approximate Nearest Neighbor (ANN) search algorithms are the mathematical engines powering modern vector databases. 
Rather than scanning every coordinate vector in memory, these algorithms organize high-dimensional spaces 
into **navigable skip-graphs (HNSW)**, **Voronoi clustering partitions (IVF)**, and **compressed subspace codebooks (PQ)**.

Understanding the internal mechanics of these algorithms is required to tune vector search speed, 
memory efficiency, and recall precision in enterprise RAG systems.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Taxonomy of Modern Vector Indexing Algorithms", divider="rainbow")
st.markdown("""
The architecture below details how the primary indexing algorithms structure search spaces 
and execute candidate vector traversal.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    algo_bp_dot = """
    digraph AlgorithmTaxonomy {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_graph {
            label = "1. Graph-Based (HNSW)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            hnsw_top [label="Layer 2: Sparse Top Layer\\nLong-range highway skips", fillcolor="#0F172A", color="#38BDF8"];
            hnsw_mid [label="Layer 1: Intermediate Layer\\nClustering neighborhoods", fillcolor="#0F172A", color="#38BDF8"];
            hnsw_bot [label="Layer 0: Dense Bottom Layer\\nAll vectors + high local connectivity", fillcolor="#0F172A", color="#10B981"];

            hnsw_top -> hnsw_mid [label="Greedy Descent"];
            hnsw_mid -> hnsw_bot [label="Local Greedy Search"];
        }

        subgraph cluster_cluster {
            label = "2. Partitioning (IVF)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            ivf_centroids [label="K-Means Voronoi Centroids\\n(nlist Centroid Vectors)", fillcolor="#334155", color="#F59E0B"];
            ivf_probe [label="Nearest Centroid Selection\\n(nprobe Closest Centroids)", fillcolor="#334155", color="#F59E0B"];
            ivf_lists [label="Inverted Buckets\\nScan only vectors in target cells", fillcolor="#334155", color="#10B981"];

            ivf_centroids -> ivf_probe -> ivf_lists;
        }

        subgraph cluster_quant {
            label = "3. Quantization (PQ / SQ)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            pq_split [label="Subspace Decomposition\\nSplit 1536 dims into M=64 subspaces", fillcolor="#334155", color="#8B5CF6"];
            pq_codebook [label="Codebook Centroids (256 per subspace)\\nMap each sub-vector to 8-bit byte", fillcolor="#334155", color="#8B5CF6"];
            pq_adc [label="Asymmetric Distance (ADC)\\nPrecomputed lookup table, zero decoding", fillcolor="#334155", color="#10B981"];

            pq_split -> pq_codebook -> pq_adc;
        }
    }
    """
    st.graphviz_chart(algo_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production, algorithms are rarely used in isolation for large collections. 
State-of-the-art vector engines employ **hybrid combinations**:
1. **IVF-PQ (Inverted File with Product Quantization)**: Used in FAISS and Milvus to index 100M+ vectors on a single machine by combining coarse cluster partitioning with 85% memory compression.
2. **HNSW-SQ (HNSW with Scalar Quantization)**: Standard in **Qdrant** and **Pinecone**, combining the rapid logarithmic graph navigation of HNSW with Int8 coordinate quantization, cutting RAM in half while preserving 97%+ recall.
3. **SCaNN (Google)**: Uses anisotropic vector quantization, penalizing parallel quantization error to maximize inner product recall.
""")

st.divider()

# ==============================================================================
# SECTION 2: HNSW
# ==============================================================================

st.header("2. HNSW (Hierarchical Navigable Small World)", divider="rainbow")
st.markdown("""
**HNSW** is currently the gold-standard algorithm for vector similarity search, 
offering superior query throughput (QPS) and recall across diverse high-dimensional datasets.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Mechanics")
    st.markdown("""
    * **Multi-Layer Skip-List Concept**:
      HNSW generalizes 1D Skip-Lists to multi-dimensional graphs.
      * **Top Layers (Sparse)**: Contain a few long-range nodes, acting as highways that span the entire vector space in a few hops.
      * **Bottom Layers (Dense)**: Layer 0 contains all vectors in the collection with dense local neighbor connections.
    * **Greedy Downhill Traversal**:
      Search starts at a fixed entry point in the highest layer, greedily hops to the neighbor closest to the query vector, 
      and drops down to the next layer once a local optimum is reached.
    * **Key Hyperparameters**:
      * `M`: Maximum number of bidirectional links per node (typically 16 to 64). Higher M increases recall and build time.
      * `efConstruction`: Size of candidate priority queue during index building (typically 64 to 200).
      * `efSearch`: Size of candidate priority queue during query time (typically 16 to 128).
    """)

with col2:
    st.subheader("HNSW Multi-Layer Diagram")
    hnsw_dot = """
    digraph HNSWLayers {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4];
        node [shape=circle, style="filled", fontname="Helvetica", fontsize=8, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8", width=0.4];
        edge [fontname="Helvetica", fontsize=7, fontcolor="#94A3B8", color="#38BDF8"];

        subgraph cluster_l2 {
            label = "Layer 2 (Highway Skips)";
            color = "#334155";
            style = "rounded";
            a2 [label="A", fillcolor="#EC4899"];
            c2 [label="C", fillcolor="#EC4899"];
            a2 -> c2 [label="Skip Hop", penwidth=2.0];
        }

        subgraph cluster_l1 {
            label = "Layer 1 (Intermediate)";
            color = "#334155";
            style = "rounded";
            a1 [label="A"];
            b1 [label="B"];
            c1 [label="C"];
            a1 -> b1 -> c1;
        }

        subgraph cluster_l0 {
            label = "Layer 0 (All Vectors - Dense)";
            color = "#334155";
            style = "rounded";
            a0 [label="A"];
            b0 [label="B"];
            c0 [label="C"];
            d0 [label="D"];
            e0 [label="E"];
            a0 -> b0 -> c0;
            b0 -> d0;
            c0 -> e0;
        }

        a2 -> a1 [style=dashed, color="#94A3B8"];
        c2 -> c1 [style=dashed, color="#94A3B8"];
        a1 -> a0 [style=dashed, color="#94A3B8"];
        c1 -> c0 [style=dashed, color="#94A3B8"];
    }
    """
    st.graphviz_chart(hnsw_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: IVF
# ==============================================================================

st.header("3. IVF (Inverted File Index and Voronoi Partitions)", divider="rainbow")
st.markdown("""
**IVF (Inverted File Index)** is a space-partitioning indexing algorithm that clusters vectors 
into Voronoi cells to prune non-relevant vectors during search.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("How IVF Operates")
    st.markdown("""
    1. **Training Phase (Clustering)**:
       Uses k-means clustering to divide the high-dimensional space into `nlist` clusters (e.g., 1,024 or 4,096 centroids).
    2. **Inverted List Ingestion**:
       Every vector in the dataset is assigned to its nearest centroid. The index stores an inverted list of vector IDs for each centroid.
    3. **Query Execution**:
       * The query vector is compared against all `nlist` centroids.
       * The algorithm selects the `nprobe` closest centroids.
       * Only vectors inside those `nprobe` inverted buckets are evaluated.
    """)

with col2:
    st.subheader("Key Tradeoffs of IVF")
    st.markdown("""
    * **Pruning Efficiency**:
      If `nlist = 1000` and `nprobe = 10`, the search evaluates only **1% of all vectors in the collection**, achieving a 100x speedup.
    * **The Boundary Problem**:
      Vectors near the edge of a Voronoi cell can have true nearest neighbors across the boundary in an adjacent unprobed cell. Increasing `nprobe` fixes this but increases latency.
    * **Low Memory Footprint**:
      Unlike HNSW, IVF requires almost zero index memory beyond the centroid coordinates.
    """)

st.divider()

# ==============================================================================
# SECTION 4: PQ
# ==============================================================================

st.header("4. PQ (Product Quantization and Asymmetric Distance)", divider="rainbow")
st.markdown("""
**Product Quantization (PQ)** is a lossy vector compression algorithm that reduces high-dimensional 
floating-point vectors into compact byte arrays while enabling fast distance calculations.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Product Quantization Pipeline")
    st.markdown("""
    1. **Subspace Splitting**:
       A 1,536-dimensional vector is divided into M smaller sub-vectors (e.g., M = 64 sub-vectors, each of dimension 24).
    2. **Subspace Codebook Training**:
       Runs k-means on each subspace to learn 256 centroid prototypes.
    3. **Byte Encoding**:
       Since 256 = 2^8, each 24-dimensional sub-vector is replaced by a single **8-bit byte** (the index of its nearest centroid).
    4. **Memory Reduction**:
       The original vector (`1,536 * 4 bytes = 6,144 bytes`) is compressed into **64 bytes**, achieving a **96x compression ratio**!
    """)

with col2:
    st.subheader("Asymmetric Distance Computation (ADC)")
    st.markdown("""
    When querying with PQ:
    * The query vector remains **uncompressed** (full precision).
    * The engine precalculates distances between the query sub-vectors and all 256 codebook centroids, storing them in a small lookup table.
    * Distance to any stored compressed vector is computed simply by summing M table lookups:
      `Dist(Q, V) = Table_1[V_sub1] + Table_2[V_sub2] + ... + Table_M[V_subM]`.
    * Requires **zero decompression overhead** and runs entirely in L1/L2 cache.
    """)

st.divider()

# ==============================================================================
# SECTION 5: FLAT INDEXES
# ==============================================================================

st.header("5. Flat Indexes: Exact Brute-Force Baselines", divider="rainbow")
st.markdown("""
A **Flat Index** (`IndexFlatL2` or `IndexFlatIP`) performs exhaustive linear scans without approximation.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Baseline Role")
    st.markdown("""
    * **Ground Truth for Evaluation**:
      Every ANN algorithm must be benchmarked against a Flat index to compute empirical Recall@K.
    * **Zero Maintenance Complexity**:
      No training phases, no centroid drift, no graph rebuilding, and zero additional RAM overhead.
    * **Deterministic Reproducibility**:
      Always returns the mathematically identical nearest neighbors for a given query vector.
    """)

with col2:
    st.subheader("Performance Profile")
    st.markdown("""
    * **Latency**: Scales strictly as O(N * d).
    * **Small Scale (< 50k vectors)**: Latency is < 5 ms on modern hardware.
    * **Enterprise Scale (1M+ vectors)**: Latency exceeds 50-100 ms, making it unacceptable for interactive RAG chatbot interfaces.
    """)

st.divider()

# ==============================================================================
# SECTION 6: ANN SEARCH
# ==============================================================================

st.header("6. ANN Search: Comparative Tradeoffs & Hybrid Algorithms", divider="rainbow")
st.markdown("""
Comparing the core properties of vector indexing algorithms across production engineering metrics:
""")

comp_matrix_df = pd.DataFrame([
    {"Algorithm", "Search Latency", "Recall @ 10", "Memory Overhead", "Build Time", "Best Used When"},
    {"Flat", "Slow (50-200ms)", "100.0%", "0% (Raw coordinates)", "Instant (<1s)", "Small datasets (<50k) or ground-truth benchmarking"},
    {"IVF-Flat", "Fast (3-8ms)", "90 - 95%", "~2% (Centroid vectors)", "Fast (seconds)", "Medium datasets where RAM overhead must be minimized"},
    {"HNSW", "Ultra-Fast (1-3ms)", "97 - 99%", "50 - 100% (Graph links)", "Slow (minutes)", "Latency-critical RAG chatbots requiring maximum recall"},
    {"IVF-PQ", "Fast (2-5ms)", "80 - 90%", "-85% (Massive savings)", "Moderate (clustering)", "Tens of millions of vectors on limited RAM budgets"},
    {"HNSW-SQ8", "Ultra-Fast (1-2ms)", "96 - 98%", "-50% (Int8 quantization)", "Moderate", "Enterprise production standard (Qdrant/Pinecone default)"}
])

st.dataframe(comp_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Multi-Layer Hierarchical Graph & Quantized Search Engine", divider="rainbow")
st.markdown("""
Below is an educational implementation demonstrating multi-layer graph skip navigation 
and inverted cluster search in production Python.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Tuple

class SimulatedHNSWEngine:
    """
    Demonstrates multi-layer skip-graph traversal mechanics
    and candidate priority queue exploration.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        # Multi-layer adjacency lists: layer_index -> {node_id -> list_of_neighbor_ids}
        self.layers: List[Dict[str, List[str]]] = [{}, {}]
        self.entry_point: str = ""

    def insert(self, doc_id: str, vector: List[float], top_layer: bool = False) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        self.vectors[doc_id] = v / norm if norm > 0 else v

        # Add to base layer (all vectors)
        if doc_id not in self.layers[0]:
            self.layers[0][doc_id] = []

        # Promote select vectors to top skip layer
        if top_layer:
            self.layers[1][doc_id] = []
            if not self.entry_point:
                self.entry_point = doc_id

    def connect_edge(self, layer: int, node_a: str, node_b: str) -> None:
        self.layers[layer][node_a].append(node_b)
        self.layers[layer][node_b].append(node_a)

    def search_greedy_layer(self, query_vec: np.ndarray, start_node: str, layer: int) -> str:
        """Greedy search for the closest node in a single graph layer."""
        current_node = start_node
        best_sim = float(np.dot(query_vec, self.vectors[current_node]))

        changed = True
        while changed:
            changed = False
            for neighbor in self.layers[layer].get(current_node, []):
                sim = float(np.dot(query_vec, self.vectors[neighbor]))
                if sim > best_sim:
                    best_sim = sim
                    current_node = neighbor
                    changed = True
        return current_node

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        qv = qv / np.linalg.norm(qv)

        if not self.entry_point:
            return []

        # 1. Skip layer highway descent
        current_closest = self.search_greedy_layer(qv, self.entry_point, layer=1)
        # 2. Dense base layer local search
        final_best = self.search_greedy_layer(qv, current_closest, layer=0)

        # Collect neighbors in bottom layer for top-K candidates
        candidates = [final_best] + self.layers[0].get(final_best, [])
        scored = [(c, float(np.dot(qv, self.vectors[c]))) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [{"id": d_id, "score": round(s, 4)} for d_id, s in scored[:top_k]]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Graph Traversal and Voronoi Cell Routing Simulator", divider="rainbow")
st.markdown("""
Interact with the internal mechanics of HNSW and IVF algorithms. 
Inspect how HNSW descends through multi-layer skip graphs and how tuning `nprobe` in IVF balances search recall against vector pruning efficiency.
""")

tab_hnsw, tab_ivf = st.tabs(["HNSW Skip-Graph Explorer", "IVF Voronoi Partition Simulator"])

with tab_hnsw:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("HNSW Parameters")
        ef_val = st.slider("efSearch (Queue Exploration Depth)", 4, 64, 16)
        num_layers_val = st.slider("Graph Hierarchy Layers", 2, 4, 3)

    with col2:
        st.subheader("Simulated Traversal Path")
        hops = AlgorithmTraversalSimulator.simulate_hnsw_walk(layers=num_layers_val, ef_search=ef_val)
        st.dataframe(pd.DataFrame(hops), use_container_width=True, hide_index=True)
        st.caption("Notice how upper layers evaluate very few nodes before handing off the local neighborhood to lower dense layers.")

with tab_ivf:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("IVF Parameters")
        nlist_val = st.slider("Total Voronoi Clusters (nlist)", 50, 500, 100, step=50)
        nprobe_val = st.slider("Centroids to Probe (nprobe)", 1, 20, 4)
        total_vecs_val = st.selectbox("Corpus Vectors", [10000, 50000, 100000, 500000], index=2)

    with col2:
        st.subheader("Pruning & Recall Metrics")
        ivf_res = AlgorithmTraversalSimulator.simulate_ivf_probe(nlist=nlist_val, nprobe=nprobe_val, total_vectors=total_vecs_val)
        ivf_df = pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in ivf_res.items()])
        st.dataframe(ivf_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: How does HNSW prevent greedy search from getting permanently stuck in local optima?"):
    st.markdown("""
    **Answer**:
    HNSW avoids local optima through two key architectural mechanisms:
    1. **Navigable Small World (NSW) Properties**: Edges are constructed using a heuristic that favors both short-range clustering links and long-range shortcuts, maintaining small average path lengths across the graph.
    2. **Multi-Layer Skip Hierarchy**: High layers provide coarse, long-range skip bridges across distant regions of the space. By the time search descends to the dense Layer 0, the traversal is already positioned in the correct immediate semantic basin of attraction.
    """)

with st.expander("Q2: What is the boundary problem in IVF indexes, and how does tuning 'nprobe' solve it?"):
    st.markdown("""
    **Answer**:
    In an IVF index, vectors are assigned strictly to their nearest centroid Voronoi cell. If a query vector falls near the boundary 
    between two Voronoi cells, true nearest neighbors may be located just across the cell border in an adjacent cluster. 
    If `nprobe = 1`, that adjacent cluster is ignored, causing immediate recall loss. 
    Increasing `nprobe` instructs the query coordinator to explore multiple adjacent cluster centroids (e.g., `nprobe = 8`), 
    recovering boundary neighbors at the expense of scanning more vectors.
    """)

with st.expander("Q3: How does Asymmetric Distance Computation (ADC) in Product Quantization work without decompressing vectors?"):
    st.markdown("""
    **Answer**:
    In ADC, the query vector Q is left uncompressed in full floating-point precision, while database vectors V are compressed into M byte codes. 
    The engine splits Q into M sub-vectors and calculates Euclidean distances from each sub-vector to the 256 pre-trained centroids in that subspace codebook. 
    These distances are stored in an M x 256 lookup table. 
    Evaluating the distance from Q to any compressed vector V requires simply summing M precomputed float values from the table using V's byte codes as indices, 
    avoiding all decompression overhead.
    """)

with st.expander("Q4: Why does HNSW consume significantly more memory than IVF-Flat or Flat indexes?"):
    st.markdown("""
    **Answer**:
    In Flat and IVF-Flat indexes, vectors are stored as contiguous float arrays with minimal metadata (e.g., cluster ID pointers). 
    In HNSW, every vector node maintains bidirectional graph links to M neighbors across multiple layers. 
    Storing integer node IDs and pointers for 16 to 64 connections per node introduces 50% to 100% additional memory overhead 
    on top of the raw vector coordinates.
    """)

with st.expander("Q5: What is the advantage of HNSW-SQ8 (Scalar Quantization) over uncompressed HNSW in production?"):
    st.markdown("""
    **Answer**:
    HNSW-SQ8 quantizes Float32 coordinates into 8-bit integers (Int8) using linear scaling between min and max bounds. 
    This cuts the vector storage requirement by 75% (from 4 bytes per dimension to 1 byte), while maintaining the HNSW graph structure. 
    Distances are calculated using integer SIMD vector instructions, which are faster on modern CPUs. 
    The resulting system achieves a 50% reduction in total RAM consumption with less than 2% drop in Recall@10.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Training IVF Centroids on Incomplete or Unrepresentative Data**  
    Running IVF k-means clustering on a small initial sample of 1,000 vectors and then inserting 1,000,000 vectors 
    leads to severe centroid drift. The Voronoi cells become unbalanced, causing query recall to drop drastically.
    """)
    st.error("""
    **Trap 2: Setting HNSW 'M' Too High (e.g., M=128) on High-Dimensional Vectors**  
    Increasing M from 32 to 128 yields negligible recall improvements while doubling index RAM consumption 
    and slowing down index construction by 4x. Standard production values range between 16 and 48.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Setting IVF 'nprobe = 1' in Production Workloads**  
    Probing only a single centroid (`nprobe = 1`) yields blazing-fast search (< 1 ms), but drops recall to 60%-70% 
    due to Voronoi boundary clipping. Always calibrate `nprobe` to balance latency against your target recall threshold.
    """)
    st.warning("""
    **Trap 4: Expecting PQ Alone to Preserve Fine-Grained Semantic Distinctions**  
    Aggressive Product Quantization (e.g., compressing 1536 dims into 32 bytes) degrades distance resolution. 
    Top-ranked documents may be ranked incorrectly unless paired with a re-ranking step using uncompressed vectors.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: HNSW Navigation")
        st.caption("How does traversal work?")
        st.markdown("**Back**: Greedily hops across long-range highway links in sparse upper layers, descending downhill into dense Layer 0 for fine-grained local search.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: IVF nprobe Parameter")
        st.caption("What does nprobe control?")
        st.markdown("**Back**: Number of Voronoi cluster centroids inspected per query. Higher nprobe increases recall but scans more candidate vectors.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: PQ Subspace Division")
        st.caption("How does PQ compress vectors?")
        st.markdown("**Back**: Divides vectors into M subspaces, clusters each subspace into 256 centroids, and represents each sub-vector with an 8-bit byte.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: ADC in PQ")
        st.caption("Why is ADC fast?")
        st.markdown("**Back**: Precomputes query-to-centroid distances in a lookup table. Evaluates stored vectors via table lookups with zero decompression.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Index Algorithms:")

c1 = st.checkbox("I understand the multi-layer skip-graph traversal mechanics of HNSW.", value=True)
c2 = st.checkbox("I understand how IVF partitions high-dimensional space into Voronoi cells and how nprobe affects recall.", value=True)
c3 = st.checkbox("I know how Product Quantization (PQ) achieves 85%+ memory compression and how Asymmetric Distance Computation works.", value=True)
c4 = st.checkbox("I understand the role of Flat exact indices as ground-truth benchmarks.", value=True)
c5 = st.checkbox("I can evaluate hybrid architectures like IVF-PQ and HNSW-SQ8 for enterprise production scaling.", value=True)
c6 = st.checkbox("I can implement a simulated hierarchical graph navigation and candidate priority queue in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Index Algorithms and understand how to architect and tune graph, cluster, and quantized indexes for production RAG systems.")
