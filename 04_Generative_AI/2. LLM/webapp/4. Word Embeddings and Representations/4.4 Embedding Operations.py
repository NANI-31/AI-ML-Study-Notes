import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.4 Embedding Operations (Cosine, Euclidean, Dot Product, ANN, Clustering, Visualization)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .takeaway-box {
        background-color: #0B132B;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #1C1917;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .modern-callout {
        background-color: #042F2E;
        border-left: 4px solid #14B8A6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-title">4.4 Embedding Operations: Geometric Algebra, ANN Indexing, Clustering, and Projection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Metric Formulas to Production Vector Databases (HNSW / IVF-PQ) and Dimensionality Reduction (PCA / UMAP) — The Mathematical Mechanics of High-Dimensional Search.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Geometric Linear Algebra Pipeline for Dense Retrieval", divider="rainbow")

st.markdown("""
Once text is mapped into continuous vectors by an encoder (e.g., BERT, BGE, OpenAI `text-embedding-3`), 
all downstream reasoning—semantic search, retrieval-augmented generation (RAG), topic discovery, and deduplication—is performed 
via **geometric vector space operations**.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("VECS", "Raw Embedding Tensors\nQuery q in R^D, Corpus X in R^[N, D]\nDimension D in [256, 4096]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    
    with arch_graph.subgraph(name="cluster_metrics") as c:
        c.attr(bgcolor="#0F172A", label="Distance & Proximity Metrics", fontcolor="#38BDF8", color="#38BDF8")
        c.node("COS", "Cosine Similarity\nDot(u, v) / (||u|| * ||v||)\nAngular separation on unit sphere", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("L2", "Euclidean (L2) Distance\n||u - v||_2 = sqrt(Sum((u_i - v_i)^2))\nSpatial distance", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("DOT", "Inner Dot Product\nSum(u_i * v_i)\nEqual to Cosine if L2-normalized", shape="box", fillcolor="#1E293B", color="#38BDF8")

    with arch_graph.subgraph(name="cluster_search") as c:
        c.attr(bgcolor="#0F172A", label="Search & Indexing Engine", fontcolor="#A855F7", color="#A855F7")
        c.node("KNN", "Exact Brute-Force k-NN\nO(N * D) compute\n100% recall, high latency", shape="box", fillcolor="#1E293B", color="#A855F7")
        c.node("ANN", "Approximate NN (HNSW / IVF-PQ)\nO(log N) compute\nVector DBs: Pinecone, Qdrant, Milvus", shape="box", fillcolor="#042F2E", color="#14B8A6")

    with arch_graph.subgraph(name="cluster_cluster") as c:
        c.attr(bgcolor="#0F172A", label="Unsupervised Discovery", fontcolor="#10B981", color="#10B981")
        c.node("CLUST", "Clustering (K-Means / HDBSCAN)\nTopic grouping and deduplication", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("VIZ", "Dimensionality Reduction (PCA / UMAP)\nProject R^D -> R^2 / R^3 visualization", shape="box", fillcolor="#312E81", color="#818CF8")

    # Edges
    arch_graph.edge("VECS", "COS")
    arch_graph.edge("VECS", "L2")
    arch_graph.edge("VECS", "DOT")
    arch_graph.edge("COS", "KNN")
    arch_graph.edge("COS", "ANN")
    arch_graph.edge("DOT", "ANN")
    arch_graph.edge("ANN", "CLUST")
    arch_graph.edge("CLUST", "VIZ")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Fundamental Axioms of Vector Space Operations
    * **Direction vs. Magnitude**: In unnormalized spaces, dot product mixes angular semantic alignment with vector length. Normalizing vectors (||v||_2 = 1.0) unifies Cosine Similarity, Dot Product, and Euclidean Distance.
    * **The Scale Frontier (O(N) vs O(log N))**:
      * *Brute Force (Exact k-NN)*: Computes dot products against all N vectors. Perfect recall, but linearly slows down as N exceeds 100,000 vectors.
      * *Approximate Nearest Neighbors (ANN)*: Builds hierarchical graphs (HNSW) or inverted cluster indices (IVF-PQ) to achieve sub-5ms search over 100,000,000+ vectors.
    * **The Curse of Dimensionality**: In high dimensions (D >= 768), Euclidean distances concentrate: the distance between the nearest neighbor and the farthest neighbor approaches a constant ratio, making raw Euclidean metrics sensitive to noise.
    """)

# ==============================================================================
# SECTION 2: 1. COSINE SIMILARITY
# ==============================================================================
st.header("2. Cosine Similarity: Directional Alignment on the Hypersphere", divider="rainbow")

st.markdown("""
**Cosine Similarity** measures the cosine of the angle between two multi-dimensional vectors, 
evaluating whether they point in the same directional orientation regardless of vector magnitude.
""")

cos_col1, cos_col2 = st.columns([1, 1])

with cos_col1:
    st.markdown("""
    ### Mathematical Formulation
    For vectors u, v in R^D:
    ```
    Cosine_Similarity(u, v) = (u . v) / (||u||_2 * ||v||_2)
    ```
    Where:
    * u * v = Sum_{i=1}^{D} (u_i * v_i) is the Euclidean dot product.
    * ||u||_2 = sqrt{Sum_{i=1}^{D} u_i^2} is the L2 Euclidean norm.

    ### Value Spectrum
    * **+1.0 (Parallel)**: Identical semantic direction (theta = 0 deg).
    * **0.0 (Orthogonal)**: Uncorrelated vectors (theta = 90 deg).
    * **-1.0 (Anti-parallel)**: Exactly opposite semantic direction (theta = 180 deg).

    ### Cosine Distance
    While cosine similarity measures closeness (1.0 = \text{identical}), distance metrics require 0.0 = \text{identical}:
    ```
    Cosine_Distance(u, v) = 1.0 - Cosine_Similarity(u, v)
    ```
    """)

with cos_col2:
    st.markdown("""
    ### Why Text Systems Standardize on Cosine
    * **Length Invariance**: A 300-word product description and a 20-word summary of the same product may have drastically different vector magnitudes if pooled naively. Cosine similarity evaluates whether their semantic direction matches, neutralizing verbosity bias.
    * **Isomorphism with Dot Product**: If all embeddings are unit-normalized (||u||_2 = 1.0) upon ingestion:
      ```
      Cosine_Similarity(u, v) = u . v
      ```
      This eliminates expensive denominator divisions during search!
    """)
    st.info("Tip: Always normalize your embeddings immediately after model inference before storing them in your database.")

# ==============================================================================
# SECTION 3: 2. EUCLIDEAN DISTANCE
# ==============================================================================
st.header("3. Euclidean (L2) Distance: Spatial Separation and Magnitude Sensitivity", divider="rainbow")

st.markdown("""
**Euclidean Distance (L2 norm)** calculates the straight-line physical separation between two points in D-dimensional continuous space.
""")

euc_col1, euc_col2 = st.columns([1, 1])

with euc_col1:
    st.markdown("""
    ### Mathematical Formulation
    ```
    d_L2(u, v) = ||u - v||_2 = sqrt(Sum_{i=1}^{D} (u_i - v_i)^2)
    ```
    
    ### Relationship to Cosine Similarity
    When vectors are L2-normalized to the unit hypersphere (||u||_2 = ||v||_2 = 1.0):
    ```
    ||u - v||_2^2 = ||u||_2^2 + ||v||_2^2 - 2 (u . v)
                  = 1 + 1 - 2 Cosine_Similarity(u, v)
                  = 2 * (1 - Cosine_Similarity(u, v))
    ```
    Therefore:
    ```
    ||u - v||_2 = sqrt(2 * (1 - Cosine_Similarity(u, v)))
    ```
    """)

with euc_col2:
    st.markdown("""
    ### Metric Equivalence Table on Normalized Embeddings
    """)
    df_metric_eq = pd.DataFrame({
        "Angle (degrees)": ["0° (Identical)", "30°", "60°", "90° (Orthogonal)", "180° (Opposite)"],
        "Cosine Similarity": ["1.0000", "0.8660", "0.5000", "0.0000", "-1.0000"],
        "Cosine Distance (1 - Cos)": ["0.0000", "0.1340", "0.5000", "1.0000", "2.0000"],
        "Euclidean L2 Distance": ["0.0000", "0.5176", "1.0000", "1.4142 (sqrt(2))", "2.0000"],
        "Dot Product (Normalized)": ["1.0000", "0.8660", "0.5000", "0.0000", "-1.0000"]
    })
    st.dataframe(df_metric_eq, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 4: 3. DOT PRODUCT
# ==============================================================================
st.header("4. Dot Product: Unnormalized Alignment and Vector Length Bias", divider="rainbow")

st.markdown("""
The **Dot Product (Inner Product)** is the most computationally efficient algebraic operation in deep learning hardware, 
powering hardware tensor cores and GPU matrix multiplications (`cublasSgemm`).
""")

dot_col1, dot_col2 = st.columns([1, 1])

with dot_col1:
    st.markdown("""
    ### Mathematical Formulation
    ```
    Dot_Product(u, v) = u . v = Sum_{i=1}^{D} u_i * v_i
    ```
    Geometrically:
    ```
    u . v = ||u||_2 * ||v||_2 * cos(theta)
    ```
    
    ### The Frequency / Length Bias Trap
    If embeddings are **not normalized**, the dot product heavily rewards vectors with large magnitudes:
    * Frequent words and longer documents often acquire larger embedding norms during training.
    * An irrelevant document with ||v|| = 10.0 and low cosine similarity (cos(theta) = 0.3) yields u . v = 3.0.
    * A highly relevant document with ||v|| = 2.0 and high cosine similarity (cos(theta) = 0.9) yields u . v = 1.8.
    * The unnormalized search engine incorrectly ranks the irrelevant document higher!
    """)

with dot_col2:
    st.markdown("""
    ### When to Use Raw Dot Product in Modern LLMs:
    1. **Self-Attention Query-Key Scores**:
       In Transformer attention, the dot product Q K^T / sqrt(d_k) intentionally allows vector magnitude to express confidence.
    2. **Token Logit Generation**:
       The final LM head computes `logits = hidden_state @ W_vocab.T`. Unnormalized dot products allow frequent tokens to maintain higher prior probabilities.
    3. **Learned Magnitude Retrieval Models**:
       Models like `ColBERT` and `DensePassageRetriever` (DPR) explicitly train without L2 normalization, allowing token weights to encode importance.
    """)

# ==============================================================================
# SECTION 5: 4. NEAREST-NEIGHBOR SEARCH (k-NN & ANN)
# ==============================================================================
st.header("5. Nearest-Neighbor Search: From Exact Brute Force to Production ANN", divider="rainbow")

st.markdown("""
Given a query vector q in R^D and a database of N document vectors X in R^[N, D], 
the retrieval goal is to identify the k most similar vectors.
""")

ann_col1, ann_col2 = st.columns([1, 1])

with ann_col1:
    st.markdown("""
    ### 1. Exact k-NN (Flat Brute Force)
    * Computes distance between query q and every single vector x_i in X.
    * Time Complexity: O(N * D).
    * Memory Complexity: O(N * D).
    * **Recall**: 100% (Guaranteed exact ground truth).
    * **Bottleneck**: At N = 10,000,000 and D = 1536, searching requires 61.4 billion floating-point operations per query (approx 150-300ms on server CPUs).

    ### 2. Approximate Nearest Neighbors (ANN)
    Sacrifices <1-2% recall in exchange for **100x to 1000x speedups**:
    * **HNSW (Hierarchical Navigable Small World)**:
      Multi-layer proximity graph where upper layers skip long distances and bottom layers perform fine-grained greedy search (O(log N) search time). The gold standard for low latency.
    * **IVF (Inverted File Index)**:
      Partitions vector space into Voronoi cells using K-Means. At query time, only searches vectors inside the n_{probe} closest centroids.
    * **PQ (Product Quantization)**:
      Splits D-dimensional vectors into M sub-vectors and quantizes each to codebook centroids, compressing vector memory by 16x-64x.
    """)

with ann_col2:
    st.markdown("""
    ### Production ANN Algorithms Comparison
    """)
    df_ann = pd.DataFrame({
        "Algorithm Family": ["Flat (Brute Force)", "HNSW (Graph-based)", "IVF-Flat (Clustering)", "IVF-PQ (Quantization)", "ScaNN (Anisotropic PQ)"],
        "Search Latency": ["150ms+ (Slow, O(N))", "1-3ms (Ultra-fast, O(log N))", "5-15ms (Fast)", "2-8ms (Fast)", "1-4ms (Ultra-fast)"],
        "Recall @ 10": ["100% (Exact)", "98-99.5%", "92-97%", "85-93%", "95-98%"],
        "Memory Footprint": ["1x (Raw vectors)", "1.5x - 2.5x (Graph links)", "1.1x (Centroids)", "0.05x - 0.1x (64x compression)", "0.1x - 0.2x"],
        "Standard Library": ["FAISS IndexFlatIP", "Hnswlib / FAISS / Qdrant", "FAISS IndexIVFFlat", "FAISS IndexIVFPQ", "Google ScaNN"]
    })
    st.dataframe(df_ann, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 6: 5. SEMANTIC SIMILARITY & HUBNESS
# ==============================================================================
st.header("6. Semantic Similarity: Threshold Calibration and the Hubness Trap", divider="rainbow")

st.markdown("""
Converting continuous cosine similarity scores ([-1.0, 1.0]) into real-world business actions 
(e.g., *"Is this document a duplicate?"* or *"Should this customer question trigger an automated refund?"*) 
requires addressing **metric calibration** and the **Hubness Phenomenon**.
""")

sem_col1, sem_col2 = st.columns([1, 1])

with sem_col1:
    st.markdown("""
    ### Calibrating Semantic Similarity Thresholds
    A common novice mistake is assuming \text{Cosine} > 0.5 means two sentences are similar. 
    Because modern encoders map texts into compact directional cones (anisotropy), similarity scores are skewed:
    
    * **Cosine > 0.92**: Near-verbatim paraphrase or duplicate statement.
    * **Cosine 0.82 - 0.92**: Strong semantic equivalence; same subject and intent.
    * **Cosine 0.70 - 0.82**: Thematic topical relevance; related concepts but different facts.
    * **Cosine 0.55 - 0.70**: Loose contextual association (same broad industry or genre).
    * **Cosine < 0.55**: Irrelevant or disjoint concepts.
    """)

with sem_col2:
    st.markdown("""
    ### The Hubness Problem in High Dimensions
    In high-dimensional spaces (D >= 768), geometric distortion causes certain outlier vectors—called **Hubs**—to appear as the nearest neighbor to thousands of unrelated queries.
    
    * **Mechanism**: High-dimensional variance asymmetries cause the global mean vector to be geometrically closer to the center of the manifold than peripheral points.
    * **Symptom in Production**: An FAQ answering *"How to update payment settings"* repeatedly ranks as top-3 for queries about *"How to change account password"* and *"Where is my order"*.
    * **Production Remedy**:
      1. **Mean Centering & Whitening**: Subtract the global corpus mean vector mu = (1 / N) * Sum(x_i) before indexing.
      2. **Cross-Encoder Reranking**: Filter top-50 results through a cross-encoder to eliminate false hubs.
    """)

# ==============================================================================
# SECTION 7: 6. CLUSTERING EMBEDDINGS
# ==============================================================================
st.header("7. Clustering Embeddings: K-Means, Agglomerative, and Density Partitioning", divider="rainbow")

st.markdown("""
Unsupervised clustering of dense embeddings enables **topic modeling**, **corpus deduplication**, 
and **curation of diverse pre-training datasets** for LLMs.
""")

clust_col1, clust_col2 = st.columns([1, 1])

with clust_col1:
    st.markdown("""
    ### Dominant Clustering Paradigms
    1. **Spherical K-Means**:
       * Modifies classical K-Means to re-normalize centroids to unit length (||c_k|| = 1.0) at every iteration.
       * Maximizes cosine similarity rather than minimizing Euclidean distance.
       * Fast (O(K * N * D)), perfect for large-scale document categorization.
    2. **Hierarchical Agglomerative Clustering (HAC)**:
       * Builds a bottom-up tree of clusters using cosine affinity.
       * Generates multi-level taxonomic trees (e.g., Science -> Physics -> Quantum Electrodynamics).
       * Computationally expensive: O(N^2 log N) without spatial indexing.
    3. **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**:
       * Groups points based on local density and explicitly identifies **noise / outliers** (-1 label).
       * Does not force every point into a cluster, making it ideal for discovering organic conversation topics.
    """)

with clust_col2:
    st.markdown("""
    ### Modern LLM Pipeline Application: Synthetic Data Filtering
    """)
    df_clust_use = pd.DataFrame({
        "LLM Pipeline Stage": ["Pre-training Data Curation", "RAG Knowledge Base Deduplication", "User Query Topic Modeling", "Instruction Tuning Curation"],
        "Clustering Strategy": ["Spherical K-Means (K=10,000)", "DBSCAN / Cosine Threshold (0.95)", "HDBSCAN on UMAP Projections", "K-Means Diversity Subsampling"],
        "Objective": [
            "Balance web corpus across balanced domains (math, code, news)",
            "Remove duplicate scraped articles from enterprise vector store",
            "Detect sudden spikes in user complaints or novel error queries",
            "Select 50,000 maximally diverse instruction prompts for SFT"
        ]
    })
    st.dataframe(df_clust_use, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 8: 7. EMBEDDING VISUALIZATION
# ==============================================================================
st.header("8. Embedding Visualization: Dimensionality Reduction via PCA, t-SNE, and UMAP", divider="rainbow")

st.markdown("""
Because humans cannot visually perceive 1024-dimensional manifolds, 
**dimensionality reduction algorithms** project vectors into 2D or 3D coordinate planes for qualitative inspection.
""")

viz_col1, viz_col2 = st.columns([1, 1])

with viz_col1:
    st.markdown("""
    ### The Three Dominant Reduction Algorithms
    1. **PCA (Principal Component Analysis)**:
       * **Linear Projection**: Computes eigenvectors of the covariance matrix (X^T X).
       * **Preserves**: Global variance and macro-structure.
       * **Flaw**: Cannot unroll complex non-linear manifolds; distinct clusters collapse onto each other.
       * **Speed**: Extremely fast, deterministic.
    2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
       * **Probabilistic Local Matching**: Models pairwise similarities as Gaussian distributions in high-D and Student-t in 2D.
       * **Preserves**: Fine-grained local neighborhood clusters.
       * **Flaw**: Destroys global distances (cluster-to-cluster distance is meaningless); non-deterministic; slow (O(N^2)).
    3. **UMAP (Uniform Manifold Approximation and Projection)**:
       * **Riemannian Geometry & Fuzzy Simplicial Sets**: Modern gold standard.
       * **Preserves**: Both local cluster cohesion and global inter-cluster relationships.
       * **Speed**: Significantly faster than t-SNE (O(N log N)), scales to millions of vectors.
    """)

with viz_col2:
    st.markdown("""
    ### Dimensionality Reduction Comparison Matrix
    """)
    df_dim_comp = pd.DataFrame({
        "Attribute": ["Type", "Global Geometry Preservation", "Local Cluster Preservation", "Algorithmic Speed", "Deterministic", "Production Primary Use"],
        "PCA": ["Linear Orthogonal", "Excellent", "Poor", "Extremely Fast", "Yes", "Preprocessing / Noise Filtering"],
        "t-SNE": ["Non-linear Probabilistic", "Poor (Distorted)", "Excellent", "Slow", "No (Seed dependent)", "Qualitative Research Inspection"],
        "UMAP": ["Non-linear Topological", "Good to Excellent", "State of the Art", "Fast", "No (Seed dependent)", "Topic Modeling (BERTopic), Visualization"]
    })
    st.dataframe(df_dim_comp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Pure Python Vector Index, Exact k-NN, and PCA Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. An **Exact k-NN Vector Index** supporting Cosine, Euclidean, and Dot Product distance metrics.
2. A **Pure-Numpy Spherical K-Means Clustering Algorithm**.
3. A **Pure-Numpy Principal Component Analysis (PCA) 2D Projection Engine**.
""")

st.code('''
import numpy as np
from typing import List, Dict, Tuple, Optional

class VectorMathEngine:
    """
    Production-grade educational linear algebra engine demonstrating:
    1. Vector distance metrics (Cosine, Euclidean, Dot Product).
    2. Exact brute-force Top-K search.
    3. Spherical K-Means clustering.
    4. Principal Component Analysis (PCA) projection to 2D.
    """
    def __init__(self, vectors: np.ndarray, documents: Optional[List[str]] = None):
        """
        Args:
            vectors: Matrix of shape [N, D]
            documents: Optional list of N text labels/documents
        """
        self.vectors = vectors.astype(np.float32)
        self.N, self.D = self.vectors.shape
        self.documents = documents if documents is not None else [f"Doc_{i}" for i in range(self.N)]
        
        # Precompute normalized vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        self.normalized_vectors = self.vectors / norms

    def search_exact(self, query: np.ndarray, top_k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        """Performs exact brute-force Top-K nearest neighbor search."""
        q = query.astype(np.float32).flatten()
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        
        if metric == "cosine":
            # Dot product with pre-normalized corpus vectors
            scores = np.dot(self.normalized_vectors, q_norm)
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.documents[i], float(scores[i])) for i in ranked_indices]
            
        elif metric == "euclidean":
            # L2 Euclidean distance: sqrt(sum((x - y)^2))
            diffs = self.vectors - q
            distances = np.linalg.norm(diffs, axis=1)
            ranked_indices = np.argsort(distances)[:top_k]
            return [(self.documents[i], float(distances[i])) for i in ranked_indices]
            
        elif metric == "dot_product":
            scores = np.dot(self.vectors, q)
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.documents[i], float(scores[i])) for i in ranked_indices]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def spherical_kmeans(self, k: int = 3, max_iter: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Spherical K-Means on unit-normalized vectors.
        Returns (cluster_labels, normalized_centroids).
        """
        rng = np.random.RandomState(seed)
        init_indices = rng.choice(self.N, size=k, replace=False)
        centroids = self.normalized_vectors[init_indices].copy()
        
        labels = np.zeros(self.N, dtype=np.int32)
        for _ in range(max_iter):
            # Assign each vector to closest centroid via cosine similarity
            sims = np.dot(self.normalized_vectors, centroids.T)
            new_labels = np.argmax(sims, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update centroids to mean of assigned vectors and re-normalize
            for j in range(k):
                assigned = self.normalized_vectors[labels == j]
                if len(assigned) > 0:
                    mean_v = np.mean(assigned, axis=0)
                    centroids[j] = mean_v / (np.linalg.norm(mean_v) + 1e-9)
                    
        return labels, centroids

    def project_pca_2d(self) -> np.ndarray:
        """Projects N x D vectors down to N x 2 via Principal Component Analysis."""
        # Mean center the data
        centered = self.vectors - np.mean(self.vectors, axis=0)
        
        # Compute Singular Value Decomposition (SVD)
        # centered = U @ S @ Vt
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        
        # Top 2 principal components
        top_components = vt[:2].T # Shape: [D, 2]
        projected = np.dot(centered, top_components) # Shape: [N, 2]
        return projected
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Vector Geometry & Metric Comparison Lab", divider="rainbow")

st.markdown("""
Compare how **Cosine Similarity**, **Euclidean Distance**, and **Dot Product** rank documents in real-time, 
and inspect how vectors with different magnitudes alter ranking outcomes.
""")

wb_mode = st.radio("Select Interactive Workbench Mode", ["Metric Ranking Simulator", "PCA 2D Cluster Explorer"], horizontal=True)

if wb_mode == "Metric Ranking Simulator":
    st.markdown("#### The Magnitude Trap: Unnormalized Dot Product vs Cosine")
    
    # Setup test vectors with varying magnitudes
    doc_catalog = [
        {"name": "Doc A (Short, Perfect Match)", "vector": np.array([1.0, 1.0, 0.0, 0.0]) * 1.0, "desc": "Aligned angle, low magnitude"},
        {"name": "Doc B (Long, Poor Match)", "vector": np.array([1.0, 0.1, 1.5, 1.2]) * 4.0, "desc": "Poor angle, massive magnitude"},
        {"name": "Doc C (Medium, Good Match)", "vector": np.array([1.0, 0.8, 0.2, 0.0]) * 2.0, "desc": "Good angle, moderate magnitude"},
        {"name": "Doc D (Opposite Topic)", "vector": np.array([-1.0, -1.0, 0.1, 0.0]) * 1.5, "desc": "Negative angle"}
    ]
    
    query_vec = np.array([1.0, 1.0, 0.0, 0.0]) # Perfect match to Doc A
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        st.markdown("**Query Vector Direction**: `[1.0, 1.0, 0.0, 0.0]`")
        selected_metric = st.selectbox("Search Metric", ["Cosine Similarity", "Euclidean (L2) Distance", "Raw Dot Product"])
        
    scores_list = []
    for d in doc_catalog:
        v = d["vector"]
        v_norm = v / np.linalg.norm(v)
        q_norm = query_vec / np.linalg.norm(query_vec)
        
        cos_sim = float(np.dot(v_norm, q_norm))
        euc_dist = float(np.linalg.norm(v - query_vec))
        dot_prod = float(np.dot(v, query_vec))
        
        scores_list.append({
            "Document": d["name"],
            "Magnitude ||v||": f"{np.linalg.norm(v):.2f}",
            "Cosine Similarity": f"{cos_sim:.4f}",
            "Euclidean Distance": f"{euc_dist:.4f}",
            "Raw Dot Product": f"{dot_prod:.4f}",
            "Characteristics": d["desc"]
        })
        
    df_scores = pd.DataFrame(scores_list)
    
    if selected_metric == "Cosine Similarity":
        df_scores = df_scores.sort_values("Cosine Similarity", ascending=False)
    elif selected_metric == "Euclidean (L2) Distance":
        df_scores = df_scores.sort_values("Euclidean Distance", ascending=True)
    else:
        df_scores = df_scores.sort_values("Raw Dot Product", ascending=False)
        
    st.dataframe(df_scores, use_container_width=True, hide_index=True)
    
    if selected_metric == "Raw Dot Product":
        st.warning("Notice the Flaw: Raw Dot Product ranked Doc B as top match solely because its vector magnitude was huge (||v|| = 7.71), even though Doc A was a 100% exact directional match!")
    elif selected_metric == "Cosine Similarity":
        st.success("Cosine Similarity correctly ranked Doc A as #1 (Cosine = 1.0000), completely unaffected by length differences!")

else:
    st.markdown("#### Synthetic Corpus 2D PCA Projections & Cluster Partitions")
    
    cluster_k = st.slider("Select K Clusters for Spherical K-Means", min_value=2, max_value=4, value=3)
    
    # Generate 3 semantic clusters
    np.random.seed(42)
    c1 = np.random.randn(15, 8) + [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Finance
    c2 = np.random.randn(15, 8) + [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Technology
    c3 = np.random.randn(15, 8) + [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Medical
    all_vecs = np.vstack([c1, c2, c3])
    labels_true = ["Finance"]*15 + ["Tech"]*15 + ["Medical"]*15
    
    # Normalize
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norm_vecs = all_vecs / norms
    
    # PCA projection
    centered = norm_vecs - np.mean(norm_vecs, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords_2d = np.dot(centered, vt[:2].T)
    
    # Cluster assignment via spherical kmeans
    rng = np.random.RandomState(42)
    centroids = norm_vecs[rng.choice(len(norm_vecs), cluster_k, replace=False)].copy()
    for _ in range(10):
        sims = np.dot(norm_vecs, centroids.T)
        assigned = np.argmax(sims, axis=1)
        for j in range(cluster_k):
            sub = norm_vecs[assigned == j]
            if len(sub) > 0:
                m = np.mean(sub, axis=0)
                centroids[j] = m / np.linalg.norm(m)
                
    chart_df = pd.DataFrame({
        "PCA_X": coords_2d[:, 0],
        "PCA_Y": coords_2d[:, 1],
        "Domain": labels_true,
        "Cluster": [f"Cluster {a+1}" for a in assigned]
    })
    
    st.scatter_chart(chart_df, x="PCA_X", y="PCA_Y", color="Cluster", use_container_width=True)
    st.caption("2D projection of 8-dimensional hypersphere showing clean spatial separation among semantic domains.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does HNSW (Hierarchical Navigable Small World) achieve logarithmic search complexity O(log N) compared to brute-force O(N)?",
        "a": "HNSW constructs a multi-layer graph inspired by Skip-Lists. The topmost layers have long-range sparse edges between distant vector clusters, allowing the search query to make rapid, coarse jumps across the vector space. As the search descends into lower layers, edge density increases, transitioning to fine-grained local neighborhood exploration. This prevents the algorithm from evaluating distant points, reducing search time from O(N) to O(log N) while preserving 98%+ recall."
    },
    {
        "q": "Under what exact mathematical condition is Euclidean distance monotonically equivalent to Cosine distance?",
        "a": "Euclidean distance is monotonically equivalent to Cosine distance if and only if all vectors are L2-normalized to unit length (||u|| = ||v|| = 1.0). In that case, ||u - v||_2 = sqrt(2 * (1 - cos(theta))). Because sqrt(2 * (1 - x)) is a strictly decreasing monotonic function of x, any pair of vectors that minimizes Euclidean distance will simultaneously maximize Cosine similarity."
    },
    {
        "q": "What is the Hubness Problem in high-dimensional vector spaces, and how does it degrade retrieval quality?",
        "a": "The Hubness Problem is an intrinsic geometric artifact of high-dimensional spaces (D >= 512). Due to concentration of measure and minor dimensional variances, certain central vectors ('Hubs') lie closer to the center of the manifold than peripheral points and appear as the top nearest neighbor to an extraordinarily large fraction of random queries. In production search, this causes specific irrelevant documents to erroneously rank in top results across completely unrelated user queries."
    },
    {
        "q": "Why is Product Quantization (PQ) crucial for scaling vector databases past 100 million vectors?",
        "a": "Raw floating-point vectors consume significant RAM: 100 million 1536-dimensional float32 vectors require 614 GB of memory. Product Quantization divides each vector into M sub-vectors (e.g., M=64 sub-vectors of dimension 24), runs K-Means on each sub-space to create 256 centroids (represented as an 8-bit byte), and replaces the raw vector with M 1-byte indices. This compresses 614 GB down to 6.4 GB (a 96x reduction), allowing the entire dataset to reside in RAM for fast search."
    },
    {
        "q": "Why does t-SNE fail to preserve global structure compared to UMAP when visualizing document embeddings?",
        "a": "t-SNE uses a Student-t distribution in the low-dimensional space to solve the crowding problem, but it enforces no mathematical constraints on distances between distant clusters. Consequently, the relative positions and distances between different clusters in a t-SNE plot are arbitrary and random. UMAP is formulated on fuzzy simplicial sets and Riemannian geometry, optimizing a cross-entropy objective that preserves both local neighbor connectivity and global inter-cluster topology."
    }
]

for i, qa in enumerate(qas, 1):
    with st.expander(f"Q{i}: {qa['q']}"):
        st.markdown(f"**Answer**: {qa['a']}")

st.markdown("""
### Production Traps & Failure Modes
""")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1 (Fatal): Querying Unnormalized Vectors with Dot Product in FAISS**
    * *Failure*: Creating a `faiss.IndexFlatIP` (Inner Product index) and adding raw unnormalized embeddings.
    * *Fix*: Inner Product does not calculate Cosine Similarity unless vectors are strictly L2-normalized. Always execute `faiss.normalize_L2(vectors)` before adding to an IndexFlatIP index.
    """)
    st.error("""
    **Trap 2 (Fatal): Choosing Euclidean L2 for Text Without Unit Normalization**
    * *Failure*: Using raw Euclidean distance across embeddings produced from documents of different lengths.
    * *Fix*: Longer documents with higher activation norms will be penalized and appear artificially distant from short queries. Normalize all vectors to unit spheres before computing L2 distances.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Setting HNSW efSearch Too Low**
    * *Failure*: Setting `efSearch = k` in HNSW indexes to minimize latency, causing recall to drop below 80%.
    * *Fix*: Set `efSearch` between 64 and 128 for production retrieval. The latency penalty is only ~0.5ms, but recall jumps from 80% to 99%.
    """)
    st.warning("""
    **Trap 4 (Memory Trap): Running PCA on Out-of-Core Data Without Incremental PCA**
    * *Failure*: Calling `sklearn.decomposition.PCA` directly on a 50GB vector matrix, causing an immediate Out-Of-Memory (OOM) crash.
    * *Fix*: Use `IncrementalPCA` (IPCA) which processes vectors in streaming mini-batches via incremental Singular Value Decomposition.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Cosine vs. Dot Product**\n\nIdentical when vectors are L2-normalized (||v|| = 1.0). Unnormalized dot product introduces severe length bias.")
with fc_col2:
    st.info("**HNSW Search Scale**\n\nMulti-layer proximity graph that achieves O(log N) search latency, enabling sub-3ms retrieval across millions of vectors.")
with fc_col3:
    st.info("**Product Quantization (PQ)**\n\nSplits vectors into sub-vectors and encodes them with 8-bit centroid IDs, compressing memory by 16x-64x.")
with fc_col4:
    st.info("**UMAP Superiority**\n\nPreserves both local clusters and global topology significantly better and faster than t-SNE.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of embedding linear algebra, nearest-neighbor algorithms, and clustering operations:
""")

c1 = st.checkbox("I can derive the mathematical equivalence between Cosine Distance and Euclidean Distance on unit-normalized vectors.")
c2 = st.checkbox("I understand why unnormalized Dot Product introduces length and frequency bias in semantic retrieval.")
c3 = st.checkbox("I know how HNSW multi-layer graphs achieve O(log N) search complexity compared to brute-force O(N).")
c4 = st.checkbox("I can describe Product Quantization (PQ) and explain how it achieves 90%+ RAM compression in vector databases.")
c5 = st.checkbox("I understand the Hubness problem in high dimensions and know how mean-centering mitigates it.")
c6 = st.checkbox("I can articulate the architectural trade-offs between PCA, t-SNE, and UMAP for embedding visualization.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered vector space linear algebra, ANN search indexing, clustering, and dimensional projection for foundation AI systems!")
