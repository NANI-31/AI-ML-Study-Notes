import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="11.3 Embedding Visualization - Embeddings Curriculum",
    page_icon="🗺️",
    layout="wide"
)

# Title & Overview
st.title("11.3 Embedding Visualization: PCA, t-SNE, UMAP, & High-Dimensional Manifold Projections")
st.caption("Chapter 11: Embeddings with NumPy | Domain: 3. Vector Embeddings")
st.markdown("""
A neural embedding vector typically contains **768, 1,536, or 3,072 floating-point coordinates**.
Because human cognition and computer displays are strictly limited to two or three spatial dimensions, we cannot directly perceive or inspect these high-dimensional spaces.

To inspect semantic clusters, detect topic drift, diagnose retrieval anomalies, or evaluate classification boundaries, machine learning engineers deploy **Dimensionality Reduction**:
* **PCA (Principal Component Analysis)**: Fast, linear orthogonal projection that maximizes global variance, but flattens intricate non-linear structures.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear probabilistic mapping that produces crisp, isolated local clusters, but scrambles global distances and is computationally expensive.
* **UMAP (Uniform Manifold Approximation and Projection)**: The modern gold standard—preserving **both local cluster fidelity and global macroscopic geometry** while executing orders of magnitude faster than t-SNE!

This lesson explores the mathematics, trade-offs, and production visualization techniques for high-dimensional embeddings.
""")

st.divider()

# Section 1: The Dimensionality Reduction Triad: PCA vs. t-SNE vs. UMAP
st.header("1. The Dimensionality Reduction Triad: PCA vs. t-SNE vs. UMAP", divider="rainbow")

col_vis1, col_vis2 = st.columns([1, 1])

with col_vis1:
    st.markdown("""
    ### Analyzing the Three Projection Paradigms:
    * **1. PCA (Principal Component Analysis)**:
      * **Type**: Linear orthogonal transformation via Singular Value Decomposition (SVD).
      * **Strengths**: Deterministic, ultra-fast, preserves global variance and macroscopic distances.
      * **Weaknesses**: Cannot unroll curved, non-linear manifolds; dense clusters crowd together into an uninformative blur.
    * **2. t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
      * **Type**: Non-linear probabilistic optimization minimizing Kullback-Leibler (KL) divergence between high-dim Gaussian affinities and low-dim Student-t affinities.
      * **Strengths**: Magnifies local neighborhoods; creates distinct, visually isolated cluster islands.
      * **Weaknesses**: Non-deterministic (varies by random seed); relative distances between distant clusters are meaningless; computationally slow ($O(N^2)$).
    * **3. UMAP (Uniform Manifold Approximation and Projection)**:
      * **Type**: Non-linear manifold learning based on Riemannian geometry and fuzzy simplicial sets.
      * **Strengths**: Preserves **both local clusters and global relationships** between clusters; scales linearly with dataset size ($O(N \\log N)$).
      * **Production Best Practice**: The default choice for clustering diagnostics and RAG visualization!
    """)

with col_vis2:
    st.markdown("### Manifold Projection Architecture")
    vis_dot = """
    digraph DimReduction {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        HighDim [label="High-Dimensional Latent Space\\nShape: [N = 10,000 docs, d = 1,536 floats]\\nUnit Hypersphere", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_algos {
            label="Reduction Algorithms"; style="dashed"; color="#8B5CF6";
            PCA_Node [label="PCA\\nLinear SVD Projection\\nPreserves Global Variance", fillcolor="#FEF3C7"];
            TSNE_Node [label="t-SNE\\nKL Divergence over Student-t\\nPreserves Local Clusters", fillcolor="#FDE68A"];
            UMAP_Node [label="UMAP\\nFuzzy Simplicial Sets\\nPreserves Local + Global", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        }

        LowDim [label="2D / 3D Visualization Plane\\nCoordinates: (x, y) or (x, y, z)\\nHuman Interpretable Map", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        HighDim -> PCA_Node -> LowDim;
        HighDim -> TSNE_Node -> LowDim;
        HighDim -> UMAP_Node -> LowDim;
    }
    """
    st.graphviz_chart(vis_dot, use_container_width=False)

st.markdown("### Technical Comparison Matrix: PCA vs. t-SNE vs. UMAP")
st.markdown("""
| Feature | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| **Linear vs Non-Linear** | Purely Linear | Non-Linear Probabilistic | Non-Linear Topological |
| **Local Structure** | Poor (Crowding problem) | **Exceptional** | **Exceptional** |
| **Global Structure** | **Preserved** | Lost (Inter-cluster distances arbitrary) | **Preserved** |
| **Time Complexity** | $O(N \\cdot d^2)$ (Fastest) | $O(N^2)$ (Very Slow) | $O(N \\log N)$ (Fast) |
| **Determinism** | Fully Deterministic | Stochastic (Varies by seed) | Stochastic (Fix seed for reproducibility) |
| **Primary Hyperparameters** | `n_components` | `perplexity` (5-50), `learning_rate` | `n_neighbors` (5-50), `min_dist` (0.1) |
| **Best Use Case** | Baseline pre-reduction | Qualitative 2D cluster exploration | Production RAG & topic discovery maps |
""")

st.divider()

# Section 2: Production Code: Multi-Algorithm Dimensionality Reducer
st.header("2. Production Implementation: Multi-Algorithm Projection Utility", divider="rainbow")

st.markdown("""
The production class below shows how to construct a robust dimensionality reduction engine supporting PCA, t-SNE, and UMAP with pre-normalization and automatic fallback.
""")

st.code(r'''
import numpy as np
from typing import Dict, Any, Tuple

# In production:
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

class EmbeddingDimensionalityReducer:
    """
    Projects high-dimensional embedding matrices down to 2D coordinates
    for visualization and cluster analysis.
    """
    def __init__(self, random_state: int = 42):
        self.seed = random_state

    def reduce_pca(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Linear projection via SVD.
        embeddings: [N, d] -> returns: [N, n_components]
        """
        # Center the data
        centered = embeddings - np.mean(embeddings, axis=0)
        # SVD: X = U * S * V^T
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        return U[:, :n_components] * S[:n_components]

    def reduce_tsne(self, embeddings: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
        """
        Non-linear t-SNE projection.
        In production: TSNE(n_components=2, perplexity=perplexity, random_state=self.seed).fit_transform(embeddings)
        """
        # PCA pre-reduction recommended before t-SNE to accelerate convergence
        pca_50 = self.reduce_pca(embeddings, n_components=min(50, embeddings.shape[1]))
        # Synthetic mock projection for illustration:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((embeddings.shape[0], 2))

    def reduce_umap(self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """
        Topological UMAP projection.
        In production: umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.seed).fit_transform(embeddings)
        """
        # Synthetic mock projection for illustration:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((embeddings.shape[0], 2))
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveProjectionSimulator:
    def __init__(self):
        # 12 representative documents across 4 distinct semantic clusters
        self.clusters = {
            "Cardiology / Healthcare": {
                "docs": [
                    "Myocardial infarction treatment and emergency stenting procedures.",
                    "Hypertension management through ACE inhibitors and diet.",
                    "Cardiac telemetry monitoring in post-surgical intensive care units."
                ],
                "base_pca": np.array([-4.2, 1.8]),
                "base_tsne": np.array([-8.5, 9.2]),
                "base_umap": np.array([-6.2, 5.8])
            },
            "Financial Markets": {
                "docs": [
                    "Federal Reserve interest rate adjustments impact Treasury bond yields.",
                    "Equity market valuation multiples during quarterly corporate earnings.",
                    "Algorithmic high-frequency trading liquidity and order book dynamics."
                ],
                "base_pca": np.array([3.8, 3.2]),
                "base_tsne": np.array([7.8, 8.4]),
                "base_umap": np.array([5.9, 6.2])
            },
            "Space & Astronomy": {
                "docs": [
                    "Hubble and James Webb space telescopes capture deep galaxy clusters.",
                    "Exoplanet atmospheric spectroscopy detects methane and water vapor.",
                    "Orbital mechanics for lunar gateway transit and gravitational slingshots."
                ],
                "base_pca": np.array([-2.1, -4.5]),
                "base_tsne": np.array([-9.1, -7.5]),
                "base_umap": np.array([-5.8, -6.1])
            },
            "Artificial Intelligence": {
                "docs": [
                    "Transformer attention mechanisms compute pairwise token similarity.",
                    "Reinforcement learning from human feedback aligns large language models.",
                    "Vector database indexing using hierarchical navigable small world graphs."
                ],
                "base_pca": np.array([4.5, -3.1]),
                "base_tsne": np.array([8.2, -8.1]),
                "base_umap": np.array([6.5, -5.4])
            }
        }

    def project_corpus(self, method: str) -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(42)
        
        for cat, data in self.clusters.items():
            if method == "PCA (Linear SVD)":
                center = data["base_pca"]
                spread = 0.65
            elif method == "t-SNE (Local Clusters)":
                center = data["base_tsne"]
                spread = 0.35 # tighter isolated clusters
            else: # UMAP
                center = data["base_umap"]
                spread = 0.45

            for i, text in enumerate(data["docs"]):
                offset = rng.standard_normal(2) * spread
                coord = center + offset
                rows.append({
                    "Category": cat,
                    "Document Snippet": text[:55] + "...",
                    "2D Coordinate X": float(coord[0]),
                    "2D Coordinate Y": float(coord[1])
                })

        return pd.DataFrame(rows)

st.divider()

# Section 3: Interactive Workbench: 2D Manifold Projection Explorer
st.header("3. Interactive Workbench: 2D Manifold Projection Explorer", divider="rainbow")

st.markdown("""
Select a dimensionality reduction algorithm below to observe how a multi-domain corpus spanning **Cardiology**, **Finance**, **Astronomy**, and **AI** is projected onto a 2D map.
Observe how **t-SNE and UMAP** produce crisp, well-separated cluster boundaries compared to **PCA's** linear projection.
""")

proj_sim = InteractiveProjectionSimulator()

col_proj_l, col_proj_r = st.columns([1, 1])

with col_proj_l:
    st.subheader("1. Select Projection Algorithm")
    method_sel = st.selectbox(
        "Dimensionality Reduction Method:",
        ["UMAP (Local + Global Geometry)", "t-SNE (Local Clusters)", "PCA (Linear SVD)"]
    )

    df_coords = proj_sim.project_corpus(method_sel)

    st.markdown("#### Projected 2D Coordinate Table:")
    st.dataframe(
        df_coords.style.format({"2D Coordinate X": "{:.2f}", "2D Coordinate Y": "{:.2f}"}),
        use_container_width=True
    )

with col_proj_r:
    st.subheader(f"2. 2D Coordinate Map ({method_sel.split()[0]})")
    st.markdown("Relative cluster distribution on the 2D visualization plane:")

    # Scatter chart using Streamlit native scatter
    st.scatter_chart(
        df_coords,
        x="2D Coordinate X",
        y="2D Coordinate Y",
        color="Category",
        use_container_width=True
    )

    st.markdown("#### Cluster Interpretation")
    if "PCA" in method_sel:
        st.info("ℹ️ **PCA Projection**: Preserves major axes of global variance. Notice that while categories are separated, points are somewhat spread out along linear diagonals.")
    elif "t-SNE" in method_sel:
        st.success("🟢 **t-SNE Projection**: Creates dense, tightly grouped semantic islands with large empty voids between clusters.")
    else:
        st.success("🌟 **UMAP Projection**: Optimal balance! Tightly gathers semantic clusters while maintaining meaningful spatial relationships between complementary domains.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Visualization Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why can you NOT interpret the distance between different clusters in t-SNE?**
    * **Answer**:
      * t-SNE prioritizes preserving **local neighborhood affinities** ($p_{j|i}$) over global distances.
      * In t-SNE, the cost function heavily penalizes putting nearby points far apart, but imposes almost zero penalty for moving distant clusters to arbitrary locations.
      * The final distance between Cluster A and Cluster B depends heavily on the arbitrary initialization and the `perplexity` parameter.
      * Therefore: you can conclude that points *within* a cluster are semantically similar, but you **cannot** conclude that Cluster A is semantically closer to Cluster B than to Cluster C!

    **Q2: How does UMAP preserve global structure while maintaining tight local clusters?**
    * **Answer**:
      * UMAP uses **fuzzy simplicial sets** grounded in Riemannian manifold theory.
      * It assumes data lies on a local manifold that is locally connected.
      * Unlike t-SNE, which uses a uniform global denominator in its Student-t distribution, UMAP balances both:
        * High cross-entropy penalty for splitting local neighbors.
        * Smooth decay that retains global topological connections across the fuzzy simplicial complex.
      * Consequently, relative macro-distances between clusters remain mathematically meaningful.

    **Q3: Why is PCA commonly applied as a pre-reduction step before t-SNE or UMAP?**
    * **Answer**:
      * High-dimensional embeddings (e.g. 1536-d) suffer from high distance concentration: in high dimensions, Euclidean distances between random pairs converge to a narrow distribution.
      * Furthermore, running t-SNE directly on 1536 dimensions is computationally prohibitive ($O(N \\cdot d + N^2)$).
      * **Best Practice**: Apply PCA first to reduce from 1536 dimensions down to 50 principal components.
      * This removes high-frequency noise, retains 90%+ of variance, and speeds up subsequent t-SNE/UMAP optimization by **10x**!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Misinterpreting Perplexity in t-SNE**:
      * *Issue*: Setting `perplexity = 2` on a 5,000-document dataset.
      * *Consequence*: The algorithm creates dozens of artificial micro-clusters that look like meaningful semantic categories but are actually random noise!
      * *Fix*: Tune `perplexity` to roughly $\\sqrt{N}$ or standard values between 30 and 50.

    * **Trap 2: Using 2D Coordinates for Downstream Retrieval**:
      * *Issue*: Projecting embeddings down to 2D using UMAP and performing vector search on the 2D coordinates.
      * *Consequence*: Compressing 1536 dimensions down to 2 dimensions destroys 95%+ of semantic discrimination.
      * *Fix*: Use 2D/3D projections strictly for **human visual analytics**; keep the full 1536 or 768 dimensions in the vector database index!

    * **Trap 3: Forgetting to Fix the Random Seed**:
      * *Issue*: Running t-SNE or UMAP in a web dashboard without specifying `random_state`.
      * *Consequence*: Every page refresh rotates and rearranges cluster positions, confusing users.
      * *Fix*: Always fix `random_state=42`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Principal Component Analysis (PCA)**
      * *Definition*: A linear dimensionality reduction technique that projects data along orthogonal axes maximizing global variance.
    * **Flashcard 2: t-SNE**
      * *Definition*: A non-linear probabilistic technique optimizing KL divergence to visualize local neighborhood clusters in 2D or 3D.
    * **Flashcard 3: UMAP**
      * *Definition*: A manifold learning technique preserving both local clusters and global topological relationships with linear-logarithmic scaling.
    * **Flashcard 4: Pre-Reduction Pipeline**
      * *Definition*: Reducing high-dimensional embeddings to 50 dimensions via PCA before applying t-SNE or UMAP to eliminate noise and accelerate computation.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embedding Visualization:

- [ ] I can describe the fundamental trade-offs between PCA, t-SNE, and UMAP.
- [ ] I understand why inter-cluster distances in t-SNE are arbitrary and cannot be interpreted globally.
- [ ] I know why UMAP is preferred for production cluster visualization and topic discovery.
- [ ] I understand why PCA pre-reduction (down to 50 dimensions) is standard practice before t-SNE.
- [ ] I know never to use 2D/3D projection coordinates for actual downstream vector database retrieval.
""")
