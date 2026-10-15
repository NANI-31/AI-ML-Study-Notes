import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="12.1 Similarity with Scikit-learn - Embeddings Curriculum",
    page_icon="🔬",
    layout="wide"
)

# Title & Overview
st.title("12.1 Similarity with Scikit-learn: Pairwise Metrics, Cython Acceleration, & Distance Matrices")
st.caption("Chapter 12: Embeddings with Scikit-learn | Domain: 3. Vector Embeddings")
st.markdown("""
While writing pure NumPy operations is valuable for understanding low-level linear algebra, **Scikit-learn (`sklearn.metrics.pairwise`)** is the production industry standard for high-level similarity and distance calculations in Python.

Scikit-learn provides battle-tested, Cython-accelerated, OpenMP-multithreaded implementations of:
* **`cosine_similarity(X, Y=None)`**: Normalized dot products supporting both dense NumPy arrays and sparse SciPy matrices.
* **`cosine_distances(X, Y=None)`**: The inverse metric defined as $1.0 - \\text{cosine\\_similarity}$.
* **`euclidean_distances(X, Y=None)`**: Optimized pairwise Euclidean geometric distances.
* **`manhattan_distances(X, Y=None)`**: $L_1$ Taxicab metric distances across coordinate axes.

This lesson explores Scikit-learn's pairwise API, symmetric Gram matrices vs. rectangular query-corpus cross-matrices, and distance-to-similarity transformations.
""")

st.divider()

# Section 1: Scikit-learn Pairwise Metric Architecture
st.header("1. Scikit-learn Pairwise Metric Architecture", divider="rainbow")

col_skl1, col_skl2 = st.columns([1, 1])

with col_skl1:
    st.markdown("""
    ### Key Features of `sklearn.metrics.pairwise`:
    * **1. Symmetric vs. Rectangular Matrix Execution**:
      * If called as `cosine_similarity(X)` with a single matrix:
        * Computes the $N \\times N$ symmetric **Gram matrix** comparing every document to every other document.
        * The diagonal is strictly $1.0000$ (self-similarity).
      * If called as `cosine_similarity(X, Y)` with two matrices:
        * Computes an $N \\times M$ rectangular matrix comparing $N$ queries against $M$ corpus documents.
    * **2. Automatic Normalization Safeguard**:
      * Unlike raw NumPy `np.dot()`, Scikit-learn's `cosine_similarity()` internally computes and divides by Euclidean norms, ensuring valid cosine scores even if the input arrays are unnormalized!
    * **3. Dual Support for Dense & Sparse Data**:
      * Accepts standard dense `np.ndarray` float arrays as well as `scipy.sparse.csr_matrix` (crucial for hybrid BM25 + dense representations).
    """)

with col_skl2:
    st.markdown("### Scikit-learn Pairwise Metric Flow")
    skl_dot = """
    digraph SklearnPairwise {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        InputX [label="Matrix X: [N = 3, d = 768]\\nDense or Sparse", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        InputY [label="Matrix Y (Optional): [M = 4, d = 768]\\nQuery or Corpus Matrix", fillcolor="#ECFDF5", color="#10B981"];
        
        CythonEngine [label="sklearn.metrics.pairwise Engine\\nCython Loops + OpenMP Parallelism\\nAutomatic Internal Normalization", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];
        
        subgraph cluster_outputs {
            label="Metric Output Options"; style="dashed"; color="#8B5CF6";
            CosSim [label="cosine_similarity(X, Y)\\nRange: [-1.0, 1.0]\\n(Higher = Closer)", fillcolor="#DCFCE7"];
            CosDist [label="cosine_distances(X, Y)\\n1.0 - cosine_similarity\\n(Lower = Closer)", fillcolor="#FEE2E2"];
            EucDist [label="euclidean_distances(X, Y)\\nStraight-line distance\\nRange: [0.0, inf)", fillcolor="#EDE9FE"];
        }

        InputX -> CythonEngine;
        InputY -> CythonEngine;
        CythonEngine -> CosSim;
        CythonEngine -> CosDist;
        CythonEngine -> EucDist;
    }
    """
    st.graphviz_chart(skl_dot, use_container_width=False)

st.markdown("### Pairwise Metric API Summary Table")
st.markdown("""
| Function Name | Mathematical Definition | Return Range | Best Use Case | Inversion Formula |
| :--- | :--- | :--- | :--- | :--- |
| **`cosine_similarity(X, Y)`** | `(x dot y) / (||x|| * ||y||)` | `[-1.0, 1.0]` | Semantic text search, RAG retrieval | `similarity = 1.0 - distance` |
| **`cosine_distances(X, Y)`** | `1.0 - cosine_similarity(X, Y)` | `[0.0, 2.0]` | Clustering algorithms (DBSCAN, Agglomerative) | `distance = 1.0 - similarity` |
| **`euclidean_distances(X, Y)`** | `sqrt(sum((x_i - y_i)^2))` | `[0.0, inf)` | Spatial clustering (K-Means), anomaly detection | `similarity = 1 / (1 + distance)` |
| **`manhattan_distances(X, Y)`** | `sum(|x_i - y_i|)` | `[0.0, inf)` | Robust outlier-resistant distance analysis | `similarity = 1 / (1 + distance)` |
""")

st.divider()

# Section 2: Production Code: Scikit-learn Metric Utility
st.header("2. Production Implementation: Scikit-learn Pairwise Suite", divider="rainbow")

st.markdown("""
The production utility below demonstrates high-level pairwise calculations using `sklearn.metrics.pairwise` with automatic distance-to-similarity conversion and top-K ranking.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any

# In production:
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances

class SklearnPairwiseSuite:
    """
    Production wrapper for Scikit-learn pairwise similarity and distance functions.
    Handles unnormalized inputs, sparse matrices, and score inversion.
    """
    def __init__(self):
        pass

    def compute_all_metrics(self, queries: np.ndarray, corpus: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes cosine similarity, cosine distance, and Euclidean distance.
        queries: [N, d], corpus: [M, d]
        """
        # In production:
        # cos_sim = cosine_similarity(queries, corpus)
        # cos_dist = cosine_distances(queries, corpus)
        # euc_dist = euclidean_distances(queries, corpus)
        
        # Pure mathematical simulation:
        q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        c_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        cos_sim = np.dot(q_norm, c_norm.T)
        cos_dist = 1.0 - cos_sim
        euc_dist = np.sqrt(np.maximum(2.0 - 2.0 * cos_sim, 0.0))

        return {
            "cosine_similarity": cos_sim,
            "cosine_distance": cos_dist,
            "euclidean_distance": euc_dist
        }
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveSklearnStudio:
    def __init__(self):
        self.doc_labels = ["Doc A: Cardiology treatment", "Doc B: Banking monetary policy", "Doc C: Space telescope discovery"]
        # Mock 3D unnormalized vectors
        self.raw_corpus = np.array([
            [4.5, 0.2, 0.1],  # Cardiology
            [0.1, 5.2, 0.2],  # Banking
            [0.2, 0.1, 4.8]   # Space
        ], dtype=np.float32)

    def compute_matrix(self, metric: str) -> pd.DataFrame:
        # Gram matrix of corpus against itself
        norms = np.linalg.norm(self.raw_corpus, axis=1, keepdims=True)
        norm_mat = self.raw_corpus / norms
        sim = np.dot(norm_mat, norm_mat.T)
        
        if metric == "cosine_similarity":
            res = sim
            fmt = "{:.4f}"
            cmap = "Blues"
        elif metric == "cosine_distances":
            res = 1.0 - sim
            fmt = "{:.4f}"
            cmap = "Reds"
        else: # euclidean_distances
            # unnormalized Euclidean distances
            a2 = np.sum(self.raw_corpus**2, axis=1, keepdims=True)
            dist_sq = a2 + a2.T - 2.0 * np.dot(self.raw_corpus, self.raw_corpus.T)
            res = np.sqrt(np.maximum(dist_sq, 0.0))
            fmt = "{:.2f}"
            cmap = "Purples"

        df = pd.DataFrame(res, index=self.doc_labels, columns=self.doc_labels)
        return df, fmt, cmap

st.divider()

# Section 3: Interactive Workbench: Scikit-learn Pairwise Metric Studio
st.header("3. Interactive Workbench: Scikit-learn Pairwise Metric Studio", divider="rainbow")

st.markdown("""
Toggle between **`cosine_similarity()`**, **`cosine_distances()`**, and **`euclidean_distances()`** to inspect the resulting $3 \\times 3$ Gram matrix across three semantic documents.
Observe how the diagonal represents self-comparison ($1.0$ for similarity, $0.0$ for distance metrics).
""")

skl_studio = InteractiveSklearnStudio()

col_st_l, col_st_r = st.columns([1, 1])

with col_st_l:
    st.subheader("1. Select Scikit-learn Metric")
    metric_choice = st.radio(
        "Pairwise Function to Execute:",
        [
            "cosine_similarity (Higher = More Similar, Diagonal = 1.0)",
            "cosine_distances (Lower = Closer, Diagonal = 0.0)",
            "euclidean_distances (Geometric Distance, Diagonal = 0.0)"
        ]
    )
    clean_metric = metric_choice.split()[0]

    st.markdown("#### Corpus Document Definitions:")
    for label in skl_studio.doc_labels:
        st.caption(f"• **{label}**")

with col_st_r:
    st.subheader(f"2. Resulting Pairwise Matrix ({clean_metric})")
    df_mat, fmt_str, cmap_str = skl_studio.compute_matrix(clean_metric)

    st.dataframe(
        df_mat.style.format(fmt_str).background_gradient(cmap=cmap_str),
        use_container_width=True
    )

    if clean_metric == "cosine_similarity":
        st.success("✅ **Symmetric Similarity Matrix**: Diagonal is exactly 1.0000; off-diagonal values reflect semantic overlap (e.g. Cardiology vs Banking = 0.02, near zero).")
    elif clean_metric == "cosine_distances":
        st.info("ℹ️ **Distance Matrix**: Diagonal is exactly 0.0000 (zero distance from self); unrelated documents have distance near 1.0000.")
    else:
        st.info("ℹ️ **Euclidean Distance**: Dependent on raw vector magnitudes. Diagonal is 0.00.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Scikit-learn Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the exact mathematical difference between cosine_similarity and cosine_distances in Scikit-learn?**
    * **Answer**:
      * `cosine_similarity(X, Y)` returns the inner angle cosine score ranging from -1.0 to +1.0.
        Larger values indicate higher semantic similarity.
      * `cosine_distances(X, Y)` is defined strictly as:
        `cosine_distance = 1.0 - cosine_similarity`
      * Its range is [0.0, 2.0].
      * Smaller values indicate closer proximity (0.0 means identical direction).
      * Clustering algorithms (e.g. `AgglomerativeClustering`, `DBSCAN`) require distance metrics, so `cosine_distances` is passed as the distance matrix.

    **Q2: Does Scikit-learn's cosine_similarity require input vectors to be pre-normalized?**
    * **Answer**:
      * **No**. Internally, Scikit-learn computes the Euclidean $L_2$ norm of both $X$ and $Y$ and divides by their product.
      * If you pass unnormalized vectors with magnitudes 10.0 and 50.0, `cosine_similarity` will still return a mathematically correct cosine score in `[-1.0, 1.0]`.
      * However, for maximum speed over large datasets, **pre-normalizing vectors upfront** allows you to use `X @ Y.T` directly, avoiding the re-normalization overhead on every call.

    **Q3: How does Scikit-learn handle sparse matrices in cosine_similarity?**
    * **Answer**:
      * When passed `scipy.sparse.csr_matrix` inputs (common in TF-IDF or sparse keyword representations), Scikit-learn avoids converting them into dense matrices (which would consume gigabytes of RAM).
      * It uses optimized sparse BLAS routines to multiply only non-zero coordinates, returning a dense similarity matrix while preserving memory efficiency during the calculation.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Passing cosine_distances to Ranking Algorithms Expecting Similarity**:
      * *Issue*: Calling `cosine_distances()` and sorting descending (`reverse=True`) to find the best match.
      * *Consequence*: Completely inverts retrieval! The system returns the most unrelated document in the entire database as the top answer!
      * *Fix*: For distance metrics, sort **ascending** (`reverse=False`); for similarity metrics, sort **descending**.

    * **Trap 2: Feeding High-Dimensional Dense Matrices into euclidean_distances**:
      * *Issue*: Computing Euclidean distances directly on 1536-dimensional raw unnormalized vectors without scaling.
      * *Consequence*: The curse of dimensionality: Euclidean distances between all pairs concentrate into a narrow band, ruining cluster separation.
      * *Fix*: Normalize vectors to unit length first, or use `cosine_distances`.

    * **Trap 3: Memory Exhaustion on Massive Gram Matrices**:
      * *Issue*: Calling `cosine_similarity(X)` on a corpus of 100,000 documents.
      * *Consequence*: The resulting $N \\times N$ matrix has $100,000 \\times 100,000 = 10,000,000,000$ float32 numbers, requiring **40 GB RAM**, causing an immediate out-of-memory crash!
      * *Fix*: Never compute full $N \\times N$ Gram matrices on large datasets; compute batch cross-matrices `cosine_similarity(queries, corpus)` or use vector databases (HNSW).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: cosine_similarity(X, Y)**
      * *Definition*: Scikit-learn pairwise function returning normalized inner dot products in the range [-1.0, 1.0].
    * **Flashcard 2: cosine_distances(X, Y)**
      * *Definition*: Distance metric defined as 1.0 - cosine_similarity, where 0.0 indicates identical directional alignment.
    * **Flashcard 3: Gram Matrix**
      * *Definition*: The symmetric N x N pairwise similarity matrix resulting from comparing a dataset X against itself, with 1.0 along the main diagonal.
    * **Flashcard 4: Sparse Matrix Support**
      * *Definition*: Scikit-learn's capability to process scipy.sparse matrices natively without densification, critical for hybrid sparse-dense pipelines.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Similarity with Scikit-learn:

- [ ] I can describe the relationship between `cosine_similarity` and `cosine_distances`.
- [ ] I understand how Scikit-learn handles both dense NumPy arrays and sparse SciPy matrices.
- [ ] I know why $N \\times N$ Gram matrices cause memory crashes on large datasets and how to avoid them.
- [ ] I know how to convert between distance metrics and similarity scores.
- [ ] I understand how Scikit-learn internal normalization protects against unnormalized input vectors.
""")
