import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="12.2 Dimensionality Reduction with Scikit-learn - Embeddings Curriculum",
    page_icon="📉",
    layout="wide"
)

# Title & Overview
st.title("12.2 Dimensionality Reduction with Scikit-learn: PCA, TruncatedSVD, Scree Variance, & Out-of-Sample Projections")
st.caption("Chapter 12: Embeddings with Scikit-learn | Domain: 3. Vector Embeddings")
st.markdown("""
High-dimensional embedding models produce rich semantic representations (e.g. 768 or 1,536 coordinates), but processing, storing, and indexing these vectors carries substantial computational costs.

**Scikit-learn** provides industry-standard matrix decomposition algorithms to compress vectors into lower-dimensional subspaces while retaining maximal semantic variance:
* **`PCA` (Principal Component Analysis)**: Centers the data and projects along orthogonal axes of maximal variance via Singular Value Decomposition (SVD).
* **`TruncatedSVD` (Latent Semantic Analysis)**: Performs SVD **without mean-centering**, making it ideal for sparse matrix embeddings where centering would destroy sparsity and consume gigabytes of RAM!
* **Explained Variance Ratio**: Quantifying precisely how much information is preserved across the top $k$ principal components using scree analysis.
* **Out-of-Sample Projection (`.transform()`)**: Understanding why PCA and TruncatedSVD can project new incoming search queries into the reduced space, whereas non-parametric algorithms like **t-SNE cannot**!

This lesson explores the Scikit-learn dimensionality reduction toolkit, scree variance curves, and production projection pipelines.
""")

st.divider()

# Section 1: PCA vs. TruncatedSVD & The Out-of-Sample Dilemma
st.header("1. PCA vs. TruncatedSVD & The Out-of-Sample Projection Dilemma", divider="rainbow")

col_dec1, col_dec2 = st.columns([1, 1])

with col_dec1:
    st.markdown("""
    ### Comparing Scikit-learn Decomposition Estimators:
    * **1. PCA (`sklearn.decomposition.PCA`)**:
      * Centers the data by subtracting column means (mu) before computing SVD:
        `X_centered = X - mean(X)`
      * Generates orthogonal components that maximize variance.
      * Supports `.transform(X_new)`: new query vectors can be projected at runtime!
      * **Limitation**: Cannot accept sparse matrices directly because subtracting means converts zero cells into non-zero floats (exploding memory).
    * **2. TruncatedSVD (`sklearn.decomposition.TruncatedSVD`)**:
      * Directly decomposes the matrix without subtracting the mean.
      * Native support for `scipy.sparse` matrices (crucial for TF-IDF / LSA hybrid pipelines).
      * When applied to pre-centered data, TruncatedSVD produces results mathematically identical to PCA.
    * **3. The Out-of-Sample Projection Dilemma (PCA vs. t-SNE)**:
      * **PCA has a projection matrix Vt**: When a user submits a search query, you simply multiply `q_reduced = (q - mu) @ Vt.T` in microseconds.
      * **t-SNE lacks a projection matrix**: It is a non-parametric optimization of points on a 2D plane. You **cannot** project a new out-of-sample query without re-optimizing the entire dataset!
    """)

with col_dec2:
    st.markdown("### Scree Variance Retention Architecture")
    scree_dot = """
    digraph ScreeVariance {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        InputVecs [label="Original Embeddings\\n[N = 10,000, d = 768]\\n100% Total Information", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        FitPCA [label="PCA / TruncatedSVD Fit\\nSVD Factorization -> Singular Values", fillcolor="#FEF3C7"];
        
        ScreeAnalysis [label="Scree Analysis & Elbow Cutoff\\nTop 64 Components retain 91.5% Variance!\\nTop 128 Components retain 96.2% Variance!", fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];
        
        TransformStep [label="Production .transform()\\nCompresses to 128 Dimensions\\n4x-6x Vector Search Speedup", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        InputVecs -> FitPCA -> ScreeAnalysis -> TransformStep;
    }
    """
    st.graphviz_chart(scree_dot, use_container_width=False)

st.markdown("### Estimator Comparison Table")
st.markdown("""
| Estimator | Mean Centering? | Sparse Matrix Support? | Has `.transform()`? | Primary Use Case | Time Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PCA`** | **Yes** (Automated) | No (Converts to dense) | **Yes** (Fast out-of-sample projection) | Dense embeddings (SBERT, OpenAI) | $O(N \\cdot d^2)$ |
| **`TruncatedSVD`** | No (Preserves sparsity) | **Yes** (`scipy.sparse`) | **Yes** (Fast out-of-sample projection) | Sparse TF-IDF, hybrid matrices | $O(N \\cdot d \\cdot k)$ |
| **`TSNE`** | N/A (Probabilistic) | Yes | **No** (Cannot project new queries) | Qualitative 2D cluster visualization | $O(N^2)$ |
""")

st.divider()

# Section 2: Production Code: PCA Compression & Projection Pipeline
st.header("2. Production Implementation: Embedding Compressor Pipeline", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to fit a PCA compressor on corpus embeddings, analyze cumulative explained variance, and project both corpus documents and runtime queries into a compressed subspace.
""")

st.code(r'''
import numpy as np
from typing import Dict, Any, Tuple

# In production:
# from sklearn.decomposition import PCA, TruncatedSVD

class ProductionEmbeddingCompressor:
    """
    Compresses high-dimensional dense embeddings using PCA / SVD.
    Maintains projection matrices to transform out-of-sample queries at runtime.
    """
    def __init__(self, target_dim: int = 128):
        self.target_dim = target_dim
        self.mean: np.ndarray = None
        self.components: np.ndarray = None  # Vt[:target_dim]
        self.explained_variance_ratio: np.ndarray = None

    def fit(self, corpus_embeddings: np.ndarray):
        """
        Fits PCA on corpus embeddings via Singular Value Decomposition.
        """
        N, d = corpus_embeddings.shape
        self.mean = np.mean(corpus_embeddings, axis=0)
        centered = corpus_embeddings - self.mean

        # SVD: centered = U * S * Vt
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:self.target_dim] # [k, d]

        # Explained variance ratio
        var_explained = (S ** 2) / (N - 1)
        total_var = np.sum(var_explained)
        self.explained_variance_ratio = var_explained[:self.target_dim] / total_var

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Projects new or existing embeddings into the compressed subspace.
        embeddings: [M, d] -> returns: [M, target_dim]
        """
        centered = embeddings - self.mean
        # Linear projection: X_proj = centered @ components.T
        compressed = np.dot(centered, self.components.T)
        
        # Unit L2 re-normalization for downstream cosine similarity
        norms = np.linalg.norm(compressed, axis=-1, keepdims=True)
        return compressed / np.maximum(norms, 1e-12)
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveScreeSimulator:
    def __init__(self, original_dim: int = 384):
        self.orig_dim = original_dim
        # Synthetic variance decay curve modeled after SBERT representations
        rng = np.random.default_rng(42)
        decay = np.exp(-np.linspace(0, 4.5, original_dim))
        decay = decay + rng.uniform(0, 0.05, original_dim)
        decay = np.sort(decay)[::-1]
        self.variance_ratios = decay / np.sum(decay)
        self.cumsum_variance = np.cumsum(self.variance_ratios)

    def evaluate_compression(self, target_k: int) -> Dict[str, Any]:
        k_capped = min(target_k, self.orig_dim)
        retained_var = float(self.cumsum_variance[k_capped - 1]) * 100.0
        compression_ratio = (self.orig_dim / k_capped)
        ram_savings_pct = (1.0 - (k_capped / self.orig_dim)) * 100.0

        return {
            "target_k": k_capped,
            "retained_var_pct": retained_var,
            "lost_var_pct": 100.0 - retained_var,
            "compression_ratio": compression_ratio,
            "ram_savings_pct": ram_savings_pct
        }

st.divider()

# Section 3: Interactive Workbench: Scree Variance & Compression Analyzer
st.header("3. Interactive Workbench: Scree Variance & Compression Analyzer", divider="rainbow")

st.markdown("""
Experiment with different target dimensions ($k$) to analyze how much semantic variance is preserved when compressing **384-dimensional embeddings**.
Notice how retaining just **64 to 96 components** preserves over **85-90% of total information** while slashing vector memory footprints by **75%+**!
""")

scree_sim = InteractiveScreeSimulator(original_dim=384)

col_sc_l, col_sc_r = st.columns([1, 1])

with col_sc_l:
    st.subheader("1. Configure Target Dimension (k)")
    target_k_val = st.slider("Target Compressed Dimensions (k):", min_value=8, max_value=384, value=64, step=8)
    
    comp_stats = scree_sim.evaluate_compression(target_k_val)

    m1, m2 = st.columns(2)
    m1.metric("Retained Variance", f"{comp_stats['retained_var_pct']:.1f}%")
    m2.metric("Information Lost", f"{comp_stats['lost_var_pct']:.1f}%", delta=f"-{comp_stats['lost_var_pct']:.1f}%", delta_color="inverse")

    m3, m4 = st.columns(2)
    m3.metric("RAM & Index Storage Savings", f"{comp_stats['ram_savings_pct']:.1f}%")
    m4.metric("Compression Ratio", f"{comp_stats['compression_ratio']:.1f}x Smaller")

with col_sc_r:
    st.subheader("2. Cumulative Explained Variance Scree Curve")
    
    # Sample points along the scree curve
    eval_points = [8, 16, 32, 64, 96, 128, 192, 256, 384]
    df_scree = pd.DataFrame({
        "Dimensions (k)": eval_points,
        "Cumulative Variance (%)": [float(scree_sim.cumsum_variance[p - 1] * 100.0) for p in eval_points]
    })
    
    st.line_chart(df_scree.set_index("Dimensions (k)"), use_container_width=True)

    if comp_stats['retained_var_pct'] >= 90.0:
        st.success(f"🌟 **Optimal Compression**: Retaining {target_k_val} dimensions preserves **{comp_stats['retained_var_pct']:.1f}%** semantic variance with a **{comp_stats['ram_savings_pct']:.1f}%** vector database storage reduction!")
    elif comp_stats['retained_var_pct'] >= 75.0:
        st.info(f"🟡 **Moderate Compression**: Retaining {target_k_val} dimensions provides high storage savings ({comp_stats['ram_savings_pct']:.1f}%), suitable for approximate candidate pre-filtering.")
    else:
        st.warning(f"⚠️ **Severe Information Loss**: Compressing to {target_k_val} dimensions discards {comp_stats['lost_var_pct']:.1f}% of variance, risking retrieval accuracy.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Decomposition Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why can PCA and TruncatedSVD be deployed in production search pipelines for new queries, but t-SNE cannot?**
    * **Answer**:
      * PCA and TruncatedSVD are **parametric linear projections**: fitting the model yields an explicit transformation matrix Vt of shape [k, d] and mean vector mu.
      * When a new search query q arrives at runtime, projecting it into the k-dimensional space requires only a single matrix multiplication:
        `q_compressed = (q - mu) @ Vt.T`. This takes less than **0.05 milliseconds**.
      * **t-SNE**, by contrast, is a **non-parametric optimization problem**: it optimizes the 2D coordinates of existing points via gradient descent over KL divergence. It does not learn a transformation function.
      * There is no formula to project an out-of-sample query without adding the query to the corpus and re-running the entire gradient descent optimization over all points!

    **Q2: What is the Scree Test and how do you determine the optimal number of dimensions k?**
    * **Answer**:
      * The **Scree Plot** plots the eigenvalues (or explained variance ratio) against component index.
      * Engineers look for the **"Elbow"**: the inflection point where the rate of explained variance drops sharply and flattens into a long plateau.
      * **Rule of Thumb for Dense Retrieval**: Select $k$ such that the cumulative explained variance reaches **85% to 92%**.
      * Slicing past the elbow yields diminishing semantic returns while increasing vector memory footprints linearly.

    **Q3: When should an engineer use TruncatedSVD instead of standard PCA?**
    * **Answer**:
      * PCA centers the input matrix by subtracting column means: X - mu.
      * If your input is a **sparse matrix** (e.g. `scipy.sparse.csr_matrix` containing 100,000 documents with a 50,000-term TF-IDF vocabulary), 99% of entries are zero.
      * Subtracting the mean turns all zeros into non-zero floats, instantly turning a 100 MB sparse matrix into a **40 GB dense matrix**, triggering an immediate Out-of-Memory crash!
      * **TruncatedSVD** performs SVD directly without centering, preserving sparsity throughout the factorization.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting to Re-Normalize after PCA Projection**:
      * *Issue*: Projecting unit-normalized vectors via `pca.transform()` and computing dot products directly.
      * *Consequence*: The projected vectors no longer lie on the unit hypersphere ($L_2 \\text{ norm} \\neq 1.0$). Dot products become distorted by coordinate magnitude rather than angular similarity!
      * *Fix*: Always call `v_proj / np.linalg.norm(v_proj)` immediately after projection.

    * **Trap 2: Fitting PCA on Test Queries**:
      * *Issue*: Calling `pca.fit_transform(queries)` on the query set rather than using the pre-fitted corpus PCA model.
      * *Consequence*: Data leakage and misaligned coordinate spaces: queries and corpus documents end up in different, non-overlapping coordinate manifolds!
      * *Fix*: Fit PCA strictly on the training corpus: `pca.fit(corpus)`, then call `pca.transform()` on queries.

    * **Trap 3: Over-Compressing Dense Vectors for Final Generation**:
      * *Issue*: Compressing 768-d vectors down to 16 dimensions for final RAG retrieval.
      * *Consequence*: Compressing past the variance knee loses nuanced semantic distinctions, causing fine-grained queries to fail.
      * *Fix*: Target 64 to 256 dimensions, or use Matryoshka Representation Learning (MRL).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Explained Variance Ratio**
      * *Definition*: The percentage of total dataset variance captured by each individual principal component in PCA / SVD.
    * **Flashcard 2: Out-of-Sample Projection**
      * *Definition*: The ability of parametric models (PCA, TruncatedSVD) to map new runtime vectors into the learned compressed subspace via .transform().
    * **Flashcard 3: TruncatedSVD**
      * *Definition*: Singular Value Decomposition executed without mean centering, preserving memory efficiency for sparse input matrices.
    * **Flashcard 4: Scree Elbow**
      * *Definition*: The inflection point on the cumulative variance curve indicating optimal dimension cutoff where additional dimensions yield diminishing returns.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Dimensionality Reduction with Scikit-learn:

- [ ] I can describe the mathematical and architectural differences between PCA and TruncatedSVD.
- [ ] I understand why PCA supports out-of-sample `.transform()` projections while t-SNE does not.
- [ ] I know how to use the Scree cumulative variance curve to select the optimal target dimension $k$.
- [ ] I understand why $L_2$ re-normalization is mandatory immediately following PCA projection.
- [ ] I know why TruncatedSVD is required when processing sparse TF-IDF or hybrid embedding matrices.
""")
