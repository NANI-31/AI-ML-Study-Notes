import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.4 Embedding Properties - Vector DB Curriculum",
    page_icon="🔬",
    layout="wide"
)

# Title & Overview
st.title("2.4 Embedding Properties: Geometry, Normalization, Anisotropy & Space Quality")
st.caption("Chapter 2: Understanding Vectors and Embeddings | Domain: 4. Vector Database")
st.markdown("""
To design high-performance vector databases and diagnostic retrieval pipelines, data engineers must understand 
the **geometric and statistical properties** of high-dimensional embedding spaces.

Embeddings are not just arbitrary numbers; they live on complex mathematical manifolds. 
Understanding how **angular similarity** relates to **Euclidean distance**, why **L2 normalization** is the 
universal standard in SIMD hardware acceleration, how **anisotropy (representation collapse)** degrades retrieval, 
and how to detect the **hubness problem** is what separates junior practitioners from principal retrieval architects.
""")

st.divider()

# Section 1: Semantic Similarity vs. Embedding Distance
st.header("1. Semantic Similarity vs. Embedding Distance", divider="rainbow")

st.markdown("""
In vector databases, search algorithms operate on either **similarity functions** (higher score is better) 
or **distance metrics** (lower score is better).
""")

col_prop1, col_prop2 = st.columns([1, 1])

with col_prop1:
    st.markdown("""
    ### 1. The Core Metrics Formulated
    * **Cosine Similarity**:
      * Measures the cosine of the angle $\\theta$ between two vectors, regardless of their magnitude:
        `cos_sim(u, v) = (u . v) / (||u|| * ||v||)`.
      * Bounded range: `[-1.0, 1.0]`. In text embeddings with non-negative activations, typically `[0.0, 1.0]`.
    * **Cosine Distance**:
      * The complementary distance metric: `cos_dist(u, v) = 1.0 - cos_sim(u, v)`.
      * Range: `[0.0, 2.0]`. Identical vectors have distance `0.0`.
    * **Euclidean Distance (L2)**:
      * The straight-line cartesian distance between two coordinate points:
        `L2(u, v) = sqrt(sum((u_i - v_i)^2))`.
      * Sensitive to vector magnitude (document length).
    * **Dot Product (Inner Product)**:
      * Direct sum of element-wise products: `dot(u, v) = sum(u_i * v_i) = ||u|| * ||v|| * cos(theta)`.
      * If both vectors are L2-normalized (`||u|| = ||v|| = 1.0`), then **Dot Product == Cosine Similarity**!
    """)

with col_prop2:
    st.markdown("### Hypersphere Geometry: Angular vs Euclidean Chord")
    geom_dot = """
    digraph HypersphereGeometry {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Origin [label="Vector Space Origin (0, 0, ..., 0)", fillcolor="#EFF6FF", color="#3B82F6", shape=circle];
        
        subgraph cluster_sphere {
            label="Unit Hypersphere Surface (||v|| = 1.0)"; style="dashed"; color="#10B981";
            VecU [label="Vector u\\n(Normalized Point)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            VecV [label="Vector v\\n(Normalized Point)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        }

        Origin -> VecU [label="||u|| = 1.0"];
        Origin -> VecV [label="||v|| = 1.0\\nAngle theta"];
        VecU -> VecV [label="Euclidean Chord Distance\\n||u - v||^2 = 2 - 2*(u . v)", color="#DC2626", style="bold"];
    }
    """
    st.graphviz_chart(geom_dot, use_container_width=False)

st.divider()

# Section 2: Mathematical Proof of L2 Normalization
st.header("2. Why L2 Normalization Powers Vector Database Engines", divider="rainbow")

st.markdown("""
Why do production vector databases insist that you **L2-normalize** your vectors before indexing? 
Because of a remarkable mathematical equivalence between Euclidean distance and Dot product on the unit hypersphere:
""")

st.markdown("""
**The Unit Normalization Equivalence Proof:**
1. Consider the squared Euclidean distance between two vectors `u` and `v`:
   `||u - v||^2 = sum((u_i - v_i)^2) = sum(u_i^2 - 2*u_i*v_i + v_i^2)`.
2. Grouping terms:
   `||u - v||^2 = sum(u_i^2) + sum(v_i^2) - 2 * sum(u_i * v_i) = ||u||^2 + ||v||^2 - 2 * (u . v)`.
3. If both vectors are L2-normalized such that `||u|| = 1.0` and `||v|| = 1.0`:
   `||u - v||^2 = 1 + 1 - 2 * (u . v) = 2 - 2 * cos_sim(u, v)`.
4. Therefore:
   * **Maximizing Cosine Similarity** is mathematically identical to **Minimizing Euclidean Distance**!
   * **Hardware Impact**: Computing `sqrt(...)` and norms requires tens of CPU clock cycles. A pure dot product `(u . v)` compiles into a single fused multiply-add (FMA) instruction on modern SIMD AVX-512 and CUDA cores, accelerating query throughput by **300% to 500%**!
""")

st.divider()

# Section 3: Intrinsic Dimensionality & Embedding Quality Flaws
st.header("3. Intrinsic Dimensionality, Anisotropy & The Hubness Problem", divider="rainbow")

st.markdown("""
High-dimensional spaces suffer from statistical pathologies that can silently degrade search precision:
""")

col_flaw1, col_flaw2, col_flaw3 = st.columns(3)

with col_flaw1:
    st.markdown("""
    #### 1. Intrinsic vs. Ambient Dimension
    * **Ambient Dimension**: The number of coordinates in the raw vector (e.g. 1536 floats).
    * **Intrinsic Dimension**: The true degrees of freedom needed to capture the semantic manifold.
    * **The Insight**: Most NLP manifolds have an intrinsic dimension of only **30 to 80 dimensions**!
    * **Techniques**:
      * Matryoshka Representation Learning (MRL) isolates intrinsic signals in early dimensions.
      * PCA / UMAP compresses vectors while preserving 95%+ of neighborhood topology.
    """)

with col_flaw2:
    st.markdown("""
    #### 2. The Anisotropy Collapse Problem
    * **What is Anisotropy?**:
      * In un-fine-tuned transformers (like raw BERT), vectors collapse into a **narrow high-dimensional cone**.
      * All pairs of vectors exhibit artificially high cosine similarity (e.g. 0.85 to 0.95), even when completely unrelated!
    * **Root Cause**:
      * Word frequency bias and dominant singular vectors in attention projection matrices.
    * **Fix**:
      * Contrastive learning fine-tuning, mean-centering, or PCA whitening.
    """)

with col_flaw3:
    st.markdown("""
    #### 3. The Hubness Problem
    * **What is Hubness?**:
      * A geometric phenomenon where a small subset of vectors ("hubs") emerge as the nearest neighbors to a vast, disproportionate number of queries!
    * **Impact on Vector DBs**:
      * Certain generic documents (e.g. legal disclaimers, boilerplate headers) dominate search results across unrelated topics.
    * **Fix**:
      * Cross-document length penalty, centering vector spaces, and MMR (Maximal Marginal Relevance) diversification.
    """)

st.divider()

# Section 4: Interactive Vector Geometry & Anisotropy Simulator
st.header("4. Interactive Vector Geometry & Anisotropy Simulator", divider="rainbow")

st.markdown("""
Manipulate vector angles, magnitudes, and space distribution to observe the direct relationship between 
**Cosine Similarity**, **Euclidean Distance**, **Dot Product**, and space isotropy:
""")

col_sim_in, col_sim_out = st.columns([1, 1])

with col_sim_in:
    st.subheader("Vector Geometry Controls")
    angle_deg = st.slider("Angle Between Vectors (Degrees)", min_value=0.0, max_value=180.0, value=30.0, step=5.0)
    angle_rad = np.radians(angle_deg)
    
    col_mag1, col_mag2 = st.columns(2)
    with col_mag1:
        mag_u = st.slider("Magnitude of Vector u (||u||)", 0.2, 3.0, 1.0, 0.1)
    with col_mag2:
        mag_v = st.slider("Magnitude of Vector v (||v||)", 0.2, 3.0, 1.0, 0.1)
        
    normalize_first = st.checkbox("Enforce L2 Unit Normalization (Set ||u|| = ||v|| = 1.0)", value=True)
    if normalize_first:
        eff_mag_u = 1.0
        eff_mag_v = 1.0
    else:
        eff_mag_u = mag_u
        eff_mag_v = mag_v

with col_sim_out:
    st.subheader("Calculated Geometric Properties")
    
    # 2D coordinates representing the vectors with the chosen angle and magnitudes
    vec_u = np.array([eff_mag_u, 0.0])
    vec_v = np.array([eff_mag_v * np.cos(angle_rad), eff_mag_v * np.sin(angle_rad)])
    
    cos_sim = float(np.dot(vec_u, vec_v) / (np.linalg.norm(vec_u) * np.linalg.norm(vec_v)))
    cos_dist = 1.0 - cos_sim
    dot_prod = float(np.dot(vec_u, vec_v))
    l2_dist = float(np.linalg.norm(vec_u - vec_v))
    sq_l2_dist = l2_dist ** 2
    
    m_a, m_b = st.columns(2)
    with m_a:
        st.metric("Cosine Similarity", f"{cos_sim:.4f}", f"cos({angle_deg:.0f}°)")
        st.metric("Cosine Distance (1 - sim)", f"{cos_dist:.4f}")
    with m_b:
        st.metric("Dot Product (Inner Product)", f"{dot_prod:.4f}")
        st.metric("Euclidean Distance (L2)", f"{l2_dist:.4f}", f"Squared L2: {sq_l2_dist:.4f}")
        
    if normalize_first:
        st.success(f"**Unit Equivalence Verified!**\n\n`Squared L2 (2 - 2*dot)` = 2 - 2*({dot_prod:.4f}) = **{2 - 2*dot_prod:.4f}** (Matches `{sq_l2_dist:.4f}` exactly!)")
    else:
        st.warning(f"**Vectors are unnormalized!** Notice how Dot Product ({dot_prod:.4f}) deviates from Cosine Similarity ({cos_sim:.4f}) due to vector magnitudes.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Space Quality, Anisotropy & Hubness Diagnostics", divider="rainbow")

st.markdown("""
The following complete Python diagnostic suite evaluates an embedding collection for 
**L2 normalization compliance**, **anisotropy (cosine cone collapse)**, and **hubness score**:
""")

code_diag = '''import numpy as np
from typing import Dict, Any

class EmbeddingSpaceDiagnostics:
    """Diagnostic suite for evaluating vector space health in vector databases."""
    def __init__(self, vectors: np.ndarray):
        # Shape: (N, d)
        self.vectors = vectors
        self.num_vectors, self.dimension = vectors.shape

    def check_normalization(self, tolerance: float = 1e-4) -> Dict[str, Any]:
        """Checks if vectors are strictly L2-normalized."""
        norms = np.linalg.norm(self.vectors, axis=1)
        is_normalized = np.all(np.abs(norms - 1.0) < tolerance)
        return {
            "is_normalized": bool(is_normalized),
            "mean_norm": float(np.mean(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms))
        }

    def measure_anisotropy(self, sample_size: int = 200) -> Dict[str, Any]:
        """Measures average cosine similarity between random pairs to detect cone collapse."""
        idx_a = np.random.choice(self.num_vectors, size=sample_size, replace=True)
        idx_b = np.random.choice(self.num_vectors, size=sample_size, replace=True)
        
        # Normalize for evaluation
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        unit_vecs = self.vectors / np.where(norms > 0, norms, 1.0)
        
        sims = np.sum(unit_vecs[idx_a] * unit_vecs[idx_b], axis=1)
        mean_sim = float(np.mean(sims))
        
        # In a healthy isotropic space, mean cosine similarity of random pairs is close to 0.0
        # If mean_sim > 0.6, the space suffers from severe anisotropy
        status = "Healthy (Isotropic)" if mean_sim < 0.3 else "Degraded (Anisotropic Cone Collapse)"
        return {
            "mean_random_cosine_similarity": mean_sim,
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
            "space_health": status
        }

    def detect_hubness(self, top_k: int = 5) -> Dict[str, Any]:
        """Calculates k-occurrence frequency (how often a node appears as a top-k neighbor)."""
        # Pairwise dot product (for unit vectors)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        unit_vecs = self.vectors / np.where(norms > 0, norms, 1.0)
        sim_matrix = np.dot(unit_vecs, unit_vecs.T)
        
        # Zero out diagonal (self-match)
        np.fill_diagonal(sim_matrix, -1.0)
        
        # Get top-k neighbors for each vector
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -top_k:]
        
        # Count occurrences
        counts = np.bincount(top_k_indices.flatten(), minlength=self.num_vectors)
        max_occurrences = int(np.max(counts))
        skewness = float(np.std(counts) / (np.mean(counts) + 1e-6))
        
        return {
            "max_hub_appearances": max_occurrences,
            "mean_appearances": float(np.mean(counts)),
            "hubness_skewness": skewness,
            "hubness_risk": "High (Hub present)" if max_occurrences > (3 * top_k) else "Low"
        }

# Demonstration
np.random.seed(42)
N, d = 500, 128

# 1. Healthy Isotropic Space
healthy_space = np.random.randn(N, d)
norms = np.linalg.norm(healthy_space, axis=1, keepdims=True)
healthy_space = healthy_space / norms

# 2. Degraded Anisotropic Space (Vectors biased toward a single dominant direction)
bias_direction = np.ones(d) * 2.0
collapsed_space = (np.random.randn(N, d) * 0.2) + bias_direction
collapsed_norms = np.linalg.norm(collapsed_space, axis=1, keepdims=True)
collapsed_space = collapsed_space / collapsed_norms

diag_healthy = EmbeddingSpaceDiagnostics(healthy_space)
diag_collapsed = EmbeddingSpaceDiagnostics(collapsed_space)

print("Healthy Space Diagnostics:")
print(diag_healthy.measure_anisotropy())

print("\\nCollapsed (Anisotropic) Space Diagnostics:")
print(diag_collapsed.measure_anisotropy())
'''

st.code(code_diag, language="python")

if st.button("🚀 Run Vector Space Diagnostic Suite", type="primary"):
    exec_scope = {}
    exec(code_diag, globals(), exec_scope)
    
    h_res = exec_scope["diag_healthy"].measure_anisotropy()
    c_res = exec_scope["diag_collapsed"].measure_anisotropy()
    
    st.success("Diagnostics executed successfully!")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Healthy Isotropic Space Report**:")
        st.json(h_res)
    with col_d2:
        st.markdown("**Collapsed Anisotropic Space Report**:")
        st.json(c_res)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Mathematically prove why maximizing Cosine Similarity is equivalent to minimizing Euclidean Distance for unit vectors."""
a1 = """* For two vectors `u` and `v`, the squared Euclidean distance is:
  `||u - v||^2 = ||u||^2 + ||v||^2 - 2 * (u . v)`.
* If both vectors are L2-normalized such that `||u|| = 1.0` and `||v|| = 1.0`:
  `||u - v||^2 = 1 + 1 - 2 * (u . v) = 2 - 2 * cos_sim(u, v)`.
* Because `2` is a constant and `-2` is a negative multiplier:
  * As `cos_sim(u, v)` increases toward `1.0`, `||u - v||^2` monotonically decreases toward `0.0`.
* Thus, the nearest neighbor according to Euclidean distance is guaranteed to be the exact same nearest neighbor according to Cosine similarity on unit vectors."""

q2 = """2. What is the Anisotropy problem in deep language representations, and why does it harm vector database retrieval?"""
a2 = """* **What It Is**: Anisotropy occurs when learned embeddings collapse into a narrow cone in vector space rather than occupying the full hypersphere uniformly.
* **Why It Happens**: Pre-trained language models without contrastive fine-tuning suffer from dominant token frequency biases, which align the representations of most words along a few dominant singular dimensions.
* **Why It Harms Retrieval**:
  * Unrelated documents have high baseline cosine similarities (e.g., 0.85).
  * The dynamic range of similarity scores collapses into a narrow band (`[0.85, 0.95]`), reducing the discriminative power of Top-K ranking and making score thresholding nearly impossible.
* **Solution**: Contrastive fine-tuning (e.g. SimCSE, BGE, OpenAI v3) or post-processing via PCA whitening / mean centering."""

q3 = """3. What is the Hubness Problem in high-dimensional vector search, and how do you mitigate it?"""
a3 = """* **Definition**: In high-dimensional spaces (e.g., `d > 500`), certain data points ("hubs") act as nearest neighbors to an abnormally large number of queries, while other valid points ("orphans") are never retrieved.
* **Root Causes**:
  * Boundary concentration of points and local density variations near the hypersphere boundary.
  * Generic boilerplate text chunks (e.g., privacy policies, copyright notices, table of contents headers) that share weak semantic links with many disparate topics.
* **Mitigation**:
  1. Centering vectors and whitening the embedding space.
  2. Applying Maximal Marginal Relevance (MMR) to diversify retrieved sets.
  3. Pre-indexing heuristic filters to strip repetitive boilerplate from document corpora."""

q4 = """4. Does Cosine Distance satisfy the mathematical properties of a true metric?"""
a4 = """* **Strictly speaking, NO.**
* Cosine Distance defined as `D_cos(u, v) = 1 - cos_sim(u, v)` violates the **Triangle Inequality**:
  `D(u, w) <= D(u, v) + D(v, w)`.
  * *Counterexample*: Two perpendicular vectors `u` and `v` (`D(u, v) = 1.0`), and `v` and `w` (`D(v, w) = 1.0`), where `u` and `w` are opposite (`D(u, w) = 2.0`). But for intermediate angles, the triangle inequality can fail.
* **Angular Distance** `D_ang(u, v) = arccos(cos_sim(u, v)) / pi` is a true metric satisfying all axioms (including triangle inequality).
* **Production Implication**: Algorithms requiring strict metric spaces (like VP-Trees or BK-Trees) fail on raw cosine distance; however, HNSW graphs work robustly on cosine distance because they rely on greedy neighbor hops rather than strict metric tree pruning."""

q5 = """5. How does Matryoshka Representation Learning (MRL) allow flexible embedding dimensions without retraining?"""
a5 = """* Standard models optimize a single loss over the full vector `v in R^d`.
* MRL optimizes multiple nested loss functions simultaneously during training:
  `Loss_total = Loss(v_{1:64}) + Loss(v_{1:128}) + Loss(v_{1:256}) + Loss(v_{1:512}) + Loss(v_{1:d})`.
* This forces the transformer to pack the most fundamental semantic discriminators into the first 64 and 128 dimensions, with subsequent dimensions capturing subtle stylistic nuances.
* As a result, you can slice the vector `v[:512]`, re-normalize it to unit length, and achieve 98%+ of the full 1536-d retrieval recall while cutting vector database RAM by 66%."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Embedding Properties Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Inner Product Metric Without Pre-Normalization**  
> When creating a vector collection in Milvus, Qdrant, or Pinecone, selecting `MetricType.IP` (Inner Product) 
> assumes vectors are already normalized. If unnormalized vectors are inserted, long chunks with larger L2 norms 
> will systematically achieve higher inner product scores regardless of actual semantic relevance. 
> Always enforce unit normalization before ingestion when using Inner Product.

> [!WARNING]
> **Trap 2: Truncating Non-MRL Embeddings**  
> Attempting to save RAM by arbitrarily slicing a standard embedding vector (e.g. `vec[:256]` on an older model 
> not trained with MRL) destroys the geometric manifold, causing search recall to collapse from 95% down to 20%! 
> Only truncate embeddings if the model explicitly certifies Matryoshka Representation Learning (MRL) support.

> [!WARNING]
> **Trap 3: Score Threshold Traps Across Different Metrics**  
> A similarity threshold of `0.75` on Cosine Similarity does NOT mean the same thing on Dot Product or Euclidean Distance. 
> In Euclidean distance, smaller values are closer (`0.0` is identical). On unnormalized dot products, scores can be `+45.0`. 
> Never hardcode retrieval filters across metric types without understanding their mathematical bounds.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is L2 Normalization?"):
        st.markdown("""
        **L2 Unit Normalization**:  
        Scaling a vector by its Euclidean norm (`v / ||v||`) so its length equals 1.0, projecting it onto the unit 
        hypersphere where Dot Product equals Cosine Similarity.
        """)
    with st.expander("🗂️ Flashcard: What is Anisotropy?"):
        st.markdown("""
        **Anisotropy**:  
        A degenerate condition where embeddings cluster inside a narrow cone, causing high baseline similarity 
        across unrelated texts and weakening retrieval ranking.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Hubness Problem?"):
        st.markdown("""
        **Hubness Problem**:  
        A high-dimensional geometric artifact where certain vectors emerge as nearest neighbors to an excessive 
        number of queries, drowning out genuine matches.
        """)
    with st.expander("🗂️ Flashcard: Why is Dot Product Faster than Cosine in Hardware?"):
        st.markdown("""
        **Hardware Acceleration**:  
        Dot products on pre-normalized vectors use single-cycle SIMD multiply-accumulate instructions without 
        expensive square roots or norm divisions.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can mathematically prove why maximizing Cosine Similarity equals minimizing Euclidean distance on unit vectors.", value=True)
chk2 = st.checkbox("I understand the exact relationship between Cosine Similarity, Cosine Distance, and Dot Product.")
chk3 = st.checkbox("I know how to diagnose and detect Anisotropy in an embedding space.")
chk4 = st.checkbox("I can explain the Hubness Problem and identify methods to mitigate it in production.")
chk5 = st.checkbox("I understand why Matryoshka Representation Learning (MRL) enables loss-free dimension slicing.")
chk6 = st.checkbox("I can implement a vector space diagnostic script in Python using NumPy.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Incredible! You have mastered Embedding Properties and completed Chapter 2 of the Vector Database curriculum!")
