import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.1 Vectors - Embeddings Curriculum",
    page_icon="📐",
    layout="wide"
)

# Title & Overview
st.title("2.1 Vectors: Scalars, Dimensions, Magnitude, Direction, & High-Dimensional Geometry")
st.caption("Chapter 2: Mathematical Foundations | Domain: 3. Vector Embeddings")
st.markdown("""
To understand how modern AI models compare ideas, retrieve knowledge, and cluster documents, 
one must first master the language of **linear algebra and vector geometry**.

At its core:
* A **Scalar** is a single quantity representing magnitude alone (e.g., price = $45.0, temperature = 22.5).
* A **Vector** is an ordered sequence of numbers possessing **both magnitude (length) and direction** in an $N$-dimensional space.
* In language models, each dimension of an embedding vector represents an abstract semantic latent axis (e.g., formality, temporal tense, factual domains).

However, vector geometry in **high dimensions (384, 768, 1536 dimensions)** behaves in ways that defy human 3D intuition: 
random high-dimensional vectors are almost universally **orthogonal**, and geometric volume concentrates entirely on the outer surface of hyperspheres.

This lesson explores vector representation, magnitude calculation (L1 vs. L2 norms), unit direction, and high-dimensional geometry.
""")

st.divider()

# Section 1: Foundations of Vectors & High Dimensions
st.header("1. Vector Geometry: From 2D Space to 1536D Hyperspaces", divider="rainbow")

col_vg1, col_vg2 = st.columns([1, 1])

with col_vg1:
    st.markdown("""
    ### Fundamental Mathematical Concepts:
    * **1. Vector Dimensions**:
      A point in D-dimensional space: v = [v1, v2, ..., vD]. In embedding models, D ranges from 384 (compact models) to 3072 (frontier encoders).
    * **2. Vector Magnitude (L2 Euclidean Norm)**:
      The geometric length from origin (0, 0, ..., 0) to coordinate v:
      `||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_D^2)`
    * **3. Unit Direction Vectors**:
      Dividing a vector by its magnitude produces a **unit vector** (norm = 1.0), stripping away scale and preserving pure semantic orientation.
    * **4. The High-Dimensional Orthogonality Phenomenon**:
      In 2D space, two random vectors have a broad spread of angles. In 1536D space, virtually **any two randomly sampled vectors are nearly perpendicular (cosine similarity near 0.0)** because the probability mass concentrates in equatorial planes.
    """)

with col_vg2:
    st.markdown("### Vector Decomposition")
    vec_dot = """
    digraph VectorStructure {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawVec [label="Vector v\\n[3.0, 4.0]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Magnitude [label="Magnitude ||v||\\nsqrt(3^2 + 4^2) = 5.0", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        UnitDir [label="Unit Direction Vector v_hat\\n[3/5, 4/5] = [0.6, 0.8]\\n(Norm = 1.0)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        RawVec -> Magnitude [label="Compute Norm"];
        RawVec -> UnitDir [label="v / ||v||"];
    }
    """
    st.graphviz_chart(vec_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Vector Mathematics in Python", divider="rainbow")
st.markdown("""
The following Python module demonstrates vector representation, norm computations (L1 vs L2), unit normalization, and the high-dimensional orthogonality simulator.
""")

class ProductionVectorGeometryEngine:
    """
    Mathematical engine for vector calculations, norm derivations,
    and high-dimensional geometric distributions.
    """
    @staticmethod
    def compute_vector_norms(v: np.ndarray) -> Dict[str, float]:
        """Calculates L1 (Manhattan) and L2 (Euclidean) norms."""
        l1_norm = float(np.sum(np.abs(v)))
        l2_norm = float(np.sqrt(np.sum(v ** 2)))
        max_norm = float(np.max(np.abs(v)))
        return {
            "l1_norm": round(l1_norm, 4),
            "l2_norm": round(l2_norm, 4),
            "max_norm": round(max_norm, 4)
        }

    @staticmethod
    def normalize_to_unit_vector(v: np.ndarray) -> np.ndarray:
        """Scales vector to unit Euclidean length (L2 norm = 1.0)."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def simulate_high_dim_orthogonality(dimensions: int, num_pairs: int = 1000) -> np.ndarray:
        """
        Samples random Gaussian vector pairs and computes their cosine similarities
        to demonstrate high-dimensional orthogonality concentration.
        """
        # Sample random vectors from standard normal distribution
        v1 = np.random.randn(num_pairs, dimensions)
        v2 = np.random.randn(num_pairs, dimensions)

        # Normalize rows to unit length
        v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

        # Compute pairwise dot products (cosine similarities)
        cosines = np.sum(v1_norm * v2_norm, axis=1)
        return cosines

code_snippet = r'''
import numpy as np

class VectorMath:
    @staticmethod
    def l2_norm(v: np.ndarray) -> float:
        return float(np.sqrt(np.sum(v ** 2)))

    @staticmethod
    def to_unit_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @staticmethod
    def random_cosine_distribution(dim: int, n_samples: int = 1000) -> np.ndarray:
        v1 = np.random.randn(n_samples, dim)
        v2 = np.random.randn(n_samples, dim)
        v1_u = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_u = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        return np.sum(v1_u * v2_u, axis=1) # Cosine similarities
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Vector Geometry Workbench
st.header("3. Interactive Vector Geometry & High-Dimension Workbench", divider="rainbow")
st.markdown("""
Experiment with vector magnitude and direction in 3D space, or test the **High-Dimensional Orthogonality Simulator** to see how random vectors behave as dimensions scale from 2 to 1536.
""")

engine = ProductionVectorGeometryEngine()

tab_v1, tab_v2 = st.tabs(["1. 3D Vector Coordinate Inspector", "2. High-Dimensional Orthogonality Simulator"])

with tab_v1:
    st.markdown("### Interactive 3D Vector Geometry")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        x_val = st.slider("Coordinate X (Dim 1):", min_value=-10.0, max_value=10.0, value=3.0, step=0.5)
    with col_c2:
        y_val = st.slider("Coordinate Y (Dim 2):", min_value=-10.0, max_value=10.0, value=4.0, step=0.5)
    with col_c3:
        z_val = st.slider("Coordinate Z (Dim 3):", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)

    test_vec = np.array([x_val, y_val, z_val])
    norms = engine.compute_vector_norms(test_vec)
    unit_vec = engine.normalize_to_unit_vector(test_vec)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("L2 Euclidean Magnitude", f"{norms['l2_norm']:.4f}")
    col_m2.metric("L1 Manhattan Magnitude", f"{norms['l1_norm']:.4f}")
    col_m3.metric("L-Infinity Max Coordinate", f"{norms['max_norm']:.4f}")

    st.markdown("##### Unit Vector Representation (Direction Only):")
    st.code(f"Unit Vector: [{unit_vec[0]:.4f}, {unit_vec[1]:.4f}, {unit_vec[2]:.4f}] (Length = {np.linalg.norm(unit_vec):.4f})")

with tab_v2:
    st.markdown("### High-Dimensional Orthogonality Simulator")
    st.markdown("""
    Select an embedding dimension. The simulator generates 2,000 random vectors and plots the distribution of pairwise cosine similarities.
    Notice how in **2D**, random angles are widely dispersed between -1.0 and +1.0. In **768D or 1536D**, the distribution **collapses sharply to 0.00**!
    """)

    dim_selected = st.select_slider(
        "Choose Vector Dimensionality (D):",
        options=[2, 5, 25, 128, 384, 768, 1536],
        value=384
    )

    cosines = engine.simulate_high_dim_orthogonality(dim_selected, num_pairs=2000)
    mean_cos = float(np.mean(cosines))
    std_cos = float(np.std(cosines))

    col_st1, col_st2, col_st3 = st.columns(3)
    col_st1.metric("Selected Dimension", f"{dim_selected}D")
    col_st2.metric("Mean Cosine Similarity", f"{mean_cos:.5f}")
    col_st3.metric("Standard Deviation (Spread)", f"{std_cos:.5f}", delta="Sharp Concentration" if std_cos < 0.05 else "Wide Dispersion")

    # Histogram representation via pandas
    counts, bin_edges = np.histogram(cosines, bins=25, range=(-1.0, 1.0))
    hist_df = pd.DataFrame({
        "Cosine Bin Range": [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(len(counts))],
        "Frequency Count": counts
    })
    st.bar_chart(hist_df.set_index("Cosine Bin Range"), use_container_width=True)
    st.caption(f"In {dim_selected} dimensions, 99% of random vector pairs share a cosine similarity within {std_cos*2.5:.3f} of zero.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Vector Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the geometric intuition behind the L1 norm vs. L2 norm?**
    * **Answer**:
      * **L1 Norm (Manhattan / Taxicab)**: The sum of absolute coordinate differences: `sum(|x_i|)`. Represents the grid distance traveled along axis-aligned city streets.
      * **L2 Norm (Euclidean)**: The straight-line distance "as the crow flies": `sqrt(sum(x_i^2))`. In linear algebra and machine learning, L2 is the standard definition of vector magnitude because it is rotationally invariant.

    **Q2: Why are random vectors in high-dimensional spaces (e.g. 1536D) almost always orthogonal?**
    * **Answer**:
      * Consider the dot product of two random unit vectors u and v with independent zero-mean components:
        `u . v = sum(u_i * v_i)`.
      * By the Central Limit Theorem, this sum of D independent random variables has mean 0 and variance 1/D.
      * As dimension D increases to 768 or 1536, the standard deviation shrinks to approximately 1 / sqrt(1536) = 0.025.
      * Almost the entirety of the high-dimensional hypersphere volume is concentrated within an equatorial band of angle theta near 90 degrees relative to any reference vector.

    **Q3: Why do embedding models typically use 768, 1024, or 1536 dimensions rather than 50 or 100,000?**
    * **Answer**:
      * **Too Low (e.g. 50D)**: Information bottleneck. Lacks capacity to distinguish subtle semantic nuances across millions of concepts, leading to high collision rates.
      * **Too High (e.g. 50,000D)**: Computationally prohibitive. Memory consumption and dot product latency scale linearly with dimension (O(D)), while HNSW indexing slows dramatically.
      * **The Sweet Spot (384 - 1536D)**: Provides sufficient representational capacity for human language while fitting within high-speed GPU cache lines and vector DB memory budgets.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Confusing Vector Magnitude with Semantic Importance**:
      * *Issue*: Assuming that a vector with magnitude 15.0 is "more important" or "more informative" than a vector with magnitude 2.0.
      * *Consequence*: Magnitude in unnormalized text models is often an artifact of token frequency or sentence length rather than semantic significance.
      * *Fix*: Normalize all vectors to unit length so comparison is strictly directional.

    * **Trap 2: Applying 3D Spatial Intuition to High-Dimensional Hyperspheres**:
      * *Issue*: Assuming that a 1536D hypersphere has most of its volume near the center (origin).
      * *Consequence*: In reality, virtually 100% of the volume of a high-dimensional ball resides in a microscopic thin shell near the surface.
      * *Fix*: Rely on mathematical derivations rather than spatial visualization when designing distance metrics.

    * **Trap 3: Overflow / Underflow in Float16 Norm Computations**:
      * *Issue*: Computing `sum(x^2)` in `Float16` on high-dimensional vectors with large values.
      * *Consequence*: Numbers exceed 65,504, overflowing to `inf` and producing `NaN` unit vectors.
      * *Fix*: Compute vector norms in `Float32` before casting back to lower precision.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Vector Magnitude (L2 Norm)**
      * *Definition*: The Euclidean length of a vector from origin to point, calculated as the square root of the sum of squared coordinates.
    * **Flashcard 2: Unit Vector**
      * *Definition*: A vector scaled to have an L2 norm of exactly 1.0, preserving pure directional orientation.
    * **Flashcard 3: Orthogonality in High Dimensions**
      * *Definition*: The mathematical property where random vectors in high-dimensional spaces have cosine similarities clustering tightly around 0.0.
    * **Flashcard 4: Scalar vs. Vector**
      * *Definition*: A scalar is a single real number with magnitude only; a vector is an ordered sequence of numbers with magnitude and direction.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Vector Foundations:

- [ ] I can explain the difference between a scalar and a vector.
- [ ] I know how to calculate the L1 norm and L2 norm of a coordinate vector.
- [ ] I understand why normalizing a vector to unit length isolates semantic direction.
- [ ] I can explain why random vectors in 1536-dimensional space are nearly orthogonal.
- [ ] I understand the representational and computational trade-offs of embedding dimensionality.
""")
