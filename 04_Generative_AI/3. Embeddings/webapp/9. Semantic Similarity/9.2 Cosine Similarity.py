import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import math
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="9.2 Cosine Similarity - Embeddings Curriculum",
    page_icon="📐",
    layout="wide"
)

# Title & Overview
st.title("9.2 Cosine Similarity: Angular Geometry, Magnitude Invariance, & Score Interpretation")
st.caption("Chapter 9: Semantic Similarity | Domain: 3. Vector Embeddings")
st.markdown("""
In Euclidean geometry, distance is typically measured as straight-line length (Euclidean Distance).
In high-dimensional **Natural Language Processing**, however, straight-line distance is deceptive: a short 5-word sentence and a 50-word paragraph explaining the exact same concept will have radically different Euclidean lengths simply because of text volume!

**Cosine Similarity** resolves this by measuring the **cosine of the angle theta** between two vectors, completely ignoring their raw magnitudes:
* Two vectors pointing in the exact same direction have an angle of 0 degrees and a cosine similarity of **1.00**.
* Two vectors pointing at right angles (perpendicular) have an angle of 90 degrees and a cosine similarity of **0.00** (orthogonal / unrelated).
* Two vectors pointing in diametrically opposite directions have an angle of 180 degrees and a cosine similarity of **-1.00**.

Because modern transformer embedding models produce unit-normalized vectors ($||u|| = 1.0$), **Cosine Similarity simplifies to a blazing-fast inner dot product**:
`Cosine Similarity = u dot v`

This lesson explores the mathematics, angular interpretation, and production engineering of Cosine Similarity.
""")

st.divider()

# Section 1: The Geometry of Cosine Similarity
st.header("1. The Geometry of Cosine Similarity & Magnitude Invariance", divider="rainbow")

col_geo1, col_geo2 = st.columns([1, 1])

with col_geo1:
    st.markdown("""
    ### Mathematical Formulation (Plain Text):
    Given two high-dimensional vectors `u` and `v` of dimension `d`:
    ```
    Cosine Similarity(u, v) = (u dot v) / (||u|| * ||v||)
                            = sum(u_i * v_i) / (sqrt(sum(u_i^2)) * sqrt(sum(v_i^2)))
    ```

    ### Why Magnitude Invariance is Critical:
    * **Length Bias in Euclidean Distance**:
      * Suppose `u` is the vector for *"Python programming"* (magnitude = 2.0).
      * Suppose `v` is the vector for a 500-word tutorial on Python (magnitude = 10.0).
      * Euclidean distance `||u - v||` is large (~8.2) because of length differences!
      * Cosine similarity normalizes both vectors to unit length, evaluating only the **directional orientation (the angle theta)**.
    * **Score Interpretation in Embedding Space**:
      * **0.90 to 1.00**: Near-identical semantic duplicates or direct answers.
      * **0.75 to 0.89**: Strong topical alignment; highly relevant passage.
      * **0.50 to 0.74**: Broad conceptual overlap; same general domain.
      * **Below 0.50**: Low relevance or unrelated noise.
    """)

with col_geo2:
    st.markdown("### Angular Vector Projection")
    cos_dot = """
    digraph CosineGeometry {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Origin [label="Coordinate Origin (0, 0)", shape=point, width=0.2];
        
        VecA [label="Vector u: 'How to learn Python?'\\nLength = 1.0", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        VecB [label="Vector v: 'Best Python tutorial'\\nLength = 2.5 (Longer text)", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        
        Origin -> VecA [label="u", color="#3B82F6", penwidth=2];
        Origin -> VecB [label="v", color="#10B981", penwidth=2];
        
        Angle [label="Angle theta < 15 deg\\ncos(theta) = 0.965\\nNear-perfect semantic match!", shape=ellipse, fillcolor="#FEF3C7", color="#D97706"];
        VecA -> Angle [style=dashed, dir=none];
        VecB -> Angle [style=dashed, dir=none];
    }
    """
    st.graphviz_chart(cos_dot, use_container_width=False)

st.markdown("### Angular Metric Reference Guide")
st.markdown("""
| Angle Theta (Degrees) | Cosine Similarity Score | Geometric Relationship | Semantic NLP Interpretation |
| :--- | :--- | :--- | :--- |
| **0.0 deg** | **1.0000** | Collinear (Identical direction) | Identical meaning or duplicate sentences |
| **30.0 deg** | **0.8660** | Tight acute angle | Direct answer, close paraphrase, high relevance |
| **45.0 deg** | **0.7071** | Acute angle | Same topic, moderate relevance |
| **60.0 deg** | **0.5000** | Wide acute angle | Broad domain overlap, weak relevance |
| **90.0 deg** | **0.0000** | Orthogonal (Perpendicular) | Completely unrelated concepts |
| **180.0 deg** | **-1.0000** | Diametrically opposite | Opposite directions (rare in dense NLP models) |
""")

st.divider()

# Section 2: Production Code: High-Throughput Cosine Engine
st.header("2. Production Implementation: Vectorized Cosine Engine", divider="rainbow")

st.markdown("""
The production utility below demonstrates both pairwise cosine similarity between two single vectors and vectorized matrix-matrix cosine similarity between a batch of queries and a corpus matrix.
""")

st.code(r'''
import numpy as np
from typing import Union, Tuple

class ProductionCosineEngine:
    """
    High-performance Cosine Similarity Engine.
    Computes pairwise and batched cross-matrix similarity with unit L2 normalization.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes a 1D vector or 2D matrix across rows."""
        norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norm, self.eps)

    def pairwise_similarity(self, u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """
        Computes cosine similarity and angular distance in degrees.
        Returns: (cosine_similarity, angle_degrees)
        """
        u_norm = self.normalize(u).flatten()
        v_norm = self.normalize(v).flatten()
        
        cos_sim = float(np.dot(u_norm, v_norm))
        # Clip to [-1.0, 1.0] to prevent floating point domain errors in arccos
        cos_sim_clipped = float(np.clip(cos_sim, -1.0, 1.0))
        angle_rad = math.acos(cos_sim_clipped)
        angle_deg = math.degrees(angle_rad)
        
        return cos_sim_clipped, angle_deg

    def batch_cosine_similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity matrix between N queries and M corpus documents.
        queries: [N, d]
        corpus:  [M, d]
        Returns: [N, M] similarity matrix
        """
        q_norm = self.normalize(queries)
        c_norm = self.normalize(corpus)
        # Dot product of normalized matrices yields full cosine similarity matrix
        return np.dot(q_norm, c_norm.T)
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveCosineWorkbench:
    def __init__(self):
        self.eps = 1e-12
        self.benchmarks = {
            "Close Paraphrase": {
                "s1": "How do I learn Python programming effectively?",
                "s2": "What is the best way to study Python code?",
                "angle_deg": 14.5
            },
            "Question and Answer Passage": {
                "s1": "What causes hypertension in adults?",
                "s2": "High blood pressure develops when arterial walls face sustained pressure.",
                "angle_deg": 26.2
            },
            "Related but Different Concepts": {
                "s1": "Electric cars battery charging infrastructure.",
                "s2": "Solar panel photovoltaic efficiency in summer.",
                "angle_deg": 58.0
            },
            "Completely Unrelated Domains": {
                "s1": "Quantum computing qubit entanglement experiments.",
                "s2": "Traditional sourdough bread baking techniques.",
                "angle_deg": 88.5
            }
        }

    def compute_cosine(self, angle_deg: float, mag_u: float, mag_v: float) -> Dict[str, Any]:
        rad = math.radians(angle_deg)
        cos_val = math.cos(rad)
        dot_product = cos_val * mag_u * mag_v
        euclidean_dist = math.sqrt(mag_u**2 + mag_v**2 - 2 * mag_u * mag_v * cos_val)
        
        return {
            "angle_deg": angle_deg,
            "angle_rad": rad,
            "cosine_similarity": cos_val,
            "dot_product": dot_product,
            "euclidean_distance": euclidean_dist,
            "mag_u": mag_u,
            "mag_v": mag_v
        }

st.divider()

# Section 3: Interactive Workbench: Cosine Angular Simulator
st.header("3. Interactive Workbench: Cosine Angular Geometry Simulator", divider="rainbow")

st.markdown("""
Experiment with the relationship between **Angular Separation (theta)**, **Vector Magnitudes**, **Dot Product**, and **Cosine Similarity**.
Notice that no matter how much you change the magnitudes of vector `u` and vector `v`, the **Cosine Similarity remains completely unchanged**!
""")

cos_sim = InteractiveCosineWorkbench()

col_cos_l, col_cos_r = st.columns([1, 1])

with col_cos_l:
    st.subheader("1. Geometric & Linguistic Scenarios")
    scenario_mode = st.radio("Configuration Mode:", ["Pre-set NLP Sentence Pairs", "Manual Angle & Magnitude Controls"])

    if scenario_mode == "Pre-set NLP Sentence Pairs":
        pair_key = st.selectbox("Select Benchmark Sentence Pair:", list(cos_sim.benchmarks.keys()))
        p_info = cos_sim.benchmarks[pair_key]
        st.markdown(f"**Sentence 1**: `{p_info['s1']}`")
        st.markdown(f"**Sentence 2**: `{p_info['s2']}`")
        angle_input = p_info["angle_deg"]
        mag_u_val = 1.0
        mag_v_val = 1.0
    else:
        angle_input = st.slider("Separation Angle Theta (Degrees):", min_value=0.0, max_value=180.0, value=30.0, step=1.0)
        col_m1, col_m2 = st.columns(2)
        mag_u_val = col_m1.slider("Magnitude of Vector u:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        mag_v_val = col_m2.slider("Magnitude of Vector v:", min_value=0.1, max_value=10.0, value=4.5, step=0.1)

with col_cos_r:
    st.subheader("2. Metric Calculation")
    calc = cos_sim.compute_cosine(angle_input, mag_u_val, mag_v_val)

    m1, m2 = st.columns(2)
    m1.metric("Angle Theta", f"{calc['angle_deg']:.1f} deg")
    m2.metric("Cosine Similarity (cos theta)", f"{calc['cosine_similarity']:.4f}")

    m3, m4 = st.columns(2)
    m3.metric("Raw Dot Product (u dot v)", f"{calc['dot_product']:.2f}")
    m4.metric("Euclidean Distance (||u - v||)", f"{calc['euclidean_distance']:.2f}")

    st.markdown("#### Interpretation Key")
    if calc['cosine_similarity'] >= 0.85:
        st.success(f"🟢 **High Semantic Similarity ({calc['cosine_similarity']:.4f})**: Vectors share a tight directional cone (theta = {calc['angle_deg']:.1f} deg). Indicates strong paraphrasing or high-confidence retrieval.")
    elif calc['cosine_similarity'] >= 0.65:
        st.info(f"🟡 **Moderate Topical Alignment ({calc['cosine_similarity']:.4f})**: Vectors share the same general domain but may discuss different specific sub-aspects.")
    else:
        st.warning(f"🔴 **Low / Unrelated Similarity ({calc['cosine_similarity']:.4f})**: Vectors are wide-angled (theta = {calc['angle_deg']:.1f} deg), indicating semantic divergence.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Cosine Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Cosine Similarity preferred over Euclidean Distance for text embeddings?**
    * **Answer**:
      * In text embeddings, vector direction represents semantic topic, while vector magnitude reflects text length, word frequency, or token count.
      * Euclidean distance is highly sensitive to vector length: a 10-word query and a 300-word relevant document will have a huge Euclidean distance simply because the document vector is longer.
      * Cosine similarity normalizes both vectors to unit length, evaluating only the angle between them.
      * Note: If vectors are already unit-normalized ($||u|| = ||v|| = 1.0$), Euclidean distance and Cosine similarity are monotonically related:
        `||u - v||^2 = 2 - 2 * Cosine(u, v)`. In this normalized state, both yield identical nearest-neighbor rankings!

    **Q2: Can Cosine Similarity between text embeddings be negative?**
    * **Answer**:
      * Mathematically, cosine ranges from -1.0 to +1.0.
      * In practice, with modern transformer embedding models (SBERT, OpenAI), negative cosine similarities are **extremely rare** (usually bounded between 0.0 and 1.0).
      * Why? Because neural network activation functions (ReLU, GELU) produce primarily non-negative hidden activations, causing the vast majority of vectors to inhabit the positive orthant of the hypersphere.

    **Q3: How do vector databases compute Cosine Similarity at massive scale?**
    * **Answer**:
      * Calculating square roots and vector norms for millions of comparisons is computationally expensive.
      * Vector databases (ChromaDB, FAISS, Milvus) solve this by **pre-normalizing all vectors to unit length at ingestion time**: `v = v / ||v||`.
      * Once vectors are unit-normalized, the denominator `||u|| * ||v||` equals 1.0.
      * The database then computes Cosine Similarity using a pure **Inner Product (BLAS GEMM matrix multiplication)**, executing billions of comparisons per second using AVX-512 and GPU Tensor Cores!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Floating Point Overflow / Domain Errors in arccos**:
      * *Issue*: Computing `math.acos(np.dot(u, v))` where `u` and `v` are identical unit vectors.
      * *Consequence*: Due to float32 rounding errors, `np.dot(u, v)` might equal `1.0000001`, causing `math.acos()` to raise `ValueError: math domain error`.
      * *Fix*: Always clamp the dot product: `np.clip(cos_val, -1.0, 1.0)` before calling `acos`.

    * **Trap 2: Computing Full Norms in Hot Retrieval Loops**:
      * *Issue*: Recomputing `np.linalg.norm(v)` inside the query search loop for thousands of stored vectors.
      * *Consequence*: Slows down retrieval by 5x-10x.
      * *Fix*: Store pre-normalized vectors in memory and execute pure dot products.

    * **Trap 3: Inverting Distance and Similarity in Vector DB Configurations**:
      * *Issue*: Selecting `metric="cosine"` in a vector database and treating the returned score as a similarity score rather than a distance score.
      * *Consequence*: In many databases (e.g. ChromaDB, FAISS), the "cosine" metric returns **Cosine Distance** (`1.0 - Cosine Similarity`), where 0.0 means identical! Confusing distance with similarity inverts ranking thresholds.
      * *Fix*: Verify whether the API returns Cosine Similarity (`[-1, 1]`) or Cosine Distance (`[0, 2]`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Cosine Similarity**
      * *Definition*: The measure of directional alignment between two vectors, calculated as the dot product divided by the product of their Euclidean norms.
    * **Flashcard 2: Magnitude Invariance**
      * *Definition*: The mathematical property of cosine similarity where scaling a vector by a constant does not alter its similarity score.
    * **Flashcard 3: Orthogonality**
      * *Definition*: When two vectors have an angle of 90 degrees (cosine similarity = 0.0), indicating complete semantic independence.
    * **Flashcard 4: Pre-Normalization Optimization**
      * *Definition*: Enforcing unit length on all vectors during ingestion so cosine similarity reduces to a fast inner dot product.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Cosine Similarity:

- [ ] I can write the plain text mathematical formula for Cosine Similarity.
- [ ] I understand why magnitude invariance makes cosine similarity ideal for variable-length texts.
- [ ] I can interpret cosine similarity score ranges (0.90+ for paraphrases, 0.75+ for relevant passages).
- [ ] I understand why pre-normalizing vectors allows vector databases to use blazing-fast dot products.
- [ ] I know how to avoid floating-point domain errors in `arccos` by clamping values to `[-1.0, 1.0]`.
""")
