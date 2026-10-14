import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.3 Similarity Measures - Embeddings Curriculum",
    page_icon="📏",
    layout="wide"
)

# Title & Overview
st.title("2.3 Similarity Measures: Cosine Similarity, Dot Product, Euclidean, Manhattan, & Metric Selection Guide")
st.caption("Chapter 2: Mathematical Foundations | Domain: 3. Vector Embeddings")
st.markdown("""
Once text is converted into dense vector coordinates, **retrieval and classification boil down to computing geometric distances**.
However, choosing the wrong similarity metric can catastrophically degrade retrieval quality.

There are four primary similarity and distance measures used in machine learning:
1. **Cosine Similarity**: Measures the cosine of the angle between two vectors, completely ignoring differences in vector length. Standard for text embeddings.
2. **Dot Product (Inner Product)**: Multiplies coordinates directly. Dependent on both angular alignment and vector lengths.
3. **Euclidean Distance (L2)**: Measures the straight-line physical distance between coordinates. Highly sensitive to vector length.
4. **Manhattan Distance (L1)**: Measures grid-based coordinate differences. Useful for sparse feature spaces and outlier resistance.

Critically, **when vectors are L2-normalized to unit length (norm = 1.0), Cosine Similarity, Dot Product, and Euclidean Distance become mathematically equivalent in their ranking order**.

This lesson explores the mathematical formulas, geometric trade-offs, and production selection criteria for each metric.
""")

st.divider()

# Section 1: Comparative Mathematical Formulations
st.header("1. Mathematical Metrics & The Unit-Sphere Equivalence", divider="rainbow")

col_sm1, col_sm2 = st.columns([1, 1])

with col_sm1:
    st.markdown("""
    ### Core Metric Formulations:
    * **1. Cosine Similarity**:
      `Cosine_Sim(u, v) = dot(u, v) / (norm(u) * norm(v))`
      * *Range*: [-1.0, +1.0]. Measures pure semantic orientation.
    * **2. Cosine Distance**:
      `Cosine_Dist(u, v) = 1.0 - Cosine_Sim(u, v)` (Range: [0.0, 2.0]).
    * **3. Euclidean Distance (L2)**:
      `L2_Dist(u, v) = sqrt( sum( (u_i - v_i)^2 ) )`
      * *Range*: [0.0, infinity). Sensitive to vector magnitude.
    * **4. Dot Product**:
      `Dot_Product(u, v) = sum( u_i * v_i )`
      * Fastest hardware execution on GPUs and SIMD registers.
    * **5. The Unit-Sphere Equivalence**:
      When `||u|| = ||v|| = 1.0`:
      `L2_Dist^2 = ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * dot(u, v) = 2 - 2 * Cosine_Sim`
      Therefore, **minimizing Euclidean Distance produces the exact same ranking as maximizing Cosine Similarity!**
    """)

with col_sm2:
    st.markdown("### Metric Decision Flowchart")
    metric_dot = """
    digraph MetricFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Data [label="Input Vectors\\n(Embeddings / Features)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        CheckNorm [label="Are vectors L2-normalized\\nto unit length (norm = 1)?", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        DotProd [label="Use Dot Product\\n(Fastest SIMD/GPU speed;\\nEquiv to Cosine)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        LengthMatters [label="Does vector magnitude\\nencode semantic meaning?", fillcolor="#EDE9FE", color="#7C3AED"];
        Cosine [label="Use Cosine Similarity\\n(Invar to doc length)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        Euclidean [label="Use Euclidean Distance (L2)\\n(e.g. Computer Vision / Clustering)", fillcolor="#FEE2E2", color="#DC2626"];

        Data -> CheckNorm;
        CheckNorm -> DotProd [label="Yes (Normalized)"];
        CheckNorm -> LengthMatters [label="No (Unnormalized)"];
        LengthMatters -> Cosine [label="No (Length is noise)"];
        LengthMatters -> Euclidean [label="Yes (Length is signal)"];
    }
    """
    st.graphviz_chart(metric_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Similarity Engine in Python", divider="rainbow")
st.markdown("""
The following Python class computes all five similarity and distance metrics and verifies the unit-sphere mathematical equivalence.
""")

class ProductionSimilarityEngine:
    """
    Computes Cosine Similarity, Cosine Distance, Dot Product,
    Euclidean Distance (L2), and Manhattan Distance (L1).
    """
    @staticmethod
    def compute_all_metrics(u: np.ndarray, v: np.ndarray) -> Dict[str, float]:
        """Calculates all standard distance and similarity metrics between two vectors."""
        norm_u = float(np.linalg.norm(u))
        norm_v = float(np.linalg.norm(v))
        
        dot_product = float(np.dot(u, v))
        
        if norm_u == 0 or norm_v == 0:
            cos_sim = 0.0
        else:
            cos_sim = float(dot_product / (norm_u * norm_v))

        cos_dist = float(1.0 - cos_sim)
        l2_dist = float(np.sqrt(np.sum((u - v) ** 2)))
        l1_dist = float(np.sum(np.abs(u - v)))

        return {
            "cosine_similarity": round(cos_sim, 4),
            "cosine_distance": round(cos_dist, 4),
            "dot_product": round(dot_product, 4),
            "euclidean_l2": round(l2_dist, 4),
            "manhattan_l1": round(l1_dist, 4),
            "norm_u": round(norm_u, 4),
            "norm_v": round(norm_v, 4)
        }

code_snippet = r'''
import numpy as np

class SimilarityMetrics:
    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.linalg.norm(u - v))

    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v))

    @staticmethod
    def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sum(np.abs(u - v)))
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Similarity Metric Comparator Workbench
st.header("3. Interactive Similarity Metric Comparator Workbench", divider="rainbow")
st.markdown("""
Manipulate coordinates and lengths for **Vector A** and **Vector B**. 
Toggle **Unit Normalization** to see how Cosine Similarity, Dot Product, and Euclidean Distance converge to exact ranking equivalence!
""")

engine = ProductionSimilarityEngine()

col_v_in1, col_v_in2 = st.columns([1, 1])

with col_v_in1:
    st.markdown("#### Vector A Configuration")
    a_x = st.slider("A: Dimension 1", min_value=-5.0, max_value=5.0, value=3.0, step=0.5)
    a_y = st.slider("A: Dimension 2", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)
    a_z = st.slider("A: Dimension 3", min_value=-5.0, max_value=5.0, value=2.0, step=0.5)
    a_scale = st.slider("A: Length Multiplier (Scale)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

with col_v_in2:
    st.markdown("#### Vector B Configuration")
    b_x = st.slider("B: Dimension 1", min_value=-5.0, max_value=5.0, value=2.5, step=0.5)
    b_y = st.slider("B: Dimension 2", min_value=-5.0, max_value=5.0, value=1.2, step=0.5)
    b_z = st.slider("B: Dimension 3", min_value=-5.0, max_value=5.0, value=1.8, step=0.5)
    b_scale = st.slider("B: Length Multiplier (Scale)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

apply_normalization = st.toggle("Apply L2 Unit Normalization to Both Vectors (norm = 1.0)", value=False)

raw_vec_a = np.array([a_x, a_y, a_z]) * a_scale
raw_vec_b = np.array([b_x, b_y, b_z]) * b_scale

if apply_normalization:
    eval_vec_a = raw_vec_a / np.linalg.norm(raw_vec_a)
    eval_vec_b = raw_vec_b / np.linalg.norm(raw_vec_b)
else:
    eval_vec_a = raw_vec_a
    eval_vec_b = raw_vec_b

metrics = engine.compute_all_metrics(eval_vec_a, eval_vec_b)

st.markdown("### Metric Calculation Dashboard")

col_res1, col_res2, col_res3, col_res4 = st.columns(4)
col_res1.metric("Cosine Similarity", f"{metrics['cosine_similarity']:.4f}", help="Range: [-1.0, +1.0] (1.0 = identical angle)")
col_res2.metric("Dot Product", f"{metrics['dot_product']:.4f}", help="dot(u, v)")
col_res3.metric("Euclidean Distance (L2)", f"{metrics['euclidean_l2']:.4f}", help="Straight line distance")
col_res4.metric("Cosine Distance", f"{metrics['cosine_distance']:.4f}", help="1.0 - Cosine Similarity")

# Verification Table
st.markdown("### Metric Comparison & Normalization Status")
comp_df = pd.DataFrame([
    {"Metric": "Cosine Similarity", "Value": f"{metrics['cosine_similarity']:.4f}", "Properties": "Length Invariant [-1, +1]", "Ideal Use Case": "Text embeddings, NLP, document retrieval"},
    {"Metric": "Dot Product", "Value": f"{metrics['dot_product']:.4f}", "Properties": "Sensitive to scale; Fastest compute", "Ideal Use Case": "Pre-normalized vector search, matrix BLAS"},
    {"Metric": "Euclidean Distance (L2)", "Value": f"{metrics['euclidean_l2']:.4f}", "Properties": "Strict metric; Sensitive to length", "Ideal Use Case": "Computer vision, image features, physical clustering"},
    {"Metric": "Manhattan Distance (L1)", "Value": f"{metrics['manhattan_l1']:.4f}", "Properties": "Grid-based; Outlier resistant", "Ideal Use Case": "Sparse keyword counts, high-dimensional grids"}
])
st.dataframe(comp_df, use_container_width=True)

if apply_normalization:
    st.success(
        f"Unit Normalization Active! Notice that `dot_product` ({metrics['dot_product']:.4f}) equals "
        f"`cosine_similarity` ({metrics['cosine_similarity']:.4f}), and Euclidean Distance squared equals "
        f"`2 - 2 * cos_sim` = {2 - 2 * metrics['cosine_similarity']:.4f} == {metrics['euclidean_l2']**2:.4f}!"
    )
else:
    st.info("Tip: Toggle 'Apply L2 Unit Normalization' above to see how dot product and cosine similarity become identical.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Metric Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Cosine Similarity the standard metric for text embeddings rather than Euclidean Distance?**
    * **Answer**:
      * In text, two documents discussing the exact same topic may have drastically different lengths: Document A is a 50-word abstract, while Document B is a 5,000-word comprehensive whitepaper.
      * In an unnormalized vector space, Document B will have a significantly larger Euclidean norm because it contains more token activations.
      * **Euclidean Distance** between A and B will be large, falsely classifying them as dissimilar.
      * **Cosine Similarity** evaluates only the angle theta between vectors, completely ignoring length and correctly recognizing that they share the same topical direction.

    **Q2: What is the proof that Euclidean Distance ranking equals Cosine Similarity ranking for unit vectors?**
    * **Answer**:
      * Let u and v be unit vectors such that norm(u) = 1.0 and norm(v) = 1.0.
      * Expanding the squared Euclidean distance:
        `||u - v||^2 = ||u||^2 + ||v||^2 - 2 * (u . v) = 1 + 1 - 2 * (u . v) = 2 - 2 * cos(theta)`.
      * Since 2 is a constant and the relation is strictly monotonically decreasing with respect to cos(theta):
        **Minimizing Euclidean distance is mathematically identical to maximizing Cosine similarity**.

    **Q3: Why do production vector databases prefer Dot Product over Cosine Similarity for indexed search?**
    * **Answer**:
      * Cosine similarity requires computing the norms of both vectors: `dot(u, v) / (norm(u) * norm(v))`, which involves division and square roots (O(D) additional operations).
      * If all vectors are pre-normalized to unit length at ingestion time, vector databases can execute pure **Dot Product** (`dot(u, v)`).
      * Modern CPU and GPU architectures (AVX-512, Intel AMX, NVIDIA Tensor Cores) execute dot products via Fused Multiply-Add (FMA) instructions at maximum silicon throughput.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Using Dot Product on Unnormalized Embeddings for Semantic Search**:
      * *Issue*: Indexing unnormalized vectors and using dot product search.
      * *Consequence*: Long, verbose documents with large norms dominate all search queries, drowning out precise, short answers.
      * *Fix*: Always L2-normalize vectors prior to indexing if using dot product search.

    * **Trap 2: Assuming Cosine Distance is a True Metric**:
      * *Issue*: Assuming Cosine Distance (1.0 - cos(theta)) satisfies the Triangle Inequality.
      * *Consequence*: It does not satisfy the triangle inequality: `d(A, C) > d(A, B) + d(B, C)` can occur, which breaks certain metric tree search indexes (e.g. VP-Trees).
      * *Fix*: Use Angular Distance (arccos(cos(theta)) / pi) if a formal mathematical metric is required.

    * **Trap 3: Mixing Distance Types within the Same Vector Index**:
      * *Issue*: Ingesting vectors normalized for Cosine search, but querying the index with Euclidean distance without updating HNSW configuration.
      * *Consequence*: Inverted nearest-neighbor rankings and degraded retrieval precision.
      * *Fix*: Align index metric configuration (`metric_type: "COSINE"` or `"IP"`) across ingestion and query clients.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Cosine Similarity**
      * *Definition*: The measure of angular alignment between two vectors, defined as dot(u, v) / (||u|| * ||v||), ranging from -1.0 to +1.0.
    * **Flashcard 2: Unit-Sphere Equivalence**
      * *Definition*: The mathematical identity stating that for unit vectors (norm = 1.0), ||u - v||^2 = 2 - 2 * cos(theta), making Euclidean and Cosine rankings identical.
    * **Flashcard 3: Dot Product (Inner Product)**
      * *Definition*: The sum of coordinate products sum(u_i * v_i), representing the fastest hardware-accelerated similarity operation.
    * **Flashcard 4: Euclidean Distance (L2)**
      * *Definition*: The geometric straight-line distance between two coordinates in space: sqrt(sum((u_i - v_i)^2)).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Similarity Measures:

- [ ] I can describe the mathematical formulas and ranges for Cosine, Dot Product, Euclidean, and Manhattan metrics.
- [ ] I understand why Cosine similarity is preferred for text embeddings where document length varies.
- [ ] I can derive the proof that Euclidean distance ranking equals Cosine similarity ranking for unit vectors.
- [ ] I understand why vector databases pre-normalize vectors to leverage high-speed GPU dot products.
- [ ] I have completed Chapter 2: Mathematical Foundations!
""")
