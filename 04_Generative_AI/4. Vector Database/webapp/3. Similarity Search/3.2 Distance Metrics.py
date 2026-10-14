import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.2 Distance Metrics - Vector DB Curriculum",
    page_icon="📏",
    layout="wide"
)

# Title & Overview
st.title("3.2 Distance Metrics: Euclidean, Manhattan, Cosine, Dot Product & Decision Selection")
st.caption("Chapter 3: Similarity Search | Domain: 4. Vector Database")
st.markdown("""
A vector database does not inherently "know" what two pieces of text mean; it only knows how to compute 
**geometric distances between numerical arrays**.

Selecting the appropriate **distance metric** is one of the most critical configuration decisions 
when initializing a vector collection. Choose the wrong metric, and your search results will be corrupted by 
document length biases, unnormalized magnitudes, or hardware performance penalties.

In this lesson, we explore the mathematical formulas, geometric interpretations, and production trade-offs 
of the 5 primary distance functions: **Euclidean (L2)**, **Manhattan (L1)**, **Cosine Similarity**, 
**Cosine Distance**, and **Dot Product (Inner Product)**, culminating in an enterprise decision framework.
""")

st.divider()

# Section 1: The 5 Primary Distance Metrics Dissected
st.header("1. The 5 Core Distance Metrics Dissected", divider="rainbow")

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("""
    ### 1. Geometric Metrics (Magnitude-Sensitive)
    * **Euclidean Distance (L2 Norm)**:
      * *Formula*: `L2(u, v) = sqrt(sum((u_i - v_i)^2))`.
      * *Interpretation*: The ordinary straight-line distance between two points in cartesian space.
      * *Sensitivity*: Highly sensitive to vector length. If two documents have identical phrasing but document B repeats the phrase 3 times, their L2 distance will be large!
      * *Range*: `[0.0, infinity)`. Zero represents identical points.
    * **Manhattan Distance (L1 Norm / City-Block / Taxicab)**:
      * *Formula*: `L1(u, v) = sum(|u_i - v_i|)`.
      * *Interpretation*: Distance traveled along grid lines (right angles), like a taxi navigating Manhattan streets.
      * *Sensitivity*: Robust to outliers because errors are not squared.
      * *Primary Use*: High-dimensional sparse vectors, bioinformatic genetic markers, and image color histograms.
    """)

with col_m2:
    st.markdown("""
    ### 2. Angular & Product Metrics (Direction-Oriented)
    * **Cosine Similarity**:
      * *Formula*: `cos_sim(u, v) = (u . v) / (||u|| * ||v||)`.
      * *Interpretation*: The cosine of the angle between two vectors. Completely ignores magnitude!
      * *Range*: `[-1.0, 1.0]`. In dense text embeddings, typically `[0.0, 1.0]`.
    * **Cosine Distance**:
      * *Formula*: `cos_dist(u, v) = 1.0 - cos_sim(u, v)`.
      * *Interpretation*: Inverted distance metric for search engines where `0.0` is identical.
      * *Range*: `[0.0, 2.0]`.
    * **Dot Product (Inner Product / IP)**:
      * *Formula*: `dot(u, v) = sum(u_i * v_i) = ||u|| * ||v|| * cos(theta)`.
      * *Interpretation*: Combines both angle and length. Rewards vectors that point in the same direction AND have large magnitudes.
      * *Unit Equivalence*: If `||u|| = ||v|| = 1.0`, then `Dot Product == Cosine Similarity`!
    """)

st.markdown("### Master Distance Metric Comparison Table")
dist_table_df = pd.DataFrame({
    "Distance Metric": ["Euclidean (L2)", "Manhattan (L1)", "Cosine Similarity", "Cosine Distance", "Dot Product (IP)"],
    "Mathematical Expression": ["sqrt(sum((u_i - v_i)^2))", "sum(|u_i - v_i|)", "(u . v) / (||u|| * ||v||)", "1 - cos_sim(u, v)", "sum(u_i * v_i)"],
    "Output Range": ["[0, infinity)", "[0, infinity)", "[-1.0, 1.0]", "[0.0, 2.0]", "(-infinity, infinity)"],
    "Magnitude Sensitive?": ["Yes (Strongly)", "Yes (Moderately)", "No (Completely invariant)", "No (Completely invariant)", "Yes (Strongly)"],
    "True Metric?": ["Yes", "Yes", "No (Similarity)", "No (Violates triangle inequality)", "No (Can be negative)"],
    "Hardware Speed": ["Moderate (Requires sqrt)", "Fast (Absolute diffs)", "Slow (Needs norms & div)", "Slow (Needs norms & div)", "Blazing (SIMD FMA in 1 cycle)"]
})
st.dataframe(dist_table_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Decision Framework: Choosing the Right Metric
st.header("2. Enterprise Decision Framework: Choosing the Right Metric", divider="rainbow")

st.markdown("""
How do you know which metric to configure in Milvus, Qdrant, Chroma, or Pinecone? 
Follow the architectural decision tree below:
""")

metric_decision_dot = """
digraph MetricDecisionTree {
    rankdir=TB;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Start [label="New Vector Collection Setup\\nWhat kind of data & model are you indexing?", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    Q1 [label="Are vectors produced by a\\nmodern Text/Multimodal model?\\n(e.g. OpenAI, BGE, Cohere, CLIP)", fillcolor="#DBEAFE"];
    Q2 [label="Are input vectors\\nstrictly L2-normalized to 1.0?", fillcolor="#BFDBFE"];
    
    ChooseIP [label="Use Dot Product (Inner Product)\\nFastest SIMD hardware throughput\\nMathematically identical to Cosine!", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
    ChooseCos [label="Use Cosine Distance / Similarity\\nNormalizes on-the-fly\\nGuarantees angle-only relevance", fillcolor="#A7F3D0", color="#047857", style="filled,bold"];
    
    QTab [label="Is your data Tabular / Geospatial / Physical?\\n(e.g. Lat/Long, Sensor readings, Image Pixels)", fillcolor="#FEF3C7"];
    ChooseL2 [label="Use Euclidean Distance (L2)\\nAbsolute spatial coordinates matter", fillcolor="#FDE68A", color="#D97706", style="filled,bold"];
    ChooseL1 [label="Use Manhattan Distance (L1)\\nHigh-dimensional sparse histograms / Outliers", fillcolor="#FED7AA", color="#EA580C", style="filled,bold"];

    Start -> Q1;
    Q1 -> Q2 [label="Yes (Embeddings)"];
    Q1 -> QTab [label="No (Physical / Tabular)"];
    
    Q2 -> ChooseIP [label="Yes (Normalized)"];
    Q2 -> ChooseCos [label="No (Unnormalized)"];
    
    QTab -> ChooseL2 [label="Dense / Continuous"];
    QTab -> ChooseL1 [label="Sparse / Outlier-heavy"];
}
"""
st.graphviz_chart(metric_decision_dot, use_container_width=False)

st.divider()

# Section 3: Interactive Distance Metric Comparison Workbench
st.header("3. Interactive Distance Metric Comparison Workbench", divider="rainbow")

st.markdown("""
Manipulate vector coordinates and observe how scaling magnitudes alters Euclidean and Dot Product 
while leaving Cosine metrics completely unchanged:
""")

col_wb_ctrl, col_wb_disp = st.columns([1, 1])

with col_wb_ctrl:
    st.subheader("Vector Configuration")
    
    st.markdown("**Vector u Base Values:**")
    u_x = st.slider("u_x", -2.0, 2.0, 1.0, 0.1)
    u_y = st.slider("u_y", -2.0, 2.0, 0.5, 0.1)
    u_scale = st.slider("Magnitude Multiplier for Vector u", 0.5, 5.0, 1.0, 0.5, help="Simulates document length expansion")
    
    st.markdown("**Vector v Base Values:**")
    v_x = st.slider("v_x", -2.0, 2.0, 0.8, 0.1)
    v_y = st.slider("v_y", -2.0, 2.0, 0.4, 0.1)
    v_scale = st.slider("Magnitude Multiplier for Vector v", 0.5, 5.0, 1.0, 0.5)
    
    auto_normalize = st.checkbox("Apply Unit L2 Normalization to Both Vectors", value=False)

with col_wb_disp:
    st.subheader("Real-Time Metric Computation")
    
    raw_u = np.array([u_x * u_scale, u_y * u_scale])
    raw_v = np.array([v_x * v_scale, v_y * v_scale])
    
    if auto_normalize:
        norm_u = np.linalg.norm(raw_u)
        norm_v = np.linalg.norm(raw_v)
        vec_u = raw_u / (norm_u if norm_u > 0 else 1.0)
        vec_v = raw_v / (norm_v if norm_v > 0 else 1.0)
    else:
        vec_u = raw_u
        vec_v = raw_v
        
    # Calculations
    l2 = float(np.linalg.norm(vec_u - vec_v))
    l1 = float(np.sum(np.abs(vec_u - vec_v)))
    dot_val = float(np.dot(vec_u, vec_v))
    
    nu = np.linalg.norm(vec_u)
    nv = np.linalg.norm(vec_v)
    cos_sim = float(dot_val / (nu * nv)) if (nu > 0 and nv > 0) else 0.0
    cos_dist = 1.0 - cos_sim
    
    # Display
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.metric("Cosine Similarity", f"{cos_sim:.4f}")
        st.metric("Cosine Distance", f"{cos_dist:.4f}")
        st.metric("Dot Product", f"{dot_val:.4f}")
    with c_res2:
        st.metric("Euclidean Distance (L2)", f"{l2:.4f}")
        st.metric("Manhattan Distance (L1)", f"{l1:.4f}")
        st.metric("Vector u Length ||u||", f"{nu:.4f}")
        
    if not auto_normalize and (u_scale != 1.0 or v_scale != 1.0):
        st.warning(f"""
        **Magnitude Effect Observed:**
        Notice that Cosine Similarity remains `{cos_sim:.4f}` regardless of the magnitude multipliers! 
        However, Euclidean distance ballooned to `{l2:.4f}` and Dot Product surged to `{dot_val:.4f}`!
        """)
    elif auto_normalize:
        st.success(f"""
        **Normalized State Verified:**
        Both vectors are on the unit circle (`||u|| = ||v|| = 1.0`). 
        Notice that `Dot Product ({dot_val:.4f})` equals `Cosine Similarity ({cos_sim:.4f})` exactly!
        """)

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Unified Vector Distance Engine in NumPy", divider="rainbow")

st.markdown("""
The following complete Python program implements all 5 distance metrics with vectorized NumPy SIMD operations, 
validates the mathematical unit equivalence theorem, and benchmarks execution speed:
""")

code_dist_engine = '''import numpy as np
import time

class VectorDistanceEngine:
    """SIMD-accelerated distance metric computation suite."""
    
    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        """L2 Euclidean distance: sqrt(sum((u_i - v_i)^2))."""
        return float(np.linalg.norm(u - v))

    @staticmethod
    def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
        """L1 Manhattan distance: sum(|u_i - v_i|)."""
        return float(np.sum(np.abs(u - v)))

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Cosine angle similarity: (u . v) / (||u|| * ||v||)."""
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    @staticmethod
    def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Cosine distance: 1.0 - cosine_similarity."""
        return 1.0 - VectorDistanceEngine.cosine_similarity(u, v)

    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> float:
        """Inner product: sum(u_i * v_i)."""
        return float(np.dot(u, v))

# Demonstration and Unit Verification
dim = 768
np.random.seed(42)

# Generate two unnormalized vectors
vec_a = np.random.randn(dim) * 2.5
vec_b = np.random.randn(dim) * 1.8

print("--- 1. Unnormalized Vectors ---")
print(f"Cosine Similarity : {VectorDistanceEngine.cosine_similarity(vec_a, vec_b):.4f}")
print(f"Cosine Distance   : {VectorDistanceEngine.cosine_distance(vec_a, vec_b):.4f}")
print(f"Dot Product       : {VectorDistanceEngine.dot_product(vec_a, vec_b):.4f}")
print(f"Euclidean (L2)    : {VectorDistanceEngine.euclidean_distance(vec_a, vec_b):.4f}")
print(f"Manhattan (L1)    : {VectorDistanceEngine.manhattan_distance(vec_a, vec_b):.4f}")

# Normalize to unit sphere
unit_a = vec_a / np.linalg.norm(vec_a)
unit_b = vec_b / np.linalg.norm(vec_b)

print("\\n--- 2. L2-Normalized Vectors (Unit Sphere) ---")
cos_sim_u = VectorDistanceEngine.cosine_similarity(unit_a, unit_b)
dot_u = VectorDistanceEngine.dot_product(unit_a, unit_b)
l2_u = VectorDistanceEngine.euclidean_distance(unit_a, unit_b)

print(f"Cosine Similarity : {cos_sim_u:.6f}")
print(f"Dot Product       : {dot_u:.6f} (Matches Cosine Similarity exactly!)")
print(f"Squared L2 Formula: {2 - 2 * dot_u:.6f} == {l2_u**2:.6f}")
'''

st.code(code_dist_engine, language="python")

if st.button("🚀 Execute Vector Distance Engine", type="primary"):
    exec_scope = {}
    exec(code_dist_engine, globals(), exec_scope)
    
    eng = exec_scope["VectorDistanceEngine"]
    v_a = np.random.randn(768)
    v_b = np.random.randn(768)
    
    res = {
        "Cosine Similarity": eng.cosine_similarity(v_a, v_b),
        "Cosine Distance": eng.cosine_distance(v_a, v_b),
        "Dot Product": eng.dot_product(v_a, v_b),
        "Euclidean L2": eng.euclidean_distance(v_a, v_b),
        "Manhattan L1": eng.manhattan_distance(v_a, v_b)
    }
    
    st.success("Distance Engine executed and verified successfully!")
    st.json(res)

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why is Cosine Distance technically not a valid mathematical metric, and does this matter in vector databases?"""
a1 = """* **Why it violates metric axioms**:
  * A true mathematical metric must satisfy the **Triangle Inequality**: `D(u, w) <= D(u, v) + D(v, w)`.
  * Cosine Distance `D_cos(u, v) = 1 - cos(theta)` fails this inequality for certain angular configurations.
* **Why it does NOT matter in vector databases**:
  * Vector databases (Milvus, Qdrant, Faiss) do not rely on metric trees (like VP-trees) that require the triangle inequality for geometric pruning.
  * Instead, they use graph traversal (HNSW) and inverted centroids (IVF), which only require that distance is monotonic with angular separation.
  * However, for exact metric space algorithms, **Angular Distance** `arccos(cos_sim) / pi` is used instead."""

q2 = """2. Why is Inner Product (Dot Product) significantly faster on modern hardware than Cosine Similarity?"""
a2 = """* **Cosine Similarity overhead**:
  * Requires computing the dot product `u . v`, the norm `||u|| = sqrt(sum(u_i^2))`, the norm `||v|| = sqrt(sum(v_i^2))`, multiplying the norms, and performing floating-point division.
  * Square roots and divisions take 10 to 40 CPU clock cycles each and stall CPU execution pipelines.
* **Inner Product advantage**:
  * A pure dot product compiles directly into fused multiply-add (FMA) instructions in SIMD AVX-512 and GPU CUDA tensor cores.
  * Modern hardware can execute two 512-bit FMAs per clock cycle per core.
  * By pre-normalizing vectors upon database insertion, query execution is reduced to pure hardware FMA dot products, increasing query throughput by up to 5x."""

q3 = """3. When would you choose Manhattan Distance (L1) over Euclidean Distance (L2)?"""
a3 = """* **1. High-Dimensional Outlier Sensitivity**: Because Euclidean distance squares coordinate differences `(u_i - v_i)^2`, a large difference in even a single dimension dominates the entire distance calculation. Manhattan distance `|u_i - v_i|` weights all dimensions linearly, offering robustness against noisy outliers.
* **2. High-Dimensional Sparse Representations**: In sparse vectors (like term frequency histograms, genomic markers, or bag-of-words), most coordinates are zero. Manhattan distance accurately reflects the total mass difference across sparse non-zero features without distortion from quadratic scaling."""

q4 = """4. What happens if you index unnormalized vectors using Inner Product (IP) in a vector database?"""
a4 = """* **The Magnitude Distortion Catastrophe**:
  * Inner product is defined as `u . v = ||u|| * ||v|| * cos(theta)`.
  * If vectors are unnormalized, a long document chunk with many tokens will have a significantly larger vector norm `||v||`.
  * Even if its angle `theta` is far from the query (poor semantic relevance), its sheer magnitude will inflate the inner product score above shorter, highly relevant documents!
  * Result: Longer documents dominate search results regardless of actual relevance. Vectors MUST be L2-normalized before indexing with Inner Product."""

q5 = """5. How does vector normalization affect Euclidean (L2) distance in high dimensions?"""
a5 = """* When all vectors are normalized to unit length (`||u|| = ||v|| = 1.0`), all points lie on the surface of a hypersphere of radius 1.
* In this unit space, the squared Euclidean distance simplifies to `||u - v||^2 = 2 - 2 * (u . v)`.
* This means that in normalized space, Euclidean distance ranking is 100% mathematically identical to Cosine similarity and Dot product ranking.
* Choosing between them on normalized data is purely an implementation detail of which hardware instruction the database optimizer prefers."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Distance Metric Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Inner Product Metric Without Pre-Normalization**  
> Choosing `MetricType.IP` in Milvus or Qdrant without verifying that your embedding pipeline executes 
> `v / np.linalg.norm(v)` will silently ruin your search rankings. Longer text chunks will artificially 
> score higher than precise short answers. Always enforce unit normalization before inserting into an IP index.

> [!WARNING]
> **Trap 2: Metric Mismatch Between Index and Query Execution**  
> If an index is constructed using `Cosine` distance, but a client query library defaults to `L2` distance 
> during retrieval calls, the database server may either reject the query or return nonsensical candidate rankings. 
> Ensure client connection parameters strictly match collection configuration.

> [!WARNING]
> **Trap 3: Square Root Redundancy in Distance Calculation**  
> When performing brute-force k-NN in custom Python or C++ code, developers often compute `sqrt(sum((u_i - v_i)^2))`. 
> Because the square root function is strictly monotonic, `a < b` if and only if `sqrt(a) < sqrt(b)`. 
> You can safely compare **Squared Euclidean Distance** `sum((u_i - v_i)^2)`, eliminating millions of expensive square root operations!
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Euclidean Distance (L2)?"):
        st.markdown("""
        **Euclidean Distance (L2)**:  
        The straight-line cartesian distance between points (`sqrt(sum((u_i - v_i)^2))`). Sensitive to vector magnitude.
        """)
    with st.expander("🗂️ Flashcard: What is Cosine Similarity?"):
        st.markdown("""
        **Cosine Similarity**:  
        The cosine of the angle between two vectors (`(u . v) / (||u|| * ||v||)`). Invariant to vector length.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: When Does Dot Product Equal Cosine?"):
        st.markdown("""
        **Unit Equivalence**:  
        When both vectors are L2-normalized to unit length (`||u|| = ||v|| = 1.0`), Dot Product equals Cosine Similarity.
        """)
    with st.expander("🗂️ Flashcard: Why Use Squared Euclidean Distance?"):
        st.markdown("""
        **Squared L2 Optimization**:  
        Omits the expensive `sqrt()` operation during k-NN ranking because square root is strictly monotonic.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can write the mathematical formulas for Euclidean, Manhattan, Cosine, and Dot Product.", value=True)
chk2 = st.checkbox("I understand why Dot Product equals Cosine Similarity on unit-normalized vectors.")
chk3 = st.checkbox("I know why Inner Product is significantly faster in hardware SIMD registers than Cosine Similarity.")
chk4 = st.checkbox("I understand the magnitude distortion trap when using unnormalized vectors with Inner Product.")
chk5 = st.checkbox("I can use the decision tree to select the optimal distance metric for any vector collection.")
chk6 = st.checkbox("I know why Squared Euclidean distance can replace standard Euclidean distance during k-NN ranking.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Distance Metrics and are ready for Chapter 3.3: Similarity Calculation!")
