import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="7.3 Similarity Measures - RAG Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class MetricComparator:
    """Computes and compares Cosine Similarity, Euclidean Distance, and Dot Product across vectors."""
    def __init__(self):
        pass

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> Dict[str, float]:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        # Dot Product
        dot_prod = float(np.dot(u, v))

        # Cosine Similarity
        if norm_u > 0 and norm_v > 0:
            cosine_sim = float(dot_prod / (norm_u * norm_v))
        else:
            cosine_sim = 0.0

        # Cosine Distance
        cosine_dist = max(0.0, 1.0 - cosine_sim)

        # Euclidean L2 Distance
        l2_dist = float(np.linalg.norm(u - v))

        # Squared L2 Distance
        l2_sq = float(np.sum((u - v) ** 2))

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "cosine_distance": round(cosine_dist, 4),
            "dot_product": round(dot_prod, 4),
            "euclidean_l2_distance": round(l2_dist, 4),
            "squared_l2_distance": round(l2_sq, 4)
        }

# Title & Overview
st.title("7.3 Similarity Measures: Cosine, Euclidean L2 & Dot Product Mechanics")
st.caption("Chapter 7: Embeddings | Domain: 6. RAG")
st.markdown("""
Once text passages are projected into high-dimensional coordinate space, the vector database must compute 
**how close** the user's query vector is to every candidate document vector.

The choice of mathematical **similarity or distance measure** dictates the entire retrieval geometry: 
whether length discrepancies are ignored, how threshold cutoffs behave, and whether distance evaluations 
can be accelerated using CPU SIMD hardware instructions.

In this lesson, you will master:
1. **The 3 Core Mathematical Metrics**: Cosine Similarity, Euclidean (L2) Distance, and Dot Product (Inner Product).
2. **Length Invariance Dynamics**: Why Cosine Similarity neutralizes document length differences.
3. **Hardware Acceleration**: Why L2-normalized vectors allow Dot Product to execute in nanoseconds via AVX-512.
4. **Distance vs Similarity Conversion**: Transforming distance metrics into bounded confidence score thresholds.
5. **Production Failure Traps**: Avoiding unnormalized dot product skews, metric inversion bugs, and arbitrary thresholding.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Similarity & Distance Calculation Subsystem", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph SimilarityMetricsBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_input {
            label="Tier 1: Query & Document Vector Pairs";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            query_vec [label="Query Vector q\\n[0.12, -0.45, ..., 0.88] (1536-d)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            doc_vec [label="Candidate Document Vector d\\n[0.15, -0.42, ..., 0.81] (1536-d)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_metric_engine {
            label="Tier 2: Metric Computation Engine (SIMD AVX-512)";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            cosine_calc [label="1. Cosine Metric Engine\\ncos(theta) = (q . d) / (||q|| * ||d||)\\nAngle orientation; length invariant", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            dot_calc [label="2. Dot Product Engine\\nIP = Sum(q_i * d_i)\\nHardware SIMD; fast when normalized", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            l2_calc [label="3. Euclidean L2 Engine\\nL2 = Sqrt(Sum((q_i - d_i)^2))\\nStraight-line spatial distance", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            query_vec -> cosine_calc;
            doc_vec -> cosine_calc;
            query_vec -> dot_calc;
            doc_vec -> dot_calc;
            query_vec -> l2_calc;
            doc_vec -> l2_calc;
        }

        subgraph cluster_scoring {
            label="Tier 3: Score Calibration & Threshold Filtering";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            calibrator [label="Threshold Filter & Score Normalizer\\nConverts distances to confidence [0.0, 1.0]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            final_ranked [label="Ranked Candidate Heap (Top-K)\\nPasses verified top passages to LLM prompt", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            cosine_calc -> calibrator;
            dot_calc -> calibrator;
            l2_calc -> calibrator;
            calibrator -> final_ranked;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Cosine Similarity
st.header("2. Cosine Similarity: Directional Angular Proximity and Length Invariance", divider="rainbow")
st.markdown("""
**Cosine Similarity** evaluates the cosine of the angle between two high-dimensional vectors, 
completely ignoring their relative magnitudes:

* **Formulation**: `Cosine Similarity = (u . v) / (||u|| * ||v||)`.
* **Value Range**: Bounded strictly between `-1.0` (diametrically opposite direction) and `+1.0` (identical direction). In practice, because text embeddings frequently occupy non-negative or narrow cones, values typically range from `0.0` to `1.0`.
* **Length Invariance Advantage**: A 10-word summary and a 500-word essay discussing the identical concept will produce vectors pointing in virtually the same direction, but the essay's vector might have a larger magnitude. Cosine Similarity cancels out magnitude discrepancies, ensuring short queries match long passages accurately.
""")
st.divider()

# Section 3: Euclidean Distance
st.header("3. Euclidean Distance: Straight-Line Coordinate Space Separation", divider="rainbow")
st.markdown("""
**Euclidean Distance (L2 Distance)** measures the straight-line geometric distance between two points in multidimensional space:

* **Formulation**: `L2 Distance = Sqrt(Sum (u_i - v_i)^2)`.
* **Squared Euclidean Distance**: `Squared L2 = Sum (u_i - v_i)^2`.
* **Sensitivity to Magnitude**: Unlike Cosine Similarity, Euclidean Distance is heavily sensitive to vector magnitude. Two vectors pointing in the exact same direction will register a large Euclidean Distance if one has magnitude 1.0 and the other has magnitude 5.0!
* **Optimal Use Cases**: Computer vision feature embeddings, facial recognition, and spatial geographic coordinates where scale carries fundamental meaning.
""")
st.divider()

# Section 4: Dot Product
st.header("4. Dot Product: Inner Product and L2-Normalized Acceleration", divider="rainbow")
st.markdown("""
The **Dot Product (Inner Product - IP)** computes the sum of element-wise coordinate multiplications:

* **Formulation**: `Dot Product = Sum(u_i * v_i)`.
* **The Equivalence Principle**: When both vectors `u` and `v` are L2-normalized to unit magnitude (`||u|| = 1.0` and `||v|| = 1.0`), the denominator of the Cosine Similarity formula becomes `1.0 * 1.0 = 1.0`. Therefore, **Normalized Dot Product is mathematically identical to Cosine Similarity**:
  `Cosine Similarity = Dot Product (under unit normalization)`.
* **Hardware Performance Impact**: Computing Cosine Similarity requires calculating vector magnitudes and square roots. By pre-normalizing vectors during ingestion, query execution simplifies to a pure multiply-accumulate loop, executing up to **4x faster** via CPU AVX-512 or ARM NEON SIMD instructions.
""")
st.divider()

# Section 5: Similarity Scoring
st.header("5. Similarity Scoring: Threshold Filtering and Score Calibration", divider="rainbow")
st.markdown("""
In production RAG pipelines, developers often configure minimum threshold filters 
(e.g., *"Only pass chunks with similarity >= 0.75 to the LLM"*):

* **Score Calibration Across Models**: Similarity scores are not universal probabilities. A score of 0.75 in Cohere Embed v3 represents very high semantic confidence, whereas a 0.75 in OpenAI `ada-002` represented marginal relevance.
* **The Danger of Rigid Hard Cutoffs**: Setting an arbitrary threshold of 0.80 can cause silent empty responses when user queries use colloquial phrasing.
* **Production Best Practice**: Combine Top-K candidate extraction with dynamic thresholding, or deploy a second-stage Cross-Encoder Reranker that produces calibrated relevance probabilities.
""")
st.divider()

# Section 6: Distance vs Similarity
st.header("6. Distance vs Similarity: Inversion Formulas and Metric Spaces", divider="rainbow")
st.markdown("""
A frequent source of production bugs is confusing **Distance** (where 0.0 is closest) with **Similarity** (where 1.0 is closest):
""")

st.markdown("""
| Metric Name | Metric Category | Perfect Match | Opposite / Distant | Sorting Order |
| :--- | :--- | :---: | :---: | :--- |
| **Cosine Similarity** | Similarity Score | `+1.0` | `-1.0` (or `0.0`) | Descending (`ORDER BY score DESC`) |
| **Cosine Distance** | Distance Metric | `0.0` | `2.0` | Ascending (`ORDER BY distance ASC`) |
| **Euclidean L2 Distance** | Distance Metric | `0.0` | `Infinity` | Ascending (`ORDER BY distance ASC`) |
| **Normalized Dot Product** | Similarity Score | `+1.0` | `-1.0` | Descending (`ORDER BY score DESC`) |
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Different vector databases standardize on different default conventions:
* **ChromaDB**: Defaults to **Cosine Distance** (`hnsw:space: cosine`), where `Distance = 1.0 - Cosine_Similarity`. Top hits have values near 0.0!
* **Qdrant**: Supports explicit metric types (`Cosine`, `Dot`, `Euclid`).
* **Milvus**: Uses metric tags (`L2`, `IP`, `COSINE`).
Understanding your database's default metric space prevents inverted query sorting bugs.
""")
st.divider()

# Section 7: Production Implementation
st.header("7. Production Implementation: Hardware-Accelerated SIMD Distance Engine", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating vectorized distance calculations across all three metrics.
""")

st.code('''
import numpy as np
from typing import Dict, Any

class SIMDDistanceEngine:
    """Computes similarity and distance metrics using vectorized NumPy operations."""
    def __init__(self):
        pass

    def evaluate_pair(self, u: np.ndarray, v: np.ndarray) -> Dict[str, float]:
        # Compute magnitudes
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        # Dot product
        dot_val = float(np.dot(u, v))

        # Cosine similarity & distance
        if norm_u > 0 and norm_v > 0:
            cosine_sim = float(dot_val / (norm_u * norm_v))
        else:
            cosine_sim = 0.0
        cosine_dist = max(0.0, 1.0 - cosine_sim)

        # Euclidean L2 distance & squared L2
        diff = u - v
        l2_dist = float(np.linalg.norm(diff))
        l2_sq = float(np.sum(diff ** 2))

        return {
            "dot_product": round(dot_val, 4),
            "cosine_similarity": round(cosine_sim, 4),
            "cosine_distance": round(cosine_dist, 4),
            "euclidean_distance": round(l2_dist, 4),
            "squared_euclidean": round(l2_sq, 4)
        }
''', language="python")
st.divider()

# Section 8: Interactive Workbench
st.header("8. Interactive Workbench: 3-Metric Geometric Distance & Scoring Laboratory", divider="rainbow")
st.markdown("""
Manipulate 2D coordinate vectors to observe how changing vector directions and magnitudes affects 
Cosine Similarity, Dot Product, and Euclidean Distance simultaneously.
""")

col_u1, col_u2 = st.columns(2)
with col_u1:
    st.subheader("Vector U Coordinates")
    u_x = st.slider("U_x:", min_value=-2.0, max_value=2.0, value=1.0, step=0.1)
    u_y = st.slider("U_y:", min_value=-2.0, max_value=2.0, value=0.5, step=0.1)
    vec_u = np.array([u_x, u_y], dtype=np.float32)

with col_u2:
    st.subheader("Vector V Coordinates")
    v_x = st.slider("V_x:", min_value=-2.0, max_value=2.0, value=0.8, step=0.1)
    v_y = st.slider("V_y:", min_value=-2.0, max_value=2.0, value=0.4, step=0.1)
    vec_v = np.array([v_x, v_y], dtype=np.float32)

comparator = MetricComparator()
res = comparator.evaluate(vec_u, vec_v)

st.subheader("Computed Metric Values")
col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Cosine Similarity (Angle)", f"{res['cosine_similarity']}")
col_res1.caption(f"Cosine Distance: **{res['cosine_distance']}**")

col_res2.metric("Dot Product (Angle + Scale)", f"{res['dot_product']}")
col_res2.caption(f"Norm U: {np.linalg.norm(vec_u):.2f} | Norm V: {np.linalg.norm(vec_v):.2f}")

col_res3.metric("Euclidean L2 Distance (Space)", f"{res['euclidean_l2_distance']}")
col_res3.caption(f"Squared L2: **{res['squared_l2_distance']}**")

if res["cosine_similarity"] > 0.95 and res["euclidean_l2_distance"] > 0.5:
    st.info("💡 **Key Observation**: Vectors point in virtually the same direction (Cosine ~ 1.0), but their different lengths create a noticeable Euclidean distance gap!")
st.divider()

# Section 9: Technical Interview Preparation & Production Traps
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: Under what condition is Dot Product mathematically identical to Cosine Similarity?", expanded=True):
    st.markdown("""
    **Answer:**
    Dot Product is identical to Cosine Similarity when both vectors are **L2-normalized to unit magnitude** (`||u|| = 1.0` and `||v|| = 1.0`). 
    The Cosine Similarity formula divides the dot product by the product of vector norms `||u|| * ||v||`. When both norms equal 1.0, 
    the denominator simplifies to 1.0, leaving `Cosine Similarity = Dot Product`.
    """)

with st.expander("Q2: Why is Cosine Similarity generally preferred over Euclidean Distance for text embeddings?"):
    st.markdown("""
    **Answer:**
    In text retrieval, vector magnitude frequently reflects extraneous attributes (such as token repetition, word count, or sentence length) 
    rather than core semantic meaning. Cosine Similarity normalizes vector length, evaluating strictly the directional angle between 
    concepts. A brief search query and a detailed paragraph on the same topic will have high Cosine Similarity, whereas Euclidean Distance 
    would penalize the magnitude gap.
    """)

with st.expander("Q3: Why do vector databases compute Squared Euclidean Distance instead of Euclidean Distance?"):
    st.markdown("""
    **Answer:**
    Standard Euclidean Distance requires evaluating `sqrt(Sum (u_i - v_i)^2)`. The square root function is computationally expensive 
    across millions of vector comparisons. Because the square root is a strictly monotonic increasing function, relative candidate 
    rankings are completely preserved without computing the square root. Omitting it increases distance calculation throughput by 20–30%.
    """)

with st.expander("Q4: How does ChromaDB define Cosine Distance, and what value indicates a perfect match?"):
    st.markdown("""
    **Answer:**
    ChromaDB defines Cosine Distance as `Distance = 1.0 - Cosine_Similarity`. 
    * A **perfect match** (identical direction) yields a distance of `0.0`.
    * Orthogonal vectors (90-degree angle) yield a distance of `1.0`.
    * Opposite vectors (180-degree angle) yield a distance of `2.0`.
    """)

with st.expander("Q5: What is the risk of using unnormalized vectors with Dot Product indexing?"):
    st.markdown("""
    **Answer:**
    If vectors are not normalized, Dot Product is heavily dominated by vector magnitude rather than semantic angle. 
    A long, verbose document with large coordinate values will achieve a higher dot product than a short, highly relevant document 
    simply due to magnitude inflation, severely degrading retrieval precision.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Inverting Distance Metrics in Application Ranking Logic (Worst Hit Ranked First)**
    Writing application filtering logic that assumes a distance metric (like ChromaDB's Cosine Distance) is a similarity score, sorting results with `ORDER BY distance DESC`. The application returns the most irrelevant, diametrically opposite documents as top results!
    """)
    st.error("""
    **Trap 2: Using Dot Product Search on Unnormalized Vectors (Magnitude Dominance)**
    Configuring a vector database collection to use Inner Product (`ip`) indexing while ingesting unnormalized vectors. Massive, repetitive documents dominate search results regardless of conceptual relevance to the user's query.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Applying Identical Score Thresholds Across Different Models (False Emptiness)**
    Hardcoding a threshold filter (`score >= 0.82`) based on OpenAI embeddings, and then switching to Cohere or BGE. Different embedding models have distinct score distributions; a valid match in BGE might peak at 0.76, resulting in zero returned documents.
    """)
    st.warning("""
    **Trap 4: Forgetting to Re-Normalize Query Vectors in SIMD Pipelines (Scale Drift)**
    Pre-normalizing all corpus vectors during ingestion, but failing to normalize user query vectors at search time. Dot Product calculations become skewed by query length variations.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: When does Dot Product equal Cosine Similarity?")
        st.markdown("**Back**: When both vectors are L2-normalized to unit magnitude (norm = 1.0).")
    with st.container(border=True):
        st.markdown("**Front**: What does a Cosine Distance of 0.0 represent in ChromaDB?")
        st.markdown("**Back**: An exact match (cosine similarity = 1.0, distance = 1 - 1 = 0).")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Cosine Similarity length-invariant?")
        st.markdown("**Back**: It divides the dot product by vector magnitudes, isolating the directional angle from length differences.")
    with st.container(border=True):
        st.markdown("**Front**: Why omit the square root in Euclidean Distance ranking?")
        st.markdown("**Back**: Square root is monotonic and does not alter relative order; omitting it saves substantial CPU cycles.")

st.divider()

# Section 10: Self-Assessment Checklist
st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Similarity and Distance Measures before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the mathematical formulation and behavior of Cosine Similarity, Euclidean Distance, and Dot Product.", value=True, key="chk_meas_1")
    chk2 = st.checkbox("I know why L2 normalization allows Cosine Similarity to be computed as an accelerated Dot Product.", value=True, key="chk_meas_2")
    chk3 = st.checkbox("I understand why Cosine Similarity is length-invariant and well-suited for natural language retrieval.", value=True, key="chk_meas_3")
with c2:
    chk4 = st.checkbox("I know the difference between similarity scores (higher is closer) and distance metrics (lower is closer).", value=True, key="chk_meas_4")
    chk5 = st.checkbox("I understand the risks of uncalibrated hard threshold cutoffs across different embedding models.", value=True, key="chk_meas_5")
    chk6 = st.checkbox("I can identify production traps such as inverted distance rankings and unnormalized dot product skew.", value=True, key="chk_meas_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Similarity Measures Mastery Demonstrated!** You are ready to advance to Lesson 7.4 in the curriculum.")
