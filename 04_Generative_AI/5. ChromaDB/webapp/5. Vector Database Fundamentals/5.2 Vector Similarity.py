import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="5.2 Vector Similarity - ChromaDB Curriculum",
    page_icon="📐",
    layout="wide"
)

# Title & Overview
st.title("5.2 Vector Similarity: Cosine, Euclidean L2, Inner Product & Distance Calibration")
st.caption("Chapter 5: Vector Database Fundamentals | Domain: 5. ChromaDB")
st.markdown("""
Every search query evaluated inside ChromaDB resolves to a single geometric calculation: 
**Measuring the distance between the query vector and candidate document vectors**.

Selecting the wrong distance metric when initializing a ChromaDB collection (`metadata={"hnsw:space": ...}`) 
will degrade search relevance, scramble result rankings, and cause similarity threshold filters to fail.

Furthermore, software engineers transitioning from search algorithms to vector databases frequently stumble 
over the **Distance vs. Similarity Dilemma**:
* ChromaDB returns **Distances** (where `0.0` represents an exact identical match).
* Business logic, ranking algorithms, and UI dashboards expect **Similarities** (where `1.0` or `100%` represents an exact match).

In this lesson, we dissect the three core distance metrics (Cosine, Euclidean L2, and Inner Product) without LaTeX equations, 
explore mathematical inversion formulas, calibrate similarity thresholds, and test an interactive 
Live Metric Space Comparator and Calibrator.
""")

st.divider()

# Section 1: The Three Core Distance Spaces in ChromaDB
st.header("1. The Three Core Distance Spaces in ChromaDB", divider="rainbow")

col_m_desc, col_m_top = col_m_desc, col_m_top = st.columns([1.1, 0.9])

with col_m_desc:
    st.markdown("""
    ### Metric Spaces Defined
    
    * **1. Cosine Distance (`metadata={"hnsw:space": "cosine"}`)**:
      * *Measurement*: Angular difference between vector trajectories.
      * *Length Sensitivity*: **Completely invariant to vector magnitude**. Whether a passage is 20 words or 500 words, only the directional orientation is evaluated.
      * *Range*: `[0.0, 2.0]` (0.0 = Identical direction, 1.0 = Orthogonal/Unrelated, 2.0 = Diametrically opposite).
      * *Best For*: Natural language text embeddings (OpenAI, MiniLM, BGE, Cohere).
    
    * **2. Squared Euclidean Distance (`metadata={"hnsw:space": "l2"}`)**:
      * *Measurement*: Straight-line Cartesian distance between coordinate endpoints.
      * *Length Sensitivity*: **Highly sensitive to vector magnitude**. Longer vectors appear further away even if pointing in the same direction.
      * *Range*: `[0.0, +infinity)`.
      * *Best For*: Computer vision embeddings, spatial coordinates, unnormalized numerical features.
    
    * **3. Inner Product (`metadata={"hnsw:space": "ip"}`)**:
      * *Measurement*: Dot product projection of one vector onto another.
      * *Key Invariant*: For unit-normalized vectors (length = 1.0), Inner Product distance is mathematically equivalent to Cosine Distance!
      * *Range*: `(-infinity, +infinity)`.
      * *Best For*: Pre-normalized unit vectors where dot products maximize SIMD CPU throughput.
    """)

with col_m_top:
    st.markdown("### Metric Space Decision Guide")
    metric_dot = """
    digraph MetricDecision {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Start [label="Choose Distance Metric\\n(metadata={'hnsw:space': ...})", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QType [label="Are vectors natural\\nlanguage text embeddings?", shape=diamond, fillcolor="#FEF3C7"];

        Cosine [label="hnsw:space = 'cosine'\\n(Default & Industry Standard)\\nAngular Invariance", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        QNorm [label="Are vectors guaranteed\\nto be unit-normalized?", shape=diamond, fillcolor="#EDE9FE"];

        IP [label="hnsw:space = 'ip'\\n(Inner Product)\\nFastest SIMD Dot Product", fillcolor="#A7F3D0", color="#059669"];

        L2 [label="hnsw:space = 'l2'\\n(Squared Euclidean)\\nGeometric Cartesian Distance", fillcolor="#FEE2E2", color="#DC2626"];

        Start -> QType;
        QType -> Cosine [label="Yes (NLP)"];
        QType -> QNorm [label="No (General)"];
        QNorm -> IP [label="Yes (Norm=1.0)"];
        QNorm -> L2 [label="No (Raw Magnitude)"];
    }
    """
    st.graphviz_chart(metric_dot, use_container_width=False)

st.divider()

# Section 2: Distance vs. Similarity: The Inversion Formulas
st.header("2. Distance vs. Similarity: The Inversion Formulas", divider="rainbow")

col_inv1, col_inv2 = col_inv1, col_inv2 = st.columns([1, 1])

with col_inv1:
    st.markdown("""
    ### Why ChromaDB Returns Distances
    * In geometry and graph search, nearest-neighbor algorithms minimize a cost function:
      `find argmin(distance(query, candidate))`.
    * Therefore, ChromaDB's `results['distances']` dictionary always returns **Distances**, where:
      * **Smaller numbers = Better matches**.
      * **Zero = Perfect match**.
    
    ### Converting Distances to Business Similarities
    To display confidence percentages or apply intuitive similarity cutoffs in business applications:
    
    * **For Cosine Space (`hnsw:space = 'cosine'`)**:
      ```python
      # Range: 1.0 (Identical) down to -1.0 (Opposite)
      cosine_similarity = 1.0 - distance

      # As a human-friendly percentage (0% to 100%):
      confidence_pct = max(0.0, (1.0 - distance)) * 100
      ```
    
    * **For Euclidean Space (`hnsw:space = 'l2'`)**:
      ```python
      # Monotonically scales from 1.0 (dist=0) to 0.0 (dist=infinity)
      euclidean_similarity = 1.0 / (1.0 + distance)
      ```
    """)

with col_inv2:
    st.markdown("### Production Distance Threshold Guidelines")
    
    THRESH_GUIDE = [
        {
            "Semantic Tier": "Exact / Near-Exact Match",
            "Cosine Distance": "< 0.15",
            "Cosine Similarity": "> 85%",
            "Typical Scenario": "Direct paraphrase, identical technical entity lookup."
        },
        {
            "Semantic Tier": "Strong Contextual Match",
            "Cosine Distance": "0.15 – 0.35",
            "Cosine Similarity": "65% – 85%",
            "Typical Scenario": "Related technical concept, answering the question well."
        },
        {
            "Semantic Tier": "Weak / Boundary Match",
            "Cosine Distance": "0.35 – 0.55",
            "Cosine Similarity": "45% – 65%",
            "Typical Scenario": "Tangentially related topic; high risk of hallucination."
        },
        {
            "Semantic Tier": "Irrelevant / Noise",
            "Cosine Distance": "> 0.55",
            "Cosine Similarity": "< 45%",
            "Typical Scenario": "Orthogonal concept; should be filtered out before LLM."
        }
    ]
    st.dataframe(pd.DataFrame(THRESH_GUIDE), use_container_width=True, hide_index=True)

st.divider()

# Section 3: Clean Python Implementation of Distance Math
st.header("3. Clean Python Implementation of Distance Math", divider="rainbow")

st.code("""import numpy as np
from typing import Tuple

def compute_all_metrics(vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[float, float, float]:
    \"\"\"Calculates Cosine Distance, Euclidean L2 Distance, and Inner Product.\"\"\"
    # 1. Euclidean L2 Distance
    l2_dist = float(np.sum((vec_a - vec_b) ** 2))
    
    # 2. Normalize to unit length for Cosine & IP
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 2.0, l2_dist, 0.0
        
    u_a = vec_a / norm_a
    u_b = vec_b / norm_b
    
    # 3. Inner Product on unit vectors
    dot_prod = float(np.dot(u_a, u_b))
    
    # 4. Cosine Distance
    cosine_dist = max(0.0, 1.0 - dot_prod)
    
    return cosine_dist, l2_dist, dot_prod

# Example demonstration
a = np.array([0.2, 0.8, -0.4], dtype=np.float32)
b = np.array([0.25, 0.75, -0.38], dtype=np.float32)

cos_d, l2_d, ip_s = compute_all_metrics(a, b)
print(f"Cosine Distance: {cos_d:.4f} (Similarity: {(1.0 - cos_d)*100:.1f}%)")
print(f"Euclidean L2 Distance: {l2_d:.4f}")
print(f"Inner Product: {ip_s:.4f}")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Metric Space Comparator & Calibrator
st.header("4. Interactive Workbench: Live Metric Space Comparator & Calibrator", divider="rainbow")

st.markdown("""
Manipulate two simulated high-dimensional vectors to see how Cosine Distance, Euclidean L2, 
and Inner Product behave simultaneously. Test distance threshold cutoffs to see which documents qualify.
""")

col_ctrl_a, col_ctrl_b = col_ctrl_a, col_ctrl_b = st.columns(2)

with col_ctrl_a:
    st.subheader("1. Vector Point A (Query)")
    a_x = st.slider("Coordinate X (A):", min_value=-1.0, max_value=1.0, value=0.6, step=0.1)
    a_y = st.slider("Coordinate Y (A):", min_value=-1.0, max_value=1.0, value=0.8, step=0.1)
    a_scale = st.slider("Vector Magnitude Scale (A):", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

with col_ctrl_b:
    st.subheader("2. Vector Point B (Candidate Document)")
    b_x = st.slider("Coordinate X (B):", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)
    b_y = st.slider("Coordinate Y (B):", min_value=-1.0, max_value=1.0, value=0.7, step=0.1)
    b_scale = st.slider("Vector Magnitude Scale (B):", min_value=0.5, max_value=3.0, value=2.0, step=0.5)

# Build vectors
vec_1 = np.array([a_x, a_y], dtype=np.float32) * a_scale
vec_2 = np.array([b_x, b_y], dtype=np.float32) * b_scale

# Metric math
l2_sq = float(np.sum((vec_1 - vec_2) ** 2))
norm_1 = np.linalg.norm(vec_1)
norm_2 = np.linalg.norm(vec_2)

if norm_1 > 0 and norm_2 > 0:
    cos_sim = float(np.dot(vec_1, vec_2) / (norm_1 * norm_2))
    cos_dist = max(0.0, 1.0 - cos_sim)
    ip_val = float(np.dot(vec_1, vec_2))
else:
    cos_dist = 2.0
    ip_val = 0.0

st.divider()

col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric(label="Cosine Distance (`hnsw:space='cosine'`)", value=f"{cos_dist:.4f}", delta="Invariant to magnitude scale")
col_res2.metric(label="Euclidean L2 Squared (`hnsw:space='l2'`)", value=f"{l2_sq:.4f}", delta="Distorted by magnitude scale")
col_res3.metric(label="Raw Dot Product (`hnsw:space='ip'`)", value=f"{ip_val:.4f}", delta="Multiplies lengths")

st.markdown("### Production Distance Threshold Evaluation")
thresh_cutoff = st.slider("Filter: Maximum Allowed Cosine Distance (Strictness):", min_value=0.05, max_value=0.80, value=0.35, step=0.05)

if cos_dist <= thresh_cutoff:
    st.success(f"🟢 **DOCUMENT ACCEPTED**: Cosine Distance `{cos_dist:.4f}` is within allowable threshold `{thresh_cutoff}`! Passed to LLM context.")
else:
    st.error(f"🔴 **DOCUMENT REJECTED**: Cosine Distance `{cos_dist:.4f}` exceeds allowable threshold `{thresh_cutoff}`! Pruned to prevent hallucination.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is Cosine Distance preferred over Euclidean Distance for text embeddings?"):
    st.markdown("""
    **Answer**:
    * Text embeddings capture **semantic topic orientation** rather than vector length.
    * A 5-word sentence and a 50-word paragraph discussing the same concept will point in the exact same geometric direction, but the 50-word paragraph may have higher word token activation magnitudes.
    * **Euclidean L2 distance penalizes difference in magnitude**, mistakenly declaring the two passages to be distant.
    * **Cosine distance evaluates only the angular direction**, correctly identifying that both passages discuss the exact same topic regardless of text length.
    """)

with st.expander("Q2: How do you configure a ChromaDB collection to use Inner Product or Euclidean distance?"):
    st.markdown("""
    **Answer**:
    Pass the `"hnsw:space"` key inside the collection's `metadata` dictionary upon creation:
    ```python
    # For Cosine:
    client.create_collection("kb_cos", metadata={"hnsw:space": "cosine"})

    # For Euclidean L2:
    client.create_collection("kb_l2", metadata={"hnsw:space": "l2"})

    # For Inner Product:
    client.create_collection("kb_ip", metadata={"hnsw:space": "ip"})
    ```
    Note: This setting is permanent and cannot be altered after the collection is created.
    """)

with st.expander("Q3: What is the mathematical relationship between Cosine Similarity and Dot Product?"):
    st.markdown("""
    **Answer**:
    * `Cosine Similarity(A, B) = Dot_Product(A, B) / (Magnitude(A) * Magnitude(B))`.
    * When vectors A and B are **unit-normalized** (their Euclidean norm equals 1.0, so `Magnitude = 1.0`):
      `Cosine Similarity(A, B) = Dot_Product(A, B)`.
    * This allows high-performance vector databases to normalize vectors once at ingestion time, and then compute lightning-fast dot products during search queries.
    """)

with st.expander("Q4: Why can't you compare distance values across different distance metrics?"):
    st.markdown("""
    **Answer**:
    * Cosine distance is bounded between `0.0` and `2.0`. A distance of `0.2` indicates strong semantic proximity.
    * Euclidean distance is unbounded (`[0.0, +infinity)`), depending entirely on the dimensionality and magnitude of the vectors. In a 1,536-dimensional space, a Euclidean distance of `0.2` might represent virtually identical vectors, while `2.0` might also be very close!
    * Similarity thresholds must be calibrated independently for the specific metric space being used.
    """)

with st.expander("Q5: What is a Similarity Cutoff Threshold, and why is it essential in RAG?"):
    st.markdown("""
    **Answer**:
    * By default, vector search returns the **Top-K closest documents**, even if all documents in the database are completely irrelevant to the user's query!
    * For example, asking *"What is the HR vacation policy?"* against a database containing only Python code will still return the 5 "closest" Python files (with a terrible distance like `0.85`).
    * A **Similarity Cutoff Threshold** discards any retrieved document whose distance exceeds a strict threshold (e.g., `distance > 0.40`).
    * If all candidates are rejected, the system gracefully responds *"I could not find relevant documentation,"* preventing the LLM from hallucinating an answer.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Filtering for `distance > 0.8` Expecting High Similarity**  
    Forgetting that Chroma returns distance instead of similarity causes filters to reject the best documents and keep the worst ones! Filter for `distance < threshold`.
    """)
    st.warning("""
    **Trap 2: Using Euclidean Distance on Unnormalized Text Embeddings**  
    Using L2 distance on text embeddings causes short sentences and long sentences about the same topic to appear far apart due to magnitude differences.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Omitting Distance Cutoff in Production RAG**  
    Returning raw Top-K results when the user asks an out-of-domain question forces the LLM to synthesize answers from irrelevant noise, inducing hallucinations.
    """)
    st.warning("""
    **Trap 4: Attempting to Change `hnsw:space` on an Existing Collection**  
    Calling `collection.modify(metadata={"hnsw:space": "l2"})` does not rebuild graph edges. Distance metrics are immutable post-creation.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does Cosine distance measure?"):
        st.markdown("""
        **Angular Direction**:  
        Measures angle between vectors; invariant to text length and magnitude. Range [0, 2].
        """)
    with st.expander("🗂️ Flashcard: Why does Chroma return distance?"):
        st.markdown("""
        **Cost Minimization**:  
        ANN graph traversal minimizes distance; 0.0 is perfect match, higher is worse.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to convert Cosine to similarity?"):
        st.markdown("""
        **Similarity = 1.0 - Distance**:  
        Converts Chroma's cosine distance into a 0.0 to 1.0 confidence score.
        """)
    with st.expander("🗂️ Flashcard: What is a Similarity Cutoff?"):
        st.markdown("""
        **Hallucination Shield**:  
        Discards Top-K hits if distance > threshold, preventing irrelevant context reaching LLMs.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the mathematical behavior of Cosine distance, Euclidean L2 distance, and Inner Product.", value=True)
chk2 = st.checkbox("I know why Cosine distance is the standard metric space for natural language text embeddings.", value=True)
chk3 = st.checkbox("I know how to configure `hnsw:space` in collection metadata.", value=True)
chk4 = st.checkbox("I understand why ChromaDB returns distances (0.0 = match) and how to invert them to similarities.", value=True)
chk5 = st.checkbox("I know how to calibrate Similarity Cutoff Thresholds to shield RAG pipelines from hallucinations.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 5.3: Vector Search in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 5.2: Vector Similarity and are ready for Chapter 5.3: Vector Search!")
