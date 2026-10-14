import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="4.1 Understanding Embeddings - ChromaDB Curriculum",
    page_icon="🧭",
    layout="wide"
)

# Title & Overview
st.title("4.1 Understanding Embeddings: Geometric Representation, Dimensionality & Semantic Distance")
st.caption("Chapter 4: Embeddings and Vector Representations | Domain: 5. ChromaDB")
st.markdown("""
At the core of ChromaDB and modern Generative AI is a profound mathematical transformation: 
**Translating human language into geometric space**.

Computers cannot measure the conceptual similarity between the phrases *"database connection pool"* 
and *"socket reuse manager"* using ASCII character codes. Keyword search fails because words with identical meanings 
use completely different letters.

An **Embedding** bridges this gap: it converts unstructured text into a **high-dimensional vector of floating-point numbers** 
where spatial proximity reflects semantic similarity. In this geometric space:
* Phrases with synonymous meanings cluster tightly together.
* Opposing or unrelated concepts are separated by wide geometric distances.
* Vector distances (Cosine, Euclidean L2, Inner Product) enable sub-millisecond Approximate Nearest Neighbor (ANN) search.

In this lesson, we explore how text becomes vectors, compare embedding dimensionalities, dissect distance metrics 
without LaTeX equations, and test an interactive Semantic Vector Geometry Explorer.
""")

st.divider()

# Section 1: What is an Embedding? The Text-to-Vector Pipeline
st.header("1. What is an Embedding? The Text-to-Vector Pipeline", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1.1, 0.9])

with col_pipe1:
    st.markdown("""
    ### The Three Stages of Embedding Generation
    
    * **Stage 1: Subword Tokenization**:
      * The input string is sliced into tokens (e.g. BPE or WordPiece tokens) and mapped to integer IDs:
        `"ChromaDB is fast"` -> `[1045, 8920, 2003, 3435]`.
    
    * **Stage 2: Transformer Encoder Forward Pass**:
      * Tokens pass through stacked self-attention transformer layers (BERT, RoBERTa, MiniLM, or Ada-002).
      * Each token attends to surrounding tokens, capturing contextual nuances (e.g., distinguishing bank of a river vs financial bank).
    
    * **Stage 3: Mean Pooling & L2 Normalization**:
      * The transformer outputs token representations. A pooling layer averages the contextualized token vectors into a single sentence vector.
      * The vector is divided by its Euclidean norm ($||v||_2 = 1.0$), projecting it onto a unit hypersphere where dot product equals cosine similarity.
    """)

with col_pipe2:
    st.markdown("### Text-to-Vector Transformation Pipeline")
    trans_dot = """
    digraph TextToVector {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawText [label="Raw String\\n'Kubernetes StatefulSets'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Tokenizer [label="Subword Tokenizer\\n[24501, 8812, 1045]", fillcolor="#FEF3C7", color="#D97706"];

        Transformer [label="Transformer Encoder\\nMulti-Head Attention", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Pooling [label="Mean Pooling + L2 Normalization\\n(384 / 1536 Floats)", fillcolor="#D1FAE5", color="#059669"];

        UnitVector [label="Unit Hypersphere Vector\\n[0.082, -0.412, 0.991, ...]", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        RawText -> Tokenizer -> Transformer -> Pooling -> UnitVector;
    }
    """
    st.graphviz_chart(trans_dot, use_container_width=False)

st.divider()

# Section 2: Dimensionality Spectrum: Capacity vs Memory Overhead
st.header("2. Dimensionality Spectrum: Capacity vs Memory Overhead", divider="rainbow")

st.markdown("""
Different embedding models produce vectors with vastly different dimensionalities. 
Balancing representational fidelity against memory overhead is a primary architectural trade-off:
""")

DIM_SPECTRUM = [
    {
        "Model Architecture": "all-MiniLM-L6-v2 (Default Chroma)",
        "Vector Dimensions": "384 Dimensions",
        "Memory per 1M Vectors": "~1.5 GB RAM",
        "Inference Speed": "Ultra-Fast (3ms on CPU via ONNX)",
        "Best Application": "Local development, edge devices, high-throughput lightweight search."
    },
    {
        "Model Architecture": "bge-base-en-v1.5 / all-mpnet-base-v2",
        "Vector Dimensions": "768 Dimensions",
        "Memory per 1M Vectors": "~3.1 GB RAM",
        "Inference Speed": "Medium (12ms on CPU / 2ms on GPU)",
        "Best Application": "Enterprise knowledge bases, technical documentation, legal search."
    },
    {
        "Model Architecture": "OpenAI text-embedding-3-small",
        "Vector Dimensions": "1536 Dimensions (or truncated)",
        "Memory per 1M Vectors": "~6.1 GB RAM",
        "Inference Speed": "Cloud API (15–35ms HTTP call)",
        "Best Application": "Multi-lingual enterprise RAG, high-precision general QA."
    },
    {
        "Model Architecture": "OpenAI text-embedding-3-large / Cohere v3",
        "Vector Dimensions": "3072 Dimensions",
        "Memory per 1M Vectors": "~12.3 GB RAM",
        "Inference Speed": "Cloud API (30–60ms HTTP call)",
        "Best Application": "Complex reasoning, scientific literature, financial filings."
    }
]

st.dataframe(pd.DataFrame(DIM_SPECTRUM), use_container_width=True, hide_index=True)

st.divider()

# Section 3: Semantic Distance Metrics: Cosine, L2 & Inner Product
st.header("3. Semantic Distance Metrics: Cosine, L2 & Inner Product", divider="rainbow")

col_dist1, col_dist2 = col_dist1, col_dist2 = st.columns([1, 1])

with col_dist1:
    st.markdown("""
    ### 1. Cosine Distance (`"cosine"`)
    * Measures the **angular difference** between two vector directions, completely ignoring vector magnitude.
    * Range: `[0.0, 2.0]`
      * Distance `0.0`: Identical semantic direction (angle = 0 deg).
      * Distance `1.0`: Orthogonal / unrelated concepts (angle = 90 deg).
      * Distance `2.0`: Diametrically opposite concepts (angle = 180 deg).
    * **Industry Standard**: Ideal for natural language text embeddings.
    """)

with col_dist2:
    st.markdown("""
    ### 2. Squared Euclidean Distance (`"l2"`)
    * Measures the straight-line distance between two points in high-dimensional Cartesian space.
    * Sensitive to vector length/magnitude.
    * Range: `[0.0, +infinity)`
      * Distance `0.0`: Identical coordinates.
    
    ### 3. Inner Product (`"ip"`)
    * Measures the dot product of two vectors.
    * For unit-normalized vectors ($||v||=1$), Inner Product distance equals Cosine Distance!
    """)

st.divider()

# Section 4: Interactive Workbench: 3D Semantic Vector Geometry Explorer
st.header("4. Interactive Workbench: 3D Semantic Vector Geometry Explorer", divider="rainbow")

st.markdown("""
Interact with simulated high-dimensional text vectors projected into 3D geometric coordinates. 
Select concept pairs to calculate real-time **Cosine Distance** and **Angular Separation**.
""")

col_p_sel1, col_p_sel2 = col_p_sel1, col_p_sel2 = st.columns(2)

CONCEPTS = {
    "Auto: 'Electric car battery warranty'": [0.85, 0.45, 0.12],
    "Auto: 'Automobile motor charging speed'": [0.82, 0.48, 0.16],
    "Culinary: 'Italian pasta recipe with garlic'": [-0.65, 0.72, 0.22],
    "Culinary: 'Spaghetti bolognese cooking guide'": [-0.62, 0.75, 0.19],
    "Tech: 'PostgreSQL database connection pooling'": [0.15, -0.78, 0.60],
    "Tech: 'Postgres socket manager buffer size'": [0.18, -0.75, 0.63]
}

with col_p_sel1:
    st.subheader("1. Vector Point A")
    c1_name = st.selectbox("Select Concept A:", list(CONCEPTS.keys()), index=0)
    v1 = np.array(CONCEPTS[c1_name])
    v1_norm = v1 / np.linalg.norm(v1)

with col_p_sel2:
    st.subheader("2. Vector Point B")
    c2_name = st.selectbox("Select Concept B:", list(CONCEPTS.keys()), index=1)
    v2 = np.array(CONCEPTS[c2_name])
    v2_norm = v2 / np.linalg.norm(v2)

# Calculate geometric metrics
cos_sim = float(np.dot(v1_norm, v2_norm))
cos_dist = max(0.0, 1.0 - cos_sim)
angle_deg = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180.0 / np.pi)
l2_dist = float(np.linalg.norm(v1_norm - v2_norm))

st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(label="Cosine Distance", value=f"{cos_dist:.4f}", delta="0 = Identical Direction")
col_m2.metric(label="Angular Separation", value=f"{angle_deg:.1f} deg", delta="Close clustering" if angle_deg < 25 else "Distant concepts")
col_m3.metric(label="Euclidean L2 Distance", value=f"{l2_dist:.4f}", delta="Cartesian Separation")

if cos_dist < 0.05:
    st.success("🟢 **TIGHT SEMANTIC CLUSTER**: Concepts share near-identical semantic orientation! Highly likely to rank in Top-1 search.")
elif cos_dist < 0.35:
    st.info("🟡 **RELATED CONCEPTS**: Concepts belong to adjacent semantic categories.")
else:
    st.error("🔴 **ORTHOGONAL / UNRELATED**: Concepts drift far apart in geometric space. Distance exceeds typical retrieval match thresholds.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why are embedding vectors normalized to unit length (L2 norm = 1.0) before indexing?"):
    st.markdown("""
    **Answer**:
    * When vectors are normalized to unit length, their Euclidean magnitude is exactly 1.0 ($||v|| = 1.0$).
    * Under unit normalization, **Cosine Similarity equals the Dot Product**:
      `cosine_similarity(A, B) = dot_product(A, B)`.
    * This allows vector databases to replace slow trigonometric normalization divisions with **fast SIMD dot products**, multiplying ANN search throughput on AVX-512 and ARM Neon CPUs.
    """)

with st.expander("Q2: What is the 'Curse of Dimensionality' in vector databases, and how does it affect ChromaDB?"):
    st.markdown("""
    **Answer**:
    * As vector dimensionality increases (e.g., from 384d to 3072d), the volume of the geometric space grows exponentially.
    * Points become sparse and distances between the nearest neighbor and the furthest neighbor converge to the same ratio.
    * **Impact on ChromaDB**:
      1. Higher dimensions consume more RAM (3,072 floats = ~12 KB per vector).
      2. Distance calculations require more CPU/GPU FLOPs.
      3. HNSW graph traversals require larger beam widths (`efSearch`) to maintain high recall.
    """)

with st.expander("Q3: What is Matryoshka Representation Learning (MRL), used in OpenAI text-embedding-3?"):
    st.markdown("""
    **Answer**:
    * Traditional embedding models store semantic information diffused equally across all vector dimensions. Truncating the vector destroys its meaning.
    * **Matryoshka Embeddings (MRL)** train the model so that the **most critical semantic features are packed into the earliest dimensions** (e.g. the first 256 or 512 dimensions).
    * This allows engineers to truncate a 1,536-dimensional vector down to 512 dimensions, cutting RAM and storage costs by 66% while retaining 98%+ of retrieval accuracy!
    """)

with st.expander("Q4: Can you compare vectors produced by different embedding models in the same ChromaDB collection?"):
    st.markdown("""
    **Answer**:
    **Absolutely NOT**.
    * Every embedding model learns a unique, arbitrary coordinate space.
    * Dimension 42 in OpenAI `text-embedding-3-small` has zero mathematical relation to Dimension 42 in `all-MiniLM-L6-v2`.
    * Comparing vectors from different models yields meaningless random distances.
    * All vectors inside a ChromaDB collection must originate from the **exact same model and version**.
    """)

with st.expander("Q5: What is Mean Pooling in sentence transformer embeddings?"):
    st.markdown("""
    **Answer**:
    * Transformer models output a separate vector for each token in the sentence (e.g. 50 token vectors).
    * **Mean Pooling** averages all contextualized token vectors across the sentence length (weighted by the attention mask, ignoring padding tokens) into a single unified sentence vector.
    * This single pooled vector represents the holistic semantic meaning of the complete text chunk.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Querying with a Different Model Than Ingestion**  
    Ingesting documents using OpenAI and querying with local MiniLM causes dimensional mismatch errors or garbage search rankings. Models must match!
    """)
    st.warning("""
    **Trap 2: Over-Dimensionalizing Lightweight Knowledge Bases**  
    Using 3,072-dimensional models for small FAQ bots wastes 8x RAM and adds unnecessary cloud API latency without noticeable recall gain. Use 384d–768d.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Confusing Cosine Distance with Cosine Similarity**  
    In ChromaDB, results return **Distances**, where `0.0 = identical`. Filtering for `distance > 0.8` accidentally selects the most irrelevant documents!
    """)
    st.warning("""
    **Trap 4: Forgetting the Max Token Context Window of Embedding Models**  
    Passing a 2,000-word chapter to `all-MiniLM-L6-v2` silently truncates text beyond 256 tokens, dropping the latter 85% of your document from the index.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is an embedding?"):
        st.markdown("""
        **Semantic Vector**:  
        Dense numerical float array where spatial closeness represents conceptual similarity.
        """)
    with st.expander("🗂️ Flashcard: What is Mean Pooling?"):
        st.markdown("""
        **Sentence Aggregation**:  
        Averages contextual token vectors into a single vector representing the entire chunk.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What are Matryoshka embeddings?"):
        st.markdown("""
        **Elastic Truncation**:  
        Packs core semantics into early dimensions, allowing vector truncation to save 66% RAM.
        """)
    with st.expander("🗂️ Flashcard: What does Cosine Distance 0.0 mean?"):
        st.markdown("""
        **Identical Direction**:  
        Zero angle difference; represents maximum possible semantic similarity.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand what embeddings are and how transformer encoders convert text into dense vectors.", value=True)
chk2 = st.checkbox("I understand the dimensionality spectrum (384d to 3072d) and its trade-offs between RAM and accuracy.", value=True)
chk3 = st.checkbox("I understand how Cosine distance, Euclidean L2 distance, and Inner Product are calculated.", value=True)
chk4 = st.checkbox("I know why unit normalization allows dot product to replace cosine similarity for high-speed search.", value=True)
chk5 = st.checkbox("I know that vectors from different embedding models cannot be mixed or compared.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 4.2: Embedding Models in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 4.1: Understanding Embeddings and are ready for Chapter 4.2: Embedding Models!")
