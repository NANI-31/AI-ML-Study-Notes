import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.1 Vector Basics - Vector DB Curriculum",
    page_icon="📐",
    layout="wide"
)

# Title & Overview
st.title("2.1 Vector Basics: Scalars, Dimensionality & Multimodal Vector Representations")
st.caption("Chapter 2: Understanding Vectors and Embeddings | Domain: 4. Vector Database")
st.markdown("""
At the heart of modern artificial intelligence and vector databases lies a single foundational primitive: **the vector**.
Every piece of unstructured human communication—whether an 80-page financial contract, a high-resolution satellite image, 
a recorded customer call, or a repository of Python code—must be translated into a mathematical vector before an algorithm 
can search, compare, cluster, or reason about it.

In this lesson, we establish the mathematical foundations of vectors, compare scalars against multi-dimensional coordinates, 
explore the surprising geometric properties of high-dimensional spaces (d = 768 to 3072), and examine how 
text, images, and audio are mapped into unified vector spaces.
""")

st.divider()

# Section 1: Scalar vs. Vector & Vector Dimensionality
st.header("1. Scalar vs. Vector & Dimensionality Foundations", divider="rainbow")

col_sc1, col_sc2 = st.columns([1, 1])

with col_sc1:
    st.markdown("""
    ### 1. Mathematical Definitions
    * **Scalar**:
      * A single numerical value representing magnitude or quantity without spatial orientation.
      * *Examples*: Temperature (72.5), file size (4.2 MB), token count (512), similarity score (0.89).
      * Belongs to 1-dimensional real space: `s in R`.
    * **Vector**:
      * An ordered sequence of n numbers (components or coordinates) that defines both magnitude and direction in an n-dimensional geometric space.
      * *Example*: `v = [x_1, x_2, x_3, ..., x_d] in R^d`.
      * In computer systems, a vector is stored as a contiguous 1D array of floating-point numbers (`float32` or `float16`).
    * **Vector Dimensionality (d)**:
      * The number of elements (axes) comprising the vector.
      * *1D Vector*: `[42.0]` (degenerate vector, essentially a scalar).
      * *2D Vector*: `[x, y]` (cartesian plane coordinates).
      * *3D Vector*: `[x, y, z]` (physical spatial coordinates or RGB color channels).
      * *High-Dimensional Vector*: `[v_1, v_2, ..., v_1536]` (AI semantic embedding vector).
    """)

with col_sc2:
    st.markdown("### The Dimensionality Spectrum in Data Engineering")
    dim_df = pd.DataFrame({
        "Type": ["Scalar", "2D Vector", "3D Vector", "Low-Dim Embedding", "Modern LLM Embedding", "Large Multimodal"],
        "Dimensions (d)": ["1", "2", "3", "64 - 128", "768 - 1536", "3072 - 4096"],
        "Memory (float32)": ["4 Bytes", "8 Bytes", "12 Bytes", "256 - 512 Bytes", "3.0 - 6.14 KB", "12.2 - 16.3 KB"],
        "Primary Use Case": [
            "Metadata filtering flags, timestamps",
            "GIS Latitude/Longitude coordinates",
            "3D CAD models, spatial coordinates",
            "Word2Vec, FastText, collaborative filtering",
            "BERT, BGE, OpenAI text-embedding-3-small",
            "text-embedding-3-large, Llama 3, Gemini Vision"
        ]
    })
    st.dataframe(dim_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **Key Takeaway**: A 1536-dimensional embedding vector is simply a point on the surface of a 1536-dimensional sphere. 
    Each dimension captures an abstract latent semantic feature learned during deep neural network training.
    """)

st.divider()

# Section 2: Multimodal Representation Vectors (Text, Image, Audio)
st.header("2. Multimodal Representation: Unifying Text, Images & Audio", divider="rainbow")

st.markdown("""
How can a vector database search for an audio recording using a text description, or find product photos 
matching an audio voice query? The secret is **Multimodal Joint Vector Spaces**.
""")

col_repr1, col_repr2 = st.columns([1, 1])

with col_repr1:
    st.markdown("""
    ### How Raw Modalities Become Vectors
    * **Text Representation**:
      * *Raw Form*: Strings of characters/words.
      * *Vectorization*: Tokenizer breaks string into token IDs -> Transformer layers apply self-attention -> 
        Mean pooling aggregates hidden states into a single d-dimensional vector (e.g., 768-d).
    * **Image Representation**:
      * *Raw Form*: 2D grid of pixels with 3 color channels (Height x Width x 3 RGB ints).
      * *Vectorization*: Vision Transformer (ViT) slices image into 16x16 pixel patches -> Linear projection -> 
        Self-attention layers -> CLS token output yields a d-dimensional vector (e.g., 512-d or 1024-d).
    * **Audio Representation**:
      * *Raw Form*: Continuous 1D pressure wave sampled at 16kHz or 44.1kHz.
      * *Vectorization*: Short-Time Fourier Transform (STFT) produces a 2D Mel-Spectrogram -> 
        Audio Spectrogram Transformer (AST) / Whisper encoder extracts temporal-frequency features into a d-dimensional vector.
    """)

with col_repr2:
    st.markdown("### Multimodal Joint Embedding Space Architecture")
    multi_dot = """
    digraph MultimodalVectors {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        TextIn [label="Text Snippet\\n'Golden retriever in park'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        ImageIn [label="Image File\\nphoto_dog_running.jpg", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        AudioIn [label="Audio Waveform\\nbarking_audio.wav", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        TextEnc [label="Text Transformer\\n(CLIP Text Encoder)", fillcolor="#DBEAFE"];
        VisionEnc [label="Vision Transformer\\n(CLIP ViT-B/32)", fillcolor="#D1FAE5"];
        AudioEnc [label="Audio Encoder\\n(AudioCLIP / CLAP)", fillcolor="#FDE68A"];

        subgraph cluster_space {
            label="Joint Shared Semantic Space (R^512)"; style="dashed"; color="#7C3AED";
            v_text [label="v_text = [0.42, -0.18, ..., 0.77]", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
            v_img [label="v_img  = [0.41, -0.19, ..., 0.75]", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
            v_aud [label="v_aud  = [0.39, -0.16, ..., 0.72]", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
            VDB [label="Vector Database Storage & Index\\n(Cosine Sim > 0.88 across all three!)", fillcolor="#DDD6FE", color="#6D28D9", style="filled,bold"];
        }

        TextIn -> TextEnc -> v_text -> VDB;
        ImageIn -> VisionEnc -> v_img -> VDB;
        AudioIn -> AudioEnc -> v_aud -> VDB;
    }
    """
    st.graphviz_chart(multi_dot, use_container_width=False)

st.divider()

# Section 3: Geometry of High-Dimensional Spaces & Curse of Dimensionality
st.header("3. High-Dimensional Geometry & The Curse of Dimensionality", divider="rainbow")

st.markdown("""
Human intuition is anchored in 2D and 3D space. When moving to **768, 1536, or 3072 dimensions**, 
geometry behaves in counter-intuitive ways that dictate how vector databases must be designed:
""")

col_geom1, col_geom2, col_geom3 = st.columns(3)

with col_geom1:
    st.markdown("""
    #### 1. The Empty Space Phenomenon
    * In d-dimensional space, the volume of a hypercube of width 1 is `1^d = 1`.
    * The volume of an inscribed hypersphere of radius 0.5 shrinks exponentially: as `d -> infinity`, sphere volume approaches **0**!
    * **Consequence**: Almost the entire volume of high-dimensional space is located in the **thin crust (corners)** of the bounding cube.
    """)

with col_geom2:
    st.markdown("""
    #### 2. Orthogonality of Random Vectors
    * In 2D, two randomly chosen vectors have an average angle of 90 degrees, but with high variance.
    * In 1536 dimensions, **ANY two random vectors are almost guaranteed to be exactly orthogonal (perpendicular)**!
    * Their cosine similarity is concentrated tightly around 0.000.
    * **Consequence**: Any non-zero cosine similarity (e.g., 0.35) represents meaningful signal rather than random noise.
    """)

with col_geom3:
    st.markdown("""
    #### 3. Distance Concentration
    * As dimensions grow, the ratio between the distance to the farthest point and the distance to the nearest point approaches 1:
      `lim (dist_max - dist_min) / dist_min = 0`.
    * **Consequence**: Traditional B-trees and KD-trees degrade into full O(N) table scans.
    * Vector databases must use specialized graph indexing (HNSW) and cosine angle metrics rather than raw cartesian partitioning.
    """)

st.divider()

# Section 4: Interactive Multimodal Vector Workbench
st.header("4. Interactive Multimodal Vector Workbench", divider="rainbow")

st.markdown("""
Explore how vectors from different modalities interact in high-dimensional space. 
Adjust vector components, inspect distance metrics, and observe how dimensionality alters geometric distance distributions:
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("Synthetic Multimodal Vector Generator")
    wb_dim = st.selectbox(
        "Dimensionality (d)",
        options=[8, 64, 256, 768, 1536],
        index=3,
        help="Higher dimensions demonstrate distance concentration and near-orthogonality"
    )
    
    np.random.seed(42)
    # Generate anchor vector (representing concept: 'Dog in the park')
    anchor_vec = np.random.randn(wb_dim)
    anchor_vec = anchor_vec / np.linalg.norm(anchor_vec)
    
    # Generate 3 candidate vectors with controlled noise
    noise_level_sem = st.slider("Semantic Variation Noise (Text: 'Puppy playing outdoors')", 0.05, 0.90, 0.15, 0.05)
    noise_level_img = st.slider("Visual Variation Noise (Image: dog_running.jpg)", 0.05, 0.90, 0.25, 0.05)
    
    # Synthetic related vectors
    sem_vec = anchor_vec + np.random.randn(wb_dim) * noise_level_sem
    sem_vec = sem_vec / np.linalg.norm(sem_vec)
    
    img_vec = anchor_vec + np.random.randn(wb_dim) * noise_level_img
    img_vec = img_vec / np.linalg.norm(img_vec)
    
    # Completely unrelated random vector
    unrelated_vec = np.random.randn(wb_dim)
    unrelated_vec = unrelated_vec / np.linalg.norm(unrelated_vec)

with col_wb2:
    st.subheader("Geometric Proximity Analysis")
    
    # Compute metrics
    cos_sem = float(np.dot(anchor_vec, sem_vec))
    cos_img = float(np.dot(anchor_vec, img_vec))
    cos_unrelated = float(np.dot(anchor_vec, unrelated_vec))
    
    l2_sem = float(np.linalg.norm(anchor_vec - sem_vec))
    l2_img = float(np.linalg.norm(anchor_vec - img_vec))
    l2_unrelated = float(np.linalg.norm(anchor_vec - unrelated_vec))
    
    results_df = pd.DataFrame({
        "Comparison Target": [
            "Query Anchor: 'Dog in the park'",
            "Target A: 'Puppy playing outdoors' (Text)",
            "Target B: photo_dog_running.jpg (Image)",
            "Target C: 'Quantum thermodynamics paper' (Unrelated)"
        ],
        "Modality": ["Text", "Text", "Image", "Text"],
        "Cosine Similarity": [1.0000, cos_sem, cos_img, cos_unrelated],
        "Cosine Distance (1 - sim)": [0.0000, 1.0 - cos_sem, 1.0 - cos_img, 1.0 - cos_unrelated],
        "Euclidean Distance (L2)": [0.0000, l2_sem, l2_img, l2_unrelated],
        "Retrieval Verdict": ["Self Anchor", "Strong Match", "Valid Multimodal Match", "Excluded by Threshold"]
    })
    
    st.dataframe(
        results_df.style.format({
            "Cosine Similarity": "{:.4f}",
            "Cosine Distance (1 - sim)": "{:.4f}",
            "Euclidean Distance (L2)": "{:.4f}"
        }).background_gradient(subset=["Cosine Similarity"], cmap="YlGn"),
        use_container_width=True,
        hide_index=True
    )
    
    st.metric(
        "Random Noise Dot Product (Orthogonality Proof)",
        f"{cos_unrelated:.4f}",
        delta=f"Expected theoretical mean: 0.000 in {wb_dim}-d"
    )

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Multimodal Vector Representation & Normalization", divider="rainbow")

st.markdown("""
The following self-contained Python script demonstrates how raw features from different modalities 
are transformed, normalized to unit length, and compared using vector arithmetic:
""")

code_impl = '''import numpy as np

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Projects a vector onto the unit hypersphere (L2 norm = 1.0)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_metrics(v1: np.ndarray, v2: np.ndarray) -> dict:
    """Calculates Cosine Similarity, Cosine Distance, and Euclidean Distance."""
    # Ensure vectors are 1D arrays
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    # Cosine similarity between normalized vectors is simply the dot product
    cos_sim = float(np.dot(v1_norm, v2_norm))
    cos_dist = 1.0 - cos_sim
    l2_dist = float(np.linalg.norm(v1_norm - v2_norm))
    
    return {
        "cosine_similarity": cos_sim,
        "cosine_distance": cos_dist,
        "euclidean_distance": l2_dist
    }

# Demonstration with synthetic 768-dimensional multimodal embeddings
dimension = 768
np.random.seed(101)

# Simulating multimodal embeddings (CLIP-like shared latent space)
text_query = np.random.randn(dimension)
image_candidate = text_query + np.random.randn(dimension) * 0.20  # Semantically close
audio_candidate = text_query + np.random.randn(dimension) * 0.35  # Moderately close
unrelated_candidate = np.random.randn(dimension)                  # Semantic noise

print("1. Text Query vs Image Candidate:")
print(compute_metrics(text_query, image_candidate))

print("\\n2. Text Query vs Audio Candidate:")
print(compute_metrics(text_query, audio_candidate))

print("\\n3. Text Query vs Unrelated Candidate:")
print(compute_metrics(text_query, unrelated_candidate))
'''

st.code(code_impl, language="python")

if st.button("▶️ Execute Multimodal Vector Pipeline in Sandbox", type="primary"):
    exec_scope = {}
    exec(code_impl, globals(), exec_scope)
    compute_metrics_fn = exec_scope["compute_metrics"]
    
    t_q = np.random.randn(768)
    i_c = t_q + np.random.randn(768) * 0.20
    u_c = np.random.randn(768)
    
    res_img = compute_metrics_fn(t_q, i_c)
    res_unrel = compute_metrics_fn(t_q, u_c)
    
    st.success("Pipeline executed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("**Image Candidate Match Results**:")
        st.json(res_img)
    with col_out2:
        st.markdown("**Unrelated Noise Results**:")
        st.json(res_unrel)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the geometric difference between a scalar, a 1D vector, and a high-dimensional vector in vector databases?"""
a1 = """* **Scalar**: A magnitude with zero spatial direction (e.g. `price = 29.99` or `doc_length = 400`). Handled by traditional relational B-Trees or column stores.
* **1D Vector**: A coordinate with one degree of freedom `[x]`. Mathematically equivalent to a scalar, allowing linear sorting (`x_1 < x_2`).
* **High-Dimensional Vector**: An ordered coordinate tuple in d-dimensional space (`d >= 128`, commonly 768 to 3072). It cannot be sorted along a single 1D axis. Proximity requires geometric metrics (cosine angle, Euclidean distance) and specialized sub-linear graph indexing (HNSW)."""

q2 = """2. Why are two randomly generated vectors in 1536-dimensional space almost exactly orthogonal?"""
a2 = """* By the central limit theorem, the dot product of two independent, standardized random vectors in d dimensions is the sum of d independent zero-mean variables.
* The expected dot product is 0.0, and the standard deviation scales inversely with the square root of d (`1 / sqrt(d)`).
* In 1536 dimensions, `1 / sqrt(1536) approx 0.0255`. Thus, over 99.7% of random vector pairs have a cosine similarity within `[-0.076, +0.076]`.
* This property enables vector databases to distinguish genuine semantic alignment from random noise with high statistical confidence."""

q3 = """3. What is the Curse of Dimensionality, and why does it invalidate KD-Trees for vector databases?"""
a3 = """* In high dimensions, space expands exponentially. The ratio of hypersphere volume to hypercube volume rapidly approaches zero, concentrating all data points near the outer boundaries.
* Distances between the nearest neighbor and the farthest point become nearly identical (`(dist_max - dist_min) / dist_min -> 0`).
* Spatial partitioning trees like KD-Trees split space along hyperplanes. In high dimensions (typically `d > 20`), almost every tree branch must be visited to guarantee finding the nearest neighbor, causing search complexity to degenerate from `O(log N)` to `O(N)` full table scan.
* Vector databases solve this using graph-based small-world networks (HNSW) and inverted cluster centroids (IVF) rather than spatial grid partitioning."""

q4 = """4. How does CLIP map image pixel matrices and text character sequences into the exact same vector space?"""
a4 = """* CLIP (Contrastive Language-Image Pre-training) employs a dual-encoder architecture: a Vision Transformer (ViT) for images and a Transformer for text.
* Both encoders project their respective representations into a shared latent embedding space of identical dimensionality (e.g. 512 dimensions).
* The models are trained simultaneously on hundreds of millions of (image, caption) pairs using InfoNCE contrastive loss:
  * For matching pairs, the cosine similarity of their vectors is maximized toward 1.0.
  * For non-matching pairs, cosine similarity is minimized toward 0.0.
* As a result, the embedding of the text 'golden retriever' lies geometrically close to the embedding of a picture of a golden retriever."""

q5 = """5. Why is vector L2-normalization standard practice before inserting into vector databases?"""
a5 = """* For any two L2-normalized vectors `u` and `v` (where `||u|| = ||v|| = 1.0`), their squared Euclidean distance is directly related to their Cosine Similarity:
  `||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u . v) = 1 + 1 - 2*(u . v) = 2 - 2 * cos_sim(u, v)`.
* Furthermore, Cosine Similarity simplifies to a pure dot product: `cos_sim(u, v) = u . v`.
* In hardware, computing a pure dot product (`u . v`) is dramatically faster than computing square roots and vector norms on every query.
* By pre-normalizing vectors upon insertion, vector databases replace expensive cosine distance formulas with ultra-fast SIMD dot products."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Vector Representation Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Dimensionality Mismatch Production Crash**  
> Vector databases enforce strict schema dimensionality on collection creation. 
> If a collection is created with `dimension=768` (e.g., using `bge-base-en-v1.5`), and an ingestion job mistakenly 
> sends 1536-dimensional embeddings from `text-embedding-3-small`, the entire batch write will fail immediately. 
> Always store the embedding model ID in your collection metadata and validate vector shape before ingestion.

> [!WARNING]
> **Trap 2: Normalization Omission on Inner Product Indexes**  
> If an index is configured for `Dot Product` (Inner Product) to maximize search throughput, but input vectors 
> are unnormalized, longer documents with larger vector magnitudes will dominate search rankings regardless 
> of actual semantic relevance. Always L2-normalize vectors prior to indexing with Dot Product.

> [!WARNING]
> **Trap 3: Mixing Encoders from Different Model Families**  
> An embedding vector from OpenAI `text-embedding-3-small` (1536-d) has zero semantic relationship with a 1536-d vector 
> from Cohere or Amazon Titan. Their coordinate spaces are completely distinct. Never query a collection using an encoder 
> different from the one used during ingestion.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Vector?"):
        st.markdown("""
        **Vector**:  
        An ordered array of floating-point values representing a point in d-dimensional space, capturing latent semantic meaning across text, vision, or audio.
        """)
    with st.expander("🗂️ Flashcard: What is Orthogonality in High Dimensions?"):
        st.markdown("""
        **High-Dimensional Orthogonality**:  
        The geometric property where randomly chosen vectors in high dimensions (e.g. d = 1536) have a dot product near zero, allowing real semantic alignments to stand out clearly.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why do KD-Trees Fail for Vector DBs?"):
        st.markdown("""
        **Curse of Dimensionality on KD-Trees**:  
        As dimensions increase, distance concentration forces KD-trees to visit almost every leaf node, degrading search time from O(log N) to O(N) full scans.
        """)
    with st.expander("🗂️ Flashcard: Why Pre-Normalize Vectors?"):
        st.markdown("""
        **L2 Normalization Advantage**:  
        Normalizing vectors to unit length transforms Cosine Similarity into a simple dot product, eliminating expensive square root and norm operations during query execution.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can clearly distinguish between scalars, 1D vectors, and high-dimensional vectors.", value=True)
chk2 = st.checkbox("I understand the memory footprint of different embedding dimensions (384, 768, 1536, 3072).")
chk3 = st.checkbox("I understand how text, images, and audio are projected into shared joint vector spaces.")
chk4 = st.checkbox("I can explain why two random high-dimensional vectors have a cosine similarity close to zero.")
chk5 = st.checkbox("I know why KD-trees fail in high dimensions due to the Curse of Dimensionality.")
chk6 = st.checkbox("I understand why pre-normalizing vectors to unit length converts Cosine Similarity into an efficient Dot Product.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Vector Basics and are ready for Chapter 2.2: Embeddings!")
