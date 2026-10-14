import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="11.3 Storage Optimization - Vector DB Curriculum",
    page_icon="🗜️",
    layout="wide"
)

# Title & Overview
st.title("11.3 Storage Optimization: Quantization, Reduced Precision & Disk Indices")
st.caption("Chapter 11: Vector Database Storage | Domain: 4. Vector Database")
st.markdown("""
At scale, storing raw 32-bit floating-point embeddings (`float32`) becomes the single largest driver of vector database infrastructure costs. 

For 100 million vectors of dimension 1,536:
* Raw `float32` vectors require **614.4 GB of RAM**, costing upwards of $2,000 to $4,000 per month in cloud memory instances.
* By applying **Scalar Quantization (SQ8)**, memory drops to **153.6 GB** (4x reduction).
* By applying **Product Quantization (PQ)**, memory drops to **~38 GB** (16x reduction).
* By applying **Binary Quantization (BQ)**, memory drops to **19.2 GB** (32x reduction).

Furthermore, modern vector storage architectures like **DiskANN** and **Two-Stage Re-ranking** allow storing raw vectors on high-speed NVMe SSDs 
while keeping lightweight quantized sketches in RAM, achieving sub-10 millisecond latency on billion-scale collections.

In this lesson, we explore the mathematics and mechanics of vector quantization, compare **Float16, Int8, PQ, and 1-bit Binary Quantization**, 
examine Matryoshka embeddings, and experiment with an interactive Vector Quantization & Error Lab.
""")

st.divider()

# Section 1: The Spectrum of Vector Quantization
st.header("1. The Spectrum of Vector Quantization", divider="rainbow")

col_spec1, col_spec2 = col_spec1, col_spec2 = st.columns([1.1, 0.9])

with col_spec1:
    st.markdown("""
    ### The Four Quantization Levels
    
    1. **Half-Precision (Float16 / BFloat16)**:
       * Drops precision from 32 bits to 16 bits per dimension.
       * **Compression**: **2x (50% RAM savings)**.
       * **Recall Impact**: **Virtually 0% loss** (>99.9% identical retrieval). Modern CPUs and GPUs natively accelerate FP16 math.
    
    2. **Scalar Quantization (SQ8 / Int8)**:
       * Maps the continuous floating-point range `[min, max]` of each dimension into 256 discrete 8-bit integer buckets (`uint8` or `int8`):
         ```text
         q_val = round( 255 * (x - min) / (max - min) )
         ```
       * **Compression**: **4x (75% RAM savings)**.
       * **Recall Impact**: Typically <0.5% recall loss. Distances are computed via fast SIMD integer instructions (`VPDPBUSD`).
    
    3. **Product Quantization (PQ)**:
       * Splits vectors into `M` sub-vectors and replaces each sub-vector with an 8-bit cluster centroid index.
       * **Compression**: **8x to 32x RAM savings**.
       * **Recall Impact**: 3% to 8% loss. Used for massive collections (50M+ vectors).
    
    4. **Binary Quantization (1-bit BQ)**:
       * Stores only the sign bit of each dimension: `1 if x > 0 else 0`.
       * **Compression**: **32x RAM savings** (1536 floats become 192 bytes!).
       * **Distance Computation**: Ultra-fast bitwise `XOR` followed by `POPCOUNT` (hardware instruction).
    """)

with col_spec2:
    st.markdown("### Quantization Compression Hierarchy")
    quant_dot = """
    digraph QuantHierarchy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        FP32 [label="Raw Float32\\n4 bytes / dim (6,144 bytes / 1536-dim)\\n100% Precision | High RAM Cost", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        FP16 [label="Float16 (Half Precision)\\n2 bytes / dim (3,072 bytes)\\n2x Savings | 99.9% Recall", fillcolor="#FEF3C7", color="#D97706"];

        SQ8 [label="Scalar Quantization (Int8)\\n1 byte / dim (1,536 bytes)\\n4x Savings | 99.2% Recall", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        PQ [label="Product Quantization (PQ-64)\\n0.125 byte / dim (96 bytes)\\n32x - 64x Savings | 92% Recall", fillcolor="#E0E7FF", color="#3B82F6"];

        BQ [label="Binary Quantization (1-bit)\\n1 bit / dim (192 bytes)\\n32x Savings | Hamming Bitwise", fillcolor="#FCE7F3", color="#DB2777"];

        FP32 -> FP16 [label="Truncate mantissa"];
        FP16 -> SQ8 [label="Linear bucket mapping"];
        SQ8 -> PQ [label="Sub-space codebooks"];
        SQ8 -> BQ [label="Sign thresholding (x > 0)"];
    }
    """
    st.graphviz_chart(quant_dot, use_container_width=False)

st.divider()

# Section 2: Disk-Native Index Architecture: DiskANN & Two-Stage Search
st.header("2. Disk-Native Index Architecture: DiskANN & Two-Stage Search", divider="rainbow")

col_disk1, col_disk2 = col_disk1, col_disk2 = st.columns([1, 1])

with col_disk1:
    st.markdown("""
    ### Why Two-Stage Search Solves the RAM Bottleneck
    In traditional HNSW, both the graph edges and the raw vectors must reside in RAM. When data exceeds RAM, the system crashes.
    
    Modern disk-native vector engines (such as **Microsoft DiskANN**, LanceDB, and Qdrant) use a **Two-Stage Retrieval Architecture**:
    
    * **Stage 1 (In-Memory Fast Candidate Filtering)**:
      * Compressed vectors (SQ8 or Binary) and compressed graph indices reside in RAM.
      * The query navigates the graph in RAM in ~2ms to retrieve the top `N_candidates` (e.g. top 100).
    * **Stage 2 (SSD Direct Re-ranking)**:
      * The engine performs 100 fast asynchronous random reads (`io_uring`) to NVMe SSD to fetch the **exact raw Float32 vectors** for those 100 candidates.
      * Computes exact distances and reranks down to the final top 10.
    * **Result**: Achieves **99.5% exact recall** while slashing RAM requirements by **80% to 90%**!
    """)

with col_disk2:
    st.markdown("""
    ### Matryoshka Representation Learning (MRL)
    Modern embedding models (e.g. OpenAI `text-embedding-3-small`, `text-embedding-3-large`, `nomic-embed-text`) are trained with **Matryoshka Representation Learning (MRL)**:
    
    * The training loss forces the **first dimensions of the vector** to carry the most essential semantic information.
    * You can simply **slice the vector** to fewer dimensions without retraining:
      ```python
      # OpenAI text-embedding-3-small (Default: 1536 dim)
      # Truncate to first 512 dimensions:
      short_vector = full_vector[:512]

      # Renormalize to unit length for cosine similarity:
      short_vector = short_vector / np.linalg.norm(short_vector)
      ```
    * **Impact**: Slicing from 1,536 to 512 dimensions cuts storage and computation by **66.7%**, while retaining **98%+ of retrieval accuracy**!
    """)

st.divider()

# Section 3: Comprehensive Storage Optimization Comparison
st.header("3. Comprehensive Optimization Strategy Comparison", divider="rainbow")

st.markdown("""
Compare the five primary storage optimization techniques across compression ratio, recall retention, and compute complexity:
""")

OPT_STRATEGIES = [
    {
        "Technique": "Float16 (Half Precision)",
        "Compression": "2x (50% savings)",
        "Recall Retention": "> 99.9%",
        "Compute Acceleration": "Native GPU / AVX512 FP16",
        "Best For": "Universal production drop-in replacement"
    },
    {
        "Technique": "Scalar Quantization (SQ8)",
        "Compression": "4x (75% savings)",
        "Recall Retention": "98.5% - 99.5%",
        "Compute Acceleration": "AVX-512 VNNI / NEON Int8 SIMD",
        "Best For": "Enterprise collections (1M - 20M vectors)"
    },
    {
        "Technique": "Product Quantization (PQ)",
        "Compression": "8x - 32x (88% - 97% savings)",
        "Recall Retention": "88.0% - 94.0%",
        "Compute Acceleration": "L1 cache table lookups (ADC)",
        "Best For": "Billion-scale web search & recommendation"
    },
    {
        "Technique": "Binary Quantization (BQ)",
        "Compression": "32x (96.9% savings)",
        "Recall Retention": "85.0% - 92.0% (Higher with rescoring)",
        "Compute Acceleration": "Hardware POPCOUNT & XOR",
        "Best For": "High-throughput first-stage candidate retrieval"
    },
    {
        "Technique": "Matryoshka Slicing (MRL)",
        "Compression": "3x (e.g. 1536 -> 512 dims)",
        "Recall Retention": "96.5% - 98.5%",
        "Compute Acceleration": "3x fewer FLOPs per comparison",
        "Best For": "Models supporting MRL (OpenAI 3, Nomic)"
    }
]

st.dataframe(pd.DataFrame(OPT_STRATEGIES), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Vector Quantization & Error Lab
st.header("4. Interactive Workbench: Vector Quantization & Error Lab", divider="rainbow")

st.markdown("""
Observe how quantization transforms a high-dimensional vector, analyze the **reconstruction error (RMSE)**, 
and measure cosine fidelity against the uncompressed original vector.
""")

col_lab_cfg, col_lab_disp = st.columns([1, 1.2])

with col_lab_cfg:
    st.subheader("1. Vector Configuration")
    
    lab_dims = st.selectbox("Vector Dimensions (D):", [128, 384, 768, 1536], index=2)
    lab_seed = st.number_input("Random Seed (for reproducibility):", min_value=1, max_value=9999, value=42)
    
    # Generate synthetic normalized embedding vector
    np.random.seed(lab_seed)
    raw_vec = np.random.randn(lab_dims).astype(np.float32)
    raw_vec = raw_vec / np.linalg.norm(raw_vec)  # L2 normalize
    
    selected_quant = st.selectbox(
        "2. Apply Quantization Technique:",
        [
            "Float16 (Half Precision)",
            "Scalar Quantization (SQ8 / Int8)",
            "Binary Quantization (1-bit BQ)",
            "Matryoshka Truncation (50% Dims)"
        ]
    )

with col_lab_disp:
    st.subheader("2. Quantization Transformation Results")
    
    if selected_quant == "Float16 (Half Precision)":
        quant_vec = raw_vec.astype(np.float16)
        reconstructed = quant_vec.astype(np.float32)
        bytes_per_dim = 2.0
        tech_note = "Mantissa truncated from 23 bits to 10 bits."
    elif selected_quant == "Scalar Quantization (SQ8 / Int8)":
        v_min, v_max = float(raw_vec.min()), float(raw_vec.max())
        # Quantize to 0..255 uint8
        sq_scaled = np.round(255.0 * (raw_vec - v_min) / (v_max - v_min)).astype(np.uint8)
        # Dequantize back to float
        reconstructed = (sq_scaled.astype(np.float32) / 255.0) * (v_max - v_min) + v_min
        # Renormalize
        reconstructed = reconstructed / np.linalg.norm(reconstructed)
        bytes_per_dim = 1.0
        tech_note = f"Scaled to 256 discrete bins between [{v_min:.3f}, {v_max:.3f}]."
    elif selected_quant == "Binary Quantization (1-bit BQ)":
        # 1 if x > 0 else 0
        bq_bits = (raw_vec > 0).astype(np.uint8)
        # Reconstruct as +1 / -1 normalized
        reconstructed = np.where(bq_bits == 1, 1.0, -1.0).astype(np.float32)
        reconstructed = reconstructed / np.linalg.norm(reconstructed)
        bytes_per_dim = 1.0 / 8.0
        tech_note = "Stored as 1 bit per dimension (sign bit). Evaluated via Hamming bitwise POPCOUNT."
    else:  # Matryoshka
        half_d = lab_dims // 2
        sliced = raw_vec[:half_d]
        reconstructed_short = sliced / np.linalg.norm(sliced)
        # Pad with zeros for dimensional metrics
        reconstructed = np.zeros(lab_dims, dtype=np.float32)
        reconstructed[:half_d] = reconstructed_short
        bytes_per_dim = 2.0  # 50% dimensions at float32
        tech_note = f"Truncated from {lab_dims} dims to {half_d} dims and renormalized."

    # Compute fidelity metrics
    cos_sim = float(np.dot(raw_vec, reconstructed))
    rmse = float(np.sqrt(np.mean((raw_vec - reconstructed)**2)))
    orig_bytes = lab_dims * 4
    quant_bytes = int(lab_dims * bytes_per_dim)
    savings_pct = (1.0 - (quant_bytes / orig_bytes)) * 100.0

    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.metric(
            label="Cosine Similarity to Original",
            value=f"{cos_sim:.5f}",
            delta=f"{(1.0 - cos_sim) * 100:.3f}% Distortion"
        )
        st.metric(
            label="Root Mean Squared Error (RMSE)",
            value=f"{rmse:.5f}"
        )
    with col_met2:
        st.metric(
            label="Memory Footprint per Vector",
            value=f"{quant_bytes} Bytes",
            delta=f"-{savings_pct:.1f}% RAM (Original: {orig_bytes}B)"
        )
        st.metric(
            label="Compression Factor",
            value=f"{orig_bytes / quant_bytes:.1f}x Smaller"
        )

    st.info(f"**Mechanism**: {tech_note}")

st.subheader("3. Visualizing Component Values: First 16 Dimensions")
df_comp = pd.DataFrame({
    "Dim Index": [f"D{i}" for i in range(16)],
    "Original Float32": [round(float(x), 4) for x in raw_vec[:16]],
    "Reconstructed": [round(float(x), 4) for x in reconstructed[:16]],
    "Delta Error": [round(float(abs(a - b)), 5) for a, b in zip(raw_vec[:16], reconstructed[:16])]
})
st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the mathematical intuition behind Binary Quantization (BQ) and why does it work for embeddings?"):
    st.markdown("""
    **Answer**:
    In high-dimensional spherical embeddings, vectors are distributed across the surface of a unit hypersphere. 
    * By taking only the sign of each dimension (`+1` if `x > 0`, `-1` if `x <= 0`), Binary Quantization projects each vector onto one of the `2^D` vertices of a hypercube.
    * For high dimensions (e.g., `D = 1,536`), the angular separation between points is remarkably well-preserved by the fraction of agreeing sign bits.
    * The cosine similarity between two binary vectors is directly proportional to their **Hamming Distance**:  
      `Cosine_Sim = 1.0 - (2.0 * Hamming_Dist / D)`.
    * Because modern CPUs have native SIMD instructions (`_mm512_popcnt_epi64`) that count matching bits in a single CPU cycle, BQ searches can evaluate **tens of millions of vectors per millisecond**.
    """)

with st.expander("Q2: How does Scalar Quantization (SQ8) compute distance without dequantizing back to float32?"):
    st.markdown("""
    **Answer**:
    In Scalar Quantization, each vector dimension is mapped to an 8-bit integer: `v_i = scale * q_i + offset`.
    When computing dot products between query `Q` and quantized vector `V`:
    1. The dot product can be expanded algebraically:  
       `<Q, V> = sum(Q_i * (scale * q_i + offset)) = scale * sum(Q_i * q_i) + offset * sum(Q_i)`.
    2. The term `sum(Q_i)` depends only on the query and is computed once.
    3. The term `sum(Q_i * q_i)` is computed using **integer-floating point fused multiply-add (VNNI) instructions** in CPU registers.
    The database never expands all 1,536 dimensions back into float32 in RAM, maximizing cache locality.
    """)

with st.expander("Q3: What is the Two-Stage Retrieval pattern and how does it prevent recall degradation when using lossy quantization?"):
    st.markdown("""
    **Answer**:
    Lossy quantization (like PQ or BQ) achieves extreme speed and memory compression but suffers a 5-10% recall penalty if used alone.
    
    The **Two-Stage Pattern** solves this:
    * **Stage 1 (Coarse Filter)**: Retrieve `K_coarse = 100` candidates using fast quantized vectors in RAM (taking <1-2ms). Because you retrieve 100 candidates instead of 10, the true top-10 are almost certainly present in the 100-candidate pool.
    * **Stage 2 (Exact Re-ranking)**: Fetch the exact uncompressed `float32` vectors for only those 100 candidates from disk or SSD cache, compute exact Euclidean or cosine distances, and return the true top 10.
    * **Result**: You achieve the **speed and memory efficiency of 16x quantization** with the **100% precision of exact float32 search**.
    """)

with st.expander("Q4: Can any embedding model use Matryoshka Representation Learning (MRL) dimension slicing?"):
    st.markdown("""
    **Answer**:
    **No**. Standard embedding models (like BERT, Sentence-Transformers, or older ada-002) distribute semantic information unpredictably across all dimensions. Slicing off dimensions from a non-MRL model destroys semantic fidelity, causing recall to collapse to near zero.
    
    Only models explicitly trained with a Matryoshka loss function (such as OpenAI `text-embedding-3`, Nomic Embed, BGE-M3, and Cohere Embed v3) can be truncated. In MRL models, the loss function evaluates loss at multiple prefix sub-vector lengths during training, guaranteeing that the first `K` dimensions form a valid semantic vector.
    """)

with st.expander("Q5: What is the impact of vector normalization when applying Scalar Quantization?"):
    st.markdown("""
    **Answer**:
    Vectors should always be **L2-normalized to unit length** (`||v|| = 1.0`) before applying Scalar Quantization.
    * When vectors have unit length, the dynamic range of each component is strictly bounded in `[-1.0, 1.0]`, allowing optimal allocation of the 256 quantization bins.
    * If vectors have unconstrained magnitudes (common in raw unnormalized dot product models), outlier dimensions with large magnitudes compress all smaller dimensions into a few coarse quantization bins, dramatically increasing reconstruction error.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Truncating Non-MRL Embeddings**  
    Slicing dimensions (`vec[:256]`) on a legacy model like `all-MiniLM-L6-v2` destroys search accuracy. Only truncate models explicitly documented to support Matryoshka Representation Learning.
    """)
    st.warning("""
    **Trap 2: Forgetting to Re-Normalize After Quantization**  
    Quantizing and reconstructing vectors introduces minor scaling deviations (`norm != 1.0`). In cosine similarity search, always re-normalize reconstructed vectors before computing distance.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Binary Quantization on Low Dimensions (< 384)**  
    Binary Quantization relies on high-dimensional concentration of measure. On 64 or 128 dimensions, 1-bit quantization loses too much angular resolution, dropping recall below 60%.
    """)
    st.warning("""
    **Trap 4: Premature Optimization on Small Collections (< 100k)**  
    Quantization adds operational complexity and training overhead. If your dataset fits easily in 4 GB of RAM, stick with standard `float32` or `float16`. Optimize when data exceeds 1M+ vectors.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Scalar Quantization (SQ8)?"):
        st.markdown("""
        **SQ8 Mechanics**:  
        Maps continuous float32 values into 256 discrete 8-bit integer bins, cutting RAM by 4x with <0.5% recall loss.
        """)
    with st.expander("🗂️ Flashcard: How does Binary Quantization compute distance?"):
        st.markdown("""
        **Hamming Bitwise**:  
        Stores 1 bit per dimension (sign). Distances are computed via hardware `XOR` and `POPCOUNT` bit instructions.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Two-Stage Retrieval?"):
        st.markdown("""
        **Two-Stage Search**:  
        Uses fast quantized vectors in RAM to filter top-100 candidates, then reranks with exact float32 vectors from NVMe SSD.
        """)
    with st.expander("🗂️ Flashcard: What is Matryoshka Representation Learning (MRL)?"):
        st.markdown("""
        **MRL Truncation**:  
        Embedding models trained so the first prefix dimensions contain core semantic meaning, allowing safe dimension slicing.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the compression and recall trade-offs across Float16, SQ8, PQ, and Binary Quantization.", value=True)
chk2 = st.checkbox("I can implement Scalar Quantization (SQ8) linear bucket mapping in Python.", value=True)
chk3 = st.checkbox("I know how Binary Quantization leverages Hamming distance and hardware POPCOUNT instructions.", value=True)
chk4 = st.checkbox("I understand how Two-Stage Retrieval combines in-memory quantization with on-disk exact reranking.", value=True)
chk5 = st.checkbox("I know how Matryoshka Representation Learning (MRL) allows slicing embedding dimensions safely.", value=True)
chk6 = st.checkbox("I can design a cost-optimized vector storage architecture for 100M+ embeddings.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completely mastered Chapter 11: Vector Database Storage and are ready for Chapter 12: Document Processing for Vector Databases!")
