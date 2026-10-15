import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="21.3 Memory Optimization - Embeddings Curriculum",
    page_icon="💾",
    layout="wide"
)

# Title & Overview
st.title("21.3 Memory Optimization: Float16, Scalar Quantization (SQ8), Binary Quantization, & Matryoshka Truncation")
st.caption("Chapter 21: Embedding Optimization | Domain: 3. Vector Embeddings")
st.markdown("""
As a vector database scales into millions of records, the dominant cost shifts from API inference to **RAM (Random Access Memory)**.
Vector indexes like HNSW require vectors and graph edges to reside directly in memory for sub-millisecond retrieval.

Consider the raw mathematics of full-precision storage:
* A 1536-dimensional vector stored as **Float32** consumes **6,144 bytes**.
* Storing **10,000,000 vectors** requires **61.4 GB of RAM** just for raw vector floats, ballooning to over **100 GB** once graph edges and metadata payloads are included!
* Renting high-memory cloud RAM instances costs hundreds of dollars per month.

To smash through the memory wall, modern vector systems deploy **Vector Compression & Quantization**:
* **Float16**: Halves RAM footprint (2 bytes/dim) with zero perceptible accuracy loss.
* **Scalar Quantization (SQ8)**: Compresses floats into 8-bit integers (1 byte/dim), reducing RAM by 75%.
* **Binary Quantization (1-Bit BQ)**: Compresses floats into single bits, achieving a **32x reduction** in RAM and executing searches via lightning-fast CPU hardware POPCNT instructions!
* **Matryoshka Representation Learning (MRL)**: Truncates high-dimensional vectors (e.g. 1536 to 512) while preserving 98%+ of retrieval accuracy.

This lesson explores the mathematics, memory tradeoffs, and implementation patterns of vector compression.
""")

st.divider()

# Section 1: The Memory Scaling Wall & Quantization Taxonomy
st.header("1. The Memory Scaling Wall & The Quantization Hierarchy", divider="rainbow")

col_mem1, col_mem2 = st.columns([1, 1])

with col_mem1:
    st.markdown("""
    ### The 4 Quantization Tiers
    * **1. Float32 (Full Precision - Baseline)**:
      * 4 bytes per dimension. Standard output from PyTorch and APIs.
      * Highest mathematical fidelity, but heaviest memory footprint.
    * **2. Float16 / Bfloat16 (Half Precision)**:
      * 2 bytes per dimension (**2x compression / 50% RAM savings**).
      * Cosine similarity accuracy retention: **99.99%**.
    * **3. Scalar Quantization (SQ8 / Int8)**:
      * Maps continuous floats in `[-1.0, 1.0]` onto 256 discrete integer buckets `[-128, 127]` (1 byte/dim).
      * **4x compression / 75% RAM savings**.
      * Accuracy retention: **99.0% - 99.5%**.
    * **4. Binary Quantization (1-Bit BQ)**:
      * Maps values $> 0$ to `1` and $\\le 0$ to `0`.
      * Consumes only 1 bit per dimension (**32x compression / 96.9% RAM savings**).
      * Calculates distances using bitwise XOR and hardware **POPCNT** (population count) instructions, achieving **10x to 40x faster search speeds**!
    """)

with col_mem2:
    st.markdown("### The Vector Compression Hierarchy")
    comp_dot = """
    digraph CompressionHierarchy {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        F32 [label="Float32 (Baseline)\\n4 bytes / dim (6,144 bytes / vector)\\n10M Vectors = 61.4 GB RAM", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        F16 [label="Float16 (Half Precision)\\n2 bytes / dim (3,072 bytes / vector)\\n10M Vectors = 30.7 GB RAM (2x saved)", fillcolor="#FED7AA"];
        SQ8 [label="Scalar Quantization (Int8)\\n1 byte / dim (1,536 bytes / vector)\\n10M Vectors = 15.4 GB RAM (4x saved)", fillcolor="#FEF08A"];
        BQ [label="Binary Quantization (1-Bit)\\n0.125 bytes / dim (192 bytes / vector)\\n10M Vectors = 1.9 GB RAM (32x saved!)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        F32 -> F16 [label="Cast float16 (99.9% recall)"];
        F16 -> SQ8 [label="Scale to int8 (99.2% recall)"];
        SQ8 -> BQ [label="Sign thresholding (>0 -> 1)\\nHardware POPCNT XOR search", color="#059669"];
    }
    """
    st.graphviz_chart(comp_dot, use_container_width=False)

st.markdown("### Memory Footprint & Scaling Comparison (10,000,000 Vectors, d=1536)")
st.markdown("""
| Precision Format | Bits / Dimension | Bytes / Vector | RAM for 10M Vectors | Compression Ratio | Recall Retention | Search Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Float32** | 32 bits | 6,144 bytes | **61.4 GB** | 1.0x (Baseline) | 100% | 1.0x |
| **Float16** | 16 bits | 3,072 bytes | **30.7 GB** | **2.0x (50% saved)** | 99.9% | 1.2x |
| **Scalar Int8 (SQ8)** | 8 bits | 1,536 bytes | **15.4 GB** | **4.0x (75% saved)** | 99.2% | 2.5x |
| **Binary (1-Bit BQ)** | 1 bit | 192 bytes | **1.9 GB** | **32.0x (96.9% saved!)** | 95.0% - 97.0% | **20x - 40x (POPCNT)** |
| **MRL (512 dim Float32)**| 32 bits | 2,048 bytes | **20.5 GB** | **3.0x** | 98.5% | 3.0x |
| **MRL 512 dim + BQ** | 1 bit | 64 bytes | **0.64 GB** | **96.0x (Ultra-compact!)**| 94.0% | **50x** |
""")

st.divider()

# Section 2: Production Code: Pure-Python Quantization Suite
st.header("2. Production Code: Pure-Python Quantization & Bitpacking Suite", divider="rainbow")

st.markdown("""
The production utility below implements Float16 casting, Scalar Int8 Quantization, Binary Bitpacking using `np.packbits`, and Matryoshka dimension truncation.
""")

st.code(r'''
import numpy as np
from typing import Dict, Any, Tuple

class VectorQuantizationSuite:
    """
    Implements production vector compression and quantization methods:
    - Float16 half-precision casting
    - Scalar Quantization (SQ8: float32 -> int8)
    - Binary Quantization (1-bit BQ: float32 -> uint8 bitpack)
    - Matryoshka Representation Learning (MRL) dimension truncation
    """
    @staticmethod
    def to_float16(vector: np.ndarray) -> np.ndarray:
        return vector.astype(np.float16)

    @staticmethod
    def scalar_quantize_int8(vector: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Quantizes normalized vector [-1.0, 1.0] into int8 [-128, 127].
        Returns (int8_array, min_val, scale) for dequantization if needed.
        """
        min_val = float(np.min(vector))
        max_val = float(np.max(vector))
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        
        # Scale to [0, 255] then shift to int8 [-128, 127]
        quantized = np.clip(np.round((vector - min_val) / scale) - 128, -128, 127).astype(np.int8)
        return quantized, min_val, scale

    @staticmethod
    def binary_quantize_bitpack(vector: np.ndarray) -> np.ndarray:
        """
        1-Bit Binary Quantization:
        Values > 0 become 1; values <= 0 become 0.
        Packs 8 boolean bits into a single uint8 byte.
        Reduces 1536 floats (6144 bytes) to 192 bytes!
        """
        binary_bools = vector > 0
        bitpacked = np.packbits(binary_bools)
        return bitpacked

    @staticmethod
    def hamming_distance(bitpacked_a: np.ndarray, bitpacked_b: np.ndarray) -> int:
        """
        Computes Hamming distance between two binary vectors using XOR and POPCNT.
        """
        xor_result = np.bitwise_xor(bitpacked_a, bitpacked_b)
        # Count set bits across packed uint8 bytes
        return int(sum(bin(byte).count("1") for byte in xor_result))

    @staticmethod
    def matryoshka_truncate(vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Matryoshka Representation Learning (MRL) truncation:
        Slices the first target_dim coordinates and re-normalizes to unit sphere.
        """
        truncated = vector[:target_dim]
        norm = np.linalg.norm(truncated)
        return truncated / max(1e-12, norm)
''', language="python")

st.divider()

# Section 3: Interactive Quantization Studio
st.header("3. Interactive Workbench: Live Quantization & RAM Calculator", divider="rainbow")

st.markdown("""
Experiment with different quantization formats and vector dimensions to see how they impact **RAM consumption**, **Storage Bytes**, and **Search Speedup**.
""")

col_q_in, col_q_out = st.columns([1, 1])

with col_q_in:
    st.subheader("1. Vector Database Scale")
    scale_vectors = st.number_input(
        "Corpus Vector Count ($N$):",
        min_value=100000,
        max_value=100000000,
        value=5000000,
        step=500000,
        help="Number of documents/chunks in the vector index."
    )

    vector_dim = st.selectbox(
        "Embedding Vector Dimensions ($D$):",
        [384, 768, 1024, 1536, 3072],
        index=3,
        help="384: MiniLM / BGE-small. 768: BGE-base / Nomic. 1536: OpenAI small. 3072: OpenAI large."
    )

    st.subheader("2. Quantization Technique")
    quant_format = st.selectbox(
        "Select Compression Format:",
        [
            "Full Precision Float32 (Baseline)",
            "Half Precision Float16 (2x Compression)",
            "Scalar Quantization Int8 SQ8 (4x Compression)",
            "Binary Quantization 1-Bit BQ (32x Compression)",
            "Matryoshka Truncation (1536 -> 512 dim Float32)",
            "Matryoshka (512 dim) + Binary Quantization (96x Compression!)"
        ],
        index=3
    )

with col_q_out:
    st.subheader("3. RAM Telemetry & Performance Impact")

    # Calculate memory
    if "Float32 (Baseline)" in quant_format:
        bytes_per_dim = 4.0
        eff_dim = vector_dim
        recall_est = "100.0%"
        speed_est = "1.0x (Baseline)"
    elif "Float16" in quant_format:
        bytes_per_dim = 2.0
        eff_dim = vector_dim
        recall_est = "99.9%"
        speed_est = "1.2x"
    elif "Int8 SQ8" in quant_format:
        bytes_per_dim = 1.0
        eff_dim = vector_dim
        recall_est = "99.2%"
        speed_est = "2.5x"
    elif "Binary Quantization 1-Bit" in quant_format:
        bytes_per_dim = 0.125 # 1 bit
        eff_dim = vector_dim
        recall_est = "96.5%"
        speed_est = "25.0x (POPCNT hardware)"
    elif "Matryoshka Truncation" in quant_format:
        bytes_per_dim = 4.0
        eff_dim = 512
        recall_est = "98.8%"
        speed_est = f"{vector_dim / 512.0:.1f}x"
    else: # MRL 512 + BQ
        bytes_per_dim = 0.125
        eff_dim = 512
        recall_est = "94.5%"
        speed_est = "40.0x"

    bytes_per_vector = int(eff_dim * bytes_per_dim)
    total_bytes = scale_vectors * bytes_per_vector
    total_gb = total_bytes / (1024.0**3)

    baseline_bytes = scale_vectors * (vector_dim * 4.0)
    baseline_gb = baseline_bytes / (1024.0**3)

    comp_ratio = baseline_bytes / float(total_bytes)
    ram_saved_gb = max(0.0, baseline_gb - total_gb)

    m1, m2, m3 = st.columns(3)
    m1.metric("RAM Required", f"{total_gb:.2f} GB", delta=f"-{ram_saved_gb:.2f} GB" if ram_saved_gb > 0 else "Baseline")
    m2.metric("Compression Ratio", f"{comp_ratio:.1f}x", help="Relative memory reduction")
    m3.metric("Estimated Recall", recall_est)

    st.markdown("#### Hardware Cost & Specifications Breakdown")
    df_hw = pd.DataFrame([
        {"Specification": "Bytes per Individual Vector", "Baseline (Float32)": f"{vector_dim * 4:,} bytes", "Selected Format": f"{bytes_per_vector:,} bytes"},
        {"Specification": "Total Vector RAM (Without Graph)", "Baseline (Float32)": f"{baseline_gb:.2f} GB", "Selected Format": f"{total_gb:.2f} GB"},
        {"Specification": "Estimated Cloud RAM Cost", "Baseline (Float32)": f"${baseline_gb * 4.50:.0f} / month", "Selected Format": f"${total_gb * 4.50:.0f} / month"},
        {"Specification": "Search Speed Multiplier", "Baseline (Float32)": "1.0x", "Selected Format": speed_est}
    ])
    st.dataframe(df_hw, use_container_width=True)

    if comp_ratio >= 10.0:
        st.success(f"🚀 **Ultra-Compact Footprint**: Reduced RAM by **{comp_ratio:.1f}x**! A 5-million vector database fits in just **{total_gb:.2f} GB** of RAM.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Quantization Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How can 1-Bit Binary Quantization (BQ) achieve 96%+ recall while discarding 96.9% of the floating-point precision?**
    * **Answer**: In high-dimensional spaces ($d=1536$), semantic orientation is governed by the **signs of coordinates across orthogonal hyperplanes** rather than exact float magnitudes.
    * If a vector has positive activations in dimensions representing 'medical' and negative activations in 'sports', the binary bitmask `[1, 0, 1, ...]` retains the exact quadrant coordinate of the vector.
    * Furthermore, production systems use **Two-Stage Re-scoring (Over-sampling)**:
      1. Stage 1: Fast Hamming distance search over binary vectors retrieves top $K=50$ candidates in $< 1\\text{ms}$.
      2. Stage 2: Re-rank the top 50 candidates using original float32 vectors to achieve 99%+ final recall!

    **Q2: What is Matryoshka Representation Learning (MRL) and how does it allow arbitrary dimension truncation?**
    * **Answer**: Traditional embedding models distribute information diffusely across all dimensions. If you slice off the second half, the vector is destroyed.
    * **MRL** (introduced by OpenAI in `text-embedding-3` and Nomic in `nomic-embed-text`) modifies the training loss function to force early dimensions to carry the highest information density (like Russian nesting dolls).
    * Slicing the vector from 1536 to 512 dimensions retains **over 98.5% of full retrieval performance** while saving 66% of disk and RAM space!

    **Q3: What is the hardware mechanism that makes Binary Quantization vector search so fast on modern CPUs?**
    * **Answer**: Calculating cosine similarity between Float32 vectors requires hundreds of floating-point multiply-accumulate (FMAC) instructions.
    * For binary vectors, similarity is calculated using **Hamming Distance**:
      1. A bitwise `XOR` instruction finds differing bits.
      2. A single hardware **`POPCNT`** (Population Count) CPU instruction counts the number of 1-bits in a 64-bit register in a single clock cycle.
      * This allows modern x86/ARM CPUs to evaluate millions of vector distances per second without touching a GPU!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Applying Binary Quantization to Low-Dimensional Models**:
      * *Issue*: Applying 1-bit BQ to a 128-dimensional embedding model.
      * *Consequence*: Severe recall degradation (recall drops below 60%). BQ requires high dimensionality ($d \\ge 768$, ideally $d \\ge 1536$) where high-dimensional geometry compensates for 1-bit quantization.
      * *Fix*: Only use Binary Quantization on large models ($d \\ge 768$ or $d=1536$). Use Int8 SQ8 for smaller models.

    * **Trap 2: Forgetting to Unit-Normalize After Matryoshka Dimension Truncation**:
      * *Issue*: Slicing the first 512 dimensions of an OpenAI vector without recalculating `v / np.linalg.norm(v)`.
      * *Consequence*: The vector is no longer a unit vector; dot product search produces distorted distance scores.
      * *Fix*: Always re-normalize truncated vectors to unit L2 norm immediately after slicing.

    * **Trap 3: Quantizing Vectors Without an Outlier Preservation Strategy**:
      * *Issue*: Global scalar quantization across vectors that contain massive outlier coordinate spikes.
      * *Consequence*: Quantization buckets are dominated by the outliers, compressing 99% of normal coordinate values into 2 or 3 bins.
      * *Fix*: Use per-vector min/max scaling or outlier-clipping quantizers.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Binary Quantization (BQ)**
      * *Definition*: Compressing continuous float coordinates into single bits based on sign ($>0 \\rightarrow 1$), achieving 32x RAM reduction and hardware POPCNT search.
    * **Flashcard 2: Scalar Quantization (SQ8)**
      * *Definition*: Quantizing 32-bit floats into 8-bit signed integers, reducing vector memory by 75% with 99%+ recall retention.
    * **Flashcard 3: Matryoshka Representation Learning (MRL)**
      * *Definition*: Training embedding models so that the initial $D$ dimensions capture the primary semantic signal, allowing arbitrary dimension truncation.
    * **Flashcard 4: Two-Stage Re-scoring (Over-sampling)**
      * *Definition*: Fast candidate retrieval of top 50 items using quantized vectors followed by exact re-ranking of top 5 items using full-precision vectors.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Memory Optimization:

- [ ] I can calculate the RAM footprint of an arbitrary corpus in Float32, Float16, Int8, and Binary formats.
- [ ] I understand how Binary Quantization achieves 32x compression and why it requires $d \\ge 768$.
- [ ] I know how hardware POPCNT and bitwise XOR accelerate binary Hamming distance lookups.
- [ ] I can explain the mechanics of Matryoshka Representation Learning (MRL) dimension truncation.
- [ ] I understand the two-stage re-scoring pattern combining binary retrieval with float32 re-ranking.
""")
