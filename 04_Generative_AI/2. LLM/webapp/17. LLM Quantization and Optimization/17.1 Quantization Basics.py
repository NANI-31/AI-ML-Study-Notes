import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="17.1 Quantization Basics - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionQuantizer:
    """
    Self-contained educational Quantizer implementing:
    - Symmetric uniform quantization
    - Asymmetric uniform quantization with zero-point offset
    - Exact scale and zero-point derivation
    - Mean Squared Error (MSE) reconstruction metric
    """
    def __init__(self):
        pass

    def quantize_symmetric(self, tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Symmetric quantization around zero:
        Scale S = max(|tensor|) / (2^(bits-1) - 1)
        q = clamp(round(tensor / S), -q_max, q_max)
        dequant = q * S
        """
        q_max = (2 ** (bits - 1)) - 1
        abs_max = float(np.max(np.abs(tensor)))
        scale = abs_max / q_max if abs_max > 0 else 1.0
        
        q = np.clip(np.round(tensor / scale), -q_max, q_max).astype(np.int32)
        dequant = q * scale
        return q, scale, dequant

    def quantize_asymmetric(self, tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Asymmetric quantization between [min, max]:
        Scale S = (max - min) / (2^bits - 1)
        Zero-point Z = round(-min / S)
        q = clamp(round(tensor / S) + Z, 0, 2^bits - 1)
        dequant = (q - Z) * S
        """
        q_max = (2 ** bits) - 1
        t_min = float(np.min(tensor))
        t_max = float(np.max(tensor))
        
        diff = t_max - t_min
        scale = diff / q_max if diff > 0 else 1.0
        zero_point = int(np.clip(np.round(-t_min / scale), 0, q_max))

        q = np.clip(np.round(tensor / scale) + zero_point, 0, q_max).astype(np.int32)
        dequant = (q - zero_point) * scale
        return q, scale, zero_point, dequant

    def compute_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Computes Mean Squared Quantization Error."""
        return float(np.mean((original - reconstructed) ** 2))

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("17.1 Quantization Basics")
st.caption("Chapter 17: LLM Quantization and Optimization | Domain: 2. Large Language Models")

st.markdown("""
A 70-billion-parameter model in standard 16-bit floating point precision requires **140 Gigabytes of GPU VRAM** 
just to hold its static weights. Adding the KV Cache pushes memory well beyond 160 GB, requiring dual H100 enterprise nodes.
**Quantization** is the mathematical discipline of mapping continuous high-precision floating-point numbers 
into discrete, compact low-bit integer representations (e.g. 8-bit, 4-bit, or ternary).

By compressing weights from 16 bits down to 8 or 4 bits, quantization delivers a **2x to 4x reduction in memory**, 
slashes memory-bandwidth pressure, and unlocks integer tensor core acceleration (INT8 / INT4 GEMM) with near-zero degradation in reasoning accuracy.

In this lesson, you will master:
1. The mathematical definition of Quantization: scale factors, zero-point offsets, and clipping.
2. Floating-point formats: FP32, FP16, and BF16 bit allocations and dynamic ranges.
3. Integer formats: INT8 and INT4 uniform quantization grids.
4. Symmetric vs. Asymmetric Quantization: zero-centered efficiency vs. arbitrary interval flexibility.
5. Weight Quantization: compressing static parameters offline.
6. Activation Quantization & The Outlier Dimension Challenge: why activations are 10x harder to quantize than weights.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Numerical Precision Hierarchy & Quantization Grid", divider="rainbow")

st.markdown("""
Quantization maps a continuous real-valued interval onto a discrete integer grid.
""")

quant_blueprint_dot = """
digraph QuantizationGrid {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_continuous {
        label = "Continuous Domain (FP32 / BF16)";
        color = "#334155";
        fontcolor = "#94A3B8";
        RealWeights [label="Original Weights W\\nContinuous Float Tensors\\nRange: [-3.842, +4.119]", fillcolor="#0F172A"];
        MinMax [label="Calibration: Find [alpha, beta]\\nalpha = min(W), beta = max(W)", fillcolor="#0369A1"];
        RealWeights -> MinMax;
    }

    subgraph cluster_mapping {
        label = "Quantization Mapping Operator";
        color = "#334155";
        fontcolor = "#94A3B8";
        ComputeScale [label="Compute Scale & Zero-Point\\nS = (beta - alpha) / (2^b - 1)\\nZ = round(-alpha / S)", fillcolor="#047857"];
        ClampRound [label="Quantize: q = clamp(round(W / S) + Z)\\nInteger Tensor: [-128..127] or [0..255]", fillcolor="#047857"];
        MinMax -> ComputeScale -> ClampRound;
    }

    subgraph cluster_dequant {
        label = "Runtime Execution";
        color = "#334155";
        fontcolor = "#94A3B8";
        Dequant [label="Dequantization: W_hat = (q - Z) * S\\nApproximates Original Weights", fillcolor="#B45309"];
        INT_GEMM [label="Fast INT8 Tensor Cores\\n2x-4x Higher Memory Bandwidth", fillcolor="#7C3AED"];
        ClampRound -> Dequant;
        ClampRound -> INT_GEMM;
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(quant_blueprint_dot, use_container_width=True)

st.markdown("""
### Core Mathematical Operators
* **Scale Factor (S)**: A positive floating-point number that maps an integer step to a real value increment.
* **Zero-Point (Z)**: An integer offset that maps the real value `0.0` exactly onto an integer coordinate.
* **Quantize Function**: `q = clamp(round(x / S) + Z, q_min, q_max)`
* **Dequantize Function**: `x_approx = (q - Z) * S`
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - WHAT IS QUANTIZATION?
# ==============================================================================
st.header("2. What is Quantization?", divider="rainbow")

st.markdown("""
Quantization is the process of approximating continuous signals with a finite set of discrete values.
In neural networks, weight tensors `W` and activation tensors `X` are traditionally stored as 16-bit or 32-bit floats.
Quantization maps these values into lower-bit formats (`b` bits) where each value can only take one of `2^b` discrete levels.
""")

col_wq1, col_wq2 = st.columns(2)
with col_wq1:
    st.subheader("Symmetric vs Asymmetric Quantization")
    st.markdown("""
    * **Symmetric Quantization**:
      * The real-value range is centered symmetrically around zero: `[-alpha, +alpha]`.
      * The zero-point is mathematically fixed: `Z = 0`.
      * `Scale S = max(|x|) / (2^(b-1) - 1)`.
      * **Fast Execution**: Eliminates zero-point subtraction during matrix multiplication (`W @ x`). Standard for weights.
    * **Asymmetric Quantization**:
      * Maps arbitrary intervals `[x_min, x_max]` to `[0, 2^b - 1]`.
      * Uses an explicit integer zero-point `Z`.
      * Ideal for activations that are strictly non-negative (e.g. after ReLU or GELU).
    """)

with col_wq2:
    st.subheader("The Fundamental Tradeoff")
    st.markdown("""
    * **Benefits**: 
      * 50% to 75% GPU memory reduction.
      * Memory-bandwidth bound decode speed jumps by 2x to 3x.
      * Faster hardware integer ALUs.
    * **Cost**: Introduces **Quantization Noise** (`Error = x - x_approx`). 
      If quantization is too aggressive (< 3 bits), perplexity surges and the model loses reasoning coherence.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - FP32
# ==============================================================================
st.header("3. FP32 (Full Precision IEEE 754)", divider="rainbow")

st.markdown("""
**FP32 (Single Precision)** allocates 32 bits per number:
```text
[ Sign: 1 bit ] [ Exponent: 8 bits ] [ Fraction / Mantissa: 23 bits ]
```
""")

col_fp32_1, col_fp32_2 = st.columns(2)
with col_fp32_1:
    st.markdown("#### Numerical Characteristics")
    st.markdown("""
    * **Dynamic Range**: `~1.4e-45` to `~3.4e38` (huge dynamic range governed by 8 exponent bits).
    * **Precision**: 23 mantissa bits give **~7 decimal digits of precision**.
    * **Memory Footprint**: **4 bytes per parameter**.
    * A 70B model requires **280 GB of memory** in FP32!
    """)

with col_fp32_2:
    st.markdown("#### Status in Modern Generative AI")
    st.markdown("""
    * **Obsolete for LLM Storage and Serving**: Consumes double the memory of BF16 with zero empirical accuracy gain.
    * **Preserved in Optimizers**: FP32 is still widely used internally for master weights and AdamW accumulator buffers during training to prevent gradient rounding error.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - FP16
# ==============================================================================
st.header("4. FP16 (Half Precision)", divider="rainbow")

st.markdown("""
**FP16** allocates 16 bits per number:
```text
[ Sign: 1 bit ] [ Exponent: 5 bits ] [ Fraction / Mantissa: 10 bits ]
```
""")

col_fp16_1, col_fp16_2 = st.columns(2)
with col_fp16_1:
    st.markdown("#### The Narrow Exponent Hazard")
    st.markdown("""
    * Because FP16 has only **5 exponent bits**, its maximum representable value is only **65,504**!
    * Minimum positive normal value is `~6.1e-5`.
    * **Underflow**: Values smaller than `1e-5` round to zero.
    * **Overflow**: Values exceeding `65,504` become `+Infinity` or `NaN`.
    """)

with col_fp16_2:
    st.markdown("#### FP16 in Practice")
    st.markdown("""
    * Widely used on older Volta/Turing architectures (V100, T4, RTX 2080).
    * Requires dynamic **Loss Scaling** during training.
    * In deep 70B+ LLMs, intermediate activation spikes easily exceed 65,504, causing sudden `NaN` crashes during FP16 inference!
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - BF16
# ==============================================================================
st.header("5. BF16 (Brain Floating Point)", divider="rainbow")

st.markdown("""
Developed by Google Brain and popularized across Ampere/Hopper GPUs, **BF16** reallocates bits to match FP32's dynamic range:
```text
[ Sign: 1 bit ] [ Exponent: 8 bits ] [ Fraction / Mantissa: 7 bits ]
```
""")

col_bf16_1, col_bf16_2 = st.columns(2)
with col_bf16_1:
    st.markdown("#### Why BF16 Replaced FP16 as the Industry Standard")
    st.markdown("""
    * By maintaining the **exact same 8-bit exponent as FP32**, BF16 shares the massive dynamic range of `1e-38` to `1e38`.
    * It is mathematically impossible for transformer activations to overflow or underflow in BF16.
    * Zero loss scaling required; 100% training and inference numerical stability.
    """)

with col_bf16_2:
    st.markdown("#### Hardware Support")
    st.markdown("""
    * Native on modern architectures: NVIDIA Ampere (A100, RTX 3000+), Hopper (H100), Blackwell (B200), Google TPUs (v2+).
    * Today, virtually all frontier models (Llama-3, Mistral, Gemma) are pre-trained and released natively in **BF16**.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - INT8
# ==============================================================================
st.header("6. INT8 (8-Bit Integer Quantization)", divider="rainbow")

st.markdown("""
**INT8** represents values as signed 8-bit integers: `[-128, +127]` (256 discrete levels, **1 byte per parameter**).
""")

col_int8_1, col_int8_2 = st.columns(2)
with col_int8_1:
    st.subheader("The Sweet Spot of Model Quantization")
    st.markdown("""
    * **50% Memory Reduction**: Compresses a 7B model from 14 GB down to **7 GB**.
    * **Lossless Accuracy**: Extensive benchmarking across MMLU, GSM8K, and HumanEval demonstrates that INT8 quantization causes **< 0.1% accuracy degradation**.
    * **Tensor Core Speedup**: INT8 GEMM execution on modern GPUs runs at **2x higher TFLOPS** than 16-bit floating point.
    """)

with col_int8_2:
    st.subheader("Uniform vs Block-Wise INT8")
    st.markdown("""
    * **Per-Tensor INT8**: 1 scale factor for an entire layer. Vulnerable to outliers.
    * **Per-Channel / Block-Wise INT8**: 1 scale factor per output neuron (channel) or per block of 128 weights. Completely isolates local variance.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - INT4
# ==============================================================================
st.header("7. INT4 (4-Bit Integer Quantization)", divider="rainbow")

st.markdown("""
**INT4** allocates only 4 bits per parameter, providing exactly `2^4 = 16` discrete levels (**0.5 bytes per parameter**).
""")

col_int4_1, col_int4_2 = st.columns(2)
with col_int4_1:
    st.markdown("#### Extreme Compression Economics")
    st.markdown("""
    * **75% Memory Reduction**: An 8B model fits in **3.5 GB to 4 GB** VRAM.
    * A massive 70B model fits into **under 38 GB**, enabling execution on a single consumer RTX 3090/4090 GPU or Mac Studio!
    """)

with col_int4_2:
    st.markdown("#### The 16-Bin Challenge")
    st.markdown("""
    * Representing complex neural weights with only 16 numbers is extremely delicate.
    * Naive uniform INT4 causes severe truncation error.
    * Requires advanced algorithms: **AWQ**, **GPTQ**, or **NF4 (NormalFloat 4)** with block-wise scaling (block size 64 or 128) to preserve reasoning capability.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - WEIGHT QUANTIZATION
# ==============================================================================
st.header("8. Weight-Only Quantization (W8A16 & W4A16)", divider="rainbow")

st.markdown("""
In **Weight-Only Quantization**, model weights are stored in low precision (INT8 or INT4), 
while activation tensors remain in high precision (FP16 or BF16).
""")

st.markdown("""
#### W4A16 Execution Flow (e.g. AWQ, GPTQ, llama.cpp)
```text
Stored in VRAM:    W_int4  [4-bit integers] (Static 75% memory reduction)
Transfer to Cache: W_int4 streams over GPU memory bus (3x-4x higher effective bandwidth!)
Inside Tensor ALU: W_dequant = W_int4 * Scale  (Dequantized to FP16 in registers)
Compute:           Y = X_fp16 @ W_dequant
```
* **Why W4A16 is so popular**: During memory-bandwidth-bound decode, generation speed is limited by how fast weights stream from HBM. 
  Transferring 4-bit weights accelerates generation speed by up to **2x to 3x** with zero activation quantization complexity!
""")

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - ACTIVATION QUANTIZATION & THE OUTLIER DIMENSION PROBLEM
# ==============================================================================
st.header("9. Activation Quantization & The Outlier Dimension Problem", divider="rainbow")

st.markdown("""
Why not quantize both Weights AND Activations to 8-bit (`W8A8`) or 4-bit (`W4A4`) to unlock pure integer tensor cores?
Because of the **Emergent Outlier Dimension Phenomenon (Dettmers et al., LLM.int8(), 2022)**.
""")

col_out1, col_out2 = st.columns(2)
with col_out1:
    st.subheader("The Emergent Outlier Phenomenon")
    st.markdown("""
    * When transformer models scale beyond **6.7 Billion parameters**, a sudden phase shift occurs.
    * In hidden activation vectors, a tiny subset of dimensions (< 0.1% of channels) suddenly explode to **100x higher magnitudes** (e.g. `+50.0` vs normal `0.5`).
    * These outliers are strictly concentrated in specific coordinate dimensions across all tokens.
    """)

with col_out2:
    st.subheader("Why Outliers Break Standard INT8")
    st.markdown("""
    * If an activation vector contains `[0.1, -0.2, 0.4, +60.0]`:
    * A symmetric 8-bit quantizer must set `Scale S = 60.0 / 127 = 0.472`.
    * All normal activations (`0.1, -0.2, 0.4`) round to **zero**!
    * **Total representation collapse**: Perplexity blows up to infinity.
    * **Remedy**: **SmoothQuant** (migrating outliers into weights) or mixed-precision decomposition (**LLM.int8()**).
    """)

st.divider()

# ==============================================================================
# SECTION 10: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Symmetric & Asymmetric Uniform Quantizer in Python", divider="rainbow")

st.markdown("""
The following self-contained class implements exact symmetric and asymmetric uniform quantization operators, 
deriving scale factors and measuring reconstruction Mean Squared Error (MSE).
""")

st.code("""
import numpy as np
from typing import Tuple

class ProductionQuantizer:
    \"\"\"
    Production-grade educational Quantizer.
    \"\"\"
    def quantize_symmetric(self, tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, np.ndarray]:
        q_max = (2 ** (bits - 1)) - 1
        abs_max = float(np.max(np.abs(tensor)))
        scale = abs_max / q_max if abs_max > 0 else 1.0
        q = np.clip(np.round(tensor / scale), -q_max, q_max).astype(np.int32)
        dequant = q * scale
        return q, scale, dequant

    def quantize_asymmetric(self, tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, int, np.ndarray]:
        q_max = (2 ** bits) - 1
        t_min, t_max = float(np.min(tensor)), float(np.max(tensor))
        scale = (t_max - t_min) / q_max if t_max > t_min else 1.0
        zero_point = int(np.clip(np.round(-t_min / scale), 0, q_max))
        q = np.clip(np.round(tensor / scale) + zero_point, 0, q_max).astype(np.int32)
        dequant = (q - zero_point) * scale
        return q, scale, zero_point, dequant

    def compute_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        return float(np.mean((original - reconstructed) ** 2))
""", language="python")

st.divider()

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Quantization Error Visualizer & Dynamic Bit-Sizer", divider="rainbow")

quantizer = ProductionQuantizer()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Quantization Operator Controls")
    u_bits = st.select_slider("Target Bit Width", options=[2, 3, 4, 8, 16], value=8)
    u_mode = st.radio("Quantization Mode", ["Symmetric (Zero-Centered, Z=0)", "Asymmetric (Arbitrary Range, Zero-Point Z)"])
    u_outlier = st.checkbox("Inject Emergent Outlier Dimension (+25.0)", value=False)

    # Generate synthetic normal weights
    np.random.seed(42)
    sample_weights = np.random.normal(0, 0.5, size=200).astype(np.float32)
    if u_outlier:
        sample_weights[10] = 25.0  # Outlier dimension

    if "Symmetric" in u_mode:
        q_tensor, scale_val, dequant_tensor = quantizer.quantize_symmetric(sample_weights, bits=u_bits)
        zp_val = 0
    else:
        q_tensor, scale_val, zp_val, dequant_tensor = quantizer.quantize_asymmetric(sample_weights, bits=u_bits)

    mse_err = quantizer.compute_mse(sample_weights, dequant_tensor)

with wb_col2:
    st.subheader("Quantization Error Telemetry")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Scale Factor (S)", f"{scale_val:.6f}", f"Zero-Point (Z): {zp_val}")
    with col_m2:
        st.metric("Reconstruction MSE", f"{mse_err:.6f}", f"Bit Width: {u_bits}-bit")

    st.markdown("##### Original vs Reconstructed Tensor Slice:")
    comp_df = pd.DataFrame({
        "Index": np.arange(30),
        "Original Float": sample_weights[:30],
        "Dequantized": dequant_tensor[:30]
    })
    st.line_chart(comp_df.set_index("Index"))

    if u_outlier and u_bits <= 4:
        st.error("🚨 Notice how the single +25.0 outlier caused severe quantization error across all normal dimensions!")

st.divider()

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why is BF16 preferred over FP16 for foundation LLMs?"):
        st.write("""
        FP16 has only 5 exponent bits, capping its maximum representable value at 65,504. In deep multi-billion-parameter models, activation spikes frequently exceed 65,504, causing sudden overflow to Infinity or NaN.
        BF16 allocates 8 exponent bits—the exact same as FP32—giving it an identical dynamic range (`1e38`). 
        It completely eliminates overflow and underflow hazards without requiring loss scaling.
        """)
    with st.expander("Q2: What is the mathematical difference between Symmetric and Asymmetric quantization?"):
        st.write("""
        **Symmetric Quantization** centers the quantization grid symmetrically around zero (`Z = 0`). Scale is `max(|x|) / (2^(b-1) - 1)`. 
        Because zero-point is zero, it simplifies hardware GEMM operations by eliminating zero-point subtraction terms.
        **Asymmetric Quantization** maps an arbitrary `[min, max]` interval to `[0, 2^b - 1]`, computing an explicit integer zero-point `Z = round(-min / S)`. 
        It is more precise for skewed distributions (such as post-ReLU activations) but adds integer offset arithmetic during matrix multiplications.
        """)
    with st.expander("Q3: What is the 'Emergent Outlier Dimension' phenomenon in activations?"):
        st.write("""
        Discovered by Dettmers et al. (2022), when language models scale beyond 6.7B parameters, a tiny fraction (< 0.1%) of hidden activation dimensions systematically explode to 100x higher magnitudes (e.g. +50.0).
        Because these outliers dominate the min/max range, standard per-tensor INT8 quantization scales down normal values to zero, causing complete model collapse.
        """)

with q2:
    with st.expander("Q4: Why does Weight-Only (W4A16) quantization accelerate inference on single-user decode?"):
        st.write("""
        Single-user token decode is memory-bandwidth bound. Compute cores spend most of their time idling while waiting for weights to transfer from High Bandwidth Memory (HBM) to SRAM.
        In W4A16, weights are 4x smaller on the memory bus. 
        Transferring 4-bit weights across the bus and dequantizing them into FP16 inside on-chip registers accelerates generation speed by 2x to 3x because the memory-bandwidth bottleneck is reduced by 75%.
        """)
    with st.expander("Q5: What is Block-Wise Quantization and why is it superior to Per-Tensor?"):
        st.write("""
        Per-tensor quantization uses a single scale factor for an entire millions-of-weights matrix, making the entire layer vulnerable to clipping if one outlier exists.
        Block-wise quantization partitions the tensor into small independent blocks (e.g. 64 or 128 contiguous elements) and derives a dedicated scale factor for each block.
        Outliers are isolated to their local block, preventing precision corruption across the rest of the tensor.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Attempting W4A4 Quantization Without Advanced SmoothQuant**
    * **Failure**: Applying naive 4-bit quantization to both weights and activations simultaneously.
    * **Impact**: Total catastrophic representation collapse; perplexity exceeds 1,000 within 5 tokens due to activation outliers.
    * **Remedy**: Use W4A16 (weights 4-bit, activations 16-bit) or use SmoothQuant W8A8.
    """)
    st.error("""
    **🚨 Production Trap 2: Running FP16 on Deep Architectures (> 30B)**
    * **Failure**: Loading a 70B model in native FP16 on older GPU drivers.
    * **Impact**: Sporadic `NaN` loss and gibberish generation caused by activations exceeding 65,504 in middle transformer blocks.
    * **Remedy**: Always standardize on `torch.bfloat16`.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Omitting Block-Wise Quantization for INT4**
    * **Failure**: Quantizing a layer with per-tensor INT4.
    * **Impact**: Severe truncation error because 16 discrete bins cannot span the full dynamic range of millions of weights.
    * **Remedy**: Enforce block size `group_size = 128` in AWQ or GPTQ.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Neglecting Zero-Point Memory Overhead in Asymmetric INT4**
    * **Failure**: Using asymmetric INT4 with small block sizes (32) while storing scales and zero-points in FP32.
    * **Impact**: Quantization constants consume 1.5 bits/param, largely defeating the purpose of 4-bit compression.
    * **Remedy**: Use symmetric quantization (where `Z = 0`) or Double Quantization (QLoRA).
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: BF16 Exponent Bits"):
        st.write("How many exponent bits does BF16 allocate?")
        st.info("8 exponent bits (the exact same as FP32).")
    with st.expander("⚡ Flashcard 2: INT8 Discrete Levels"):
        st.write("How many discrete values can an INT8 integer represent?")
        st.info("256 discrete levels (`[-128, +127]`).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Symmetric Zero-Point Value"):
        st.write("What is the zero-point `Z` in symmetric quantization?")
        st.info("`Z = 0` (strictly zero-centered).")
    with fc_col2:
        with st.expander("⚡ Flashcard 4: Outlier Emergence Threshold"):
            st.write("At what model scale do activation outliers emerge systematically?")
            st.info("Around 6.7 Billion parameters (Dettmers et al., 2022).")

st.divider()

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Quantization Basics before advancing to Lesson 17.2 (Quantization Methods):
""")

chk1 = st.checkbox("I can derive scale S and zero-point Z for symmetric and asymmetric uniform quantization.", value=True)
chk2 = st.checkbox("I understand why BF16 prevents the overflow and underflow issues of FP16.", value=True)
chk3 = st.checkbox("I know why INT8 quantization causes virtually zero accuracy loss on foundation models.", value=True)
chk4 = st.checkbox("I understand why Weight-Only (W4A16) accelerates memory-bound decode by up to 3x.", value=True)
chk5 = st.checkbox("I can explain the Emergent Outlier Dimension problem in activations at 6.7B+ scale.", value=True)
chk6 = st.checkbox("I know why block-wise quantization isolates outliers and preserves precision.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 17.2: Quantization Methods.")
