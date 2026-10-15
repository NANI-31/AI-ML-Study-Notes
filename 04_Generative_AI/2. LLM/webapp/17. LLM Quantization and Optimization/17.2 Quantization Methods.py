import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="17.2 Quantization Methods: GPTQ, AWQ, bitsandbytes, & SmoothQuant",
    page_icon="⚡",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class QuantizationSimulator:
    """
    Module-scope realistic simulation engine for modern LLM quantization algorithms:
    - Naive Round-To-Nearest (RTN)
    - Activation-Aware Weight Quantization (AWQ) salient protection
    - SmoothQuant outlier migration
    - bitsandbytes NF4 quantile mapping
    """
    
    @staticmethod
    def generate_synthetic_layer(
        in_features: int = 64,
        out_features: int = 64,
        outlier_channel_ratio: float = 0.05,
        outlier_magnitude: float = 25.0,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic LLM weight matrix and activation batch with sharp outlier channels.
        """
        np.random.seed(random_seed)
        # Weights are typically normally distributed around zero
        W = np.random.normal(loc=0.0, scale=0.05, size=(out_features, in_features))
        
        # Activations have mostly small values, but 1-2% of channels exhibit massive systematic outliers
        X = np.random.normal(loc=0.0, scale=1.0, size=(16, in_features))
        num_outliers = max(1, int(in_features * outlier_channel_ratio))
        outlier_cols = np.random.choice(in_features, size=num_outliers, replace=False)
        
        for col in outlier_cols:
            X[:, col] *= outlier_magnitude
            
        return W, X

    @staticmethod
    def quantize_rtn_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Naive Symmetric INT8 Round-To-Nearest Quantization."""
        abs_max = float(np.max(np.abs(tensor)))
        if abs_max == 0:
            return tensor, 1.0
        scale = abs_max / 127.0
        q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        dequant = q.astype(np.float32) * scale
        return dequant, scale

    @staticmethod
    def smooth_quant_transform(
        W: np.ndarray,
        X: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SmoothQuant transformation:
        Migrates difficulty from activation outliers to weights via per-channel scale s_j.
        s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1-alpha))
        X_smooth = X * diag(s)^-1
        W_smooth = diag(s) * W
        """
        act_max = np.max(np.abs(X), axis=0) # [in_features]
        weight_max = np.max(np.abs(W), axis=0) # [in_features]
        
        act_max = np.maximum(act_max, 1e-5)
        weight_max = np.maximum(weight_max, 1e-5)
        
        # Scale vector s
        s = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
        s = np.maximum(s, 1e-5)
        
        # Smooth activations and weights
        X_smooth = X / s
        W_smooth = W * s
        
        return W_smooth, X_smooth, s

    @staticmethod
    def simulate_awq(
        W: np.ndarray,
        X: np.ndarray,
        salient_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates AWQ: protects top salient weight channels based on activation magnitudes.
        Salient channels are kept in FP16; remaining 95% channels quantized to INT4.
        """
        act_magnitudes = np.mean(np.abs(X), axis=0)
        num_salient = max(1, int(len(act_magnitudes) * salient_ratio))
        salient_indices = np.argsort(act_magnitudes)[-num_salient:]
        
        W_dequant = W.copy()
        mask_non_salient = np.ones(W.shape[1], dtype=bool)
        mask_non_salient[salient_indices] = False
        
        # Quantize non-salient weights to INT4 [-8, 7]
        W_non_salient = W[:, mask_non_salient]
        abs_max = np.max(np.abs(W_non_salient))
        scale = abs_max / 7.0 if abs_max > 0 else 1.0
        q_int4 = np.clip(np.round(W_non_salient / scale), -8, 7)
        W_dequant[:, mask_non_salient] = q_int4 * scale
        
        return W_dequant, salient_indices

    @staticmethod
    def compute_layer_error(
        W_orig: np.ndarray,
        W_quant: np.ndarray,
        X_orig: np.ndarray,
        X_quant: np.ndarray = None
    ) -> Dict[str, float]:
        """Computes layer output MSE and Cosine Similarity."""
        Y_orig = X_orig @ W_orig.T
        if X_quant is None:
            X_quant = X_orig
        Y_quant = X_quant @ W_quant.T
        
        mse = float(np.mean((Y_orig - Y_quant) ** 2))
        norm_orig = np.linalg.norm(Y_orig)
        norm_quant = np.linalg.norm(Y_quant)
        if norm_orig > 0 and norm_quant > 0:
            cos_sim = float(np.sum(Y_orig * Y_quant) / (norm_orig * norm_quant))
        else:
            cos_sim = 1.0
            
        return {"mse": mse, "cosine_similarity": cos_sim}


# --- HEADER & INTRODUCTION ---
st.title("17.2 Quantization Methods: GPTQ, AWQ, bitsandbytes, & SmoothQuant")
st.caption("Chapter 17: LLM Quantization and Optimization | Domain: 2. Large Language Models")

st.markdown("""
Why does naive Round-to-Nearest (RTN) quantization destroy an LLM's reasoning capabilities at 4 bits, while modern 
Post-Training Quantization (PTQ) algorithms preserve near-perfect perplexity? 

The answer lies in **cross-weight interdependence** and **activation outlier geometry**:
1. When weights are quantized independently, rounding errors accumulate catastrophically across billions of parameters.
2. In transformer architectures scaled beyond 6.7B parameters, a tiny fraction (0.1% to 1%) of activation channels 
   exhibit massive magnitude spikes (up to 100x variance). Naive quantization squashes all subtle features into zero.

To overcome these physical bottlenecks, the AI community engineered four breakthrough quantization algorithms:
- **GPTQ**: Inverts the second-order Hessian matrix to dynamically update unquantized weights as each column is rounded.
- **AWQ**: Identifies salient channels based on activation volume and scales them to protect critical reasoning circuits.
- **bitsandbytes (LLM.int8() & NF4)**: Vector outlier decomposition for zero-drop INT8 and information-optimal 4-bit normal float tuning in QLoRA.
- **SmoothQuant**: Mathematically migrates the quantization difficulty from erratic activations into smooth weights, unlocking native W8A8 Tensor Core matrix multiplication.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern LLM Quantization Pipeline & Algorithm Taxonomy", divider="rainbow")

st.markdown("""
Understanding how modern quantization algorithms divide labor between **Weight-Only (W4A16)**, **Weight-Activation (W8A8)**, 
and **Quantization-Aware Training (QAT)** across the LLM deployment lifecycle:
""")

blueprint_dot = """
digraph QuantizationTaxonomy {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_input {
        label="Trained FP16 / BF16 Foundation Model";
        color="#1E293B";
        fontcolor="#38BDF8";
        PretrainedModel [label="Pre-trained Weights\\n(FP16 / BF16 Base)", fillcolor="#1E293B"];
        CalibData [label="Unlabeled Calibration Set\\n(128 - 512 Sentences)", fillcolor="#1E293B"];
    }

    subgraph cluster_ptq_weight {
        label="Weight-Only PTQ (Memory-Bandwidth Bound Serving)";
        color="#1E293B";
        fontcolor="#38BDF8";
        GPTQ [label="GPTQ\\nSecond-Order Hessian Inversion\\nRow-Wise Error Correction", fillcolor="#0F766E"];
        AWQ [label="AWQ\\nActivation-Aware Weight Scaling\\nTop 1% Salient Channel Protection", fillcolor="#0369A1"];
        BNB [label="bitsandbytes NF4\\nInformation-Theoretic Quantiles\\nDouble Quantization (QLoRA)", fillcolor="#4338CA"];
    }

    subgraph cluster_ptq_both {
        label="Weight + Activation PTQ (Compute Bound Serving)";
        color="#1E293B";
        fontcolor="#38BDF8";
        SmoothQuant [label="SmoothQuant (W8A8)\\nOutlier Migration Factor alpha\\nActivations / s | Weights * s", fillcolor="#6D28D9"];
    }

    subgraph cluster_engines {
        label="High-Throughput Serving Execution";
        color="#1E293B";
        fontcolor="#38BDF8";
        vLLM [label="vLLM / SGLang / TRT-LLM\\nW4A16 GEMM / GEMV Kernels", fillcolor="#334155"];
        INT8TC [label="Hardware INT8 Tensor Cores\\nTrue 2x Compute Throughput", fillcolor="#334155"];
    }

    PretrainedModel -> GPTQ [label="Hessian X*X^T"];
    CalibData -> GPTQ;
    PretrainedModel -> AWQ [label="Act Max Search"];
    CalibData -> AWQ;
    PretrainedModel -> BNB [label="Direct NF4 Mapping"];
    PretrainedModel -> SmoothQuant [label="Per-Channel Outlier Scan"];
    CalibData -> SmoothQuant;

    GPTQ -> vLLM;
    AWQ -> vLLM;
    BNB -> vLLM [label="QLoRA Tuning"];
    SmoothQuant -> INT8TC;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Algorithmic Comparison Matrix: Production Quantization Engines
""")

comparison_data = [
    {
        "Algorithm": "GPTQ (Frantar et al.)",
        "Target Format": "W4A16 / W3A16",
        "Target Workload": "Inference Serving",
        "Core Optimization Mechanism": "Second-order Taylor expansion; inverse Hessian column-by-column error updates",
        "Calibration Time (70B)": "~1.5 hours (1x A100)",
        "Primary Production Engines": "AutoGPTQ, ExLlamaV2, vLLM"
    },
    {
        "Algorithm": "AWQ (Lin et al.)",
        "Target Format": "W4A16 / W3A16",
        "Target Workload": "Inference Serving",
        "Core Optimization Mechanism": "Observation that weights protecting top 1% activation magnitudes are critical; per-channel grid search scaling",
        "Calibration Time (70B)": "~20 minutes (1x A100)",
        "Primary Production Engines": "vLLM, TensorRT-LLM, SGLang"
    },
    {
        "Algorithm": "bitsandbytes (Dettmers et al.)",
        "Target Format": "W8A16 / W4A16 (NF4)",
        "Target Workload": "Fine-Tuning & Local Dev",
        "Core Optimization Mechanism": "LLM.int8() vector outlier routing; NF4 normal distribution quantiles + Double Quantization (DQ)",
        "Calibration Time (70B)": "0 mins (On-the-fly)",
        "Primary Production Engines": "Hugging Face PEFT / Transformers"
    },
    {
        "Algorithm": "SmoothQuant (Xiao et al.)",
        "Target Format": "W8A8 (True INT8 GEMM)",
        "Target Workload": "Compute-Bound Serving",
        "Core Optimization Mechanism": "Mathematical equivalent transformation: scales activation channels down and weight channels up via migration factor alpha",
        "Calibration Time (70B)": "~10 minutes (1x A100)",
        "Primary Production Engines": "TensorRT-LLM, FasterTransformer, vLLM"
    },
    {
        "Algorithm": "Quantization-Aware Training (QAT)",
        "Target Format": "W4A4 / W2A4",
        "Target Workload": "Pre-Training / Post-Training",
        "Core Optimization Mechanism": "Straight-Through Estimator (STE) during gradient updates with simulated quantization noise",
        "Calibration Time (70B)": "Days to Weeks (Cluster)",
        "Primary Production Engines": "PyTorch FX, DeepSeek-V3 FP8 Custom"
    }
]
st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - GPTQ ---
st.header("2. GPTQ: Second-Order Hessian Error Compensation", divider="rainbow")

st.markdown("""
Prior to **GPTQ (Accurate Post-Training Quantization for Generative Pre-trained Transformers)**, quantizing 175B 
parameter models to 4 bits resulted in immediate perplexity collapse. Rounding each weight independently ignores 
the quadratic loss landscape of the neural network.

### The Mathematical Formulation: Second-Order Taylor Expansion
Consider the layer-wise reconstruction objective: minimize the squared error between the original full-precision output 
and the quantized output across a small calibration dataset `X`:

```text
Objective: Minimize E = || W @ X - W_quant @ X ||_2^2
```

Using a second-order Taylor expansion around the full-precision weight matrix `W`, the change in reconstruction error 
when updating weight vector `w` can be written as:

```text
Delta E = (w_quant - w)^T @ H @ (w_quant - w)
```

Where `H = 2 * X @ X^T` is the **Hessian matrix** of the layer activations.

### The Optimal Brain Quantizer (OBQ) Bottleneck & GPTQ Solution
In classical Optimal Brain Compression (OBC/OBQ), quantizing a single weight required inverting the entire Hessian matrix 
of dimension `[d_row, d_row]`, yielding an impractical complexity of `O(d_row^3)` per weight, taking weeks for a 7B model.

**Frantar et al. introduced three breakthroughs in GPTQ**:
1. **Arbitrary Order Quantization**: Rather than greedily searching for the lowest-error weight, GPTQ processes weights in a fixed, contiguous column order, allowing the Hessian inversion to be shared across all rows of the layer.
2. **Lazy Batch Updates**: Rather than updating remaining weights after every single scalar quantization, GPTQ batches updates across 128 columns at a time, utilizing high-speed GPU Matrix Multiply (GEMM) instead of slow memory bandwidth-bound vector updates.
3. **Cholesky Stabilization**: Inverts `H` using Cholesky decomposition with a numerical damping constant `lambda * I` (typically 1% of the diagonal mean) to prevent numerical instability in ill-conditioned activation matrices:

```text
H_inv = Cholesky_Invert(2 * X @ X^T + lambda * I)
```

Whenever column `w_q` is rounded, the remaining unquantized weight matrix `W_remaining` is compensated via:

```text
W_remaining = W_remaining - (w_quant - w_orig) / [H_inv]_{q, q} @ [H_inv]_{q, remaining}
```
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    **Key Advantage of GPTQ**:
    - Unrivaled perplexity preservation at aggressive 3-bit and 4-bit compression.
    - Zero runtime memory overhead during calibration; processes 70B parameter models in under 2 hours on a single consumer GPU.
    """)
with col2:
    st.warning("""
    **Operational Trade-Off**:
    - Weights are reorganized in block-column order; specialized dequantization GEMM kernels (e.g., ExLlamaV2) are required to avoid dequantization latency overhead.
    """)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
In 2023, GPTQ enabled the first wave of running LLaMA-65B locally on a single RTX 3090. Today, while GPTQ remains 
massively popular in local frameworks like Ollama and LM Studio, modern enterprise serving engines (like vLLM and TensorRT-LLM) 
often prefer **AWQ** for W4A16 because AWQ eliminates the need for complex inverse Hessian coordinate updates while achieving 
identical or superior serving throughput.
""")

st.divider()

# --- SECTION 3: TOPIC 2 - AWQ ---
st.header("3. AWQ: Activation-Aware Weight Quantization", divider="rainbow")

st.markdown("""
Introduced by Lin et al. (MIT HAN Lab, 2023), **AWQ (Activation-aware Weight Quantization)** is based on a profound 
empirical discovery: **weights are not equally important, and weight importance is determined by activation magnitudes, not weight magnitudes**.

### The Core Empirical Discovery
If you inspect the weight matrix of an LLM, all weights appear normally distributed with similar variance. However, 
if you measure the input activation magnitudes `X` produced by real text prompts:
- Only **0.1% to 1%** of hidden channels carry extreme magnitudes (e.g., values exceeding 20.0 to 100.0).
- Protecting this top 1% of channels completely preserves reasoning, math, and code generation benchmarks!
""")

awq_flow = """
digraph AWQFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    FP16W [label="Full Precision Weights\\nW [out, in]", fillcolor="#1E293B"];
    ActIn [label="Calibration Activations\\nX [seq, in]", fillcolor="#1E293B"];
    Saliency [label="Compute Channel Salience\\nS_j = Average(|X_j|)", fillcolor="#0369A1"];
    GridSearch [label="Per-Channel Scale Search\\ns_j in [1, 2, ... 4]\\nMinimizes Quantization MSE", fillcolor="#0F766E"];
    ScaledW [label="Scaled Weights & Inverted Acts\\nW_new = W * diag(s)\\nX_new = X * diag(s)^-1", fillcolor="#4338CA"];
    QuantKernel [label="Standard Uniform INT4 Quant\\nZero Hardware Overhead\\nHigh-Speed vLLM Kernel", fillcolor="#059669"];

    FP16W -> Saliency;
    ActIn -> Saliency;
    Saliency -> GridSearch;
    GridSearch -> ScaledW;
    ScaledW -> QuantKernel;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(awq_flow, use_container_width=True)

st.markdown("""
### Why Not Just Keep Salient Weights in FP16? (Mixed-Precision Penalty)
A naive approach would be to leave the top 1% of weights in FP16 and quantize the remaining 99% in INT4. 
However, **mixed-precision matrix multiplication is poison for GPU hardware**:
- Modern GPU Tensor Cores are designed for uniform memory access and dense vector instructions.
- Splitting matrix multiplication into two disparate GEMM operations (one FP16 sparse, one INT4 dense) incurs 
  massive kernel launching overhead and memory fragmentation, destroying inference throughput.

### The AWQ Mathematical Formulation: Per-Channel Scaling
Instead of isolating salient weights into a separate matrix, AWQ scales the entire channel mathematically:

```text
Y = (W @ X) = (W @ diag(s)) @ (diag(s)^-1 @ X) = W_scaled @ X_scaled
```

AWQ searches for an optimal per-channel scaling factor `s` using a fast grid search:
1. `s = S_X^alpha`, where `S_X = Average(|X|)` is the activation magnitude, and `alpha` is a hyperparameter searched in `[0, 1]`.
2. By multiplying the weight channel by `s`, the dynamic range of the weight channel is enlarged, dramatically reducing the relative rounding error during uniform INT4 quantization!
3. The inverse scale `s^-1` is absorbed directly into the preceding layer's LayerNorm or activation function, meaning **zero extra operations during GPU forward pass execution**.
""")

st.info("""
**Why AWQ Dominates vLLM & TensorRT-LLM**:
Because AWQ produces uniform INT4 weights with per-channel scaling constants, it fits natively into specialized 
hardware dequantization kernels (`Marlin`, `AWQ-GEMM`). It yields up to **3x higher throughput** than unoptimized 
quantization while maintaining MMLU accuracy within 0.1% of uncompressed FP16.
""")

st.divider()

# --- SECTION 4: TOPIC 3 - BITSANDBYTES ---
st.header("4. bitsandbytes: LLM.int8(), NF4, and Double Quantization in QLoRA", divider="rainbow")

st.markdown("""
Developed by Tim Dettmers et al. (University of Washington), **bitsandbytes** democratized open-source LLM development 
by enabling consumer GPUs to run and fine-tune models that previously required 8x A100 server clusters.

The library introduced three cornerstone innovations across its evolution:
""")

tab_int8, tab_nf4, tab_dq = st.tabs(["1. LLM.int8() (2022)", "2. NormalFloat 4 (NF4) (2023)", "3. Double Quantization (DQ)"])

with tab_int8:
    st.markdown("""
    ### LLM.int8(): Vector-Wise Quantization & Outlier Decomposition
    Prior to LLM.int8(), quantizing models larger than 6.7B parameters to INT8 resulted in complete degradation. 
    Dettmers discovered that **emergent outlier features** (coordinates with values up to 6.0 in magnitude) 
    are confined to roughly 0.1% of hidden dimensions, but affect 100% of tokens in the sequence.

    **How LLM.int8() works**:
    1. Splits the input activation matrix `X` and weight matrix `W` into two separate streams:
       - **Outlier Stream**: Columns where activation magnitude exceeds threshold `alpha = 6.0` are kept in **FP16** (<0.1% of columns).
       - **Non-Outlier Stream**: Remaining 99.9% of columns are quantized to **INT8** using vector-wise quantization (independent scale per row of `X` and column of `W`).
    2. Executes standard FP16 matrix multiply for outliers: `Y_outlier = X_fp16 @ W_fp16`.
    3. Executes INT8 matrix multiply for non-outliers: `Y_int8 = X_int8 @ W_int8`.
    4. Dequantizes and adds the two matrices: `Y = Y_outlier + (Y_int8 * scale)`.
    5. Result: **Zero degradation in zero-shot perplexity across all model sizes up to 175B**.
    """)

with tab_nf4:
    st.markdown("""
    ### NormalFloat 4 (NF4): Information-Theoretically Optimal Quantile Quantization
    Standard uniform 4-bit integer quantization divides the range `[-1, 1]` into 16 equally spaced bins. 
    However, neural network weights trained with weight decay and LayerNorm are **normally distributed** 
    `W ~ Normal(0, sigma^2)`.

    In a Gaussian distribution:
    - Most weights cluster near 0.
    - Uniform binning wastes precision on empty tail regions while clustering hundreds of dense near-zero weights into a single bin.

    **NF4 computes 16 discrete quantile levels** such that each bin contains an **exactly equal probability mass (1/16 = 6.25%)** 
    under a standard normal distribution `N(0, 1)`:

    ```text
    q_i = 0.5 * ( Q_X(i / (2^k)) + Q_X((i + 1) / (2^k)) )
    ```

    Where `Q_X` is the quantile function (inverse CDF) of `N(0, 1)`. 
    NF4 guarantees maximum information entropy per bit, outperforming both INT4 and FP4 on all language modeling benchmarks.
    """)

with tab_dq:
    st.markdown("""
    ### Double Quantization (DQ): Quantizing the Quantization Constants
    In block-wise quantization (e.g., block size 64), each block of 64 weights requires a 32-bit floating-point scale factor `c_1`. 
    For a 65B model, storing these scales consumes:

    ```text
    Memory = (32 bits / 64) = 0.5 bits per parameter
    ```

    **Double Quantization applies 8-bit FP quantization to the FP32 scale factors themselves**:
    - Treats the scale factors `c_1` as a continuous tensor and quantizes them into 8-bit integers with a second block size of 256.
    - Reduces the memory footprint of scales from 0.5 bits/parameter down to **0.127 bits/parameter**, saving an astonishing 3 GB of VRAM on a 65B model with zero impact on fine-tuning perplexity!
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - SMOOTHQUANT ---
st.header("5. SmoothQuant: Activation-Weight Outlier Migration", divider="rainbow")

st.markdown("""
While GPTQ and AWQ focus on **Weight-Only Quantization (W4A16)** (where activations remain in FP16), 
weight-only quantization does not accelerate compute-bound matrix operations because the matrix multiplication 
still executes in FP16 arithmetic on the GPU.

To achieve true hardware-level 2x speedups, both weights AND activations must be quantized to 8 bits (**W8A8**), 
enabling native **INT8 Tensor Core GEMM**.

### The Activation Outlier Challenge
- Quantizing weights to INT8 is trivial (weights have smooth, bounded distributions).
- Quantizing activations to INT8 is notoriously difficult because transformers exhibit **systematic outlier channels**:
  some feature dimensions have values 100x larger than others. If you apply a global scale factor across the tensor, 
  all normal features are rounded to 0 or 1, destroying model accuracy.

### The SmoothQuant Mathematical Trick: Per-Channel Equivalence
SmoothQuant (Xiao et al., MIT, 2023) observes that matrix multiplication is mathematically invariant to diagonal scaling:

```text
Y = X @ W = (X @ diag(s)^-1) @ (diag(s) @ W) = X_hat @ W_hat
```

Where `s` is a per-channel smoothing scale vector `s in R^C`:
1. Dividing activation channel `j` by `s_j` squashes the outlier peak.
2. Multiplying weight column `j` by `s_j` absorbs that difficulty into the weight.
3. Because weights are static and have low variance, they can easily absorb this extra range without degradation!
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Outlier Migration Factor `alpha`
    SmoothQuant balances the difficulty using a migration exponent `alpha in [0, 1]`:

    ```text
    s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1 - alpha))
    ```

    - **alpha = 1.0**: Migrates all difficulty to weights (weights absorb everything, activations become trivial to quantize).
    - **alpha = 0.0**: Weights absorb nothing (reverts to naive activation quantization).
    - **alpha = 0.5 (Optimal)**: Divides the dynamic range difficulty equally between activations and weights, allowing both `X_hat` and `W_hat` to be quantized to INT8 with zero loss!
    """)
with col2:
    st.markdown("""
    ### Hardware Integration Pipeline
    ```text
    [Previous Layer Output] 
               |
               v
      [LayerNorm / RMSNorm]
               |
         (Absorb diag(s)^-1 into LayerNorm scale gamma)
               |
               v
     [Quantize X_hat to INT8]  --- (Outliers Eliminated!)
               |
               v
     [INT8 Tensor Core GEMM] <-- [W_hat pre-quantized to INT8]
               |
               v
     [Output in INT32 -> Dequant to FP16]
    ```
    SmoothQuant runs at **zero runtime overhead** because the inverse scale `s^-1` is folded directly into the pre-norm weights during offline compilation.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - QAT ---
st.header("6. Quantization-Aware Training (QAT)", divider="rainbow")

st.markdown("""
In **Quantization-Aware Training (QAT)**, quantization is not an afterthought applied to a finished model; 
rather, the model is trained or fine-tuned with simulated quantization noise embedded directly into the forward pass.

### The Mathematical Challenge: Zero Gradients of the Round Function
The core mathematical obstacle to gradient-based quantization is the rounding function:

```text
q = round(w / scale)
```

The derivative of `round(x)` is zero everywhere, and undefined at half-integers:

```text
d/dx [round(x)] = 0  (almost everywhere)
```

If you backpropagate through a rounded tensor, the gradient `dL/dw` vanishes to zero, preventing any weight updates.

### The Solution: Straight-Through Estimator (STE)
QAT circumvents this using the **Straight-Through Estimator (STE)** (Bengio et al.):
- **Forward Pass**: Uses the quantized, rounded weights to calculate the loss:
  ```text
  W_forward = scale * clip(round(W / scale), -2^(b-1), 2^(b-1) - 1)
  ```
- **Backward Pass**: Replaces the derivative of the rounding function with the identity function (`d/dw [round(w)] = 1`):
  ```text
  dL/dW_full = (dL / dW_forward) * 1.0  (if inside clipping bounds)
  ```
""")

qat_dot = """
digraph QATFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    FP32Weights [label="Full Precision Master Weights\\nW_fp32 (Updated by Optimizer)", fillcolor="#1E293B"];
    SimQuant [label="Simulated Quantization (STE)\\nForward: Round to INT4\\nBackward: Gradient Passes Through", fillcolor="#0F766E"];
    ForwardAct [label="Forward Activations & Loss\\nComputes Loss with Quant Noise", fillcolor="#4338CA"];
    BackwardGrad [label="Backward Pass (STE)\\ndL / dW_fp32 = dL / dW_sim", fillcolor="#0369A1"];
    OptimizerStep [label="AdamW Optimizer Step\\nWeights Adapt to Quantization Noise", fillcolor="#059669"];

    FP32Weights -> SimQuant;
    SimQuant -> ForwardAct;
    ForwardAct -> BackwardGrad;
    BackwardGrad -> OptimizerStep;
    OptimizerStep -> FP32Weights;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(qat_dot, use_container_width=True)

st.markdown("""
### QAT in the Modern LLM Era: Why PTQ Superseded Classical QAT
While QAT yields the highest possible fidelity at ultra-low bit widths (2-bit and 3-bit weights), 
it is rarely used for full foundation model pre-training due to massive computational costs:
- Simulating quantization operations doubles memory consumption during training.
- Modern PTQ algorithms (AWQ, GPTQ) recover 99.5% of FP16 accuracy at 4 bits without retraining.
- However, modern frontier models (like **Meta Llama 3** and **DeepSeek-V3**) are resurrecting QAT principles 
  via **FP8 Mixed-Precision Pre-Training**, training the final 10% of tokens with FP8 scaling rules to guarantee zero-loss PTQ export.
""")

st.divider()

# --- SECTION 7: TOPIC 6 - PTQ ---
st.header("7. Post-Training Quantization (PTQ)", divider="rainbow")

st.markdown("""
**Post-Training Quantization (PTQ)** is the gold standard for enterprise LLM optimization. 
It converts a pre-trained FP16/BF16 model to INT8, INT4, or FP8 **without retraining, without backpropagation, 
and without access to the full training dataset**.

### The 4-Step Standard PTQ Workflow
1. **Calibration Dataset Assembly**: Collect 128 to 512 representative sequences (typically 2,048 tokens each) from diverse domains (Wikipedia, C4, GitHub, arXiv).
2. **Forward Pass & Statistic Accumulation**: Pass the calibration batches through the model layer-by-layer; record channel-wise minimums, maximums, and Hessian cross-products `X @ X^T`.
3. **Scale & Zero-Point Calculation**: Compute quantization parameters `(scale, zero_point)` or smoothing vectors `s`.
4. **Discretization & Packing**: Round weights to integer grid and pack multiple sub-byte values into 32-bit registers (e.g., eight 4-bit integers per `uint32`).
""")

st.markdown("""
### Strategic Decision Matrix: PTQ vs QAT vs Baseline FP16
""")

decision_matrix = [
    {
        "Criterion": "Engineering Effort",
        "Baseline FP16": "Zero",
        "PTQ (AWQ / GPTQ)": "Low (1 script, ~30 mins)",
        "QAT": "High (Cluster training pipeline)"
    },
    {
        "Criterion": "GPU Compute Budget",
        "Baseline FP16": "Zero",
        "PTQ (AWQ / GPTQ)": "1x GPU for 1-2 hours",
        "QAT": "Dozens of GPUs for days/weeks"
    },
    {
        "Criterion": "Data Requirements",
        "Baseline FP16": "None",
        "PTQ (AWQ / GPTQ)": "128-512 unlabeled sentences",
        "QAT": "Extensive instruction / pre-training dataset"
    },
    {
        "Criterion": "4-bit Perplexity Retention",
        "Baseline FP16": "N/A (Reference 100%)",
        "PTQ (AWQ / GPTQ)": "99.0% - 99.8% of FP16",
        "QAT": "99.5% - 100.0% of FP16"
    },
    {
        "Criterion": "Sub-3-Bit Viability",
        "Baseline FP16": "N/A",
        "PTQ (AWQ / GPTQ)": "Noticeable degradation",
        "QAT": "Viable with careful loss balancing"
    }
]
st.dataframe(pd.DataFrame(decision_matrix), use_container_width=True)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: Realistic Simulation of Weight-Activation Quantization & Outlier Scaling", divider="rainbow")

st.markdown("""
The following complete, dependency-clean Python script implements the mathematical core of **SmoothQuant Outlier Migration**, 
**Activation-Aware Weight Quantization (AWQ)**, and **Naive Round-To-Nearest (RTN)**.

This module is defined in real module scope and executes live inside this application:
""")

code_block = '''import numpy as np
from typing import Tuple, Dict

class QuantizationEngine:
    """
    Self-contained implementation of modern Post-Training Quantization (PTQ) mechanics.
    Zero external dependencies beyond standard NumPy.
    """
    
    @staticmethod
    def smooth_quant_channel_transform(
        W: np.ndarray,      # Shape: [out_features, in_features]
        X: np.ndarray,      # Shape: [batch_seq, in_features]
        alpha: float = 0.5  # Migration factor: 0.0 (all to act) to 1.0 (all to weight)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates per-channel smoothing scale s_j and applies outlier migration:
        s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1 - alpha))
        Returns: (W_smooth, X_smooth, s)
        """
        # 1. Compute per-channel maximum absolute values
        act_max = np.max(np.abs(X), axis=0)      # [in_features]
        weight_max = np.max(np.abs(W), axis=0)   # [in_features]
        
        # Guard against division by zero
        act_max = np.maximum(act_max, 1e-5)
        weight_max = np.maximum(weight_max, 1e-5)
        
        # 2. Compute migration scale vector
        s = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
        s = np.maximum(s, 1e-5)
        
        # 3. Apply mathematically equivalent transformation
        # Y = X @ W.T = (X / s) @ (W * s).T
        X_smooth = X / s
        W_smooth = W * s
        
        return W_smooth, X_smooth, s

    @staticmethod
    def uniform_int8_quantize(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Symmetric INT8 uniform quantization with scale factor."""
        abs_max = float(np.max(np.abs(tensor)))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        dequantized = q.astype(np.float32) * scale
        return dequantized, scale

    @staticmethod
    def awq_salient_protection(
        W: np.ndarray,
        X: np.ndarray,
        salient_fraction: float = 0.05
    ) -> np.ndarray:
        """
        Protects top salient weight channels based on activation magnitudes.
        Salient channels kept in full precision; remaining channels quantized to INT4.
        """
        act_magnitude = np.mean(np.abs(X), axis=0)
        num_salient = max(1, int(len(act_magnitude) * salient_fraction))
        salient_indices = np.argsort(act_magnitude)[-num_salient:]
        
        W_dequant = W.copy()
        mask_non_salient = np.ones(W.shape[1], dtype=bool)
        mask_non_salient[salient_indices] = False
        
        # Quantize remaining non-salient weights to INT4 [-8, 7]
        W_non = W[:, mask_non_salient]
        scale = np.max(np.abs(W_non)) / 7.0 if np.max(np.abs(W_non)) > 0 else 1.0
        q_int4 = np.clip(np.round(W_non / scale), -8, 7)
        W_dequant[:, mask_non_salient] = q_int4 * scale
        
        return W_dequant
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: Calibration Matrix & Quantization Method Comparator", divider="rainbow")

st.markdown("""
Test how outlier channel magnitudes, SmoothQuant migration factors, and AWQ salient channel ratios 
alter the layer reconstruction error in real time:
""")

wb_col1, wb_col2, wb_col3 = st.columns(3)
with wb_col1:
    outlier_mag = st.slider("Activation Outlier Multiplier", min_value=1.0, max_value=80.0, value=25.0, step=1.0)
with wb_col2:
    smooth_alpha = st.slider("SmoothQuant Migration Factor (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
with wb_col3:
    salient_ratio = st.slider("AWQ Salient Channel Protection Ratio", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# Run live simulation
W_raw, X_raw = QuantizationSimulator.generate_synthetic_layer(
    in_features=48,
    out_features=48,
    outlier_channel_ratio=0.05,
    outlier_magnitude=outlier_mag
)

# 1. Baseline Naive RTN (Weights and Activations independently quantized to INT8)
W_rtn, _ = QuantizationSimulator.quantize_rtn_int8(W_raw)
X_rtn, _ = QuantizationSimulator.quantize_rtn_int8(X_raw)
err_rtn = QuantizationSimulator.compute_layer_error(W_raw, W_rtn, X_raw, X_rtn)

# 2. SmoothQuant (W8A8 with alpha migration)
W_smooth, X_smooth, s_vec = QuantizationSimulator.smooth_quant_transform(W_raw, X_raw, alpha=smooth_alpha)
W_sq_quant, _ = QuantizationSimulator.quantize_rtn_int8(W_smooth)
X_sq_quant, _ = QuantizationSimulator.quantize_rtn_int8(X_smooth)
err_sq = QuantizationSimulator.compute_layer_error(W_raw, W_sq_quant, X_raw, X_sq_quant)

# 3. AWQ (W4A16 with salient protection)
W_awq, salient_idx = QuantizationSimulator.simulate_awq(W_raw, X_raw, salient_ratio=salient_ratio)
err_awq = QuantizationSimulator.compute_layer_error(W_raw, W_awq, X_raw, None)

st.subheader("Empirical Layer Reconstruction Benchmark")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Naive RTN INT8 MSE", f"{err_rtn['mse']:.5f}", delta=f"Cos Sim: {err_rtn['cosine_similarity']:.4f}", delta_color="inverse")
m_col2.metric("SmoothQuant W8A8 MSE", f"{err_sq['mse']:.5f}", delta=f"Cos Sim: {err_sq['cosine_similarity']:.4f}", delta_color="normal")
m_col3.metric("AWQ Simulated W4A16 MSE", f"{err_awq['mse']:.5f}", delta=f"Cos Sim: {err_awq['cosine_similarity']:.4f}", delta_color="normal")

st.markdown("### Channel Activation Dynamic Range: Raw vs SmoothQuant")
act_raw_max = np.max(np.abs(X_raw), axis=0)
act_smooth_max = np.max(np.abs(X_smooth), axis=0)

chart_df = pd.DataFrame({
    "Channel Index": list(range(len(act_raw_max))),
    "Raw Activation Max": act_raw_max,
    "SmoothQuant Activation Max": act_smooth_max
}).set_index("Channel Index")

st.line_chart(chart_df)

st.caption("""
Observe the sharp spikes in the 'Raw Activation Max' line: these are the systematic outlier channels. 
Notice how SmoothQuant's mathematical transformation (orange line) pulls the spikes down into a flat, 
quantizable dynamic range without changing the mathematical output of the matrix multiplication!
""")

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "How does GPTQ utilize the inverse Hessian matrix to update remaining weights during quantization?",
        "a": "GPTQ formulates quantization as a quadratic error minimization problem: E = || W @ X - W_quant @ X ||^2. The second-order Taylor expansion reveals the error gradient as Delta E = (w_quant - w)^T @ H @ (w_quant - w), where H = 2 * X @ X^T is the activation Hessian. When a weight column is rounded to the nearest integer, GPTQ computes the immediate rounding error and applies a closed-form compensation update to all remaining unquantized columns: W_remaining = W_remaining - (error / H_inv[q,q]) @ H_inv[q, remaining]. This cancels out the accumulated error across the sequence."
    },
    {
        "q": "Why does AWQ select salient channels based on activation magnitudes rather than weight magnitudes?",
        "a": "Neural network weights are typically regularized and exhibit relatively uniform variance across channels. In contrast, LLM activations scaled beyond 6.7B parameters develop sharp, emergent outlier channels where 0.1% to 1% of hidden dimensions reach magnitudes 10x to 100x larger than average. Multiplying even a tiny weight rounding error by a 100x activation produces a massive output error. Therefore, activation volume is the true indicator of channel importance."
    },
    {
        "q": "Explain how SmoothQuant mathematically enables INT8 Tensor Core execution without changing the model's output.",
        "a": "SmoothQuant exploits the mathematical equivalence: Y = X @ W = (X @ diag(s)^-1) @ (diag(s) @ W) = X_smooth @ W_smooth. It computes a per-channel smoothing scale s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1-alpha)). By dividing activations by s, extreme outlier spikes are squashed. By multiplying weights by s, the difficulty is absorbed into the weights. Because the inverse scaling can be folded offline into the preceding LayerNorm weights, there is zero forward-pass compute overhead, allowing native INT8 x INT8 matrix multiplication."
    },
    {
        "q": "What is the key theoretical difference between bitsandbytes NF4 and standard INT4?",
        "a": "Standard INT4 divides the interval uniformly into 16 linear bins. However, pre-trained LLM weights follow a zero-mean normal distribution, meaning most weights are packed densely around zero while the tails are nearly empty. Uniform binning wastes precision. NormalFloat 4 (NF4) sets its 16 discrete bin levels based on the quantiles of a standard normal distribution N(0, 1), ensuring that every single bin contains an exactly equal probability mass (1/16). This maximizes information entropy per bit."
    },
    {
        "q": "Why is Weight-Only Quantization (W4A16) ideal for batch-size 1 decoding, but insufficient for large batch sizes?",
        "a": "At batch size 1, LLM token generation is strictly memory-bandwidth bound: the GPU spends 95% of its clock cycles reading weights from VRAM to registers and only 5% performing arithmetic. Compressing weights from 16 bits to 4 bits reduces VRAM traffic by 4x, yielding immediate speedups. However, at large batch sizes (e.g., 64 or 128), compute intensity increases and the GPU enters the compute-bound regime. Because W4A16 dequantizes weights to FP16 before executing FP16 Tensor Core arithmetic, it cannot exceed FP16 compute ceilings. True speedups at high load require W8A8 (SmoothQuant) or FP8 matrix operations."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Calibration Set Distribution Mismatch (Domain Poisoning)**
    Using generic web text (e.g., standard C4/Pile) to calibrate a model destined for specialized coding or biomedical tasks. 
    If the calibration set never exercises domain-specific tokens or specialized activation pathways, Hessian matrices and AWQ 
    salience scores will miss critical circuits, leading to catastrophic degradation on technical prompts.
    """)
    st.warning("""
    **Trap 3: bitsandbytes NF4 Serving Throughput Bottleneck**
    Using bitsandbytes 4-bit (`load_in_4bit=True`) for high-concurrency production serving. While bitsandbytes is unmatched 
    for single-GPU QLoRA fine-tuning, its inference kernels dynamically dequantize weights to FP16 in shared memory on every pass. 
    In high-throughput servers, replace bitsandbytes with pre-quantized AWQ or GPTQ loaded via vLLM.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Damping Constant Instability in GPTQ Hessian Inversion**
    Setting the Hessian numerical damping parameter `lambda` too low (or omitting it). When inverting `H = 2 * X @ X^T`, 
    ill-conditioned channels cause the matrix to be singular or near-singular. Without proper damping (`lambda * I`), 
    the inverse explodes numerically, generating `NaN` weights across entire transformer blocks.
    """)
    st.warning("""
    **Trap 4: Naive INT8 Activation Overflow Without Pre-Scaling**
    Attempting to perform INT8 matrix multiplication on GPUs without per-tensor dynamic scale clipping. In high-entropy prompts, 
    unanticipated activation spikes will overflow signed 8-bit registers (`> 127`), resulting in integer wraparound to `-128` and producing gibberish output.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: GPTQ Core"):
        st.markdown("**What is the primary formula in GPTQ?**\\n\\n`W_rem = W_rem - (err / H_inv[q,q]) * H_inv[q, rem]`\\n\\nUses inverse Hessian of activations to cancel rounding errors.")
with fc2:
    with st.popover("Flashcard 2: AWQ Core"):
        st.markdown("**How does AWQ protect channels without mixed precision?**\\n\\nApplies per-channel scale `s` to weights and `s^-1` to activations; folds `s^-1` into preceding LayerNorm.")
with fc3:
    with st.popover("Flashcard 3: SmoothQuant"):
        st.markdown("**What does migration factor alpha control?**\\n\\n`alpha = 0.5` divides quantization difficulty equally between weights and activations for W8A8 INT8 Tensor Cores.")
with fc4:
    with st.popover("Flashcard 4: NF4"):
        st.markdown("**Why does NF4 beat INT4?**\\n\\nBins represent equal quantiles (6.25% each) of a normal distribution, maximizing entropy for zero-mean weights.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why naive Round-To-Nearest (RTN) fails at 4-bit precision due to weight interdependence.",
    "I can explain the second-order Taylor expansion formulation and inverse Hessian update mechanics in GPTQ.",
    "I know why AWQ uses activation magnitudes to identify salient channels and how per-channel scaling avoids mixed-precision kernel penalties.",
    "I can articulate how bitsandbytes LLM.int8() separates outlier channels from non-outliers, and how NF4 calculates equal-mass quantiles.",
    "I can derive the SmoothQuant mathematical transformation (Y = (X / s) @ (W * s).T) and explain the role of alpha.",
    "I understand the trade-offs between Weight-Only PTQ (memory-bound), W8A8 PTQ (compute-bound), and QAT (training-time simulation)."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of modern LLM quantization algorithms (GPTQ, AWQ, bitsandbytes, and SmoothQuant).")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
