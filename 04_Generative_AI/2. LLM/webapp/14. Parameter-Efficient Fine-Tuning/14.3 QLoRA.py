import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="14.3 QLoRA - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionQLoRAEngine:
    """
    Self-contained educational implementation of QLoRA (Quantized Low-Rank Adaptation):
    - 4-bit NormalFloat (NF4) theoretical codebook
    - Block-wise quantization & Double Quantization calculation
    - Dual precision dequantization on the fly to BF16 for forward projection
    """
    # Exact 16 quantiles of the standard normal distribution N(0, 1) normalized to [-1.0, 1.0]
    NF4_CODEBOOK = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ], dtype=np.float32)

    def __init__(self, block_size: int = 64, double_quant_block: int = 256):
        self.block_size = block_size
        self.double_quant_block = double_quant_block

    def quantize_nf4(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes continuous weights into 4-bit NF4 indices and scale factors.
        Input: [N] 1D weight tensor.
        Output: indices [N] (integers 0..15), scales [N // block_size].
        """
        n_elements = len(weights)
        n_blocks = int(np.ceil(n_elements / self.block_size))
        
        # Pad to multiple of block size
        padded_len = n_blocks * self.block_size
        padded_w = np.pad(weights, (0, padded_len - n_elements), mode="constant")
        blocks = padded_w.reshape(n_blocks, self.block_size)
        
        # Scale c_1: max absolute value per block
        scales = np.max(np.abs(blocks), axis=-1, keepdims=True)
        scales[scales == 0] = 1.0
        
        # Normalize to [-1.0, 1.0]
        norm_blocks = blocks / scales
        
        # Find closest NF4 codebook index
        # [n_blocks, block_size, 1] vs [1, 1, 16]
        diffs = np.abs(norm_blocks[:, :, np.newaxis] - self.NF4_CODEBOOK[np.newaxis, np.newaxis, :])
        indices = np.argmin(diffs, axis=-1)
        
        return indices.flatten()[:n_elements], scales.flatten()

    def dequantize_nf4(self, indices: np.ndarray, scales: np.ndarray, original_len: int) -> np.ndarray:
        """
        Dequantizes 4-bit indices back to continuous 16-bit representation on the fly.
        """
        n_blocks = len(scales)
        padded_len = n_blocks * self.block_size
        padded_indices = np.pad(indices, (0, padded_len - len(indices)), mode="constant")
        
        # Codebook lookup
        quant_vals = self.NF4_CODEBOOK[padded_indices].reshape(n_blocks, self.block_size)
        dequant_blocks = quant_vals * scales[:, np.newaxis]
        return dequant_blocks.flatten()[:original_len]

    def calculate_memory_savings(self, param_count_b: float) -> Dict[str, float]:
        """Calculates exact memory savings across FP16, NF4, and Double Quantization."""
        params = param_count_b * 1e9
        # FP16: 16 bits = 2 bytes
        fp16_gb = (params * 2) / (1024**3)
        # NF4 with single quant: 4 bits + (32 bits / 64) = 4.5 bits/param
        nf4_single_bits = 4.0 + (32.0 / self.block_size)
        nf4_single_gb = (params * (nf4_single_bits / 8.0)) / (1024**3)
        # NF4 with Double Quant: 4 bits + (8 bits / 64) + (32 bits / (64 * 256)) = ~4.127 bits/param
        nf4_dq_bits = 4.0 + (8.0 / self.block_size) + (32.0 / (self.block_size * self.double_quant_block))
        nf4_dq_gb = (params * (nf4_dq_bits / 8.0)) / (1024**3)

        return {
            "fp16_gb": fp16_gb,
            "nf4_single_gb": nf4_single_gb,
            "nf4_dq_gb": nf4_dq_gb,
            "bits_per_param_dq": nf4_dq_bits
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("14.3 QLoRA (Quantized Low-Rank Adaptation)")
st.caption("Chapter 14: Parameter-Efficient Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
In May 2023, Tim Dettmers et al. published **QLoRA (Quantized Low-Rank Adaptation)**, delivering arguably 
the most impactful democratization breakthrough in open-source AI history.
Prior to QLoRA, fine-tuning a 65-billion-parameter foundation model required over 780 GB of enterprise GPU VRAM across multiple A100 nodes.
QLoRA reduced that requirement to **under 48 GB**, making 65B/70B fine-tuning achievable on a single GPU workstation.

Crucially, QLoRA proved mathematically and empirically that **quantizing base weights to 4 bits does NOT degrade fine-tuning accuracy** 
when paired with 16-bit LoRA adapters.

In this lesson, you will master:
1. The paradigm of Quantized Fine-Tuning: separating 4-bit storage precision from 16-bit compute precision.
2. 4-bit block-wise quantization mechanics and dynamic dequantization on the fly.
3. **NF4 (NormalFloat 4)**: Why standard INT4 fails on neural weights, and how information-theoretic quantile binning achieves optimal precision.
4. **Double Quantization (DQ)**: Quantizing the quantization constants to reclaim 0.37 bits per parameter.
5. The complete QLoRA training pipeline: Paged Optimizers with CUDA Unified Memory to eradicate memory spikes.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The 4-Bit Quantized Training Pipeline & Memory Flow", divider="rainbow")

st.markdown("""
QLoRA pairs compressed 4-bit storage in GPU VRAM with dynamic 16-bit dequantization inside tensor cores during forward and backward passes.
""")

qlora_dot = """
digraph QLoRABlueprint {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_storage {
        label = "Static GPU VRAM Storage (4-Bit NF4)";
        color = "#334155";
        fontcolor = "#94A3B8";
        NF4Weights [label="Base Weights W_0\\n(4-bit NF4 + Double Quant)\\n3.5 GB for 7B Model", fillcolor="#0F172A"];
        Scales [label="Double-Quantized Scales\\n8-bit c_1 + 32-bit c_2\\n0.127 bits/param", fillcolor="#0F172A"];
        NF4Weights -> Scales;
    }

    subgraph cluster_compute {
        label = "Tensor Core Execution (16-Bit BF16)";
        color = "#334155";
        fontcolor = "#94A3B8";
        Dequant [label="On-the-Fly Dequantization\\nNF4 -> BF16\\n(Transient Cache in SRAM)", fillcolor="#0369A1"];
        BaseForward [label="Base Forward: W_dequant @ x\\n[Batch, Seq_Len, d_out]", fillcolor="#0369A1"];
        LoRAForward [label="LoRA Adapter Branch\\n(alpha / r) * B @ (A @ x)\\n16-bit BF16 Trainable", fillcolor="#047857"];
        
        Dequant -> BaseForward;
    }

    subgraph cluster_paged {
        label = "Optimizer Memory (CUDA Unified)";
        color = "#334155";
        fontcolor = "#94A3B8";
        PagedAdam [label="Paged AdamW 32-bit\\nAuto-evicts to CPU RAM on peaks", fillcolor="#B45309"];
    }

    Output [label="Summed Output h\\nh = W_dequant x + B A x", fillcolor="#7C3AED"];

    NF4Weights -> Dequant [label="Read 4-bit"];
    BaseForward -> Output;
    LoRAForward -> Output;
    LoRAForward -> PagedAdam [label="Gradients strictly on A and B"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(qlora_dot, use_container_width=True)

st.markdown("""
### The Three Core Innovations
1. **4-bit NormalFloat (NF4)**: Information-theoretically optimal quantile distribution tailored specifically for zero-mean Gaussian neural weights.
2. **Double Quantization (DQ)**: Compresses the block quantization constants themselves from 32-bit floats to 8-bit integers, saving ~3GB on a 70B model.
3. **Paged Optimizers**: Leverages CUDA Unified Memory page tables to seamlessly evict optimizer states to host CPU RAM during memory-intensive sequence peaks.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - QUANTIZED FINE-TUNING
# ==============================================================================
st.header("2. Quantized Fine-Tuning", divider="rainbow")

st.markdown("""
Before QLoRA, quantizing a model to 4 bits was strictly a **post-training deployment technique** (e.g. GPTQ, AWQ). 
Attempting to fine-tune 4-bit weights was impossible because:
1. Low-bit integers cannot accumulate small gradient updates (`Delta W < 0.001` rounds to zero).
2. Quantization noise creates severe gradient variance, destabilizing backpropagation.
""")

col_qft1, col_qft2 = st.columns(2)
with col_qft1:
    st.subheader("The QLoRA Paradigm Shift: Dual Precision")
    st.markdown("""
    * **Storage Precision (4-bit)**: The multi-billion base model weights are stored in GPU VRAM strictly as 4-bit NF4 integers.
    * **Compute Precision (16-bit BF16)**: When an input token batch enters a linear layer, the 4-bit weights are **dequantized on the fly into BF16** inside GPU register/SRAM caches.
    * **Matrix Multiplication**: Executed in standard 16-bit floating point.
    * **Adapter Accumulation**: Gradients are backpropagated **strictly to the 16-bit LoRA adapter matrices**, preserving full mathematical gradient precision!
    """)

with col_qft2:
    st.subheader("Why No Accuracy Is Lost")
    st.markdown("""
    * Dettmers et al. performed extensive evaluations across 1,000+ models.
    * QLoRA with 4-bit base weights matched 16-bit full fine-tuning and 16-bit LoRA performance on MMLU, Vicuna-Bench, and AlpacaEval.
    * The base model only needs to preserve its static representations; the 16-bit adapter provides all necessary adaptation flexibility.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - 4-BIT QUANTIZATION
# ==============================================================================
st.header("3. 4-Bit Quantization", divider="rainbow")

st.markdown("""
A 4-bit number can represent exactly `2^4 = 16` discrete values.
To map continuous floating-point weights into 16 bins without catastrophic information loss, QLoRA uses **Block-Wise Quantization**.
""")

st.markdown("""
#### Block-Wise Quantization Mechanics
Instead of finding a single min/max scaling constant for an entire 16-million-weight layer (where one outlier destroys precision for the entire tensor):
1. The weight tensor is partitioned into contiguous blocks of size `B = 64`.
2. For each block `i`, compute scaling constant `c_i = max(|W_i|)`.
3. Normalize block weights: `W_norm = W_i / c_i in [-1.0, 1.0]`.
4. Quantize each normalized weight to the nearest 4-bit codebook bin.
""")

st.code("""
# Block-wise quantization concept
block_size = 64
for block in weights.reshape(-1, block_size):
    scale = torch.max(torch.abs(block))
    normalized = block / scale
    quant_indices = quantize_to_16_bins(normalized)
    # Store 4-bit indices and 32-bit scale
""", language="python")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - NF4 (NORMALFLOAT 4)
# ==============================================================================
st.header("4. NF4 (NormalFloat 4)", divider="rainbow")

st.markdown("""
Why does standard integer quantization (`INT4` or `FP4`) underperform on deep neural networks?
""")

col_nf1, col_nf2 = st.columns(2)
with col_nf1:
    st.markdown("#### The Flaw of Uniform INT4")
    st.markdown("""
    * **Uniform Spacing**: `INT4` divides the interval `[-1.0, 1.0]` into 16 evenly spaced bins: `-8, -7, ..., 0, ..., 7`.
    * **The Weight Distribution Mismatch**: Pretrained neural network weights are **normally distributed with zero mean: `W ~ N(0, sigma^2)`**.
    * **Severe Quantization Waste**: Most weights cluster tightly around `0.0`. Uniform `INT4` places most bins in the empty tails where almost zero weights exist, while giving only 2 or 3 bins to the dense center where 80% of the weights reside!
    """)

with col_nf2:
    st.markdown("#### The Information-Theoretic Solution: NF4")
    st.markdown("""
    * **Equal Probability Mass**: NF4 computes 16 quantiles `q_i` such that each bin contains **exactly 1/16th of the probability mass** of a standard Gaussian distribution:
      ```text
      P(q_i <= X <= q_{i+1}) = 1 / 16 = 6.25%
      ```
    * **Exact Zero Representation**: NF4 is engineered to contain an exact discrete `0.0` bin to preserve padding and zero-mask values.
    * **Result**: Maximizes Shannon entropy and minimizes mean squared quantization error (MSE) for normally distributed weights.
    """)

st.markdown("""
#### The Exact 16 Quantile Values of NF4:
```text
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```
*Notice how bins are densely packed near 0.0 (e.g. -0.09, 0.0, 0.08) and widely spaced at the extremes (-1.0, -0.69).*
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - DOUBLE QUANTIZATION
# ==============================================================================
st.header("5. Double Quantization (DQ)", divider="rainbow")

st.markdown("""
Block-wise quantization solves outlier clipping, but it creates a secondary memory footprint: **the quantization constants**.
""")

st.markdown("""
### Quantization Constant Overhead
If we use block size `B_1 = 64` and store each scaling factor `c_1` in FP32 (32 bits):
```text
Overhead = 32 bits / 64 parameters = 0.5 bits per parameter
```
This increases the effective memory of a 4-bit model to `4.5 bits/param`—a **12.5% memory penalty** just for the scale factors!
""")

col_dq1, col_dq2 = st.columns(2)
with col_dq1:
    st.subheader("The Double Quantization Mechanism")
    st.markdown("""
    Double Quantization quantizes the quantization constants `c_1` themselves:
    1. Group the first-level scales `c_1` into larger blocks of size `B_2 = 256`.
    2. Quantize `c_1` to **8-bit integers (FP8)**.
    3. Store a single 32-bit second-level scale `c_2` for every 256 first-level scales.
    """)

with col_dq2:
    st.subheader("The Exact Bit-Budget Savings")
    st.markdown("""
    * **First-level scales (8-bit)**: `8 bits / 64 = 0.125 bits/param`
    * **Second-level scales (32-bit)**: `32 bits / (64 * 256) = 0.0019 bits/param`
    * **Total scale overhead**: `0.125 + 0.0019 = 0.127 bits/param`
    * **Net Memory Footprint**: Drops from `4.5 bits` down to **`4.127 bits per parameter`**!
    * **Total Savings**: **Saves ~3 GB of VRAM on a 70B model**, often the exact margin needed to fit within an A100 40GB or RTX 3090/4090 GPU!
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - QLORA TRAINING PIPELINE
# ==============================================================================
st.header("6. QLoRA Training Pipeline", divider="rainbow")

st.markdown("""
The complete QLoRA pipeline integrates `bitsandbytes`, Hugging Face `transformers`, `peft`, and Paged Optimizers into a unified stack.
""")

st.code("""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Configure 4-bit NormalFloat and Double Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Dequantize to BF16 for forward pass
)

# 2. Load 4-bit Base Model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Prepare for k-bit training (freezes base weights, casts LayerNorms to FP32)
model = prepare_model_for_kbit_training(model)

# 4. Attach 16-bit LoRA Adapters across all linear projections
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
""", language="python")

st.markdown("""
### Paged Optimizers: The Safety Net Against Out-Of-Memory Spikes
* **The Problem**: During training on sequences with variable lengths, a sudden long sample spikes activation memory, triggering a CUDA Out-Of-Memory error even if the average batch fits comfortably in VRAM.
* **The Paged Optimizer Solution**: Uses `paged_adamw_8bit` or `paged_adamw_32bit`. When GPU memory exceeds 95%, CUDA Unified Memory automatically evicts optimizer state pages to host CPU RAM over PCIe, swapping them back when the batch backward step finishes.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End NF4 Quantizer & Dequantizer in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates the complete mathematical quantization and dequantization cycle:
NF4 codebook lookup, block scaling, and on-the-fly reconstruction.
""")

st.code("""
import numpy as np
from typing import Tuple

class ProductionQLoRAEngine:
    \"\"\"
    Production-grade educational NF4 Quantizer and Dequantizer.
    \"\"\"
    NF4_CODEBOOK = np.array([
        -1.0, -0.6961928, -0.5250731, -0.3949175,
        -0.2844414, -0.1847734, -0.0910500, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0
    ], dtype=np.float32)

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def quantize_nf4(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_elements = len(weights)
        n_blocks = int(np.ceil(n_elements / self.block_size))
        padded = np.pad(weights, (0, n_blocks * self.block_size - n_elements))
        blocks = padded.reshape(n_blocks, self.block_size)
        
        scales = np.max(np.abs(blocks), axis=-1, keepdims=True)
        scales[scales == 0] = 1.0
        norm_blocks = blocks / scales
        
        # Match nearest quantile index
        diffs = np.abs(norm_blocks[:, :, np.newaxis] - self.NF4_CODEBOOK[np.newaxis, np.newaxis, :])
        indices = np.argmin(diffs, axis=-1)
        return indices.flatten()[:n_elements], scales.flatten()

    def dequantize_nf4(self, indices: np.ndarray, scales: np.ndarray, original_len: int) -> np.ndarray:
        n_blocks = len(scales)
        padded_idx = np.pad(indices, (0, n_blocks * self.block_size - len(indices)))
        quant_vals = self.NF4_CODEBOOK[padded_idx].reshape(n_blocks, self.block_size)
        return (quant_vals * scales[:, np.newaxis]).flatten()[:original_len]
""", language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: NF4 Quantizer Visualizer & VRAM Budget Calculator", divider="rainbow")

engine = ProductionQLoRAEngine(block_size=64)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("1. VRAM Sizer Across Model Scales")
    model_scale = st.selectbox(
        "Base Model Scale",
        [7.0, 13.0, 34.0, 70.0],
        index=0,
        format_func=lambda x: f"{x} Billion Parameters (e.g. Llama-3-{int(x) if x.is_integer() else x}B)"
    )
    savings = engine.calculate_memory_savings(model_scale)

    st.metric("16-Bit Base Weight VRAM", f"{savings['fp16_gb']:.2f} GB", "16 bits/param")
    st.metric("QLoRA 4-bit NF4 + Double Quant VRAM", f"{savings['nf4_dq_gb']:.2f} GB", f"{savings['bits_per_param_dq']:.3f} bits/param")

    vram_diff = savings["fp16_gb"] - savings["nf4_dq_gb"]
    st.success(f"⚡ QLoRA frees up **{vram_diff:.2f} GB of GPU memory** on this model!")

with wb_col2:
    st.subheader("2. Real-Time NormalFloat 4 Quantization Simulator")
    st.markdown("Generate synthetic Gaussian weights and watch the NF4 quantizer preserve distributions.")

    sample_size = st.slider("Weight Vector Size (Parameters)", min_value=64, max_value=512, value=128, step=64)
    np.random.seed(42)
    synthetic_w = np.random.normal(0, 0.25, size=sample_size).astype(np.float32)

    indices, scales = engine.quantize_nf4(synthetic_w)
    reconstructed_w = engine.dequantize_nf4(indices, scales, sample_size)

    mse = float(np.mean((synthetic_w - reconstructed_w)**2))
    st.metric("Quantization Mean Squared Error (MSE)", f"{mse:.6f}", "Optimal Normal Distribution Preservation")

    sim_df = pd.DataFrame({
        "Index": np.arange(len(synthetic_w))[:40],
        "Original Float32": synthetic_w[:40],
        "Dequantized NF4": reconstructed_w[:40]
    })
    st.line_chart(sim_df.set_index("Index"))

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: What is NormalFloat 4 (NF4) and why does it outperform standard INT4?"):
        st.write("""
        Pretrained neural network weights follow a normal distribution centered at zero.
        Standard INT4 divides values into uniformly spaced bins, wasting quantization levels on the low-probability tails while failing to represent the dense cluster near zero.
        NF4 constructs 16 discrete levels where each bin contains equal probability mass under N(0, 1). 
        This maximizes information entropy and minimizes mean squared reconstruction error for deep learning weights.
        """)
    with st.expander("Q2: How does Double Quantization work and how much memory does it save?"):
        st.write("""
        Block-wise quantization stores a 32-bit floating-point scale factor for every block of 64 weights, adding 0.5 bits/parameter overhead.
        Double Quantization quantizes these first-level scale factors to 8-bit integers (FP8) with a block size of 256.
        This reduces the scale overhead from 0.5 bits down to 0.127 bits per parameter, saving 0.37 bits/param (~3 GB VRAM on a 70B model).
        """)
    with st.expander("Q3: How are gradients computed through 4-bit weights during backpropagation?"):
        st.write("""
        Gradients are NOT computed for 4-bit weights.
        The 4-bit base model weights are strictly frozen (`requires_grad=False`).
        During the forward pass, 4-bit weights are dequantized on the fly into BF16 to compute layer activations. 
        Backpropagation flows gradients through the 16-bit LoRA adapter matrices, which update with standard floating-point precision.
        """)

with q2:
    with st.expander("Q4: What is a Paged Optimizer and how does it prevent OOM crashes?"):
        st.write("""
        Paged Optimizers use CUDA Unified Memory to automatically allocate memory pages across GPU VRAM and CPU system RAM.
        When activation peaks occur during long-context training, the driver automatically pages optimizer state memory out to CPU RAM over PCIe.
        Once the activation tensors are freed after the backward pass, the optimizer pages are brought back to GPU memory, eliminating transient OOM crashes.
        """)
    with st.expander("Q5: What is the computational speed tradeoff of QLoRA vs standard 16-bit LoRA?"):
        st.write("""
        QLoRA is typically 15% to 30% slower per training step than 16-bit LoRA.
        This overhead stems from on-the-fly dequantization kernels converting 4-bit weights to 16-bit BF16 before every matrix multiplication.
        However, the ability to train 70B models on accessible hardware overwhelmingly offsets the slight latency penalty.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Setting bnb_4bit_compute_dtype to FP32**
    * **Failure**: Leaving compute dtype as FP32 on modern Ampere/Hopper GPUs.
    * **Impact**: Dequantizing weights to FP32 doubles activation memory and disables native BF16 tensor core acceleration, causing 3x slower training.
    * **Remedy**: Always set `bnb_4bit_compute_dtype=torch.bfloat16`.
    """)
    st.error("""
    **🚨 Production Trap 2: Attempting to Merge LoRA into 4-bit Base Weights**
    * **Failure**: Executing `model.merge_and_unload()` on a 4-bit quantized base model.
    * **Impact**: 4-bit discrete quantization levels cannot store continuous float additions without disastrous rounding noise.
    * **Remedy**: Load the unquantized base model in 16-bit (BF16), then call `merge_and_unload()`.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Missing prepare_model_for_kbit_training**
    * **Failure**: Attaching LoRA directly to a 4-bit model without calling `prepare_model_for_kbit_training`.
    * **Impact**: LayerNorm and embedding layers remain in low precision, causing catastrophic gradient divergence and NaN loss within 10 steps.
    * **Remedy**: Always wrap with `prepare_model_for_kbit_training(model)` to cast normalization layers to FP32.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Running QLoRA on GPUs Older Than Turing (GTX 1080 / P100)**
    * **Failure**: Attempting QLoRA on Pascal GPUs lacking native INT4/BF16 support.
    * **Impact**: BitsAndBytes falls back to software emulation or crashes with CUDA kernel launch errors.
    * **Remedy**: Ensure target hardware is Ampere or newer (RTX 3000+, A100, H100) for full BF16/NF4 support.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: NF4 Core Principle"):
        st.write("Why does NF4 assign equal probability mass to each bin?")
        st.info("To maximize information entropy and minimize quantization error for zero-mean normally distributed weights.")
    with st.expander("⚡ Flashcard 2: Net Bit-Rate of QLoRA"):
        st.write("What is the exact bit-budget per parameter of NF4 with Double Quantization?")
        st.info("Approximately 4.127 bits per parameter (4 bits weight + 0.127 bits scale).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Paged Optimizer Mechanism"):
        st.write("What hardware feature enables Paged Optimizers?")
        st.info("CUDA Unified Memory page tables between GPU VRAM and CPU RAM.")
    with st.expander("⚡ Flashcard 4: Compute Precision in QLoRA"):
        st.write("What precision are matrix multiplications executed in during QLoRA?")
        st.info("16-bit precision (BF16 or FP16), via on-the-fly dequantization.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of QLoRA before advancing to Lesson 14.4 (Other PEFT Methods):
""")

chk1 = st.checkbox("I understand why NF4 outperforms uniform INT4 for neural network weights.", value=True)
chk2 = st.checkbox("I know how Double Quantization reduces scale constant overhead to 0.127 bits/param.", value=True)
chk3 = st.checkbox("I understand dual precision: 4-bit storage in VRAM, 16-bit compute in tensor cores.", value=True)
chk4 = st.checkbox("I know how Paged Optimizers eliminate transient activation memory spikes.", value=True)
chk5 = st.checkbox("I understand why prepare_model_for_kbit_training is required to cast LayerNorm to FP32.", value=True)
chk6 = st.checkbox("I know why merging adapters requires dequantizing the base model to 16-bit first.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 14.4: Other PEFT Methods.")
