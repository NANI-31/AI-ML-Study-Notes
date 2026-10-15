"""
7.3 Long Context
================
Interactive Master Lesson exploring Context Windows, Position Extrapolation,
Positional Interpolation (Linear, NTK-Aware, YaRN), and Modern Long-Context Engineering.
"""

from typing import Dict, List, Tuple
import graphviz
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="7.3 Long Context - LLM Architecture",
    page_icon="📜",
    layout="wide",
)

# Custom CSS for aesthetic styling & dark theme typography
st.markdown(
    """
<style>
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .callout-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TOP NAVIGATION / HERO
# -----------------------------------------------------------------------------
st.title("📜 7.3 Long Context")
st.caption(
    "Chapter 7: Positional Encoding and Position Information | Module 3 of 3"
)

st.markdown(
    """
Modern frontier Large Language Models have expanded from early 2,048-token context limits
(GPT-3) to massive windows spanning 128,000 tokens (Llama 3), 1,000,000 tokens (Llama 3.1, Gemini 1.5 Pro),
and beyond. Expanding context windows requires overcoming fundamental barriers:
**O(N^2) attention computation**, **KV cache GPU VRAM explosion**, **position extrapolation failure**,
and the empirical **Lost in the Middle** phenomenon.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & CONTEXT SCALING FRONTIER
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & Context Scaling Frontier", divider="rainbow")

st.markdown(
    """
### The Evolution of Context Window Capacities
The history of context scaling represents a simultaneous breakthrough in mathematics (RoPE scaling),
algorithmic attention kernels (FlashAttention), and distributed cluster communication (Ring Attention).
"""
)

# Architectural Graphviz Diagram
arch_dot = graphviz.Digraph(
    "long_context_arch",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node(
    "c1",
    "2k - 4k Tokens (2020-2022)\\nBERT (512), GPT-3 (2k), Llama 1 (2k)\\nStandard Dense Attention, Learned Abs PE",
    fillcolor="#1E293B",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "c2",
    "8k - 32k Tokens (2023)\\nLlama 2 (4k), Mistral 7B (8k), CodeLlama (16k)\\nFlashAttention-2, Linear RoPE Interpolation",
    fillcolor="#1E293B",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "c3",
    "128k Tokens (2024)\\nLlama 3 (8k to 128k), GPT-4 Turbo (128k)\\nYaRN, NTK-Aware Scaling, theta=500k",
    fillcolor="#0369A1",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "c4",
    "1M - 10M Tokens (2024-Present)\\nGemini 1.5 Pro (2M), Llama 3.1 (128k/1M)\\nFlashAttention-3, Ring Attention, Deep Context Distillation",
    fillcolor="#047857",
    fontcolor="#F8FAFC",
)

arch_dot.edge("c1", "c2", label="IO-Aware FlashAttention")
arch_dot.edge("c2", "c3", label="Frequency Interpolation (YaRN)")
arch_dot.edge("c3", "c4", label="Distributed Ring Attention and GQA")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Frontier LLMs achieve 128k to 1M token contexts through a unified 4-pillar system:
1. **Grouped-Query Attention (GQA)**: Reduces Key-Value cache memory footprint by 4x to 8x.
2. **YaRN (Yet another RoPE extensioN)**: Frequency-dependent interpolation ensuring high-frequency local detail is preserved while low-frequency global coordinates are stretched.
3. **Chunked Prefill & FlashAttention-3**: Eliminates IO memory bottlenecks by computing attention in GPU SRAM blocks without materializing the N x N attention matrix in HBM.
4. **Targeted Long-Context Continual Pretraining**: Training on high-quality long-context curricula (books, code repos, multi-turn dialogues) with synthetic needle-in-a-haystack retrieval tasks.
"""
)

# Comparison Table
context_scaling_df = pd.DataFrame(
    {
        "Model": ["GPT-2", "GPT-3", "Llama 1", "Mistral 7B", "Llama 3 70B", "Llama 3.1", "Gemini 1.5 Pro"],
        "Context Window": ["1,024", "2,048", "2,048", "8,192", "8,192", "131,072", "2,000,000+"],
        "Positional Strategy": ["Learned Absolute", "Learned Absolute", "Standard RoPE (10k)", "RoPE (10k) + SWA", "RoPE (500k base)", "YaRN / Scaled RoPE", "Proprietary Recurrent / Long RoPE"],
        "KV Cache (FP16 per Batch)": ["~0.03 GB", "~0.3 GB", "~0.5 GB", "~1.2 GB", "~12 GB", "~130 GB (GQA)", "Multi-Node Sharded"],
        "Attention Backend": ["Standard Matrix", "Standard Matrix", "xFormers", "FlashAttention-2", "FlashAttention-2", "FlashAttention-3", "Distributed TPU Ring Attention"],
    }
)
st.dataframe(context_scaling_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: CONTEXT WINDOW DYNAMICS & MEMORY BOTTLENECK
# -----------------------------------------------------------------------------
st.header("2. Context Window Dynamics & Memory Bottlenecks", divider="rainbow")

st.markdown(
    """
A model's context window defines the maximum number of consecutive tokens it can ingest and reason over in a single forward pass.
Scaling the context window exposes two distinct physical bottlenecks:

### 1. Quadratic Attention Computational Complexity: O(N^2)
For sequence length `N`, computing all pairwise dot products `Q * K^T` requires:
* Floating point operations: `FLOPs = 4 * B * L * H * N^2 * d_k`
* At `N = 2,048`: `N^2 = 4,194,304` operations per head.
* At `N = 128,000`: `N^2 = 16,384,000,000` operations per head (a 3,900x increase in compute!).

### 2. Linear KV Cache Memory Growth: O(N)
During autoregressive generation, past Key and Value states must be cached in GPU high-bandwidth memory (HBM).
For a model with `L` layers, `H_kv` KV heads, `d_k` head dimension, batch size `B`, and sequence length `N`:

```text
KV_Cache_Bytes = 2 (Key + Value) * B * N * L * H_kv * d_k * Bytes_Per_Element
```

For Llama 3 70B (`L = 80`, `H_kv = 8`, `d_k = 128`, FP16 = 2 bytes):
* Per token per batch: `2 * 1 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB / token`.
* At `N = 8,192`: `2.6 GB` VRAM.
* At `N = 128,000`: `40.96 GB` VRAM **per single concurrent user** just for the KV cache!
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: POSITION EXTRAPOLATION VS. OUT-OF-DISTRIBUTION FAILURES
# -----------------------------------------------------------------------------
st.header("3. Position Extrapolation vs. Out-of-Distribution Failures", divider="rainbow")

st.markdown(
    """
### What Happens When You Pass N > L_train?
If a model trained on context length `L_train = 4,096` receives a prompt of `8,192` tokens without position scaling:
* **Perplexity Explosion**: Test cross-entropy loss immediately shoots to infinity.
* **Phase Incoherence in RoPE**: The rotation angle for position `m` is `m * theta_i`. When `m > L_train`, rotation vectors encounter angle intervals `(theta_i * m)` that the model was never trained to associate with attention routing.
* **Logit Range Distortion**: Queries and Keys produce wildly out-of-distribution inner products, causing Softmax attention distributions to become either completely uniform or degenerate into single-token attractors (attention sinks).
"""
)

col_ext1, col_ext2 = st.columns(2)
with col_ext1:
    st.markdown(
        """
        **Direct Extrapolation Failure Mechanism:**
        * High-frequency channels wrap around multiple complete rotations.
        * Low-frequency channels explore previously unseen coordinate quadrants.
        * Multi-layer residual streams accumulate positional error exponentially across depth.
        """
    )
with col_ext2:
    st.markdown(
        """
        **Empirical Evidence:**
        * Testing Llama 2 (trained at 4k) at 5k tokens without scaling yields 100% gibberish or loop generation.
        * ALiBi exhibits graceful decay but struggles with complex needle retrieval past 2x training length.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 4: LONG-CONTEXT ARCHITECTURAL CHALLENGES
# -----------------------------------------------------------------------------
st.header("4. Long-Context Architectural Challenges (KV Cache, Compute, Lost in the Middle)", divider="rainbow")

st.markdown(
    """
Extending context length is not purely a memory challenge; it fundamentally impacts reasoning and information retrieval:
"""
)

st.markdown(
    """
### 1. The "Lost in the Middle" Phenomenon (Liu et al., 2023)
Even when a model can technically ingest 64k or 128k tokens without crashing, empirical evaluation demonstrates a U-shaped performance curve:
* Information placed at the **beginning** of the prompt (Primacy effect) is retrieved with high accuracy (>95%).
* Information placed at the **very end** of the prompt (Recency effect) is retrieved with high accuracy (>95%).
* Information placed in the **middle 20% to 80%** of the prompt drops drastically in retrieval fidelity (often falling below 40% accuracy).

### 2. Needle In A Haystack (NIAH) Benchmark
Modern evaluations test long-context fidelity by inserting a single synthetic fact ("The secret passkey is 94821") at varying depths (0% to 100%) across varying context lengths (1k to 128k tokens) and demanding precise retrieval.
Frontier models require explicit long-context fine-tuning on diverse document layouts to achieve solid green NIAH heatmaps.

### 3. Cumulative Softmax Dilution
As context grows to 100,000 tokens, standard Softmax normalizes across 100,000 logits. The denominator sum becomes enormous, causing probability mass to diffuse across thousands of irrelevant background tokens, drowning out weak semantic signals.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: CONTEXT-LENGTH HARDWARE & ALGORITHMIC LIMITATIONS
# -----------------------------------------------------------------------------
st.header("5. Context-Length Hardware & Algorithmic Limitations", divider="rainbow")

st.markdown(
    """
Scaling context is governed by physical hardware bounds: Memory Bandwidth and Memory Capacity.
"""
)

limitations_data = [
    {
        "Component": "GPU High Bandwidth Memory (HBM)",
        "Limitation": "H100 has 80GB HBM. Storing 70B model weights in 16-bit consumes 140GB (requires 2 GPUs minimum). Adding 128k KV caches for a batch of 4 consumes another 160GB.",
        "Engineering Mitigation": "Grouped-Query Attention (GQA), FP8 / INT4 KV cache quantization, PagedAttention (vLLM).",
    },
    {
        "Component": "Memory Bandwidth (HBM to SRAM Transfer)",
        "Limitation": "Autoregressive decode step is memory-bandwidth bound. Fetching the entire KV cache from HBM for every generated token saturates bus speeds (3.35 TB/s on H100).",
        "Engineering Mitigation": "FlashDecoding, Speculative Decoding, Medusa heads to verify multiple tokens per KV cache fetch.",
    },
    {
        "Component": "Inter-GPU Interconnect (NVLink)",
        "Limitation": "Sequences exceeding single-node capacity (8 GPUs) must split context across nodes via InfiniBand, incurring microsecond latency penalties.",
        "Engineering Mitigation": "Ring Attention, DeepSpeed Ulysses, Context Parallelism (CP).",
    },
]
st.dataframe(pd.DataFrame(limitations_data), use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 6: POSITION INTERPOLATION (PI, NTK-AWARE, YARN ROPE SCALING)
# -----------------------------------------------------------------------------
st.header("6. Position Interpolation (PI, NTK-Aware, YaRN RoPE Scaling)", divider="rainbow")

st.markdown(
    """
To extend RoPE contexts without training from scratch, researchers developed mathematical interpolation techniques:
"""
)

st.markdown(
    """
### 1. Linear Position Interpolation (Chen et al., 2023)
Instead of extrapolating to unseen positions `m > L_train`, compress position coordinates by a constant scaling factor `s = L_target / L_train`:

```text
m_interpolated = m / s
```
* **Advantage**: The model never encounters rotation angles larger than those seen during pretraining.
* **Fatal Flaw**: Compressing all frequencies equally causes high frequencies (which encode fine-grained local word order) to become overcrowded, degrading grammar and code parsing.

### 2. NTK-Aware RoPE Scaling (bloc97, 2023)
Inspired by Neural Tangent Kernel theory, NTK scaling notes that low frequencies matter for global position, while high frequencies matter for local position.
Instead of scaling position index `m`, NTK scales the base frequency `theta`:

```text
theta_new = theta_base * s ^ (d / (d - 2))
```
* High frequencies are barely modified (preserving local syntactic discrimination).
* Low frequencies are heavily stretched (expanding global context window).
* Enables zero-shot extension from 4k to 16k with zero fine-tuning!

### 3. YaRN (Yet another RoPE extensioN, Peng et al., 2023)
YaRN divides dimensions into three distinct frequency bands:
1. **High Frequencies (Wavelength < L_train)**: Zero interpolation. Keep original RoPE angles.
2. **Low Frequencies (Wavelength > L_train)**: Full linear interpolation.
3. **Medium Frequencies**: Smooth ramp function transitioning between the two.
Additionally, YaRN applies an attention temperature multiplier `sqrt(t)` to combat Softmax entropy dilution.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: LONG-CONTEXT OPTIMIZATION
# -----------------------------------------------------------------------------
st.header("7. Long-Context Optimization (FlashAttention-3, Chunked Prefill, Ring Attention)", divider="rainbow")

st.markdown(
    """
State-of-the-art long-context serving stacks deploy three critical algorithmic innovations:
"""
)

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    st.subheader("FlashAttention-3")
    st.markdown(
        """
        * Exploits Hopper H100 Tensor Memory Accelerator (TMA) and FP8 asynchronous pipelines.
        * Tiles Q, K, V blocks into on-chip SRAM (228 KB per SM).
        * Computes Softmax online via safe running-max updates, never writing the N x N attention matrix to DRAM.
        * Delivers 1.5x - 2x speedup over FlashAttention-2 at 128k lengths.
        """
    )

with col_opt2:
    st.subheader("Chunked Prefill")
    st.markdown(
        """
        * A 64k token prompt would normally monopolize GPU execution for seconds, blocking all concurrent decode streams.
        * Chunked prefill splits large prompts into 2k or 4k chunks, interleaving them with autoregressive decode iterations.
        * Prevents Time-To-First-Token (TTFT) spikes and stabilizes cluster throughput.
        """
    )

with col_opt3:
    st.subheader("Ring Attention")
    st.markdown(
        """
        * Distributes a 1M token sequence across a ring of 16 to 128 GPUs.
        * While GPU `i` computes attention between its local Query block and its current Key-Value block, it asynchronously transmits the KV block to GPU `(i+1)` via NVLink.
        * Perfectly overlaps communication with computation, enabling near-linear context scaling to 10M tokens.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE YARN & NTK ROPE SCALING IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("8. Production-Grade YaRN & NTK RoPE Scaling Implementation", divider="rainbow")

st.markdown(
    """
Vectorized PyTorch implementation of NTK-Aware and YaRN frequency scaling for production LLMs:
"""
)

code_snippet = """import torch
import torch.nn as nn
import math

class ScaledRotaryEmbedding(nn.Module):
    # Production NTK and YaRN scaled RoPE module
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_type: str = "yarn",
        scaling_factor: float = 4.0,  # e.g. 4x expansion (4k -> 16k)
        original_max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        
        self._compute_inv_freq()

    def _compute_inv_freq(self):
        if self.scaling_type == "none":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif self.scaling_type == "linear":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            inv_freq = inv_freq / self.scaling_factor
        elif self.scaling_type == "ntk":
            # NTK-Aware scaling: scale base frequency
            base_ntk = self.base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base_ntk ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif self.scaling_type == "yarn":
            # YaRN: Frequency-band dependent ramp
            pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            inv_freq = 1.0 / pos_freqs
            
            # Wavelength = 2 * pi * pos_freqs
            wavelengths = 2 * math.pi * pos_freqs
            
            # Compute interpolation ramp parameters
            low = self.original_max_position_embeddings / 32.0
            high = self.original_max_position_embeddings
            
            ramp = torch.clamp((wavelengths - low) / (high - low), 0.0, 1.0)
            # Blend: (1 - ramp) * unscaled + ramp * (linear scaled)
            inv_freq = (1.0 - ramp) * inv_freq + ramp * (inv_freq / self.scaling_factor)
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")
            
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE LONG-CONTEXT CAPACITY & ATTENTION DECAY WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Long-Context Capacity & Attention Decay Workbench", divider="rainbow")

st.markdown(
    """
Model the GPU VRAM consumption of the Key-Value (KV) Cache and compare effective
rotation frequencies under **Standard RoPE**, **Linear Interpolation**, **NTK-Aware**, and **YaRN**.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    target_tokens = st.select_slider(
        "Target Context Length (Tokens)",
        options=[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576],
        value=32768,
    )
    batch_size = st.slider("Batch Size (Concurrent Streams)", min_value=1, max_value=16, value=2)
with col_ctrl2:
    model_family = st.selectbox(
        "Model Architecture",
        options=["Llama 3 8B (32L, 8KV, 128D)", "Llama 3 70B (80L, 8KV, 128D)", "Mistral 7B (32L, 8KV, 128D)"],
        index=1,
    )
    precision_bytes = st.selectbox("KV Cache Precision", options=["FP16 (2 Bytes)", "FP8 (1 Byte)", "INT4 (0.5 Bytes)"], index=0)

# Parsing model specs
if "70B" in model_family:
    layers, kv_heads, head_dim = 80, 8, 128
else:
    layers, kv_heads, head_dim = 32, 8, 128

bytes_per_elem = 2.0 if "FP16" in precision_bytes else (1.0 if "FP8" in precision_bytes else 0.5)

# Calculate VRAM
kv_cache_bytes = 2 * batch_size * target_tokens * layers * kv_heads * head_dim * bytes_per_elem
kv_cache_gb = kv_cache_bytes / (1024 ** 3)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total KV Cache VRAM", f"{kv_cache_gb:.2f} GB")
with col_m2:
    tokens_per_sec_overhead = target_tokens / 1000.0
    st.metric("Attention Flops Scale", f"{(target_tokens / 4096.0) ** 2:.1f}x vs 4k")
with col_m3:
    scaling_req = target_tokens / 4096.0
    st.metric("Required RoPE Expansion", f"{scaling_req:.1f}x")

# RoPE Frequency Spectrum Comparison Table
dim = head_dim
indices = np.arange(0, dim, 2)
unscaled_freq = 1.0 / (10000.0 ** (indices / dim))
linear_freq = unscaled_freq / max(1.0, scaling_req)
ntk_base = 10000.0 * (scaling_req ** (dim / (dim - 2)))
ntk_freq = 1.0 / (ntk_base ** (indices / dim))

freq_comparison_df = pd.DataFrame(
    {
        "Channel Dimension": [f"Dim {i}-{i+1}" for i in indices[:6]],
        "Standard RoPE (10k)": np.round(unscaled_freq[:6], 5),
        "Linear Interpolated": np.round(linear_freq[:6], 5),
        "NTK-Aware Scaled": np.round(ntk_freq[:6], 5),
    }
)
st.subheader("Frequency Scaling Spectrum (First 6 Head Subspaces)")
st.dataframe(freq_comparison_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: What is the mathematical difference between Position Interpolation and Position Extrapolation?"):
    st.markdown(
        """
        **Answer:**
        * **Position Extrapolation** extends the context by evaluating the model on position indices `m > L_train`. The model encounters rotation angles and distance values outside its training distribution, causing severe attention phase distortion and perplexity explosion.
        * **Position Interpolation (PI)** scales the position indices by `s = L_target / L_train`, evaluating positions as `m / s`. This bounds all input angles within `[0, L_train]`. While this avoids out-of-distribution values, uniform linear scaling compresses high frequencies, degrading local syntax understanding unless mitigated by NTK or YaRN.
        """
    )

with st.expander("Q2: Why does Grouped-Query Attention (GQA) dramatically assist Long-Context serving?"):
    st.markdown(
        """
        **Answer:**
        In standard Multi-Head Attention (MHA), every query head has its own dedicated key and value head (`H_kv = H_q`).
        At long contexts (e.g. 128k), the KV cache consumes tens of gigabytes of VRAM per batch.
        GQA shares a single Key-Value head across a group of Query heads (e.g., 8 Q heads per KV head).
        This cuts KV cache memory size and memory bandwidth transfers by **8x**, allowing long-context serving on single-node GPU clusters.
        """
    )

with st.expander("Q3: Explain the 'Lost in the Middle' problem and how modern models mitigate it."):
    st.markdown(
        """
        **Answer:**
        Transformer autoregressive attention naturally prioritizes the prompt prefix (system instructions, initial tokens) and prompt suffix (recent tokens before generation), while attention weights across middle tokens suffer from diffusion and decay.
        **Mitigations**:
        1. Multi-stage continual pretraining on long documents with synthetic needle-in-a-haystack tasks inserted uniformly across 0% to 100% depth.
        2. YaRN temperature scaling on attention logits to prevent entropy collapse.
        3. Bidirectional pretraining objectives for encoder-decoder or prefix segments.
        """
    )

with st.expander("Q4: How does Ring Attention enable million-token contexts without OOM?"):
    st.markdown(
        """
        **Answer:**
        Ring Attention splits the `N` tokens across `P` GPUs.
        Each GPU holds `N / P` tokens of Q, K, and V.
        In each step of the ring, GPU `p` computes attention between its local `Q` block and its current `K, V` block. Concurrently, it passes its `K, V` block to neighbor `p+1` and receives a new block from `p-1`.
        Because communication is completely overlapped with matrix computation, memory per GPU remains `O(N / P)` instead of `O(N)`.
        """
    )

with st.expander("Q5: Why does naive linear RoPE interpolation hurt code generation and math reasoning?"):
    st.markdown(
        """
        **Answer:**
        Code syntax and mathematical expressions depend critically on exact, fine-grained local token ordering (e.g., distinguishing `i + 1` from `1 + i`, or exact bracket matching).
        Naive linear interpolation divides all rotation frequencies by `s`. High-frequency dimensions (which oscillate rapidly to capture adjacent token syntax) are compressed, reducing the angular distance between neighboring tokens and making the model insensitive to local syntax errors.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: OOM During Long-Context Prefill Due to Activation Materialization**
        *Root Cause*: Even with FlashAttention, naive intermediate activation caching in custom PyTorch layers can allocate gigabytes of temporary memory during 64k prefill.
        *Production Fix*: Enable activation checkpointing (gradient checkpointing) and utilize chunked prefill to cap maximum forward tensor memory.
        """
    )
    st.warning(
        """
        **Trap 2: Ignoring Attention Logit Temperature Scaling with YaRN**
        *Root Cause*: Scaling context length by 16x without adjusting attention logit scale causes Softmax distribution entropy to shift, degrading generation coherence.
        *Production Fix*: Apply the YaRN temperature correction factor `1.0 / (sqrt(t))` to Query-Key dot products before Softmax.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Needle in a Haystack False Confidence**
        *Root Cause*: Passing a single synthetic needle test (e.g., retrieving a random 5-digit number) does not guarantee the model can perform multi-hop reasoning or aggregate dispersed facts across 128k tokens.
        *Production Fix*: Evaluate using multi-hop retrieval benchmarks and long-context summarization datasets alongside synthetic NIAH.
        """
    )
    st.warning(
        """
        **Trap 4: Mismatched RoPE Base Frequency During Serving**
        *Root Cause*: Training a model with `theta_base = 500,000` (like Llama 3) but serving it via an inference engine configured with default `theta_base = 10,000`.
        *Production Fix*: Ensure `rope_theta` in `config.json` is explicitly parsed and honored by vLLM, TensorRT-LLM, or Hugging Face.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: O(N^2) Attention FLOPs"):
        st.write("Why does doubling context length quadruple attention computation?")
        st.info("Full self-attention computes an N x N similarity matrix. Doubling N yields (2N)^2 = 4N^2 dot products.")

    with st.expander("⚡ Flashcard 2: NTK Scaling Core Insight"):
        st.write("What does NTK-aware scaling adjust instead of token position indices?")
        st.info("It scales the base frequency theta of RoPE, preserving high frequencies while stretching low frequencies.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: KV Cache Growth Rate"):
        st.write("Does KV cache memory scale quadratically or linearly with sequence length?")
        st.info("Linearly O(N). Each new token appends one Key and one Value vector to the cache per head per layer.")

    with st.expander("⚡ Flashcard 4: Lost in the Middle"):
        st.write("Where in a long prompt does information retrieval performance suffer most?")
        st.info("In the middle 20% to 80% of the input text (between the prompt beginning and end).")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the dual bottlenecks of long-context LLMs: O(N^2) compute and O(N) KV cache memory.",
    "I can explain why naive RoPE position extrapolation fails due to out-of-distribution phase wrap.",
    "I can mathematically distinguish Linear Interpolation, NTK-Aware Scaling, and YaRN frequency bands.",
    "I understand why Grouped-Query Attention (GQA) is essential for serving 128k context windows.",
    "I can explain the 'Lost in the Middle' retrieval curve and how continual pretraining addresses it.",
    "I understand how Ring Attention and Chunked Prefill prevent OOM in distributed long-context inference.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Long Context Dynamics, Position Interpolation, and Scaled RoPE Architectures."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
