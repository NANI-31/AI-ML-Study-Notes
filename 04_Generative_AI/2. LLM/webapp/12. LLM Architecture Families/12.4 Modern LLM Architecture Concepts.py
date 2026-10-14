"""
12.4 Modern LLM Architecture Concepts
=====================================
Interactive Master Lesson exploring RMSNorm, SwiGLU Gated MLPs, Rotary Position Embeddings (RoPE),
KV Cache Dynamics, Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Mixture of Experts (MoE).
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
    page_title="12.4 Modern LLM Architecture Concepts - LLM Architecture",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS for aesthetic typography & dark theme styling
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
st.title("🚀 12.4 Modern LLM Architecture Concepts")
st.caption(
    "Chapter 12: LLM Architecture Families | Module 4 of 4"
)

st.markdown(
    """
Between 2020 and 2026, the standard transformer architecture underwent a profound structural transformation.
Modern open-weight and frontier foundation models (Llama 3, Mistral, Gemma 2, DeepSeek-V2/V3, Qwen 2.5)
have converged on an ultra-optimized blueprint: **RMSNorm**, **SwiGLU Gated MLPs**, **Rotary Position Embeddings (RoPE)**,
**Grouped-Query Attention (GQA)**, and **Sparse Mixture of Experts (MoE)**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & MODERN FOUNDATION BLUEPRINT
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Modern Foundation Blueprint", divider="rainbow")

st.markdown(
    """
### The 2024-2026 State-of-the-Art Transformer Architecture
Every modern foundation model combines five standardized architectural enhancements:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "modern_transformer_stack",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("input", "Input Token IDs [B, T]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("embed", "Token Embedding (No Additive PE)\\nW_emb in R^[V, d]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("rmsnorm1", "1. Pre-RMSNorm\\nVariance scaling only (eps=1e-5)", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("gqa", "2. Grouped-Query Attention (GQA)\\nRoPE on Q and K | 8x KV Compression", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("rmsnorm2", "3. Post-Attention RMSNorm\\nStabilizes residual stream", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("swiglu", "4. SwiGLU Gated MLP / MoE\\n8/3 * d expansion | Top-2 Gating", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("final_norm", "5. Final RMSNorm + LM Head\\nLogit Soft-Capping (cap=30.0)", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("input", "embed")
arch_dot.edge("embed", "rmsnorm1")
arch_dot.edge("rmsnorm1", "gqa")
arch_dot.edge("gqa", "rmsnorm2")
arch_dot.edge("rmsnorm2", "swiglu")
arch_dot.edge("swiglu", "final_norm")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Convergence on GQA**: Multi-Head Attention (MHA) has been universally abandoned in 2024-2026 foundation models. Every model from Llama 3 8B/70B to Gemma 2 and Mistral Large uses **Grouped-Query Attention (GQA)** with 8 Key-Value heads, slashing inference VRAM footprint by 4x to 8x.
* **Sparse Mixture of Experts (MoE)**: Frontier models (DeepSeek-V2, Mixtral 8x22B, Grok-1) scale total parameters to hundreds of billions while routing each token through only a fraction of active parameters (e.g. activating 2 of 8 experts), achieving 70B performance at 13B inference latency!
"""
)

# Modern LLM Architectural Blueprint Matrix Table
blueprint_matrix_df = pd.DataFrame(
    {
        "Architectural Component": ["Normalization", "Positional Information", "Attention Routing", "FFN Activation", "Inference KV Cache", "Parameter Routing"],
        "Legacy Vaswani (2017)": ["Post-LayerNorm", "Additive Sinusoidal PE", "Multi-Head Attention (MHA)", "Standard ReLU / GELU", "Uncompressed MHA KV Cache", "Dense (100% Active)"],
        "Modern Frontier (2024-2026)": ["Pre-RMSNorm (No mean, no bias)", "Rotary Position Embeddings (RoPE)", "Grouped-Query Attention (GQA)", "SwiGLU Bilinear Gating", "Paged GQA + Quantization (FP8)", "Sparse MoE (Top-2 / Shared)"],
        "Adopted By": ["Llama 3, Mistral, Gemma, Qwen", "Llama 3, Mistral, Gemma, Qwen", "Llama 3, Mistral, Gemma, Qwen", "Llama 3, Mistral, Gemma, Qwen", "vLLM, TensorRT-LLM, SGLang", "Mixtral, DeepSeek-V3, DBRX"],
    }
)
st.dataframe(blueprint_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: RMSNORM
# -----------------------------------------------------------------------------
st.header("2. RMSNorm (Root Mean Square Normalization & Pre-LN Stability)", divider="rainbow")

st.markdown(
    """
Zhang & Sennrich (2019) introduced **RMSNorm** by proving that mean centering in LayerNorm is redundant:

```text
RMS(x) = sqrt( 1/d * SUM_{i=1}^{d} x_i^2 + eps )
RMSNorm(x) = ( x / RMS(x) ) * gamma
```
* **Computational Advantage**: Eliminates mean calculation `mu` and mean subtraction `(x - mu)`. Eliminates learnable additive bias `beta`.
* **Hardware Efficiency**: Requires 1 single memory pass over the input vector instead of 2 passes, delivering a **10% to 50% speedup** on GPU memory-bandwidth bound layers.
* **Pre-LN Placement**: Placed strictly before Attention and MLP sub-blocks, ensuring the residual stream maintains an unhindered gradient highway.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: SWIGLU ACTIVATION DYNAMICS
# -----------------------------------------------------------------------------
st.header("3. SwiGLU Activation Dynamics & Gated Bilinear Units (Shazeer, 2020)", divider="rainbow")

st.markdown(
    """
Noam Shazeer introduced **SwiGLU** (Swish-Gated Linear Unit), replacing standard activation functions:

```text
SwiGLU(x) = ( SiLU(x * W_gate) * (x * W_up) ) * W_down
```
Where `SiLU(x) = x * sigmoid(x)`.

### Why Bilinear Gating Outperforms GELU
* The `SiLU(x * W_gate)` branch acts as an adaptive continuous valve, dynamically scaling the feature channels projected by `(x * W_up)`.
* **The 8/3 Scaling Law**: Because SwiGLU uses three weight matrices (`W_gate`, `W_up`, `W_down`) instead of two (`W1`, `W2`), the intermediate hidden dimension is set to:
  ```text
  d_ff = round( (8/3) * d_model )
  ```
  This preserves the exact parameter count of a standard 4x FFN while delivering superior language modeling loss.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: GATED MLPS
# -----------------------------------------------------------------------------
st.header("4. Gated MLPs & Dynamic Channel Routing", divider="rainbow")

st.markdown(
    """
Gated MLPs introduce multiplicative interaction between linear projections:
* In standard MLPs, activations are computed statically through monotonic non-linearities: `Act(X * W1) * W2`.
* In Gated MLPs, the model learns a **bilinear product** between two distinct projections of the input, enabling the network to implement conditional logic and routing directly within the feed-forward layer.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: ROTARY POSITION EMBEDDINGS (ROPE)
# -----------------------------------------------------------------------------
st.header("5. Rotary Position Embeddings (RoPE 2D Orthogonal Rotations)", divider="rainbow")

st.markdown(
    """
RoPE (Su et al., 2021) applies a multiplicative 2D orthogonal rotation to Query and Key vectors:

```text
q_m = R_m * q,    k_n = R_n * k
<q_m, k_n> = q^T * (R_m^T * R_n) * k = q^T * R_{n - m} * k
```
* **Inner Product Property**: The resulting attention similarity depends strictly on the relative token separation `(n - m)`.
* **Zero Norm Distortion**: Because rotation matrices are orthogonal (`||R x|| = ||x||`), RoPE does not alter vector lengths or pollute semantic token embeddings.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: KEY-VALUE (KV) CACHE MEMORY SCALING
# -----------------------------------------------------------------------------
st.header("6. Key-Value (KV) Cache Memory Scaling & Autoregressive Serving", divider="rainbow")

st.markdown(
    """
During autoregressive text generation, the KV cache stores past Keys and Values to prevent `O(N^2)` recomputation:

```text
KV_Cache_Size_Bytes = 2 * Batch_Size * Sequence_Length * Num_Layers * Num_KV_Heads * Head_Dim * Bytes_Per_Element
```

For Llama 3 70B (`Layers = 80`, `H_kv = 8`, `d_k = 128`, `FP16 = 2 bytes`):
* Single token: `2 * 1 * 1 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB / token`.
* At `128,000` tokens: **40.96 GB VRAM** for a single sequence!
This extreme memory consumption motivated the transition to Multi-Query and Grouped-Query Attention.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: GROUPED-QUERY ATTENTION (GQA)
# -----------------------------------------------------------------------------
st.header("7. Grouped-Query Attention (GQA Ratio & Memory-Compute Balance)", divider="rainbow")

st.markdown(
    """
Ainslie et al. (2023) introduced **Grouped-Query Attention (GQA)** as the ideal Pareto frontier between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):

```text
MHA:  32 Query Heads <--> 32 KV Heads   (1:1 Ratio, High Memory)
GQA:  32 Query Heads <-->  8 KV Heads   (4:1 Ratio, Llama 3 / Mistral)
MQA:  32 Query Heads <-->  1 KV Head    (32:1 Ratio, Falcon)
```

### The 8x KV Cache Reduction
In Llama 3 (32 Q heads, 8 KV heads):
* Every group of 4 Query heads shares a single Key-Value head.
* **Cuts KV cache memory size and memory bandwidth transfers by 4x**.
* Maintains near-identical perplexity and reasoning benchmark scores to full MHA.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: MULTI-QUERY ATTENTION (MQA)
# -----------------------------------------------------------------------------
st.header("8. Multi-Query Attention (MQA Single-Head Compression)", divider="rainbow")

st.markdown(
    """
Shazeer (2019) introduced **Multi-Query Attention (MQA)**:
* All `H_q` Query heads share a single global Key head and a single global Value head (`H_kv = 1`).
* **Maximum Memory Compression**: Slashes KV cache memory by up to 32x or 64x.
* **Downside**: In massive models (>70B), compressing all heads to a single KV head causes slight capacity loss on complex multi-document reasoning tasks. Hence, the industry converged on GQA as the optimal middle ground.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: MIXTURE OF EXPERTS (MOE)
# -----------------------------------------------------------------------------
st.header("9. Mixture of Experts (MoE Routing, Top-k Gating & Shared Experts)", divider="rainbow")

st.markdown(
    """
Sparse Mixture of Experts replaces the dense feed-forward network with `E` independent expert sub-networks
and a learned router:

```text
G(x) = Softmax( TopK( x * W_gate, k ) )
Output(x) = SUM_{i in TopK} G(x)_i * Expert_i(x)
```

### Modern MoE Paradigms (DeepSeek-V2 / Mixtral)
* **Top-2 Routing**: E.g. Mixtral 8x7B routes each token through the top 2 of 8 experts. Total parameters: 47B; active parameters per token: **13B**!
* **Fine-Grained Experts + Shared Experts (DeepSeek-V2/V3)**:
  * Partitions standard experts into smaller granular pieces (e.g. 160 tiny experts).
  * Allocates a set of **always-active Shared Experts** that capture universal language patterns, allowing routed experts to specialize strictly in specialized domains.
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE MODERN FOUNDATION ARCHITECTURE
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Modern Foundation Architecture (GQA, SwiGLU, MoE & RMSNorm)", divider="rainbow")

st.markdown(
    """
A complete PyTorch reference implementation combining GQA, RMSNorm, SwiGLU, and Sparse Top-2 MoE routing:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class SparseTop2MoE(nn.Module):
    # Modern Top-2 Sparse Mixture of Experts layer
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        hidden_dim = int(8.0 * d_model / 3.0)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, d_model]
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])
        
        # Compute router logits and top-k weights
        router_logits = self.router(flat_x)
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Dispatch tokens to experts
        out = torch.zeros_like(flat_x)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            weight = weights[:, i].unsqueeze(-1)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    token_slice = flat_x[mask]
                    expert_out = self.experts[e](token_slice)
                    out[mask] += weight[mask] * expert_out
                    
        return out.view(orig_shape)

class GroupedQueryAttention(nn.Module):
    # Production GQA layer with 4:1 query-to-kv ratio
    def __init__(self, d_model: int, num_q_heads: int = 32, num_kv_heads: int = 8):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads to match query heads (GQA broadcast)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE MODERN LLM BLUEPRINT & MOE WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Modern LLM Blueprint & MoE Routing Simulator Workbench", divider="rainbow")

st.markdown(
    """
Model the KV cache reduction of **GQA vs. MHA** and simulate **Sparse MoE Top-k token routing**.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    attn_type = st.selectbox("Attention Architecture", options=["Multi-Head Attention (MHA: 32 Q, 32 KV)", "Grouped-Query Attention (GQA: 32 Q, 8 KV)", "Multi-Query Attention (MQA: 32 Q, 1 KV)"], index=1)
    seq_len_sim = st.select_slider("Serving Context Length (Tokens)", options=[2048, 4096, 8192, 32768, 128000], value=8192)
with col_ctrl2:
    moe_experts_count = st.slider("Total MoE Experts (E)", min_value=4, max_value=16, value=8, step=2)
    moe_top_k = st.slider("Active Experts per Token (k)", min_value=1, max_value=4, value=2)

# Calculate KV Cache
if "MHA" in attn_type:
    kv_heads = 32
elif "GQA" in attn_type:
    kv_heads = 8
else:
    kv_heads = 1

layers_sim = 32
head_dim_sim = 128
bytes_per_tok = 2 * 1 * layers_sim * kv_heads * head_dim_sim * 2  # FP16
total_kv_mb = (bytes_per_tok * seq_len_sim) / (1024 ** 2)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("KV Cache per Stream", f"{total_kv_mb:.1f} MB")
with col_m2:
    mha_baseline_mb = ((2 * 1 * layers_sim * 32 * head_dim_sim * 2) * seq_len_sim) / (1024 ** 2)
    st.metric("VRAM Savings Factor", f"{mha_baseline_mb / max(0.1, total_kv_mb):.1f}x vs MHA")
with col_m3:
    st.metric("Active Compute Ratio (MoE)", f"{(moe_top_k / moe_experts_count) * 100:.1f}%")

st.subheader("Architectural Comparison: KV Cache Scaling Across Context Windows")
context_windows = [2048, 4096, 8192, 16384, 32768, 65536, 128000]
mha_curve = [((2 * layers_sim * 32 * head_dim_sim * 2) * c) / (1024 ** 3) for c in context_windows]
gqa_curve = [((2 * layers_sim * 8 * head_dim_sim * 2) * c) / (1024 ** 3) for c in context_windows]
mqa_curve = [((2 * layers_sim * 1 * head_dim_sim * 2) * c) / (1024 ** 3) for c in context_windows]

kv_scaling_df = pd.DataFrame(
    {
        "Context Tokens": context_windows,
        "MHA (32 KV Heads) GB": np.round(mha_curve, 2),
        "GQA (8 KV Heads) GB": np.round(gqa_curve, 2),
        "MQA (1 KV Head) GB": np.round(mqa_curve, 2),
    }
)
st.line_chart(kv_scaling_df.set_index("Context Tokens"))

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why did GQA supersede MHA in all modern open-weight LLMs?"):
    st.markdown(
        """
        **Answer:**
        In standard MHA, each query head maintains its own Key and Value head.
        At 128k context lengths, the KV cache consumes over 40GB per batch, making large-batch serving impossible.
        GQA groups query heads (e.g. 4 query heads per 1 KV head), slashing KV cache memory and bandwidth transfers by **4x to 8x** while matching full MHA perplexity and benchmark reasoning scores.
        """
    )

with st.expander("Q2: What is the computational intuition behind SwiGLU 8/3 expansion?"):
    st.markdown(
        """
        **Answer:**
        A standard MLP has 2 matrices (`W1 in R^[d, 4d]`, `W2 in R^[4d, d]`), totaling `8 * d^2` parameters.
        SwiGLU has 3 matrices (`W_gate`, `W_up`, `W_down`), each of shape `[d, d_ff]`. Total parameters = `3 * d * d_ff`.
        To match `8 * d^2`:
        `3 * d * d_ff = 8 * d^2  ==>  d_ff = (8/3) * d approx 2.67 * d`.
        This guarantees that SwiGLU retains the exact same parameter budget and arithmetic cost as a standard 4x MLP.
        """
    )

with st.expander("Q3: How does Sparse MoE achieve high parameter counts with low latency?"):
    st.markdown(
        """
        **Answer:**
        MoE separates model capacity (parameter count) from inference compute (FLOPs per token).
        In Mixtral 8x7B (47B total parameters), a learned router evaluates each token and routes it through only the top 2 of 8 experts.
        Because each token only executes FLOPs for 2 experts, inference compute matches a small **13B parameter model**, while parameter knowledge capacity matches a much larger network.
        """
    )

with st.expander("Q4: What is Expert Routing Collapse in MoE and how is it prevented?"):
    st.markdown(
        """
        **Answer:**
        In early training, the router might arbitrarily favor 1 or 2 experts. Because these experts receive more tokens, they receive more gradient updates and learn faster, causing the router to send even more tokens to them in a self-reinforcing loop. The remaining experts starve and never learn.
        **Mitigation**: Adding an auxiliary **Load Balancing Loss** (`alpha * sum(f_i * P_i)`) that penalizes uneven token distribution across experts.
        """
    )

with st.expander("Q5: What is the purpose of Shared Experts in modern MoE architectures (DeepSeek-V2/V3)?"):
    st.markdown(
        """
        **Answer:**
        In traditional MoE (Mixtral), all experts are routed. Basic grammatical syntax and common language tokens are redundantly memorized across multiple experts, wasting capacity.
        DeepSeek introduces dedicated **Shared Experts** that are always active for all tokens to handle universal language modeling, allowing the routed experts to specialize strictly in niche knowledge and reasoning domains.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Memory Imbalance in Distributed MoE (Expert Parallelism)**
        *Root Cause*: If tokens are routed unevenly, GPUs hosting popular experts run out of memory (OOM) or stall the cluster.
        *Production Fix*: Implement token dropping limits (capacity factor) or deploy auxiliary load balancing loss during pretraining.
        """
    )
    st.warning(
        """
        **Trap 2: Naive KV Head Broadcasting in Custom Kernels**
        *Root Cause*: Copying and allocating memory for repeated KV heads before attention.
        *Production Fix*: Modern FlashAttention supports GQA natively without broadcasting or copying tensors in memory. Always pass un-repeated KV tensors to native GQA kernels.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Omitting Epsilon in RMSNorm with Half-Precision**
        *Root Cause*: Setting `eps = 1e-12` in FP16/BF16 RMSNorm.
        *Production Fix*: Causes division by zero on zero-padded vectors. Always set `eps = 1e-5` or `1e-6`.
        """
    )
    st.warning(
        """
        **Trap 4: Miscalculating SwiGLU Intermediate Dimensions**
        *Root Cause*: Setting `hidden_dim = 4 * d_model` in SwiGLU, accidentally increasing FFN parameter count by 50%.
        *Production Fix*: Always apply the `8/3 * d_model` formula (and round to the nearest multiple of 256 for Tensor Core alignment).
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: GQA Query-to-KV Ratio in Llama 3"):
        st.write("What is the ratio of Query heads to KV heads in Llama 3 70B?")
        st.info("4:1 (64 Query heads to 8 Key-Value heads).")

    with st.expander("⚡ Flashcard 2: SwiGLU Activation Formula"):
        st.write("What is the mathematical formulation of SwiGLU?")
        st.info("w_down( silu(w_gate(x)) * w_up(x) ).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: MoE Load Balancing Loss"):
        st.write("Why is load balancing loss added to MoE training?")
        st.info("To prevent expert routing collapse and ensure all experts receive training tokens.")

    with st.expander("⚡ Flashcard 4: RMSNorm vs LayerNorm Formula"):
        st.write("What is the core difference between RMSNorm and LayerNorm?")
        st.info("RMSNorm eliminates mean calculation and mean centering, normalizing solely by root-mean-square variance.")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the modern foundation LLM blueprint: RMSNorm, SwiGLU, RoPE, GQA, and MoE.",
    "I can explain why RMSNorm executes faster than standard LayerNorm.",
    "I understand why SwiGLU sets its intermediate hidden dimension to 8/3 * d_model.",
    "I know how Grouped-Query Attention (GQA) reduces KV cache memory by 4x to 8x.",
    "I understand the difference between Multi-Head Attention, Grouped-Query Attention, and Multi-Query Attention.",
    "I can explain how Sparse Mixture of Experts (MoE) decouples parameter capacity from inference latency.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Modern LLM Architecture Concepts, GQA, SwiGLU, RMSNorm, and Mixture of Experts."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
