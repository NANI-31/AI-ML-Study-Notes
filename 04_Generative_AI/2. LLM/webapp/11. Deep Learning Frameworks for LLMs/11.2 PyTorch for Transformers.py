"""
11.2 PyTorch for Transformers
=============================
Interactive Master Lesson exploring Embeddings, Linear Projections, LayerNorm/RMSNorm,
Dropout Dynamics, Multi-Head Attention, Transformer Blocks, and Causal Loss Functions in PyTorch.
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
    page_title="11.2 PyTorch for Transformers - LLM Architecture",
    page_icon="🧩",
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
st.title("🧩 11.2 PyTorch for Transformers")
st.caption(
    "Chapter 11: Deep Learning Frameworks for LLMs | Module 2 of 4"
)

st.markdown(
    """
Constructing a production foundation Large Language Model requires translating theoretical
transformer equations into optimized PyTorch modules. From **nn.Embedding** and **bias-free nn.Linear**
to **RMSNorm**, **FlashAttention integration**, **SwiGLU MLPs**, and **fused cross-entropy losses**,
every component must balance numerical stability with high GPU Tensor Core efficiency.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & COMPONENT HIERARCHY
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & Transformer Component Hierarchy", divider="rainbow")

st.markdown(
    """
### The Module Hierarchy of Modern Decoder-Only Models
A standard modern decoder-only foundation LLM is organized as a strict hierarchical assembly of PyTorch modules:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "transformer_hierarchy",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("model", "CausalLMModel\\nTop-Level Module Container", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("embed", "Token Embedding\\nnn.Embedding(V, d)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("blocks", "nn.ModuleList([Block_0, ..., Block_{L-1}])\\nStacked Transformer Layers", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("norm_final", "Final RMSNorm\\nRMSNorm(d)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("head", "LM Head Unembedding\\nnn.Linear(d, V, bias=False)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("block_inner", "Inside Each Block:\\n- Input RMSNorm\\n- Self-Attention (Q, K, V, Out)\\n- Post-Attn RMSNorm\\n- SwiGLU MLP (Gate, Up, Down)", fillcolor="#047857", fontcolor="#F8FAFC")

arch_dot.edge("model", "embed")
arch_dot.edge("model", "blocks")
arch_dot.edge("model", "norm_final")
arch_dot.edge("model", "head")
arch_dot.edge("blocks", "block_inner", label="Repeated L Times")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Bias-Free Linear Layers**: Modern models (Llama 3, Mistral, Gemma 2, Chinchilla) completely eliminate biases (`bias=False` in all `nn.Linear` layers). Removing additive bias vectors enhances training stability, reduces parameter count, and speeds up GEMM operations.
* **Pre-RMSNorm over Post-LayerNorm**: Modern architectures replace standard LayerNorm with **RMSNorm** (Root Mean Square Normalization). Removing the mean subtraction operation cuts memory reads and delivers a 10-15% speedup without degrading accuracy.
* **Zero Dropout in Pretraining**: Early transformers applied `nn.Dropout(0.1)` everywhere. Frontier models set `dropout = 0.0` across all pretraining layers; web corpora are so vast (15T tokens) that regularization is redundant, and zero dropout maximizes arithmetic throughput.
"""
)

# Component Comparison Table
comp_df = pd.DataFrame(
    {
        "Layer Component": ["Token Embeddings", "Linear Projections", "Normalization", "Dropout", "Attention Core", "FFN Activation"],
        "Original Vaswani (2017)": ["Learned Embedding + Sinusoidal PE", "nn.Linear(bias=True)", "Post-LayerNorm (mean & std)", "Dropout(0.1)", "Standard Dot-Product Softmax", "ReLU / GELU (4 * d)"],
        "Modern Llama 3 (2024)": ["nn.Embedding (Untied, RoPE in QK)", "nn.Linear(bias=False)", "Pre-RMSNorm (std only, eps=1e-5)", "Dropout(0.0) (Disabled)", "Grouped-Query FlashAttention", "SwiGLU (8/3 * d, Gated)"],
        "PyTorch Implementation": ["nn.Embedding(vocab, d)", "nn.Linear(d_in, d_out, bias=False)", "Custom RMSNorm(dim)", "nn.Dropout(p=0.0)", "F.scaled_dot_product_attention", "silu(gate) * up"],
    }
)
st.dataframe(comp_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: EMBEDDING LAYERS
# -----------------------------------------------------------------------------
st.header("2. Embedding Layers (nn.Embedding vs. Tied Unembedding Projections)", divider="rainbow")

st.markdown(
    """
`nn.Embedding` acts as a lookup table that maps discrete token indices to continuous vectors:

```text
Input:  [Batch_Size, Sequence_Length] (Integer token IDs)
Output: [Batch_Size, Sequence_Length, Hidden_Dimension] (Dense vectors)
```

### Weight Tying in PyTorch
In weight-tied models (GPT-2, T5):
```python
self.lm_head.weight = self.embed_tokens.weight
```
* **Pros**: Saves `V * d_model * 2 bytes` of VRAM (over 1 GB in 128k vocabulary models).
* **Cons**: In modern 70B+ foundation models, input and output manifolds differ; untying them yields slightly lower perplexity and better multilingual representations.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: LINEAR LAYERS
# -----------------------------------------------------------------------------
st.header("3. Linear Layers (nn.Linear, Bias Elimination & Fused Projections)", divider="rainbow")

st.markdown(
    """
Linear layers perform affine transformations: `Y = X * W^T (+ b)`.
In modern LLMs:
* **`bias=False`**: Eliminating bias terms reduces memory transfers and stabilizes gradients.
* **Fused QKV Projections**: Instead of three separate matrix multiplies for Query, Key, and Value:
  ```python
  # Separate: 3 GEMM launches
  q = self.q_proj(x)
  k = self.k_proj(x)
  v = self.v_proj(x)

  # Fused: 1 GEMM launch
  qkv = self.qkv_proj(x) # Shape: [B, T, (H_q + 2 * H_kv) * d_k]
  ```
  A single fused GEMM launch drastically improves GPU Tensor Core utilization by reducing kernel launch latency.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: NORMALIZATION LAYERS
# -----------------------------------------------------------------------------
st.header("4. Normalization Layers (LayerNorm vs. RMSNorm vs. Gemma Scaling)", divider="rainbow")

st.markdown(
    """
### RMSNorm Mathematical Formulation (Zhang & Sennrich, 2019)
Standard LayerNorm centers inputs by subtracting the mean `mu`, then scales by variance `sigma`:
```text
LN(x) = ( (x - mu) / sqrt(sigma^2 + eps) ) * gamma + beta
```
RMSNorm hypothesizes that the scaling property (variance normalization) provides 99% of the stabilization benefit, while mean centering is redundant:

```text
RMS(x) = sqrt( 1/d * SUM_{i=1}^{d} x_i^2 + eps )
RMSNorm(x) = ( x / RMS(x) ) * gamma
```
* Eliminates mean calculation and subtraction.
* Eliminates learnable additive bias `beta`.
* Saves GPU memory bandwidth while providing identical training stability.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: REGULARIZATION & DROPOUT DYNAMICS
# -----------------------------------------------------------------------------
st.header("5. Regularization & Dropout Dynamics (Pretraining Zero-Dropout Standard)", divider="rainbow")

st.markdown(
    """
### Why Modern LLMs Abandoned Dropout
* **Historically (2017-2020)**: `nn.Dropout(p=0.1)` was randomly zeroing out 10% of activations to prevent co-adaptation.
* **Modern Reality**: Modern foundation models train on massive non-repetitive internet datasets (10T to 15T tokens). The model never encounters the same document twice, so **overfitting during pretraining does not occur**.
* **Drawbacks of Dropout**: Dropout adds computational overhead, reduces effective model capacity, and complicates multi-GPU gradient all-reduce synchronization. Setting `dropout=0.0` is now standard.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: ATTENTION LAYERS
# -----------------------------------------------------------------------------
st.header("6. Attention Layers (Scaled Dot-Product Attention & Multi-Head Projections)", divider="rainbow")

st.markdown(
    """
PyTorch 2.0 introduced `F.scaled_dot_product_attention` (SDPA), which automatically dispatches
to the fastest available hardware backend:
1. **FlashAttention-2 / FlashAttention-3**: Fused SRAM tiling for modern NVIDIA GPUs.
2. **Memory-Efficient Attention (xFormers)**: Cutlass-based block attention.
3. **C++ Native Vectorized Math**: CPU and legacy GPU fallback.

```python
# Modern SDPA invocation:
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,
    dropout_p=0.0,
    is_causal=True
)
```
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: TRANSFORMER BLOCKS & RESIDUAL STREAMS
# -----------------------------------------------------------------------------
st.header("7. Transformer Blocks & Residual Stream Integration", divider="rainbow")

st.markdown(
    """
A complete decoder block combines self-attention with a gated Feed-Forward Network (FFN):

```text
Residual Stream:
x = x + SelfAttention(RMSNorm(x))
x = x + SwiGLU_MLP(RMSNorm(x))
```

### SwiGLU Gated Multi-Layer Perceptron (Shazeer, 2020)
Replaces standard two-layer MLPs (`ReLU(x * W1) * W2`) with a three-matrix gated bilinear structure:

```text
SwiGLU(x) = ( SiLU(x * W_gate) * (x * W_up) ) * W_down
```
To preserve the parameter count of standard 4x MLPs, the intermediate hidden dimension is scaled to `8/3 * d_model`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: LOSS FUNCTIONS
# -----------------------------------------------------------------------------
st.header("8. Loss Functions (Causal Cross-Entropy & Label Smoothing)", divider="rainbow")

st.markdown(
    """
Causal language modeling aligns predictions so step `t` predicts target `t+1`:

```python
# Efficient shifted loss implementation
shift_logits = logits[..., :-1, :].contiguous()
shift_targets = target_ids[..., 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_targets.view(-1),
    ignore_index=-100
)
```
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE COMPLETE DECODER-ONLY TRANSFORMER
# -----------------------------------------------------------------------------
st.header("9. Production-Grade Complete Decoder-Only Transformer (PyTorch Implementation)", divider="rainbow")

st.markdown(
    """
A complete, standalone PyTorch module implementing a mini-Llama 3 architecture with RMSNorm, SwiGLU, and SDPA:
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

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class ModernDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        hidden_dim = int(8.0 * d_model / 3.0)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # 1. Pre-RMSNorm Attention
        norm_x = self.attn_norm(x)
        q = self.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # PyTorch 2.0 SDPA with causal flash attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn_out)
        
        # 2. Pre-RMSNorm MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x

class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([ModernDecoderBlock(d_model, num_heads) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.token_embeddings(input_ids)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE TRANSFORMER BLOCK WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive Transformer Block & Forward Activation Workbench", divider="rainbow")

st.markdown(
    """
Simulate forward activation tensor shapes and inspect parameter allocations across
the **Embedding**, **Attention**, **SwiGLU MLP**, and **LM Head** sub-blocks.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    sim_d_model = st.select_slider("Hidden Dimension (d_model)", options=[512, 1024, 2048, 4096], value=1024)
    sim_heads = st.select_slider("Attention Heads", options=[4, 8, 16, 32], value=8)
with col_ctrl2:
    sim_layers = st.slider("Number of Transformer Blocks (L)", min_value=2, max_value=32, value=8)
    sim_vocab = st.select_slider("Vocabulary Size", options=[32000, 50257, 128256], value=32000)

# Parameter computations per block
head_dim = sim_d_model // sim_heads
qkv_params = 3 * (sim_d_model * sim_d_model)
out_proj_params = sim_d_model * sim_d_model
hidden_dim = int(8.0 * sim_d_model / 3.0)
mlp_params = 3 * (sim_d_model * hidden_dim)  # gate, up, down
norm_params = 2 * sim_d_model  # attn_norm + mlp_norm
per_block_params = qkv_params + out_proj_params + mlp_params + norm_params

total_model_params = (sim_vocab * sim_d_model) + (sim_layers * per_block_params) + sim_d_model + (sim_vocab * sim_d_model)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total Model Parameters", f"{total_model_params / 1e6:.1f} Million")
with col_m2:
    st.metric("Params per Block", f"{per_block_params / 1e6:.2f} Million")
with col_m3:
    st.metric("SwiGLU Hidden Dim", f"{hidden_dim} channels")

arch_breakdown_df = pd.DataFrame(
    {
        "Sub-Block Component": ["Token Embeddings", f"Self-Attention ({sim_layers} blocks)", f"SwiGLU MLP ({sim_layers} blocks)", f"RMSNorms ({sim_layers * 2 + 1})", "LM Head Unembedding"],
        "Parameter Count": [
            sim_vocab * sim_d_model,
            sim_layers * (qkv_params + out_proj_params),
            sim_layers * mlp_params,
            (sim_layers * 2 + 1) * sim_d_model,
            sim_vocab * sim_d_model,
        ],
        "Percentage Share (%)": [
            np.round(100.0 * (sim_vocab * sim_d_model) / total_model_params, 1),
            np.round(100.0 * (sim_layers * (qkv_params + out_proj_params)) / total_model_params, 1),
            np.round(100.0 * (sim_layers * mlp_params) / total_model_params, 1),
            np.round(100.0 * ((sim_layers * 2 + 1) * sim_d_model) / total_model_params, 1),
            np.round(100.0 * (sim_vocab * sim_d_model) / total_model_params, 1),
        ],
    }
)

st.subheader("Architectural Parameter Allocation")
st.dataframe(arch_breakdown_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is bias=False used in all linear layers of modern transformers?"):
    st.markdown(
        """
        **Answer:**
        1. **Parameter Efficiency**: In models with 70B+ parameters, bias vectors add millions of unnecessary parameters.
        2. **Numerical Stability**: Removing additive bias constants prevents slow coordinate drift and improves learning rate robustness.
        3. **Kernel Speed**: Pure matrix multiplications `Y = X * W^T` execute faster on Tensor Cores than operations requiring element-wise bias additions.
        """
    )

with st.expander("Q2: Why did SwiGLU replace standard ReLU/GELU activations in foundation models?"):
    st.markdown(
        """
        **Answer:**
        SwiGLU computes `(SiLU(x * W_gate) * (x * W_up)) * W_down`.
        The element-wise gating mechanism allows the network to dynamically scale or suppress individual activation channels based on input context (similar to an attention gate).
        Empirically (Shazeer, 2020), SwiGLU consistently improves downstream perplexity and benchmark reasoning scores over standard GELU activations with equal compute budgets.
        """
    )

with st.expander("Q3: What is the primary operational advantage of PyTorch 2.0 SDPA?"):
    st.markdown(
        """
        **Answer:**
        `F.scaled_dot_product_attention` automatically selects the optimal hardware kernel (FlashAttention-2, Memory-Efficient Attention, or C++ vectorized math) based on tensor shapes and GPU architecture.
        It computes attention directly in GPU SRAM without materializing the `N x N` attention matrix in high-bandwidth memory (HBM), reducing memory consumption from `O(N^2)` to `O(N)` and accelerating speed by 2x to 4x.
        """
    )

with st.expander("Q4: Why does RMSNorm execute faster than standard LayerNorm?"):
    st.markdown(
        """
        **Answer:**
        Standard LayerNorm requires two statistical passes over the input: one to compute the mean, and a second to compute the variance.
        RMSNorm computes only the root-mean-square (variance), completely omitting mean calculation and mean subtraction.
        This cuts GPU memory reads and eliminates synchronization barriers across the reduction dimension, delivering a 10% to 50% faster forward/backward pass.
        """
    )

with st.expander("Q5: Why is Dropout set to 0.0 in modern foundation model pretraining?"):
    st.markdown(
        """
        **Answer:**
        Dropout was invented to prevent overfitting on small datasets.
        In modern pretraining, models train on 15 trillion tokens of diverse web text in a single epoch. The model never encounters identical training documents, making pretraining overfitting impossible.
        Disabling dropout preserves full model capacity, speeds up GEMM operations, and prevents distributed all-reduce communication noise.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Forgetting is_causal=True in SDPA**
        *Root Cause*: Calling `F.scaled_dot_product_attention(q, k, v)` without setting `is_causal=True` or providing a causal mask.
        *Production Fix*: The attention kernel executes bidirectional attention, allowing future tokens to attend to subsequent tokens and causing complete training failure.
        """
    )
    st.warning(
        """
        **Trap 2: Mismatched Reshaping in Multi-Head Attention**
        *Root Cause*: Permuting heads without calling `.contiguous()` before `.view()`.
        *Production Fix*: Raises `RuntimeError: view size is not compatible with stride`. Always write: `x.transpose(1, 2).contiguous().view(B, T, C)`.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Target ID Shift Misalignment in Cross-Entropy Loss**
        *Root Cause*: Passing unshifted inputs and targets into `F.cross_entropy`.
        *Production Fix*: Shift logits: `logits[:, :-1, :]` and shift targets: `targets[:, 1:]`.
        """
    )
    st.warning(
        """
        **Trap 4: Missing Epsilon in Custom RMSNorm**
        *Root Cause*: Evaluating `torch.rsqrt(variance)` without adding `eps`.
        *Production Fix*: If an input token is all zeros (e.g. padding), variance is zero, producing `1 / sqrt(0) = inf` and destroying the model. Always add `eps = 1e-6`.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: RMSNorm vs LayerNorm"):
        st.write("What statistical step does RMSNorm eliminate compared to LayerNorm?")
        st.info("Mean calculation and mean centering (x - mu).")

    with st.expander("⚡ Flashcard 2: SwiGLU Hidden Ratio"):
        st.write("What is the typical hidden dimension ratio for SwiGLU in modern LLMs?")
        st.info("8/3 * d_model (maintains parameter count of standard 4x MLPs).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Linear Bias Standard"):
        st.write("Are biases included in linear projections in modern foundation models?")
        st.info("No, bias=False across all linear layers.")

    with st.expander("⚡ Flashcard 4: PyTorch SDPA Kernel"):
        st.write("What kernel backend does F.scaled_dot_product_attention prioritize on Hopper/Ampere?")
        st.info("FlashAttention-2 / FlashAttention-3.")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the modular component hierarchy of a modern decoder-only foundation LLM.",
    "I can explain why modern architectures eliminate biases (bias=False) in all nn.Linear layers.",
    "I can derive the mathematical formulation of RMSNorm and explain its speed advantage over LayerNorm.",
    "I know how the SwiGLU gated activation functions and why its hidden dimension is 8/3 * d_model.",
    "I understand why Dropout is set to 0.0 in modern large-scale foundation pretraining.",
    "I can implement a complete, idiomatic Transformer block using PyTorch 2.0 SDPA.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered PyTorch Transformer Components, RMSNorm, SwiGLU, and SDPA Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
