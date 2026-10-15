"""
7.2 Positional Encoding Methods
===============================
Interactive Master Lesson exploring Sinusoidal, Learned, Relative, RoPE,
and ALiBi positional encoding paradigms in modern Large Language Models.
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
    page_title="7.2 Positional Encoding Methods - LLM Architecture",
    page_icon="📐",
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
st.title("📐 7.2 Positional Encoding Methods")
st.caption(
    "Chapter 7: Positional Encoding and Position Information | Module 2 of 3"
)

st.markdown(
    """
In transformer self-attention, the dot-product similarity between query and key vectors
is inherently permutation-equivariant. Without explicit injection of coordinate or distance
information, the model cannot distinguish between identical words arranged in different orders.
Over the evolution of Large Language Models, positional encoding has transformed from
static additive signals into dynamic inner-product rotations (RoPE) and causal attention slope biases (ALiBi).
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & PARADIGM SHIFT
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & Paradigm Shift", divider="rainbow")

st.markdown(
    """
### The Evolutionary Trajectory of Position Representation
The history of positional encoding reflects a fundamental architectural realization:
**language understanding relies on relative pairwise distances between tokens rather than absolute index values**.
"""
)

# Architectural Graphviz Diagram
arch_dot = graphviz.Digraph(
    "pos_encoding_paradigm",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node(
    "vaswani",
    "Vaswani et al. (2017)\\nSinusoidal Additive PE\\nFixed Frequency Waves",
    fillcolor="#1E293B",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "learned",
    "Devlin / Radford (2018-2019)\\nLearned Absolute Pos\\nBERT, GPT-2, GPT-3",
    fillcolor="#1E293B",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "relative",
    "Shaw / Raffel (2018-2020)\\nT5 Relative Attention\\nDistance Bias Tables",
    fillcolor="#1E293B",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "rope",
    "Su et al. (2021-Present)\\nRotary Position Embeddings (RoPE)\\nLlama 2/3, Mistral, Gemma, Qwen",
    fillcolor="#0369A1",
    fontcolor="#F8FAFC",
)
arch_dot.node(
    "alibi",
    "Press et al. (2021)\\nAttention with Linear Biases (ALiBi)\\nMPT, BLOOM, Falcon",
    fillcolor="#047857",
    fontcolor="#F8FAFC",
)

arch_dot.edge("vaswani", "learned", label="Data-driven params")
arch_dot.edge("learned", "relative", label="Relative shift")
arch_dot.edge("vaswani", "rope", label="Frequency rotation in QK space")
arch_dot.edge("relative", "alibi", label="Non-parametric slope bias")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Modern state-of-the-art foundation LLMs (Llama 3, Mistral, Gemma 2, DeepSeek-V2, Qwen 2.5)
have almost entirely universally converged upon **Rotary Position Embeddings (RoPE)**.
Why?
* **Zero Input Embedding Pollution**: Absolute positional embeddings pollute the bottom token embeddings via vector addition. RoPE modifies Query and Key vectors directly within self-attention heads.
* **Inner Product Geometry**: The dot product between rotated vectors satisfies `(R_m q)^T (R_n k) = q^T R_{n-m} k`, which depends strictly on relative offset `(m - n)`.
* **Long-Context Extensibility**: RoPE allows base-frequency scaling (NTK-aware, YaRN, RoPE frequency interpolation) to expand 4k context windows to 128k or 1M tokens with minimal fine-tuning.
"""
)

# Comparison Table
pos_matrix_df = pd.DataFrame(
    {
        "Method": ["Sinusoidal", "Learned Absolute", "T5 Relative Bias", "RoPE", "ALiBi"],
        "Applied At": ["Input Layer (Sum)", "Input Layer (Sum)", "Attention Logits", "Q and K Vectors", "Attention Logits"],
        "Form": ["Deterministic Waves", "Trainable Matrix", "Trainable Scalar Table", "Orthogonal 2D Rotations", "Static Slope Bias (-m * d)"],
        "Extrapolation Ability": ["Weak (> L_train fails)", "Zero (Hard Max Length)", "Moderate (Bucketed)", "High (With Scaling)", "High (Direct Extrapolation)"],
        "Major LLMs Used": ["Original Transformer", "BERT, GPT-2, GPT-3", "T5, Flan-T5", "Llama 1/2/3, Mistral, Qwen", "MPT-7B/30B, BLOOM"],
    }
)
st.dataframe(pos_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: SINUSOIDAL POSITIONAL ENCODING
# -----------------------------------------------------------------------------
st.header("2. Sinusoidal Positional Encoding (Vaswani et al., 2017)", divider="rainbow")

st.markdown(
    """
Vaswani et al. introduced fixed sinusoidal waves using sine and cosine functions across geometric frequency progressions.
For token index `pos` and channel dimension `i in [0, d_model / 2)`:

```text
PE(pos, 2i)     = sin(pos / (10000 ^ (2i / d_model)))
PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d_model)))
```

### Key Mathematical Characteristics
1. **Geometric Wavelength Spectrum**: Wavelengths form a geometric progression from `2 * pi` to `10000 * 2 * pi`. Low dimensions cycle rapidly across adjacent tokens; high dimensions cycle slowly across entire sequence spans.
2. **Linear Shift Invariance**: By standard trigonometric angle addition:
   `cos(alpha + beta) = cos(alpha)cos(beta) - sin(alpha)sin(beta)`
   `sin(alpha + beta) = sin(alpha)cos(beta) + cos(alpha)sin(beta)`
   There exists a fixed linear transformation matrix `M_k` such that `PE(pos + k) = M_k * PE(pos)`. This property theoretically allows the attention mechanism to attend to relative positions.
3. **Decay with Distance**: The inner product `PE(pos)^T PE(pos + k)` naturally decreases as the relative separation `k` increases.
"""
)

col_sin1, col_sin2 = st.columns(2)
with col_sin1:
    st.markdown(
        """
        **Strengths:**
        * Deterministic, zero learnable parameters.
        * Smooth decay over token separation.
        * Generalizes across any sequence length without retraining parameters.
        """
    )
with col_sin2:
    st.markdown(
        """
        **Limitations & Failure Modes in Modern LLMs:**
        * **Input Corruption**: Directly adding `PE` to token embeddings distorts semantic clustering in early layers.
        * **Attention Dilution**: After layer normalization and dense projections `W_Q, W_K`, absolute position signals fade across deep 80-layer networks.
        * **Poor Long-Range Extrapolation**: Empirically, attention weights fail when evaluated on sequences longer than training lengths without fine-tuning.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 3: LEARNED POSITIONAL EMBEDDINGS
# -----------------------------------------------------------------------------
st.header("3. Learned Positional Embeddings (BERT, GPT-2)", divider="rainbow")

st.markdown(
    """
Rather than using static trigonometric formulas, models like BERT, GPT-2, and GPT-3 allocate a dedicated learnable parameter matrix:

```text
W_pos in R^[L_max, d_model]
x_input[i] = Token_Embeddings[x_i] + W_pos[i]
```

Where `L_max` is the pre-configured maximum context length (e.g., 512 for BERT, 2048 for GPT-3).
"""
)

st.markdown(
    """
### Why the Industry Adopted and Then Abandoned Learned Embeddings
* **Adoption Rationale (2018-2020)**: Gradient descent directly optimizes position vectors to maximize language modeling loss. Empirically, BERT and GPT-2 scored slightly higher downstream accuracy than pure sinusoidal models on fixed-length benchmarks.
* **Fatal Architectural Flaw (The Extrapolation Wall)**: Because `W_pos` is an explicit lookup table indexed by integer position `i`, the model has **zero representation** for index `i >= L_max`.
  If trained on 2048 tokens, running token 2049 causes an index out-of-bounds error or requires arbitrary position resets.
* **Modern LLM Consensus**: Learned absolute embeddings have been completely abandoned in 2023-2026 foundation models in favor of RoPE.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: RELATIVE POSITIONAL ENCODING
# -----------------------------------------------------------------------------
st.header("4. Relative Positional Encoding (Shaw et al., T5 Relative Bias)", divider="rainbow")

st.markdown(
    """
Relative positional encoding recognizes that what matters for syntax and reasoning is not whether a token is at position 100 or 200,
but whether the modifier is 1 token to the left of the noun.

### Shaw et al. Formulation (2018)
Instead of modifying input vectors, relative position biases are injected into the attention logits:

```text
Attention_Score(i, j) = (q_i * k_j^T + q_i * a_{i, j}^K) / sqrt(d_k)
```

Where `a_{i, j}^K` is a learnable embedding representing the relative distance `clip(i - j, -k_max, +k_max)`.

### T5 Relative Position Bias (Raffel et al., 2020)
T5 simplified this into a scalar bias added directly to raw attention scores:

```text
Attention_Score(i, j) = (q_i * k_j^T) / sqrt(d_k) + b_{i - j}
```

T5 uses a logarithmic bucketing scheme:
* Relative offsets `0 to 7` have exact individual bins.
* Larger offsets are assigned to exponentially growing buckets up to a max distance of 128.
* All distances beyond 128 share a single bucket.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: ROTARY POSITION EMBEDDINGS (ROPE) THEORY & MATHEMATICS
# -----------------------------------------------------------------------------
st.header("5. Rotary Position Embeddings (RoPE) Theory & Mathematics", divider="rainbow")

st.markdown(
    """
Introduced by Jianlin Su et al. (2021) in the *RoFormer* paper, **RoPE (Rotary Position Embedding)** unites absolute position injection with pure relative distance behavior.

### Mathematical Formulation
Given query vector `q_m` at position `m` and key vector `k_n` at position `n`, we seek an operation `f(q, m)` and `f(k, n)` such that their inner product depends solely on relative distance `(m - n)`:

```text
<f(q, m), f(k, n)> = g(q, k, m - n)
```

In 2D space, rotating a vector `x = [x_1, x_2]` by an angle `m * theta` using rotation matrix `R_{m, theta}`:

```text
R_{m, theta} = [
  cos(m * theta)   -sin(m * theta)
  sin(m * theta)    cos(m * theta)
]
```

By the orthogonal property of rotation matrices:
```text
(R_{m, theta} * q)^T * (R_{n, theta} * k) = q^T * (R_{m, theta}^T * R_{n, theta}) * k
                                          = q^T * R_{n - m, theta} * k
```
The resulting attention logit depends **exclusively** on the token separation `(n - m)`!
"""
)

st.info(
    "Key Insight: RoPE is NOT additive. It does not add coordinates to vectors. "
    "Instead, it acts as a multiplicative orthogonal rotation in 2D coordinate sub-planes of the Query and Key vectors."
)

# -----------------------------------------------------------------------------
# SECTION 6: ROTARY POSITIONAL EMBEDDINGS IMPLEMENTATION & 2D ROTATION PLANES
# -----------------------------------------------------------------------------
st.header("6. Rotary Positional Embeddings Implementation & 2D Rotation Planes", divider="rainbow")

st.markdown(
    """
### High-Dimensional Chunking into 2D Subspaces
A head dimension of `d = 128` is partitioned into `d / 2 = 64` independent 2-dimensional planes.
Each plane `i in [0, d/2 - 1]` is assigned a base frequency:

```text
theta_i = 10000 ^ (-2i / d)
```

For token position `m`, each 2D vector chunk `[x_{2i}, x_{2i + 1}]` is rotated:

```text
x_rotated_{2i}     = x_{2i} * cos(m * theta_i) - x_{2i + 1} * sin(m * theta_i)
x_rotated_{2i + 1} = x_{2i} * sin(m * theta_i) + x_{2i + 1} * cos(m * theta_i)
```

### Why RoPE Scales Seamlessly
1. **Long-Term Decay**: As relative distance `|m - n|` increases, the sum of inner products across multiple frequency planes naturally decays, stabilizing self-attention.
2. **Frequency Scaling for Context Expansion**:
   To stretch context length from 8k to 128k (like Llama 3):
   * Base frequency `theta_base` is scaled up from `10,000` to `500,000` or `5,000,000`.
   * High frequencies (local syntax) remain sensitive, while low frequencies (document-level context) slow down.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: ATTENTION WITH LINEAR BIASES (ALIBI)
# -----------------------------------------------------------------------------
st.header("7. Attention with Linear Biases (ALiBi)", divider="rainbow")

st.markdown(
    """
Introduced by Ofir Press et al. (2021) in *Train Short, Test Long*, **ALiBi** eliminates positional embeddings entirely from input representations and query-key transformations.

### ALiBi Core Formulation
ALiBi adds a non-learnable, static linear penalty directly to the pre-softmax attention logits:

```text
Attention_Logits(i, j) = (q_i * k_j^T) / sqrt(d_k) - m * (i - j)
```

Where:
* `(i - j)` is the non-negative distance between query position `i` and key position `j` (for causal autoregressive attention where `i >= j`).
* `m` is a head-specific, non-learnable geometric slope constant.
"""
)

# ALiBi Slope Distribution
st.markdown(
    """
### Geometric Slope Selection Across Heads
For a multi-head model with `H = 8` heads, slopes `m` are fixed powers of 2:

```text
Slopes = [2^(-8/8), 2^(-7/8), 2^(-6/8), ..., 2^(-1/8)]
       = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
```
* **Heads with Large Slopes (e.g. m = 0.5)**: Strictly penalize distant tokens; focus exclusively on immediate local context (syntax, compound words).
* **Heads with Gentle Slopes (e.g. m = 1/256)**: Minimal distance penalty; allow long-range thematic attention.

### Zero-Shot Extrapolation Strength
ALiBi models trained on 1,024 tokens can extrapolate to 8,192 tokens during inference without any fine-tuning, because the linear penalty cleanly penalizes unseen long distances without encountering out-of-distribution phase wraps.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE PYTORCH IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("8. Production-Grade PyTorch Implementation (RoPE & ALiBi Modules)", divider="rainbow")

st.markdown(
    """
Below is a standalone, vectorized implementation of Rotary Positional Embeddings (RoPE)
and ALiBi attention masking matching the architecture used in modern open-weight LLMs:
"""
)

code_snippet = """import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    # Production RoPE implementation matching Llama 3 architecture
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Calculate inverse frequencies: theta_i = base^(-2i / dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def build_alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor:
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power) + get_slopes(2 * closest_power)[0::2][:n - closest_power]

    slopes = torch.tensor(get_slopes(num_heads)).unsqueeze(1).unsqueeze(2)
    distance = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    distance = -torch.abs(distance)
    alibi_bias = slopes * distance.unsqueeze(0)
    return alibi_bias
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE POSITIONAL WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Positional Workbench: Decay & Inner Product Heatmaps", divider="rainbow")

st.markdown(
    """
Explore how relative position modifies Query-Key inner products under **Vaswani Sinusoidal**,
**RoPE (Rotary)**, and **ALiBi** across sequence distances.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    seq_length_slider = st.slider("Sequence Length (Tokens)", min_value=16, max_value=64, value=32, step=8)
    head_dim_slider = st.selectbox("Head Dimension (d_k)", options=[32, 64, 128], index=1)
with col_ctrl2:
    rope_base = st.selectbox("RoPE Base Frequency (theta_base)", options=[10000.0, 500000.0], index=0)
    alibi_slope_val = st.slider("ALiBi Slope Factor (m)", min_value=0.01, max_value=0.5, value=0.125, step=0.025)

# Interactive computation
positions = np.arange(seq_length_slider)
dim = head_dim_slider

# Compute Sinusoidal PE matrix
sinusoidal_mat = np.zeros((seq_length_slider, dim))
for p in range(seq_length_slider):
    for i in range(0, dim, 2):
        denom = 10000.0 ** (i / dim)
        sinusoidal_mat[p, i] = np.sin(p / denom)
        sinusoidal_mat[p, i + 1] = np.cos(p / denom)

# Normalized Sinusoidal inner product matrix: PE * PE^T
sin_similarity = np.dot(sinusoidal_mat, sinusoidal_mat.T)
norm = np.linalg.norm(sinusoidal_mat, axis=1, keepdims=True)
sin_similarity_norm = sin_similarity / (norm * norm.T + 1e-9)

# Compute RoPE relative decay for uniform unit vector
inv_freq = 1.0 / (rope_base ** (np.arange(0, dim, 2) / dim))
rope_sim = np.zeros((seq_length_slider, seq_length_slider))
for i in range(seq_length_slider):
    for j in range(seq_length_slider):
        delta = i - j
        rope_sim[i, j] = np.mean(np.cos(delta * inv_freq))

# Compute ALiBi distance penalty
alibi_penalty = np.zeros((seq_length_slider, seq_length_slider))
for i in range(seq_length_slider):
    for j in range(seq_length_slider):
        if j <= i:
            alibi_penalty[i, j] = -alibi_slope_val * (i - j)
        else:
            alibi_penalty[i, j] = -1e9  # Causal mask

col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.subheader("Sinusoidal PE Similarity")
    st.caption("Normalized inner product PE[i] dot PE[j]")
    st.dataframe(pd.DataFrame(np.round(sin_similarity_norm[:8, :8], 3)))
with col_w2:
    st.subheader("RoPE Relative Dot Product")
    st.caption("Mean relative rotation cos((i-j) * theta)")
    st.dataframe(pd.DataFrame(np.round(rope_sim[:8, :8], 3)))
with col_w3:
    st.subheader("ALiBi Attention Logit Bias")
    st.caption("Linear slope penalty -m * |i-j|")
    st.dataframe(pd.DataFrame(np.round(alibi_penalty[:8, :8], 2)))

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why did modern LLMs transition from Learned Absolute Embeddings to RoPE?"):
    st.markdown(
        """
        **Answer:**
        Learned absolute embeddings (BERT, GPT-2) store a fixed weight matrix `W_pos in R^[L_max, d]`.
        1. **Strict Context Boundaries**: They have zero mathematical capability to extrapolate beyond `L_max`.
        2. **Relative Distance Blindness**: Attention similarity depends on absolute coordinates rather than translation-invariant distance `(i - j)`.
        3. **Embedding Pollution**: Adding position vectors directly to lexical embeddings dilutes semantic vector spaces in early layers.
        RoPE solves all three by applying orthogonal 2D rotations directly to Query and Key vectors inside attention heads, ensuring inner products depend purely on relative distance `(m - n)`.
        """
    )

with st.expander("Q2: Mathematically, why does RoPE satisfy relative position properties?"):
    st.markdown(
        """
        **Answer:**
        RoPE represents vectors in 2D planes as complex numbers: `q_m = q * exp(i * m * theta)`.
        The inner product between rotated vectors corresponds to the real part of the complex conjugate product:
        `Re(q_m * k_n^*) = Re(q * exp(i * m * theta) * k^* * exp(-i * n * theta)) = Re(q * k^* * exp(i * (m - n) * theta))`.
        Because the exponent contains only `(m - n)`, the dot product depends solely on relative token separation.
        """
    )

with st.expander("Q3: Compare RoPE vs. ALiBi in terms of zero-shot length extrapolation."):
    st.markdown(
        """
        **Answer:**
        * **ALiBi**: Provides superior out-of-the-box zero-shot extrapolation without fine-tuning. Because the linear penalty `-m * (i - j)` is continuous and monotonic, testing on 8k tokens when trained on 2k works cleanly.
        * **RoPE**: Can fail zero-shot when evaluated on contexts significantly longer than training because high-frequency rotations encounter unseen phase configurations. However, with frequency scaling (NTK-aware, YaRN), RoPE adapts with minimal fine-tuning and offers higher representational fidelity than ALiBi.
        """
    )

with st.expander("Q4: Does RoPE modify the Value (V) vectors in self-attention?"):
    st.markdown(
        """
        **Answer:**
        **No.** RoPE is strictly applied to **Query (Q)** and **Key (K)** tensors.
        The attention weights `Softmax((Q K^T) / sqrt(d_k))` encode the positional distance. The Value vectors represent pure content information to be aggregated. Applying RoPE to Value vectors would inappropriately rotate the output representations across token positions.
        """
    )

with st.expander("Q5: What is the purpose of increasing the RoPE base frequency theta from 10,000 to 500,000?"):
    st.markdown(
        """
        **Answer:**
        Increasing the base frequency `theta_base` reduces the rotation speed of lower frequency dimensions.
        At standard `theta = 10,000`, long sequences (> 8k tokens) cause low-frequency dimensions to wrap around full periods `2 * pi`, causing the model to confuse distant tokens with nearby tokens. Increasing `theta_base` expands the wavelength so long sequences remain within unique phase intervals.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Applying RoPE to Padded Tokens Without Proper Index Alignment**
        *Root Cause*: If position IDs are generated sequentially `0, 1, ..., N` on left-padded sequences in batched inference, padding tokens consume valid position coordinates.
        *Production Fix*: Position IDs must be assigned based strictly on actual token positions, masking out pads or utilizing packed sequence indices.
        """
    )
    st.warning(
        """
        **Trap 2: Mismatched RoPE Dimension vs. Head Dimension**
        *Root Cause*: Applying RoPE across the entire hidden dimension instead of head dimension `d_k = d_model / num_heads`.
        *Production Fix*: Ensure `dim` in `RotaryEmbedding` matches `head_dim` (e.g. 128 in Llama 3), and ensure `dim` is an even number.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Inverting Query / Key Rotation Direction**
        *Root Cause*: Using opposite rotation signs or conjugate mismatch between `Q` and `K`, causing distance to appear as `(m + n)` instead of `(m - n)`.
        *Production Fix*: Both Query and Key must receive identical forward rotations: `R_m q` and `R_n k`. The transposition in `q^T k` automatically produces `(m - n)`.
        """
    )
    st.warning(
        """
        **Trap 4: Naive Extrapolation Without RoPE Scaling**
        *Root Cause*: Passing 32k tokens directly into a model trained on 8k with standard RoPE causes perplexity to explode to infinity.
        *Production Fix*: Always engage frequency interpolation (NTK-aware, Linear, or YaRN) when exceeding pretraining context limits.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: RoPE Orthogonality"):
        st.write("Does RoPE change the L2 norm of Query and Key vectors?")
        st.info("No. 2D rotation matrices are orthogonal (|R x| = |x|). RoPE preserves vector norms perfectly.")

    with st.expander("⚡ Flashcard 2: ALiBi Trainability"):
        st.write("Are ALiBi slope values learned during gradient descent?")
        st.info("No. ALiBi slopes are non-learnable, fixed geometric constants set prior to training.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Value Vector RoPE"):
        st.write("Why are Value vectors excluded from RoPE rotation?")
        st.info("Positional routing happens via Q-K dot products. Rotating V would corrupt semantic output vectors.")

    with st.expander("⚡ Flashcard 4: Vaswani Angle Addition"):
        st.write("What mathematical property allowed Vaswani PE to represent relative shifts?")
        st.info("Trigonometric angle addition: cos(a+b) and sin(a+b) can be computed via linear combinations.")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why transformer self-attention is permutation-equivariant without positional signals.",
    "I can contrast Vaswani Sinusoidal vs. Learned Absolute vs. RoPE vs. ALiBi positional approaches.",
    "I understand the 2D complex rotation mathematics proving that (R_m q)^T (R_n k) = q^T R_{n-m} k.",
    "I understand why RoPE is applied to Query and Key tensors but never to Value tensors.",
    "I know how ALiBi implements distance penalties using non-learnable geometric head slopes.",
    "I can explain why increasing theta_base from 10k to 500k facilitates long-context scaling in Llama 3.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered all Positional Encoding Methods across Classical and Modern Foundation LLMs."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
