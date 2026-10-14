"""
12.2 Decoder Models
===================
Interactive Master Lesson exploring Causal Transformer Decoders, the GPT Family Evolution,
Causal Language Modeling, Autoregressive Generation, KV Caches, and Decoder-Only Dominance.
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
    page_title="12.2 Decoder Models - LLM Architecture",
    page_icon="🔮",
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
st.title("🔮 12.2 Decoder Models")
st.caption(
    "Chapter 12: LLM Architecture Families | Module 2 of 4"
)

st.markdown(
    """
The decoder-only causal transformer is the undisputed structural backbone of modern Artificial General Intelligence.
From **GPT-1** and **GPT-2** to **GPT-4**, **Llama 3**, and **Mistral**, the simplicity of next-token autoregressive
modeling paired with masked attention has demonstrated unmatched scaling predictability, emergent reasoning,
and in-context zero-shot generalizability.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & CAUSAL DECODER EMPIRE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Causal Decoder Empire", divider="rainbow")

st.markdown(
    """
### The Pure Causal Dataflow Stream
Decoder-only networks eliminate encoder cross-attention entirely, processing input prompts and generated completions through a single unified causal backbone:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "decoder_architecture_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("tokens", "Input Tokens\\n[x_1, ..., x_t]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("causal_mask", "Causal Attention Mask\\nUpper-Triangular -inf", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("blocks", "Causal Blocks (L)\\nSelf-Attention + SwiGLU", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("kv_cache", "KV Cache Storage\\nPast Keys & Values", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("head", "LM Head\\nNext-Token Logits z_{t+1}", fillcolor="#7C3AED", fontcolor="#F8FAFC")

arch_dot.edge("tokens", "blocks")
arch_dot.edge("causal_mask", "blocks", label="Constrains Attention")
arch_dot.edge("blocks", "kv_cache", label="Appends Keys/Values")
arch_dot.edge("blocks", "head", label="Final Hidden State")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Complete Convergence on Decoders**: Virtually all state-of-the-art frontier models (OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Meta Llama 3, Google Gemma 2, Mistral Large) are decoder-only causal transformers.
* **Why the Encoder Died for Generation**: Encoder-decoder models (like T5) require separate encoder and decoder parameter towers. At 70B+ scale, splitting parameters creates memory and communication bottlenecks during distributed inference, whereas decoder-only models allow 100% of parameters to be utilized for both understanding and generation simultaneously.
"""
)

# GPT Family Evolution Table
gpt_family_df = pd.DataFrame(
    {
        "Model": ["GPT-1 (2018)", "GPT-2 (2019)", "GPT-3 (2020)", "GPT-4 (2023)", "Llama 3 70B (2024)"],
        "Parameters": ["117 Million", "1.5 Billion", "175 Billion", "~1.8 Trillion (MoE)", "70 Billion (Dense)"],
        "Context Window": ["512", "1,024", "2,048", "128,000", "128,000"],
        "Normalization": ["Post-LayerNorm", "Pre-LayerNorm", "Pre-LayerNorm + Alt Heads", "Pre-RMSNorm (Assumed)", "Pre-RMSNorm + SwiGLU"],
        "Attention Backend": ["Standard Vanilla", "Standard Vanilla", "Sparse Alternating", "FlashAttention / Custom MoE", "Grouped-Query Attention (GQA)"],
    }
)
st.dataframe(gpt_family_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: GPT ARCHITECTURE EVOLUTION
# -----------------------------------------------------------------------------
st.header("2. GPT Architecture Evolution (GPT-1, GPT-2, GPT-3, GPT-4)", divider="rainbow")

st.markdown(
    """
The history of the Generative Pre-trained Transformer (GPT) family marks the progression of modern AI:

### 1. GPT-1 (Radford et al., 2018): Generative Pretraining
Proved that pretraining a 12-layer decoder on unannotated BooksCorpus and fine-tuning with a linear classifier head outperformed heavily supervised specialized models across 9 NLP tasks.

### 2. GPT-2 (Radford et al., 2019): Task-Agnostic Zero-Shot Learners
Scaled to 1.5B parameters and 40GB WebText. Discovered that language models can perform downstream translation, summarization, and question answering **zero-shot** without any parameter fine-tuning, simply by conditioning on textual prompts.
Moved LayerNorm to the input of each sub-block (**Pre-LayerNorm**).

### 3. GPT-3 (Brown et al., 2020): Few-Shot In-Context Learning
Scaled to 175B parameters. Demonstrated that large models exhibit powerful **in-context learning**: providing 3-5 examples inside the prompt enables the model to execute novel tasks with near-fine-tuned accuracy.

### 4. GPT-4 (OpenAI, 2023): Sparse Mixture of Experts
Transitioned from dense 175B networks to a massive **Sparse Mixture of Experts (MoE)** architecture, activating a subset of expert networks per token while achieving human-level performance on legal, medical, and scientific exams.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: CAUSAL LANGUAGE MODELING OBJECTIVE
# -----------------------------------------------------------------------------
st.header("3. Causal Language Modeling Objective & Upper-Triangular Masking", divider="rainbow")

st.markdown(
    """
Causal language modeling trains the network to model the probability of sequences via the exact chain rule:

```text
P(x_1, x_2, ..., x_T) = PRODUCT_{t=1}^{T} P(x_t | x_{<t})
```

### The Upper-Triangular Mask
To ensure token `t` cannot peek at future tokens `t+1, t+2, ...`, the attention score matrix is masked before Softmax:

```text
Mask_{i, j} = 0      if j <= i (Past & Current)
Mask_{i, j} = -inf   if j > i  (Future)

Attention_Weights = Softmax( (Q * K^T) / sqrt(d_k) + Mask )
```
Adding `-inf` forces the Softmax denominator exponent `exp(-inf) = 0.0`, strictly zeroing out attention to all future tokens.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: AUTOREGRESSIVE GENERATION & KV CACHE
# -----------------------------------------------------------------------------
st.header("4. Autoregressive Generation & The KV Cache Decoding Loop", divider="rainbow")

st.markdown(
    """
During text generation, the model predicts one token at a time in a serial loop:

```text
Step 1: Pass [x_1]             -> sample x_2
Step 2: Pass [x_1, x_2]        -> sample x_3
Step 3: Pass [x_1, x_2, x_3]   -> sample x_4
```

### The Quadratic Waste of Naive Generation
Without caching, generating token 1,000 requires recomputing the forward pass over all 999 previous tokens from scratch, resulting in `O(N^2)` matrix multiplications!

### The Key-Value (KV) Cache Solution
Because past tokens do not change, their Key and Value vectors across all layers are computed once and appended to a persistent GPU buffer:
* At step `t`, only the single new token `x_t` is passed into the model.
* Only one Query vector `q_t` is projected.
* `q_t` attends against the cached past Keys and Values:
  ```text
  K_total = concat(K_cached, k_t)
  V_total = concat(V_cached, v_t)
  ```
* Reduces generation compute from **O(N^2) to O(N)**!
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: DECODER-ONLY ARCHITECTURAL DOMINANCE
# -----------------------------------------------------------------------------
st.header("5. Decoder-Only Architectural Dominance (Why It Triumphed in Foundation Models)", divider="rainbow")

st.markdown(
    """
Why did the AI industry converge on decoder-only models rather than Encoder-Decoder (T5) or Encoders (BERT)?

1. **Unified Multi-Task Interface**: Everything in human intelligence (reasoning, code, translation, tool use, conversation) can be cast as next-token prediction.
2. **Serving Throughput & KV Cache Efficiency**: In Encoder-Decoder models, the encoder output must be held in memory and attended to across every single decode step. A decoder-only model maintains a single unified KV cache.
3. **Training Flop Efficiency**: Decoders calculate a loss on every single token of the document during pretraining. In contrast, encoder-decoder models only compute loss on the target span, wasting pretraining compute on the prefix.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: PRODUCTION-GRADE CAUSAL DECODER WITH KV CACHING
# -----------------------------------------------------------------------------
st.header("6. Production-Grade Causal Decoder Architecture with KV Caching", divider="rainbow")

st.markdown(
    """
A complete, standalone PyTorch module demonstrating modern causal self-attention with explicit Key-Value caching:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CausalSelfAttentionWithKVCache(nn.Module):
    # Production Causal Self-Attention with KV Cache for fast autoregressive generation
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # x shape: [batch_size, seq_len, d_model]
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Append to KV cache if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # When caching during 1-token decode step, T = 1 and query attends to all past keys
        is_causal = (T > 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out)
        return out, new_kv
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 7: INTERACTIVE KV CACHE & GENERATION WORKBENCH
# -----------------------------------------------------------------------------
st.header("7. Interactive Autoregressive KV Cache & Generation Simulator Workbench", divider="rainbow")

st.markdown(
    """
Compare computational complexity and memory bandwidth with and without a **Key-Value Cache**
over sequence generation steps.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    gen_tokens = st.slider("Tokens to Generate", min_value=16, max_value=256, value=64, step=16)
    model_param_count = st.select_slider("Model Architecture", options=[7, 13, 70], value=7)
with col_ctrl2:
    batch_sim = st.slider("Batch Size", min_value=1, max_value=16, value=2)
    prompt_len = st.slider("Prompt Length (Tokens)", min_value=16, max_value=512, value=128, step=16)

# Computations
steps_range = np.arange(1, gen_tokens + 1)
# Without KV cache: step t computes (prompt_len + t) tokens
flops_no_cache = [2 * model_param_count * 1e9 * (prompt_len + t) for t in steps_range]
# With KV cache: step t computes 1 token (attending to prompt_len + t)
flops_with_cache = [2 * model_param_count * 1e9 * 1 for t in steps_range]

total_no_cache_flops = sum(flops_no_cache)
total_with_cache_flops = sum(flops_with_cache)
speedup = total_no_cache_flops / total_with_cache_flops

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total Compute With KV Cache", f"{total_with_cache_flops:.2e} FLOPs")
with col_m2:
    st.metric("Total Compute Without Cache", f"{total_no_cache_flops:.2e} FLOPs")
with col_m3:
    st.metric("KV Cache Theoretical Speedup", f"{speedup:.1f}x Faster")

cache_plot_df = pd.DataFrame(
    {
        "Generated Token Step": steps_range,
        "Compute Without Cache (FLOPs)": [f / 1e12 for f in flops_no_cache],
        "Compute With KV Cache (FLOPs)": [f / 1e12 for f in flops_with_cache],
    }
)

st.subheader("Step-by-Step FLOPs Scaling Comparison (TFLOPS per Step)")
st.line_chart(cache_plot_df.set_index("Generated Token Step"))

# -----------------------------------------------------------------------------
# SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: What is the primary computational difference between prompt prefill and token decode?"):
    st.markdown(
        """
        **Answer:**
        * **Prefill Phase**: Ingests the entire prompt of length `N` in parallel. Uses compute-bound GEMM matrix multiplications (`[B, N, d] * [d, d]`), saturating GPU Tensor Cores.
        * **Decode Phase**: Generates 1 token at a time sequentially. Uses memory-bandwidth bound GEMV matrix-vector operations (`[B, 1, d] * [d, d]`), where the GPU must stream all model weights from HBM for every token.
        """
    )

with st.expander("Q2: Why does the KV Cache grow linearly with generation length?"):
    st.markdown(
        """
        **Answer:**
        Every generated token appends one Key vector and one Value vector per head per layer to the cache.
        For a model with `L` layers, `H_kv` heads, and head dimension `d_k`, each new token adds `2 * L * H_kv * d_k * bytes_per_element` to VRAM.
        Across `N` generated tokens, KV cache memory scales as `O(N)`.
        """
    )

with st.expander("Q3: Why did GPT-2 shift from Post-LayerNorm to Pre-LayerNorm?"):
    st.markdown(
        """
        **Answer:**
        In Post-LayerNorm (GPT-1), normalization is placed after the residual addition (`x = LN(x + f(x))`).
        In deep networks, gradients passing back through Post-LN diminish, requiring warmup schedules and careful initialization to avoid divergence.
        Pre-LayerNorm (`x = x + f(LN(x))`) keeps the residual identity stream unimpeded, allowing gradients to backpropagate directly across depth without attenuation.
        """
    )

with st.expander("Q4: Can a causal decoder model be used for sentence classification?"):
    st.markdown(
        """
        **Answer:**
        **Yes.** In modern causal decoders, classification is performed by:
        1. Taking the hidden state of the final token (which has attended to the entire sequence due to causal masking).
        2. Passing that final vector into a linear classification head.
        Alternatively, models are prompted with instruction templates and asked to generate the target label token directly (e.g. `"positive"` or `"negative"`).
        """
    )

with st.expander("Q5: What is the difference between Multi-Query Attention (MQA) and Multi-Head Attention (MHA)?"):
    st.markdown(
        """
        **Answer:**
        In MHA, each query head has a unique key and value head (`H_q = H_kv = 32`).
        In MQA, all query heads share a single Key and Value head (`H_kv = 1`).
        This reduces KV cache memory consumption by **32x**, significantly increasing batch capacity during long-sequence autoregressive decoding.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Applying is_causal=True During 1-Token Decode with KV Cache**
        *Root Cause*: Passing `is_causal=True` to `F.scaled_dot_product_attention` when query length `T = 1` and key length `K > 1`.
        *Production Fix*: Setting `is_causal=True` forces PyTorch to apply a lower-triangular mask on a non-square matrix, inadvertently masking out all cached past keys! Set `is_causal=False` when query length is 1.
        """
    )
    st.warning(
        """
        **Trap 2: KV Cache Memory Fragmentation in Batched Serving**
        *Root Cause*: Allocating contiguous GPU memory blocks for maximum context lengths.
        *Production Fix*: Over 60% of VRAM is wasted on unallocated tail slots. Deploy PagedAttention (vLLM) to page KV memory dynamically.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Off-by-One Error in Causal Masking**
        *Root Cause*: Setting mask diagonal index incorrectly, allowing token `t` to attend to token `t+1`.
        *Production Fix*: The model cheats during pretraining and fails completely during autoregressive inference. Always assert mask diagonal is 1: `torch.triu(..., diagonal=1)`.
        """
    )
    st.warning(
        """
        **Trap 4: Forgetting to Slice Position IDs During Caching**
        *Root Cause*: Passing sequential position IDs `[0, ..., T]` during decode steps instead of the current scalar position offset.
        *Production Fix*: RoPE rotation angles will be calculated at position 0 instead of position `t`, corrupting spatial distance encodings.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: KV Cache Complexity Benefit"):
        st.write("What complexity reduction does the KV Cache achieve during generation?")
        st.info("Reduces generation compute from O(N^2) to O(N).")

    with st.expander("⚡ Flashcard 2: Causal Mask Value"):
        st.write("What value is placed in the upper-triangular attention mask before Softmax?")
        st.info("-inf (so that exp(-inf) = 0.0 in Softmax).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Pre-LN Pioneer"):
        st.write("Which GPT model standardized Pre-LayerNorm over Post-LayerNorm?")
        st.info("GPT-2 (2019).")

    with st.expander("⚡ Flashcard 4: Decode Operational Intensity"):
        st.write("Is token decode compute-bound or memory-bandwidth bound?")
        st.info("Memory-bandwidth bound (GEMV with operational intensity ~1 FLOP/Byte).")

# -----------------------------------------------------------------------------
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("9. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the unified causal dataflow of decoder-only transformer architectures.",
    "I can explain why upper-triangular masking prevents future information leakage in causal modeling.",
    "I understand why the KV Cache reduces autoregressive generation complexity from O(N^2) to O(N).",
    "I can explain why decoder-only architectures superseded encoder-decoder models in foundation AI.",
    "I understand the difference between compute-bound prefill and memory-bound decode phases.",
    "I know how to implement causal self-attention with Key-Value cache support in PyTorch.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Decoder Architectures, GPT Evolution, and Autoregressive KV Caching."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
