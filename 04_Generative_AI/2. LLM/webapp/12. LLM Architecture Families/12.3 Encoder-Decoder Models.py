"""
12.3 Encoder-Decoder Models
===========================
Interactive Master Lesson exploring Dual-Tower Sequence-to-Sequence Architectures,
T5 (Text-to-Text Transfer Transformer), Cross-Attention Mechanics, and Modern Multimodal Paradigms.
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
    page_title="12.3 Encoder-Decoder Models - LLM Architecture",
    page_icon="🔄",
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
st.title("🔄 12.3 Encoder-Decoder Models")
st.caption(
    "Chapter 12: LLM Architecture Families | Module 3 of 4"
)

st.markdown(
    """
The dual-tower **Encoder-Decoder** architecture was the original formulation introduced in Vaswani et al. (2017)
for machine translation and popularized by **T5 (Text-to-Text Transfer Transformer)**. By coupling an unmasked,
bidirectional encoder with an autoregressive, causal decoder via **Cross-Attention**, sequence-to-sequence
models excel at transformation tasks where a distinct source text is mapped to a target output.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & DUAL-TOWER PARADIGM
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Dual-Tower Sequence-to-Sequence Paradigm", divider="rainbow")

st.markdown(
    """
### The Dual-Tower Information Flow
Input prompts pass through an unmasked bidirectional encoder to produce memory keys and values, which are repeatedly queried by the autoregressive decoder through cross-attention:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "enc_dec_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("src", "Source Tokens\\n[translate English to German: ...]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("encoder", "Bidirectional Encoder (L_enc)\\nUnmasked Self-Attention", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("memory", "Encoder Memory States\\nH_enc in R^[B, S, d]", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("decoder", "Autoregressive Decoder (L_dec)\\n1. Causal Self-Attn\\n2. Cross-Attention\\n3. SwiGLU / FFN", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("target", "Target Tokens\\n[Das ist das Haus]", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("src", "encoder")
arch_dot.edge("encoder", "memory")
arch_dot.edge("memory", "decoder", label="K_enc, V_enc")
arch_dot.edge("target", "decoder", label="Q_dec")
arch_dot.edge("decoder", "target", label="Next Token Logits")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Where Encoder-Decoders Thrive Today**: While pure decoders dominate text LLMs, the encoder-decoder structure is the premier architecture for **Multimodal AI** (e.g. Vision-Language Models like Florence-2, Whisper for Speech-to-Text). The encoder ingests dense continuous signals (audio frames, image patches) bidirectionally, while the decoder generates autoregressive text.
* **T5's Enduring Legacy**: T5 proved that all NLP tasks (classification, regression, translation, summarization) can be unified into a single text-to-text formulation, laying the direct conceptual groundwork for prompt engineering in ChatGPT.
"""
)

# Encoder-Decoder Family Table
enc_dec_matrix_df = pd.DataFrame(
    {
        "Model": ["Original Vaswani (2017)", "T5 (Raffel et al., 2020)", "BART (Lewis et al., 2019)", "Flan-T5 (Chung et al., 2022)", "Whisper (OpenAI, 2022)"],
        "Encoder Structure": ["6 Layers, Sinusoidal PE", "12-24 Layers, Relative Bias", "12-24 Layers, Learned Abs PE", "Fine-tuned T5 on 1,800 tasks", "Audio ConvNet + Transformer"],
        "Decoder Structure": ["6 Layers with Cross-Attn", "12-24 Layers with Cross-Attn", "12-24 Layers with Cross-Attn", "12-24 Layers with Cross-Attn", "Autoregressive Text Decoder"],
        "Pretraining Objective": ["Supervised Translation", "Span Corruption (Sentinel Tokens)", "Arbitrary Denoising (Permute/Mask)", "Multi-task Instruction Tuning", "Weakly Supervised Audio-to-Text"],
        "Primary Niche": ["Academic Machine Translation", "Universal Text-to-Text Benchmark", "Summarization & Translation", "Instruction following without RLHF", "State-of-the-art Speech Recognition"],
    }
)
st.dataframe(enc_dec_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: T5: TEXT-TO-TEXT TRANSFER TRANSFORMER
# -----------------------------------------------------------------------------
st.header("2. T5: Text-to-Text Transfer Transformer (Raffel et al., 2020)", divider="rainbow")

st.markdown(
    """
Introduced by Google Research, **T5** conducted the most comprehensive empirical study of transformer pretraining in history:

### 1. Span Corruption Pretraining (The Sentinel Token Objective)
Rather than masking individual 15% tokens like BERT, T5 masks contiguous spans of text (averaging 3 tokens per span) with unique sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...):
```text
Original: "Thank you for inviting me to your party last week"
Input:    "Thank you <extra_id_0> me to your party <extra_id_1> week"
Target:   "<extra_id_0> for inviting <extra_id_1> last <extra_id_2>"
```
The decoder is trained to predict **only the missing spans**, minimizing target token generation overhead during pretraining.

### 2. Relative Position Bucketing
T5 eliminated absolute positional embeddings entirely, injecting scalar biases directly into self-attention and cross-attention logits based on logarithmic relative distance buckets.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: TEXT-TO-TEXT MODELING FRAMEWORK
# -----------------------------------------------------------------------------
st.header("3. Text-to-Text Modeling Framework (Universal Task Unification)", divider="rainbow")

st.markdown(
    """
T5 cast every natural language problem into a standardized text-in, text-out interface:

* **Translation**:
  `"translate English to German: That is good." -> "Das ist gut."`
* **Classification (Sentiment)**:
  `"sst2 sentence: The acting was superb." -> "positive"`
* **Regression (Similarity)**:
  `"stsb sentence1: The dog ran. sentence2: The canine sprinted." -> "4.8"`
* **Summarization**:
  `"summarize: In a historic vote today, the senate passed..." -> "Senate passes bill."`

By utilizing string labels instead of specialized classification heads (`softmax(W_c * h)`), a single model architecture and training loop solved the entire spectrum of natural language tasks.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: SEQUENCE-TO-SEQUENCE GENERATION DYNAMICS
# -----------------------------------------------------------------------------
st.header("4. Sequence-to-Sequence Generation Dynamics", divider="rainbow")

st.markdown(
    """
Execution proceeds in two distinct stages:

### Stage 1: The Encoder Forward Pass (Parallel & Bidirectional)
* The source prompt `[x_1, ..., x_S]` is processed bidirectionally by the encoder once.
* Produces encoder hidden representation matrix `H_enc in R^[B, S, d]`.
* Keys and Values for cross-attention are projected once and cached:
  ```text
  K_enc = W_k_cross * H_enc,    V_enc = W_v_cross * H_enc
  ```

### Stage 2: The Autoregressive Decoder Loop (Serial & Causal)
* The decoder starts with the start-of-sequence token `<BOS>`.
* At step `t`, Query `q_t` is computed from the current decoder state.
* The decoder attends to its own past generated tokens via Causal Self-Attention.
* The decoder attends to the **entire static encoder memory** via Cross-Attention.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: ENCODER-DECODER CROSS-ATTENTION MECHANICS
# -----------------------------------------------------------------------------
st.header("5. Encoder-Decoder Cross-Attention Mechanics", divider="rainbow")

st.markdown(
    """
Cross-attention is the bridge where the decoder extracts conditioning information from the encoder:

```text
Query (Q): Sourced from the DECODER's previous layer  [B, H, T_dec, d_k]
Key (K):   Sourced from the ENCODER's final memory    [B, H, T_enc, d_k]
Value (V): Sourced from the ENCODER's final memory    [B, H, T_enc, d_k]

Cross_Attention(Q, K, V) = Softmax( (Q * K^T) / sqrt(d_k) ) * V
```

### Critical Architectural Distinctions:
1. **Unmasked Softmax**: Cross-attention is **never causally masked**. Every decoder token has full uninhibited access to all source tokens in the encoder memory.
2. **Static Key/Value Cache**: `K_enc` and `V_enc` do not grow or change during autoregressive decoding; they remain completely static throughout the entire generation trajectory!
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: PRODUCTION-GRADE ENCODER-DECODER ARCHITECTURE
# -----------------------------------------------------------------------------
st.header("6. Production-Grade Encoder-Decoder Transformer with Cross-Attention", divider="rainbow")

st.markdown(
    """
A complete PyTorch implementation demonstrating cross-attention binding between an encoder memory tower and a causal decoder:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    # Cross-Attention: Queries from Decoder, Keys/Values from Encoder
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, dec_states: torch.Tensor, enc_memory: torch.Tensor, enc_mask: torch.Tensor = None) -> torch.Tensor:
        B, T_dec, _ = dec_states.shape
        _, T_enc, _ = enc_memory.shape

        # Query from decoder, Key/Value from encoder
        q = self.q_proj(dec_states).view(B, T_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc_memory).view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_memory).view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)

        # Unmasked cross-attention (mask only padding tokens from source)
        pad_mask = enc_mask.unsqueeze(1).unsqueeze(2) if enc_mask is not None else None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_dec, self.d_model)
        return self.out_proj(attn_out)

class Seq2SeqDecoderBlock(nn.Module):
    # Complete Decoder Block combining Self-Attention, Cross-Attention, and FFN
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, dec_x: torch.Tensor, enc_memory: torch.Tensor, enc_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Causal Self-Attention
        norm_x = self.self_norm(dec_x)
        self_out, _ = self.self_attn(norm_x, norm_x, norm_x, is_causal=True)
        dec_x = dec_x + self_out

        # 2. Unmasked Cross-Attention to Encoder Memory
        norm_x = self.cross_norm(dec_x)
        cross_out = self.cross_attn(norm_x, enc_memory, enc_mask=enc_mask)
        dec_x = dec_x + cross_out

        # 3. Feed-Forward Network
        dec_x = dec_x + self.ffn(self.ffn_norm(dec_x))
        return dec_x
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 7: INTERACTIVE CROSS-ATTENTION & ALIGNMENT WORKBENCH
# -----------------------------------------------------------------------------
st.header("7. Interactive Cross-Attention & Sequence Alignment Workbench", divider="rainbow")

st.markdown(
    """
Simulate a Machine Translation alignment matrix showing how decoder target tokens
attend back to source encoder tokens through **Cross-Attention**.
"""
)

source_sentence = "The intelligent system processes continuous data"
target_sentence = "Das intelligente System verarbeitet kontinuierliche Daten"

src_tokens = source_sentence.split()
trg_tokens = target_sentence.split()

# Synthetic cross-attention alignment matrix
align_matrix = np.zeros((len(trg_tokens), len(src_tokens)))
for i, t_tok in enumerate(trg_tokens):
    align_matrix[i, min(i, len(src_tokens) - 1)] = 0.85
    if i > 0:
        align_matrix[i, min(i - 1, len(src_tokens) - 1)] = 0.10
    if i < len(src_tokens) - 1:
        align_matrix[i, min(i + 1, len(src_tokens) - 1)] = 0.05

align_df = pd.DataFrame(
    np.round(align_matrix, 2),
    index=[f"Target: '{w}'" for w in trg_tokens],
    columns=[f"Source: '{w}'" for w in src_tokens],
)

st.subheader("Cross-Attention Alignment Heatmap (Decoder Query x Encoder Key)")
st.caption("Rows represent generated target tokens; columns represent source encoder tokens.")
st.dataframe(align_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: What is the key difference between Self-Attention and Cross-Attention?"):
    st.markdown(
        """
        **Answer:**
        * **Self-Attention**: Query (Q), Key (K), and Value (V) tensors are all projected from the same sequence of hidden states. It models internal dependencies within that single sequence.
        * **Cross-Attention**: Query (Q) is projected from the decoder's current representations, while Key (K) and Value (V) are projected from the encoder's final memory output. It allows the decoder to condition on external context.
        """
    )

with st.expander("Q2: Why does Cross-Attention not require a causal mask?"):
    st.markdown(
        """
        **Answer:**
        Causal masks prevent tokens from peeking at future tokens in autoregressive generation.
        In Cross-Attention, the Keys and Values originate from the source prompt in the encoder. The source prompt has already been completely read and processed. Every decoder token should have full, uninhibited visibility into the entire source context, making causal masking unnecessary.
        """
    )

with st.expander("Q3: How did T5's Span Corruption pretraining improve on BERT's token masking?"):
    st.markdown(
        """
        **Answer:**
        BERT replaced individual 15% tokens with `[MASK]`, predicting them independently.
        T5 masked contiguous spans (averaging 3 tokens) with a single sentinel token (`<extra_id_0>`).
        The decoder was trained to generate only the missing spans, reducing target generation sequence length and saving pretraining compute while teaching the model to predict variable-length phrase insertions.
        """
    )

with st.expander("Q4: Why does an Encoder-Decoder model consume more memory during serving than a Decoder-Only model?"):
    st.markdown(
        """
        **Answer:**
        In addition to storing the causal decoder's KV cache, the serving system must keep the encoder's entire Key and Value memory matrix (`K_enc, V_enc`) allocated in GPU VRAM across the entire generation process for every concurrent user stream.
        """
    )

with st.expander("Q5: Where are Encoder-Decoder models primarily utilized in modern production AI (2026)?"):
    st.markdown(
        """
        **Answer:**
        1. **Speech-to-Text**: Whisper (audio spectrogram encoder -> text decoder).
        2. **Vision-Language**: Florence-2 / Donut (image patch encoder -> text decoder).
        3. **Machine Translation**: High-precision low-resource sequence-to-sequence systems.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Applying Causal Masking to Cross-Attention**
        *Root Cause*: Passing `is_causal=True` to the cross-attention SDPA call.
        *Production Fix*: The decoder accidentally masks out late source encoder tokens, crippling translation and summarization fidelity. Cross-attention must always be unmasked.
        """
    )
    st.warning(
        """
        **Trap 2: Recomputing Encoder Representations on Every Generation Step**
        *Root Cause*: Running the encoder inside the autoregressive decode loop.
        *Production Fix*: The encoder memory is static; compute it once during prefill and cache `K_enc` and `V_enc` for all subsequent decode iterations.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Mismatched Padding Mask in Cross-Attention**
        *Root Cause*: Passing decoder padding masks into the cross-attention kernel instead of the source encoder padding mask.
        *Production Fix*: Cross-attention attends over encoder length `S`; the mask must correspond to source token valid positions.
        """
    )
    st.warning(
        """
        **Trap 4: Forgetting the Prefix Task String in T5**
        *Root Cause*: Fine-tuning or prompting T5 without the explicit task prefix (e.g. `"translate English to German: "`).
        *Production Fix*: T5's pretraining conditioning relies heavily on the task prompt prefix to route weights to the correct task manifold.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Cross-Attention Query Source"):
        st.write("Where do Query vectors come from in cross-attention?")
        st.info("From the Decoder's hidden states.")

    with st.expander("⚡ Flashcard 2: Cross-Attention Key/Value Source"):
        st.write("Where do Key and Value vectors come from in cross-attention?")
        st.info("From the Encoder's final memory representations.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: T5 Pretraining Objective"):
        st.write("What was T5's self-supervised pretraining objective?")
        st.info("Span corruption using sentinel tokens (<extra_id_0>).")

    with st.expander("⚡ Flashcard 4: Leading Speech-to-Text Model"):
        st.write("What popular open-weight audio model uses an encoder-decoder architecture?")
        st.info("OpenAI Whisper.")

# -----------------------------------------------------------------------------
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("9. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the dual-tower information flow between bidirectional encoders and causal decoders.",
    "I can derive the mathematical formulation of Cross-Attention where Q is from decoder and K,V from encoder.",
    "I understand why Cross-Attention is unmasked while Self-Attention in the decoder is causally masked.",
    "I understand T5's text-to-text task formulation and span corruption pretraining.",
    "I know why encoder-decoder models remain the dominant architecture for Multimodal and Speech AI.",
    "I can implement a complete sequence-to-sequence decoder block with cross-attention in PyTorch.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Encoder-Decoder Models, T5, and Cross-Attention Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
