"""
12.1 Encoder Models
===================
Interactive Master Lesson exploring Bidirectional Transformer Encoders,
BERT, RoBERTa, DistilBERT, Contextual Representations, and Masked Language Modeling (MLM).
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
    page_title="12.1 Encoder Models - LLM Architecture",
    page_icon="🔍",
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
st.title("🔍 12.1 Encoder Models")
st.caption(
    "Chapter 12: LLM Architecture Families | Module 1 of 4"
)

st.markdown(
    """
Encoder-only architectures represent the breakthrough that proved transfer learning in NLP.
By utilizing unmasked, **bidirectional self-attention** and training under the **Masked Language Modeling (MLM)**
objective, models like **BERT**, **RoBERTa**, and **DistilBERT** produce deep, context-sensitive
dense vector representations ideal for classification, search retrieval, and semantic embeddings.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE BIDIRECTIONAL PARADIGM
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Bidirectional Paradigm", divider="rainbow")

st.markdown(
    """
### Unconstrained Bidirectional Attention Flow
Unlike causal autoregressive decoders, encoder tokens attend simultaneously to all preceding and following tokens:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "encoder_architecture_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("tokens", "Input Tokens\\n[CLS] The [MASK] jumped [SEP]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("embed", "Embeddings\\nToken + Pos + Segment", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("encoder", "Bidirectional Transformer\\nL Layers (Full N x N Attention)", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("cls_out", "[CLS] Representation\\nSequence Classification", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("mlm_out", "[MASK] Token Vector\\nPredict Target Word", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("tokens", "embed")
arch_dot.edge("embed", "encoder")
arch_dot.edge("encoder", "cls_out", label="h_[CLS] in R^[d]")
arch_dot.edge("encoder", "mlm_out", label="h_[MASK] in R^[d]")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Why Decoders Dominant Over Encoders**: In the 2018-2021 era, NLP research was split between Encoders (BERT for understanding) and Decoders (GPT for generation). Today, **decoder-only models have largely superseded encoders** even for representation tasks, because unified next-token prediction scales smoothly to generative chat, tool use, and in-context learning.
* **Modern Encoders as Embedding Backbones**: Encoders survive today primarily as dedicated dense embedding models (e.g. `bge-large-en`, `e5-mistral`, `nomic-embed-text`) powering Retrieval-Augmented Generation (RAG) vector databases.
"""
)

# Encoder Family Comparison Table
encoder_matrix_df = pd.DataFrame(
    {
        "Model": ["BERT-Base", "BERT-Large", "RoBERTa-Base", "DistilBERT", "DeBERTa-v3"],
        "Parameters": ["110 Million", "340 Million", "125 Million", "66 Million", "86 Million"],
        "Pretraining Objective": ["MLM (15%) + NSP", "MLM (15%) + NSP", "Dynamic MLM (No NSP)", "Knowledge Distillation (Triple Loss)", "Disentangled Attention + RTD"],
        "Training Tokens": ["3.3 Billion", "3.3 Billion", "30 Billion (10x BERT)", "3.3 Billion", "160 Billion"],
        "Primary Niche": ["Historical baseline", "Historical benchmark", "NLU benchmark standard", "Edge / Low-latency CPU inference", "State-of-the-art encoder representations"],
    }
)
st.dataframe(encoder_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: BERT ARCHITECTURE & FOUNDATIONS
# -----------------------------------------------------------------------------
st.header("2. BERT Architecture & Foundations (Devlin et al., 2018)", divider="rainbow")

st.markdown(
    """
**BERT (Bidirectional Encoder Representations from Transformers)** revolutionized NLP by combining deep bi-directional representations with pretraining and fine-tuning:

### 1. Three-Part Additive Embeddings
BERT constructs input representations by summing three distinct vector tables:
```text
Input = Token_Embeddings[x] + Segment_Embeddings[s] + Position_Embeddings[pos]
```
* **Token Embeddings**: 30,522 WordPiece vocabulary.
* **Segment Embeddings**: Binary identifier (`0` or `1`) distinguishing Sentence A from Sentence B.
* **Learned Absolute Position Embeddings**: Trainable matrix up to 512 tokens.

### 2. Dual Pretraining Tasks
* **Masked Language Modeling (MLM)**: 15% of tokens are selected: 80% replaced with `[MASK]`, 10% replaced with a random word, 10% kept unchanged.
* **Next Sentence Prediction (NSP)**: Binary classification predicting whether Sentence B logically follows Sentence A in the original document.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: ROBERTA: ROBUSTLY OPTIMIZED BERT PRETRAINING
# -----------------------------------------------------------------------------
st.header("3. RoBERTa: Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)", divider="rainbow")

st.markdown(
    """
Liu et al. (Meta AI) demonstrated that original BERT was severely undertrained, introducing **RoBERTa**:

### Four Critical Algorithmic Improvements
1. **Dynamic Masking**: Instead of fixing mask positions during preprocessing, RoBERTa generates fresh random masks dynamically every time a sequence is fed to the model.
2. **Elimination of Next Sentence Prediction (NSP)**: Proved that NSP hurts downstream performance; training on full contiguous paragraphs without NSP yields superior representations.
3. **Larger Mini-Batches**: Scaled batch size from 256 to 8,000 sequences, significantly stabilizing Adam optimization.
4. **Byte-Pair Encoding (BPE)**: Replaced WordPiece with a 50,000-token byte-level BPE vocabulary matching GPT-2.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: DISTILBERT: KNOWLEDGE DISTILLATION
# -----------------------------------------------------------------------------
st.header("4. DistilBERT: Knowledge Distillation & Model Compression (Sanh et al., 2019)", divider="rainbow")

st.markdown(
    """
DistilBERT compresses BERT-Base by **40% in size** while retaining **97% of its language understanding performance** and running **60% faster**:

### Knowledge Distillation Architecture
* **Structural Halving**: Takes BERT-Base (12 layers) and removes every second layer, resulting in a compact 6-layer architecture.
* **Triple Loss Function**:
  ```text
  L_total = alpha_ce * L_ce + alpha_mlm * L_mlm + alpha_cos * L_cos
  ```
  1. **Distillation Loss (`L_ce`)**: KL-divergence forcing student Softmax logits to match teacher Softmax logits at high temperature (`T = 2.0`).
  2. **Supervised Masked Language Loss (`L_mlm`)**: Hard ground truth next-token cross-entropy.
  3. **Cosine Embedding Loss (`L_cos`)**: Forces student hidden states to align with teacher hidden state vectors.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: DENSE ENCODER REPRESENTATIONS
# -----------------------------------------------------------------------------
st.header("5. Dense Encoder Representations & Contextual Embeddings", divider="rainbow")

st.markdown(
    """
Unlike static Word2Vec vectors where the word "bank" has a single static coordinate,
encoders produce **dynamic contextual representations**:
* `"I deposited cash in the bank"` -> `h_bank` coordinates lie near financial institutions.
* `"The river bank was muddy"` -> `h_bank` coordinates lie near geography and nature.

### The Pooling Dilemma (CLS vs. Mean Pooling)
When using encoders to produce a single sequence-level vector for semantic search:
* **`[CLS]` Token Pooling**: Uses the first hidden state `h_0`. Works well if trained with a classification head, but collapses into narrow anisotropic cones if un-finetuned.
* **Mean Pooling**: Averages all token vectors across sequence length: `h_mean = 1/T sum h_t`. Empirically delivers superior semantic similarity scores in modern vector retrieval benchmarks (Sentence-BERT).
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: MASKED LANGUAGE MODELING (MLM) OBJECTIVE
# -----------------------------------------------------------------------------
st.header("6. Masked Language Modeling (MLM) Objective & Cloze Task Formulation", divider="rainbow")

st.markdown(
    """
MLM formulates pretraining as a Cloze test:

```text
Input:  "The capital of [MASK] is Paris."
Target: "France"
```

### The 80 / 10 / 10 Corruption Heuristic
If the model only saw `[MASK]` tokens during pretraining, it would never learn representations for natural words encountered during downstream fine-tuning.
For the 15% chosen tokens:
* **80% of the time**: Replaced with `[MASK]`.
* **10% of the time**: Replaced with a random dictionary token (forces the model to maintain contextual representations for every token, detecting anomalies).
* **10% of the time**: Left unchanged (biases representations toward the actual observed word).
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: PRODUCTION-GRADE BERT FEATURE EXTRACTOR IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("7. Production-Grade BERT MLM & Feature Extractor Implementation", divider="rainbow")

st.markdown(
    """
A complete PyTorch implementation demonstrating bidirectional encoder pooling and masked prediction:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniBERTEncoder(nn.Module):
    # Minimal Bidirectional Transformer Encoder with Mean Pooling
    def __init__(self, vocab_size: int = 30522, d_model: int = 768, num_layers: int = 6, num_heads: int = 12):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(512, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [batch_size, seq_len]
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Additive embeddings
        h = self.embeddings(input_ids) + self.pos_embeddings(positions)
        h = self.norm(h)
        
        # Unmasked bidirectional attention
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h_encoded = self.transformer(h, src_key_padding_mask=pad_mask)
        
        # 1. Masked token logits
        mlm_logits = self.mlm_head(h_encoded)
        
        # 2. Mean-pooled sentence embedding
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(h_encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
        else:
            mean_pooled = h_encoded.mean(dim=1)
            
        return mlm_logits, mean_pooled
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 8: INTERACTIVE MASKED LANGUAGE MODELING WORKBENCH
# -----------------------------------------------------------------------------
st.header("8. Interactive Masked Language Modeling & Attention Density Workbench", divider="rainbow")

st.markdown(
    """
Experiment with how bidirectional context resolves masked tokens and compare
**`[CLS]` Token Vectors** vs. **Mean Pooling Embeddings**.
"""
)

sample_cloze = st.selectbox(
    "Select Cloze Sentence Context",
    options=[
        "The quick brown fox jumps over the [MASK] dog.",
        "Artificial intelligence will transform modern [MASK] forever.",
        "The financial [MASK] approved the mortgage application.",
        "We sat along the river [MASK] fishing at sunset.",
    ],
    index=0,
)

# Simulated predictions
sim_candidates = {
    "The quick brown fox jumps over the [MASK] dog.": {"lazy": 0.88, "sleeping": 0.08, "black": 0.02, "barking": 0.01},
    "Artificial intelligence will transform modern [MASK] forever.": {"society": 0.65, "technology": 0.22, "science": 0.08, "medicine": 0.04},
    "The financial [MASK] approved the mortgage application.": {"bank": 0.74, "institution": 0.18, "advisor": 0.05, "officer": 0.02},
    "We sat along the river [MASK] fishing at sunset.": {"bank": 0.92, "side": 0.05, "shore": 0.02, "water": 0.01},
}

cands = sim_candidates[sample_cloze]
words = list(cands.keys())
probs = list(cands.values())

col_b1, col_b2 = st.columns(2)
with col_b1:
    st.subheader("Top Predicted Mask Completions")
    st.dataframe(pd.DataFrame({"Token": words, "Confidence": [f"{p*100:.1f}%" for p in probs]}), use_container_width=True)
with col_b2:
    st.subheader("Simulated Bidirectional Attention Heatmap")
    tokens_list = sample_cloze.split()
    n_tok = len(tokens_list)
    # Synthetic dense bidirectional attention matrix (fully connected)
    attn_matrix = np.full((n_tok, n_tok), 1.0 / n_tok)
    st.caption("Bidirectional attention matrix is 100% unmasked (N x N complete visibility).")
    st.dataframe(pd.DataFrame(np.round(attn_matrix[:6, :6], 3)), use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why cannot BERT generate long autoregressive text natively?"):
    st.markdown(
        """
        **Answer:**
        BERT was trained with full bidirectional self-attention: every token attends to all future tokens.
        When generating text autoregressively (one token at a time), future tokens do not yet exist.
        If you attempt to sample tokens sequentially with BERT, you violate the bidirectional representation distribution it was trained on, resulting in severe degradation and repetitive nonsense.
        """
    )

with st.expander("Q2: Why did RoBERTa eliminate Next Sentence Prediction (NSP)?"):
    st.markdown(
        """
        **Answer:**
        In original BERT, half the training examples combined two random sentences from different documents to train the NSP binary classification head.
        Liu et al. demonstrated that breaking contiguous document narrative harmed the model's ability to learn long-range discourse structure. Removing NSP and training on full contiguous text blocks improved performance on virtually all downstream benchmarks.
        """
    )

with st.expander("Q3: What are the three losses used to train DistilBERT?"):
    st.markdown(
        """
        **Answer:**
        1. **Distillation Loss (Soft Target Cross-Entropy)**: Matches student output probabilities to teacher probabilities at high temperature.
        2. **Masked Language Modeling Loss (Hard Targets)**: Cross-entropy with true masked token ground truth.
        3. **Cosine Embedding Loss**: Forces the student's hidden activation vectors to point in the same directional orientation as the teacher's vectors.
        """
    )

with st.expander("Q4: Why is Mean Pooling preferred over [CLS] pooling for semantic search embeddings?"):
    st.markdown(
        """
        **Answer:**
        In un-finetuned BERT, the `[CLS]` token hidden state suffers from representation anisotropy (vectors are clustered in a narrow cone in latent space, producing artificially high cosine similarity for completely unrelated sentences).
        Averaging all token vectors (Mean Pooling) distributes contextual information evenly, yielding far superior separation on semantic similarity and vector retrieval benchmarks (Sentence-BERT).
        """
    )

with st.expander("Q5: Why did the AI industry converge on Decoder-Only models over Encoders for general LLMs?"):
    st.markdown(
        """
        **Answer:**
        1. **Unified Generative Architecture**: Decoders can perform both understanding (prompt compression) and generation in a single continuous forward pass.
        2. **In-Context Learning**: Next-token prediction enables few-shot prompt adaptation without fine-tuning.
        3. **Scaling Law Predictability**: Decoders exhibit smoother power-law scaling across compute budgets than masked models.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Padding Tokens Polluting Mean-Pooled Embeddings**
        *Root Cause*: Executing `h.mean(dim=1)` without masking out `<pad>` tokens.
        *Production Fix*: The zeros or padding vectors pull the mean embedding toward the origin. Always multiply by the attention mask before summing, and divide by the unmasked token count.
        """
    )
    st.warning(
        """
        **Trap 2: Exceeding BERT's 512-Token Absolute Position Limit**
        *Root Cause*: Passing sequences longer than 512 tokens into BERT or RoBERTa.
        *Production Fix*: BERT uses a fixed 512-entry learned position table; passing 513 tokens triggers instant index out-of-bounds errors. Chunk long documents or utilize ModernBERT/DeBERTa.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Using [CLS] Token Directly Without Fine-Tuning**
        *Root Cause*: Using raw `h[:, 0, :]` as an off-the-shelf vector database embedding.
        *Production Fix*: Raw `[CLS]` vectors have poor cosine separation. Either fine-tune with contrastive loss (SimCSE) or use Mean Pooling.
        """
    )
    st.warning(
        """
        **Trap 4: Forgetting the 80/10/10 Rule When Implementing Custom MLM**
        *Root Cause*: Replacing 100% of selected tokens with `[MASK]`.
        *Production Fix*: The model never learns to refine real words and degrades during downstream fine-tuning where `[MASK]` tokens are absent.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: BERT Attention Nature"):
        st.write("Is BERT's self-attention causal or bidirectional?")
        st.info("Bidirectional (unmasked N x N attention matrix).")

    with st.expander("⚡ Flashcard 2: DistilBERT Size vs Performance"):
        st.write("How much smaller is DistilBERT compared to BERT-Base?")
        st.info("40% smaller (6 layers vs 12 layers) while retaining 97% performance.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: RoBERTa Dynamic Masking"):
        st.write("What was RoBERTa's dynamic masking innovation?")
        st.info("Generating random masks on-the-fly during data loading instead of pre-static masks.")

    with st.expander("⚡ Flashcard 4: Modern Encoder Role"):
        st.write("What is the primary commercial use case of encoder models in 2026?")
        st.info("Dense vector embeddings for RAG and semantic search.")

# -----------------------------------------------------------------------------
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("10. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the difference between bidirectional encoders and causal autoregressive decoders.",
    "I know how the 80/10/10 Masked Language Modeling (MLM) corruption heuristic functions.",
    "I can explain why RoBERTa eliminated Next Sentence Prediction (NSP) and scaled batch sizes.",
    "I understand the triple-loss distillation framework behind DistilBERT.",
    "I know why Mean Pooling is preferred over [CLS] pooling for dense vector search.",
    "I can explain why decoder-only architectures superseded encoders for foundation AI systems.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Encoder Architectures, BERT, RoBERTa, DistilBERT, and Masked Language Modeling."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
