import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="8.2 Tokens and Embeddings - Embeddings Curriculum",
    page_icon="🧩",
    layout="wide"
)

# Title & Overview
st.title("8.2 Tokens to Sentence Embeddings: Token-Level Manifolds, Attention Weights, & Pooling Operators")
st.caption("Chapter 8: Tokenization and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
A Transformer encoder does **not** natively output a single sentence embedding.
Instead, when an input sequence of $N$ tokens passes through 12 or 24 self-attention layers, the final hidden state is a **two-dimensional tensor matrix**:
`Last Hidden State Shape = [Batch Size, Sequence Length N, Hidden Dimension d]`

Every individual token ($t_1, t_2, ..., t_N$) holds its own contextual vector of dimension $d$ (e.g., 768 or 1024 floats).
To perform vector search, clustering, or classification, downstream systems require a **single vector** of shape `[Hidden Dimension d]`.

**Pooling** is the mathematical operator that collapses the variable-length sequence of token vectors into a single fixed-size representation.

This lesson explores the mathematics and trade-offs of token aggregation: **Mean Pooling**, **CLS Token Pooling**, **Max Pooling**, **Attention-Weighted Pooling**, and the alternative **ColBERT Late Interaction** paradigm.
""")

st.divider()

# Section 1: The Token Aggregation Problem & Pooling Taxonomy
st.header("1. Token-Level Manifolds & Pooling Operators", divider="rainbow")

col_pool1, col_pool2 = st.columns([1, 1])

with col_pool1:
    st.markdown("""
    ### The 4 Primary Pooling Operators:
    * **1. Masked Mean Pooling (SBERT Standard)**:
      * Averages the contextual vectors of all non-padding tokens across the sequence:
        `Sentence_Vector = sum(H_i * mask_i) / sum(mask_i)`
      * Smooths out token-level fluctuations and captures the overall thematic centroid of the sentence.
    * **2. CLS Token Pooling (Classification Standard)**:
      * Extracts the final hidden state of the first token `[CLS]`:
        `Sentence_Vector = H_0`
      * Common in models trained with explicit sequence-level contrastive objectives (e.g. OpenAI `text-embedding-3`, BERT classifiers).
    * **3. Max Pooling**:
      * Takes the coordinate-wise maximum value across all token vectors:
        `Sentence_Vector_j = max(H_1,j, H_2,j, ..., H_N,j)`
      * Extracts the most extreme activations, capturing salient keywords, but vulnerable to outlier noise.
    * **4. Attention-Weighted Pooling**:
      * Learns a lightweight attention head that scores token importance:
        `Sentence_Vector = sum(softmax(a_i) * H_i)`
      * Dynamically gives greater weight to content words while down-weighting syntax words.
    """)

with col_pool2:
    st.markdown("### Matrix Pooling Projection")
    pool_dot = """
    digraph PoolingFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        SeqMatrix [label="Token Hidden States Matrix\\nShape: [seq_len = 5, hidden_dim = 768]\\n[CLS], 'deep', 'learning', 'models', [SEP]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_operators {
            label="Pooling Operators"; style="dashed"; color="#8B5CF6";
            MeanOp [label="Mean Pooling\\nsum(H * mask) / tokens", fillcolor="#EDE9FE"];
            CLSOp [label="CLS Pooling\\nSlice H[0, :]", fillcolor="#FEF3C7"];
            MaxOp [label="Max Pooling\\nmax(H, dim=seq)", fillcolor="#FDE68A"];
        }

        UnifiedVec [label="Dense Sentence Embedding\\nShape: [768 Float32 Dimensions]\\nUnit L2 Normalized", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        SeqMatrix -> MeanOp -> UnifiedVec [label="Thematic centroid"];
        SeqMatrix -> CLSOp -> UnifiedVec [label="Supervised head"];
        SeqMatrix -> MaxOp -> UnifiedVec [label="Peak activations"];
    }
    """
    st.graphviz_chart(pool_dot, use_container_width=False)

st.markdown("### Detailed Comparison of Pooling Architectures")
st.markdown("""
| Pooling Strategy | Formula Summary | Resilience to Noise | Captures Keyword Spikes? | Context Window Sensitivity | Primary Deployed Models |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Pooling** | Average across valid tokens | **Very High** | Moderate | Uniform across sequence | SBERT, BGE, E5, GTE |
| **CLS Pooling** | Hidden state of token 0 | High (if fine-tuned) | Low | Sensitive to position 0 | OpenAI text-embedding-3, BERT |
| **Max Pooling** | Dimension-wise max | Low | **Very High** | Driven by extreme values | Classical NLP SBERT variants |
| **Attention-Weighted** | Softmax-weighted sum | High | High | Driven by query context | GTR, Instructor, ColBERT |
| **ColBERT (No Pooling)** | Retains all token vectors | **Frontier** | **Extreme** | Full token interaction | ColBERT v2, RAGatouille |
""")

st.divider()

# Section 2: Production Code: Multi-Strategy Pooling Engine
st.header("2. Production Implementation: Multi-Strategy Tensor Pooling Engine", divider="rainbow")

st.markdown("""
The production class below implements raw PyTorch-equivalent pooling operations across a 3D token hidden state tensor, demonstrating masked mean pooling, CLS slicing, and dimension-wise max pooling.
""")

st.code(r'''
import numpy as np
from typing import Dict, Any, List

class MultiStrategyPooler:
    """
    Executes multiple pooling algorithms on token hidden states:
    Mean Pooling (attention-masked), CLS Pooling, and Max Pooling.
    """
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def _l2_norm(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def pool_mean(self, hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Masked Mean Pooling:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for pad)
        """
        # Expand mask: [batch, seq_len, 1]
        expanded_mask = np.expand_dims(attention_mask, axis=-1)
        # Sum masked embeddings
        sum_embeddings = np.sum(hidden_states * expanded_mask, axis=1)
        # Count non-pad tokens per sequence
        sum_mask = np.maximum(np.sum(expanded_mask, axis=1), 1e-9)
        pooled = sum_embeddings / sum_mask
        return self._l2_norm(pooled) if self.normalize else pooled

    def pool_cls(self, hidden_states: np.ndarray) -> np.ndarray:
        """CLS Token Pooling: Extracts position 0."""
        cls_vecs = hidden_states[:, 0, :]
        return self._l2_norm(cls_vecs) if self.normalize else cls_vecs

    def pool_max(self, hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Masked Max Pooling: Computes dimension-wise maximum."""
        expanded_mask = np.expand_dims(attention_mask, axis=-1)
        # Masked values set to large negative number before max
        masked_states = np.where(expanded_mask == 1, hidden_states, -1e9)
        max_vecs = np.max(masked_states, axis=1)
        return self._l2_norm(max_vecs) if self.normalize else max_vecs
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractivePoolingSimulator:
    def __init__(self):
        # 4 semantic coordinates: [AI/Tech, Biology, Finance, General]
        self.dim_labels = ["AI/Tech", "Biology", "Finance", "General Syntax"]
        
        # Synthetic contextual token hidden states for a sample sentence:
        # Sentence: "[CLS] Deep learning models predict biological proteins [SEP]"
        self.token_vectors = {
            "[CLS]": np.array([0.45, 0.40, 0.10, 0.60]),
            "Deep": np.array([0.90, 0.05, 0.08, 0.15]),
            "learning": np.array([0.88, 0.08, 0.10, 0.12]),
            "models": np.array([0.82, 0.12, 0.15, 0.18]),
            "predict": np.array([0.70, 0.45, 0.30, 0.20]),
            "biological": np.array([0.15, 0.95, 0.02, 0.08]),
            "proteins": np.array([0.10, 0.98, 0.01, 0.05]),
            "[SEP]": np.array([0.20, 0.20, 0.10, 0.70])
        }

    def compute_pooling(self, strategy: str) -> np.ndarray:
        keys = list(self.token_vectors.keys())
        matrix = np.array([self.token_vectors[k] for k in keys]) # [8, 4]
        
        if strategy == "Mean Pooling (All Content Tokens)":
            # Exclude special tokens [CLS] and [SEP]
            content_vecs = np.array([self.token_vectors[k] for k in keys if k not in ["[CLS]", "[SEP]"]])
            pooled = np.mean(content_vecs, axis=0)
        elif strategy == "CLS Token Pooling":
            pooled = self.token_vectors["[CLS]"]
        elif strategy == "Max Pooling":
            content_vecs = np.array([self.token_vectors[k] for k in keys if k not in ["[CLS]", "[SEP]"]])
            pooled = np.max(content_vecs, axis=0)
        else: # Biology-Weighted Pooling
            weights = np.array([0.05, 0.10, 0.10, 0.10, 0.15, 0.25, 0.25, 0.0]) # heavily weight bio
            weights = weights / np.sum(weights)
            pooled = np.sum(matrix * weights[:, np.newaxis], axis=0)

        # Unit L2 normalize
        return pooled / np.linalg.norm(pooled)

st.divider()

# Section 3: Interactive Workbench: Token Pooling Visualizer
st.header("3. Interactive Workbench: Token Pooling Visualizer", divider="rainbow")

st.markdown("""
Inspect how individual token vectors combine into a single sentence vector.
The sentence is: **"[CLS] Deep learning models predict biological proteins [SEP]"**.
Observe how changing the pooling strategy dramatically alters the balance between the **AI/Tech** and **Biology** coordinate dimensions in the final sentence embedding!
""")

pool_sim = InteractivePoolingSimulator()

col_pool_l, col_pool_r = st.columns([1, 1])

with col_pool_l:
    st.subheader("1. Token Hidden States Matrix")
    st.markdown("Each token has a 4-dimensional contextual activation vector:")
    
    df_tokens = pd.DataFrame(pool_sim.token_vectors).T
    df_tokens.columns = pool_sim.dim_labels
    st.dataframe(df_tokens.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)

    strategy_choice = st.radio(
        "Select Pooling Strategy:",
        [
            "Mean Pooling (All Content Tokens)",
            "CLS Token Pooling",
            "Max Pooling",
            "Attention-Weighted Pooling (Focus on Domain Entities)"
        ]
    )

with col_pool_r:
    st.subheader("2. Resulting Sentence Embedding")
    final_vec = pool_sim.compute_pooling(strategy_choice)
    
    df_res = pd.DataFrame({
        "Latent Concept Dimension": pool_sim.dim_labels,
        "Normalized Activation": final_vec
    })
    
    st.dataframe(
        df_res.style.format({"Normalized Activation": "{:.4f}"}).background_gradient(subset=["Normalized Activation"], cmap="Greens"),
        use_container_width=True
    )

    m1, m2 = st.columns(2)
    m1.metric("L2 Euclidean Norm", f"{np.linalg.norm(final_vec):.6f}")
    dominant_dim = pool_sim.dim_labels[np.argmax(final_vec)]
    m2.metric("Dominant Concept", dominant_dim)

    if strategy_choice == "CLS Token Pooling":
        st.info("ℹ️ **CLS Pooling**: Reflects the initial sequence token prior. If not fine-tuned with contrastive loss, CLS vectors tend to be diffuse across dimensions.")
    elif strategy_choice == "Max Pooling":
        st.info("ℹ️ **Max Pooling**: Retains peak spikes across both AI and Biology dimensions, capturing strong keyword activations.")
    else:
        st.success("✅ **Balanced Representation**: Successfully balances AI method and Biological domain within the same unit-normalized vector.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Pooling Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does Mean Pooling outperform CLS pooling on Semantic Textual Similarity (STS) benchmarks for open-source models?**
    * **Answer**:
      * In base BERT models, the `[CLS]` token is pre-trained only on Next Sentence Prediction (NSP). It never learns to summarize the semantic content of all tokens.
      * Mean pooling averages all token representations across the sequence, ensuring every content word contributes to the final vector.
      * Empirical STS benchmarks (Reimers & Gurevych, 2019) demonstrated that Mean Pooling yields a **15-20% higher Spearman correlation** than CLS pooling when models are fine-tuned on sentence pairs.

    **Q2: What is the ColBERT (Contextualized Late Interaction) paradigm, and how does it bypass pooling altogether?**
    * **Answer**:
      * Traditional bi-encoders compress all token vectors into a single pooled vector, inevitably losing fine-grained keyword nuance.
      * **ColBERT** retains **all token embeddings** for both query and document (no pooling).
      * At query time, it computes **Late Interaction (MaxSim)**: for every query token, it finds the maximum dot product with any document token, and sums these maxima:
        `Score = sum_{q in Query} max_{d in Document} (E_q dot E_d)`
      * This provides Cross-Encoder-level accuracy with Bi-Encoder retrieval speed!

    **Q3: How do you implement Masked Mean Pooling in PyTorch without memory leaks?**
    * **Answer**:
      * Expand attention mask to 3D: `expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()`.
      * Multiply and sum: `summed = torch.sum(hidden_states * expanded, dim=1)`.
      * Clamp mask sum to avoid zero division: `counts = torch.clamp(expanded.sum(dim=1), min=1e-9)`.
      * Divide: `pooled = summed / counts`.
      * Wrap inside `torch.no_grad()` to prevent gradient graph accumulation during inference.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Naive torch.mean() without Attention Masking**:
      * *Issue*: Computing `torch.mean(last_hidden_state, dim=1)` across padded sequences.
      * *Consequence*: The padding tokens (zeros or pad embeddings) contribute to the average, corrupting short sentences in large batches.
      * *Fix*: Always mask with attention mask before computing sums.

    * **Trap 2: Forgetting L2 Normalization after Pooling**:
      * *Issue*: Feeding unnormalized pooled vectors into inner-product vector indexes.
      * *Consequence*: Sentence length directly affects vector magnitude, corrupting cosine distance calculations.
      * *Fix*: Always call `F.normalize(pooled, p=2, dim=1)` after pooling.

    * **Trap 3: Using CLS Pooling on Models Trained with Mean Pooling**:
      * *Issue*: Loading `BAAI/bge-small-en-v1.5` and taking `last_hidden_state[:, 0, :]`.
      * *Consequence*: Causes severe retrieval hit-rate collapse because BGE was trained with Mean Pooling objectives.
      * *Fix*: Inspect model configuration (`modules.json` in Hugging Face checkpoint) to determine the expected pooling mode.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Mean Pooling**
      * *Definition*: Aggregating all contextual token vectors into a sentence embedding by averaging over valid non-pad tokens.
    * **Flashcard 2: CLS Pooling**
      * *Definition*: Slicing the vector corresponding to the initial [CLS] token to represent the entire sequence.
    * **Flashcard 3: Max Pooling**
      * *Definition*: Taking the coordinate-wise maximum value across all tokens in a sequence to capture peak keyword activations.
    * **Flashcard 4: ColBERT MaxSim**
      * *Definition*: A late-interaction scoring operator that computes the sum of maximal token-to-token dot products without pooling.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Tokens and Embeddings:

- [ ] I can describe the shape transformations from raw tokens to pooled sentence vectors.
- [ ] I understand the mathematical differences between Mean, CLS, Max, and Attention pooling.
- [ ] I know why attention-masked Mean Pooling is the gold standard for Sentence Transformers.
- [ ] I can implement Masked Mean Pooling from scratch in PyTorch.
- [ ] I understand how ColBERT's late interaction paradigm circumvents pooling limitations.
""")
