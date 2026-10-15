import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.1 Text to Embedding - Embeddings Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("3.1 Text to Embedding: Tokenization Pipeline, Neural Projection, & Vector Generation")
st.caption("Chapter 3: Text Embeddings Basics | Domain: 3. Vector Embeddings")
st.markdown("""
How does a string of human words like `"Python is a programming language"` become an array of numbers like `[0.021, -0.143, 0.782, ...]`?
This is not a random hashing function; it is a **multi-stage neural transformation pipeline** governed by deep transformer encoders.

The production **Text-to-Embedding Generation Pipeline** executes five distinct stages:
1. **Input Normalization & Subword Tokenization**: Splitting text into subword token IDs (e.g., via Byte-Pair Encoding or WordPiece).
2. **Initial Embedding Lookup**: Mapping token IDs to static vocabulary coordinate vectors.
3. **Transformer Self-Attention Encoding**: Allowing each token to attend to all other tokens, injecting rich contextual meaning into ambiguous words.
4. **Sentence Pooling (Mean Pooling vs. CLS Pooling)**: Aggregating all contextual token vectors into a single, unified sentence vector.
5. **L2 Unit Normalization**: Projecting the pooled vector onto the unit hypersphere (magnitude = 1.0) for high-speed cosine comparison.

This lesson explores each stage of the embedding lifecycle, detailing why proper pooling masks and tokenization handling are critical to retrieval accuracy.
""")

st.divider()

# Section 1: The 5-Stage Embedding Pipeline Architecture
st.header("1. The 5-Stage Neural Transformation Lifecycle", divider="rainbow")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("""
    ### Engineering Breakdown:
    * **Stage 1 — Subword Tokenization**:
      Splits text into known subwords and prefixes/suffixes, appending special tokens:
      `[CLS] Python is a programming language [SEP]`.
    * **Stage 2 — Static Token Embedding Matrix**:
      Retrieves static vector weights from an embedding table of shape (Vocabulary_Size x Embedding_Dimension).
    * **Stage 3 — Transformer Contextual Layers**:
      12 to 24 multi-head self-attention layers compute contextual interactions: each word's representation shifts dynamically based on surrounding tokens.
    * **Stage 4 — Attention-Masked Mean Pooling**:
      Averages only the valid non-padding token vectors:
      `v_pooled = sum(v_i * mask_i) / sum(mask_i)`.
    * **Stage 5 — L2 Normalization**:
      Scales the pooled vector to unit length: `v_final = v_pooled / norm(v_pooled)`.
    """)

with col_p2:
    st.markdown("### End-to-End Pipeline Flow")
    pipe_dot = """
    digraph TextToEmbPipe {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        TextIn [label="1. Raw Input Text String\\n'Python is a programming language'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Tokenize [label="2. Subword Tokenizer\\n[101, 18750, 2003, 1037, 4731, 2653, 102]", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Transformer [label="3. Transformer Encoder (BERT/RoBERTa)\\nContextual Hidden States (Seq_Len x 768)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
        Pooling [label="4. Masked Mean Pooling\\nAverage hidden states across tokens", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        Norm [label="5. L2 Normalization\\nUnit Hypersphere Projection (norm = 1.0)", fillcolor="#DCFCE7", color="#16A34A"];
        VectorOut [label="Final Dense Embedding Vector\\n[0.021, -0.143, 0.782, ..., 0.091]", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        TextIn -> Tokenize -> Transformer -> Pooling -> Norm -> VectorOut;
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Embedding Pipeline in Python", divider="rainbow")
st.markdown("""
The following Python class simulates the five-stage text-to-embedding pipeline, implementing token ID lookup, attention-weighted mean pooling, and L2 unit normalization.
""")

class ProductionTextToEmbeddingPipeline:
    """
    Simulates the 5-stage neural embedding generation pipeline:
    Tokenization, contextual hidden state simulation, masked pooling, and normalization.
    """
    def __init__(self, hidden_dim: int = 8):
        self.hidden_dim = hidden_dim

    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """Simulates subword tokenization with special tokens."""
        words = text.strip().split()
        tokens = [("[CLS]", 101)]
        for i, w in enumerate(words):
            # Generate deterministic token ID from word hash
            token_id = 1000 + (abs(hash(w.lower())) % 8000)
            tokens.append((w, token_id))
        tokens.append(("[SEP]", 102))
        return tokens

    def compute_hidden_states(self, tokens: List[Tuple[str, int]]) -> np.ndarray:
        """Simulates contextual hidden states from transformer encoder (Seq_Len x Hidden_Dim)."""
        np.random.seed(42)
        seq_len = len(tokens)
        hidden_states = np.zeros((seq_len, self.hidden_dim), dtype=np.float64)

        for i, (word, token_id) in enumerate(tokens):
            np.random.seed(token_id)
            # Base token representation
            base_vec = np.random.uniform(-0.5, 0.5, self.hidden_dim)
            # Add positional / contextual bias
            context_bias = np.sin(np.arange(self.hidden_dim) * (i + 1)) * 0.2
            hidden_states[i] = base_vec + context_bias

        return hidden_states

    def pool_and_normalize(
        self,
        hidden_states: np.ndarray,
        tokens: List[Tuple[str, int]],
        strategy: str = "Mean Pooling"
    ) -> Tuple[np.ndarray, float]:
        """
        Applies sentence pooling (Mean Pooling vs CLS Pooling)
        and enforces L2 unit normalization.
        """
        if strategy == "CLS Pooling":
            # Select first token [CLS]
            pooled = hidden_states[0]
        else: # Mean Pooling
            # Exclude [CLS] and [SEP] special tokens from mean pooling
            content_states = hidden_states[1:-1]
            if len(content_states) > 0:
                pooled = np.mean(content_states, axis=0)
            else:
                pooled = hidden_states[0]

        raw_norm = float(np.linalg.norm(pooled))
        normalized_vec = pooled / raw_norm if raw_norm > 0 else pooled
        return normalized_vec, raw_norm

code_snippet = r'''
import numpy as np

class TextToEmbeddingPipeline:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate_embedding(self, text: str, pooling: str = "mean") -> np.ndarray:
        # 1. Tokenize text with attention mask
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # 2. Transformer forward pass
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state # (Batch, Seq_Len, Hidden_Dim)
        
        # 3. Masked Mean Pooling (crucial: ignore padding tokens)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 4. L2 Normalization
        normalized = pooled / pooled.norm(dim=-1, keepdim=True)
        return normalized.squeeze().cpu().numpy()
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Pipeline Visualizer Workbench
st.header("3. Interactive Pipeline Visualizer & Step-by-Step Inspector", divider="rainbow")
st.markdown("""
Trace an input sentence through every transformation stage: **Tokenization -> Contextual Hidden States -> Pooling -> Unit Normalization**.
""")

pipeline_engine = ProductionTextToEmbeddingPipeline(hidden_dim=8)

sample_sentences = [
    "Python is an interpreted high-level programming language.",
    "Machine learning embeddings map semantics to geometric vectors.",
    "Neural search engines retrieve documents using cosine similarity.",
    "Deep transformer encoders contextualize words through multi-head self-attention."
]

input_text = st.selectbox("Select sample sentence or enter custom text below:", sample_sentences)
custom_text = st.text_input("Custom text (optional):", value=input_text)
active_text = custom_text if custom_text.strip() else input_text

col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    pooling_method = st.radio("Sentence Pooling Strategy:", ["Mean Pooling (Recommended)", "CLS Pooling"])
with col_opt2:
    st.info(
        "**Mean Pooling** averages all content tokens, capturing balanced semantic breadth. "
        "**CLS Pooling** extracts the first token vector [CLS], common in older BERT classifiers."
    )

if st.button("Execute Text-to-Embedding Pipeline", type="primary"):
    st.markdown("#### Step 1: Subword Tokenization & Token IDs")
    token_list = pipeline_engine.tokenize(active_text)
    
    token_df = pd.DataFrame([
        {"Token Index": i, "Subword Token": t[0], "Token ID": t[1], "Type": "Special Token" if t[0] in ["[CLS]", "[SEP]"] else "Content Word"}
        for i, t in enumerate(token_list)
    ])
    st.dataframe(token_df, use_container_width=True)

    st.markdown("#### Step 2: Contextual Hidden States Matrix (Simulated 8-Dim)")
    hidden_matrix = pipeline_engine.compute_hidden_states(token_list)
    hidden_df = pd.DataFrame(hidden_matrix, columns=[f"Dim_{j+1}" for j in range(8)], index=[t[0] for t in token_list])
    st.dataframe(hidden_df.style.format("{:.3f}").background_gradient(cmap="coolwarm", axis=1), use_container_width=True)

    st.markdown("#### Step 3: Sentence Pooling & L2 Normalization")
    final_vector, pre_norm = pipeline_engine.pool_and_normalize(hidden_matrix, token_list, strategy=pooling_method)
    
    col_out1, col_out2, col_out3 = st.columns(3)
    col_out1.metric("Pre-Norm Magnitude", f"{pre_norm:.4f}")
    col_out2.metric("Post-Norm L2 Length", f"{np.linalg.norm(final_vector):.4f}", delta="Unit Vector")
    col_out3.metric("Vector Dimensionality", f"{len(final_vector)} Dimensions")

    st.markdown("##### Final Output Continuous Vector Coordinates:")
    out_df = pd.DataFrame([final_vector], columns=[f"Dim_{j+1}" for j in range(8)])
    st.dataframe(out_df.style.format("{:.4f}").background_gradient(cmap="viridis", axis=1), use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Pipeline Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is Masked Mean Pooling and why must the attention mask be included when averaging token vectors?**
    * **Answer**:
      * In deep learning batch processing, sequences of varying lengths are padded with `[PAD]` tokens (token ID 0) to equal the maximum sequence length.
      * If you naively call `np.mean(hidden_states, axis=0)`, the zero-padding vectors are included in the denominator and numerator, severely diluting the embedding magnitude and shifting coordinates toward zero.
      * **Masked Mean Pooling**: Multiplies hidden states by the binary `attention_mask` (1 for real tokens, 0 for padding) and divides strictly by the sum of real tokens:
        `pooled = sum(hidden_states * mask) / sum(mask)`.

    **Q2: Why does Mean Pooling outperform [CLS] token pooling for Sentence Transformers?**
    * **Answer**:
      * In the original BERT model, the `[CLS]` token was trained specifically for Next Sentence Prediction (NSP) classification, not semantic similarity.
      * Empirical research (Reimers & Gurevych, *Sentence-BERT*, 2019) proved that taking the average (mean) of all contextualized token representations captures significantly richer sentence semantics than relying on the single `[CLS]` token alone.

    **Q3: How do transformer embedding models handle texts longer than their maximum context window (e.g. 512 tokens)?**
    * **Answer**:
      * Standard BERT/MiniLM models strictly truncate text beyond 512 tokens, silently discarding the rest.
      * **Production Solutions**:
        1. **Text Chunking**: Split long documents into 256-512 token chunks with 15% overlap, embed each chunk independently, and store them with parent document metadata.
        2. **Hierarchical / Late Chunking**: Embed long documents using extended-context models (e.g., Jina-v2 or Voyage with 8,192 token windows) and average paragraph embeddings.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting the Attention Mask during Batch Mean Pooling**:
      * *Issue*: Averaging across padding tokens in batch inference.
      * *Consequence*: Short sentences padded with 400 zeros become severely degraded in magnitude and fail similarity search.
      * *Fix*: Always multiply by `attention_mask` before computing the mean.

    * **Trap 2: Ignoring Subword Token Splitting on Out-of-Vocabulary Terms**:
      * *Issue*: Expecting the tokenizer to return a single token for specialized domain terms like `"neuroblastoma"`.
      * *Consequence*: The tokenizer splits it into `["neuro", "##blast", "##oma"]`. If code assumes 1 word = 1 token, alignment fails.
      * *Fix*: Always operate on token IDs and rely on subword pooling.

    * **Trap 3: Omitting Special Tokens ([CLS] and [SEP])**:
      * *Issue*: Passing raw token sequences to BERT without mandatory start `[CLS]` and end `[SEP]` markers.
      * *Consequence*: Degrades self-attention accuracy; transformer positional embeddings expect standard boundary delimiters.
      * *Fix*: Always use `tokenizer.encode(add_special_tokens=True)`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Masked Mean Pooling**
      * *Definition*: Averaging contextual token hidden states while strictly ignoring zero-padding tokens using the attention mask.
    * **Flashcard 2: Subword Tokenization**
      * *Definition*: Splitting words into frequent subword units (BPE/WordPiece) to handle out-of-vocabulary terms gracefully.
    * **Flashcard 3: Self-Attention Contextualization**
      * *Definition*: The transformer mechanism where token representations dynamically incorporate information from all other tokens in the sequence.
    * **Flashcard 4: [CLS] Pooling**
      * *Definition*: Using the hidden state of the first special classification token as the representation for the entire sequence.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Text-to-Embedding Pipelines:

- [ ] I can describe the 5 stages of the text-to-embedding neural transformation lifecycle.
- [ ] I understand how transformer self-attention contextualizes ambiguous words.
- [ ] I know why Masked Mean Pooling is superior to naive average pooling and [CLS] pooling.
- [ ] I can explain why padding tokens must be masked out before computing sentence vectors.
- [ ] I understand how long texts exceeding context limits must be chunked prior to embedding.
""")
