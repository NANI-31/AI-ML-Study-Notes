import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="8.1 Tokenization Basics - Embeddings Curriculum",
    page_icon="🔤",
    layout="wide"
)

# Title & Overview
st.title("8.1 Tokenization Basics: Subwords, Vocabulary Encodings, & Token-ID Projections")
st.caption("Chapter 8: Tokenization and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Before a neural network can generate a single floating-point coordinate for a sentence, it must first solve a fundamental computer science challenge: **How do we convert continuous natural human language into discrete numerical symbols?**

Computers do not read words; they process numerical matrices. The bridge between human text and neural tensor representations is **Tokenization**.

Historically, NLP attempted:
1. **Character-level Tokenization**: Tiny vocabulary (128 ASCII bytes), but sequences become excessively long, and individual letters lack semantic meaning.
2. **Word-level Tokenization**: Words hold meaning, but vocabularies explode to millions of words, and novel or misspelled words immediately crash as **Out-of-Vocabulary (OOV)** tokens (`<UNK>`).
3. **Subword Tokenization (BPE / WordPiece / Unigram)**: The universal foundation of all modern embedding models! Frequently used words remain whole words; rare, technical, or compound words are broken into meaningful subword morphemes.

This lesson explores subword tokenization mechanics, vocabulary lookups, and token-to-vector projections.
""")

st.divider()

# Section 1: The Evolution of Tokenization
st.header("1. The Evolution of Tokenization: Words vs. Characters vs. Subwords", divider="rainbow")

col_tok1, col_tok2 = st.columns([1, 1])

with col_tok1:
    st.markdown("""
    ### Why Modern Embeddings Rely Exclusively on Subwords:
    * **The Out-Of-Vocabulary (OOV) Catastrophe**:
      * In word-level tokenization, if the vocabulary is 50,000 words, seeing a domain-specific word like *"bioinformatics"* or a typo *"reciept"* results in `<UNK>` (Unknown). An `<UNK>` token completely wipes out vector meaning!
    * **The Subword Revolution (Sennrich et al., 2016)**:
      * Subword algorithms (Byte-Pair Encoding, WordPiece) break rare words into reusable morphological components:
        `"unpredictability"` -> `["un", "predict", "able", "ity"]`
      * Even if the model has never encountered the full word, it understands the composite semantics of its subword roots!
    * **Token IDs and Lookup Matrix**:
      * Each subword string maps to an integer **Token ID** in a fixed vocabulary (Vocab Size V approx 30,522 for BERT, 100,277 for OpenAI cl100k_base).
      * The model retrieves initial token vectors by indexing into an **Embedding Matrix** W of shape [Vocab Size V, Hidden Dim d].
    """)

with col_tok2:
    st.markdown("### Tokenization to Embedding Matrix Lookup")
    tok_dot = """
    digraph TokenFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawStr [label="Raw Text: 'Deep learning is unbelievable'\\n(4 Words)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Subwords [label="Subword Tokenizer (BPE/WordPiece)\\n['Deep', 'learning', 'is', 'un', '##believ', '##able']\\n(6 Subwords)", fillcolor="#FEF3C7"];
        
        SpecialTokens [label="Special Token Insertion\\n[CLS], 'Deep', ..., '##able', [SEP]", fillcolor="#FDE68A"];
        
        TokenIDs [label="Token IDs (Integers)\\n[101, 2784, 4083, 2003, 4895, 29045, 3085, 102]", fillcolor="#D1FAE5"];
        
        LookupMatrix [label="Embedding Lookup Matrix W (V x d)\\nExtracts initial row vectors\\nShape: [seq_len, hidden_dim]", fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];

        RawStr -> Subwords -> SpecialTokens -> TokenIDs -> LookupMatrix;
    }
    """
    st.graphviz_chart(tok_dot, use_container_width=False)

st.markdown("### Tokenization Paradigms Comparison")
st.markdown("""
| Tokenization Type | Vocab Size V | Sequence Length for 100 Words | Out-of-Vocabulary (OOV) Rate | Semantic Meaning per Token | Primary Implementations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Character-Level** | ~256 bytes | **~500 - 600 tokens** | 0% | Near-Zero | Canine, ByT5 |
| **Word-Level** | 500,000+ | ~100 tokens | **High (2% - 8%)** | High | Word2Vec, GloVe, FastText |
| **Subword (WordPiece)** | ~30,000 | **~125 - 140 tokens** | **0% (Infinite Open Vocab)** | **High** | BERT, SBERT, BAAI BGE |
| **Byte-Pair Encoding (BPE)** | ~50,000 - 100k | **~120 - 130 tokens** | **0% (Byte Fallback)** | **High** | RoBERTa, GPT-4, LLaMA |
""")

st.divider()

# Section 2: Production Code: Subword Decomposition Engine
st.header("2. Production Implementation: Subword Tokenizer Simulator", divider="rainbow")

st.markdown("""
The production utility below demonstrates how modern subword tokenizers decompose text into subwords, assign numerical vocabulary IDs, compute token-to-word compression ratios, and track sequence context footprints.
""")

st.code(r'''
from typing import List, Dict, Any, Tuple

class SubwordTokenizerSimulator:
    """
    Simulates WordPiece / BPE subword decomposition and token ID assignment.
    Demonstrates prefix merging (##), special token attachment, and compression metrics.
    """
    def __init__(self):
        # Sample vocabulary mapping for illustration
        self.vocab = {
            "[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102,
            "deep": 2784, "learning": 4083, "is": 2003, "fast": 3435,
            "neural": 15758, "network": 2897, "model": 2944, "models": 4275,
            "un": 4895, "##believ": 29045, "##able": 3085, "##ing": 2075,
            "vector": 7254, "##s": 2015, "embed": 12890, "##ding": 4652
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize_and_encode(self, text: str) -> Dict[str, Any]:
        """
        Decomposes input words into subwords and retrieves corresponding token IDs.
        """
        words = text.lower().strip().split()
        tokens = ["[CLS]"]
        
        for w in words:
            if w in self.vocab:
                tokens.append(w)
            elif w == "unbelievable":
                tokens.extend(["un", "##believ", "##able"])
            elif w == "embeddings":
                tokens.extend(["embed", "##ding", "##s"])
            elif w == "vectors":
                tokens.extend(["vector", "##s"])
            else:
                tokens.append(w) # Fallback token

        tokens.append("[SEP]")
        
        token_ids = [self.vocab.get(t, 1000 + abs(hash(t)) % 25000) for t in tokens]
        
        return {
            "word_count": len(words),
            "token_count": len(tokens),
            "tokens": tokens,
            "token_ids": token_ids,
            "compression_ratio": len(tokens) / max(len(words), 1)
        }
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveTokenizerWorkbench:
    def __init__(self):
        self.mock_subword_splits = {
            "unbelievable": ["un", "##believ", "##able"],
            "embeddings": ["embed", "##ding", "##s"],
            "embedding": ["embed", "##ding"],
            "transformers": ["transform", "##ers"],
            "bioinformatics": ["bio", "##inform", "##atics"],
            "multimodal": ["multi", "##modal"],
            "tokenization": ["token", "##ization"],
            "hyperparameter": ["hyper", "##param", "##eter"],
            "vectorization": ["vector", "##ization"]
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        raw_words = text.strip().split()
        subwords = ["[CLS]"]
        
        for w in raw_words:
            clean_w = w.lower().strip(".,!?:;")
            if clean_w in self.mock_subword_splits:
                subwords.extend(self.mock_subword_splits[clean_w])
            else:
                subwords.append(clean_w)
                
        subwords.append("[SEP]")
        
        token_ids = [101 if t == "[CLS]" else (102 if t == "[SEP]" else 2000 + abs(hash(t)) % 28000) for t in subwords]
        
        return {
            "words": raw_words,
            "word_count": len(raw_words),
            "tokens": subwords,
            "token_count": len(subwords),
            "token_ids": token_ids,
            "ratio": len(subwords) / max(len(raw_words), 1)
        }

st.divider()

# Section 3: Interactive Workbench: Subword Tokenizer Explorer
st.header("3. Interactive Workbench: Subword Tokenizer Explorer", divider="rainbow")

st.markdown("""
Type or select an input sentence containing complex technical terms. Observe how the subword tokenizer decomposes compound and inflected words into morphological subwords with continuation prefixes (`##`), attaches special tokens (`[CLS]` and `[SEP]`), and computes vocabulary token IDs.
""")

tok_bench = InteractiveTokenizerWorkbench()

col_tok_l, col_tok_r = st.columns([1, 1])

with col_tok_l:
    st.subheader("1. Input Text String")
    sample_choice = st.selectbox(
        "Select Sample Sentence or Type Below:",
        [
            "Deep learning transformers use unbelievable tokenization for embeddings.",
            "Multimodal vectorization requires bioinformatics and hyperparameter tuning.",
            "Dense embeddings represent continuous semantic geometries."
        ]
    )

    custom_text = st.text_area("Input Text to Tokenize:", value=sample_choice, height=90)
    tok_data = tok_bench.process_text(custom_text)

    st.markdown("#### Subword Token Breakdown")
    token_display = " ".join([f"`{t}`" for t in tok_data["tokens"]])
    st.markdown(f"**Tokens**: {token_display}")

with col_tok_r:
    st.subheader("2. Tokenizer Statistics & Encodings")
    m1, m2 = st.columns(2)
    m1.metric("Word Count", f"{tok_data['word_count']} words")
    m2.metric("Total Tokens (incl. [CLS]/[SEP])", f"{tok_data['token_count']} tokens")

    m3, m4 = st.columns(2)
    m3.metric("Tokens per Word Ratio", f"{tok_data['ratio']:.2f}x")
    m4.metric("Context Window Consumption", f"{tok_data['token_count']} / 512 tokens")

    st.markdown("#### Token to Numerical Vocabulary ID Mapping")
    df_tok = pd.DataFrame({
        "Position": range(len(tok_data["tokens"])),
        "Subword Token": tok_data["tokens"],
        "Token ID (Integer)": tok_data["token_ids"]
    })
    st.dataframe(df_tok, use_container_width=True)

    if tok_data['ratio'] > 1.5:
        st.warning(f"⚠️ **High Subword Fragmentation ({tok_data['ratio']:.2f}x)**: Technical terminology is split into multiple subwords. Ensure your chunk length allows for ~1.4x token expansion over raw word count!")
    else:
        st.success(f"✅ **Efficient Tokenization ({tok_data['ratio']:.2f}x)**: High subword reuse preserves context window economy.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Tokenization Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the difference between Byte-Pair Encoding (BPE) and WordPiece tokenization?**
    * **Answer**:
      * Both are bottom-up subword algorithms that begin with characters and iteratively merge frequent pairs.
      * **BPE (Byte-Pair Encoding)**: Merges the pair of symbols that has the highest raw co-occurrence frequency in the training corpus. Used by GPT-2, GPT-4, RoBERTa, LLaMA.
      * **WordPiece**: Instead of raw frequency, WordPiece chooses the pair that maximizes the likelihood of the training data according to a language model (maximizing mutual information). Marked by continuation prefix `##` (e.g. `transform`, `##ers`). Used by BERT and standard Sentence Transformers.

    **Q2: Why is the rule of thumb '1 word = 1.3 tokens' important when sizing chunk boundaries for RAG?**
    * **Answer**:
      * Embedding models enforce a hard maximum sequence length (e.g. 512 tokens).
      * If an engineer naively configures text chunking using a Python word split (`len(text.split()) == 500`), the text actually consumes `500 * 1.3 = 650 tokens`.
      * The final 138 tokens will be silently clipped by the model, destroying the end of every chunk!
      * Always use a token-aware tokenizer (e.g. `tiktoken` or Hugging Face `tokenizer.tokenize`) rather than character or word approximations.

    **Q3: What role do special tokens ([CLS], [SEP], <s>, </s>) play in embedding architectures?**
    * **Answer**:
      * `[CLS]` (Classification / Sequence Start): Positioned at index 0. In models trained for CLS pooling, its final hidden state acts as the pooled sentence vector.
      * `[SEP]` (Separator): Positioned at the end of each sentence or between query and passage. Tells self-attention layers where text segments terminate.
      * They occupy real token positions and consume 2 tokens of the model's sequence length budget.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Assuming 1 Token Equals 1 Word**:
      * *Issue*: Configuring chunk size to 512 words on a model with a 512-token limit.
      * *Consequence*: Causes severe silent truncation; 20-30% of content is discarded.
      * *Fix*: Target 350-400 words per chunk to guarantee staying below 512 tokens.

    * **Trap 2: Multilingual Token Inflation**:
      * *Issue*: Ingesting non-English documents (e.g. Chinese, Japanese, Hindi) with an English-centric tokenizer.
      * *Consequence*: The tokenizer lacks native subwords and falls back to byte-level representations, taking 3-5 tokens per single character! A 100-character sentence consumes 400 tokens, exhausting the context window instantly.
      * *Fix*: Deploy multilingual models with balanced multilingual vocabularies (e.g. `XLM-RoBERTa` or `bge-m3`).

    * **Trap 3: Mismatching Tokenizer and Model Weights**:
      * *Issue*: Tokenizing text with `bert-base-uncased` tokenizer and feeding the token IDs into a `roberta-base` model.
      * *Consequence*: Token IDs point to completely different words in the respective vocabulary tables, producing nonsensical garbage embeddings.
      * *Fix*: Always instantiate tokenizer and model from the identical Hugging Face model checkpoint.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Subword Tokenization**
      * *Definition*: Decomposing text into reusable morphemic subwords, eliminating Out-of-Vocabulary errors while maintaining compact vocabulary sizes.
    * **Flashcard 2: Token ID**
      * *Definition*: The unique integer index mapped to a specific subword in the embedding model's vocabulary table.
    * **Flashcard 3: Embedding Matrix (W_e)**
      * *Definition*: The weight matrix of shape [Vocab Size, Hidden Dim] used to look up initial dense coordinate vectors for input token IDs.
    * **Flashcard 4: Token Inflation**
      * *Definition*: The expansion of text into excessive tokens that occurs when an English-trained tokenizer processes code or non-Latin scripts.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Tokenization Basics:

- [ ] I can articulate why subword tokenization replaced word-level and character-level approaches.
- [ ] I understand how Token IDs index into the model's Embedding Matrix to retrieve dense vectors.
- [ ] I know why 1 word typically expands to ~1.3 tokens in English and much higher in other languages.
- [ ] I know how to calculate context window consumption accurately using token-aware tools.
- [ ] I understand the function of special tokens (`[CLS]`, `[SEP]`) in transformer input sequences.
""")
