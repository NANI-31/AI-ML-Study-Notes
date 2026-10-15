import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="3.1 Tokenization Basics - LLM Curriculum",
    page_icon="🧩",
    layout="wide"
)

# Title & Overview
st.title("3.1 Tokenization Basics: The Gateway from Human Language to LLM Representations")
st.caption("Chapter 3: Tokenization | Domain: 2. Large Language Models")
st.markdown("""
Tokenization is the foundational translation bridge between human natural language and the numerical world of deep learning.
Large Language Models do not possess eyes or intuitive linguistic awareness; they cannot read characters, words, or sentences directly.
Instead, tokenizers discretize continuous text streams into integer token IDs mapped from a fixed mathematical vocabulary.
Mastering tokenization mechanics is crucial for understanding model context windows, latency economics, compute complexity, and cross-lingual performance.

In this lesson, you will master:
1. The end-to-end transformation pipeline from raw characters to token IDs, dense embeddings, and transformer attention heads.
2. The granular trade-offs of character-level, word-level, subword-level, and sentence-level tokenization.
3. Why token IDs are non-semantic discrete indices and how vocabulary size constraints dictate neural network architecture.
4. The structural mechanics of special control tokens (BOS, EOS, PAD, CLS, SEP) and batched attention masks.
5. Production implications including tokenization inflation, context window budgeting, cost estimation, and non-prose edge cases.
""")

st.divider()

# ─── Section 1: The End-to-End LLM Input Pipeline ───────────────────────────
st.header("1. The End-to-End LLM Input Pipeline", divider="rainbow")

col_pipe_a, col_pipe_b = col_pipe_a, col_pipe_b = st.columns([1.1, 0.9])

with col_pipe_a:
    st.markdown("""
    ### From Human Characters to Mathematical Tensors
    Neural network layers cannot perform matrix multiplications directly against Unicode strings.
    Before a transformer model can calculate a single attention weight, text must pass through a strict multi-stage ingestion pipeline:

    1. **Raw Text Input**: The user sends a string such as `"I love Python"`.
    2. **Token Segmentation**: The tokenizer breaks the text into discrete textual subunits: `["I", " love", " Python"]`.
    3. **Vocabulary Lookup (Token IDs)**: Each token string is mapped to a unique integer index in a static lookup table: `[40, 1842, 11454]`.
    4. **Embedding Layer**: The integer IDs index into a massive weight matrix of shape `[Vocab_Size, Hidden_Dimension]`, retrieving dense continuous vectors.
    5. **Positional Encoding**: Sequential ordering vectors are added to inject word order information.
    6. **Transformer Blocks**: The resulting continuous tensors feed into multi-head self-attention and feed-forward networks.
    """)

with col_pipe_b:
    st.markdown("### Pipeline Architecture Flow")
    dot_pipeline = """
    digraph TokenPipeline {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        // Tier 1: Ingress & Client Input
        subgraph cluster_ingress {
            label="Tier 1: Client Ingress";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            style="rounded";
            color="#334155";
            raw [label="User Input Text\\n'I love Python'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        // Tier 2: Tokenization Engine
        subgraph cluster_tokenizer {
            label="Tier 2: Tokenizer Engine (Byte-Pair Encoding)";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            style="rounded";
            color="#334155";
            seg [label="Subword Segmentation\\n['I', ' love', ' Python']", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            lookup [label="Vocabulary Index Lookup\\n[40, 1842, 11454]", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            vocab_db [label="Static Vocabulary Table\\n(32k - 128k Entries)", shape=cylinder, fillcolor="#1E293B", fontcolor="#FDE68A", color="#D97706"];
        }

        // Tier 3: Embedding & Positional Encoding
        subgraph cluster_embeddings {
            label="Tier 3: Continuous Latent Space Projection";
            bgcolor="#1E293B";
            fontcolor="#A78BFA";
            style="rounded";
            color="#334155";
            emb [label="Embedding Lookup Matrix\\nW_emb [Vocab x 4096]", shape=cylinder, fillcolor="#4C1D95", fontcolor="#EDE9FE", color="#8B5CF6"];
            pos [label="Rotary / Absolute Position\\nPositional Vectors Added", fillcolor="#4C1D95", fontcolor="#EDE9FE", color="#8B5CF6"];
        }

        // Tier 4: Transformer Execution Core
        subgraph cluster_transformer {
            label="Tier 4: Deep Learning Attention Engine";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            style="rounded";
            color="#334155";
            attn [label="Multi-Head Self-Attention\\nQ, K, V Projections (Hidden d=4096)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        }

        raw -> seg [label="UTF-8 String"];
        seg -> lookup [label="Subword Tokens"];
        vocab_db -> lookup [label="Index Mapping", style="dashed", color="#D97706"];
        lookup -> emb [label="Token IDs [int64]"];
        emb -> pos [label="Dense Vectors"];
        pos -> attn [label="Input Tensor [1, Seq, 4096]"];
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(dot_pipeline, use_container_width=True)

st.markdown("### Token Array & Tensor Layout")
dot_token_array = """
digraph TokenArray {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9];

    token_array [shape=plaintext, label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Array Index</b></font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">0</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">1</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">2</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">3</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Subword Token</b></font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"I"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"love"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"Python"</font></td>
                <td bgcolor="#64748B"><font color="#FFFFFF">[PAD]</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Token ID (Vocab)</b></font></td>
                <td bgcolor="#D97706"><font color="#FFFFFF">40</font></td>
                <td bgcolor="#D97706"><font color="#FFFFFF">1842</font></td>
                <td bgcolor="#D97706"><font color="#FFFFFF">11454</font></td>
                <td bgcolor="#475569"><font color="#CBD5E1">0</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Attention Mask</b></font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#991B1B"><font color="#FCA5A5">0</font></td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_token_array, use_container_width=True)

st.divider()

# ─── Section 2: Core Tokenization Concepts & Subtle Traps ─────────────────────
st.header("2. What is Tokenization? Core Principles & Traps", divider="rainbow")

col_prin_a, col_prin_b = col_prin_a, col_prin_b = st.columns([1, 1])

with col_prin_a:
    st.markdown("""
    ### What Constitutes a Token?
    A token is not synonymous with a word or a syllable. Depending on the model's vocabulary, a single token can be:
    * A full English word: `"apple"`
    * A subword morpheme: `"un"`, `"break"`, `"able"`
    * A single ASCII or Unicode character: `"a"`, `"!"`, `"π"`
    * A whitespace sequence or indentation tab: `"    "` (4 spaces)
    * A raw byte: `\\xe2\\x9c\\x93` (UTF-8 byte sequence for checkmark)
    * A reserved control code: `<|endoftext|>`, `[PAD]`

    ### Critical Trap: Whitespace Sensitivity
    Computers treat whitespace as explicit data. In modern subword tokenizers (like GPT's BPE or Llama's SentencePiece), 
    **a word with a preceding space is a completely different token from the same word without one**:
    * `"love"` (no leading space) -> Token ID `1842`
    * `" love"` (with leading space, represented as `Ġlove` in BPE) -> Token ID `3021`
    
    If your software strips or injects whitespace unexpectedly, you alter the underlying token sequences and disrupt model attention.
    """)

with col_prin_b:
    st.markdown("""
    ### Tokenization is Strictly Model-Specific
    There is no universal "standard" tokenization dictionary. Tokenization algorithms, vocabulary tables, 
    and integer assignments are trained alongside the specific neural network:
    * Token ID `101` in BERT (`bert-base-uncased`) represents the special token `[CLS]`.
    * Token ID `101` in GPT-2 represents the character `"e"`.
    * Token ID `101` in Llama 3 represents `"#"`.
    
    **Rule**: Never pass token IDs generated by one model into another model. Every model requires its own dedicated tokenizer binary.

    ### Tokenization vs Traditional Text Preprocessing
    Traditional NLP pipelines relied on aggressive lossy preprocessing:
    * Lowercasing (collapsing `"Apple"` and `"apple"`)
    * Punctuation removal (stripping commas, hyphens, and periods)
    * Stop-word removal (deleting `"is"`, `"the"`, `"and"`)
    * Stemming/Lemmatization (truncating words to grammatical roots)

    Modern LLMs operate on **zero-loss raw text**. Casing signals proper nouns and acronyms; punctuation conveys syntactic pauses and code structure; stop words preserve grammatical nuances. Modern tokenizers retain every single character.
    """)

st.divider()

# ─── Section 3: The Granularity Spectrum: Character vs Word vs Subword ────────
st.header("3. The Granularity Spectrum: Character, Word, and Subword", divider="rainbow")

st.markdown("""
Every tokenization strategy represents a fundamental engineering trade-off between **vocabulary size** (memory overhead in the embedding table) 
and **sequence length** (compute overhead in quadratic attention layers).
""")

col_g1, col_g2, col_g3 = col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.markdown("""
    #### 1. Character Tokenization
    Decomposes text into raw individual characters (`list(text)`).
    * **Vocab Size**: Ultra-compact (~256 ASCII or ~2,000 Unicode characters).
    * **Out-of-Vocabulary (OOV)**: Zero OOV risk. Can represent any new word, typo, or neologism.
    * **Sequence Length**: Extremely long. A 100-word paragraph becomes 600+ tokens.
    * **Semantic Density**: Weak. Individual characters (`'c'`, `'a'`, `'t'`) hold zero innate semantic meaning.
    * **Attention Penalty**: Catastrophic quadratic memory explosion due to sequence length expansion.
    """)

with col_g2:
    st.markdown("""
    #### 2. Word Tokenization
    Splits text on whitespace boundaries (`text.split()`).
    * **Vocab Size**: Massive (500,000+ to millions of distinct vocabulary entries).
    * **Out-of-Vocabulary (OOV)**: Severe vulnerability. Typos, slang, or inflections trigger `[UNK]`.
    * **Sequence Length**: Minimal. 1 token per word.
    * **Semantic Density**: Very high per token.
    * **Punctuation Flaw**: Naive splitting attaches punctuation to adjacent words (`"python."` becomes a distinct token from `"python"`).
    """)

with col_g3:
    st.markdown("""
    #### 3. Subword Tokenization
    The universal standard for modern LLMs (BPE, WordPiece, SentencePiece).
    * **Vocab Size**: Balanced sweet spot (32,000 to 128,000 tokens).
    * **Out-of-Vocabulary (OOV)**: Zero OOV risk. Falls back to sub-morphemes or byte representations.
    * **Sequence Length**: Balanced (approx. 1.3 tokens per English word).
    * **Semantic Density**: High for frequent terms; compositional for rare terms (`"playingly"` -> `"playing" + "ly"`).
    * **Adoption**: GPT-4, Llama 3, Claude 3, DeepSeek, Gemini, Mistral.
    """)

st.markdown("### Code Demonstration: Character vs Word vs Subword Splitting")

demo_code = '''# Comparing Naive Splitting Flaws
text = "Machine learning is fascinating, isn't it?"

# 1. Character Tokenization
char_tokens = list(text)
print("Char Token Count:", len(char_tokens))  # 42 tokens!
print("First 10 Chars:", char_tokens[:10])

# 2. Word Tokenization via whitespace
word_tokens = text.split()
print("Word Token Count:", len(word_tokens))  # 6 tokens
print("Word Tokens:", word_tokens)
# Flaw: "fascinating," has comma attached; "it?" has question mark attached!

# 3. Subword decomposition concept
# Rare or compound words are broken into meaningful morphological roots
rare_word = "unbelievably"
subwords = ["un", "believ", "ably"]  # Reusable sub-units across the vocabulary
print("Subword units for rare word:", subwords)'''

st.code(demo_code, language="python")

st.divider()

# ─── Section 4: Sentence Tokenization & Segmentation Challenges ───────────────
st.header("4. Sentence Tokenization & Segmentation Challenges", divider="rainbow")

col_sent_a, col_sent_b = col_sent_a, col_sent_b = st.columns([1, 1])

with col_sent_a:
    st.markdown("""
    ### What is Sentence Tokenization?
    Sentence tokenization (or sentence boundary detection) divides a document into grammatically complete sentence units.
    While modern autoregressive LLMs ingest subwords directly, sentence segmentation remains essential in production pipelines:
    * **RAG Chunking**: Slicing large knowledge base articles at natural semantic sentence boundaries rather than arbitrary character counts.
    * **Sentence Pair Models**: Cross-encoders for reranking (e.g., Query vs Document sentence matching).
    * **Translation Pipelines**: Aligning source and target sentences in parallel corpora.

    ### Why Simple Period Splitting Fails
    Splitting text on `.` fails immediately in real-world corpora:
    * **Abbreviations & Titles**: `"Dr. Watson visited the U.S.A. yesterday."` contains periods that do not mark terminal boundaries.
    * **Floating Point Numbers**: `"$12.50 per share"` contains a period that must not trigger a split.
    * **Ellipses & Punctuation Clusters**: `"Wait... what happened?!"`
    * **Quotation Boundaries**: `'"I agree," said Alice. "Let us proceed."'`
    """)

with col_sent_b:
    st.markdown("""
    ### Rule-Based Sentence Segmentation Code
    Modern sentence segmenters use regular expressions or statistical Punkt segmenters to avoid false splits:
    """)
    sent_code = '''import re

def robust_sentence_tokenize(text: str) -> list[str]:
    # Lookahead/lookbehind pattern avoiding abbreviations like Dr., Mr., U.S.A.
    pattern = r'(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?|\\!)\\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

document = "Dr. Smith bought 3.5 lbs of apples. He paid $7.50 at the U.S. market! Did he leave?"
sentences = robust_sentence_tokenize(document)

for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent}")
# Correctly yields 3 sentences without false splits on Dr., 3.5, $7.50, or U.S.'''
    st.code(sent_code, language="python")

st.divider()

# ─── Section 5: Token IDs, Embeddings, and Vocabulary Constraints ─────────────
st.header("5. Token IDs, Embeddings, and the Vocabulary Trade-off", divider="rainbow")

col_vocab_a, col_vocab_b = col_vocab_a, col_vocab_b = st.columns([1.1, 0.9])

with col_vocab_a:
    st.markdown("""
    ### Token IDs are Arbitrary Indices, Not Semantic Vectors
    A common misconception is that token IDs contain geometric or semantic information.
    **Token IDs are simply lookup keys into a table**:
    * If `"king"` is ID `501` and `"queen"` is ID `502`, their numerical closeness is purely coincidental.
    * ID `501` and ID `502` have zero mathematical distance or angle until they are mapped through the **Embedding Matrix**.
    
    The embedding layer is a weight matrix `W_emb` of shape `[Vocab_Size, Hidden_Dimension]`, where `Vocab_Size` is vocabulary size (e.g., 50,257 to 128,256) and `Hidden_Dimension` is embedding dimension (e.g., 4096).
    The integer token ID acts as a row index: selecting row `i` retrieves that token's `d`-dimensional continuous dense vector.

    ### Why Not Make Vocabulary Infinitely Large?
    If larger vocabularies yield shorter sequences and fewer subword fragments, why not create a 5-million-word vocabulary?
    * **Memory Footprint**: An embedding matrix with `Vocab_Size = 1,000,000` and `Hidden_Dim = 4096` requires `1,000,000 * 4096 * 2 bytes (fp16) = 8.19 GB` of GPU VRAM solely for the input embedding and output LM head projection layers!
    * **Softmax Bottleneck**: During inference decoding, the final unembedding layer computes a softmax over all `Vocab_Size` logits at every single token step. A huge vocabulary severely throttles decoding throughput.
    * **Sparse Training**: Infrequently occurring words in a massive vocabulary receive few gradient updates, leading to poorly trained representations.
    """)

with col_vocab_b:
    st.markdown("### Token ID vs Dense Embedding Comparison")
    comp_df = pd.DataFrame({
        "Attribute": ["Data Type", "Dimensionality", "Semantic Content", "Trainable?", "Role in Pipeline"],
        "Token ID": ["Integer (int32/int64)", "Scalar (1 number)", "Zero (arbitrary index)", "No (fixed mapping)", "Vocab lookup index"],
        "Embedding Vector": ["Floating point (fp16/bf16)", "Dense vector (d=4096)", "High (cosine similarity)", "Yes (updated via backprop)", "Attention input tensor"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Vocabulary Size Comparison Across Leading LLMs
    * **BERT**: 30,522 tokens (WordPiece)
    * **GPT-2**: 50,257 tokens (Byte-level BPE)
    * **GPT-4**: 100,277 tokens (tiktoken `cl100k_base`)
    * **Llama 3**: 128,256 tokens (tiktoken byte-fallback BPE)
    * **Gemma**: 256,000 tokens (SentencePiece)
    """)

st.markdown("### Dense Embedding Vector Table & Weight Matrix Lookup")
dot_vector_lookup = """
digraph EmbeddingLookup {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9];
    edge [color="#38BDF8", penwidth=1.5, arrowsize=0.8, fontname="Helvetica", fontsize=8, fontcolor="#38BDF8"];

    token_id [shape=plaintext, label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr><td bgcolor="#D97706"><font color="#FFFFFF"><b>Discrete Token ID: 11454</b><br/>(Subword: "Python")</font></td></tr>
            <tr><td bgcolor="#0F172A"><font color="#94A3B8">Integer Index (Scalar int64)</font></td></tr>
        </table>
    >];

    dense_vector [shape=plaintext, label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#1E293B">
            <tr><td colspan="6" bgcolor="#7C3AED"><font color="#FFFFFF"><b>Embedding Matrix Row: W_emb[11454, :] (Hidden Dimension d=4096)</b></font></td></tr>
            <tr bgcolor="#334155">
                <td><font color="#94A3B8">dim[0]</font></td>
                <td><font color="#94A3B8">dim[1]</font></td>
                <td><font color="#94A3B8">dim[2]</font></td>
                <td><font color="#94A3B8">dim[3]</font></td>
                <td><font color="#94A3B8">...</font></td>
                <td><font color="#94A3B8">dim[4095]</font></td>
            </tr>
            <tr bgcolor="#0F172A">
                <td><font color="#A78BFA">+0.214</font></td>
                <td><font color="#A78BFA">-0.432</font></td>
                <td><font color="#A78BFA">+0.729</font></td>
                <td><font color="#A78BFA">+0.015</font></td>
                <td><font color="#64748B">...</font></td>
                <td><font color="#A78BFA">-0.118</font></td>
            </tr>
        </table>
    >];

    token_id -> dense_vector [label="Row Index Lookup W_emb[11454]"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_vector_lookup, use_container_width=True)

st.divider()

# ─── Section 6: Special Tokens & The Attention Mask Architecture ──────────────
st.header("6. Special Tokens & The Attention Mask Architecture", divider="rainbow")

col_sp_a, col_sp_b = col_sp_a, col_sp_b = st.columns([1, 1])

with col_sp_a:
    st.markdown("""
    ### Control Codes of the Transformer Architecture
    Special tokens are reserved indices that instruct the model regarding sequence boundaries, classification targets, and batching layout:

    * `[BOS]` / `<s>` / `<|begin_of_text|>`: **Beginning of Sequence**. Informs the autoregressive model that a new sequence is starting.
    * `[EOS]` / `</s>` / `<|end_of_text|>`: **End of Sequence**. When the model outputs this token during text generation, the inference engine halts the generation loop.
    * `[CLS]`: **Classification Token**. In encoder models like BERT, the final hidden state of `[CLS]` aggregates document-level representation for classification.
    * `[SEP]`: **Separator Token**. Distinguishes sentence pairs in question answering or cross-encoding tasks (`[CLS] Question [SEP] Context [SEP]`).
    * `[PAD]`: **Padding Token**. Equalizes varying sequence lengths across a batch so tensors form rectangular matrices.
    * `[UNK]`: **Unknown Token**. Fallback placeholder when a token cannot be represented.
    """)

with col_sp_b:
    st.markdown("""
    ### The Attention Mask: Preventing Padding Pollution
    When packing multiple sentences into a batch, shorter sentences are padded with `[PAD]` tokens.
    Without an attention mask, self-attention layers would compute attention weights across meaningless padding tokens!

    The **Attention Mask** is a binary tensor matching the input shape:
    * `1`: Indicates a valid token that self-attention should attend to.
    * `0`: Indicates a padding token that must be masked out with negative infinity (-10000.0 or -inf) before softmax, zeroing its attention weight.
    """)
    pad_code = '''# Batched Encoding with Attention Masking
batch_sentences = [
    "I love deep learning",     # 4 tokens
    "Neural networks work"      # 3 tokens
]

# Padded to max_len = 5:
# Sentence 1: ['I', 'love', 'deep', 'learning', '[PAD]']
input_ids = [
    [101, 1842, 2784, 4083, 0],
    [3214, 2891, 1940, 0,    0]
]

# Attention Mask (1 = Attend, 0 = Ignore):
attention_mask = [
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0]
]
# Mask ensures padded positions receive zero attention weight!'''
    st.code(pad_code, language="python")

st.markdown("### Batched Sequence Alignment & Attention Masking Grid")
dot_batched_mask = """
digraph BatchedMask {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9];

    batch_table [shape=plaintext, label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#1E293B">
            <tr><td colspan="6" bgcolor="#334155"><font color="#FFFFFF"><b>Batched Input Tensors and Attention Mask Grid (Batch Size = 2, Max Len = 5)</b></font></td></tr>
            <tr bgcolor="#1E293B">
                <td bgcolor="#334155"><font color="#94A3B8"><b>Batch Sequence</b></font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">Pos 0</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">Pos 1</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">Pos 2</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">Pos 3</font></td>
                <td bgcolor="#0F172A"><font color="#38BDF8">Pos 4 (Pad)</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Seq 0 (Tokens)</b></font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"I"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"love"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"deep"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"learning"</font></td>
                <td bgcolor="#475569"><font color="#94A3B8">[PAD]</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Seq 0 (Mask)</b></font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#991B1B"><font color="#FCA5A5">0 (Masked)</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Seq 1 (Tokens)</b></font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"Neural"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"networks"</font></td>
                <td bgcolor="#0284C7"><font color="#FFFFFF">"work"</font></td>
                <td bgcolor="#475569"><font color="#94A3B8">[PAD]</font></td>
                <td bgcolor="#475569"><font color="#94A3B8">[PAD]</font></td>
            </tr>
            <tr>
                <td bgcolor="#334155"><font color="#94A3B8"><b>Seq 1 (Mask)</b></font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#059669"><font color="#FFFFFF">1</font></td>
                <td bgcolor="#991B1B"><font color="#FCA5A5">0 (Masked)</font></td>
                <td bgcolor="#991B1B"><font color="#FCA5A5">0 (Masked)</font></td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_batched_mask, use_container_width=True)

st.divider()

# ─── Section 7: Subword Tokenization Families: BPE, WordPiece, SentencePiece ──
st.header("7. Subword Tokenization Families: BPE, WordPiece, and SentencePiece", divider="rainbow")

col_fam1, col_fam2, col_fam3 = col_fam1, col_fam2, col_fam3 = st.columns(3)

with col_fam1:
    st.markdown("""
    #### Byte Pair Encoding (BPE)
    * **Algorithm**: Starts with individual characters/bytes. Iteratively counts all adjacent token pairs and merges the most frequent pair into a new token until desired vocab size is reached.
    * **Whitespace**: Pre-tokenizes on whitespace first (GPT-2, GPT-4, Llama 3).
    * **Used In**: GPT series, RoBERTa, Llama, Falcon.
    * **Key Trait**: Frequency-driven bottom-up merging.
    """)

with col_fam2:
    st.markdown("""
    #### WordPiece
    * **Algorithm**: Similar to BPE, but instead of merging based on raw pair frequency, it selects merges that maximize the likelihood of the training data according to a language model.
    * **Subword Prefix**: Denotes subwords with leading `##` (e.g., `"play"` + `"##ing"`).
    * **Used In**: BERT, DistilBERT, Electra.
    * **Key Trait**: Likelihood-driven vocabulary selection.
    """)

with col_fam3:
    st.markdown("""
    #### SentencePiece
    * **Algorithm**: Treats raw text as a continuous stream of Unicode characters without requiring language-dependent whitespace pre-tokenization.
    * **Meta-symbol**: Replaces whitespace with a meta-symbol (usually ` `).
    * **Used In**: T5, mT5, ALBERT, Gemma, LLaMA.
    * **Key Trait**: Fully language-agnostic; lossless reversible de-tokenization.
    """)

st.divider()

# ─── Section 8: Standalone Python Implementation: Production Tokenizer ────────
st.header("8. Standalone Python Implementation: Minimal Subword Tokenizer", divider="rainbow")

st.markdown("""
Below is a standalone, pure-Python implementation of a dictionary-driven subword tokenizer demonstrating encoding, decoding, 
special token injection, and attention mask generation without external dependencies:
""")

standalone_tok_code = '''from typing import List, Dict, Tuple

class ProductionMockTokenizer:
    """A self-contained demonstration subword tokenizer with padding and masking."""
    def __init__(self):
        # Base vocabulary with special tokens and subwords
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.subwords = {
            "i": 10, "love": 11, "python": 12, "machine": 13, 
            "learning": 14, "deep": 15, "neural": 16, "networks": 17,
            "un": 18, "break": 19, "able": 20, "play": 21, "ing": 22
        }
        self.vocab = {**self.special_tokens, **self.subwords}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def tokenize_word(self, word: str) -> List[str]:
        """Greedy subword decomposition."""
        w = word.lower()
        if w in self.vocab:
            return [w]
        # Check simple prefix/suffix splits
        for prefix in ["un"]:
            if w.startswith(prefix) and len(w) > len(prefix):
                remainder = w[len(prefix):]
                if remainder in self.vocab:
                    return [prefix, remainder]
                for suffix in ["able", "ing"]:
                    if remainder.endswith(suffix):
                        stem = remainder[:-len(suffix)]
                        if stem in self.vocab:
                            return [prefix, stem, suffix]
        return [w if w in self.vocab else "[UNK]"]

    def encode(self, text: str, max_length: int = 8) -> Tuple[List[int], List[int]]:
        """Encodes text with [CLS], [SEP], padding, and attention mask."""
        words = text.split()
        tokens = ["[CLS]"]
        for w in words:
            tokens.extend(self.tokenize_word(w))
        tokens.append("[SEP]")
        
        # Truncate if exceeds max_length
        tokens = tokens[:max_length]
        
        # Map to IDs
        token_ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]
        
        # Calculate attention mask before padding
        attention_mask = [1] * len(token_ids)
        
        # Apply padding
        pad_length = max_length - len(token_ids)
        token_ids.extend([self.vocab["[PAD]"]] * pad_length)
        attention_mask.extend([0] * pad_length)
        
        return token_ids, attention_mask

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back into readable string, filtering padding."""
        tokens = [self.inv_vocab.get(i, "[UNK]") for i in ids if i != self.vocab["[PAD]"]]
        filtered = [t for t in tokens if t not in ["[CLS]", "[SEP]"]]
        return " ".join(filtered)

# Execution test
tok = ProductionMockTokenizer()
ids, mask = tok.encode("I love deep learning", max_length=8)
print("Token IDs:     ", ids)             # [2, 10, 11, 15, 14, 3, 0, 0]
print("Attention Mask:", mask)            # [1, 1, 1, 1, 1, 1, 0, 0]
print("Decoded String:", tok.decode(ids)) # "i love deep learning"'''

st.code(standalone_tok_code, language="python")

st.markdown("### Industry Standard: Hugging Face AutoTokenizer Execution")
st.markdown("""
In real-world production engineering, rather than maintaining manual subword dictionaries, teams leverage optimized C++/Rust-backed tokenizers via the Hugging Face `transformers` ecosystem:
""")

hf_code = '''from transformers import AutoTokenizer

# Load industry-standard pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Machine learning engineers build scalable AI systems."

# 1. Direct subword decomposition
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Segmented Tokens:", tokens)
print("Vocabulary IDs:  ", token_ids)

# 2. Batched encoding with automatic padding and PyTorch tensor generation
encoded = tokenizer([text], padding=True, return_tensors="pt")
print("Tensors (input_ids):     ", encoded["input_ids"])
print("Tensors (attention_mask):", encoded["attention_mask"])'''

st.code(hf_code, language="python")

st.divider()

# ─── Section 9: Cross-Cutting Engineering Implications ────────────────────────
st.header("9. Cross-Cutting Engineering Implications", divider="rainbow")

col_imp1, col_imp2, col_imp3 = col_imp1, col_imp2, col_imp3 = st.columns(3)

with col_imp1:
    st.markdown("""
    #### 1. LLM API Billing & Cost Economics
    * **Tokens, Not Characters**: Commercial LLM APIs (OpenAI, Anthropic, Google) charge strictly per token, not per character or word.
    * **Input vs Output Multiplier**: Output generation is 3x to 5x more computationally expensive than prompt ingestion.
    * **Prompt Verbosity Tax**: Padded prompt templates, massive zero-shot few-shot examples, and verbose system instructions consume the same billing rate.
    * **Schema Waste**: Sending large raw JSON schemas can burn thousands of tokens before answering the user query.
    """)

with col_imp2:
    st.markdown("""
    #### 2. Context Window & Chunking Limits
    * **Word Count != Token Budget**: A 128,000 token context window does NOT hold 128,000 English words. It holds roughly 90,000 to 98,000 words.
    * **RAG Chunking Vulnerability**: Chunking text documents strictly on character count (e.g., `chunk_size = 1000`) creates unpredictable token spans when text shifts from English prose to code or tables.
    * **Silent Truncation**: Truncating inputs on tokenizers must always preserve special tokens like `[SEP]` and `[EOS]` to prevent model hallucinations.
    """)

with col_imp3:
    st.markdown("""
    #### 3. Numbers, Code, and Multilingual Bias
    * **Arithmetic Degradation**: LLMs struggle with math because numbers are often split haphazardly (e.g., `"2026"` may be `20` + `26`).
    * **Source Code Indentation**: Four spaces can be 1 token or 4 tokens depending on tokenizer vocabulary; indentation burns context rapidly.
    * **Multilingual Inflation**: A sentence that takes 10 tokens in English may take 40 tokens in Hindi, Japanese, or Arabic due to vocabulary training bias!
    """)

st.divider()

# ─── Section 10: Interactive Workbench: Granularity, Padding & Cost Simulator ──
st.header("10. Interactive Workbench: Granularity, Padding & Cost Simulator", divider="rainbow")

st.markdown("""
Experiment with different input texts and models to observe token fragmentation, attention masking, and estimated API inference costs.
""")

col_in, col_out = col_in, col_out = st.columns([1, 1.2])

with col_in:
    st.subheader("Configuration Panel")
    
    preset = st.selectbox(
        "Select Text Preset or Custom:",
        [
            "Standard English Prose",
            "Python Code Snippet",
            "Number & Arithmetic Sequence",
            "Multilingual Comparison",
            "Custom Text"
        ]
    )
    
    if preset == "Standard English Prose":
        default_val = "Transformers utilize self-attention mechanisms to process tokens simultaneously."
    elif preset == "Python Code Snippet":
        default_val = "def calculate_loss(predictions, targets):\n    return np.mean((predictions - targets) ** 2)"
    elif preset == "Number & Arithmetic Sequence":
        default_val = "Calculate: 149204 + 98312 = 247516."
    elif preset == "Multilingual Comparison":
        default_val = "Hello world! नमस्ते दुनिया! こんにちは世界！"
    else:
        default_val = "Unbelievable breakthroughs in artificial intelligence occur daily."

    user_text = st.text_area("Input Text:", value=default_val, height=100)
    
    target_model = st.selectbox(
        "Target Architecture:",
        ["GPT-4 (cl100k_base BPE)", "Llama 3 (128k BPE)", "BERT (WordPiece)", "Character-Level Tokenizer"]
    )
    
    batch_padding = st.slider("Target Batch Sequence Length:", min_value=16, max_value=64, value=32, step=4)
    price_per_m = st.number_input("API Price per 1 Million Tokens ($):", min_value=0.1, max_value=50.0, value=2.50, step=0.5)

with col_out:
    st.subheader("Tokenizer Diagnostics & Tensors")
    
    # Simulation calculations
    words = user_text.split()
    chars = len(user_text)
    
    if target_model == "Character-Level Tokenizer":
        tokens_sim = list(user_text)
        token_count = len(tokens_sim)
        subword_ratio = token_count / max(1, len(words))
    elif target_model == "BERT (WordPiece)":
        tokens_sim = []
        for w in words:
            if len(w) > 7:
                tokens_sim.extend([w[:4], "##" + w[4:]])
            else:
                tokens_sim.append(w)
        token_count = len(tokens_sim)
        subword_ratio = token_count / max(1, len(words))
    else:
        # BPE models
        tokens_sim = []
        for w in words:
            if len(w) > 8:
                tokens_sim.extend([w[:4], w[4:8], w[8:]])
            elif len(w) > 5:
                tokens_sim.extend([w[:3], w[3:]])
            else:
                tokens_sim.append(w)
        token_count = len(tokens_sim)
        subword_ratio = token_count / max(1, len(words))
        
    pad_needed = max(0, batch_padding - token_count)
    active_tokens = min(token_count, batch_padding)
    simulated_mask = [1] * active_tokens + [0] * pad_needed
    
    # Cost calculation for 100k runs
    cost_100k = (token_count * 100_000 / 1_000_000) * price_per_m
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Tokens", f"{token_count}")
    with m2:
        st.metric("Tokens / Word Ratio", f"{subword_ratio:.2f}")
    with m3:
        st.metric("Cost per 100k Calls", f"${cost_100k:.4f}")
        
    st.markdown("#### Token Stream Decomposition")
    # Display tokens visually
    display_tokens = [f"`{t}`" for t in tokens_sim[:20]]
    st.markdown(" ".join(display_tokens) + (" ..." if len(tokens_sim) > 20 else ""))
    
    st.markdown("#### Simulated Attention Mask (Padded to Length)")
    st.code(f"attention_mask = {simulated_mask[:32]}", language="python")

st.divider()

# ─── Section 11: Technical Interview Preparation & Production Traps ───────────
st.header("11. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why did modern LLMs universally converge on Subword Tokenization rather than character or word levels?"):
    st.markdown("""
    **Answer**:
    * **Solves the Vocabulary vs Sequence Length Trilemma**: Word tokenization requires millions of words, creating an uncontrollable embedding parameter footprint and catastrophic Out-Of-Vocabulary (OOV) failures on rare words.
    * **Avoids Attention Quadratic Explosion**: Character tokenization blows up sequence lengths by 5x-8x, incurring massive quadratic O(N^2) compute costs in self-attention layers while offering poor semantic density per token.
    * **Compositional Morphological Representation**: Subword tokenization (BPE, SentencePiece) provides the optimal compromise: frequent words receive dedicated single-token IDs, while rare or novel terms decompose into constituent morphemes or UTF-8 bytes with zero OOV failures.
    """)

with st.expander("Q2: What is the exact mechanical purpose of the Attention Mask when processing padded token batches?"):
    st.markdown("""
    **Answer**:
    * **Batch Normalization Requirement**: Tensors processed across GPUs must have identical dimensions; sequences shorter than the batch maximum are padded with `[PAD]` tokens.
    * **Attention Weight Contamination**: Without masking, standard softmax self-attention would compute affinity weights between valid sentence tokens and empty `[PAD]` placeholders.
    * **Softmax Invalidation via Large Negative Constants**: The attention mask passes `1` for valid tokens and `0` for padding. In the attention computation Softmax(Q K^T / sqrt(d_k) + M), positions where M=0 receive -infinity (or -10000.0), ensuring their post-softmax probability is strictly 0.0.
    """)

with st.expander("Q3: Why are Token IDs integers that do not encode semantic similarity, and how is semantic geometry created?"):
    st.markdown("""
    **Answer**:
    * **Arbitrary Index Nature**: Token IDs are simple array row offsets. Token ID 100 and Token ID 101 have no semantic proximity whatsoever; their numerical distance is mathematically meaningless to the model.
    * **Projection into Dense Latent Space**: Semantic meaning is introduced exclusively when the integer ID queries the model's **Embedding Layer** (matrix W_emb of shape [Vocab_Size, Hidden_Dimension]).
    * **Gradient-Trained Geometric Manifolds**: During pretraining, backpropagation continuously adjusts the dense vector coordinates in the embedding matrix so that semantically related concepts cluster together in high-dimensional vector space.
    """)

with st.expander("Q4: How does subword tokenization degrade arithmetic capabilities and numerical reasoning in LLMs?"):
    st.markdown("""
    **Answer**:
    * **Haphazard Number Splitting**: Standard BPE tokenizers group digits based on corpus frequency rather than positional number systems. For instance, `10425` might be tokenized as `104` + `25`, whereas `1042` might be a single token.
    * **Lack of Positional Alignment**: Because digit groupings vary unpredictably across multi-digit numbers, the transformer cannot learn clean column-wise addition or carry-over arithmetic across consistent token slots.
    * **Modern Mitigations**: Cutting-edge models (such as Llama 3 and GPT-4) enforce regex rules during tokenizer training to force individual single-digit tokenization for all numerical sequences.
    """)

with st.expander("Q5: What is SentencePiece and why is it superior for multilingual language models compared to whitespace-based BPE?"):
    st.markdown("""
    **Answer**:
    * **Zero Pre-tokenization Dependency**: Classical BPE tokenizers depend on language-specific rule engines or whitespace splitting before running BPE merges. This fails completely on languages without explicit whitespace boundaries (e.g., Chinese, Japanese, Thai).
    * **Raw UTF-8 Byte Stream Processing**: SentencePiece treats the entire input as an uninterrupted stream of Unicode characters, encoding whitespace explicitly as an underscored meta-symbol (`_`).
    * **Reversible & Language-Agnostic**: De-tokenization requires nothing more than joining tokens and replacing `_` with a space, ensuring flawless cross-lingual handling across high-resource and low-resource scripts alike.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Cross-Model Token ID Bleed**  
    Feeding token IDs produced by one model's tokenizer directly into a different model's architecture. Because vocabulary tables differ completely across models, this corrupts generation into unintelligible gibberish.
    """)
    st.warning("""
    **Trap 2: Multilingual Tokenization Inflation**  
    Assuming 1,000 words always equals ~1,300 tokens. Non-Latin scripts (Hindi, Arabic, Japanese) often consume 3x to 6x more tokens per word due to English-biased vocabularies, causing unexpected latency and billing shocks.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unmasked Padding Leakage**  
    Failing to pass the `attention_mask` tensor during batched inference. Without attention masking, `[PAD]` tokens participate in self-attention calculations, degrading downstream classification and generation accuracy.
    """)
    st.warning("""
    **Trap 4: Naive Word-Count Chunking in RAG**  
    Chunking knowledge base documents by raw character or word counts rather than tokenizer token counts. This frequently overflows the model's hard context window during inference, causing silent truncation of vital context.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Subword Morphemes?"):
        st.markdown("""
        **Subword Morphemes**:  
        Sub-linguistic units (roots, prefixes, suffixes) that allow tokenizers to represent rare or unseen words compositionally with zero Out-Of-Vocabulary failures.
        """)
    with st.expander("🗂️ Flashcard: Out-of-Vocabulary (OOV)?"):
        st.markdown("""
        **Out-of-Vocabulary (OOV)**:  
        A word or symbol missing from a model's static vocabulary dictionary, forcing replacement with an uninformative `[UNK]` token in word-level architectures.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Attention Mask?"):
        st.markdown("""
        **Attention Mask**:  
        A binary indicator tensor (1s for valid tokens, 0s for padding) that forces self-attention layers to assign zero probability to meaningless `[PAD]` positions.
        """)
    with st.expander("🗂️ Flashcard: Byte Fallback?"):
        st.markdown("""
        **Byte Fallback**:  
        A safety mechanism in modern BPE and SentencePiece tokenizers that decomposes unknown Unicode characters into raw UTF-8 bytes, ensuring 100% text coverage.
        """)

st.divider()

# ─── Section 12: Self-Assessment Checklist ────────────────────────────────────
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural trade-offs between character, word, subword, and sentence tokenization.", value=True)
chk2 = st.checkbox("I can explain why token IDs are arbitrary integer indices and how embedding layers map them to continuous dense vector space.", value=True)
chk3 = st.checkbox("I understand the mechanics of special tokens (BOS, EOS, PAD, CLS, SEP) and how attention masks neutralize padding.", value=True)
chk4 = st.checkbox("I know the architectural differences and training mechanics of BPE, WordPiece, and SentencePiece tokenizers.", value=True)
chk5 = st.checkbox("I can analyze the production impacts of tokenization on API pricing, context limits, numbers, and multilingual inflation.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 3.2: Subword Tokenization!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Tokenization Basics. Ready for Chapter 3.2: Subword Tokenization!")
