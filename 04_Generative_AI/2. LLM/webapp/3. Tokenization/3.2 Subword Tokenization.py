import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="3.2 Subword Tokenization - LLM Curriculum",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# Title Block and Executive Overview
# ==============================================================================
st.title("3.2 Subword Tokenization: BPE, WordPiece, Unigram, and Byte-Level BBPE")
st.caption("Chapter 3: Tokenization | Domain: 2. Large Language Models")

st.markdown("""
Subword tokenization is the computational foundation of modern Large Language Models. 
Neither pure character-level tokenization (which produces excessively long sequences and explodes quadratic attention costs) 
nor pure word-level tokenization (which creates unbounded, sparse vocabularies and catastrophic out-of-vocabulary failures) 
is viable for foundation models.

Subword algorithms solve this fundamental tension by discovering a data-driven statistical vocabulary: 
frequently occurring words remain whole single tokens, while rare words, typos, and morphologically complex words 
are decomposed into meaningful constituent subunits.

In this lesson, you will master:
1. The 7 core subword tokenization paradigms and how they construct vocabularies from raw training corpora.
2. The exact algorithmic mechanics of Byte Pair Encoding (BPE) merges, WordPiece mutual information scoring, and Unigram EM pruning.
3. Why SentencePiece and Byte-Level BPE (BBPE) eliminated the `<UNK>` token and revolutionized multilingual processing.
4. The architectural trade-offs of vocabulary sizing: how expanding from 32k to 128k tokens balances embedding memory against inference speed.
5. A runnable pure Python implementation of a Byte-Pair Encoding engine with merge rank tracking and compression evaluation.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Subword Tokenization Spectrum
# ==============================================================================
st.header("1. Architectural Blueprint: The Subword Tokenization Spectrum", divider="rainbow")

st.markdown("""
Observe how natural language strings transition through pre-tokenization regex splitting, statistical subword decomposition, 
and byte-fallback encoding into discrete token IDs:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    spectrum_dot = """
    digraph SubwordSpectrum {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_input {
            label="Raw Text Ingress";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            Raw [label="Raw Text String\\n'unbelievable 🚀 $450'", fillcolor="#1E293B"];
            PreSplit [label="Regex Pre-Splitting\\n['unbelievable', ' ', '🚀', ' ', '$450']", fillcolor="#1E293B"];
            Raw -> PreSplit;
        }

        subgraph cluster_algorithms {
            label="Statistical Subword Engines";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            
            Algo_BPE [label="BPE (Frequency Merges)\\n['un', 'believ', 'able']", fillcolor="#0C4A6E", color="#38BDF8"];
            Algo_WP [label="WordPiece (Likelihood Score)\\n['un', '##believ', '##able']", fillcolor="#1E1B4B", color="#818CF8"];
            Algo_UG [label="Unigram (EM Pruning)\\nProbabilistic Viterbi Path", fillcolor="#312E81", color="#A78BFA"];
            Algo_BBPE [label="Byte-Level BBPE (UTF-8 Fallback)\\nGuaranteed 0 UNK Tokens", fillcolor="#064E3B", color="#34D399"];
            
            PreSplit -> Algo_BPE;
            PreSplit -> Algo_WP;
            PreSplit -> Algo_UG;
            PreSplit -> Algo_BBPE;
        }

        subgraph cluster_output {
            label="Discrete Model Tensors";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            
            VocabTable [label="Static Vocabulary Table\\nSize: 32,000 to 128,256 IDs", fillcolor="#065F46", color="#34D399"];
            TokenIDs [label="Integer Token ID Tensor\\n[2841, 19402, 1204, 95821]", fillcolor="#047857", color="#34D399"];
            
            Algo_BBPE -> VocabTable -> TokenIDs;
        }
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(spectrum_dot, use_container_width=True)

with col_diag2:
    st.markdown("#### The Subword Engineering Objective")
    st.markdown("""
    * **Optimal Information Density**: Maximize the number of UTF-8 characters represented per token while keeping vocabulary size computationally manageable.
    * **Zero Out-of-Vocabulary (OOV)**: Ensure every possible input string—including code syntax, foreign scripts, math symbols, and emojis—can be encoded into valid token IDs without failing.
    * **Morphological Coherence**: Align subword boundaries with linguistic affixes (e.g., `pre-`, `un-`, `-tion`, `-ing`) to facilitate syntactic and semantic generalization in deep attention layers.
    * **Detokenization Invertibility**: Guarantee that detokenizing token IDs reproduces the original raw string without losing whitespace or casing.
    """)

# Data Structure Comparison Matrix
st.markdown("#### Data Structure Architecture: Comprehensive Subword Algorithm Matrix")

subword_table_dot = """
digraph SubwordTable {
    bgcolor="#0B1120";
    rankdir=TB;

    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=plaintext];

    TableNode [label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#0F172A" colspan="5"><font color="#38BDF8"><b>Subword Tokenization Algorithms: Architectural Comparison</b></font></td>
            </tr>
            <tr bgcolor="#334155">
                <td><b>Algorithm</b></td>
                <td><b>Vocabulary Construction</b></td>
                <td><b>Pair Selection Criterion</b></td>
                <td><b>Continuation Marker</b></td>
                <td><b>Frontier Models</b></td>
            </tr>
            <tr>
                <td bgcolor="#0C4A6E"><b>Byte Pair Encoding (BPE)</b></td>
                <td>Bottom-up greedy pair merges</td>
                <td>Max bigram frequency in training corpus</td>
                <td>Trailing whitespace or end-of-word marker</td>
                <td>GPT-2, GPT-3, RoBERTa</td>
            </tr>
            <tr>
                <td bgcolor="#1E1B4B"><b>WordPiece</b></td>
                <td>Bottom-up probabilistic merging</td>
                <td>Max mutual information: Count(AB) / (Count(A)*Count(B))</td>
                <td>Prefix '##' (e.g. '##ing')</td>
                <td>BERT, DistilBERT, Electra</td>
            </tr>
            <tr>
                <td bgcolor="#78350F"><b>Unigram Language Model</b></td>
                <td>Top-down loss-driven pruning</td>
                <td>Minimizes decrease in corpus log-likelihood</td>
                <td>Whitespace meta-symbol '_' (U+2581)</td>
                <td>T5, ALBERT, MarianMT</td>
            </tr>
            <tr>
                <td bgcolor="#064E3B"><b>SentencePiece</b></td>
                <td>Language-agnostic stream processing</td>
                <td>Wraps BPE or Unigram on raw bytes</td>
                <td>Reversible whitespace meta-symbol '_'</td>
                <td>Llama 1, Llama 2, Mistral 7B</td>
            </tr>
            <tr>
                <td bgcolor="#047857"><b>Byte-Level BPE (BBPE)</b></td>
                <td>Bottom-up merge over 256 UTF-8 bytes</td>
                <td>Greedy byte-bigram frequency with regex rules</td>
                <td>Direct byte mapping (Ġ for space)</td>
                <td>GPT-4 (cl100k/o200k), Llama 3 (128k)</td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(subword_table_dot, use_container_width=True)

st.divider()

# ==============================================================================
# Section 2: 1. BPE (Byte Pair Encoding)
# ==============================================================================
st.header("2. BPE: The Bottom-Up Greedy Statistical Merge Paradigm", divider="rainbow")

st.markdown("""
Originally developed as a general-purpose data compression algorithm by Philip Gage in 1994, 
**Byte Pair Encoding (BPE)** was adapted for natural language processing by Sennrich, Haddow, and Birch (2016). 
It represents the algorithmic bedrock upon which modern autoregressive foundation models are built.
""")

col_bpe1, col_bpe2 = st.columns(2)

with col_bpe1:
    st.markdown("#### The Core Philosophy of BPE")
    st.markdown("""
    BPE operates via an iterative, bottom-up greedy clustering process:
    
    1. **Initialization**: The base vocabulary is initialized containing only individual characters (or bytes) observed in the training corpus, plus a special end-of-word boundary marker (such as `</w>`).
    2. **Corpus Segmentation**: Every word in the training corpus is split into individual characters.
    3. **Bigram Frequency Counting**: The algorithm scans the entire segmented corpus and counts the frequency of all adjacent symbol pairs (bigrams).
    4. **Greedy Merge**: The single most frequent bigram `(c_1, c_2)` is identified and merged into a new unified symbol `c_{12}`.
    5. **Vocabulary Augmentation**: The new symbol `c_{12}` is added to the vocabulary, and the merge rule `c_1 + c_2 -> c_{12}` is recorded in an ordered merge table.
    6. **Iteration**: Steps 3 through 5 repeat iteratively until the vocabulary reaches a predetermined target size (e.g., 32,000 or 50,000 tokens).
    """)

with col_bpe2:
    st.markdown("#### Why BPE Transformed Language Modeling")
    st.markdown("""
    * **Solves the Out-of-Vocabulary Crisis**: Any unknown word can always be decomposed into its individual base characters, ensuring the model never completely fails to ingest a sequence.
    * **Dynamic Compression**: Common words like `"the"`, `"programming"`, and `"neural"` are merged into single tokens, minimizing the number of sequential tokens required to express documents.
    * **Subword Sharing**: Rare morphological variations (e.g. `"unprecedentedly"`) share base tokens with common roots (`"un"`, `"precedent"`, `"ed"`, `"ly"`), allowing embedding representations to generalize across related words.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
State-of-the-art foundation models such as Llama 3 (Meta) and GPT-4 (OpenAI) rely entirely on BPE variants. 
However, modern implementations abandoned character-level BPE in favor of **Byte-Level BPE** with massive vocabularies 
(128,256 tokens in Llama 3; 200,000 tokens in GPT-4o), pairing BPE with specialized pre-tokenization regexes to prevent 
merges across numbers, whitespace, and punctuation.
""")

bpe_comp_df = pd.DataFrame({
    "Criterion": [
        "Base Ingestion Unit",
        "Vocabulary Size",
        "Merge Granularity",
        "Punctuation Merges",
        "Handling of Unknown Words"
    ],
    "Classical BPE (Sennrich 2016)": [
        "Unicode characters after whitespace splitting",
        "Small (16,000 to 32,000 tokens)",
        "Aggressive word-internal merges",
        "Separated via external rule-based pre-tokenizers",
        "Decomposes to character-level tokens"
    ],
    "Modern Frontier BPE (Llama 3 / GPT-4)": [
        "Raw UTF-8 bytes (0-255)",
        "Massive (128k to 200k tokens)",
        "Regex-constrained subwords and byte clusters",
        "Strictly isolated via regex rules (digits never merge with letters)",
        "100% universal byte-fallback (zero UNK tokens)"
    ]
})
st.dataframe(bpe_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Byte Pair Encoding Training Mechanics
# ==============================================================================
st.header("3. Byte Pair Encoding: Step-by-Step Training Mechanics and Inference Lookup", divider="rainbow")

st.markdown("""
To understand how BPE creates a vocabulary, examine a concrete step-by-step trace of the training algorithm 
over a sample corpus of words and their frequencies.
""")

col_mech1, col_mech2 = st.columns(2)

with col_mech1:
    st.markdown("#### Step-by-Step Training Trace")
    st.markdown("""
    Consider a toy training corpus with word frequencies:
    * `"low": 5`, `"lower": 2`, `"newest": 6`, `"widest": 3`
    
    **Initial Segmentation (Base Vocab: {d, e, i, l, n, o, r, s, t, w, </w>}):**
    * `l o w </w>` (count 5)
    * `l o w e r </w>` (count 2)
    * `n e w e s t </w>` (count 6)
    * `w i d e s t </w>` (count 3)
    
    **Iteration 1: Count Symbol Bigrams**
    * Pair `('s', 't')` appears in `"newest"` (6) and `"widest"` (3) = **9 occurrences**.
    * Pair `('e', 's')` appears in `"newest"` (6) and `"widest"` (3) = **9 occurrences**.
    * *Tie-break action*: Merge `('e', 's') -> 'es'`.
    * Corpus updated: `n e w es t </w>`, `w i d es t </w>`.
    
    **Iteration 2: Merge ('es', 't') -> 'est' (Frequency = 9)**
    * Corpus updated: `n e w est </w>`, `w i d est </w>`.
    
    **Iteration 3: Merge ('est', '</w>') -> 'est</w>' (Frequency = 9)**
    * The suffix `"est</w>"` is now a permanent vocabulary token!
    
    **Iteration 4: Merge ('l', 'o') -> 'lo' (Frequency = 7)**
    * Pair `('l', 'o')` appears in `"low"` (5) + `"lower"` (2) = 7.
    * Corpus updated: `lo w </w>`, `lo w e r </w>`.
    """)

with col_mech2:
    st.markdown("#### Inference Tokenization: Ordered Merge Lookup")
    st.markdown("""
    During inference, tokenizing a new, unseen word (e.g. `"lowest"`) does **NOT** count corpus frequencies. 
    Instead, it applies the recorded merge rules **in the exact historical rank order** in which they were learned:
    
    1. Input string is split into individual characters: `['l', 'o', 'w', 'e', 's', 't', '</w>']`.
    2. Check all adjacent pairs against the learned **Merge Table**:
       * Rank 1 rule: `('e', 's') -> 'es'` => `['l', 'o', 'w', 'es', 't', '</w>']`
       * Rank 2 rule: `('es', 't') -> 'est'` => `['l', 'o', 'w', 'est', '</w>']`
       * Rank 3 rule: `('est', '</w>') -> 'est</w>'` => `['l', 'o', 'w', 'est</w>']`
       * Rank 4 rule: `('l', 'o') -> 'lo'` => `['lo', 'w', 'est</w>']`
       * Rank 5 rule: `('lo', 'w') -> 'low'` => `['low', 'est</w>']`
    3. Final output tokens: `['low', 'est</w>']`.
    4. Map tokens to integer IDs using the vocabulary index table.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern high-throughput inference engines do not iterate through Python lists of merge rules. 
Libraries like OpenAI's `tiktoken` (written in Rust) compile the BPE merge table into a highly optimized 
**Trie (Prefix Tree) and Hash Map**. This allows tokenization to run in O(N) time directly over raw memory buffers, 
processing tens of megabytes of text per second per CPU core.
""")

trace_df = pd.DataFrame({
    "Merge Rank": [1, 2, 3, 4, 5],
    "Bigram Merged": ["('e', 's')", "('es', 't')", "('est', '</w>')", "('l', 'o')", "('lo', 'w')"],
    "New Vocabulary Symbol": ["es", "est", "est</w>", "lo", "low"],
    "Corpus Frequency": [9, 9, 9, 7, 7],
    "Linguistic Role": ["Stem/Affix connector", "Superlative suffix base", "Terminal superlative suffix", "Root prefix", "Complete common root"]
})
st.dataframe(trace_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. WordPiece
# ==============================================================================
st.header("4. WordPiece: Likelihood-Driven Segmentation and the BERT Subword Paradigm", divider="rainbow")

st.markdown("""
Developed by Schuster and Nakajima (2012) for Japanese and Korean speech recognition and adopted by Google for 
the **BERT** architecture (Devlin et al., 2018), **WordPiece** is a bottom-up subword algorithm that differs 
fundamentally from BPE in its merge selection criterion.
""")

col_wp1, col_wp2 = col_wp1, col_wp2 = st.columns(2)

with col_wp1:
    st.markdown("#### The Mathematical Formulation of WordPiece")
    st.markdown("""
    While BPE greedily merges the symbol pair with the highest raw occurrence count, WordPiece selects the pair that 
    **maximizes the log-likelihood of the language model** on the training data:
    
    `Score(A, B) = Count(AB) / (Count(A) * Count(B))`
    
    * **Mutual Information Intuition**:
      * The numerator `Count(AB)` measures how often symbols A and B appear together.
      * The denominator `Count(A) * Count(B)` penalizes pairs composed of individually frequent symbols.
      * If letter `'e'` and letter `'s'` appear trillions of times independently throughout English, their high raw co-occurrence might just be random statistical noise.
      * WordPiece only merges A and B if they appear together **far more frequently than would be expected by random chance**.
    """)

with col_wp2:
    st.markdown("#### The Continuation Subword Prefix ('##')")
    st.markdown("""
    WordPiece distinguishes between the beginning of a word and continuation subwords using the `##` prefix:
    * `"playing"` is tokenized as `["play", "##ing"]`.
    * `"unbelievable"` is tokenized as `["un", "##beli", "##eva", "##ble"]`.
    
    **Inference Algorithm (MaxMatch / Greedy Longest-Match-First):**
    Unlike BPE, WordPiece tokenizes each word during inference by scanning from the beginning and greedily finding 
    the **longest prefix that exists in the vocabulary**. If a match is found, it moves to the remaining substring 
    and searches for the longest continuation prefix starting with `##`. If any character cannot be matched, 
    the entire word is replaced with `[UNK]`.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Why did modern generative LLMs abandon WordPiece in favor of Byte-Level BPE?
1. **The `[UNK]` Vulnerability**: WordPiece operates on Unicode characters after whitespace splitting. Any unseen character or corrupt Unicode sequence triggers `[UNK]`, which breaks autoregressive generative models.
2. **The `##` Formatting Overhead**: The explicit `##` prefix complicates raw byte generation and code tokenization.
3. **Lossless Detokenization**: WordPiece loses original whitespace variations, making exact round-trip string reconstruction difficult.
""")

wp_comp_df = pd.DataFrame({
    "Dimension": [
        "Selection Criterion",
        "Scoring Formula",
        "Inference Mechanism",
        "Subword Identifier",
        "Failure Mode on Unknown Glyph",
        "Primary Model Adoption"
    ],
    "Byte Pair Encoding (BPE)": [
        "Raw frequency counting",
        "Score = Count(A, B)",
        "Ordered merge table application",
        "End-of-word marker or trailing space",
        "Falls back to single characters/bytes",
        "GPT-2, GPT-4, Llama 3"
    ],
    "WordPiece (Google)": [
        "Unigram language model likelihood",
        "Score = Count(AB) / (Count(A) * Count(B))",
        "Greedy Longest-Match-First (MaxMatch)",
        "Continuation prefix '##'",
        "Emits catastrophic [UNK] token",
        "BERT, DistilBERT, MobileBERT"
    ]
})
st.dataframe(wp_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Unigram Tokenization
# ==============================================================================
st.header("5. Unigram Tokenization: Probabilistic Pruning and Subword Regularization", divider="rainbow")

st.markdown("""
Introduced by Taku Kudo (2018), the **Unigram Language Model Tokenizer** flips the entire subword training paradigm upside down. 
Instead of starting with individual characters and iteratively **merging** (bottom-up), Unigram starts with a massive 
candidate vocabulary and iteratively **prunes** symbols (top-down).
""")

col_ug1, col_ug2 = st.columns(2)

with col_ug1:
    st.markdown("#### The Top-Down Pruning Mechanism")
    st.markdown("""
    1. **Initialization**: Initialize a massive seed vocabulary V containing hundreds of thousands of candidate substrings (all characters plus frequent n-grams from the corpus).
    2. **Expectation-Maximization (EM)**: Optimize unigram token probabilities P(x) to maximize the likelihood of the training corpus:
       `L = Sum_{w in Corpus} log (Sum_{x_1...x_k in Seg(w)} Product_i P(x_i))`
    3. **Compute Pruning Loss**: For every token x in V, calculate how much the total corpus log-likelihood would decrease if token x were deleted from the vocabulary.
    4. **Iterative Truncation**: Discard the bottom 10% to 20% of tokens that contribute the least to corpus likelihood (while strictly protecting single characters to prevent OOV).
    5. **Convergence**: Repeat steps 2 through 4 until the vocabulary reaches the desired target size (e.g. 32,000 tokens).
    """)

with col_ug2:
    st.markdown("#### Viterbi Search and Subword Regularization")
    st.markdown("""
    * **Inference Decoding via Viterbi**:
      Because multiple valid segmentations exist for any word, Unigram uses dynamic programming (the Viterbi algorithm) 
      to find the single sequence of tokens that maximizes the joint probability:
      `Optimal_Segmentation = argmax_{x_1...x_k} Product_i P(x_i)`
    
    * **Subword Regularization (Data Augmentation)**:
      Unlike BPE (which is strictly deterministic), Unigram can sample different segmentations for the same text 
      during pretraining according to their multinomial probability:
      * Standard: `["international", "ization"]`
      * Sampled variant: `["inter", "national", "ization"]`
      This acts as powerful statistical regularization, making the downstream language model remarkably robust to typos, noise, and dialectal variations.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Google's foundation architectures—including **T5 (Text-to-Text Transfer Transformer)**, **SentencePiece Unigram**, 
and multilingual models—use Unigram tokenization. The ability to perform subword regularization during pretraining 
is one of the key reasons T5 exhibits exceptional cross-lingual transfer across non-Latin scripts.
""")

ug_comp_df = pd.DataFrame({
    "Property": [
        "Directionality",
        "Mathematical Objective",
        "Deterministic vs Probabilistic",
        "Inference Algorithm",
        "Data Augmentation Capability"
    ],
    "Byte Pair Encoding (BPE)": [
        "Bottom-up: Merges characters into words",
        "Heuristic frequency bigram maximization",
        "Strictly deterministic (1 fixed segmentation)",
        "Ordered merge lookup",
        "None (requires BPE-Dropout extension)"
    ],
    "Unigram Language Model": [
        "Top-down: Prunes huge candidate vocabulary",
        "Maximum Likelihood Estimation via EM algorithm",
        "Inherently probabilistic with token priors P(x)",
        "Dynamic programming Viterbi search",
        "Native Subword Regularization sampling"
    ]
})
st.dataframe(ug_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. SentencePiece
# ==============================================================================
st.header("6. SentencePiece: Pre-tokenization-Free Curation and Raw Stream Modeling", divider="rainbow")

st.markdown("""
Prior to **SentencePiece** (Kudo & Richardson, 2018), every NLP tokenizer relied on language-specific rule-based 
pre-tokenization (e.g. splitting text on whitespace using Python regex or Moses). This created a fatal linguistic flaw: 
languages such as Chinese, Japanese, and Thai do not use spaces between words, requiring complex morphological analyzers 
(Jieba, MeCab) that introduced cascading errors.
""")

col_sp1, col_sp2 = st.columns(2)

with col_sp1:
    st.markdown("#### Treating Text as a Raw Character Stream")
    st.markdown("""
    SentencePiece revolutionizes pre-tokenization by eliminating language-specific heuristics:
    
    1. **Whitespace as a Visible Token**: SentencePiece treats whitespace like any other character. It replaces spaces with the visible meta-symbol **`_` (U+2581 Lower One Eighth Block)**:
       * Raw input: `"Hello world"`
       * Normalized stream: `" Hello_world"`
    2. **Lossless Detokenization**:
       Because all whitespace information is preserved directly inside the token stream, detokenization is completely trivial:
       `Detokenized_String = "".join(tokens).replace("_", " ").strip()`
       There is zero ambiguity about where spaces were originally located, eliminating complex regex detokenization rules.
    3. **Language Agnostic**: Works identically on English, Arabic, Chinese, Python code, and Japanese with zero language-specific configuration.
    """)

with col_sp2:
    st.markdown("#### Dual Engine Architecture")
    st.markdown("""
    SentencePiece is not an algorithm itself; it is a **universal framework** that wraps two distinct underlying subword algorithms:
    * **SentencePiece BPE**: Uses BPE merges directly over the character stream containing the `_` meta-symbol.
    * **SentencePiece Unigram**: Uses Unigram pruning over the character stream.
    
    **Direct File Pipeline**:
    SentencePiece trains directly from raw text files without requiring pre-split sentences or corpora, outputting a 
    single self-contained model file (`tokenizer.model`) containing both the vocabulary and the merge/pruning rules.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
SentencePiece is the standard tokenizer backend for **Llama 1, Llama 2, Mistral 7B, and Google Gemma**. 
By encoding spaces as `_`, these foundation models maintain flawless formatting fidelity across indentation-sensitive 
code (Python, YAML) and whitespace-free Asian scripts.
""")

sp_comp_df = pd.DataFrame({
    "Feature": [
        "Language Assumptions",
        "Handling of Whitespace",
        "Detokenization Process",
        "Asian Script Compatibility",
        "Production Format"
    ],
    "Traditional Tokenizers (Moses / SpaCy)": [
        "Assumes whitespace separates words",
        "Discarded or collapsed during pre-processing",
        "Lossy; requires heuristic spacing restoration",
        "Requires external dictionary segmenters",
        "Fragmented code + vocab files"
    ],
    "SentencePiece Framework": [
        "Zero language assumptions; pure raw stream",
        "Preserved as visible meta-symbol U+2581 (_)",
        "100% Lossless; trivial string join and replace",
        "Native out-of-the-box support for all scripts",
        "Single compiled binary model artifact (.model)"
    ]
})
st.dataframe(sp_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Byte-level Tokenization
# ==============================================================================
st.header("7. Byte-Level Tokenization (BBPE): Eliminating the UNK Token and Universal Encoding", divider="rainbow")

st.markdown("""
Standard character-level BPE had one major vulnerability: if an input text contained an exceedingly rare Unicode glyph 
(e.g., an obscure hieroglyphic, a rare emoji, or corrupted data) that was never seen during pretraining, the tokenizer 
had no choice but to map it to the catastrophic `<UNK>` token.

**Byte-Level BPE (BBPE)**—introduced by Radford et al. (2019) for GPT-2—completely solved this by changing the fundamental 
atoms of tokenization from Unicode characters to **raw UTF-8 bytes**.
""")

col_bbpe1, col_bbpe2 = st.columns(2)

with col_bbpe1:
    st.markdown("#### The 256-Byte Atomic Universe")
    st.markdown("""
    Every character in every human writing system, every programming language, and every emoji is represented digitally 
    as a sequence of 1 to 4 bytes in the UTF-8 encoding standard:
    * ASCII letters (`'A'`) = 1 byte (`0x41`)
    * Accented letters (`'é'`) = 2 bytes (`0xC3 0xA9`)
    * Chinese characters (`'中'`) = 3 bytes (`0xE4 0xB8 0xAD`)
    * Emojis (`'🚀'`) = 4 bytes (`0xF0 0x9F 0x9A 0x80`)
    
    **The Byte-Level Guarantee:**
    By initializing the base vocabulary with all **256 fundamental byte values (0x00 to 0xFF)**, any conceivable string 
    can be expressed as a combination of base bytes:
    `Vocab_Base = {0x00, 0x01, ..., 0xFF}   # Exactly 256 base tokens`
    **It is mathematically impossible for an input string to be Out-of-Vocabulary!** The `<UNK>` token is completely eliminated.
    """)

with col_bbpe2:
    st.markdown("#### Preventing Degenerate Merges: Regex Pre-Splitting")
    st.markdown("""
    If BPE were run naively across raw byte streams without constraints, it would merge across unrelated word boundaries:
    e.g., merging `"dog."` with `" The"` to form `"dog. The"`.
    
    To prevent this, modern Byte-Level BPE employs a **regex splitting rule** that partitions the text before byte merges:
    ```python
    # GPT-4 / Llama 3 Style Regex Partitioning
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+|[\\s]+"
    ```
    * **Letters (`\\p{L}+`)** are kept together.
    * **Digits (`\\p{N}{1,3}`)** are partitioned into small groups, preventing numbers from fusing with words and enabling arithmetic reasoning.
    * **Punctuation** is isolated from alphanumeric tokens.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
All premier foundation models—**GPT-4, Claude 3.5, Llama 3, Mistral, and DeepSeek**—use Byte-Level BPE. 
In Llama 3 and GPT-4, the tokenizer maps raw bytes to printable Unicode representations (e.g. using `Ġ` to represent space) 
to ensure human readability during logging and debugging.
""")

bbpe_comp_df = pd.DataFrame({
    "Dimension": [
        "Atomic Base Units",
        "Base Vocab Size",
        "Out-of-Vocabulary Rate",
        "Handling of Unknown Emojis",
        "Handling of Binary / Raw Bytes",
        "Modern Frontier Standard"
    ],
    "Unicode BPE (Sennrich 2016)": [
        "Unicode Code Points (Characters)",
        "Thousands of initial characters",
        "Non-zero (emits <unk> on unseen characters)",
        "Replaces with <unk> or question marks",
        "Fails completely",
        "Legacy (BART / MarianMT)"
    ],
    "Byte-Level BPE (GPT-4 / Llama 3)": [
        "Raw UTF-8 Bytes (0-255)",
        "Exactly 256 atomic bytes",
        "Strictly 0.00% (Mathematically impossible)",
        "Decomposes into 4 valid constituent byte tokens",
        "Natively ingests arbitrary binary streams",
        "Universal standard for all modern LLMs"
    ]
})
st.dataframe(bbpe_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Token Vocabulary Construction
# ==============================================================================
st.header("8. Token Vocabulary Construction: Size Dynamics, Compression Ratio, and Memory", divider="rainbow")

st.markdown("""
Designing the vocabulary size (V) of a foundation model is one of the most critical architectural decisions 
in deep learning engineering. It governs the trade-off between **context window compression** and **GPU parameter memory**.
""")

col_vc1, col_vc2 = st.columns(2)

with col_vc1:
    st.markdown("#### The Fundamental Vocabulary Size Dilemma")
    st.markdown("""
    * **The Case for Small Vocabularies (V = 32,000$, e.g. Llama 1/2)**:
      * The model's embedding matrix and final linear head have shape:
        `W_emb in R^{[V, Hidden_Dimension]}`
      * With $V = 32,000$ and D = 4,096, the embedding parameters consume only 131 million parameters (approx 260 MB in FP16).
      * However, small vocabularies suffer from **low compression**: documents take more tokens to represent, exploding quadratic self-attention compute (O(N^2)) and slowing down inference.
    
    * **The Case for Large Vocabularies (V = 128,000$ to $256,000$, e.g. Llama 3, Gemma 2)**:
      * Larger vocabularies compress text significantly better (fewer tokens per document).
      * A 128k vocabulary generates answers in **20% to 30% fewer autoregressive decoding steps**, slashing serving latency!
      * Downside: The embedding weights in Gemma 2 ($V = 256,000, D = 4,096$) consume over **1 billion parameters** just for token lookup!
    """)

with col_vc2:
    st.markdown("#### The Multilingual 'Token Tax'")
    st.markdown("""
    In smaller English-dominated vocabularies (like GPT-2's 50k vocab), English words average **~4 characters per token**. 
    However, languages with non-Latin scripts (Hindi, Arabic, Japanese, Telugu) were severely penalized:
    * *English sentence*: `"Artificial intelligence is transforming medicine."` -> **6 tokens**.
    * *Same sentence in Hindi*: `"कृत्रिम बुद्धिमत्ता चिकित्सा को बदल रही है।"` -> **28 tokens!**
    
    This **Token Tax** meant non-English users consumed context windows 4x faster and paid 4x more API costs for the exact same semantic content. 
    Modern models solve this by training on massively multilingual corpora and allocating tens of thousands of dedicated merge slots to global scripts.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Meta quadrupled the vocabulary in **Llama 3 from 32,000 to 128,256 tokens**, yielding an average **15% improvement in compression** 
across English text and up to **35% compression improvement** across programming code and multilingual documents.
""")

vocab_comp_df = pd.DataFrame({
    "Model Family": ["GPT-2", "Llama 1 / 2", "Mistral 7B", "GPT-4 (cl100k)", "Llama 3", "Gemma 2"],
    "Vocabulary Size": ["50,257", "32,000", "32,768", "100,277", "128,256", "256,000"],
    "Algorithm Backend": ["Byte-Level BPE", "SentencePiece BPE", "Byte-Level BPE", "Byte-Level BPE", "Byte-Level BPE", "SentencePiece BPE"],
    "Embedding Parameters (FP16)": ["~80 MB", "~260 MB", "~268 MB", "~820 MB", "~1.05 GB", "~2.10 GB"],
    "English Compression (chars/token)": ["~3.8", "~3.7", "~3.8", "~4.4", "~4.5", "~4.6"],
    "Multilingual Efficiency": ["Poor", "Moderate", "Moderate", "High", "Exceptional", "Superhuman"]
})
st.dataframe(vocab_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: Production Implementation: Pure Python BPE Engine
# ==============================================================================
st.header("9. Production Implementation: Pure Python Byte-Pair Encoding Engine", divider="rainbow")

st.markdown("""
The self-contained Python implementation below demonstrates the complete BPE lifecycle: 
frequency counting over character streams, greedy merge extraction, merge rank tracking, 
and inference tokenization with exact detokenization verification:
""")

st.code('''from typing import List, Dict, Tuple, Set

class ProductionBPETokenizer:
    """
    A pure Python implementation of the Byte-Pair Encoding (BPE) algorithm.
    Demonstrates corpus training, ordered merge rank creation, subword tokenization,
    and lossless detokenization.
    """

    def __init__(self):
        self.vocab: Dict[int, str] = {}
        self.inverse_vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], int] = {}  # (symbol_a, symbol_b) -> merge_rank
        self.end_of_word = "</w>"

    def _get_corpus_stats(self, corpus_words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of all adjacent symbol pairs across the segmented corpus."""
        pairs: Dict[Tuple[str, str], int] = {}
        for word, freq in corpus_words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_corpus(self, pair: Tuple[str, str], corpus_words: Dict[str, int]) -> Dict[str, int]:
        """Applies a newly learned merge rule across all words in the corpus."""
        new_corpus: Dict[str, int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in corpus_words.items():
            new_word = word.replace(bigram, replacement)
            new_corpus[new_word] = freq
        return new_corpus

    def train(self, texts: List[str], num_merges: int = 10) -> None:
        """
        Trains BPE merge rules from an input list of training texts.
        """
        # 1. Initialize words segmented into characters with end-of-word marker
        word_counts: Dict[str, int] = {}
        for text in texts:
            for raw_word in text.strip().split():
                segmented = " ".join(list(raw_word)) + " " + self.end_of_word
                word_counts[segmented] = word_counts.get(segmented, 0) + 1

        # 2. Base vocabulary contains all unique characters
        unique_chars: Set[str] = set()
        for word in word_counts.keys():
            unique_chars.update(word.split())
            
        current_id = 0
        for char in sorted(unique_chars):
            self.vocab[current_id] = char
            self.inverse_vocab[char] = current_id
            current_id += 1

        # 3. Iterative greedy merge loop
        corpus = word_counts
        for rank in range(num_merges):
            pairs = self._get_corpus_stats(corpus)
            if not pairs:
                break
            # Find the most frequent adjacent bigram
            best_pair = max(pairs, key=pairs.get)
            
            # Record merge rule with its historical rank
            self.merges[best_pair] = rank
            new_token = "".join(best_pair)
            self.vocab[current_id] = new_token
            self.inverse_vocab[new_token] = current_id
            current_id += 1
            
            # Apply merge to corpus
            corpus = self._merge_corpus(best_pair, corpus)

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenizes a single word using the learned ordered merge table."""
        symbols = list(word) + [self.end_of_word]
        
        while len(symbols) > 1:
            # Find all candidate pairs in current symbols that exist in merge table
            candidate_pairs = [
                (symbols[i], symbols[i + 1]) 
                for i in range(len(symbols) - 1) 
                if (symbols[i], symbols[i + 1]) in self.merges
            ]
            if not candidate_pairs:
                break  # No more valid merges possible
            
            # Pick the candidate pair that was learned earliest (lowest rank)
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            
            # Apply merge to current symbols
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append("".join(best_pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            
        return symbols

    def encode(self, text: str) -> List[int]:
        """Encodes full text into an array of integer token IDs."""
        token_ids = []
        for word in text.split():
            tokens = self.tokenize_word(word)
            for t in tokens:
                if t in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[t])
                else:
                    # In true byte-level BPE, fallback handles characters natively
                    token_ids.append(-1)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes integer token IDs back into natural text."""
        tokens = [self.vocab.get(tid, "") for tid in token_ids]
        raw_text = "".join(tokens).replace(self.end_of_word, " ")
        return raw_text.strip()

# Verification demonstration
if __name__ == "__main__":
    training_data = [
        "low lower lowest newest wider widest",
        "low lower low newest wider"
    ]
    bpe = ProductionBPETokenizer()
    bpe.train(training_data, num_merges=6)
    
    sample_word = "lowest"
    tokens = bpe.tokenize_word(sample_word)
    ids = bpe.encode(sample_word)
    decoded = bpe.decode(ids)
    
    print(f"Tokenized '{sample_word}':", tokens)
    print("Token IDs:", ids)
    print("Decoded string:", decoded)
    assert decoded == sample_word, "Detokenization must exactly match original string!"
''', language="python")

st.divider()

# ==============================================================================
# Section 10: Interactive Workbench: Subword Tokenizer Simulator
# ==============================================================================
st.header("10. Interactive Workbench: Subword Tokenizer Simulator & Compression Benchmark", divider="rainbow")

st.markdown("""
Experiment with how different natural language passages, code snippets, and foreign scripts are segmented 
across simulated subword algorithms. Observe token counts, compression ratios (bytes per token), and subword boundaries:
""")

col_wb_in1, col_wb_in2 = st.columns([1.2, 0.8])

with col_wb_in1:
    user_workbench_text = st.text_area(
        "Enter Text to Tokenize:",
        value="Artificial intelligence models utilize Byte-Pair Encoding (BPE) to tokenize programming code: def compute_attention(Q, K, V): return softmax(Q @ K.T) @ V",
        height=100
    )

with col_wb_in2:
    selected_algorithm = st.selectbox(
        "Select Subword Algorithm Family:",
        [
            "Byte-Level BPE (Llama 3 / GPT-4)",
            "WordPiece (BERT style)",
            "SentencePiece Unigram (T5 style)",
            "Character-Level (Baseline)"
        ]
    )
    
    vocab_size_choice = st.select_slider(
        "Simulated Vocabulary Size:",
        options=["32,000 (Small)", "50,257 (Medium)", "100,277 (Large)", "128,256 (Frontier)"],
        value="128,256 (Frontier)"
    )

# Simulation segmentation logic
def simulate_subword_segmentation(text: str, algo: str, vocab_scale: str) -> Tuple[List[str], List[int]]:
    words = text.split(" ")
    tokens = []
    
    # Vocabulary size scale factor: larger vocab = longer subwords merged
    scale = 4 if "128,256" in vocab_scale else (3 if "100,277" in vocab_scale else (2 if "50,257" in vocab_scale else 1))
    
    if "Byte-Level BPE" in algo:
        for idx, w in enumerate(words):
            prefix = " " if idx > 0 else ""
            if len(w) <= 4:
                tokens.append(f"{prefix}{w}")
            elif len(w) <= 8 and scale >= 2:
                tokens.append(f"{prefix}{w}")
            elif scale >= 4 and len(w) <= 12:
                tokens.append(f"{prefix}{w}")
            else:
                # Subword split
                mid = len(w) // 2
                tokens.append(f"{prefix}{w[:mid]}")
                tokens.append(f"{w[mid:]}")
    elif "WordPiece" in algo:
        for w in words:
            if len(w) <= 5:
                tokens.append(w)
            else:
                tokens.append(w[:4])
                tokens.append(f"##{w[4:]}")
    elif "SentencePiece" in algo:
        for w in words:
            if len(w) <= 6:
                tokens.append(f"_{w}")
            else:
                tokens.append(f"_{w[:4]}")
                tokens.append(w[4:])
    else:  # Character level
        for c in text:
            tokens.append(c)
            
    # Generate mock deterministic IDs
    token_ids = [abs(hash(t)) % 100000 for t in tokens]
    return tokens, token_ids

sim_tokens, sim_ids = simulate_subword_segmentation(user_workbench_text, selected_algorithm, vocab_size_choice)

# Metrics calculation
utf8_bytes = len(user_workbench_text.encode("utf-8"))
token_count = len(sim_tokens)
bytes_per_token = utf8_bytes / token_count if token_count > 0 else 0.0

st.markdown("#### Tokenization Diagnostics and Compression Metrics")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Raw Characters", f"{len(user_workbench_text)}")
with col_m2:
    st.metric("Total UTF-8 Bytes", f"{utf8_bytes} bytes")
with col_m3:
    st.metric("Generated Token Count", f"{token_count} tokens")
with col_m4:
    st.metric("Compression Efficiency", f"{bytes_per_token:.2f} bytes/token")

st.markdown("#### Visualized Subword Token Boundaries")

# Render colorized tokens
token_pill_html = ""
colors = ["#1E3A8A", "#065F46", "#831843", "#78350F", "#312E81", "#134E4A"]

for idx, tok in enumerate(sim_tokens):
    c = colors[idx % len(colors)]
    disp_tok = tok.replace(" ", "&middot;").replace("_", "&lowbar;")
    token_pill_html += f'<span style="background-color: {c}; color: #F8FAFC; padding: 3px 8px; margin: 3px; border-radius: 6px; font-family: monospace; display: inline-block;">{disp_tok} <small style="color: #94A3B8;">[{sim_ids[idx]}]</small></span>'

st.markdown(token_pill_html, unsafe_allow_html=True)

st.markdown("#### Token Breakdown Table")
token_inspect_df = pd.DataFrame({
    "Token Index": list(range(len(sim_tokens))),
    "Token String": sim_tokens,
    "Token ID": sim_ids,
    "Character Length": [len(t) for t in sim_tokens]
})
st.dataframe(token_inspect_df.head(15), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 11: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Subword tokenization is a prime interview topic for Staff and Principal Machine Learning Engineers. 
Master the underlying systems trade-offs and real-world production vulnerabilities:
""")

st.subheader("Principal / Staff-Level Interview Questions")

with st.expander("Q1: Why does Byte-Level BPE eliminate the UNK token, and how does it handle multi-byte Unicode characters?"):
    st.markdown("""
    **Direct Answer:**
    Byte-Level BPE (BBPE) shifts the base vocabulary from Unicode characters to raw UTF-8 bytes:
    
    1. **Base Completeness**: In UTF-8, every conceivable symbol—including rare emojis, mathematical symbols, and non-Latin glyphs—is encoded into a sequence of between 1 and 4 bytes (values 0 to 255).
    2. **Atomic Ingestion**: Because the tokenizer's base vocabulary contains all 256 possible byte values, any input string can always be represented as a sequence of base byte tokens. An unknown symbol cannot exist, making the `<UNK>` token obsolete.
    3. **Multi-byte Characters**: When an uncommon emoji (e.g. `'🚀'`, encoded as 4 bytes: `0xF0 0x9F 0x9A 0x80`) has not been merged into a single dedicated token, the tokenizer simply emits the 4 individual constituent byte tokens. Downstream transformer self-attention learns to compose these 4 byte tokens into the unified semantic representation of the rocket emoji.
    """)

with st.expander("Q2: What is the mathematical difference in merge selection between BPE and WordPiece?"):
    st.markdown("""
    **Direct Answer:**
    The difference lies in **raw frequency counting** vs. **mutual information likelihood scoring**:
    
    * **BPE Selection Criterion**:
      `Score_BPE(A, B) = Count(A, B)`
      BPE greedily selects the pair that appears most frequently in the training corpus. It does not normalize by the individual frequencies of A or B.
    * **WordPiece Selection Criterion**:
      `Score_WordPiece(A, B) = Count(A, B) / (Count(A) * Count(B))`
      WordPiece normalizes the pair count by the product of individual token counts, maximizing the training corpus likelihood under a unigram language model. This penalizes pairs composed of individually frequent characters (like `'e'` and `'s'`) unless they co-occur at a rate far higher than expected by independent chance.
    """)

with st.expander("Q3: How does Unigram tokenization enable subword regularization during model pretraining?"):
    st.markdown("""
    **Direct Answer:**
    Unlike BPE, which produces a single deterministic segmentation for any word based on rigid merge ranks, the Unigram Language Model estimates probability priors P(x) for all vocabulary tokens:
    
    1. During inference, it uses dynamic programming (Viterbi) to select the single segmentation that maximizes the product of P(x_i).
    2. During pretraining, instead of always selecting the top-1 Viterbi path, the tokenizer samples alternative valid segmentations according to their probability:
       `P(x_1...x_k | Text) proportional to Product_i P(x_i)^alpha`
       Where alpha is a smoothing temperature.
    3. **Impact**: A word like `"internationalization"` might be tokenized as `["international", "ization"]` in epoch 1, and as `["inter", "national", "ization"]` in epoch 2. This acts as automatic morphological data augmentation, making the model highly robust to spelling variations and noisy text.
    """)

with st.expander("Q4: What is the architectural impact of quadrupling vocabulary size from 32k to 128k tokens on GPU memory and latency?"):
    st.markdown("""
    **Direct Answer:**
    The trade-off balances **embedding VRAM overhead** against **sequence compression speed**:
    
    * **Embedding Matrix VRAM (Negative Impact)**:
      The embedding table W_emb and final output lm_head have shape [V, D]. For a model with hidden dimension D = 4,096 in 16-bit precision:
      * At V = 32,000: 32,000 * 4,096 * 2 bytes ~ 262 MB.
      * At V = 128,000: 128,000 * 4,096 * 2 bytes ~ 1.05 GB (plus another 1.05 GB if lm_head is untied).
    * **Inference Latency & Context Compression (Positive Impact)**:
      A 128k vocabulary provides much higher information density per token, compressing documents into ~20% to 30% fewer tokens. Because autoregressive generation produces one token per step, a 20% reduction in token count directly translates to **20% fewer sequential GPU memory-streaming forward passes**, resulting in significantly faster text generation.
    """)

with st.expander("Q5: Why did SentencePiece adopt the whitespace meta-symbol U+2581 ('_') and how does it guarantee lossless detokenization?"):
    st.markdown("""
    **Direct Answer:**
    Classical tokenizers treat whitespace as a delimiter and discard it during pre-tokenization. Reconstructing original spacing required brittle heuristic rules, which frequently corrupted indentation in Python code, math formulas, and whitespace-free languages.
    
    SentencePiece replaces all spaces with the visible Unicode meta-symbol `_` (U+2581) before performing subword segmentation:
    * It treats text as a single unbroken character stream.
    * Every token explicitly records whether it was preceded by a space or attached to the previous word.
    * **Lossless Detokenization**: Detokenization requires zero heuristics: concatenating all token strings and replacing `_` with a standard space reconstructs the exact original text with 100% bit-for-bit fidelity.
    """)

st.subheader("Production Engineering Traps")

st.error("""
🚨 **Trap 1 (Error): Inconsistent Pre-tokenization Regex Across Training and Serving**
If a tokenizer is trained using a regex pattern that isolates digits (e.g. `\\p{N}{1,3}`) but the production inference pipeline uses standard whitespace splitting, numbers like `"12345"` will be tokenized into completely different subword IDs. The model will receive invalid token IDs, producing garbled numerical predictions and failing at arithmetic.
*Remediation*: Always package and distribute the pre-tokenization regex directly inside the tokenizer model artifact (e.g., in `tokenizer.json`).
""")

st.error("""
🚨 **Trap 2 (Error): Vocabulary Size Desynchronization (Tokenizer vs. Model Weight Tensor)**
When fine-tuning an existing model and adding custom tokens (e.g., adding `<|tool_call|>` or new entity tags), the tokenizer vocabulary size increases from 32,000 to 32,005. If the model's embedding matrix W_emb and lm_head are not resized via `model.resize_token_embeddings(len(tokenizer))`, inference will crash with a CUDA IndexOutOfBounds error as soon as a new token ID is generated.
*Remediation*: Enforce CI/CD assertions verifying that `len(tokenizer) == model.config.vocab_size` before serializing model checkpoints.
""")

st.warning("""
⚠️ **Trap 3 (Warning): The Multilingual "Token Tax" in Small Vocabularies**
Deploying an English-optimized 32k vocabulary in non-English enterprise regions (e.g., Japan, India, Middle East) causes extreme token inflation: documents require 3x to 4x more tokens than English equivalents. This increases API billing costs by 400% and cuts the effective context window by 75%.
*Remediation*: Benchmark the tokenizer's compression ratio across target regional languages before selecting a foundation model.
""")

st.warning("""
⚠️ **Trap 4 (Warning): Arithmetic Failure Caused by Subword Number Chunking**
Older tokenizers like GPT-3's BPE inconsistently merged digits based on corpus frequency (e.g. `"12"` was a single token, but `"34"` was split into `"3"` and `"4"`). This prevented attention heads from learning regular place-value positional relationships.
*Remediation*: Modern foundation tokenizers (GPT-4, Llama 3) strictly enforce single-digit or three-digit regex rules, ensuring uniform token representations for mathematical reasoning.
""")

st.subheader("Concept Flashcards")

col_fc1, col_fc2 = st.columns(2)

with col_fc1:
    with st.container():
        st.info("**Byte Pair Encoding (BPE)**")
        st.caption("A bottom-up subword algorithm that iteratively merges the most frequent adjacent symbol bigrams in the training corpus into new vocabulary entries.")

    with st.container():
        st.info("**WordPiece Likelihood Criterion**")
        st.caption("A bottom-up subword algorithm that merges pairs maximizing the mutual information score: Count(AB) / (Count(A) * Count(B)).")

with col_fc2:
    with st.container():
        st.info("**Unigram Top-Down Pruning**")
        st.caption("A subword algorithm that starts with an over-complete candidate vocabulary and iteratively prunes symbols that contribute least to training corpus log-likelihood.")

    with st.container():
        st.info("**Compression Ratio (Bytes/Token)**")
        st.caption("The fundamental efficiency metric of a tokenizer: Total UTF-8 Bytes divided by Total Tokens. Higher values mean faster inference and lower compute costs.")

st.divider()

# ==============================================================================
# Section 12: Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of subword tokenization algorithms and vocabulary construction. 
Complete all 6 checklist items:
""")

cb1 = st.checkbox("1. I understand how BPE builds vocabularies through bottom-up greedy bigram frequency merges.", key="chk_32_1")
cb2 = st.checkbox("2. I can explain the mutual information scoring criterion that distinguishes WordPiece from BPE.", key="chk_32_2")
cb3 = st.checkbox("3. I know how Unigram tokenization prunes vocabularies top-down and enables subword regularization.", key="chk_32_3")
cb4 = st.checkbox("4. I understand how SentencePiece eliminates language-specific pre-tokenizers by treating whitespace as U+2581 (_).", key="chk_32_4")
cb5 = st.checkbox("5. I can explain why Byte-Level BPE (BBPE) over 256 base bytes guarantees 0% out-of-vocabulary UNK tokens.", key="chk_32_5")
cb6 = st.checkbox("6. I understand the architectural trade-offs between vocabulary size, embedding VRAM, and autoregressive decoding latency.", key="chk_32_6")

if cb1 and cb2 and cb3 and cb4 and cb5 and cb6:
    st.success("🎉 Outstanding! You have mastered all foundational subword tokenization algorithms and vocabulary construction dynamics!")
