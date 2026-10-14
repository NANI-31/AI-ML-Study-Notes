import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="2.1 Text Processing - LLM Curriculum",
    page_icon="📝",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("2.1 Text Processing Foundations for Modern LLMs")
st.caption("Chapter 2: NLP Foundations for LLMs | Domain: 2. Large Language Models")

st.markdown("""
Before language models can process natural language, raw character streams must undergo systematic 
**lexical processing, normalization, and structural segmentation**. 

While classical NLP relied on aggressive, destructive normalization techniques—such as case folding, punctuation 
stripping, stopword filtering, and heuristic stemming—modern Foundation LLMs take the opposite architectural stance: 
**lossless pre-tokenization**. Retaining casing, punctuation, and grammatical stopwords is essential for preserving 
syntactic dependency trees, programming syntax, and subtle semantic nuance.

In this lesson, you will master:
1. The 8 fundamental text processing operations and how they evolved from classical NLP to modern LLMs.
2. Why modern foundation models (Llama 3, GPT-4, Mistral) strictly preserve casing, punctuation, and stopwords.
3. The mathematical mechanics of Unicode normalization (NFC, NFD, NFKC, NFKD) and byte-fallback encoding.
4. Stemming vs Lemmatization: the trade-offs between heuristic affix chopping and morphological dictionaries.
5. Production-grade Python implementations comparing destructive preprocessing with lossless tokenization.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: Legacy Preprocessing vs Modern LLMs
# ==============================================================================
st.header("1. Architectural Blueprint: Legacy Preprocessing vs Modern LLM Tokenization", divider="rainbow")

st.markdown("""
Observe the fundamental paradigm shift between classical NLP preprocessing (destructive filtering designed to fit 
sparse matrices on constrained RAM) and modern LLM pre-tokenization curation (lossless preservation designed for deep self-attention):
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    pipeline_dot = """
    digraph TextProcessingComparison {
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
            label="Raw Input Stream";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            RawText [label="Raw Unstructured Text\\n'Dr. Smith paid $450 to Apple Inc.!'", fillcolor="#1E293B"];
        }

        subgraph cluster_classical {
            label="Path A: Classical Destructive NLP";
            style=filled;
            color="#1E293B";
            fontcolor="#EF4444";
            C1 [label="1. Lowercasing\\n'dr. smith paid $450 to apple inc.!'", fillcolor="#450A0A", color="#EF4444"];
            C2 [label="2. Punctuation Strip\\n'dr smith paid 450 to apple inc'", fillcolor="#450A0A", color="#EF4444"];
            C3 [label="3. Stopword Removal\\n'smith paid 450 apple inc'", fillcolor="#450A0A", color="#EF4444"];
            C4 [label="4. Porter Stemming\\n['smith', 'pai', '450', 'appl', 'inc']", fillcolor="#7F1D1D", color="#EF4444"];
            C_Out [label="Destroyed Context Matrix\\n(Lost casing, syntax, negation)", fillcolor="#991B1B", color="#EF4444"];
        }

        subgraph cluster_modern {
            label="Path B: Modern Lossless LLM Pre-Tokenization";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            M1 [label="1. Unicode NFC Normalization\\n(Canonical composite UTF-8 bytes)", fillcolor="#064E3B", color="#34D399"];
            M2 [label="2. Regex Pre-Splitting\\n(Preserves casing, punctuation, symbols)", fillcolor="#064E3B", color="#34D399"];
            M3 [label="3. Byte-Pair Encoding (BPE)\\n(Maps subwords to dense vocabulary IDs)", fillcolor="#065F46", color="#34D399"];
            M4 [label="4. Full Context Retention\\n(Casing, grammar, indentation preserved)", fillcolor="#047857", color="#34D399"];
            M_Out [label="Preserved Latent Context\\n(Feeds into Transformer Attention)", fillcolor="#064E3B", color="#34D399"];
        }

        RawText -> C1;
        C1 -> C2 -> C3 -> C4 -> C_Out;

        RawText -> M1;
        M1 -> M2 -> M3 -> M4 -> M_Out;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(pipeline_dot, use_container_width=True)

with col_diag2:
    st.markdown("### The Paradigm Shift")
    st.markdown("""
    * **Classical NLP Goal**: Minimize vocabulary size to avoid the curse of dimensionality in sparse bag-of-words and TF-IDF matrices.
    * **Classical Consequence**: Destroyed syntax, lost negation words (*not*, *never*), corrupted code indentation, and conflated proper nouns (*Apple* the company vs *apple* the fruit).
    * **Modern LLM Goal**: Retain 100% of character, casing, and syntactic signals so deep self-attention heads can resolve polysemy, tone, and code semantics.
    """)

st.divider()

# ==============================================================================
# Section 2: 1. Text Normalization
# ==============================================================================
st.header("2. Text Normalization: Unicode Equivalence, NFKC/NFC Forms & Canonical Decomposition", divider="rainbow")

st.markdown("""
Text normalization transforms text into a standardized canonical format to eliminate superficial variations. 
In modern software systems, text normalization focuses primarily on **Unicode Normalization**. 
The Unicode standard allows the same visible character to be encoded in multiple distinct byte sequences:
* **Canonical Equivalence**: Characters that represent the exact same abstract glyph (e.g., `é` as a single code point `U+00E9` vs `e` followed by combining acute accent `U+0065 U+0301`).
* **Compatibility Equivalence**: Characters that represent different formatting of the same character (e.g., ligature `ﬁ` vs individual letters `f` and `i`, or superscript `2` vs digit `2`).

### The Four Unicode Normalization Forms
1. **NFC (Normalization Form C)**: Canonical Decomposition followed by Canonical Composition (recommended for web and LLMs). Combines base characters with accents into single composite code points.
2. **NFD (Normalization Form D)**: Canonical Decomposition. Separates composite characters into base characters and independent combining marks.
3. **NFKC (Normalization Form KC)**: Compatibility Decomposition followed by Canonical Composition. Normalizes ligatures (`ﬁ` -> `fi`) and full-width characters.
4. **NFKD (Normalization Form KD)**: Compatibility Decomposition. Decomposes both composite glyphs and compatibility variants.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Pre-Tokenization NFC Standardization**: Foundation models (Llama 3, Mistral, GPT-4) normalize input text using **NFC** before byte-pair encoding. If raw text contains unnormalized combining characters, NFC composes them, ensuring identical tokens are generated regardless of client OS or keyboard layout.
* **Why NFKC is Used Cautiously**: While NFKC simplifies ligatures, aggressive compatibility decomposition can alter mathematical symbols or code syntax (e.g., converting superscript exponents into standard digits, corrupting algebraic formulas). Modern LLM data pipelines use standard NFC or lightweight customized NFKC rules.
* **Byte-Fallback Safety**: Modern tokenizers fall back directly to raw UTF-8 byte tokens, ensuring that even unnormalized or corrupted byte streams are ingested without raising exceptions or emitting `<unk>` tokens.
""")

df_norm = pd.DataFrame({
    "Dimension": [
        "Normalization Focus",
        "Unicode Form Employed",
        "Handling Accents & Diacritics",
        "Ligature & Symbol Preservation",
        "Out-of-Vocabulary Impact"
    ],
    "Legacy NLP (Rule-Based & Statistical)": [
        "Aggressive ASCII folding; stripping accents via unidecode",
        "Ad-hoc regex replaces or forced ASCII transliteration",
        "Accents stripped ('café' -> 'cafe', 'résumé' -> 'resume')",
        "Ligatures deleted or corrupted; symbols stripped",
        "Frequent <unk> tokens on foreign scripts and emojis"
    ],
    "Modern Foundation LLMs (Llama 3 / GPT-4)": [
        "Lossless Unicode NFC canonical composition",
        "Standard NFC precomposed form; minimal NFKC",
        "Accents preserved perfectly as distinct native subwords or bytes",
        "Full preservation of emojis, mathematical symbols, and ligatures",
        "Zero <unk>; byte-fallback maps arbitrary bytes to vocabulary"
    ]
})
st.dataframe(df_norm, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Lowercasing
# ==============================================================================
st.header("3. Lowercasing: Case Folding, Named Entity Preservation & Subword Sensitivity", divider="rainbow")

st.markdown("""
Lowercasing (also known as Case Folding) converts all alphabetical characters in a string to lowercase. 
In classical NLP, lowercasing was universally mandated to collapse redundant vocabulary dimensions—ensuring 
that `"The"`, `"the"`, and `"THE"` shared a single bag-of-words index.

### The Catastrophic Cost of Blind Lowercasing
In modern AI systems, case folding destroys vital semantic and syntactic information:
1. **Named Entity Disambiguation**: `"Apple"` (multinational technology corporation) vs `"apple"` (edible fruit); `"May"` (calendar month or surname) vs `"may"` (modal auxiliary verb); `"White House"` (presidential residence) vs `"white house"` (any painted home).
2. **Sentiment & Urgency Modality**: `"I am NOT happy"` conveys decisive emphasis compared to `"I am not happy"`. ALL-CAPS text signals screaming, urgency, or legal liability warnings.
3. **Programming Code Semantics**: In Python, C++, and Java, casing dictates identifier types: `class OrderProcessor` (PascalCase class definition), `def process_order` (snake_case function), `MAX_TIMEOUT = 30` (SCREAMING_SNAKE_CASE constant). Lowercasing destroys code executability.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Case-Sensitive Subword Vocabularies**: Modern foundation tokenizers maintain separate, dedicated token IDs for title-case, lower-case, and upper-case tokens. For example, in Tiktoken / Llama 3:
  * `" apple"` -> Token ID `17163`
  * `" Apple"` -> Token ID `12314`
  * `" APPLE"` -> Token ID `42456`
* **Autonomous Capitalization Modeling**: Because casing is retained, LLMs natively learn capitalization rules, grammatical sentence boundaries, and syntactic capitalization without external rules.
""")

df_case = pd.DataFrame({
    "Dimension": [
        "Casing Treatment",
        "Vocabulary Impact",
        "Entity Disambiguation",
        "Programming Code Support",
        "Sentiment & Emphasis"
    ],
    "Legacy NLP (BERT uncased / Word2Vec)": [
        "Forced lowercasing (all text converted to lowercase)",
        "Slashed vocabulary size by 40% to fit constrained RAM",
        "Completely destroyed; 'Apple' and 'apple' map to identical vector",
        "Fails on case-sensitive code syntax and variable scoping",
        "Stripped all capitalization emphasis and urgency nuances"
    ],
    "Modern Foundation LLMs (Llama 3 / GPT-4)": [
        "Strictly case-sensitive; 100% case preservation",
        "Dedicated subwords for capitalized, lowercase, and all-caps words",
        "Preserved natively; 'Apple' attends to corporate context",
        "Full support for PascalCase, camelCase, snake_case, and constants",
        "Accurately interprets capitalized anger, legal disclaimers, and emphasis"
    ]
})
st.dataframe(df_case, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Punctuation Handling
# ==============================================================================
st.header("4. Punctuation Handling: Syntactic Delimiters, Special Tokens & Code Formatting", divider="rainbow")

st.markdown("""
Punctuation marks (`.`, `,`, `!`, `?`, `;`, `:`, `-`, `""`, `()`) organize text into structured syntactic hierarchies. 
In classical bag-of-words pipelines, punctuation marks were stripped via regex (`re.sub(r'[^\\w\\s]', '', text)`) because 
they had no standalone semantic weight in frequency-based document vectors.

### Punctuation as Syntactic Routing Tokens
In transformer architectures, punctuation marks are not noise; they are critical **attention routing anchors**:
* **Periods (`.`) & Question Marks (`?`)**: Serve as natural boundary signals, allowing self-attention heads to constrain clause-level dependencies.
* **Commas (`,`) & Semicolons (`;`)**: Route grammatical parentheticals and subordinate clauses.
* **Quotes (`""`) & Parentheses (`()`)**: Act as syntactic delimiters enclosing citations, direct speech, and function arguments.
* **Hyphens & Dashes (`-`, `—`)**: Disambiguate compound adjectives (`"state-of-the-art"`) and parenthetical em-dashes.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Dedicated Punctuation Subwords**: Modern tokenizers dedicate individual token IDs to punctuation marks and common whitespace-punctuation pairs (e.g. `","`, `" ("`, `").\n"`).
* **Indentation Preservation**: In programming languages (especially Python and YAML), whitespace and indentation (`\t`, four spaces) define execution scope. Stripping punctuation or collapsing whitespace breaks code compilation.
* **Structured Generation Control**: Punctuation marks (such as curly braces `{}` and quotes `""`) guide grammar-constrained decoding (CFGs) when extracting structured JSON payloads.
""")

df_punct = pd.DataFrame({
    "Dimension": [
        "Punctuation Policy",
        "Impact on Syntax Parsing",
        "Code Indentation & Formatting",
        "Clause-Level Attention",
        "Dialogue & Direct Quotes"
    ],
    "Legacy NLP (TF-IDF / Naive Bayes)": [
        "Stripped entirely via regex string filtering",
        "Destroyed sentence boundaries; created run-on word bags",
        "Stripped tabs and newlines; rendered source code completely unparseable",
        "Zero attention routing; all words treated as an unordered collection",
        "Stripped quote marks; lost distinction between speaker and statement"
    ],
    "Modern Foundation LLMs": [
        "100% preserved; treated as high-priority structural tokens",
        "Provides explicit syntactic anchors for multi-head attention",
        "Preserves whitespace and tabs, enabling Python / YAML compilation",
        "Attention heads specifically track clause boundaries and commas",
        "Accurately parses dialogue speaker attribution and nested quotes"
    ]
})
st.dataframe(df_punct, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Stop Words
# ==============================================================================
st.header("5. Stop Words: Information Density vs. Attention Routing & Why LLMs Retain Function Words", divider="rainbow")

st.markdown("""
Stop words are high-frequency grammatical function words—such as articles (*the*, *a*, *an*), prepositions (*in*, *on*, *at*), 
conjunctions (*and*, *but*), and auxiliary verbs (*is*, *was*). In classical information retrieval, stop words were discarded 
using hardcoded lists (e.g. NLTK's 179 English stopwords) because their high frequency diluted term-frequency signals in sparse vectors.

### The Fatal Flaw of Stop Word Removal
Removing stop words fundamentally corrupts semantic meaning:
1. **Negation Reversal**: In `"The medication is not effective"`, stripping the stop word `"not"` reverses the clinical diagnosis into `"medication effective"`, introducing catastrophic medical risk.
2. **Grammatical Polysemy**: In Shakespeare's iconic line `"To be or not to be"`, every single word is a stop word. Stripping stop words produces an empty string (`""`).
3. **Relational Directionality**: In `"flights from Boston to Chicago"`, the prepositions `"from"` and `"to"` define origin and destination. Stripping them yields `"flights Boston Chicago"`, confusing the route.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Zero Stop Word Removal**: Modern foundation models never remove stop words during pretraining, fine-tuning, or inference.
* **Induction Heads & Syntactic Glue**: Transformer attention heads specifically utilize function words as relational glue. Induction heads track articles and prepositions to bind nouns to verbs and resolve coreference chains across multiple sentences.
* **Softmax Attention Masking**: Rather than permanently deleting stop words, the self-attention mechanism dynamically computes how much weight to place on function words versus semantic nouns conditioned on the specific task.
""")

df_stop = pd.DataFrame({
    "Dimension": [
        "Stop Word Treatment",
        "Handling of Negations ('not', 'never')",
        "Directional Prepositions ('from', 'to')",
        "Linguistic Fluency",
        "Grammatical Relations"
    ],
    "Legacy NLP (NLTK Stopwords / BoW)": [
        "Aggressively deleted using static lists of 100-500 words",
        "Often stripped, flipping sentence polarity and sentiment",
        "Deleted; confused travel routes and relational predicates",
        "Produced disjointed, robotic keyword fragments",
        "Completely destroyed grammatical subject-verb-object dependencies"
    ],
    "Modern Foundation LLMs": [
        "100% retained; zero stopword removal",
        "Preserved with high attention weights on negation tokens",
        "Directly guides relational vectors and argument binding",
        "Produces natural, fluent, human-grade prose",
        "Self-attention heads dynamically route information through function words"
    ]
})
st.dataframe(df_stop, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Stemming
# ==============================================================================
st.header("6. Stemming: Heuristic Affix Stripping (Porter/Snowball) vs. Subword Tokenization", divider="rainbow")

st.markdown("""
Stemming is a crude, rule-based heuristic process that strips inflectional and derivational affixes (prefixes and suffixes) 
from words to reduce them to a common root stem. 
The most famous algorithms—the **Porter Stemmer (1980)** and **Snowball Stemmer**—apply sequential cascading rewrite rules 
based on consonant-vowel patterns (e.g. replacing `-sses` with `-ss`, `-ies` with `-i`, and stripping `-ing` or `-ed`).

### Fundamental Flaws of Stemming
Because stemmers rely on mechanical rules without linguistic dictionaries or context, they suffer from two major failure modes:
1. **Over-Stemming**: Chopping off too much, conflating distinct semantic concepts:
   * `"universal"`, `"universe"`, and `"university"` all reduced to the stem `"univers"`.
   * `"organization"` and `"organ"` both stemmed to `"organ"`.
2. **Under-Stemming**: Failing to connect words with common etymological roots due to irregular spelling:
   * `"ran"` and `"run"` remain distinct stems.
   * `"cylinder"` and `"cylindrical"` fail to merge.
3. **Non-Word Stems**: Generates invalid, misspelled strings (e.g. `"business"` -> `"busi"`, `"trouble"` -> `"troubl"`).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Subword Tokenization Replaces Stemming**: Modern foundation models completely eliminate stemming. Subword algorithms (BPE, WordPiece) break words down into recurring sub-units (e.g., `"un" + "believ" + "able"`), naturally discovering morphological roots without generating invalid non-word stems.
* **Shared Embedding Subspaces**: Because subword tokens share high-dimensional latent space representations, the model learns that `"run"`, `"running"`, and `"ran"` inhabit nearby geometric regions while preserving tense and grammatical nuance.
""")

df_stem = pd.DataFrame({
    "Dimension": [
        "Reduction Algorithm",
        "Output Validity",
        "Over-Stemming Risk",
        "Irregular Verbs ('run' vs 'ran')",
        "Modern LLM Status"
    ],
    "Heuristic Stemming (Porter / Snowball)": [
        "Heuristic regex affix stripping without dictionary verification",
        "Frequently outputs non-words ('busi', 'oper', 'commun')",
        "High; conflates 'university' with 'universe'",
        "Fails completely; cannot connect irregular morphological variants",
        "Obsolete; never used in modern generative foundation models"
    ],
    "Modern Foundation Tokenization (Subword BPE)": [
        "Data-driven subword segmentation trained on frequency statistics",
        "100% valid subwords or byte representations",
        "Zero over-stemming; retains distinct tokens for 'university' and 'universe'",
        "Captured natively via shared continuous geometric embedding vectors",
        "Universal standard; subword BPE preserves morphological nuance"
    ]
})
st.dataframe(df_stem, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Lemmatization
# ==============================================================================
st.header("7. Lemmatization: Morphological Vocabulary Dictionaries vs. Contextual Polysemy", divider="rainbow")

st.markdown("""
Lemmatization is a linguistically grounded process that reduces inflected forms of a word to its canonical base dictionary form, 
known as the **lemma**. 
Unlike stemming, lemmatization relies on **complete morphological dictionaries** (such as WordNet) and requires 
**Part-of-Speech (POS) disambiguation**:
* The word `"meeting"` lemmatized as a **Noun** remains `"meeting"`.
* The word `"meeting"` lemmatized as a **Verb** (`"We are meeting tomorrow"`) reduces to the lemma `"meet"`.
* The irregular past tense `"better"` lemmatized as an **Adjective** reduces to `"good"`.

### Computational Overhead & Contextual Limits
While lemmatization avoids the spelling corruption of stemming, it introduces significant pipeline dependencies:
1. **Dependency on POS Taggers**: If the upstream POS tagger mislabels a word (tagging a verb as a noun), lemmatization produces the wrong lemma.
2. **High Latency Overhead**: Morphological dictionary lookups and lemmatizers (e.g. Spacy, WordNet) are slow and CPU-bound, making them impractical for multi-terabyte web pretraining datasets.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Implicit Contextual Polysemy**: Foundation models do not need to reduce words to lemmas because the **Transformer attention layers implicitly compute contextualized representations**. The token `"meeting"` in a business conference context develops an internal vector representation completely distinct from `"meeting"` in an encounter context across the residual stream.
* **Lossless Tense Preservation**: Lemmatizing `"was running"` to `"be run"` destroys grammatical tense and aspect. LLMs retain the exact grammatical form to generate fluent, temporally consistent answers.
""")

df_lem = pd.DataFrame({
    "Dimension": [
        "Reduction Mechanism",
        "Linguistic Accuracy",
        "POS Tagging Dependency",
        "Processing Speed",
        "Grammatical Tense Preservation"
    ],
    "Classical Lemmatization (WordNet / Spacy)": [
        "Morphological lexical lookup using WordNet database and POS tags",
        "High; always produces a valid dictionary base word (lemma)",
        "Strictly required; fails or degrades without accurate POS tagger",
        "Slow; dictionary lookups introduce substantial CPU bottlenecks",
        "Destroys tense; reduces 'was', 'were', 'is' to generic 'be'"
    ],
    "Modern Foundation LLMs": [
        "Direct subword tokens mapped into contextualized residual vectors",
        "Native contextual comprehension; resolves polysemy via self-attention",
        "Zero dependency; POS and syntax emerge implicitly in transformer layers",
        "Extremely fast; GPU matrix operations during byte-pair encoding",
        "100% preserved; retains exact past/present/future tense distinctions"
    ]
})
st.dataframe(df_lem, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Sentence Segmentation
# ==============================================================================
st.header("8. Sentence Segmentation: Rule-Based Sentence Boundaries vs. Continuous Attention Streams", divider="rainbow")

st.markdown("""
Sentence Segmentation (Sentence Boundary Disambiguation - SBD) divides continuous text documents into discrete sentence units. 
In natural language, sentence segmentation is surprisingly difficult because the primary sentence terminator—the period (`.`)—is 
heavily overloaded across multiple linguistic roles:
1. **Abbreviations & Titles**: `"Dr. Smith"`, `"U.S.A."`, `"Inc."`, `"e.g."`, `"vs."`.
2. **Numeric Decimals**: `"$4,500.50"`, `"Version 3.14"`.
3. **Ellipses & Punctuation Clusters**: `"Wait... what?!"`.
4. **Email Addresses & URLs**: `"support@company.com"`, `"https://openai.com"`.

### Classical SBD Algorithms
Classical tools (Punkt in NLTK, Spacy Dependency Parser) use unsupervised statistical models or decision trees to determine 
whether a period represents an abbreviation or a true sentence boundary.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Continuous Attention Streams**: Foundation models do not require explicit sentence boundary splitting during training or inference. They ingest continuous streams of text packed into fixed context windows (e.g. 8,192 tokens per training sample), learning sentence boundaries natively through self-attention patterns.
* **Sentence Boundaries in RAG Chunking**: While LLMs do not require sentence segmentation internally, **production RAG pipelines rely heavily on SBD** to chunk large PDFs into semantically coherent passages. Naively splitting text on arbitrary character offsets splits sentences in half, causing vector embeddings to lose semantic coherence.
""")

df_sbd = pd.DataFrame({
    "Dimension": [
        "Segmentation Method",
        "Abbreviation Disambiguation",
        "Cross-Sentence Attention",
        "Model Ingestion Scope",
        "Role in Modern Systems"
    ],
    "Classical SBD (NLTK Punkt / Spacy)": [
        "Heuristic decision trees or unsupervised abbreviation frequency tables",
        "Prone to false splits on unlisted acronyms (e.g. splitting 'Inc. on Monday')",
        "Zero; classical models processed each sentence in complete isolation",
        "Sentence-level input buffers (50-100 tokens max)",
        "Core requirement for legacy parsers and feature extractors"
    ],
    "Modern Foundation LLMs + RAG": [
        "Self-attention naturally detects clause boundaries via learned weights",
        "Handles arbitrary acronyms and code notation seamlessly in context",
        "Full cross-sentence attention across 100,000+ tokens simultaneously",
        "Continuous document streams packed into large context windows",
        "Used externally in RAG pipelines to create semantically clean chunk boundaries"
    ]
})
st.dataframe(df_sbd, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: 8. Word Segmentation
# ==============================================================================
st.header("9. Word Segmentation: White-Space Assumptions vs. Byte-Pair Encoding Across Global Languages", divider="rainbow")

st.markdown("""
Word segmentation segments continuous character sequences into discrete lexical units (words). 
In Western languages (English, Spanish, French), word boundaries are conveniently marked by whitespace. 
However, applying whitespace segmentation globally is a severe linguistic failure:
* **Non-Segmented Scripts**: Languages such as Chinese, Japanese, Thai, and Lao write sentences as continuous character strings without whitespace (e.g., Chinese: `"我喜欢学习机器学习"`).
* **Compound Nouns**: Languages like German, Dutch, and Swedish join words into massive compounds without spaces (e.g., German: `"Donaudampfschifffahrtsgesellschaftskapitän"` - Danube steamship company captain).
* **Hyphenated Words & URLs**: Modern text contains URLs, hashtags (`#MachineLearning`), and compound terms (`"cross-validation"`, `"end-to-end"`).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Subword Byte-Pair Encoding (BPE)**: Modern foundation models (Llama 3, Tiktoken, Mistral) resolve global word segmentation through subword BPE and SentencePiece. Instead of assuming whitespace boundaries, BPE operates on raw UTF-8 bytes, discovering optimal subword units statistically across all global scripts.
* **Seamless Multilingual Processing**: A single BPE tokenizer segments English words, decompounds German compound nouns into constituent words, and segments Chinese characters into meaningful semantic multi-character words—using the exact same algorithm.
""")

df_ws = pd.DataFrame({
    "Dimension": [
        "Segmentation Primitive",
        "Non-Whitespace Scripts (Chinese/Thai)",
        "Compound Word Handling (German)",
        "Domain-Specific Slang & Hashtags",
        "Global Consistency"
    ],
    "Legacy Word Tokenization (Whitespace / Jieba)": [
        "Whitespace splitting for English; specialized external dictionary tools for Asian scripts",
        "Requires language-specific tools (Jieba for Chinese, Mecab for Japanese)",
        "Fails; treats German compounds as massive single unknown words",
        "Splits hashtags into single unknown tokens or fails on punctuation",
        "Fragmented; requires maintaining separate segmentation packages per language"
    ],
    "Modern Subword BPE / Tiktoken": [
        "Data-driven statistical Byte-Pair Encoding over raw UTF-8 byte streams",
        "Processes Chinese, Japanese, and Thai natively into common subword tokens",
        "Automatically decomposes long German compounds into constituent roots",
        "Segments hashtags and code identifiers into clean constituent subwords",
        "Single unified tokenizer processes 100+ global languages universally"
    ]
})
st.dataframe(df_ws, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Preprocessing vs Lossless Tokenization
# ==============================================================================
st.header("10. Production Implementation: Preprocessing vs Lossless Tokenization in Python", divider="rainbow")

st.markdown("""
The following complete, runnable Python script contrasts classical destructive NLP preprocessing 
with a modern enterprise-grade **Lossless Pre-Tokenization Pipeline** featuring Unicode NFC normalization, 
abbreviation-safe sentence boundary disambiguation, and whitespace preservation for code:
""")

st.code(r'''import re
import unicodedata
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# 1. Classical Destructive Preprocessing Pipeline (Legacy NLP)
# ----------------------------------------------------------------------
def classical_destructive_preprocess(text: str) -> List[str]:
    """
    Simulates classical NLP: Lowercasing, punctuation stripping,
    stopword filtering, and heuristic affix truncation.
    """
    # Step 1: Lowercase
    lowered = text.lower()
    
    # Step 2: Strip all punctuation
    no_punct = re.sub(r'[^\w\s]', '', lowered)
    
    # Step 3: Split on whitespace
    tokens = no_punct.split()
    
    # Step 4: Remove stopwords
    stopwords = {"the", "a", "an", "is", "was", "to", "of", "and", "in", "on", "not", "he", "it"}
    filtered = [t for t in tokens if t not in stopwords]
    
    # Step 5: Heuristic pseudo-stemming (e.g. strip 'ing', 'ed', 'ly')
    stemmed = [re.sub(r'(ing|ed|ly|es|s)$', '', t) for t in filtered]
    return stemmed

# ----------------------------------------------------------------------
# 2. Modern Lossless Pre-Tokenization Pipeline (Foundation LLM Standard)
# ----------------------------------------------------------------------
class ModernLosslessPreprocessor:
    """
    Enterprise preprocessor for LLM fine-tuning, RAG chunking, and curation.
    Preserves casing, punctuation, code indentation, and Unicode integrity.
    """
    ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "corp", "e.g", "i.e", "vs", "etc", "u.s"}

    @classmethod
    def normalize_unicode_nfc(cls, text: str) -> str:
        """Applies canonical composite Unicode normalization."""
        return unicodedata.normalize("NFC", text)

    @classmethod
    def clean_unprintable_controls(cls, text: str) -> str:
        """Strips corrupt control characters while preserving tabs and newlines for code."""
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r")

    @classmethod
    def segment_sentences_safely(cls, text: str) -> List[str]:
        """Abbreviation-aware sentence boundary segmentation protecting decimals and titles."""
        # Protect decimal numbers (e.g. 4.5 -> 4<DEC>5)
        protected = re.sub(r'(\d+)\.(\d+)', r'\1<DEC>\2', text)
        
        # Split on sentence terminators followed by whitespace
        chunks = re.split(r'([.!?]+)(\s+)', protected)
        sentences = []
        current = ""
        
        i = 0
        while i < len(chunks):
            current += chunks[i]
            if i + 1 < len(chunks):
                delimiter = chunks[i + 1]
                ws = chunks[i + 2] if i + 2 < len(chunks) else ""
                
                # Check for abbreviation
                words = re.findall(r'\b[a-zA-Z]+\b', current)
                last_word = words[-1].lower() if words else ""
                
                if last_word in cls.ABBREVIATIONS:
                    current += delimiter + ws
                else:
                    sentences.append(current + delimiter)
                    current = ""
                i += 3
            else:
                i += 1
                
        if current.strip():
            sentences.append(current)
            
        return [s.replace("<DEC>", ".").strip() for s in sentences if s.strip()]

# ----------------------------------------------------------------------
# 3. Demonstration & Comparative Audit
# ----------------------------------------------------------------------
sample_input = """
Dr. Smith paid $4,500.50 to Apple Inc. on 12/05/2026! 
He was NOT satisfied with the compute server.
\tdef check_status(code=500):
\t\treturn "Failed" if code != 200 else "OK"
"""

legacy_tokens = classical_destructive_preprocess(sample_input)
modern_clean = ModernLosslessPreprocessor.normalize_unicode_nfc(sample_input)
modern_sentences = ModernLosslessPreprocessor.segment_sentences_safely(modern_clean)

print("--- Classical Destructive Preprocessing Result ---")
print("Tokens:", legacy_tokens)
print("Notice: Negation 'NOT' deleted, code destroyed, 'Apple' lost casing!\n")

print("--- Modern Lossless Pre-Tokenization Result ---")
print(f"Segmented Sentences ({len(modern_sentences)} total):")
for idx, s in enumerate(modern_sentences, 1):
    print(f"[{idx}] {repr(s)}")
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: Preprocessing Ablation & Information Loss
# ==============================================================================
st.header("11. Interactive Workbench: Preprocessing Ablation & Information Loss Simulator", divider="rainbow")

st.markdown("""
Test the direct impact of various preprocessing filters on lexical preservation, token volume, and semantic fidelity. 
Observe how aggressive classical techniques corrupt negation words, proper nouns, and code indentation.
""")

col_wb1, col_wb2 = st.columns([1.2, 1])

with col_wb1:
    user_input_text = st.text_area(
        "Enter Raw Input Text to Preprocess:",
        value="Dr. John Smith paid $4,500.50 to Apple Inc. on 12/05/2026! He was NOT satisfied with the server...\n\tdef compute_tax(rate=0.15):\n\t\treturn 4500.50 * rate",
        height=140
    )

with col_wb2:
    st.markdown("##### Preprocessing Configuration")
    cfg_casing = st.radio("Casing Treatment:", ["Preserve Casing (LLM Standard)", "Force Lowercase (Classical NLP)"], horizontal=True)
    cfg_punct = st.radio("Punctuation Treatment:", ["Preserve All Punctuation (LLM)", "Strip All Punctuation (Classical)"], horizontal=True)
    cfg_stopwords = st.checkbox("Filter Common Stopwords (Classical BoW)", value=False)
    cfg_sbd = st.checkbox("Run Abbreviation-Aware Sentence Segmentation", value=True)

# Execute Transformation
processed_text = user_input_text

if "Force Lowercase" in cfg_casing:
    processed_text = processed_text.lower()

if "Strip All Punctuation" in cfg_punct:
    processed_text = "".join(ch for ch in processed_text if ch.isalnum() or ch.isspace())

tokens_list = processed_text.split()
stop_set = {"the", "was", "with", "to", "on", "he", "not", "def", "return"}
filtered_count = 0

if cfg_stopwords:
    retained_tokens = []
    for t in tokens_list:
        if t.lower() in stop_set:
            filtered_count += 1
        else:
            retained_tokens.append(t)
    tokens_list = retained_tokens

st.markdown("#### Preprocessing Telemetry")

met_c1, met_c2, met_c3, met_c4 = st.columns(4)
met_c1.metric("Original Characters", len(user_input_text))
met_c2.metric("Resulting Tokens", len(tokens_list))
met_c3.metric("Stopwords Deleted", filtered_count)
computed_fidelity = 100 - (filtered_count * 10) - (25 if "Strip" in cfg_punct else 0) - (15 if "Lowercase" in cfg_casing else 0)
met_c4.metric("Estimated Semantic Fidelity", f"{max(computed_fidelity, 5)}%")

col_out1, col_out2 = st.columns(2)

with col_out1:
    st.subheader("Resulting Token Stream")
    st.write(tokens_list)

with col_out2:
    st.subheader("Sentence Segmentation")
    if cfg_sbd:
        import re
        s_chunks = re.split(r'(?<=[.!?])\s+(?=[A-Zdef])', user_input_text)
        for idx, s in enumerate(s_chunks):
            st.info(f"**Segment {idx + 1}**: `{s.strip()}`")
    else:
        st.write("Sentence boundary segmentation disabled.")

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Text Processing Interview Questions")

q_col1, q_col2 = st.columns(2)

with q_col1:
    with st.expander("Q1: Why did modern LLMs abandon classical stopword removal?"):
        st.markdown("""
        **Core Architectural Answer:**  
        In classical bag-of-words and TF-IDF pipelines, stop words (e.g., *the*, *is*, *in*, *not*) were deleted because their high document frequency diluted sparse term vectors.
        
        In modern transformers, **stop words are essential syntactic and semantic routing tokens**:
        1. **Negation Flipping**: In `"The contract is not legally binding"`, deleting `"not"` inverts the entire legal meaning into `"contract legally binding"`.
        2. **Induction Heads**: Attention heads specifically route information through prepositions (*from*, *to*) to identify origin and destination in flight schedules or relational queries.
        3. **Linguistic Fluency**: Generating natural language requires function words to maintain proper grammar, syntax, and voice.
        """)

    with st.expander("Q2: What is the difference between Unicode NFC and NFD, and why does it matter for LLMs?"):
        st.markdown("""
        **Answer:**  
        * **NFC (Normalization Form C)**: Composes decomposed base characters and combining accents into single precomposed code points (e.g. `e` + `´` becomes `é` as code point `U+00E9`).
        * **NFD (Normalization Form D)**: Decomposes composite characters into individual base characters and combining marks (`é` becomes `U+0065` and `U+0301`).
        
        **LLM Production Consequence**: If a training dataset contains a mixture of NFC and NFD, the tokenizer will produce completely different token sequences for the exact same word (e.g. 1 token for NFC `é`, but 2 tokens for NFD `e` + accent). This fragments vocabulary statistics and increases inference token costs. Modern data pipelines enforce strict NFC normalization before tokenization.
        """)

    with st.expander("Q3: Why is Byte-Pair Encoding (BPE) superior to rule-based stemming?"):
        st.markdown("""
        **Answer:**  
        * **Stemming (Porter/Snowball)**: Uses heuristic suffix-stripping rules without a dictionary. It produces non-words (`"university"` -> `"univers"`) and conflates unrelated concepts (over-stemming `"organ"` and `"organization"` to the same stem).
        * **Subword BPE**: Data-driven, statistical segmentation trained directly on token co-occurrence across trillions of tokens. It naturally discovers true morphological roots, preserves valid spellings, and handles compound words, slang, and code seamlessly without heuristic rules.
        """)

with q_col2:
    with st.expander("Q4: How does blind lowercasing corrupt modern AI programming and entity extraction systems?"):
        st.markdown("""
        **Answer:**  
        Blind lowercasing (`text.lower()`) destroys three critical semantic dimensions:
        1. **Proper Nouns vs Common Nouns**: Conflates `"Apple"` (corporation) with `"apple"` (fruit), or `"May"` (month/name) with `"may"` (modal verb).
        2. **Code Syntax Scoping**: In programming languages, casing defines syntax rules—classes use `PascalCase`, functions use `snake_case`, and constants use `SCREAMING_SNAKE_CASE`. Lowercasing breaks code syntax and causes runtime compilation failures.
        3. **Sentiment & Urgency**: Removes user capitalization emphasis (`"I said NO!"` vs `"I said no"`).
        """)

    with st.expander("Q5: How does Sentence Boundary Disambiguation (SBD) impact RAG retrieval pipelines?"):
        st.markdown("""
        **Answer:**  
        In Retrieval-Augmented Generation (RAG), documents must be split into chunks before generating vector embeddings. 
        If chunking is performed using naive character counts (e.g. every 500 characters), sentences are split in half across chunk boundaries. 
        This introduces severe incomplete thoughts into vector embeddings, degrading cosine retrieval similarity. 
        Applying abbreviation-aware SBD guarantees that chunk boundaries fall strictly between complete, semantically cohesive sentences.
        """)

st.divider()

st.subheader("Production Traps & Processing Gotchas")

t_col1, t_col2 = st.columns(2)

with t_col1:
    st.error("""
    **Production Trap 1 (Critical): Blind Regex Punctuation Stripping on Code Datasets**  
    * **Failure Mode**: Running `re.sub(r'[^\\w\\s]', '', text)` on pretraining corpora containing source code or markdown.
    * **Impact**: Destroys programming syntax—stripping colons, brackets, tabs, and indentation—rendering Python and YAML code completely invalid.
    * **Remediation**: Never strip punctuation from LLM pretraining or fine-tuning datasets. Rely on modern byte-fallback tokenizers to encode punctuation losslessly.
    """)

    st.error("""
    **Production Trap 2 (Critical): Accidental Stop Word Filtering in Legal or Medical RAG**  
    * **Failure Mode**: Applying legacy NLTK stopword filtering before embedding patient clinical notes or legal contracts.
    * **Impact**: Stripping negations (*not*, *no*, *never*) inverses diagnoses (*"patient not allergic"* becomes *"patient allergic"*).
    * **Remediation**: Strictly ban stop word removal in modern LLM data pipelines. Foundation embeddings require full syntactic context.
    """)

with t_col2:
    st.warning("""
    **Production Trap 3 (Caution): Unicode NFD Explosion in Multilingual Processing**  
    * **Failure Mode**: Ingesting text normalized with NFD (Canonical Decomposition) into tokenizers optimized for NFC.
    * **Impact**: Drastically increases token fertility on accented European, Indic, and Asian scripts; a single word explodes from 1 token to 4 tokens.
    * **Remediation**: Always enforce `unicodedata.normalize('NFC', text)` during data curation and ingestion pipelines.
    """)

    st.warning("""
    **Production Trap 4 (Caution): Naive Period Splitting in Sentence Segmentation**  
    * **Failure Mode**: Splitting sentences on raw period delimiters (`text.split('.')`).
    * **Impact**: Erroneously splits on decimal numbers (`$4.50`), abbreviations (`Dr. Smith`, `Inc.`), and URLs, fragmenting context chunks.
    * **Remediation**: Use abbreviation-aware regular expressions or neural sentence segmenters (such as Spacy Sentencizer) that protect decimals and known abbreviations.
    """)

st.divider()

st.subheader("Core Text Processing Flashcards")

flash_choice = st.selectbox(
    "Select Concept Flashcard to Review:",
    [
        "Unicode NFC Normalization",
        "Lossless Pre-Tokenization",
        "Over-Stemming vs Under-Stemming",
        "Sentence Boundary Disambiguation (SBD)"
    ]
)

if flash_choice == "Unicode NFC Normalization":
    st.info("""
    **Unicode NFC Normalization**: Canonical Decomposition followed by Canonical Composition. Merges base characters and combining accents into single composite code points, ensuring identical byte representations regardless of platform.
    """)
elif flash_choice == "Lossless Pre-Tokenization":
    st.info("""
    **Lossless Pre-Tokenization**: The modern LLM principle of preserving 100% of casing, punctuation, whitespace, and stopwords, allowing deep self-attention layers to resolve syntactic and semantic relationships without information loss.
    """)
elif flash_choice == "Over-Stemming vs Under-Stemming":
    st.info("""
    **Over-Stemming vs Under-Stemming**: Over-stemming occurs when words with different meanings are incorrectly reduced to the same root stem (e.g. 'universe' and 'university'). Under-stemming occurs when words with the same root fail to merge (e.g. 'run' and 'ran').
    """)
else:
    st.info("""
    **Sentence Boundary Disambiguation (SBD)**: Identifying where sentences end and begin while distinguishing true terminators from decimal numbers, acronyms, and abbreviations.
    """)

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of text processing foundations for Large Language Models:")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    cb1 = st.checkbox("I can articulate why modern foundation models reject classical stopword removal and lowercasing.", value=True)
    cb2 = st.checkbox("I understand the differences between Unicode NFC, NFD, and NFKC normalization forms.", value=True)
    cb3 = st.checkbox("I can explain why punctuation marks serve as essential attention routing anchors in transformers.", value=True)

with cb_col2:
    cb4 = st.checkbox("I understand the failure modes of heuristic stemming (over-stemming and under-stemming).", value=True)
    cb5 = st.checkbox("I know why lemmatization is computationally expensive and why subword BPE replaces it.", value=True)
    cb6 = st.checkbox("I can implement an abbreviation-aware sentence segmentation algorithm that protects decimal numbers.", value=True)

if all([cb1, cb2, cb3, cb4, cb5, cb6]):
    st.success("🎯 **Lesson 2.1 Complete! You have demonstrated comprehensive mastery of text processing foundations and lossless curation for Large Language Models.**")
