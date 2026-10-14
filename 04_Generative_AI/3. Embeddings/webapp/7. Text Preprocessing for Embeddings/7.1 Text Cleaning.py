import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import unicodedata
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="7.1 Text Cleaning - Embeddings Curriculum",
    page_icon="🧹",
    layout="wide"
)

# Title & Overview
st.title("7.1 Text Cleaning for Embeddings: Whitespace Normalization, Unicode Forms, & Casing Tradeoffs")
st.caption("Chapter 7: Text Preprocessing for Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
A common misconception in modern NLP is that *"Transformer embedding models are so powerful that raw text can be passed directly without cleaning."*
In reality, embedding quality is exceptionally vulnerable to textual noise:
* Inconsistent Unicode representations (e.g. `e` + combining acute vs precomposed `é`) produce completely different subword token IDs, fragmenting semantic coordinates.
* Hidden control characters (`\\x00`, `\\r`, zero-width spaces `\\u200b`) generate unexpected out-of-vocabulary or unknown token embeddings.
* Indiscriminate lowercasing can destroy critical named-entity distinctions (e.g., *"Apple"* the company vs *"apple"* the fruit; *"US"* the country vs *"us"* the pronoun).

This lesson explores rigorous text sanitization specifically tailored for dense neural embeddings: whitespace consolidation, Unicode normalization (NFC/NFKC), casing strategy, and deduplication.
""")

st.divider()

# Section 1: Core Text Cleaning Foundations
st.header("1. Core Text Cleaning Operations for Embedding Pipelines", divider="rainbow")

col_cln1, col_cln2 = st.columns([1, 1])

with col_cln1:
    st.markdown("""
    ### Essential Sanitization Operations:
    * **1. Whitespace Consolidation**:
      * Tab characters, consecutive spaces, and varied newline combinations (`\\r\\n`, `\\n`, `\\r`) waste positional tokens and confuse attention.
      * *Action*: Replace all whitespace runs with single spaces; normalize paragraph breaks to double newlines `\\n\\n`.
    * **2. Unicode Normalization (NFC / NFKC)**:
      * Characters can be represented in multiple byte sequences (e.g. ligature `ﬁ` vs `f` + `i`; compatibility characters).
      * `unicodedata.normalize('NFKC', text)` converts compatibility characters into standard canonical equivalents.
    * **3. Non-Printable & Control Character Removal**:
      * Invisible bytes (e.g. zero-width joiners `\\u200d`, byte order marks `\\ufeff`) pollute tokenizer inputs.
      * *Action*: Filter characters using regex `[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F-\\x9F]`.
    * **4. The Casing Dilemma: To Lowercase or Not?**:
      * In classical bag-of-words (TF-IDF), lowercasing was mandatory to merge vocabulary.
      * In modern cased transformer models (e.g. `all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `text-embedding-3`), **DO NOT LOWERCASE UNCONDITIONALLY**! Lowercasing destroys acronyms (*"AIDS"*, *"NASA"*), stock tickers (*"AMZN"*), and proper names.
    """)

with col_cln2:
    st.markdown("### The Cleaning Pipeline Flow")
    clean_dot = """
    digraph CleanFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawIn [label="Dirty Raw Text\\n'Apple Inc.\\u00a0\\u200b announced\\n\\nnew\\tMacBook'\\n(Mixed Unicode & Tabs)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        UnicodeStep [label="1. Unicode Normalization\\nNFKC canonical form decomposition", fillcolor="#FEF3C7"];
        
        ControlStep [label="2. Control Byte Stripping\\nRemove zero-width spaces & non-printables", fillcolor="#FDE68A"];
        
        SpaceStep [label="3. Whitespace Consolidation\\nRegex collapse spaces & newlines", fillcolor="#D1FAE5"];
        
        CaseCheck [label="4. Context-Preserving Case Check\\nRetain acronyms & proper nouns", fillcolor="#EDE9FE", color="#8B5CF6"];
        
        CleanOut [label="Clean Ingestion-Ready Text\\n'Apple Inc. announced new MacBook'\\nOptimal Tokenizer Economy", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        RawIn -> UnicodeStep -> ControlStep -> SpaceStep -> CaseCheck -> CleanOut;
    }
    """
    st.graphviz_chart(clean_dot, use_container_width=False)

st.markdown("### Unicode Normalization Standards Comparison")
st.markdown("""
| Normalization Form | Name | Mechanism | Best Use Case | Example Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **NFC** | Canonical Decomposition + Canonical Composition | Composes characters into single precomposed code points | General text storage, database indexing | `e` + `´` -> `é` |
| **NFD** | Canonical Decomposition | Splits precomposed characters into base letter + accents | Accent stripping, linguistic phonetic search | `é` -> `e` + `´` |
| **NFKC** | Compatibility Decomposition + Canonical Composition | Collapses visual compatibility glyphs into standard equivalents | **Recommended for Embedding Pipelines** | `ﬁ` (ligature) -> `fi`, `²` -> `2` |
| **NFKD** | Compatibility Decomposition | Decomposes compatibility characters without recombining | Deep search tokenization | `ﬁ` -> `f` + `i` |
""")

st.divider()

# Section 2: Production Code: Robust Text Cleaner
st.header("2. Production Implementation: Enterprise Text Sanitizer", divider="rainbow")

st.markdown("""
The production utility below implements a comprehensive cleaning pipeline tailored specifically for embedding generation. It preserves critical punctuation and casing while sanitizing formatting noise and normalizing Unicode.
""")

st.code(r'''
import unicodedata
import re
from typing import str

class EmbeddingTextCleaner:
    """
    Production-grade text sanitizer for embedding models.
    Normalizes Unicode, eliminates invisible control bytes, consolidates whitespace,
    and deduplicates repetitive whitespace artifacts without destroying semantic casing.
    """
    def __init__(self, apply_nfkc: bool = True, preserve_newlines: bool = True):
        self.apply_nfkc = apply_nfkc
        self.preserve_newlines = preserve_newlines
        # Regex to match non-printable ASCII control characters except \n and \t
        self.control_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b-\u200f]')

    def clean(self, text: str) -> str:
        """Executes full cleaning chain on input text string."""
        if not text:
            return ""

        # 1. Unicode Normalization (NFKC)
        if self.apply_nfkc:
            text = unicodedata.normalize('NFKC', text)

        # 2. Strip non-printable and zero-width control characters
        text = self.control_chars_pattern.sub('', text)

        # 3. Replace non-breaking spaces with standard space
        text = text.replace('\xa0', ' ').replace('\u202f', ' ')

        # 4. Whitespace and newline consolidation
        if self.preserve_newlines:
            # Normalize carriage returns
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            # Collapse multiple newlines (more than 2) into 2
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Collapse multiple spaces or tabs into a single space
            text = re.sub(r'[ \t]{2,}', ' ', text)
        else:
            text = re.sub(r'\s+', ' ', text)

        return text.strip()
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveTextCleaner:
    def __init__(self):
        self.control_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b-\u200f]')

    def clean(self, text: str, do_nfkc: bool, strip_control: bool, collapse_space: bool, lowercase: bool) -> Tuple[str, Dict[str, Any]]:
        orig_len = len(text)
        cleaned = text
        
        if do_nfkc:
            cleaned = unicodedata.normalize('NFKC', cleaned)
            
        if strip_control:
            cleaned = self.control_pattern.sub('', cleaned)
            cleaned = cleaned.replace('\xa0', ' ')
            
        if collapse_space:
            cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
            cleaned = re.sub(r'\s{2,}', ' ', cleaned)
            
        if lowercase:
            cleaned = cleaned.lower()
            
        cleaned = cleaned.strip()
        new_len = len(cleaned)
        
        stats = {
            "original_length": orig_len,
            "cleaned_length": new_len,
            "bytes_removed": orig_len - new_len,
            "char_reduction_pct": ((orig_len - new_len) / max(orig_len, 1)) * 100.0
        }
        return cleaned, stats

st.divider()

# Section 3: Interactive Workbench: Text Sanitization Laboratory
st.header("3. Interactive Workbench: Text Sanitization & Unicode Laboratory", divider="rainbow")

st.markdown("""
Test how raw text containing hidden control characters, ligatures, and messy spacing is transformed by different cleaning passes.
Toggle sanitization steps to observe character reductions and inspect the resulting clean text.
""")

sample_dirty_text = "Apple\u00a0\u200bInc. announced   the\tnew \uFB01nancial   report\n\n\n\nfor Q4.   Revenue: $89.5B.\r\nNASA confirmed   satellite\u202ftelemetry."

cleaner_sim = InteractiveTextCleaner()

col_cln_l, col_cln_r = st.columns([1, 1])

with col_cln_l:
    st.subheader("1. Configure Cleaning Passes")
    user_input = st.text_area("Raw Input Text (Contains Ligatures, Non-Breaking Spaces, & Tabs):", value=sample_dirty_text, height=140)
    
    do_nfkc = st.checkbox("1. Apply NFKC Unicode Normalization (converts ligatures like \uFB01 -> fi)", value=True)
    strip_control = st.checkbox("2. Strip Zero-Width Spaces (\\u200b) & Non-Printable Bytes", value=True)
    collapse_space = st.checkbox("3. Collapse Multiple Spaces & Newlines into Single Spacing", value=True)
    do_lower = st.checkbox("4. Force Lowercase (Caution: May destroy acronyms like NASA / Apple)", value=False)

with col_cln_r:
    st.subheader("2. Sanitization Inspection")
    clean_result, stats = cleaner_sim.clean(user_input, do_nfkc, strip_control, collapse_space, do_lower)
    
    st.markdown("#### Cleaned Output Text:")
    st.text_area("Output Preview:", value=clean_result, height=100, disabled=True)

    m1, m2 = st.columns(2)
    m1.metric("Original Characters", f"{stats['original_length']}")
    m2.metric("Cleaned Characters", f"{stats['cleaned_length']}", delta=f"-{stats['bytes_removed']} chars", delta_color="normal")

    m3, m4 = st.columns(2)
    m3.metric("Reduction Percentage", f"{stats['char_reduction_pct']:.1f}%")
    m4.metric("Casing Mode", "Lowercased (Uncased)" if do_lower else "Preserved Casing (Cased)")

    if do_lower:
        st.warning("⚠️ **Entity Loss Warning**: Lowercasing converted `NASA` -> `nasa` and `Apple` -> `apple`. Modern cased embedding models may lose distinction between corporate entities and generic dictionary nouns.")
    else:
        st.success("✅ **Semantic Preservation**: Acronyms (NASA), currency symbols, and proper nouns remain intact for high-fidelity cased embeddings.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Cleaning Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why should you NOT lowercase text when feeding modern embedding models like text-embedding-3 or BGE?**
    * **Answer**:
      * Classical NLP models (Word2Vec, TF-IDF) frequently used uncased vocabularies because capitalization artificially split word counts (e.g. `The` vs `the`).
      * Modern Transformer embedding models are **cased**: their tokenizers contain distinct subwords for capitalized and lowercase tokens.
      * Capitalization conveys vital semantic signals:
        1. **Proper Nouns & Named Entities**: `Apple` (technology company) vs `apple` (fruit).
        2. **Acronyms**: `AIDS` (medical disease) vs `aids` (assists); `US` (United States) vs `us` (pronoun).
      * Lowercasing text collapses these distinctions, measurably degrading entity-specific retrieval.

    **Q2: What is Unicode NFKC normalization and why is it preferred over NFC for embeddings?**
    * **Answer**:
      * **NFC (Canonical Composition)** combines base characters with diacritics (e.g. `e` + `´` becomes `é`), but leaves compatibility characters untouched.
      * **NFKC (Compatibility Decomposition + Canonical Composition)** decomposes typographic compatibility variants into standard characters:
        * Ligatures: `ﬁ` becomes `fi`.
        * Formatting glyphs: `²` becomes `2`, full-width roman numerals become standard Latin letters.
      * Because tokenizers are rarely trained on obscure ligature code points, NFKC ensures that words with ligatures tokenize into familiar subword tokens rather than unknown `<unk>` tokens.

    **Q3: How do zero-width spaces (\u200b) and non-breaking spaces (\u00a0) corrupt tokenization?**
    * **Answer**:
      * Zero-width characters often leak into text from copy-pasting rich text editors or web scrapers.
      * Tokenizers treat a zero-width space as a distinct unicode character. When attached to a word (e.g. `\u200bword`), the tokenizer fails to match `word` in its vocabulary and breaks it into fragmented byte-level tokens, substantially altering the resulting embedding vector.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Stripping All Punctuation via Regex**:
      * *Issue*: Applying regex `re.sub(r'[^a-zA-Z0-9]', ' ', text)` on raw inputs.
      * *Consequence*: Destroys question marks (`?`), hyphens in compound words (`state-of-the-art`), decimal points in numbers (`$10.5M`), and code syntax. Transformers rely heavily on punctuation for clause boundary attention!
      * *Fix*: Never strip semantic punctuation; only remove non-printable control characters.

    * **Trap 2: Normalizing Text at Ingestion but Passing Dirty Text at Query Time**:
      * *Issue*: Cleaning documents with NFKC, but allowing users to submit queries with unnormalized ligatures or weird whitespaces.
      * *Consequence*: Disjoint tokenization between queries and documents introduces an artificial semantic gap.
      * *Fix*: Wrap cleaning logic into a shared library used identically across both indexing and query pipelines.

    * **Trap 3: Aggressive Stopword Removal**:
      * *Issue*: Removing stopwords (`and`, `the`, `is`, `not`) before passing text to Sentence Transformers.
      * *Consequence*: Completely inverts sentence meaning! *"Drug is safe"* and *"Drug is not safe"* become identical after stopword removal. Transformers need stopwords for contextual attention.
      * *Fix*: Never remove stopwords before feeding transformer embedding models.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: NFKC Normalization**
      * *Definition*: Unicode standard that normalizes compatibility characters (ligatures, superscript numbers) into standard canonical characters.
    * **Flashcard 2: Cased Embedding Models**
      * *Definition*: Embedding models whose vocabularies distinguish between capitalized and lowercase tokens to preserve named entity and acronym semantics.
    * **Flashcard 3: Zero-Width Characters**
      * *Definition*: Invisible non-printing Unicode code points (\u200b, \ufeff) that corrupt tokenization if not sanitized.
    * **Flashcard 4: Stopword Preservation**
      * *Definition*: The requirement to retain function words (not, but, is) so transformer self-attention can resolve negation and logical structure.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Text Cleaning for Embeddings:

- [ ] I understand why lowercasing text damages retrieval quality in cased embedding models.
- [ ] I know the difference between NFC and NFKC Unicode normalization.
- [ ] I can articulate why stopword removal is detrimental to transformer self-attention.
- [ ] I understand how zero-width and control characters distort subword tokenization.
- [ ] I know how to implement an automated text sanitization utility in Python.
""")
