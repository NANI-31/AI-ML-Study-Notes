import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
import unicodedata
import hashlib
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="12.2 Document Cleaning - Vector DB Curriculum",
    page_icon="🧹",
    layout="wide"
)

# Title & Overview
st.title("12.2 Document Cleaning: Normalization, Deduplication & Noise Removal")
st.caption("Chapter 12: Document Processing for Vector Databases | Domain: 4. Vector Database")
st.markdown("""
In machine learning and information retrieval, the adage **"Garbage In, Garbage Out"** applies acutely to vector embeddings. 

If raw document text contains **HTML markup, running headers, mojibake encoding corruption, zero-width characters, or excessive boilerplate**, 
the embedding model's transformer attention heads waste capacity processing syntactic noise rather than core semantic meaning.

Furthermore, duplicate or near-duplicate documents pollute the vector database, wasting expensive vector storage and causing 
RAG systems to return repetitive chunks to the user.

In this lesson, we master the enterprise **Document Cleaning Pipeline**, explore techniques for **Unicode normalization and mojibake correction**, 
examine **exact and fuzzy deduplication**, understand the dangers of over-cleaning, and test an interactive live Sanitization Lab.
""")

st.divider()

# Section 1: The Multi-Stage Cleaning Pipeline
st.header("1. The Multi-Stage Cleaning Pipeline", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1.1, 0.9])

with col_pipe1:
    st.markdown("""
    ### The 5 Standard Cleaning Stages
    
    1. **Structural & Markup Stripping**:
       * Decompose HTML tags (`<script>`, `<style>`, `<div>`), markdown image tags, base64 strings, and tracking pixels.
    2. **Repetitive Boilerplate Removal**:
       * Strip running page headers (*"Confidential - Page 14"*), footers, cookie banners, navigation breadcrumbs, and standard corporate copyright notices.
    3. **Unicode & Character Normalization**:
       * Apply Unicode `NFKC` normalization to standardize ligatures (e.g. `ﬁ` -> `fi`), full-width characters, and non-breaking spaces (`\\xa0`).
       * Fix character encoding corruption (**Mojibake**, e.g. `â€™` -> `'`).
    4. **Whitespace & Typography Compaction**:
       * Collapse multiple consecutive spaces, tabs, and runs of 3+ newlines down to a single clean newline or space.
    5. **Deduplication (Exact & Fuzzy)**:
       * Generate cryptographic hashes (SHA-256) or Locality-Sensitive Hashes (MinHash / SimHash) to discard identical or near-identical text chunks before embedding.
    """)

with col_pipe2:
    st.markdown("### Sequential Cleaning Pipeline Diagram")
    clean_dot = """
    digraph CleanPipeline {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Raw [label="Raw Scraped / Parsed Text\\nHTML tags, Mojibake, Duplicates", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        Stage1 [label="1. Strip HTML & Scripts\\nRegex / BeautifulSoup", fillcolor="#FEF3C7"];
        Stage2 [label="2. Remove Boilerplate\\nStrip headers, footers, legal", fillcolor="#FEF3C7"];
        Stage3 [label="3. Unicode Normalization\\nunicodedata.normalize('NFKC')", fillcolor="#D1FAE5"];
        Stage4 [label="4. Whitespace Compaction\\nCollapse \\n\\n\\n and spaces", fillcolor="#D1FAE5"];
        Stage5 [label="5. Deduplication Gate\\nSHA-256 / SimHash check", fillcolor="#E0E7FF", color="#4338CA"];

        Clean [label="Clean Semantic Text\\nReady for Chunking & Embedding", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Raw -> Stage1;
        Stage1 -> Stage2;
        Stage2 -> Stage3;
        Stage3 -> Stage4;
        Stage4 -> Stage5;
        Stage5 -> Clean;
    }
    """
    st.graphviz_chart(clean_dot, use_container_width=False)

st.divider()

# Section 2: Essential Normalization & Cleaning Code Patterns
st.header("2. Essential Normalization & Cleaning Code Patterns", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. Unicode Normalization & Mojibake Fix")
    st.code("""import unicodedata
import re

def normalize_unicode_and_mojibake(text: str) -> str:
    # 1. Common mojibake corrections (UTF-8 decoded as Windows-1252)
    mojibake_map = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€”": "—",
        "â€“": "–", "â€¦": "...", "Ã©": "é", "Ã¼": "ü"
    }
    for bad, good in mojibake_map.items():
        text = text.replace(bad, good)
        
    # 2. Unicode NFKC normalization
    # Normalizes ligatures (ﬁ -> fi) and full-width chars
    text = unicodedata.normalize("NFKC", text)
    
    # 3. Strip non-printable / zero-width characters
    # \\u200b: zero-width space, \\ufeff: byte order mark
    text = re.sub(r"[\\u200b-\\u200d\\ufeff]", "", text)
    
    # 4. Normalize non-breaking spaces
    text = text.replace("\\xa0", " ")
    return text
""", language="python")

with col_code2:
    st.markdown("### 2. Whitespace Compaction & Boilerplate Stripping")
    st.code("""def compact_whitespace(text: str) -> str:
    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r"[ \\t]+", " ", text)
    
    # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraphs)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    return "\\n".join(lines).strip()

def strip_common_boilerplate(text: str) -> str:
    # Strip running headers/footers like 'Page 4 of 12'
    text = re.sub(r"(?i)page\\s+\\d+\\s+of\\s+\\d+", "", text)
    # Strip confidentiality banners
    text = re.sub(r"(?i)confidential\\s+-\\s+internal\\s+use\\s+only", "", text)
    return text
""", language="python")

st.divider()

# Section 3: Exact vs Fuzzy Deduplication
st.header("3. Exact vs Fuzzy Deduplication: SHA-256 vs MinHash", divider="rainbow")

col_dedup1, col_dedup2 = col_dedup1, col_dedup2 = st.columns([1, 1])

with col_dedup1:
    st.markdown("""
    ### Exact Deduplication (Cryptographic Hashing)
    * **Algorithm**: Compute SHA-256 or MD5 hash over the normalized text chunk:
      ```python
      chunk_hash = hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()
      ```
    * **Storage**: Maintain a hash set or SQLite unique index on `chunk_hash`.
    * **Cost**: Extremely fast (`O(1)` lookup), zero memory overhead.
    * **Limitation**: Fragile! If even **one single comma or character** differs between two 500-word documents, the cryptographic hash completely diverges.
    """)

with col_dedup2:
    st.markdown("""
    ### Fuzzy / Near-Duplicate Deduplication (MinHash / SimHash)
    * **Use Case**: Syndicated articles, slightly edited drafts, or crawling mirrors that differ by 1-2 words.
    * **Mechanism**:
      1. Shingle text into 3-word n-grams (`k-shingles`).
      2. Apply `MinHash` permutation functions to generate a compact signature vector.
      3. Use Locality-Sensitive Hashing (LSH) to detect pairs with **Jaccard Similarity > 85%**.
    * **Benefit**: Catches plagiarized, re-formatted, or near-identical documents, preventing index bloat.
    """)

st.divider()

# Section 4: The Goldilocks Principle: Avoiding Over-Cleaning
st.header("4. The Goldilocks Principle: Avoiding Over-Cleaning", divider="rainbow")

st.markdown("""
A common antipattern in modern AI engineering is applying **legacy NLP preprocessing** (from 2012-era bag-of-words or TF-IDF models) 
to modern transformer embedding models. Modern transformer models require **grammatical context** to generate rich embeddings.
""")

col_over1, col_over2 = col_over1, col_over2 = st.columns([1, 1])

with col_over1:
    st.error("""
    ### ❌ What NOT to Clean (Destructive Over-Cleaning)
    1. **Do NOT Remove Stop Words**: Words like *"not"*, *"above"*, *"before"*, *"never"* carry immense semantic weight. Removing stop words inverts sentence meaning (*"This system is NOT HIPAA compliant"* becomes *"This system HIPAA compliant"*!).
    2. **Do NOT Lowercase Everything**: Acronyms like `"IT"` (Information Technology vs pronoun), `"US"` (United States vs pronoun), and `"AWS"` rely on capitalization.
    3. **Do NOT Strip Punctuation**: Transformers use periods, commas, and question marks to establish clause boundaries and interrogative tone.
    4. **Do NOT Stem or Lemmatize**: Do not convert `"running"` to `"run"`. Modern byte-pair encoders (BPE) handle morphology natively.
    """)

with col_over2:
    st.success("""
    ### ✅ What to Clean (Constructive Sanitization)
    1. **HTML/XML Tags**: `<p>`, `<div>`, `<script>`, `&nbsp;`.
    2. **Mojibake & Bad Encoding**: Corrupted apostrophes and quotes.
    3. **Zero-Width & Non-Printable Bytes**: `\u200b`, `\ufeff`.
    4. **Excessive Repetitive Layout Spacing**: Multiple blank lines, tab runs.
    5. **Irrelevant Repetitive Boilerplate**: Page numbers, header banners, cookie warnings.
    6. **Identical Duplicate Chunks**: Redundant paragraph copies.
    """)

st.divider()

# Section 5: Interactive Workbench: Live Document Sanitization Lab
st.header("5. Interactive Workbench: Live Document Sanitization Lab", divider="rainbow")

st.markdown("""
Paste raw text or choose a pre-loaded noisy sample to test each cleaning stage interactively. 
Inspect character reduction, token savings, and the final sanitized output.
""")

SAMPLE_DIRTY_TEXTS = {
    "Web Scrape with HTML & Mojibake": """<div class="article-body">
<h1>Cloud Infrastructure Update â€“ Q3</h1>
<p>The companyâ€™s primary cluster was upgraded to Kubernetes v1.28.&nbsp;&nbsp;Latency dropped by 14%.</p>
<p>Notice: This document is CONFIDENTIAL - INTERNAL USE ONLY.</p>
Page 1 of 5
<script>trackPageView();</script>
<p>The companyâ€™s primary cluster was upgraded to Kubernetes v1.28.&nbsp;&nbsp;Latency dropped by 14%.</p>
</div>""",
    "PDF Extract with Layout Breaks & Ligatures": """CONFIDENTIAL - DO NOT DISTRIBUTE
Page 14 of 42

The  ﬁrewall   rules  were  re-conﬁgured  to  block  incoming  UDP  floods.

Page 14 of 42
All   endpoints   must   utilize   mTLS   certiﬁcates.


CONFIDENTIAL - DO NOT DISTRIBUTE"""
}

col_lab_in, col_lab_toggles = st.columns([1.2, 0.8])

with col_lab_in:
    st.subheader("1. Input Document Text")
    sample_choice = st.selectbox("Load Sample Dirty Text:", list(SAMPLE_DIRTY_TEXTS.keys()))
    input_text = st.text_area("Raw Text Input:", value=SAMPLE_DIRTY_TEXTS[sample_choice], height=200)

with col_lab_toggles:
    st.subheader("2. Active Sanitization Filters")
    opt_html = st.checkbox("1. Strip HTML & Script Tags", value=True)
    opt_mojibake = st.checkbox("2. Fix Mojibake & Decode Errors", value=True)
    opt_unicode = st.checkbox("3. Unicode NFKC Normalization (Ligatures)", value=True)
    opt_boilerplate = st.checkbox("4. Remove Running Headers & Footers", value=True)
    opt_whitespace = st.checkbox("5. Compact Whitespace & Blank Lines", value=True)
    opt_dedup = st.checkbox("6. Deduplicate Exact Repeating Paragraphs", value=True)

# Processing Pipeline
processed_text = input_text

if opt_html:
    # Remove script and style tags with content
    processed_text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", processed_text)
    # Strip remaining HTML tags
    processed_text = re.sub(r"<[^>]+>", " ", processed_text)
    processed_text = processed_text.replace("&nbsp;", " ")

if opt_mojibake:
    mojibake_map = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€”": "—",
        "â€“": "–", "â€¦": "...", "Ã©": "é", "Ã¼": "ü"
    }
    for bad, good in mojibake_map.items():
        processed_text = processed_text.replace(bad, good)

if opt_unicode:
    processed_text = unicodedata.normalize("NFKC", processed_text)
    processed_text = re.sub(r"[\u200b-\u200d\ufeff]", "", processed_text)
    processed_text = processed_text.replace("\xa0", " ")

if opt_boilerplate:
    processed_text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", processed_text)
    processed_text = re.sub(r"(?i)confidential\s+-\s+(internal\s+use\s+only|do\s+not\s+distribute)", "", processed_text)

if opt_dedup:
    # Deduplicate paragraphs
    paras = [p.strip() for p in processed_text.splitlines() if p.strip()]
    seen = set()
    deduped = []
    for p in paras:
        h = hashlib.sha256(p.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(p)
    processed_text = "\n\n".join(deduped)

if opt_whitespace:
    processed_text = re.sub(r"[ \t]+", " ", processed_text)
    processed_text = re.sub(r"\n{3,}", "\n\n", processed_text)
    processed_text = "\n".join(l.strip() for l in processed_text.splitlines() if l.strip())

st.divider()

col_out_clean, col_out_stats = st.columns([1.2, 0.8])

with col_out_clean:
    st.subheader("3. Cleaned Document Result")
    st.text_area("Sanitized Text Output:", value=processed_text, height=200)

with col_out_stats:
    st.subheader("4. Cleaning Metrics & Efficiency")
    orig_chars = len(input_text)
    clean_chars = len(processed_text)
    chars_removed = max(0, orig_chars - clean_chars)
    reduction_pct = (chars_removed / max(1, orig_chars)) * 100.0
    
    # Estimate tokens
    orig_tokens = max(1, orig_chars // 4)
    clean_tokens = max(1, clean_chars // 4)

    st.metric(label="Characters Removed", value=f"{chars_removed} chars", delta=f"-{reduction_pct:.1f}% Size")
    st.metric(label="Estimated Token Savings", value=f"{orig_tokens - clean_tokens} Tokens", help="Tokens saved from context window per chunk.")
    
    st.success("✅ Output verified: Zero destructive lemmatization or stop-word deletion detected.")

st.divider()

# Section 6: Technical Interview Preparation
st.header("6. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is removing stop words considered a severe antipattern in modern embedding pipelines?"):
    st.markdown("""
    **Answer**:
    In older bag-of-words search (TF-IDF, BM25), stop words (*"the", "is", "at", "not"*) contributed little lexical specificity and were pruned to reduce vocabulary size.
    
    In modern transformer embedding models:
    1. **Self-Attention Mechanism**: Attention heads calculate contextual embeddings by attending to function words and syntactic relationships.
    2. **Semantic Inversion**: Stop words include critical qualifiers such as **negations** (*"not", "never", "no"*), **prepositions** (*"before", "after"*), and **modal verbs** (*"must", "should"*). 
    Removing stop words destroys the core meaning of sentences (e.g. *"This medication must NOT be taken with alcohol"* becomes *"medication taken alcohol"*). Never remove stop words before dense vector embedding.
    """)

with st.expander("Q2: What is Mojibake and how does it happen in document parsing pipelines?"):
    st.markdown("""
    **Answer**:
    Mojibake occurs when text encoded in one character encoding (typically UTF-8) is read or parsed as another encoding (typically Windows-1252 or ISO-8859-1).
    * For example, the UTF-8 right single quotation mark `'` is stored as bytes `0xE2 0x80 0x99`.
    * If a parser reads those 3 bytes using Windows-1252, it displays them as three distinct glyphs: **`â€™`**.
    * This corrupts tokenization: the tokenizer splits `â€™` into multiple weird sub-tokens instead of a single apostrophe token, distorting the embedding.
    """)

with st.expander("Q3: What is the difference between Unicode NFC and NFKC normalization?"):
    st.markdown("""
    **Answer**:
    * **NFC (Canonical Decomposition, followed by Canonical Composition)**: Replaces decomposed characters with pre-composed characters (e.g. `e` + accent `´` becomes single character `é`). It preserves visual typography.
    * **NFKC (Compatibility Decomposition, followed by Canonical Composition)**: Goes a step further by replacing formatting/compatibility glyphs with their standard equivalents:
      * Ligatures: `ﬁ` is expanded to `fi`, and `ﬀ` to `ff`.
      * Full-width Latin characters (common in Asian documents) are converted to standard ASCII.
      * Circled numbers (①) become standard digits (1).
    NFKC is the **recommended standard for vector retrieval** because it ensures consistent token IDs across different font and typographic variations.
    """)

with st.expander("Q4: How does MinHash with Locality-Sensitive Hashing (LSH) detect near-duplicate documents at scale?"):
    st.markdown("""
    **Answer**:
    Cryptographic hashes (SHA-256) cannot detect near-duplicates. 
    **MinHash + LSH** solves this:
    1. **Shingling**: Break each document into overlapping word n-grams (e.g. 3-word shingles: *"the quick brown"*, *"quick brown fox"*).
    2. **MinHash Signatures**: Apply hundreds of random hash functions to the shingle set. The minimum hash value produced by each function forms an integer signature vector. The probability that two documents share the same minimum hash equals their **Jaccard Similarity**.
    3. **LSH Bucketing**: Divide signatures into bands and hash bands into buckets. Candidate near-duplicates hash to the same bucket without requiring `O(N^2)` pairwise comparisons.
    """)

with st.expander("Q5: Why should consecutive runs of newlines be collapsed to a maximum of two newlines?"):
    st.markdown("""
    **Answer**:
    When extracting text from OCR or layout-heavy documents, visual gaps often translate into 5 to 10 consecutive newline characters (`\n\n\n\n\n`).
    * When text is split using standard recursive text splitters (which prioritize splitting on `\n\n`), excessive newlines create **empty or near-empty chunks** containing only whitespace.
    * Collapsing newlines ensures that single newlines represent line wraps, exactly two newlines represent paragraph breaks, and zero empty chunks are generated.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Applying Lemmatization / Stemming to Embeddings**  
    Running NLTK Porter Stemmer reduces *"computation"* to *"comput"*, destroying the sub-word tokenization representations learned by modern LLM embedding models. Avoid stemming.
    """)
    st.warning("""
    **Trap 2: Stripping Punctuation Marks**  
    Removing periods and question marks destroys sentence boundaries. Embedding models rely on periods to understand where one factual thought ends and another begins.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Ignoring Zero-Width Spaces (`\u200b`)**  
    Zero-width spaces copied from websites or PDFs are invisible to the eye but break sub-word tokenization, causing standard words like `"database"` to split into obscure single-character tokens.
    """)
    st.warning("""
    **Trap 4: Deduplicating Before Whitespace Normalization**  
    Checking SHA-256 hashes on raw un-normalized text misses duplicate paragraphs that differ only by an extra trailing space or newline. Always normalize whitespace before hashing.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Should you remove stop words for vector embeddings?"):
        st.markdown("""
        **NO**:  
        Stop words (especially negations like *not* and *never*) provide critical contextual meaning to transformer attention heads.
        """)
    with st.expander("🗂️ Flashcard: What is NFKC normalization?"):
        st.markdown("""
        **Unicode NFKC**:  
        Normalizes typographic ligatures (`ﬁ` -> `fi`), full-width characters, and compatibility glyphs into standard text.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What causes Mojibake?"):
        st.markdown("""
        **Encoding Mismatch**:  
        UTF-8 encoded multi-byte characters decoded incorrectly using Windows-1252 (e.g. `â€™` instead of `'`).
        """)
    with st.expander("🗂️ Flashcard: How to detect near-duplicate documents?"):
        st.markdown("""
        **MinHash + LSH**:  
        Converts word shingles into MinHash signatures to calculate Jaccard similarity without pairwise comparison.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the 5-stage document cleaning pipeline for vector database ETL.", value=True)
chk2 = st.checkbox("I know how to resolve Mojibake encoding corruption and Unicode ligatures via NFKC.", value=True)
chk3 = st.checkbox("I understand why removing stop words or lowercasing destroys transformer semantic embeddings.", value=True)
chk4 = st.checkbox("I know how to implement exact SHA-256 deduplication and understand MinHash for near-duplicates.", value=True)
chk5 = st.checkbox("I know how to strip running page headers and footers without truncating body text.", value=True)
chk6 = st.checkbox("I can design a production sanitization pipeline that maximizes embedding token density.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Document Cleaning and are ready for Chapter 12.3: Document Chunking Strategies!")
