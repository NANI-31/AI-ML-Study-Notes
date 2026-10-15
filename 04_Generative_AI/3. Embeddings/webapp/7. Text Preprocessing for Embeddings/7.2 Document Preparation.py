import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="7.2 Document Preparation - Embeddings Curriculum",
    page_icon="📑",
    layout="wide"
)

# Title & Overview
st.title("7.2 Document Preparation: Format Sanitization, PDF De-Noising, & Structural Extraction")
st.caption("Chapter 7: Text Preprocessing for Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In real-world enterprise RAG pipelines, text rarely arrives as clean, standalone sentences. Instead, it originates from **heterogeneous document formats**:
* **PDFs**: Full of repeated running headers/footers (*"Confidential - Page 14 of 50"*), mid-word line-wrap hyphens (*"con- / temporary"*), and scrambled multi-column text.
* **HTML Pages**: Contaminated with navigation menus, cookie consent banners, boilerplate scripts, and nested styling tags.
* **Markdown & Technical Docs**: Containing YAML frontmatter headers, code blocks, raw image URLs, and ASCII tables that require specialized linear representation.

If these formatting artifacts are left uncleaned, they pollute vector representations: documents cluster by their shared company footer or PDF boilerplate rather than their actual intellectual content!

This lesson explores techniques for preparing HTML, Markdown, and PDF documents for high-fidelity vector embedding.
""")

st.divider()

# Section 1: Format-Specific Ingestion Challenges
st.header("1. Format-Specific Ingestion & De-Noising Strategies", divider="rainbow")

col_doc1, col_doc2 = st.columns([1, 1])

with col_doc1:
    st.markdown("""
    ### 1. PDF Ingestion Pitfalls:
    * **Running Headers & Footers**: Every single page includes page numbers, legal disclaimers, or document titles. When embedded, these repeated tokens pull every page towards the same boilerplate vector!
      * *Remedy*: Identify recurring top/bottom line patterns across page streams and strip them via regex or coordinate bounding boxes.
    * **Hyphenated Line Breaks**: Words split across lines (`"trans- \\n formation"`) become two distinct, broken subwords (`trans` and `formation`).
      * *Remedy*: Regex de-hyphenation: `re.sub(r'(\\w+)-\\n(\\w+)', r'\\1\\2', text)`.
    * **Multi-Column Scrambling**: Naive text extractors read across columns horizontally, interleaving sentences from Column A and Column B.
      * *Remedy*: Use layout-aware extractors (e.g. `pymupdf` with column sorting, or OCR models with reading-order layout detection).

    ### 2. HTML Ingestion:
    * Strip `<script>`, `<style>`, `<nav>`, and `<footer>` tags prior to text extraction.
    * Convert structural tags (`<h1>` to `<h3>`, `<li>`) into readable Markdown equivalents rather than flattening all text into an undifferentiated block.

    ### 3. Markdown & Technical Documentation:
    * Strip YAML frontmatter headers (`--- ... ---`).
    * Linearize tables: Convert raw ASCII/markdown tables into natural language sentences (e.g. *"In 2023, Revenue was $10M"*) to improve embedding retrieval.
    """)

with col_doc2:
    st.markdown("### Document Ingestion & De-Noising Flow")
    doc_dot = """
    digraph DocPrepFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawPDF [label="Raw Document Stream\\nRepeated Headers, Footers, Hyphens, HTML tags", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        BoilerplateFilter [label="1. Boilerplate / Header Stripper\\nRemove repeated page stamps & navbars", fillcolor="#FEF3C7"];
        
        Dehyphen [label="2. Hyphenation Stitching\\n'algo- \\n rithm' -> 'algorithm'", fillcolor="#FDE68A"];
        
        MarkupNormalize [label="3. Markup Linearization\\nHTML to clean markdown; tabular synthesis", fillcolor="#D1FAE5"];
        
        PreparedDoc [label="Prepared Document Text\\nCohesive prose retaining structural hierarchy", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        RawPDF -> BoilerplateFilter -> Dehyphen -> MarkupNormalize -> PreparedDoc;
    }
    """
    st.graphviz_chart(doc_dot, use_container_width=False)

st.markdown("### Document Preprocessing Checklist")
st.markdown("""
| Source Format | Primary Artifacts | Cleansing Action | Risk If Ignored |
| :--- | :--- | :--- | :--- |
| **PDF** | Repeated headers, page numbers, split hyphens | Strip top/bottom 5% margin text; regex stitch hyphens | Boilerplate vector clustering; broken subwords |
| **HTML** | Navbars, cookie banners, script tags | Drop `<nav>`, `<header>`, `<footer>`, `<script>`; extract main tag | 40%+ token budget wasted on navigation links |
| **Markdown** | YAML frontmatter, raw image links | Strip YAML metadata; convert image captions to text | Syntactic metadata polluting semantic text |
| **Database Dumps** | Null values, raw JSON brackets | Flatten JSON into descriptive key-value sentences | Dense vector unable to interpret raw syntax braces |
""")

st.divider()

# Section 2: Production Code: Multi-Format Document Preprocessor
st.header("2. Production Implementation: Document Preprocessing Engine", divider="rainbow")

st.markdown("""
The production class below demonstrates modular document preparation for PDFs, HTML, and Markdown, including automated de-hyphenation, boilerplate header stripping, and tag removal.
""")

st.code(r'''
import re
from typing import List, Dict, Any

class DocumentPreprocessor:
    """
    Production-grade document sanitizer for diverse enterprise inputs.
    Cleans PDF artifacts (split hyphens, repeated page footers),
    HTML tags, and Markdown metadata before vectorization.
    """
    def __init__(self):
        # Regex for line-broken hyphenated words: word- \n word
        self.hyphen_pattern = re.compile(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)')
        # Regex for page numbers and running footers
        self.footer_pattern = re.compile(r'(?i)page\s+\d+\s+of\s+\d+|confidential\s+-\s+internal\s+use\s+only')

    def clean_pdf_text(self, text: str) -> str:
        """Removes hyphenated line breaks and repeated page footer stamps."""
        # Stitch split hyphenated words across lines
        text = self.hyphen_pattern.sub(r'\1\2', text)
        # Remove common running footer stamps
        text = self.footer_pattern.sub('', text)
        # Normalize redundant newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clean_html_text(self, html_text: str) -> str:
        """Strips HTML tags while preserving paragraph spacing."""
        # Strip script and style blocks entirely
        text = re.sub(r'(?is)<(script|style|nav|footer)[^>]*>.*?</\1>', ' ', html_text)
        # Replace block tags with newlines
        text = re.sub(r'(?i)<(p|div|h[1-6]|li)[^>]*>', '\n', text)
        # Strip remaining inline tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clean_markdown_text(self, md_text: str) -> str:
        """Strips YAML frontmatter headers and raw image URLs."""
        # Strip YAML frontmatter: --- ... ---
        text = re.sub(r'(?s)^---\s*\n.*?\n---\s*\n', '', md_text)
        # Strip markdown images: ![alt](url) -> keep alt text
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        return text.strip()
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveDocPreprocessor:
    def __init__(self):
        self.hyphen_pattern = re.compile(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)')
        self.footer_pattern = re.compile(r'(?i)page\s+\d+\s+of\s+\d+|confidential\s+-\s+internal\s+use\s+only')

    def clean_pdf(self, text: str, fix_hyphens: bool, strip_footers: bool) -> Tuple[str, Dict[str, Any]]:
        orig_words = len(text.split())
        cleaned = text
        if fix_hyphens:
            cleaned = self.hyphen_pattern.sub(r'\1\2', cleaned)
        if strip_footers:
            cleaned = self.footer_pattern.sub('', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        new_words = len(cleaned.split())
        return cleaned, {"orig_words": orig_words, "clean_words": new_words}

    def clean_html(self, html_text: str) -> Tuple[str, Dict[str, Any]]:
        orig_len = len(html_text)
        # Strip script, style, nav, footer
        text = re.sub(r'(?is)<(script|style|nav|footer)[^>]*>.*?</\1>', ' ', html_text)
        text = re.sub(r'(?i)<(p|div|h[1-6]|li)[^>]*>', '\n', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text, {"orig_chars": orig_len, "clean_chars": len(text)}

st.divider()

# Section 3: Interactive Workbench: Document Preparation Studio
st.header("3. Interactive Workbench: Multi-Format Document Preparation Studio", divider="rainbow")

st.markdown("""
Select a document format contaminated with real-world artifacts (PDF running footers and line breaks, or messy HTML tags).
Observe how the preprocessor sanitizes formatting noise into clean prose ready for tokenization.
""")

sample_pdf = """The deep learning model achieved state-of-the-art per-
formance on multi-lingual benchmarks.
CONFIDENTIAL - INTERNAL USE ONLY
Page 14 of 50
Its transformer architecture deploys self-atten-
tion to model long-range sequence dependencies across tokens.
CONFIDENTIAL - INTERNAL USE ONLY
Page 15 of 50"""

sample_html = """<nav><a href="/home">Home</a> | <a href="/docs">Docs</a></nav>
<div class="main-body">
  <h1>Attention Mechanisms</h1>
  <p>Self-attention calculates dynamic pairwise weights between tokens.</p>
  <script>console.log("analytics tracking loaded");</script>
  <p>This allows transformers to process sequences in parallel without recur-rence.</p>
</div>
<footer>Copyright 2026 AI Corp. All rights reserved.</footer>"""

prep_sim = InteractiveDocPreprocessor()

format_tab = st.radio("Choose Document Format to Prepare:", ["PDF Text Stream", "HTML Web Page"], horizontal=True)

col_in_doc, col_out_doc = st.columns([1, 1])

if format_tab == "PDF Text Stream":
    with col_in_doc:
        st.subheader("1. Raw PDF Text Stream")
        pdf_input = st.text_area("Input PDF Text (with split hyphens & page stamps):", value=sample_pdf, height=180)
        fix_hy = st.checkbox("Fix Line-Break Hyphens ('per- \\n formance' -> 'performance')", value=True)
        strip_ft = st.checkbox("Strip Repeated Running Footers & Page Stamps", value=True)

    with col_out_doc:
        st.subheader("2. Prepared Clean Text")
        clean_pdf_out, p_stats = prep_sim.clean_pdf(pdf_input, fix_hy, strip_ft)
        st.text_area("Sanitized PDF Output:", value=clean_pdf_out, height=140, disabled=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Original Word Count", f"{p_stats['orig_words']}")
        m2.metric("Clean Word Count", f"{p_stats['clean_words']}", delta=f"{p_stats['clean_words'] - p_stats['orig_words']} words")
        st.success("✅ **De-Hyphenation Complete**: Broken words stitched back together into single unified tokens!")

else:
    with col_in_doc:
        st.subheader("1. Raw HTML Markup Stream")
        html_input = st.text_area("Input HTML Markup (with navigation, scripts, & footer):", value=sample_html, height=180)

    with col_out_doc:
        st.subheader("2. Extracted Semantic Text")
        clean_html_out, h_stats = prep_sim.clean_html(html_input)
        st.text_area("Extracted Text Output:", value=clean_html_out, height=140, disabled=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Original Markup Chars", f"{h_stats['orig_chars']}")
        m2.metric("Clean Semantic Chars", f"{h_stats['clean_chars']}", delta=f"-{h_stats['orig_chars'] - h_stats['clean_chars']} chars", delta_color="normal")
        st.success("✅ **Markup Stripped**: Navigation links, analytics scripts, and footer copyright stripped completely!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Preparation Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why do repeated running headers and footers in PDF documents cause severe failure modes in vector retrieval?**
    * **Answer**:
      * In a 100-page document, the text *"Acme Corp Confidential - Q3 Financial Report - Page X"* appears 100 times.
      * When pages are chunked and embedded, this boilerplate text accounts for 15-20% of every chunk's token budget.
      * In vector space, all 100 chunks are strongly pulled toward each other along the coordinate axes corresponding to the boilerplate terms.
      * When a user searches for an unrelated topic (e.g. *"Acme quarterly budget"*), the boilerplate terms dominate cosine similarity, causing massive false-positive retrieval across unrelated pages.

    **Q2: How does mid-word hyphenation in PDFs corrupt subword tokenization?**
    * **Answer**:
      * In print layout, words are split with a hyphen at the right margin (e.g. `trans-` at end of line 1, `formation` at start of line 2).
      * Raw PDF text extractors extract this literally as `"trans- \n formation"`.
      * Instead of producing a single token `transformation`, the tokenizer generates three separate tokens: `trans`, `-`, and `formation`.
      * The resulting embedding loses the exact semantic meaning of the compound word, degrading keyword and dense retrieval.

    **Q3: How should tabular data in HTML or Markdown documents be prepared for embedding models?**
    * **Answer**:
      * Raw markdown table syntax (`| Year | Revenue | EBITDA |`) is poorly understood by standard sentence embedding models.
      * The transformer attention mechanism treats the vertical bars `|` and alignment dashes `---` as punctuation noise.
      * **Best Practice**: Linearize rows into natural descriptive sentences:
        *"In 2023, Acme Corp recorded a Revenue of $10.5M with an EBITDA of $2.1M."*
      * Sentence linearization produces embeddings that match conversational search queries with significantly higher cosine accuracy.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Reading Multi-Column PDFs Linearly**:
      * *Issue*: Extracting PDF text without column layout awareness, reading left-to-right across two columns.
      * *Consequence*: Interleaves alternating phrases from two completely different articles or paragraphs, rendering chunks completely incoherent.
      * *Fix*: Use layout-aware PDF parsers (e.g. PyMuPDF with column sorting, or MinerU/Marker).

    * **Trap 2: Stripping All Markdown Structure Blindly**:
      * *Issue*: Stripping `# Headers` and `- Bullet points` so that everything becomes a single unformatted string.
      * *Consequence*: Destroys the hierarchical topic structure that tells the model where one section ends and another begins.
      * *Fix*: Retain markdown heading hierarchy (`## Header`) to provide contextual anchor tokens.

    * **Trap 3: Preserving Raw Code Blocks in General Document Indexes**:
      * *Issue*: Embedding 500 lines of raw JSON or C++ stack traces using a standard natural-language embedding model.
      * *Consequence*: The model's vocabulary breaks code syntax into thousands of individual punctuation and bracket tokens, hitting token limits instantly.
      * *Fix*: Route code blocks to code-specific embedding models (e.g. `bge-code` or `voyage-code-2`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Running Boilerplate**
      * *Definition*: Repetitive headers, footers, or copyright banners that artificially cluster diverse chunks together if not stripped.
    * **Flashcard 2: Hyphen Stitching**
      * *Definition*: The preprocessing technique that detects and merges line-broken hyphenated words back into single coherent vocabulary terms.
    * **Flashcard 3: Table Linearization**
      * *Definition*: Converting structured row-column table cells into natural language prose sentences to maximize semantic retrieval matching.
    * **Flashcard 4: Layout-Aware Extraction**
      * *Definition*: PDF parsing that respects visual bounding boxes and reading order, preventing multi-column text interleaving.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Document Preparation:

- [ ] I can describe the primary failure modes caused by PDF running footers in vector indexes.
- [ ] I understand how line-broken hyphens corrupt subword tokenization and know how to fix them.
- [ ] I know why raw HTML navigation and script tags must be stripped before chunking.
- [ ] I understand how table linearization converts tabular data into high-affinity embedding text.
- [ ] I know why multi-column PDFs require layout-aware extraction algorithms.
""")
