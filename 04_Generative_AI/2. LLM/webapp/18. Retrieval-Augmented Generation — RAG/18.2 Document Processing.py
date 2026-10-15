import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.2 Document Processing - LLM Engineering",
    page_icon="📄",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class DocumentProcessor:
    """
    Self-contained production preprocessing engine for enterprise RAG:
    - Unicode normalization and de-hyphenation
    - HTML boilerplate and tag stripping
    - Table serialization into Markdown format
    - PII redaction (email, SSN, credit cards)
    - Metadata enrichment
    """

    @staticmethod
    def dehyphenate_text(text: str) -> str:
        """Fixes words split across line breaks by hyphens."""
        # Matches word ending with hyphen followed by newline and next word segment
        return re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)

    @staticmethod
    def strip_html_boilerplate(html_text: str) -> str:
        """Removes script tags, style blocks, and HTML tags while preserving line breaks."""
        clean = re.sub(r'<(script|style|nav|footer|header)[^>]*>.*?</\1>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<br\s*/?>', '\n', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</p>', '\n\n', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', '', clean)
        return clean

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalizes irregular spaces, tabs, and excessive newlines."""
        # Replace multiple horizontal spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3 or more newlines with two newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def redact_pii(text: str) -> Tuple[str, int]:
        """Redacts emails, SSNs, and credit card numbers for compliance."""
        redactions = 0
        
        # Email redaction
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        matches = len(re.findall(email_pattern, text))
        redactions += matches
        text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
        
        # SSN redaction (XXX-XX-XXXX)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        matches = len(re.findall(ssn_pattern, text))
        redactions += matches
        text = re.sub(ssn_pattern, '[REDACTED_SSN]', text)
        
        # Credit card redaction (16 digits)
        cc_pattern = r'\b(?:\d{4}[ -]?){3}\d{4}\b'
        matches = len(re.findall(cc_pattern, text))
        redactions += matches
        text = re.sub(cc_pattern, '[REDACTED_CARD]', text)
        
        return text, redactions

    @staticmethod
    def serialize_table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
        """Converts extracted table cell matrix into clean Markdown table format."""
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, separator_line] + data_lines)


# --- HEADER & INTRODUCTION ---
st.title("18.2 Document Processing: Multi-Format Ingestion, Layout Analysis, & Metadata Enrichment")
st.caption("Chapter 18: Retrieval-Augmented Generation — RAG | Domain: 2. Large Language Models")

st.markdown("""
A Retrieval-Augmented Generation system is only as good as the raw text it ingests. In production enterprise environments, 
information does not arrive as clean, formatted UTF-8 strings; it is trapped inside **multi-column PDFs**, 
**nested Word documents (DOCX)**, **scanned TIFF images**, and **bloated HTML pages**.

Naively extracting text using primitive file readers strips away semantic hierarchy, scrambles two-column reading orders, 
flattens critical financial tables into unstructured gibberish, and discards document metadata.

This lesson covers the data engineering discipline of document processing: multi-format extraction, vision-based layout analysis, 
table serialization into Markdown, text cleaning transformations, and metadata enrichment.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Multi-Format Document Ingestion & Extraction Pipeline", divider="rainbow")

st.markdown("""
How modern enterprise RAG ingestion platforms ingest heterogeneous file formats and convert them into standardized, chunk-ready Markdown:
""")

pipeline_dot = """
digraph DocumentPipeline {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_raw {
        label="Raw Enterprise Formats";
        color="#1E293B";
        fontcolor="#38BDF8";
        PDF [label="Digital / Scanned PDF\\n(Multi-Column, Tables)", fillcolor="#1E293B"];
        DOCX [label="Word (.docx)\\n(XML Styles, Revisions)", fillcolor="#1E293B"];
        HTML [label="Web / HTML\\n(Navbars, Ads, Boilerplate)", fillcolor="#1E293B"];
        MD [label="Markdown (.md)\\n(Headers, Frontmatter)", fillcolor="#1E293B"];
    }

    subgraph cluster_extract {
        label="Format-Specific Extraction";
        color="#1E293B";
        fontcolor="#38BDF8";
        LayoutLM [label="Layout Analysis & OCR\\n(Marker / Surya / PyMuPDF)\\nPreserves Reading Order", fillcolor="#0F766E"];
        DocxParser [label="XML Style Parser\\n(python-docx)\\nMaps Heading1/2 Hierarchy", fillcolor="#0369A1"];
        Trafilatura [label="DOM Scraper\\n(Trafilatura / Readability)\\nStrips Nav & Boilerplate", fillcolor="#4338CA"];
    }

    subgraph cluster_clean {
        label="Sanitization & Normalization";
        color="#1E293B";
        fontcolor="#38BDF8";
        Sanitizer [label="Text Cleaning Engine\\nDe-hyphenation across lines\\nUnicode NFKC Normalization", fillcolor="#6D28D9"];
        PIIFilter [label="PII & Compliance Masking\\nRegex Masking (SSN, Cards)\\nGDPR & HIPAA Guardrails", fillcolor="#7F1D1D"];
    }

    subgraph cluster_output {
        label="Standardized Chunk Input";
        color="#1E293B";
        fontcolor="#38BDF8";
        StandardMD [label="Clean Markdown Document\\nWith Embedded Tables\\nand Rich Metadata Dict", fillcolor="#059669"];
    }

    PDF -> LayoutLM -> Sanitizer;
    DOCX -> DocxParser -> Sanitizer;
    HTML -> Trafilatura -> Sanitizer;
    MD -> Sanitizer;
    Sanitizer -> PIIFilter -> StandardMD;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(pipeline_dot, use_container_width=True)

st.markdown("""
### Enterprise Document Formats Comparison Matrix
""")

format_matrix = [
    {
        "Format": "PDF (Digital)",
        "Core Technical Challenge": "Coordinate-based glyph layout; multi-column reading order; scrambled tables",
        "Recommended Parsers": "PyMuPDF (fitz), pdfplumber, Marker, LlamaParse",
        "Target Output": "Structured Markdown with Markdown tables"
    },
    {
        "Format": "PDF (Scanned)",
        "Core Technical Challenge": "Image skew, low DPI, handwriting, lack of text layer",
        "Recommended Parsers": "PaddleOCR, Tesseract, Surya, AWS Textract",
        "Target Output": "OCR text with bounding box coordinates"
    },
    {
        "Format": "Word (.docx)",
        "Core Technical Challenge": "Zipped XML structures; tracked changes; custom paragraph styles",
        "Recommended Parsers": "python-docx, Unstructured, Pandoc",
        "Target Output": "Hierarchical Markdown preserving H1/H2 headings"
    },
    {
        "Format": "HTML / Web",
        "Core Technical Challenge": "JavaScript rendering; navigation menus; cookie banners; ads",
        "Recommended Parsers": "Trafilatura, Readability-lxml, BeautifulSoup4",
        "Target Output": "Article body text stripped of DOM clutter"
    },
    {
        "Format": "Markdown (.md)",
        "Core Technical Challenge": "Preserving YAML frontmatter; code blocks; tables",
        "Recommended Parsers": "Python-Markdown, Mistune",
        "Target Output": "Native clean text ready for hierarchical chunking"
    }
]
st.dataframe(pd.DataFrame(format_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - DOCUMENT LOADING ---
st.header("2. Document Loading: Unified Loaders, Lazy Streaming, & Directory Crawlers", divider="rainbow")

st.markdown("""
At the entry point of the ingestion pipeline, **Document Loaders** abstract disparate file system protocols 
into a unified in-memory data structure:

```python
class Document:
    page_content: str
    metadata: Dict[str, Any] # source, page, author, created_at, access_tier
```

### Lazy Loading vs Eager Loading in Large Corpora
- **Eager Loading (`load()`)**: Reads all 50,000 documents into RAM simultaneously. For an enterprise legal repository, 
  this consumes 80+ GB of memory, triggering Python out-of-memory crashes.
- **Lazy Loading (`lazy_load()`)**: Implements Python generators (`yield Document`) that stream documents one file at a time. 
  Memory consumption remains constant at under 200 MB regardless of whether you are indexing 100 documents or 10,000,000 documents.
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    **Metadata Enrichment at Load Time**:
    Always inject file-level metadata during the initial read:
    - `source_file`: Absolute path or URL
    - `doc_id`: Unique cryptographic hash (e.g., SHA-256 of file bytes)
    - `created_timestamp`: File creation or modification date
    - `access_control_level`: Security clearance classification
    """)
with col2:
    st.warning("""
    **MIME-Type Sniffing**:
    Never rely solely on file extensions (e.g. `.pdf`). Enterprise users frequently save HTML error pages or text files 
    with `.pdf` or `.docx` extensions. Use magic byte sniffing (`python-magic`) to determine the true binary format before launching a parser.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - PDF PROCESSING ---
st.header("3. PDF Processing: Layout Analysis, Reading Order, OCR, & Table Serialization", divider="rainbow")

st.markdown("""
The **Portable Document Format (PDF)** was created in 1993 as a digital paper specification. It was engineered 
to render vector coordinates identically across printers, **not to preserve semantic reading order**.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. The Multi-Column Reading Order Problem")
    st.markdown("""
    Digital PDFs store text as disjointed coordinate snippets: `(x, y, "text")`.
    - Naive extractors sort text by vertical coordinate `y` alone.
    - In a two-column academic paper, reading horizontally across a `y`-slice weaves the two columns together:
      ```text
      Naive Output: "The proposed model achieves [Col 1] while in baseline testing [Col 2] 94.2% accuracy on benchmarks [Col 1] earlier models failed [Col 2]..."
      ```
    - The resulting chunk is semantically corrupted and completely misinforms the embedding model.
    - **Solution**: Modern parsers (like **Marker** and **LayoutLM**) execute computer vision object detection to segment pages into column bounding boxes before reading top-to-bottom within each column.
    """)

with col2:
    st.subheader("2. Table Extraction & Serialization")
    st.markdown("""
    Tables contain the highest density of critical enterprise information (e.g. quarterly revenues, dosages, pricing tiers).
    - If a PDF parser strips table grid lines, row coordinates become unaligned strings of floating numbers.
    - **Solution**: Table extraction models (**pdfplumber**, **Table-Transformer**) identify cell bounding boxes and serialize the matrix into **Markdown Table format**:
      ```markdown
      | Metric | Q3 2024 | Q3 2025 |
      | --- | --- | --- |
      | Revenue | $4.2M | $5.8M |
      ```
    - LLMs have seen millions of Markdown tables during pretraining; keeping tables in Markdown preserves exact row-column associations!
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - HTML PROCESSING ---
st.header("4. HTML Processing: Boilerplate Removal, DOM Traversal, & Main Body Extraction", divider="rainbow")

st.markdown("""
Extracting documentation from company wikis (Confluence, Notion) or public websites requires stripping away 
the vast ocean of web noise that surrounds actual article content.

In a typical web page:
- **80% of tokens** consist of navigation menus, header bars, cookie popups, footer disclaimers, and tracking scripts.
- **20% of tokens** represent the actual informative body text.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Why Raw HTML Destroys Vector Search
    If you embed raw HTML strings:
    - The top tokens are `<div>`, `class="nav-link"`, `href="..."`, and `<script>`.
    - Every page from your company portal will share the exact same 1,000 tokens of navigation and footer text!
    - At search time, every query matches the generic navigation boilerplate, causing search precision to collapse.
    """)
with col2:
    st.markdown("""
    ### Main Body Extraction Heuristics
    Modern text scrapers (such as **Trafilatura** and **Readability.js**) use text density algorithms:
    1. Parse the DOM tree into a node hierarchy.
    2. Calculate the text-to-tag ratio for each container element.
    3. Prune `<nav>`, `<header>`, `<footer>`, `<aside>`, and `<form>` subtrees.
    4. Convert headings (`<h1>` to `<h6>`) and lists (`<ul>`, `<ol>`) into Markdown, discarding CSS styles and JavaScript.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - WORD DOCUMENTS ---
st.header("5. Word Documents (DOCX): XML Structures, Tracked Changes, & Style Hierarchies", divider="rainbow")

st.markdown("""
Microsoft Word documents (`.docx`) are zipped packages containing an internal XML structure 
(`word/document.xml`, `word/styles.xml`).

### Key DOCX Parsing Engineering Tasks
1. **Style-Driven Heading Mapping**: Unlike raw text, Word documents contain explicit paragraph style tags:
   - `<w:pStyle w:val="Heading1"/>` -> Map directly to `# Heading 1`
   - `<w:pStyle w:val="Heading2"/>` -> Map directly to `## Heading 2`
   - Preserving this structure allows downstream chunkers to split on true document sections rather than arbitrary character counts.
2. **Tracked Changes & Comments**:
   - In enterprise legal documents, tracked changes often contain deleted draft text or unaccepted edits.
   - A naive XML scraper will extract both the deleted clause AND the replacement clause, creating contradictory legal text!
   - Production parsers must explicitly accept or reject revisions based on ingestion configuration before extracting body text.
3. **Embedded Tables**: Word tables are cleanly defined in XML (`<w:tbl>`, `<w:tr>`, `<w:tc>`), making them trivial to extract into clean Markdown grids compared to coordinate-based PDFs.
""")

st.divider()

# --- SECTION 6: TOPIC 5 - MARKDOWN ---
st.header("6. Markdown: Structural Superiority, Frontmatter, & Hierarchy Preservation", divider="rainbow")

st.markdown("""
**Markdown** is the universal target representation for all document ingestion pipelines. 
Every production RAG architecture converts PDFs, Word documents, and HTML pages into clean Markdown before passing them to the chunker.

### Why Markdown is the Ideal RAG Format
- **Human & LLM Readability**: All frontier foundation models (GPT-4o, Llama 3, Claude 3.5) were pretrained on billions of lines of Markdown from GitHub, Wikipedia, and StackOverflow.
- **Zero Token Overhead**: Unlike HTML or XML, Markdown indicates hierarchy with single characters (`#`, `##`, `*`, `|`), minimizing token consumption in the context window.
- **YAML Frontmatter Support**: Supports structured document metadata at the very top of the file:
  ```markdown
  ---
  title: Annual Cybersecurity Audit
  department: InfoSec
  classification: Internal Confidental
  effective_date: 2026-03-15
  ---
  # Section 1: Executive Summary
  ...
  ```
  This frontmatter can be parsed immediately into the vector database metadata index for instant filtering!
""")

st.divider()

# --- SECTION 7: TOPIC 6 - TEXT CLEANING ---
st.header("7. Text Cleaning: Whitespace Normalization, Unicode De-hyphenation, & PII Redaction", divider="rainbow")

st.markdown("""
Even after extracting clean text, real-world documents contain low-level encoding artifacts and compliance hazards 
that must be sanitized:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown(r"""
    ### 1. De-Hyphenation Across Line Breaks
    In justified PDF text, long words are split across line breaks with a hyphen:
    ```text
    "Retrieval-aug-
    mented generation..."
    ```
    If not cleaned, the tokenizer splits this into `aug` and `mented`, destroying the semantic embedding for "augmented".
    - Regex fix: `re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)`

    ### 2. Unicode Normalization (NFKC)
    Documents frequently contain non-standard Unicode characters:
    - Curly quotes (`“`, `”`) vs straight quotes (`"`).
    - Non-breaking spaces (`\xa0`) vs standard space (` `).
    - Ligatures (`fi` encoded as a single character `\ufb01`).
    - Applying `unicodedata.normalize('NFKC', text)` ensures uniform character encoding.
    """)

with col2:
    st.markdown("""
    ### 3. PII (Personally Identifiable Information) Redaction
    Under GDPR, HIPAA, and CCPA compliance standards, sensitive customer or patient information must never be 
    embedded into vector databases or sent to external LLM APIs:
    - **Social Security Numbers (SSN)**: Redact to `[REDACTED_SSN]`
    - **Credit Card Numbers**: Redact to `[REDACTED_CARD]`
    - **Customer Email Addresses**: Redact to `[REDACTED_EMAIL]`
    - Using regular expressions or enterprise Named Entity Recognition (NER) models (e.g. Microsoft Presidio) before chunking guarantees strict regulatory compliance.
    """)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: Multi-Format Document Preprocessing & Cleaning Engine in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements a production document preprocessing pipeline. 
It demonstrates de-hyphenation, HTML boilerplate cleaning, table serialization, and regex PII redaction:
""")

code_block = r'''import re
import unicodedata
from typing import List, Tuple, Dict

class DocumentSanitizer:
    """
    Production-grade text cleaning and sanitization pipeline for enterprise RAG.
    Zero external dependencies beyond standard Python libraries.
    """
    
    @staticmethod
    def clean_text_pipeline(raw_text: str) -> Tuple[str, Dict[str, int]]:
        """
        Executes full sanitization pipeline:
        1. Unicode NFKC normalization
        2. Line-break de-hyphenation
        3. Whitespace normalization
        4. PII redaction
        """
        stats = {"pii_redacted": 0, "hyphens_fixed": 0}
        
        # 1. Unicode normalization (converts ligatures, non-breaking spaces)
        text = unicodedata.normalize("NFKC", raw_text)
        
        # 2. De-hyphenate words split across line breaks
        hyphen_matches = len(re.findall(r'(\b\w+)-\s*\n\s*(\w+\b)', text))
        stats["hyphens_fixed"] = hyphen_matches
        text = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)
        
        # 3. Normalize horizontal whitespace and excessive blank lines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 4. Redact PII (SSN, Email, Credit Cards)
        pii_patterns = [
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'\b(?:\d{4}[ -]?){3}\d{4}\b', '[REDACTED_CARD]')
        ]
        
        for pattern, replacement in pii_patterns:
            matches = len(re.findall(pattern, text))
            stats["pii_redacted"] += matches
            text = re.sub(pattern, replacement, text)
            
        return text.strip(), stats

    @staticmethod
    def html_to_markdown(html_snippet: str) -> str:
        """Strips scripts, navigation, and converts basic tags to clean text."""
        # Remove script and style tags
        clean = re.sub(r'<(script|style|nav|footer)[^>]*>.*?</\1>', '', html_snippet, flags=re.DOTALL | re.IGNORECASE)
        # Convert break tags and paragraphs
        clean = re.sub(r'<br\s*/?>', '\n', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</p>', '\n\n', clean, flags=re.IGNORECASE)
        # Remove remaining tags
        clean = re.sub(r'<[^>]+>', '', clean)
        return clean.strip()
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: Raw Document Sanitizer & Table Preservation Lab", divider="rainbow")

st.markdown("""
Test real-world document cleaning transformations, PII redactions, and table serializations live:
""")

sample_inputs = {
    "Dirty Multi-Column PDF Text": (
        "In our multi-cen-\n"
        "ter clinical study, pa-\n"
        "tients receiving the new com-\n"
        "pound showed marked im-\n"
        "provement.\n\n"
        "Contact primary investigator Dr. Sarah at sarah.jenkins@biotech-research.org or call office.\n"
        "Patient ID 440-21-9984 was admitted with baseline blood glucose 110 mg/dL."
    ),
    "Bloated HTML Web Page": (
        "<nav><a href='/home'>Home</a> | <a href='/products'>Products</a> | <a href='/login'>Login</a></nav>\n"
        "<div class='cookie-banner'><p>We use cookies. Click accept to continue.</p></div>\n"
        "<article>\n"
        "<h1>Q4 Financial Overview</h1>\n"
        "<p>Total revenue grew by 24% year-over-year, driven by enterprise cloud adoption.<br>Operating margins reached 32%.</p>\n"
        "</article>\n"
        "<footer><p>Copyright 2026 Acme Corp. All rights reserved.</p></footer>"
    )
}

wb_col1, wb_col2 = st.columns(2)
with wb_col1:
    selected_sample = st.selectbox("Select Dirty Input Sample", list(sample_inputs.keys()))
    raw_input_text = st.text_area("Raw Text Input", sample_inputs[selected_sample], height=180)

with wb_col2:
    st.markdown("**Sanitization Filters & Guardrails**")
    apply_dehyphen = st.checkbox("De-hyphenate Broken Line Breaks", value=True)
    apply_html_strip = st.checkbox("Strip HTML Tags & Boilerplate", value=True)
    apply_pii_redact = st.checkbox("Redact PII (Emails, SSNs, Cards)", value=True)
    apply_ws_norm = st.checkbox("Normalize Whitespace & Blank Lines", value=True)

# Execute interactive pipeline
processed_text = raw_input_text
total_redactions = 0

if apply_html_strip:
    processed_text = DocumentProcessor.strip_html_boilerplate(processed_text)
if apply_dehyphen:
    processed_text = DocumentProcessor.dehyphenate_text(processed_text)
if apply_pii_redact:
    processed_text, total_redactions = DocumentProcessor.redact_pii(processed_text)
if apply_ws_norm:
    processed_text = DocumentProcessor.normalize_whitespace(processed_text)

st.subheader("Sanitized Output & Token Efficiency")
m1, m2, m3 = st.columns(3)
m1.metric("Raw Characters", len(raw_input_text))
m2.metric("Cleaned Characters", len(processed_text), delta=f"{len(processed_text) - len(raw_input_text)} chars")
m3.metric("PII Items Redacted", total_redactions)

st.text_area("Sanitized Markdown Output", processed_text, height=160)

st.markdown("### Interactive Table Serialization Lab")
st.caption("Demonstrating how tabular cell matrices are transformed into LLM-friendly Markdown:")
table_headers = ["Product Tier", "Users Included", "Annual Cost", "SLA Guarantee"]
table_rows = [
    ["Starter", "Up to 10", "$1,200", "99.5%"],
    ["Professional", "Up to 100", "$9,600", "99.9%"],
    ["Enterprise", "Unlimited", "$45,000", "99.99%"]
]
serialized_table_md = DocumentProcessor.serialize_table_to_markdown(table_headers, table_rows)
st.code(serialized_table_md, language="markdown")

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why is extracting semantic text from digital PDFs fundamentally harder than from Word (.docx) or HTML?",
        "a": "Word documents (.docx) store semantic XML tag trees (such as <w:pStyle w:val='Heading1'/>), and HTML represents text inside hierarchical DOM elements (<h1>, <p>, <table>). In contrast, digital PDFs are visual coordinate print files that only store disjointed character placement operations: (x, y, 'glyph'). A PDF has no intrinsic concept of a paragraph, reading order, table cell, or header. Extracting coherent text requires layout analysis models to reconstruct column reading boundaries and table grids from scratch."
    },
    {
        "q": "What is the multi-column reading order problem in PDF parsing and how do modern layout models solve it?",
        "a": "When a digital PDF contains two or more columns, naive extractors sort text solely by vertical coordinate (y-axis). This interweaves the text of adjacent columns on the same horizontal plane, producing scrambled sentences. Modern parsers (such as Marker, LayoutLM, or Surya) use computer vision object detection models to identify column bounding boxes on the page image, sort the columns left-to-right, and then extract the text top-to-bottom within each distinct column box."
    },
    {
        "q": "Why should tabular data in documents be serialized into Markdown format rather than raw whitespace text?",
        "a": "If table gridlines are stripped, numerical data becomes a loose stream of floating values where the association between row headers, column headers, and data cells is lost. Modern LLMs were pretrained on vast quantities of Markdown tables from GitHub and Wikipedia. Serializing tables into standard Markdown ('| Header 1 | Header 2 |') preserves structural relationships, allowing the LLM to understand which numbers belong to which metrics."
    },
    {
        "q": "How does web page boilerplate (navbars, cookie popups, footers) destroy vector search accuracy?",
        "a": "In uncleaned web crawls, navigation bars and footer legal disclaimers can account for up to 80% of a page's tokens. Because every page on the company portal shares identical navigation tokens, embedding models cluster all pages close to each other in vector space. When a user asks a question, the navigation links create false semantic matches, crowding out actual informative content chunks."
    },
    {
        "q": "What is the importance of de-hyphenation across line breaks before chunking?",
        "a": "Justified text in books, legal briefs, and academic PDFs frequently splits long words across line breaks with a hyphen (e.g. 'trans- \n former'). If not repaired, subword tokenizers treat the two pieces as distinct tokens ('trans' and 'former' or fragmented subwords), destroying the semantic embedding of the term and causing zero-recall on keyword or vector searches for the word."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Memory Exhaustion via Eager Loading of Massive Document Batches**
    Using `loader.load()` to parse 50,000 PDF documents into a single Python list. Memory consumption explodes, triggering OS OOM kills. 
    Always use lazy streaming generators (`loader.lazy_load()`) that process and index one document at a time with fixed memory usage.
    """)
    st.warning("""
    **Trap 3: Ingesting Unaccepted Tracked Changes from Legal Word Documents**
    Scraping raw XML from `.docx` legal contracts without inspecting revision tags. Tracked changes often contain clauses marked for deletion. 
    Ingesting deleted clauses can cause the RAG system to quote invalid, voided contract terms.
    """)

with col_trap2:
    st.error("""
    **Trap 2: PII Ingestion Leading to Regulatory Compliance Violations**
    Ingesting raw employee records or customer support logs containing Social Security Numbers or credit card digits into public vector databases. 
    Under GDPR and HIPAA, storing unmasked PII in searchable vector stores is an immediate regulatory violation. Implement regex/NER redaction before embedding.
    """)
    st.warning("""
    **Trap 4: Over-Aggressive Regex Stripping of Programming Code or Math**
    Applying broad cleaning rules that strip tabs, indentation, or symbols like `<`, `>`, or `_`. In technical manuals, this destroys 
    Python code indentation and scrambles mathematical equations. Keep cleaning selective and domain-aware.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: PDF Nature"):
        st.markdown("**Why are PDFs hard to parse?**\\n\\nPDFs store visual coordinates (x, y, glyph), not semantic paragraphs or reading order.")
with fc2:
    with st.popover("Flashcard 2: Table Format"):
        st.markdown("**Best format for tables?**\\n\\nMarkdown tables (`| col | col |`), natively understood by LLM attention heads.")
with fc3:
    with st.popover("Flashcard 3: Lazy Load"):
        st.markdown("**Eager vs Lazy Loading?**\\n\\nLazy loading streams documents one-by-one via generators, keeping RAM consumption constant.")
with fc4:
    with st.popover("Flashcard 4: De-hyphenate"):
        st.markdown("**What is de-hyphenation?**\\n\\nRecombining words split across lines (`infor-\\nmation` -> `information`) before tokenization.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why digital PDFs lack semantic hierarchy and how layout analysis models reconstruct column reading orders.",
    "I know how to serialize extracted tables into Markdown grids to preserve row-column relationships for LLMs.",
    "I can explain why lazy loading is mandatory when processing enterprise document repositories exceeding thousands of files.",
    "I understand the necessity of boilerplate stripping (navbars, footers, scripts) for preventing vector cluster distortion.",
    "I can implement regex de-hyphenation across line breaks and explain why it protects tokenization fidelity.",
    "I understand how PII redaction (SSNs, cards, emails) must be applied prior to embedding for GDPR/HIPAA compliance."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of multi-format document processing, layout analysis, table serialization, and data sanitization.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
