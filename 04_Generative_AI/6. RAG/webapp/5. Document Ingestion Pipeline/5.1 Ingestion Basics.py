import streamlit as st
import pandas as pd
import re
import html
import unicodedata

st.set_page_config(
    page_title="5.1 Ingestion Basics - RAG Master Curriculum",
    page_icon="🧹",
    layout="wide"
)

# Title & Overview
st.title("5.1 Ingestion Basics: The Data Prep Pipeline, Text Normalization, & Content Sanitization")
st.caption("Chapter 5: Document Ingestion Pipeline | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
In classical data science, the golden rule is *"Garbage In, Garbage Out"*. 
In Retrieval-Augmented Generation, the reality is even more severe: **"Garbage In, Hallucination Out"**. 
If raw documents ingested into your vector database are polluted with broken PDF hyphenation line-breaks, 
HTML entities (`&amp;`, `&#39;`), zero-width unicode spaces, and repetitive tracking disclaimers, 
embedding models generate noisy, corrupted vectors that fail to match clean user queries.
The **Document Ingestion Pipeline** is the asynchronous ETL (Extract, Transform, Load) backbone of RAG: 
extracting raw data, stripping boilerplate noise, repairing line-wrap artifacts, 
normalizing unicode, and enriching documents with structured metadata.
This lesson covers the fundamentals of text sanitization, normalization, and data preparation.
""")

st.divider()

# Section 1: Anatomy of the Ingestion Pipeline
st.header("1. The Five Stages of the Ingestion ETL Pipeline", divider="rainbow")
st.markdown("""
Every document entering a production RAG knowledge store must pass through five structured preparation phases:
""")

ingestion_stages = pd.DataFrame([
    {
        "Ingestion Stage": "1. Extraction & Ingestion",
        "Primary Operation": "Extracts raw text strings, byte buffers, and embedded images from source files (S3, Confluence, PDFs).",
        "Artifact Produced": "Raw unstructured text string with source file path."
    },
    {
        "Ingestion Stage": "2. Boilerplate & Clutter Stripping",
        "Primary Operation": "Removes website cookie notices, navbars, recurring PDF running headers/footers, and disclaimers.",
        "Artifact Produced": "Filtered article body containing only core informative prose."
    },
    {
        "Ingestion Stage": "3. Text Normalization",
        "Primary Operation": "Fixes broken PDF word hyphenations, unescapes HTML entities, and standardizes unicode (NFKC).",
        "Artifact Produced": "Clean linguistic text matching human reading flow."
    },
    {
        "Ingestion Stage": "4. Whitespace & Layout Standardization",
        "Primary Operation": "Collapses redundant tabs, fixes carriage returns (`\\r\\n` to `\\n`), and preserves markdown table grids.",
        "Artifact Produced": "Uniform layout with standard paragraph separators (`\\n\\n`)."
    },
    {
        "Ingestion Stage": "5. Metadata Extraction & Provenance",
        "Primary Operation": "Computes SHA-256 content hashes, parses author/timestamps, and attaches security access tags.",
        "Artifact Produced": "Standardized `Document(page_content, metadata)` object ready for chunking."
    }
])

st.dataframe(ingestion_stages, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Cost of Dirty Data in Vector Spaces")
    st.markdown("""
    * **Broken PDF Hyphenation**: Words split across line wraps (*`"dis- \n count"`*) are tokenized as separate tokens (`dis`, `-`, `count`). A vector search for `"discount"` will often miss the chunk completely!
    * **Unicode Ghost Characters**: Zero-width spaces (`\u200b`) and non-breaking spaces (`\xa0`) create out-of-vocabulary tokens that bloat context windows and distort attention.
    * **HTML Clutter**: Unstripped tags (`<div class="header-v2">`) inject high-frequency code tokens that drag the semantic vector away from business meaning toward generic web code.
    """)

with col2:
    st.subheader("Ingestion ETL Pipeline Diagram")
    etl_chart = """
    digraph IngestionETL {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin=0.15];
        edge [fontname="Helvetica", fontsize=9];

        Raw [label="1. Raw Documents (PDF / HTML / S3)", fillcolor="#F3F4F6", color="#4B5563"];
        Strip [label="2. Boilerplate Stripping\\n(Remove headers, footers, navbars)", fillcolor="#EDE9FE", color="#6D28D9"];
        Norm [label="3. Text Normalization\\n(Fix line hyphens, unescape HTML)", fillcolor="#FEF3C7", color="#D97706"];
        White [label="4. Whitespace Clean\\n(Collapse redundant spaces)", fillcolor="#DCFCE7", color="#16A34A"];
        Meta [label="5. Metadata Enrichment\\n(SHA-256 hash, timestamps, RBAC)", fillcolor="#E0F2FE", color="#0284C7"];
        Ready [label="Clean Document Objects\\n(Ready for Semantic Chunking)", fillcolor="#D1FAE5", color="#059669"];

        Raw -> Strip -> Norm -> White -> Meta -> Ready;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(etl_chart, use_container_width=False)

st.divider()

# Section 2: Production Code: Pure Python Text Sanitization Pipeline
st.header("2. Production Code: Pure Python Text Sanitization Pipeline", divider="rainbow")
st.markdown("""
Here is a complete, production-grade text sanitization pipe that repairs hyphenation, cleans unicode, unescapes HTML, and collapses whitespace:
""")

st.code("""
import re
import html
import unicodedata
from typing import Dict, Any

class ProductionTextSanitizer:
    def __init__(self):
        # Regex to fix hyphenated line breaks in PDFs (e.g. "con-\\n tractor" -> "contractor")
        self.hyphen_regex = re.compile(r"(\\w+)-\\s*\\n\\s*(\\w+)")
        # Regex to remove non-printable control characters (except newline and tab)
        self.control_char_regex = re.compile(r"[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]")

    def repair_pdf_hyphens(self, text: str) -> str:
        \"\"\"Recombines words split across page/line boundaries.\"\"\"
        return self.hyphen_regex.sub(r"\\1\\2", text)

    def normalize_unicode_and_html(self, text: str) -> str:
        \"\"\"Standardizes unicode to NFKC form and unescapes HTML entities.\"\"\"
        # Unescape HTML (&amp; -> &, &quot; -> ")
        text = html.unescape(text)
        # Normalize unicode (converts special ligature glyphs fi -> f i, full-width chars)
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width spaces (\u200b) and non-breaking spaces (\xa0)
        text = text.replace("\\u200b", "").replace("\\xa0", " ")
        return text

    def clean_whitespace(self, text: str) -> str:
        \"\"\"Collapses redundant spaces while preserving paragraph breaks.\"\"\"
        # Replace carriage returns
        text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
        # Collapse multiple horizontal spaces into single space
        text = re.sub(r"[ \\t]+", " ", text)
        # Collapse 3 or more consecutive newlines into double newline (paragraph break)
        text = re.sub(r"\\n{3,}", "\\n\\n", text)
        return text.strip()

    def sanitize(self, raw_text: str) -> Dict[str, Any]:
        initial_chars = len(raw_text)
        
        # Pipeline execution
        t1 = self.control_char_regex.sub("", raw_text)
        t2 = self.repair_pdf_hyphens(t1)
        t3 = self.normalize_unicode_and_html(t2)
        cleaned_text = self.clean_whitespace(t3)
        
        return {
            "cleaned_text": cleaned_text,
            "original_char_count": initial_chars,
            "cleaned_char_count": len(cleaned_text),
            "characters_removed": initial_chars - len(cleaned_text)
        }

# Demonstration Execution
sanitizer = ProductionTextSanitizer()
dirty_sample = \"\"\"
The em- 
ployee agrees to receive a 15&percnt; dis-   \\t
count on all hard-  \\n ware purchases&period;\\xa0\\xa0\\xa0
\\n\\n\\n\\n
Contact support&colon; admin&commat;corp&period;com
\"\"\"

res = sanitizer.sanitize(dirty_sample)
print("--- SANITIZED TEXT ---")
print(res["cleaned_text"])
print(f"\\nChar Reduction: {res['characters_removed']} bytes of noise removed.")
""", language="python")

st.divider()

# Section 3: Interactive Text Sanitization Playground
st.header("3. Interactive Text Sanitization Playground", divider="rainbow")
st.markdown("""
Test the sanitization pipeline on common real-world text extraction bugs:
""")

dirty_presets = {
    "Broken PDF Line-Wrap Hyphenation": "The company pro- \n vides compre- \n hensive health- \n care coverage for all full-time staff.",
    "Cluttered HTML Entities & Unicode Noise": "Terms &amp; Conditions&colon; All sales are final&period;\xa0\xa0No refunds on custom orders&excl;\u200b",
    "Excessive Whitespace & Carriage Returns": "Quarterly\r\n\r\n\r\n\r\nRevenue:\t\t\t$4.5 Billion.\r\n\r\n\r\nOperating Margin:\t\t24%."
}

selected_preset = st.selectbox("Select Dirty Sample:", list(dirty_presets.keys()))
input_dirty = st.text_area("Dirty Input Text:", dirty_presets[selected_preset], height=80)

if st.button("Execute Normalization & Sanitization Pipeline", type="primary"):
    # Apply cleaning logic
    t = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", input_dirty)
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t).replace("\xa0", " ").replace("\u200b", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    cleaned = re.sub(r"\n{3,}", "\n\n", t).strip()

    st.subheader("Sanitization Results")
    c1, c2 = st.columns(2)
    c1.metric("Original Size", f"{len(input_dirty)} chars")
    c2.metric("Cleaned Size", f"{len(cleaned)} chars")

    st.success("✅ **Normalized Linguistic Text**:")
    st.code(cleaned, language="text")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Ingestion Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does broken word hyphenation in PDFs degrade vector embedding quality?**
    * **Answer**: In PDFs, words that wrap onto the next line are split with a trailing hyphen and newline (e.g. `"inter- \n est"`). Embedding tokenizers operate on subwords; encountering `"inter-"` and `"est"` creates two arbitrary fragments instead of the single token `"interest"`. When a user later searches for *"interest rates"*, the vector similarity distance is significantly penalized because the primary keyword was mathematically fragmented in vector space.
    
    **Q2: What is Unicode NFKC normalization and why is it essential for RAG?**
    * **Answer**: In Unicode, visually identical characters can have completely different byte representations. For example, the ligature glyph `"ﬁ"` (single character U+FB01) vs the two characters `"f"` and `"i"` (U+0066 U+0069), or full-width Asian characters vs standard ASCII. **NFKC (Normalization Form KC)** decomposes compatibility characters into canonical equivalents, ensuring that text extracted from diverse international PDFs maps to the exact same subword tokens used by modern LLM tokenizers.
    
    **Q3: What is the risk of aggressively lowercasing all text during document ingestion?**
    * **Answer**: In classical NLP (bag-of-words, TF-IDF), lowercasing was standard practice. In modern transformer architectures (GPT-4, Claude), lowercasing destroys vital semantic signals: distinguishing between the acronym **AIDS** (the disease) and the word **aids** (help), or **APPLE** (the company) and **apple** (the fruit). Modern BPE tokenizers are cased; text should preserve original capitalization.
    """)

with traps_tab:
    st.error("""
    **Trap 1: Stripping all newline characters (`\n`) into a single continuous run of text**  
    Replacing all newlines with spaces destroys table alignments, header boundaries, and paragraph separations. An LLM reading a flattened table cannot tell where rows begin or end! Always preserve double newlines (`\n\n`) as structural paragraph breaks.
    """)
    st.error("""
    **Trap 2: Neglecting deduplication prior to vector embedding**  
    If duplicate drafts of a company handbook exist in your S3 bucket, ingesting both creates redundant vector clusters. When queried, your Top-5 results will be filled with 5 identical copies of the same paragraph! Always compute content hashes for deduplication.
    """)

with flashcards_tab:
    st.info("**Text Normalization**: Standardizing character representations, repairing broken hyphenations, and unescaping markup entities prior to chunking.")
    st.info("**Unicode NFKC**: Canonical and compatibility unicode normalization standard converting visually equivalent ligatures and glyphs into standard ASCII characters.")
    st.info("**Boilerplate Removal**: Stripping repetitive, non-informative elements (headers, footers, navbars, cookie banners) prior to embedding generation.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
checklist = [
    "I understand the five core stages of the Document Ingestion ETL lifecycle.",
    "I know how broken line-wrap hyphenation degrades vector embedding precision.",
    "I can explain why Unicode NFKC normalization is required for multi-source text.",
    "I know why modern RAG pipelines preserve case rather than lowercasing all text.",
    "I can implement a production text sanitization pipeline in pure Python."
]

for item in checklist:
    st.checkbox(item, value=False)
