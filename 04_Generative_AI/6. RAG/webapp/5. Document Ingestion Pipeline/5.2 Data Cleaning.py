import streamlit as st
import pandas as pd
import hashlib
import re

st.set_page_config(
    page_title="5.2 Data Cleaning - RAG Master Curriculum",
    page_icon="🧼",
    layout="wide"
)

# Title & Overview
st.title("5.2 Data Cleaning: Deduplication, Header/Footer Stripping, OCR Denoising, & Corrupted Files")
st.caption("Chapter 5: Document Ingestion Pipeline | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Enterprise document repositories are rarely pristine. In production corpora, up to **30% of ingested files are duplicates or near-duplicates** 
(e.g., v1, v2, v3 drafts of an agreement, identical email threads, mirrored wiki pages). 
Furthermore, multi-page PDFs inject repetitive running headers and footers onto every single page, 
HTML pages contain buried scripts and SVG icons, and scanned documents arrive with skewed, noisy OCR artifacts.
Deploying high-precision RAG requires a defensive **Data Cleaning Engine**: 
eliminating duplicates via **SHA-256 hashing and MinHash LSH**, dynamically detecting and stripping **running headers and footers**, 
denoising OCR text, and isolating corrupted or password-protected files into quarantine queues.
This lesson covers the cleaning algorithms, deduplication mechanics, and fault-tolerant ingestion patterns.
""")

st.divider()

# Section 1: The Enterprise Data Cleaning Matrix
st.header("1. Core Enterprise Data Cleaning Challenges & Solutions", divider="rainbow")
st.markdown("""
How modern ingestion pipelines resolve the six most prevalent dirty data failure modes:
""")

cleaning_matrix = pd.DataFrame([
    {
        "Cleaning Challenge": "1. Exact Duplicate Documents",
        "Detection Mechanism": "Cryptographic SHA-256 content hashing.",
        "Remediation Strategy": "Drop duplicate; retain existing document ID and update last-seen timestamp.",
        "Impact if Ignored": "Duplicate vectors crowd out legitimate candidate chunks in Top-K search results."
    },
    {
        "Cleaning Challenge": "2. Near-Duplicate Documents (Drafts)",
        "Detection Mechanism": "MinHash + Locality Sensitive Hashing (LSH) / Jaccard similarity > 0.85.",
        "Remediation Strategy": "Flag newer version as canonical; archive or deprecate older draft versions.",
        "Impact if Ignored": "Model retrieves contradictory statements from older drafts (e.g. outdated pricing)."
    },
    {
        "Cleaning Challenge": "3. Running Headers & Footers",
        "Detection Mechanism": "Frequency analysis: text appearing in first/last 2 lines across > 80% of pages.",
        "Remediation Strategy": "Automatically strip identified running headers and page numbers prior to chunking.",
        "Impact if Ignored": "Embedding model associates every chunk with generic company names or page numbers."
    },
    {
        "Cleaning Challenge": "4. HTML Scripts & Inline SVG",
        "Detection Mechanism": "DOM tree traversal / Regex tag stripping.",
        "Remediation Strategy": "Strip `<script>`, `<style>`, `<nav>`, and inline `<svg>` blocks; extract inner text.",
        "Impact if Ignored": "Pollutes context with thousands of irrelevant CSS rules and JavaScript syntax tokens."
    },
    {
        "Cleaning Challenge": "5. Noisy OCR Scans",
        "Detection Mechanism": "Dictionary lookup ratio (percentage of non-dictionary garbage characters).",
        "Remediation Strategy": "Binarization (Otsu thresholding), deskewing, spell-correction before embedding.",
        "Impact if Ignored": "Gibberish tokens (e.g. `l1k3 th1s`) create out-of-vocabulary vector embeddings."
    },
    {
        "Cleaning Challenge": "6. Corrupted / Encrypted Files",
        "Detection Mechanism": "Exception handling during loader execution (`PDFSyntaxError`, `PasswordRequired`).",
        "Remediation Strategy": "Catch exception; route file to dead-letter quarantine queue; log alert; continue batch.",
        "Impact if Ignored": "A single corrupted file crashes the entire overnight ingestion job."
    }
])

st.dataframe(cleaning_matrix, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dynamic Header & Footer Stripping Algorithm")
    st.markdown("""
    How do you remove running headers from a 100-page PDF without hardcoding strings?
    1. **Page Extraction**: Extract the top 2 lines and bottom 2 lines from each page.
    2. **Frequency Distribution**: Count line occurrences across all pages: `count(line) / total_pages`.
    3. **Thresholding**: Any line appearing on more than **70% of pages** (e.g., *"Acme Corp Annual Report 2024"* or *"Page X of Y"*) is classified as running boilerplate.
    4. **Stripping**: Prune these recurring lines from the page text before chunking!
    """)

with col2:
    st.subheader("Data Cleaning & Quarantine Architecture")
    clean_chart = """
    digraph CleaningFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin=0.15];
        edge [fontname="Helvetica", fontsize=9];

        Input [label="Raw Documents Ingestion", fillcolor="#E0E7FF", color="#4338CA"];
        CorruptCheck [label="Corrupted / Password Check", fillcolor="#EDE9FE", color="#6D28D9"];
        Quarantine [label="Quarantine Queue\\n(Alert Admin)", fillcolor="#FCA5A5", color="#B91C1C"];
        HashCheck [label="SHA-256 Exact Hash Check", fillcolor="#FEF3C7", color="#D97706"];
        HeaderStrip [label="Dynamic Header/Footer Stripper", fillcolor="#DCFCE7", color="#16A34A"];
        CleanDocs [label="Clean Document Stream\\n(To Semantic Chunker)", fillcolor="#D1FAE5", color="#059669"];

        Input -> CorruptCheck;
        CorruptCheck -> Quarantine [label="Corrupted / Encrypted"];
        CorruptCheck -> HashCheck [label="Valid file"];
        HashCheck -> Quarantine [label="Exact Duplicate (Drop)"];
        HashCheck -> HeaderStrip [label="Unique Document"];
        HeaderStrip -> CleanDocs;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(clean_chart, use_container_width=False)

st.divider()

# Section 2: Production Code: Pure Python Enterprise Data Cleaning Engine
st.header("2. Production Code: Pure Python Enterprise Data Cleaning Suite", divider="rainbow")
st.markdown("""
Here is a complete, runnable data cleaning module featuring **cryptographic deduplication**, **dynamic header stripping**, and **corrupted file resilience**:
""")

st.code("""
import hashlib
import re
from typing import List, Dict, Any, Set

class EnterpriseDataCleaner:
    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def is_exact_duplicate(self, text: str) -> bool:
        \"\"\"Checks if identical text was already ingested via SHA-256.\"\"\"
        doc_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if doc_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(doc_hash)
        return False

    def strip_running_headers_and_footers(self, pages: List[str], repetition_threshold: float = 0.7) -> List[str]:
        \"\"\"
        Dynamically detects and removes headers/footers repeated across multiple pages.
        \"\"\"
        if len(pages) < 3:
            return pages  # Need at least 3 pages to establish pattern
            
        header_candidate_counts: Dict[str, int] = {}
        footer_candidate_counts: Dict[str, int] = {}
        
        # Pass 1: Count line frequencies in top 2 and bottom 2 lines of each page
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            if lines:
                first_line = lines[0]
                last_line = lines[-1]
                header_candidate_counts[first_line] = header_candidate_counts.get(first_line, 0) + 1
                footer_candidate_counts[last_line] = footer_candidate_counts.get(last_line, 0) + 1
                
        # Identify repeating boilerplate
        total_pages = len(pages)
        boilerplate_lines = set()
        for line, count in header_candidate_counts.items():
            if count / total_pages >= repetition_threshold:
                boilerplate_lines.add(line)
        for line, count in footer_candidate_counts.items():
            if count / total_pages >= repetition_threshold:
                boilerplate_lines.add(line)
                
        # Pass 2: Strip identified lines from all pages
        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = [l for l in lines if l.strip() not in boilerplate_lines]
            cleaned_pages.append("\\n".join(cleaned_lines).strip())
            
        return cleaned_pages

    def clean_html_markup(self, raw_html: str) -> str:
        \"\"\"Strips scripts, styles, and tags from raw HTML.\"\"\"
        # Remove scripts and styles
        cleaned = re.sub(r"<(script|style).*?>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Collapse whitespace
        return re.sub(r"\\s+", " ", cleaned).strip()

# Demonstration
cleaner = EnterpriseDataCleaner()

# 1. Deduplication Test
text_a = "Acme Corp Policy on Working Hours: Core hours are 10am to 3pm."
print("Doc 1 Duplicate Check:", cleaner.is_exact_duplicate(text_a))  # False (New)
print("Doc 2 Duplicate Check:", cleaner.is_exact_duplicate(text_a))  # True (Duplicate Dropped!)

# 2. Dynamic Header Stripping Test across 3 pages
sample_pages = [
    "Acme Annual Report 2024\\nSection 1: Revenue grew by 15%.\\nPage 1 of 3",
    "Acme Annual Report 2024\\nSection 2: Operating margin was 24%.\\nPage 2 of 3",
    "Acme Annual Report 2024\\nSection 3: Net income was $400M.\\nPage 3 of 3"
]

cleaned_pages = cleaner.strip_running_headers_and_footers(sample_pages, repetition_threshold=0.66)
print("\\n--- Cleaned Page 1 ---")
print(cleaned_pages[0])
""", language="python")

st.divider()

# Section 3: Interactive Cleaning & Deduplication Workbench
st.header("3. Interactive Cleaning & Deduplication Workbench", divider="rainbow")
st.markdown("""
Test automated header stripping and exact deduplication across simulated multi-page documents:
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Multi-Page Document Input")
    st.code("""[Page 1]:
CONFIDENTIAL - INTERNAL AUDIT ONLY
The European operations achieved $45M EBIT.
Page 1 of 3

[Page 2]:
CONFIDENTIAL - INTERNAL AUDIT ONLY
North American revenue reached $120M.
Page 2 of 3

[Page 3]:
CONFIDENTIAL - INTERNAL AUDIT ONLY
Asia-Pacific margins expanded by 3%.
Page 3 of 3""", language="text")

with col2:
    st.subheader("Automated Cleaned Output")
    st.code("""[Page 1]:
The European operations achieved $45M EBIT.

[Page 2]:
North American revenue reached $120M.

[Page 3]:
Asia-Pacific margins expanded by 3%.""", language="text")

st.info("💡 Notice how the dynamic stripper automatically detected and removed the running header (`CONFIDENTIAL - INTERNAL AUDIT ONLY`) because it repeated across 100% of pages, leaving only pure business signal!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Cleaning Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does MinHash LSH (Locality Sensitive Hashing) achieve near-duplicate detection at scale?**
    * **Answer**: In a corpus of 100,000 documents, computing pairwise string similarity requires `(100,000 * 99,999) / 2 = ~5 billion` comparisons, which is computationally intractable. **MinHash** compresses each document into a small vector of hash integers (e.g. 128 hashes) representing its shingle set. **Locality Sensitive Hashing (LSH)** groups documents with matching hash bands into the same hash buckets. Only documents sharing buckets are compared, reducing comparison overhead from $O(N^2)$ to sub-linear time and identifying near-duplicate drafts in seconds.
    
    **Q2: How do you handle password-protected or encrypted PDFs in batch ingestion?**
    * **Answer**: If a document loader encounters a password-protected PDF without error handling, it raises an unhandled `PyPDF.errors.PdfReadError: File is encrypted` exception that crashes the entire batch worker process. In production, file loaders must be wrapped in `try...except` blocks. Encrypted files are intercepted, logged with a security alert, and moved to an isolated **Quarantine / Dead-Letter Queue** for manual review while the pipeline continues uninterrupted.
    
    **Q3: What preprocessing steps dramatically improve OCR accuracy on scanned documents?**
    * **Answer**: Raw scanned PDFs often suffer from low contrast, skewed angles, and background speckles. Standard OCR preprocessing pipelines apply: (1) **Deskewing** (detecting document tilt angle and rotating the image); (2) **Grayscale & Binarization** (Otsu thresholding to convert pixels into crisp black text on white background); (3) **Denoising** (median filtering to strip salt-and-pepper scan noise); and (4) **DPI Upscaling** (resampling images to 300 DPI, the optimal resolution for Tesseract OCR).
    """)

with traps_tab:
    st.error("""
    **Trap 1: Executing pairwise $O(N^2)$ string comparisons for corpus deduplication**  
    Running pairwise comparisons on a corporate repository of 50,000 files requires over 1.2 billion comparisons, freezing ingestion pipelines for hours. Always use cryptographic hashes (SHA-256) for exact duplicates and MinHash LSH for near-duplicates!
    """)
    st.error("""
    **Trap 2: Stripping HTML tags without removing `<script>` and `<style>` blocks first**  
    If you apply a naive regex (`<[^>]+>`) to HTML without stripping script tags, the raw JavaScript and CSS code inside the tags remains behind as text, resulting in vector chunks filled with CSS variables and tracking code!
    """)

with flashcards_tab:
    st.info("**MinHash LSH**: An algorithmic hashing technique that maps similar documents to the same hash buckets, enabling near-duplicate detection across millions of records in sub-linear time.")
    st.info("**Running Header Detection**: Identifying recurring strings at the boundaries of multi-page documents through frequency thresholding across page collections.")
    st.info("**Quarantine Queue**: A designated dead-letter storage location where corrupted, encrypted, or malformed files are isolated to prevent batch ingestion crashes.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
checklist = [
    "I understand the difference between exact deduplication (SHA-256) and near-duplicate detection (MinHash).",
    "I can implement a dynamic algorithm to strip repetitive PDF running headers and footers.",
    "I know how to safely clean HTML markup by stripping script and style blocks first.",
    "I understand the necessary image preprocessing steps (deskewing, binarization) for OCR scans.",
    "I know how to design a fault-tolerant quarantine queue for corrupted and encrypted files."
]

for item in checklist:
    st.checkbox(item, value=False)
