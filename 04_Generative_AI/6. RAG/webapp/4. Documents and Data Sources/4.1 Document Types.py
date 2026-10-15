import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="4.1 Document Types - RAG Master Curriculum",
    page_icon="📄",
    layout="wide"
)

# Title & Overview
st.title("4.1 Document Types: Unstructured, Semi-Structured, & Tabular Data in Enterprise RAG")
st.caption("Chapter 4: Documents and Data Sources | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Enterprise knowledge does not live in clean plaintext files. In real organizations, knowledge is fragmented across 
a chaotic landscape of file formats: **Unstructured Documents** (PDFs, Word docs, Markdown, plain text), 
**Semi-Structured Hierarchies** (HTML web pages, JSON payloads, XML feeds), and **Structured Tabular Stores** (CSV, Excel, SQL databases).
Each document type imposes radically different parsing requirements. 
Flattening a complex PDF with multi-column layouts into raw text scrambles reading order; 
similarly, feeding a raw CSV row (*"102, Acme, 45.2, True"*) into a vector model destroys column semantics unless serialized into natural language.
This lesson covers the characteristics, parsing tools, and serialization strategies across all major enterprise document types.
""")

st.divider()

# Section 1: The Multi-Format Document Taxonomy
st.header("1. The Multi-Format Enterprise Document Taxonomy", divider="rainbow")
st.markdown("""
How different file types are parsed and transformed for semantic retrieval:
""")

doc_types_table = pd.DataFrame([
    {
        "Document Category": "1. Markdown (.md)",
        "Formats Included": "GitHub wikis, technical documentation, Notion exports.",
        "Primary Parsing Tool": "Native markdown splitters (`MarkdownHeaderTextSplitter`).",
        "Key Advantage": "Gold standard format! Headers (`#`, `##`, `###`) provide native semantic boundary tags."
    },
    {
        "Document Category": "2. Portable Document Format (.pdf)",
        "Formats Included": "Annual 10-K reports, research papers, legal contracts, scanned invoices.",
        "Primary Parsing Tool": "PyPDF, PDFPlumber, Unstructured, LlamaParse, MinerU.",
        "Key Challenge": "Multi-column layouts, running headers/footers, embedded charts, scanned OCR needs."
    },
    {
        "Document Category": "3. Tabular & Spreadsheets (.csv, .xlsx)",
        "Formats Included": "Sales ledgers, financial models, customer lists, inventory databases.",
        "Primary Parsing Tool": "Pandas, OpenPyXL, CSV DictReader.",
        "Key Challenge": "Must serialize rows into key-value sentences (`Column: Value`) to maintain semantic meaning."
    },
    {
        "Document Category": "4. Semi-Structured (.json, .xml)",
        "Formats Included": "API responses, database dumps, configuration records.",
        "Primary Parsing Tool": "Built-in `json`, `xml.etree`, JSONLoader.",
        "Key Challenge": "Deep nested keys must be flattened or extracted into targeted metadata fields."
    },
    {
        "Document Category": "5. Web & Hypertext (.html)",
        "Formats Included": "Intranet portals, Zendesk help centers, public corporate websites.",
        "Primary Parsing Tool": "BeautifulSoup, Trafilatura, Playwright (for JavaScript rendering).",
        "Key Challenge": "Must strip boilerplate navbars, cookie banners, tracking scripts, and sidebars."
    },
    {
        "Document Category": "6. Office Documents (.docx, .pptx)",
        "Formats Included": "Meeting memos, project proposals, corporate presentation decks.",
        "Primary Parsing Tool": "`python-docx`, `python-pptx`, Unstructured.",
        "Key Challenge": "Extracting speaker notes, table cells, and shape text in logical reading order."
    }
])

st.dataframe(doc_types_table, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Tabular Trap in Vector Search")
    st.markdown("""
    * **Naive Mistake**: Ingesting a CSV row as raw comma-separated values:  
      *`"942, WidgetPro, 49.99, InStock, 2024-05-01"`*
    * **Why Vector Search Fails**: Dense embedding models are trained on natural human language. The vector for `"49.99"` has no idea whether it represents a price, an age, a weight, or a percentage!
    * **The Semantic Serialization Solution**: Reformat tabular rows into structured natural sentences:  
      *`"Product ID: 942. Name: WidgetPro. Retail Price: $49.99. Inventory Status: InStock. Restock Date: May 1, 2024."`*
    """)

with col2:
    st.subheader("Document Ingestion Flowchart")
    doc_chart = """
    digraph DocTypes {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin=0.15];
        edge [fontname="Helvetica", fontsize=9];

        Source [label="Raw Enterprise Sources\\n(PDF, Markdown, CSV, JSON, HTML)", fillcolor="#E0E7FF", color="#4338CA"];
        Parser [label="Format-Specific Extractor\\n(LlamaParse / BS4 / Pandas)", fillcolor="#EDE9FE", color="#6D28D9"];
        Cleaner [label="Sanitizer & Normalizer\\n(Strip boilerplate, fix whitespace)", fillcolor="#FEF3C7", color="#D97706"];
        Standard [label="Standard Document Object\\n(Clean text + Metadata dictionary)", fillcolor="#DCFCE7", color="#16A34A"];
        RAG [label="To Chunking & Vectorization", fillcolor="#D1FAE5", color="#059669"];

        Source -> Parser -> Cleaner -> Standard -> RAG;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(doc_chart, use_container_width=False)

st.divider()

# Section 2: Production Code: Multi-Format Text Extractor & Serializer
st.header("2. Production Code: Multi-Format Semantic Serializer", divider="rainbow")
st.markdown("""
Here is a pure Python serializer demonstrating how different data structures are converted into clean, semantically rich RAG text chunks:
""")

st.code("""
import json
import csv
import io
from typing import List, Dict, Any

class MultiFormatRAGSerializer:
    def __init__(self):
        pass

    def serialize_csv_rows(self, csv_data: str) -> List[str]:
        \"\"\"
        Converts CSV rows into semantic key-value sentences for vector embedding.
        \"\"\"
        f = io.StringIO(csv_data.strip())
        reader = csv.DictReader(f)
        serialized_chunks = []
        for row in reader:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            sentence = ". ".join(parts) + "."
            serialized_chunks.append(sentence)
        return serialized_chunks

    def serialize_json_record(self, json_data: Dict[str, Any], prefix: str = "") -> str:
        \"\"\"
        Recursively flattens nested JSON hierarchies into readable sentences.
        \"\"\"
        lines = []
        for k, v in json_data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                lines.append(self.serialize_json_record(v, prefix=full_key))
            elif isinstance(v, list):
                items_str = ", ".join(str(x) for x in v)
                lines.append(f"{full_key}: [{items_str}]")
            else:
                lines.append(f"{full_key}: {v}")
        return "\\n".join(lines)

# Demonstration
serializer = MultiFormatRAGSerializer()

# 1. Test CSV Serialization
sample_csv = \"\"\"product_name,category,price_usd,warranty_months
UltraLaptop 15,Computers,1299.99,24
SoundPro Buds,Audio,149.50,12\"\"\"

print("--- SERIALIZED CSV ROWS ---")
for chunk in serializer.serialize_csv_rows(sample_csv):
    print(f"- {chunk}")

# 2. Test JSON Serialization
sample_json = {
    "employee": "Jane Doe",
    "department": {"name": "Engineering", "team": "Platform Security"},
    "roles": ["Admin", "Deployer"]
}

print("\\n--- SERIALIZED JSON RECORD ---")
print(serializer.serialize_json_record(sample_json))
""", language="python")

st.divider()

# Section 3: Interactive Document Type Inspector & Serializer Workbench
st.header("3. Interactive Document Type Inspector & Serializer Workbench", divider="rainbow")
st.markdown("""
Compare raw file structures against their semantically converted RAG representations:
""")

doc_format = st.selectbox(
    "Choose Document Format to Inspect:",
    ["CSV Tabular Data", "Hierarchical JSON", "Markdown with Headers", "Raw HTML Page"]
)

if doc_format == "CSV Tabular Data":
    st.subheader("Raw CSV Data")
    st.code("id,company,revenue_b,growth_pct\n1,Acme,4.2,15\n2,Globex,8.5,8", language="text")
    st.subheader("Semantically Serialized RAG Chunks")
    st.code("Chunk 1: id: 1. company: Acme. revenue_b: 4.2. growth_pct: 15%.\nChunk 2: id: 2. company: Globex. revenue_b: 8.5. growth_pct: 8%.", language="text")
    st.info("💡 By prefixing every field with its column header, the embedding model correctly captures what '4.2' and '15%' mean.")

elif doc_format == "Hierarchical JSON":
    st.subheader("Raw JSON Record")
    st.code('{\n  "server": "prod-east-01",\n  "specs": {"cpu": 32, "ram_gb": 128},\n  "status": "HEALTHY"\n}', language="json")
    st.subheader("Flattened RAG Context")
    st.code("server: prod-east-01\nspecs.cpu: 32 cores\nspecs.ram_gb: 128 GB\nstatus: HEALTHY", language="text")

elif doc_format == "Markdown with Headers":
    st.subheader("Raw Markdown Text")
    st.code("# Vacation Policy\n## Eligibility\nFull time employees after 6 months.\n## Days Off\n15 days annual leave.", language="markdown")
    st.subheader("Header-Aware Chunks with Metadata")
    st.code("Chunk 1: [Header: Vacation Policy > Eligibility] Full time employees after 6 months.\nChunk 2: [Header: Vacation Policy > Days Off] 15 days annual leave.", language="text")

else:
    st.subheader("Raw HTML Source")
    st.code('<div class="navbar">Home | About</div><article><h1>Refunds</h1><p>30 days return window.</p></article>', language="html")
    st.subheader("Clean Trafilatura Extracted Markdown")
    st.code("# Refunds\n30 days return window.", language="text")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Document Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How should tabular CSV or SQL rows be transformed for dense vector search?**
    * **Answer**: Dense embedding models are trained on natural language prose, not comma-separated lists of numbers. Passing a raw CSV row (e.g. `142, Bob, 92000, Eng`) into a vector model produces poor embeddings because column semantics are lost. In production, tabular rows must be **semantically serialized**: converting each row into an explicit key-value sentence (e.g. `"Employee ID: 142. Name: Bob. Salary: $92,000. Department: Engineering."`), preserving column names alongside the row values.
    
    **Q2: Why is Markdown considered the optimal intermediate format for document ingestion?**
    * **Answer**: Markdown retains essential semantic hierarchy (headers `#`, `##`, tables, bold terms, bullet points) in lightweight, human-readable plaintext without the massive tag overhead of HTML or the layout complexity of PDF. Most modern enterprise loaders (like LlamaParse or Microsoft MarkItDown) parse complex PDFs directly into Markdown because Markdown headers provide deterministic semantic boundaries for chunking.
    
    **Q3: How do modern Vision-Language models (like ColPali) circumvent traditional PDF parsing failures?**
    * **Answer**: Traditional PDF parsing converts pages to text via OCR or PDF extractors, which inevitably scrambles multi-column layouts, breaks tables, and ignores infographics. ColPali eliminates text extraction completely: it renders each PDF page as an image, feeds the image through a Vision-Language foundation model (PaliGemma), and generates multi-vector patch embeddings directly from the page layout, allowing text queries to match directly against visual tables and diagrams without OCR.
    """)

with traps_tab:
    st.error("""
    **Trap 1: Ingesting recurring page headers, footers, and copyright notices into vector databases**  
    In PDFs, recurring footers (e.g. *"Confidential - Internal Use Only - Page 14"*) appear on every single page. If not stripped during cleaning, vector search will surface 50 chunks matching the word *"Confidential"*, crowding out real answers!
    """)
    st.error("""
    **Trap 2: Splitting complex JSON objects across chunk boundaries**  
    Splitting a JSON document by raw character counts slices across curly braces and commas, producing invalid JSON fragments that corrupt downstream parsers and erase key relationships.
    """)

with flashcards_tab:
    st.info("**Tabular Serialization**: Transforming structured database rows or CSV records into natural language key-value sentences for dense embedding.")
    st.info("**MarkdownHeaderTextSplitter**: A chunking tool that splits text based on Markdown headers (`#`, `##`), attaching parent header paths to each chunk's metadata.")
    st.info("**Boilerplate Removal**: Stripping repetitive, non-informative elements (headers, footers, navbars, cookie banners) prior to embedding generation.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
checklist = [
    "I understand the parsing differences between Unstructured, Semi-Structured, and Tabular data.",
    "I know how to semantically serialize CSV and SQL rows for vector search.",
    "I understand why Markdown is the premier intermediate format for RAG ingestion.",
    "I can explain why recurring PDF headers and footers must be stripped prior to indexing.",
    "I understand how Vision-Language models (ColPali) bypass brittle PDF OCR pipelines."
]

for item in checklist:
    st.checkbox(item, value=False)
