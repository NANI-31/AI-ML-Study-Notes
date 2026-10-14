import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class LoadedDocSnippet:
    """Represents a simulated parsed Document object from an enterprise loader."""
    def __init__(self, doc_type: str, title: str, text: str, metadata: Dict[str, Any]):
        self.doc_type = doc_type
        self.title = title
        self.text = text
        self.metadata = metadata

class MockDocumentLoaderSuite:
    """Simulates heterogeneous enterprise loaders with schema-aware parsing."""
    def __init__(self):
        pass

    def simulate_load(self, loader_type: str) -> List[LoadedDocSnippet]:
        if loader_type == "PDF Loader (PyPDFLoader)":
            return [
                LoadedDocSnippet(
                    "PDF", "Page 1 - Corporate By-laws",
                    "Article I: Governance Structure. The Board of Directors shall consist of seven voting members.",
                    {"source": "corporate_bylaws.pdf", "page": 1, "total_pages": 4, "format": "application/pdf"}
                ),
                LoadedDocSnippet(
                    "PDF", "Page 2 - Officer Duties",
                    "Article II: Executive Authority. The Chief Executive Officer shall execute all contracts.",
                    {"source": "corporate_bylaws.pdf", "page": 2, "total_pages": 4, "format": "application/pdf"}
                )
            ]
        elif loader_type == "CSV Loader (Row-by-Row)":
            return [
                LoadedDocSnippet(
                    "CSV", "Row 1: EMP-101",
                    "Employee ID: EMP-101 | Name: Alice Chen | Role: Staff AI Architect | Clearance: Top Secret",
                    {"source": "employees.csv", "row": 1, "source_column": "Name", "format": "text/csv"}
                ),
                LoadedDocSnippet(
                    "CSV", "Row 2: EMP-102",
                    "Employee ID: EMP-102 | Name: Bob Martinez | Role: Security Engineer | Clearance: Secret",
                    {"source": "employees.csv", "row": 2, "source_column": "Name", "format": "text/csv"}
                )
            ]
        elif loader_type == "JSON Loader (jq Schema Query)":
            return [
                LoadedDocSnippet(
                    "JSON", "Record: cluster-prod-east",
                    "Region: us-east-1 | Nodes: 48 | Status: HEALTHY | Version: v1.31.2-eks",
                    {"source": "infrastructure_state.json", "jq_schema": ".clusters[]", "format": "application/json"}
                ),
                LoadedDocSnippet(
                    "JSON", "Record: cluster-prod-eu",
                    "Region: eu-central-1 | Nodes: 32 | Status: HEALTHY | Version: v1.31.2-eks",
                    {"source": "infrastructure_state.json", "jq_schema": ".clusters[]", "format": "application/json"}
                )
            ]
        elif loader_type == "HTML Loader (BSHTMLLoader)":
            return [
                LoadedDocSnippet(
                    "HTML", "Corporate Careers Portal",
                    "Careers at MegaCorp. We are hiring Distributed Systems Engineers. Benefits include remote-first flexibility and equity grants.",
                    {"source": "https://megacorp.internal/careers", "title": "MegaCorp Careers", "format": "text/html"}
                )
            ]
        elif loader_type == "Markdown Loader":
            return [
                LoadedDocSnippet(
                    "Markdown", "README.md - Architecture",
                    "# System Blueprint\\nAll ingress services must authenticate against the central OIDC IdP before routing.",
                    {"source": "docs/architecture/README.md", "header_level": 1, "format": "text/markdown"}
                )
            ]
        elif loader_type == "WebBaseLoader (Live Scraping)":
            return [
                LoadedDocSnippet(
                    "Web", "API Documentation v4",
                    "Endpoint: /v1/chat/completions. Accepts JSON payload with model identifier and array of message objects.",
                    {"source": "https://api.docs.enterprise.com", "http_status": 200, "user_agent": "LangChain-WebBaseLoader"}
                )
            ]
        elif loader_type == "DirectoryLoader (Recursive)":
            return [
                LoadedDocSnippet(
                    "Directory", "file1.txt", "Config: memory_limit=16G",
                    {"source": "/var/configs/file1.txt", "extension": ".txt"}
                ),
                LoadedDocSnippet(
                    "Directory", "file2.md", "## Deployment Guide\\nDeploy with helm.",
                    {"source": "/var/configs/file2.md", "extension": ".md"}
                )
            ]
        elif loader_type == "Database Loader (SQLDatabaseLoader)":
            return [
                LoadedDocSnippet(
                    "SQL", "Row: orders_tbl: 98402",
                    "Order ID: 98402 | Customer: Acquired Healthcare Corp | Total: 142000.00 USD | Status: COMPLETED",
                    {"source": "postgresql://db.corp:5432/orders", "table": "orders_tbl", "primary_key": 98402}
                )
            ]
        return []


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="12.3 Document Loaders",
    page_icon="📥",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">12.3 Document Loaders</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Connecting enterprise data sources: PDF, Text, CSV, JSON, HTML, Markdown, Web, Directory, and Database loaders.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Multi-Format Document Ingestion Engine", divider="rainbow")

st.markdown("""
Enterprise knowledge resides across fragmented storage mediums: PDF manuals on file shares, tabular datasets in CSV, 
API responses in JSON, intranet wikis in HTML/Markdown, web portals, and relational SQL databases.
**Document Loaders** abstract away disparate file protocols, encoding formats, and network I/O, converting any data stream 
into a standardized list of LangChain `Document` objects.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LoaderArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_heterogeneous_sources {
            label="Heterogeneous Enterprise Data Stores";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            SrcPDF [label="PDF Files\\n(Manuals, Invoices, Scans)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            SrcCSV [label="CSV / TSV\\n(Tabular Records, Metrics)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            SrcJSON [label="JSON / REST\\n(API Payloads, Schemas)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            SrcHTML [label="HTML / Wikis\\n(Confluence, SharePoint)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            SrcSQL [label="SQL Databases\\n(Postgres, MySQL, Snowflake)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_specialized_loaders {
            label="Specialized LangChain Loaders";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            LoadPDF [label="PyPDFLoader / PDFPlumber\\n(Page-by-page text extraction)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            LoadCSV [label="CSVLoader\\n(Row-to-Document translation)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            LoadJSON [label="JSONLoader\\n(jq query path filtering)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            LoadHTML [label="BSHTMLLoader / WebBaseLoader\\n(DOM cleaning & tag stripping)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            LoadSQL [label="SQLDatabaseLoader\\n(Query execution & row mapping)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_unified_output {
            label="Standardized Ingestion Stream";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            UnifiedStream [label="Standardized Document Stream\\n• page_content: clean string\\n• metadata: {source, page, format, row}\\n• id: deterministic UUID", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399", penwidth=2];
        }

        SrcPDF -> LoadPDF [label="Binary stream"];
        SrcCSV -> LoadCSV [label="Delimited text"];
        SrcJSON -> LoadJSON [label="JSON tree"];
        SrcHTML -> LoadHTML [label="Raw markup"];
        SrcSQL -> LoadSQL [label="ODBC cursor"];

        LoadPDF -> UnifiedStream [label="List[Document]"];
        LoadCSV -> UnifiedStream [label="List[Document]"];
        LoadJSON -> UnifiedStream [label="List[Document]"];
        LoadHTML -> UnifiedStream [label="List[Document]"];
        LoadSQL -> UnifiedStream [label="List[Document]"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: PDF LOADERS ---
st.header("2. PDF Loaders", divider="rainbow")

st.markdown("""
PDF is the dominant format for enterprise contracts, financial prospectuses, and technical specifications. 
LangChain provides multiple specialized PDF loaders depending on document complexity:
- **`PyPDFLoader`**: Fast, pure-Python loader. Splits the PDF page-by-page, attaching `{"page": X, "source": "path"}` to metadata.
- **`PDFPlumberLoader`**: Preserves spatial formatting and extracts embedded tabular figures cleanly.
- **`UnstructuredPDFLoader`**: Uses neural document layout analysis to detect headers, multi-column reading orders, and footnotes.
""")

st.code("""from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("annual_financials_2026.pdf")
pages = loader.load()

# Each page becomes a distinct Document object
print(f"Total Pages: {len(pages)}")
print(f"Page 1 Content: {pages[0].page_content[:150]}")
print(f"Metadata: {pages[0].metadata}")
# Output: {'source': 'annual_financials_2026.pdf', 'page': 0}""", language="python")

st.divider()

# --- SECTION 3: TEXT LOADERS ---
st.header("3. Text Loaders", divider="rainbow")

st.markdown("""
The `TextLoader` is the foundational loader for plain text, `.txt`, log files, and source code.
It handles character encoding normalization (UTF-8, ISO-8859-1) and attaches the file path to the metadata.
""")

st.code("""from langchain_community.document_loaders import TextLoader

# Auto-detect encoding or specify explicitly
loader = TextLoader("server_audit.log", encoding="utf-8", autodetect_encoding=True)
docs = loader.load()""", language="python")

st.divider()

# --- SECTION 4: CSV LOADERS ---
st.header("4. CSV Loaders", divider="rainbow")

st.markdown("""
When indexing structured tabular data for RAG, dumping raw CSV files into an embedding model confuses vector similarity.
The `CSVLoader` converts each row into an independent `Document` object formatted as readable key-value pairs:
""")

st.code("""from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="customers.csv",
    source_column="customer_id",  # Designate column used for citations
    csv_args={"delimiter": ",", "quotechar": '"'}
)
row_documents = loader.load()""", language="python")

st.divider()

# --- SECTION 5: JSON LOADERS ---
st.header("5. JSON Loaders", divider="rainbow")

st.markdown("""
APIs, MongoDB dumps, and microservice events emit hierarchical JSON. 
The `JSONLoader` utilizes `jq` query syntax to extract targeted text fields from complex nested dictionaries:
""")

st.code("""from langchain_community.document_loaders import JSONLoader

# Extract all 'comment' fields from a nested list of user reviews
loader = JSONLoader(
    file_path="reviews.json",
    jq_schema=".reviews[].comment",
    text_content=True
)
comments = loader.load()""", language="python")

st.divider()

# --- SECTION 6: HTML LOADERS ---
st.header("6. HTML Loaders", divider="rainbow")

st.markdown("""
Scraping intranet wikis or downloaded web pages requires stripping boilerplate HTML syntax without losing textual hierarchy.
- **`BSHTMLLoader`**: Uses BeautifulSoup to strip raw tags, extracting text and the `<title>` element into metadata.
- **`UnstructuredHTMLLoader`**: Distinguishes between navigation sidebars, footers, and main article body content.
""")

st.divider()

# --- SECTION 7: MARKDOWN LOADERS ---
st.header("7. Markdown Loaders", divider="rainbow")

st.markdown("""
Markdown files (documentation sites, GitHub READMEs) contain structural hierarchy embedded in `#`, `##`, and `###` headers.
The `UnstructuredMarkdownLoader` parses markdown syntax, preserving headers and code blocks intact for downstream chunking.
""")

st.divider()

# --- SECTION 8: WEB PAGE LOADERS ---
st.header("8. Web Page Loaders", divider="rainbow")

st.markdown("""
The `WebBaseLoader` fetches live web pages directly over HTTP using `urllib` and BeautifulSoup.
It supports concurrent scraping across multiple URLs, header customization, and rate limiting:
""")

st.code("""from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://docs.langchain.com/oss/python/langchain/rag",
    "https://docs.langchain.com/oss/python/langgraph/overview"
]
web_loader = WebBaseLoader(web_paths=urls)
web_loader.requests_per_second = 2  # Rate limit politeness
web_docs = web_loader.load()""", language="python")

st.divider()

# --- SECTION 9: DIRECTORY LOADERS ---
st.header("9. Directory Loaders", divider="rainbow")

st.markdown("""
In enterprise deployments, documents arrive in folders containing mixed file types.
The `DirectoryLoader` scans directories recursively, using glob patterns and mapping file extensions to specialized child loaders:
""")

st.code("""from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Multi-threaded recursive loader
dir_loader = DirectoryLoader(
    path="./knowledge_repo",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    use_multithreading=True,
    max_concurrency=4,
    show_progress=True
)
docs = dir_loader.load()""", language="python")

st.divider()

# --- SECTION 10: DATABASE LOADERS ---
st.header("10. Database Loaders", divider="rainbow")

st.markdown("""
The `SQLDatabaseLoader` queries relational database engines (PostgreSQL, MySQL, Snowflake) using SQLAlchemy, 
translating table rows or custom SELECT queries directly into Document objects:
""")

st.code("""from langchain_community.document_loaders import SQLDatabaseLoader
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost:5432/corp_db")
query = "SELECT order_id, customer_name, total_amount, status FROM orders WHERE status = 'PENDING';"

db_loader = SQLDatabaseLoader(
    query=query,
    engine=engine,
    page_content_columns=["order_id", "customer_name", "status"],
    metadata_columns=["total_amount"]
)
db_docs = db_loader.load()""", language="python")

st.divider()

# --- SECTION 11: PRODUCTION IMPLEMENTATION ---
st.header("11. Production Implementation: Unified Multi-Modal Ingestion Router", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Ingestion Router that automatically inspects file extensions and dispatches 
to the optimal loader class with defensive encoding fallback handling.
""")

st.code("""import mimetypes
from typing import List, Dict, Any

class EnterpriseIngestionRouter:
    \"\"\"Dynamically selects and executes the appropriate LangChain loader.\"\"\"
    def __init__(self):
        self.supported_extensions = [".pdf", ".csv", ".json", ".txt", ".md", ".html"]

    def route_and_load(self, file_path: str, file_bytes: bytes = None) -> List[Dict[str, Any]]:
        ext = file_path.lower().split(".")[-1] if "." in file_path else ""
        ext = f".{ext}"

        if ext == ".pdf":
            # In production: PyPDFLoader(file_path).load()
            return [{"source": file_path, "type": "PDF", "pages": 2, "sample": "Parsed PDF contract page."}]
        elif ext == ".csv":
            # In production: CSVLoader(file_path).load()
            return [{"source": file_path, "type": "CSV", "rows": 50, "sample": "Row-by-row key-value documents."}]
        elif ext == ".json":
            # In production: JSONLoader(file_path, jq_schema=".").load()
            return [{"source": file_path, "type": "JSON", "records": 12, "sample": "Parsed JSON tree objects."}]
        elif ext in [".txt", ".md"]:
            # In production: TextLoader(file_path).load()
            return [{"source": file_path, "type": "Text/MD", "sample": "Raw UTF-8 text document."}]
        elif ext == ".html":
            # In production: BSHTMLLoader(file_path).load()
            return [{"source": file_path, "type": "HTML", "sample": "Cleaned DOM body content."}]
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Must be one of {self.supported_extensions}")""", language="python")

st.divider()

# --- SECTION 12: INTERACTIVE WORKBENCH ---
st.header("12. Interactive Workbench: Enterprise Loader Simulator & Content Inspector", divider="rainbow")

st.markdown("""
Select a data source loader to simulate how diverse enterprise formats are parsed into normalized LangChain `Document` objects.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Select Loader Implementation")
    selected_loader = st.selectbox(
        "Target Document Loader",
        options=[
            "PDF Loader (PyPDFLoader)",
            "CSV Loader (Row-by-Row)",
            "JSON Loader (jq Schema Query)",
            "HTML Loader (BSHTMLLoader)",
            "Markdown Loader",
            "WebBaseLoader (Live Scraping)",
            "DirectoryLoader (Recursive)",
            "Database Loader (SQLDatabaseLoader)"
        ]
    )

    st.markdown("#### Loader Mechanics:")
    if "PDF" in selected_loader:
        st.info("Splits by page boundary; attaches page indices to metadata.")
    elif "CSV" in selected_loader:
        st.info("Translates each CSV record into an independent key-value text snippet.")
    elif "JSON" in selected_loader:
        st.info("Applies jq path selection to pluck nested attributes.")
    elif "HTML" in selected_loader:
        st.info("Strips `<script>`, `<style>`, and navigation headers; extracts title.")
    elif "Database" in selected_loader:
        st.info("Converts relational SQL result sets into Document records.")

    btn_load = st.button("🚀 Execute Document Loader", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Normalized Document Output")

    if btn_load:
        suite = MockDocumentLoaderSuite()
        loaded_items = suite.simulate_load(selected_loader)

        st.metric("Documents Emitted", len(loaded_items))

        for idx, item in enumerate(loaded_items, 1):
            with st.expander(f"📄 Doc {idx}: {item.title} ({item.doc_type})", expanded=True):
                st.write(f"**Content**: `{item.text}`")
                st.json(item.metadata)
    else:
        st.info("Select a loader format and click **Execute Document Loader** to view the resulting Document objects.")

st.divider()

# --- SECTION 13: TECHNICAL INTERVIEW PREPARATION ---
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why should tabular data (CSV / SQL) be formatted into natural language key-value pairs before embedding?",
     "Embedding models are trained on continuous natural language prose. If you embed raw comma-separated text ('101,Alice,Architect,95000'), the embedding model struggles to understand column-value associations. Formatting rows into natural language ('Employee ID is 101, Name is Alice, Role is Architect') aligns with transformer attention patterns, dramatically improving semantic retrieval."),
    ("What are the key tradeoffs between PyPDFLoader, PDFPlumberLoader, and UnstructuredPDFLoader?",
     "PyPDFLoader is fast and lightweight (pure Python) but fails on complex multi-column layouts and scanned images. PDFPlumberLoader is slower but excels at extracting structured tables. UnstructuredPDFLoader uses machine learning models to detect layout hierarchies and OCR scanned pages, offering the highest quality at the expense of significant CPU compute and memory."),
    ("How does WebBaseLoader manage rate limiting and concurrency when scraping enterprise intranets?",
     "WebBaseLoader exposes requests_per_second and requests_kwargs parameters. In production, engineers configure concurrency throttles (e.g. 2-5 requests per second) and custom User-Agent headers to avoid triggering Web Application Firewall (WAF) blocks or crashing internal documentation web servers."),
    ("How does JSONLoader utilize jq syntax for data extraction?",
     "JSONLoader integrates the jq query engine. Using schemas like '.messages[].content' or '.users[] | {name: .name, bio: .biography}', developers can cherry-pick specific nested strings out of multi-megabyte JSON payloads, discarding irrelevant operational boilerplate before creating Document objects."),
    ("What happens when DirectoryLoader encounters a corrupted file in a batch of 10,000 documents?",
     "By default, an unhandled exception crashes the entire loading job. In production, DirectoryLoader must be initialized with silent_errors=True and show_progress=True, allowing the loader to log a warning for corrupted files and proceed with the remaining 9,999 documents.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Crashing on Encoding Mismatches with TextLoader. Hardcoding encoding='utf-8' crashes when encountering legacy Windows CP1252 or Latin-1 text files. Always set autodetect_encoding=True or wrap loading in fallback encoding try-catch blocks.")
    st.error("**Trap 2 (Breaking Error)**: Ingesting Giant Multi-Page PDFs as Single Documents. Using naive text extraction that dumps an entire 200-page PDF into a single Document object destroys retrieval precision. Always load page-by-page or chunk appropriately.")
with trap_c2:
    st.warning("**Trap 3 (Memory Warning)**: Unbounded Concurrency in DirectoryLoader. Setting use_multithreading=True with max_concurrency=64 on memory-intensive PDF loaders exhausts process RAM. Cap concurrency to 4-8 workers.")
    st.warning("**Trap 4 (Quality Warning)**: Ingesting Empty Whitespace Documents. Many PDFs have blank pages or decorative image pages with zero text. Ingesting empty Document objects clutters vector databases with null vectors.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Document Loader Role"):
        st.markdown("**Converts heterogeneous file and data sources into standardized LangChain Document objects.**")
    with st.expander("🎴 Flashcard 2: PyPDFLoader"):
        st.markdown("**Splits PDF page-by-page, capturing page numbers in metadata for verifiable citations.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: CSV Loader"):
        st.markdown("**Translates tabular rows into natural language key-value pairs for optimal embedding capture.**")
    with st.expander("🎴 Flashcard 4: JSONLoader + jq"):
        st.markdown("**Selectively plucks nested text fields from complex JSON trees using jq query syntax.**")

st.divider()

# --- SECTION 14: SELF-ASSESSMENT CHECKLIST ---
st.header("14. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural role of Document Loaders in RAG ingestion pipelines.", value=True)
chk2 = st.checkbox("I know how to select the right PDF loader based on formatting and layout complexity.", value=True)
chk3 = st.checkbox("I can configure CSVLoader to format tabular records into natural language documents.", value=True)
chk4 = st.checkbox("I understand how JSONLoader uses jq schemas to extract nested attributes.", value=True)
chk5 = st.checkbox("I know how to configure DirectoryLoader with multithreading and error suppression.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 Document Loader interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 12.3 Document Loaders!")
