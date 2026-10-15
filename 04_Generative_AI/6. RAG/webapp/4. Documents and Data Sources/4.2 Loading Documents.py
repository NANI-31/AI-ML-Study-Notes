import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="4.2 Loading Documents - RAG Master Curriculum",
    page_icon="📥",
    layout="wide"
)

# Title & Overview
st.title("4.2 Loading Documents: PDF, Web, CSV, Directory, Database, & API Connectors")
st.caption("Chapter 4: Documents and Data Sources | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Document Loaders are the foundational extractors of the RAG ingestion lifecycle. 
Their sole mission is to ingest raw, heterogeneous data from diverse storage backends—local files, 
cloud buckets (S3, GCS), corporate wikis (Confluence, Notion), relational databases, and live web pages—and 
standardize them into a unified list of **`Document` objects** pairing clean `page_content` with rich `metadata`.
Choosing the wrong loader can ruin a RAG pipeline before chunking even begins: 
using basic HTML scrapers on client-rendered React applications yields empty text, 
and using slow pure-Python PDF parsers on a 500-page prospectus can take minutes instead of seconds.
This lesson covers loader architectures, performance benchmarks, lazy loading, and multi-source connectors.
""")

st.divider()

# Section 1: The Loader Landscape & Performance Benchmarks
st.header("1. The Enterprise Document Loader Landscape", divider="rainbow")
st.markdown("""
How different specialized loaders extract data across enterprise sources:
""")

loaders_table = pd.DataFrame([
    {
        "Data Source": "1. Standard PDF Documents",
        "Recommended Loader": "PyMuPDFLoader (`fitz`) / PyPDFLoader",
        "Throughput Speed": "Ultra-Fast (~10-20ms per page via C-bindings)",
        "Table & Layout Extraction": "Good for native text; struggles with complex bordered tables.",
        "Key Production Consideration": "Use `PyMuPDF` over standard `pypdf` for 10x faster ingestion speed."
    },
    {
        "Data Source": "2. Complex / Scanned PDFs",
        "Recommended Loader": "PDFPlumberLoader / LlamaParse / Unstructured",
        "Throughput Speed": "Moderate to Slow (~200ms - 2s per page with OCR)",
        "Table & Layout Extraction": "Superior! Extracts tables directly into clean Markdown grids.",
        "Key Production Consideration": "Mandatory for annual financial 10-K reports and scanned invoices."
    },
    {
        "Data Source": "3. Static Web Pages",
        "Recommended Loader": "TrafilaturaLoader / WebBaseLoader (BeautifulSoup)",
        "Throughput Speed": "Fast (~100-300ms network latency)",
        "Table & Layout Extraction": "Strips boilerplate (navbars, ads, footers) automatically.",
        "Key Production Consideration": "Fails on client-rendered SPAs (React/Vue) where content loads via JS."
    },
    {
        "Data Source": "4. Dynamic Web Pages (SPAs)",
        "Recommended Loader": "PlaywrightURLLoader / SeleniumURLLoader",
        "Throughput Speed": "Slow (~1.5s - 3s per page due to browser rendering)",
        "Table & Layout Extraction": "Renders JavaScript; captures exact DOM state seen by human users.",
        "Key Production Consideration": "Requires headless Chromium binary; higher memory footprint."
    },
    {
        "Data Source": "5. Local File Directories",
        "Recommended Loader": "DirectoryLoader (with multithreading)",
        "Throughput Speed": "High (Parallel file I/O)",
        "Table & Layout Extraction": "Applies format-specific sub-loaders based on file extensions.",
        "Key Production Consideration": "Use `use_multithreading=True` and `silent_errors=True` to avoid crashing."
    },
    {
        "Data Source": "6. Relational SQL Databases",
        "Recommended Loader": "SQLDatabaseLoader / SQLAlchemy Query Loader",
        "Throughput Speed": "Database query latency",
        "Table & Layout Extraction": "Serializes SQL result rows into structured document records.",
        "Key Production Consideration": "Enforce strict row limits (`LIMIT 1000`) to avoid memory saturation."
    }
])

st.dataframe(loaders_table, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Lazy Loading vs. Eager Loading")
    st.markdown("""
    * **Eager Loading (`loader.load()`)**: Loads and parses the *entire* collection of files into memory all at once. If your directory contains 5,000 PDF documents, eager loading will exhaust server RAM and crash the process!
    * **Lazy Loading (`loader.lazy_load()`)**: Returns a Python generator that yields one `Document` at a time. This allows multi-gigabyte corpora to be processed sequentially in a constant memory footprint ($O(1)$ memory).
    """)

with col2:
    st.subheader("Multi-Source Ingestion Architecture")
    ingest_chart = """
    digraph LoaderArchitecture {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin=0.15];
        edge [fontname="Helvetica", fontsize=9];

        Sources [label="Diverse Enterprise Repositories\\n(PDFs, React Web, S3, SQL, Confluence)", fillcolor="#E0E7FF", color="#4338CA"];
        Loaders [label="Specialized Document Loaders\\n(PyMuPDF, Playwright, SQL, Directory)", fillcolor="#EDE9FE", color="#6D28D9"];
        Standard [label="Standard Document Objects\\n(page_content: str, metadata: dict)", fillcolor="#DCFCE7", color="#16A34A"];
        Pipeline [label="To Text Chunkers & Embedding Models", fillcolor="#D1FAE5", color="#059669"];

        Sources -> Loaders;
        Loaders -> Standard [label="Yields Documents (lazy_load)"];
        Standard -> Pipeline;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(ingest_chart, use_container_width=False)

st.divider()

# Section 2: Production Code: Multi-Source Custom Document Loader
st.header("2. Production Code: Pure Python Unified Document Loader", divider="rainbow")
st.markdown("""
Here is a complete, runnable custom loader demonstrating **lazy generation**, **metadata attachment**, and **multi-format handling**:
""")

st.code("""
from typing import Generator, Dict, Any, List

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(source='{self.metadata.get('source')}', chars={len(self.page_content)})"

class UnifiedEnterpriseLoader:
    def __init__(self):
        pass

    def load_from_text(self, text: str, source_uri: str) -> Document:
        \"\"\"Wraps raw string into standard Document with metadata.\"\"\"
        return Document(
            page_content=text.strip(),
            metadata={"source": source_uri, "type": "plaintext", "char_count": len(text)}
        )

    def load_from_sql_records(self, table_name: str, records: List[Dict[str, Any]]) -> List[Document]:
        \"\"\"Serializes database records into Document objects.\"\"\"
        docs = []
        for idx, row in enumerate(records):
            content = ". ".join(f"{k}: {v}" for k, v in row.items()) + "."
            docs.append(Document(
                page_content=content,
                metadata={"source": f"sql://{table_name}", "row_id": idx + 1, "type": "database_row"}
            ))
        return docs

    def lazy_load_batch(self, raw_items: List[Dict[str, str]]) -> Generator[Document, None, None]:
        \"\"\"Lazy streaming generator yielding documents one by one to save memory.\"\"\"
        for item in raw_items:
            yield Document(
                page_content=item["content"],
                metadata={"source": item["uri"], "title": item.get("title", "Untitled")}
            )

# Demonstration
loader = UnifiedEnterpriseLoader()

# 1. Text loading
doc1 = loader.load_from_text("Acme Corp SOC2 Security Compliance Overview.", "s3://compliance/soc2.txt")
print("Text Doc:", doc1)

# 2. Database loading
mock_sql_rows = [
    {"user_id": 101, "plan": "Enterprise", "seats": 50},
    {"user_id": 102, "plan": "Pro", "seats": 5}
]
sql_docs = loader.load_from_sql_records("subscriptions", mock_sql_rows)
print("\\nSQL Docs:", sql_docs)
print("Row 1 Content:", sql_docs[0].page_content)
""", language="python")

st.divider()

# Section 3: Interactive Loader Strategy Selector
st.header("3. Interactive Loader Strategy Selector", divider="rainbow")
st.markdown("""
Select your data source scenario to see the recommended loader library, dependencies, and code snippet:
""")

source_options = {
    "1,000 Local PDF Invoices": {
        "loader": "PyMuPDFLoader (with DirectoryLoader multithreading)",
        "install": "pip install pymupdf langchain-community",
        "why": "PyMuPDF is written in C and executes 10x-20x faster than pure-Python PyPDF, preventing hours of processing latency.",
        "code": "from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader\nloader = DirectoryLoader('invoices/', glob='*.pdf', loader_cls=PyMuPDFLoader, use_multithreading=True)"
    },
    "Annual 10-K Report with Complex Tables": {
        "loader": "PDFPlumberLoader or LlamaParse",
        "install": "pip install pdfplumber llama-parse",
        "why": "Standard loaders scramble multi-column financial balance sheets. PDFPlumber extracts border coordinates to output structured tables.",
        "code": "from langchain_community.document_loaders import PDFPlumberLoader\nloader = PDFPlumberLoader('reports/apple_10k.pdf')\ndocs = loader.load()"
    },
    "Client-Side React SPA Portal": {
        "loader": "PlaywrightURLLoader",
        "install": "pip install playwright && playwright install chromium",
        "why": "Standard BeautifulSoup scrapers only see empty HTML templates. Playwright spins up a headless browser to render JavaScript before scraping.",
        "code": "from langchain_community.document_loaders import PlaywrightURLLoader\nloader = PlaywrightURLLoader(urls=['https://app.corp.io/portal'])\ndocs = loader.load()"
    },
    "Live Production PostgreSQL Database": {
        "loader": "SQLDatabaseLoader (SQLAlchemy)",
        "install": "pip install sqlalchemy psycopg2-binary",
        "why": "Directly queries SQL tables and converts each row into a standardized Document object with primary key metadata.",
        "code": "from langchain_community.document_loaders import SQLDatabaseLoader\nloader = SQLDatabaseLoader('SELECT * FROM knowledge_wiki LIMIT 1000', db=sql_db)"
    }
}

selected_source = st.selectbox("Choose Data Ingestion Scenario:", list(source_options.keys()))
src_info = source_options[selected_source]

m1, m2 = st.columns(2)
m1.success(f"**Recommended Loader**: `{src_info['loader']}`")
m2.info(f"**Installation**: `{src_info['install']}`")

st.write(f"**Architectural Rationale**: {src_info['why']}")
st.code(src_info["code"], language="python")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Loader Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is PyMuPDF (`fitz`) dramatically faster than pure-Python `pypdf`?**
    * **Answer**: `pypdf` is implemented purely in interpreted Python, making byte parsing, decompression, and string decoding CPU-bound and slow. PyMuPDF is a Python wrapper around the MuPDF rendering engine written in high-performance native C. It performs PDF font parsing, stream decompression, and text rendering directly in compiled C, yielding 10x to 25x faster throughput on multi-page enterprise documents.
    
    **Q2: What is the risk of using `WebBaseLoader` on modern web applications?**
    * **Answer**: `WebBaseLoader` uses `urllib` and BeautifulSoup to fetch static HTML. Modern websites (built with React, Next.js, Vue, or Angular) load their content client-side via JavaScript APIs after the initial HTML shell is fetched. Scraping an SPA with `WebBaseLoader` returns an empty `div id="root"` shell with zero actual article text. Scraping dynamic SPAs requires headless browser automation (Playwright or Puppeteer).
    
    **Q3: How does `lazy_load()` protect microservices during batch ingestion?**
    * **Answer**: In Python, calling `load()` reads all documents into a single in-memory list (`List[Document]`). If ingesting a folder of 10,000 documents, holding all parsed strings in memory simultaneously triggers out-of-memory (OOM) fatal crashes. `lazy_load()` returns a generator yielding one document at a time. The pipeline can process, chunk, embed, and upsert each document before discarding it, maintaining a flat memory footprint regardless of dataset size.
    """)

with traps_tab:
    st.error("""
    **Trap 1: Scraping websites without rate-limiting and User-Agent headers**  
    Running multithreaded web scrapers against external documentation sites without custom User-Agent headers and delay throttling will get your corporate IP address blocked with HTTP 403 Forbidden or 429 Too Many Requests. Always respect robots.txt and add rate limits.
    """)
    st.error("""
    **Trap 2: Ignoring corrupted files in `DirectoryLoader`**  
    If a single PDF in a folder of 5,000 files is corrupted or password-protected, standard loaders crash the entire batch pipeline. Always set `silent_errors=True` so broken files are logged and skipped without halting ingestion.
    """)

with flashcards_tab:
    st.info("**PyMuPDF (fitz)**: A high-performance native C-based PDF extraction engine delivering the fastest text parsing throughput in the Python ecosystem.")
    st.info("**Lazy Loading**: An architectural pattern using generators to process documents sequentially, preventing out-of-memory crashes on large datasets.")
    st.info("**Headless Browser Scraping**: Using automated browser engines (Playwright, Selenium) to execute JavaScript and extract content from Single Page Applications (SPAs).")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
checklist = [
    "I understand the performance differences between PyPDF, PyMuPDF, and PDFPlumber.",
    "I know when headless browser scrapers (Playwright) are required over static scrapers.",
    "I understand why lazy loading (`lazy_load()`) is essential to prevent out-of-memory crashes.",
    "I know how to handle corrupted documents gracefully using `silent_errors=True`.",
    "I can implement a custom enterprise document loader in Python."
]

for item in checklist:
    st.checkbox(item, value=False)
