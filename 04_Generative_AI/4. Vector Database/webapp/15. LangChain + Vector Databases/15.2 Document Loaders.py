import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="15.2 Document Loaders - Vector DB Curriculum",
    page_icon="📂",
    layout="wide"
)

# Title & Overview
st.title("15.2 LangChain Document Loaders: PDF, Text, CSV, Web & Directory Ingestion")
st.caption("Chapter 15: LangChain + Vector Databases | Domain: 4. Vector Database")
st.markdown("""
Before documents can be embedded or stored in a vector database, raw binary streams across files, APIs, and web pages 
must be parsed into standardized **`langchain_core.documents.Document`** objects.

LangChain provides over 150 specialized **Document Loaders** implementing the **`BaseLoader`** interface:
* Standardized protocol: every loader exposes `.load()` (eager list) and **`.lazy_load()`** (memory-efficient generator).
* Preserves critical source provenance: every loaded `Document` contains clean `page_content: str` alongside structured `metadata: dict`.
* Handles multi-format edge cases: two-column PDFs, tabular CSV column mappings, HTML DOM extraction, and recursive directory crawling.

In this lesson, we explore the five core production document loaders: **`PyPDFLoader`, `TextLoader`, `CSVLoader`, `WebBaseLoader`, and `DirectoryLoader`**, 
examine lazy loading mechanics for large corpora, and test an interactive Multi-Loader Simulator.
""")

st.divider()

# Section 1: The BaseLoader Abstraction & Lazy Loading
st.header("1. The BaseLoader Abstraction & Lazy Loading", divider="rainbow")

col_base1, col_base2 = col_base1, col_base2 = st.columns([1.1, 0.9])

with col_base1:
    st.markdown("""
    ### The BaseLoader Contract
    Every document loader in LangChain inherits from `BaseLoader` and implements two key methods:
    
    1. **Eager Loading: `.load() -> List[Document]`**:
       * Reads and parses all documents entirely into RAM before returning a standard Python list.
       * *Best For*: Small files (< 50 pages) and interactive scripts.
       * *Fatal Risk*: Calling `.load()` on a folder of 10,000 PDFs causes Linux Out-Of-Memory (OOM) process termination!
    
    2. **Lazy Streaming: `.lazy_load() -> Iterator[Document]`**:
       * Uses Python generators (`yield`) to parse and return **one document at a time**.
       * Allows streaming documents directly through a text splitter into a vector store in small batches.
       * Memory consumption stays flat at **under 50 MB**, regardless of whether you ingest 10 files or 1,000,000 files!
    """)

with col_base2:
    st.markdown("### Document Loader Processing Lifecycle")
    ldr_dot = """
    digraph LoaderArchitecture {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawFiles [label="Raw Data Sources\\nPDFs / CSVs / HTML / Directories", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        BaseLdr [label="BaseLoader Protocol\\n(Parser & Metadata Enricher)", fillcolor="#FEF3C7", color="#D97706"];

        subgraph cluster_modes {
            label="Loading Execution Modes"; style="dashed"; color="#059669";
            Eager [label=".load()\\nReturns full List[Document]\\n(High RAM on large corpora)", fillcolor="#FEE2E2", color="#DC2626"];
            Lazy [label=".lazy_load()\\nYields Document generator\\n(Constant O(1) Memory)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        }

        VectorStore [label="ChromaDB Collection\\nBatch Ingestion Pipeline", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        RawFiles -> BaseLdr;
        BaseLdr -> Eager;
        BaseLdr -> Lazy;
        Lazy -> VectorStore [label="Stream in batches of 100", style="bold"];
    }
    """
    st.graphviz_chart(ldr_dot, use_container_width=False)

st.divider()

# Section 2: The Five Core Production Document Loaders
st.header("2. The Five Core Production Document Loaders", divider="rainbow")

tab_l1, tab_l2, tab_l3, tab_l4, tab_l5 = st.tabs([
    "1. PDF Loaders (PyPDF / PyMuPDF)",
    "2. TextLoader (Plain / Markdown)",
    "3. CSVLoader (Tabular Data)",
    "4. WebBaseLoader (HTML Scraping)",
    "5. DirectoryLoader (Recursive Bulk)"
])

with tab_l1:
    st.markdown("### 1. PyPDFLoader & PyMuPDFLoader")
    st.markdown("""
    * **Mechanism**: Parses PDF files page-by-page. Each page generates a distinct `Document` with `metadata={"source": "...", "page": N}`.
    * **`PyMuPDFLoader` (Recommended)**: Powered by C++ MuPDF; 5x to 10x faster than pure-Python PyPDF.
    """)
    st.code("""from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./specs/architecture_whitepaper.pdf")

# Eager load:
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")
print(f"Page 1 metadata: {pages[0].metadata}")
# Output: {'source': './specs/architecture_whitepaper.pdf', 'page': 0, 'total_pages': 14}
""", language="python")

with tab_l2:
    st.markdown("### 2. TextLoader (Plain Text & Markdown)")
    st.markdown("""
    * **Mechanism**: Reads standard plain text or Markdown files into a single `Document`.
    * **Encoding Safety**: Always specify `encoding="utf-8"` to prevent Windows cp1252 decode errors.
    """)
    st.code("""from langchain_community.document_loaders import TextLoader

# Always specify explicit UTF-8 encoding
loader = TextLoader("./notes/incident_postmortem.md", encoding="utf-8")
docs = loader.load()

print(f"Loaded document chars: {len(docs[0].page_content)}")
print(f"Source file: {docs[0].metadata['source']}")
""", language="python")

with tab_l3:
    st.markdown("### 3. CSVLoader (Tabular to Document Conversion)")
    st.markdown("""
    * **Mechanism**: Transforms each tabular row into a separate `Document`.
    * **`source_column` Parameter**: Specifies which column value should serve as the source citation identifier (e.g., ticket ID or product SKU).
    """)
    st.code("""from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./data/customer_support_tickets.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"'
    },
    source_column="ticket_id"  # Uses ticket_id in document metadata['source']
)

ticket_docs = loader.load()
# Each row becomes formatted text:
# "ticket_id: 10482\ncategory: Billing\nissue: Double charged for subscription"
""", language="python")

with tab_l4:
    st.markdown("### 4. WebBaseLoader (HTML Web Page Scraping)")
    st.markdown("""
    * **Mechanism**: Fetches web pages via `urllib` / `requests` and parses DOM text using BeautifulSoup.
    * **Header Spoofing**: Set custom User-Agent headers to prevent being blocked by anti-bot firewalls.
    """)
    st.code("""from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://docs.docker.com/get-started/"],
    header_template={
        "User-Agent": "Enterprise-RAG-Crawler/1.0 (internal research bot)"
    }
)

# Strips scripts/styles and returns cleaned HTML text
web_docs = loader.load()
print(f"Page Title in Metadata: {web_docs[0].metadata.get('title')}")
""", language="python")

with tab_l5:
    st.markdown("### 5. DirectoryLoader (Recursive Bulk Ingestion)")
    st.markdown("""
    * **Mechanism**: Traverses a directory tree and delegates files to specialized loaders based on glob patterns.
    * **Multithreading**: Set `use_multithreading=True` to parse files across CPU cores simultaneously.
    """)
    st.code("""from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

# Recursively crawl all PDF files in knowledge base
dir_loader = DirectoryLoader(
    path="./knowledge_base",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    use_multithreading=True,
    max_concurrency=4,
    show_progress=True
)

# Stream documents lazily to prevent memory spikes
for doc in dir_loader.lazy_load():
    # Process document chunk and upsert to Chroma
    pass
""", language="python")

st.divider()

# Section 3: Lazy Streaming Ingestion Pattern into ChromaDB
st.header("3. Lazy Streaming Ingestion Pattern into ChromaDB", divider="rainbow")

st.markdown("""
When building enterprise pipelines, combine **`lazy_load()`** with **`RecursiveCharacterTextSplitter`** 
to stream millions of tokens into ChromaDB with bounded RAM:
""")

st.code("""from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Initialize VectorStore
vectorstore = Chroma(
    collection_name="enterprise_docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_vault"
)

# 2. Configure Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 3. Configure Lazy Loader
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)

# 4. Stream & Batch Ingest
batch_buffer = []
BATCH_LIMIT = 200

for doc in loader.lazy_load():
    # Split single document into chunks
    chunks = splitter.split_documents([doc])
    batch_buffer.extend(chunks)
    
    if len(batch_buffer) >= BATCH_LIMIT:
        vectorstore.add_documents(batch_buffer)
        print(f"Persisted batch of {len(batch_buffer)} chunks to ChromaDB.")
        batch_buffer.clear()

# Flush remaining chunks
if batch_buffer:
    vectorstore.add_documents(batch_buffer)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Multi-Loader Simulator
st.header("4. Interactive Workbench: Multi-Loader Simulator", divider="rainbow")

st.markdown("""
Select a document loader to observe how raw source data is extracted, how text is formatted into `page_content`, 
and what metadata attributes are automatically captured.
""")

col_sim_ldr, col_sim_sample = st.columns([1, 1.2])

with col_sim_ldr:
    st.subheader("1. Choose Document Loader")
    loader_choice = st.selectbox(
        "Select Loader Type:",
        [
            "PyMuPDFLoader (Multi-Page PDF)",
            "CSVLoader (CRM Support Tickets)",
            "WebBaseLoader (HTML Knowledge Wiki)",
            "TextLoader (Markdown Runbook)",
            "DirectoryLoader (Bulk File Tree)"
        ]
    )
    
    load_mode = st.radio("Execution Mode:", ["Lazy Generator (.lazy_load())", "Eager Memory List (.load())"])

with col_sim_sample:
    st.subheader("2. Simulated Ingestion Processing")
    
    if "PyMuPDF" in loader_choice:
        raw_sample = """Page 1: [CONFIDENTIAL]
Title: Zero-Trust Gateway Architecture
Envoy proxies enforce mTLS authentication across all service-to-service links.

Page 2:
Telemetry and metrics are forwarded to Prometheus via OpenTelemetry collectors."""
        parsed_docs = [
            {"content": "Zero-Trust Gateway Architecture\nEnvoy proxies enforce mTLS authentication across all service-to-service links.", "meta": {"source": "gateway_arch.pdf", "page": 0, "total_pages": 2}},
            {"content": "Telemetry and metrics are forwarded to Prometheus via OpenTelemetry collectors.", "meta": {"source": "gateway_arch.pdf", "page": 1, "total_pages": 2}}
        ]
    elif "CSVLoader" in loader_choice:
        raw_sample = """ticket_id,priority,category,description
TICK-801,High,Auth,User unable to complete Okta SSO SAML handshake
TICK-802,Low,Billing,Invoice request for Q3 enterprise support tier"""
        parsed_docs = [
            {"content": "ticket_id: TICK-801\npriority: High\ncategory: Auth\ndescription: User unable to complete Okta SSO SAML handshake", "meta": {"source": "TICK-801", "row": 0}},
            {"content": "ticket_id: TICK-802\npriority: Low\ncategory: Billing\ndescription: Invoice request for Q3 enterprise support tier", "meta": {"source": "TICK-802", "row": 1}}
        ]
    elif "WebBaseLoader" in loader_choice:
        raw_sample = """<html><head><title>Docker Ingress</title></head><body>
<h1>Ingress Routing</h1><p>Docker swarm routes traffic using VIPs.</p>
</body></html>"""
        parsed_docs = [
            {"content": "Ingress Routing\nDocker swarm routes traffic using VIPs.", "meta": {"source": "https://docs.docker.internal/ingress", "title": "Docker Ingress"}}
        ]
    elif "TextLoader" in loader_choice:
        raw_sample = """# SRE Incident Postmortem: INC-2024-09
**Impact**: 14 minutes database connection pool exhaustion.
**Fix**: Increased maximum pool connections from 50 to 200."""
        parsed_docs = [
            {"content": raw_sample.strip(), "meta": {"source": "inc_2024_09.md", "encoding": "utf-8"}}
        ]
    else:  # DirectoryLoader
        raw_sample = """Directory Tree:
knowledge_base/
├── security/
│   ├── mtls_policy.pdf (2 pages)
│   └── sso_runbook.md (1 page)
└── infrastructure/
    └── k8s_ingress.pdf (4 pages)"""
        parsed_docs = [
            {"content": "mTLS Policy: Client certs mandatory for ingress.", "meta": {"source": "knowledge_base/security/mtls_policy.pdf", "page": 0}},
            {"content": "SSO Runbook: Okta SAML callback URL configuration.", "meta": {"source": "knowledge_base/security/sso_runbook.md"}},
            {"content": "K8s Ingress: Envoy reverse proxy gateway spec.", "meta": {"source": "knowledge_base/infrastructure/k8s_ingress.pdf", "page": 0}}
        ]

    st.text_area("Raw Source Representation:", value=raw_sample, height=140)

st.divider()

col_out_docs, col_out_stats = st.columns([1.2, 0.8])

with col_out_docs:
    st.subheader(f"Output `List[Document]` ({len(parsed_docs)} Documents Generated)")
    for idx, d in enumerate(parsed_docs):
        with st.expander(f"📦 Document {idx+1}: {d['meta']['source']}", expanded=True):
            st.markdown(f"**`page_content`**:\n\n{d['content']}")
            st.json(d["meta"])

with col_out_stats:
    st.subheader("Loader Efficiency Audit")
    mem_est = "0.4 MB (O(1) Streaming)" if "Lazy" in load_mode else f"{len(parsed_docs) * 1.2:.1f} MB (In-Memory List)"
    st.metric(label="Memory Footprint", value=mem_est)
    st.metric(label="Loaded Documents Count", value=len(parsed_docs))
    st.success("✅ Metadata integrity verified: Provenance attributes attached to every chunk.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why should production ingestion pipelines strictly use `.lazy_load()` instead of `.load()`?"):
    st.markdown("""
    **Answer**:
    * `.load()` loads all extracted documents into a monolithic Python list in memory (`return list(self.lazy_load())`). For an enterprise corpus of 50,000 PDFs, holding hundreds of thousands of document objects in memory consumes gigabytes of heap RAM, triggering Linux Out-Of-Memory (OOM) killer terminations.
    * `.lazy_load()` returns a Python **generator** that reads and yields one document at a time. By piping this generator into a batch buffer of size 200, the pipeline maintains a constant, tiny memory footprint (`O(1)` memory), scaling safely to millions of documents.
    """)

with st.expander("Q2: What is the `source_column` parameter in LangChain's `CSVLoader` and why is it crucial for RAG?"):
    st.markdown("""
    **Answer**:
    By default, `CSVLoader` populates `doc.metadata["source"]` with the filesystem path of the CSV file (`./data/tickets.csv`). 
    * If you ingest 10,000 support tickets, every single ticket will have the exact same generic source path!
    * Setting `source_column="ticket_id"` instructs the loader to use the unique primary key of that specific row (e.g. `TICK-8012`) as the source metadata.
    * When the downstream LLM cites sources, it can cite `[Ticket TICK-8012]` directly rather than a useless generic file name.
    """)

with st.expander("Q3: Why is `PyMuPDFLoader` significantly preferred over standard `PyPDFLoader`?"):
    st.markdown("""
    **Answer**:
    * `PyPDFLoader` is written in pure Python. It is CPU-bound, struggles with complex fonts, and can take 2-4 seconds per page on media-heavy PDFs.
    * `PyMuPDFLoader` uses Python C-bindings to the high-performance C/C++ MuPDF rendering engine. It parses PDFs **5x to 15x faster**, extracts layout blocks in accurate spatial reading order, and handles encrypted or corrupted PDF streams much more resiliently.
    """)

with st.expander("Q4: How do you handle encoding errors when loading heterogeneous text files with `TextLoader`?"):
    st.markdown("""
    **Answer**:
    On Windows systems, Python's default text encoding is often `cp1252` or `mbcs`, while files are encoded in `utf-8`. Attempting to load UTF-8 files triggers `UnicodeDecodeError`.
    
    **Production Best Practices**:
    1. Always explicitly pass `encoding="utf-8"`: `TextLoader(file_path, encoding="utf-8")`.
    2. For multi-encoding crawled archives, use `UnstructuredFileLoader` or wrap the loader in a charset detector (`chardet` / `charset_normalizer`) to detect the encoding prior to ingestion.
    """)

with st.expander("Q5: How does `DirectoryLoader` parallelize file ingestion across multiple CPU cores?"):
    st.markdown("""
    **Answer**:
    By setting `use_multithreading=True` and `max_concurrency=N` in `DirectoryLoader`:
    * LangChain uses Python's `concurrent.futures.ThreadPoolExecutor` to process files concurrently.
    * For I/O-bound tasks (downloading files or reading from disk) and C-extension operations (PyMuPDF parsing), multithreading releases the Global Interpreter Lock (GIL), cutting total ingestion duration by 4x to 8x on modern multi-core machines.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Calling `.load()` on Massive Directories**  
    Running `DirectoryLoader("./data").load()` on a 50 GB file tree exhausts container RAM and crashes the host. Always use `.lazy_load()`.
    """)
    st.warning("""
    **Trap 2: Forgetting `encoding="utf-8"` in TextLoader**  
    Relying on default system encoding causes pipelines to crash on non-ASCII characters (e.g. smart quotes or accents). Explicitly set `encoding="utf-8"`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Scraping Web Pages Without User-Agent Headers**  
    Running `WebBaseLoader` with default headers causes Cloudflare or AWS WAF to flag and block requests with HTTP 403 Forbidden. Configure a realistic User-Agent header.
    """)
    st.warning("""
    **Trap 4: Missing the `source_column` in CSV Ingestion**  
    Leaving `source_column` blank in `CSVLoader` tags every row with the identical CSV file path, destroying citation fidelity. Always specify a primary key column.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the difference between .load() and .lazy_load()?"):
        st.markdown("""
        **Execution Modes**:  
        * `.load()`: Loads all documents into memory at once (risk of OOM).  
        * `.lazy_load()`: Yields documents one at a time via generator (constant O(1) RAM).
        """)
    with st.expander("🗂️ Flashcard: Why choose PyMuPDFLoader?"):
        st.markdown("""
        **PyMuPDF Speed**:  
        Built on high-performance C++ MuPDF; parses PDFs 5x-10x faster than pure-Python PyPDF.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What does source_column do in CSVLoader?"):
        st.markdown("""
        **Row Citation**:  
        Designates a specific row column (e.g. ticket_id) as the document metadata source attribute.
        """)
    with st.expander("🗂️ Flashcard: How to parallelize DirectoryLoader?"):
        st.markdown("""
        **Multithreading**:  
        Configure `use_multithreading=True` and `max_concurrency=4` to utilize multiple CPU threads.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the BaseLoader abstraction and why `.lazy_load()` prevents OOM crashes.", value=True)
chk2 = st.checkbox("I know how to parse multi-page PDFs using high-performance `PyMuPDFLoader`.", value=True)
chk3 = st.checkbox("I can configure `CSVLoader` with custom delimiters and `source_column` row identifiers.", value=True)
chk4 = st.checkbox("I know how to scrape web pages safely with `WebBaseLoader` using custom User-Agent headers.", value=True)
chk5 = st.checkbox("I can configure multithreaded directory crawling via `DirectoryLoader`.", value=True)
chk6 = st.checkbox("I can implement a memory-bounded streaming ingestion pipeline into ChromaDB.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Document Loaders and are ready for Chapter 15.3: Text Splitters!")
