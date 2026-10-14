import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import json
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="12.1 Data Sources - Vector DB Curriculum",
    page_icon="📄",
    layout="wide"
)

# Title & Overview
st.title("12.1 Data Sources: Ingestion Architectures for PDF, DOCX, CSV, JSON, HTML & SQL")
st.caption("Chapter 12: Document Processing for Vector Databases | Domain: 4. Vector Database")
st.markdown("""
A vector database is only as good as the **quality, structure, and fidelity of the text fed into its embedding pipeline**.

In real-world enterprise applications, knowledge rarely arrives as pristine plain text strings. It exists trapped across 
diverse legacy formats:
* Complex **multi-column PDF reports** with headers, footers, and scanned charts.
* Relational **SQL database tables** and transactional **CSV rows**.
* Structured **JSON API responses** and semi-structured payloads.
* Rich **Word documents (`.docx`)** with hierarchical heading outlines.
* Cluttered **HTML web pages** packed with navigation menus, cookie banners, and scripts.

Transforming these multi-format sources into clean, semantically dense document chunks with rich metadata is the 
foundation of the vector ETL (Extract, Transform, Load) pipeline.

In this lesson, we explore production parsers for all 8 foundational data sources, examine metadata extraction best practices, 
and test an interactive Multi-Format Ingestion Workbench.
""")

st.divider()

# Section 1: The Multi-Format Ingestion Architecture
st.header("1. The Multi-Format Ingestion Architecture", divider="rainbow")

col_arch1, col_arch2 = col_arch1, col_arch2 = st.columns([1.1, 0.9])

with col_arch1:
    st.markdown("""
    ### The Three Ingestion Modalities
    1. **Unstructured Narrative (PDF, DOCX, TXT, HTML)**:
       * Primary goal: Extract linear text flow while preserving semantic hierarchy (Titles > H1 > H2 > Paragraphs) and stripping non-content noise (page numbers, headers, ads).
    2. **Structured & Semi-Structured (CSV, JSON, SQL Records)**:
       * Primary goal: Transform tabular rows into coherent natural language descriptions or key-value narratives so embedding models can capture semantic relationships.
    3. **Live Streaming Sources (Web crawlers, Database CDC)**:
       * Primary goal: Listen to Change Data Capture (CDC) events via Kafka or Webhooks and automatically insert, update, or tombstone vectors in real time.
    """)

with col_arch2:
    st.markdown("### Document Parsing & Enrichment Pipeline")
    flow_dot = """
    digraph IngestionFlow {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Sources [label="Raw Ingestion Sources\\nPDF / DOCX / CSV / SQL", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Parsers [label="Specialized Parser\\n(PyMuPDF / docx / BeautifulSoup)", fillcolor="#FEF3C7", color="#D97706"];

        Sanitizer [label="Text Sanitizer\\nStrip Boilerplate & Noise", fillcolor="#D1FAE5", color="#059669"];

        MetadataEngine [label="Metadata Enricher\\nAttach file, page, author, date", fillcolor="#E0E7FF", color="#4338CA"];

        ChromaPayload [label="ChromaDB Ready\\n(ids, docs, metadatas)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Sources -> Parsers;
        Parsers -> Sanitizer;
        Sanitizer -> MetadataEngine;
        MetadataEngine -> ChromaPayload;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# Section 2: Code Patterns for the 8 Core Data Sources
st.header("2. Code Patterns for the Eight Core Data Sources", divider="rainbow")

tab_src1, tab_src2, tab_src3, tab_src4 = st.tabs([
    "1. PDF & Word Documents",
    "2. Tabular CSV & Database Records",
    "3. JSON Payloads & API Data",
    "4. HTML Web Pages & Plain Text"
])

with tab_src1:
    st.markdown("### PDF Extraction (PyMuPDF / fitz)")
    st.code("""import fitz  # PyMuPDF

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(file_path)
    extracted_chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if len(text) > 30:  # Skip blank pages
            extracted_chunks.append({
                "text": text,
                "metadata": {
                    "source": file_path,
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "file_type": "pdf"
                }
            })
    return extracted_chunks
""", language="python")

    st.markdown("### Word Document Extraction (python-docx)")
    st.code("""import docx

def extract_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = docx.Document(file_path)
    paragraphs = []
    
    for i, p in enumerate(doc.paragraphs):
        clean_text = p.text.strip()
        if clean_text:
            paragraphs.append({
                "text": clean_text,
                "metadata": {
                    "source": file_path,
                    "paragraph_index": i,
                    "style": p.style.name,
                    "file_type": "docx"
                }
            })
    return paragraphs
""", language="python")

with tab_src2:
    st.markdown("### CSV Tabular Linearization")
    st.markdown("""
    Directly embedding raw CSV rows (`"102, Acme Corp, 45000, active"`) yields poor similarity because embedding models 
    struggle to understand positional column semantics without headers.  
    **Best Practice**: Linearize rows into structured descriptive sentences:
    """)
    st.code("""import pandas as pd

def process_csv_for_vectors(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    records = []
    
    for idx, row in df.iterrows():
        # Linearize into natural language description
        narrative = (
            f"Customer Account: {row['company_name']}. "
            f"Industry: {row['industry']}. "
            f"Annual Contract Value: ${row['acv']:,}. "
            f"Account Status: {row['status']}. "
            f"Key Notes: {row['notes']}"
        )
        records.append({
            "id": f"cust_{row['customer_id']}",
            "text": narrative,
            "metadata": {
                "customer_id": int(row["customer_id"]),
                "industry": str(row["industry"]),
                "acv": float(row["acv"]),
                "status": str(row["status"])
            }
        })
    return records
""", language="python")

    st.markdown("### Database Records (PostgreSQL / SQLite via SQLAlchemy)")
    st.code("""from sqlalchemy import create_engine, text

def ingest_sql_table(db_url: str, table_name: str) -> List[Dict[str, Any]]:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT id, title, body, author, created_at FROM {table_name}"))
        rows = result.fetchall()
        
    return [
        {
            "id": f"sql_{r.id}",
            "text": f"Title: {r.title}\\nContent: {r.body}",
            "metadata": {"author": r.author, "created_at": str(r.created_at)}
        }
        for r in rows
    ]
""", language="python")

with tab_src3:
    st.markdown("### JSON Hierarchical Serialization")
    st.markdown("""
    Nested JSON files (e.g., API payloads, Jira tickets) must be flattened and transformed into clear key-value strings:
    """)
    st.code("""import json

def serialize_json_object(data: Dict[str, Any], root_key: str = "") -> str:
    \"\"\"Recursively transforms nested JSON dict into readable key-value narrative.\"\"\"
    lines = []
    for k, v in data.items():
        full_key = f"{root_key}.{k}" if root_key else k
        if isinstance(v, dict):
            lines.append(serialize_json_object(v, full_key))
        elif isinstance(v, list):
            lines.append(f"{full_key}: " + ", ".join(str(item) for item in v))
        else:
            lines.append(f"{full_key}: {v}")
    return "\\n".join(lines)

# Example output:
# product.title: Quantum Server X
# product.specs.ram_gb: 128
# product.specs.cpu_cores: 64
""", language="python")

with tab_src4:
    st.markdown("### HTML Web Page Extraction (BeautifulSoup + Readability)")
    st.code("""from bs4 import BeautifulSoup
import re

def clean_html_webpage(html_content: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Strip scripts, stylesheets, navbars, and footers
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
        
    # Extract page title
    title = soup.title.string.strip() if soup.title else "Untitled Page"
    
    # Extract plain text with normalized whitespace
    raw_text = soup.get_text(separator=" ")
    clean_text = re.sub(r"\\s+", " ", raw_text).strip()
    
    return {
        "text": f"Page Title: {title}\\n\\n{clean_text}",
        "metadata": {
            "source_url": url,
            "title": title,
            "char_count": len(clean_text)
        }
    }
""", language="python")

st.divider()

# Section 3: Essential Metadata Provenance Design
st.header("3. Essential Metadata Provenance Design", divider="rainbow")

st.markdown("""
When parsing any raw data source, you must attach **Provenance Metadata** to the generated chunks. 
Provenance metadata enables citation generation, access control, and source auditing.
""")

col_meta_req, col_meta_code = st.columns([1, 1])

with col_meta_req:
    st.markdown("""
    ### The 6 Core Enterprise Metadata Attributes
    Every document chunk should capture:
    1. **`source`**: File path, URL, or database table origin.
    2. **`doc_id` / `parent_id`**: Unique ID of the overarching document to group related chunks.
    3. **`chunk_index`**: Zero-based ordinal index of the chunk within the document (`0, 1, 2...`).
    4. **`page_number`**: Physical page (for PDFs/DOCX) to generate user citations (`[Page 14]`).
    5. **`created_timestamp`**: Unix epoch integer representing publication or ingestion time.
    6. **`access_role` / `tenant_id`**: Clearance tags for deterministic security filtering.
    """)

with col_meta_code:
    st.markdown("### The Provenance Payload Schema")
    st.code("""# Standardized ChromaDB Ingestion Payload
chroma_payload = {
    "ids": [f"doc_{doc_id}_chunk_{i}"],
    "documents": [chunk_text],
    "metadatas": [{
        "source": "contracts/vendor_agreement_2024.pdf",
        "parent_doc_id": "VEND-8812",
        "chunk_index": i,
        "page": 4,
        "total_pages": 18,
        "created_epoch": 1709251200,
        "author": "Legal Team",
        "tenant_id": "corp_us_west",
        "clearance_level": 2
    }]
}
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Multi-Format Parser & Ingestion Lab
st.header("4. Interactive Workbench: Multi-Format Parser & Ingestion Lab", divider="rainbow")

st.markdown("""
Test real-time parsing and metadata serialization across different data sources. 
Select a data format to view how raw input is transformed into a Chroma-ready chunk and metadata object.
""")

col_lab_sel, col_lab_sample = st.columns([1, 1.2])

with col_lab_sel:
    st.subheader("1. Select Data Source Type")
    source_type = st.selectbox(
        "Choose Ingestion Format:",
        [
            "PDF Document (Technical Specification)",
            "CSV Database Export (Customer CRM)",
            "JSON API Response (Product Catalog)",
            "HTML Web Article (Knowledge Base)",
            "Plain Text / Markdown Note"
        ]
    )

with col_lab_sample:
    st.subheader("2. Raw Source Data Input")
    
    if source_type == "PDF Document (Technical Specification)":
        raw_input_text = """[Header: Confidential - Project Titan]
Page 4 of 12

1.3 Kubernetes Ingress Architecture
The cluster routes ingress traffic through an Envoy proxy gateway configured with mTLS. 
Each ingress route enforces rate limiting of 500 requests per minute per tenant API key.
Telemetry is exported to Prometheus via OpenTelemetry sidecars.

[Footer: Copyright 2024 Titan Systems - Page 4]"""
    elif source_type == "CSV Database Export (Customer CRM)":
        raw_input_text = """customer_id,name,tier,acv,region,status
10482,Acme Analytics,Enterprise,120000,US-West,Active
10483,Global Logistics,Mid-Market,45000,EU-Central,Active"""
    elif source_type == "JSON API Response (Product Catalog)":
        raw_input_text = json.dumps({
            "product_id": "HW-GPU-990",
            "name": "Nvidia H100 Tensor Core GPU",
            "category": "Datacenter Hardware",
            "specs": {
                "memory_gb": 80,
                "memory_type": "HBM3",
                "interconnect_gbps": 900
            },
            "pricing": {"msrp": 30000, "currency": "USD"}
        }, indent=2)
    elif source_type == "HTML Web Article (Knowledge Base)":
        raw_input_text = """<html>
<head><title>Password Reset Protocol</title></head>
<body>
<nav><a href="/home">Home</a> | <a href="/login">Login</a></nav>
<h1>Password Reset Protocol</h1>
<p>To reset an active directory password, submit a ticket to the IT Helpdesk with your employee badge ID.</p>
<p>Temporary passwords expire after 24 hours.</p>
<footer>Cookie Policy | Privacy Terms</footer>
</body></html>"""
    else:
        raw_input_text = """# Incident Post-Mortem: OUTAGE-2024-03
**Date**: 2024-03-12  
**Author**: SRE Team  
**Root Cause**: Memory leak in Redis cache pod caused connection pool starvation."""

    st.code(raw_input_text, language="html" if "HTML" in source_type else ("json" if "JSON" in source_type else "text"))

# Parsing Engine
st.divider()

col_out_chunk, col_out_meta = st.columns([1.2, 0.8])

with col_out_chunk:
    st.subheader("3. Transformed Document Text Chunk")
    
    if "PDF" in source_type:
        # Strip headers and footers
        lines = [l for l in raw_input_text.splitlines() if not l.startswith("[Header") and not l.startswith("[Footer")]
        parsed_chunk = "\n".join(lines).strip()
        parsed_meta = {
            "source": "specs/project_titan_spec.pdf",
            "page": 4,
            "total_pages": 12,
            "section": "1.3 Kubernetes Ingress Architecture",
            "format": "pdf"
        }
    elif "CSV" in source_type:
        parsed_chunk = (
            "Customer Record 10482: Acme Analytics. Tier: Enterprise. "
            "Annual Contract Value: $120,000. Region: US-West. Status: Active."
        )
        parsed_meta = {
            "source": "exports/crm_customers.csv",
            "customer_id": 10482,
            "tier": "Enterprise",
            "acv": 120000.0,
            "region": "US-West",
            "format": "csv"
        }
    elif "JSON" in source_type:
        parsed_chunk = (
            "Product Name: Nvidia H100 Tensor Core GPU\n"
            "Category: Datacenter Hardware\n"
            "Specs: 80GB HBM3 Memory, 900 GB/s Interconnect\n"
            "Pricing MSRP: $30,000 USD"
        )
        parsed_meta = {
            "source": "api/products/HW-GPU-990.json",
            "product_id": "HW-GPU-990",
            "category": "Datacenter Hardware",
            "price_usd": 30000.0,
            "format": "json"
        }
    elif "HTML" in source_type:
        parsed_chunk = (
            "Document Title: Password Reset Protocol\n\n"
            "To reset an active directory password, submit a ticket to the IT Helpdesk with your employee badge ID. "
            "Temporary passwords expire after 24 hours."
        )
        parsed_meta = {
            "source": "https://wiki.corp.internal/kb/password-reset",
            "title": "Password Reset Protocol",
            "format": "html"
        }
    else:
        parsed_chunk = raw_input_text.strip()
        parsed_meta = {
            "source": "postmortems/2024-03-outage.md",
            "author": "SRE Team",
            "incident_id": "OUTAGE-2024-03",
            "format": "markdown"
        }

    st.success("Extracted Semantic Text:")
    st.text_area("Chunk Content:", value=parsed_chunk, height=180)

with col_out_meta:
    st.subheader("4. Extracted Provenance Metadata")
    st.json(parsed_meta)
    
    st.info(f"""
    **Validation Check**:  
    * Text Length: `{len(parsed_chunk)} chars` (~`{len(parsed_chunk)//4} tokens`)  
    * Metadata Keys: `{len(parsed_meta)} primitive fields`  
    * Compatible with `collection.add()`: **Yes**
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is raw text extraction from multi-column PDFs notoriously difficult, and how do you resolve it?"):
    st.markdown("""
    **Answer**:
    PDF files are graphic layout instructions (displaying glyphs at exact `x, y` coordinates on a canvas), not structured semantic documents. 
    * **The Multi-Column Failure**: Naive text extractors read across horizontal coordinate lines, interleaving Column 1 and Column 2 (`Line 1 Col 1 -> Line 1 Col 2 -> Line 2 Col 1`), producing completely scrambled nonsense.
    * **Solution**: Use layout-aware visual parsers:
      1. **PyMuPDF (`get_text('blocks')`)**: Groups text into spatial bounding boxes and sorts blocks by reading order.
      2. **LayoutParser / Marker / Unstructured.io**: Uses lightweight computer vision models to identify columns, tables, headers, and paragraph boundaries before extracting text.
    """)

with st.expander("Q2: Why should tabular CSV rows be transformed into natural language sentences rather than embedded raw?"):
    st.markdown("""
    **Answer**:
    Embedding models (e.g., text-embedding-3, BGE) are trained on natural language syntax (subject, verb, object), not raw delimiter-separated values.
    * A raw row like `"14, 520, 89.2, True"` provides zero semantic context to the self-attention heads; the model has no way of knowing whether 520 is a price, a latency, or a customer ID.
    * Linearizing the row into `"Order ID 14: Latency was 89.2 ms with 520 concurrent connections and SSL enabled: True"` provides strong contextual cues that align directly with user queries (*"What was our latency under 500 connections?"*).
    """)

with st.expander("Q3: How do you handle scanned PDF documents that contain text rendered as images?"):
    st.markdown("""
    **Answer**:
    Standard text extractors return empty strings for scanned documents. 
    The ingestion pipeline must incorporate an **Optical Character Recognition (OCR) fallback**:
    1. Check extracted text density: If a PDF page has `< 20` characters, flag it as a raster scan.
    2. Render the page to a 300 DPI image (`page.get_pixmap()`).
    3. Pass the image through an OCR engine (Tesseract, PaddleOCR, or Amazon Textract).
    4. Store an ingestion metadata tag: `{"is_ocr": True}` for auditability.
    """)

with st.expander("Q4: What is the risk of leaving HTML tags and JavaScript inside webpage vector chunks?"):
    st.markdown("""
    **Answer**:
    1. **Context Window Waste**: In RAG, LLM context windows are limited. Injecting `<script>`, `<div>`, and inline CSS wastes tokens on irrelevant syntax.
    2. **Semantic Contamination**: Boilerplate words like `"Home"`, `"Contact Us"`, `"Privacy Policy"`, or class names like `class="btn-primary"` distort vector embeddings, pulling the chunk toward web development concepts rather than the page's actual subject matter.
    3. **Token Limit Truncation**: HTML bloat can push valuable factual paragraphs past the embedding model's maximum token limit (e.g. 512 tokens), permanently discarding core content.
    """)

with st.expander("Q5: How do you design an incremental Change Data Capture (CDC) ingestion pipeline from an SQL database to ChromaDB?"):
    st.markdown("""
    **Answer**:
    1. **WAL / CDC Listener**: Use Debezium or PostgreSQL logical replication to stream row changes (`INSERT`, `UPDATE`, `DELETE`) to a message queue (Kafka / RabbitMQ).
    2. **Event Consumer**:
       * On `INSERT` or `UPDATE`: Format the row into a text chunk, generate the embedding, and execute `collection.upsert(ids=[row_id], documents=[chunk], metadatas=[meta])`.
       * On `DELETE`: Execute `collection.delete(ids=[row_id])` to prune stale vectors immediately.
    3. Guarantees that vector search results reflect relational database state within seconds.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Retaining Running Headers and Footers Across Every Page**  
    Leaving `"Confidential - Company Internal - Page X"` in every extracted PDF chunk pollutes embeddings with identical noise, degrading vector distinction. Strip repetitive headers.
    """)
    st.warning("""
    **Trap 2: Ignoring Document Table Structures**  
    Flattening complex multi-column tables into naive space-separated text scrambles row and column associations. Extract tables using specialized tools (`pdfplumber` / markdown tables).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Ingesting Raw JSON Keys Without Schema Context**  
    Extracting raw JSON values without their parent key hierarchy loses critical meaning (e.g. extracting `"price": 10` without knowing whether it belongs to `"wholesale"` or `"retail"`).
    """)
    st.warning("""
    **Trap 4: Forgetting to Record the Source Page Number**  
    Failing to record the physical page number in chunk metadata makes it impossible for the downstream RAG assistant to provide clickable page citations to users.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: How to handle multi-column PDFs?"):
        st.markdown("""
        **Multi-Column Solution**:  
        Use layout-aware parsers (PyMuPDF blocks or LayoutParser) to sort text by visual bounding boxes, not raw coordinate lines.
        """)
    with st.expander("🗂️ Flashcard: How to embed CSV rows?"):
        st.markdown("""
        **Linearization**:  
        Transform tabular rows into natural language sentences containing column headers to provide semantic context.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is essential for chunk metadata?"):
        st.markdown("""
        **Core Metadata**:  
        `source`, `doc_id`, `chunk_index`, `page_number`, `created_timestamp`, and `tenant_id`.
        """)
    with st.expander("🗂️ Flashcard: Why strip HTML scripts and navbars?"):
        st.markdown("""
        **Boilerplate Stripping**:  
        Prevents wasting context window tokens and stops web navigation keywords from contaminating semantic vectors.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to extract text and page metadata from PDF, DOCX, CSV, JSON, HTML, and SQL.", value=True)
chk2 = st.checkbox("I understand why multi-column PDF layouts require layout-aware block extraction.", value=True)
chk3 = st.checkbox("I can linearize structured CSV rows into semantically descriptive natural language sentences.", value=True)
chk4 = st.checkbox("I know how to clean HTML pages using BeautifulSoup to strip scripts and navigation noise.", value=True)
chk5 = st.checkbox("I understand the 6 essential provenance metadata attributes required for enterprise RAG.", value=True)
chk6 = st.checkbox("I can design an incremental Change Data Capture (CDC) pipeline from SQL to a vector database.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Data Sources and are ready for Chapter 12.2: Document Cleaning & Normalization!")
