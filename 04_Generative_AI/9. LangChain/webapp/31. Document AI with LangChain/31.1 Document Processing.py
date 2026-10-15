import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="31.1 Document Processing - LangChain Architecture",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for document processing telemetry
st.markdown("""
<style>
    .doc-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-fmt {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DocumentParserSimulator:
    """
    Simulation engine showcasing multi-format document parsing,
    metadata preservation, and structural token densities.
    """
    FORMAT_PRESETS = {
        "PDF Invoice": {
            "ext": ".pdf",
            "loader": "PyPDFLoader / PDFPlumber",
            "raw_size_kb": 450,
            "extracted_text_sample": "INVOICE #9921\nVendor: Acme Cloud Services\nDate: 2026-10-12\nTotal Due: $4,250.00\nLine items: EC2 Instance ($2,100), RDS Cluster ($2,150)",
            "metadata": {"source": "invoice_9921.pdf", "page": 1, "producer": "Adobe PDF Engine"}
        },
        "Word Contract (.docx)": {
            "ext": ".docx",
            "loader": "Docx2txtLoader / Unstructured",
            "raw_size_kb": 120,
            "extracted_text_sample": "SECTION 4: INDEMNIFICATION\nThe Service Provider agrees to defend, indemnify, and hold harmless the Client against third-party IP claims.",
            "metadata": {"source": "master_services_agreement.docx", "author": "Legal Counsel", "paragraphs": 42}
        },
        "Excel Financials (.xlsx)": {
            "ext": ".xlsx",
            "loader": "UnstructuredExcelLoader",
            "raw_size_kb": 280,
            "extracted_text_sample": "Quarter: Q3-2026 | Revenue: $14.2M | OpEx: $9.8M | Net Margin: 31%\nQuarter: Q4-2026 | Revenue: $16.8M | OpEx: $10.5M | Net Margin: 37.5%",
            "metadata": {"source": "q3_q4_financials.xlsx", "sheet_name": "Summary_P&L", "rows": 120}
        },
        "CSV Customer Log (.csv)": {
            "ext": ".csv",
            "loader": "CSVLoader (Row-as-Doc)",
            "raw_size_kb": 65,
            "extracted_text_sample": "user_id: usr_881\nevent: checkout_completed\namount: 149.99\ncurrency: USD\ntimestamp: 2026-10-13T14:22:00Z",
            "metadata": {"source": "orders_export.csv", "row": 44, "columns": 5}
        },
        "Markdown README (.md)": {
            "ext": ".md",
            "loader": "MarkdownHeaderTextSplitter",
            "raw_size_kb": 35,
            "extracted_text_sample": "# Microservices Architecture\n## Authentication Service\nHandles OAuth2 JWT issuance with RS256 signing.",
            "metadata": {"source": "README.md", "header_level_1": "Microservices Architecture", "header_level_2": "Authentication Service"}
        },
        "HTML Knowledge Base (.html)": {
            "ext": ".html",
            "loader": "BSHTMLLoader / UnstructuredHTMLLoader",
            "raw_size_kb": 95,
            "extracted_text_sample": "Knowledge Article: Configuring SSO\n1. Navigate to Settings > Identity Provider.\n2. Paste SAML Metadata XML.\n3. Save and test assertion.",
            "metadata": {"source": "kb_sso_config.html", "title": "Configuring SSO", "language": "en"}
        },
        "JSON Event Payload (.json)": {
            "ext": ".json",
            "loader": "JSONLoader (with jq_schema)",
            "raw_size_kb": 18,
            "extracted_text_sample": '{"event_id": "evt_9912", "service": "billing", "action": "charge_failed", "customer_tier": "enterprise"}',
            "metadata": {"source": "audit_stream.json", "jq_pointer": ".events[]"}
        }
    }

    def parse_document(self, preset_key: str) -> Dict[str, Any]:
        doc = self.FORMAT_PRESETS.get(preset_key, self.FORMAT_PRESETS["PDF Invoice"])
        tokens = len(doc["extracted_text_sample"].split()) * 2
        return {
            "name": preset_key,
            "ext": doc["ext"],
            "loader": doc["loader"],
            "raw_size_kb": doc["raw_size_kb"],
            "text": doc["extracted_text_sample"],
            "metadata": doc["metadata"],
            "estimated_tokens": tokens
        }


# --- LESSON HEADER ---
st.title("31.1 Document Processing Across Heterogeneous Formats")
st.caption("Chapter 31: Document AI with LangChain | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Enterprise knowledge is rarely clean, plain text. Real-world corporate knowledge assets are scattered across
complex, semi-structured document formats: scanned and digital PDFs, Word agreements, multi-tab Excel sheets,
CSV database dumps, Markdown tech docs, raw HTML support pages, and nested JSON audit payloads.

In this lesson, you will master:
1. **PDF Processing**: Extracting multi-column text, metadata, and embedded tabular grids.
2. **Word Documents**: Preserving typographic hierarchy, headers, and bullet structures.
3. **Excel Files**: Serializing multi-sheet financial workbooks into LLM-digestible chunks.
4. **CSV Files**: Balancing row-as-document and table-as-document ingestion strategies.
5. **Markdown**: Leveraging structural header delimiters for semantic boundary splitting.
6. **HTML**: Stripping DOM boilerplate, scripts, and navigation chrome while preserving content.
7. **JSON**: Navigating deeply nested schemas using jq path pointers.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Heterogeneous Document Ingestion Engine", divider="rainbow")
st.markdown("""
A robust Document AI ingestion engine accepts heterogeneous file streams, routes each file to a dedicated
format loader, normalizes extracted text into standard LangChain `Document` schemas, and attaches immutable metadata.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    doc_proc_dot = """
    digraph DocProcessingBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_ingress_files {
            label="1. Ingress File Stream";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            raw_pdf [label="PDF Document\\nDigital or Scanned", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            raw_office [label="Office Files\\nWord (.docx) & Excel (.xlsx)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            raw_web_code [label="Web & Data\\nHTML, Markdown, CSV, JSON", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_format_parsers {
            label="2. Specialized LangChain Loaders";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            pypdf [label="PyPDFLoader / PDFPlumber\\nPage-level slicing & tables", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            docx_loader [label="Docx2txt / UnstructuredWord\\nParagraphs & heading styles", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            excel_loader [label="UnstructuredExcelLoader\\nSheet-to-table serialization", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            html_json_loader [label="BSHTMLLoader & JSONLoader\\nDOM cleaner & jq pointer", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_normalized_output {
            label="3. Standardized LangChain Document Schema";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            norm_doc [label="List[Document]\\npage_content: str\\nmetadata: {'source', 'format', 'page', 'author'}", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            downstream_rag [label="Downstream RAG & Splitters\\nReady for chunking & embedding", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        raw_pdf -> pypdf [label="PDF bytes"];
        raw_office -> docx_loader [label=".docx"];
        raw_office -> excel_loader [label=".xlsx"];
        raw_web_code -> html_json_loader [label=".html / .json"];
        pypdf -> norm_doc [label="Emits Documents"];
        docx_loader -> norm_doc [label="Emits Documents"];
        excel_loader -> norm_doc [label="Emits Documents"];
        html_json_loader -> norm_doc [label="Emits Documents"];
        norm_doc -> downstream_rag [label="Pipes to Chunkers"];
    }
    """
    st.graphviz_chart(doc_proc_dot, use_container_width=True)

# --- SECTION 2: PDF PROCESSING ---
st.header("2. PDF Processing", divider="rainbow")
st.markdown("""
PDFs represent the most ubiquitous and historically painful format in Document AI.
PDF files are not semantic document trees; they are drawing instructions specifying where glyphs and lines are rendered on a 2D canvas.
""")

col_pdf1, col_pdf2 = st.columns(2)
with col_pdf1:
    st.markdown("""
    #### Loader Selection Guide:
    - **`PyPDFLoader`**: Fast, lightweight, pure-Python. Excellent for modern digital PDFs with clean single-column text. Extracts text per page with `page` metadata.
    - **`PDFPlumberLoader`**: Preserves spatial layout, coordinates, and extracted table grids as markdown tables.
    - **`UnstructuredPDFLoader`**: Detects document elements (titles, narrative text, tables, headers, footers).
    - **`AmazonTextractPDFLoader` / Azure Document Intelligence**: Enterprise cloud OCR for scanned physical papers, handwritten notes, and complex multi-page financial forms.
    """)

with col_pdf2:
    with st.container(border=True):
        st.markdown("📄 **PyPDFLoader in Action**")
        st.code("""
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("annual_report_2026.pdf")
docs = loader.load()

# Each Document corresponds to a single page
print(f"Loaded {len(docs)} pages.")
print(docs[0].metadata) # {'source': 'annual_report_2026.pdf', 'page': 0}
        """, language="python")

# --- SECTION 3: WORD DOCUMENTS ---
st.header("3. Word Documents", divider="rainbow")
st.markdown("""
Microsoft Word files (`.docx`) contain rich XML structure underneath. Unlike PDFs, Word documents maintain
explicit paragraph boundaries, heading levels (Heading 1, Heading 2), and table definitions.
""")

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.markdown("""
    #### Key Capabilities:
    - **`Docx2txtLoader`**: Fast, dependency-free text extraction. Converts docx paragraphs into plain text while discarding images.
    - **`UnstructuredWordDocumentLoader`**: Preserves semantic hierarchy, converting Word heading styles into structured element tags.
    - **Metadata Extraction**: Extracts author, revision date, title, and page count from OpenXML metadata properties.
    """)

with col_w2:
    with st.container(border=True):
        st.markdown("📝 **Docx2txtLoader Usage**")
        st.code("""
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("contract_draft.docx")
docs = loader.load()

# Emits full document text with preserved paragraph breaks
print(docs[0].page_content[:200])
        """, language="python")

# --- SECTION 4: EXCEL FILES ---
st.header("4. Excel Files", divider="rainbow")
st.markdown("""
Spreadsheets (`.xlsx`, `.xls`) contain tabular grids across multiple sheets, formulas, and merged cells.
Dumping raw cell coordinates into an LLM causes severe confusion. Effective Excel processing serializes tables
into Markdown tables or natural language sentences.
""")

st.subheader("Comparison of Excel Serialization Strategies")
excel_comp = pd.DataFrame({
    "Strategy": ["CSV Flattening (Sheet-by-Sheet)", "Markdown Table Serialization", "Natural Language Row-Templating"],
    "How it Works": ["Converts each sheet into a comma-delimited string", "Converts cell ranges into `| Col1 | Col2 |` tables", "Converts row to sentence: `In Year X, Region Y produced Revenue Z.`"],
    "LLM Comprehension": ["Moderate (Comma noise in numeric data)", "High (Models excel at markdown table parsing)", "Highest (Zero formatting ambiguity for LLM)"],
    "Token Efficiency": ["High (Compact)", "Moderate (Markdown delimiters add tokens)", "Lowest (Verbose English prose)"]
})
st.dataframe(excel_comp, use_container_width=True)

# --- SECTION 5: CSV FILES ---
st.header("5. CSV Files", divider="rainbow")
st.markdown("""
Comma-Separated Values (CSV) are the standard export format for database tables and logs.
LangChain provides two distinct paradigms for CSV loading:
""")

col_csv1, col_csv2 = st.columns(2)
with col_csv1:
    with st.container(border=True):
        st.markdown("🎯 **Row-as-Document Pattern**")
        st.caption("Default `CSVLoader` behavior")
        st.markdown("""
        Each row becomes a distinct `Document`:
        ```text
        product_id: 1042
        name: Enterprise Firewall
        price: $2,400
        category: Security
        ```
        - **Advantage**: Perfect for semantic search over catalog items or customer tickets.
        """)

with col_csv2:
    with st.container(border=True):
        st.markdown("📊 **Table-as-Document Pattern**")
        st.caption("Loaded as single markdown/csv string")
        st.markdown("""
        All rows loaded together into a single context:
        - **Advantage**: Enables global aggregation questions (e.g. 'What was the average price?').
        - **Limitation**: Bounded by maximum model context length.
        """)

# --- SECTION 6: MARKDOWN ---
st.header("6. Markdown", divider="rainbow")
st.markdown("""
Markdown (`.md`) is the native language of LLMs and developer documentation.
Unlike unstructured text, Markdown contains explicit semantic headers (`#`, `##`, `###`) that provide
the ideal splitting boundaries for RAG chunkers.
""")

col_md1, col_md2 = st.columns(2)
with col_md1:
    st.markdown("""
    #### `MarkdownHeaderTextSplitter`
    Instead of chunking by fixed character count (which arbitrarily cuts sentences in half),
    `MarkdownHeaderTextSplitter` splits strictly along markdown headers and injects the header hierarchy
    directly into chunk metadata:
    ```json
    {
      "page_content": "Install with npm install @acme/core...",
      "metadata": {
        "Header 1": "Installation Guide",
        "Header 2": "Node.js Setup"
      }
    }
    ```
    """)

with col_md2:
    with st.container(border=True):
        st.markdown("📑 **Header-Aware Chunking Code**")
        st.code("""
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = splitter.split_text(markdown_content)
        """, language="python")

# --- SECTION 7: HTML ---
st.header("7. HTML", divider="rainbow")
st.markdown("""
Scraped web pages contain vast amounts of navigational noise, tracking scripts, CSS stylesheets, and cookie banners.
Ingesting raw HTML wastes tokens and confuses model reasoning.
""")

col_html1, col_html2 = st.columns(2)
with col_html1:
    st.markdown("""
    #### Cleaning Pipeline for HTML Documents:
    1. **DOM Parsing**: Parse with BeautifulSoup (`BSHTMLLoader`).
    2. **Element Stripping**: Automatically remove `<script>`, `<style>`, `<nav>`, `<footer>`, and `<header>` tags.
    3. **Semantic Extraction**: Extract the main content container (`<main>`, `<article>`, or `#content`).
    4. **Link & Anchor Preservation**: Convert `<a href="...">text</a>` into markdown links `[text](url)` to maintain research citations.
    """)

with col_html2:
    with st.container(border=True):
        st.markdown("🌐 **BSHTMLLoader in Action**")
        st.code("""
from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("web_article.html", open_encoding="utf-8")
docs = loader.load()

# Emits clean textual body with page title metadata
print(docs[0].metadata) # {'source': 'web_article.html', 'title': 'Cloud Security 2026'}
        """, language="python")

# --- SECTION 8: JSON ---
st.header("8. JSON", divider="rainbow")
st.markdown("""
JSON is the native data format of microservice APIs and event buses.
LangChain's `JSONLoader` allows developers to extract specific nested attributes using **jq query syntax**
rather than ingesting monolithic, bracket-heavy payloads.
""")

col_j1, col_j2 = st.columns(2)
with col_j1:
    st.markdown("""
    #### jq Pointer Selection
    Given a complex JSON payload:
    ```json
    {
      "results": [
        {"id": "A1", "summary": "DB Connection dropped"},
        {"id": "A2", "summary": "High memory warning"}
      ]
    }
    ```
    Specifying `jq_schema=".results[].summary"` extracts each incident summary as an independent `Document` object!
    """)

with col_j2:
    with st.container(border=True):
        st.markdown("📦 **JSONLoader with jq Syntax**")
        st.code("""
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="audit_events.json",
    jq_schema=".results[]",
    content_key="summary",
    metadata_func=lambda record, meta: {"event_id": record.get("id")}
)
docs = loader.load()
        """, language="python")

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Multi-Format Document Ingestion Router", divider="rainbow")
st.markdown("""
Below is a runnable multi-format document ingestion router demonstrating automated file extension detection,
loader dispatch, and unified metadata normalization.
""")

prod_doc_code = '''from typing import List, Dict, Any
from langchain_core.documents import Document

class MultiFormatDocumentRouter:
    """Production document dispatcher routing files by extension to specialized parsers."""

    def parse_content(self, filename: str, raw_content: str) -> List[Document]:
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if ext == "json":
            import json
            data = json.loads(raw_content)
            # Flatten list of dicts into Document stream
            if isinstance(data, list):
                return [Document(page_content=str(item), metadata={"source": filename, "index": i}) for i, item in enumerate(data)]
            return [Document(page_content=str(data), metadata={"source": filename})]
            
        elif ext == "md":
            # Header-aware markdown parsing
            return [Document(page_content=raw_content, metadata={"source": filename, "format": "markdown"})]
            
        elif ext == "csv":
            lines = raw_content.strip().split("\\n")
            header = lines[0] if lines else ""
            return [
                Document(page_content=line, metadata={"source": filename, "header": header, "row": i})
                for i, line in enumerate(lines[1:])
            ]
            
        else: # Fallback text parser
            return [Document(page_content=raw_content, metadata={"source": filename, "format": "raw_text"})]

router = MultiFormatDocumentRouter()
# Example invocation
documents = router.parse_content("config.json", \'[{"server": "us-east", "status": "active"}]\')
'''
st.code(prod_doc_code, language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Multi-Format Document Parser & Token Density Simulator", divider="rainbow")
st.markdown("""
Inspect how diverse enterprise document formats are transformed into standardized LangChain `Document` schemas.
Evaluate metadata preservation and extracted token densities across formats.
""")

parser_engine = DocumentParserSimulator()

col_pb1, col_pb2 = st.columns([1, 1])

with col_pb1:
    st.subheader("📁 Document Selection")
    chosen_preset = st.selectbox(
        "Choose an Enterprise Document Preset:",
        list(parser_engine.FORMAT_PRESETS.keys())
    )
    parse_btn = st.button("🚀 Ingest & Parse Document", type="primary", use_container_width=True)

with col_pb2:
    st.subheader("📊 Document Normalization Telemetry")
    if parse_btn:
        res = parser_engine.parse_document(chosen_preset)
        
        st.markdown(
            f"<div class='doc-card'><span class='badge-fmt'>{res['ext'].upper()} PARSED</span>"
            f"<br><br><b>Recommended Loader:</b> <code>{res['loader']}</code>"
            f"<br><b>Raw File Size:</b> {res['raw_size_kb']} KB | <b>Estimated Tokens:</b> {res['estimated_tokens']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("##### 📄 Extracted `page_content`")
        st.code(res["text"], language="text")

        st.markdown("##### 🏷️ Preserved Document `metadata`")
        st.json(res["metadata"])
    else:
        st.info("Select a document format preset and click 'Ingest & Parse Document' to inspect normalization.")

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is PDF text extraction notoriously difficult compared to HTML or Word documents?"):
    st.markdown("""
    **Answer**: PDF is fundamentally a visual page description language designed for printing, not semantic understanding.
    Text is stored as individual character glyphs positioned at coordinates (X, Y). A PDF has no intrinsic concept of
    paragraphs, multi-column reading order, or table structures. A naive parser reading left-to-right across a two-column
    scientific paper will interleave words from column 1 and column 2, destroying reading comprehension. Specialized
    layout-aware parsers (PDFPlumber, LayoutParser, OCR) are required to reconstruct reading order.
    """)

with st.expander("Q2: When should you load a CSV file using row-as-document vs table-as-document?"):
    st.markdown("""
    **Answer**: Use **Row-as-Document** (`CSVLoader`) when building semantic search over distinct entities (e.g., product catalogs,
    support tickets, employee profiles) where queries target individual items ('Find tickets regarding VPN timeout').
    Use **Table-as-Document** (or SQL database querying) when queries involve multi-row mathematical aggregation ('What was the
    average invoice amount in Q3?'), because semantic similarity search over individual row embeddings cannot perform mathematical summations.
    """)

with st.expander("Q3: How does `MarkdownHeaderTextSplitter` prevent semantic fragmentation in RAG?"):
    st.markdown("""
    **Answer**: Fixed-size character splitters arbitrarily slice text across paragraph and heading boundaries, separating code examples
    from their introductory explanations. `MarkdownHeaderTextSplitter` respects the author's logical structure, chunking along `#` and `##`
    delimiters and encoding the entire heading path into chunk metadata. This ensures that a chunk about 'Prerequisites' always carries
    metadata indicating it belongs under 'Chapter 3: Installation'.
    """)

with st.expander("Q4: Why is raw HTML ingestion discouraged in production RAG pipelines?"):
    st.markdown("""
    **Answer**: Raw HTML contains 60% to 80% non-content noise: `<script>`, `<style>`, navigation links, cookie banners, and CSS class names.
    This noise inflates embedding vector storage costs, dilutes semantic similarity scores, and exhausts context windows during generation.
    Production systems always clean HTML with DOM parsers (BeautifulSoup) or HTML-to-Markdown converters before chunking.
    """)

with st.expander("Q5: What role does `jq` play in LangChain's `JSONLoader`?"):
    st.markdown("""
    **Answer**: `jq` is a query language for JSON. In enterprise APIs, JSON payloads are heavily nested with operational metadata,
    wrappers, and headers. `JSONLoader` uses `jq_schema` pointers to zero in on the exact textual payload (e.g. `.data.items[].description`)
    and convert each matching object into an independent LangChain `Document`, filtering out irrelevant envelope metadata.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Scanned PDF Silent Failure)**: Attempting to parse a scanned image PDF with `PyPDFLoader`. The loader returns empty strings without throwing an error because it cannot read pixels. Always implement OCR fallbacks.")
    st.warning("**Trap 2 (Monolithic Document Ingestion)**: Loading an entire 300-page PDF into a single `Document` object and embedding it as one vector. Vectors cannot represent 300 pages of divergent topics; always split by page and chunk.")
with col_trap2:
    st.error("**Trap 3 (Dropping Metadata During Loading)**: Extracting text into a string and discarding the file path, page number, and author metadata. Downstream citations and multi-tenant access control become completely impossible.")
    st.warning("**Trap 4 (Memory Exhaustion on Large Spreadsheets)**: Loading a 2-million-row CSV directly into memory via `CSVLoader`. Use streaming iterators or database engines (DuckDB, SQLite) for large datasets.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the core difference between PyPDFLoader and PDFPlumber?")
        st.caption("Flip for answer:")
        st.info("**Back**: PyPDFLoader extracts raw digital text per page; PDFPlumber preserves 2D spatial layouts, coordinates, and extracts embedded table grids.")
    with st.container(border=True):
        st.markdown("**Front**: What does MarkdownHeaderTextSplitter do?")
        st.caption("Flip for answer:")
        st.info("**Back**: Splits markdown along header tags (`#`, `##`) and stores the heading hierarchy in the chunk's metadata.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why is DOM stripping necessary for HTML ingestion?")
        st.caption("Flip for answer:")
        st.info("**Back**: It eliminates navigational boilerplate, CSS, and scripts that dilute vector embeddings and waste context tokens.")
    with st.container(border=True):
        st.markdown("**Front**: How does JSONLoader extract specific nested fields?")
        st.caption("Flip for answer:")
        st.info("**Back**: By specifying a `jq_schema` pointer expression (e.g. `.records[].comment`) that isolates the target data.")

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural differences between PyPDFLoader, PDFPlumber, and OCR.", value=True)
c2 = st.checkbox("I know when to select row-as-document vs table-as-document for CSV processing.", value=True)
c3 = st.checkbox("I can implement header-aware markdown splitting with MarkdownHeaderTextSplitter.", value=True)
c4 = st.checkbox("I know how to clean HTML boilerplate using BeautifulSoup prior to embedding.", value=True)
c5 = st.checkbox("I understand how jq schema pointers navigate nested JSON payloads in JSONLoader.", value=True)
c6 = st.checkbox("I recognize why immutable metadata preservation is mandatory across all document loaders.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 31.2: Document Pipelines.")
