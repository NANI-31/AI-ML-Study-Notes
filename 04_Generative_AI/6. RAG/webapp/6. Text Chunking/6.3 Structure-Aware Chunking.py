import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="6.3 Structure-Aware Chunking - RAG Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class StructureParser:
    """Simulates structure-aware parsing across Markdown and Source Code formats."""
    def __init__(self):
        pass

    def parse_markdown(self, md_text: str) -> List[Dict[str, Any]]:
        sections = []
        current_h1 = "Overview"
        current_h2 = "General"
        buffer = []
        
        for line in md_text.strip().split("\n"):
            if line.startswith("# "):
                if buffer:
                    sections.append({"h1": current_h1, "h2": current_h2, "text": "\n".join(buffer).strip()})
                    buffer = []
                current_h1 = line.replace("# ", "").strip()
                current_h2 = "General"
            elif line.startswith("## "):
                if buffer:
                    sections.append({"h1": current_h1, "h2": current_h2, "text": "\n".join(buffer).strip()})
                    buffer = []
                current_h2 = line.replace("## ", "").strip()
            else:
                if line.strip():
                    buffer.append(line)
        if buffer:
            sections.append({"h1": current_h1, "h2": current_h2, "text": "\n".join(buffer).strip()})
        return sections

    def parse_python_code(self, code_text: str) -> List[Dict[str, Any]]:
        chunks = []
        lines = code_text.strip().split("\n")
        current_scope = "module"
        buffer = []
        
        for line in lines:
            if line.startswith("class ") or (line.startswith("def ") and not line.startswith("    def ")):
                if buffer:
                    chunks.append({"scope": current_scope, "code": "\n".join(buffer).strip()})
                    buffer = []
                current_scope = line.split("(")[0].split(":")[0].strip()
            buffer.append(line)
        if buffer:
            chunks.append({"scope": current_scope, "code": "\n".join(buffer).strip()})
        return chunks

# Title & Overview
st.title("6.3 Structure-Aware Chunking: Formats, Syntax Trees & Layouts")
st.caption("Chapter 6: Text Chunking | Domain: 6. RAG")
st.markdown("""
Enterprise knowledge does not exist in plain, unformatted ASCII strings. It resides in structured formats: 
**Markdown documentation**, **nested HTML pages**, **multi-column PDF reports**, **source code repositories**, 
and **financial tabular sheets**.

Naive chunking treats these rich documents as flat character streams, obliterating table columns, bisecting 
code functions midway through execution scopes, and stripping crucial section header context.

**Structure-Aware Chunking** leverages Document Object Models (DOM), Abstract Syntax Trees (AST), and 
layout-aware vision models to partition documents strictly along natural structural boundaries, preserving 
heading hierarchies and semantic integrity.

In this lesson, you will master:
1. **The Structural Spectrum**: Adapting splitting strategies for Markdown, HTML, PDF, Code, and Tables.
2. **Abstract Syntax Tree (AST) Splitting**: Parsing programming languages using tree-sitter to bound classes and functions.
3. **Table & Schema Linearization**: Preventing row/column scrambling in financial balance sheets and markdown grids.
4. **Heading-Aware Breadcrumb Injection**: Carrying parent section titles down into nested bullet points.
5. **Production Failure Traps**: Avoiding broken HTML tags, scrambled PDF multi-columns, and code syntax invalidation.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Structure-Aware Document Partitioning Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph StructureAwareBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_formats {
            label="Tier 1: Structured Document Ingestion Formats";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            md_doc [label="Markdown (.md)\\nHeadings (#, ##), lists, bold, links", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            code_doc [label="Source Code (.py, .ts, .go)\\nClasses, functions, docstrings, AST", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            pdf_doc [label="Complex PDF Documents\\nMulti-column layout, headers, footnotes", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            table_doc [label="Tabular Data (HTML / CSV / Sheets)\\nRow-column relationships, headers", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_parsers {
            label="Tier 2: Structure-Aware Lexers & Syntax Tree Parsers";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            md_splitter [label="MarkdownHeaderTextSplitter\\nTraverses header levels [H1, H2, H3]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            ast_splitter [label="Tree-Sitter AST Splitter\\nBounds code by function / class syntax trees", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            layout_splitter [label="Layout-Aware PDF Parser\\nVision bounding-box & column de-weaving", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            table_linearizer [label="Table Linearizer & Summarizer\\nConverts matrix rows into self-contained records", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            md_doc -> md_splitter;
            code_doc -> ast_splitter;
            pdf_doc -> layout_splitter;
            table_doc -> table_linearizer;
        }

        subgraph cluster_enriched {
            label="Tier 3: Breadcrumb Injection & Cohesive Chunk Output";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            enriched_chunks [label="Enriched Semantic Chunks\\nPayload: {breadcrumb: 'Auth > SAML > Config', text: '...', scope: 'Function'}", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            md_splitter -> enriched_chunks;
            ast_splitter -> enriched_chunks;
            layout_splitter -> enriched_chunks;
            table_linearizer -> enriched_chunks;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Markdown Chunking
st.header("2. Markdown Chunking: Header Hierarchies and Nested Lists", divider="rainbow")
st.markdown("""
Markdown is the de facto standard for software documentation, wikis, and technical blogs:
* **Header Hierarchy Awareness**: Tools like LangChain's `MarkdownHeaderTextSplitter` split documents along header lines (`#`, `##`, `###`).
* **Breadcrumb Preservation**: When text under an `### Endpoint Authentication` subheader is chunked, the splitter attaches the parent headers as metadata:
  `{"Header 1": "API Reference", "Header 2": "Users Service", "Header 3": "Endpoint Authentication"}`.
* **Preserving Lists and Blocks**: Guarantees that bulleted lists and indented code blocks within a section are not severed arbitrarily.
""")
st.divider()

# Section 3: HTML Chunking
st.header("3. HTML Chunking: DOM Tree Traversal and Semantic Tags", divider="rainbow")
st.markdown("""
Raw web pages contain substantial non-content noise (navigation bars, footers, cookie banners, advertising scripts):
* **Semantic Tag Extraction**: Effective HTML chunkers filter out `<nav>`, `<header>`, `<footer>`, and `<script>` elements, focusing strictly on `<main>`, `<article>`, `<p>`, and `<h1>-<h6>` containers.
* **DOM Hierarchy Traversal**: Traverses the Document Object Model tree, grouping paragraphs that reside within the identical `<section>` or `<article>` element.
* **Cleaning Tag Artifacts**: Converts clean DOM elements into Markdown or structured text to avoid polluting embedding vectors with angle brackets and CSS class names.
""")
st.divider()

# Section 4: PDF Chunking
st.header("4. PDF Chunking: Multi-Column Layouts, Bounding Boxes & Reading Order", divider="rainbow")
st.markdown("""
PDF is an optical display format designed for print layout, **not text semantics**:
* **The Multi-Column Trap**: Naive text extractors read across horizontal coordinate lines, interleaving Column 1 and Column 2 sentences into a scrambled word salad!
* **Layout-Aware Parsing**: Modern pipelines (such as `Unstructured.io`, `Marker`, or PyMuPDF layout analysis) detect bounding boxes and reconstruct the true human reading order.
* **Headers and Footers**: Detects recurring page headers (*"Chapter 3: Memory Systems"*) and page numbers, purging them so they do not artificially sever conceptual paragraphs across page transitions.
""")
st.divider()

# Section 5: Code Chunking
st.header("5. Code Chunking: Tree-Sitter AST and Scope Boundaries", divider="rainbow")
st.markdown("""
Source code cannot be chunked using paragraph splitters without breaking programming logic:
* **Abstract Syntax Tree (AST) Parsing**: Using language-specific grammars (via `tree-sitter`), code chunkers identify syntactic scopes: classes, functions, methods, and interface declarations.
* **Preserving Complete Functional Units**: Every chunk encapsulates a complete function signature, its docstring comments, and the full implementation block.
* **Context Prepending**: When a method inside a class is chunked, the class signature and imported dependencies are prepended to the chunk metadata.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production developer coding assistants (such as GitHub Copilot and Cursor), code chunking is tightly coupled 
with repository-level Symbol Graphs (Language Server Protocol / LSP). When a function is retrieved, the chunker 
automatically retrieves its caller hierarchy, type definitions, and interface contracts, packing them into context.
""")
st.divider()

# Section 6: Table Chunking
st.header("6. Table Chunking: Row Linearization and Column Continuity", divider="rainbow")
st.markdown("""
Tables (financial balance sheets, product matrices, pricing tiers) represent two-dimensional relational data that 
completely falls apart when flattened into raw text:
* **The Flattening Problem**: Splitting a table across page boundaries leaves rows without column headers, making numbers meaningless.
* **Table Linearization**: Converts each row into a standalone declarative sentence:
  *"In Q3 2025, Cloud Infrastructure revenue was $4.2B, representing a 22% year-over-year increase."*
* **Markdown Table Preservation**: Keeps small tables intact as complete Markdown tables, ensuring header rows are repeated if a massive table must be partitioned across chunks.
""")
st.divider()

# Section 7: Section-Based Chunking
st.header("7. Section-Based Chunking", divider="rainbow")
st.markdown("""
In formal reports, legal agreements, and scientific research publications, documents follow strict topical sections:
* **Logical Chunk Bounding**: Chunks are bounded by formal sections (e.g., *Abstract*, *Methodology*, *Results*, *Governing Law*).
* **Variable Length Accommodation**: Unlike naive fixed splitters that force every chunk to be 500 characters, section-based chunking allows chunks to expand or contract according to the author's logical section boundaries.
""")
st.divider()

# Section 8: Heading-Aware Chunking
st.header("8. Heading-Aware Chunking: Breadcrumb Metadata Injection", divider="rainbow")
st.markdown("""
Heading-aware chunking solves the context isolation trap by injecting contextual path breadcrumbs directly 
into the vector payload:
""")

col_head1, col_head2 = st.columns(2)
with col_head1:
    st.markdown("""
    ### Raw Isolated Chunk (Naive)
    ```text
    * Minimum retention: 365 days
    * Cold archive: S3 Glacier
    * Approval required: CISO
    ```
    *Problem*: The vector embedding has no idea what policy or department these rules govern!
    """)
with col_head2:
    st.markdown("""
    ### Enriched Heading-Aware Chunk
    ```text
    [Document: Compliance Handbook]
    [Path: IT Governance > Audit Logs > Data Retention]
    
    * Minimum retention: 365 days
    * Cold archive: S3 Glacier
    * Approval required: CISO
    ```
    *Result*: Vector search matches queries for *"IT Governance compliance retention"* with 98% confidence!
    """)
st.divider()

# Section 9: Production Implementation
st.header("9. Production Implementation: AST-Aware Markdown & Code Document Splitter", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating structure-aware splitting for Markdown documents 
with automatic header breadcrumb injection.
""")

st.code('''
from typing import List, Dict, Any

class MarkdownBreadcrumbSplitter:
    """Structure-aware Markdown splitter preserving heading breadcrumb metadata."""
    def __init__(self, max_section_length: int = 500):
        self.max_section_length = max_section_length

    def split_markdown(self, document_id: str, markdown_text: str) -> List[Dict[str, Any]]:
        chunks = []
        current_h1 = "Document Root"
        current_h2 = "General"
        buffer_lines = []

        def flush_buffer():
            if not buffer_lines:
                return
            body = "\\n".join(buffer_lines).strip()
            if body:
                breadcrumb = f"{current_h1} > {current_h2}"
                payload_text = f"Breadcrumb: {breadcrumb}\\n\\n{body}"
                chunks.append({
                    "chunk_id": f"{document_id}_chk_{len(chunks) + 1:03d}",
                    "breadcrumb": breadcrumb,
                    "h1": current_h1,
                    "h2": current_h2,
                    "content": payload_text
                })
            buffer_lines.clear()

        for line in markdown_text.splitlines():
            if line.startswith("# "):
                flush_buffer()
                current_h1 = line.replace("# ", "").strip()
                current_h2 = "General"
            elif line.startswith("## "):
                flush_buffer()
                current_h2 = line.replace("## ", "").strip()
            else:
                buffer_lines.append(line)
                if sum(len(l) for l in buffer_lines) >= self.max_section_length:
                    flush_buffer()

        flush_buffer()
        return chunks
''', language="python")
st.divider()

# Section 10: Interactive Workbench
st.header("10. Interactive Workbench: Structure-Preserving Document Parsing Simulator", divider="rainbow")
st.markdown("""
Test structure-aware parsing across Markdown and Source Code formats to observe how headings and code scopes 
are preserved without mid-function bisection.
""")

parser = StructureParser()

doc_format = st.radio("Choose Document Format to Inspect:", ["Markdown Architecture Document", "Python Source Code"])

if doc_format == "Markdown Architecture Document":
    sample_md = """# Authentication Subsystem
## OAuth2 Configuration
Configure client IDs and redirect URIs in the centralized identity management portal. Ensure PKCE is enabled for all public single-page applications.

## Token Validation
All incoming requests to the API gateway must validate JWT signatures against the JWKS endpoint. Expired tokens must return an HTTP 401 Unauthorized status immediately.
"""
    parsed_sections = parser.parse_markdown(sample_md)
    st.subheader(f"Parsed Markdown Sections ({len(parsed_sections)} Chunks)")
    for s in parsed_sections:
        with st.container(border=True):
            st.markdown(f"**Breadcrumb**: `{s['h1']} > {s['h2']}`")
            st.text(s["text"])
else:
    sample_py = """class TokenValidator:
    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url

    def validate(self, token: str) -> bool:
        # Validate JWT signature against JWKS
        return len(token) > 20

def generate_session_cookie(user_id: str) -> str:
    # Generate cryptographic session token
    return f"sess_{user_id}_secure"
"""
    parsed_code = parser.parse_python_code(sample_py)
    st.subheader(f"Parsed Code Functional Scopes ({len(parsed_code)} Chunks)")
    for c in parsed_code:
        with st.container(border=True):
            st.markdown(f"**Scope**: `{c['scope']}`")
            st.code(c["code"], language="python")
st.divider()

# Section 11: Technical Interview Preparation & Production Traps
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: Why is Structure-Aware chunking superior to plain Character chunking for enterprise documents?", expanded=True):
    st.markdown("""
    **Answer:**
    Enterprise documents (Markdown, PDFs, code, HTML) rely on structural conventions to convey meaning. Character chunking slices 
    arbitrarily across tokens, severing table rows from headers, cutting code functions midway through loops, and stripping bulleted 
    lists from their parent section headers. Structure-aware chunking parses the document's syntactic layout (DOM, AST, or Markdown headers), 
    ensuring each chunk represents an integral semantic unit with attached structural breadcrumb metadata.
    """)

with st.expander("Q2: How should tabular data (tables) be handled in a RAG ingestion pipeline?"):
    st.markdown("""
    **Answer:**
    Standard text embedding models struggle with raw flattened tables because spatial grid relationships are lost. 
    Best practice approaches include:
    1. **Row Linearization**: Transforming each row into a structured natural language statement including the table name and column headers.
    2. **Markdown Grid Preservation**: Preserving the table in clean Markdown table syntax with repeated column header rows.
    3. **Two-Tier Summarization**: Using an LLM to generate a natural language summary of the table for vector retrieval, while storing the raw table in payload storage for LLM generation.
    """)

with st.expander("Q3: What role does Tree-Sitter play in code-aware chunking?"):
    st.markdown("""
    **Answer:**
    Tree-Sitter is a concrete syntax tree parsing library. Unlike regex splitters, Tree-Sitter parses source code into complete 
    syntactic nodes (class declarations, method definitions, decorator blocks). A code chunker uses Tree-Sitter to bound chunks 
    strictly around complete functional units, ensuring an LLM never receives an incomplete code fragment with missing opening braces 
    or severed parameter lists.
    """)

with st.expander("Q4: How do you prevent multi-column reading order corruption in PDF documents?"):
    st.markdown("""
    **Answer:**
    Standard text extractors read across horizontal coordinate lines, interleaving text from Column 1 with Column 2. 
    To prevent this, use layout-aware document parsers (such as Unstructured, Marker, or Microsoft OCR Document Intelligence) 
    that segment pages into visual bounding boxes, identify column boundaries, and reconstruct the true human reading sequence 
    before chunking occurs.
    """)

with st.expander("Q5: What is Header Breadcrumb Injection, and why does it prevent anaphoric context loss?"):
    st.markdown("""
    **Answer:**
    In structured documents, lower-level bullet points often omit the subject entity (e.g., `* Port: 5432`, `* Pool size: 20`). 
    If chunked in isolation, the embedding vector has no semantic association with the entity (PostgreSQL). Breadcrumb injection 
    prepends the full header hierarchy (`[Database > PostgreSQL > Connection Settings]`) to the chunk text, ensuring the entity 
    is embedded alongside its attributes.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Horizontal PDF Text Extraction Scrambling Multi-Column Research Papers (Word Salad)**
    Using basic `pypdf` extraction on dual-column scientific publications. Text lines from column 1 and column 2 are merged alternately into single unreadable sentences, destroying semantic embeddings. Always deploy layout-aware bounding-box extractors.
    """)
    st.error("""
    **Trap 2: Cutting Source Code Mid-Function Leaving Dangling Scopes (Syntax Invalidation)**
    Using character-based splitters on Python or TypeScript codebases. Functions are split across chunk boundaries, resulting in syntactically invalid code snippets that cause code-generation LLMs to hallucinate missing variables.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Flattening Financial Tables Without Column Headers (Meaningless Numbers)**
    Chunking a 50-row financial statement across page breaks without repeating column titles. Chunks 2, 3, and 4 contain raw floating-point numbers without headers (*"$4.2M"*, *"$1.1M"*), making it impossible for the LLM to know whether a number represents revenue, expense, or tax.
    """)
    st.warning("""
    **Trap 4: Ingesting Raw HTML Boilerplate With Navigation and Footer Clutter (Vector Noise)**
    Chunking raw scraped HTML without stripping `<nav>`, `<footer>`, and cookie banner elements. Top-K search queries frequently retrieve generic site navigation links rather than actual product documentation.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does `MarkdownHeaderTextSplitter` preserve context?")
        st.markdown("**Back**: It segments along header levels (`#`, `##`, `###`) and attaches parent titles as metadata breadcrumbs to every chunk.")
    with st.container(border=True):
        st.markdown("**Front**: What library is standard for AST-based code chunking?")
        st.markdown("**Back**: Tree-sitter, which builds complete syntax trees to bound chunks strictly around full classes and functions.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is the best strategy for chunking large tables?")
        st.markdown("**Back**: Linearizing each row into a self-contained sentence, or preserving markdown tables with repeated header rows.")
    with st.container(border=True):
        st.markdown("**Front**: Why do naive PDF extractors fail on academic papers?")
        st.markdown("**Back**: They read horizontally across dual columns, weaving separate paragraphs into garbled sentences.")

st.divider()

# Section 12: Self-Assessment Checklist
st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Structure-Aware Chunking before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand how structure-aware chunking preserves formatting and syntax across Markdown, Code, and HTML.", value=True, key="chk_struct_1")
    chk2 = st.checkbox("I can explain why AST-based parsing is necessary for chunking source code repositories.", value=True, key="chk_struct_2")
    chk3 = st.checkbox("I understand how layout-aware parsers prevent multi-column PDF reading order corruption.", value=True, key="chk_struct_3")
with c2:
    chk4 = st.checkbox("I know how to linearize tabular data to preserve row-column relationships.", value=True, key="chk_struct_4")
    chk5 = st.checkbox("I understand the role of Header Breadcrumb Injection in mitigating context isolation.", value=True, key="chk_struct_5")
    chk6 = st.checkbox("I can identify production traps such as scrambled PDF columns and broken HTML boilerplate.", value=True, key="chk_struct_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Structure-Aware Chunking Mastery Demonstrated!** You are ready to advance to Lesson 6.4 in the curriculum.")
