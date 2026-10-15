import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="19.2 Document Prompting | GenAI Roadmap",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .doc-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("📄 19.2 Document Prompting — PDFs, Complex Tables, Forms & Charts")
st.markdown(
    """
Welcome to **Lesson 19.2: Document Prompting**.  
The world's highest-value enterprise knowledge lives inside **PDFs**, **scanned government forms**, **SEC financial filings**, and **dense balance sheets**. 

Traditional text extractors (`pypdf`, `pdfminer`) discard 2D visual layouts, hopelessly scrambling multi-column articles and merging adjacent table cells into unreadable strings. Modern **Visual Document Understanding (VDU)** leverages multimodal VLMs to read documents as visual layouts, preserving table hierarchies, checkboxes, and signatures with high fidelity.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Visual Layout Imperative in Document Ingestion
# ==============================================================================
st.header("1. Text-Only Extraction vs. Visual Document Understanding (VDU)", divider="rainbow")

col_vd1, col_vd2 = st.columns(2)

with col_vd1:
    with st.container(border=True):
        st.subheader("❌ Traditional Text Extraction (`pypdf` / Raw OCR)")
        st.markdown(
            """
**Extracted Text Stream:**
```text
Item Qty Price Total Discount Widgets 10 $5.00 $50.00 10%
Gadgets 2 $20.00 $40.00 Subtotal $90.00 Tax $7.20 Total $97.20
```
**Why Downstream Systems Fail:**
- Bounding boxes and cell coordinates are lost.
- Multi-column newspaper formats interleave sentences across columns.
- Table headers dissociate from their corresponding values.
- Checkboxes, radio buttons, and hand-signed signature lines are completely invisible.
"""
        )

with col_vd2:
    with st.container(border=True):
        st.subheader("✅ Visual Document Understanding (Multimodal VLM)")
        st.markdown(
            """
**Native VLM Page Ingestion:**
- The page is rendered as an image and ingested via high-detail vision tokens.
- Cross-attention heads understand the **2D spatial proximity** between headers, borders, and numerical cells.
- Accurately parses multi-page tables, merged header cells, and handwritten annotations.
- Outputs clean, deterministic Markdown tables or typed JSON trees.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Core Document Archetypes & Extraction Strategies
# ==============================================================================
st.header("2. Core Enterprise Document Archetypes", divider="rainbow")
st.markdown("Specialized prompting strategies tailored to distinct layout topologies:")

doc_archetypes_df = pd.DataFrame({
    "Document Archetype": [
        "1. Financial Filings (SEC 10-K / Balance Sheets)",
        "2. Structured Forms (IRS W-2, 1099, CMS-1500)",
        "3. Multi-Column Academic Papers",
        "4. Technical Architectural Diagrams",
        "5. Low-Contrast Scanned Receipts & Invoices"
    ],
    "Visual Layout Challenge": [
        "Multi-level column headers, indented rows, footnotes referencing specific lines",
        "Tight bounding box fields, checked vs unchecked checkboxes, handwritten notes",
        "Two-column reading orders, floating figures, embedded equations",
        "Directed arrows, cloud provider logos, nested networking VPC boundaries",
        "Thermal paper fading, crumpling, skewed orientation, coffee stains"
    ],
    "Target Prompt Specification": [
        "Require markdown table output + explicit footnoted row mappings and sign conventions (e.g. `(450)` = negative)",
        "Prompt for key-value JSON schema mirroring the physical form's box numbers (e.g. `box_1_wages`)",
        "Instruct model to read down column 1 before starting column 2; transcribe section headers",
        "Instruct model to extract graph adjacency lists (`source -> edge_protocol -> target`)",
        "Mandate: 'Transcribe visible characters verbatim; state UNREADABLE for faded regions'"
    ]
})
st.dataframe(doc_archetypes_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Visual PDF Ingestion & Normalization Pipeline", divider="rainbow")
st.markdown("How enterprise documents are rendered, ingested, and structured into verified tables:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        pdf [label="Multi-Page Enterprise PDF\n(SEC Filing / Form)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        rasterizer [label="Page Rasterizer\n(PyMuPDF 300 DPI PNG)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        vlm [label="Multimodal VLM\n(Layout-Aware Vision)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        extractor [label="Structured Extractor\n(Markdown / Pydantic JSON)", fillcolor="#EAB308", fontcolor="#000000"]
        math_check [label="Checksum Verifier\n(Subtotal + Tax = Total)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        db [label="Enterprise ERP / Data Lake", fillcolor="#10B981", fontcolor="#FFFFFF"]

        pdf -> rasterizer -> vlm -> extractor -> math_check -> db
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Document Parsing Workbench
# ==============================================================================
st.header("4. Interactive Document Parsing Workbench", divider="rainbow")
st.markdown("Select an enterprise document scenario and observe how layout-aware prompts extract clean data:")

doc_choice = st.selectbox(
    "Select Document Extraction Scenario",
    [
        "SEC 10-K Condensed Balance Sheet (Nested Multi-Period Table)",
        "IRS Form W-2 Wage Statement (Structured Boxed Fields)",
        "Clinical Blood Chemistry Laboratory Panel (Reference Ranges)"
    ]
)

col_dw1, col_dw2 = st.columns([1, 1])

with col_dw1:
    st.subheader("📄 Simulated Document Page Layout")
    if "Balance Sheet" in doc_choice:
        st.markdown(
            """
```text
=============================================================
                    CONSOLIDATED BALANCE SHEETS
                  (in millions, except share data)
=============================================================
ASSETS:                                September 30,  September 24,
                                           2026           2025
Current assets:
  Cash and cash equivalents . . . . . . $  29,965      $  28,420
  Marketable securities . . . . . . . .    31,590         24,659
  Accounts receivable, net. . . . . . .    29,501         28,184
  Inventories . . . . . . . . . . . . .     6,331          4,946
Total current assets. . . . . . . . . . $  97,387      $  86,209
=============================================================
```
"""
        )
    elif "W-2" in doc_choice:
        st.markdown(
            """
```text
+-----------------------------------+------------------------+
| a  Employee's social security #   | b  Employer EIN        |
|    ***-**-8492                    |    12-3456789          |
+-----------------------------------+------------------------+
| 1  Wages, tips, other compensation| 2  Federal income tax  |
|    $ 142,500.00                   |    $ 24,650.00         |
+-----------------------------------+------------------------+
| 3  Social security wages          | 4  Social security tax |
|    $ 142,500.00                   |    $  8,835.00         |
+-----------------------------------+------------------------+
| c  Employer's name & address:     | e  Employee's name:    |
|    Acme Enterprise Tech Inc       |    Jane Doe            |
+-----------------------------------+------------------------+
```
"""
        )
    else:
        st.markdown(
            """
```text
=============================================================
               COMPREHENSIVE METABOLIC PANEL
=============================================================
Test Name           Result    Flag    Reference Range   Units
Glucose, Fasting     108       H          70 - 99       mg/dL
Creatinine           0.92                 0.60 - 1.20   mg/dL
eGFR                  88                   > 60         mL/min
Potassium            4.3                  3.5 - 5.0     mmol/L
Sodium               139                  135 - 145     mmol/L
=============================================================
```
"""
        )

with col_dw2:
    st.subheader("📤 Extracted Structured Representation")
    if "Balance Sheet" in doc_choice:
        extracted = {
            "statement_type": "CONSOLIDATED_BALANCE_SHEET",
            "reporting_currency": "USD",
            "scale": "MILLIONS",
            "periods": ["2026-09-30", "2025-09-24"],
            "line_items": [
                {"item": "Cash and cash equivalents", "values": {"2026": 29965, "2025": 28420}},
                {"item": "Marketable securities", "values": {"2026": 31590, "2025": 24659}},
                {"item": "Accounts receivable, net", "values": {"2026": 29501, "2025": 28184}},
                {"item": "Inventories", "values": {"2026": 6331, "2025": 4946}},
                {"item": "Total current assets", "values": {"2026": 97387, "2025": 86209}}
            ],
            "checksum_verified": True
        }
        st.json(extracted)
        st.success("✅ Multi-period financial table mapped into structured time-series records.")

    elif "W-2" in doc_choice:
        extracted = {
            "form_type": "IRS_W2",
            "tax_year": 2026,
            "employee": {"name": "Jane Doe", "ssn_masked": "***-**-8492"},
            "employer": {"name": "Acme Enterprise Tech Inc", "ein": "12-3456789"},
            "box_1_wages": 142500.00,
            "box_2_federal_tax": 24650.00,
            "box_3_ss_wages": 142500.00,
            "box_4_ss_tax": 8835.00
        }
        st.json(extracted)
        st.success("✅ Clean tax document extraction keyed to official box identifiers.")

    else:
        extracted = {
            "panel": "COMPREHENSIVE_METABOLIC_PANEL",
            "abnormal_findings_count": 1,
            "results": [
                {"test": "Glucose, Fasting", "value": 108, "units": "mg/dL", "flag": "HIGH", "ref_low": 70, "ref_high": 99},
                {"test": "Creatinine", "value": 0.92, "units": "mg/dL", "flag": "NORMAL", "ref_low": 0.60, "ref_high": 1.20},
                {"test": "eGFR", "value": 88, "units": "mL/min", "flag": "NORMAL", "ref_low": 60, "ref_high": None},
                {"test": "Potassium", "value": 4.3, "units": "mmol/L", "flag": "NORMAL", "ref_low": 3.5, "ref_high": 5.0},
                {"test": "Sodium", "value": 139, "units": "mmol/L", "flag": "NORMAL", "ref_low": 135, "ref_high": 145}
            ]
        }
        st.json(extracted)
        st.warning("⚠️ High Fasting Glucose (108 mg/dL) automatically detected and flagged.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Visual PDF Ingestion Script", divider="rainbow")
st.markdown("Python script rendering PDF pages to high-resolution PNG images for multimodal VLM ingestion:")

st.code(
    '''import fitz  # PyMuPDF
import base64
from typing import List

def convert_pdf_page_to_base64_png(pdf_path: str, page_number: int = 0, dpi: int = 300) -> str:
    """
    Renders a PDF page to high-DPI image in memory and returns a base64 string
    for submission to multimodal vision APIs.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    
    # 300 DPI zoom matrix ensures fine print and table borders are crystal clear
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Encode pixmap bytes directly to base64 string
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

DOCUMENT_EXTRACTION_PROMPT = """<system>
You are an expert Document Understanding and Table Extraction Specialist.
Analyze the rendered document page.
Rules:
1. Extract all tables into GitHub-flavored Markdown tables.
2. Preserve column headers, row hierarchy, and indented categories.
3. For scanned forms, map visible field boxes to a JSON schema.
4. Do not hallucinate obscured numbers; write [ILLEGIBLE] for low-contrast text.
</system>
"""

print("PDF Visual Ingestion utility ready for 300 DPI high-detail extraction.")
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why does native Visual PDF ingestion outperform traditional OCR + text LLM pipelines on financial tables?", expanded=True):
        st.markdown(
            """
1. **Preservation of 2D Spatial Geometry**: Traditional OCR flattens text into a 1D sequence of words, losing track of column alignments, merged header spans, and indented accounting sub-totals.
2. **Visual Clues (Lines & Borders)**: Financial tables use subtle visual elements—such as single underlines (subtotals) and double underlines (grand totals)—to signal accounting hierarchies. A visual VLM reads these graphical indicators directly, whereas text OCR discards them.
3. **Multi-Column Immunity**: Visual models process reading order naturally without jumping across vertical gutter dividers.
"""
        )
    with st.expander("Q2: How do you prevent token overflow when processing 100-page enterprise PDF documents?"):
        st.markdown(
            """
Passing 100 pages as high-detail images into a single prompt consumes 100,000+ tokens and causes massive latency.
**Production Multi-Page Architecture**:
1. **Visual Page Classifier / Router**: Run a lightweight text or low-detail image pass to classify pages (e.g. identify which 3 pages contain the financial tables or contract terms).
2. **Targeted Extraction**: Send only the candidate pages to high-detail multimodal extraction.
3. **Map-Reduce Stitching**: Process pages in parallel and stitch the resulting JSON tables together via deterministic Python logic.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we extract tables from scanned PDFs at 72 DPI to save image file size?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, 72 DPI is standard web resolution and uploads faster."*
- **Winning Response**: *"Never. At 72 DPI, small 8pt or 9pt financial table typography (decimal points, negative signs in parentheses, superscripts) is rendered as blurry pixel noise. A decimal point disappears, turning '$12.50' into '$1250'—a catastrophic 100x financial error. Production document rasterizers must render at **minimum 200 DPI to 300 DPI** to guarantee that punctuation and numbers are sharply defined before VLM patchification."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Visual Document Understanding (VDU)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Ingesting documents as 2D visual layouts to preserve table borders, multi-column reading orders, and checkboxes.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Rasterization DPI**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Converting PDF pages to 300 DPI images to ensure fine typography, decimal points, and borders remain crisp for vision models.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Page Routing Pipeline**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Classifying and filtering multi-page PDFs to submit only high-value pages to expensive high-detail vision extraction.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 19.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Layout Preservation**: Use Visual Document Understanding to preserve 2D geometry and table cells.
2. **High DPI Resolution**: Render documents at 300 DPI to prevent blurred decimal points and symbols.
3. **Page Classification**: Filter multi-page PDFs to extract only relevant pages and control token cost.
4. **Structured Schemas**: Prompt for typed JSON or clean Markdown tables with mathematical checksum validation.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why text-only PDF extractors fail on multi-column and complex tabular layouts.", value=True, key="chk_19_2_1")
    st.checkbox("I know how to render PDF pages to high-DPI base64 images using PyMuPDF.", value=True, key="chk_19_2_2")
    st.checkbox("I can structure prompts for complex financial balance sheets and tax forms.", value=True, key="chk_19_2_3")
    st.checkbox("I know how to design a page-routing architecture for 100+ page enterprise documents.", value=True, key="chk_19_2_4")
