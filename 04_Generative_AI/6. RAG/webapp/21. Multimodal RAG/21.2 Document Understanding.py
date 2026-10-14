import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="21.2 Document Understanding in RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class VisualDocumentParserEngine:
    """Enterprise Visual Document Layout Parsing & Element Extraction Engine."""

    def __init__(self):
        # Sample document layout elements
        self.elements = {
            "PDF Images": {
                "element_id": "elem_pdf_img_01",
                "type": "PDF Vector Graphic",
                "bbox": [54, 120, 480, 360],
                "confidence": 0.96,
                "parsed_data": "Extracted high-res vector diagram (300 DPI): Semiconductor Die Floorplan showing 8x HBM3e memory stacks surrounding the central compute tile."
            },
            "Tables": {
                "element_id": "elem_table_02",
                "type": "Bordered Tabular Matrix",
                "bbox": [72, 410, 520, 680],
                "confidence": 0.98,
                "parsed_data": "| Metric | Q3 2024 | Q3 2025 | Growth YoY |\n| :--- | :---: | :---: | :---: |\n| Cloud ARR | $3.2B | $4.8B | +50.0% |\n| Gross Margin | 68.2% | 72.1% | +390 bps |"
            },
            "Charts": {
                "element_id": "elem_chart_03",
                "type": "Multi-Line Trend Plot",
                "bbox": [50, 200, 500, 450],
                "confidence": 0.93,
                "parsed_data": "Chart Type: Line Plot. X-Axis: Epochs (1-100), Y-Axis: Validation Loss. Trend: Loss monotonically decreases from 2.84 to 0.42, reaching convergence at epoch 78."
            },
            "Diagrams": {
                "element_id": "elem_diag_04",
                "type": "Architectural Flowchart",
                "bbox": [80, 150, 510, 580],
                "confidence": 0.91,
                "parsed_data": "Nodes: [Client, API Gateway, Vector DB, Reranker, VLM]. Directed Edges: Client -> Gateway -> Vector DB -> Reranker -> VLM -> Client. Protocol: gRPC mTLS."
            },
            "Screenshots": {
                "element_id": "elem_screen_05",
                "type": "Web UI Application Viewport",
                "bbox": [0, 0, 1280, 720],
                "confidence": 0.89,
                "parsed_data": "UI Hierarchy: Navigation Bar (Home, Analytics, Settings), Active Modal Dialog 'Confirm Key Rotation', Primary Action Button 'Authorize (FIDO2)'."
            }
        }

    def parse_document_element(self, element_category: str) -> Dict[str, Any]:
        """Extracts deep semantic metadata and structured text from visual document elements."""
        elem = self.elements.get(element_category)
        if not elem:
            return {"status": "NOT_FOUND"}
            
        return {
            "category": element_category,
            "element_id": elem["element_id"],
            "element_type": elem["type"],
            "bounding_box_coords": elem["bbox"],
            "parser_confidence": elem["confidence"],
            "extracted_representation": elem["parsed_data"]
        }


# --- Streamlit Layout ---

st.title("21.2 Document Understanding in RAG")
st.caption("Visual Layout Analysis: Extracting PDF Images, Tabular Matrices, Benchmark Charts, Flowcharts & UI Screenshots")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Visual Document Layout Parsing & Deep Semantic Extraction Pipeline", divider="rainbow")

st.markdown(
    """
    Real-world enterprise documents are not plain text files; they are complex visual artifacts formatted in PDF, PPTX, or image formats.
    A modern **Document Layout Analysis (DLA)** pipeline decomposes complex documents into functional visual primitives:
    **Embedded Images**, **Structured Tables**, **Statistical Charts**, **System Flowcharts**, and **Application Screenshots**.
    
    Rather than treating a page as an unstructured text soup, the pipeline isolates each element's 2D bounding box, executes specialized
    extractors (OCR, Table Transformers, Chart Parsers), and injects both the structured text and cropped image tokens into the multimodal index.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Document_Understanding {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_page_raster {
            label = "1. DOCUMENT RASTERIZATION & PAGE SEGMENTATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            raw_pdf [label="Raw Enterprise Document\\n(Multi-Page Scanned PDF / Slide Deck)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            rasterizer [label="High-Res Page Rasterizer (300 DPI)\\nPyMuPDF / PDF2Image", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            dla_model [label="Document Layout Analysis (DLA)\\nLayoutLMv3 / YOLO-Doc / Surya\\n(Bounding Box Detection & Classification)", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            raw_pdf -> rasterizer -> dla_model;
        }

        subgraph cluster_specialized_extractors {
            label = "2. SPECIALIZED VISUAL EXTRACTORS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            ext_pdf_img [label="1. PDF Image Extractor\\nVector Graphic Crop & High-Res Feature Encoding", fillcolor="#0F172A", fontcolor="#38BDF8"];
            ext_table [label="2. Table Transformer (TATR)\\nCell Coordinate Detection & Markdown Table Output", fillcolor="#0F172A", fontcolor="#38BDF8"];
            ext_chart [label="3. Chart Parser (DePlot)\\nAxis OCR, Data Series Points & Trend Summaries", fillcolor="#0F172A", fontcolor="#38BDF8"];
            ext_diagram [label="4. Diagram / Schematic Parser\\nNode-Edge Graph Extraction & Directional Flow", fillcolor="#0F172A", fontcolor="#38BDF8"];
            ext_screen [label="5. Screenshot UI Parser\\nDOM Tree Recovery & Component Bounding Boxes", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            dla_model -> ext_pdf_img [label="Detect Image"];
            dla_model -> ext_table [label="Detect Table"];
            dla_model -> ext_chart [label="Detect Chart"];
            dla_model -> ext_diagram [label="Detect Diagram"];
            dla_model -> ext_screen [label="Detect Screenshot"];
        }

        subgraph cluster_multimodal_manifest {
            label = "3. MULTIMODAL MANIFEST & HYBRID CHUNKING";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            manifest_builder [label="Multimodal Manifest Assembly\\n- Standardized Markdown Tables\\n- Visual Crop Bounding Boxes\\n- Extracted Graph Relations", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            vector_index [label="Multimodal Index\\n(ColPali Patches + Dense Text Embeddings)", shape=cylinder, fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            ext_pdf_img -> manifest_builder;
            ext_table -> manifest_builder;
            ext_chart -> manifest_builder;
            ext_diagram -> manifest_builder;
            ext_screen -> manifest_builder;
            manifest_builder -> vector_index;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: PDF IMAGES ---
st.header("2. PDF Images (High-Resolution Rasterization, Embedded Vector Extraction & Figure Cropping)", divider="rainbow")

st.markdown(
    """
    PDFs store visual illustrations in two distinct formats:
    - **Raster Bitmaps (JPEG/PNG streams)**: Embedded within the PDF file structure, frequently compressed.
    - **Vector Drawing Directives (PostScript/PDF Paths)**: Composed of bezier curves, lines, and text labels rather than pixels. Naive rasterization at 72 DPI blurs fine lines.
    
    ### Production Extraction Pipeline:
    1. **High-Resolution Rasterization (300 DPI)**: Renders vector drawing directives at sufficient pixel density for vision models to read 6-point font labels.
    2. **Bounding Box Figure Cropping**: Isolates the figure bounding box and strips away surrounding page headers and footers.
    3. **Visual Vector Embedding**: Encodes the cropped image patch using Vision Transformers (ViT) into the multimodal index.
    """
)

# --- SECTION 3: TABLES ---
st.header("3. Tables (Table Structure Recognition, Cell Coordinate Extraction, Markdown/HTML Conversion)", divider="rainbow")

st.markdown(
    """
    Tables represent the densest concentration of numerical facts in enterprise reports.
    Extracting tables via standard OCR produces scrambled strings where column headers are severed from data rows.
    
    ### Table Structure Recognition (TSR):
    1. **Cell Coordinate Detection**: Models like **TATR (Table Transformer)** predict row and column bounding box separators.
    2. **Spanning Cell Resolution**: Detects merged headers spanning multiple columns (e.g., *'Fiscal Year 2024'* spanning *'Q1'*, *'Q2'*, *'Q3'*).
    3. **Canonical Serialization**: Converts the 2D matrix into a standard GitHub Markdown table or HTML table string, preserving explicit row-column relationships for LLMs.
    """
)

# --- SECTION 4: CHARTS ---
st.header("4. Charts (Bar, Line, and Scatter Plots: OCR Data Point Extraction, Trend Summarization & Numerical Grounding)", divider="rainbow")

st.markdown(
    """
    Charts (bar charts, line plots, scatter graphs) communicate trends, slopes, and statistical distributions visually.
    A text prompt like *"What was the peak validation loss?"* cannot be answered unless chart elements are decoded:
    
    - **DePlot & Pix2Struct Paradigms**: Visual language models trained specifically to translate chart images directly into linearized data tables (*'X: 10, Y: 2.84'*).
    - **Extrema & Slope Extraction**: Detecting axis min/max values, logarithmic scales, and inflection points.
    - **Dual Indexing**: Indexing both the raw chart image (for VLM spatial inspection) and the extracted numerical table (for LLM reasoning).
    """
)

# --- SECTION 5: DIAGRAMS ---
st.header("5. Diagrams (Flowcharts, Architectural Schematics, Entity-Relationship Graphs & Arrow Flow Mapping)", divider="rainbow")

st.markdown(
    """
    Flowcharts, system architecture schematics, and entity-relationship diagrams convey **relational logic and directional flows**:
    
    - **Node-Edge Extraction**: Detecting visual boxes (services, databases, entities) and labeled connecting arrows (protocols, request directions).
    - **Topological Sorting**: Translating a circular flowchart into an ordered sequence of events (*'Step 1: Client sends request -> Step 2: Gateway verifies JWT -> Step 3: DB writes record'*).
    - **Graph Encoding**: Serializing diagram topologies into Mermaid.js or Graphviz DOT strings for dense vector indexing.
    """
)

# --- SECTION 6: SCREENSHOTS ---
st.header("6. Screenshots (UI Layout Parsing, Text Bounding Boxes, Viewport Cropping & Component Identification)", divider="rainbow")

st.markdown(
    """
    In software documentation, IT troubleshooting, and customer support, user manuals rely heavily on application screenshots:
    
    - **UI Component Hierarchy**: Detecting buttons, text input boxes, dropdown selectors, and modal popups.
    - **Visual Status Cues**: Recognizing active vs disabled buttons, red error banners, and highlighted menu items.
    - **DOM Anchor Alignment**: Correlating visual UI elements with software API actions to guide multi-turn agentic task completion.
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Visual Document Parsing & Layout Understanding Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `VisualDocumentParserEngine`.
    It handles document layout primitives across PDF images, tables, charts, flowcharts, and application screenshots,
    extracting bounding box coordinates, parser confidence metrics, and structured representations.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class VisualDocumentParserEngine:
    """Enterprise Visual Document Layout Parsing & Element Extraction Engine."""

    def __init__(self):
        self.elements = {
            "PDF Images": {
                "element_id": "elem_pdf_img_01",
                "type": "PDF Vector Graphic",
                "bbox": [54, 120, 480, 360],
                "confidence": 0.96,
                "parsed_data": "Extracted high-res vector diagram (300 DPI): Semiconductor Die Floorplan showing 8x HBM3e memory stacks surrounding the central compute tile."
            },
            "Tables": {
                "element_id": "elem_table_02",
                "type": "Bordered Tabular Matrix",
                "bbox": [72, 410, 520, 680],
                "confidence": 0.98,
                "parsed_data": "| Metric | Q3 2024 | Q3 2025 | Growth YoY |\\n| :--- | :---: | :---: | :---: |\\n| Cloud ARR | $3.2B | $4.8B | +50.0% |\\n| Gross Margin | 68.2% | 72.1% | +390 bps |"
            },
            "Charts": {
                "element_id": "elem_chart_03",
                "type": "Multi-Line Trend Plot",
                "bbox": [50, 200, 500, 450],
                "confidence": 0.93,
                "parsed_data": "Chart Type: Line Plot. X-Axis: Epochs (1-100), Y-Axis: Validation Loss. Trend: Loss monotonically decreases from 2.84 to 0.42, reaching convergence at epoch 78."
            },
            "Diagrams": {
                "element_id": "elem_diag_04",
                "type": "Architectural Flowchart",
                "bbox": [80, 150, 510, 580],
                "confidence": 0.91,
                "parsed_data": "Nodes: [Client, API Gateway, Vector DB, Reranker, VLM]. Directed Edges: Client -> Gateway -> Vector DB -> Reranker -> VLM -> Client. Protocol: gRPC mTLS."
            },
            "Screenshots": {
                "element_id": "elem_screen_05",
                "type": "Web UI Application Viewport",
                "bbox": [0, 0, 1280, 720],
                "confidence": 0.89,
                "parsed_data": "UI Hierarchy: Navigation Bar (Home, Analytics, Settings), Active Modal Dialog 'Confirm Key Rotation', Primary Action Button 'Authorize (FIDO2)'."
            }
        }

    def parse_document_element(self, element_category: str) -> Dict[str, Any]:
        """Extracts deep semantic metadata and structured text from visual document elements."""
        elem = self.elements.get(element_category)
        if not elem:
            return {"status": "NOT_FOUND"}
            
        return {
            "category": element_category,
            "element_id": elem["element_id"],
            "element_type": elem["type"],
            "bounding_box_coords": elem["bbox"],
            "parser_confidence": elem["confidence"],
            "extracted_representation": elem["parsed_data"]
        }
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Element Document Layout Analyzer & Visual Extractor", divider="rainbow")

st.markdown(
    """
    Inspect the visual document parsing pipeline across heterogeneous visual layout elements.
    Select an element category to observe bounding box coordinates, extraction confidence, and structured representations.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("📑 Visual Element Selection")
    
    element_category = st.selectbox(
        "Select Layout Element Category to Inspect:",
        ["PDF Images", "Tables", "Charts", "Diagrams", "Screenshots"],
        index=1
    )
    
    engine = VisualDocumentParserEngine()
    parsed_result = engine.parse_document_element(element_category)
    
    st.markdown("#### Bounding Box & Parser Telemetry")
    st.info(
        f"**Element ID**: `{parsed_result['element_id']}`  \n"
        f"**Primitive Type**: {parsed_result['element_type']}  \n"
        f"**Bounding Box [x1, y1, x2, y2]**: `{parsed_result['bounding_box_coords']}`  \n"
        f"**Extractor Confidence**: `{parsed_result['parser_confidence'] * 100:.1f}%`"
    )

with wb_col2:
    st.subheader("🔍 Structured Semantic Output")
    
    m1, m2 = st.columns(2)
    m1.metric("Category", parsed_result["category"])
    m2.metric("Parser Confidence", f"{parsed_result['parser_confidence'] * 100:.1f}%")
    
    st.markdown("#### Extracted Representation (Injected into Context)")
    if element_category == "Tables":
        st.markdown(parsed_result["extracted_representation"])
    else:
        st.code(parsed_result["extracted_representation"], language="text")

st.markdown("### Document Layout Primitive Taxonomy")
taxonomy_df = pd.DataFrame([
    {
        "Primitive": k,
        "Sub-Type": v["type"],
        "Sample Bounding Box": str(v["bbox"]),
        "Typical Extraction Model": "LayoutLMv3 + OCR" if "Image" in k or "Screenshot" in k else ("TATR (Table Transformer)" if "Table" in k else "DePlot / Pix2Struct"),
        "Primary Output Format": "Markdown Table" if "Table" in k else ("Mermaid / Graph DOT" if "Diagram" in k else "Natural Language Caption / Dense ViT Patches")
    }
    for k, v in engine.elements.items()
])
st.dataframe(taxonomy_df, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why is Document Layout Analysis (DLA) necessary before chunking enterprise PDFs?**"):
    st.markdown(
        """
        Naive chunking operates on character offsets or line breaks. In a multi-column PDF with embedded sidebars, tables, and image captions:
        1. Slicing across columns combines text from left and right columns horizontally, destroying grammatical and semantic coherence.
        2. Slicing through tables severs rows from column headers.
        3. Image captions are concatenated with body text paragraphs without demarcation.
        
        **Document Layout Analysis (DLA)** detects 2D bounding boxes and groups visual elements into distinct semantic blocks *before* chunking, preserving reading order.
        """
    )

with st.expander("**Q2: How do Table Transformers (TATR) recognize complex merged table structures?**"):
    st.markdown(
        """
        Table structure recognition operates through two sequential object detection tasks:
        1. **Table Detection (TD)**: Identifies the global bounding box of the table on the page.
        2. **Table Structure Recognition (TSR)**: Predicts bounding boxes for every row, column, and spanning cell.
        
        By computing the geometric intersections of row and column bands, the model reconstructs the full 2D grid matrix, properly assigning spanning headers to multiple child columns.
        """
    )

with st.expander("**Q3: What is the DePlot paradigm for chart understanding in RAG?**"):
    st.markdown(
        """
        **DePlot** (Liu et al., Google Research) is a visual language model trained to translate chart images (bar charts, line graphs, scatter plots) directly into linearized data tables.
        
        Rather than asking an LLM to guess trends from messy OCR text, DePlot converts the visual chart into:
        `Entity | Year | Metric` rows.
        The resulting structured table can be fed directly to a standard text LLM or Text-to-SQL engine for precise numerical reasoning.
        """
    )

with st.expander("**Q4: How do you extract directional flow from software architectural diagrams?**"):
    st.markdown(
        """
        Extracting directional flow requires combining:
        1. **Object Detection**: Identifies boxes, cylinder database icons, and text labels.
        2. **Arrow Keypoint Detection**: Locates arrow heads, arrow tails, and path segments.
        3. **Graph Formulation**: Maps arrow tails as source nodes and arrow heads as destination nodes, constructing an adjacency list serialized as a Directed Acyclic Graph (DAG) in Mermaid.js syntax.
        """
    )

with st.expander("**Q5: What are the primary failure modes of OCR on low-contrast screenshots?**"):
    st.markdown(
        """
        Screenshots of dark-mode IDEs or low-contrast mobile apps cause standard OCR engines to hallucinate characters:
        - Light gray text on dark gray backgrounds fails thresholding algorithms.
        - Antialiased font smoothing blurs small character edges (confusing *'l'*, *'1'*, and *'I'*).
        - Iconographic buttons (e.g., trash can icons for delete) contain zero text, rendering OCR completely blind to critical UI actions.
        
        Production systems require native Vision Transformers trained on full screenshot viewports.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Flattening Tables into Unformatted Text Strings**  
        **Antipattern**: Extracting tables as space-delimited text without markdown column boundaries.  
        **Consequence**: The LLM loses column alignment, associating Column 4 values with Column 1 headers.  
        **Production Fix**: Enforce strict serialization into GitHub Flavored Markdown tables or HTML `<table>` markup.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Discarding Bounding Box Coordinates During Ingestion**  
        **Antipattern**: Keeping only extracted text and throwing away pixel bounding box coordinates.  
        **Consequence**: The system cannot cite visual regions of interest or render visual bounding boxes on source PDFs for user verification.  
        **Production Fix**: Store `[page_num, x1, y1, x2, y2]` coordinates in document chunk metadata payloads.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Slicing Documents by Character Offsets Instead of Layout Blocks**  
        **Antipattern**: Running fixed 500-token text splitters over raw PDF text streams.  
        **Consequence**: Bects tables and flowchart blocks mid-sentence, destroying structural coherence.  
        **Production Fix**: Implement layout-aware chunkers that treat tables, charts, and diagrams as atomic, indivisible chunks.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Relying Solely on Text Captions for Flowchart Logic**  
        **Antipattern**: Summarizing an intricate 20-step deployment flowchart with a one-sentence text caption.  
        **Consequence**: Discards branch conditions, rollback steps, and error handling pathways.  
        **Production Fix**: Extract explicit node-edge graph representations and preserve both text summaries and high-resolution crops.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Document Layout Analysis (DLA)?")
        st.markdown("**Back**: A computer vision process that detects 2D bounding boxes and classifies page regions into semantic primitives: paragraphs, headers, tables, charts, and figures.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is Table Structure Recognition (TSR)?")
        st.markdown("**Back**: The process of identifying rows, columns, and spanning cells within a table bounding box to convert image tables into structured Markdown or HTML.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is the DePlot model?")
        st.markdown("**Back**: A specialized vision-language model that converts visual chart plots (bar, line, scatter) directly into linearized data tables for downstream LLM reasoning.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is an Atomic Layout Chunk?")
        st.markdown("**Back**: An ingestion rule dictating that structured tables, diagrams, and figures must never be bisected by character token splitters, but preserved as unified chunks.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of visual document understanding and layout analysis pipelines:")

chk1 = st.checkbox("I understand why Document Layout Analysis (DLA) is mandatory before chunking complex enterprise PDFs.", value=True)
chk2 = st.checkbox("I can implement Table Structure Recognition (TSR) pipelines that serialize visual tables into Markdown.", value=True)
chk3 = st.checkbox("I understand how DePlot translates visual charts and line plots into linearized numerical tables.", value=True)
chk4 = st.checkbox("I know how to extract directional node-edge flows from architectural schematics and flowcharts.", value=True)
chk5 = st.checkbox("I understand the challenges of UI screenshot parsing and DOM hierarchy recovery in software manuals.", value=True)
chk6 = st.checkbox("I know how to preserve 2D bounding box metadata coordinates to enable visual grounding and region citations.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Visual Document Understanding and Layout Parsing in RAG.")
