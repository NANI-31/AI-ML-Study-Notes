import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="30.2 Image Applications - LangChain Systems",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual intelligence telemetry
st.markdown("""
<style>
    .vision-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-vdu {
        background-color: #059669;
        color: #ECFDF5;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-ocr {
        background-color: #2563EB;
        color: #EFF6FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-qa {
        background-color: #D97706;
        color: #FFFBEB;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class VisualDocumentWorkbenchSimulator:
    """
    Simulates visual intelligence workflows:
    zero-shot classification, deep defect inspection, OCR extraction,
    Visual Document Understanding (VDU), and visual question answering (VQA).
    """
    def __init__(self):
        self.preset_documents = {
            "Commercial Vendor Invoice (VDU)": {
                "category": "Financial Document / Accounts Payable",
                "confidence": 0.992,
                "inspection_summary": "Clean scan, zero skew, valid digital signature detected.",
                "ocr_tokens_extracted": 348,
                "extracted_fields": {
                    "invoice_number": "INV-2026-9841",
                    "vendor_name": "Apex Cloud Systems Inc.",
                    "billing_date": "2026-10-01",
                    "total_amount": "$14,850.00",
                    "line_items": [
                        {"item": "Managed Kubernetes Node Hours", "qty": 720, "amount": "$4,320.00"},
                        {"item": "Enterprise LangSmith Seat Licenses", "qty": 25, "amount": "$6,250.00"},
                        {"item": "High-Throughput Vector DB Tier", "qty": 1, "amount": "$4,280.00"}
                    ]
                },
                "vqa_answer": "The total payable sum is $14,850.00, due within 30 days of the October 1, 2026 invoice date."
            },
            "Industrial Circuit Board QA (Inspection)": {
                "category": "Hardware Assembly Quality Audit",
                "confidence": 0.965,
                "inspection_summary": "Soldering anomaly detected at capacitor C42; traces otherwise within IPC-A-610 tolerance.",
                "ocr_tokens_extracted": 42,
                "extracted_fields": {
                    "board_serial": "PCB-SN-88329-REV3",
                    "inspection_status": "FLAGGED_FOR_MANUAL_REVIEW",
                    "defect_coordinates": {"ymin": 340, "xmin": 612, "ymax": 395, "xmax": 678},
                    "defect_type": "Cold Solder Joint (Bridging risk)"
                },
                "vqa_answer": "Capacitor C42 shows insufficient thermal wetting on the positive terminal pad, posing a high risk of vibration-induced disconnection."
            },
            "Q3 Revenue Growth Trend Chart (VQA)": {
                "category": "Business Analytics / Data Visualization",
                "confidence": 0.988,
                "inspection_summary": "Dual-axis bar and line chart displaying monthly recurring revenue and gross margin percentages.",
                "ocr_tokens_extracted": 115,
                "extracted_fields": {
                    "chart_title": "Q3 2026 Enterprise SaaS Expansion",
                    "peak_revenue_month": "September ($2.4M)",
                    "average_margin": "78.4%",
                    "growth_trajectory": "+28% Quarter-over-Quarter"
                },
                "vqa_answer": "Gross margin peaked in August at 81.2% before stabilizing at 78.4% in September due to new data center provisioning."
            }
        }

    def simulate_document_pipeline(self, doc_key: str) -> Dict[str, Any]:
        return self.preset_documents.get(doc_key, self.preset_documents["Commercial Vendor Invoice (VDU)"])


# --- LESSON HEADER ---
st.title("30.2 Image Applications: OCR, Classification, Document AI & VQA")
st.markdown("""
Vision-Language Models have transformed computer vision from fragmented specialized pipelines (YOLO for detection, Tesseract for OCR, ResNet for classification)
into unified zero-shot visual reasoning systems. In LangChain, developers build high-precision enterprise applications
spanning document understanding, quality audits, technical chart extraction, and Visual Question Answering (VQA).
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Visual Intelligence & OCR Pipelines", divider="rainbow")
st.markdown("""
An enterprise visual intelligence architecture ingests heterogeneous images (scanned PDFs, smartphone photos, camera feeds)
and routes them through a structured multimodal extraction pipeline:
1. **Preprocessing & Normalization**: Scales, auto-rotates, and denoises image assets before tokenization.
2. **Visual Feature Extraction**: Vision encoders decompose spatial layout, typography, tabular structures, and graphical entities.
3. **Structured Schema Grounding**: Pydantic models enforce type validation and range constraints on extracted fields.
4. **Validation & Verification**: Cross-references optical numbers against arithmetic sums (e.g., verifying invoice subtotals against total).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VisualIntelligence_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_input {
            label = "Visual Ingestion Layer";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            raw_img    [label="Raw Image Input\\nScanned PDF / JPEG / PNG", fillcolor="#0F172A"];
            preprocess [label="Preprocessing & Validation\\nOrientation, Deskew, Contrast, Resolution Check", fillcolor="#1E3A8A"];
        }

        subgraph cluster_vlm {
            label = "Multimodal Vision-Language Reasoning";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            vlm_engine [label="Frontier Vision Model\\nClaude 3.5 Sonnet / GPT-4o / Gemini 1.5 Pro", fillcolor="#0369A1"];
            prompt_eng [label="System Prompt & Task Framing\\n'Extract tabular grid and verify column arithmetic'", fillcolor="#0284C7"];
        }

        subgraph cluster_output {
            label = "Structured Extraction & Downstream Consumer";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            pydantic_val [label="Pydantic Type Validation\\nStrict Schema Enforcement & Bounding Boxes", fillcolor="#047857"];
            arithmetic   [label="Mathematical Reconciliation\\nAssert: Subtotal + Tax == Total", fillcolor="#059669"];
            enterprise   [label="Downstream Database / ERP\\nPostgres / SAP / Salesforce", fillcolor="#10B981"];
        }

        raw_img    -> preprocess;
        preprocess -> vlm_engine;
        prompt_eng -> vlm_engine;
        vlm_engine -> pydantic_val [label="Structured JSON Response"];
        pydantic_val -> arithmetic [label="Validated Entity Graph"];
        arithmetic   -> enterprise [label="Verified Payload Ingestion"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)


# --- SECTION 2: ZERO-SHOT IMAGE CLASSIFICATION & STRUCTURED TAGGING ---
st.header("2. Zero-Shot Image Classification and Structured Tagging", divider="rainbow")
st.markdown("""
Traditional image classification required training custom Convolutional Neural Networks (CNNs) on thousands of labeled examples per class.
With LangChain and Vision-Language Models, zero-shot classification is achieved by prompting the model with a fixed taxonomy
enforced by a Pydantic `Enum` or `Literal` schema.
""")

col_cls1, col_cls2 = st.columns(2)
with col_cls1:
    st.markdown("""
    ### Why VLMs Outperform Classical Classifiers
    - **Open-Vocabulary Flexibility**: Classify into novel categories instantly without retraining or collecting thousands of labeled training images.
    - **Hierarchical Classification**: Classify both macro category (e.g. "Medical Imaging") and micro sub-category (e.g. "Contrast-Enhanced Chest CT Scan").
    - **Reasoning Rationales**: In addition to the category label, the model outputs a textual explanation detailing the visual evidence supporting its decision.
    """)

with col_cls2:
    st.markdown("### Zero-Shot Image Classification Implementation")
    st.code("""
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class DocumentCategory(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    ID_CARD = "id_card"
    TECHNICAL_DIAGRAM = "technical_diagram"
    LEGAL_CONTRACT = "legal_contract"
    UNCLASSIFIED = "unclassified"

class ClassificationResult(BaseModel):
    primary_category: DocumentCategory
    confidence_score: float = Field(ge=0.0, le=1.0)
    visual_cues: list[str] = Field(description="Visual indicators observed in the image")
    requires_human_review: bool

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_classifier = llm.with_structured_output(ClassificationResult)
    """, language="python")


# --- SECTION 3: DEEP IMAGE ANALYSIS & COMPARATIVE INSPECTION ---
st.header("3. Deep Image Analysis and Comparative Inspection", divider="rainbow")
st.markdown("""
Enterprise inspection tasks demand granular visual anomaly detection: identifying micro-cracks in industrial components,
evaluating retail product shelf-stocking, or performing before-and-after construction audits.
""")

col_ia1, col_ia2 = st.columns(2)
with col_ia1:
    st.markdown("### Comparative Inspection Architecture")
    st.markdown("""
    When comparing two images (e.g., golden master reference vs manufactured unit), pass both image URLs in a single `HumanMessage`.
    The model cross-attends across spatial patches of both images simultaneously:
    """)
    st.code("""
comparison_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Compare Image 1 (Golden Reference) with Image 2 (Sample from Line 4). Identify all missing components, solder bridges, or misalignments."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
)
    """, language="python")

with col_ia2:
    st.markdown("### Defect Localization Schema")
    st.code("""
class AnomalySeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class DetectedAnomaly(BaseModel):
    description: str
    severity: AnomalySeverity
    bounding_box: list[int] = Field(
        description="[ymin, xmin, ymax, xmax] coordinates normalized 0-1000"
    )
    recommended_action: str

class QualityAuditReport(BaseModel):
    pass_inspection: bool
    total_anomalies: int
    anomalies: list[DetectedAnomaly]
    """, language="python")


# --- SECTION 4: OCR WORKFLOWS & HYBRID TEXT-VISION PIPELINES ---
st.header("4. OCR Workflows and Hybrid Text-Vision Pipelines", divider="rainbow")
st.markdown("""
Optical Character Recognition (OCR) converts pixel arrays into textual tokens.
Engineering teams face a critical architectural decision: **Traditional OCR (Tesseract / PaddleOCR / AWS Textract)** vs **Native VLM End-to-End OCR (GPT-4o / Claude 3.5 Sonnet)**.
""")

col_ocr_tbl, col_ocr_diag = st.columns(2)
with col_ocr_tbl:
    st.markdown("### Architectural Trade-off Matrix")
    ocr_tradeoffs = {
        "Metric": ["Word Accuracy", "Layout Understanding", "Handwriting Recognition", "Inference Cost", "Latency", "Structured Output"],
        "Tesseract OCR": ["82% - 88%", "Poor (Linear stream)", "Very Low", "Zero ($0 / open-source)", "50ms - 150ms", "None (Raw text)"],
        "AWS Textract": ["94% - 97%", "Good (Table blocks)", "Moderate", "$0.015 / page", "800ms - 2000ms", "JSON block tree"],
        "Frontier VLM": ["98% - 99.5%", "State of the Art", "High (Contextual)", "$0.005 - $0.02 / page", "600ms - 1500ms", "Native Pydantic"]
    }
    st.dataframe(pd.DataFrame(ocr_tradeoffs), use_container_width=True, hide_index=True)

with col_ocr_diag:
    st.markdown("### Hybrid Strategy: OCR-Augmented VLM Prompt")
    st.markdown("""
    In high-throughput enterprise pipelines, combine fast local OCR with VLM reasoning:
    1. Run fast OCR to extract raw text tokens and character coordinates.
    2. Pass both the image **and** the OCR text stream into the VLM prompt.
    3. The VLM resolves OCR misspellings using contextual visual disambiguation.
    """)
    st.code("""
hybrid_prompt = f\"\"\"
The following raw OCR text was extracted by Tesseract:
```
{raw_ocr_stream}
```
Using the attached image to resolve any illegible characters,
reconstruct the complete structured invoice table.
\"\"\"
    """, language="python")


# --- SECTION 5: VISUAL DOCUMENT UNDERSTANDING (VDU) & FORM EXTRACTION ---
st.header("5. Visual Document Understanding and Form Extraction", divider="rainbow")
st.markdown("""
Visual Document Understanding (VDU) goes beyond character extraction: it models spatial geometry, hierarchical nesting,
tabular alignment, and semantic key-value relationships in complex enterprise forms (bills of lading, tax filings, healthcare claims).
""")

col_vdu1, col_vdu2 = st.columns(2)
with col_vdu1:
    st.subheader("Tabular Grid & Key-Value Extraction")
    st.markdown("""
    Multi-column invoices frequently break classical text parsers because text wrapping interleaves adjacent column items.
    VLMs preserve 2D coordinate geometry, correctly binding item descriptions to their corresponding quantities and prices.
    """)
    st.code("""
class InvoiceLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float

class EnterpriseInvoiceSchema(BaseModel):
    invoice_id: str
    vendor_tax_id: str
    issue_date: str
    due_date: str
    currency: str
    line_items: list[InvoiceLineItem]
    subtotal: float
    tax_amount: float
    grand_total: float
    """, language="python")

with col_vdu2:
    st.subheader("Deterministic Reconciliation Guardrails")
    st.markdown("""
    Never trust raw VLM arithmetic blindly. Always enforce a deterministic verification layer in Python post-extraction:
    """)
    st.code("""
def verify_invoice_arithmetic(inv: EnterpriseInvoiceSchema) -> tuple[bool, str]:
    computed_subtotal = sum(item.line_total for item in inv.line_items)
    computed_total = computed_subtotal + inv.tax_amount
    
    # Check within 2 cents tolerance for rounding
    if abs(computed_total - inv.grand_total) > 0.02:
        return False, f"Discrepancy: Extracted total {inv.grand_total} != computed {computed_total}"
        
    return True, "Arithmetic perfectly reconciled"
    """, language="python")


# --- SECTION 6: VISUAL QUESTION ANSWERING (VQA) ON CHARTS & SCHEMAS ---
st.header("6. Visual Question Answering (VQA) on Charts and Technical Schemas", divider="rainbow")
st.markdown("""
Charts, architectural blueprints, and scientific figures encode dense quantitative information that text-based RAG cannot interpret.
Visual Question Answering (VQA) chains allow users to ask conversational, multi-turn questions about charts and schematics.
""")

col_vqa1, col_vqa2 = st.columns(2)
with col_vqa1:
    st.markdown("""
    ### Chart Reasoning Patterns
    - **Dual-Axis Cross-Referencing**: Correlating values between a primary left axis (e.g. Dollar Revenue) and secondary right axis (e.g. Growth Percentage).
    - **Trend Extrapolation**: Identifying inflection points, outliers, and trend slopes without accessing underlying CSV data.
    - **Color & Legend Mapping**: Accurately mapping visual legend swatches (e.g. blue dashed vs green solid line) to corresponding data series.
    """)

with col_vqa2:
    st.markdown("### VQA Interactive Chain Pattern")
    st.code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

def query_technical_chart(chart_base64: str, question: str, vlm_model) -> str:
    messages = [
        SystemMessage(content="You are an expert quantitative data visualization analyst. Provide exact numbers, trend analyses, and cite specific visual indicators."),
        HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_base64}"}}
        ])
    ]
    response = vlm_model.invoke(messages)
    return response.content
    """, language="python")


# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Visual Inspection & Structured Extraction Engine", divider="rainbow")
st.markdown("""
A production-grade Visual Document Understanding engine handles base64 encoding, schema binding,
retry backoffs on schema validation errors, and deterministic reconciliation.
""")

st.code("""
import base64
from typing import Type, TypeVar, Optional, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

T = TypeVar("T", bound=BaseModel)

class ProductionVisualDocumentExtractor:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def extract_structured_document(
        self,
        image_bytes: bytes,
        schema: Type[T],
        mime_type: str = "image/png",
        instruction: Optional[str] = None
    ) -> Tuple[Optional[T], Optional[str]]:
        \"\"\"
        Extracts structured data from an image buffer according to a Pydantic schema.
        Returns (parsed_instance, error_message).
        \"\"\"
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        system_prompt = (
            "You are a high-precision document extraction engine. "
            "Extract all text, numbers, dates, and tables exactly as printed. "
            "Do not hallucinate or extrapolate missing values."
        )
        
        user_prompt = instruction or "Extract all structured information from this document image."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                        "detail": "high"
                    }
                }
            ]
        )
        
        structured_llm = self.llm.with_structured_output(schema)
        
        try:
            result = structured_llm.invoke([SystemMessage(content=system_prompt), message])
            return result, None
        except Exception as e:
            return None, str(e)
""", language="python")


# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Visual Document Analyzer & VQA Simulator", divider="rainbow")
st.markdown("""
Test visual intelligence extraction across multiple document types:
a commercial invoice, an industrial QA circuit board audit, and an executive growth trend chart.
Observe Pydantic schema parsing, OCR token counts, and conversational visual question answering.
""")

sim = VisualDocumentWorkbenchSimulator()

col_sim_cfg, col_sim_res = st.columns([1, 1])

with col_sim_cfg:
    st.subheader("Document Selection & Query")
    doc_selection = st.selectbox(
        "Choose Inspection Scenario:",
        list(sim.preset_documents.keys()),
        index=0
    )
    
    custom_question = st.text_input(
        "Visual Question Answering Query:",
        value="What is the primary conclusion or total sum shown in this document?"
    )
    
    run_btn = st.button("🔍 Run Visual Extraction Pipeline", use_container_width=True)

with col_sim_res:
    st.subheader("Structured Extraction Telemetry")
    if run_btn:
        doc_data = sim.simulate_document_pipeline(doc_selection)
        
        st.markdown(
            f"<div class='vision-card'>"
            f"<b>Category:</b> <span class='badge-vdu'>{doc_data['category']}</span><br><br>"
            f"<b>Classification Confidence:</b> {doc_data['confidence'] * 100:.1f}%<br>"
            f"<b>OCR Tokens Ingested:</b> {doc_data['ocr_tokens_extracted']} tokens<br>"
            f"<b>Inspection Audit:</b> {doc_data['inspection_summary']}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("### Structured Schema Output (JSON)")
        st.json(doc_data["extracted_fields"])
        
        st.markdown("### Visual Question Answering (VQA) Response")
        st.success(f"**VLM Response**: {doc_data['vqa_answer']}")
    else:
        st.info("Select a document scenario and click Run to test structured visual extraction.")


# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why do Vision-Language Models outperform classical OCR on multi-column tables?"):
    st.markdown("""
    **Answer**: Traditional OCR systems (like Tesseract) rasterize text line-by-line horizontally without understanding semantic layouts.
    On multi-column layouts, they frequently read across columns, concatenating unrelated text lines. VLMs process 2D spatial patches
    holistically, recognizing column dividers, gutters, and header boundaries to preserve tabular integrity.
    """)

with st.expander("Q2: How do you prevent hallucinations when extracting numerical fields from financial documents?"):
    st.markdown("""
    **Answer**: Use a multi-defense approach:
    1. Set temperature to `0.0`.
    2. Enforce strict Pydantic schemas using `.with_structured_output(...)`.
    3. Include explicit system instructions: 'If a value is obscured or absent, return null instead of estimating.'
    4. Implement programmatic Python arithmetic reconciliation: assert that line items sum up to the extracted grand total.
    """)

with st.expander("Q3: What is the token cost difference between 'low' and 'high' detail modes in GPT-4o?"):
    st.markdown("""
    **Answer**: In `low` detail mode, the image is scaled to a single 512x512 tile costing a flat 85 tokens.
    In `high` detail mode, the image is scaled and split into multiple 512x512 tiles, costing 170 tokens per tile plus an 85 token base.
    A high-resolution document requiring 6 tiles costs `85 + (6 * 170) = 1,105 tokens`.
    """)

with st.expander("Q4: When should you use a dedicated OCR engine (like AWS Textract) alongside a VLM?"):
    st.markdown("""
    **Answer**: In high-volume compliance environments (e.g., millions of pages/day), running pure frontier VLMs on every page may be cost-prohibitive.
    A hybrid architecture uses fast OCR for standard text and triggers the VLM only for complex unstructured tables, handwritten annotations,
    or flagged edge cases.
    """)

with st.expander("Q5: How can bounding boxes extracted from a VLM be verified for accuracy?"):
    st.markdown("""
    **Answer**: Bounding box coordinates normalized to `[0, 1000]` can be mapped back to original image pixel coordinates:
    `x_pixel = (xmin / 1000) * original_width`. A programmatic OpenCV or PIL script can crop the bounded region and verify
    that character contents match the label via lightweight local OCR.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Trusting VLM Arithmetic Without Re-Check)**: Accepting computed totals from model outputs without deterministic Python code verification. Models can hallucinate decimal places.")
    st.warning("**Trap 2 (Ignoring Image Rotation & Skew)**: Ingesting upside-down or 90-degree rotated smartphone photos. Always run an orientation detector before model inference.")
with col_trap2:
    st.error("**Trap 3 (Sending Uncompressed 300 DPI Scans)**: Transmitting 20MB raw TIFF/PNG files directly over HTTP. Always compress to high-quality WebP or JPEG (150-200 DPI) to prevent payload timeouts.")
    st.warning("**Trap 4 (Omitting Structured Pydantic Schemas)**: Asking the model for freeform Markdown tables. Freeform tables require brittle regex parsing that breaks on edge-case columns.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is Visual Document Understanding (VDU)?")
        st.caption("Flip for answer:")
        st.info("**Back**: The extraction and reasoning over visual structure, layout, typography, and tabular data in document images.")
    with st.container(border=True):
        st.markdown("**Front**: How should numerical totals in invoices be validated?")
        st.caption("Flip for answer:")
        st.info("**Back**: Using deterministic Python arithmetic verification: asserting subtotal + tax == grand total.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary limitation of classical Tesseract OCR?")
        st.caption("Flip for answer:")
        st.info("**Back**: Inability to understand 2D semantic layouts; frequently merges text across multiple adjacent columns.")
    with st.container(border=True):
        st.markdown("**Front**: How are bounding boxes mapped from normalized coordinates (0-1000) to pixels?")
        st.caption("Flip for answer:")
        st.info("**Back**: Multiply normalized coordinate by image dimension: `pixel = (coord / 1000) * dimension`.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand zero-shot image classification using Pydantic Enums in LangChain.", value=True)
c2 = st.checkbox("I can build multi-image comparative inspection prompts for before/after visual audits.", value=True)
c3 = st.checkbox("I know the trade-offs between classical OCR engines and native VLM extraction.", value=True)
c4 = st.checkbox("I can design Visual Document Understanding (VDU) schemas for complex tabular forms.", value=True)
c5 = st.checkbox("I always implement deterministic Python arithmetic verification on extracted financial totals.", value=True)
c6 = st.checkbox("I know how to build Visual Question Answering (VQA) chains for technical charts and schematics.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 30.3: Multimodal Agents.")
