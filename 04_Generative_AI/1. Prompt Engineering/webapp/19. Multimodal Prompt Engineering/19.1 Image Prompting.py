import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="19.1 Image Prompting | GenAI Roadmap",
    page_icon="🖼️",
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
    .vision-chip {
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
st.title("🖼️ 19.1 Image Prompting — Vision-Language Models, VQA, OCR & Spatial Reasoning")
st.markdown(
    """
Welcome to **Lesson 19.1: Image Prompting**.  
Generative AI has evolved beyond text-only interfaces into **Vision-Language Models (VLMs)** such as GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro.

Visual prompting is the discipline of structuring text instructions alongside images: extracting text via **deep OCR**, performing **Visual Question Answering (VQA)**, identifying **fine-grained anomalies**, and predicting **spatial bounding box coordinates** for object localization.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Vision-Language Model (VLM) Architecture
# ==============================================================================
st.header("1. How Vision-Language Models Process Images", divider="rainbow")

col_vlm1, col_vlm2 = st.columns(2)

with col_vlm1:
    with st.container(border=True):
        st.subheader("🧩 Image Patchification & Visual Tokens")
        st.markdown(
            """
- An image is not fed into an LLM as raw pixel matrices.
- A **Vision Encoder** (typically a Vision Transformer / ViT like CLIP or SigLIP) divides the image into a grid of small patches (e.g. 14x14 pixels).
- Each patch is projected through a linear layer into a **dense visual embedding vector**.
- These visual vectors are treated by the LLM exactly like text token embeddings, allowing self-attention heads to attend freely across words and image regions.
"""
        )

with col_vlm2:
    with st.container(border=True):
        st.subheader("📐 High-Detail vs. Low-Detail Tiling")
        st.markdown(
            """
- **Low-Detail Mode:** Resizes the image to a fixed low resolution (e.g. 512x512) and uses ~85 tokens. Best for general scene classification or quick sentiment checks.
- **High-Detail Mode:** Crops the high-resolution image into multiple 512x512 tiles plus a thumbnail. Consumes ~765 to 1,500+ tokens. Mandatory for reading fine-print OCR, receipt line items, and inspecting complex diagrams.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Core Vision Prompting Tasks Matrix
# ==============================================================================
st.header("2. Core Vision Prompting Tasks Matrix", divider="rainbow")
st.markdown("Structuring visual prompts across primary computer vision applications:")

vision_tasks_df = pd.DataFrame({
    "Vision Task": [
        "1. Visual Question Answering (VQA)",
        "2. High-Precision OCR",
        "3. Defect & Anomaly Inspection",
        "4. Spatial Grounding & Bounding Boxes",
        "5. Chart & Infographic Reasoning"
    ],
    "Input Prompt Formulation": [
        "Provide image + explicit analytical query asking for evidence-grounded facts",
        "Instruct model to transcribe all visible text verbatim, preserving row breaks and tables",
        "Provide domain reference rubric (e.g. scratch, dent, discoloration) + inspect image",
        "Instruct model to return object coordinates normalized to [ymin, xmin, ymax, xmax] on a 0-1000 scale",
        "Ask model to extract numerical axis labels, legend mappings, and data points into JSON"
    ],
    "Best Practice Constraint": [
        "Mandate: 'If details are blurry or occluded, state UNCERTAIN rather than guessing'",
        "Mandate: 'Do not correct spelling errors found in the physical image; transcribe raw OCR'",
        "Require bounding box coordinates surrounding the detected defect",
        "Enforce strict 4-tuple integer arrays formatted as JSON",
        "Verify data by recalculating column sums before generating narrative takeaway"
    ]
})
st.dataframe(vision_tasks_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Vision-Language Encoding Pipeline", divider="rainbow")
st.markdown("How images and text instructions flow through the multimodal transformer:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        img [label="High-Res Image Input\n(Receipt / Diagram / Photo)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        vit [label="Vision Transformer (ViT)\nPatchifier (14x14 grid)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        v_tokens [label="Visual Tokens\n(Tiled Grid Vectors)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        prompt [label="Text System Prompt\n& Extraction Schema", fillcolor="#EAB308", fontcolor="#000000"]
        multimodal_llm [label="Multimodal Transformer\nCross-Attention Core", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        out [label="Structured JSON\nwith Bounding Boxes", fillcolor="#10B981", fontcolor="#FFFFFF"]

        img -> vit -> v_tokens -> multimodal_llm
        prompt -> multimodal_llm
        multimodal_llm -> out
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Image Prompting Workbench
# ==============================================================================
st.header("4. Interactive Image Prompting Workbench", divider="rainbow")
st.markdown("Test prompt structures across different visual inspection archetypes:")

image_archetype = st.selectbox(
    "Select Image Inspection Scenario",
    [
        "Scanned Restaurant Receipt (OCR & Tax Extraction)",
        "Industrial Manufacturing Defect Inspection",
        "AWS Cloud Architecture Diagram Analysis"
    ]
)

col_iw1, col_iw2 = st.columns([1, 1])

with col_iw1:
    st.subheader("🖼️ Simulated Visual Artifact")
    if image_archetype == "Scanned Restaurant Receipt (OCR & Tax Extraction)":
        st.markdown(
            """
```text
+-----------------------------------+
|          BISTRO PARISIEN          |
|        Date: 2026-10-12           |
|                                   |
| 1x Espresso Doppio        $ 4.50  |
| 2x Croissant Beurre       $ 8.00  |
| 1x Tartine Saumon        $ 16.50  |
|                                   |
| Subtotal:                $ 29.00  |
| Tax (8.5%):              $  2.47  |
| Total:                   $ 31.47  |
+-----------------------------------+
```
"""
        )
    elif image_archetype == "Industrial Manufacturing Defect Inspection":
        st.markdown(
            """
```text
+-----------------------------------+
|  [Titanium Turbine Blade Photo]   |
|                                   |
|   (o) Crack detected at edge      |
|       Coordinates: [320, 410]     |
|       Length: ~4.2mm              |
|                                   |
|   Surface Finish: Micro-scratched |
+-----------------------------------+
```
"""
        )
    else:
        st.markdown(
            """
```text
+-----------------------------------+
|  CloudFront CDN                   |
|       |                           |
|       v                           |
|  ALB (Application Load Balancer)  |
|       |                           |
|       +---> ECS Microservice A    |
|       +---> ECS Microservice B    |
|                |                  |
|                v                  |
|          Amazon RDS Aurora        |
+-----------------------------------+
```
"""
        )

with col_iw2:
    st.subheader("📤 Extracted Structured VLM Payload")
    if image_archetype == "Scanned Restaurant Receipt (OCR & Tax Extraction)":
        extracted_data = {
            "merchant": "BISTRO PARISIEN",
            "transaction_date": "2026-10-12",
            "line_items": [
                {"name": "Espresso Doppio", "qty": 1, "price": 4.50},
                {"name": "Croissant Beurre", "qty": 2, "price": 8.00},
                {"name": "Tartine Saumon", "qty": 1, "price": 16.50}
            ],
            "subtotal": 29.00,
            "tax_amount": 2.47,
            "total_usd": 31.47,
            "tax_rate_verified": True
        }
        st.json(extracted_data)
        st.success("✅ Perfect OCR tabular line-item extraction with mathematical validation.")

    elif image_archetype == "Industrial Manufacturing Defect Inspection":
        extracted_data = {
            "component": "Titanium Turbine Blade",
            "defect_identified": True,
            "defect_type": "STRUCTURAL_STRESS_FRACTURE",
            "severity": "CRITICAL_REJECT",
            "bounding_box_normalized": [310, 400, 345, 435],
            "estimated_length_mm": 4.2,
            "recommended_action": "SCRAP_AND_LOG_METALLURGY_BATCH"
        }
        st.json(extracted_data)
        st.warning("⚠️ High-severity manufacturing defect flagged with bounding box coordinates.")

    else:
        extracted_data = {
            "architecture_type": "Web Microservice Cluster",
            "edge_layer": "AWS CloudFront",
            "ingress": "Application Load Balancer (ALB)",
            "compute": ["ECS Microservice A", "ECS Microservice B"],
            "persistence": "Amazon RDS Aurora",
            "single_point_of_failure_audit": "ALB is redundant across AZs, but Aurora should verify multi-AZ replication configuration."
        }
        st.json(extracted_data)
        st.info("ℹ️ Architectural topology mapped directly from visual diagram.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Multimodal Vision API Payloads", divider="rainbow")
st.markdown("Python code formatting base64-encoded images for OpenAI and Claude vision endpoints:")

st.code(
    '''import base64
from typing import Dict, Any

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_vision_chat_payload(base64_image_str: str, user_question: str) -> Dict[str, Any]:
    """
    Constructs an OpenAI-compatible multimodal message array with high-detail image tiling.
    """
    return {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert OCR and visual inspection engine. Output strictly valid JSON."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_str}",
                            "detail": "high"  # High detail enables 512x512 tile cropping
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1
    }

print("Vision Chat Payload Builder compiled successfully with high-detail tiling parameters.")
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
    with st.expander("Q1: What is the difference between setting `detail: 'low'` and `detail: 'high'` in OpenAI vision API requests?", expanded=True):
        st.markdown(
            """
- **`detail: 'low'`**: Downsamples the image to a low 512x512 resolution and costs a flat **85 tokens**. It is fast and cheap, but cannot read small text, blurred barcodes, or subtle engineering schematic lines.
- **`detail: 'high'`**: First scales the image to fit within 2048x2048, then divides it into multiple 512x512 squares. Each square costs **170 tokens** plus an initial 85 base tokens. An image tiled into 4 squares costs `(4 * 170) + 85 = 765 tokens`. High detail is mandatory for OCR, document analysis, and fine-grained visual defect detection.
"""
        )
    with st.expander("Q2: How do you prompt a Vision-Language Model to perform spatial bounding box localization?"):
        st.markdown(
            """
Instruct the model to output coordinates normalized to a fixed integer scale (typically `[0, 1000]`):
- Specify the standard coordinate convention: `[ymin, xmin, ymax, xmax]` where `[0, 0]` is top-left and `[1000, 1000]` is bottom-right.
- Require the output to be a list of typed JSON objects: `{"label": "defect", "box_2d": [ymin, xmin, ymax, xmax]}`.
- Modern VLMs (Claude 3.5, Gemini 1.5, GPT-4o) have strong spatial attention alignment and can accurately locate objects within pixel boundaries.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we rely on Vision LLMs to extract exact numbers from complex blurry graphs without verification?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, GPT-4o has human-level vision and always reads the exact pixel values."*
- **Winning Response**: *"Never. When charts have dense gridlines, overlapping bars, or compressed JPEG artifacts, VLMs frequently misread values by 5-15% because fine details blur across adjacent patch embeddings. For high-stakes financial data, prompt the model to extract the raw data points into a table, calculate checksums (e.g. asserting that percentages sum to 100%), and flag low-contrast text regions as 'UNCERTAIN' for human verification."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Patchification**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The process in Vision Transformers (ViT) of dividing an image into a grid of small pixel squares and projecting them into token embeddings.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**detail='high' Tiling**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Cropping a high-resolution image into multiple 512x512 tiles (170 tokens each) to preserve fine text and OCR readability.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Spatial Grounding**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Prompting a VLM to predict normalized bounding box coordinates [ymin, xmin, ymax, xmax] to localize specific visual objects.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 19.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Multimodal Tokens**: Vision encoders patchify images into vectors that interact directly with text tokens.
2. **Detail Mode Trade-Off**: Use low-detail (85 tokens) for classification, high-detail (700-1500 tokens) for OCR.
3. **Structured Extraction**: Always prompt for typed JSON schemas with explicit verification checksums.
4. **Spatial Grounding**: Use normalized bounding box coordinates for object and defect localization.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand how Vision Transformers (ViT) convert image patches into token embeddings.", value=True, key="chk_19_1_1")
    st.checkbox("I know the token cost and resolution mechanics of high-detail vs. low-detail tiling.", value=True, key="chk_19_1_2")
    st.checkbox("I can implement spatial bounding box extraction prompts.", value=True, key="chk_19_1_3")
    st.checkbox("I know how to build multimodal base64 image chat payloads in Python.", value=True, key="chk_19_1_4")
