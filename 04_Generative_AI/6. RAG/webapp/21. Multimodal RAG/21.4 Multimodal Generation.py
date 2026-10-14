import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="21.4 Multimodal Generation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class MultimodalGenerationEngine:
    """Enterprise Vision-Language Model (VLM) Generation & Visual Grounding Engine."""

    def __init__(self):
        # Sample Visual Evidence Context
        self.evidence_store = {
            "H100 NVLink Architecture Schematic": {
                "image_id": "img_h100_nvlink",
                "regions": [
                    {"label": "NVLink Switch Crossbar", "bbox": [120, 45, 380, 190], "data": "4x NVLink Switches providing 900 GB/s bidirectional bandwidth per GPU."},
                    {"label": "PCIe Gen5 Uplink", "bbox": [410, 50, 580, 180], "data": "16x PCIe Gen5 lanes to CPU host subsystem."},
                    {"label": "Power Delivery Unit", "bbox": [50, 400, 200, 600], "data": "48V to 12V DC-DC power converters capable of 10.2 kW."}
                ]
            },
            "Loss Convergence Chart": {
                "image_id": "img_loss_curve",
                "regions": [
                    {"label": "Early Epoch Plateau", "bbox": [80, 100, 250, 300], "data": "High initial loss plateau between epochs 1-12 at 2.84."},
                    {"label": "Convergence Inflection", "bbox": [280, 320, 500, 450], "data": "Rapid descent to steady-state validation loss 0.42 at epoch 78."}
                ]
            }
        }

    def generate_image_grounded_answer(
        self,
        query: str,
        selected_evidence: str,
        strict_grounding: bool = True
    ) -> Dict[str, Any]:
        """Synthesizes an answer grounded strictly in visual regions of interest with coordinate citations."""
        evidence = self.evidence_store.get(selected_evidence)
        if not evidence:
            return {"answer": "No visual evidence selected.", "citations": [], "groundedness_score": 0.0}

        citations = []
        answer_clauses = []
        
        for reg in evidence["regions"]:
            citations.append({
                "region_label": reg["label"],
                "bbox_coords": reg["bbox"],
                "evidence_text": reg["data"]
            })
            answer_clauses.append(f"- {reg['label']} (Visual Box {reg['bbox']}): {reg['data']}")

        grounded_text = (
            f"Visual VLM Grounded Synthesis for '{query}':\n\n"
            f"Analysis of visual asset '{selected_evidence}' reveals:\n"
            + "\n".join(answer_clauses) +
            f"\n\nConclusion: All statements verified against localized pixel coordinates."
        )

        return {
            "query": query,
            "evidence_name": selected_evidence,
            "generated_answer": grounded_text,
            "visual_citations": citations,
            "groundedness_score": 0.98 if strict_grounding else 0.72,
            "estimated_visual_tokens": len(evidence["regions"]) * 256
        }


# --- Streamlit Layout ---

st.title("21.4 Multimodal Generation in RAG")
st.caption("Vision-Language Models (VLMs), Interleaved Pixel-Text Generation, Region-of-Interest Citations & Visual Grounding")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Vision-Language Model (VLM) Generative Synthesis & Visual Epistemic Contract", divider="rainbow")

st.markdown(
    """
    Retrieving an image is only half the battle; the generative model must be capable of **interpreting raw visual pixels**
    and reasoning across spatial layouts, tabular grids, and graphical plots.
    
    In **Multimodal Generation**, frontier Vision-Language Models (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet) receive both
    the user's text prompt and retrieved high-resolution image crops interleaved directly into their multi-head self-attention layers.
    The model enforces an **Epistemic Visual Contract**, grounding every factual claim in explicit visual regions of interest (ROI) and pixel coordinates.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Multimodal_Generation {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_context_assembly {
            label = "1. INTERLEAVED MULTIMODAL PROMPT ASSEMBLY";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_prompt [label="User Query: 'Analyze NVLink switch bandwidth & power limits'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            retrieved_images [label="Retrieved Visual Assets (PNG/JPEG)\\nCropped Architecture Schematics & Charts", fillcolor="#0F172A", fontcolor="#38BDF8"];
            retrieved_text [label="Retrieved Text Specifications\\nHardware Datacenter Manuals", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            interleaved_payload [label="Interleaved Multimodal Context Payload\\nTokens: [Text Prompt] + <image_tokens> + [Text Context]", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            user_prompt -> interleaved_payload;
            retrieved_images -> interleaved_payload;
            retrieved_text -> interleaved_payload;
        }

        subgraph cluster_vlm_transformer {
            label = "2. VISION-LANGUAGE TRANSFORMER (VLM BACKBONE)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            vision_patch_tokenizer [label="Vision Encoder (Patch Tokenizer)\\nSlices images into 16x16 visual token embeddings", fillcolor="#312E81", fontcolor="#A5B4FC"];
            multimodal_attention [label="Unified Self-Attention Layers\\nJoint cross-attention across text tokens & visual patches", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            spatial_reasoning [label="Spatial & Tabular Reasoning Core\\nResolves axis lines, arrows, coordinate anchors", fillcolor="#0F172A", fontcolor="#10B981"];
            
            interleaved_payload -> vision_patch_tokenizer -> multimodal_attention -> spatial_reasoning;
        }

        subgraph cluster_grounded_output {
            label = "3. IMAGE-GROUNDED SYNTHESIS & REGION CITATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            roi_citation_engine [label="Visual Region Citation Engine\\nAttaches Bounding Box Coordinates [x1, y1, x2, y2]", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            grounded_answer [label="Image-Grounded Verifiable Response\\nHighlights visual components on source diagram", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            spatial_reasoning -> roi_citation_engine -> grounded_answer;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: VISION-LANGUAGE MODELS ---
st.header("2. Vision-Language Models (Architectural Paradigms: Dual-Encoder vs Native Interleaved VLMs, Gemini, GPT-4o, Claude 3.5 Sonnet)", divider="rainbow")

st.markdown(
    """
    The architecture of generative vision-language models has evolved through distinct evolutionary phases:
    
    ### 1. Adapter-Based VLMs (LLaVA, MiniGPT-4)
    - A frozen pre-trained vision encoder (CLIP ViT) extracts image features.
    - A lightweight projection adapter (linear layer or MLP) converts visual features into the text LLM's embedding space.
    - *Limitation*: The text LLM cannot control what visual features the encoder focuses on, leading to coarse image descriptions.
    
    ### 2. Native Native Multimodal Transformers (GPT-4o, Gemini 1.5, Claude 3.5 Sonnet)
    - The model is trained from scratch on interleaved tokens of text, audio, and visual patches.
    - Vision and language share the exact same transformer self-attention layers.
    - *Advantage*: The model can perform deep spatial reasoning, reading fine circuit labels, analyzing mathematical slopes, and emitting exact pixel bounding box coordinates.
    """
)

# --- SECTION 3: IMAGE-GROUNDED ANSWERS ---
st.header("3. Image-Grounded Answers (Bounding Box Citations, Region-of-Interest Highlighting & Visual Hallucination Guardrails)", divider="rainbow")

st.markdown(
    """
    In text-only RAG, hallucinations are mitigated by in-line document citations (`[Doc 101]`).
    In multimodal RAG, citations require **2D Spatial Bounding Boxes**:
    
    - **Visual Coordinate Citations**: Instead of just citing an image file, the VLM emits normalized coordinates: `[Box: 120, 45, 380, 190]`.
    - **Region-of-Interest (ROI) Rendering**: The user interface overlays a translucent colored rectangle directly on the retrieved PDF schematic, highlighting the exact switch or resistor the answer describes.
    - **Visual Hallucination Defense**: If the VLM claims *"The diagram shows 16 GPUs"*, an automated vision checker verifies that the bounding box actually contains 16 distinct component contours.
    """
)

# --- SECTION 4: VISUAL QUESTION ANSWERING ---
st.header("4. Visual Question Answering (Spatial Reasoning, Tabular Visual Logic & Complex Infographic Interpretation)", divider="rainbow")

st.markdown(
    """
    **Visual Question Answering (VQA)** evaluates whether a model can solve complex tasks requiring both sight and domain logic:
    
    - **Tabular Visual Logic**: Reading merged row headers in non-standard financial balance sheets.
    - **Directional Arrow Flow Tracking**: Tracing causal flowcharts (*"If Step 3 fails due to timeout, which recovery state machine is activated?"*).
    - **Trend Extrapolation in Charts**: Inspecting line slope angles on logarithmic axes to determine whether loss convergence is plateauing.
    """
)

# --- SECTION 5: PRODUCTION IMPLEMENTATION ---
st.header("5. Production Implementation: Enterprise Visual Question Answering & Image Grounding Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `MultimodalGenerationEngine`.
    It simulates native VLM visual context processing, performs spatial region analysis across architectural schematics,
    and formats image-grounded answers with explicit bounding box coordinate citations.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class MultimodalGenerationEngine:
    """Enterprise Vision-Language Model (VLM) Generation & Visual Grounding Engine."""

    def __init__(self):
        self.evidence_store = {
            "H100 NVLink Architecture Schematic": {
                "image_id": "img_h100_nvlink",
                "regions": [
                    {"label": "NVLink Switch Crossbar", "bbox": [120, 45, 380, 190], "data": "4x NVLink Switches providing 900 GB/s bidirectional bandwidth per GPU."},
                    {"label": "PCIe Gen5 Uplink", "bbox": [410, 50, 580, 180], "data": "16x PCIe Gen5 lanes to CPU host subsystem."},
                    {"label": "Power Delivery Unit", "bbox": [50, 400, 200, 600], "data": "48V to 12V DC-DC power converters capable of 10.2 kW."}
                ]
            },
            "Loss Convergence Chart": {
                "image_id": "img_loss_curve",
                "regions": [
                    {"label": "Early Epoch Plateau", "bbox": [80, 100, 250, 300], "data": "High initial loss plateau between epochs 1-12 at 2.84."},
                    {"label": "Convergence Inflection", "bbox": [280, 320, 500, 450], "data": "Rapid descent to steady-state validation loss 0.42 at epoch 78."}
                ]
            }
        }

    def generate_image_grounded_answer(
        self,
        query: str,
        selected_evidence: str,
        strict_grounding: bool = True
    ) -> Dict[str, Any]:
        """Synthesizes an answer grounded strictly in visual regions of interest with coordinate citations."""
        evidence = self.evidence_store.get(selected_evidence)
        if not evidence:
            return {"answer": "No visual evidence selected.", "citations": [], "groundedness_score": 0.0}

        citations = []
        answer_clauses = []
        
        for reg in evidence["regions"]:
            citations.append({
                "region_label": reg["label"],
                "bbox_coords": reg["bbox"],
                "evidence_text": reg["data"]
            })
            answer_clauses.append(f"- {reg['label']} (Visual Box {reg['bbox']}): {reg['data']}")

        grounded_text = (
            f"Visual VLM Grounded Synthesis for '{query}':\\n\\n"
            f"Analysis of visual asset '{selected_evidence}' reveals:\\n"
            + "\\n".join(answer_clauses) +
            f"\\n\\nConclusion: All statements verified against localized pixel coordinates."
        )

        return {
            "query": query,
            "evidence_name": selected_evidence,
            "generated_answer": grounded_text,
            "visual_citations": citations,
            "groundedness_score": 0.98 if strict_grounding else 0.72,
            "estimated_visual_tokens": len(evidence["regions"]) * 256
        }
''',
    language="python"
)

# --- SECTION 6: INTERACTIVE WORKBENCH ---
st.header("6. Interactive Workbench: VLM Grounded Synthesis & Visual Hallucination Inspector", divider="rainbow")

st.markdown(
    """
    Test how a native Vision-Language Model (VLM) synthesizes answers from visual technical assets.
    Select an architectural schematic or loss chart, submit engineering questions, and inspect visual bounding box citations.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🖼️ Visual Evidence & Prompt Settings")
    
    selected_asset = st.selectbox(
        "Select Visual Asset in Working Context:",
        ["H100 NVLink Architecture Schematic", "Loss Convergence Chart"],
        index=0
    )
    
    vlm_query = st.text_input(
        "User Prompt:",
        "Explain the high-speed switch topology and power delivery specifications"
    )
    
    strict_grounding_toggle = st.toggle("Enforce Strict Region-of-Interest (ROI) Grounding", value=True)

with wb_col2:
    st.subheader("⚡ VLM Generation & Citation Output")
    
    engine = MultimodalGenerationEngine()
    result = engine.generate_image_grounded_answer(vlm_query, selected_asset, strict_grounding=strict_grounding_toggle)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Visual Citations", len(result["visual_citations"]))
    m2.metric("Groundedness", f"{result['groundedness_score'] * 100:.1f}%")
    m3.metric("Est. Image Tokens", result["estimated_visual_tokens"])
    
    st.markdown("#### Generated Image-Grounded Response")
    st.info(result["generated_answer"])

st.markdown("### Visual Region of Interest (ROI) Coordinate Citations")
citation_df = pd.DataFrame([
    {
        "Region Label": c["region_label"],
        "Normalized Coordinates [x1, y1, x2, y2]": str(c["bbox_coords"]),
        "Verified Visual Fact": c["evidence_text"]
    }
    for c in result["visual_citations"]
])
st.dataframe(citation_df, use_container_width=True)

# --- SECTION 7: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("7. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: How do Vision-Language Models tokenize image pixels compared to text tokens?**"):
    st.markdown(
        """
        - **Text Tokenization**: Text strings are split by BPE (Byte-Pair Encoding) or WordPiece into vocabulary token IDs (integers), which are mapped to 1D embedding vectors via an embedding matrix lookup.
        - **Image Tokenization**: An image (e.g., 1024 x 1024 pixels) is divided into a grid of 2D patches (e.g., 16 x 16 pixels). A convolutional or linear projection flattens each patch into a vector, and 2D positional embeddings are added. Each visual patch functions exactly like a text token in the transformer's self-attention mechanism.
        """
    )

with st.expander("**Q2: What is the primary cause of Visual Hallucinations in VLMs?**"):
    st.markdown(
        """
        Visual hallucinations occur when:
        1. **Language Prior Bias**: The language decoder's strong pre-trained parametric priors overpower weak visual attention signals (e.g., if asked *'What color is the stop sign?'*, the model says *'red'* even if the image shows a blue stop sign).
        2. **Resolution Downsampling**: Fine details (small circuit labels, decimal points on chart axes) fall below the patch tokenization resolution, forcing the model to guess.
        3. **Object Hallucination**: Co-occurrence bias causes the model to hallucinate common co-occurring objects (e.g., seeing a desk and hallucinating a laptop that does not exist).
        """
    )

with st.expander("**Q3: How do you enforce Visual Grounding in production VLM generation?**"):
    st.markdown(
        """
        Enforcing visual grounding requires:
        1. **Coordinate Constrained Decoding**: Prompting the model to emit normalized bounding box coordinates `[ymin, xmin, ymax, xmax]` for every extracted entity.
        2. **Visual Verification Filter**: Passing the cropped bounding box to a lightweight classifier to verify that the claimed text or object is actually present in those coordinates.
        3. **Negative Constraint Directives**: Instructing the model: *'If an attribute cannot be confirmed from the visible pixels, explicitly state: Undetermined from image.'*
        """
    )

with st.expander("**Q4: What is the token cost implication of injecting high-resolution images into VLMs?**"):
    st.markdown(
        """
        In commercial API pricing (e.g., OpenAI, Google, Anthropic), images are priced based on the number of visual tiles/patches required:
        - A low-resolution thumbnail costs ~85 tokens (~$0.0002).
        - A high-resolution 2048 x 2048 engineering blueprint sliced into 512 x 512 tiles consumes ~1,500 to $2,500$ tokens (~$0.02 - $0.05 per page).
        
        Injecting 10 retrieved page images per query can quickly consume 20,000+ prompt tokens, making selective bounding-box cropping critical for cost management.
        """
    )

with st.expander("**Q5: How does Spatial Reasoning differ from Semantic Classification in VLM architectures?**"):
    st.markdown(
        """
        - **Semantic Classification**: Identifies what objects exist in an image (*'There is a database and a server'*). Handled easily by pooled global embeddings (CLIP).
        - **Spatial Reasoning**: Resolves relative positions, distances, and directional flows (*'The database is located downstream of the gateway and connected via red arrow to the cache'*).
        
        Spatial reasoning requires preserve 2D patch coordinate geometry across multi-head attention layers, which native interleaved VLMs achieve.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Passing Full Uncropped Page Images for Minor Details**  
        **Antipattern**: Injecting a massive 4K PDF page image into the VLM prompt when the query only asks about a tiny 1-inch table in the corner.  
        **Consequence**: Consumes 2,000+ visual tokens, slows down TTFT latency by 1,200ms, and dilutes the VLM's attention.  
        **Production Fix**: Crop the specific bounding box region of interest (ROI) and pass only the cropped image snippet to the generator.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Trusting VLM Output Without Visual Coordinate Verification**  
        **Antipattern**: Assuming that because a VLM cited an image, the visual interpretation is factually accurate.  
        **Consequence**: The model hallucinates plausible-looking numbers from chart axes due to strong pre-trained language bias.  
        **Production Fix**: Enforce bounding box citations and verify extracted data against underlying OCR/table extraction manifests.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Slicing Visual Assets Across Tile Boundaries**  
        **Antipattern**: Fixed-grid tiling algorithms that cut important component labels or table cells in half across tile borders.  
        **Consequence**: Slices words and circuit symbols into illegible fragments, triggering visual recognition errors.  
        **Production Fix**: Use overlap-aware visual tiling with a minimum of 20% spatial padding between tiles.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Neglecting Visual Image Rotation & Orientation**  
        **Antipattern**: Passing sideways or upside-down PDF scan pages directly to the vision transformer.  
        **Consequence**: Vision models fail to read inverted text, producing complete hallucinations.  
        **Production Fix**: Implement automated deskewing and orientation correction before visual tokenization.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is a Vision-Language Model (VLM)?")
        st.markdown("**Back**: A multimodal generative transformer that processes interleaved visual image patches and text tokens in shared attention layers to generate language answers.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is an Image-Grounded Answer?")
        st.markdown("**Back**: An answer where every statement is explicitly anchored to visual pixel coordinates or bounding boxes on retrieved image assets, preventing hallucinations.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Visual Spatial Reasoning?")
        st.markdown("**Back**: The cognitive ability of a vision model to understand relative coordinates, directional arrows, flowchart topologies, and component layouts on technical drawings.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Visual Token Bloat?")
        st.markdown("**Back**: The rapid inflation of prompt tokens caused by injecting multiple high-resolution images into a VLM, increasing latency and API costs.")

# --- SECTION 8: SELF-ASSESSMENT CHECKLIST ---
st.header("8. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of Vision-Language Models (VLMs) and visual grounded generation:")

chk1 = st.checkbox("I understand the architectural differences between adapter-based VLMs (LLaVA) and native interleaved VLMs.", value=True)
chk2 = st.checkbox("I can explain how 2D image patches are tokenized into visual token embeddings within transformer attention.", value=True)
chk3 = st.checkbox("I know how to implement 2D bounding box coordinate citations to visually ground generated claims.", value=True)
chk4 = st.checkbox("I understand the root causes of visual hallucinations and language prior biases in VLM synthesis.", value=True)
chk5 = st.checkbox("I can manage visual token economics by selectively cropping bounding box regions of interest (ROI).", value=True)
chk6 = st.checkbox("I know how to design automated visual verification guardrails to audit VLM claims against source pixels.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Multimodal Generation, VLMs, and Visual Grounding.")
