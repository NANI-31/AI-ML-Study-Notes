import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="30.1 Multimodal Models - LangChain Systems",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for multimodal telemetry
st.markdown("""
<style>
    .mm-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-vision {
        background-color: #0284C7;
        color: #F0F9FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-audio {
        background-color: #7C3AED;
        color: #F5F3FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-video {
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
class MultimodalModelSimulator:
    """
    Simulates multimodal tokenization, cross-attention projection,
    and inference across text, vision, audio, and video modalities.
    """
    def __init__(self):
        self.preset_tasks = {
            "System Architecture Diagram (Vision)": {
                "modality": "Vision (Image)",
                "input_spec": "1024x768 PNG Diagram, high-detail mode",
                "tile_count": 4,
                "token_cost": 765,
                "latency_ms": 680,
                "extracted_entities": ["Kubernetes Cluster", "Vector Store (Chroma)", "Redis Cache", "API Gateway"],
                "reasoning_output": "The architecture displays a microservice gateway load-balancing traffic across dual LangGraph agent runtimes backed by Redis checkpointers and Chroma vector storage."
            },
            "Executive Earnings Call (Audio)": {
                "modality": "Audio (Speech)",
                "input_spec": "45-second 16kHz WAV audio segment",
                "tile_count": 1,
                "token_cost": 1125,
                "latency_ms": 940,
                "extracted_entities": ["Q3 Net Margin: +14.2%", "Capex Guidance: $4.2B", "Speaker: CFO Elena Rostova"],
                "reasoning_output": "The speaker confirms that enterprise recurring revenue grew 24% year-over-year, while cloud infrastructure costs declined due to local token caching."
            },
            "Assembly Line Defect Audit (Video)": {
                "modality": "Video (Temporal Frames)",
                "input_spec": "10-second MP4 at 1 FPS (10 sampled keyframes)",
                "tile_count": 10,
                "token_cost": 2550,
                "latency_ms": 1420,
                "extracted_entities": ["Timestamp 00:04 Micro-crack detected", "Conveyor belt velocity nominal", "Defect class: C3-Fatigue"],
                "reasoning_output": "At keyframe 4 (timestamp 00:04), a 2.4mm structural hairline crack is detected on the ceramic chassis casing before the robotic arm packaging station."
            }
        }

    def simulate_inference(self, task_key: str) -> Dict[str, Any]:
        return self.preset_tasks.get(task_key, self.preset_tasks["System Architecture Diagram (Vision)"])


# --- LESSON HEADER ---
st.title("30.1 Multimodal Models: Vision, Audio & Video Integration")
st.markdown("""
Modern Generative AI architectures have evolved beyond uni-modal text processing into native multimodal systems.
In LangChain, modern chat models seamlessly ingest interleaved text, images (base64 or remote URLs), audio recordings,
and video keyframes within unified conversation messages, projecting heterogeneous sensory inputs into shared semantic vector spaces.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Unified Multimodal Foundation Pipeline", divider="rainbow")
st.markdown("""
Multimodal models project disparate sensory signals into a shared token embedding space using specialized modality encoders:
1. **Vision Projectors (ViT / MLP)**: Split 2D images into patches or tiles, projecting visual features into text-compatible token vectors.
2. **Audio Processors (Whisper / Conformer)**: Convert spectrogram acoustic features into sequential speech tokens.
3. **Temporal Frame Samplers**: Decompose video streams into uniform or scene-detected frame arrays for sequential cross-attention.
4. **Unified Transformer Backbone**: Cross-attends across all modality tokens to output structured textual or function-calling responses.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Multimodal_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_inputs {
            label = "Heterogeneous Input Modalities";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            in_text [label="User Text Prompt\\n'Explain this defect and audio notes'", fillcolor="#0F172A"];
            in_img  [label="Visual Asset\\nPNG / JPEG / WebP / PDF page", fillcolor="#0369A1"];
            in_aud  [label="Acoustic Signal\\nWAV / MP3 Voice Note", fillcolor="#6D28D9"];
            in_vid  [label="Video Clip\\nMP4 Keyframe Stream", fillcolor="#B45309"];
        }

        subgraph cluster_encoders {
            label = "Modality Encoders and Feature Projectors";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            enc_text [label="Text BPE Tokenizer\\nVocab Token IDs", fillcolor="#0F172A"];
            enc_img  [label="Vision Transformer (ViT)\\nPatch / Tile Grid Projector", fillcolor="#0284C7"];
            enc_aud  [label="Audio Spectrogram Encoder\\nConformer / Whisper Acoustic Tower", fillcolor="#7C3AED"];
            enc_vid  [label="Temporal Sampler + Frame ViT\\nTimestamped Token Sequence", fillcolor="#D97706"];
        }

        subgraph cluster_llm {
            label = "Unified Transformer Decoder & Agent Runtime";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            shared_embed [label="Shared Latent Vector Sequence\\n[Text, Vision, Audio, Video Tokens]", fillcolor="#334155"];
            vlm_core     [label="Multimodal Foundation Model\\nGPT-4o / Claude 3.5 Sonnet / Gemini 1.5 Pro", fillcolor="#047857"];
            router_tools [label="Tool Calling & Structured Output\\nJSON Schema / Bind Tools", fillcolor="#059669"];
        }

        in_text -> enc_text;
        in_img  -> enc_img;
        in_aud  -> enc_aud;
        in_vid  -> enc_vid;

        enc_text -> shared_embed [label="Text Tokens"];
        enc_img  -> shared_embed [label="Visual Patches"];
        enc_aud  -> shared_embed [label="Acoustic Tokens"];
        enc_vid  -> shared_embed [label="Temporal Tokens"];

        shared_embed -> vlm_core [label="Cross-Attention"];
        vlm_core     -> router_tools [label="Generate Response / Action"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)


# --- SECTION 2: TEXT MODELS & MULTIMODAL MODALITY ALIGNMENT ---
st.header("2. Text Models and Multimodal Modality Alignment", divider="rainbow")
st.markdown("""
Traditional Large Language Models operate strictly over discrete vocabulary tokens created by Byte-Pair Encoding (BPE)
or WordPiece tokenizers. In native multimodal models, non-text inputs are mapped into continuous latent vectors matching
the hidden dimension of the transformer decoder.
""")

col_mod1, col_mod2 = st.columns(2)
with col_mod1:
    st.subheader("Modality Alignment Strategies")
    st.markdown("""
    - **Cross-Attention Fusion**: Dedicated visual and acoustic encoders preserve their independent representations; cross-attention layers in the decoder attend directly to encoder hidden states.
    - **Linear / MLP Projection (LLaVA approach)**: A pre-trained Vision Transformer (ViT) produces image patch embeddings that pass through a multi-layer perceptron projection layer, transforming them directly into text-token embeddings.
    - **Native Early-Fusion (GPT-4o, Gemini)**: Text, pixel patches, and spectrogram frames are tokenized at the earliest layer and jointly trained across billions of interleaved tokens.
    """)

with col_mod2:
    st.subheader("LangChain HumanMessage Content Schema")
    st.markdown("""
    LangChain models multimodal inputs using a list of typed dictionaries inside `HumanMessage(content=[...])`:
    """)
    st.code("""
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this architectural blueprint:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,...",
                "detail": "high"  # 'low', 'high', or 'auto'
            }
        }
    ]
)
    """, language="python")

st.info("💡 **Production Insight**: Setting `detail: 'low'` downsamples images to a single 512x512 tile costing 85 tokens in GPT-4o, whereas `detail: 'high'` tiles the image into 512px segments at 170 tokens per tile plus 85 base tokens.")


# --- SECTION 3: VISION MODELS ---
st.header("3. Vision Models and Multi-Image Reasoning", divider="rainbow")
st.markdown("""
Vision-Language Models (VLMs) understand both photographic images and technical diagrams.
Modern frontier models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) support multi-image conversations, enabling
comparative visual inspection, change detection, and temporal before/after analysis.
""")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("### Frontier VLM Benchmarks (2026)")
    vlm_data = {
        "Model": ["Claude 3.5 Sonnet", "GPT-4o", "Gemini 1.5 Pro", "Llama 3.2 Vision 90B"],
        "Native Input": ["Images / Documents", "Images / Audio / Text", "Images / Audio / Video", "Images / Text"],
        "Max Resolution": ["Unlimited (Tiled)", "Unlimited (Tiled)", "Unlimited (Native)", "1120x1120"],
        "Context Window": ["200,000 tokens", "128,000 tokens", "2,000,000 tokens", "128,000 tokens"],
        "Cost per 1k Tiles": ["$1.60", "$2.55", "$1.30", "Self-Hosted / Open"]
    }
    st.dataframe(pd.DataFrame(vlm_data), use_container_width=True, hide_index=True)

with col_v2:
    st.markdown("### Comparative Multi-Image Code Pattern")
    st.code("""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

multi_img_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Compare Image 1 (Baseline) and Image 2 (Current Production). Report visible anomalies."},
        {"type": "image_url", "image_url": {"url": "https://cdn.corp/pcb_baseline.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://cdn.corp/pcb_current.jpg"}}
    ]
)

response = llm.invoke([multi_img_msg])
print(response.content)
    """, language="python")


# --- SECTION 4: IMAGE UNDERSTANDING & SPATIAL VISUAL GROUNDING ---
st.header("4. Image Understanding and Spatial Visual Grounding", divider="rainbow")
st.markdown("""
Beyond holistic captioning, enterprise applications require **spatial visual grounding**: extracting coordinates,
bounding boxes, and relationships between visual elements.
""")

col_sg1, col_sg2 = st.columns(2)
with col_sg1:
    st.markdown("""
    ### Spatial Grounding Primitives
    - **Normalized Bounding Boxes**: Models return coordinates in `[ymin, xmin, ymax, xmax]` normalized to `[0, 1000]` or `[0, 1]`.
    - **Visual OCR Extraction**: Extracting text directly from complex layouts, tabular grids, and illegible hand-drawn whiteboards.
    - **Visual Reasoning**: Interpreting arrows, flowchart branches, and chemical molecular structures without pre-vectorized diagrams.
    """)

with col_sg2:
    st.markdown("### Structured Grounding Schema (Pydantic)")
    st.code("""
from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    label: str = Field(description="Name of detected object or component")
    ymin: int = Field(ge=0, le=1000, description="Top boundary coordinate")
    xmin: int = Field(ge=0, le=1000, description="Left boundary coordinate")
    ymax: int = Field(ge=0, le=1000, description="Bottom boundary coordinate")
    xmax: int = Field(ge=0, le=1000, description="Right boundary coordinate")

class VisualInspectionReport(BaseModel):
    defects_found: bool
    confidence_score: float
    components: List[BoundingBox]

structured_vlm = llm.with_structured_output(VisualInspectionReport)
    """, language="python")


# --- SECTION 5: AUDIO MODELS & SPEECH PIPELINES ---
st.header("5. Audio Models and Speech Pipelines", divider="rainbow")
st.markdown("""
Integrating audio into LangChain applications historically required a two-stage pipeline: transcribing audio with
Whisper, followed by passing text to an LLM. Modern multimodal models support **direct audio ingestion**, capturing
acoustic cues such as pitch, intonation, emotional valence, and background noise.
""")

col_a1, col_a2 = st.columns(2)
with col_a1:
    st.markdown("""
    ### Two-Stage vs Native Audio Ingestion
    
    | Dimension | Two-Stage (Whisper + LLM) | Native Multimodal (Gemini / GPT-4o-Audio) |
    | :--- | :--- | :--- |
    | **Information Loss** | Loses tone, emphasis, accents, cadence | Preserves acoustic nuances and background sounds |
    | **Latency** | 1.5s - 4.0s (STT + LLM round-trips) | 300ms - 800ms end-to-end streaming |
    | **Cost Structure** | Separate Whisper API ($0.006/min) + LLM | Billed as native audio tokens |
    | **Language Switching** | May struggle with mid-sentence code switching | Smooth multi-lingual understanding |
    """)

with col_a2:
    st.markdown("### Native Audio Ingestion in LangChain")
    st.code("""
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Read and encode binary audio
with open("customer_call_clip.wav", "rb") as f:
    audio_bytes = f.read()
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

audio_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Listen to this audio snippet. Transcribe the conversation and classify the speaker's emotional state."},
        {
            "type": "media",
            "mime_type": "audio/wav",
            "data": b64_audio
        }
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
response = llm.invoke([audio_msg])
    """, language="python")


# --- SECTION 6: VIDEO UNDERSTANDING & TEMPORAL FRAME ANALYSIS ---
st.header("6. Video Understanding and Temporal Frame Analysis", divider="rainbow")
st.markdown("""
Videos represent high-dimensional spatiotemporal data. Ingesting full 60 FPS video exhausts token budgets in seconds.
LangChain video applications employ **Temporal Frame Sampling** strategies: extracting 1 frame per second (FPS) or
scene-change keyframes, attaching monotonic timestamp metadata to each visual token.
""")

col_v_code, col_v_diag = st.columns(2)
with col_v_code:
    st.markdown("### Frame Sampling Strategy Code")
    st.code("""
import cv2
import base64
from typing import List, Dict

def extract_video_keyframes(video_path: str, fps_sample_rate: float = 0.5) -> List[Dict[str, Any]]:
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / fps_sample_rate)
    
    frames_payload = []
    frame_idx = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if not success:
            break
        if frame_idx % interval == 0:
            timestamp_sec = round(frame_idx / fps, 2)
            _, buffer = cv2.imencode('.jpg', image)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
                "metadata": {"timestamp": timestamp_sec}
            })
        frame_idx += 1
    vidcap.release()
    return frames_payload
    """, language="python")

with col_v_diag:
    st.markdown("### Token Budget for Video Processing")
    video_budget_df = pd.DataFrame({
        "Duration": ["10 seconds", "30 seconds", "1 minute", "5 minutes"],
        "Sample Rate": ["1 FPS", "1 FPS", "0.5 FPS", "0.2 FPS"],
        "Total Frames": [10, 30, 30, 60],
        "Token Cost (Low Detail)": [850, 2550, 2550, 5100],
        "Token Cost (High Detail)": [7650, 22950, 22950, 45900]
    })
    st.dataframe(video_budget_df, use_container_width=True, hide_index=True)
    st.caption("Calculation based on 85 tokens/frame (low detail) vs 765 tokens/frame average (high detail).")


# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Multimodal Model Orchestrator", divider="rainbow")
st.markdown("""
A robust production multimodal handler encapsulates token estimation, payload validation, image compression,
and provider-specific format transformations into a reusable abstraction.
""")

st.code("""
import base64
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class EnterpriseMultimodalDispatcher:
    def __init__(self, primary_provider: str = "openai", openai_model: str = "gpt-4o", gemini_model: str = "gemini-1.5-pro"):
        self.provider = primary_provider
        self.openai_llm = ChatOpenAI(model=openai_model, temperature=0.1)
        self.gemini_llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.1)
        
    def estimate_vision_tokens(self, width: int, height: int, detail: str = "high") -> int:
        if detail == "low":
            return 85
        # High detail: scale to fit 2048x2048, then 768 on shortest side
        tiles_w = (width + 511) // 512
        tiles_h = (height + 511) // 512
        total_tiles = tiles_w * tiles_h
        return 85 + (170 * total_tiles)

    def build_multimodal_message(
        self,
        prompt_text: str,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        image_mime: str = "image/jpeg",
        audio_mime: str = "audio/wav",
        detail: str = "high"
    ) -> HumanMessage:
        content_payload: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        
        if image_bytes:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{b64_img}",
                    "detail": detail
                }
            })
            
        if audio_bytes:
            b64_aud = base64.b64encode(audio_bytes).decode("utf-8")
            if self.provider == "gemini":
                content_payload.append({
                    "type": "media",
                    "mime_type": audio_mime,
                    "data": b64_aud
                })
            else:
                # Fallback for OpenAI input audio format
                content_payload.append({
                    "type": "input_audio",
                    "input_audio": {"data": b64_aud, "format": "wav"}
                })
                
        return HumanMessage(content=content_payload)

    def invoke(self, message: HumanMessage) -> str:
        if self.provider == "openai":
            response = self.openai_llm.invoke([message])
        else:
            response = self.gemini_llm.invoke([message])
        return str(response.content)
""", language="python")


# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Vision & Audio Multimodal Input Simulator", divider="rainbow")
st.markdown("""
Experiment with multimodal inference across Vision, Audio, and Video modalities.
Inspect real-time token footprint estimations, cross-attention processing latency, and structured reasoning outputs.
""")

simulator = MultimodalModelSimulator()

col_cfg, col_res = st.columns([1, 1])

with col_cfg:
    st.subheader("Modality Selection & Parameters")
    selected_task = st.selectbox(
        "Select Multimodal Inspection Scenario:",
        list(simulator.preset_tasks.keys()),
        index=0
    )
    
    detail_level = st.radio("Vision Detail Mode:", ["high", "low"], index=0, horizontal=True)
    sample_rate = st.slider("Video / Audio Sample Frequency:", min_value=0.2, max_value=2.0, value=1.0, step=0.2)
    
    execute_btn = st.button("🚀 Execute Multimodal Inference", use_container_width=True)

with col_res:
    st.subheader("Multimodal Execution Telemetry")
    if execute_btn:
        result = simulator.simulate_inference(selected_task)
        
        # Adjust token cost if detail is low
        tokens = result["token_cost"] if detail_level == "high" else int(result["token_cost"] * 0.3)
        
        st.markdown(
            f"<div class='mm-card'>"
            f"<b>Modality:</b> <span class='badge-vision'>{result['modality']}</span><br><br>"
            f"<b>Input Specification:</b> <code>{result['input_spec']}</code><br>"
            f"<b>Tile / Frame Count:</b> {result['tile_count']}<br>"
            f"<b>Token Consumption:</b> ~{tokens} tokens<br>"
            f"<b>Inference Latency:</b> {result['latency_ms']} ms"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.caption("Extracted Semantic Entities:")
        for entity in result["extracted_entities"]:
            st.markdown(f"- 📌 **{entity}**")
            
        st.success(f"Reasoning Synthesis: {result['reasoning_output']}")
    else:
        st.info("Select a multimodal inspection scenario and execute inference to observe modality tokens and output reasoning.")


# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How do vision models calculate token consumption for high-resolution images?"):
    st.markdown("""
    **Answer**: In models like GPT-4o, an image is first scaled to fit within a 2048x2048 bounding box. Next, the shortest side is scaled to 768px.
    The resulting image is partitioned into 512x512 pixel tiles. Each 512px tile costs 170 tokens, added to a fixed base cost of 85 tokens.
    For example, a 1024x1024 image splits into 4 tiles: `(4 * 170) + 85 = 765 tokens`.
    """)

with st.expander("Q2: What is the risk of using base64 data URIs versus remote presigned URLs in LangChain multimodal messages?"):
    st.markdown("""
    **Answer**: Base64 encoding inflates payload file size by 33%. Transmitting large images or audio clips directly in HTTP payloads increases network latency
    and risks hitting client-side request body limits. Presigned cloud storage URLs (S3, GCS) minimize client latency, letting the LLM provider fetch the asset directly.
    """)

with st.expander("Q3: How does native multimodal audio processing differ from Whisper-transcribed pipelines?"):
    st.markdown("""
    **Answer**: Whisper pipelines transcribe speech to text, stripping vocal tone, pitch, volume, background ambient noise, and emotion.
    Native multimodal audio models process acoustic spectrogram tokens directly, allowing the model to detect hesitation, sarcasm, urgency,
    and acoustic background anomalies that pure text strips away.
    """)

with st.expander("Q4: Why should video files be sampled into discrete keyframes instead of streamed continuously to LLMs?"):
    st.markdown("""
    **Answer**: Full 30 FPS video generates 1,800 frames per minute, consuming hundreds of thousands of tokens and crashing context windows within seconds.
    Sampling at 0.5 - 1.0 FPS or using scene-detection algorithms captures essential state transitions while keeping token usage within manageable enterprise budgets.
    """)

with st.expander("Q5: What is 'spatial grounding' and how does structured output enable it in LangChain?"):
    st.markdown("""
    **Answer**: Spatial grounding is the capability of a vision model to associate semantic concepts with exact 2D pixel coordinates.
    By binding a Pydantic schema with coordinate constraints (`ymin`, `xmin`, `ymax`, `xmax`), LangChain forces the model to return verified bounding box coordinates
    for UI navigation, robotic manipulation, or visual defect inspection.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Unbounded Image Resolution Ingestion)**: Sending raw 40-megapixel camera images to VLMs, causing extreme latency spikes and runaway token billing without resolution caps.")
    st.warning("**Trap 2 (Detail Mode Misconfiguration)**: Using `detail: 'high'` on low-information images (icons, simple logos), wasting 700+ tokens per image when `detail: 'low'` (85 tokens) is sufficient.")
with col_trap2:
    st.error("**Trap 3 (Sending Unsupported Media Mime Types)**: Passing incompatible container formats (e.g. MKV or raw PCM) without conversion, triggering hard 400 Bad Request API rejections.")
    st.warning("**Trap 4 (Omitting Modality Timestamps)**: Submitting video frames without explicit timestamp annotations, preventing the model from performing temporal sequencing or chronometry.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the base token cost for low-detail images in GPT-4o?")
        st.caption("Flip for answer:")
        st.info("**Back**: Exactly 85 tokens (scaled down to a 512x512 representation).")
    with st.container(border=True):
        st.markdown("**Front**: What message format in LangChain supports multimodal content?")
        st.caption("Flip for answer:")
        st.info("**Back**: `HumanMessage(content=[{'type': 'text', ...}, {'type': 'image_url', ...}])`.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the recommended frame sampling rate for long video analysis?")
        st.caption("Flip for answer:")
        st.info("**Back**: 0.2 to 1.0 frames per second (FPS), or dynamic scene-change detection.")
    with st.container(border=True):
        st.markdown("**Front**: How are bounding box coordinates normalized in modern vision models?")
        st.caption("Flip for answer:")
        st.info("**Back**: `[ymin, xmin, ymax, xmax]` normalized to `[0, 1000]` or `[0.0, 1.0]`.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural differences between cross-attention encoders and native early-fusion multimodal models.", value=True)
c2 = st.checkbox("I can structure LangChain HumanMessage payloads with interleaved text, base64 images, and media URLs.", value=True)
c3 = st.checkbox("I know how image tiling works and how to optimize token costs using 'low' vs 'high' detail modes.", value=True)
c4 = st.checkbox("I understand spatial grounding and how to extract bounding box coordinates via Pydantic schemas.", value=True)
c5 = st.checkbox("I know how to sample video keyframes with timestamps to avoid context window exhaustion.", value=True)
c6 = st.checkbox("I can implement an enterprise multimodal dispatcher supporting both OpenAI and Gemini vision models.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 30.2: Image Applications.")
