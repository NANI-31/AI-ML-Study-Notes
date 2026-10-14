import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="21.1 Multimodal Fundamentals",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class MultimodalFundamentalsEngine:
    """Enterprise Multimodal Embedding & Retrieval Engine projecting text and visual assets into a shared vector space."""

    def __init__(self):
        # Multimodal knowledge store containing text documents and visual assets
        self.assets = [
            {
                "id": "img_arch_01",
                "type": "Image (Architecture Diagram)",
                "title": "H100 GPU Cluster Interconnect Topology",
                "caption": "High-speed NVLink and InfiniBand quantum-2 network switch fabric connecting 8x H100 SXM5 accelerators with 3.2 Tbps bisectional bandwidth.",
                "vector_tags": ["gpu", "interconnect", "nvlink", "hardware", "switch", "cluster"]
            },
            {
                "id": "txt_doc_01",
                "type": "Text Document",
                "title": "GPU Server Power & Thermal Specifications",
                "caption": "Each 8-way GPU chassis requires 10.2 kW of power delivery under maximum tensor core workload, supported by rear-door heat exchangers.",
                "vector_tags": ["power", "thermal", "cooling", "hardware", "datacenter"]
            },
            {
                "id": "img_chart_02",
                "type": "Image (Benchmark Chart)",
                "title": "Inference Latency vs Batch Size Benchmark",
                "caption": "Line chart comparing FP8 vs FP16 throughput on Llama-3-70B; shows 2.4x latency reduction at batch size 32 using FP8 quantization.",
                "vector_tags": ["benchmark", "latency", "throughput", "fp8", "batch", "quantization"]
            },
            {
                "id": "img_diagram_03",
                "type": "Image (System Diagram)",
                "title": "Distributed Attention Ring Pipeline",
                "caption": "Circular schematic illustrating RingAttention key-value tensor communication across 16 nodes for 1M context window scaling.",
                "vector_tags": ["attention", "ring", "context", "distributed", "tensor"]
            },
            {
                "id": "txt_doc_02",
                "type": "Text Document",
                "title": "Context Window Scaling Principles",
                "caption": "RingAttention divides sequence tokens across GPU workers in a logical ring, overlapping communication with attention block computation.",
                "vector_tags": ["attention", "ring", "context", "scaling", "sequence"]
            }
        ]

    def _project_pseudo_vector(self, terms: List[str]) -> np.ndarray:
        """Deterministic 32-dimensional mock embedding projection."""
        vec = np.zeros(32, dtype=float)
        for term in terms:
            for char in term.lower():
                idx = ord(char) % 32
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search_multimodal(self, query: str, modality_filter: str = "All Modalities", top_k: int = 3) -> List[Dict[str, Any]]:
        """Cross-modal retrieval: embeds text query and scores both text chunks and visual image embeddings."""
        q_tokens = query.lower().split()
        q_vec = self._project_pseudo_vector(q_tokens)
        
        candidates = []
        for asset in self.assets:
            if modality_filter != "All Modalities":
                if modality_filter == "Images Only" and "Image" not in asset["type"]:
                    continue
                if modality_filter == "Text Only" and "Text" not in asset["type"]:
                    continue
                    
            asset_vec = self._project_pseudo_vector(asset["vector_tags"] + asset["title"].lower().split())
            sim = float(np.dot(q_vec, asset_vec))
            
            candidates.append({
                "id": asset["id"],
                "type": asset["type"],
                "title": asset["title"],
                "caption": asset["caption"],
                "similarity_score": round(sim, 4)
            })
            
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


# --- Streamlit Layout ---

st.title("21.1 Multimodal Fundamentals in RAG")
st.caption("Beyond Text: Shared Latent Spaces, Image Embeddings, Vision Transformers & Cross-Modal Retrieval Pipelines")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Unified Multimodal RAG Architecture (Cross-Modal Vector Spaces)", divider="rainbow")

st.markdown(
    """
    Text-only RAG operates under a fundamental blind spot: over **50% of high-value enterprise knowledge** is locked inside visual assets:
    architectural schematics, circuit diagrams, benchmark charts, financial infographics, and UI screenshots.
    Converting an intricate diagram into a crude text caption discards spatial relationships, exact coordinates, and visual structure.
    
    **Multimodal RAG** projects both textual tokens and high-resolution image patches into a **unified shared vector space** (via CLIP, SigLIP, or ColPali).
    A natural language user query directly retrieves visual images, diagrams, and text chunks within a single unified nearest-neighbor search.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Multimodal_Fundamentals {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_dual_ingestion {
            label = "1. MULTIMODAL INGESTION & FEATURE EXTRACTION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            text_corpus [label="Text Corpus (PDFs, Markdown, Specs)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            image_corpus [label="Visual Corpus (Schematics, Charts, Screenshots)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            
            text_encoder [label="Text Transformer Encoder\\n(CLIP Text / ColPali Text)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            vision_encoder [label="Vision Transformer (ViT)\\n(Patch Projection: 16x16 tokens)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            text_corpus -> text_encoder;
            image_corpus -> vision_encoder;
        }

        subgraph cluster_shared_latent {
            label = "2. UNIFIED SHARED VECTOR SPACE (CROSS-MODAL PROJECTION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            latent_space [label="Unified Latent Vector Space (d=1024)\\nCos(v_text, v_image) aligns semantic meaning\\n(Text query matches visual diagram)", shape=cylinder, fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            text_encoder -> latent_space [label="Project Text Embeddings"];
            vision_encoder -> latent_space [label="Project Visual Embeddings"];
        }

        subgraph cluster_runtime_retrieval {
            label = "3. RUNTIME CROSS-MODAL RETRIEVAL";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="User Query: 'How does NVLink connect 8 GPUs?'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_vector [label="Query Embedding (Text Encoder)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            ann_search [label="Approximate Nearest Neighbors Search\\n(Returns BOTH Image Diagrams & Text Chunks)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            user_query -> query_vector;
            latent_space -> ann_search [label="Index Probe"];
            query_vector -> ann_search;
        }

        subgraph cluster_vlm_generation {
            label = "4. VISION-LANGUAGE SYNTHESIS (VLM)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            vlm_generator [label="Frontier Vision-Language Model (VLM)\\n(GPT-4o / Gemini 1.5 Pro / Claude 3.5 Sonnet)\\nInspects Raw Image Pixels + Text Evidence", fillcolor="#1E3A8A", fontcolor="#10B981"];
            multimodal_answer [label="Image-Grounded Synthesized Answer\\nWith Visual Bounding Box Citations", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            ann_search -> vlm_generator [label="Inject Image + Text Chunks"];
            vlm_generator -> multimodal_answer;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: WHAT IS MULTIMODAL RAG? ---
st.header("2. What is Multimodal RAG? (Beyond Text-Only RAG, Visual Knowledge Representation & System Paradigms)", divider="rainbow")

st.markdown(
    """
    Traditional text-based RAG assumes that all human knowledge can be faithfully serialized into ASCII or UTF-8 strings.
    In technical and scientific domains, this assumption completely breaks down:
    
    - **Information Loss in Tabular & Graphic Data**: An intricate circuit diagram or architecture map contains hundreds of directional relationships. A human or OCR description like *"A diagram showing server interconnects"* discards 95% of the information.
    - **Visual Spatial Reasoning**: Queries like *"Which switch connects Node 3 to the storage fabric?"* require spatial coordinate understanding, not merely keyword presence.
    
    ### The Three Paradigms of Multimodal RAG:
    """
)

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    with st.container(border=True):
        st.markdown("#### Paradigm 1: Text-to-Text with Image Captioning")
        st.markdown(
            """
            - **Ingestion**: Every image is captioned using an offline VLM (e.g., LLaVA).
            - **Retrieval**: Standard text vector search retrieves the text caption.
            - **Limitation**: Highly lossy; the captioner cannot anticipate every detail a user might ask about.
            """
        )
with col_p2:
    with st.container(border=True):
        st.markdown("#### Paradigm 2: Shared Cross-Modal Embeddings (CLIP)")
        st.markdown(
            """
            - **Ingestion**: Images and text are mapped into a single geometric vector space via dual encoders (CLIP/SigLIP).
            - **Retrieval**: Text queries directly retrieve raw image vectors via cosine similarity.
            - **Advantage**: Fast retrieval without captioning latency.
            """
        )
with col_p3:
    with st.container(border=True):
        st.markdown("#### Paradigm 3: ColPali (Late-Interaction Multi-Vector)")
        st.markdown(
            """
            - **Ingestion**: High-res document page images are tokenized directly into patch vectors via Vision LLMs (PaliGemma).
            - **Retrieval**: Text query tokens match visual patch vectors via late interaction (MaxSim).
            - **Advantage**: Completely eliminates OCR and parsing errors.
            """
        )

# --- SECTION 3: TEXT + IMAGE RETRIEVAL ---
st.header("3. Text + Image Retrieval (Dual-Encoder Projections, Joint Scoring & Modality Fusion)", divider="rainbow")

st.markdown(
    """
    In a true multimodal retrieval system, the index contains heterogeneous objects: text paragraphs and visual image assets.
    When a user submits a natural language query, the retriever computes similarity across all objects concurrently.
    
    ### Joint Modality Fusion Scoring
    When ranking candidates, systems apply **Modality Calibration Weights**:
    
    `Final_Score = alpha * Cosine(q_text, doc_vec) + (1 - alpha) * CrossModal_Prior`
    
    - If the user query contains visual intent words (*"diagram"*, *"schematic"*, *"chart"*, *"screenshot"*), the router dynamically boosts the weight of image assets over plain text passages.
    """
)

# --- SECTION 4: IMAGE EMBEDDINGS ---
st.header("4. Image Embeddings (Vision Transformers, CLIP ViT-L/14, OpenCLIP & Dense Visual Feature Maps)", divider="rainbow")

st.markdown(
    """
    How does a computer transform raw RGB pixels into a semantic vector?
    Modern multimodal RAG relies on **Vision Transformers (ViT)**:
    
    1. **Patch Extraction**: A 224 x 224 image is sliced into a grid of non-overlapping 16 x 16 pixel patches (14 x 14 = 196 total patches).
    2. **Linear Projection & Position Embeddings**: Each patch is flattened into a 1D vector and added to a 1D learnable position embedding to preserve 2D spatial coordinates.
    3. **Transformer Self-Attention Layers**: Self-attention heads compute cross-patch correlations, allowing the model to recognize global shapes, edges, and text within the image.
    4. **CLS Token Output**: The final `[CLS]` token output serves as the dense semantic image embedding vector (v_image in R^768 or R^1024).
    """
)

# --- SECTION 5: MULTIMODAL EMBEDDINGS ---
st.header("5. Multimodal Embeddings (Unified Shared Vector Spaces: Nomic Embed Vision, ImageBind & ColPali Multi-Vector)", divider="rainbow")

st.markdown(
    """
    The ultimate milestone in multimodal representation is the **Unified Multimodal Vector Space**:
    
    - **ImageBind (Meta AI)**: Binds 6 modalities (Text, Image/Video, Audio, Depth, Thermal, IMU) into a single joint vector space, allowing an audio recording of engine noise to directly retrieve an automotive engine schematic.
    - **Nomic Embed Vision**: A fully open-source dense multimodal embedding model aligned directly with text embeddings, enabling sub-millisecond cross-modal vector search.
    - **ColPali (Hugging Face / Illuin)**: Rather than pooling an entire image into a single 1D vector, ColPali produces an array of 1,024 multi-vectors representing visual tokens, enabling fine-grained keyword matching directly on image text patches.
    """
)

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Unified Multimodal Embedding & Retrieval Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `MultimodalFundamentalsEngine`.
    It implements deterministic cross-modal vector projection, heterogeneous asset indexing (text documents and technical diagrams),
    modality-specific filtering, and joint cosine similarity ranking.
    """
)

st.code(
    '''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class MultimodalFundamentalsEngine:
    """Enterprise Multimodal Embedding & Retrieval Engine projecting text and visual assets into a shared vector space."""

    def __init__(self):
        self.assets = [
            {
                "id": "img_arch_01",
                "type": "Image (Architecture Diagram)",
                "title": "H100 GPU Cluster Interconnect Topology",
                "caption": "High-speed NVLink and InfiniBand quantum-2 network switch fabric connecting 8x H100 SXM5 accelerators with 3.2 Tbps bisectional bandwidth.",
                "vector_tags": ["gpu", "interconnect", "nvlink", "hardware", "switch", "cluster"]
            },
            {
                "id": "txt_doc_01",
                "type": "Text Document",
                "title": "GPU Server Power & Thermal Specifications",
                "caption": "Each 8-way GPU chassis requires 10.2 kW of power delivery under maximum tensor core workload, supported by rear-door heat exchangers.",
                "vector_tags": ["power", "thermal", "cooling", "hardware", "datacenter"]
            },
            {
                "id": "img_chart_02",
                "type": "Image (Benchmark Chart)",
                "title": "Inference Latency vs Batch Size Benchmark",
                "caption": "Line chart comparing FP8 vs FP16 throughput on Llama-3-70B; shows 2.4x latency reduction at batch size 32 using FP8 quantization.",
                "vector_tags": ["benchmark", "latency", "throughput", "fp8", "batch", "quantization"]
            },
            {
                "id": "img_diagram_03",
                "type": "Image (System Diagram)",
                "title": "Distributed Attention Ring Pipeline",
                "caption": "Circular schematic illustrating RingAttention key-value tensor communication across 16 nodes for 1M context window scaling.",
                "vector_tags": ["attention", "ring", "context", "distributed", "tensor"]
            },
            {
                "id": "txt_doc_02",
                "type": "Text Document",
                "title": "Context Window Scaling Principles",
                "caption": "RingAttention divides sequence tokens across GPU workers in a logical ring, overlapping communication with attention block computation.",
                "vector_tags": ["attention", "ring", "context", "scaling", "sequence"]
            }
        ]

    def _project_pseudo_vector(self, terms: List[str]) -> np.ndarray:
        """Deterministic 32-dimensional mock embedding projection."""
        vec = np.zeros(32, dtype=float)
        for term in terms:
            for char in term.lower():
                idx = ord(char) % 32
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search_multimodal(self, query: str, modality_filter: str = "All Modalities", top_k: int = 3) -> List[Dict[str, Any]]:
        """Cross-modal retrieval: embeds text query and scores both text chunks and visual image embeddings."""
        q_tokens = query.lower().split()
        q_vec = self._project_pseudo_vector(q_tokens)
        
        candidates = []
        for asset in self.assets:
            if modality_filter != "All Modalities":
                if modality_filter == "Images Only" and "Image" not in asset["type"]:
                    continue
                if modality_filter == "Text Only" and "Text" not in asset["type"]:
                    continue
                    
            asset_vec = self._project_pseudo_vector(asset["vector_tags"] + asset["title"].lower().split())
            sim = float(np.dot(q_vec, asset_vec))
            
            candidates.append({
                "id": asset["id"],
                "type": asset["type"],
                "title": asset["title"],
                "caption": asset["caption"],
                "similarity_score": round(sim, 4)
            })
            
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]
''',
    language="python"
)

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: Cross-Modal Similarity Explorer & Joint Embedding Projector", divider="rainbow")

st.markdown(
    """
    Test how natural language text queries retrieve technical diagrams, benchmark charts, and text documents
    within a shared cross-modal vector index. Adjust modality filters and inspect candidate similarity rankings.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔍 Query & Modality Controls")
    
    preset_query = st.selectbox(
        "Select Multimodal Query Preset:",
        [
            "Preset 1 (Visual Diagram): Show me the NVLink hardware interconnect diagram for GPU clusters",
            "Preset 2 (Visual Chart): What are the FP8 benchmark latency curves at batch size 32?",
            "Preset 3 (Text + Image): How does RingAttention distribute sequence tensors across nodes?",
            "Preset 4 (Custom Query): Enter your own query below"
        ],
        index=0
    )
    
    if "Custom" in preset_query:
        active_query = st.text_input("Active Query:", "Datacenter cooling specifications and thermal heat exchangers")
    else:
        active_query = preset_query.split(": ", 1)[1]
        st.text_input("Active Query:", active_query, disabled=True)

    modality_filter = st.radio(
        "Modality Scope Filter:",
        ["All Modalities", "Images Only", "Text Only"],
        horizontal=True
    )
    
    top_k_val = st.slider("Retrieval Candidate Top-K:", min_value=1, max_value=5, value=3, step=1)

with wb_col2:
    st.subheader("⚡ Cross-Modal Retrieval Results")
    
    engine = MultimodalFundamentalsEngine()
    results = engine.search_multimodal(active_query, modality_filter=modality_filter, top_k=top_k_val)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Retrieved Assets", len(results))
    top_score = results[0]["similarity_score"] if results else 0.0
    m2.metric("Top-1 Cosine Match", f"{top_score:.4f}")
    img_count = sum(1 for r in results if "Image" in r["type"])
    m3.metric("Visual Assets Found", img_count)
    
    for r in results:
        with st.container(border=True):
            st.markdown(f"**[{r['type']}] {r['title']}** (Cosine Score: `{r['similarity_score']}`)")
            st.caption(r["caption"])

st.markdown("### Cross-Modal Candidate Manifest Table")
manifest_df = pd.DataFrame(results)
st.dataframe(manifest_df, use_container_width=True)

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why does naive OCR text extraction fail for complex technical infographics and charts?**"):
    st.markdown(
        """
        OCR extracts characters as a flat sequence of words. In a complex chart or schematic:
        1. **Spatial Disconnect**: The OCR engine might read an axis label, then a data point, then an unrelated legend label, destroying spatial alignment.
        2. **Loss of Non-Textual Semantics**: Colors, line slopes, arrow directions, and flowchart branching are non-textual and completely lost.
        3. **Scrambled Data**: Multi-column tables or grouped bar charts lose their columnar headers, producing nonsensical strings like *'45.2 Q1 12.8 Net'*.
        
        Multimodal RAG solves this by processing images as 2D patch tokens through Vision Transformers (ViT) or ColPali.
        """
    )

with st.expander("**Q2: How does CLIP align text and image embeddings in a shared vector space?**"):
    st.markdown(
        """
        CLIP (Contrastive Language-Image Pretraining) uses **dual transformer encoders**: a Text Transformer and a Vision Transformer (ViT).
        During pre-training on hundreds of millions of (image, text) pairs:
        1. A batch of `N` images produces vectors v_1, ..., v_N.
        2. A batch of `N` paired texts produces vectors u_1, ..., u_N.
        3. A **contrastive symmetric cross-entropy loss** maximizes the cosine similarity of true positive pairs v_i . u_i while minimizing cosine similarity for all `N^2 - N` incorrect pairs.
        
        This aligns visual semantic concepts (*pictures of dogs*) with textual concepts (*'dog'*) into the exact same latent space.
        """
    )

with st.expander("**Q3: What is ColPali and how does it revolutionize multimodal document retrieval?**"):
    st.markdown(
        """
        Traditional multimodal systems compress an entire high-resolution document page into a single 1D vector (e.g., 1024 dimensions), causing severe information loss.
        
        **ColPali** adapts the ColBERT late-interaction paradigm to vision:
        1. A high-resolution PDF page is processed by a Vision-Language Model (PaliGemma), producing an array of patch tokens (~1,024 multi-vectors).
        2. Query text tokens `q_j` interact with all document visual patch vectors `d_i` via **MaxSim Late Interaction**:
        `Score = Sum_j ( max_i ( q_j . d_i ) )`.
        3. Query tokens align directly with the exact visual image patch containing the text or diagram, achieving state-of-the-art document recall with zero OCR pre-processing.
        """
    )

with st.expander("**Q4: What are the storage and memory trade-offs between dense CLIP vectors and ColPali multi-vectors?**"):
    st.markdown(
        """
        - **Dense CLIP Vectors**: 1 vector per image (~1,024 floats = 4 KB). Highly compact; 1 million images require only 4 GB of RAM. Fast approximate nearest neighbor search via HNSW.
        - **ColPali Multi-Vectors**: ~1,024 vectors per image (1,024 x 128 floats = 512 KB). 1 million images require ~500 GB of storage.
        
        While ColPali delivers vastly superior retrieval precision, it requires multi-vector indexing frameworks (like Vespa, Qdrant multi-vector, or binary quantization) to manage index size in enterprise production.
        """
    )

with st.expander("**Q5: How does a Vision-Language Model (VLM) generate answers from retrieved images?**"):
    st.markdown(
        """
        Modern frontier models (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet) are native **Vision-Language Models**:
        1. Retrieved image files (PNG/JPEG) are tokenized by a vision encoder into image patch embeddings.
        2. Image tokens and prompt text tokens are interleaved into the transformer's multi-head self-attention layers.
        3. The model attends directly to image pixels, enabling it to read axis numbers, trace circuit wiring, and synthesize textual answers with exact region-of-interest citations.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Downsampling Images Below Legibility Thresholds**  
        **Antipattern**: Resizing technical schematics and PDF pages to 224 x 224 pixels to fit standard CLIP models.  
        **Consequence**: Text, chart axis numbers, and fine diagram lines become completely illegible, destroying visual retrieval precision.  
        **Production Fix**: Employ dynamic patch models (ColPali or SigLIP 384+) that support high-resolution inputs and tiled multi-crop processing.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Captioning Every Image Synchronously at Ingestion**  
        **Antipattern**: Generating full paragraph captions for every PDF figure using an expensive commercial VLM during ingestion.  
        **Consequence**: Adds massive GPU cost and hours of ingestion latency; captions miss fine details that users ask about later.  
        **Production Fix**: Store raw images in shared cross-modal vector spaces; only invoke generative VLMs at runtime for the top retrieved visual assets.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Ignoring Modality-Specific Query Intent**  
        **Antipattern**: Using fixed candidate top-K without checking if the user explicitly requested a visual diagram or chart.  
        **Consequence**: A user asking *"Show me the architecture diagram"* receives 4 blocks of text and zero visual images.  
        **Production Fix**: Detect visual intent keywords and apply dynamic modality calibration weights to boost visual assets.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Storing Raw Image Blobs Directly Inside Vector Databases**  
        **Antipattern**: Encoding 5MB JPEG binary payloads directly inside vector database payload columns.  
        **Consequence**: Bloats index memory, slows down HNSW graph traversal, and triggers database out-of-memory crashes.  
        **Production Fix**: Store images in cloud object storage (Amazon S3 / GCS) and store only image URLs and bounding box coordinates in the vector database.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Multimodal RAG?")
        st.markdown("**Back**: An architecture that ingests, indexes, and retrieves both text and visual assets (diagrams, charts, screenshots) in a unified shared vector space for VLM synthesis.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is a Vision Transformer (ViT)?")
        st.markdown("**Back**: A neural network that divides an image into a grid of non-overlapping patches (e.g., 16x16 pixels) and processes them as visual tokens using standard transformer self-attention.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is ColPali?")
        st.markdown("**Back**: A late-interaction multimodal model that represents document page images as multi-vector patch embeddings, eliminating OCR and allowing query tokens to match visual patches directly.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is MaxSim Late Interaction?")
        st.markdown("**Back**: An alignment score that computes the maximum cosine similarity between each query token and all document image patches, summing the maximum scores for high-precision retrieval.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of multimodal fundamentals, shared vector spaces, and vision models:")

chk1 = st.checkbox("I understand why text-only RAG fails to capture structural, tabular, and visual schematic knowledge.", value=True)
chk2 = st.checkbox("I can differentiate between Image Captioning RAG, Shared Embedding RAG (CLIP), and ColPali Late-Interaction.", value=True)
chk3 = st.checkbox("I understand how Vision Transformers (ViT) divide raw image pixels into visual patch tokens.", value=True)
chk4 = st.checkbox("I know how dual contrastive encoders (CLIP/SigLIP) align text and image vectors in a shared latent space.", value=True)
chk5 = st.checkbox("I understand the storage and latency trade-offs between single dense vectors and ColPali multi-vector representations.", value=True)
chk6 = st.checkbox("I know how to implement joint cross-modal retrieval and modality calibration weighting in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Multimodal RAG Fundamentals and Shared Latent Spaces.")
