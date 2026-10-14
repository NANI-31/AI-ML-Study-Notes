import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="21.3 Multimodal Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class CrossModalRetrievalEngine:
    """Enterprise Cross-Modal Retrieval Orchestrator supporting Text-to-Image, Image-to-Text, Reverse Image, and MaxSim Late-Interaction."""

    def __init__(self):
        # Multimodal Asset Repository
        self.visual_assets = [
            {"id": "img_k8s_arch", "type": "Image", "title": "Kubernetes Control Plane Topology", "tags": ["kubernetes", "k8s", "control", "plane", "etcd", "apiserver"], "caption": "Detailed diagram of API Server, ETCD cluster, Scheduler, and Controller Manager."},
            {"id": "img_gpu_mesh", "type": "Image", "title": "GPU NVLink Mesh Interconnect", "tags": ["gpu", "nvlink", "hardware", "switch", "cluster"], "caption": "Full-mesh 900 GB/s bidirectional NVLink crossbar connecting 8 accelerators."},
            {"id": "img_auth_flow", "type": "Image", "title": "OAuth2 PKCE Authentication Flow", "tags": ["oauth2", "pkce", "security", "token", "auth"], "caption": "Sequence diagram showing Code Challenge, Auth Code exchange, and Access Token emission."}
        ]
        self.text_assets = [
            {"id": "txt_k8s_spec", "type": "Text", "title": "Kubernetes High Availability Best Practices", "tags": ["kubernetes", "k8s", "control", "plane", "etcd", "ha"], "caption": "ETCD requires a minimum of 3 odd-numbered replicas for quorum. Control plane nodes must span distinct failure domains."},
            {"id": "txt_gpu_spec", "type": "Text", "title": "NVLink Bandwidth & Thermal Envelopes", "tags": ["gpu", "nvlink", "hardware", "cooling"], "caption": "Total power dissipation for NVLink switches requires active liquid cooling loops maintaining cold-plate temperatures below 45C."},
            {"id": "txt_auth_spec", "type": "Text", "title": "Zero-Trust Token Lifetime Governance", "tags": ["oauth2", "security", "token", "auth"], "caption": "Access tokens expire within 15 minutes; refresh tokens require rotation with cryptographic replay detection."}
        ]

    def _score_tokens(self, q_tokens: List[str], target_tags: List[str]) -> float:
        """Computes overlap score simulating normalized cosine similarity in joint space."""
        intersection = len(set(q_tokens).intersection(set(target_tags)))
        return float(intersection / (len(q_tokens) + 1e-5))

    def retrieve(self, query: str, mode: str = "Cross-Modal (Unified)", top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes retrieval across specified modality path."""
        q_tokens = [w.lower() for w in query.split()]
        candidates = []
        
        pool = []
        if mode == "Text-to-Image (Image Search)":
            pool = self.visual_assets
        elif mode == "Image-to-Text (Reverse Lookup)":
            pool = self.text_assets
        elif mode == "Image-to-Image (Visual Similarity)":
            pool = self.visual_assets
        else: # Cross-Modal (Unified)
            pool = self.visual_assets + self.text_assets

        for item in pool:
            score = self._score_tokens(q_tokens, item["tags"] + item["title"].lower().split())
            # Simulating MaxSim alignment boost
            maxsim_score = min(1.0, score + 0.35 if score > 0 else 0.12)
            candidates.append({
                "id": item["id"],
                "type": item["type"],
                "title": item["title"],
                "caption": item["caption"],
                "similarity_score": round(maxsim_score, 4)
            })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


# --- Streamlit Layout ---

st.title("21.3 Multimodal Retrieval in RAG")
st.caption("Bidirectional Cross-Modal Search: Text-to-Image, Image-to-Text, Reverse Image Search & ColPali MaxSim Late Interaction")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Heterogeneous Cross-Modal Retrieval & ColPali Late-Interaction Architecture", divider="rainbow")

st.markdown(
    """
    Multimodal retrieval is inherently **bidirectional and heterogeneous**.
    A user may provide a textual query seeking visual diagrams (**Text-to-Image**),
    or upload an error screenshot to retrieve the corresponding troubleshooting wiki (**Image-to-Text**),
    or search for similar circuit diagrams using a query schematic (**Image-to-Image**).
    
    The diagram below models the complete **Cross-Modal Retrieval Topology**, contrasting traditional pooled bi-encoders
    against modern multi-vector **ColPali MaxSim Late-Interaction** scoring.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Multimodal_Retrieval {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_query_input {
            label = "1. QUERY INGRESS (MULTIMODAL QUERY MODES)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            mode_t2i [label="Mode A: Text Query\\n'Show NVLink 8-GPU mesh'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            mode_i2t [label="Mode B: Query Image\\n(Upload Error Screenshot / Architecture JPEG)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_encoder_tier {
            label = "2. DUAL-MODAL ENCODING & MULTI-VECTOR TOKENIZATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            text_encoder [label="Text Tokenizer & Encoder\\nGenerates Q Token Embeddings: [q_1, q_2, ..., q_m]", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            vision_patch_encoder [label="Vision Patch Encoder (PaliGemma)\\nGenerates Document Patch Array: [d_1, d_2, ..., d_n]", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            mode_t2i -> text_encoder;
            mode_i2t -> vision_patch_encoder;
        }

        subgraph cluster_late_interaction {
            label = "3. LATE-INTERACTION SCORING ENGINE (MAXSIM)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            maxsim_matrix [label="ColPali MaxSim Matrix\\nScore = Sum_j ( max_i ( q_j . d_i ) )\\nEvery query token matches its best visual patch", fillcolor="#312E81", fontcolor="#A5B4FC"];
            dense_ann [label="Fast Pruning (HNSW Bi-Encoder)\\nPrunes index to Top-50 candidate documents", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            text_encoder -> dense_ann;
            vision_patch_encoder -> dense_ann;
            dense_ann -> maxsim_matrix [label="Top Candidates"];
        }

        subgraph cluster_retrieval_output {
            label = "4. CROSS-MODAL FUSED CANDIDATE POOL";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            fused_ranked_results [label="Ranked Multimodal Candidates\\n- Visual Images & Schematics\\n- Grounding Text Documentation", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            maxsim_matrix -> fused_ranked_results;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: IMAGE SEARCH ---
st.header("2. Image Search (Reverse Image Lookup, Feature Vector Nearest Neighbors & Structural Similarity)", divider="rainbow")

st.markdown(
    """
    **Image Search (Reverse Image Lookup)** allows users to query with an image rather than text words:
    
    - **Visual Feature Extraction**: A query image is passed through a vision transformer to extract dense visual feature maps.
    - **Nearest Neighbor Graph Traversal**: The vector database computes cosine distance between the query image embedding and all indexed asset embeddings.
    - **Structural Similarity**: Enables finding identical architectural diagrams that have been slightly edited, re-cropped, or translated into different languages.
    """
)

# --- SECTION 3: TEXT-TO-IMAGE RETRIEVAL ---
st.header("3. Text-to-Image Retrieval (Natural Language Queries against Visual Assets & Caption Matching)", divider="rainbow")

st.markdown(
    """
    **Text-to-Image Retrieval** is the most common user workflow: a user writes a conceptual description (*"Show me the cluster storage topology"*)
    and expects the system to surface visual image files:
    
    - **Dual Embedding Alignment**: Works by projecting the text query into the CLIP or ColPali latent space where visual diagrams reside.
    - **Hybrid Caption Fusion**: Combines visual vector search with BM25 keyword matching over image captions and OCR text banners.
    - **Resolution Prioritization**: Rerankers favor high-resolution diagrams over low-resolution thumbnail graphics.
    """
)

# --- SECTION 4: IMAGE-TO-TEXT RETRIEVAL ---
st.header("4. Image-to-Text Retrieval (Query Image as Context against Text Knowledge Bases & Caption Generation)", divider="rainbow")

st.markdown(
    """
    In customer engineering and field maintenance, users submit photos or error screenshots asking: *"What does this error code mean and how do I fix it?"*:
    
    - **OCR & Visual Entity Parsing**: Extracting text error strings and visual warning icons from the query image.
    - **Cross-Modal Querying**: The query image embedding probes the text knowledge base, surfacing relevant engineering runbooks and incident recovery guides.
    """
)

# --- SECTION 5: CROSS-MODAL RETRIEVAL ---
st.header("5. Cross-Modal Retrieval (Unified Multi-Vector Indexing, Modality-Agnostic Top-K & Late Interaction MaxSim)", divider="rainbow")

st.markdown(
    """
    In unified **Cross-Modal Retrieval**, the query and the corpus can each contain any mix of text and imagery:
    
    ### ColPali MaxSim Formulation:
    Instead of collapsing an image into a single 1D vector, ColPali produces an array of patch vectors:
    
    `MaxSim_Score(Query, Document) = Sum_j [ max_i ( q_j . d_i ) ]`
    
    For every query token `q_j`, the engine finds the single visual image patch `d_i` that produces the maximum dot product, and sums those maximum scores across all query tokens.
    This guarantees that if a user searches for *"ETCD quorum"*, the score peaks when a query token matches the exact visual patch where *"ETCD"* is drawn on a diagram.
    """
)

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Enterprise Cross-Modal Retrieval Orchestrator", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `CrossModalRetrievalEngine`.
    It orchestrates multi-directional cross-modal queries across text-to-image, image-to-text, and unified modalities,
    simulating MaxSim late-interaction alignment scoring.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class CrossModalRetrievalEngine:
    """Enterprise Cross-Modal Retrieval Orchestrator."""

    def __init__(self):
        self.visual_assets = [
            {"id": "img_k8s_arch", "type": "Image", "title": "Kubernetes Control Plane Topology", "tags": ["kubernetes", "k8s", "control", "plane", "etcd", "apiserver"], "caption": "Detailed diagram of API Server, ETCD cluster, Scheduler, and Controller Manager."},
            {"id": "img_gpu_mesh", "type": "Image", "title": "GPU NVLink Mesh Interconnect", "tags": ["gpu", "nvlink", "hardware", "switch", "cluster"], "caption": "Full-mesh 900 GB/s bidirectional NVLink crossbar connecting 8 accelerators."},
            {"id": "img_auth_flow", "type": "Image", "title": "OAuth2 PKCE Authentication Flow", "tags": ["oauth2", "pkce", "security", "token", "auth"], "caption": "Sequence diagram showing Code Challenge, Auth Code exchange, and Access Token emission."}
        ]
        self.text_assets = [
            {"id": "txt_k8s_spec", "type": "Text", "title": "Kubernetes High Availability Best Practices", "tags": ["kubernetes", "k8s", "control", "plane", "etcd", "ha"], "caption": "ETCD requires a minimum of 3 odd-numbered replicas for quorum. Control plane nodes must span distinct failure domains."},
            {"id": "txt_gpu_spec", "type": "Text", "title": "NVLink Bandwidth & Thermal Envelopes", "tags": ["gpu", "nvlink", "hardware", "cooling"], "caption": "Total power dissipation for NVLink switches requires active liquid cooling loops maintaining cold-plate temperatures below 45C."},
            {"id": "txt_auth_spec", "type": "Text", "title": "Zero-Trust Token Lifetime Governance", "tags": ["oauth2", "security", "token", "auth"], "caption": "Access tokens expire within 15 minutes; refresh tokens require rotation with cryptographic replay detection."}
        ]

    def _score_tokens(self, q_tokens: List[str], target_tags: List[str]) -> float:
        """Computes overlap score simulating normalized cosine similarity in joint space."""
        intersection = len(set(q_tokens).intersection(set(target_tags)))
        return float(intersection / (len(q_tokens) + 1e-5))

    def retrieve(self, query: str, mode: str = "Cross-Modal (Unified)", top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes retrieval across specified modality path."""
        q_tokens = [w.lower() for w in query.split()]
        candidates = []
        
        pool = []
        if mode == "Text-to-Image (Image Search)":
            pool = self.visual_assets
        elif mode == "Image-to-Text (Reverse Lookup)":
            pool = self.text_assets
        elif mode == "Image-to-Image (Visual Similarity)":
            pool = self.visual_assets
        else: # Cross-Modal (Unified)
            pool = self.visual_assets + self.text_assets

        for item in pool:
            score = self._score_tokens(q_tokens, item["tags"] + item["title"].lower().split())
            maxsim_score = min(1.0, score + 0.35 if score > 0 else 0.12)
            candidates.append({
                "id": item["id"],
                "type": item["type"],
                "title": item["title"],
                "caption": item["caption"],
                "similarity_score": round(maxsim_score, 4)
            })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]
''',
    language="python"
)

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: Bidirectional Cross-Modal Retrieval Laboratory & MaxSim Visualizer", divider="rainbow")

st.markdown(
    """
    Test bidirectional cross-modal retrieval across all four operational modes:
    Text-to-Image, Image-to-Text, Visual Similarity, and Unified Cross-Modal search.
    Observe how query keywords align with visual diagrams and textual specifications.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔀 Retrieval Mode & Query Controls")
    
    retrieval_mode = st.selectbox(
        "Select Retrieval Mode:",
        [
            "Cross-Modal (Unified)",
            "Text-to-Image (Image Search)",
            "Image-to-Text (Reverse Lookup)",
            "Image-to-Image (Visual Similarity)"
        ],
        index=0
    )
    
    query_text = st.text_input(
        "Enter Query String / Image Identifier:",
        "Kubernetes ETCD control plane architecture"
    )
    
    top_k_count = st.slider("Top-K Candidates:", min_value=1, max_value=6, value=3, step=1)

with wb_col2:
    st.subheader("⚡ Cross-Modal Candidate Telemetry")
    
    engine = CrossModalRetrievalEngine()
    results = engine.retrieve(query_text, mode=retrieval_mode, top_k=top_k_count)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Retrieved Chunks", len(results))
    top_s = results[0]["similarity_score"] if results else 0.0
    m2.metric("MaxSim Score", f"{top_s:.4f}")
    img_c = sum(1 for r in results if r["type"] == "Image")
    m3.metric("Visual Assets", img_c)
    
    for r in results:
        with st.container(border=True):
            st.markdown(f"**[{r['type']}] {r['title']}** (Score: `{r['similarity_score']}`)")
            st.caption(r["caption"])

st.markdown("### Ranked Cross-Modal Candidate Manifest")
st.dataframe(pd.DataFrame(results), use_container_width=True)

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: What is the MaxSim operator in ColPali and how does it compute token-to-patch alignment?**"):
    st.markdown(
        """
        In ColPali late interaction:
        1. Query tokens are embedded as vectors q_1, ..., q_m in R^d.
        2. The document page image is embedded as visual patch vectors d_1, ..., d_n in R^d.
        3. The **MaxSim Operator** computes the dot product of query token `q_j` with every patch vector `d_i`, finds the single maximum dot product, and sums those maximums across all query tokens:
        `MaxSim = Sum_j [ max_i ( q_j . d_i ) ]`.
        
        This enables token-level alignment directly to visual page regions without relying on brittle OCR character sequences.
        """
    )

with st.expander("**Q2: How does Cross-Modal Retrieval prevent modality bias in unified vector spaces?**"):
    st.markdown(
        """
        In joint embedding spaces (like CLIP), vectors exhibit a **Modality Gap**: image embeddings cluster in one sub-space while text embeddings cluster in a slightly offset sub-space.
        As a result, text queries often score higher against text documents than against relevant image diagrams.
        
        Production systems calibrate modality scores:
        1. **Centroid Centering**: Subtracting the modality cluster mean from text and image vectors before cosine calculation.
        2. **Temperature Scaling**: Applying learned temperature factors tau_image and tau_text to equalize score distributions.
        """
    )

with st.expander("**Q3: What are the engineering challenges of indexing multi-vector embeddings at million-document scale?**"):
    st.markdown(
        """
        Storing ~1,024 vectors per document page consumes 500 KB per page, requiring 500 GB of memory for 1 million pages.
        
        Production scaling techniques:
        1. **Two-Stage Retrieval**: First retrieve the top 100 documents using a lightweight single-vector dense embedding (HNSW), then evaluate expensive MaxSim multi-vector scoring strictly on the top 100 candidates.
        2. **Binary Quantization (BQ)**: Quantizing 128-dimensional float vectors into 128 bits, reducing memory footprint by 32x while speeding up dot product calculations using hardware XOR instructions.
        """
    )

with st.expander("**Q4: What is Reverse Image Search and how does it function in technical documentation RAG?**"):
    st.markdown(
        """
        Reverse Image Search allows engineers to upload an error dialog screenshot or wiring schematic to find documentation.
        The query image is passed through a vision transformer to produce an image embedding vector v_query.
        The vector database performs approximate nearest neighbor search to find stored images with the closest visual feature representations.
        """
    )

with st.expander("**Q5: How do you evaluate cross-modal retrieval recall and precision?**"):
    st.markdown(
        """
        Standard text metrics (NDCG@10, Recall@K, MRR) are adapted to cross-modal test benchmarks:
        - **Text-to-Image Recall@K**: Did a text query place the gold-standard schematic within the top K results?
        - **Visual IoU Attribution**: Did the highest-scoring ColPali visual patch overlap with the ground-truth visual bounding box containing the answer?
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Ignoring the CLIP Modality Gap**  
        **Antipattern**: Calculating raw cosine similarity between text queries and image vectors without calibration.  
        **Consequence**: Text documents score systematically higher than images due to vector cluster offsets, starving the LLM of visual diagrams.  
        **Production Fix**: Apply centroid subtraction or learnable temperature scaling to calibrate text-image similarity distributions.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Exhaustive MaxSim Scoring over Entire Corpora**  
        **Antipattern**: Calculating full token-to-patch MaxSim matrix multiplications across 100,000 document images.  
        **Consequence**: Requires billions of dot products per query, inflating search latency to several seconds.  
        **Production Fix**: Deploy a two-stage pipeline: fast dense ANN pruning to Top-50 candidates, followed by MaxSim reranking.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Slicing Visual Assets from Surrounding Text Context**  
        **Antipattern**: Indexing image diagrams without referencing the surrounding document section or paragraph context.  
        **Consequence**: The image loses critical contextual framing (e.g., product model numbers or software version constraints).  
        **Production Fix**: Prepend section headers and document metadata to image caption chunks.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Discarding Bounding Box Patch Coordinates**  
        **Antipattern**: Storing multi-vector patch embeddings without their 2D coordinate map on the source page.  
        **Consequence**: The system cannot display visual bounding boxes highlighting the retrieved diagram to human users.  
        **Production Fix**: Preserve patch coordinate arrays `(row, col)` alongside patch vectors.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Text-to-Image Retrieval?")
        st.markdown("**Back**: A cross-modal search where a natural language text query directly retrieves relevant visual images and diagrams from a shared vector space.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the Modality Gap in CLIP?")
        st.markdown("**Back**: A geometric phenomenon where text embeddings and image embeddings occupy disjoint sub-spaces, causing uncalibrated cross-modal cosine similarity scores.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Late Interaction in Multimodal Retrieval?")
        st.markdown("**Back**: Delaying vector aggregation until retrieval time, allowing every query token to compute similarity against every document visual patch via MaxSim.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Binary Quantization (BQ)?")
        st.markdown("**Back**: Compressing floating-point vectors into binary bit arrays, slashing memory consumption by 32x while accelerating dot product scoring via hardware XOR instructions.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of cross-modal retrieval, bidirectional search, and ColPali late interaction:")

chk1 = st.checkbox("I understand the bidirectional workflows: Text-to-Image, Image-to-Text, and Visual Similarity search.", value=True)
chk2 = st.checkbox("I can explain how the ColPali MaxSim operator computes token-to-patch alignment across visual page images.", value=True)
chk3 = st.checkbox("I know how to identify and calibrate the CLIP modality gap to prevent text bias in unified vector stores.", value=True)
chk4 = st.checkbox("I understand the two-stage retrieval architecture (Dense ANN pruning + Late Interaction MaxSim reranking).", value=True)
chk5 = st.checkbox("I can implement Binary Quantization (BQ) to scale multi-vector multimodal retrieval to enterprise volume.", value=True)
chk6 = st.checkbox("I know how to evaluate cross-modal recall and precision using visual IoU and patch attribution benchmarks.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Multimodal Retrieval and Cross-Modal Search Architectures.")
