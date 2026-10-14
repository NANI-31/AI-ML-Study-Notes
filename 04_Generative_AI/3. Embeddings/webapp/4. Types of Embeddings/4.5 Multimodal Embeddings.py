import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.5 Multimodal Embeddings - Embeddings Curriculum",
    page_icon="🌌",
    layout="wide"
)

# Title & Overview
st.title("4.5 Multimodal Embeddings: Joint Latent Spaces, CLIP, SigLIP, & Cross-Modal Alignment")
st.caption("Chapter 4: Types of Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Traditional embedding models operate strictly within a single modality: text-in, text-vector-out.
However, human knowledge and real-world intelligence span multiple sensory channels: text, images, audio, video, sensor readings, and 3D spatial models.

**Multimodal Embeddings** bridge heterogeneous modalities by mapping disparate data types into a **shared, unified metric space (Joint Latent Space)**. In this shared space:
* The vector representation of the text description *"A golden retriever catching a tennis ball in the park"*
* The vector representation of an image showing a golden retriever jumping for a yellow ball
* The vector representation of an audio recording of a dog barking excitedly

...all inhabit nearly the **same coordinates** and share high cosine similarity!

Pioneered by OpenAI's **CLIP (Contrastive Language-Image Pretraining)** and refined by Google's **SigLIP (Sigmoid Loss for Language-Image Pre-Training)** and Meta's **ImageBind**, multimodal embeddings power modern visual search, zero-shot image classification, text-to-image generation (Stable Diffusion / DALL-E), and multimodal RAG.
""")

st.divider()

# Section 1: Joint Latent Space Foundations
st.header("1. Joint Latent Space Foundations & Modality Encoders", divider="rainbow")

col_mm1, col_mm2 = st.columns([1, 1])

with col_mm1:
    st.markdown("""
    ### How Multimodal Alignment Works:
    * **Separate Dual Encoders**:
      * **Text Encoder**: Typically a Transformer (like GPT or RoBERTa) tokenizing text and outputting a pooled text representation vector.
      * **Vision Encoder**: Typically a Vision Transformer (ViT) or ResNet processing non-overlapping image patches into visual tokens.
      * **Audio Encoder**: A spectrogram transformer (e.g. AST, Whisper encoder) turning mel-spectrogram frequencies into audio tokens.
      * **Video Encoder**: Spatio-temporal 3D-convolutions or Video Transformer blocks capturing frame sequences and motion dynamics.
    * **Linear / MLP Projection Heads**:
      * Each modality encoder outputs native hidden vectors of arbitrary dimensionality (e.g. text hidden dim = 768, vision hidden dim = 1024).
      * A learned linear projection matrix projects both representations into a **common shared dimension d** (e.g., d = 512 or 768).
    * **Unit L2 Normalization**:
      * Vectors from all encoders are L2-normalized onto a shared hypersphere.
      * Similarity between any two modalities is computed directly using the dot product:
        `Cosine Similarity = Vector_Text dot Vector_Image`
    """)

with col_mm2:
    st.markdown("### Dual-Encoder Multimodal Architecture")
    mm_dot = """
    digraph MultimodalSpace {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_inputs {
            label="Input Modalities"; style="dashed"; color="#94A3B8";
            TxtIn [label="Text: 'A cute red fox'\\n(Tokens)", fillcolor="#EFF6FF", color="#3B82F6"];
            ImgIn [label="Image: fox.jpg\\n(224x224 RGB)", fillcolor="#ECFDF5", color="#10B981"];
            AudIn [label="Audio: fox_call.wav\\n(Mel-Spectrogram)", fillcolor="#FEF3C7", color="#F59E0B"];
        }

        TxtEnc [label="Text Transformer\\n(e.g., RoBERTa / BERT)", fillcolor="#DBEAFE"];
        ImgEnc [label="Vision Transformer\\n(ViT-B/32 or ViT-L/14)", fillcolor="#D1FAE5"];
        AudEnc [label="Audio Transformer\\n(Audio Spectrogram Tr.)", fillcolor="#FDE68A"];

        ProjTxt [label="Projection Head W_t\\n(768 -> 512)", fillcolor="#BFDBFE"];
        ProjImg [label="Projection Head W_v\\n(1024 -> 512)", fillcolor="#A7F3D0"];
        ProjAud [label="Projection Head W_a\\n(768 -> 512)", fillcolor="#FCD34D"];

        JointSpace [label="Unified L2 Hypersphere (d = 512)\\nHigh Cosine Similarity for Semantically Matched Concepts", shape=ellipse, fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];

        TxtIn -> TxtEnc -> ProjTxt -> JointSpace [label="v_text"];
        ImgIn -> ImgEnc -> ProjImg -> JointSpace [label="v_image"];
        AudIn -> AudEnc -> ProjAud -> JointSpace [label="v_audio"];
    }
    """
    st.graphviz_chart(mm_dot, use_container_width=False)

st.divider()

# Section 2: Contrastive Learning: CLIP vs. SigLIP vs. ImageBind
st.header("2. Training Multimodal Encoders: CLIP vs. SigLIP vs. ImageBind", divider="rainbow")

st.markdown("""
How do we train separate neural networks so that an image of an apple and the word "apple" land on identical coordinates?
The secret lies in **Contrastive Learning**.
""")

col_loss1, col_loss2 = st.columns([1, 1])

with col_loss1:
    st.markdown("""
    ### 1. OpenAI CLIP (Contrastive Language-Image Pretraining)
    * Given a batch of N (image, text) pairs:
      * There are N positive pairs (matching image and caption).
      * There are N * (N - 1) negative pairs (unmatched images and captions in the batch).
    * CLIP computes an N x N cosine similarity matrix scaled by a learnable temperature parameter:
      `Logits = (Image_Embeddings @ Text_Embeddings.T) / temperature`
    * It applies **Symmetric Cross-Entropy (InfoNCE)** along both rows (image-to-text) and columns (text-to-image).
    * **Limitation**: Requires massive batch sizes (e.g. 32,768) because negatives come strictly from within the batch!
    
    ### 2. Google SigLIP (Sigmoid Loss for Language-Image Pretraining)
    * Replaces the global Softmax normalization with pairwise independent **Sigmoid Binary Cross-Entropy**.
    * Evaluates each of the N x N pair cells independently as a binary classification (positive if i == j, negative if i != j).
    * **Major Advantage**: Eliminates the all-gather communication bottleneck across GPUs; memory scales linearly with batch size, outperforming CLIP at smaller batch sizes!
    """)

with col_loss2:
    st.markdown("""
    ### 3. Meta ImageBind: Holistic Multi-Modal Binding
    * While CLIP binds Text to Image, **ImageBind** uses Image as the central binding anchor to unify **6 modalities**:
      1. Images / Video
      2. Text
      3. Audio
      4. Depth maps
      5. Thermal infrared images
      6. IMU inertial motion sensors
    * ImageBind aligns each non-image modality directly with the image space:
      `Audio <-> Image` and `Text <-> Image`.
    * **Emergent Cross-Modal Property**: Even though the model was never directly trained on `Audio <-> Text` pairs, you can search audio recordings using pure text queries zero-shot, because both are independently aligned with the shared visual space!
    """)

st.markdown("### Comparison Table: Multimodal Alignment Architectures")
st.markdown("""
| Architecture | Encoders | Projection Dim | Loss Function | Key Strength | Primary Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI CLIP** | ViT + Transformer | 512 / 768 | Symmetric InfoNCE Softmax | Master baseline for multimodal zero-shot | Zero-shot image classification, visual search |
| **Google SigLIP** | ViT + Transformer | 768 / 1152 | Sigmoid Binary Cross-Entropy | Linear GPU memory scaling, superior retrieval | Modern multimodal vision-language models (PaliGemma) |
| **Meta ImageBind** | ViT + AST + Modality Heads | 1024 | Pairwise Contrastive with Image Anchor | Unifies 6 sensory modalities zero-shot | Cross-sensory search, audio-visual robotics |
| **OpenAI CLIP ViT-L/14@336px** | High-res ViT | 768 | Contrastive InfoNCE | Fine-grained visual feature discrimination | Diffusion model conditioning (Stable Diffusion) |
""")

st.divider()

# Section 3: Production Code: Cross-Modal Visual Search Engine
st.header("3. Production Implementation: Cross-Modal Vector Search Engine", divider="rainbow")

st.markdown("""
The production class below demonstrates a clean, robust multimodal index engine. It showcases how query text embeddings are mapped against indexed visual assets in a shared normalized latent space.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class MultimodalSearchEngine:
    """
    A production-grade in-memory Multimodal Vector Search Engine.
    Emulates dual-encoder inference (e.g. CLIP / SigLIP) with joint L2 projection.
    """
    def __init__(self, embedding_dim: int = 512, seed: int = 42):
        self.dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.image_catalog: List[Dict[str, Any]] = []
        self.image_embeddings: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalizes a vector or matrix across rows."""
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def register_image(self, image_id: str, title: str, category: str, synthetic_semantic_features: np.ndarray):
        """Indexes an image asset with its vision-encoder projected vector."""
        norm_vec = self._normalize(synthetic_semantic_features.astype(np.float32))
        self.image_catalog.append({
            "image_id": image_id,
            "title": title,
            "category": category,
            "vector": norm_vec
        })
        if len(self.image_embeddings) == 0:
            self.image_embeddings = norm_vec.reshape(1, -1)
        else:
            self.image_embeddings = np.vstack([self.image_embeddings, norm_vec.reshape(1, -1)])

    def search_by_text(self, text_query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Computes cross-modal dot product similarity between text query and visual catalog.
        Returns top_k ranked visual matches.
        """
        if len(self.image_catalog) == 0:
            return []
        
        norm_query = self._normalize(text_query_vector.astype(np.float32)).reshape(-1)
        # Cosine similarity is direct dot product since both are unit-normalized
        scores = np.dot(self.image_embeddings, norm_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.image_catalog[idx]
            results.append({
                "image_id": item["image_id"],
                "title": item["title"],
                "category": item["category"],
                "similarity_score": float(scores[idx])
            })
        return results
''', language="python")

# Actual Python implementation for the interactive workbench
class MultimodalSearchEngine:
    def __init__(self, embedding_dim: int = 8, seed: int = 42):
        self.dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.image_catalog: List[Dict[str, Any]] = []
        self.image_embeddings: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def register_image(self, image_id: str, title: str, category: str, semantic_vector: np.ndarray):
        norm_vec = self._normalize(semantic_vector.astype(np.float32))
        self.image_catalog.append({
            "image_id": image_id,
            "title": title,
            "category": category,
            "vector": norm_vec
        })
        if len(self.image_embeddings) == 0:
            self.image_embeddings = norm_vec.reshape(1, -1)
        else:
            self.image_embeddings = np.vstack([self.image_embeddings, norm_vec.reshape(1, -1)])

    def search_by_text(self, text_query_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if len(self.image_catalog) == 0:
            return []
        norm_query = self._normalize(text_query_vector.astype(np.float32)).reshape(-1)
        scores = np.dot(self.image_embeddings, norm_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = self.image_catalog[idx]
            results.append({
                "image_id": item["image_id"],
                "title": item["title"],
                "category": item["category"],
                "similarity_score": float(scores[idx])
            })
        return results

st.divider()

# Section 4: Interactive Workbench: Cross-Modal Semantic Retrieval
st.header("4. Interactive Workbench: Zero-Shot Cross-Modal Retrieval Simulator", divider="rainbow")

st.markdown("""
Test zero-shot multimodal retrieval! Select or customize a text search query and observe how the dual-encoder system aligns the query representation with visual and acoustic catalog entries in a shared latent space.
""")

# Pre-defined semantic basis coordinates in 8-dimensional space
# Dimensions: [Canine/Dog, Nature/Outdoor, Technology/Computer, Automotive/Car, Music/Acoustic, Food/Cooking, Space/Astronomy, Medical/Health]
dim_labels = ["Canine", "Outdoor Nature", "Computing Tech", "Automotive", "Audio Music", "Culinary", "Astronomy", "Medical"]

sample_assets = [
    {
        "id": "IMG_001",
        "title": "Golden Retriever catching frisbee in sunlit meadow",
        "category": "Visual (Image)",
        "vector": np.array([0.92, 0.85, 0.05, 0.02, 0.01, 0.08, 0.01, 0.02])
    },
    {
        "id": "IMG_002",
        "title": "Modern data center server rack with blinking LED cables",
        "category": "Visual (Image)",
        "vector": np.array([0.01, 0.04, 0.95, 0.12, 0.05, 0.01, 0.15, 0.03])
    },
    {
        "id": "IMG_003",
        "title": "Electric supercar charging at high-speed station",
        "category": "Visual (Image)",
        "vector": np.array([0.02, 0.10, 0.45, 0.92, 0.04, 0.02, 0.05, 0.01])
    },
    {
        "id": "IMG_004",
        "title": "Artisanal sourdough bread with butter and rosemary",
        "category": "Visual (Image)",
        "vector": np.array([0.05, 0.18, 0.02, 0.01, 0.02, 0.94, 0.01, 0.05])
    },
    {
        "id": "AUD_005",
        "title": "Acoustic folk guitar solo recorded in studio",
        "category": "Acoustic (Audio)",
        "vector": np.array([0.01, 0.15, 0.08, 0.02, 0.96, 0.04, 0.02, 0.01])
    },
    {
        "id": "IMG_006",
        "title": "Deep field telescope view of distant spiral galaxies",
        "category": "Visual (Image)",
        "vector": np.array([0.01, 0.20, 0.25, 0.01, 0.02, 0.01, 0.96, 0.02])
    }
]

engine = MultimodalSearchEngine(embedding_dim=8)
for asset in sample_assets:
    engine.register_image(asset["id"], asset["title"], asset["category"], asset["vector"])

col_ctrl, col_res = st.columns([1, 1])

with col_ctrl:
    st.subheader("1. Configure Text Query")
    query_choice = st.selectbox(
        "Select Test Text Query:",
        [
            "Playful puppy running outside on grass",
            "High performance cloud server infrastructure",
            "Zero emission battery electric sports vehicle",
            "Freshly baked gourmet loaf of bread",
            "Relaxing melodic acoustic guitar track",
            "Cosmic nebula and distant star clusters"
        ]
    )

    query_vector_presets = {
        "Playful puppy running outside on grass": np.array([0.90, 0.82, 0.02, 0.01, 0.01, 0.05, 0.01, 0.02]),
        "High performance cloud server infrastructure": np.array([0.02, 0.03, 0.94, 0.15, 0.02, 0.01, 0.12, 0.04]),
        "Zero emission battery electric sports vehicle": np.array([0.01, 0.08, 0.40, 0.95, 0.03, 0.01, 0.04, 0.01]),
        "Freshly baked gourmet loaf of bread": np.array([0.03, 0.12, 0.01, 0.01, 0.02, 0.95, 0.01, 0.06]),
        "Relaxing melodic acoustic guitar track": np.array([0.01, 0.12, 0.05, 0.01, 0.95, 0.03, 0.02, 0.01]),
        "Cosmic nebula and distant star clusters": np.array([0.01, 0.15, 0.20, 0.01, 0.01, 0.01, 0.95, 0.02])
    }

    q_vec = query_vector_presets[query_choice]
    
    top_k = st.slider("Top K Retrieved Assets:", min_value=1, max_value=6, value=4)

    st.markdown("#### Projected Query Latent Vector (Normalized)")
    norm_q = q_vec / np.linalg.norm(q_vec)
    df_q = pd.DataFrame({
        "Latent Concept": dim_labels,
        "Activation Magnitude": [f"{v:.4f}" for v in norm_q]
    })
    st.dataframe(df_q, use_container_width=True)

with col_res:
    st.subheader("2. Cross-Modal Similarity Matches")
    matches = engine.search_by_text(q_vec, top_k=top_k)
    
    match_df = pd.DataFrame(matches)
    match_df["Rank"] = range(1, len(matches) + 1)
    match_df = match_df[["Rank", "image_id", "category", "title", "similarity_score"]]
    match_df.rename(columns={
        "image_id": "Asset ID",
        "category": "Modality",
        "title": "Asset Description",
        "similarity_score": "Cosine Similarity"
    }, inplace=True)
    
    st.dataframe(
        match_df.style.format({"Cosine Similarity": "{:.4f}"}).background_gradient(subset=["Cosine Similarity"], cmap="Blues"),
        use_container_width=True
    )

    top_hit = matches[0]
    st.success(f"**Top Cross-Modal Match**: [{top_hit['image_id']}] {top_hit['title']} (Score: {top_hit['similarity_score']:.4f})")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Multimodal Traps & Edge Cases", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do dual-encoder models like CLIP enable zero-shot image classification without fine-tuning?**
    * **Answer**:
      * In traditional vision classifiers, the final layer has a fixed set of classes (e.g. 1000 ImageNet categories with a Softmax output).
      * In CLIP, zero-shot classification is converted into a **cross-modal retrieval problem**:
        1. Synthesize text prompts for all candidate classes: `"A photo of a {class_name}"`.
        2. Pass all prompt strings through the Text Encoder to generate a matrix of target vectors.
        3. Pass the test image through the Vision Encoder to get the image vector.
        4. Compute cosine similarity between the image vector and all class text vectors.
        5. Predict the class that yields the maximum cosine similarity score.
      * This allows classifying arbitrary, novel concepts never seen in training without modifying network weights!

    **Q2: What is the primary difference between CLIP InfoNCE Loss and SigLIP Loss?**
    * **Answer**:
      * **CLIP (InfoNCE)**: Uses a Softmax denominator that sums exponential similarities across all N negatives in the batch.
        This introduces a global dependency: all GPU workers must synchronize and communicate their full negative matrices across the interconnect (all-gather), creating a communication bottleneck.
      * **SigLIP (Sigmoid Loss)**: Treats every cell in the N x N cross-modal matrix as an independent binary classification using the Sigmoid function. Positive cells (matching pairs) are pushed to 1; negative cells are pushed to 0.
        This removes the global denominator, completely eliminates GPU all-gather communication, and scales memory linearly with batch size.

    **Q3: How does ImageBind achieve cross-modal binding without pairwise training data between all modalities?**
    * **Answer**:
      * Direct training data between all modalities (e.g. thermal infrared paired with audio recordings) is virtually non-existent.
      * ImageBind solves this by using **Image as a universal binding pivot**.
      * Paired datasets exist for `(Image, Text)`, `(Image, Audio)`, `(Image, Depth)`, and `(Image, Thermal)`.
      * By independently mapping each modality into the visual coordinate space, the non-visual modalities automatically align with each other transitively!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: The 'Modality Gap' Phenomenon**:
      * *Issue*: Even when trained to convergence, text vectors and image vectors occupy separate geometric clusters on the hypersphere, separated by an offset cone!
      * *Consequence*: Cosine similarity between image and text is systematically lower than image-image or text-text similarity.
      * *Fix*: Apply temperature scaling or calibrate the cross-modal threshold; do not use raw Euclidean distance directly across modalities.

    * **Trap 2: Ignoring Visual Resolution Limits**:
      * *Issue*: Standard CLIP (ViT-B/32) processes 224x224 images divided into 32x32 patches (yielding only 7x7 = 49 tokens).
      * *Consequence*: Tiny visual details, text inside images (OCR), or small objects are completely lost in patch pooling.
      * *Fix*: Upgrade to ViT-L/14@336px or deploy vision-language models with dynamic high-resolution patching (e.g. SigLIP-So400M).

    * **Trap 3: Prompt Template Fragility in Zero-Shot Search**:
      * *Issue*: Querying with raw single words (e.g. `"dog"`) rather than prompt-engineered templates.
      * *Consequence*: Causes up to 5-10% accuracy degradation because the text encoder was trained on full captions.
      * *Fix*: Use prompt ensembling: average embeddings over multiple templates (`"A photo of a {x}"`, `"A clean picture showing a {x}"`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Joint Latent Space**
      * *Definition*: A shared high-dimensional vector space where vectors from different modalities (text, image, audio) representing the same semantic concept are mapped to nearby coordinates.
    * **Flashcard 2: Dual Encoder**
      * *Definition*: An architecture featuring distinct specialized encoders for each modality (e.g. ViT for images, Transformer for text) connected to shared projection heads.
    * **Flashcard 3: SigLIP**
      * *Definition*: Sigmoid Loss for Language-Image Pretraining; replaces global Softmax with pairwise sigmoid cross-entropy, eliminating GPU all-gather bottlenecks.
    * **Flashcard 4: Modality Gap**
      * *Definition*: An empirical geometric phenomenon where embeddings from different modalities inhabit disjoint cones on the hypersphere even after contrastive alignment.
    """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Multimodal Embeddings:

- [ ] I can explain the mechanics of a dual-encoder architecture with linear projection heads.
- [ ] I understand how contrastive loss (InfoNCE) aligns positive image-text pairs while repelling negative pairs.
- [ ] I can articulate the architectural difference and efficiency gains of SigLIP over standard CLIP.
- [ ] I understand how Meta ImageBind uses the visual modality as an anchor to bind 6 diverse sensory modalities.
- [ ] I know how zero-shot image classification is implemented as a cross-modal nearest-neighbor search.
- [ ] I am aware of the Modality Gap phenomenon and how to handle cross-modal score calibration in production.
""")
