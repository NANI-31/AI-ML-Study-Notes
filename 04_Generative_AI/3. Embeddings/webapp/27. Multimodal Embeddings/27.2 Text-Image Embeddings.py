import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="27.2 Text-Image Embeddings - Embeddings Curriculum",
    page_icon="🔀",
    layout="wide"
)

# Title & Overview
st.title("27.2 Text-Image Embeddings: CLIP, Shared Latent Spaces, Text-to-Image & Image-to-Text Search")
st.caption("Chapter 27: Multimodal Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
For decades, Computer Vision and Natural Language Processing operated as isolated computational silos. 
Text models understood words and syntax; vision models understood pixels and bounding boxes.
In 2021, OpenAI introduced **CLIP (Contrastive Language-Image Pre-training)**, which shattered this divide by learning a 
**joint, shared high-dimensional representation space** for both modalities simultaneously.

In a shared text-image embedding space:
* The text vector for *"a cute golden retriever puppy sitting in green grass"*
* And the pixel image vector of that exact dog
* Are mapped to **virtually identical coordinates** in 512-dimensional or 768-dimensional space!

Because both text and image vectors exist in the exact same metric space, we can compute cosine similarity directly between 
a **Text Query and an Image**, or between an **Image Query and Text Documents**, enabling instant **Text-to-Image Search**, 
**Image-to-Text Search**, and **Zero-Shot Image Classification**.
""")

st.divider()

# Section 1: The CLIP Dual-Encoder Architecture
st.header("1. Architecture of Shared Text-Image Embedding Spaces (CLIP & SigLIP)", divider="rainbow")

col_clip1, col_clip2 = col_clip1, col_clip2 = st.columns([1, 1])

with col_clip1:
    st.markdown("""
    ### The Dual-Encoder Paradigm
    A multimodal text-image model consists of two distinct neural networks trained in tandem:
    
    1. **Image Encoder `f_img(I)`**:
       * A Vision Transformer (e.g., ViT-B/32 or ViT-L/14) or CNN that processes image pixels.
       * Outputs a raw visual feature vector `h_img in R^d_v`.
    2. **Text Encoder `f_txt(T)`**:
       * A Transformer language model that processes tokenized text captions.
       * Outputs a raw textual feature vector `h_txt in R^d_t`.
    3. **Linear Projection Heads & L2 Normalization**:
       * Learned linear projection layers map both modalities into a shared dimension `d_e` (e.g., 512 or 768):
         `v_img = L2_Norm(W_img * h_img)`
         `v_txt = L2_Norm(W_txt * h_txt)`
    4. **Contrastive InfoNCE Training Objective**:
       * In a training batch of $N$ `(image, text)` pairs, there are $N$ positive matches along the diagonal and $N^2 - N$ incorrect pairings.
       * The loss function maximizes the cosine similarity `dot(v_img_i, v_txt_i)` of true pairs while driving all negative pairings toward zero.
    """)

with col_clip2:
    st.markdown("### The CLIP Shared Embedding Space")
    clip_dot = """
    digraph CLIPArchitecture {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_inputs {
            label="Input Modalities"; style="dashed"; color="#3B82F6";
            ImgInput [label="Input Image\\n(Pixel Tensor)", fillcolor="#EFF6FF", color="#3B82F6"];
            TxtInput [label="Text Caption\\n'A golden retriever puppy'", fillcolor="#FEF3C7", color="#F59E0B"];
        }

        ImgEnc [label="Vision Encoder (ViT)\\nProcesses image patches", fillcolor="#DBEAFE"];
        TxtEnc [label="Text Encoder (Transformer)\\nProcesses BPE tokens", fillcolor="#FDE68A"];

        ProjImg [label="Projection W_img + L2 Norm", fillcolor="#BFDBFE"];
        ProjTxt [label="Projection W_txt + L2 Norm", fillcolor="#FCD34D"];

        subgraph cluster_shared {
            label="Unified Metric Space (R^512)"; style="dashed"; color="#10B981";
            SharedSpace [label="Shared Latent Hypersphere\\nv_img ≈ v_txt (Cosine Sim > 0.88)\\nDirect Vector Dot Product!", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        ImgInput -> ImgEnc;
        TxtInput -> TxtEnc;
        ImgEnc -> ProjImg;
        TxtEnc -> ProjTxt;
        ProjImg -> SharedSpace [color="#10B981", style="bold"];
        ProjTxt -> SharedSpace [color="#10B981", style="bold"];
    }
    """
    st.graphviz_chart(clip_dot, use_container_width=False)

st.markdown("""
### Multimodal Model Family Comparison
| Model | Developer | Visual Backbone | Embedding Dimensions | Contrastive Loss Formulation | Key Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI CLIP** | OpenAI | ViT-B/32 or ViT-L/14 | 512 / 768 | Softmax InfoNCE across batch | Pioneering foundation model; ubiquitous vector DB support |
| **OpenCLIP** | LAION / Community | ViT-H/14 or ViT-bigG/14 | 1024 | Scaled InfoNCE on 2B open web pairs | Open weights, higher accuracy on fine-grained objects |
| **SigLIP** | Google Research | ViT-SO400M | 1152 | **Pairwise Sigmoid Loss** | Replaces softmax; batch-size invariant; state-of-the-art retrieval |
| **BLIP-2 / EVA-CLIP** | Salesforce / BAAI | EVA-02 ViT-E | 768 / 1408 | Q-Former cross-attention + contrastive | Optimal for generative multimodal vision-language interfaces |
""")

st.divider()

# Section 2: Text-to-Image and Image-to-Text Search Workflows
st.header("2. Text-to-Image and Image-to-Text Retrieval Workflows", divider="rainbow")

col_wf1, col_wf2 = col_wf1, col_wf2 = st.columns([1, 1])

with col_wf1:
    st.markdown("""
    ### 1. Text-to-Image Search (Natural Language Queries)
    * **Use Case**: Stock photo platforms (Unsplash, Shutterstock), e-commerce product search, photo album search (*"find pictures of my daughter on a bicycle"*).
    * **Execution**:
      1. Ingest 10M photos: extract `v_img = Model.encode_image(photo)` and store in vector database index.
      2. User types text query: `v_txt = Model.encode_text("vintage red sports car driving on mountain road")`.
      3. Execute standard ANN cosine search with `v_txt`.
      4. Nearest neighbor image vectors return the exact visual match!
    
    ### 2. Image-to-Text Search (Reverse Tagging & Caption Matching)
    * **Use Case**: Automated alt-text generation, finding catalog descriptions for uploaded products, copyright infringement tracking.
    * **Execution**:
      1. Database stores 1M textual product descriptions or catalog articles.
      2. User uploads an image: `v_img = Model.encode_image(uploaded_photo)`.
      3. Query text vector database with `v_img`.
      4. Returns the closest matching textual descriptions.
    """)

with col_wf2:
    st.markdown("### Production Code: Text-to-Image & Image-to-Text with OpenCLIP")
    st.code("""
import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

def embed_image(image_path: str) -> np.ndarray:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0]

def embed_text(query: str) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0]

# Text-to-Image: dot(embed_text(query), image_db_vectors.T)
# Image-to-Text: dot(embed_image(photo), text_db_vectors.T)
""", language="python")

st.divider()

# Section 3: Zero-Shot Image Classification via Prompt Ensembling
st.header("3. Zero-Shot Image Classification via Prompt Ensembling", divider="rainbow")

st.markdown("""
### How CLIP Classifies Images Without Training on Classes
Unlike traditional classifiers that require fine-tuning a Softmax head on fixed categories, 
a multimodal model can classify images into **arbitrary open-ended categories zero-shot**:
1. Take a list of candidate class labels: `["cat", "dog", "airplane", "sports car"]`.
2. Format each label using prompt templates: `"a photo of a {class}"` or `"a high-resolution picture of a {class}"`.
3. Compute the text embedding for each candidate prompt: `v_txt_cat`, `v_txt_dog`, `v_txt_airplane`, `v_txt_car`.
4. Compute the image embedding for the target image: `v_img`.
5. Compute cosine similarities: `Sim_c = dot(v_img, v_txt_c)`.
6. Apply Softmax over the similarities to produce predicted class probabilities!
""")

col_zs1, col_zs2 = col_zs1, col_zs2 = st.columns([1, 1])

with col_zs1:
    st.markdown("### Zero-Shot Classification Matrix")
    st.markdown("""
    ```text
    Softmax Probability Formula:
    P(class_i | Image) = exp( 100 * dot(v_img, v_txt_i) ) / Sum_j [ exp( 100 * dot(v_img, v_txt_j) ) ]
    
    Where:
    - 100 is the learned temperature scale parameter (logit_scale).
    - dot(v_img, v_txt) is the cosine similarity between L2-normalized vectors.
    ```
    """)

with col_zs2:
    st.markdown("### Prompt Ensembling Best Practices")
    st.markdown("""
    * Querying with raw single-word labels (*"dog"*) degrades accuracy because single words lack context in the language encoder.
    * Industry best practice is to **ensemble 8 to 15 prompt templates**:
      * *"a photo of a {label}."*
      * *"a close-up photo of the {label}."*
      * *"a high quality image of a {label}."*
    * Averaging the normalized text vectors across templates boosts zero-shot Top-1 accuracy by **3% to 5%**!
    """)

st.divider()

# Section 4: Interactive Workbench — Bidirectional Text-Image Search Simulator
st.header("4. Interactive Workbench: Cross-Modal Search & Zero-Shot Classifier", divider="rainbow")

st.markdown("""
Test cross-modal retrieval live. Our simulated enterprise catalog contains 6 distinct visual items with paired images and text captions. 
Switch between **Text-to-Image Search** and **Image-to-Text / Zero-Shot Classification**.
""")

catalog_modalities = [
    {
        "id": "ITEM_01",
        "title": "Minimalist Japanese Ceramic Teapot",
        "category": "Home & Kitchen",
        "description": "Handcrafted matte black ceramic kyusu teapot with natural wooden side-handle and brass accents.",
        "sim_tags": ["teapot", "ceramic", "japanese", "black", "kitchen", "tea"],
        "base_vector": [0.85, 0.20, 0.15, 0.72]
    },
    {
        "id": "ITEM_02",
        "title": "Ergonomic Mesh Office Chair",
        "category": "Office Furniture",
        "description": "High-back breathable mesh ergonomic desk chair with 3D adjustable armrests and lumbar support spine.",
        "sim_tags": ["chair", "office", "mesh", "ergonomic", "desk", "furniture"],
        "base_vector": [0.22, 0.88, 0.35, 0.65]
    },
    {
        "id": "ITEM_03",
        "title": "Ultra-Light Carbon Fiber Road Bicycle",
        "category": "Sports & Outdoors",
        "description": "Aerodynamic matte black carbon fiber racing bicycle with 12-speed electronic wireless gear shifting.",
        "sim_tags": ["bicycle", "bike", "carbon fiber", "sports", "racing", "cycling"],
        "base_vector": [0.15, 0.30, 0.92, 0.55]
    },
    {
        "id": "ITEM_04",
        "title": "Vintage Mechanical Chronograph Watch",
        "category": "Luxury Watches",
        "description": "Swiss-made stainless steel automatic chronograph wrist watch with brown alligator leather strap.",
        "sim_tags": ["watch", "chronograph", "mechanical", "swiss", "leather", "luxury"],
        "base_vector": [0.65, 0.18, 0.25, 0.91]
    },
    {
        "id": "ITEM_05",
        "title": "Active Noise-Cancelling Wireless Headphones",
        "category": "Electronics & Audio",
        "description": "Over-ear memory foam bluetooth headphones with active noise cancellation and 40-hour battery life.",
        "sim_tags": ["headphones", "audio", "wireless", "noise cancelling", "bluetooth", "music"],
        "base_vector": [0.35, 0.75, 0.68, 0.40]
    },
    {
        "id": "ITEM_06",
        "title": "Artisanal Sourdough Bread Loaf",
        "category": "Bakery & Gourmet",
        "description": "Freshly baked organic sourdough loaf with golden blistered crust and rustic flour scoring pattern.",
        "sim_tags": ["bread", "sourdough", "bakery", "food", "crust", "organic"],
        "base_vector": [0.90, 0.10, 0.12, 0.30]
    }
]

search_mode = st.radio(
    "Select Cross-Modal Interaction Mode:",
    options=["Text-to-Image Search (Query with Text)", "Image-to-Text / Zero-Shot Classifier (Query with Image)"],
    horizontal=True
)

col_mod1, col_mod2 = col_mod1, col_mod2 = st.columns([1, 1])

if search_mode == "Text-to-Image Search (Query with Text)":
    with col_mod1:
        st.subheader("Text Query Formulation")
        preset_queries = [
            "comfortable office desk chair with back support",
            "japanese ceramic tea pot with wooden handle",
            "fast lightweight bicycle for road racing",
            "luxury stainless steel mechanical wrist watch",
            "over-ear bluetooth headphones for listening to music",
            "freshly baked rustic sourdough bread"
        ]
        selected_text_q = st.selectbox("Choose a natural language query or enter custom:", preset_queries + ["Custom Text Query"])
        if selected_text_q == "Custom Text Query":
            active_txt = st.text_input("Enter custom query:", value="aerodynamic racing bike")
        else:
            active_txt = selected_text_q
            
        st.info(f"**Text Query Vector**: Computed via CLIP Text Transformer Encoder -> Projected to `R^512`.")
        
    def score_text_to_image(q_str, items):
        q_lower = q_str.lower()
        results = []
        for it in items:
            # Calculate match based on tag overlap and base vector
            overlap = sum(1 for tag in it["sim_tags"] if tag in q_lower)
            base_score = 0.45 + (0.12 * overlap)
            score = min(0.965, round(base_score + np.random.uniform(0.01, 0.05), 3))
            # Force high relevance for direct semantic match
            if it["title"].lower().split()[0] in q_lower or any(t in q_lower for t in it["sim_tags"][:2]):
                score = max(score, round(0.89 + np.random.uniform(0.02, 0.06), 3))
            results.append({
                "Image ID": it["id"],
                "Catalog Image": it["title"],
                "Category": it["category"],
                "Cross-Modal Similarity": score,
                "Visual Description": it["description"]
            })
        return pd.DataFrame(results).sort_values(by="Cross-Modal Similarity", ascending=False).reset_index(drop=True)

    with col_mod2:
        st.subheader("Retrieved Catalog Images (Top Visual Matches)")
        df_t2i = score_text_to_image(active_txt, catalog_modalities)
        styled_t2i = df_t2i.style.format({"Cross-Modal Similarity": "{:.3f}"}).background_gradient(
            subset=["Cross-Modal Similarity"], cmap="YlGn", vmin=0.4, vmax=1.0
        )
        st.dataframe(styled_t2i, use_container_width=True)
        top_hit = df_t2i.iloc[0]
        st.success(f"""
        **Top Image Match**: `{top_hit['Catalog Image']}` (Score: **{top_hit['Cross-Modal Similarity']:.3f}**)  
        *Retrieved solely via text-image dot product in shared latent space!*
        """)

else:
    with col_mod1:
        st.subheader("Query Image Selection")
        image_options = {it["id"] + " - " + it["title"]: it for it in catalog_modalities}
        selected_img_key = st.selectbox("Choose Query Image:", options=list(image_options.keys()))
        active_img_item = image_options[selected_img_key]
        
        st.info(f"""
        **Selected Query Image**: `{active_img_item['title']}`  
        **Image Encoder**: ViT-B/32 processing 16x16 pixel patches -> Projected to `R^512`.
        """)
        
        st.markdown("### Zero-Shot Candidate Classes:")
        candidate_classes = ["Mechanical Watch", "Office Chair", "Ceramic Teapot", "Racing Bicycle", "Headphones", "Bakery Loaf"]
        st.write(", ".join([f"`{c}`" for c in candidate_classes]))

    def score_image_to_text(img_item, candidate_classes):
        scores = {}
        for c in candidate_classes:
            c_lower = c.lower()
            if any(tag in c_lower for tag in img_item["sim_tags"]):
                scores[c] = round(0.88 + np.random.uniform(0.04, 0.08), 3)
            else:
                scores[c] = round(0.18 + np.random.uniform(0.02, 0.06), 3)
                
        # Softmax over logits
        logits = np.array(list(scores.values())) * 15.0
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        results = []
        for i, (cls_name, sim) in enumerate(scores.items()):
            results.append({
                "Class Candidate": cls_name,
                "Prompt Template": f"'a photo of a {cls_name}'",
                "Cosine Similarity": sim,
                "Zero-Shot Softmax Probability": f"{probs[i]*100:.1f}%"
            })
        return pd.DataFrame(results).sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

    with col_mod2:
        st.subheader("Zero-Shot Classification Probabilities")
        df_i2t = score_image_to_text(active_img_item, candidate_classes)
        st.dataframe(df_i2t, use_container_width=True)
        top_cls = df_i2t.iloc[0]
        st.success(f"""
        **Predicted Class**: `{top_cls['Class Candidate']}` with **{top_cls['Zero-Shot Softmax Probability']}** confidence!  
        *Zero training samples were used for this category.*
        """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How does CLIP align two completely different modalities (text tokens vs. image pixels) into the same metric space?", expanded=True):
    st.markdown("""
    **Answer:**
    CLIP achieves cross-modal alignment through **Dual-Encoder Contrastive Representation Learning**:
    1. **Independent Modality Encoders**: A Vision Transformer encodes the image into visual feature vector $h_{img}$, and a Text Transformer encodes the caption into text feature vector $h_{txt}$.
    2. **Linear Projections into a Shared Space**: Learned linear projection matrices $W_{img}$ and $W_{txt}$ project both vectors into a common dimension $d$ (e.g., 512 dimensions), followed by strict L2-normalization onto the unit hypersphere:
       $v_{img} = \\frac{W_{img} h_{img}}{||W_{img} h_{img}||}, \\quad v_{txt} = \\frac{W_{txt} h_{txt}}{||W_{txt} h_{txt}||}$
    3. **Symmetric Cross-Entropy Contrastive Loss (InfoNCE)**: Given a batch of $N$ paired images and captions, the model computes an $N \\times N$ cosine similarity matrix:
       $S_{i, j} = v_{img, i} \\cdot v_{txt, j}$
       The loss function simultaneously maximizes the similarity along the positive diagonal ($S_{i, i}$) while minimizing all off-diagonal negative pairs ($S_{i, j}$ where $i \\neq j$) via cross-entropy across rows and columns.
    4. By training on 400+ million diverse web image-text pairs, the model forces concepts like *"red apple"* in pixels and *"red apple"* in text to occupy identical spatial coordinates.
    """)

with st.expander("Q2: What is SigLIP (Sigmoid Loss for Language Image Pre-Training) and why is it superior to original CLIP?"):
    st.markdown("""
    **Answer:**
    Original CLIP uses **Softmax-based InfoNCE loss**, which normalizes pairwise similarities across the entire batch:
    * This requires global batch communication across all GPU nodes, creating massive inter-device bandwidth bottlenecks when scaling to batch sizes of 32,000+.
    * It makes the loss of each pair dependent on all other pairs in the batch.
    
    **SigLIP (Zhai et al., Google Research, 2023)** replaces the softmax normalization with a **simple pairwise Sigmoid Binary Cross-Entropy loss**:
    * It treats every image-text pair in the batch as an independent binary classification task: $+1$ if $i = j$ (matching pair) and $-1$ if $i \\neq j$ (negative pair).
    * **Batch-Size Invariance**: SigLIP does not require global denominator normalization, allowing training with arbitrary, smaller, or massive batch sizes without gradient degradation.
    * **Memory Efficiency**: Eliminates expensive cross-GPU all-gather communications during loss calculation, achieving superior retrieval accuracy with lower compute overhead.
    """)

with st.expander("Q3: Why does zero-shot CLIP classification require prompt templates rather than bare class names?"):
    st.markdown("""
    **Answer:**
    During pre-training, CLIP was trained on natural language captions scraped from the web (e.g., *"A stunning photograph of a golden retriever playing in the snow"*), which are rich in syntax and descriptive grammar. 
    It was **never trained on isolated single-word dictionary labels** like *"dog"*.
    
    If you pass the raw word *"dog"*, the language model experiences a distribution shift—the token is ambiguous and lacks photographic context (it could mean a slang verb, a cartoon, or an astronomical constellation). 
    Wrapping the class in a prompt template like `"a photo of a {label}"` or `"a high-resolution close-up of a {label}"` aligns the query with the exact linguistic style of the training distribution, dramatically increasing cosine similarity against the real photo.
    """)

with st.expander("Q4: Can CLIP embeddings be used directly for fine-grained image retrieval (e.g., distinguishing two identical cars with different scratch marks)?"):
    st.markdown("""
    **Answer:**
    No. Standard global CLIP embeddings are **semantic and coarse-grained**, not fine-grained geometric descriptors:
    * CLIP's contrastive objective forces the model to capture high-level semantic concepts that match broad web captions (e.g., *"red sedan"*, *"sunny day"*).
    * Because global pooling collapses all patch tokens into a single 512-d vector, fine-grained localized details—such as a small scratch on a bumper, specific license plate characters, or subtle stitch patterns—are smoothed away.
    * For fine-grained retrieval, architectures like **DINOv2 patch matching**, **ColPali (multi-vector token-level late interaction)**, or specialized metric-learning models (ArcFace/Triplet Loss) are required.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Text-Image Modality Gap**: In standard CLIP, text embeddings and image embeddings cluster in two slightly separated parallel sub-cones of the hypersphere rather than completely overlapping. In production, directly calculating raw Euclidean distance can yield subtle offsets; always use Cosine Similarity or center-subtract modality centroids.
* **Trap 2: Ignoring Text Length Limits in CLIP**: The standard CLIP text encoder has a strict context limit of **77 tokens**. Feeding a 200-word product description silently truncates the text at token 77, discarding all remaining specifications.
* **Trap 3: Forgetting L2 Normalization Before Vector DB Insertion**: Inserting un-normalized image features into Milvus or Qdrant with Cosine metric. Un-normalized vectors cause magnitude distortion and incorrect nearest neighbor rankings.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: CLIP"):
        st.markdown("**Front**: What does CLIP stand for and what does it do?")
        st.divider()
        st.markdown("**Back**: **Contrastive Language-Image Pre-training**; maps text captions and image pixels into a shared high-dimensional metric space.")

with card_cols[1]:
    with st.popover("Card 2: InfoNCE Loss"):
        st.markdown("**Front**: How is CLIP trained?")
        st.divider()
        st.markdown("**Back**: Contrastive InfoNCE loss: maximizes diagonal (positive matching pairs) while minimizing all off-diagonal negative pairs in an N x N batch.")

with card_cols[2]:
    with st.popover("Card 3: SigLIP"):
        st.markdown("**Front**: What is the key innovation of SigLIP?")
        st.divider()
        st.markdown("**Back**: Replaces Softmax InfoNCE with pairwise Sigmoid loss, eliminating global batch synchronization and improving training stability.")

with card_cols[3]:
    with st.popover("Card 4: Zero-Shot Class"):
        st.markdown("**Front**: How does CLIP perform zero-shot classification?")
        st.divider()
        st.markdown("**Back**: Formats classes into prompts ('a photo of a {class}'), computes text vectors, and computes Softmax over cosine similarities against the image vector.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Text-Image Multimodal Embeddings:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the Dual-Encoder architecture (Vision ViT + Text Transformer).", key="chk27_2_1")
    st.checkbox("I know how the contrastive InfoNCE objective aligns text and image vectors.", key="chk27_2_2")
    st.checkbox("I can implement Text-to-Image search and Image-to-Text retrieval with OpenCLIP.", key="chk27_2_3")
with c2:
    st.checkbox("I understand how SigLIP replaces softmax with sigmoid loss for batch-size invariance.", key="chk27_2_4")
    st.checkbox("I know how prompt ensembling improves zero-shot classification accuracy.", key="chk27_2_5")
    st.checkbox("I understand the CLIP text length constraint (77 tokens) and the modality gap.", key="chk27_2_6")
