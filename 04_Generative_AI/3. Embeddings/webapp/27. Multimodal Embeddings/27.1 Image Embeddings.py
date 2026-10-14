import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="27.1 Image Embeddings - Embeddings Curriculum",
    page_icon="🖼️",
    layout="wide"
)

# Title & Overview
st.title("27.1 Image Embeddings: Image-to-Vector, Visual Similarity, Search & Clustering")
st.caption("Chapter 27: Multimodal Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Just as words and sentences are converted into high-dimensional numerical vectors to capture semantic meaning, 
**images are converted into dense feature vectors** to capture visual concepts—such as shapes, textures, objects, 
color palettes, spatial compositions, and high-level semantics.

Before deep learning, computer vision relied on brittle handcrafted descriptors (SIFT, SURF, Color Histograms). 
Today, **Deep Neural Networks (ResNets, ConvNeXts)** and **Vision Transformers (ViT, DINOv2)** project raw pixels 
into dense embedding vectors (typically 512 to 1024 dimensions) where **Euclidean or Cosine distance directly correlates with visual and conceptual similarity**.

This foundational capability powers reverse image search (e.g., Google Lens), visual duplicate detection in e-commerce, 
automated product recommendation, medical image diagnosis, and unsupervised visual clustering.
""")

st.divider()

# Section 1: Image-to-Vector Architecture
st.header("1. How Images Become Vectors: CNNs vs. Vision Transformers (ViT)", divider="rainbow")

col_arch1, col_arch2 = col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    ### The Image-to-Vector Transformation Pipeline
    An image begins as a 3D tensor of pixel values `(Height x Width x 3 Channels)` with values from 0 to 255. 
    To convert this tensor into a single 1D vector `R^d`:
    
    1. **Convolutional Neural Networks (ResNet, ConvNeXt)**:
       * Raw pixels pass through hierarchical convolutional layers.
       * Early layers detect edges and textures; middle layers detect contours and parts; final layers detect complex objects.
       * **Global Average Pooling (GAP)** collapses the final 2D feature map into a single dense vector (e.g., 2048-dim for ResNet-50).
    2. **Vision Transformers (ViT, DINOv2)**:
       * The image is sliced into a grid of non-overlapping patches (e.g., `16x16` pixels).
       * Each patch is flattened and linearly projected into a token embedding vector.
       * A learnable `[CLS]` (Classification) token is prepended.
       * All patch tokens pass through transformer self-attention layers.
       * The final output state of the `[CLS]` token serves as the **global image embedding**.
    3. **Self-Supervised DINOv2 (State of the Art)**:
       * Trained without text labels using student-teacher self-distillation.
       * Produces universal image embeddings that capture fine-grained geometry and semantic class boundaries simultaneously.
    """)

with col_arch2:
    st.markdown("### Vision Transformer (ViT) Patch Embedding Flow")
    vit_dot = """
    digraph ViTPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawImg [label="Raw Input Image\\n(224 x 224 x 3 RGB)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Patches [label="Patch Extraction\\n196 Patches of 16x16 Pixels", fillcolor="#DBEAFE"];
        
        LinearProj [label="Linear Projection + Position Embed\\n196 Patch Tokens + 1 [CLS] Token", fillcolor="#E0E7FF"];

        TransformerBlocks [label="Transformer Encoder\\n12 Layers of Multi-Head Self-Attention\\nPatches attend to each other", fillcolor="#FEF3C7", color="#F59E0B"];

        CLSToken [label="Extract [CLS] Token Head\\n(Global Semantic Summary)", fillcolor="#FDE68A"];

        L2Norm [label="L2 Normalization Unit\\n||v_img|| = 1.0", fillcolor="#D1FAE5"];
        FinalVec [label="Dense Image Embedding Vector\\nv_img in R^768 (or R^1024)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        RawImg -> Patches;
        Patches -> LinearProj;
        LinearProj -> TransformerBlocks;
        TransformerBlocks -> CLSToken;
        CLSToken -> L2Norm;
        L2Norm -> FinalVec;
    }
    """
    st.graphviz_chart(vit_dot, use_container_width=False)

st.markdown("""
### Vision Backbone Model Comparison for Image Embeddings
| Architecture | Exemplar Model | Embedding Dimensions | Pre-training Objective | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Self-Supervised ViT** | **Meta DINOv2** | 768 (Base) / 1536 (Giant) | Self-distillation (DINO) on 142M curated images | State-of-the-art visual feature representation; captures fine-grained parts without text labels |
| **Vision-Language Dual**| **OpenAI CLIP (ViT-B/32)** | 512 | Cross-modal text-image contrastive loss | Jointly aligned with text; optimal for zero-shot text search |
| **Modern CNN** | **ConvNeXt-V2** | 1024 | Masked Autoencoder (FCMAE) | Extremely fast GPU inference, native variable resolution support |
| **Classic Residual Net**| **ResNet-50** | 2048 | Supervised ImageNet-1k classification | Ubiquitous legacy benchmark; compact CPU footprint |
""")

st.divider()

# Section 2: Visual Similarity & Reverse Image Search Pipeline
st.header("2. Visual Similarity & Reverse Image Search Pipeline", divider="rainbow")

st.markdown("""
### The Reverse Image Search Workflow
In production reverse image search engines (e.g., Pinterest visual search, Amazon product visual match, copyright infringement scanners):
1. **Offline Ingestion**:
   * Millions of catalog images are pre-processed (resized, normalized).
   * Feature vectors are computed via a Vision Transformer and L2-normalized.
   * Vectors are inserted into an Approximate Nearest Neighbor (ANN) index (HNSW or IVFPQ) in a vector database like Milvus, Qdrant, or Pinecone.
2. **Online Querying**:
   * A user uploads an arbitrary query photo taken with their smartphone.
   * The query photo is embedded in sub-15ms using the exact same vision backbone.
   * HNSW index executes cosine similarity search, scanning 10,000,000 vectors in sub-5ms to return top-k visually nearest products.
""")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### Production Code: PyTorch Image Embedding Extractor")
    st.code("""
import torch
import torchvision.transforms as T
from PIL import Image
import timm

# Load pre-trained DINOv2 Vision Transformer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
model = model.to(device).eval()

# Standard ImageNet / DINO preprocessing transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_image_embedding(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass through Vision Transformer
        features = model(tensor) # Shape: (1, 768)
        
        # L2-normalize to unit hypersphere
        norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
        
    return norm_features.cpu().numpy()[0]
""", language="python")

with col_code2:
    st.markdown("### Production Search in Vector Store")
    st.code("""
import chromadb

# Initialize ChromaDB vector client
client = chromadb.PersistentClient(path="./image_catalog_db")
collection = client.get_or_create_collection(
    name="product_images",
    metadata={"hnsw:space": "cosine"}
)

def visual_search(query_image_path: str, top_k: int = 5):
    # 1. Extract normalized embedding vector
    q_vec = extract_image_embedding(query_image_path).tolist()
    
    # 2. Query vector database
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    return results
""", language="python")

st.divider()

# Section 3: Visual Clustering & Duplicate Detection
st.header("3. Unsupervised Visual Clustering & Near-Duplicate Detection", divider="rainbow")

col_clust1, col_clust2 = col_clust1, col_clust2 = st.columns([1, 1])

with col_clust1:
    st.markdown("""
    ### 1. Near-Duplicate Detection
    * **The Problem**: Marketplaces receive millions of duplicate seller listings with slight image alterations (watermarks, crops, minor color grading, compression artifacts).
    * **Dense Vector Thresholding**:
      * Because Vision Transformer embeddings are invariant to mild crops and JPEG compression, two images of the same product maintain:
        `CosineSim(img1, img2) > 0.95`
      * Any listing with cosine similarity above 0.95 is flagged automatically as an exact or near duplicate.
    
    ### 2. Unsupervised Visual Clustering
    * Applying algorithms like **K-Means** or **DBSCAN** directly on image embedding vectors partitions unlabeled image collections into intuitive visual themes:
      * Cluster 1: Red running sneakers & athletic footwear
      * Cluster 2: Leather office handbags & satchels
      * Cluster 3: Modern minimalist wooden coffee tables
    * Enables automated visual catalog organization and auto-tagging without human labeling!
    """)

with col_clust2:
    st.markdown("### Visual Clustering Pipeline")
    cluster_dot = """
    digraph VisualClustering {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawDataset [label="Unlabeled Image Dataset\\n100,000 Catalog Photos", fillcolor="#EFF6FF", color="#3B82F6"];
        
        ViTExtractor [label="Vision Transformer Encoder\\nExtracts 768-dim L2 Vector per Image", fillcolor="#DBEAFE"];

        ClusterAlgo [label="Clustering Engine (K-Means / DBSCAN)\\nGroups by High-Dimensional Density", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        C1 [label="Cluster 1: Footwear\\n(Centroid: Athletics)", fillcolor="#D1FAE5"];
        C2 [label="Cluster 2: Furniture\\n(Centroid: Minimalist)", fillcolor="#FCE7F3"];
        C3 [label="Cluster 3: Electronics\\n(Centroid: Laptops)", fillcolor="#E0E7FF"];

        RawDataset -> ViTExtractor;
        ViTExtractor -> ClusterAlgo;
        ClusterAlgo -> C1;
        ClusterAlgo -> C2;
        ClusterAlgo -> C3;
    }
    """
    st.graphviz_chart(cluster_dot, use_container_width=False)

st.divider()

# Section 4: Interactive Workbench — Image Similarity & Reverse Search Sandbox
st.header("4. Interactive Workbench: Reverse Visual Search & Cluster Simulator", divider="rainbow")

st.markdown("""
Experience how Vision Transformer embeddings power reverse image search and visual clustering. 
Select an uploaded query image from our catalog to see its feature vector and inspect nearest visual neighbors.
""")

catalog_items = [
    {"id": "IMG_01", "name": "Crimson Nike Running Sneaker", "category": "Footwear", "color": "Red/White", "style": "Sport", "vector_norm": [0.82, 0.15, 0.12, 0.95]},
    {"id": "IMG_02", "name": "Scarlet Adidas Marathon Shoe", "category": "Footwear", "color": "Red/Black", "style": "Sport", "vector_norm": [0.80, 0.18, 0.14, 0.92]},
    {"id": "IMG_03", "name": "Burgundy Formal Leather Oxford", "category": "Footwear", "color": "Dark Red/Brown", "style": "Formal", "vector_norm": [0.65, 0.45, 0.20, 0.70]},
    {"id": "IMG_04", "name": "Cerulean Blue Trail Hiking Boot", "category": "Footwear", "color": "Blue/Grey", "style": "Outdoor", "vector_norm": [0.25, 0.78, 0.60, 0.85]},
    {"id": "IMG_05", "name": "Apple MacBook Pro M3 Space Grey", "category": "Electronics", "color": "Grey/Black", "style": "Tech", "vector_norm": [0.10, 0.12, 0.88, 0.20]},
    {"id": "IMG_06", "name": "Dell XPS 15 Silver Laptop", "category": "Electronics", "color": "Silver", "style": "Tech", "vector_norm": [0.12, 0.15, 0.85, 0.22]},
    {"id": "IMG_07", "name": "Scandinavian Oak Dining Table", "category": "Furniture", "color": "Natural Wood", "style": "Home", "vector_norm": [0.15, 0.82, 0.18, 0.35]},
    {"id": "IMG_08", "name": "Walnut Mid-Century Coffee Table", "category": "Furniture", "color": "Dark Wood", "style": "Home", "vector_norm": [0.18, 0.80, 0.22, 0.38]}
]

col_sim_left, col_sim_right = col_sim_left, col_sim_right = st.columns([1, 1])

with col_sim_left:
    st.subheader("Step 1: Select Query Image")
    query_options = {item["id"] + " - " + item["name"]: item for item in catalog_items}
    selected_query_key = st.selectbox("Choose Query Image:", options=list(query_options.keys()))
    query_item = query_options[selected_query_key]
    
    st.info(f"""
    **Query Item**: `{query_item['name']}`  
    **Visual Category**: `{query_item['category']}` | **Dominant Palette**: `{query_item['color']}` | **Style**: `{query_item['style']}`
    """)
    
    st.markdown("### Simulated ViT Feature Vector Head (First 8 Dimensions):")
    # Generate 8 realistic floats based on seed
    np.random.seed(int(query_item["id"].split("_")[1]) * 17)
    sample_vec = np.random.uniform(-0.35, 0.35, 8)
    sample_vec = sample_vec / np.linalg.norm(sample_vec)
    st.code("[" + ", ".join([f"{v:.4f}" for v in sample_vec]) + ", ... 768 total dimensions]", language="python")
    
    st.caption("Extracted via Vision Transformer (DINOv2 / ViT-B/16) and projected onto the unit hypersphere.")

# Compute cosine similarity across catalog
def compute_visual_similarity(q_item, catalog):
    results = []
    q_features = np.array(q_item["vector_norm"])
    q_norm = q_features / np.linalg.norm(q_features)
    
    for item in catalog:
        c_features = np.array(item["vector_norm"])
        c_norm = c_features / np.linalg.norm(c_features)
        sim = np.dot(q_norm, c_norm)
        
        # Perfect match gets 1.0
        if item["id"] == q_item["id"]:
            sim = 1.000
        else:
            sim = round(float(sim), 3)
            
        results.append({
            "ID": item["id"],
            "Product Name": item["name"],
            "Category": item["category"],
            "Visual Similarity": sim,
            "Is Duplicate (>0.95)": "YES (Duplicate)" if sim >= 0.95 and item["id"] != q_item["id"] else ("Exact Target" if sim == 1.0 else "No")
        })
    return pd.DataFrame(results).sort_values(by="Visual Similarity", ascending=False).reset_index(drop=True)

with col_sim_right:
    st.subheader("Step 2: Nearest Visual Neighbors (Cosine Similarity)")
    df_results = compute_visual_similarity(query_item, catalog_items)
    
    styled_results = df_results.style.format({"Visual Similarity": "{:.3f}"}).background_gradient(
        subset=["Visual Similarity"], cmap="YlGn", vmin=0.3, vmax=1.0
    )
    st.dataframe(styled_results, use_container_width=True)
    
    top_neighbor = df_results.iloc[1]
    st.success(f"""
    **Top Retrieved Match**: `{top_neighbor['Product Name']}` (Similarity: **{top_neighbor['Visual Similarity']:.3f}**)  
    Notice how items in the same visual category (*{query_item['category']}*) and color palette (*{query_item['color']}*) naturally rank highest!
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How does a Vision Transformer (ViT) generate a single 1D vector from an image of 224x224 pixels?", expanded=True):
    st.markdown("""
    **Answer:**
    A Vision Transformer converts an image into a single 1D vector through the following sequential pipeline:
    1. **Patch Tokenization**: The 2D image `(224 x 224 x 3)` is sliced into a sequence of $N$ non-overlapping patches of size $P \\times P$ (e.g., $16 \\times 16$ pixels, yielding $N = (224/16)^2 = 196$ patches).
    2. **Linear Embedding**: Each flattened patch is projected via a learned linear matrix into a $D$-dimensional token vector (e.g., $D = 768$).
    3. **Classification Token ([CLS]) & Positional Embeddings**: A special learnable `[CLS]` token is prepended to the sequence of 196 patch tokens, and 1D learnable position embeddings are added to preserve spatial layout.
    4. **Self-Attention Layers**: The sequence of 197 tokens passes through 12+ layers of Multi-Head Self-Attention, allowing every image patch to attend to all other patches across the visual field.
    5. **Pooling**: The output representation corresponding to the `[CLS]` token (or mean pooling across all patch tokens) is extracted as the global image embedding vector and L2-normalized.
    """)

with st.expander("Q2: Why does DINOv2 outperform supervised ImageNet-trained models for image embeddings?"):
    st.markdown("""
    **Answer:**
    * **Supervised ImageNet Models (Class Collapse)**: Models trained with cross-entropy loss on 1,000 discrete classes learn to discard visual features that are irrelevant to class prediction (e.g., background textures, fine-grained object geometry, spatial scale). This causes different items within the same broad class to collapse into the same vector region, hurting fine-grained visual search.
    * **DINOv2 (Self-Supervised Distillation)**: Trained without labels using self-distillation on 142M curated images. By forcing a student network to predict local and global patch representations produced by an exponential moving average teacher network, DINOv2 learns rich, pixel-level semantic geometry. Its embeddings preserve object parts, depth, pose, and texture, making them vastly superior for nearest-neighbor visual search and clustering.
    """)

with st.expander("Q3: What metric and indexing structure should be used to search across 50 million image vectors in real-time?"):
    st.markdown("""
    **Answer:**
    * **Distance Metric**: **Cosine Similarity** or **Inner Product (IP)** over strictly L2-normalized vectors (`||v|| = 1.0`). Normalization guarantees that dot product mathematically equals cosine similarity, which executes in nanoseconds via SIMD hardware instructions.
    * **Indexing Structure**: An Approximate Nearest Neighbor (ANN) index using **HNSW (Hierarchical Navigable Small World)** or **IVF-PQ (Inverted File with Product Quantization)**:
      * For lowest latency (<5ms) and highest recall (>98%): **HNSW** with `M=32, efConstruction=200`.
      * For memory-constrained scale (50M 768-d vectors in RAM): **IVF-PQ** or **Scalar Quantization (SQ8)** to compress each 768-d vector from 3,072 bytes (float32) down to 96 bytes, reducing RAM consumption from 150 GB down to under 5 GB.
    """)

with st.expander("Q4: How do you detect visually near-duplicate images with crops or watermarks?"):
    st.markdown("""
    **Answer:**
    Near-duplicate detection uses dense vision embeddings rather than cryptographic hashes (MD5/SHA256, which fail on a single modified pixel):
    1. Pass images through a robust Vision Transformer (e.g., DINOv2 or CLIP).
    2. Because self-attention attends to the dominant global object rather than peripheral pixels, a cropped, compressed, or watermarked duplicate produces a cosine similarity **> 0.94 to 0.97** against the original.
    3. In production, an automated pipeline flags any candidate pair with `CosineSim >= 0.95` as a near-duplicate, optionally falling back to local patch matching (RANSAC or geometric verification) for final legal confirmation.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Searching Without L2 Normalization**: Storing raw un-normalized image features and using Euclidean distance. Features from images with high brightness or complex backgrounds will have larger vector norms, skewing distance calculations. Always L2-normalize.
* **Trap 2: Ignoring Aspect Ratio Distortions During Preprocessing**: Forcibly resizing a tall banner image (800x200) into a square (224x224) distorts aspect ratios, stretching visual features and degrading similarity precision. Use center cropping with padding.
* **Trap 3: Using Supervised Classification Models for Open-Domain Retrieval**: Using ResNet-50 trained strictly on ImageNet-1k to search industrial medical parts. The model tries to map x-rays and machine components to dogs and cars, producing meaningless embeddings.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: ViT [CLS] Token"):
        st.markdown("**Front**: What is the purpose of the [CLS] token in Vision Transformers?")
        st.divider()
        st.markdown("**Back**: It serves as a global sequence aggregator that attends to all image patches, producing the final global image embedding vector.")

with card_cols[1]:
    with st.popover("Card 2: DINOv2"):
        st.markdown("**Front**: Why is DINOv2 widely used for image embeddings?")
        st.divider()
        st.markdown("**Back**: It is self-supervised (no labels needed), capturing fine-grained visual geometry, textures, and object semantics without class collapse.")

with card_cols[2]:
    with st.popover("Card 3: Duplicate Threshold"):
        st.markdown("**Front**: What cosine similarity threshold is typically used for near-duplicate images?")
        st.divider()
        st.markdown("**Back**: `CosineSim >= 0.95` on normalized ViT embeddings reliably identifies crops, compression, and watermarked copies.")

with card_cols[3]:
    with st.popover("Card 4: Visual Clustering"):
        st.markdown("**Front**: How does K-Means cluster images using embeddings?")
        st.divider()
        st.markdown("**Back**: It computes centroids directly in 768-d vector space, grouping images with similar visual features without human annotations.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Image Embeddings:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand how Vision Transformers (ViT) convert 2D image patches into a 1D vector.", key="chk27_1_1")
    st.checkbox("I know why self-supervised models (DINOv2) outperform supervised ImageNet models for embeddings.", key="chk27_1_2")
    st.checkbox("I can implement an image embedding extraction pipeline with PyTorch and timm.", key="chk27_1_3")
with c2:
    st.checkbox("I understand the production reverse image search architecture with HNSW vector stores.", key="chk27_1_4")
    st.checkbox("I know how cosine thresholding detects near-duplicate images with crops or watermarks.", key="chk27_1_5")
    st.checkbox("I can apply unsupervised visual clustering (K-Means/DBSCAN) on image vectors.", key="chk27_1_6")
