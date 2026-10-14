import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="28.1 Recommendation Basics - Embeddings Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("28.1 Recommendation Basics: User Embeddings, Item Embeddings & Two-Tower Candidate Generation")
st.caption("Chapter 28: Embeddings for Recommendations | Domain: 3. Vector Embeddings")
st.markdown("""
Modern recommendation systems (YouTube, Netflix, Spotify, Amazon, TikTok) operate on an unprecedented scale: 
they must match **hundreds of millions of active users** against **billions of candidate items** in less than 50 milliseconds.

Traditional collaborative filtering (e.g., matrix factorization or user-item co-occurrence matrices) suffers from severe 
cold-start problems, high memory footprint, and inability to incorporate rich real-time context.
**Embedding-based Recommender Systems** solve this by representing both **Users** and **Items** as dense vectors 
in the same metric space:
* **Item Embeddings**: Capture content attributes, genres, textual descriptions, tags, and historical engagement patterns.
* **User Embeddings**: Capture user demographics, long-term preferences, and real-time session interaction history.
* **Similarity Matching**: The predicted affinity between a user and an item is simply their vector dot product: 
  `Score(u, i) = dot(v_user, v_item)`.

By indexing all item vectors in an Approximate Nearest Neighbor (ANN) vector database, **Candidate Generation** reduces 
a billion-item catalog down to the top 100 high-affinity candidates in sub-5ms!
""")

st.divider()

# Section 1: The Two-Tower Neural Recommender Architecture
st.header("1. The Two-Tower Neural Network Architecture", divider="rainbow")

col_two1, col_two2 = col_two1, col_two2 = st.columns([1, 1])

with col_two1:
    st.markdown("""
    ### How Two-Tower Recommenders Work
    Popularized by Google (YouTube DNN) and Pinterest (PinSage), the **Two-Tower Neural Network** splits the system into two independent models:
    
    1. **The User Tower (Query Tower) `U(x_user)`**:
       * Takes dynamic user inputs: User ID, age, country, device, search history, and the sequence of last 10 clicked item IDs.
       * Runs through dense neural layers to output a normalized 128-d or 256-d vector: `v_user`.
       * Computed **dynamically at request time** (or streamed via feature stores).
    2. **The Item Tower (Candidate Tower) `I(x_item)`**:
       * Takes static item features: Item ID, category, textual description, tags, pricing, and visual thumbnail embeddings.
       * Outputs a normalized vector: `v_item`.
       * Pre-computed **offline in batch** and indexed in a vector database.
    3. **Training Objective (Cross-Entropy with Sampled Negatives)**:
       * When a user clicks or purchases item $i$, `(v_user, v_item)` is a positive pair.
       * All other items in the batch are treated as in-batch negative pairs.
       * Loss maximizes `dot(v_user, v_pos)` while minimizing `dot(v_user, v_neg)`.
    """)

with col_two2:
    st.markdown("### Two-Tower Architecture Diagram")
    two_tower_dot = """
    digraph TwoTower {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_user_tower {
            label="User Tower (Query Tower)"; style="dashed"; color="#3B82F6";
            UserFeatures [label="User Features\\nUser ID, Recent Clicks, Device, Locale", fillcolor="#EFF6FF", color="#3B82F6"];
            UserMLP [label="User Deep Neural Net (MLP)\\nProcesses Sequence & Context", fillcolor="#DBEAFE"];
            UserVec [label="User Embedding Vector\\nv_user in R^128 (L2-norm)", fillcolor="#BFDBFE", color="#1D4ED8", style="filled,bold"];
            UserFeatures -> UserMLP -> UserVec;
        }

        subgraph cluster_item_tower {
            label="Item Tower (Candidate Tower)"; style="dashed"; color="#10B981";
            ItemFeatures [label="Item Features\\nItem ID, Category, Text, Tags, Visuals", fillcolor="#ECFDF5", color="#10B981"];
            ItemMLP [label="Item Deep Neural Net (MLP)\\nProcesses Catalog Attributes", fillcolor="#D1FAE5"];
            ItemVec [label="Item Embedding Vector\\nv_item in R^128 (L2-norm)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            ItemFeatures -> ItemMLP -> ItemVec;
        }

        DotProduct [label="Dot Product Scoring\\nScore = dot(v_user, v_item)\\nHNSW ANN Search (5ms)", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];
        TopCandidates [label="Top 100 Candidates\\nHigh-Recall Funnel Output", fillcolor="#FDE68A"];

        UserVec -> DotProduct;
        ItemVec -> DotProduct;
        DotProduct -> TopCandidates;
    }
    """
    st.graphviz_chart(two_tower_dot, use_container_width=False)

st.markdown("""
### Two-Tower vs. Cross-Attention Recommender Comparison
| Dimension | Stage 1: Two-Tower (Candidate Generation) | Stage 2: Deep Ranking (Cross-Attention / DLRM) |
| :--- | :--- | :--- |
| **Primary Goal** | **Broad Recall**: Filter 10,000,000 items -> 100 candidates | **Precision**: Rank 100 candidates -> Top 10 display slots |
| **Interaction Point** | Vectors interact ONLY via final dot product | Features interact across every layer of deep neural network |
| **Latency SLA** | **Sub-5 milliseconds** (via HNSW index) | 20 - 40 milliseconds |
| **Item Pre-computation**| **Yes**: All item vectors pre-indexed in vector DB | **No**: Computes live pairwise forward passes per candidate |
| **Catalog Scope** | Entire enterprise catalog (millions to billions) | Small candidate pool (50 to 500 items) |
""")

st.divider()

# Section 2: Building User Embeddings from Interaction History
st.header("2. Constructing Dynamic User Embeddings", divider="rainbow")

col_usr1, col_usr2 = col_usr1, col_usr2 = st.columns([1, 1])

with col_usr1:
    st.markdown("""
    ### Methods to Compute User Embeddings
    How does a platform know what a user likes without manual profile questionnaires?
    
    1. **Mean Pooling with Exponential Time Decay**:
       * The user's representation is the weighted average of the embeddings of the last $k$ items they engaged with:
         `v_user = Sum_{i=1}^k [ w_i * v_item_i ]`
       * Where `w_i = exp(-lambda * delta_t)` gives higher weight to items clicked in the last 10 minutes over items clicked 3 weeks ago!
    2. **Sequential Attention Models (SASRec, BERT4Rec)**:
       * The sequence of clicked item embeddings `[v_1, v_2, ..., v_t]` is fed into a causal transformer.
       * Self-attention learns transition dynamics: buying a camera predicts buying a tripod and memory card next!
    3. **Two-Tower Neural Inference**:
       * The User Tower processes both static demographic features and the pooled interaction vector to predict immediate intent.
    """)

with col_usr2:
    st.markdown("### Production Code: Time-Decayed User Vector Aggregator")
    st.code("""
import numpy as np
import time

def compute_user_embedding(clicked_items: list[dict], decay_half_life_hours: float = 24.0) -> np.ndarray:
    \"\"\"Aggregates recent item vectors into a dynamic user embedding.\"\"\"
    now = time.time()
    half_life_seconds = decay_half_life_hours * 3600.0
    
    weighted_vectors = []
    weights = []
    
    for item in clicked_items:
        # Calculate time delta in seconds
        delta_t = now - item['timestamp']
        # Exponential decay: weight = 0.5 ** (delta_t / half_life)
        weight = 0.5 ** (delta_t / half_life_seconds)
        
        vec = np.array(item['embedding'], dtype=np.float32)
        weighted_vectors.append(vec * weight)
        weights.append(weight)
        
    if not weighted_vectors:
        return np.zeros(128, dtype=np.float32)
        
    # Weighted average
    user_vec = np.sum(weighted_vectors, axis=0) / np.sum(weights)
    
    # L2-normalize onto unit hypersphere
    return user_vec / np.linalg.norm(user_vec)
""", language="python")

st.divider()

# Section 3: Candidate Generation & Retrieval at Scale
st.header("3. Scaling Candidate Generation to 100 Million Items", divider="rainbow")

st.markdown("""
### Why Vector Databases Power Modern RecSys
In classic recommendations, evaluating every candidate required calculating dot products across the entire database. 
For 100 million items, $O(N)$ dot products take hundreds of milliseconds on CPU clusters.

By indexing item vectors in an **Approximate Nearest Neighbor (ANN) index** (e.g., Milvus, Qdrant, Vespa, or Pinecone):
1. **Sub-linear Search Complexity**: HNSW explores small-world graph highways to find top-k nearest items in O(log N) time.
2. **Metadata Filtering**: Easily restrict candidates to in-stock items, items shipping to user's country, or items the user hasn't already purchased:
   `filter={"status": "in_stock", "region": "US"}`
3. **Partitioned Routing**: High-traffic catalogs partition items into category clusters (e.g., Electronics, Fashion, Books), querying only the relevant partitions based on user segment.
""")

st.divider()

# Section 4: Interactive Workbench — Two-Tower Recommender Simulator
st.header("4. Interactive Workbench: Two-Tower Candidate Generation Simulator", divider="rainbow")

st.markdown("""
Explore how dynamic user interaction history shapes user embeddings and retrieves personalized candidate items live!
Choose a pre-configured user profile or customize recent clicks to see the resulting user vector and candidate retrieval ranking.
""")

catalog_rec_items = [
    {"id": "TECH_01", "name": "Kubernetes Production Best Practices Guide", "genre": "DevOps", "vector": [0.88, 0.12, 0.15, 0.92]},
    {"id": "TECH_02", "name": "Terraform Infrastructure as Code Mastery", "genre": "DevOps", "vector": [0.85, 0.18, 0.20, 0.89]},
    {"id": "TECH_03", "name": "Docker Container Security & Hardening", "genre": "DevOps", "vector": [0.82, 0.22, 0.18, 0.91]},
    {"id": "DATA_01", "name": "Deep Learning with PyTorch & Transformers", "genre": "AI/ML", "vector": [0.15, 0.92, 0.35, 0.85]},
    {"id": "DATA_02", "name": "Vector Databases & Semantic Search Systems", "genre": "AI/ML", "vector": [0.35, 0.88, 0.40, 0.90]},
    {"id": "DATA_03", "name": "Computer Vision with DINOv2 & Vision Transformers", "genre": "AI/ML", "vector": [0.12, 0.95, 0.25, 0.80]},
    {"id": "WEB_01", "name": "Next.js 15 Full-Stack Server Components", "genre": "Web Dev", "vector": [0.20, 0.25, 0.92, 0.75]},
    {"id": "WEB_02", "name": "TypeScript Advanced Type System Architecture", "genre": "Web Dev", "vector": [0.25, 0.30, 0.89, 0.78]},
    {"id": "WEB_03", "name": "Tailwind CSS & Modern Micro-Frontends", "genre": "Web Dev", "vector": [0.18, 0.20, 0.95, 0.70]},
    {"id": "EXEC_01", "name": "SaaS Unit Economics & Enterprise Pricing Strategy", "genre": "Business", "vector": [0.35, 0.40, 0.30, 0.25]}
]

user_profiles = {
    "Profile 1: Cloud & DevOps Architect": {
        "recent_clicks": ["TECH_01", "TECH_02", "TECH_03"],
        "description": "User frequently reads Kubernetes and Terraform deployment guides."
    },
    "Profile 2: AI / Machine Learning Researcher": {
        "recent_clicks": ["DATA_01", "DATA_02", "DATA_03"],
        "description": "User studies deep learning architectures, PyTorch, and vector search engines."
    },
    "Profile 3: Modern Web & Frontend Specialist": {
        "recent_clicks": ["WEB_01", "WEB_02", "WEB_03"],
        "description": "User builds high-performance user interfaces with Next.js and TypeScript."
    },
    "Profile 4: Hybrid Engineer (DevOps + Vector Search)": {
        "recent_clicks": ["TECH_01", "DATA_02"],
        "description": "User is building Kubernetes infrastructure for vector database deployments."
    }
}

selected_profile_key = st.selectbox("Choose a User Profile:", options=list(user_profiles.keys()))
profile_data = user_profiles[selected_profile_key]

col_rec_left, col_rec_right = col_rec_left, col_rec_right = st.columns([1, 1])

# Calculate dynamic user vector
def compute_simulated_user_vector(clicked_ids, catalog):
    vectors = []
    for cid in clicked_ids:
        item = next(it for it in catalog if it["id"] == cid)
        vectors.append(item["vector"])
    mean_vec = np.mean(vectors, axis=0)
    return mean_vec / np.linalg.norm(mean_vec)

user_vector = compute_simulated_user_vector(profile_data["recent_clicks"], catalog_rec_items)

with col_rec_left:
    st.subheader("Dynamic User Profile & Interaction State")
    st.info(f"**Selected Persona**: `{selected_profile_key}`  \n**Behavioral Pattern**: {profile_data['description']}")
    
    st.markdown("**Recent Clicked Items (Input to User Tower):**")
    for cid in profile_data["recent_clicks"]:
        item = next(it for it in catalog_rec_items if it["id"] == cid)
        st.markdown(f"• `{item['id']}` - **{item['name']}** *({item['genre']})*")
        
    st.markdown("### Synthesized User Vector (Normalized 4-d projection):")
    st.code("[" + ", ".join([f"{v:.4f}" for v in user_vector]) + "]", language="python")

# Compute recommendation affinities
def rank_candidates(u_vec, catalog, clicked_ids):
    results = []
    for item in catalog:
        i_vec = np.array(item["vector"])
        i_norm = i_vec / np.linalg.norm(i_vec)
        score = round(float(np.dot(u_vec, i_norm)), 3)
        
        is_clicked = item["id"] in clicked_ids
        
        results.append({
            "Item ID": item["id"],
            "Title": item["name"],
            "Genre": item["genre"],
            "Affinity Score": score,
            "History Status": "Already Engaged" if is_clicked else "Unseen Candidate"
        })
    return pd.DataFrame(results).sort_values(by="Affinity Score", ascending=False).reset_index(drop=True)

with col_rec_right:
    st.subheader("Top Candidate Generation (Dot Product ANN Output)")
    df_ranked = rank_candidates(user_vector, catalog_rec_items, profile_data["recent_clicks"])
    
    styled_df = df_ranked.style.format({"Affinity Score": "{:.3f}"}).background_gradient(
        subset=["Affinity Score"], cmap="YlGn", vmin=0.3, vmax=1.0
    )
    st.dataframe(styled_df, use_container_width=True)
    
    unseen_top = df_ranked[df_ranked["History Status"] == "Unseen Candidate"].iloc[0]
    st.success(f"""
    **Top Recommended Unseen Item**: `{unseen_top['Title']}` *(Affinity: {unseen_top['Affinity Score']:.3f})*  
    *Automatically surfaced by vector dot product matching in shared metric space!*
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What is the Two-Tower Architecture in Recommender Systems, and why is it preferred for Candidate Generation?", expanded=True):
    st.markdown("""
    **Answer:**
    The **Two-Tower Architecture** separates representation learning into two decoupled sub-networks:
    1. **User Tower**: Ingests user context, history, and real-time signals to output a single embedding vector $v_{user} \\in \\mathbb{R}^d$.
    2. **Item Tower**: Ingests catalog attributes, tags, text, and visual features to output item vector $v_{item} \\in \\mathbb{R}^d$.
    
    **Why it is preferred for Candidate Generation**:
    * **Decoupled Pre-computation**: All item vectors can be computed offline in batch and indexed inside an Approximate Nearest Neighbor (ANN) index (e.g., HNSW).
    * **Blazing Fast Request-Time Inference**: At request time, the system only needs to run a single forward pass through the User Tower to obtain $v_{user}$, and then executes an ANN search over billions of items in sub-5ms!
    * Complex cross-attention networks (which evaluate user and item together) cannot pre-compute representations and are far too slow for million-item catalogs.
    """)

with st.expander("Q2: How do Two-Tower models handle the 'Cold-Start' problem for new items and new users?"):
    st.markdown("""
    **Answer:**
    * **New Items (Item Cold-Start)**: Because the Item Tower ingests content features (title, description embeddings, category tags, images) rather than relying solely on historical collaborative interaction matrix IDs, a brand new item immediately receives a high-quality embedding vector the second it is published! It can be retrieved immediately based on semantic content similarity.
    * **New Users (User Cold-Start)**: When a user has zero click history, the User Tower relies on initial contextual features (sign-up demographics, country, device, referral source, onboarding topic choices) to construct a baseline user vector. As soon as the user makes their first click, the user vector dynamically updates via time-decayed streaming aggregation.
    """)

with st.expander("Q3: What loss function is typically used to train Two-Tower Recommenders?"):
    st.markdown("""
    **Answer:**
    Two-Tower recommenders are typically trained using **Sampled Softmax with In-Batch Negatives** (a variant of InfoNCE contrastive loss):
    ```text
    Loss = -log [ exp(dot(v_u, v_pos) / tau) / Sum_j exp(dot(v_u, v_j) / tau) ]
    ```
    Where:
    * `dot(v_u, v_pos)` is the predicted score of the positive item clicked by user $u$.
    * The denominator sums over all items present in the training batch across all parallel users (in-batch negatives).
    * `tau` is a temperature parameter that controls score sharpness.
    * To correct for popularity bias (where frequently clicked items appear as false negatives too often), industry systems apply **LogQ correction** (subtracting $\\log P_i$ from logits).
    """)

with st.expander("Q4: Why can't Candidate Generation directly return the final Top 10 recommendations to the user?"):
    st.markdown("""
    **Answer:**
    Candidate Generation is optimized solely for **high recall** (narrowing 100M items to 100 candidates), not precision or business constraints:
    1. **Lack of Fine-Grained Feature Crosses**: Two-tower models only interact via a single dot product, missing non-linear interactions (e.g., *"user only buys this brand on weekends during discount campaigns"*).
    2. **Diversity and Freshness**: Pure dot product retrieval leads to echo chambers—retrieving 10 identical variants of the same product.
    3. **Business Guardrails**: Real systems require subsequent **Ranking and Re-ranking stages** to enforce deduplication, inventory availability checks, profitability margins, and legal age restrictions.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Stale User Vectors**: Updating user embeddings only in daily batch jobs. If a user spends 15 minutes browsing baby strollers, a batch-updated system keeps recommending mechanical tools until tomorrow. Modern RecSys updates user vectors in real-time streaming pipelines.
* **Trap 2: Popularity Collapse**: Without popularity correction (LogQ) during in-batch negative training, the model over-indexes on global blockbuster items, recommending viral pop songs to heavy metal enthusiasts.
* **Trap 3: Forgetting Deduplication & History Filtering**: Failing to filter out items the user already purchased, wasting premium screen real estate showing a user the laptop they bought yesterday.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Two-Tower Model"):
        st.markdown("**Front**: What are the two towers in a Two-Tower recommender?")
        st.divider()
        st.markdown("**Back**: The **User Tower** (computes user/query vector dynamically) and the **Item Tower** (computes item vectors offline).")

with card_cols[1]:
    with st.popover("Card 2: Candidate Generation"):
        st.markdown("**Front**: What is the primary metric of Candidate Generation?")
        st.divider()
        st.markdown("**Back**: **High Recall** — distilling millions of items into 50-100 high-affinity candidates in sub-5ms.")

with card_cols[2]:
    with st.popover("Card 3: Cold-Start"):
        st.markdown("**Front**: How do embedding models solve the item cold-start problem?")
        st.divider()
        st.markdown("**Back**: Content features (text, tags, visual thumbnails) allow the item tower to compute an accurate vector before anyone has clicked it.")

with card_cols[3]:
    with st.popover("Card 4: Time Decay"):
        st.markdown("**Front**: Why apply exponential time decay to user clicks?")
        st.divider()
        st.markdown("**Back**: To prioritize real-time in-session intent over older historical clicks from weeks ago.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Recommendation Basics:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the Two-Tower neural architecture (User Tower vs. Item Tower).", key="chk28_1_1")
    st.checkbox("I know how vector dot product scores user-item affinity.", key="chk28_1_2")
    st.checkbox("I can implement dynamic time-decayed user vector aggregation.", key="chk28_1_3")
with c2:
    st.checkbox("I understand how ANN indices scale Candidate Generation to millions of items.", key="chk28_1_4")
    st.checkbox("I know how content embeddings solve item cold-start.", key="chk28_1_5")
    st.checkbox("I understand why subsequent Ranking & Diversification stages are mandatory.", key="chk28_1_6")
