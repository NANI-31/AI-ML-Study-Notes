import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="1.3 Real-World Applications - Embeddings Curriculum",
    page_icon="🌍",
    layout="wide"
)

# Title & Overview
st.title("1.3 Real-World Applications: Search Engines, Enterprise Chatbots, Fraud Detection, & Multimodal Systems")
st.caption("Chapter 1: Introduction to Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Vector embeddings are no longer confined to academic NLP research—they are the **operational foundation of the world's most scalable production systems**.

From Google's search algorithms and Spotify's recommendation engines to credit card fraud detection and cross-modal image search, 
vector representations power mission-critical enterprise workflows:
1. **Next-Generation Search Engines**: Connecting user search queries to billions of web documents via hybrid dense-sparse retrieval.
2. **Enterprise Chatbots & QA**: Grounding conversational agents with accurate, real-time factual knowledge via RAG.
3. **Dual-Encoder Recommendation Engines**: Matching user preferences with vast catalog items at sub-10ms latency.
4. **Behavioral Anomaly & Fraud Detection**: Projecting financial transactions into latent space; transactions distant from a user's normal geometric cluster are flagged as fraudulent.
5. **Cross-Modal Image & Video Retrieval**: Mapping text descriptions and visual pixels into a single unified geometric space (e.g., CLIP / SigLIP).

This lesson explores production case studies, two-tower architectural patterns, and latent anomaly detection.
""")

st.divider()

# Section 1: Real-World Architectural Case Studies
st.header("1. Core Enterprise Production Architectural Patterns", divider="rainbow")

col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("""
    ### Three Dominant Architectural Paradigms:
    * **1. Two-Tower Recommendation Architecture (YouTube / Spotify)**:
      * *User Tower*: Encodes user watch history, demographics, and active time into vector u.
      * *Item Tower*: Encodes video metadata, transcripts, and tags into vector v.
      * *Inference*: Similarity score is computed as dot(u, v) over millions of items via ANN indexing in under 5ms.
    * **2. Transaction Latent Anomaly Detection (Visa / Mastercard)**:
      * Normal user purchasing habits form a tight cluster around a centroid vector.
      * High-risk transactions (e.g., rapid overseas jewelry purchases) land far from the normal cluster centroid (z-score > 3.0), triggering automated card locks.
    * **3. Joint Multimodal Embedding Space (CLIP)**:
      * Text encoder and Image encoder are trained contrastively to project into the exact same vector space.
    """)

with col_app2:
    st.markdown("### Two-Tower Production Architecture")
    rec_dot = """
    digraph TwoTower {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_user {
            label="User Side"; style="dashed"; color="#3B82F6";
            UserFeatures [label="User Profile & History", fillcolor="#EFF6FF"];
            UserTower [label="User Tower (Neural Network)", fillcolor="#DBEAFE", style="filled,bold"];
            UserVec [label="User Vector (128-dim)", fillcolor="#93C5FD", style="filled,bold"];
            UserFeatures -> UserTower -> UserVec;
        }

        subgraph cluster_item {
            label="Item Catalog Side"; style="dashed"; color="#10B981";
            ItemFeatures [label="Catalog Item Metadata", fillcolor="#ECFDF5"];
            ItemTower [label="Item Tower (Neural Network)", fillcolor="#D1FAE5", style="filled,bold"];
            ItemVec [label="Item Vector (128-dim)", fillcolor="#6EE7B7", style="filled,bold"];
            ItemFeatures -> ItemTower -> ItemVec;
        }

        ANN [label="Vector Dot-Product ANN Matcher\\nTop-100 Recommendations (< 5ms)", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        UserVec -> ANN;
        ItemVec -> ANN;
    }
    """
    st.graphviz_chart(rec_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Application Engine in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates two-tower vector matching, transaction outlier anomaly detection, and cross-modal simulated retrieval.
""")

class ProductionApplicationEngine:
    """
    Demonstrates real-world embedding use cases:
    Two-tower recommendation, fraud/anomaly detection, and cross-modal search.
    """
    def __init__(self):
        # Baseline normal transaction behavior centroid (6 dimensions)
        self.normal_user_centroid = np.array([0.45, 0.50, 0.12, 0.88, 0.35, 0.20])
        self.normal_user_centroid /= np.linalg.norm(self.normal_user_centroid)

    def detect_fraud_anomaly(self, transaction_vec: np.ndarray, threshold_distance: float = 0.65) -> Tuple[bool, float]:
        """
        Calculates cosine distance from normal transaction behavior centroid.
        Cosine Distance = 1.0 - Cosine Similarity
        """
        t_norm = transaction_vec / np.linalg.norm(transaction_vec)
        cos_sim = float(np.dot(self.normal_user_centroid, t_norm))
        cos_dist = 1.0 - cos_sim
        is_fraud = cos_dist >= threshold_distance
        return is_fraud, round(cos_dist, 4)

    @staticmethod
    def two_tower_recommend(user_vector: np.ndarray, catalog_items: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Ranks catalog items by dot product with user representation vector."""
        u_norm = user_vector / np.linalg.norm(user_vector)
        scored = []
        for item in catalog_items:
            v_norm = item["vector"] / np.linalg.norm(item["vector"])
            score = float(np.dot(u_norm, v_norm))
            scored.append({**item, "match_score": round(score, 4)})
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        return scored[:top_k]

code_snippet = r'''
import numpy as np

class AnomalyEmbeddingDetector:
    def __init__(self, user_centroid: np.ndarray):
        self.centroid = user_centroid / np.linalg.norm(user_centroid)

    def is_anomaly(self, txn_vector: np.ndarray, distance_threshold: float = 0.60) -> bool:
        v_norm = txn_vector / np.linalg.norm(txn_vector)
        cosine_distance = 1.0 - float(np.dot(self.centroid, v_norm))
        return cosine_distance > distance_threshold
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Real-World Simulation Workbench
st.header("3. Interactive Real-World Simulation Workbench", divider="rainbow")
st.markdown("""
Select a real-world enterprise domain and test embedding behavior: **Behavioral Fraud Detection** or **E-Commerce Two-Tower Recommendation**.
""")

app_engine = ProductionApplicationEngine()

domain_tab1, domain_tab2 = st.tabs(["1. Financial Fraud & Anomaly Detection", "2. E-Commerce Two-Tower Recommendation"])

with domain_tab1:
    st.markdown("### Transaction Vector Outlier Detection")
    st.markdown("""
    The user's historical transactions (grocery shopping, local utility bills, digital subscriptions) form a tight baseline centroid. 
    Incoming transactions are embedded into behavioral vectors and compared against this profile.
    """)

    sample_txns = {
        "Transaction A: Weekend Grocery Supermarket ($68, Local Store)": np.array([0.46, 0.49, 0.14, 0.86, 0.33, 0.22]),
        "Transaction B: Monthly Internet Utility Bill ($55, Recurring Online)": np.array([0.42, 0.52, 0.10, 0.90, 0.36, 0.18]),
        "Transaction C: Coffee Shop Downtown ($6.50, Contactless Terminal)": np.array([0.48, 0.47, 0.15, 0.84, 0.32, 0.25]),
        "Transaction D (Suspicious): Luxury Watch Store ($4,800, Foreign Terminal)": np.array([-0.30, -0.70, 0.65, 0.10, -0.45, 0.80]),
        "Transaction E (Suspicious): High-Risk Cryptocurrency Exchange Transfer ($2,500)": np.array([0.05, -0.85, 0.80, -0.20, 0.70, -0.50])
    }

    selected_txn = st.selectbox("Select incoming transaction to evaluate:", list(sample_txns.keys()))
    txn_vector = sample_txns[selected_txn]

    col_an1, col_an2 = st.columns([1, 1])
    with col_an1:
        fraud_threshold = st.slider("Anomaly Cosine Distance Threshold:", min_value=0.20, max_value=0.90, value=0.60, step=0.05)

    is_anomaly, dist_score = app_engine.detect_fraud_anomaly(txn_vector, threshold_distance=fraud_threshold)

    st.markdown("##### Security Evaluation Trace:")
    col_met1, col_met2 = st.columns([1, 1])
    with col_met1:
        st.metric("Cosine Distance from User Centroid", f"{dist_score:.4f}", delta=f"{'OUTLIER' if is_anomaly else 'NORMAL'}", delta_color="inverse" if is_anomaly else "normal")
    with col_met2:
        if is_anomaly:
            st.error(f"🚨 FRAUD ALERT TRIGGERED: Transaction distance ({dist_score:.4f}) exceeds threshold ({fraud_threshold:.2f}). Transaction paused for 2FA challenge.")
        else:
            st.success(f"✅ TRANSACTION APPROVED: Behavior conforms to historical profile (Distance: {dist_score:.4f} <= {fraud_threshold:.2f}).")

with domain_tab2:
    st.markdown("### Two-Tower Personalized Item Recommendation")
    
    user_personas = {
        "Persona 1: Tech Enthusiast & Developer (Loves GPUs, AI, Cloud Infra)": np.array([0.90, 0.85, 0.10, 0.15]),
        "Persona 2: Outdoor Adventurer & Hiker (Loves Camping, Backpacks, Trail Gear)": np.array([0.10, 0.15, 0.92, 0.88]),
        "Persona 3: Executive Business Analyst (Loves Finance, Laptops, Productivity)": np.array([0.75, 0.25, 0.20, 0.80])
    }

    catalog = [
        {"name": "NVIDIA RTX 4090 GPU", "category": "Hardware", "vector": np.array([0.92, 0.88, 0.08, 0.10])},
        {"name": "Ultra-lightweight 2-Person Camping Tent", "category": "Outdoor", "vector": np.array([0.08, 0.12, 0.95, 0.85])},
        {"name": "Mechanical Keyboard & Ergonomic Mouse", "category": "Accessories", "vector": np.array([0.80, 0.70, 0.15, 0.30])},
        {"name": "Waterproof Gore-Tex Hiking Boots", "category": "Footwear", "vector": np.array([0.12, 0.10, 0.89, 0.92])},
        {"name": "Executive Leather Laptop Briefcase", "category": "Luggage", "vector": np.array([0.65, 0.30, 0.25, 0.85])}
    ]

    chosen_user = st.selectbox("Select user profile for personalized recommendation:", list(user_personas.keys()))
    u_vector = user_personas[chosen_user]

    top_recs = app_engine.two_tower_recommend(u_vector, catalog, top_k=3)

    st.markdown("##### Top Personalized Recommendations (Generated in < 2ms):")
    rec_df = pd.DataFrame([
        {"Rank": i+1, "Product": r["name"], "Category": r["category"], "Cosine Match Score": f"{r['match_score']:.4f}"}
        for i, r in enumerate(top_recs)
    ])
    st.dataframe(rec_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Production Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do Two-Tower models decouple query computation from catalog indexing at scale?**
    * **Answer**:
      * In a two-tower neural network, the **Query Tower** and the **Item Tower** are independent sub-networks that produce embeddings in the same dimensional space.
      * **Asynchronous Indexing**: The item tower embeds the catalog of 100M items *offline* once per day, storing them in a distributed vector index.
      * **Real-Time Serving**: At query time, only the lightweight query tower runs online to produce the single user vector u.
      * Sub-10ms retrieval is achieved by querying the pre-built index via ANN dot products, avoiding the need to run heavy neural network passes over catalog items during user requests.

    **Q2: How does Latent Space Anomaly Detection differ from traditional rule-based fraud detection?**
    * **Answer**:
      * **Rule-Based Systems**: Use static boolean thresholds (`amount > $1,000 AND country != 'US'`). Sophisticated fraudsters easily evade fixed rules by structuring transactions at $995.
      * **Latent Space Embeddings**: Encode multi-dimensional behavior (merchant category, time of day, geolocation delta, device fingerprint, transaction velocity) into a single continuous vector.
      * A transaction of $50 can be flagged as fraudulent if its latent vector is geometrically anomalous relative to the user's habitual pattern.

    **Q3: How does CLIP align text and visual modalities in a shared embedding space?**
    * **Answer**:
      * CLIP (Contrastive Language-Image Pretraining) trains an image encoder (ViT) and a text encoder (Transformer) simultaneously on hundreds of millions of (image, caption) pairs.
      * It uses **InfoNCE contrastive loss**: pulling the image vector and corresponding caption vector close together (maximizing cosine similarity) while pushing apart unrelated image-text pairs in the batch.
      * This allows text queries like `"a photo of a Golden Retriever running in snow"` to directly retrieve matching images via vector similarity.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Concept Drift in User Profiles**:
      * *Issue*: Retaining a static user embedding calculated six months ago.
      * *Consequence*: User recommendations stay frozen on past seasonal interests (e.g. holiday gifts) rather than current intent.
      * *Fix*: Implement exponential decay weighting on historical events to continuously update user profile embeddings.

    * **Trap 2: Ignoring Cold-Start Latency for Heavy Query Towers**:
      * *Issue*: Designing a complex 12-layer transformer for the real-time query tower.
      * *Consequence*: Real-time inference latency exceeds the 50ms SLA, causing mobile client timeouts.
      * *Fix*: Keep query towers lightweight (e.g. 2-4 layers or distilled models) and quantize weights with TensorRT/ONNX.

    * **Trap 3: Domain Vocabulary Mismatch in Pre-Trained Embeddings**:
      * *Issue*: Using general web embeddings (e.g. standard MiniLM) on specialized biomedical or legal discovery corpora.
      * *Consequence*: The model fails to recognize critical medical relationships (e.g. disease-drug interactions).
      * *Fix*: Fine-tune embedding encoders on domain-specific contrastive pairs (e.g. BioBERT / PubMedBERT).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Two-Tower Architecture**
      * *Definition*: A dual-encoder neural framework where query and candidate entities are encoded by separate networks into a shared vector space for fast ANN matching.
    * **Flashcard 2: Contrastive Learning (InfoNCE)**
      * *Definition*: A training objective that maximizes the cosine similarity of positive pairs while minimizing the similarity of negative pairs in vector space.
    * **Flashcard 3: Latent Anomaly Detection**
      * *Definition*: Detecting abnormal events or fraud by measuring the geometric distance between a candidate vector and normal behavioral cluster centroids.
    * **Flashcard 4: Cross-Modal Alignment**
      * *Definition*: Projecting different data modalities (e.g. images, audio, text) into a single unified embedding space so items across modalities can be compared directly.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Real-World Embedding Applications:

- [ ] I can describe the Two-Tower architectural pattern used by YouTube and Spotify.
- [ ] I understand how latent space distance detects financial fraud and behavioral anomalies.
- [ ] I know how contrastive learning (CLIP) maps images and text into a shared geometric space.
- [ ] I can explain why offline item embedding + online query embedding satisfies sub-10ms SLAs.
- [ ] I have completed Chapter 1: Introduction to Embeddings!
""")
