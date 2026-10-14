import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="28.3 Recommendation Pipeline - Embeddings Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("28.3 Recommendation Pipeline: Ingestion, Retrieval, Deep Ranking & Business Serving")
st.caption("Chapter 28: Embeddings for Recommendations | Domain: 3. Vector Embeddings")
st.markdown("""
In commercial web-scale applications (e.g., Netflix home screen, Amazon recommendations, TikTok FYP), 
recommending content is not a simple single-formula search call. 
Instead, it is an **orchestrated multi-stage machine learning pipeline** operating under strict sub-50ms latency budgets.

The end-to-end production pipeline consists of four distinct engineering stages:
1. **Offline Ingestion & Item Embedding Generation**: Continuously transforming catalog items into high-dimensional vectors and updating the vector database.
2. **Online User Embedding Generation**: Dynamically aggregating real-time session events into an active user vector in sub-5ms.
3. **High-Recall Candidate Retrieval**: Executing Approximate Nearest Neighbor (ANN) search across millions of items to extract the top 100 to 500 candidates.
4. **Deep Ranking, Filtering & Business Serving**: Re-scoring candidates with a heavy Deep Learning Recommendation Model (DLRM), applying business rules (inventory checks, frequency capping, sponsored boost), and serving the final 10 personalized recommendations.
""")

st.divider()

# Section 1: The 4-Stage Production RecSys Funnel
st.header("1. The 4-Stage Production Recommendation Funnel", divider="rainbow")

col_fun1, col_fun2 = col_fun1, col_fun2 = st.columns([1, 1])

with col_fun1:
    st.markdown("""
    ### Architectural Stages & SLA Budgets
    * **Stage 1: Candidate Generation (Vector ANN Retrieval)**:
      * *Scope*: 10,000,000+ items -> 200 candidates.
      * *Latency*: **< 5 ms**.
      * *Mechanism*: Dot product between real-time user vector and pre-indexed item vectors in HNSW vector database.
    * **Stage 2: Scoring / Deep Ranking (Heavy Cross-Attention / DLRM)**:
      * *Scope*: 200 candidates -> 50 top items.
      * *Latency*: **15 - 25 ms**.
      * *Mechanism*: Deep Neural Network (e.g., DLRM, DeepFM, or Cross-Encoder) evaluating 100+ fine-grained features (user-item cross features, context, hour of day).
    * **Stage 3: Re-ranking & Business Optimization**:
      * *Scope*: 50 items -> 20 items.
      * *Latency*: **3 - 5 ms**.
      * *Mechanism*: Diversity injection (MMR), freshness re-weighting, sponsored item interleaving, profit margin optimization.
    * **Stage 4: Post-Processing & Presentation**:
      * *Scope*: 20 items -> Final 10 UI slots.
      * *Latency*: **< 2 ms**.
      * *Mechanism*: Deduplication (filtering items already purchased or seen in current session), frequency capping, and localization formatting.
    """)

with col_fun2:
    st.markdown("### The Production Recommendation Funnel")
    funnel_dot = """
    digraph RecSysFunnel {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Catalog [label="Full Enterprise Catalog\\n10,000,000 Items", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Stage1 [label="Stage 1: Candidate Retrieval (ANN)\\nTwo-Tower Vector Search\\nFilter: 10M -> 200 items (3ms)", fillcolor="#DBEAFE"];

        Stage2 [label="Stage 2: Deep Ranking (DLRM)\\nPredicts P(Click) & P(Conversion)\\nFilter: 200 -> 50 items (20ms)", fillcolor="#FEF3C7", color="#F59E0B"];

        Stage3 [label="Stage 3: Re-ranking & Business Logic\\nMMR Diversity + Margin Boost\\nFilter: 50 -> 20 items (3ms)", fillcolor="#FDE68A"];

        Stage4 [label="Stage 4: Post-Processing & Dedup\\nFrequency Capping + History Purge\\nOutput: Top 10 UI Slots (1ms)", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        UI [label="User Mobile App / Web Carousel\\nTotal End-to-End Latency: 28ms", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        Catalog -> Stage1;
        Stage1 -> Stage2;
        Stage2 -> Stage3;
        Stage3 -> Stage4;
        Stage4 -> UI;
    }
    """
    st.graphviz_chart(funnel_dot, use_container_width=False)

st.divider()

# Section 2: Real-Time Streaming Architecture & Feature Stores
st.header("2. Real-Time Streaming Architecture & Feature Stores", divider="rainbow")

st.markdown("""
To support dynamic recommendations where a user's latest click immediately influences what they see on the next page view, 
modern tech platforms deploy an event-driven streaming infrastructure:
""")

col_feat1, col_feat2 = col_feat1, col_feat2 = st.columns([1, 1])

with col_feat1:
    st.markdown("""
    ### Key Infrastructure Components
    1. **Event Streaming (Apache Kafka / AWS Kinesis)**:
       * Captures raw user actions (`click`, `scroll`, `add_to_cart`, `purchase`) as JSON events with microsecond timestamps.
    2. **Stream Processing (Apache Flink / Spark Streaming)**:
       * Maintains a sliding session window (e.g., last 15 minutes of clicks).
       * Computes dynamic running aggregations (recent category affinity, time-decayed vector centroids).
    3. **Real-Time Feature Store (Redis / Feast / Hopsworks)**:
       * Stores real-time user feature vectors and historical aggregates with sub-2ms key-value read latency.
    4. **Vector Database (Milvus / Qdrant / Vespa)**:
       * Maintains the HNSW index of all pre-computed item vectors, updated continually as new products enter the catalog.
    """)

with col_feat2:
    st.markdown("### Production Streaming RecSys Architecture")
    stream_dot = """
    digraph StreamingArchitecture {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserDevice [label="User App\\nGenerates Clicks", fillcolor="#EFF6FF", color="#3B82F6"];
        Kafka [label="Kafka Event Bus\\nTopic: user_clicks", fillcolor="#FED7AA", color="#F97316"];
        Flink [label="Apache Flink\\nAggregates Session Vectors", fillcolor="#FEE2E2"];
        Feast [label="Feature Store (Redis)\\nStores User Vector\\n(Read SLA: 1ms)", fillcolor="#DBEAFE", color="#1D4ED8"];
        VectorDB [label="Vector DB (Milvus)\\nCatalog Item Embeddings\\n(HNSW Search: 3ms)", fillcolor="#D1FAE5", color="#10B981"];
        RecService [label="RecSys Microservice\\nCombines & Ranks", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserDevice -> Kafka;
        Kafka -> Flink;
        Flink -> Feast;
        UserDevice -> RecService [label="GET /recommendations"];
        RecService -> Feast [label="Fetch user vector"];
        RecService -> VectorDB [label="ANN Search"];
        VectorDB -> RecService;
        RecService -> UserDevice [label="Return Top 10 JSON"];
    }
    """
    st.graphviz_chart(stream_dot, use_container_width=False)

st.divider()

# Section 3: Deep Ranking, Business Rules & Serving Guardrails
st.header("3. Deep Ranking & Business Serving Guardrails", divider="rainbow")

col_rk1, col_rk2 = col_rk1, col_rk2 = st.columns([1, 1])

with col_rk1:
    st.markdown("""
    ### Why Vector Similarity Alone Is Not Enough
    Vector similarity tells you that Item A is semantically related to User U. 
    However, commercial success requires optimizing for **Expected Business Value**:
    
    ```text
    Utility_Score = P(Click) * P(Purchase | Click) * Profit_Margin - Return_Risk_Penalty
    ```
    
    ### Mandatory Business Serving Filters:
    * **Inventory & Geolocation Filter**: Reject items that are out of stock or cannot be delivered to the user's postal code.
    * **Frequency Capping**: If a user has seen Item X three times in the last 48 hours without clicking, suppress it for 7 days.
    * **Recent Purchase Suppression**: Immediately purge items the user bought in the last 30 days (unless consumable like groceries).
    * **Category Diversity Constraint**: Ensure no single category (e.g., *Headphones*) occupies more than 40% of the carousel slots.
    * **Sponsored Boosting**: Boost paid merchant listings by a controlled multiplier while preserving relevance.
    """)

with col_rk2:
    st.markdown("### Production Code: Multi-Objective Ranking & Filtering Engine")
    st.code("""
def post_process_recommendations(
    candidates: list[dict],
    user_purchase_history: set[str],
    user_impressions: dict[str, int],
    max_slots: int = 5,
    max_per_category: int = 2
) -> list[dict]:
    \"\"\"Applies production business filters and multi-objective scoring.\"\"\"
    final_recs = []
    category_counts = {}
    
    for item in candidates:
        item_id = item['id']
        category = item['category']
        
        # 1. Filter out recently purchased items
        if item_id in user_purchase_history:
            continue
            
        # 2. Frequency capping: suppress if seen >= 3 times without click
        if user_impressions.get(item_id, 0) >= 3:
            continue
            
        # 3. Category diversity guardrail
        if category_counts.get(category, 0) >= max_per_category:
            continue
            
        # 4. Compute composite commercial score
        # Base vector affinity + Margin boost + Sponsor weight
        commercial_score = (
            item['vector_affinity'] * 0.5 + 
            item['margin_score'] * 0.3 + 
            item.get('sponsor_boost', 1.0) * 0.2
        )
        item['final_score'] = round(commercial_score, 3)
        
        final_recs.append(item)
        category_counts[category] = category_counts.get(category, 0) + 1
        
        if len(final_recs) == max_slots:
            break
            
    return final_recs
""", language="python")

st.divider()

# Section 4: Interactive Workbench — End-to-End Pipeline Simulator
st.header("4. Interactive Workbench: Complete Recommendation Pipeline Stepper", divider="rainbow")

st.markdown("""
Step through the entire **5-stage production recommendation lifecycle** live. 
Select a user profile, trigger candidate retrieval, inspect the DLRM scoring stage, watch business rules filter 
out ineligible items, and observe the final personalized recommendations presented to the user.
""")

pipeline_users = {
    "User A: Premium Tech Professional": {
        "user_id": "USR_9921",
        "purchased_ids": ["PROD_MONITOR_01"],
        "impressions_history": {"PROD_MOUSE_01": 4, "PROD_KB_01": 1},
        "session_intent": "Home Office Productivity & Ergonomics",
        "user_vector": [0.82, 0.25, 0.18, 0.90]
    },
    "User B: Gaming & Audio Enthusiast": {
        "user_id": "USR_4412",
        "purchased_ids": ["PROD_AUDIO_02"],
        "impressions_history": {"PROD_MONITOR_02": 3},
        "session_intent": "High-Performance Gaming & Spatial Sound",
        "user_vector": [0.15, 0.88, 0.70, 0.40]
    }
}

catalog_candidates = [
    {"id": "PROD_KB_01", "name": "Keychron Q1 Pro Wireless Mechanical Keyboard", "category": "Keyboards", "vector": [0.85, 0.22, 0.20, 0.88], "stock": 45, "margin": 0.85, "sponsor": 1.1},
    {"id": "PROD_KB_02", "name": "Logitech MX Mechanical Mini Minimalist Keyboard", "category": "Keyboards", "vector": [0.80, 0.20, 0.18, 0.92], "stock": 12, "margin": 0.70, "sponsor": 1.0},
    {"id": "PROD_KB_03", "name": "Ergodox EZ Custom Split Ergonomic Keyboard", "category": "Keyboards", "vector": [0.78, 0.15, 0.12, 0.95], "stock": 0, "margin": 0.90, "sponsor": 1.0}, # Out of stock!
    {"id": "PROD_MOUSE_01", "name": "Logitech MX Master 3S Wireless Mouse", "category": "Mice", "vector": [0.83, 0.28, 0.15, 0.91], "stock": 80, "margin": 0.75, "sponsor": 1.0}, # Frequency capped for User A!
    {"id": "PROD_MOUSE_02", "name": "Razer DeathAdder V3 Pro Esports Gaming Mouse", "category": "Mice", "vector": [0.20, 0.92, 0.65, 0.45], "stock": 50, "margin": 0.80, "sponsor": 1.2},
    {"id": "PROD_MONITOR_01", "name": "Dell UltraSharp 34-inch Curved USB-C Hub Monitor", "category": "Monitors", "vector": [0.88, 0.18, 0.25, 0.85], "stock": 10, "margin": 0.65, "sponsor": 1.0}, # Already purchased by User A!
    {"id": "PROD_AUDIO_01", "name": "Sony WH-1000XM5 Noise Cancelling Headphones", "category": "Audio", "vector": [0.35, 0.85, 0.75, 0.50], "stock": 25, "margin": 0.82, "sponsor": 1.15},
    {"id": "PROD_AUDIO_02", "name": "Sennheiser HD 660S2 Open-Back Audiophile Headphones", "category": "Audio", "vector": [0.10, 0.95, 0.60, 0.35], "stock": 5, "margin": 0.90, "sponsor": 1.0}
]

selected_u_key = st.selectbox("Select Active User Context:", options=list(pipeline_users.keys()))
active_u = pipeline_users[selected_u_key]

col_pipe_a, col_pipe_b = col_pipe_a, col_pipe_b = st.columns([1, 1])

# Stage 1: Vector Affinity calculation
def run_stage1_retrieval(u_vec, catalog):
    res = []
    for it in catalog:
        it_v = np.array(it["vector"])
        norm_v = it_v / np.linalg.norm(it_v)
        sim = float(np.dot(u_vec, norm_v))
        res.append({
            **it,
            "vector_affinity": round(sim, 3)
        })
    return sorted(res, key=lambda x: x["vector_affinity"], reverse=True)

stage1_candidates = run_stage1_retrieval(active_u["user_vector"], catalog_candidates)

with col_pipe_a:
    st.subheader("Stage 1 & 2: User Ingestion & Candidate Retrieval")
    st.info(f"""
    **User ID**: `{active_u['user_id']}`  
    **Detected Session Intent**: *"{active_u['session_intent']}"*  
    **Past Purchases**: `{', '.join(active_u['purchased_ids'])}`  
    **Impression Counts**: `{active_u['impressions_history']}`
    """)
    
    st.markdown("### Top Stage 1 Vector Candidates (Before Business Filtering):")
    st.dataframe(pd.DataFrame([
        {"Item": c["name"], "Category": c["category"], "Vector Sim": f"{c['vector_affinity']:.3f}", "Stock": c["stock"]}
        for c in stage1_candidates[:6]
    ]), use_container_width=True, hide_index=True)

# Stage 3 & 4: Deep Business Rules & Post-Processing
def run_stage3_business_rules(candidates, user_data):
    final_list = []
    audit_log = []
    category_counts = {}
    
    for it in candidates:
        iid = it["id"]
        cat = it["category"]
        
        # Check stock
        if it["stock"] <= 0:
            audit_log.append({"Item": it["name"], "Status": "REJECTED", "Reason": "Out of Stock (stock = 0)"})
            continue
            
        # Check already purchased
        if iid in user_data["purchased_ids"]:
            audit_log.append({"Item": it["name"], "Status": "REJECTED", "Reason": "Already Purchased by User"})
            continue
            
        # Check frequency capping (>= 3 impressions)
        if user_data["impressions_history"].get(iid, 0) >= 3:
            audit_log.append({"Item": it["name"], "Status": "REJECTED", "Reason": f"Frequency Capped ({user_data['impressions_history'][iid]} prior views without click)"})
            continue
            
        # Check category limit (max 2 per category)
        if category_counts.get(cat, 0) >= 2:
            audit_log.append({"Item": it["name"], "Status": "REJECTED", "Reason": f"Category Limit Exceeded (Max 2 for {cat})"})
            continue
            
        # Compute composite score
        score = it["vector_affinity"] * 0.5 + it["margin"] * 0.3 + (it["sponsor"] - 1.0) * 0.5
        audit_log.append({"Item": it["name"], "Status": "APPROVED", "Reason": f"Score: {score:.3f} (Ranked)"})
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        final_list.append({
            **it,
            "final_score": round(score, 3)
        })
        
    final_list = sorted(final_list, key=lambda x: x["final_score"], reverse=True)
    return final_list, audit_log

final_recs, audit_decisions = run_stage3_business_rules(stage1_candidates, active_u)

with col_pipe_b:
    st.subheader("Stage 3 & 4: Business Rules & Final Serving Carousel")
    
    st.markdown("### Business Rule Audit Decisions:")
    st.dataframe(pd.DataFrame(audit_decisions), use_container_width=True, hide_index=True)
    
    st.markdown("### Final Delivered Recommendations (Top UI Slots):")
    for rank, item in enumerate(final_recs, 1):
        st.success(f"""
        **#{rank}: {item['name']}**  
        *Category*: `{item['category']}` | *Vector Affinity*: `{item['vector_affinity']:.3f}` | *Profit Margin*: `{item['margin'] * 100:.0f}%` | **Final Commercial Score**: `{item['final_score']:.3f}`
        """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What is the end-to-end latency budget of an enterprise recommendation pipeline, and how is it allocated?", expanded=True):
    st.markdown("""
    **Answer:**
    In high-traffic consumer web applications (Amazon, Netflix, YouTube), the total latency budget for the recommendation API is typically **under 50 milliseconds**:
    1. **Network & Ingress Gateway (2 - 5 ms)**: TLS termination, authentication, and routing.
    2. **User Feature Fetching (2 - 4 ms)**: Retrieving real-time session features and the user vector from an in-memory cache / feature store (Redis).
    3. **Candidate Generation via Vector ANN (3 - 6 ms)**: Querying an HNSW vector database (Milvus/Qdrant) across 10M+ items to retrieve the top 200 candidates.
    4. **Deep Ranking Model (DLRM / Heavy MLP) (15 - 25 ms)**: Evaluating cross-features and computing P(Click) / P(Purchase) on the 200 candidates using a GPU inference cluster (Triton).
    5. **Business Rules, Diversification & Serving (3 - 5 ms)**: MMR diversity, frequency capping, stock availability pruning, and JSON response assembly.
    """)

with st.expander("Q2: Why do production systems decouple Candidate Generation from Deep Ranking instead of using a single model?"):
    st.markdown("""
    **Answer:**
    Decoupling Candidate Generation from Deep Ranking solves the fundamental **Accuracy vs. Computational Complexity trade-off**:
    * **Candidate Generation (Broad Recall)**: Evaluates millions of items. To meet sub-5ms SLAs, it *must* use decoupled Two-Tower vector representations where item vectors are pre-computed offline and similarity is computed via dot product over an HNSW index. However, dot products miss complex non-linear feature interactions (e.g., *"user only clicks high-margin items on weekends when using iOS"*).
    * **Deep Ranking (Fine-Grained Precision)**: Evaluates complex cross-features (user historical conversion rate on this specific merchant, price sensitivity, exact category match) through deep neural networks (DLRM, DeepFM, Transformer Cross-Attention). Because these cross-features require live joint forward passes, the model is computationally expensive ($O(K)$) and can only execute on a small candidate pool of 100 to 500 items.
    """)

with st.expander("Q3: How does a Feature Store bridge the gap between offline training and real-time inference?"):
    st.markdown("""
    **Answer:**
    A **Feature Store** (e.g., Feast, Hopsworks, Tecton) provides a unified interface for feature engineering across two storage layers:
    1. **Offline Store (Parquet / Snowflake / BigQuery)**: Stores historical feature snapshots used for training dataset creation, guaranteeing time-travel point-in-time correctness without future data leakage.
    2. **Online Store (Redis / DynamoDB)**: Stores the latest real-time values of user and item features (e.g., *"number of purchases in last 24h"* or *"latest session vector"*), delivering sub-2ms key-value lookups for live inference.
    * This prevents **Training-Serving Skew**—the common production bug where features are calculated with slightly different definitions in SQL during training versus in Python during live serving.
    """)

with st.expander("Q4: What is Frequency Capping and why is it essential in recommendation serving?"):
    st.markdown("""
    **Answer:**
    **Frequency Capping** is an essential business guardrail that limits the number of times a specific item or ad is shown to a user within a specified time window:
    * Without frequency capping, a recommender system with high vector affinity toward an unclicked item will repeatedly place that item in prime screen real estate across every single page refresh.
    * This creates user fatigue, wastes valuable impression slots, and damages customer engagement.
    * In production, the serving layer queries user impression logs: if an item has been shown $N$ times (typically 3 to 5 impressions) without user engagement, it is suppressed from recommendations for a cooldown period (e.g., 7 days).
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Serving Out-of-Stock Items**: Recommending a product with a 0.98 similarity score that is out of stock in the user's regional fulfillment center, resulting in high click rates followed by immediate purchase drop-off.
* **Trap 2: Training-Serving Feature Skew**: Calculating the user's average purchase value with a 30-day window during training, but using a 7-day window in the live feature store, causing the ranking model's weights to output invalid probabilities.
* **Trap 3: Omitting Re-ranking Diversity**: Displaying 5 nearly identical versions of the same mechanical keyboard, boring the user and missing opportunities to cross-sell related categories (wrist rests, keycaps, desk mats).
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: 4-Stage Funnel"):
        st.markdown("**Front**: What are the 4 stages of the recommendation funnel?")
        st.divider()
        st.markdown("**Back**: 1. Candidate Retrieval (ANN), 2. Deep Ranking (DLRM), 3. Re-ranking & Business Rules, 4. Post-processing & Serving.")

with card_cols[1]:
    with st.popover("Card 2: Feature Store"):
        st.markdown("**Front**: What problem does a Feature Store solve?")
        st.divider()
        st.markdown("**Back**: Prevents training-serving skew by providing a unified definition for offline training datasets and low-latency online inference lookups.")

with card_cols[2]:
    with st.popover("Card 3: Frequency Capping"):
        st.markdown("**Front**: What is Frequency Capping?")
        st.divider()
        st.markdown("**Back**: Suppressing items that have been displayed multiple times (e.g., 3+ times) without user engagement to prevent user fatigue.")

with card_cols[3]:
    with st.popover("Card 4: Latency SLA"):
        st.markdown("**Front**: What is the standard end-to-end latency SLA for recommendations?")
        st.divider()
        st.markdown("**Back**: Sub-50 milliseconds total, with Candidate Generation taking under 5ms and Deep Ranking taking 15-25ms.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Production Recommendation Pipelines:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4-stage recommendation funnel and latency budget allocation.", key="chk28_3_1")
    st.checkbox("I know how streaming pipelines (Kafka + Flink + Redis) update user vectors in real-time.", key="chk28_3_2")
    st.checkbox("I can implement business rule filters (inventory, frequency capping, history suppression).", key="chk28_3_3")
with c2:
    st.checkbox("I know the role of a Feature Store in preventing training-serving skew.", key="chk28_3_4")
    st.checkbox("I understand why candidate generation is decoupled from deep ranking.", key="chk28_3_5")
    st.checkbox("I can formulate multi-objective commercial scoring combining relevance and margin.", key="chk28_3_6")
