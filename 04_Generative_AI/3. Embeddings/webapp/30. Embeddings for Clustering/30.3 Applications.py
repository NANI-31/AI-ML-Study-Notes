import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="30.3 Applications of Semantic Clustering - Embeddings Curriculum",
    page_icon="🏢",
    layout="wide"
)

# Title & Overview
st.title("30.3 Applications: Customer Segmentation, Document Grouping, News Aggregation & Topic Discovery")
st.caption("Chapter 30: Embeddings for Clustering | Domain: 3. Vector Embeddings")
st.markdown("""
Unsupervised semantic clustering is one of the highest-ROI capabilities in modern enterprise artificial intelligence. 
While supervised classification requires thousands of expensive, human-annotated training samples, **semantic clustering operates on raw, unlabeled text streams** 
to surface latent patterns, emerging risks, and market intelligence:

* **Customer Segmentation**: Discovering behavioral personas and feature desire clusters from open-ended NPS surveys and feedback forms.
* **Document Grouping**: Automatically organizing chaotic multi-terabyte corporate drives and PDF repositories into intuitive semantic folders.
* **News & Event Grouping**: Clustering thousands of breaking news articles from hundreds of publishers into single, cohesive event stories (e.g., Google News).
* **Support-Ticket Incident Grouping**: Grouping live support tickets in real-time to detect zero-day system outages hours before monitoring dashboards trigger.
* **Topic Discovery (BERTopic)**: Extracting structured topic models, keyword hierarchies, and topic evolution over time without brittle Dirichlet priors (LDA).
""")

st.divider()

# Section 1: The Five Core Application Blueprints
st.header("1. Production Blueprints for the Five Core Applications", divider="rainbow")

col_app1, col_app2 = col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("""
    ### 1. Customer Feedback Segmentation
    * **Objective**: Group open-ended customer feedback into actionable product personas.
    * **Workflow**: Customer feedback embeddings are clustered into themes:
      * Cluster 1: *"Performance & Load Time Complaints"*
      * Cluster 2: *"Mobile App UX & Navigation Confusion"*
      * Cluster 3: *"Enterprise SSO & Security Requests"*
    * **Action**: Automatically routed to respective Product Managers.
    
    ### 2. Enterprise Document Grouping
    * **Objective**: Ingest uncataloged file shares (contracts, engineering specs, whitepapers) and construct a navigable virtual directory hierarchy.
    * **Workflow**: Uses Hierarchical Agglomerative clustering to build a tree of folders and sub-folders based on semantic similarity.
    
    ### 3. Real-Time News Event Grouping (Google News Paradigm)
    * **Objective**: Ingest 50,000 articles per hour from 500 news outlets and group all articles reporting on the *exact same real-world event* into a single story card.
    * **Workflow**: Combines high cosine similarity threshold (`> 0.88`) with a tight 24-hour temporal sliding window to merge coverage across outlets.
    """)

with col_app2:
    st.markdown("""
    ### 4. Support-Ticket Incident Clustering
    * **Objective**: Detect emerging production outages before automated infrastructure alerts fire.
    * **Workflow**: As users submit tickets, embeddings are streamed into a rolling HDBSCAN density clustering pipeline.
    * If a new cluster forms with > 15 tickets within 10 minutes (e.g., all mentioning checkout errors), an automated **Sev-1 Incident Bridge** is paged!
    
    ### 5. BERTopic Topic Discovery Engine
    * **The Modern Replacement for LDA**:
      1. **Embeddings**: Sentence Transformers extract semantic vectors.
      2. **Dimensionality Reduction**: UMAP reduces dimensions to 5-d.
      3. **Clustering**: HDBSCAN finds dense topic clusters and isolates noise.
      4. **c-TF-IDF**: Extracts the most discriminative words per cluster.
      5. **LLM Representation**: Prompt an LLM to generate a natural topic summary.
    """)

st.markdown("### The Modern BERTopic Architecture")
bertopic_dot = """
digraph BERTopicWorkflow {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    RawDocs [label="Raw Unlabeled Documents\\n(100,000 texts)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
    
    Embedder [label="1. Embedding Model\\n(BGE / SBERT)\\n1024-dim dense vectors", fillcolor="#DBEAFE"];

    UMAP [label="2. UMAP Projection\\nReduces 1024-d -> 5-d\\nPreserves local topology", fillcolor="#FEF3C7", color="#F59E0B"];

    HDBSCAN [label="3. HDBSCAN Clustering\\nIdentifies dense clusters\\nFilters outlier noise (-1)", fillcolor="#FDE68A"];

    cTFIDF [label="4. c-TF-IDF Engine\\nExtracts top keywords\\nper cluster", fillcolor="#D1FAE5", color="#10B981"];

    LLMTopic [label="5. LLM Topic Synthesis\\nOutputs human titles & summaries", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

    RawDocs -> Embedder;
    Embedder -> UMAP;
    UMAP -> HDBSCAN;
    HDBSCAN -> cTFIDF;
    cTFIDF -> LLMTopic;
}
"""
st.graphviz_chart(bertopic_dot, use_container_width=False)

st.divider()

# Section 2: Production Code: The BERTopic Topic Discovery Pipeline
st.header("2. Code Implementation: The Production BERTopic Pipeline", divider="rainbow")

st.code("""
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# 1. Step 1: Pre-trained Sentence Transformer for dense semantics
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 2. Step 2: UMAP for non-linear dimensionality reduction
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# 3. Step 3: HDBSCAN for density clustering and noise extraction
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# 4. Step 4: Tokenizer with English stop-words removal
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# 5. Assemble and fit the unified BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=5,
    verbose=True
)

# Fit on unlabelled text corpus
topics, probabilities = topic_model.fit_transform(corpus_texts)

# Inspect discovered topic hierarchy
topic_info = topic_model.get_topic_info()
print(topic_info.head())
""", language="python")

st.divider()

# Section 3: Interactive Workbench — 5-Domain Semantic Clustering Suite
st.header("3. Interactive Workbench: 5-Domain Semantic Clustering Suite", divider="rainbow")

st.markdown("""
Experience semantic clustering applied across all 5 enterprise domains live! 
Choose an application domain to inspect the unorganized inputs, generate semantic clusters, extract representative keywords, 
and observe automated operational insights.
""")

demo_domains = {
    "1. Breaking News Story Aggregator (Google News)": {
        "description": "Groups disparate news articles from Bloomberg, Reuters, and TechCrunch reporting on the same real-world events.",
        "items": [
            {"title": "OpenAI releases GPT-4o with multimodal voice capabilities", "outlet": "TechCrunch", "event": "OpenAI Launch"},
            {"title": "OpenAI introduces omni model supporting real-time audio and vision", "outlet": "Reuters", "event": "OpenAI Launch"},
            {"title": "Sam Altman announces new flagship GPT-4o model for all users", "outlet": "Bloomberg", "event": "OpenAI Launch"},
            {"title": "Federal Reserve holds interest rates steady amid stubborn inflation", "outlet": "Wall Street Journal", "event": "Fed Rate Decision"},
            {"title": "Fed leaves benchmark rates unchanged as policymakers monitor economic data", "outlet": "CNBC", "event": "Fed Rate Decision"},
            {"title": "Central bank keeps borrowing costs at 23-year high in unanimous vote", "outlet": "Financial Times", "event": "Fed Rate Decision"},
            {"title": "SpaceX Starship completes successful fourth test flight and ocean splashdown", "outlet": "Ars Technica", "event": "Starship Launch"},
            {"title": "Starship survives atmospheric re-entry during historic fourth orbital test", "outlet": "SpaceNews", "event": "Starship Launch"},
            {"title": "Elon Musk celebrates Starship splashdown in Indian Ocean following booster return", "outlet": "Reuters", "event": "Starship Launch"}
        ]
    },
    "2. Support-Ticket Outage Detector": {
        "description": "Detects real-time production outages by clustering incoming customer bug reports before dashboards trigger.",
        "items": [
            {"title": "Cannot process credit card payments, checkout button spins forever", "outlet": "Customer Support", "event": "Payment Gateway Outage"},
            {"title": "Stripe webhook 504 gateway timeout on customer purchase order", "outlet": "SRE Pager", "event": "Payment Gateway Outage"},
            {"title": "All orders failing at payment processing screen with network error", "outlet": "Zendesk", "event": "Payment Gateway Outage"},
            {"title": "How do I invite team members to our workspace and assign admin roles?", "outlet": "Zendesk", "event": "User Management Inquiry"},
            {"title": "Where is the button to add a secondary account administrator?", "outlet": "Customer Support", "event": "User Management Inquiry"},
            {"title": "Can I transfer project ownership to another developer email address?", "outlet": "Zendesk", "event": "User Management Inquiry"}
        ]
    },
    "3. Customer Persona & Feedback Clustering": {
        "description": "Clusters unstructured survey text into actionable user personas and feature priorities.",
        "items": [
            {"title": "The mobile app is painfully slow on 4G networks, please optimize caching", "outlet": "App Store Review", "event": "Mobile Performance"},
            {"title": "App startup time takes over 8 seconds on older iPhone models", "outlet": "Play Store Review", "event": "Mobile Performance"},
            {"title": "Navigation transitions lag and drop frames during search scrolling", "outlet": "NPS Survey", "event": "Mobile Performance"},
            {"title": "Our compliance department requires SOC2 report and SSO integration", "outlet": "Enterprise Inquiry", "event": "Security Compliance"},
            {"title": "Do you support Okta SAML 2.0 and automated SCIM user provisioning?", "outlet": "Enterprise RFP", "event": "Security Compliance"},
            {"title": "We need data residency guarantees for EU GDPR data storage in Frankfurt", "outlet": "Legal Review", "event": "Security Compliance"}
        ]
    }
}

selected_demo_key = st.selectbox("Select Application Domain to Inspect:", list(demo_domains.keys()))
active_demo = demo_domains[selected_demo_key]

st.info(f"**Domain Objective**: {active_demo['description']}")

col_stream, col_clusters = col_stream, col_clusters = st.columns([1, 1])

with col_stream:
    st.subheader("Incoming Raw Unstructured Stream")
    df_raw = pd.DataFrame(active_demo["items"])[["title", "outlet"]]
    df_raw.columns = ["Incoming Text / Headline", "Source / Channel"]
    st.dataframe(df_raw, use_container_width=True, height=280)

# Simulate Clustering on the active demo
items_list = active_demo["items"]
texts_list = [it["title"] for it in items_list]

# Use simple TF-IDF + K-Means on the items to group them live
unique_events = list(dict.fromkeys([it["event"] for it in items_list]))
k_target = len(unique_events)

vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(texts_list)

km_demo = KMeans(n_clusters=k_target, random_state=42, n_init=10)
demo_labels = km_demo.fit_predict(X_tfidf)

# Group items into clusters
grouped_results = {}
for i, item in enumerate(items_list):
    c_lbl = demo_labels[i]
    if c_lbl not in grouped_results:
        grouped_results[c_lbl] = []
    grouped_results[c_lbl].append(item)

with col_clusters:
    st.subheader(f"Discovered Semantic Clusters ({k_target} Discovered Topics)")
    
    for c_id, members in grouped_results.items():
        event_name = members[0]["event"]
        count = len(members)
        
        with st.container():
            st.markdown(f"### Story Group {c_id + 1}: **{event_name}** `({count} Articles/Tickets)`")
            for m in members:
                st.caption(f"• **[{m['outlet']}]** {m['title']}")
            
            # Action trigger
            if "Outage" in event_name:
                st.error("🚨 **AUTOMATED INCIDENT TRIGGERED**: High-density cluster detected! Alerting On-Call Payment SRE team via PagerDuty.")
            elif "OpenAI" in event_name or "Starship" in event_name or "Fed" in event_name:
                st.success(f"📰 **Google News Story Card**: Grouped {count} international press reports into a unified event card.")
            else:
                st.info(f"📋 **Product Roadmap Tag**: Auto-routed to Product Manager for feature triage.")
            st.divider()

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How does BERTopic differ from traditional Latent Dirichlet Allocation (LDA) for topic discovery?", expanded=True):
    st.markdown("""
    **Answer:**
    * **Latent Dirichlet Allocation (LDA)**:
      * Bag-of-words probabilistic model that treats documents as mixtures of topics and topics as mixtures of words.
      * *Major Flaws*: Completely ignores word order, context, and syntax; treats *"not good"* the same as *"good not"*; struggles with short texts (tweets, search queries, customer tickets) due to word sparsity; requires pre-specifying the exact number of topics $\\alpha, \\beta$.
    * **BERTopic**:
      * Leverages pre-trained transformer embeddings that capture contextual semantic meaning and polysemy.
      * Uses non-linear manifold reduction (UMAP) and density clustering (HDBSCAN) to discover natural cluster shapes without assuming a fixed number of topics.
      * Uses **c-TF-IDF** to extract keywords that specifically differentiate one cluster from all others.
      * Produces dramatically more coherent, human-interpretable topics on both short and long texts with zero vocabulary pre-filtering.
    """)

with st.expander("Q2: How does a Google News-style news aggregator group breaking news articles across publishers in real-time?"):
    st.markdown("""
    **Answer:**
    Production breaking news aggregation employs a **Two-Tier Spatio-Temporal Clustering Pipeline**:
    1. **Sliding Time Window (Temporal Filtering)**: Restrict clustering candidates to articles published within a sliding 24-to-48-hour window. Articles about the same topic published two years apart should not be grouped into the same news event card.
    2. **Dense Semantic Embedding + Named Entity Filtering**:
       * Encode article headlines and lead paragraphs via a multilingual dense encoder.
       * Extract Named Entities (e.g., *OpenAI, Sam Altman, GPT-4o*).
       * Require both **high cosine similarity (> 0.85)** AND **shared core entity overlap** (Jaccard similarity on entities > 0.60).
    3. **Incremental Centroid Updating (Single-Pass Clustering)**:
       * When a new article arrives, compute similarity to existing active story clusters.
       * If similarity exceeds threshold, append to the story card and update the story centroid.
       * If similarity is below threshold, initialize a new breaking story cluster.
    """)

with st.expander("Q3: How can streaming support-ticket clustering detect zero-day outages before automated alerts?"):
    st.markdown("""
    **Answer:**
    Automated infrastructure alerts (Datadog, Prometheus) only fire when predefined metric thresholds are breached (e.g., CPU > 90% or Error Rate > 2%). 
    However, silent logical failures—such as a front-end checkout button failing to click or a 3rd-party payment gateway silently dropping requests—often report 200 OK HTTP codes and trigger zero alerts.
    
    **Streaming Semantic Clustering Triage**:
    1. Inbound customer complaints are vectorized in real-time as they arrive in Zendesk or Intercom.
    2. Maintain a 15-minute sliding density buffer.
    3. If a sudden surge of semantically similar tickets forms a tight cluster (e.g., cosine density spikes 5x above baseline, and 15+ tickets mention checkout spin within 10 minutes), the system flags an **Emerging Incident Anomaly**.
    4. The cluster medoid is sent directly to the engineering on-call channel, diagnosing outages up to 45 minutes faster than traditional human escalation paths.
    """)

with st.expander("Q4: Why does BERTopic use UMAP rather than PCA or t-SNE before clustering?"):
    st.markdown("""
    **Answer:**
    * **PCA (Principal Component Analysis)**: A purely linear projection. It fails to preserve non-linear manifold curvatures and folds complex high-dimensional semantic clusters on top of each other.
    * **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Excellent for 2D visualization, but it only preserves local neighborhood distances, completely destroying global distances and cluster densities. Furthermore, t-SNE does not easily allow projecting new unseen data points.
    * **UMAP (Uniform Manifold Approximation and Projection)**:
      * Preserves both **local neighborhood structure** AND **global inter-cluster distances**.
      * Significantly faster than t-SNE ($O(N \\log N)$).
      * Supports arbitrary output dimensions (typically reducing 1024-d down to 5 or 10 dimensions for HDBSCAN).
      * Provides a fast `transform()` method to project new incoming documents into the learned embedding space without retraining.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Clustering News Articles Without Temporal Decay**: Grouping articles about "Interest Rates Unchanged" published in 2021 with articles published today into the same breaking news card. Always enforce temporal decay windows.
* **Trap 2: Ignoring Stop-Words and Punctuation in c-TF-IDF**: Passing raw text into c-TF-IDF without removing domain-generic terms. In customer service, every cluster will have top keywords like *"please"*, *"help"*, *"issue"*, *"hello"*, making clusters look identical. Always use domain-tuned stop-word filters.
* **Trap 3: Forgetting That Outliers in HDBSCAN Can Starve Minor Topics**: Setting `min_cluster_size` too high in HDBSCAN (e.g., `min_cluster_size=100` on a 500-sample dataset). Over 80% of points will be classified as noise (`-1`), hiding legitimate emerging micro-topics.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: BERTopic"):
        st.markdown("**Front**: What are the 5 core stages of BERTopic?")
        st.divider()
        st.markdown("**Back**: 1. Embeddings (SBERT), 2. Dimensionality reduction (UMAP), 3. Clustering (HDBSCAN), 4. Keyword extraction (c-TF-IDF), 5. LLM representation.")

with card_cols[1]:
    with st.popover("Card 2: News Aggregator"):
        st.markdown("**Front**: How do news aggregators group articles into story cards?")
        st.divider()
        st.markdown("**Back**: Combines dense semantic cosine similarity (> 0.85) with named entity overlap and a 24-hour sliding temporal decay window.")

with card_cols[2]:
    with st.popover("Card 3: Outage Detection"):
        st.markdown("**Front**: How does ticket clustering detect silent outages?")
        st.divider()
        st.markdown("**Back**: Monitors real-time density spikes: if an unexpected cluster of identical complaints forms within 10 minutes, an automated incident is paged.")

with card_cols[3]:
    with st.popover("Card 4: UMAP vs PCA"):
        st.markdown("**Front**: Why is UMAP preferred over PCA for text embeddings?")
        st.divider()
        st.markdown("**Back**: UMAP is non-linear and preserves both local neighborhoods and global cluster separation, preventing dense clusters from collapsing.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Semantic Clustering Applications:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 5 core applications of semantic clustering.", key="chk30_3_1")
    st.checkbox("I know the complete 5-stage BERTopic architecture.", key="chk30_3_2")
    st.checkbox("I can implement real-time incident detection via ticket clustering.", key="chk30_3_3")
with c2:
    st.checkbox("I understand how news aggregators combine embeddings with temporal decay.", key="chk30_3_4")
    st.checkbox("I know why UMAP is superior to PCA and t-SNE for clustering preparation.", key="chk30_3_5")
    st.checkbox("I can apply domain-specific stop-word filtering in c-TF-IDF pipelines.", key="chk30_3_6")
