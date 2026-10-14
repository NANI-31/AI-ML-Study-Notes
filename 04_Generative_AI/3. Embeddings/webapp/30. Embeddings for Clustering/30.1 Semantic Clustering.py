import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="30.1 Semantic Clustering - Embeddings Curriculum",
    page_icon="🧩",
    layout="wide"
)

# Title & Overview
st.title("30.1 Semantic Clustering: Unsupervised Grouping, Similarity Manifolds & Cluster Labeling")
st.caption("Chapter 30: Embeddings for Clustering | Domain: 3. Vector Embeddings")
st.markdown("""
In enterprise data landscapes, over 80% of text data is **unlabeled, noisy, and unorganized** (e.g., millions of customer support transcripts, 
employee survey feedback, open-ended reviews, internal Slack conversations, and uncataloged PDF archives).
Manually reading and tagging these vast corpora with human annotators is prohibitively slow and expensive.

**Semantic Clustering** applies unsupervised machine learning algorithms directly onto dense embedding vectors:
1. **Generate Embeddings**: Pass unstructured text through a pre-trained foundation model to obtain continuous vector representations.
2. **Calculate Similarity**: Compute cosine distances on the unit hypersphere where spatial proximity indicates semantic affinity.
3. **Apply Clustering**: Group vectors into coherent geometric partitions without requiring any predefined labels or taxonomies.
4. **Analyze & Label Clusters**: Extract central medoids and run class-based TF-IDF (c-TF-IDF) or LLM summarization to automatically discover 
   **emerging themes, trending customer pain points, and natural topic taxonomies**.
""")

st.divider()

# Section 1: The 4-Stage Semantic Clustering Lifecycle
st.header("1. The 4-Stage Semantic Clustering Lifecycle", divider="rainbow")

col_life1, col_life2 = col_life1, col_life2 = st.columns([1, 1])

with col_life1:
    st.markdown("""
    ### The 4 Core Stages
    * **Stage 1: Vector Extraction & L2 Normalization**:
      * Ingest unlabeled corpus: `D = {doc_1, doc_2, ..., doc_N}`.
      * Pass text through an encoder (e.g., `text-embedding-3-small` or `BGE-Large-en-v1.5`).
      * Project every vector onto the unit hypersphere: `v_i = v_i / ||v_i||_2`.
    * **Stage 2: Metric Distance Computation**:
      * In normalized space, Euclidean distance squared is directly proportional to Cosine distance:
        `||v_i - v_j||_2^2 = 2 - 2 * dot(v_i, v_j)`.
      * Allows standard algorithms (K-Means) to perform spherical cosine clustering natively!
    * **Stage 3: High-Dimensional Partitioning**:
      * Group vectors into clusters such that intra-cluster distance is minimized and inter-cluster separation is maximized.
    * **Stage 4: Automated Cluster Labeling & Theme Extraction**:
      * Find the **Medoid** (the actual document closest to the cluster centroid).
      * Prompt an LLM with the medoids: *"Analyze these 5 representative customer complaints and output a concise 4-word theme title."*
    """)

with col_life2:
    st.markdown("### Semantic Clustering Pipeline Flow")
    flow_dot = """
    digraph SemanticClustering {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UnlabeledData [label="Unlabeled Text Corpus\\n100,000 Support Inquiries", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Encoder [label="Embeddings Engine\\nGenerates 1536-dim Vectors\\nL2 Normalization Unit", fillcolor="#DBEAFE"];

        ClusterEngine [label="Clustering Engine\\nSpherical K-Means / HDBSCAN\\nPartitions Hypersphere", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        ClusterA [label="Cluster 1 (18% of data)\\nCentroid: Billing Discrepancies", fillcolor="#D1FAE5"];
        ClusterB [label="Cluster 2 (34% of data)\\nCentroid: Kubernetes 504 Timeouts", fillcolor="#FCE7F3"];
        ClusterC [label="Cluster 3 (22% of data)\\nCentroid: SAML SSO Login Errors", fillcolor="#E0E7FF"];

        AutoLabeler [label="LLM Topic Labeler (c-TF-IDF)\\nAssigns Human-Readable Names", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UnlabeledData -> Encoder;
        Encoder -> ClusterEngine;
        ClusterEngine -> ClusterA;
        ClusterEngine -> ClusterB;
        ClusterEngine -> ClusterC;
        ClusterA -> AutoLabeler;
        ClusterB -> AutoLabeler;
        ClusterC -> AutoLabeler;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# Section 2: Evaluating Cluster Quality & The Silhouette Score
st.header("2. Cluster Validation Metrics: Silhouette Score & Davies-Bouldin", divider="rainbow")

st.markdown("""
In unsupervised learning, there is no ground truth $y$ to compute Accuracy or F1 score. 
Instead, we rely on intrinsic geometric cluster validation metrics:
""")

col_val1, col_val2 = col_val1, col_val2 = st.columns([1, 1])

with col_val1:
    st.markdown("""
    ### 1. The Silhouette Coefficient
    For a given data point $i$:
    * $a(i)$ = Mean distance between point $i$ and all other points in the **same cluster** (intra-cluster cohesion).
    * $b(i)$ = Mean distance between point $i$ and all points in the **nearest neighboring cluster** (inter-cluster separation).
    
    ```text
    Silhouette_Score(i) = (b(i) - a(i)) / max(a(i), b(i))
    ```
    
    * **Range**: $-1.0$ to $+1.0$.
    * **Score > 0.50**: Well-separated, dense, high-quality clusters.
    * **Score ~ 0.0**: Overlapping clusters (borderline points).
    * **Score < 0.0**: Sample assigned to the wrong cluster.

    ### 2. Davies-Bouldin Index
    * Measures the ratio of within-cluster scatter to between-cluster separation.
    * **Lower values are better** (zero indicates optimal clustering).
    """)

with col_val2:
    st.markdown("### Production Code: Automated Topic Discovery & Evaluation")
    st.code("""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI

client = OpenAI()

def cluster_and_evaluate(embeddings: np.ndarray, texts: list[str], k: int = 5):
    # 1. Fit Spherical K-Means on normalized vectors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    # 2. Evaluate cluster cohesion
    score = silhouette_score(embeddings, labels, metric='cosine')
    
    # 3. Find medoid (representative document) per cluster
    cluster_summaries = {}
    for cluster_id in range(k):
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_vecs = embeddings[cluster_indices]
        
        # Centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        # Find item with highest cosine similarity to centroid
        sims_to_centroid = np.dot(cluster_vecs, centroid)
        medoid_idx = cluster_indices[np.argmax(sims_to_centroid)]
        
        cluster_summaries[cluster_id] = {
            "size": int(np.sum(cluster_mask)),
            "medoid_text": texts[medoid_idx]
        }
        
    return labels, score, cluster_summaries
""", language="python")

st.divider()

# Section 3: Interactive Workbench — Semantic Clustering Studio
st.header("3. Interactive Workbench: Live Semantic Clustering Studio", divider="rainbow")

st.markdown("""
Run **Spherical K-Means** live on a real-world enterprise dataset of customer support inquiries and IT incident reports.
Adjust the number of clusters $K$, inspect the resulting clusters and their calculated Silhouette score, and observe 
how representative medoids reveal distinct operational topics automatically!
""")

unlabeled_corpus = [
    # Topic A: Billing & Refund issues
    {"text": "Why was my credit card billed twice for the monthly subscription renewal?", "true_theme": "Billing & Payments"},
    {"text": "I need to update our corporate VAT number on the downloadable invoice PDF.", "true_theme": "Billing & Payments"},
    {"text": "How do I cancel our annual plan and receive a pro-rated refund to our card?", "true_theme": "Billing & Payments"},
    {"text": "Our payment method was rejected by our issuing bank due to cross-border limits.", "true_theme": "Billing & Payments"},
    {"text": "Where can our accounts department download past receipt statements?", "true_theme": "Billing & Payments"},
    
    # Topic B: DevOps & Container failures
    {"text": "Kubernetes pods are crashing with exit code 137 OOMKilled during traffic spikes.", "true_theme": "DevOps & Cloud"},
    {"text": "Envoy ingress proxy is intermittently returning 504 Gateway Timeout errors.", "true_theme": "DevOps & Cloud"},
    {"text": "Docker container fails to mount persistent volume claim on AWS EBS gp3.", "true_theme": "DevOps & Cloud"},
    {"text": "Prometheus node exporter reports high memory consumption above 95% on worker nodes.", "true_theme": "DevOps & Cloud"},
    {"text": "How do I configure horizontal pod autoscaling stabilization window in helm charts?", "true_theme": "DevOps & Cloud"},
    
    # Topic C: Identity & Security access
    {"text": "User lost their two-factor authentication mobile app and cannot log into the portal.", "true_theme": "Identity & Security"},
    {"text": "How do we rotate compromised AWS IAM access keys and revoke active sessions?", "true_theme": "Identity & Security"},
    {"text": "Encountering SAML 2.0 signature validation mismatch error when logging in with Okta.", "true_theme": "Identity & Security"},
    {"text": "Suspicious login attempt detected from an unrecognized IP address in eastern Europe.", "true_theme": "Identity & Security"},
    {"text": "Please reset the administrative password for the service account root user.", "true_theme": "Identity & Security"}
]

# Generate synthetic 16-d semantic embeddings for the corpus
np.random.seed(42)
topic_centers = {
    "Billing & Payments": np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1]),
    "DevOps & Cloud": np.array([0.1, 0.1, 0.9, 0.8, 0.7, 0.9, 0.1, 0.1, 0.1, 0.9, 0.8, 0.9, 0.1, 0.9, 0.8, 0.1]),
    "Identity & Security": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.9])
}

X_corpus = []
for doc in unlabeled_corpus:
    c = topic_centers[doc["true_theme"]]
    noise = np.random.normal(0, 0.12, 16)
    vec = c + noise
    X_corpus.append(vec / np.linalg.norm(vec))

X_corpus = np.array(X_corpus)
all_texts = [d["text"] for d in unlabeled_corpus]

col_k_ctrl, col_k_view = col_k_ctrl, col_k_view = st.columns([1, 1])

with col_k_ctrl:
    st.subheader("Clustering Configuration")
    k_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=5, value=3, step=1)
    
    # Fit K-Means
    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_assignments = km.fit_predict(X_corpus)
    
    # Silhouette
    sil_score = silhouette_score(X_corpus, cluster_assignments, metric="cosine")
    
    st.metric("Overall Silhouette Score (Cosine)", f"{sil_score:.3f}")
    
    if sil_score > 0.45:
        st.success("High Cluster Quality: Distinct, well-separated semantic groupings!")
    else:
        st.warning("Moderate Separation: Some overlap between cluster boundaries.")

# Find medoids per cluster
cluster_table = []
for c_id in range(k_clusters):
    indices = np.where(cluster_assignments == c_id)[0]
    count = len(indices)
    
    if count > 0:
        c_vecs = X_corpus[indices]
        centroid = km.cluster_centers_[c_id]
        centroid_norm = centroid / np.linalg.norm(centroid)
        sims = np.dot(c_vecs, centroid_norm)
        medoid_idx = indices[np.argmax(sims)]
        medoid_str = all_texts[medoid_idx]
    else:
        medoid_str = "Empty"
        
    cluster_table.append({
        "Cluster ID": f"Cluster {c_id + 1}",
        "Item Count": count,
        "Cluster Medoid (Representative Sample)": medoid_str
    })

with col_k_view:
    st.subheader(f"Discovered Clusters (K = {k_clusters})")
    st.dataframe(pd.DataFrame(cluster_table), use_container_width=True, hide_index=True)
    
    st.caption("""
    *Medoid Discovery*: The representative sample shown for each cluster is the exact document located closest 
    to the high-dimensional centroid. Notice how K=3 cleanly splits the data into Billing, DevOps, and Security!
    """)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What is Spherical K-Means and why is it preferred over standard Euclidean K-Means for text embeddings?", expanded=True):
    st.markdown("""
    **Answer:**
    Standard K-Means minimizes the sum of squared **Euclidean distances** between points and cluster centroids. 
    In raw high-dimensional text embeddings, longer documents with high token counts can have larger vector norms, meaning Euclidean distance is confounded by document length rather than semantic angle.
    
    **Spherical K-Means** solves this:
    1. All data vectors are normalized onto the unit hypersphere: $||x_i||_2 = 1.0$.
    2. During centroid updates, the new centroid is re-normalized onto the unit sphere after every iteration:
       $\\mu_k = \\frac{\\sum_{i \\in C_k} x_i}{||\\sum_{i \\in C_k} x_i||_2}$
    3. On unit vectors, maximizing Cosine Similarity is mathematically identical to minimizing Euclidean distance squared:
       $||x - y||_2^2 = 2 - 2(x \\cdot y) = 2(1 - \\cos(x, y))$.
    4. Spherical K-Means clusters data strictly based on **semantic orientation** rather than vector magnitude, delivering significantly cleaner text topic boundaries.
    """)

with st.expander("Q2: How do you choose the optimal number of clusters K when working with unlabeled embeddings?"):
    st.markdown("""
    **Answer:**
    Determining optimal $K$ in unsupervised clustering requires combining heuristic metrics:
    1. **The Silhouette Method**: Compute the average Silhouette score across $K = 2$ to $K = 30$. The $K$ value that produces the global maximum silhouette score represents the most cohesive, well-separated partition.
    2. **The Elbow Method (Inertia Curve)**: Plot the sum of squared distances to centroids (inertia) versus $K$. Identify the "elbow" point where adding more clusters yields diminishing returns in inertia reduction.
    3. **Domain Density (HDBSCAN Fallback)**: If the number of natural topics is unknown, use density-based clustering like **HDBSCAN**, which does not require specifying $K$ upfront and automatically discovers the natural number of dense clusters while isolating noise points!
    """)

with st.expander("Q3: What is the difference between a Cluster Centroid and a Cluster Medoid?"):
    st.markdown("""
    **Answer:**
    * **Centroid**: The mathematical average (mean) of all vectors in the cluster: $\\mu = \\frac{1}{|C|} \\sum_{x \\in C} x$.
      * *Limitation*: The centroid is a synthetic floating-point vector that does **not correspond to any real document** in your dataset. You cannot "read" a centroid.
    * **Medoid**: The actual, existing document in the cluster whose vector has the smallest distance (highest cosine similarity) to the centroid:
      $x_{medoid} = \\arg\\max_{x \\in C} (x \\cdot \\mu)$.
      * *Strength*: The medoid is an authentic, human-readable document that serves as the perfect representative exemplar of the entire cluster for prompt engineering and human review.
    """)

with st.expander("Q4: How does c-TF-IDF (Class-based TF-IDF) extract human-readable topic labels for embedding clusters?"):
    st.markdown("""
    **Answer:**
    **c-TF-IDF (Maarten Grootendorst, BERTopic)** treats each discovered cluster as a single composite "mega-document" by concatenating all text samples within that cluster.
    It then modifies the traditional TF-IDF formula:
    ```text
    c-TF-IDF_score = (tf_{t, c}) * log( 1 + (A / f_t) )
    ```
    Where:
    * $tf_{t, c}$ is the frequency of word $t$ in cluster $c$.
    * $f_t$ is the frequency of word $t$ across all clusters.
    * $A$ is the average number of words per cluster.
    This extracts words that are **frequently mentioned inside this specific cluster but rare in all other clusters**, producing crisp, discriminative topic labels (e.g., *["oomkilled", "cgroup", "exit 137"]*) without calling external LLM APIs!
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Clustering Without Pre-Filtering Boilerplate / Email Signatures**: Clustering raw customer emails containing legal disclaimers (*"This message contains confidential information..."*). The disclaimer becomes the dominant semantic signal, grouping unrelated tickets together into a useless "Confidentiality Notice" cluster.
* **Trap 2: Forcing K-Means on Uneven Cluster Densities**: Using K-Means when the data contains one massive cluster (70% of tickets) and multiple tiny niche topics. K-Means assumes spherical, equal-variance clusters and will artificially fracture the big cluster while misclassifying the small ones. Use HDBSCAN instead.
* **Trap 3: Neglecting Outlier / Noise Handling**: K-Means forces every single vector—including gibberish, spam, and corrupted strings—into one of the $K$ clusters. These outliers pull centroids away from true semantic centers.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Spherical K-Means"):
        st.markdown("**Front**: Why normalize vectors for Spherical K-Means?")
        st.divider()
        st.markdown("**Back**: Unit normalization ensures Euclidean distance corresponds strictly to cosine angle, clustering by semantic concept rather than document word count.")

with card_cols[1]:
    with st.popover("Card 2: Silhouette Score"):
        st.markdown("**Front**: What does a Silhouette Score > 0.50 indicate?")
        st.divider()
        st.markdown("**Back**: Strong cluster quality: points are tightly clustered internally and well-separated from neighboring clusters.")

with card_cols[2]:
    with st.popover("Card 3: Medoid"):
        st.markdown("**Front**: What is a Cluster Medoid?")
        st.divider()
        st.markdown("**Back**: The actual real document in the dataset closest to the cluster center, providing a human-readable exemplar of the topic.")

with card_cols[3]:
    with st.popover("Card 4: c-TF-IDF"):
        st.markdown("**Front**: What is c-TF-IDF in BERTopic?")
        st.divider()
        st.markdown("**Back**: Class-based TF-IDF treats clusters as single documents to extract the most distinctive, discriminative keywords defining each topic.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Semantic Clustering:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4 stages of the semantic clustering lifecycle.", key="chk30_1_1")
    st.checkbox("I know why Spherical K-Means is used on L2-normalized embeddings.", key="chk30_1_2")
    st.checkbox("I can evaluate cluster separation using the Silhouette Coefficient.", key="chk30_1_3")
with c2:
    st.checkbox("I understand the distinction between cluster centroids and cluster medoids.", key="chk30_1_4")
    st.checkbox("I know how c-TF-IDF automatically extracts discriminative cluster labels.", key="chk30_1_5")
    st.checkbox("I can implement end-to-end unsupervised topic discovery in scikit-learn.", key="chk30_1_6")
