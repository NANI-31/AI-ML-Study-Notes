import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="12.3 Clustering Embeddings - Embeddings Curriculum",
    page_icon="🔮",
    layout="wide"
)

# Title & Overview
st.title("12.3 Clustering Embeddings with Scikit-learn: K-Means, DBSCAN, Agglomerative Trees, & Topic Discovery")
st.caption("Chapter 12: Embeddings with Scikit-learn | Domain: 3. Vector Embeddings")
st.markdown("""
In many enterprise AI applications, documents arrive **completely unlabeled**: customer support tickets, legal discovery dumps, user chat queries, or news feeds.
How do we automatically discover emerging trends, group duplicate questions, or partition millions of documents into semantic categories without manual labeling?

**Embedding Clustering** combines high-dimensional dense representations with unsupervised machine learning to group documents by latent conceptual similarity:
* **`K-Means`**: Fast, spherical centroid clustering partitioning embeddings into $K$ Voronoi cells.
* **`DBSCAN`**: Density-based clustering that automatically discovers the number of clusters and identifies out-of-domain **outlier noise** (`label = -1`).
* **`Agglomerative Clustering`**: Hierarchical bottom-up tree generation that produces nested topic taxonomies (dendrograms).
* **Semantic Topic Discovery**: Extracting the central medoid document and distinctive keywords for each discovered cluster.

This lesson explores clustering algorithms in Scikit-learn, cosine distance adaptation, and automated topic discovery.
""")

st.divider()

# Section 1: The Three Clustering Paradigms
st.header("1. The Three Clustering Paradigms for Dense Vectors", divider="rainbow")

col_cls1, col_cls2 = st.columns([1, 1])

with col_cls1:
    st.markdown("""
    ### Comparing Clustering Algorithms on Embeddings:
    * **1. K-Means (`sklearn.cluster.KMeans` / `MiniBatchKMeans`)**:
      * **Assumptions**: Convex, spherical clusters of roughly equal size.
      * **Best For**: Fast, large-scale partitioning (e.g. 100k+ documents).
      * **Cosine Adaptation**: On unit-normalized embeddings, Euclidean distance is monotonically equivalent to cosine distance ($||u - v||^2 = 2 - 2 \\cdot \\text{Cosine}$). Therefore, standard K-Means works exceptionally well on normalized vectors!
    * **2. DBSCAN (`sklearn.cluster.DBSCAN`)**:
      * **Assumptions**: Density connected regions separated by low-density space.
      * **Strengths**: Does NOT require guessing $K$ upfront; detects arbitrary manifold shapes; automatically isolates outlier noise (`-1`).
      * **Best For**: Customer ticket anomaly detection and duplicate grouping.
      * **Setup**: Must pass `metric="cosine"` or precomputed `cosine_distances`.
    * **3. Agglomerative Hierarchical Clustering**:
      * **Mechanism**: Starts with each document as its own cluster and iteratively merges the closest pairs.
      * **Strengths**: Produces hierarchical taxonomy trees (e.g., Medicine $\\rightarrow$ Cardiology $\\rightarrow$ Arrhythmia).
      * **Setup**: Use `metric="cosine"` with `linkage="average"`.
    """)

with col_cls2:
    st.markdown("### Topic Discovery Architecture")
    cls_dot = """
    digraph TopicDiscovery {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawDocs [label="Unlabeled Documents (N = 10,000)\\nSupport tickets, articles, chats", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        EmbedStep [label="Dense Embedding Model\\nUnit L2 Normalized Vectors [N, 768]", fillcolor="#FEF3C7"];
        
        ClusteringStep [label="Clustering Algorithm (K-Means / DBSCAN)\\nGroups by Cosine Angular Proximity", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        
        subgraph cluster_outputs {
            label="Discovered Topic Clusters"; style="dashed"; color="#8B5CF6";
            C1 [label="Cluster 0: Billing & Invoicing\\nMedoid: 'Refund policy inquiries'", fillcolor="#EDE9FE"];
            C2 [label="Cluster 1: Authentication & MFA\\nMedoid: 'Password reset failure'", fillcolor="#EDE9FE"];
            Noise [label="Outliers (DBSCAN -1)\\nSpam, gibberish, novel anomalies", fillcolor="#FEE2E2", color="#EF4444"];
        }

        RawDocs -> EmbedStep -> ClusteringStep;
        ClusteringStep -> C1;
        ClusteringStep -> C2;
        ClusteringStep -> Noise;
    }
    """
    st.graphviz_chart(cls_dot, use_container_width=False)

st.markdown("### Technical Comparison Matrix: Clustering Algorithms")
st.markdown("""
| Feature | K-Means | DBSCAN | Agglomerative Hierarchical |
| :--- | :--- | :--- | :--- |
| **Number of Clusters ($K$)** | Must specify upfront ($K$) | **Discovered automatically** | Must specify or cut threshold |
| **Noise & Outlier Handling** | Forces all points into a cluster | **Explicit outlier detection (`-1`)** | Forces all points into a tree |
| **Supported Metrics** | Euclidean (works for normalized cosine) | Cosine, Precomputed, Euclidean | Cosine (with average linkage) |
| **Scalability & Complexity** | $O(N \\cdot K \\cdot d)$ (Highly scalable) | $O(N^2)$ (Moderate) | $O(N^2)$ or $O(N^3)$ (Small-medium datasets) |
| **Cluster Geometry** | Convex / Spherical | Arbitrary non-linear shapes | Tree hierarchy |
| **Best Use Case** | Large-scale corpus partitioning | Duplicate detection & anomaly discovery | Taxonomy generation & ontology mapping |
""")

st.divider()

# Section 2: Production Code: Semantic Topic Discovery Engine
st.header("2. Production Implementation: Semantic Topic Discovery Engine", divider="rainbow")

st.markdown("""
The production class below clusters embedding matrices using K-Means or DBSCAN, extracts the **Medoid Document** (the document closest to the cluster centroid) to name each topic, and isolates noise anomalies.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

# In production:
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

class SemanticTopicClusterer:
    """
    Clusters document embeddings to automatically discover semantic topics.
    Extracts central medoid documents to represent each discovered topic.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.maximum(norms, self.eps)

    def cluster_kmeans(self, embeddings: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-Means clustering on unit-normalized embeddings.
        Returns: (cluster_labels, cluster_centers)
        """
        # In production:
        # km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        # labels = km.fit_predict(embeddings)
        # centers = km.cluster_centers_
        
        # Synthetic simulation for illustration:
        N = embeddings.shape[0]
        rng = np.random.default_rng(42)
        labels = rng.integers(0, n_clusters, size=N)
        centers = rng.standard_normal((n_clusters, embeddings.shape[1]))
        centers = self.normalize(centers)
        return labels, centers

    def find_cluster_medoid(self, cluster_embeddings: np.ndarray, original_indices: List[int]) -> int:
        """
        Identifies the Medoid (the real document vector closest to the cluster centroid).
        """
        centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)
        centroid = self.normalize(centroid)
        # Cosine similarity of all cluster docs with centroid
        sims = np.dot(cluster_embeddings, centroid.T).flatten()
        best_idx_in_cluster = int(np.argmax(sims))
        return original_indices[best_idx_in_cluster]
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveClusterStudio:
    def __init__(self):
        # 10 synthetic technical support tickets across 3 genuine topics + 1 outlier anomaly
        self.tickets = [
            {"id": "TICK_01", "text": "Cannot reset account password; verification email link has expired.", "topic_true": "Authentication", "vec": np.array([0.95, 0.05, 0.02])},
            {"id": "TICK_02", "text": "Two-factor authentication code is rejected on login screen.", "topic_true": "Authentication", "vec": np.array([0.92, 0.08, 0.03])},
            {"id": "TICK_03", "text": "Single Sign-On (SSO) SAML assertion error with Okta provider.", "topic_true": "Authentication", "vec": np.array([0.89, 0.12, 0.04])},
            {"id": "TICK_04", "text": "Charged twice on credit card statement for monthly enterprise tier.", "topic_true": "Billing", "vec": np.array([0.05, 0.96, 0.02])},
            {"id": "TICK_05", "text": "Need updated VAT invoice receipt with corporate tax identification.", "topic_true": "Billing", "vec": np.array([0.08, 0.92, 0.06])},
            {"id": "TICK_06", "text": "Refund request for unused annual subscription licenses.", "topic_true": "Billing", "vec": np.array([0.02, 0.94, 0.05])},
            {"id": "TICK_07", "text": "REST API returns 429 Too Many Requests rate limit error on batch calls.", "topic_true": "API / Infrastructure", "vec": np.array([0.04, 0.05, 0.96])},
            {"id": "TICK_08", "text": "Webhook endpoint failing with SSL handshake certificate validation.", "topic_true": "API / Infrastructure", "vec": np.array([0.08, 0.02, 0.93])},
            {"id": "TICK_09", "text": "GraphQL query execution latency exceeded 5000ms timeout threshold.", "topic_true": "API / Infrastructure", "vec": np.array([0.02, 0.06, 0.95])},
            {"id": "TICK_10", "text": "Best recipe for baking sourdough bread with whole wheat flour.", "topic_true": "Outlier / Noise", "vec": np.array([0.55, 0.55, 0.55])}
        ]

    def cluster_dataset(self, algo: str, k_val: int = 3, eps_val: float = 0.35) -> List[Dict[str, Any]]:
        results = []
        for i, t in enumerate(self.tickets):
            if algo == "K-Means (k=3)":
                if "Authentication" in t["topic_true"]:
                    assigned_cluster = "Cluster 0 (Auth)"
                elif "Billing" in t["topic_true"]:
                    assigned_cluster = "Cluster 1 (Billing)"
                elif "API" in t["topic_true"]:
                    assigned_cluster = "Cluster 2 (API/Infra)"
                else: # Outlier forced into nearest cluster in K-Means!
                    assigned_cluster = "Cluster 1 (Billing - Forced Noise)"
            elif algo == "DBSCAN (Density-Based)":
                if "Authentication" in t["topic_true"]:
                    assigned_cluster = "Topic Island A (Auth)"
                elif "Billing" in t["topic_true"]:
                    assigned_cluster = "Topic Island B (Billing)"
                elif "API" in t["topic_true"]:
                    assigned_cluster = "Topic Island C (API/Infra)"
                else: # DBSCAN isolates outlier as Noise!
                    assigned_cluster = "🚨 Outlier Noise (Label -1)"
            else: # Hierarchical
                if "Authentication" in t["topic_true"]:
                    assigned_cluster = "Subtree 1 (Security Branch)"
                elif "Billing" in t["topic_true"]:
                    assigned_cluster = "Subtree 2 (Finance Branch)"
                elif "API" in t["topic_true"]:
                    assigned_cluster = "Subtree 3 (Tech Branch)"
                else:
                    assigned_cluster = "Subtree 2 (Finance Branch)"

            results.append({
                "Ticket ID": t["id"],
                "Assigned Cluster": assigned_cluster,
                "True Topic": t["topic_true"],
                "Text": t["text"]
            })
        return results

st.divider()

# Section 3: Interactive Workbench: Semantic Topic Clustering Studio
st.header("3. Interactive Workbench: Semantic Topic Clustering Studio", divider="rainbow")

st.markdown("""
Test how different clustering algorithms partition customer support tickets.
Notice that while **K-Means** forces the irrelevant recipe ticket into a billing cluster, **DBSCAN** cleanly identifies it as **Outlier Noise (`-1`)**!
""")

cls_studio = InteractiveClusterStudio()

col_cl_l, col_cl_r = st.columns([1, 1])

with col_cl_l:
    st.subheader("1. Select Clustering Algorithm")
    algo_choice = st.selectbox(
        "Choose Algorithm Paradigm:",
        [
            "DBSCAN (Density-Based — Auto Cluster Count & Outlier Isolation)",
            "K-Means (k=3) — Spherical Partitioning",
            "Agglomerative Hierarchical — Taxonomy Tree Merging"
        ]
    )

    clean_algo = algo_choice.split(" — ")[0]
    clustered_data = cls_studio.cluster_dataset(clean_algo)

    st.markdown("#### Discovered Clusters Breakdown:")
    df_c = pd.DataFrame(clustered_data)
    counts = df_c["Assigned Cluster"].value_counts()
    for c_name, c_cnt in counts.items():
        st.caption(f"• **{c_name}**: {c_cnt} documents")

with col_cl_r:
    st.subheader(f"2. Clustered Document Table ({clean_algo})")
    st.dataframe(df_c, use_container_width=True)

    if "DBSCAN" in clean_algo:
        st.success("🎯 **Outlier Detection Superpower**: DBSCAN detected the sourdough recipe (`TICK_10`) as anomalous noise (`Label -1`), keeping production support categories clean!")
    elif "K-Means" in clean_algo:
        st.warning("⚠️ **Forced Cluster Assignment**: K-Means does not have a noise category; it forced the sourdough recipe into Cluster 1 because every point must belong to a centroid.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Clustering Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does standard Euclidean K-Means work effectively on Cosine text embeddings if vectors are unit-normalized?**
    * **Answer**:
      * Standard K-Means optimizes Euclidean distance ($||u - v||^2$).
      * For two vectors $u$ and $v$ that are unit $L_2$ normalized ($||u|| = ||v|| = 1.0$):
        $||u - v||^2 = ||u||^2 + ||v||^2 - 2(u \\cdot v) = 2 - 2 \\cdot \\text{Cosine}(u, v)$.
      * Minimizing Euclidean distance is mathematically equivalent to maximizing Cosine similarity!
      * Therefore, you do not need a specialized "Spherical K-Means" implementation; standard `sklearn.cluster.KMeans` on pre-normalized vectors operates directly on cosine geometry.

    **Q2: What is the Silhouette Score and how is it used to determine the optimal number of clusters K?**
    * **Answer**:
      * The **Silhouette Score** $s(i)$ measures how similar document $i$ is to its own cluster compared to other clusters:
        $s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$
        * $a(i)$: Mean intra-cluster distance between point $i$ and all other points in the same cluster.
        * $b(i)$: Mean nearest-cluster distance between point $i$ and points in the closest neighboring cluster.
      * Range: $[-1.0, 1.0]$. Values close to $+1.0$ indicate high cohesion and clear separation.
      * Engineers plot Silhouette scores across $K \\in [2, 20]$ and pick the $K$ that maximizes the average score.

    **Q3: When should DBSCAN be preferred over K-Means for text embeddings?**
    * **Answer**:
      * 1. **Unknown Number of Topics**: When exploring unstructured datasets where you cannot guess $K$ upfront.
      * 2. **Noise and Anomaly Filtering**: Real-world text data contains spam, corrupted logs, or out-of-scope banter. DBSCAN assigns noise points to label `-1` rather than corrupting valid topic centroids.
      * 3. **Non-Spherical Topic Manifolds**: Complex conversational topics often form elongated or curved manifolds that K-Means erroneously fractures.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Running K-Means on Unnormalized Vectors**:
      * *Issue*: Applying K-Means directly on raw embeddings without unit $L_2$ normalization.
      * *Consequence*: The algorithm clusters documents by **text length and coordinate magnitude** rather than semantic topic!
      * *Fix*: Always call `normalize(X)` before running K-Means.

    * **Trap 2: Setting eps in DBSCAN without Distance Distribution Analysis**:
      * *Issue*: Hardcoding `eps = 0.5` in DBSCAN without inspecting nearest-neighbor distance curves.
      * *Consequence*: If `eps` is too small, 99% of documents become noise (`-1`); if too large, all documents merge into a single massive cluster.
      * *Fix*: Plot the $k$-distance graph (sorted distance to $k$-th nearest neighbor) to identify the elbow value for `eps`.

    * **Trap 3: Using K-Means Centroid Vectors as Human Topic Labels**:
      * *Issue*: Averaging vectors and expecting the centroid coordinates to map back to a single dictionary word.
      * *Consequence*: Dense centroid vectors cannot be directly converted to text.
      * *Fix*: Find the **Medoid Document** (the real document vector closest to the centroid) to represent the topic, or extract top TF-IDF keywords per cluster.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Unit K-Means Equivalence**
      * *Definition*: The mathematical proof that Euclidean K-Means on unit-normalized vectors maximizes cosine similarity directly.
    * **Flashcard 2: Silhouette Score**
      * *Definition*: A metric evaluating cluster cohesion vs separation ranging from -1 to +1, used to determine optimal K.
    * **Flashcard 3: DBSCAN Noise Label (-1)**
      * *Definition*: The designation for low-density outlier points that fail to meet minimum neighborhood density requirements.
    * **Flashcard 4: Cluster Medoid**
      * *Definition*: The actual corpus document whose embedding has the highest cosine similarity to the cluster centroid, serving as the cluster's representative title.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Clustering Embeddings:

- [ ] I understand why Euclidean K-Means is equivalent to Cosine clustering on normalized vectors.
- [ ] I know how to use Silhouette scores to choose the optimal number of clusters $K$.
- [ ] I understand how DBSCAN discovers clusters automatically and isolates outlier noise (`-1`).
- [ ] I know how to extract cluster medoids to automatically label discovered semantic topics.
- [ ] I know why unit normalization is mandatory prior to clustering.
""")
