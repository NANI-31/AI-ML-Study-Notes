import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="30.2 Clustering Algorithms - Embeddings Curriculum",
    page_icon="🔬",
    layout="wide"
)

# Title & Overview
st.title("30.2 Clustering Algorithms: K-Means, DBSCAN, Hierarchical & Gaussian Mixture Models")
st.caption("Chapter 30: Embeddings for Clustering | Domain: 3. Vector Embeddings")
st.markdown("""
Not all clustering algorithms perform equally when operating on **continuous, high-dimensional embedding spaces** (typically 768 to 1536 dimensions).

Different clustering paradigms make fundamentally different assumptions about geometric shape, cluster density, and noise:
* **K-Means**: Assumes convex, spherical clusters of roughly equal variance; forces every sample into a cluster.
* **DBSCAN & HDBSCAN**: Groups points based on dense spatial connectivity; explicitly separates true clusters from unstructured noise points.
* **Hierarchical (Agglomerative)**: Constructs a deterministic bottom-up tree of semantic clusters, generating multi-level topic taxonomies.
* **Gaussian Mixture Models (GMM)**: Soft, probabilistic clustering where documents can belong partially to multiple topics simultaneously.

Selecting the optimal algorithm depends on whether you know the number of clusters in advance, whether your data contains outliers, 
and whether you require hard partitions or soft multi-topic distributions.
""")

st.divider()

# Section 1: Deep Algorithmic Mechanics on Dense Vectors
st.header("1. Deep Algorithmic Breakdown on High-Dimensional Manifolds", divider="rainbow")

col_alg1, col_alg2 = col_alg1, col_alg2 = st.columns([1, 1])

with col_alg1:
    st.markdown("""
    ### 1. K-Means & Spherical K-Means
    * **Mechanism**: Partitions space into $K$ Voronoi cells by iteratively updating centroids to the mean of assigned vectors.
    * **High-Dim Behavior**: When normalized to the unit sphere ($||v|| = 1$), it minimizes cosine distance.
    * **Best for**: Large corpora (scales to millions via `MiniBatchKMeans`) when $K$ is approximately known in advance.
    * **Weakness**: Cannot detect noise; forces outliers into the nearest cluster; assumes equal-sized spherical clusters.
    
    ### 2. DBSCAN & HDBSCAN (Density-Based)
    * **Mechanism**: Finds core points with at least `min_samples` neighbors within radius `eps`. Expands clusters along continuous density paths.
    * **The Curse of Dimensionality in DBSCAN**: In 1536 dimensions, all pairwise distances concentrate around $\\sqrt{2} \\approx 1.414$, making standard `eps` hyperparameter tuning extremely fragile!
    * **Industry Solution**: Use **UMAP** to project embeddings to 5-15 dimensions, then run **HDBSCAN** (Hierarchical DBSCAN).
    * **Best for**: Discovering the natural number of topics and discarding unclassifiable noise/spam into a `-1` label.
    """)

with col_alg2:
    st.markdown("""
    ### 3. Agglomerative Hierarchical Clustering
    * **Mechanism**: Starts with every document as its own cluster. Recursively merges the two closest clusters according to a linkage criterion:
      * **Ward's Linkage**: Minimizes the total within-cluster variance.
      * **Average / Complete Linkage**: Measures average or maximum pairwise distance between clusters.
    * **Best for**: Creating hierarchical taxonomies and subtopic trees (e.g., *DevOps -> Containers -> Kubernetes -> Pod Networking*).
    * **Limitation**: $O(N^2)$ memory and $O(N^3)$ compute complexity limits scaling to $< 20,000$ documents.
    
    ### 4. Gaussian Mixture Models (GMM)
    * **Mechanism**: Models the data as a mixture of $K$ multivariate Gaussian distributions using the Expectation-Maximization (EM) algorithm.
    * **Soft Clustering Advantage**: Instead of a hard binary assignment, GMM outputs a probability vector:
      `[P(Topic_1) = 0.65, P(Topic_2) = 0.30, P(Topic_3) = 0.05]`.
    * Ideal for multi-faceted documents discussing multiple related concepts.
    """)

st.markdown("""
### Algorithmic Comparison Matrix for Embedding Clustering
| Algorithm | Scalability (Complexity) | Requires Specifying K? | Handles Noise / Outliers? | Cluster Geometry | Primary Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spherical K-Means** | **Fastest: $O(N \\cdot K \\cdot D)$** | **Yes** | No (Forces all into clusters) | Spherical Voronoi cells | Hard cluster IDs + Centroids |
| **HDBSCAN (via UMAP)**| Moderate: $O(N \\log N)$ | **No (Discovers K naturally)**| **Yes (Labels noise as -1)** | Arbitrary non-linear shapes | Hard IDs + Noise filter |
| **Agglomerative (Ward)**| Slow: $O(N^2)$ Memory | Optional (Cut tree at height) | No | Hierarchical nested tree | Full topic tree (Dendrogram) |
| **Gaussian Mixture (GMM)**| Moderate: $O(N \\cdot K \\cdot D)$ | **Yes** | Coarse (Low density probabilities) | Ellipsoidal (Soft Gaussian) | Soft probabilities across topics |
""")

st.divider()

# Section 2: Code Implementation Across the 4 Algorithms
st.header("2. Code Implementation: The 4-Algorithm Clustering Suite", divider="rainbow")

st.code("""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Assume X is an L2-normalized embedding matrix: X = X / np.linalg.norm(X, axis=1, keepdims=True)

# 1. Spherical K-Means (K=5)
kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# 2. DBSCAN (Cosine Metric)
# eps: maximum cosine distance (e.g. 0.45); min_samples: minimum points in neighborhood
dbscan = DBSCAN(eps=0.45, min_samples=3, metric='cosine')
labels_dbscan = dbscan.fit_predict(X) # Noise points assigned label -1

# 3. Agglomerative Hierarchical Clustering (Ward linkage requires Euclidean on normalized vectors)
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X)

# 4. Gaussian Mixture Model (Soft Probabilistic Clustering)
gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=42)
gmm.fit(X)
soft_probabilities = gmm.predict_proba(X) # Shape: (N, 5)
labels_gmm = np.argmax(soft_probabilities, axis=1)
""", language="python")

st.divider()

# Section 3: Interactive Workbench — The 4-Algorithm Clustering Arena
st.header("3. Interactive Workbench: The 4-Algorithm Clustering Arena", divider="rainbow")

st.markdown("""
Test how all four algorithms cluster the same set of enterprise document embeddings. 
Switch algorithms, tune their hyperparameters live, and compare how they handle cluster separation, noise detection, and group sizes.
""")

# Synthetic Multi-Topic Dataset with Noise
synthetic_corpus_data = [
    # Cluster 1: Payment & Billing
    {"text": "Why was our credit card charged twice for the monthly cloud subscription?", "domain": "Billing"},
    {"text": "We need an updated VAT invoice for our finance department audit.", "domain": "Billing"},
    {"text": "How do I cancel our annual plan and request a pro-rated refund?", "domain": "Billing"},
    {"text": "Credit card payment declined by bank due to international limit.", "domain": "Billing"},
    {"text": "Where can our accountant download past payment receipts?", "domain": "Billing"},
    
    # Cluster 2: DevOps & Containers
    {"text": "Kubernetes pods are failing with OOMKilled exit code 137 under load.", "domain": "DevOps"},
    {"text": "Envoy ingress proxy intermittently returns 504 Gateway Timeout.", "domain": "DevOps"},
    {"text": "Docker container fails to mount persistent volume on AWS EBS.", "domain": "DevOps"},
    {"text": "How to tune Horizontal Pod Autoscaler stabilization window in Helm?", "domain": "DevOps"},
    {"text": "Prometheus alert firing for memory pressure on worker nodes.", "domain": "DevOps"},
    
    # Cluster 3: Security & IAM
    {"text": "User lost MFA authenticator app and cannot complete login.", "domain": "Security"},
    {"text": "How do we rotate compromised AWS IAM secret access keys?", "domain": "Security"},
    {"text": "SAML 2.0 signature validation mismatch error with Okta SSO.", "domain": "Security"},
    {"text": "Suspicious login attempt detected from an unknown overseas IP.", "domain": "Security"},
    {"text": "Revoke API access tokens for a contractor who left the company.", "domain": "Security"},
    
    # Noise / Outliers (Random unclassifiable queries)
    {"text": "What is the capital of Australia and the local time zone in Sydney?", "domain": "Noise / Outlier"},
    {"text": "Best Italian pasta recipe with garlic, olive oil, and fresh basil.", "domain": "Noise / Outlier"}
]

# Generate synthetic 16-d semantic embeddings
np.random.seed(99)
prototypes = {
    "Billing": np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1]),
    "DevOps": np.array([0.1, 0.1, 0.9, 0.8, 0.7, 0.9, 0.1, 0.1, 0.1, 0.9, 0.8, 0.9, 0.1, 0.9, 0.8, 0.1]),
    "Security": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.9]),
    "Noise / Outlier": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
}

X_bench = []
for item in synthetic_corpus_data:
    base = prototypes[item["domain"]]
    noise = np.random.normal(0, 0.11, 16)
    vec = base + noise
    X_bench.append(vec / np.linalg.norm(vec))

X_bench = np.array(X_bench)
doc_texts = [d["text"] for d in synthetic_corpus_data]

col_algo_cfg, col_algo_res = col_algo_cfg, col_algo_res = st.columns([1, 1])

with col_algo_cfg:
    st.subheader("Algorithm & Hyperparameter Selection")
    chosen_algo = st.selectbox(
        "Choose Clustering Algorithm:",
        ["Spherical K-Means", "DBSCAN (Density-Based)", "Agglomerative Hierarchical", "Gaussian Mixture Model (GMM)"]
    )
    
    if chosen_algo == "Spherical K-Means":
        k_val = st.slider("Number of Clusters (K):", 2, 6, 3)
        model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = model.fit_predict(X_bench)
        
    elif chosen_algo == "DBSCAN (Density-Based)":
        eps_val = st.slider("Epsilon (Cosine Neighborhood Radius):", 0.20, 0.80, 0.45, 0.05)
        min_samples_val = st.slider("Min Samples in Neighborhood:", 2, 5, 3)
        model = DBSCAN(eps=eps_val, min_samples=min_samples_val, metric="cosine")
        labels = model.fit_predict(X_bench)
        
    elif chosen_algo == "Agglomerative Hierarchical":
        k_hier = st.slider("Number of Clusters (Cut Height):", 2, 6, 3)
        linkage_type = st.selectbox("Linkage Metric:", ["ward", "average", "complete"])
        model = AgglomerativeClustering(n_clusters=k_hier, linkage=linkage_type)
        labels = model.fit_predict(X_bench)
        
    else: # GMM
        k_gmm = st.slider("Number of Gaussian Components (K):", 2, 6, 3)
        cov_type = st.selectbox("Covariance Type:", ["diag", "spherical", "tied"])
        model = GaussianMixture(n_components=k_gmm, covariance_type=cov_type, random_state=42)
        model.fit(X_bench)
        soft_probs = model.predict_proba(X_bench)
        labels = model.predict(X_bench)

    # Compute validation metrics
    unique_labels = set(labels)
    n_clusters_found = len(unique_labels - {-1})
    noise_count = sum(labels == -1)
    
    st.markdown("### Algorithmic Execution Metrics:")
    m1, m2 = st.columns(2)
    m1.metric("Clusters Discovered", n_clusters_found)
    m2.metric("Outliers / Noise Identified", noise_count)
    
    if n_clusters_found > 1:
        # Exclude noise for silhouette if applicable
        if -1 in unique_labels:
            non_noise_mask = (labels != -1)
            if len(set(labels[non_noise_mask])) > 1:
                sil = silhouette_score(X_bench[non_noise_mask], labels[non_noise_mask], metric="cosine")
                st.metric("Silhouette Score (Excl. Noise)", f"{sil:.3f}")
        else:
            sil = silhouette_score(X_bench, labels, metric="cosine")
            st.metric("Silhouette Score", f"{sil:.3f}")

with col_algo_res:
    st.subheader("Cluster Assignments & Noise Detection")
    
    # Build results table
    results_list = []
    for i, doc_item in enumerate(synthetic_corpus_data):
        lbl = labels[i]
        lbl_str = f"Cluster {lbl + 1}" if lbl != -1 else "NOISE / OUTLIER (-1)"
        
        row = {
            "Assigned Cluster": lbl_str,
            "True Category": doc_item["domain"],
            "Document Text": doc_item["text"][:75] + "..."
        }
        results_list.append(row)
        
    df_algo_res = pd.DataFrame(results_list)
    st.dataframe(df_algo_res, use_container_width=True, height=350)
    
    if chosen_algo == "DBSCAN (Density-Based)" and noise_count > 0:
        st.success(f"""
        **Noise Handling Demonstrated**: Notice how DBSCAN correctly labeled the two random culinary & geography questions as **NOISE (-1)** without forcing them into technical clusters!
        """)
    elif chosen_algo == "Spherical K-Means":
        st.caption("""
        *Notice*: K-Means forces every sample—including the pasta recipe and Australia question—into one of the clusters, distorting cluster centroids.
        """)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why does standard Euclidean DBSCAN struggle in raw 1536-dimensional embedding space, and how does the BERTopic architecture solve it?", expanded=True):
    st.markdown("""
    **Answer:**
    * **The Curse of Dimensionality & Distance Concentration**: In high-dimensional spaces (e.g., 1536 dimensions), pairwise Euclidean distances between random vectors concentrate tightly within a narrow numerical band (e.g., between 1.38 and 1.44). The ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1.0.
    * Because density is defined by the number of points within radius `eps`, a tiny change in `eps` (e.g., from 0.42 to 0.44) causes DBSCAN to swing erratically from classifying **100% of points as noise** to merging **100% of points into a single giant mega-cluster**!
    
    **The BERTopic Solution**:
    1. **Non-Linear Manifold Reduction (UMAP)**: Project 1536-dimensional embeddings down to 5 to 15 dimensions. UMAP preserves both local and global neighborhood topology while destroying the curse of dimensionality.
    2. **Hierarchical Density-Based Clustering (HDBSCAN)**: HDBSCAN operates across varying density thresholds, automatically discovering clusters of different densities without needing a fixed global `eps` parameter, cleanly filtering noise.
    """)

with st.expander("Q2: When should Gaussian Mixture Models (GMM) be used instead of K-Means for text documents?"):
    st.markdown("""
    **Answer:**
    GMM should be selected when documents are **thematically polysemous** (multi-intent) and hard boundary cuts are inappropriate:
    * In K-Means, a document discussing *"Kubernetes Security and Okta SAML Integration"* must be assigned 100% to either *DevOps* or *Security*.
    * GMM assigns **soft posterior probabilities**:
      `P(DevOps | x) = 0.52, P(Security | x) = 0.44, P(Billing | x) = 0.04`
    * This allows downstream systems to route the document to both engineering queues or flag it as cross-disciplinary.
    * *Trade-off*: GMM requires fitting full or diagonal covariance matrices, requiring $O(K \\cdot D^2)$ parameters. In 1536 dimensions, you must use `covariance_type='diag'` or reduce dimensions via PCA/UMAP first to prevent covariance matrix singularity.
    """)

with st.expander("Q3: What are the trade-offs between Ward's linkage and Average linkage in Agglomerative Hierarchical clustering?"):
    st.markdown("""
    **Answer:**
    * **Ward's Linkage**:
      * Merges the pair of clusters that leads to the smallest increase in total within-cluster variance.
      * *Properties*: Produces balanced, compact, spherical clusters of roughly equal sizes. Highly robust to noise.
      * *Limitation*: Mathematically requires Euclidean distance (equivalent to cosine distance on normalized vectors).
    * **Average Linkage (UPGMA)**:
      * Merges clusters with the smallest average pairwise distance between all member points.
      * *Properties*: Can handle non-spherical, elongated clusters and arbitrary distance metrics (e.g., pre-computed Cosine similarity matrices).
      * *Limitation*: Can suffer from "chaining" effects where outlier points slowly attach one-by-one to an existing cluster.
    """)

with st.expander("Q4: How do you scale K-Means to cluster 50 million text embeddings in production?"):
    st.markdown("""
    **Answer:**
    To cluster 50 million 768-d vectors (which consume ~150 GB of RAM):
    1. **MiniBatch K-Means**: Uses small random sub-samples (e.g., 2,048 samples per step) to iteratively update cluster centroids using convex gradient steps, avoiding loading the entire dataset into memory.
    2. **GPU-Accelerated Clustering (FAISS / RAPIDS cuML)**: FAISS provides `faiss.Clustering` executing highly optimized Spherical K-Means directly on GPU VRAM, clustering 10 million vectors in under 2 minutes.
    3. **Vector Quantization (IVF-PQ Centroids)**: Vector databases like Milvus and Qdrant use K-Means clustering as the core indexing primitive for Inverted File indexes (`nlist = 4096`), training centroids on a sample and assigning millions of vectors in parallel.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Running GMM with Full Covariance on 1536 Dimensions**: Setting `covariance_type='full'` on 1536-d vectors. Fitting a $1536 \\times 1536$ covariance matrix per cluster requires over 2.3 million parameters per component, causing immediate out-of-memory crashes or singular matrix errors. Always use `covariance_type='diag'`.
* **Trap 2: Using Agglomerative Clustering on Large Datasets**: Calling `AgglomerativeClustering` on 100,000 documents. The $O(N^2)$ distance matrix requires storing 10 billion floats (~40 GB RAM) and takes hours to link. Cap hierarchical clustering at $N \\le 15,000$.
* **Trap 3: Forgetting That K-Means Has No Noise Concept**: Using K-Means to discover emerging customer issues. Malicious bot spam and corrupted strings will be forced into customer topics, corrupting the cluster medoids.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: DBSCAN Noise"):
        st.markdown("**Front**: How does DBSCAN handle unclassifiable outliers?")
        st.divider()
        st.markdown("**Back**: It assigns them a label of **-1 (Noise)**, protecting true semantic clusters from corruption.")

with card_cols[1]:
    with st.popover("Card 2: UMAP + HDBSCAN"):
        st.markdown("**Front**: Why reduce dimensions with UMAP before HDBSCAN?")
        st.divider()
        st.markdown("**Back**: Overcomes the curse of dimensionality where high-dimensional distance concentration makes density estimation impossible.")

with card_cols[2]:
    with st.popover("Card 3: GMM Soft Clustering"):
        st.markdown("**Front**: What is the primary output of a Gaussian Mixture Model?")
        st.divider()
        st.markdown("**Back**: Soft posterior probabilities across all K components, allowing documents to belong partially to multiple topics.")

with card_cols[3]:
    with st.popover("Card 4: Ward's Linkage"):
        st.markdown("**Front**: What does Ward's Linkage minimize in Hierarchical clustering?")
        st.divider()
        st.markdown("**Back**: The increase in total within-cluster variance, generating compact, balanced, and robust cluster trees.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Clustering Algorithms on Embeddings:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4 clustering paradigms (K-Means, DBSCAN, Hierarchical, GMM).", key="chk30_2_1")
    st.checkbox("I know why UMAP is applied prior to HDBSCAN in BERTopic.", key="chk30_2_2")
    st.checkbox("I can implement density-based noise filtering with DBSCAN.", key="chk30_2_3")
with c2:
    st.checkbox("I understand soft probabilistic topic assignment with GMMs.", key="chk30_2_4")
    st.checkbox("I know how Agglomerative clustering builds hierarchical topic dendrograms.", key="chk30_2_5")
    st.checkbox("I understand how to scale K-Means to millions of vectors using MiniBatch / FAISS.", key="chk30_2_6")
