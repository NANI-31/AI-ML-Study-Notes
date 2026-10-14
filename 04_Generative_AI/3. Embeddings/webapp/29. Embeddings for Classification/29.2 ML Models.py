import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

st.set_page_config(
    page_title="29.2 ML Models for Embeddings - Embeddings Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("29.2 ML Models: Logistic Regression, SVM, Random Forest, KNN & Neural Networks")
st.caption("Chapter 29: Embeddings for Classification | Domain: 3. Vector Embeddings")
st.markdown("""
Once text is transformed into dense high-dimensional vectors, choosing the right downstream machine learning algorithm 
is crucial for production efficiency, accuracy, and latency.

Different machine learning architectures interact with **dense, continuous, high-dimensional manifolds** in fundamentally different ways:
* **Linear Models (Logistic Regression & Linear SVM)**: Exploit Cover's Theorem to draw flat, highly regularized hyperplanes in microseconds.
* **Instance-Based Models (KNN)**: Classify samples based directly on nearest neighbors in metric space (analogous to vector database queries).
* **Tree-Based Models (Random Forest & XGBoost)**: Make axis-aligned orthogonal cuts; often struggle on dense embeddings without dimensionality reduction.
* **Neural Networks (MLP Classifier Heads)**: Learn non-linear feature interactions through hidden layers and non-linear activations.

This lesson analyzes the mathematical characteristics, empirical trade-offs, and production considerations of the five core classification architectures.
""")

st.divider()

# Section 1: Detailed Model Breakdown on Dense Embeddings
st.header("1. Comparative Analysis of the 5 Core Classifiers", divider="rainbow")

col_m1, col_m2 = col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("""
    ### 1. Logistic Regression (The Industry Default)
    * **How it works**: Learns a linear weight matrix $W$ and bias $b$, projecting input vectors into class logits, followed by Softmax:
      `P(y = c | x) = exp(W_c * x + b_c) / Sum_j exp(W_j * x + b_j)`
    * **Strengths**: Blazing fast training (< 1s on CPU); calibrated class probabilities (`predict_proba`); lightweight memory footprint (few kilobytes).
    * **When to use**: 90% of production text classification tasks!
    
    ### 2. Support Vector Machines (Linear SVM)
    * **How it works**: Solves a convex optimization problem to find the **maximum margin hyperplane** that separates classes while maximizing the distance to the closest data points (support vectors).
    * **Strengths**: Highly robust to extreme high dimensionality; resistant to outliers.
    * **Limitation**: Does not natively output calibrated probabilities (requires Platt scaling).
    * **When to use**: High-dimensional, sparse or small datasets where maximizing margin prevents overfitting.
    
    ### 3. K-Nearest Neighbors (KNN)
    * **How it works**: Non-parametric algorithm that memorizes training vectors. At inference, computes Cosine distance to all training vectors and assigns the majority vote of the $k$ closest neighbors.
    * **Strengths**: Zero training time; highly interpretable (you can inspect the exact training samples that justified the prediction).
    * **Limitation**: Inference scales linearly with dataset size $O(N \\cdot D)$ unless backed by an HNSW vector database.
    """)

with col_m2:
    st.markdown("""
    ### 4. Random Forest & Gradient Boosted Trees
    * **How it works**: An ensemble of decision trees where each node splits a single feature along an axis-aligned threshold (e.g., `feature_42 <= 0.15`).
    * **The Dense Embedding Pathology**: In dense vectors (e.g., 1536-d), semantic concepts are distributed across combinations of dimensions. Individual coordinates rarely hold standalone meaning.
    * Trees struggle to represent diagonal semantic boundaries with rectangular staircases!
    * **Remediation**: If using trees, apply **PCA** or **UMAP** to compress embeddings to 50 dimensions first.
    
    ### 5. Multi-Layer Perceptron (MLP) / Small Neural Head
    * **How it works**: Passes embeddings through 1 or 2 hidden layers (e.g., 256 neurons) with ReLU activations and Dropout:
      `h = ReLU(W_1 * x + b_1)` -> `logits = W_2 * h + b_2`
    * **Strengths**: Captures complex, non-linear interactions between embedding dimensions.
    * **When to use**: Large datasets (>10,000 samples per class) where non-linear patterns exceed linear separability.
    """)

st.markdown("""
### Comprehensive Model Benchmark Comparison Matrix
| Algorithm | Training Speed (10k samples) | Inference Latency (Single Sample) | Probability Calibration | Memory Footprint | Best Suited Scenario |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **Fastest (< 1 sec)** | **< 0.1 ms (CPU)** | **Native (Softmax)** | **Minimal (~50 KB)** | **Default choice for all production tasks** |
| **Linear SVM** | Fast (~ 2 sec) | < 0.1 ms (CPU) | Requires Platt scaling | Minimal (~50 KB) | Small datasets with tight margins |
| **K-Nearest Neighbors** | **Instant (0 sec)** | Slow (5 - 30 ms) | Weak (Frequency fraction) | Large (Stores all training vectors) | Interpretable nearest-example lookups |
| **Random Forest (100 trees)** | Moderate (15 - 45 sec) | Moderate (2 - 8 ms) | Coarse (Tree vote fraction) | Heavy (50 - 200 MB) | Tabular data with mixed embedding features |
| **MLP (2-layer Neural Head)** | Moderate (10 - 60 sec) | Fast (~ 0.5 ms CPU / GPU) | Native (Softmax) | Moderate (~1 - 5 MB) | Large datasets with non-linear class overlap |
""")

st.divider()

# Section 2: Code Implementation Across All 5 Algorithms
st.header("2. Code Implementation: The Unified Classifier Suite", divider="rainbow")

st.code("""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Standardized Suite of Classifiers on Dense Embedding Matrix X_train (shape: N, D)
classifiers = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
    "Linear SVM": LinearSVC(C=1.0, dual=False, max_iter=2000),
    "KNN (Cosine)": KNeighborsClassifier(n_neighbors=5, metric='cosine'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1),
    "MLP Neural Head": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, early_stopping=True)
}

# Train and evaluate in a unified loop
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))
""", language="python")

st.divider()

# Section 3: Interactive Workbench — The 5-Model Classifier Arena
st.header("3. Interactive Workbench: The 5-Model Classifier Arena", divider="rainbow")

st.markdown("""
Compare all five machine learning models live on a simulated multi-class IT & Support ticket dataset! 
Select a classifier to train it instantly, inspect training duration, evaluate accuracy, and test live predictions on custom user inputs.
""")

# Synthetic Multi-Class Dataset (4 Classes)
arena_dataset = [
    # Class 0: Cloud Infrastructure & DevOps
    {"text": "Kubernetes pods are failing with CrashLoopBackOff on node worker-4.", "cat": "DevOps"},
    {"text": "How do I configure Terraform state locking with AWS DynamoDB?", "cat": "DevOps"},
    {"text": "Docker daemon is unresponsive when running docker-compose build.", "cat": "DevOps"},
    {"text": "Envoy ingress proxy returns 502 bad gateway during service rollout.", "cat": "DevOps"},
    {"text": "Prometheus alert triggered for high node CPU utilization above 90 percent.", "cat": "DevOps"},
    
    # Class 1: Database & Storage
    {"text": "PostgreSQL query execution plan shows a sequential scan on large table.", "cat": "Database"},
    {"text": "Redis cluster failed over due to primary node heartbeat timeout.", "cat": "Database"},
    {"text": "How to create a composite B-Tree index in MySQL for multi-column filters?", "cat": "Database"},
    {"text": "Deadlock detected between concurrent transactions updating orders table.", "cat": "Database"},
    {"text": "MongoDB replica set sync lag is growing on secondary node.", "cat": "Database"},
    
    # Class 2: Account & Security
    {"text": "Please reset my account MFA authenticator as my phone was stolen.", "cat": "Security"},
    {"text": "Unrecognized login attempt detected from an unknown IP address in London.", "cat": "Security"},
    {"text": "How do I rotate enterprise SAML single sign-on signing certificates?", "cat": "Security"},
    {"text": "Revoke API access tokens for a former employee whose contract ended.", "cat": "Security"},
    {"text": "Suspicious privilege escalation alert triggered on root IAM user.", "cat": "Security"},
    
    # Class 3: Billing & Finance
    {"text": "We were charged twice for the annual enterprise cloud subscription.", "cat": "Billing"},
    {"text": "Where can our finance team download the VAT tax invoice for last month?", "cat": "Billing"},
    {"text": "How do I update our corporate credit card on file before the renewal date?", "cat": "Billing"},
    {"text": "Our payment method was declined due to bank transaction limits.", "cat": "Billing"},
    {"text": "What is the per-seat discount rate if we upgrade to 250 user seats?", "cat": "Billing"}
]

# Generate synthetic 16-d semantic embeddings
np.random.seed(101)
class_centroids = {
    "DevOps": np.array([0.9, 0.8, 0.1, 0.1, 0.8, 0.9, 0.1, 0.1, 0.2, 0.8, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1]),
    "Database": np.array([0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.1, 0.1, 0.9, 0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1]),
    "Security": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
    "Billing": np.array([0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
}

X_arena = []
y_arena = []

for item in arena_dataset:
    base = class_centroids[item["cat"]]
    noise = np.random.normal(0, 0.14, 16)
    vec = base + noise
    norm_vec = vec / np.linalg.norm(vec)
    X_arena.append(norm_vec)
    y_arena.append(item["cat"])

X_arena = np.array(X_arena)
y_arena = np.array(y_arena)

col_ctrl, col_view = col_ctrl, col_view = st.columns([1, 1])

with col_ctrl:
    st.subheader("Model Configuration & Training")
    selected_model_name = st.selectbox(
        "Select Model Architecture to Fit:",
        ["Logistic Regression", "Linear SVM", "KNN (k=3, Cosine)", "Random Forest (50 Trees)", "MLP Neural Head (2-Layer)"]
    )
    
    # Train the chosen model
    start_t = time.perf_counter()
    if selected_model_name == "Logistic Regression":
        active_clf = LogisticRegression(C=2.0, max_iter=200)
    elif selected_model_name == "Linear SVM":
        active_clf = LinearSVC(C=1.0, dual=False, max_iter=500)
    elif selected_model_name == "KNN (k=3, Cosine)":
        active_clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    elif selected_model_name == "Random Forest (50 Trees)":
        active_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    else:
        active_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        
    active_clf.fit(X_arena, y_arena)
    fit_duration_ms = (time.perf_counter() - start_t) * 1000
    
    preds_train = active_clf.predict(X_arena)
    acc = accuracy_score(y_arena, preds_train)
    
    st.success(f"""
    **Architecture**: `{selected_model_name}`  
    **Training Duration**: **{fit_duration_ms:.2f} ms**  
    **Training Set Accuracy**: **{acc * 100:.1f}%** ({sum(preds_train == y_arena)} / {len(y_arena)} correct)
    """)
    
    st.markdown("### Interactive Inference Test:")
    test_presets = [
        "How do I fix a CrashLoopBackOff error on a Kubernetes deployment?",
        "Postgres database is slow because of table locking during bulk updates.",
        "A malicious hacker is trying to brute-force our SSH login server.",
        "We need an official tax invoice receipt for our monthly subscription payment.",
        "Docker container memory consumption is exceeding limits."
    ]
    chosen_test = st.selectbox("Choose a Test Prompt or Enter Custom:", test_presets + ["Custom Prompt"])
    if chosen_test == "Custom Prompt":
        eval_text = st.text_input("Enter ticket text:", value="Where can I download the monthly receipt?")
    else:
        eval_text = chosen_test

with col_view:
    st.subheader("Live Inference & Decision Breakdown")
    
    # Map input to synthetic vector
    def project_test_text(t: str) -> np.ndarray:
        t_low = t.lower()
        if any(w in t_low for w in ["kubernetes", "pod", "docker", "terraform", "ingress", "envoy", "node"]):
            c = class_centroids["DevOps"]
        elif any(w in t_low for w in ["postgres", "database", "redis", "mysql", "lock", "table", "sql"]):
            c = class_centroids["Database"]
        elif any(w in t_low for w in ["security", "mfa", "login", "hacker", "ssh", "iam", "token", "saml"]):
            c = class_centroids["Security"]
        else:
            c = class_centroids["Billing"]
        noise = np.random.normal(0, 0.12, 16)
        vec = c + noise
        return vec / np.linalg.norm(vec)

    test_v = project_test_text(eval_text)
    
    inf_start = time.perf_counter()
    pred_class = active_clf.predict([test_v])[0]
    inf_duration_us = (time.perf_counter() - inf_start) * 1_000_000
    
    st.markdown(f"### Predicted Class: `{pred_class}` :trophy:")
    st.info(f"**Inference Latency**: **{inf_duration_us:.1f} microseconds** on CPU")
    
    # Check if model supports predict_proba
    if hasattr(active_clf, "predict_proba"):
        probs = active_clf.predict_proba([test_v])[0]
        df_p = pd.DataFrame({
            "Class": active_clf.classes_,
            "Confidence": [f"{p*100:.1f}%" for p in probs],
            "Score": probs
        }).sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.dataframe(df_p[["Class", "Confidence"]], use_container_width=True)
    else:
        st.caption("*(Note: Linear SVM does not compute native probabilities; uses decision function margins)*")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why do Decision Trees and Random Forests often perform worse than Logistic Regression on dense text embeddings?", expanded=True):
    st.markdown("""
    **Answer:**
    Decision Trees make **axis-aligned orthogonal splits** along a single feature at a time:
    * `If Feature_42 <= 0.12 then Branch A else Branch B`.
    * In tabular data (e.g., Age, Income, Credit Score), individual features have independent physical meanings where rectangular thresholds are natural.
    * In **dense text embeddings** (such as 1536-d vectors from OpenAI or BGE), semantic concepts are **distributed representations**—meaning is encoded as complex linear combinations across hundreds of coordinates simultaneously.
    * An axis-aligned tree must construct a massive, inefficient "staircase" of hundreds of orthogonal splits to approximate a single diagonal hyperplane.
    * Consequently, Random Forests on raw dense embeddings tend to overfit, train slowly, and achieve 3% to 8% lower accuracy than simple linear models.
    """)

with st.expander("Q2: How does K-Nearest Neighbors (KNN) classification relate to Vector Databases?", expanded=True):
    st.markdown("""
    **Answer:**
    **Vector Database retrieval is mathematically equivalent to exact or approximate K-Nearest Neighbors classification!**
    * In classic KNN, you store all training pairs $(x_i, y_i)$. When a test sample $x_{new}$ arrives, you calculate the distance to all $N$ points and return the majority vote among the top $k$.
    * In an enterprise vector database (e.g., Milvus, Qdrant, Pinecone), you store document vectors with class metadata:
      `collection.query(query_embedding=x_new, n_results=k)`
    * The vector database uses **HNSW graph traversal** to find the $k$ nearest neighbors in sub-linear $O(\\log N)$ time (instead of $O(N)$ brute-force).
    * Taking the majority class among the top $k$ retrieved metadata items provides an instant, zero-training KNN classifier directly inside your search infrastructure!
    """)

with st.expander("Q3: When would you add a Multi-Layer Perceptron (MLP) head instead of Logistic Regression?", expanded=True):
    st.markdown("""
    **Answer:**
    An MLP classifier head (e.g., 2 hidden layers with 256 and 128 neurons and ReLU activations) should be chosen over Logistic Regression when:
    1. **Multi-Modal or Multi-Attribute Fusion**: You are concatenating a text embedding with numerical metadata (e.g., user tenure, price, geolocation coordinates). An MLP learns non-linear interactions between the continuous text vector and the discrete tabular numbers.
    2. **Multi-Label Classification with Complex Dependencies**: Predicting multiple overlapping tags simultaneously (e.g., an article being both *'DevOps'*, *'Kubernetes'*, and *'Tutorial'*) where tag co-occurrence requires shared hidden representations.
    3. **Abundant Labeled Data (>20,000 samples)**: When data is abundant, an MLP can model subtle non-linear manifold curvatures that a flat hyperplane cannot capture.
    """)

with st.expander("Q4: Why is Linear SVM preferred over Logistic Regression in certain legal or clinical compliance settings?"):
    st.markdown("""
    **Answer:**
    Linear SVM optimizes the **Hinge Loss**, which strictly maximizes the geometric margin between the decision boundary and the nearest support vectors:
    * Logistic Regression tries to maximize the log-likelihood across *all* training points, meaning distant outliers can exert a small cumulative pull on the decision boundary.
    * Linear SVM only cares about the data points right along the boundary margin (the support vectors); points far away in the correct half-space contribute **zero loss**.
    * This gives Linear SVM higher stability and tighter worst-case error bounds in safety-critical classification where class separation must be as wide as possible.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Running Brute-Force KNN on 1,000,000 Samples in a Live API**: Calling `scikit-learn`'s `KNeighborsClassifier` at request time. It computes 1,000,000 dot products per request, causing 500ms+ latency and crashing under traffic spikes. Always use an ANN vector index (HNSW) for large-scale KNN.
* **Trap 2: Training Trees on 1536 Dimensions Without Pruning**: Setting `max_depth=None` on a Random Forest with 1536 dense features. Trees will expand to millions of nodes, consuming gigabytes of RAM and severely overfitting.
* **Trap 3: Forgetting Early Stopping in MLP Heads**: Training a dense neural network for 500 epochs without early stopping on validation loss. The network quickly memorizes the training embeddings and suffers catastrophic generalization drop on unseen test prompts.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Linear Models"):
        st.markdown("**Front**: Why are linear models the top choice on embeddings?")
        st.divider()
        st.markdown("**Back**: Pre-trained embeddings are already linearly separable; linear models train in seconds on CPU and execute in sub-0.1ms.")

with card_cols[1]:
    with st.popover("Card 2: Trees on Embeddings"):
        st.markdown("**Front**: Why do Random Forests struggle on dense vectors?")
        st.divider()
        st.markdown("**Back**: They make single-feature orthogonal splits, which struggle to capture diagonal concepts distributed across hundreds of continuous dimensions.")

with card_cols[2]:
    with st.popover("Card 3: Vector DB = KNN"):
        st.markdown("**Front**: How does a Vector Database implement KNN?")
        st.divider()
        st.markdown("**Back**: ANN search retrieves the top-k nearest neighbors in O(log N) time; taking the majority class label gives instant KNN classification.")

with card_cols[3]:
    with st.popover("Card 4: SVM Margin"):
        st.markdown("**Front**: What is the core optimization goal of Linear SVM?")
        st.divider()
        st.markdown("**Back**: Maximizes the geometric margin between classes, focusing strictly on boundary support vectors rather than global averages.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master ML Models for Embeddings:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand why Logistic Regression is the default choice on dense embeddings.", key="chk29_2_1")
    st.checkbox("I know why Random Forests struggle on dense vectors without PCA.", key="chk29_2_2")
    st.checkbox("I can implement a unified benchmark suite across all 5 classifiers in scikit-learn.", key="chk29_2_3")
with c2:
    st.checkbox("I understand the equivalence between HNSW Vector Search and KNN classification.", key="chk29_2_4")
    st.checkbox("I know when to deploy an MLP neural head for multi-modal feature fusion.", key="chk29_2_5")
    st.checkbox("I understand Linear SVM's maximum-margin objective and hinge loss.", key="chk29_2_6")
