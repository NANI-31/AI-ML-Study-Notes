import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="1.2 Why Embeddings are Important - Embeddings Curriculum",
    page_icon="⭐",
    layout="wide"
)

# Title & Overview
st.title("1.2 Why Embeddings are Important: Semantic Search, Recommendations, Clustering, & Duplicate Detection")
st.caption("Chapter 1: Introduction to Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Embeddings have revolutionized modern software architecture. Before neural embeddings, handling text required 
fragile regular expressions, handcrafted linguistic features, manual synonym dictionaries, and rigid keyword inverted indexes.

Today, **Vector Embeddings act as the universal semantic substrate** for intelligent applications:
1. **Semantic Search & Document Retrieval**: Answering queries based on conceptual intent rather than identical keywords.
2. **Recommendation Systems**: Matching user interest vectors with item catalog vectors in real time.
3. **Retrieval-Augmented Generation (RAG)**: Powering context retrieval for frontier LLMs to eliminate hallucinations.
4. **Unsupervised Topic Clustering**: Discovering emergent customer trends and support ticket clusters without human labeling.
5. **Few-Shot Classification**: Using pre-trained embeddings as static feature vectors to build high-accuracy classifiers with only 10 labeled examples.
6. **Near-Duplicate Detection**: Automatically identifying paraphrased questions, duplicate bug reports, and plagiarized text.

This lesson explores why embeddings are foundational to modern AI architectures and demonstrates their utility across core tasks.
""")

st.divider()

# Section 1: The Multi-Task Architecture
st.header("1. One Representation, Many Capabilities", divider="rainbow")

col_cap1, col_cap2 = st.columns([1, 1])

with col_cap1:
    st.markdown("""
    ### Universal Utility of Vector Embeddings:
    * **1. Asymmetry Handling in Search**:
      Mapping short 3-word user questions to 500-word technical manual paragraphs within a shared geometric space.
    * **2. Transfer Learning Efficiency**:
      Instead of training a billion-parameter model for every task, a single frozen embedding model transforms text into fixed vectors that feed lightweight linear classifiers, clustering algorithms, and vector stores.
    * **3. Real-Time Latency**:
      Once documents are pre-embedded and indexed in an Approximate Nearest Neighbor (ANN) index, similarity searches execute in **under 5 milliseconds** over millions of vectors.
    """)

with col_cap2:
    st.markdown("### Universal Embedding Ecosystem")
    eco_dot = """
    digraph EmbeddingEcosystem {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        TextIn [label="Raw Text / Document", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        EmbModel [label="Pre-trained Embedding Model\\n(Transforms text to dense vector)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Vector [label="Dense Vector Coordinate\\n(1536-Dimensional)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Search [label="Semantic Search & RAG", fillcolor="#DCFCE7", color="#16A34A"];
        RecSys [label="Personalized Recommendations", fillcolor="#DCFCE7", color="#16A34A"];
        Cluster [label="Unsupervised Clustering", fillcolor="#DCFCE7", color="#16A34A"];
        Dedup [label="Near-Duplicate Detection", fillcolor="#DCFCE7", color="#16A34A"];

        TextIn -> EmbModel -> Vector;
        Vector -> Search;
        Vector -> RecSys;
        Vector -> Cluster;
        Vector -> Dedup;
    }
    """
    st.graphviz_chart(eco_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Multi-Task Embedding Engine in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates how a single embedding representation powers semantic search, duplicate detection, and topic clustering.
""")

class ProductionMultiTaskEmbeddingEngine:
    """
    Demonstrates multi-task utilization of vector embeddings:
    Semantic search, near-duplicate detection, and cosine ranking.
    """
    def __init__(self):
        # Sample document corpus with pre-computed semantic features
        self.corpus = [
            {"id": "DOC-1", "title": "How to reset your password", "category": "IT Support", "vector": np.array([0.90, 0.10, 0.20, 0.15])},
            {"id": "DOC-2", "title": "Instructions for changing account credentials", "category": "IT Support", "vector": np.array([0.88, 0.12, 0.18, 0.14])},
            {"id": "DOC-3", "title": "Setting up two-factor authentication (2FA)", "category": "Security", "vector": np.array([0.70, 0.60, 0.10, 0.20])},
            {"id": "DOC-4", "title": "Quarterly financial revenue report 2024", "category": "Finance", "vector": np.array([0.10, 0.85, 0.90, 0.05])},
            {"id": "DOC-5", "title": "Corporate expense reimbursement guidelines", "category": "Finance", "vector": np.array([0.15, 0.80, 0.75, 0.10])},
            {"id": "DOC-6", "title": "Delicious employee lunch menu specials", "category": "Facilities", "vector": np.array([0.05, 0.10, 0.05, 0.95])}
        ]
        # Normalize all corpus vectors to unit length
        for item in self.corpus:
            v = item["vector"]
            item["vector"] = v / np.linalg.norm(v)

    def semantic_search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Ranks documents by cosine similarity to query vector."""
        q_norm = query_vector / np.linalg.norm(query_vector)
        scored = []
        for doc in self.corpus:
            score = float(np.dot(q_norm, doc["vector"]))
            scored.append({**doc, "similarity": round(score, 4)})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def detect_duplicates(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Identifies near-duplicate document pairs exceeding cosine threshold."""
        duplicates = []
        n = len(self.corpus)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(np.dot(self.corpus[i]["vector"], self.corpus[j]["vector"]))
                if score >= threshold:
                    duplicates.append((self.corpus[i]["title"], self.corpus[j]["title"], round(score, 4)))
        return duplicates

code_snippet = r'''
import numpy as np

class MultiTaskEmbeddingRunner:
    def semantic_search(self, query_vec: np.ndarray, doc_vectors: np.ndarray, top_k: int = 3):
        # High-speed matrix-vector dot product for normalized vectors
        scores = np.dot(doc_vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(top_indices[i], scores[top_indices[i]]) for i in range(top_k)]

    def find_near_duplicates(self, doc_vectors: np.ndarray, threshold: float = 0.95):
        # Pairwise cosine similarity matrix
        sim_matrix = np.dot(doc_vectors, doc_vectors.T)
        duplicates = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((i, j, sim_matrix[i, j]))
        return duplicates
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Multi-Task Embedding Workbench
st.header("3. Interactive Multi-Task Embedding Workbench", divider="rainbow")
st.markdown("""
Explore how the exact same vector representations power **Semantic Search**, **Near-Duplicate Detection**, and **Categorical Clustering**.
""")

task_engine = ProductionMultiTaskEmbeddingEngine()

tab_task1, tab_task2, tab_task3 = st.tabs(["1. Semantic Search & RAG", "2. Near-Duplicate Detection", "3. Corpus Cluster Explorer"])

with tab_task1:
    st.markdown("### Test Semantic Retrieval across Different Queries")
    
    preset_queries = {
        "User Query: 'I forgot my login password, how do I recover it?'": np.array([0.89, 0.11, 0.22, 0.12]),
        "User Query: 'How do I submit travel expense receipts?'": np.array([0.14, 0.82, 0.73, 0.12]),
        "User Query: 'Where can I see today's cafeteria lunch options?'": np.array([0.04, 0.08, 0.06, 0.96])
    }

    selected_q = st.selectbox("Select user search inquiry:", list(preset_queries.keys()))
    q_vec = preset_queries[selected_q]
    top_k_select = st.slider("Top-K Retrieved Documents:", min_value=1, max_value=6, value=3)

    results = task_engine.semantic_search(q_vec, top_k=top_k_select)
    
    st.markdown("##### Retrieved Ranked Documents:")
    res_display = [
        {"Rank": i+1, "Doc ID": r["id"], "Title": r["title"], "Category": r["category"], "Cosine Similarity": f"{r['similarity']:.4f}"}
        for i, r in enumerate(results)
    ]
    st.dataframe(pd.DataFrame(res_display), use_container_width=True)
    st.caption("Notice how 'I forgot my login password' successfully retrieves both Doc 1 ('reset your password') and Doc 2 ('changing account credentials') without requiring exact keyword matches.")

with tab_task2:
    st.markdown("### Near-Duplicate Detection Threshold Scanner")
    st.markdown("Near-duplicate detection prevents customer support teams from authoring redundant articles and flags duplicate Jira tickets.")
    
    dup_threshold = st.slider("Cosine Similarity Duplicate Threshold:", min_value=0.80, max_value=0.99, value=0.94, step=0.01)
    dups_found = task_engine.detect_duplicates(threshold=dup_threshold)

    if dups_found:
        st.warning(f"Found {len(dups_found)} near-duplicate document pair(s) at threshold >= {dup_threshold}:")
        dup_df = pd.DataFrame([
            {"Document 1": d[0], "Document 2": d[1], "Similarity Score": f"{d[2]:.4f}", "Status": "DUPLICATE FLAGGED"}
            for d in dups_found
        ])
        st.dataframe(dup_df, use_container_width=True)
    else:
        st.success(f"Zero duplicates detected above threshold {dup_threshold}. Corpus is cleanly deduplicated.")

with tab_task3:
    st.markdown("### Corpus Category Clustering View")
    st.markdown("Inspect how the 6 documents group into natural thematic clusters in vector space:")
    
    cluster_df = pd.DataFrame([
        {"Doc ID": d["id"], "Title": d["title"], "Domain Category": d["category"], "Vector Coordinates": np.array2string(d["vector"], precision=2)}
        for d in task_engine.corpus
    ])
    st.dataframe(cluster_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "System Architecture Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do vector embeddings solve the 'Cold Start' problem in recommendation systems?**
    * **Answer**:
      * Traditional collaborative filtering algorithms (matrix factorization) rely on historical user-item interaction logs. When a new product or article is published with zero historical clicks, it cannot be recommended (the cold-start problem).
      * **Content-Based Embedding Solution**: By embedding the text description, title, and metadata of the new product, it immediately receives a coordinates vector in the shared semantic space.
      * Users whose historical preference vectors are close in cosine similarity can immediately receive recommendations for the new product, completely bypassing the cold-start barrier.

    **Q2: What is the difference between Semantic Search and Near-Duplicate Detection thresholds?**
    * **Answer**:
      * **Semantic Search** casts a relatively wide net: documents with cosine similarity between 0.65 and 0.85 are highly relevant answers to the inquiry.
      * **Near-Duplicate Detection** requires extreme similarity: thresholds are set strictly between **0.93 and 0.98**. Lowering the threshold to 0.85 would falsely flag two distinct articles discussing the same topic as duplicates.

    **Q3: Why are embeddings preferred over fine-tuning for few-shot classification?**
    * **Answer**:
      * Fine-tuning a 7B LLM requires expensive GPU memory, hyperparameter tuning, and risks catastrophic forgetting.
      * In contrast, extracting frozen embeddings from a state-of-the-art encoder (e.g. `text-embedding-3-large`) and training a simple Logistic Regression or Linear SVM classifier on 10 examples per class trains in **under 2 seconds on a CPU**, produces zero inference hallucinations, and often matches or exceeds fine-tuned accuracy.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Setting Duplicate Detection Thresholds Too Low**:
      * *Issue*: Setting a duplicate threshold of 0.80.
      * *Consequence*: Articles on *"Setting up 2FA"* and *"Resetting password"* get merged and deleted as duplicates despite being distinct workflows.
      * *Fix*: Calibrate near-duplicate thresholds strictly above 0.94 based on validation pairs.

    * **Trap 2: Ignoring Asymmetry Between Short Queries and Long Documents**:
      * *Issue*: Embedding a 3-word query and a 10-page document with a standard symmetric model.
      * *Consequence*: Document vectors represent average topic breadth, while query vectors represent narrow intent, leading to weak similarity scores.
      * *Fix*: Use asymmetric embedding models or chunk documents into focused 256-512 token passages.

    * **Trap 3: Using K-Means Clustering on Raw 1536-Dimensional Vectors**:
      * *Issue*: Running K-Means with Euclidean distance on high-dimensional vectors without dimensionality reduction.
      * *Consequence*: The **Curse of Dimensionality** causes all pairwise Euclidean distances to converge to similar values, resulting in poor cluster quality.
      * *Fix*: Apply UMAP or PCA to reduce dimensions to 10-50 before clustering, or use cosine-based HDBSCAN.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Search**
      * *Definition*: Information retrieval based on the conceptual meaning of words rather than literal lexical string matches.
    * **Flashcard 2: Cold-Start Recommendation**
      * *Definition*: Recommending newly introduced catalog items with zero click history by matching item content embeddings to user profile embeddings.
    * **Flashcard 3: Near-Duplicate Detection**
      * *Definition*: Identifying paraphrased or redundant text by checking if cosine similarity exceeds an ultra-high threshold (e.g., > 0.94).
    * **Flashcard 4: Asymmetric Search**
      * *Definition*: A retrieval setup where query text is short and document text is long, requiring specialized query/passage instruction models.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embedding Utility & Applications:

- [ ] I can describe how embeddings power semantic search, recommendations, clustering, and duplicate detection.
- [ ] I know how content embeddings solve the cold-start problem in recommendation engines.
- [ ] I understand the difference in similarity thresholds between search (0.70) and deduplication (0.95).
- [ ] I can explain why frozen embeddings + linear classifiers are ideal for few-shot classification.
- [ ] I understand how the curse of dimensionality impacts clustering in high-dimensional vector spaces.
""")
