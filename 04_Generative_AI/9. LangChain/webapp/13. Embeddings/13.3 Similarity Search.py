import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class BenchmarkVectorItem:
    """Represents a document chunk with pre-computed normalized 4D embedding vector."""
    def __init__(self, item_id: str, text: str, topic: str, vector: np.ndarray):
        self.item_id = item_id
        self.text = text
        self.topic = topic
        self.vector = vector

class MockVectorMetricPlaygroundSimulator:
    """Executes vector similarity calculations across multiple distance kernels and threshold gates."""
    def __init__(self):
        # 4D synthetic vectors
        self.dataset = [
            BenchmarkVectorItem(
                "DOC-A1", "Production PostgreSQL failover target is under 30 seconds with synchronous replication.",
                "Database Infra", np.array([0.91, 0.12, 0.28, 0.26])
            ),
            BenchmarkVectorItem(
                "DOC-A2", "Database connection pooling guidelines recommend PgBouncer with transaction-level pooling.",
                "Database Infra", np.array([0.88, 0.18, 0.32, 0.30])
            ),
            BenchmarkVectorItem(
                "DOC-B1", "Corporate dental and vision benefits cover up to 2,500 USD annually per dependent.",
                "HR Benefits", np.array([0.15, 0.92, 0.18, 0.32])
            ),
            BenchmarkVectorItem(
                "DOC-B2", "Paid parental leave allows 16 weeks of fully compensated time off for primary caregivers.",
                "HR Benefits", np.array([0.18, 0.89, 0.22, 0.35])
            ),
            BenchmarkVectorItem(
                "DOC-C1", "Zero trust network architecture mandates continuous mutual TLS authentication on all internal APIs.",
                "Cybersecurity", np.array([0.35, 0.20, 0.90, 0.18])
            ),
            BenchmarkVectorItem(
                "DOC-C2", "Vulnerability disclosure policy rewards external bug bounty researchers up to 10,000 USD.",
                "Cybersecurity", np.array([0.30, 0.28, 0.86, 0.28])
            )
        ]
        # Normalize all dataset vectors to unit norm
        for item in self.dataset:
            norm = np.linalg.norm(item.vector)
            if norm > 0:
                item.vector = item.vector / norm

    def _get_query_vector(self, query_theme: str) -> np.ndarray:
        if query_theme == "Database Clustering & Failover":
            vec = np.array([0.94, 0.10, 0.25, 0.22])
        elif query_theme == "Health & Family Insurance Benefits":
            vec = np.array([0.12, 0.95, 0.15, 0.24])
        elif query_theme == "Zero Trust Network & Cryptography":
            vec = np.array([0.28, 0.18, 0.93, 0.15])
        else:
            vec = np.array([0.25, 0.25, 0.25, 0.25])

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def compute_ranking(self, query_theme: str, metric: str, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        q_vec = self._get_query_vector(query_theme)
        scored_items = []

        for item in self.dataset:
            # 1. Cosine Similarity: dot product of normalized vectors
            cos_sim = float(np.dot(q_vec, item.vector))

            # 2. Euclidean Distance (L2 norm)
            l2_dist = float(np.linalg.norm(q_vec - item.vector))

            # 3. Dot Product
            dot_prod = float(np.dot(q_vec, item.vector))

            if metric == "Cosine Similarity":
                score = cos_sim
                passed = score >= threshold
            elif metric == "Euclidean Distance":
                score = l2_dist
                passed = score <= threshold  # Distance: smaller is better
            else:  # Dot Product
                score = dot_prod
                passed = score >= threshold

            scored_items.append({
                "item_id": item.item_id,
                "topic": item.topic,
                "text": item.text,
                "score": round(score, 4),
                "cos_sim": round(cos_sim, 4),
                "l2_dist": round(l2_dist, 4),
                "passed_threshold": passed
            })

        # Sort based on metric
        if metric == "Euclidean Distance":
            scored_items.sort(key=lambda x: x["score"], reverse=False)
        else:
            scored_items.sort(key=lambda x: x["score"], reverse=True)

        return scored_items[:top_k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="13.3 Similarity Search",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">13.3 Similarity Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Distance kernels, metric geometry: Cosine Similarity, Euclidean Distance, Dot Product, Top-K ranking, and threshold gating.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Vector Metric Geometry & Distance Kernels", divider="rainbow")

st.markdown("""
Once text is converted into dense vectors, retrieval becomes a geometric proximity calculation.
A distance kernel evaluates the geometric alignment between a query vector and millions of candidate vectors in database storage.
The choice of similarity metric governs **length sensitivity**, **computational latency**, and **retrieval accuracy**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MetricArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_query_input {
            label="1. Inbound Query Representation";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            QueryVector [label="Query Vector (q)\\n[0.94, 0.10, 0.25, 0.22] (||q|| = 1.0)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_metric_kernels {
            label="2. Distance & Similarity Kernels";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            CosineKernel [label="Cosine Similarity\\ncos(theta) = (q · v) / (||q|| * ||v||)\\nAngular direction; length invariant", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            EuclideanKernel [label="Euclidean Distance (L2)\\ndist = sqrt(sum((q_i - v_i)^2))\\nStraight-line distance; magnitude sensitive", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            DotKernel [label="Dot Product (Inner Product)\\nprod = sum(q_i * v_i)\\nFastest hardware SIMD; requires unit norm", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_ranking_filtering {
            label="3. Top-K Ranking & Threshold Gate";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            TopKHeap [label="Priority Queue / Top-K Max-Heap\\nExtracts K closest neighbors across N records", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            ThresholdGate [label="Similarity Threshold Filter\\nDrop records where score < minimum cutoff (e.g. 0.75)\\nEliminates hallucination pollution", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_context_delivery {
            label="4. Verified Context Output";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            CleanContext [label="Clean Grounded Context Excerpts\\nDelivered to LLM prompt with confidence scores", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399", penwidth=2];
        }

        QueryVector -> CosineKernel [label="Evaluate angle"];
        QueryVector -> EuclideanKernel [label="Evaluate distance"];
        QueryVector -> DotKernel [label="Evaluate projection"];

        CosineKernel -> TopKHeap [label="Similarity scores"];
        EuclideanKernel -> TopKHeap [label="Distance values"];
        DotKernel -> TopKHeap [label="Inner products"];

        TopKHeap -> ThresholdGate [label="Top-K Candidates"];
        ThresholdGate -> CleanContext [label="Passing records only"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: COSINE SIMILARITY ---
st.header("2. Cosine Similarity", divider="rainbow")

st.markdown("""
**Cosine Similarity** evaluates the cosine of the angle between two multi-dimensional vectors:
```
cosine_similarity = (A · B) / (||A|| * ||B||)
```
- **Range**: -1.0 (opposing vectors) to +1.0 (identical angular orientation). For non-negative embeddings, values fall between 0.0 and 1.0.
- **Key Advantage**: It is **completely invariant to vector magnitude**. Whether a document is 20 words or 2,000 words, if the topical direction aligns with the query, the cosine similarity remains high.
- **Cosine Distance**: Many vector databases (e.g. Chroma, Pinecone) report distance instead of similarity:
```
cosine_distance = 1.0 - cosine_similarity
```
Where 0.0 indicates an exact identical match.
""")

st.divider()

# --- SECTION 3: EUCLIDEAN DISTANCE (L2) ---
st.header("3. Euclidean Distance", divider="rainbow")

st.markdown("""
**Euclidean Distance (L2 Norm)** measures the geometric straight-line distance between two points in multi-dimensional space:
```
euclidean_distance = sqrt( sum( (A_i - B_i)^2 ) )
```
- **Range**: 0.0 (identical vectors) to positive infinity.
- **Key Characteristic**: It is **magnitude-sensitive**. If two vectors have the exact same angular direction but different lengths, their Euclidean distance will be large.
- **Unit Norm Equivalence**: When both vectors are pre-normalized to unit length (`||A|| = 1.0` and `||B|| = 1.0`), Euclidean distance and Cosine similarity share a monotonic mathematical relationship:
```
(euclidean_distance)^2 = 2 * (1.0 - cosine_similarity)
```
Consequently, searching for the smallest L2 distance yields the exact same ranked order as searching for the largest Cosine similarity on normalized vectors!
""")

st.divider()

# --- SECTION 4: DOT PRODUCT ---
st.header("4. Dot Product", divider="rainbow")

st.markdown("""
The **Dot Product (Inner Product)** multiplies corresponding coordinates across two vectors and sums the results:
```
dot_product = sum( A_i * B_i )
```
- **Computational Efficiency**: The Dot Product requires only multiply-accumulate operations (MACs). It avoids expensive square root operations and divisions.
- **The Golden Rule**: When vectors are L2-normalized, **Dot Product equals Cosine Similarity identically**.
Production vector databases (Pinecone, Milvus, pgvector) achieve massive throughput by pre-normalizing vectors upon insertion and executing pure Dot Product hardware kernels.
""")

st.divider()

# --- SECTION 5: TOP-K RETRIEVAL ---
st.header("5. Top-k Retrieval", divider="rainbow")

st.markdown("""
Given a query vector and a database of N indexed vectors, **Top-K Retrieval** extracts the K candidates 
with the highest similarity scores.

### Exact vs Approximate Search:
""")

topk_df = pd.DataFrame([
    {
        "Search Algorithm": "Flat / Exact k-NN (Brute Force)",
        "Time Complexity": "O(N * d)",
        "Recall Accuracy": "100% (Guaranteed optimal)",
        "When to Use": "Small corpora (< 100,000 vectors); in-memory testing."
    },
    {
        "Search Algorithm": "HNSW (Hierarchical Navigable Small World)",
        "Time Complexity": "O(log N * d)",
        "Recall Accuracy": "95% - 99% (Approximate)",
        "When to Use": "The industry standard for production vector databases (100K to 100M+ vectors)."
    },
    {
        "Search Algorithm": "IVF-PQ (Inverted File with Product Quantization)",
        "Time Complexity": "O(C * d) (Clustered)",
        "Recall Accuracy": "90% - 97% (Quantized)",
        "When to Use": "Massive corpora (100M+ vectors) where RAM efficiency is prioritized over perfect recall."
    }
])
st.dataframe(topk_df, use_container_width=True)

st.divider()

# --- SECTION 6: SIMILARITY THRESHOLDS ---
st.header("6. Similarity Thresholds", divider="rainbow")

st.markdown("""
Standard RAG systems blindly inject whatever Top-K documents are returned, even when the best match has a low similarity score (e.g. 0.38).
Injecting low-relevance documents poisons the LLM context, directly causing hallucinations.

**Similarity Threshold Gating** enforces a minimum relevance floor:
- If a chunk's score exceeds the threshold (e.g. `score >= 0.75`), it is passed to the prompt.
- If zero chunks exceed the threshold, the retriever signals a **Knowledge Void**, allowing the agent to decline to answer safely.
""")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: High-Performance Vector Metric Engine & Threshold Gate", divider="rainbow")

st.markdown("""
Below is an enterprise-grade vector similarity engine written in pure NumPy, demonstrating normalized dot products, 
priority-heap Top-K ranking, and threshold gating.
""")

st.code("""import numpy as np
import heapq
from typing import List, Dict, Any, Tuple

class ProductionVectorMetricEngine:
    \"\"\"Production vector metric and top-k retrieval engine.\"\"\"
    def __init__(self, metric: str = "cosine", min_threshold: float = 0.70):
        self.metric = metric.lower()
        self.min_threshold = min_threshold

    @staticmethod
    def l2_normalize(matrix: np.ndarray) -> np.ndarray:
        \"\"\"Normalizes 2D matrix rows to unit length.\"\"\"
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def search_top_k(self,
                     query_vec: np.ndarray,
                     corpus_matrix: np.ndarray,
                     doc_metadata: List[Dict[str, Any]],
                     top_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Normalize query and corpus
        q_norm = query_vec / np.linalg.norm(query_vec)
        c_norm = self.l2_normalize(corpus_matrix)

        # 2. Vectorized batch matrix multiplication (Dot Product == Cosine)
        similarities = np.dot(c_norm, q_norm)

        # 3. Top-K extraction using min-heap
        heap = []
        for idx, score in enumerate(similarities):
            # Threshold gating
            if score >= self.min_threshold:
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, idx))
                else:
                    if score > heap[0][0]:
                        heapq.heappushpop(heap, (score, idx))

        # 4. Sort results in descending score order
        results = []
        while heap:
            score, idx = heapq.heappop(heap)
            results.append({
                "metadata": doc_metadata[idx],
                "similarity_score": round(float(score), 4)
            })

        results.reverse()
        return results""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Geometric Metric Playground & Top-K Ranking Simulator", divider="rainbow")

st.markdown("""
Compare how different distance kernels rank documents and observe how threshold gating drops irrelevant chunks.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Retrieval Parameters")
    query_scenario = st.selectbox(
        "Select Enterprise Query Theme",
        options=[
            "Database Clustering & Failover",
            "Health & Family Insurance Benefits",
            "Zero Trust Network & Cryptography"
        ]
    )

    metric_choice = st.selectbox(
        "Distance Metric Kernel",
        options=["Cosine Similarity", "Euclidean Distance", "Dot Product"]
    )

    k_slider = st.slider("Top-K Candidates", min_value=1, max_value=6, value=4)

    default_thresh = 0.70 if metric_choice != "Euclidean Distance" else 0.80
    thresh_slider = st.slider(
        f"Relevance Threshold Cutoff ({'Max Distance' if metric_choice == 'Euclidean Distance' else 'Min Similarity'})",
        min_value=0.20, max_value=1.50 if metric_choice == "Euclidean Distance" else 0.95,
        value=default_thresh,
        step=0.05
    )

    btn_search = st.button("🚀 Execute Geometric Similarity Search", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Geometric Search Ranking")

    if btn_search:
        sim = MockVectorMetricPlaygroundSimulator()
        matches = sim.compute_ranking(query_scenario, metric_choice, k_slider, thresh_slider)

        st.metric("Total Candidates Evaluated", len(matches))

        for idx, m in enumerate(matches, 1):
            passed = m["passed_threshold"]
            badge = "✅ PASS" if passed else "❌ REJECTED BY THRESHOLD"
            with st.expander(f"Rank #{idx} [{badge}]: {m['item_id']} ({m['topic']})", expanded=passed):
                c1, c2 = st.columns(2)
                c1.metric("Selected Metric Score", m["score"])
                c2.metric("Cosine Similarity", m["cos_sim"])
                st.write(f"**Content**: `{m['text']}`")
                if not passed:
                    st.caption("⚠️ This document was dropped to prevent context dilution.")
    else:
        st.info("Select parameters and click **Execute Geometric Similarity Search** to view rankings.")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why do production vector databases prefer Dot Product over Cosine Similarity internally?",
     "Cosine similarity requires calculating the magnitude (norm) of both vectors, involving square roots and divisions for every comparison. Dot product involves only multiply-accumulate operations, which execute natively on modern CPU vector units (AVX-512) and GPU Tensor Cores. By pre-normalizing all vectors to unit length during ingestion, the Dot Product produces the exact same score as Cosine Similarity at a fraction of the compute cost."),
    ("What is the difference between exact k-NN search and Approximate Nearest Neighbor (ANN) search?",
     "Exact k-NN compares the query vector against every single vector in the index (brute force, O(N * d)). It guarantees 100% recall but becomes unacceptably slow on millions of vectors. ANN algorithms (like HNSW or IVF) build hierarchical graph or cluster index structures that search only a small subset of candidate vectors (O(log N * d)), achieving 95-99% recall in under 5 milliseconds."),
    ("Why is similarity thresholding critical in enterprise conversational RAG?",
     "Top-K retrieval always returns K documents regardless of how irrelevant they are. If a user asks a question completely outside the corporate knowledge base, a top-K search will still return the K least-unrelated documents. If these are injected into the prompt, the model is likely to hallucinate an answer. A strict similarity threshold drops low-scoring chunks, allowing the system to detect when no relevant information exists and respond with an honest refusal."),
    ("How do vector databases convert distance metrics into confidence scores?",
     "Different metrics have different scales: Cosine similarity ranges from -1 to 1, while Euclidean distance ranges from 0 to infinity. Most vector stores normalize these into a 0.0 to 1.0 confidence score. For example, Chroma converts Cosine distance via: score = 1.0 - distance, while L2 distance is often normalized via: score = 1.0 / (1.0 + distance)."),
    ("What is the computational complexity of brute force search vs HNSW graph search?",
     "Brute force search scales linearly with the number of documents: O(N * d), where N is document count and d is vector dimension. In contrast, HNSW builds a multi-layer graph of vectors resembling a skip-list. Searching an HNSW index scales logarithmically: O(log N * d), enabling sub-10ms queries across hundreds of millions of vectors.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Dot Product on Unnormalized Vectors. Calculating dot products on raw, unnormalized vectors favors longer documents that have higher vector magnitudes, completely distorting semantic rankings. Always normalize vectors to unit length before using dot products.")
    st.error("**Trap 2 (Breaking Error)**: Inverting Distance and Similarity in Threshold Filters. Confusing distance (where smaller is better) with similarity (where larger is better) results in rejecting all good documents while accepting the worst ones. Always verify whether your vector database returns distance or similarity.")
with trap_c2:
    st.warning("**Trap 3 (Latency Warning)**: Using Brute Force Flat Search Beyond 100K Vectors. Flat search causes latency to scale linearly with dataset size. Beyond 100,000 vectors, always enable HNSW indexing.")
    st.warning("**Trap 4 (Quality Warning)**: Setting Static Thresholds Without Validation. Setting an arbitrary threshold (e.g. 0.85) without testing against real validation queries can cause widespread retrieval false-negatives. Calibrate thresholds empirically against a labeled test suite.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Cosine Similarity"):
        st.markdown("**Evaluates angular alignment; invariant to text and vector magnitude.**")
    with st.expander("🎴 Flashcard 2: Dot Product Rule"):
        st.markdown("**Identical to Cosine Similarity when vectors are pre-normalized to unit length.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: HNSW Algorithm"):
        st.markdown("**Hierarchical graph index enabling sub-5ms logarithmic search over millions of vectors.**")
    with st.expander("🎴 Flashcard 4: Threshold Gating"):
        st.markdown("**Drops low-similarity candidates to prevent prompt pollution and hallucinations.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the geometric formulas and trade-offs between Cosine, Euclidean, and Dot Product metrics.", value=True)
chk2 = st.checkbox("I know why Dot Product on unit-normalized vectors is the industry standard for search speed.", value=True)
chk3 = st.checkbox("I understand the computational scaling difference between brute force k-NN and HNSW ANN search.", value=True)
chk4 = st.checkbox("I can implement similarity threshold gating to eliminate low-relevance hallucination noise.", value=True)
chk5 = st.checkbox("I know how to convert between distance metrics and normalized similarity confidence scores.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 similarity search interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 13.3 Similarity Search and completed all of Chapter 13: Embeddings!")
