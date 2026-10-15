import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class DenseSemanticSearchEngine:
    """
    Simulates dense semantic retrieval using high-dimensional dense embeddings,
    cosine similarity projections, dual-tower representations, and Top-K gating.
    Demonstrates synonymy resolution, polysemy disambiguation, and out-of-vocabulary handling.
    """
    CORPUS = [
        {
            "id": "doc_med_01",
            "title": "Clinical Compensation Review",
            "text": "Physician annual compensation packages across metropolitan healthcare networks.",
            "category": "Healthcare",
            "vector": [0.89, 0.12, 0.05, 0.02, 0.91, 0.08, 0.14, 0.03]
        },
        {
            "id": "doc_fin_01",
            "title": "Commercial Banking Operations",
            "text": "Deposit cash reserves and liquid capital directly into central investment bank accounts.",
            "category": "Finance",
            "vector": [0.08, 0.92, 0.11, 0.88, 0.05, 0.15, 0.22, 0.09]
        },
        {
            "id": "doc_geo_01",
            "title": "Hydrology and River Geomorphology",
            "text": "Sediment deposition along the fertile river bank during seasonal monsoon flooding.",
            "category": "Geology",
            "vector": [0.04, 0.15, 0.95, 0.18, 0.02, 0.87, 0.12, 0.10]
        },
        {
            "id": "doc_tech_01",
            "title": "Distributed DB Outage Postmortem",
            "text": "Troubleshooting fatal memory leak error code ERR_MEM_0x8892 during WAL replication.",
            "category": "Engineering",
            "vector": [0.15, 0.20, 0.08, 0.94, 0.12, 0.18, 0.89, 0.91]
        },
        {
            "id": "doc_tech_02",
            "title": "Mobile Device Liquid Ingress",
            "text": "Hardware diagnostics and electrolytic corrosion prevention after smartphone submersion in water.",
            "category": "Engineering",
            "vector": [0.18, 0.14, 0.25, 0.82, 0.20, 0.25, 0.92, 0.78]
        }
    ]

    PRESET_QUERIES = {
        "Synonymy Probe: 'Doctor salary in hospital'": [0.88, 0.15, 0.04, 0.03, 0.90, 0.07, 0.12, 0.02],
        "Polysemy Probe (Finance): 'Deposit money in local bank branch'": [0.09, 0.90, 0.10, 0.85, 0.06, 0.12, 0.20, 0.08],
        "Polysemy Probe (Nature): 'Walk along the grassy river bank'": [0.03, 0.12, 0.94, 0.15, 0.01, 0.89, 0.10, 0.08],
        "Concept Probe: 'My phone dropped in the pool and won't turn on'": [0.20, 0.11, 0.22, 0.85, 0.18, 0.22, 0.90, 0.80],
        "Exact Token Probe: 'Fix ERR_MEM_0x8892 crash'": [0.14, 0.19, 0.07, 0.92, 0.11, 0.16, 0.88, 0.90]
    }

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @classmethod
    def search(
        cls,
        query_vector: List[float],
        top_k: int = 3,
        threshold: float = 0.50,
        distance_metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(qv)
        qv_norm = qv / norm_q if norm_q > 0 else qv

        scored_records = []
        for doc in cls.CORPUS:
            dv = np.array(doc["vector"], dtype=np.float32)
            norm_d = np.linalg.norm(dv)
            dv_norm = dv / norm_d if norm_d > 0 else dv

            if distance_metric == "cosine":
                score = float(np.dot(qv_norm, dv_norm))
                dist = 1.0 - score
            elif distance_metric == "dot_product":
                score = float(np.dot(qv, dv))
                dist = -score
            else:  # euclidean
                euclid_dist = float(np.linalg.norm(qv - dv))
                score = 1.0 / (1.0 + euclid_dist)
                dist = euclid_dist

            if score >= threshold:
                scored_records.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "score": round(score, 4),
                    "distance": round(dist, 4)
                })

        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="11.2 Semantic Search - RAG Master Curriculum",
    page_icon="🧠",
    layout="wide"
)

st.title("11.2 Semantic Search: Dense Neural Retrieval, Dual-Tower Manifolds, & Semantic Similarity")
st.caption("Chapter 11: Hybrid Search | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Traditional lexical search operates under the rigid assumption that relevant answers must contain the exact 
keywords present in a user query. Human language, however, expresses intent through synonymy, polysemy, 
and conceptual paraphrasing:
* **Synonymy**: Multiple distinct terms reference the identical semantic entity (*physician salary* vs. *doctor compensation*).
* **Polysemy**: Identical lexical tokens represent completely different concepts depending on context (*commercial bank* vs. *river bank*).
* **Conceptual Intent**: A user asking *"My phone fell in the pool and died"* seeks a document titled *"Troubleshooting Hardware Liquid Ingress"*, even though zero lexical words overlap.

**Dense Semantic Search** projects language passages into continuous, dense latent vector manifolds 
(typically 768 to 1536 dimensions). Using deep Transformer bi-encoders, it resolves linguistic meaning, 
unlocking semantic proximity retrieval at production scale.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Dense Neural Retrieval Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    semantic_bp_dot = """
    digraph DenseSemanticSearchBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_offline {
            label = "1. Offline Ingestion & Dense Indexing Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_corpus [label="Raw Text Corpus\\nDocuments, Runbooks, & Knowledge Base", fillcolor="#0F172A", color="#38BDF8"];
            doc_encoder [label="Document Bi-Encoder (f_doc)\\ne.g., text-embedding-3-small, BGE-large", fillcolor="#334155", color="#10B981"];
            dense_vectors [label="Dense Vector Storage\\n768/1536-dim Float32 Normalized Coordinates", fillcolor="#0F172A", color="#06B6D4"];
            ann_index [label="ANN Proximity Index\\nHNSW Graph / IVF Vector Index", fillcolor="#1E293B", color="#F59E0B"];

            raw_corpus -> doc_encoder [label="Tokenize & Chunk"];
            doc_encoder -> dense_vectors [label="Mean Pooling + L2 Norm"];
            dense_vectors -> ann_index [label="Build Proximity Graph"];
        }

        subgraph cluster_online {
            label = "2. Online Query Processing & Dual-Tower Execution";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="User Natural Language Query\\ne.g., 'physician earnings in metro hospitals'", fillcolor="#0F172A", color="#38BDF8"];
            query_encoder [label="Query Bi-Encoder (f_query)\\nShared / Asymmetric Tower Weights", fillcolor="#334155", color="#8B5CF6"];
            query_vector [label="Dense Query Coordinate\\nq_vec = Normalized Float32[1536]", fillcolor="#0F172A", color="#8B5CF6"];

            user_query -> query_encoder [label="Subword Tokenizer"];
            query_encoder -> query_vector [label="Forward Pass & L2 Norm"];
        }

        subgraph cluster_scoring {
            label = "3. Geometric Scoring & Top-K Gating Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            graph_traversal [label="Vector Space Proximity Walk\\nCosine Dot Product: sim = dot(q, d)", fillcolor="#334155", color="#EC4899"];
            topk_filter [label="Top-K Gating & Threshold Filter\\nscore >= threshold | Top-K Candidates", fillcolor="#334155", color="#F59E0B"];
            final_docs [label="Retrieved Semantic Documents\\nHydrated Context for Generator LLM", fillcolor="#0F172A", color="#10B981"];

            query_vector -> graph_traversal [label="Query Probe Vector"];
            ann_index -> graph_traversal [label="Neighbor Graph Traversal"];
            graph_traversal -> topk_filter [label="Ranked Distance Heap"];
            topk_filter -> final_docs [label="Hydrate Top-K Context"];
        }
    }
    """
    st.graphviz_chart(semantic_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems, dense retrieval executes via dual-tower bi-encoders where document embeddings 
are pre-computed offline and loaded into vector indices such as **Qdrant**, **Pinecone**, or **pgvector**. 
During online execution, the user query is encoded into a normalized vector in under 15ms. 
The database traverses an HNSW graph to find nearest neighbors via inner product, returning high-recall 
candidates even when zero literal terms match.
""")

st.divider()

# ==============================================================================
# SECTION 2: DENSE RETRIEVAL
# ==============================================================================

st.header("2. Dense Retrieval: Architecture, Dual Towers, and Subspace Manifolds", divider="rainbow")

st.markdown("""
**Dense retrieval** fundamentally differs from sparse lexical search by mapping every token sequence 
into a compact continuous vector space where semantic proximity corresponds directly to geometric angle.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dual-Tower Bi-Encoder Mechanics")
    st.markdown("""
    In a bi-encoder architecture, queries and documents pass through independent Transformer towers:
    * **Independent Encoders**: The query function `f_query(Q)` and document function `f_doc(D)` encode texts separately. 
      Because they do not cross-attend to each other, document vectors are generated once and stored permanently.
    * **Mean Pooling & Normalization**: The token embeddings output from the final Transformer hidden layer are 
      averaged across tokens (mean pooling) and projected onto a unit sphere (`L2 norm = 1.0`).
    * **Inner Product Computation**: On a unit sphere, cosine similarity equals the simple dot product:
      `similarity = dot(q_norm, d_norm)`. This allows SIMD-accelerated AVX-512 hardware matrix operations.
    * **Asymmetric Encodings**: Production embedding models often use asymmetric prefixes (e.g., `"search_query: "` 
      vs. `"search_document: "`) to optimize directional projections for asymmetric RAG tasks.
    """)

with col2:
    st.subheader("Bi-Encoder vs. Cross-Encoder Contrast")
    bi_vs_cross_dot = """
    digraph BiVsCross {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.35, ranksep=0.45];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=8, fontcolor="#94A3B8"];

        subgraph cluster_bi {
            label = "Bi-Encoder (Sub-millisecond Scale)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";

            q_in [label="Query Text", fillcolor="#0F172A", color="#38BDF8"];
            d_in [label="Doc Text", fillcolor="#0F172A", color="#10B981"];
            q_enc [label="Encoder Tower 1", fillcolor="#334155", color="#38BDF8"];
            d_enc [label="Encoder Tower 2", fillcolor="#334155", color="#10B981"];
            dot_op [label="Dot Product Layer", fillcolor="#0F172A", color="#F59E0B"];

            q_in -> q_enc -> dot_op;
            d_in -> d_enc -> dot_op;
        }

        subgraph cluster_cross {
            label = "Cross-Encoder (High Latency Reranker)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#EC4899";

            concat_in [label="Concat: [CLS] Query [SEP] Doc [SEP]", fillcolor="#0F172A", color="#EC4899"];
            full_trans [label="All-to-All Full Cross-Attention\\nO((N_q + N_d)^2) Compute", fillcolor="#334155", color="#EC4899"];
            cross_score [label="Scalar Relevance Logit", fillcolor="#0F172A", color="#10B981"];

            concat_in -> full_trans -> cross_score;
        }
    }
    """
    st.graphviz_chart(bi_vs_cross_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: EMBEDDING-BASED RETRIEVAL
# ==============================================================================

st.header("3. Embedding-Based Retrieval: Vector Geometries, Index Traversal, and Quantization", divider="rainbow")

st.markdown("""
Embedding-based retrieval converts retrieval into an **Approximate Nearest Neighbor (ANN)** problem across 
millions of dense vectors. Understanding vector geometry, indexing mechanisms, and precision compression 
is critical for low-latency RAG architectures.
""")

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown("""
    **Cosine Similarity Metric**
    * **Geometry**: Measures angle between two vectors regardless of magnitude.
    * **Formula**: Dot product divided by product of vector norms.
    * **Range**: -1.0 to +1.0 (typically 0.60 to 0.95 in real embeddings).
    * **Use Case**: Text semantic matching where document length varies.
    """)

with metric_cols[1]:
    st.markdown("""
    **Euclidean Distance (L2)**
    * **Geometry**: Straight-line physical distance between vector tips.
    * **Formula**: Square root of sum of squared coordinate differences.
    * **Range**: 0.0 to infinity (0.0 represents identical points).
    * **Use Case**: Spatial clustering, image embeddings, unnormalized spaces.
    """)

with metric_cols[2]:
    st.markdown("""
    **Inner Product (Dot Product)**
    * **Geometry**: Projection of vector A onto vector B multiplied by length.
    * **Formula**: Sum of element-wise coordinate multiplications.
    * **Range**: Negative infinity to positive infinity.
    * **Optimization**: Equivalent to Cosine Similarity when vectors are L2 normalized.
    """)

st.subheader("System Trade-offs Across Vector Indexing Types")
vdb_perf_df = pd.DataFrame([
    {
        "Index Architecture": "Flat (Brute Force Exact Search)",
        "Recall Accuracy": "100.0%",
        "Search Latency (1M Docs)": "45.0 ms (O(N * d))",
        "RAM Footprint": "High (uncompressed vectors)",
        "Production Role": "Golden truth verification & small datasets (<50k docs)"
    },
    {
        "Index Architecture": "HNSW (Hierarchical Navigable Small World)",
        "Recall Accuracy": "97.5% - 99.2%",
        "Search Latency (1M Docs)": "1.2 ms (O(log N))",
        "RAM Footprint": "Very High (vectors + multi-layer graph skip lists)",
        "Production Role": "Default gold standard for ultra-low latency RAG"
    },
    {
        "Index Architecture": "IVF-PQ (Inverted File + Product Quantization)",
        "Recall Accuracy": "91.0% - 95.0%",
        "Search Latency (1M Docs)": "3.5 ms",
        "RAM Footprint": "Ultra-Low (4x - 16x memory compression)",
        "Production Role": "Billion-scale enterprise vector archives with strict RAM limits"
    }
])
st.dataframe(vdb_perf_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: SEMANTIC SIMILARITY
# ==============================================================================

st.header("4. Semantic Similarity: Resolving Synonymy, Polysemy, and Out-of-Vocabulary Intent", divider="rainbow")

st.markdown("""
Semantic similarity models evaluate text at the conceptual layer rather than the surface string layer. 
This resolves the classical lexical failures while introducing unique neural behaviors that every engineer must master.
""")

col_sem1, col_sem2 = st.columns(2)

with col_sem1:
    st.subheader("The Three Superpowers of Dense Similarity")
    st.markdown("""
    1. **Synonymy Resolution**:
       * Queries using *"physician pay"* match documents discussing *"doctor compensation"* or *"MD remuneration"*.
       * The attention layers map these token sequences to the same semantic centroid in embedding space.
    2. **Polysemy Disambiguation**:
       * The ambiguous word *"bank"* changes coordinates depending on surrounding tokens:
         * *"commercial investment bank account"* -> Finance cluster (coordinates near currency, interest, loans).
         * *"grassy river bank sediment"* -> Hydrology cluster (coordinates near water, silt, nature).
    3. **Conceptual Paraphrase Matching**:
       * Complex queries like *"how do I deal with my laptop failing after coffee spilled on it?"* 
         retrieve documents titled *"Motherboard Corrosion Prevention and Liquid Cleaning"*.
    """)

with col_sem2:
    st.subheader("The Three Blind Spots of Pure Dense Similarity")
    st.markdown("""
    1. **Alphanumeric Code Blindness**:
       * Embeddings struggle with exact SKU numbers (*"SN-88219-X"*), commit hashes (*"commit 4a9f1b"*), 
         or custom error codes (*"ERR_MEM_0x8892"*). Subword tokenizers fragment them into generic tokens.
    2. **Negation Insensitivity**:
       * *"Drugs approved for asthma"* and *"Drugs NOT approved for asthma"* share 95% identical context tokens. 
         Their cosine similarity is often > 0.88, causing severe hallucination in legal and medical RAG.
    3. **Uncalibrated Score Distributions**:
       * Unlike probability values (0.0 to 1.0), cosine scores often sit in an arbitrary band (e.g., 0.72 to 0.86). 
         A score of 0.74 may indicate strong relevance in one corpus and noise in another.
    """)

st.divider()

# ==============================================================================
# SECTION 5: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("5. Production Implementation: Enterprise Dense Vector Retrieval Engine", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module demonstrating dense semantic search with pre-normalization, 
metric switching, candidate filtering, and structured result payload formatting.
""")

st.code('''
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseSemanticSearchEngine:
    """
    High-performance in-memory dense vector search engine.
    Applies L2 normalization during ingestion so cosine similarity
    reduces to hardware-accelerated dot products.
    """
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.doc_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.normalized_matrix: Optional[np.ndarray] = None

    def index_corpus(
        self,
        documents: List[Dict[str, Any]],
        embedding_matrix: np.ndarray
    ) -> None:
        """
        Ingests document records and caches L2-normalized float32 embeddings.
        """
        assert len(documents) == embedding_matrix.shape[0], "Length mismatch between docs and vectors"
        self.doc_ids = [d["id"] for d in documents]
        self.corpus_texts = [d["text"] for d in documents]
        self.metadata_store = [d.get("metadata", {}) for d in documents]

        # Compute L2 norms along row axis
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_matrix = (embedding_matrix / norms).astype(np.float32)

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Executes proximity search against normalized matrix using vectorized dot products.
        """
        if self.normalized_matrix is None or len(self.doc_ids) == 0:
            return []

        # Normalize query vector
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_unit = (query_vector / q_norm).astype(np.float32)

        # Batch dot product over entire normalized matrix
        raw_similarities = np.dot(self.normalized_matrix, q_unit)

        # Filter by threshold and retrieve top-k indices
        valid_indices = np.where(raw_similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Sort candidate indices by score descending
        sorted_subset = valid_indices[np.argsort(-raw_similarities[valid_indices])][:top_k]

        results = []
        for idx in sorted_subset:
            results.append({
                "id": self.doc_ids[idx],
                "text": self.corpus_texts[idx],
                "metadata": self.metadata_store[idx],
                "score": float(np.round(raw_similarities[idx], 4))
            })
        return results
''', language="python")

st.divider()

# ==============================================================================
# SECTION 6: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("6. Interactive Workbench: Dense Semantic Search Simulator", divider="rainbow")

st.markdown("""
Explore how dense semantic search resolves complex queries, evaluates synonymy and polysemy, 
and handles distance thresholds across a sample multi-domain corpus.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_query_label = st.selectbox(
        "Select Test Query Intent:",
        list(DenseSemanticSearchEngine.PRESET_QUERIES.keys())
    )
    metric_choice = st.selectbox(
        "Distance Metric:",
        ["cosine", "dot_product", "euclidean"]
    )
    top_k_val = st.slider("Top-K Candidates:", min_value=1, max_value=5, value=3)
    threshold_val = st.slider("Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    debug_vector_mode = st.toggle("Show Raw Vector Coordinates", value=False)

with wb_col2:
    st.markdown("### 🎯 Intent Analysis & Expected Outcome")
    if "Synonymy" in selected_query_label:
        st.info("💡 **Synonymy Challenge**: The query uses *'doctor'* and *'salary'*. The corpus contains Doc 1 (*'physician compensation'*). Zero lexical words overlap, but dense coordinates point in the same direction!")
    elif "Finance" in selected_query_label:
        st.info("💡 **Polysemy Disambiguation (Finance)**: The word *'bank'* attends to *'deposit'* and *'money'*, steering the vector directly toward Doc 2 (Commercial Banking) while rejecting Doc 3 (River Bank).")
    elif "Nature" in selected_query_label:
        st.info("💡 **Polysemy Disambiguation (Nature)**: The word *'bank'* attends to *'grassy'* and *'river'*, steering the vector directly toward Doc 3 (River Geomorphology) while rejecting Doc 2 (Commercial Banking).")
    elif "Concept" in selected_query_label:
        st.info("💡 **Conceptual Paraphrase**: User describes a broken smartphone after liquid immersion. Dense search routes query to Doc 5 (Hardware Liquid Ingress).")
    else:
        st.warning("⚠️ **Alphanumeric Code Fragility**: Query seeks exact code *ERR_MEM_0x8892*. Dense search finds Doc 4, but subword tokenization reduces score margin compared to lexical search.")

query_vec = DenseSemanticSearchEngine.PRESET_QUERIES[selected_query_label]
search_results = DenseSemanticSearchEngine.search(
    query_vector=query_vec,
    top_k=top_k_val,
    threshold=threshold_val,
    distance_metric=metric_choice
)

st.markdown("### 📊 Retrieved Candidates & Similarity Scores")
if search_results:
    res_df = pd.DataFrame(search_results)
    st.dataframe(res_df[["id", "title", "category", "score", "distance", "text"]], use_container_width=True)
else:
    st.warning("No documents exceeded the configured similarity threshold!")

if debug_vector_mode:
    st.markdown("#### 🔬 Query Vector Coordinates (Float32)")
    st.write(query_vec)

st.divider()

# ==============================================================================
# SECTION 7: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("7. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does a Transformer Bi-Encoder resolve word polysemy during embedding generation?"):
    st.markdown("""
    **Answer**: In classical static word embeddings (like Word2Vec or GloVe), a polysemous word like *"bank"* has a single static vector. 
    In modern Bi-Encoder architectures, texts pass through multi-head self-attention layers. 
    The token representation of *"bank"* dynamically attends to all other tokens in the sequence. 
    If surrounded by tokens like *"deposit"*, *"account"*, and *"money"*, self-attention weights shift its contextual state toward 
    the finance subspace. If surrounded by *"river"*, *"stream"*, and *"water"*, its contextual state shifts toward hydrology. 
    The final pooled sentence embedding reflects this situated contextual meaning rather than an isolated dictionary definition.
    """)

with st.expander("Q2: Why do dense embedding models exhibit severe blind spots with sentence negation?"):
    st.markdown("""
    **Answer**: Dense embedding models are trained using contrastive objective functions (e.g., InfoNCE, Multiple Negatives Ranking Loss) 
    that reward pulling semantically similar pairs together and pushing random negatives apart. 
    Because *"Drug X is recommended for condition Y"* and *"Drug X is NOT recommended for condition Y"* share 95% of the exact same topical vocabulary, 
    entities, and grammatical syntax, the encoder projects them into virtually identical vector coordinates (often with cosine similarity > 0.88). 
    The model focuses heavily on broad topical alignment and struggles with the fine-grained logical inversion introduced by single negation tokens.
    """)

with st.expander("Q3: What is the fundamental difference between Bi-Encoders and Cross-Encoders in latency and accuracy?"):
    st.markdown("""
    **Answer**: 
    * **Bi-Encoder**: Encodes query `f(Q)` and document `f(D)` completely independently into fixed-dimensional vectors. 
      Because document vectors are pre-computed offline, retrieval at query time requires only a single forward pass for the query 
      followed by an approximate nearest neighbor inner product search over millions of documents in under 5 milliseconds.
    * **Cross-Encoder**: Concatenates query and document into a single joint sequence `[CLS] Q [SEP] D [SEP]` and computes 
      full all-to-all cross-attention across all query and document tokens simultaneously. 
      This yields dramatically higher scoring accuracy and captures negation, but requires running a full Transformer forward pass 
      for every candidate document at query time, creating latency of hundreds of milliseconds. 
      Therefore, Cross-Encoders are used strictly as second-stage rerankers on Top-K candidates.
    """)

with st.expander("Q4: Why does L2-normalizing vectors before indexing significantly improve production query throughput?"):
    st.markdown("""
    **Answer**: The mathematical formula for cosine similarity requires computing both the dot product and the Euclidean norms 
    of both vectors: `dot(A, B) / (norm(A) * norm(B))`. 
    Calculating square roots and norms across millions of document candidates during a live query is computationally expensive. 
    By pre-normalizing all document vectors to unit length (`L2 norm = 1.0`) during ingestion and normalizing the query vector once upon entry, 
    the cosine similarity simplifies directly to `dot(A, B)`. 
    Vector databases can execute this as a pure dot product using hardware-accelerated SIMD instructions (AVX-512 / ARM NEON), 
    increasing throughput by up to 5x.
    """)

with st.expander("Q5: What is the hubness problem in high-dimensional vector spaces, and how does it impact retrieval?"):
    st.markdown("""
    **Answer**: In high-dimensional spaces (e.g., 768 or 1536 dimensions), a geometric phenomenon known as the **hubness problem** arises: 
    a small fraction of data points (called "hubs") become the nearest neighbors of an unnaturally large number of other points across the space. 
    In RAG systems, these hub vectors act as "attractor documents" that appear in retrieval result lists for many unrelated queries, 
    degrading precision. Mitigations include vector centering (subtracting the global mean vector), applying PCA whitening, 
    or using asymmetric cross-encoder rerankers to filter out irrelevant false-positive hubs.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Treating Cosine Similarity as a Calibrated Probability**
    * **Symptom**: Writing business logic that assumes a cosine similarity score of 0.80 means "80% confidence of relevance".
    * **Root Cause**: Embedding model cosine score distributions are uncalibrated and non-linear. Scores cluster tightly in arbitrary bands (e.g., 0.70 to 0.88). A score of 0.75 might represent a weak match in one domain and a top match in another.
    * **Remedy**: Never use hardcoded raw thresholds without domain validation. Calibrate thresholds using empirical precision-recall curves on labeled test splits, or use rank-based fusion (RRF).
    """)

    st.error("""
    **Trap 2: Silently Truncating Documents Exceeding Token Limits**
    * **Symptom**: Search engine consistently fails to retrieve critical information located in the second half of long documents.
    * **Root Cause**: Many popular embedding models (e.g., standard BERT-based models) have a hard 512-token context window. Passing a 3,000-word document into the embedding API silently truncates everything after token 512 without throwing an error.
    * **Remedy**: Implement rigorous chunking pipelines (recursive or semantic chunking) before generating embeddings, ensuring no text exceeds model context boundaries.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Embedding Query and Document with Mismatched Instruction Prefixes**
    * **Symptom**: Sudden drop in recall after migrating to modern asymmetric embedding models (e.g., E5 or BGE).
    * **Root Cause**: Modern embedding models require specific task prefixes (e.g., `"query: "` vs. `"passage: "`). Encoding both queries and documents without their designated prefixes degrades the directional alignment trained into the model.
    * **Remedy**: Standardize ingestion and query pipelines with explicit wrapper classes that automatically enforce prefix formatting.
    """)

    st.warning("""
    **Trap 4: Neglecting Memory Footprint in High-Dimension HNSW Indices**
    * **Symptom**: Vector database crashes with Out-Of-Memory (OOM) errors as corpus grows past 1 million documents.
    * **Root Cause**: Storing 1 million uncompressed 1536-dimensional float32 vectors requires 6.1 GB of RAM just for raw floats. Adding HNSW graph edges and metadata indexes expands RAM requirements to over 15–20 GB.
    * **Remedy**: Implement Scalar Quantization (SQ8) or Product Quantization (PQ) to compress vectors to 8-bit or 4-bit representations, reducing memory footprint by 75% with less than 2% recall degradation.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is a Dual-Tower Bi-Encoder and why is it essential for scalable RAG?")
        st.markdown("**Back**: An architecture where queries and documents are processed by independent neural towers. Because document vectors are computed once offline, query-time retrieval reduces to sub-millisecond vector dot products across millions of pre-indexed vectors.")

    with st.container(border=True):
        st.markdown("**Front**: How does contextual self-attention solve the problem of synonymy in search?")
        st.markdown("**Back**: Self-attention maps disparate surface tokens representing the same underlying concept (such as 'physician' and 'doctor') to proximal coordinates in the continuous latent manifold, enabling semantic matches without shared vocabulary.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Cosine Similarity mathematically identical to Dot Product on unit vectors?")
        st.markdown("**Back**: Cosine similarity equals `dot(A, B) / (||A|| * ||B||)`. When vectors A and B are pre-normalized to unit Euclidean length (`||A|| = ||B|| = 1.0`), the denominator equals 1.0, simplifying the operation to a pure inner product.")

    with st.container(border=True):
        st.markdown("**Front**: What is Negation Blindness in dense vector retrieval?")
        st.markdown("**Back**: A vulnerability where sentences with opposite factual polarity ('Drug X is approved' vs 'Drug X is NOT approved') receive misleadingly high cosine similarity (>0.85) because they share almost all domain tokens and grammatical structure.")

st.divider()

# ==============================================================================
# SECTION 8: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("8. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Dense Semantic Search, Dual-Tower Mechanics, and Vector Manifolds:
""")

chk1 = st.checkbox("I understand how Bi-Encoder dual towers enable sub-millisecond search by decoupling query and document encoding.", value=True)
chk2 = st.checkbox("I can articulate how contextual self-attention resolves polysemy by shifting token coordinates based on surrounding context.", value=True)
chk3 = st.checkbox("I recognize the key failure modes of dense retrieval, including alphanumeric code fragmentation and negation blindness.", value=True)
chk4 = st.checkbox("I understand why pre-normalizing vectors to unit length transforms expensive cosine similarity into SIMD-accelerated dot products.", value=True)
chk5 = st.checkbox("I know the trade-offs between exact Flat index search, HNSW graph traversal, and IVF-PQ quantized compression.", value=True)
chk6 = st.checkbox("I understand why cosine similarity scores do not represent calibrated probabilities and require empirical thresholding.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Phenomenal! You have mastered Dense Semantic Search and Neural Vector Retrieval architectures.")
