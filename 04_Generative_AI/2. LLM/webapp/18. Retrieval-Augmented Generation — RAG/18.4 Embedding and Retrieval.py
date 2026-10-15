import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.4 Embedding and Retrieval - LLM Engineering",
    page_icon="🧭",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class VectorDatabaseEngine:
    """
    Self-contained in-memory vector database engine implementing:
    - L2 Unit normalization
    - Cosine similarity and Dot Product matrix search
    - Pre-filtering metadata predicate evaluation
    - Top-k retrieval with similarity threshold cutoff
    """
    
    DOCUMENTS = [
        {
            "id": "VEC-001",
            "title": "Zero Trust Network Architecture Guide",
            "category": "Security",
            "tenant": "Engineering",
            "vector": np.array([0.88, 0.12, 0.35, 0.10, 0.25], dtype=np.float32)
        },
        {
            "id": "VEC-002",
            "title": "Quarterly Financial Revenue Breakdown",
            "category": "Finance",
            "tenant": "Executive",
            "vector": np.array([0.05, 0.92, 0.18, 0.30, 0.12], dtype=np.float32)
        },
        {
            "id": "VEC-003",
            "title": "Kubernetes Pod Autoscaling & Spot Instances",
            "category": "Infrastructure",
            "tenant": "Engineering",
            "vector": np.array([0.75, 0.20, 0.60, 0.15, 0.10], dtype=np.float32)
        },
        {
            "id": "VEC-004",
            "title": "Employee Parental Leave & Benefits Policy",
            "category": "HR",
            "tenant": "All_Company",
            "vector": np.array([0.10, 0.40, 0.12, 0.88, 0.20], dtype=np.float32)
        },
        {
            "id": "VEC-005",
            "title": "Hardware Firewall Rules & Port Allowlist",
            "category": "Security",
            "tenant": "Engineering",
            "vector": np.array([0.82, 0.15, 0.40, 0.08, 0.38], dtype=np.float32)
        },
        {
            "id": "VEC-006",
            "title": "Corporate Mergers & Acquisitions Audit",
            "category": "Finance",
            "tenant": "Executive",
            "vector": np.array([0.12, 0.85, 0.25, 0.35, 0.18], dtype=np.float32)
        }
    ]

    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @classmethod
    def search(
        cls,
        query_vector: np.ndarray,
        metric: str = "Cosine Similarity",
        top_k: int = 3,
        threshold: float = 0.0,
        category_filter: str = "All",
        tenant_filter: str = "All"
    ) -> List[Dict[str, Any]]:
        # Normalize query vector for clean metric evaluation
        q_norm = cls.normalize_vector(query_vector)
        
        candidates = []
        for doc in cls.DOCUMENTS:
            # Metadata pre-filtering
            if category_filter != "All" and doc["category"] != category_filter:
                continue
            if tenant_filter != "All" and doc["tenant"] != tenant_filter and doc["tenant"] != "All_Company":
                continue
                
            d_norm = cls.normalize_vector(doc["vector"])
            
            if metric == "Cosine Similarity":
                score = float(np.dot(q_norm, d_norm))
            elif metric == "Dot Product (Raw)":
                score = float(np.dot(query_vector, doc["vector"]))
            else: # Euclidean Distance (L2) converted to similarity
                l2_dist = float(np.linalg.norm(q_norm - d_norm))
                score = 1.0 / (1.0 + l2_dist)
                
            if score >= threshold:
                entry = dict(doc)
                entry["score"] = score
                candidates.append(entry)
                
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


# --- HEADER & INTRODUCTION ---
st.title("18.4 Embedding and Retrieval: Bi-Encoders, Vector Databases, & Similarity Metrics")
st.caption("Chapter 18: Retrieval-Augmented Generation — RAG | Domain: 2. Large Language Models")

st.markdown("""
How does a computer search millions of unstructured documents for concepts rather than exact matching keywords?

In traditional lexical search (like SQL `LIKE` or classic BM25), searching for *"automobile repairs"* fails to find documents 
mentioning *"car maintenance"* unless synonyms are explicitly hardcoded. 

**Dense Vector Embeddings** solve this by projecting text into high-dimensional geometric spaces (typically 768 to 1536 dimensions) 
where semantically similar ideas cluster in close spatial proximity regardless of phrasing. 
**Vector Databases** index these high-dimensional coordinates using **Approximate Nearest Neighbor (ANN)** graph algorithms 
to retrieve relevant passages in sub-10-millisecond response times.

This lesson covers the Bi-Encoder architecture, vector distance metrics, vector database indexing structures (HNSW and IVF-PQ), 
and production metadata filtering.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Dense Vector Embedding & Vector Database Retrieval Engine", divider="rainbow")

st.markdown("""
How high-dimensional embedding spaces and vector indexes orchestrate semantic retrieval:
""")

vector_blueprint_dot = """
digraph VectorSearchBlueprint {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_query {
        label="Query Processing";
        color="#1E293B";
        fontcolor="#38BDF8";
        UserQ [label="User Query\\n'How to configure pod autoscaling?'", fillcolor="#1E293B"];
        BiEncoder [label="Bi-Encoder Model\\n(e.g., BGE-M3 / text-embedding-3)\\nMean-Pools Hidden States", fillcolor="#0369A1"];
        NormQ [label="L2-Normalized Query Vector\\n||q|| = 1 in R^1536", fillcolor="#0F766E"];
    }

    subgraph cluster_db {
        label="Vector Database Engine";
        color="#1E293B";
        fontcolor="#38BDF8";
        PreFilter [label="Metadata Pre-Filter\\nEnforces Tenant / Role ACLs", fillcolor="#6D28D9"];
        HNSW [label="HNSW Graph Index\\nLogarithmic ANN Traversal\\nO(log N) Search Time", fillcolor="#4338CA"];
        MetricEngine [label="Distance Evaluator\\nDot Product / Cosine SIMD\\nHardware AVX-512 GEMV", fillcolor="#334155"];
    }

    subgraph cluster_output {
        label="Top-K Selection";
        color="#1E293B";
        fontcolor="#38BDF8";
        TopKQueue [label="Min-Heap Priority Queue\\nMaintains Top-K Candidates", fillcolor="#059669"];
        RetrievedChunks [label="Ranked Chunks\\nPassed to Prompt Builder", fillcolor="#047857"];
    }

    UserQ -> BiEncoder -> NormQ;
    NormQ -> PreFilter -> HNSW -> MetricEngine -> TopKQueue -> RetrievedChunks;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(vector_blueprint_dot, use_container_width=True)

st.markdown("""
### Vector Similarity Metrics Comparison Matrix
""")

metric_matrix = [
    {
        "Similarity Metric": "Cosine Similarity",
        "Mathematical Formula": "(A dot B) / (||A|| * ||B||)",
        "Range": "-1.0 to +1.0",
        "Geometric Interpretation": "Cosine of the angle theta between two vectors in hyperspace",
        "Hardware Characteristics": "Requires computing vector norms at runtime (division and square root)"
    },
    {
        "Similarity Metric": "Dot Product (Inner Product)",
        "Mathematical Formula": "sum(A_i * B_i)",
        "Range": "-inf to +inf",
        "Geometric Interpretation": "Projection of vector A onto vector B multiplied by length",
        "Hardware Characteristics": "Fastest hardware operation (pure fused multiply-add; identical to Cosine if vectors are L2-normalized)"
    },
    {
        "Similarity Metric": "Euclidean Distance (L2)",
        "Mathematical Formula": "sqrt( sum( (A_i - B_i)^2 ) )",
        "Range": "0.0 to +inf",
        "Geometric Interpretation": "Straight-line physical distance between vector coordinates",
        "Hardware Characteristics": "Sensitive to document length and token volume unless normalized"
    },
    {
        "Similarity Metric": "Manhattan Distance (L1)",
        "Mathematical Formula": "sum( |A_i - B_i| )",
        "Range": "0.0 to +inf",
        "Geometric Interpretation": "Grid distance along coordinate axes",
        "Hardware Characteristics": "Rarely used in dense semantic text search"
    }
]
st.dataframe(pd.DataFrame(metric_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - EMBEDDING MODELS ---
st.header("2. Embedding Models: Bi-Encoder Transformers, Pooling, & Dimensionality", divider="rainbow")

st.markdown("""
Modern dense semantic search relies on **Bi-Encoder transformer architectures** (pioneered by Sentence-BERT / Reimers & Gurevych, 2019).

### How Bi-Encoders Operate
Unlike Cross-Encoders (which process query and passage together), Bi-Encoders process documents and queries **independently**:
1. Passes text through a transformer encoder to produce a tensor of token embeddings: `[Batch_Size, Sequence_Length, Hidden_Dimension]`.
2. Applies a **pooling strategy** to collapse the variable-length token sequence into a single fixed-size vector `e in R^D`:
   - **Mean Pooling (Standard)**: Averages token embeddings across the sequence (excluding `<pad>` tokens).
   - **CLS Token Pooling**: Uses the first `[CLS]` token representation.
3. **The Precomputation Advantage**: Millions of enterprise document chunks can be embedded **once offline** and indexed. 
   At query time, the server encodes only the 10-word user query and performs an instant matrix-vector search!
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Asymmetric Task Prefixes")
    st.markdown("""
    User queries and documentation chunks have asymmetric structures:
    - Queries are short questions (*"How to setup MFA?"*).
    - Documents are long declarative manuals (*"Multi-factor authentication setup guide for engineers..."*).
    - Leading embedding models (**BGE-M3**, **E5**, **Voyage**) use asymmetric training prefixes:
      - Query: `"query: How to setup MFA?"`
      - Document: `"passage: Multi-factor authentication setup guide..."`
    """)

with col2:
    st.subheader("Matryoshka Representation Learning (MRL)")
    st.markdown("""
    Pioneered by Kusupati et al. (2022) and implemented in OpenAI `text-embedding-3`:
    - Embeddings are trained such that **the first N dimensions contain the majority of information**.
    - You can truncate a 1536-dimensional vector down to **256 dimensions**:
      - Reduces vector database storage and RAM by **83%**!
      - Retains over **98% of full-dimensional retrieval accuracy**.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - VECTOR DATABASES ---
st.header("3. Vector Databases: Specialized Engines, HNSW Graphs, & Quantization (IVF-PQ)", divider="rainbow")

st.markdown("""
Why can't we use traditional SQL databases (like PostgreSQL with B-Trees) for high-dimensional vector search?

In a relational database, B-Trees index scalar values along a 1-dimensional order (`A < B < C`). 
In high-dimensional spaces (e.g. 1536 dimensions), there is no natural linear order: the **Curse of Dimensionality** forces 
an exhaustive linear scan through every single document vector `O(N * D)`. 
For 10,000,000 documents, a single query requires evaluating 15 billion floating-point multiplications, taking 10+ seconds per search!

### Approximate Nearest Neighbor (ANN) Index Structures
Vector databases achieve sub-10ms search by accepting a 1% compromise in mathematical recall through specialized index structures:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. HNSW (Hierarchical Navigable Small World)")
    st.markdown("""
    - Inspired by skip-lists and small-world network graphs!
    - Builds a multi-layer graph where:
      - Top layers have long-distance edges between distant vector clusters (fast coarse routing).
      - Bottom layer contains dense, local edges between nearest neighbors.
    - Achieves **logarithmic search time `O(log N)`**!
    - The gold standard for low-latency in-memory vector search (used in **Milvus**, **Qdrant**, **Pinecone**, and **Chroma**).
    """)

with col2:
    st.subheader("2. IVF-PQ (Inverted File + Product Quantization)")
    st.markdown("""
    - **Inverted File (IVF)**: Partitions the vector space into Voronoi cells using k-means clustering. At query time, only the closest centroid clusters are searched.
    - **Product Quantization (PQ)**: Decomposes 1536-dimensional float32 vectors into 8-bit quantized byte codes:
      - Slashes VRAM footprint by **95%**!
      - Allows 100M+ vectors to fit into a single server's RAM.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - SIMILARITY SEARCH ---
st.header("4. Similarity Search: Approximate Nearest Neighbor (ANN) vs Exact KNN", divider="rainbow")

st.markdown("""
In vector search engineering, systems architects navigate the **Trade-Off Triangle**:
""")

search_triangle = """
digraph TradeOffTriangle {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    Recall [label="1. Search Recall Accuracy\\n(Target: 95% - 99% Recall)", fillcolor="#0F766E"];
    Throughput [label="2. Query Throughput (QPS) & Latency\\n(Target: < 10ms per query)", fillcolor="#0369A1"];
    Memory [label="3. Index Build Time & RAM Consumption\\n(Target: Fit within server memory budget)", fillcolor="#6D28D9"];

    Recall -> Throughput [dir=both, label="Trade-off"];
    Throughput -> Memory [dir=both, label="Trade-off"];
    Memory -> Recall [dir=both, label="Trade-off"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(search_triangle, use_container_width=True)

st.markdown("""
### Exact k-Nearest Neighbors (KNN) vs ANN
- **Exact KNN**: Computes the true Euclidean or Cosine distance against 100% of the dataset. Guaranteed 100% recall. 
  Viable only for datasets smaller than 50,000 vectors.
- **ANN (HNSW / IVF)**: Trades 1-2% of recall accuracy for a **100x to 1,000x latency acceleration**. 
  In generative RAG, an ANN recall of 96% is indistinguishable from 100% because downstream LLM reasoning tolerates minor rank variances.
""")

st.divider()

# --- SECTION 5: TOPIC 4 - COSINE SIMILARITY ---
st.header("5. Cosine Similarity: Angular Alignment, Normalization, & Metric Equivalence", divider="rainbow")

st.markdown("""
**Cosine Similarity** evaluates the cosine of the angle `theta` between two vectors in hyperspace:

```text
Cosine_Similarity(A, B) = (A dot B) / ( ||A||_2 * ||B||_2 )
```

### The Length-Invariance Property
Unlike Euclidean distance (which grows larger if a document is long and contains many repeated words), 
Cosine Similarity measures purely **angular orientation**, making it completely invariant to document length!
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Normalization Equivalence Theorem")
    st.markdown("""
    If all document vectors `d` and query vectors `q` are **L2-normalized to the unit sphere** during ingestion (`||v||_2 = 1.0`):

    ```text
    Cosine_Similarity(q, d) = (q dot d) / ( 1.0 * 1.0 ) = q dot d
    ```

    **Operational Takeaway**:
    By normalizing vectors offline, Cosine Similarity simplifies to a **pure Dot Product (Inner Product)**!
    - Eliminates expensive runtime division and square root calculations.
    - Converts vector search into high-speed GPU Matrix-Vector multiplication (`GEMV`).
    """)

with col2:
    st.subheader("Relationship to Euclidean Distance")
    st.markdown("""
    For unit-normalized vectors (`||A|| = ||B|| = 1`), Euclidean distance `||A - B||^2` is mathematically tied to Cosine Similarity:

    ```text
    ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * (A dot B) = 2 - 2 * Cosine_Similarity(A, B)
    ```

    Thus, ranking by highest Cosine Similarity produces the **exact same ranking** as ordering by lowest Euclidean Distance!
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - TOP-K RETRIEVAL ---
st.header("6. Top-K Retrieval: Score Thresholds, Priority Queues, & Context Budgets", divider="rainbow")

st.markdown("""
Once vector similarity scores are computed across the candidate pool, the engine extracts the top `K` chunks 
using a **Min-Heap Priority Queue** in `O(N log K)` time.

### Calibrating Top-K and Absolute Score Thresholds
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. The Dangers of Fixed Top-K")
    st.markdown("""
    In naive RAG systems, engineers configure `top_k = 5` statically:
    - If a user asks a question with **no relevant documents** in the database, the engine still extracts the 5 highest-scoring documents!
    - Even if their similarity scores are terrible (`0.25`), they are injected into the prompt, forcing the LLM to hallucinate from irrelevant text.
    """)

with col2:
    st.subheader("2. Score Threshold Cutoff (`score >= tau`)")
    st.markdown("""
    Production systems enforce an absolute similarity threshold (e.g. `threshold = 0.70`):
    - Chunks scoring below `0.70` are discarded before prompt formatting.
    - If zero chunks exceed the threshold, the system triggers fallback logic: *"No verified documentation exists for this query."*
    - Prevents hallucination at the source!
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - METADATA FILTERING ---
st.header("7. Metadata Filtering: Pre-Filtering vs Post-Filtering & Hybrid Predicates", divider="rainbow")

st.markdown("""
Enterprise search is rarely pure vector similarity; it is almost always **Vector Search + Relational Filtering**:
*"Find the top-3 chunks semantically matching 'bonus policy' WHERE department = 'Finance' AND year = 2026 AND access_level <= 3"*.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Post-Filtering (The High-Recall Failure)")
    st.markdown("""
    1. The vector engine performs ANN search and retrieves the top-50 nearest semantic vectors globally.
    2. The application filters those 50 chunks by `department = 'Finance'`.
    3. **Fatal Flaw**: If only 1 of the top-50 chunks happens to belong to Finance, the user receives only 1 chunk (or zero chunks!), even though thousands of relevant Finance documents exist in the database!
    """)

with col2:
    st.subheader("Pre-Filtering (Iterative Graph Filtering)")
    st.markdown("""
    1. The metadata filter evaluates first, identifying the subset of vector IDs matching the boolean predicate.
    2. Vector search traverses the HNSW graph **only through nodes that satisfy the metadata condition**.
    3. Guarantees that the engine returns the true top-K semantic matches that satisfy enterprise security and classification rules.
    """)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: In-Memory Vector Store with Cosine ANN & Pre-Filtering in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements a production-grade in-memory vector store. 
It features unit normalization, Cosine matrix similarity, metadata pre-filtering, and priority-queue Top-K selection:
""")

code_block = '''import numpy as np
from typing import List, Dict, Any

class ProductionVectorStore:
    """
    Self-contained in-memory vector database implementing L2-normalization,
    fast matrix dot product similarity, metadata pre-filtering, and thresholding.
    """
    
    def __init__(self, dimension: int = 5):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        
    def add_document(self, doc_id: str, vector: np.ndarray, meta: Dict[str, Any]):
        """Normalizes vector to unit sphere and inserts into vector matrix."""
        # 1. Normalize vector to unit length ||v|| = 1.0
        norm = np.linalg.norm(vector)
        norm_vec = (vector / norm) if norm > 0 else vector
        norm_vec = norm_vec.astype(np.float32)
        
        # 2. Append to storage
        self.doc_ids.append(doc_id)
        self.metadata.append(meta)
        if self.vectors.shape[0] == 0:
            self.vectors = norm_vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, norm_vec])
            
    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.0,
        filter_fn = None
    ) -> List[Dict[str, Any]]:
        """
        Executes pre-filtered vector similarity search:
        - Evaluates filter_fn on metadata first
        - Computes dot product on normalized vectors (equivalent to Cosine Similarity)
        - Returns ranked candidates above threshold
        """
        if self.vectors.shape[0] == 0:
            return []
            
        # 1. Normalize query vector
        q_norm = query_vector / np.linalg.norm(query_vector)
        
        # 2. Pre-filter candidate indices
        candidate_indices = []
        for idx, meta in enumerate(self.metadata):
            if filter_fn is None or filter_fn(meta):
                candidate_indices.append(idx)
                
        if not candidate_indices:
            return []
            
        # 3. Batch matrix-vector dot product (SIMD GEMV)
        sub_matrix = self.vectors[candidate_indices]
        scores = np.dot(sub_matrix, q_norm)
        
        # 4. Filter by threshold and rank
        results = []
        for local_idx, score in enumerate(scores):
            if score >= threshold:
                orig_idx = candidate_indices[local_idx]
                results.append({
                    "id": self.doc_ids[orig_idx],
                    "score": float(score),
                    "metadata": self.metadata[orig_idx]
                })
                
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: High-Dimensional Vector Search & Embedding Space Visualizer", divider="rainbow")

st.markdown("""
Interact with a live vector search engine: test how **Metric Formulas**, **Top-K Limits**, **Score Thresholds**, 
and **Metadata Pre-Filters** alter retrieval results in real time:
""")

# Define synthetic queries with distinct coordinate directions
query_presets = {
    "Cybersecurity / Zero Trust Policy": np.array([0.90, 0.10, 0.30, 0.05, 0.20], dtype=np.float32),
    "Quarterly Earnings & Revenue": np.array([0.08, 0.95, 0.15, 0.25, 0.10], dtype=np.float32),
    "Cloud Infrastructure & Kubernetes": np.array([0.70, 0.22, 0.65, 0.10, 0.15], dtype=np.float32),
    "HR Parental Leave & Childcare": np.array([0.10, 0.35, 0.10, 0.92, 0.18], dtype=np.float32)
}

wb_col1, wb_col2, wb_col3 = st.columns(3)
with wb_col1:
    selected_query_title = st.selectbox("Select Semantic Query Concept", list(query_presets.keys()))
    active_q_vec = query_presets[selected_query_title]
with wb_col2:
    selected_metric = st.selectbox("Similarity Metric Formula", ["Cosine Similarity", "Dot Product (Raw)", "Euclidean Distance (L2)"])
    top_k_val = st.slider("Top-K Candidates", min_value=1, max_value=6, value=3, step=1)
with wb_col3:
    threshold_val = st.slider("Minimum Similarity Threshold (tau)", min_value=0.0, max_value=0.95, value=0.50, step=0.05)
    category_filter_val = st.selectbox("Metadata Category Pre-Filter", ["All", "Security", "Finance", "Infrastructure", "HR"])

# Run live search
search_results = VectorDatabaseEngine.search(
    query_vector=active_q_vec,
    metric=selected_metric,
    top_k=top_k_val,
    threshold=threshold_val,
    category_filter=category_filter_val,
    tenant_filter="All"
)

st.subheader(f"Retrieved Top-{top_k_val} Documents (Filtered & Ranked)")
if search_results:
    res_table = []
    for r in search_results:
        res_table.append({
            "Doc ID": r["id"],
            "Document Title": r["title"],
            "Category": r["category"],
            "Tenant Access": r["tenant"],
            "Similarity Score": f"{r['score']:.4f}"
        })
    st.dataframe(pd.DataFrame(res_table), use_container_width=True)
else:
    st.warning("No documents satisfied both the metadata pre-filter and the minimum similarity threshold.")

st.markdown("### Vector Alignment Geometry")
st.caption("Visualizing the raw coordinate vectors across the 5 latent embedding dimensions:")
coord_df = pd.DataFrame({
    "Dimension": [f"Dim {i+1}" for i in range(5)],
    "Query Vector": VectorDatabaseEngine.normalize_vector(active_q_vec)
}).set_index("Dimension")

for r in search_results[:2]:
    coord_df[f"{r['id']} ({r['category']})"] = VectorDatabaseEngine.normalize_vector(r["vector"])

st.bar_chart(coord_df)

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why is Dot Product mathematically identical to Cosine Similarity when vectors are L2-normalized?",
        "a": "Cosine similarity is defined as CosSim(A, B) = (A dot B) / (||A||_2 * ||B||_2). If both vector A and vector B are normalized to the unit sphere during indexing (such that ||A||_2 = 1.0 and ||B||_2 = 1.0), the denominator simplifies to 1.0 * 1.0 = 1.0. Therefore, CosSim(A, B) = A dot B. Normalizing offline allows vector databases to execute high-speed inner product GEMV instructions on GPU and SIMD AVX-512 hardware without expensive runtime square root and division operations."
    },
    {
        "q": "How does the HNSW graph achieve logarithmic search time in high-dimensional vector spaces?",
        "a": "Hierarchical Navigable Small World (HNSW) builds a multi-layer graph inspired by skip-lists. The upper layers contain a sparse graph with long-range edges connecting distant vector clusters, allowing the search algorithm to greedily traverse large spatial distances in a few hops. As the search descends into lower layers, the graph becomes progressively denser with local nearest-neighbor edges. This multi-scale hierarchical routing avoids linear scanning and achieves O(log N) search complexity."
    },
    {
        "q": "What is the fatal flaw of Post-Filtering in vector databases?",
        "a": "Post-filtering executes the Approximate Nearest Neighbor (ANN) search across the entire global vector index first, retrieving top-N candidates, and then discards candidates that fail the metadata predicate (e.g. department = 'Finance'). If the metadata filter is highly selective (e.g. Finance documents represent only 1% of the database), all or nearly all of the top-N vectors will be discarded, returning zero or fewer results than requested. Pre-filtering traverses the graph strictly within valid metadata nodes, guaranteeing that top-K candidates are found."
    },
    {
        "q": "What is Matryoshka Representation Learning (MRL) and how does it optimize vector database costs?",
        "a": "MRL (Kusupati et al.) trains embedding models with a nested loss function such that the earlier dimensions (e.g. the first 256 or 512 coordinates) contain the majority of the representational entropy. This allows engineering teams to truncate 1536-dimensional vectors down to 256 dimensions. Truncation slashes vector memory and storage requirements by up to 83% while preserving over 98% of the retrieval accuracy, significantly reducing vector database hosting costs."
    },
    {
        "q": "Why do modern Bi-Encoder embedding models require separate query and passage prefixes?",
        "a": "In retrieval benchmarks, user queries are short questions (low word count, interrogative grammar), while document chunks are long statements (high word count, declarative grammar). If encoded identically, the model's semantic representations may suffer from format mismatch. Models like E5 and BGE prepend task instructions (e.g. 'query: ' vs 'passage: ') to condition the transformer attention heads to project questions and answers into compatible geometric regions."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Evaluating Un-Normalized Dot Product on Variable-Length Texts**
    Using un-normalized Dot Product as the search metric. Longer documents naturally accumulate larger L2 vector norms. 
    At query time, the dot product will automatically favor longer documents regardless of true semantic relevance. 
    Always L2-normalize vectors during ingestion if using inner product search.
    """)
    st.warning("""
    **Trap 3: Inadequate HNSW `ef_search` Setting During High-QPS Load**
    Setting the HNSW search parameter `ef_search` too low to minimize latency. While this reduces search time to under 1ms, 
    recall accuracy can plummet from 98% down to 65%, resulting in missing critical documents. Benchmark recall curves before freezing `ef_search`.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Storing Uncompressed FP32 Vectors in RAM for 10M+ Documents**
    Keeping full 1536-dimensional 32-bit floating-point vectors uncompressed in memory. 10 million vectors require 
    `10,000,000 * 1536 * 4 bytes = 61.4 GB` of pure RAM just for coordinates (plus HNSW graph overhead = ~100 GB). 
    Use Product Quantization (PQ) or Matryoshka dimension truncation to reduce memory footprint by 80%+.
    """)
    st.warning("""
    **Trap 4: Missing Score Threshold Leading to Hallucinatory Context Flooding**
    Configuring static `top_k = 5` without a similarity threshold cutoff. When users submit out-of-domain queries, 
    the engine injects low-scoring irrelevant chunks (`score < 0.35`) into the LLM prompt, prompting the model to hallucinate. 
    Always enforce a minimum score cutoff.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Dot Product vs Cosine"):
        st.markdown("**When are Dot Product and Cosine equal?**\\n\\nWhen vectors are L2-normalized to unit length (`||v|| = 1.0`).")
with fc2:
    with st.popover("Flashcard 2: HNSW Complexity"):
        st.markdown("**What is HNSW search complexity?**\\n\\n`O(log N)` logarithmic time via hierarchical skip-list graph traversal.")
with fc3:
    with st.popover("Flashcard 3: Pre vs Post Filter"):
        st.markdown("**Why pre-filter metadata?**\\n\\nPost-filtering discards top-N candidates and can yield 0 results on selective predicates.")
with fc4:
    with st.popover("Flashcard 4: MRL Truncation"):
        st.markdown("**What is MRL?**\\n\\nMatryoshka Representation Learning: truncates 1536d to 256d, saving 80%+ RAM with <2% accuracy loss.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand how Bi-Encoder transformers project queries and passages into dense latent coordinates independently.",
    "I can derive the Normalization Equivalence Theorem proving that Dot Product equals Cosine Similarity on unit vectors.",
    "I understand how HNSW graph skip-lists achieve logarithmic search complexity O(log N) compared to linear KNN.",
    "I know why post-filtering metadata fails on selective predicates and why pre-filtering graph traversal is mandatory.",
    "I can explain Matryoshka Representation Learning (MRL) and how dimension truncation reduces vector DB hosting costs.",
    "I understand the necessity of similarity score thresholding to prevent out-of-domain context flooding and hallucination."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of vector embeddings, vector databases, similarity metrics, and metadata filtering.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
