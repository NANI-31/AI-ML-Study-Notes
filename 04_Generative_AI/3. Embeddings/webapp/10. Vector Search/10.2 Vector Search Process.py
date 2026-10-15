import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
import heapq
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="10.2 Vector Search Process - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("10.2 Vector Search Process: The 6-Stage Execution Lifecycle, Heap Ranking, & Latency Budgets")
st.caption("Chapter 10: Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
When a user types a query into a search bar or AI chatbot, how does an enterprise system retrieve relevant context in under **50 milliseconds** across millions of documents?

The **Vector Search Process** coordinates a strict, sequential pipeline across **6 operational stages**:
1. **Query Ingestion**: Capturing user input, sanitizing whitespace, and injecting asymmetric task prompts (e.g. `query: `).
2. **Query Vector Encoding**: Tokenizing and forward-passing through the embedding model to produce a unit-normalized query vector.
3. **Index Traversal**: Traversing vector graph structures (HNSW) or inverted index clusters (IVF) to identify candidate nearest neighbors.
4. **Metric Calculation**: Computing inner dot products against candidate vectors.
5. **Heap-Based Top-K Ranking**: Using a bounded priority queue (min-heap of size $K$) to maintain top scores in $O(N \\log K)$ time.
6. **Payload Hydration & Assembly**: Fetching chunk text, source URLs, and metadata tags from disk storage to assemble the final response.

This lesson explores each stage of the vector search lifecycle, its computational bottlenecks, and latency budget management.
""")

st.divider()

# Section 1: The 6-Stage Retrieval Lifecycle
st.header("1. The 6-Stage Retrieval Lifecycle & Latency Budgets", divider="rainbow")

col_cyc1, col_cyc2 = st.columns([1, 1])

with col_cyc1:
    st.markdown("""
    ### Stage-by-Stage Operational Breakdown:
    * **Stage 1: Query Ingestion (1 - 2 ms)**:
      * Sanitizes user input and attaches model-specific prefixes (e.g. `"Represent this query for retrieving documents: "`).
    * **Stage 2: Query Encoding (8 - 25 ms)**:
      * **Primary Latency Bottleneck**: Running a neural forward pass through transformer layers.
      * *Optimization*: Run query encoder on GPU (FP16) or deploy optimized ONNX/TensorRT engines.
    * **Stage 3: Index Traversal (2 - 6 ms)**:
      * Navigates HNSW graph layers or IVF centroid Voronoi cells to prune non-candidate vectors.
    * **Stage 4: Similarity Computation (1 - 3 ms)**:
      * Computes inner products using SIMD / AVX-512 hardware instructions.
    * **Stage 5: Heap-Based Ranking (1 - 2 ms)**:
      * Rather than sorting all $N$ documents ($O(N \\log N)$), maintains a min-heap of size $K$ ($O(N \\log K)$).
    * **Stage 6: Payload Hydration (2 - 5 ms)**:
      * Vector IDs are mapped to SQLite / RocksDB key-value stores to retrieve raw text and metadata.
    """)

with col_cyc2:
    st.markdown("### The 6-Stage Lifecycle Architecture")
    proc_dot = """
    digraph SearchLifecycle {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        S1 [label="1. Ingest & Prefix\\nAttach 'query: '", fillcolor="#EFF6FF", color="#3B82F6"];
        S2 [label="2. Query Encoding (Encoder)\\nForward pass -> Unit L2 vector", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];
        S3 [label="3. Index Traversal (HNSW)\\nGraph pruning across layers", fillcolor="#D1FAE5"];
        S4 [label="4. Metric Dot Product\\nInner product calculation", fillcolor="#FDE68A"];
        S5 [label="5. Heap Ranking (Min-Heap)\\nBounded top-K collection", fillcolor="#EDE9FE", color="#8B5CF6"];
        S6 [label="6. Payload Hydration (RocksDB)\\nAttach text & metadata", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        S1 -> S2 -> S3 -> S4 -> S5 -> S6;
    }
    """
    st.graphviz_chart(proc_dot, use_container_width=False)

st.markdown("### Target Production Latency Budget (Target: < 45 ms)")
st.markdown("""
| Lifecycle Stage | Component | Typical CPU Latency | Typical GPU Latency | Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ingest & Prefix** | Gateway App Server | 0.5 ms | 0.5 ms | In-memory string formatting |
| **2. Query Encoding** | Transformer Model | 28.0 ms | **8.5 ms** | FP16 Half Precision / ONNX Runtime |
| **3. Index Traversal** | HNSW Graph Engine | 4.0 ms | 1.5 ms | In-memory RAM index with HNSW $M=32$ |
| **4. Metric Dot Product** | Vector Math (BLAS) | 2.5 ms | 0.8 ms | AVX-512 SIMD vector instructions |
| **5. Heap Ranking** | Min-Heap Priority Queue | 1.0 ms | 0.5 ms | C++ `std::priority_queue` |
| **6. Payload Hydration** | Key-Value Document Store | 3.0 ms | 2.0 ms | RocksDB / LMDB memory-mapped cache |
| **Total End-to-End** | **Full Pipeline** | **~39.0 ms** | **~13.8 ms** | Sub-50ms SLA achieved |
""")

st.divider()

# Section 2: Production Code: Heap-Based Vector Search Engine
st.header("2. Production Implementation: Heap-Based Vector Search Engine", divider="rainbow")

st.markdown("""
The production class below executes the complete 6-stage lifecycle using a **Min-Heap priority queue** of size $K$ to guarantee $O(N \\log K)$ time complexity, followed by payload hydration.
""")

st.code(r'''
import heapq
import numpy as np
from typing import List, Dict, Any

class ProductionVectorSearchEngine:
    """
    Executes the 6-stage vector search lifecycle with heap-based ranking
    and document payload hydration.
    """
    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim
        # Vector index: list of tuples (doc_id, normalized_vector)
        self.vector_index: List[Tuple[str, np.ndarray]] = []
        # Payload store: doc_id -> metadata dictionary
        self.payload_store: Dict[str, Dict[str, Any]] = {}

    def insert_document(self, doc_id: str, text: str, metadata: Dict[str, Any], vector: np.ndarray):
        """Indexes vector and stores payload."""
        norm_v = vector / np.maximum(np.linalg.norm(vector), 1e-12)
        self.vector_index.append((doc_id, norm_v))
        self.payload_store[doc_id] = {"text": text, **metadata}

    def search_top_k(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes bounded top-K retrieval using a Min-Heap.
        Maintains only K candidates in memory at any point.
        Complexity: O(N * log K) where N = len(index).
        """
        q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-12)
        
        # Min-heap stores tuples of (score, doc_id)
        # The lowest score among the top-K is at heap[0]
        min_heap: List[Tuple[float, str]] = []

        for doc_id, doc_vec in self.vector_index:
            score = float(np.dot(q_norm, doc_vec))
            
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (score, doc_id))
            else:
                if score > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (score, doc_id))

        # Sort the top-K items descending
        sorted_top = sorted(min_heap, key=lambda x: x[0], reverse=True)

        # Stage 6: Hydrate with stored payloads
        results = []
        for score, doc_id in sorted_top:
            payload = self.payload_store.get(doc_id, {})
            results.append({
                "doc_id": doc_id,
                "score": score,
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"}
            })
        return results
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveLifecycleTracker:
    def __init__(self):
        # Sample technical knowledge base with 4D vectors
        # Coordinates: [Cloud/Infra, AI/ML, Security, Frontend/UI]
        self.corpus = [
            {"id": "DOC_101", "title": "Kubernetes Pod Autoscaling", "text": "Horizontal Pod Autoscaler monitors CPU utilization and dynamically scales replica pods.", "vec": np.array([0.95, 0.10, 0.15, 0.02]), "cat": "DevOps"},
            {"id": "DOC_102", "title": "Transformer Attention Layers", "text": "Multi-head attention projects query, key, and value vectors into multiple representation subspaces.", "vec": np.array([0.08, 0.96, 0.05, 0.05]), "cat": "Machine Learning"},
            {"id": "DOC_103", "title": "Zero-Trust Identity Authentication", "text": "Mutual TLS and hardware security tokens enforce strict authentication at every service boundary.", "vec": np.array([0.15, 0.05, 0.95, 0.02]), "cat": "Security"},
            {"id": "DOC_104", "title": "React Server Components Rendering", "text": "Server components execute exclusively on the server, streaming HTML chunks to the browser client.", "vec": np.array([0.05, 0.05, 0.02, 0.96]), "cat": "Frontend"},
            {"id": "DOC_105", "title": "Cloud Disaster Recovery Backup", "text": "Automated snapshot replication across multi-region object storage guarantees data durability.", "vec": np.array([0.88, 0.05, 0.35, 0.01]), "cat": "DevOps"}
        ]

    def execute_lifecycle(self, raw_query: str, query_vec: np.ndarray, top_k: int) -> Dict[str, Any]:
        # Stage 1: Ingest & Prefix
        prefixed_q = f"query: {raw_query.strip()}"
        
        # Stage 2: Encode & Normalize
        q_norm = query_vec / np.linalg.norm(query_vec)
        
        # Stage 3, 4, 5: Traversal, Dot product, Heap Ranking
        min_heap = []
        for doc in self.corpus:
            d_norm = doc["vec"] / np.linalg.norm(doc["vec"])
            sim = float(np.dot(q_norm, d_norm))
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, doc))
            else:
                if sim > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (sim, doc))
                    
        sorted_hits = sorted(min_heap, key=lambda x: x[0], reverse=True)
        
        # Stage 6: Hydrate
        results = []
        for sim, d in sorted_hits:
            results.append({
                "doc_id": d["id"],
                "title": d["title"],
                "category": d["cat"],
                "text": d["text"],
                "score": sim
            })

        return {
            "prefixed_query": prefixed_q,
            "query_norm_vector": q_norm[:4].tolist(),
            "results": results
        }

st.divider()

# Section 3: Interactive Workbench: 6-Stage Lifecycle Tracker
st.header("3. Interactive Workbench: 6-Stage Vector Search Lifecycle Simulator", divider="rainbow")

st.markdown("""
Trace an incoming search query step-by-step through the 6 operational stages of the vector search engine.
Observe how the raw query text is prefixed, encoded, compared against the corpus via dot products, ranked using a min-heap, and hydrated with stored metadata!
""")

tracker = InteractiveLifecycleTracker()

col_tr_l, col_tr_r = st.columns([1, 1])

with col_tr_l:
    st.subheader("1. Ingest User Query")
    query_presets = {
        "Container scaling and pod replication": {
            "text": "Container scaling and pod replication",
            "vec": np.array([0.94, 0.08, 0.12, 0.02]),
            "target": "DevOps"
        },
        "Deep neural network attention mechanisms": {
            "text": "Deep neural network attention mechanisms",
            "vec": np.array([0.05, 0.95, 0.04, 0.04]),
            "target": "Machine Learning"
        },
        "Mutual TLS token authentication security": {
            "text": "Mutual TLS token authentication security",
            "vec": np.array([0.12, 0.04, 0.96, 0.01]),
            "target": "Security"
        }
    }

    q_choice = st.selectbox("Select Test Query:", list(query_presets.keys()))
    q_item = query_presets[q_choice]
    top_k_val = st.slider("Select Top-K Results to Return:", min_value=1, max_value=4, value=3)

    out = tracker.execute_lifecycle(q_item["text"], q_item["vec"], top_k_val)

    st.markdown("#### Lifecycle Stage Trace:")
    st.markdown(f"**Stage 1 (Prefixed Query)**: `{out['prefixed_query']}`")
    st.markdown(f"**Stage 2 (Encoded Vector)**: `[{', '.join([f'{v:.3f}' for v in out['query_norm_vector']])}]`")
    st.markdown(f"**Stage 5 (Heap Capacity)**: Bounded min-heap of size K={top_k_val}")

with col_tr_r:
    st.subheader("2. Hydrated Retrieval Results (Stage 6)")
    df_res = pd.DataFrame(out["results"])
    df_res.rename(columns={
        "doc_id": "Doc ID",
        "title": "Document Title",
        "category": "Domain",
        "score": "Cosine Similarity",
        "text": "Hydrated Content Payload"
    }, inplace=True)

    st.dataframe(
        df_res.style.format({"Cosine Similarity": "{:.4f}"}).background_gradient(subset=["Cosine Similarity"], cmap="Blues"),
        use_container_width=True
    )

    top_doc = out["results"][0]
    st.success(f"🎯 **Top Ranked Document**: [{top_doc['doc_id']}] **{top_doc['title']}** (Cosine: {top_doc['score']:.4f})")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Process Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Min-Heap used instead of a full sort when collecting Top-K results in vector search?**
    * **Answer**:
      * In a database of $N = 10,000,000$ documents, sorting all similarity scores requires $O(N \\log N)$ time, performing $\approx 2.3 \\times 10^8$ operations.
      * A **Min-Heap** of size $K$ (e.g. $K = 5$) maintains only the top 5 scores seen so far:
        1. Compare incoming document score with `heap[0]` (the smallest of the top-5).
        2. If larger, push-pop in $O(\\log K)$ time.
        3. Total time complexity is $O(N \\log K)$.
      * For $N = 10,000,000$ and $K = 5$, $\\log K \\approx 2.3$, reducing sorting overhead by **10x**!

    **Q2: What is 'Payload Hydration' and why is it separated from the vector index?**
    * **Answer**:
      * Modern vector indexes (HNSW, FAISS) reside in **fast RAM** to support sub-millisecond graph traversals.
      * Text chunks and metadata (e.g. 1 KB per chunk) are bulky. Storing 10 million raw text chunks in RAM would consume 10 GB+ of unnecessary memory, starving the CPU cache.
      * Therefore, the vector index stores **only (vector_id, vector_embedding)**.
      * Once the top $K$ IDs are identified, the system performs **Payload Hydration**: fetching the raw text and JSON metadata from a separate, disk-backed key-value store (RocksDB, LMDB, PostgreSQL) in a single fast batched lookup.

    **Q3: How do you achieve sub-50ms P95 latency in enterprise vector search?**
    * **Answer**:
      * 1. **GPU Query Encoding**: Run the query embedding model on a dedicated GPU worker with FP16 precision (reducing encoding latency from 40ms on CPU to <8ms).
      * 2. **HNSW RAM Indexing**: Keep graph edges in RAM; tune $M=16$ and `ef_search=64` for optimal latency-recall tradeoff.
      * 3. **Batch Payload Lookup**: Fetch all $K$ document payloads in a single multi-get I/O operation rather than sequential lookups.
      * 4. **Query Caching**: Cache query embeddings and top-K results for frequent queries (e.g. Redis LRU cache).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Hydrating All Document Candidates Before Filtering**:
      * *Issue*: Reading text payloads from disk for all candidates before applying top-K or threshold filters.
      * *Consequence*: Massive disk I/O bottleneck; search latency spikes by 10x-50x.
      * *Fix*: Filter and rank in pure vector space first; hydrate text ONLY for the final top-K items.

    * **Trap 2: Forgetting Asymmetric Query Instruction Prefixes**:
      * *Issue*: Passing the raw query string into models like BGE or E5 without prepending the required instruction prefix (`query: `).
      * *Consequence*: The model encodes the query in document space rather than query space, causing a 15-25% drop in retrieval hit-rate!
      * *Fix*: Implement strict wrapper classes that automatically prepend model-specific prefixes.

    * **Trap 3: Synchronous Re-Embedding of Queries**:
      * *Issue*: Making sequential HTTP calls to embedding APIs without connection pooling.
      * *Consequence*: Each query incurs 20ms of TLS handshake overhead.
      * *Fix*: Maintain persistent HTTP sessions (`urllib3.PoolManager` or `requests.Session`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Min-Heap Top-K**
      * *Definition*: A priority queue data structure that maintains the K highest-scoring items in O(N log K) time without sorting the entire corpus.
    * **Flashcard 2: Payload Hydration**
      * *Definition*: The final retrieval step of fetching raw text, URLs, and metadata attributes for the top-K vector IDs from disk storage.
    * **Flashcard 3: Query Encoding Latency**
      * *Definition*: The time required to pass the user's query through the transformer model, typically accounting for 60-80% of total search latency.
    * **Flashcard 4: Asymmetric Prompt Steering**
      * *Definition*: Prepending a task instruction ('query: ') to user queries to project them into the geometric manifold of explanatory answers.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of the Vector Search Process:

- [ ] I can describe the 6 sequential stages of the vector search execution lifecycle.
- [ ] I understand why Min-Heap priority queues are used instead of full sorting for Top-K ranking.
- [ ] I understand what Payload Hydration is and why text is stored separately from vector index RAM.
- [ ] I can articulate the latency breakdown and know how to achieve a sub-50ms P95 search budget.
- [ ] I know why asymmetric instruction prefixes are required during query ingestion.
""")
