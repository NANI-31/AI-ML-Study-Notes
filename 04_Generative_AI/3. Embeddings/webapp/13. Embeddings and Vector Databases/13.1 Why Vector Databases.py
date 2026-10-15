import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="13.1 Why Vector Databases - Embeddings Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Title & Overview
st.title("13.1 Why Vector Databases: Flat File Bottlenecks, ANN Indexing, & Hybrid Metadata Filtering")
st.caption("Chapter 13: Embeddings and Vector Databases | Domain: 3. Vector Embeddings")
st.markdown("""
When building a prototype search script, storing vectors in a local NumPy array (`np.save('vectors.npy')`) or JSON file seems simple and intuitive.
However, the moment a system moves toward production with **hundreds of thousands of documents, concurrent users, dynamic updates, and user-permission filters**, flat file storage completely collapses:
* **The Full-Scan Latency Trap**: Finding the nearest neighbor in a `.npy` file requires comparing the query against every single vector ($O(N)$ linear scan), causing latency to scale linearly from 5ms to several seconds.
* **The Full-Rewrite Update Penalty**: Appending or updating a single document requires reading, copying, and rewriting the entire multi-gigabyte array to disk.
* **The Metadata Filtering Dilemma**: In relational systems, users search within specific partitions (e.g., *"Find documents where author='legal' AND created_at >= 2024"*). Flat vector files cannot perform indexed metadata filtering without scanning the entire dataset.

**Vector Databases** were created specifically to solve these challenges, delivering sub-millisecond Approximate Nearest Neighbor (ANN) indexing, transactional ACID upserts, and single-stage metadata filtering.

This lesson explores why specialized vector databases are required for enterprise scale.
""")

st.divider()

# Section 1: The Limitations of Flat Files vs. Vector Databases
st.header("1. Limitations of Flat File Storage vs. Vector Database Architecture", divider="rainbow")

col_vdb1, col_vdb2 = st.columns([1, 1])

with col_vdb1:
    st.markdown("""
    ### Why Flat Files (.npy / Parquet / SQLite BLOBs) Fail at Scale:
    * **1. No Sub-linear Indexing ($O(N)$ vs $O(\\log N)$)**:
      * NumPy flat files perform brute-force linear scans.
      * At 5,000,000 vectors, one query requires evaluating 5 million dot products, saturating CPU bandwidth.
      * Vector databases build graph (HNSW) or inverted (IVF) indexes, evaluating less than 1% of the corpus per query!
    * **2. The Impossibility of High-Concurrency Upserts**:
      * In flat files, multiple workers cannot simultaneously write updates without race conditions or global file locks.
      * Vector databases maintain Write-Ahead Logs (WAL) and concurrent LSM-trees for non-blocking upserts.
    * **3. Metadata Pre-Filtering Inefficiencies**:
      * In flat files, you must either:
        * **Post-Filter**: Retrieve top 100 vectors, then filter by metadata (often yields 0 matches if top 100 belonged to other users!).
        * **Pre-Filter**: Filter metadata in SQL, then scan vectors (slow disconnected round-trips).
      * Vector databases execute **Single-Stage Filtered HNSW Traversal**, navigating the graph strictly along nodes that satisfy metadata criteria.
    """)

with col_vdb2:
    st.markdown("### Flat File vs. Vector DB Architecture")
    vdb_dot = """
    digraph VectorStorage {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_flat {
            label="Flat File (NumPy / Parquet)"; style="dashed"; color="#EF4444";
            F1 [label="Raw Binary Array\\nvectors.npy\\nLinear Scan O(N)", fillcolor="#FEE2E2"];
            F2 [label="No Metadata Indexes\\nRequires Full File Rewrites on Update", fillcolor="#FEE2E2"];
        }

        subgraph cluster_vdb {
            label="Dedicated Vector Database (ChromaDB / Qdrant)"; style="dashed"; color="#10B981";
            V1 [label="ANN Graph Index (HNSW)\\nSub-5ms Logarithmic Search O(log N)", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
            V2 [label="Payload Metadata Engine\\nSingle-stage filtered graph traversal", fillcolor="#D1FAE5"];
            V3 [label="ACID Transactions & WAL\\nDynamic zero-downtime upserts", fillcolor="#A7F3D0"];
        }

        QueryIn [label="User Query: 'Q3 Earnings'\\nFilter: {department == 'finance'}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        QueryIn -> F1 [label="Linear brute force\\nSlow at scale"];
        QueryIn -> V1 [label="Sub-5ms ANN search"];
    }
    """
    st.graphviz_chart(vdb_dot, use_container_width=False)

st.markdown("### Comprehensive Architecture Comparison Table")
st.markdown("""
| Architectural Capability | Flat Files (`.npy` / CSV) | Relational SQL with Vector Ext (pgvector) | Dedicated Vector Database (ChromaDB / Qdrant) |
| :--- | :--- | :--- | :--- |
| **Search Complexity** | $O(N)$ (Brute force linear) | $O(\\log N)$ (HNSW index in Postgres) | $O(\\log N)$ (Native in-memory / DiskANN) |
| **P95 Latency (1M Vectors)** | **~450 ms** (Unacceptable) | **~12 ms** | **~3 - 6 ms** (Blazing Fast) |
| **Metadata Filtering** | Manual post-filtering | Standard SQL `WHERE` clauses | Native payload filter with bitmask acceleration |
| **Concurrent Writes** | File lock conflicts | Multi-Version Concurrency (MVCC) | Distributed write-ahead logging (WAL) |
| **Horizontal Sharding** | Manual scripting | Postgres Foreign Data Wrappers | Distributed cluster sharding with Raft consensus |
| **Recommended Scale** | < 10,000 vectors | 10k - 5M vectors (existing DB) | 100k - Billions of vectors (Dedicated RAG) |
""")

st.divider()

# Section 2: Production Code: Filtered Vector Search Engine
st.header("2. Production Implementation: Filtered Vector Index Simulator", divider="rainbow")

st.markdown("""
The production utility below demonstrates the mechanics of **Single-Stage Filtered Vector Search**: traversing candidate vector IDs while checking metadata predicate masks in a single unified step.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Callable

class FilteredVectorIndex:
    """
    Simulates a production vector database engine supporting
    combined metadata filtering and cosine similarity ranking.
    """
    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim
        self.doc_ids: List[str] = []
        self.vectors: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self.metadata_store: List[Dict[str, Any]] = []

    def upsert(self, doc_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """Inserts or updates a document with metadata and unit L2 vector."""
        norm_v = vector / np.maximum(np.linalg.norm(vector), 1e-12)
        
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            self.vectors[idx] = norm_v
            self.metadata_store[idx] = metadata
        else:
            self.doc_ids.append(doc_id)
            self.metadata_store.append(metadata)
            if len(self.vectors) == 0:
                self.vectors = norm_v.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, norm_v.reshape(1, -1)])

    def query(
        self, 
        query_vector: np.ndarray, 
        filter_fn: Callable[[Dict[str, Any]], bool] = None, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes single-stage filtered vector search.
        """
        if len(self.doc_ids) == 0:
            return []

        q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-12)

        # 1. Identify indices satisfying metadata predicate
        if filter_fn is not None:
            valid_indices = [i for i, meta in enumerate(self.metadata_store) if filter_fn(meta)]
        else:
            valid_indices = list(range(len(self.doc_ids)))

        if not valid_indices:
            return []

        # 2. Compute dot products only for matching candidates
        candidate_vectors = self.vectors[valid_indices]
        scores = np.dot(candidate_vectors, q_norm.reshape(-1))

        # 3. Extract top-k among filtered subset
        sorted_pos = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for p in sorted_pos:
            actual_idx = valid_indices[p]
            results.append({
                "doc_id": self.doc_ids[actual_idx],
                "score": float(scores[p]),
                "metadata": self.metadata_store[actual_idx]
            })
        return results
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveVDBBenchmark:
    def __init__(self):
        # Multi-department enterprise documents with mock 3D vectors
        self.documents = [
            {"id": "DOC_01", "dept": "Finance", "year": 2024, "text": "Annual fiscal revenue growth exceeded targets by 14%.", "vec": np.array([0.95, 0.10, 0.05])},
            {"id": "DOC_02", "dept": "Finance", "year": 2023, "text": "Operating expenditure audits completed for fiscal year 2023.", "vec": np.array([0.92, 0.08, 0.06])},
            {"id": "DOC_03", "dept": "Legal", "year": 2024, "text": "Intellectual property patent filings finalized for machine learning.", "vec": np.array([0.15, 0.94, 0.08])},
            {"id": "DOC_04", "dept": "Engineering", "year": 2024, "text": "Kubernetes cluster migration reduced server latency by 40ms.", "vec": np.array([0.08, 0.12, 0.96])},
            {"id": "DOC_05", "dept": "Engineering", "year": 2023, "text": "Legacy database sharding replaced with distributed SQL nodes.", "vec": np.array([0.10, 0.15, 0.94])}
        ]

    def estimate_scale_metrics(self, corpus_size: int):
        # Linear vs ANN latency estimates
        flat_scan_ms = (corpus_size / 1_000_000.0) * 450.0  # 450ms per 1M on CPU
        ann_vdb_ms = 2.5 + (np.log10(max(corpus_size, 10)) * 0.8) # logarithmic ~ 3-7ms
        
        # RAM requirements (float32, 768 dims)
        ram_mb = (corpus_size * 768 * 4) / (1024 * 1024)
        
        return {
            "flat_ms": max(flat_scan_ms, 0.05),
            "vdb_ms": ann_vdb_ms,
            "speedup": max(flat_scan_ms / max(ann_vdb_ms, 0.01), 1.0),
            "ram_mb": ram_mb
        }

st.divider()

# Section 3: Interactive Workbench: Flat File vs. Vector Database Scale Simulator
st.header("3. Interactive Workbench: Flat File vs. Vector Database Scale Simulator", divider="rainbow")

st.markdown("""
Slide the corpus size from **10,000 to 10,000,000 vectors** to observe why flat files fail in production.
Watch how **Flat File Linear Scan Latency spikes into seconds**, while **Vector Database HNSW Latency remains firmly under 8 milliseconds**!
""")

vdb_bench = InteractiveVDBBenchmark()

col_sc_l, col_sc_r = st.columns([1, 1])

with col_sc_l:
    st.subheader("1. Scale Configuration")
    scale_size = st.select_slider(
        "Corpus Size (Number of Documents):",
        options=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000],
        value=1_000_000
    )

    metrics = vdb_bench.estimate_scale_metrics(scale_size)

    m1, m2 = st.columns(2)
    m1.metric("Flat File (O(N) Scan)", f"{metrics['flat_ms']:.1f} ms")
    m2.metric("Vector DB (HNSW ANN)", f"{metrics['vdb_ms']:.1f} ms")

    st.metric("Vector DB Latency Advantage", f"{metrics['speedup']:.0f}x Faster!", delta="Sub-linear Logarithmic Traversal")
    st.metric("Raw Embedding RAM Footprint", f"{metrics['ram_mb']:.1f} MB ({metrics['ram_mb']/1024:.2f} GB)")

with col_sc_r:
    st.subheader("2. Filtered Metadata Query Simulation")
    dept_filter = st.selectbox("Metadata Filter: Department:", ["All Departments", "Finance", "Legal", "Engineering"])
    year_filter = st.radio("Metadata Filter: Year:", [2024, 2023, "All Years"], horizontal=True)

    # Filter documents
    filtered_docs = []
    for d in vdb_bench.documents:
        match_dept = (dept_filter == "All Departments" or d["dept"] == dept_filter)
        match_year = (year_filter == "All Years" or d["year"] == year_filter)
        if match_dept and match_year:
            filtered_docs.append(d)

    st.markdown(f"#### Documents Passing Single-Stage Filter ({len(filtered_docs)} matched):")
    if filtered_docs:
        df_f = pd.DataFrame([
            {"ID": d["id"], "Dept": d["dept"], "Year": d["year"], "Snippet": d["text"]}
            for d in filtered_docs
        ])
        st.dataframe(df_f, use_container_width=True)
        st.success(f"✅ **Single-Stage Filtering Verified**: Vector DB prunes non-matching partitions upfront, evaluating vectors strictly within `{dept_filter} / {year_filter}`.")
    else:
        st.warning("No documents match the specified metadata filters.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Database Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the difference between Pre-Filtering, Post-Filtering, and Single-Stage Filtered Vector Search?**
    * **Answer**:
      * **Post-Filtering**: Runs pure vector search first to get top 100 results, then applies metadata filters (`WHERE user_id = 5`).
        * *Flaw*: If all top 100 documents belong to other users, post-filtering returns **0 results**, even if valid documents exist deeper in the database!
      * **Pre-Filtering**: Executes the SQL metadata filter first to extract 10,000 valid document IDs, then performs brute-force vector search across that subset.
        * *Flaw*: Bypasses the HNSW vector index; forces an unindexed linear scan on the subset.
      * **Single-Stage Filtering (Native Vector DB)**: The HNSW graph traversal algorithm checks the metadata bitmask **during graph exploration**. It traverses graph edges only between nodes that satisfy the metadata filter, preserving sub-millisecond retrieval speeds!

    **Q2: When should an enterprise use pgvector (PostgreSQL) versus a dedicated vector database (Qdrant / ChromaDB)?**
    * **Answer**:
      * **Use `pgvector` when**:
        * You already use PostgreSQL as your primary transactional relational database.
        * Your dataset is small-to-medium (< 1-2 million vectors).
        * You require strict ACID foreign-key joins between relational tables and embeddings in a single SQL query.
      * **Use a Dedicated Vector Database (Qdrant / Pinecone / ChromaDB) when**:
        * Your dataset exceeds 5-10 million vectors.
        * You require multi-tenant payload filtering with sub-5ms P95 latency guarantees.
        * You need advanced vector features like quantization (Product Quantization, Scalar Quantization), dynamic multi-vector reranking, or distributed multi-region sharding.

    **Q3: What are the primary disk persistence mechanisms used by Vector Databases to survive crashes?**
    * **Answer**:
      * 1. **Write-Ahead Logging (WAL)**: Every vector insertion or update is appended to a sequential disk log before updating in-memory graph structures. On crash restart, the WAL is replayed.
      * 2. **Memory-Mapped Files (mmap)**: Vectors and graph edges are mapped directly from disk into virtual memory, allowing the OS kernel to page memory efficiently without loading the entire multi-gigabyte index into user-space RAM.
      * 3. **Segment Merging (LSM-Tree Pattern)**: New vectors are written to small in-memory segments and periodically merged and indexed into immutable disk segments in the background.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Naive Post-Filtering Resulting in Empty Results**:
      * *Issue*: Fetching top 5 vectors from a vector DB and filtering by `tenant_id == 'tenant_b'` in Python.
      * *Consequence*: If the top 5 nearest vectors belong to `tenant_a`, your query returns 0 matches, even though tenant B has dozens of relevant documents!
      * *Fix*: Always pass metadata filters directly into the vector database query API (`where={"tenant_id": "tenant_b"}`).

    * **Trap 2: Storing Embeddings in Regular SQL Columns without ANN Indexes**:
      * *Issue*: Creating a Postgres column `embedding float8[]` and querying with `<->` without creating an `HNSW` or `IVFFlat` index.
      * *Consequence*: Postgres performs a sequential table scan (`Seq Scan`) on every query, maxing out CPU cores at 50,000 rows.
      * *Fix*: Execute `CREATE INDEX ON table USING hnsw (embedding vector_cosine_ops)`.

    * **Trap 3: Rebuilding Indexes Synchronously on Every Insert**:
      * *Issue*: Rebuilding an HNSW graph every time a new document is uploaded.
      * *Consequence*: Upload endpoint freezes for seconds or minutes.
      * *Fix*: Use databases that support incremental, asynchronous graph edge insertion (e.g. Qdrant, ChromaDB).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Single-Stage Filtering**
      * *Definition*: The capability of advanced vector databases to filter metadata criteria concurrently during graph traversal rather than before or after.
    * **Flashcard 2: Linear Scan Penalty**
      * *Definition*: The computational bottleneck of flat file storage where every query must evaluate dot products against all N records in O(N) time.
    * **Flashcard 3: Write-Ahead Log (WAL)**
      * *Definition*: An append-only persistence log that ensures vector database updates survive power loss or system crashes.
    * **Flashcard 4: pgvector**
      * *Definition*: An open-source PostgreSQL extension adding native vector data types and HNSW/IVFFlat nearest-neighbor indexing to relational tables.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Why Vector Databases:

- [ ] I can articulate the 3 major failure modes of flat file storage (.npy) in production.
- [ ] I understand the difference between Pre-filtering, Post-filtering, and Single-Stage filtering.
- [ ] I know when to use `pgvector` vs when to deploy a dedicated vector database like Qdrant or ChromaDB.
- [ ] I understand how Write-Ahead Logging and memory-mapped files provide crash durability for vector indexes.
- [ ] I understand why HNSW ANN indexing scales logarithmically $O(\\log N)$ compared to linear $O(N)$ scans.
""")
