import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple
import hashlib
import time

st.set_page_config(
    page_title="7.1 Adding Documents - Vector DB Curriculum",
    page_icon="📥",
    layout="wide"
)

# Title & Overview
st.title("7.1 Adding Documents: Batch Ingestion, Auto-Embedding, IDs & Metadata Binding")
st.caption("Chapter 7: Adding Data to a Vector Database | Domain: 4. Vector Database")
st.markdown("""
Ingesting documents into a vector database is the foundation of any retrieval pipeline. 
Whether building a company-wide RAG knowledge base from 100,000 PDF pages, indexing customer support tickets, 
or populating an e-commerce catalog, **how you structure, batch, and identify your documents** dictates 
ingestion throughput, data consistency, and retrieval quality.

In ChromaDB and modern vector databases, adding documents is not simply storing raw strings; it is an **atomic pipeline**:
* Unstructured text strings are mapped to **Unique IDs**.
* Key-value **Metadata Payloads** are attached for downstream filtering.
* The registered **Embedding Function** intercepts text, executes model inference, and produces dense vector arrays.
* Vectors and metadata are committed atomically across the **dual storage engine (SQLite + HNSWlib)**.

In this lesson, we explore single and multi-document ingestion, analyze batch size optimization, compare ID generation strategies, 
and learn how to prevent client memory crashes during large data loads.
""")

st.divider()

# Section 1: The Document Ingestion Pipeline
st.header("1. The Anatomy of Document Ingestion", divider="rainbow")

col_ing1, col_ing2 = col_ing1, col_ing2 = st.columns([1, 1])

with col_ing1:
    st.markdown("""
    ### 1. The Core Ingestion Call
    In ChromaDB, adding documents uses the unified `collection.add()` API:
    ```python
    collection.add(
        documents=["Document chunk 1...", "Document chunk 2..."],
        metadatas=[{"source": "wiki", "page": 1}, {"source": "wiki", "page": 2}],
        ids=["doc_001", "doc_002"]
    )
    ```
    * **Length Invariant**: The lengths of `documents`, `ids`, and `metadatas` must match **strictly 1-to-1**.
    * If `documents` has 10 items, `ids` must contain exactly 10 unique strings.
    
    ### 2. Auto-Embedding Interception
    * When `documents` is provided without passing `embeddings`, Chroma's collection wrapper automatically intercepts the text.
    * It passes the list of strings to the collection's bound `embedding_function` (e.g. `all-MiniLM-L6-v2` or OpenAI).
    * Model inference executes in batch, producing a matrix of shape `(N, d)`.
    * Vectors are indexed into HNSWlib; text and metadata are written to `chroma.sqlite3`.
    """)

with col_ing2:
    st.markdown("### The Ingestion Pipeline Flowchart")
    ing_dot = """
    digraph DocumentIngestionFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawDocs [label="Raw Text Documents List\\n['Text 1', 'Text 2', ...]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        IDsMeta [label="Document IDs & Metadata Payloads\\nids=['id1', 'id2'], metadatas=[{...}, {...}]", fillcolor="#DBEAFE"];
        
        Batcher [label="Batch Partitioner\\nChunks into optimal batches of 500", fillcolor="#BFDBFE"];

        subgraph cluster_dual_write {
            label="Atomic Dual-Write Pipeline"; style="dashed"; color="#059669";
            EmbedFunc [label="Embedding Function Interception\\nComputes dense vectors (N, d)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            HNSWWrite [label="HNSWlib Graph Builder\\nInserts vectors & updates links", fillcolor="#A7F3D0"];
            SQLiteWrite [label="SQLite Storage Engine\\nInserts documents, IDs & metadata", fillcolor="#6EE7B7", color="#047857", style="filled,bold"];
        }

        Success [label="Committed to Vector DB\\nSearchable in sub-5ms!", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        RawDocs -> Batcher;
        IDsMeta -> Batcher;
        Batcher -> EmbedFunc;
        Batcher -> SQLiteWrite;
        EmbedFunc -> HNSWWrite;
        HNSWWrite -> Success;
        SQLiteWrite -> Success;
    }
    """
    st.graphviz_chart(ing_dot, use_container_width=False)

st.divider()

# Section 2: Batch Ingestion Architecture & Sizing
st.header("2. Batch Ingestion Architecture & Sizing Optimization", divider="rainbow")

st.markdown("""
A common beginner mistake is attempting to add 50,000 documents in a single `collection.add()` call:
```python
# ANTI-PATTERN: DO NOT DO THIS!
collection.add(documents=all_50000_docs, ids=all_50000_ids)
```
This causes client-side memory exhaustion (OOM), HTTP payload size timeouts in client-server mode, 
and OpenAI API token-per-minute (TPM) rate limit crashes.
""")

col_batch1, col_batch2 = col_batch1, col_batch2 = st.columns([1, 1])

with col_batch1:
    st.markdown("""
    ### Sizing Trade-Offs Across Batch Sizes
    * **Tiny Batches (1 to 10 records)**:
      * **Drawback**: High transaction overhead. Each call incurs SQLite transaction commit and network round-trip overhead.
      * **Throughput**: Very slow (e.g. 50 documents/sec).
    * **Optimal Production Batches (200 to 1,000 records)**:
      * **Sweet Spot**: Maximizes SIMD batch vectorization on GPUs/CPUs.
      * Fits comfortably within HTTP request payloads (typically < 5 MB).
      * **Throughput**: Blazing (1,500 to 5,000 documents/sec).
    * **Oversized Batches (10,000+ records)**:
      * **Drawback**: High risk of HTTP payload timeouts (413 Payload Too Large) and client memory spikes.
    """)

with col_batch2:
    st.markdown("### The Production Batching Helper Pattern")
    st.code("""def batch_ingest(collection, docs, metas, ids, batch_size=500):
    \"\"\"Splits large ingestion lists into memory-safe batches.\"\"\"
    total = len(docs)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_docs = docs[i:end]
        batch_metas = metas[i:end]
        batch_ids = ids[i:end]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"Ingested batch {i//batch_size + 1}: {end}/{total}")""", language="python")

st.divider()

# Section 3: Interactive Batch Ingestion Simulator
st.header("3. Interactive Batch Ingestion Simulator", divider="rainbow")

st.markdown("""
Simulate ingesting a corpus of technical documentation into ChromaDB. 
Tune batch size and ID strategy to observe **batch execution cycles**, **memory payload size**, 
and **database record synchronization**:
""")

col_sim_ctrl, col_sim_view = st.columns([1, 1])

with col_sim_ctrl:
    st.subheader("Ingestion Parameters")
    corpus_size = st.select_slider("Total Documents to Ingest", [10, 50, 200, 500, 1000, 5000], value=200)
    batch_size = st.select_slider("Batch Size (Records per add() call)", [10, 25, 50, 100, 250, 500], value=50)
    id_strategy = st.radio("Document ID Strategy", ["Deterministic SHA-256 (Deduplicating)", "Sequential String ('doc_0001')", "UUIDv4 Random"])
    include_meta = st.checkbox("Attach Structured Metadata (`source`, `category`, `token_count`)", value=True)

with col_sim_view:
    st.subheader("Simulated Ingestion Performance & Pipeline Audit")
    
    num_batches = int(np.ceil(corpus_size / batch_size))
    est_payload_kb = (batch_size * 500 * 2) / 1024 # ~500 chars per doc + metadata
    est_total_time_s = (corpus_size * 0.002) + (num_batches * 0.02)
    throughput_dps = corpus_size / est_total_time_s if est_total_time_s > 0 else 0
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric("Total Batches Required", f"{num_batches} batches", f"Batch Size: {batch_size}")
        st.metric("Average Payload Size per Batch", f"{est_payload_kb:.1f} KB", "Safe (< 5MB limit)")
    with c_m2:
        st.metric("Estimated Ingestion Duration", f"{est_total_time_s:.2f} seconds", f"{throughput_dps:.0f} docs/sec")
        st.metric("ID Deduplication Safety", "100% Guaranteed" if "SHA-256" in id_strategy else "No (Duplicates possible)")
        
    st.markdown("#### Sample Ingested Record Preview (Batch 1, Item 1):")
    sample_text = "PostgreSQL WAL archiving configuration with pgBackRest for cloud backups."
    
    if "SHA-256" in id_strategy:
        sample_id = hashlib.sha256(sample_text.encode()).hexdigest()[:16]
    elif "Sequential" in id_strategy:
        sample_id = "doc_0001"
    else:
        sample_id = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        
    sample_record = {
        "id": sample_id,
        "document": sample_text,
        "metadata": {"source": "database_guide.md", "category": "Infra", "token_count": 14} if include_meta else {}
    }
    st.json(sample_record)

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Robust Production Batch Document Ingester", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Production Batch Document Ingester** featuring 
**safe batch chunking**, **deterministic SHA-256 ID deduplication**, **metadata binding**, and **progress logging**:
""")

code_ingester = '''import chromadb
from chromadb.config import Settings
import hashlib
from typing import List, Dict, Any

class ProductionBatchIngester:
    """Production-grade document ingester for ChromaDB."""
    def __init__(self, collection_name: str, batch_size: int = 2):
        self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.batch_size = batch_size

    @staticmethod
    def generate_deterministic_id(doc_text: str, source: str) -> str:
        """Generates idempotent SHA-256 ID to prevent duplicate pollution."""
        payload_key = f"{source}:{doc_text.strip()}"
        return hashlib.sha256(payload_key.encode("utf-8")).hexdigest()[:16]

    def ingest_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Batch ingests documents with automatic deduplication IDs."""
        assert len(documents) == len(metadatas), "Documents and metadatas length mismatch!"
        total_docs = len(documents)
        
        # Generate deterministic IDs
        ids = [
            self.generate_deterministic_id(doc, meta.get("source", "unknown"))
            for doc, meta in zip(documents, metadatas)
        ]

        print(f"Starting batch ingestion of {total_docs} documents (Batch Size: {self.batch_size})...")
        
        # Chunk into batches
        for i in range(0, total_docs, self.batch_size):
            end_idx = min(i + self.batch_size, total_docs)
            b_docs = documents[i:end_idx]
            b_metas = metadatas[i:end_idx]
            b_ids = ids[i:end_idx]

            # Ingest into Chroma
            self.collection.add(
                documents=b_docs,
                metadatas=b_metas,
                ids=b_ids
            )
            print(f"  -> Successfully committed Batch {i // self.batch_size + 1}: Items [{i}:{end_idx}]")

        print(f"Ingestion complete! Total Collection Record Count: {self.collection.count()}")

# Demonstration
sample_corpus = [
    "Kubernetes Horizontal Pod Autoscaler automatically scales pods based on CPU utilization.",
    "Terraform state locking prevents concurrent pipeline runs from corrupting infrastructure state.",
    "Prometheus alertmanager routes alerts to Slack and PagerDuty with inhibition rules.",
    "PostgreSQL vacuuming reclaims storage occupied by dead tuple versions."
]

sample_meta = [
    {"source": "k8s_guide.md", "dept": "DevOps"},
    {"source": "iac_guide.md", "dept": "DevOps"},
    {"source": "monitoring.md", "dept": "SRE"},
    {"source": "postgres.md", "dept": "Database"}
]

ingester = ProductionBatchIngester(collection_name="enterprise_wiki", batch_size=2)
ingester.ingest_documents(sample_corpus, sample_meta)

# Query to verify
hits = ingester.collection.query(query_texts=["How does autoscaling work?"], n_results=1)
print(f"\\nVerification Query Result: {hits['documents'][0][0]}")
'''

st.code(code_ingester, language="python")

if st.button("🚀 Execute Production Batch Ingester Demo", type="primary"):
    exec_scope = {}
    exec(code_ingester, globals(), exec_scope)
    
    col_cnt = exec_scope["ingester"].collection.count()
    verif_hit = exec_scope["hits"]["documents"][0][0]
    
    st.success("Batch Ingestion completed and query verified successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown(f"**Verified Collection Record Count:** `{col_cnt}`")
    with col_out2:
        st.markdown(f"**Retrieved Query Hit:** `{verif_hit}`")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the optimal batch size for ingesting documents into a Vector Database, and why does batching matter?"""
a1 = """* **Optimal Batch Size**: Typically **200 to 1,000 documents** per batch.
* **Why Batching Matters**:
  1. **SIMD Vectorization**: Deep learning embedding models (ONNX Runtime, PyTorch) achieve peak GPU/CPU throughput when processing batches of vectors simultaneously rather than single items.
  2. **Transaction Overhead**: In SQLite and WAL storage engines, every `collection.add()` incurs transaction commit and lock synchronization overhead. Batching amortizes this cost across 500 documents.
  3. **Network Serialization**: In Client-Server mode (`HttpClient`), batching eliminates thousands of individual HTTP round-trips."""

q2 = """2. What happens if you pass lists of different lengths to `documents`, `metadatas`, and `ids` in `collection.add()`?"""
a2 = """* ChromaDB enforces strict array length invariants.
* If `documents` contains 10 strings, but `ids` contains 9 strings, Chroma immediately throws a `ValueError`:
  `ValueError: Number of IDs (9) does not match number of documents (10)`.
* The transaction is rejected, and zero records from that batch are written to disk.
* **Production Best Practice**: Always validate lengths using `assert len(docs) == len(metas) == len(ids)` before calling `add()`."""

q3 = """3. Why is using deterministic SHA-256 IDs superior to random UUIDs when adding documents?"""
a3 = """* **Idempotency & Re-entrancy**:
  * In production pipelines, ingestion jobs run periodically (e.g. nightly cron sync).
  * If the pipeline uses random UUIDs (`uuid.uuid4()`), every re-run inserts duplicate vectors for identical documents.
  * Retrieval results become flooded with multiple copies of the exact same chunk!
* **Deterministic SHA-256 Solution**:
  * By computing `id = hashlib.sha256(f"{source}:{text}").hexdigest()`, re-running ingestion simply updates the existing record ID.
  * Guarantees 100% duplicate-free collections and allows skipping redundant embedding model API calls."""

q4 = """4. How does ChromaDB coordinate raw document text in SQLite with dense vectors in HNSWlib during `collection.add()`?"""
a4 = """* When `collection.add(documents=...)` is invoked:
  1. Chroma's embedding function vectorizes the text into a float32 array matrix.
  2. It generates an internal sequence of integer IDs (`uint64`) mapping to the user's string IDs.
  3. Vectors and integer IDs are passed to the C++ HNSWlib index to update graph adjacency edges.
  4. The user string IDs, raw document text strings, and JSON metadata payloads are written to the `embeddings` table in `chroma.sqlite3`.
  5. Both writes commit atomically, ensuring that every vector in the graph has matching hydrated text in SQLite."""

q5 = """5. How do you handle HTTP 413 Payload Too Large errors when using Chroma's `HttpClient` for batch ingestion?"""
a5 = """* In Client-Server mode, the Chroma server runs behind a Uvicorn / FastAPI HTTP server with a maximum request body limit (typically 10 MB).
* If a batch containing 2,000 large markdown documents exceeds 10 MB, the reverse proxy or server rejects the request with HTTP 413.
* **Mitigation**:
  * Dynamically calculate batch payload size in bytes: `sum(len(d.encode()) for d in batch_docs)`.
  * Flush the batch whenever either the record count reaches 500 **OR** the byte size reaches 2 MB, whichever comes first."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Document Ingestion Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Massive Unbounded Ingestion Memory Spike**  
> Reading 100,000 PDF documents into a single Python list and passing them into `collection.add()` causes Python's 
> process RAM to spike to tens of gigabytes, triggering the OS Out-of-Memory (OOM) killer. 
> Always stream documents using generators or chunked iterators in batches of 500.

> [!WARNING]
> **Trap 2: The Silent Empty Document Crash**  
> If an upstream PDF text extractor outputs an empty string `""` (e.g. for a scanned diagram page), 
> passing it to certain embedding functions will either crash the tokenizer or generate a zero-norm vector. 
> Always filter out empty or whitespace-only documents (`if doc.strip():`) before batching.

> [!WARNING]
> **Trap 3: Unhandled Duplicate IDs in Pure `add()` Calls**  
> If an ID already exists in the collection, calling `collection.add()` will throw an exception and abort 
> the entire batch! If you expect existing IDs, use **`collection.upsert()`** instead of `collection.add()`.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Optimal Batch Size?"):
        st.markdown("""
        **Optimal Batch Size**:  
        200 to 1,000 documents per batch, balancing SIMD vectorization speed against HTTP payload limits.
        """)
    with st.expander("🗂️ Flashcard: What is Auto-Embedding?"):
        st.markdown("""
        **Auto-Embedding**:  
        Chroma automatically vectorizes text passed to `documents` using the collection's registered embedding function.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Use SHA-256 IDs?"):
        st.markdown("""
        **SHA-256 Deduplication**:  
        Generates deterministic IDs based on text content, ensuring idempotent ingestion and preventing duplicate vectors.
        """)
    with st.expander("🗂️ Flashcard: What Invariant Governs `add()`?"):
        st.markdown("""
        **Length Invariant**:  
        `len(documents) == len(metadatas) == len(ids)` must hold true strictly, or Chroma throws a `ValueError`.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to add single and multiple documents using `collection.add()`.", value=True)
chk2 = st.checkbox("I understand why batching documents (200-1,000 per batch) is critical for performance.")
chk3 = st.checkbox("I know how Chroma automatically intercepts text and generates embeddings.")
chk4 = st.checkbox("I understand how deterministic SHA-256 IDs guarantee idempotent deduplication.")
chk5 = st.checkbox("I know how to avoid HTTP 413 Payload Too Large errors in Client-Server mode.")
chk6 = st.checkbox("I can implement a robust batch document ingester in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Adding Documents and are ready for Chapter 7.2: Adding Embeddings!")
