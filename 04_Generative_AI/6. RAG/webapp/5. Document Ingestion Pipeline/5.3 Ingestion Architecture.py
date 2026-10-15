import streamlit as st
import pandas as pd
import hashlib
import time
from typing import Dict, List, Any, Optional

st.set_page_config(
    page_title="5.3 Ingestion Architecture - RAG Master Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# Title & Overview
st.title("5.3 Ingestion Architecture: Batch, Incremental, Real-Time Ingestion, Versioning, & Re-Indexing")
st.caption("Chapter 5: Document Ingestion Pipeline | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
In prototype RAG applications, ingestion is often treated as a one-off script: a folder of PDFs is loaded, 
split into chunks, embedded via an API, and dumped into a local vector index. 
However, in enterprise production environments with millions of living documents, data sources are continuously mutating: 
contracts are amended, intranet wikis are updated daily, obsolete policies are deleted, and security classifications shift dynamically.

An enterprise **Ingestion Architecture** must handle continuous data drift with high throughput, zero downtime, 
and rock-solid consistency. It guarantees that queries retrieve strictly the latest approved versions of documents 
without breaking running queries, running up astronomical embedding API bills on unchanged files, or leaving 
orphaned "ghost vectors" in the database.

This lesson explores production-grade ingestion patterns: **Batch**, **Incremental**, **Scheduled**, and **Real-Time** 
pipelines, robust **Change Detection (CDC / SHA-256)**, **Document Versioning**, and zero-downtime **Re-Indexing Strategies**.
""")

st.divider()

# Section 1: Ingestion Paradigms
st.header("1. Ingestion Architectural Paradigms", divider="rainbow")
st.markdown("""
Depending on latency tolerance, update frequency, and data volume, enterprise RAG systems adopt one of four core ingestion architectures:
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Comparison of Ingestion Paradigms")
    paradigms_data = [
        {
            "Paradigm": "Batch Ingestion",
            "Trigger": "One-time initial load, nightly maintenance window",
            "Update Latency": "Hours to Days",
            "Throughput": "Massive (Millions of docs)",
            "Cost Efficiency": "High (optimized bulk API calls)",
            "Best For": "Initial corpus migration, quarterly regulatory archives"
        },
        {
            "Paradigm": "Incremental Ingestion",
            "Trigger": "Scheduled delta runs (every 15–60 min)",
            "Update Latency": "Minutes",
            "Throughput": "Medium (only modified deltas)",
            "Cost Efficiency": "Highest (skips unchanged files)",
            "Best For": "Confluence wikis, Notion workspaces, SharePoint drives"
        },
        {
            "Paradigm": "Scheduled Ingestion",
            "Trigger": "Cron jobs, Airflow DAGs, Celery beats",
            "Update Latency": "Predictable intervals (e.g., hourly at :00)",
            "Throughput": "Configurable batch sizes",
            "Cost Efficiency": "Very High",
            "Best For": "Daily financial statements, recurring web scrapers"
        },
        {
            "Paradigm": "Real-Time / Streaming",
            "Trigger": "Webhooks, S3 ObjectCreated events, Kafka / Debezium CDC",
            "Update Latency": "Sub-second to a few seconds",
            "Throughput": "High burst concurrency",
            "Cost Efficiency": "Lower (frequent micro-batches)",
            "Best For": "Customer support tickets, live news feeds, chat logs"
        }
    ]
    st.dataframe(pd.DataFrame(paradigms_data), use_container_width=True)

with col2:
    st.subheader("Event-Driven Real-Time Architecture")
    realtime_graph = """
    digraph RealtimeIngestion {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.12];
        edge [fontname="Helvetica", fontsize=8];

        Source [label="Data Source\\n(S3 / Notion / DB)", fillcolor="#EFF6FF", color="#3B82F6"];
        Event [label="Event Hook\\n(S3 Event / Webhook)", fillcolor="#F3E8FF", color="#9333EA"];
        Queue [label="Message Queue\\n(Kafka / AWS SQS)", fillcolor="#FEF3C7", color="#D97706"];
        Worker [label="Ingestion Worker\\n(Extract, Clean, Chunk)", fillcolor="#DCFCE7", color="#16A34A"];
        Embed [label="Batch Embedder\\n(Micro-batch buffer)", fillcolor="#FCE7F3", color="#DB2777"];
        VDB [label="Vector DB\\n(Upsert & Invalidate)", fillcolor="#CCFBF1", color="#0D9488"];

        Source -> Event -> Queue -> Worker -> Embed -> VDB;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(realtime_graph, use_container_width=False)

st.divider()

# Section 2: Change Detection & Document Versioning
st.header("2. Change Detection, Fingerprinting, & Document Versioning", divider="rainbow")
st.markdown("""
In an enterprise corpus with 500,000 documents, re-embedding every file daily would cost thousands of dollars in LLM API fees 
and take hours of unnecessary computation. An intelligent ingestion engine uses **Change Detection** to determine if a document is 
**NEW**, **MODIFIED**, **UNCHANGED**, or **DELETED**.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Change Detection Mechanisms")
    st.markdown("""
    1. **Content Fingerprinting (SHA-256 Hashing)**:
       * Compute a cryptographic SHA-256 digest over the raw extracted text content.
       * Even if a file's timestamp or filename changes (e.g. `policy_copy.pdf`), identical content hashes bypass the chunking and embedding pipeline completely.
    2. **Metadata Watermarking (ETag / Last-Modified)**:
       * Cloud object stores (S3, GCS) provide `ETag` and `LastModified` timestamps. 
       * Fast shallow check: if the source `mtime <= last_ingested_at`, skip reading the file payload entirely.
    3. **Change Data Capture (CDC)**:
       * Database sources (PostgreSQL, MongoDB) emit log-based change events via tools like Debezium or DynamoDB Streams.
       * Captures row-level INSERT, UPDATE, and DELETE actions immediately without expensive full-table polling.
    """)

with col2:
    st.subheader("Document Versioning & Ghost Vector Prevention")
    st.markdown("""
    * **The Ghost Vector Danger**: If `Document_v1` is chunked into 10 vectors, and an update changes the content so that it now yields only 7 vectors, simply adding the new vectors leaves **3 obsolete chunks** active in the index! When users query old details, the model retrieves outdated ghost vectors.
    * **Parent Document ID Tagging**: Every chunk must carry `doc_id` and `doc_version` in its vector metadata.
    * **Atomic Chunk Eviction**: Before upserting new chunks for `doc_id`, issue a batch delete for all existing vectors matching `doc_id == target_id`.
    * **Soft Deletion / Tombstoning**: Set `is_active: false` in vector metadata for instant query exclusion, followed by an asynchronous background garbage collection run.
    """)

st.divider()

# Section 3: Re-Indexing Strategies
st.header("3. Zero-Downtime Re-Indexing Strategies", divider="rainbow")
st.markdown("""
When your embedding model changes (e.g., migrating from `text-embedding-ada-002` to `text-embedding-3-large`) or your chunking strategy changes (e.g. from 1000 characters to semantic sentence splitting), 
**all existing vectors must be regenerated**. How do you rebuild the entire index without interrupting ongoing live user queries?
""")

reindex_df = pd.DataFrame([
    {
        "Strategy": "In-Place Upsert",
        "How It Works": "Updates records directly in the live collection while queries are running.",
        "Downtime": "Zero",
        "Risk": "High. During the hours of migration, queries hit a mixed hybrid state of old and new embeddings, degrading search quality.",
        "Recommended Use": "Minor incremental document additions only."
    },
    {
        "Strategy": "Blue/Green Vector Index Swap",
        "How It Works": "Build a completely new collection (Green) in parallel. Once fully populated, verified, and warmed up, point application alias/DNS from Blue to Green. Blue is destroyed.",
        "Downtime": "Zero (Instant atomic switch)",
        "Risk": "Low. Safe rollback possible if Green fails validation. Requires temporary double storage capacity.",
        "Recommended Use": "Model migrations, dimension changes, major pipeline overhauls."
    },
    {
        "Strategy": "Shadow Re-Indexing & Canary",
        "How It Works": "Shadow-route 5% of production query traffic to the newly built index alongside the primary index. Compare latency and retrieval recall metrics before 100% cutover.",
        "Downtime": "Zero",
        "Risk": "Lowest. Empirical verification under real user query workloads.",
        "Recommended Use": "Mission-critical enterprise RAG applications."
    }
])
st.dataframe(reindex_df, use_container_width=True)

st.subheader("Blue/Green Index Migration Workflow")
blue_green_chart = """
digraph BlueGreen {
    rankdir=TB;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.12];
    edge [fontname="Helvetica", fontsize=8];

    App [label="RAG Query Service\\n(Client API)", fillcolor="#EFF6FF", color="#3B82F6"];
    Alias [label="Collection Alias\\n('rag_knowledge_prod')", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
    Blue [label="Blue Index (Old)\\n(v1 Embeddings - Active)", fillcolor="#DCFCE7", color="#16A34A"];
    Green [label="Green Index (New)\\n(v2 Embeddings - Shadow)", fillcolor="#F3E8FF", color="#9333EA"];

    App -> Alias;
    Alias -> Blue [label="Active Queries", color="#16A34A", penwidth=2.0];
    Alias -> Green [label="Atomic Swap ->", style="dashed", color="#9333EA"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blue_green_chart, use_container_width=False)

st.divider()

# Section 4: Production Code Implementation
st.header("4. Production Python Architecture: Ingestion Manager", divider="rainbow")
st.markdown("""
Below is an enterprise-grade `IngestionManager` implementing SHA-256 content hashing, delta status detection, 
atomic chunk tombstoning, and blue/green collection management:
""")

st.code("""
import hashlib
from typing import Dict, List, Any, Optional

class IngestionManager:
    \"\"\"
    Production ingestion coordinator supporting incremental change detection,
    atomic document updates, and zero-downtime collection aliasing.
    \"\"\"
    def __init__(self, active_collection_name: str = "rag_docs_v1"):
        # Registry tracking: doc_id -> {"content_hash": str, "version": int, "chunks": List[str]}
        self.doc_registry: Dict[str, Dict[str, Any]] = {}
        # Simulated vector collection: collection_name -> {vector_id: {"doc_id": str, "content": str}}
        self.collections: Dict[str, Dict[str, Dict[str, Any]]] = {
            active_collection_name: {}
        }
        self.active_alias: str = active_collection_name

    def compute_hash(self, content: str) -> str:
        \"\"\"Generates SHA-256 fingerprint of the normalized text content.\"\"\"
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def detect_change(self, doc_id: str, content: str) -> str:
        \"\"\"Determines whether document is NEW, MODIFIED, or UNCHANGED.\"\"\"
        new_hash = self.compute_hash(content)
        if doc_id not in self.doc_registry:
            return "NEW"
        if self.doc_registry[doc_id]["content_hash"] != new_hash:
            return "MODIFIED"
        return "UNCHANGED"

    def ingest_document(self, doc_id: str, content: str) -> Dict[str, Any]:
        \"\"\"Ingests a document incrementally with ghost vector prevention.\"\"\"
        status = self.detect_change(doc_id, content)
        
        if status == "UNCHANGED":
            return {"doc_id": doc_id, "status": "UNCHANGED", "action": "Skipped embedding API call"}

        current_ver = self.doc_registry.get(doc_id, {}).get("version", 0) + 1
        content_hash = self.compute_hash(content)

        # 1. Purge any existing chunks for this doc_id from active collection (prevent ghost vectors)
        collection = self.collections[self.active_alias]
        existing_keys = [k for k, v in collection.items() if v["doc_id"] == doc_id]
        for k in existing_keys:
            del collection[k]

        # 2. Simulate text chunking (in real pipeline, call text splitter)
        chunks = [content[i:i+100] for i in range(0, len(content), 80)]
        
        # 3. Insert new chunks with version metadata
        new_chunk_ids = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_v{current_ver}_chunk_{idx}"
            collection[chunk_id] = {
                "doc_id": doc_id,
                "version": current_ver,
                "content": chunk_text
            }
            new_chunk_ids.append(chunk_id)

        # 4. Update local registry
        self.doc_registry[doc_id] = {
            "content_hash": content_hash,
            "version": current_ver,
            "chunk_count": len(chunks)
        }

        return {
            "doc_id": doc_id,
            "status": status,
            "version": current_ver,
            "purged_old_chunks": len(existing_keys),
            "new_chunks_indexed": len(new_chunk_ids)
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        \"\"\"Removes document and all associated chunks from active index.\"\"\"
        if doc_id not in self.doc_registry:
            return {"doc_id": doc_id, "status": "NOT_FOUND"}
            
        collection = self.collections[self.active_alias]
        keys_to_delete = [k for k, v in collection.items() if v["doc_id"] == doc_id]
        for k in keys_to_delete:
            del collection[k]
            
        del self.doc_registry[doc_id]
        return {"doc_id": doc_id, "status": "DELETED", "purged_chunks": len(keys_to_delete)}
""", language="python")

st.divider()

# Section 5: Interactive Ingestion & Versioning Workbench
st.header("5. Interactive Ingestion & Versioning Workbench", divider="rainbow")
st.markdown("""
Simulate enterprise change detection and incremental indexing in real time. 
Notice how unchanged files bypass processing, while edits increment the document version and purge obsolete chunks!
""")

if "manager" not in st.session_state:
    st.session_state.manager = {
        "registry": {
            "doc_refund_policy": {
                "hash": hashlib.sha256(b"Customers may request full refunds within 30 days of purchase.").hexdigest()[:12],
                "version": 1,
                "chunks": 2
            },
            "doc_api_guidelines": {
                "hash": hashlib.sha256(b"All API requests require Bearer token authorization in the HTTP headers.").hexdigest()[:12],
                "version": 1,
                "chunks": 2
            }
        },
        "log": []
    }

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Document Ingestion Trigger")
    selected_doc = st.selectbox(
        "Select Document ID:",
        ["doc_refund_policy", "doc_api_guidelines", "doc_new_sla_contract"]
    )
    
    default_text = "Customers may request full refunds within 30 days of purchase."
    if selected_doc == "doc_api_guidelines":
        default_text = "All API requests require Bearer token authorization in the HTTP headers."
    elif selected_doc == "doc_new_sla_contract":
        default_text = "Standard uptime guarantee is 99.9% excluding planned maintenance windows."

    input_text = st.text_area("Document Content Payload:", value=default_text, height=120)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Ingest / Sync Document", use_container_width=True):
            current_hash = hashlib.sha256(input_text.strip().encode("utf-8")).hexdigest()[:12]
            reg = st.session_state.manager["registry"]
            
            if selected_doc not in reg:
                # NEW
                chunks_count = max(1, len(input_text) // 40)
                reg[selected_doc] = {"hash": current_hash, "version": 1, "chunks": chunks_count}
                st.session_state.manager["log"].insert(0, f"✅ [NEW] {selected_doc} (v1) created with {chunks_count} chunks.")
            elif reg[selected_doc]["hash"] == current_hash:
                # UNCHANGED
                st.session_state.manager["log"].insert(0, f"⚡ [UNCHANGED] {selected_doc} identical hash ({current_hash}). Skipped embedding API call!")
            else:
                # MODIFIED
                old_ver = reg[selected_doc]["version"]
                new_ver = old_ver + 1
                chunks_count = max(1, len(input_text) // 40)
                reg[selected_doc] = {"hash": current_hash, "version": new_ver, "chunks": chunks_count}
                st.session_state.manager["log"].insert(0, f"🔄 [MODIFIED] {selected_doc} updated to v{new_ver}. Purged old chunks & re-indexed {chunks_count} chunks.")
            st.rerun()

    with col_btn2:
        if st.button("🗑️ Delete Document", use_container_width=True):
            reg = st.session_state.manager["registry"]
            if selected_doc in reg:
                del reg[selected_doc]
                st.session_state.manager["log"].insert(0, f"❌ [DELETED] {selected_doc} purged from vector index and registry.")
                st.rerun()
            else:
                st.warning("Document does not exist in registry.")

with col2:
    st.subheader("Live Document Registry & State")
    reg_items = []
    for d_id, data in st.session_state.manager["registry"].items():
        reg_items.append({
            "Doc ID": d_id,
            "Version": f"v{data['version']}",
            "SHA-256 Digest": data["hash"],
            "Active Chunks": data["chunks"]
        })
    st.dataframe(pd.DataFrame(reg_items), use_container_width=True)

    st.subheader("Ingestion Audit Log")
    if st.session_state.manager["log"]:
        for item in st.session_state.manager["log"][:6]:
            st.caption(item)
    else:
        st.info("No ingestion actions executed yet.")

st.divider()

# Section 6: Technical Interview Preparation
st.header("6. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Ingestion Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do you perform a zero-downtime embedding model migration in a production RAG system?**
    * **Answer**: You execute a **Blue/Green Index Swap**. You provision a second vector collection (Green) using the new embedding model and pipeline. You backfill the historical corpus asynchronously into Green. During backfill, any real-time CDC updates are double-written to both Blue and Green. Once Green is fully caught up and benchmarked, you switch the application pointer/alias from Blue to Green atomically (taking milliseconds). If any issues arise, rollback is instantaneous by repointing to Blue.

    **Q2: What is the 'Ghost Vector' problem, and how do you prevent it?**
    * **Answer**: The Ghost Vector problem occurs when an updated document yields fewer or different chunks than its predecessor, and the new vectors are inserted without deleting the old ones. The leftover vectors remain searchable and cause the LLM to hallucinate contradictory or superseded facts. It is prevented by: (1) attaching parent `doc_id` to every vector chunk, and (2) performing an atomic delete-by-filter (`doc_id == target_id`) prior to upserting new chunks.

    **Q3: Why is content hashing preferred over file modification timestamps (mtime) in large-scale ingestion?**
    * **Answer**: Modification timestamps are frequently modified by non-semantic actions, such as `touch`, git branch switching, backup restorations, or metadata edits. Relying on `mtime` triggers expensive re-embedding of millions of unchanged documents. Cryptographic SHA-256 content hashing guarantees that documents are re-indexed only when their actual textual content changes.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Synchronous Webhook Ingestion During Traffic Spikes**:
      * *Issue*: If your webhook handler extracts, embeds, and indexes documents synchronously on incoming HTTP requests, a burst of 500 uploads will exhaust server memory or hit embedding API rate limits (`429 Too Many Requests`), dropping requests.
      * *Fix*: Decouple ingestion via an asynchronous queue (RabbitMQ / AWS SQS / Celery). The webhook simply enqueues the file ID and returns HTTP 202 Accepted immediately.

    * **Trap 2: Ignoring Document Deletions in Incremental Crawls**:
      * *Issue*: Many incremental crawlers only look for `created_at > last_sync_time` or `updated_at > last_sync_time`. They never detect files that were deleted from the source storage!
      * *Fix*: Implement periodic reconciliation passes (comparing source ID manifests against index ID manifests) or capture source audit logs / trash bin webhooks.

    * **Trap 3: Full Table Re-embedding Instead of CDC**:
      * *Issue*: Running a daily SQL `SELECT * FROM articles` script re-embeds 1,000,000 rows when only 50 changed.
      * *Fix*: Use Change Data Capture (Debezium, WAL listeners) to stream only granular INSERT/UPDATE/DELETE events.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: CDC (Change Data Capture)**
      * *Definition*: A design pattern that detects and captures row- or document-level database modifications (INSERT, UPDATE, DELETE) at the transaction log level and streams them to downstream systems in real time.
    * **Flashcard 2: Blue/Green Index Swap**
      * *Definition*: An operational technique where a new vector index is populated in isolation and swapped with the live index via alias reassignment to eliminate downtime and query degradation during model migrations.
    * **Flashcard 3: Content Hashing (SHA-256)**
      * *Definition*: Generating a fixed-length hexadecimal digest of normalized document text to detect genuine semantic modifications while ignoring superficial file timestamp updates.
    * **Flashcard 4: Tombstoning**
      * *Definition*: Tagging obsolete vectors with a metadata flag (`is_active: false`) to instantly exclude them from retrieval queries prior to executing asynchronous physical disk deletions.
    """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Use this checklist to verify your mastery of Ingestion Architecture:

- [ ] I can articulate the key trade-offs between Batch, Incremental, Scheduled, and Real-Time ingestion.
- [ ] I understand how to prevent Ghost Vectors through parent document ID metadata and atomic purges.
- [ ] I know why SHA-256 content hashing is superior to file modification timestamps (`mtime`).
- [ ] I can design a zero-downtime Blue/Green Vector Index Swap workflow for embedding model migrations.
- [ ] I understand how Change Data Capture (CDC) enables low-latency synchronizations from relational and NoSQL databases.
- [ ] I know how to use asynchronous message queues (SQS / Kafka) to buffer embedding API rate limits during ingestion spikes.
""")
