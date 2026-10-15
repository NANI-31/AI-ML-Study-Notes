import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class VectorStoreDocItem:
    """Represents a discrete document record stored in a vector collection."""
    def __init__(self, doc_id: str, content: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.vector = vector
        self.metadata = metadata

class MockVectorStoreCRUDStudio:
    """Simulates an enterprise vector store supporting full CRUD lifecycle operations."""
    def __init__(self):
        self.collection: Dict[str, VectorStoreDocItem] = {
            "doc-001": VectorStoreDocItem(
                "doc-001", "Production Aurora PostgreSQL failover latency is targeted under 30 seconds.",
                np.array([0.91, 0.12, 0.25]),
                {"department": "DevOps", "status": "active", "version": 1}
            ),
            "doc-002": VectorStoreDocItem(
                "doc-002", "Annual employee dental coverage allowance is capped at 2,000 USD.",
                np.array([0.15, 0.92, 0.18]),
                {"department": "HR", "status": "active", "version": 1}
            ),
            "doc-003": VectorStoreDocItem(
                "doc-003", "Zero Trust access gateway requires continuous mutual TLS identity verification.",
                np.array([0.35, 0.18, 0.92]),
                {"department": "Security", "status": "active", "version": 1}
            )
        }
        self.audit_log: List[str] = [
            "INITIALIZED: Collection 'enterprise_kb' created with 3 base records."
        ]
        self.persisted_to_disk: bool = True

    def _mock_embed(self, text: str) -> np.ndarray:
        t = text.lower()
        v = np.array([0.33, 0.33, 0.33])
        if any(w in t for w in ["database", "postgres", "sql", "cluster"]):
            v = np.array([0.90, 0.15, 0.20])
        elif any(w in t for w in ["hr", "dental", "benefits", "leave"]):
            v = np.array([0.12, 0.93, 0.16])
        elif any(w in t for w in ["security", "zero trust", "tls", "auth"]):
            v = np.array([0.30, 0.18, 0.94])
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def add_or_update(self, doc_id: str, content: str, dept: str) -> str:
        vec = self._mock_embed(content)
        is_update = doc_id in self.collection
        ver = self.collection[doc_id].metadata.get("version", 1) + 1 if is_update else 1

        self.collection[doc_id] = VectorStoreDocItem(
            doc_id, content, vec,
            {"department": dept, "status": "active", "version": ver}
        )
        action = f"UPSERT (Updated)" if is_update else "INSERT (Added)"
        msg = f"{action}: Document '{doc_id}' (v{ver}) under department '{dept}'."
        self.audit_log.append(msg)
        self.persisted_to_disk = False
        return msg

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.collection:
            del self.collection[doc_id]
            self.audit_log.append(f"DELETE: Document '{doc_id}' purged permanently (GDPR compliance).")
            self.persisted_to_disk = False
            return True
        return False

    def persist(self) -> str:
        self.persisted_to_disk = True
        msg = f"PERSIST: Flushed {len(self.collection)} records to durable storage volume."
        self.audit_log.append(msg)
        return msg

    def query(self, query_text: str, dept_filter: str, top_k: int = 2) -> List[Dict[str, Any]]:
        q_vec = self._mock_embed(query_text)
        candidates = []
        for item in self.collection.values():
            if dept_filter != "All" and item.metadata.get("department") != dept_filter:
                continue
            sim = float(np.dot(q_vec, item.vector))
            candidates.append({
                "doc_id": item.doc_id,
                "content": item.content,
                "department": item.metadata.get("department"),
                "version": item.metadata.get("version"),
                "score": round(sim, 4)
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="14.3 Vector Store Operations",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">14.3 Vector Store Operations</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Complete CRUD lifecycle operations: Creating stores, adding documents, inserting pre-computed embeddings, metadata queries, GDPR deletions, upserts, and persistence.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Full Lifecycle Vector Store CRUD & Persistence Architecture", divider="rainbow")

st.markdown("""
A production vector database is not a static read-only index; it is a dynamic operational datastore.
Enterprise applications must continuously execute the complete **CRUD (Create, Read, Update, Delete)** lifecycle: 
ingesting new knowledge, updating outdated policies without index re-builds, deleting customer data for GDPR compliance, 
and flushing in-memory graphs to durable disk storage.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VectorStoreCRUD {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_create_add {
            label="1. Ingestion & Upsert (Create / Update)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            AddDocs [label="add_documents(docs, ids=[...])\\n• Batch tokenization & embedding\\n• Deterministic UUID collision check", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            AddEmbeds [label="add_embeddings(texts, embeddings, ids)\\n• Direct pre-computed vector ingestion\\n• Bypasses external API model calls", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_query_filter {
            label="2. Retrieval & Querying (Read)";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            SimQuery [label="similarity_search_with_score(q, k, filter)\\n• Pre-filtering on metadata fields\\n• Normalized similarity confidence scores", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            MMRQuery [label="max_marginal_relevance_search(q, k, lambda)\\n• Diversity re-ranking; penalizes redundancy", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_delete_purge {
            label="3. Deletion & Compliance (Delete)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#EF4444";
            fontsize=12;

            DeleteIds [label="delete(ids=['doc-001'])\\n• Permanent hard-deletion from HNSW graph\\n• GDPR Article 17 'Right to Be Forgotten'", fillcolor="#7F1D1D", fontcolor="#FFFFFF", color="#F87171"];
        }

        subgraph cluster_persistence {
            label="4. Durability & Serialization";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            PersistDisk [label="persist() / Snapshot Export\\n• Syncs WAL logs & RAM buffers to disk volume\\n• Immutable snapshot backup for disaster recovery", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399", penwidth=2];
        }

        AddDocs -> SimQuery [label="Vectorized & indexed"];
        AddEmbeds -> SimQuery [label="Direct insertion"];
        SimQuery -> MMRQuery [label="Optional diversity pass"];
        DeleteIds -> PersistDisk [label="Audit logged"];
        AddDocs -> PersistDisk [label="Flush updates"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: CREATING VECTOR STORES ---
st.header("2. Creating Vector Stores", divider="rainbow")

st.markdown("""
In LangChain, vector stores can be instantiated either by populating them with an initial set of documents 
or by connecting to an existing collection:
""")

st.code("""from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Method A: Factory constructor from existing documents
vectorstore = Chroma.from_documents(
    documents=initial_documents,
    embedding=embeddings,
    collection_name="enterprise_kb",
    persist_directory="./chroma_data"
)

# Method B: Initialize empty store connecting to an existing persistent collection
client_store = Chroma(
    collection_name="enterprise_kb",
    embedding_function=embeddings,
    persist_directory="./chroma_data"
)""", language="python")

st.divider()

# --- SECTION 3: ADDING DOCUMENTS ---
st.header("3. Adding Documents", divider="rainbow")

st.markdown("""
To add new documents incrementally to an existing vector store, use `add_documents()`:
- Automatically embeds each document's `page_content`.
- Extracts and indexes metadata key-value pairs.
- Returns the list of document IDs assigned in the store.
""")

st.code("""from langchain_core.documents import Document

new_docs = [
    Document(
        page_content="The 2026 paternity leave duration is increased to 16 weeks.",
        metadata={"department": "HR", "category": "benefits"}
    )
]

# Incremental batch addition
assigned_ids = vectorstore.add_documents(new_docs)
print(f"Successfully indexed document with ID: {assigned_ids[0]}")""", language="python")

st.divider()

# --- SECTION 4: ADDING EMBEDDINGS ---
st.header("4. Adding Embeddings", divider="rainbow")

st.markdown("""
In high-throughput ingestion pipelines, generating embeddings inside the vector store client can create bottlenecks.
`add_embeddings()` accepts **pre-computed vectors**, bypassing real-time model invocation during indexing:
""")

st.code("""# Ingesting pre-computed vectors directly
precomputed_vectors = [[0.042, -0.198, 0.812], [0.045, -0.185, 0.795]]
texts = ["Document snippet A", "Document snippet B"]
metadatas = [{"dept": "Finance"}, {"dept": "Legal"}]
ids = ["doc-101", "doc-102"]

vectorstore.add_embeddings(
    texts=texts,
    embeddings=precomputed_vectors,
    metadatas=metadatas,
    ids=ids
)""", language="python")

st.divider()

# --- SECTION 5: QUERYING VECTORS ---
st.header("5. Querying Vectors", divider="rainbow")

st.markdown("""
LangChain provides three primary search query methods:
1. **`similarity_search(query, k)`**: Returns the top k most similar `Document` objects.
2. **`similarity_search_with_score(query, k)`**: Returns tuples of `(Document, score)`, enabling threshold validation.
3. **`max_marginal_relevance_search(query, k, fetch_k, lambda_mult)`**: Enhances diversity by balancing query similarity with candidate distinctiveness.
""")

st.divider()

# --- SECTION 6: METADATA FILTERING ---
st.header("6. Metadata Filtering", divider="rainbow")

st.markdown("""
Metadata filters restrict the search space before distance calculations.
Most modern vector stores support standard boolean comparison operators:
- Equality: `{"department": "HR"}`
- Comparison operators: `{"year": {">=": 2025}}`
- Logical composition: `{"and": [{"department": "Finance"}, {"status": "approved"}]}`
""")

st.divider()

# --- SECTION 7: DELETING VECTORS ---
st.header("7. Deleting Vectors", divider="rainbow")

st.markdown("""
Under data protection regulations such as GDPR (Article 17 "Right to Be Forgotten") and CCPA, 
enterprises must have the ability to permanently purge specific records and their vector representations.
""")

st.code("""# Hard deletion by specific document IDs
vectorstore.delete(ids=["doc-101", "doc-102"])
print("Successfully purged records from vector index.")""", language="python")

st.divider()

# --- SECTION 8: UPDATING VECTORS ---
st.header("8. Updating Vectors", divider="rainbow")

st.markdown("""
When a document is updated, its text, metadata, and embedding must be synchronized in the database.
Most vector stores use **Upsert semantics**: providing an existing document ID replaces the old vector and payload.

### LangChain SQLRecordManager:
LangChain's Indexing API manages synchronization automatically:
""")

st.code("""from langchain.indexes import SQLRecordManager, index

record_manager = SQLRecordManager(
    namespace="chroma/enterprise_kb",
    db_url="sqlite:///record_manager.sql"
)
record_manager.create_schema()

# Automatically handles additions, updates, and deletion of orphaned chunks!
indexing_stats = index(
    docs_source=updated_documents,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",  # Options: 'full', 'incremental', 'none'
    source_id_key="source"
)
print(f"Indexing outcome: {indexing_stats}")
# Output: {'num_added': 4, 'num_updated': 2, 'num_deleted': 1, 'num_skipped': 40}""", language="python")

st.divider()

# --- SECTION 9: PERSISTENCE ---
st.header("9. Persistence", divider="rainbow")

st.markdown("""
Embedded vector stores (such as Chroma or FAISS) accumulate vector index modifications in memory. 
To guarantee data durability across process restarts, modifications must be explicitly persisted to disk:
""")

st.code("""# Persisting embedded store to local directory
# In Chroma v0.4+: Persistence is automatic when persist_directory is configured!
# In legacy Chroma: vectorstore.persist()
# In FAISS: vectorstore.save_local("faiss_backup")""", language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: End-to-End Enterprise Vector Store CRUD Operations Suite", divider="rainbow")

st.markdown("""
The following complete Python program implements an end-to-end vector store CRUD manager 
handling document ingestion, pre-filtering, upsert updates, GDPR deletions, and persistence.
""")

st.code("""import numpy as np
from typing import Dict, List, Any, Optional

class EnterpriseVectorStoreManager:
    \"\"\"Production harness managing full vector store CRUD lifecycle.\"\"\"
    def __init__(self, collection_name: str = "corporate_kb"):
        self.collection_name = collection_name
        self.store: Dict[str, Dict[str, Any]] = {}
        self.is_dirty: bool = False

    def upsert_document(self, doc_id: str, text: str, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        norm = np.linalg.norm(vector)
        norm_vec = vector / norm if norm > 0 else vector

        is_existing = doc_id in self.store
        self.store[doc_id] = {
            "content": text,
            "vector": norm_vec,
            "metadata": metadata
        }
        self.is_dirty = True
        return "UPDATED" if is_existing else "INSERTED"

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.store:
            del self.store[doc_id]
            self.is_dirty = True
            return True
        return False

    def query_with_filter(self, query_vec: np.ndarray, k: int = 3, filter_dict: Optional[dict] = None) -> List[Dict[str, Any]]:
        norm = np.linalg.norm(query_vec)
        q_norm = query_vec / norm if norm > 0 else query_vec

        candidates = []
        for doc_id, record in self.store.items():
            if filter_dict:
                match = True
                for fk, fv in filter_dict.items():
                    if record["metadata"].get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            score = float(np.dot(q_norm, record["vector"]))
            candidates.append({
                "id": doc_id,
                "content": record["content"],
                "metadata": record["metadata"],
                "score": round(score, 4)
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def persist_to_disk(self) -> str:
        self.is_dirty = False
        return f"Successfully flushed {len(self.store)} records to durable storage." """, language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Interactive Vector Store CRUD & Lifecycle Studio", divider="rainbow")

st.markdown("""
Perform live CRUD operations on an active vector store collection. Insert new knowledge, update existing records, 
trigger GDPR deletions, query with metadata filters, and commit modifications to persistent storage.
""")

if "crud_studio" not in st.session_state:
    st.session_state.crud_studio = MockVectorStoreCRUDStudio()

studio = st.session_state.crud_studio

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Collection CRUD Operations")

    op_mode = st.radio("Select Action", options=["Search Collection", "Add / Update Document", "Delete Document (GDPR)", "Persist Collection"])

    if op_mode == "Search Collection":
        s_query = st.text_input("Query Text", value="What is the database failover policy?")
        s_dept = st.selectbox("Department Filter", options=["All", "DevOps", "HR", "Security"])
        if st.button("🔍 Search", type="primary"):
            matches = studio.query(s_query, s_dept)
            st.session_state.last_matches = matches

    elif op_mode == "Add / Update Document":
        u_id = st.text_input("Document ID (Existing ID triggers Upsert)", value="doc-004")
        u_text = st.text_area("Document Content", value="Employees receive a 500 USD annual wellness stipend.")
        u_dept = st.selectbox("Department Category", options=["HR", "DevOps", "Security"])
        if st.button("➕ Execute Upsert", type="primary"):
            msg = studio.add_or_update(u_id, u_text, u_dept)
            st.success(msg)

    elif op_mode == "Delete Document (GDPR)":
        d_id = st.text_input("Document ID to Purge", value="doc-002")
        if st.button("🗑️ Execute Hard Deletion", type="primary"):
            if studio.delete(d_id):
                st.success(f"Purged document '{d_id}' from vector index.")
            else:
                st.error(f"Document ID '{d_id}' not found.")

    elif op_mode == "Persist Collection":
        if st.button("💾 Flush to Persistent Storage", type="primary"):
            msg = studio.persist()
            st.success(msg)

with col_wb2:
    st.subheader("📊 Live Collection State & Audit Trail")

    m1, m2 = st.columns(2)
    m1.metric("Total Records in Index", len(studio.collection))
    m2.metric("Durability Status", "✅ Persisted" if studio.persisted_to_disk else "⚠️ Unsaved Buffers")

    if op_mode == "Search Collection" and "last_matches" in st.session_state:
        st.markdown("#### Search Results:")
        for m in st.session_state.last_matches:
            with st.expander(f"📄 {m['doc_id']} (Score: {m['score']}) - Dept: {m['department']}", expanded=True):
                st.write(f"`{m['content']}`")
                st.caption(f"Version: v{m['version']}")

    st.markdown("#### 📜 Collection Audit Log:")
    for log in reversed(studio.audit_log[-5:]):
        st.write(f"- `{log}`")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What are the architectural challenges of updating or deleting vectors in an active HNSW index?",
     "HNSW builds a tightly interconnected geometric graph. When a vector is deleted or updated, removing a node leaves broken graph edges, creating isolated clusters and degrading search recall. Some engines mark nodes as soft-deleted (tombstoning) and require background index rebuilding/compaction to clean up dead nodes, which temporarily consumes extra memory and CPU."),
    ("How does LangChain's SQLRecordManager solve the orphan chunk problem during document re-indexing?",
     "When an ingested document is modified (e.g. edited from 5 pages to 3 pages), naive re-indexing inserts new chunks while leaving the old 5 chunks in the vector store. SQLRecordManager tracks a hash of every chunk and source key in a relational database. During re-indexing, it computes the diff between existing hashes and new hashes, issuing explicit delete commands for obsolete chunk IDs."),
    ("What is the difference between add_documents() and add_embeddings() in LangChain?",
     "add_documents() accepts Document objects with text, internally calls the vector store's embedding model to vectorize the text, and then inserts the vectors and metadata into the index. add_embeddings() accepts pre-computed vector arrays directly, bypassing the embedding model entirely. add_embeddings() is preferred in high-performance pipelines where embedding computation is handled by dedicated distributed GPU worker pools."),
    ("How do vector databases enforce multi-tenancy and data isolation across customer accounts?",
     "There are three primary patterns: 1) Hard Collection Isolation (each tenant gets an independent vector store collection or index), 2) Partition/Namespace Isolation (shared cluster with segregated namespaces), and 3) Metadata Pre-filtering (all vectors share a collection, but every query mandates a WHERE tenant_id = 'org-42' pre-filter). Hard collection or namespace isolation is required for SOC2 and HIPAA compliance."),
    ("What happens if an application terminates abruptly before calling persist() on an embedded vector store?",
     "In embedded vector stores that buffer writes in RAM (such as FAISS or older Chroma versions), modifications held in memory that were not flushed to disk are permanently lost upon process termination. Modern stores use Write-Ahead Logging (WAL) and automatic synchronization to guarantee durability without manual persist() calls.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Ingestion Without Providing Stable Document IDs. Allowing the vector store to assign random UUIDs on every ingestion run creates massive duplicate chunk bloat when re-indexing documents. Always provide deterministic content-hashed IDs.")
    st.error("**Trap 2 (Breaking Error)**: Tombstone Accumulation in High-Churn Vector Stores. Continuously updating vectors without index compaction leaves millions of tombstoned dead nodes in HNSW graphs, slowing search latency. Schedule periodic index rebuilding.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Generating Embeddings Inside Synchronous Request Threads. Invoking add_documents() synchronously inside web request handlers causes HTTP timeouts. Offload ingestion to background task workers.")
    st.warning("**Trap 4 (Quality Warning)**: Partial Failures in Batch Vector Ingestion. If a batch of 1,000 documents fails halfway due to an API timeout, half the documents are indexed and half are missing. Implement transactional batching or idempotent retries.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Vector Upsert"):
        st.markdown("**Inserts or updates a document using a deterministic ID, replacing obsolete vectors.**")
    with st.expander("🎴 Flashcard 2: SQLRecordManager"):
        st.markdown("**Tracks indexed chunk hashes in SQL, automatically deleting orphaned chunks upon re-indexing.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: add_embeddings()"):
        st.markdown("**Directly ingests pre-computed vector arrays, decoupling embedding compute from index insertion.**")
    with st.expander("🎴 Flashcard 4: HNSW Tombstoning"):
        st.markdown("**Soft-deleting nodes in graph indices requires periodic compaction to reclaim RAM and maintain recall.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the complete vector store CRUD lifecycle and how to create and manage collections.", value=True)
chk2 = st.checkbox("I know when to use add_embeddings() with pre-computed vectors instead of add_documents().", value=True)
chk3 = st.checkbox("I can implement similarity search, score thresholds, and Maximal Marginal Relevance queries.", value=True)
chk4 = st.checkbox("I understand how to execute GDPR Article 17 hard deletions across vector index graphs.", value=True)
chk5 = st.checkbox("I know how LangChain's SQLRecordManager eliminates duplicate and orphaned chunks.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 vector store operations interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 14.3 Vector Store Operations and completed all of Chapter 14: Vector Stores!")
