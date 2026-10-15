import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import time

st.set_page_config(
    page_title="5.2 Records - Vector DB Curriculum",
    page_icon="📝",
    layout="wide"
)

# Title & Overview
st.title("5.2 Vector Records: CRUD Operations, Tombstones, Upserts & Metadata Indexing")
st.caption("Chapter 5: Vector Database Data Model | Domain: 4. Vector Database")
st.markdown("""
The **Vector Record** (often referred to as an *Entity*, *Point*, or *Document*) is the atomic unit of storage, 
indexing, and retrieval in a vector database.

While performing CRUD (Create, Read, Update, Delete) on a relational database row is straightforward, managing vector records 
involves deep architectural trade-offs:
* **Vectors are immutable by nature**: modifying a vector alters its geometric neighborhood, requiring HNSW graph re-wiring.
* **Deletions rely on Tombstones**: removing a vector from an active graph index mid-query causes graph disconnects, requiring background compaction.
* **Updates are Upserts**: changing text requires re-embedding, updating metadata indexes, and re-indexing the vector array.

In this lesson, we explore the complete engineering lifecycle of vector records: **Creation**, **Point Lookups**, 
**In-Place Metadata Updates**, **Tombstoning Deletions**, **Content-Hash IDs**, and **Optimized Retrieval Hydration**.
""")

st.divider()

# Section 1: The Vector Record CRUD Lifecycle
st.header("1. The Engineering Mechanics of Vector Record CRUD", divider="rainbow")

col_crud1, col_crud2 = col_crud1, col_crud2 = st.columns([1, 1])

with col_crud1:
    st.markdown("""
    ### 1. Create (Insert / Upsert)
    * **Atomic Insertion Bundle**:
      * Every record insertion bundles 4 components: `{id, vector, document, metadata}`.
    * **Batching Best Practice**:
      * Never insert records one by one in high-throughput pipelines. Always batch writes (e.g. 500 to 2,000 records per batch) to maximize Write-Ahead Log (WAL) sequential write throughput.
    * **Upsert Semantics**:
      * If a record with ID `X` does not exist -> Insert it.
      * If ID `X` already exists -> Atomically overwrite its vector, document, and metadata payload.
    
    ### 2. Read (Point Lookup & Hydration)
    * **Point Lookup by ID (`get_by_id`)**:
      * Direct hash-map or B-Tree lookup returning the record in sub-millisecond time (`O(1)`).
    * **Payload Projection (Save WAN Bandwidth!)**:
      * A 1536-dimensional vector consumes 6.14 KB. Fetching 10,000 vectors transfers 61 MB of raw floating-point numbers across the network.
      * Production clients toggle `with_vectors=False` when they only require document text and metadata!
    """)

with col_crud2:
    st.markdown("""
    ### 3. Update (Vector vs. Metadata Updates)
    * **Partial Metadata-Only Updates**:
      * Updating a metadata flag (`status = 'archived'`) does NOT require touching the vector or re-wiring the HNSW graph!
      * Modern engines (Qdrant, Milvus) update the metadata inverted index in-place without triggering expensive re-indexing.
    * **Vector / Text Updates**:
      * Updating document text requires re-computing the embedding via the AI model, invalidating the old vector, and inserting a new point.
    
    ### 4. Delete (Tombstoning & Compaction)
    * **Why Not Immediate Physical Deletion?**:
      * In an HNSW graph, deleting a node breaks graph edges, potentially isolating neighboring nodes into disconnected islands.
    * **The Tombstone Protocol (Soft Delete)**:
      * Deleting marks the record ID with a **tombstone bit** in a deletion bitset.
      * During search, tombstoned nodes can still act as routing bridges, but are strictly excluded from the Top-K results heap.
      * A background **Compaction Worker** periodically rebuilds clean immutable segments, physically purging tombstoned records.
    """)

st.markdown("### The Complete Record CRUD Lifecycle")
crud_dot = """
digraph RecordCRUDLifecycle {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Create [label="1. CREATE / UPSERT\\nBatch write to WAL\\nAppend to MemTable", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
    Active [label="2. ACTIVE STATE\\nSearchable in HNSW Graph\\nMetadata indexed in Bitsets", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
    UpdateMeta [label="3a. UPDATE METADATA\\nIn-place payload update\\nZero re-indexing cost", fillcolor="#FEF3C7"];
    UpdateVec [label="3b. UPDATE TEXT/VECTOR\\nTombstone old vector\\nInsert fresh embedding", fillcolor="#FDE68A"];
    Tombstone [label="4. DELETE (TOMBSTONE)\\nMarked in deletion bitset\\nExcluded from search heap", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
    Compacted [label="5. PURGE & COMPACT\\nBackground segment merge\\nDisk space reclaimed", fillcolor="#9CA3AF"];

    Create -> Active;
    Active -> UpdateMeta -> Active;
    Active -> UpdateVec -> Active;
    Active -> Tombstone -> Compacted;
}
"""
st.graphviz_chart(crud_dot, use_container_width=False)

st.divider()

# Section 2: ID Strategies: Random UUIDs vs. Content-Hash Deduplication
st.header("2. Record ID Strategies: Random UUIDs vs. Deterministic Content Hashes", divider="rainbow")

st.markdown("""
Assigning IDs is not arbitrary. Your ID strategy dictates whether your ingestion pipeline is **idempotent** 
and immune to duplicate data:
""")

col_id1, col_id2 = col_id1, col_id2 = st.columns([1, 1])

with col_id1:
    st.markdown("""
    ### 1. Random UUIDv4 (`uuid.uuid4()`)
    * **Characteristics**:
      * Completely randomized 128-bit identifier: `'f47ac10b-58cc-4372-a567-0e02b2c3d479'`.
    * **Advantages**:
      * Can be generated independently on distributed client workers with zero coordination or collision risk.
    * **The Trap (Duplicate Poisoning)**:
      * If a user ingests the exact same document twice, two distinct UUIDs are created!
      * Your database now stores redundant duplicate vectors, splitting retrieval scores and wasting RAM.
    """)

with col_id2:
    st.markdown("""
    ### 2. Deterministic Content Hash (SHA-256)
    * **Characteristics**:
      * Compute cryptographic hash of the document text and tenant ID:
        `record_id = hashlib.sha256(f"{tenant_id}:{chunk_text}".encode()).hexdigest()[:32]`.
    * **Advantages (Automatic Deduplication & Idempotency)**:
      * **Idempotent Ingestion**: Re-running an ingestion pipeline simply upserts the same ID, guaranteeing zero duplicates!
      * Saves embedding model API costs: you can check if `record_id` exists before calling the embedding API!
    """)

st.divider()

# Section 3: Interactive Record CRUD Operations Workbench
st.header("3. Interactive Record CRUD Operations Workbench", divider="rainbow")

st.markdown("""
Execute live CRUD operations against an interactive in-memory vector database. 
Observe **record insertions**, **partial metadata edits**, **tombstone soft deletes**, and **segment compaction**:
""")

# Initialize Session State Database
if "crud_db" not in st.session_state:
    st.session_state.crud_db = {
        "rec_01": {"id": "rec_01", "text": "Kubernetes pod crashing with OOMKilled error.", "dept": "DevOps", "status": "active", "tombstoned": False},
        "rec_02": {"id": "rec_02", "text": "PostgreSQL database connection pool exhausted.", "dept": "Database", "status": "active", "tombstoned": False},
        "rec_03": {"id": "rec_03", "text": "Redis eviction policy drops keys under pressure.", "dept": "Cache", "status": "active", "tombstoned": False}
    }

col_op_in, col_op_out = st.columns([1, 1])

with col_op_in:
    st.subheader("Record Operation Console")
    op_type = st.radio("Choose CRUD Action", ["CREATE (Insert/Upsert)", "READ (Point Lookup)", "UPDATE (Metadata Only)", "DELETE (Tombstone Soft Delete)", "COMPACT (Purge Tombstones)"])
    
    if op_type == "CREATE (Insert/Upsert)":
        new_id = st.text_input("Record ID", value="rec_04")
        new_text = st.text_input("Document Text", value="How to configure AWS IAM roles for service accounts.")
        new_dept = st.selectbox("Department", ["Cloud", "DevOps", "Security", "Billing"])
        if st.button("➕ Execute Insert / Upsert", type="primary"):
            st.session_state.crud_db[new_id] = {
                "id": new_id,
                "text": new_text,
                "dept": new_dept,
                "status": "active",
                "tombstoned": False
            }
            st.success(f"Record `{new_id}` successfully committed to WAL and MemTable!")
            
    elif op_type == "READ (Point Lookup)":
        read_id = st.selectbox("Select ID to Retrieve", list(st.session_state.crud_db.keys()))
        include_vec = st.checkbox("Hydrate Dense Vector Array (Simulated)", value=False)
        if st.button("🔍 Execute Point Lookup", type="primary"):
            rec = st.session_state.crud_db.get(read_id)
            if rec and not rec["tombstoned"]:
                st.info(f"**Retrieved Record `{read_id}`:**")
                st.json(rec)
            elif rec and rec["tombstoned"]:
                st.warning(f"Record `{read_id}` exists on disk but is TOMBSTONED (Invisible to search)!")
            else:
                st.error("Record Not Found!")
                
    elif op_type == "UPDATE (Metadata Only)":
        up_id = st.selectbox("Select ID to Update", list(st.session_state.crud_db.keys()))
        up_dept = st.selectbox("New Department", ["Security", "Executive", "Infrastructure", "Archived"])
        if st.button("✏️ Execute In-Place Metadata Update", type="primary"):
            if up_id in st.session_state.crud_db:
                st.session_state.crud_db[up_id]["dept"] = up_dept
                st.success(f"Record `{up_id}` metadata updated in-place! Vector index untouched.")
                
    elif op_type == "DELETE (Tombstone Soft Delete)":
        del_id = st.selectbox("Select ID to Soft Delete", [k for k, v in st.session_state.crud_db.items() if not v["tombstoned"]])
        if st.button("🗑️ Execute Tombstone Soft Delete", type="primary"):
            st.session_state.crud_db[del_id]["tombstoned"] = True
            st.warning(f"Record `{del_id}` marked as TOMBSTONED in deletion bitset. Excluded from search!")
            
    else: # COMPACT
        st.markdown("Rebuilds immutable segments, purges all tombstoned records, and defragments memory.")
        if st.button("🧹 Run Background Compaction Job", type="primary"):
            before_cnt = len(st.session_state.crud_db)
            st.session_state.crud_db = {k: v for k, v in st.session_state.crud_db.items() if not v["tombstoned"]}
            purged = before_cnt - len(st.session_state.crud_db)
            st.success(f"Compaction completed! Purged {purged} tombstoned record(s) from disk segments.")

with col_op_out:
    st.subheader("Current Database Storage State")
    
    current_data = []
    for k, v in st.session_state.crud_db.items():
        current_data.append({
            "Record ID": v["id"],
            "Document Snippet": v["text"][:35] + "...",
            "Department": v["dept"],
            "Storage State": "🔴 Tombstoned (Soft Deleted)" if v["tombstoned"] else "🟢 Active (Indexed)"
        })
    st.dataframe(pd.DataFrame(current_data), use_container_width=True, hide_index=True)
    
    active_count = sum(1 for v in st.session_state.crud_db.values() if not v["tombstoned"])
    tomb_count = sum(1 for v in st.session_state.crud_db.values() if v["tombstoned"])
    
    col_cnt1, col_cnt2 = st.columns(2)
    with col_cnt1:
        st.metric("Active Searchable Records", active_count)
    with col_cnt2:
        st.metric("Tombstoned Records Awaiting Compaction", tomb_count)

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Complete Vector Record CRUD Manager with Tombstones", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Vector Record CRUD Manager** featuring 
**SHA-256 Content Deduplication**, **In-Place Metadata Updates**, **Tombstone Soft Deletions**, 
and **Segment Compaction**:
""")

code_crud_mgr = '''import numpy as np
import hashlib
from typing import Dict, Any, List, Optional

class VectorRecordCRUDManager:
    """Manages the full lifecycle of vector records with tombstones and deduplication."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.records: Dict[str, Dict[str, Any]] = {}
        # Tombstone bitset tracking soft-deleted IDs
        self.tombstones: set = set()

    @staticmethod
    def generate_content_id(text: str) -> str:
        """Generates deterministic SHA-256 hash ID for idempotent deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def upsert(self, text: str, vector: np.ndarray, metadata: Dict[str, Any], custom_id: Optional[str] = None) -> str:
        """Idempotent insert/upsert. Automatically revokes tombstone if re-inserted."""
        record_id = custom_id or self.generate_content_id(text)
        assert len(vector) == self.dimension, f"Vector dimension must be {self.dimension}"

        # Unit normalize
        norm = np.linalg.norm(vector)
        unit_vec = vector / (norm if norm > 0 else 1.0)

        self.records[record_id] = {
            "id": record_id,
            "text": text,
            "vector": unit_vec,
            "metadata": metadata
        }
        # If record was previously deleted, revive it
        self.tombstones.discard(record_id)
        return record_id

    def update_metadata(self, record_id: str, new_metadata: Dict[str, Any]) -> bool:
        """In-place metadata update without re-embedding or index modification."""
        if record_id in self.records and record_id not in self.tombstones:
            self.records[record_id]["metadata"].update(new_metadata)
            return True
        return False

    def delete(self, record_id: str) -> bool:
        """Tombstone soft-delete: keeps graph topology intact while hiding from search."""
        if record_id in self.records and record_id not in self.tombstones:
            self.tombstones.add(record_id)
            return True
        return False

    def compact(self) -> int:
        """Background compaction worker: physically purges all tombstoned records."""
        purged_count = len(self.tombstones)
        for rid in self.tombstones:
            del self.records[rid]
        self.tombstones.clear()
        return purged_count

    def search(self, query_vec: np.ndarray, top_k: int = 2) -> List[Dict[str, Any]]:
        """ANN search filtering out tombstoned records."""
        q_norm = query_vec / np.linalg.norm(query_vec)
        active_ids = [rid for rid in self.records if rid not in self.tombstones]
        if not active_ids:
            return []

        matrix = np.array([self.records[rid]["vector"] for rid in active_ids])
        scores = np.dot(matrix, q_norm)
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        return [{
            "id": active_ids[i],
            "score": float(scores[i]),
            "text": self.records[active_ids[i]]["text"],
            "metadata": self.records[active_ids[i]]["metadata"]
        } for i in ranked_idx]

# Demonstration
mgr = VectorRecordCRUDManager(dimension=4)
np.random.seed(42)

# 1. Create Records with deterministic IDs
id1 = mgr.upsert("PostgreSQL performance tuning.", np.array([0.9, 0.1, 0.1, 0.0]), {"dept": "DB"})
id2 = mgr.upsert("Kubernetes Helm deployments.", np.array([0.1, 0.9, 0.1, 0.0]), {"dept": "DevOps"})
print(f"Created Record 1 ID (Deterministic Hash): {id1}")

# 2. Update Metadata In-Place
mgr.update_metadata(id1, {"verified": True, "reviewer": "Alice"})
print(f"Updated Metadata for {id1}: {mgr.records[id1]['metadata']}")

# 3. Soft Delete (Tombstone)
mgr.delete(id2)
print(f"Soft-deleted Record 2 (In Tombstones: {id2 in mgr.tombstones})")

# 4. Search (Notice id2 is invisible!)
hits = mgr.search(np.array([0.1, 0.9, 0.1, 0.0]), top_k=2)
print(f"Search Results (Found {len(hits)} active hits): {[h['id'] for h in hits]}")

# 5. Background Compaction
purged = mgr.compact()
print(f"Compaction finished: permanently purged {purged} record(s).")
'''

st.code(code_crud_mgr, language="python")

if st.button("🚀 Execute CRUD Manager Demo", type="primary"):
    exec_scope = {}
    exec(code_crud_mgr, globals(), exec_scope)
    
    m_inst = exec_scope["mgr"]
    h_res = exec_scope["hits"]
    p_cnt = exec_scope["purged"]
    
    st.success("CRUD Manager executed successfully!")
    st.markdown(f"**Search Hits Retained:** `{len(h_res)}` (Soft-deleted record was cleanly excluded!)")
    st.markdown(f"**Compacted Tombstones:** `{p_cnt}` record(s) physically reclaimed.")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why do Vector Databases utilize Tombstone soft deletes instead of immediate physical in-place graph deletion?"""
a1 = """* **Graph Topology Integrity**:
  * In an HNSW graph, nodes are connected by proximity edges. Deleting a node immediately requires re-wiring all its incoming and outgoing neighbors.
  * In concurrent environments, re-wiring graphs under active search queries causes severe lock contention, thread stalls, and risk of graph disconnects (partitioning).
* **The Tombstone Solution**:
  * Deleting a vector simply flips a bit in an in-memory **Tombstone Bitset**.
  * During search, the node can still be used as a routing bridge to jump across graph regions, but is filtered out before populating the Top-K results heap.
  * Background compaction processes periodically rebuild clean immutable segments in the background without affecting read queries."""

q2 = """2. What is the danger of assigning random UUIDs instead of content-hash IDs during document chunk ingestion?"""
a2 = """* **The Duplicate Pollution Trap**:
  * In enterprise RAG, document ingestion pipelines are re-run periodically (e.g., nightly crawls or re-indexing jobs).
  * If the pipeline assigns `uuid.uuid4()`, the exact same paragraph chunk ingested twice will receive two distinct IDs.
  * The vector database now contains duplicate vectors occupying identical coordinates.
  * When a user searches, the Top-5 results will be polluted by 2 or 3 identical duplicate chunks, crowding out genuine complementary context!
* **Solution**: Use deterministic SHA-256 content hashes (`hashlib.sha256(text).hexdigest()`). Re-running ingestion simply executes an idempotent upsert on the existing ID."""

q3 = """3. Why should client applications avoid retrieving raw dense vectors when fetching search results?"""
a3 = """* In typical RAG applications, downstream Large Language Models only need the **Document Text** and **Metadata Payloads** (e.g. source URL, title).
* A 1536-dimensional float32 vector consumes **6.14 KB** of bandwidth.
* If a search returns Top-20 candidates, transmitting the vector arrays wastes **122.8 KB** of network serialization overhead per query!
* At 1,000 QPS, this wastes over **120 MB/second of network bandwidth**.
* Always set `with_vectors=False` (or equivalent SDK projection flags) to request only text and metadata."""

q4 = """4. What is the difference between an In-Place Metadata Update and a Vector Update in terms of system write amplification?"""
a4 = """* **In-Place Metadata Update**:
  * Modifies only the scalar payload store (e.g. RocksDB) and updates the metadata inverted index bitsets.
  * Zero graph re-indexing; zero GPU embedding inference; negligible write amplification.
* **Vector / Text Update**:
  * Requires generating a fresh embedding from the AI model ($0.02 - $0.13 API cost or GPU inference).
  * Requires tombstoning the old vector in the HNSW index and inserting the new vector into the Write-Ahead Log and MemTable.
  * Triggers future background segment compaction."""

q5 = """5. How do Vector Databases handle Mass Deletions (e.g., `DELETE WHERE tenant_id = 42`)?"""
a5 = """* Rather than scanning and tombstoning millions of vectors one by one, production vector databases leverage their **Metadata Inverted Index**:
  1. Retrieve the compressed bitset matching `tenant_id = 42`.
  2. Perform a bitwise OR operation with the global deletion tombstone bitset in microseconds.
  3. All millions of matching vectors immediately become invisible to subsequent search queries in sub-millisecond time.
  4. The background compaction worker reclaims physical disk space during scheduled maintenance."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Vector Record Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Forgotten Embedding Update on Text Edits**  
> Developers often update the `document` text string in metadata (e.g. fixing a typo or updating a policy paragraph) 
> but forget to call the embedding model to generate a new vector. The vector remains stuck in its old semantic position, 
> causing searches for the updated text to fail completely! Always regenerate vectors when updating source text.

> [!WARNING]
> **Trap 2: The Ghost Tombstone Memory Leak**  
> If an application frequently updates or deletes records, but background compaction is disabled or misconfigured, 
> tombstoned records continue occupying RAM in HNSW graphs indefinitely! In high-churn databases, tombstoned "ghosts" 
> can account for 60% of total RAM usage. Always schedule automated segment compaction.

> [!WARNING]
> **Trap 3: Transmitting Vectors Over WAN to Web Frontends**  
> Fetching search results with `with_vectors=True` directly into a React or Next.js web application bloats JSON 
> payloads from 2 KB to 150 KB per search, significantly degrading mobile page load times. Never send raw vector arrays 
> to frontend clients.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Tombstone in Vector DBs?"):
        st.markdown("""
        **Tombstone**:  
        A soft-delete bitset marker that excludes deleted records from search results while preserving HNSW graph connectivity until compaction.
        """)
    with st.expander("🗂️ Flashcard: Why Use Content-Hash IDs?"):
        st.markdown("""
        **Content-Hash Deduplication**:  
        Generating IDs via SHA-256 of document text guarantees idempotent ingestion and prevents duplicate vector pollution.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Set `with_vectors=False`?"):
        st.markdown("""
        **Payload Projection**:  
        Omitting dense vector arrays during retrieval saves up to 90% of network bandwidth and serialization latency.
        """)
    with st.expander("🗂️ Flashcard: What is Segment Compaction?"):
        st.markdown("""
        **Compaction**:  
        A background worker that merges immutable segments, rebuilds clean HNSW graphs, and permanently purges tombstoned records.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the 4 components of an atomic vector record insertion bundle.", value=True)
chk2 = st.checkbox("I can explain why Tombstones are used instead of immediate in-place graph deletions.")
chk3 = st.checkbox("I know how deterministic content-hash IDs eliminate duplicate vector pollution.")
chk4 = st.checkbox("I understand why metadata-only updates have zero vector re-indexing overhead.")
chk5 = st.checkbox("I know why client applications should disable vector hydration (`with_vectors=False`).")
chk6 = st.checkbox("I can implement a complete Vector Record CRUD manager with tombstones in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Vector Records and are ready for Chapter 5.3: Collections!")
