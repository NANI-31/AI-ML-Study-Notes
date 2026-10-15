import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
import hashlib

st.set_page_config(
    page_title="7.4 Data Management - Vector DB Curriculum",
    page_icon="🛠️",
    layout="wide"
)

# Title & Overview
st.title("7.4 Data Management: Inserts, Updates, Idempotent Upserts, Deletions & Deduplication")
st.caption("Chapter 7: Adding Data to a Vector Database | Domain: 4. Vector Database")
st.markdown("""
Managing the ongoing lifecycle of data in a production vector database requires far more than simply appending new documents. 
Enterprise knowledge bases constantly evolve: policies are updated, outdated articles are deprecated, bug fixes 
alter existing text, and periodic crawler jobs re-index previously ingested corpora.

Without robust **Data Management strategies**, collections suffer from **duplicate vector pollution**, **failed batch crashes**, 
and **stale retrieval hallucination**.

In this lesson, we master the four core data manipulation operations and deduplication patterns:
1. **Insert Operations (`collection.add`)** & Handling Duplicate ID Conflicts
2. **Update Operations (`collection.update`)** & Partial Field Modification
3. **Idempotent Upsert Operations (`collection.upsert`)** (The Production Gold Standard)
4. **Delete Operations (`collection.delete`)** by ID and by Metadata Filter
5. **Exact vs. Semantic Deduplication** (SHA-256 vs. Cosine Similarity Clustering)
""")

st.divider()

# Section 1: The Core Data Mutation Operations Compared
st.header("1. Core Data Mutation Operations: Add vs. Update vs. Upsert vs. Delete", divider="rainbow")

st.markdown("""
Understanding the exact semantics, failure modes, and invariants of each data operation is critical:
""")

col_op1, col_op2 = col_op1, col_op2 = st.columns([1, 1])

with col_op1:
    st.markdown("""
    ### 1. `collection.add()` (Strict Insert)
    * **Semantics**: Assumes all submitted IDs are **brand new**.
    * **Failure Mode**: If *any* submitted ID already exists in the collection, the entire batch operation **fails and raises an exception** (`UniqueConstraintError` or `ValueError`)!
    * **Use Case**: Initial bulk backfills of freshly partitioned data where duplicate IDs are guaranteed to be absent.
    
    ### 2. `collection.update()` (Strict Modification)
    * **Semantics**: Assumes all submitted IDs **already exist** in the collection.
    * **Failure Mode**: If an ID is missing, the operation will fail or skip the missing record.
    * **Partial Updates**: Allows updating metadata payloads without re-calculating embeddings.
    """)

with col_op2:
    st.markdown("""
    ### 3. `collection.upsert()` (Idempotent Standard)
    * **Semantics**: **Insert if missing, Update if present**.
    * **Failure Mode**: Immune to duplicate ID exceptions!
    * **Why It is the Production Gold Standard**:
      * Re-running an extraction pipeline, retrying a failed batch worker, or re-indexing an updated folder is completely idempotent and crash-proof.
    
    ### 4. `collection.delete()` (Point & Mass Deletion)
    * **Point Deletion**: `collection.delete(ids=["doc_1", "doc_2"])`.
    * **Filter Deletion (Mass Delete)**:
      * `collection.delete(where={"tenant_id": "org_42"})`.
      * Instantly purges all records matching the metadata condition in sub-millisecond time using inverted index bitsets!
    """)

st.markdown("### Operational Matrix Reference")
ops_matrix_df = pd.DataFrame({
    "Operation": ["collection.add()", "collection.update()", "collection.upsert()", "collection.delete()"],
    "If ID Exists": ["Throws Error (Aborts)", "Overwrites Record", "Overwrites Record", "Tombstones Record"],
    "If ID Missing": ["Inserts New Record", "Throws Error / Skips", "Inserts New Record", "No-Op (Silent Success)"],
    "Idempotent?": ["No (Fails on retry)", "No (Fails on missing)", "Yes (100% Safe to retry)", "Yes (Safe to retry)"],
    "Production Best Practice": [
        "Initial bulk load only",
        "Targeted in-place metadata changes",
        "Default for all recurring pipelines",
        "GDPR compliance & tenant offboarding"
    ]
})
st.dataframe(ops_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Complete Operations Decision Flowchart
st.header("2. Data Mutation Operations Flowchart", divider="rainbow")

st.markdown("""
Follow the decision logic below when designing data ingestion and synchronization workers:
""")

flow_ops_dot = """
digraph DataOpsFlow {
    rankdir=TB;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Start [label="Incoming Document Record\\n{id, text, metadata}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    Q_Action [label="What is the operational intent?", fillcolor="#DBEAFE"];
    
    Q_Remove [label="Is this record being removed?", fillcolor="#FEE2E2"];
    DoDelete [label="collection.delete(ids=[...])\\nor delete(where={...})", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];

    Q_Exists [label="Are you 100% certain\\nID does NOT exist?", fillcolor="#BFDBFE"];
    DoAdd [label="collection.add()\\nFast initial backfill", fillcolor="#93C5FD"];
    DoUpsert [label="collection.upsert()\\nIdempotent production standard!\\nHandles new & modified docs", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];

    Start -> Q_Action;
    Q_Action -> Q_Remove [label="Deletion"];
    Q_Action -> Q_Exists [label="Write / Ingest"];

    Q_Remove -> DoDelete [label="Yes"];
    Q_Exists -> DoAdd [label="Yes (Guaranteed clean)"];
    Q_Exists -> DoUpsert [label="No / Recurring pipeline"];
}
"""
st.graphviz_chart(flow_ops_dot, use_container_width=False)

st.divider()

# Section 3: Exact vs. Semantic Deduplication
st.header("3. Duplicate Handling: Exact Hash vs. Semantic Cosine Deduplication", divider="rainbow")

st.markdown("""
Duplicate documents degrade search quality by crowding out unique context chunks in LLM prompts. 
Enterprise architectures handle duplicates at two distinct layers:
""")

col_dedup1, col_dedup2 = col_dedup1, col_dedup2 = st.columns([1, 1])

with col_dedup1:
    st.markdown("""
    ### 1. Exact Lexical Deduplication (SHA-256)
    * **How it works**:
      * Compute cryptographic hash of the normalized text:
        `doc_id = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]`.
      * Ingest using `collection.upsert(ids=[doc_id], ...)`.
    * **Performance**:
      * Executes in microseconds on CPU.
      * Zero API costs; zero embedding model calls!
    * **Limitation**:
      * Misses near-duplicates (e.g. identical text with a minor typo or different capitalization).
    """)

with col_dedup2:
    st.markdown("""
    ### 2. Semantic Near-Duplicate Detection (Cosine Proximity)
    * **How it works**:
      * Before adding a new document chunk, query the collection using its embedding:
        `hits = collection.query(query_embeddings=[vec], n_results=1)`.
      * If `hits['distances'][0][0] < 0.05` (Cosine similarity > 0.95):
        * The document is flagged as a **near-duplicate** (paraphrase or minor edit)!
        * You can either skip ingestion or merge metadata payloads.
    * **Advantage**:
      * Catches paraphrased policies, duplicate ticket submissions, and cross-channel scraped duplicates.
    """)

st.divider()

# Section 4: Interactive Data Management Workbench
st.header("4. Interactive Data Management & Deduplication Workbench", divider="rainbow")

st.markdown("""
Execute live database operations and observe how `add()`, `update()`, and `upsert()` handle 
conflicts and deduplication across an active collection:
""")

# Initialize Session State
if "dm_workbench_db" not in st.session_state:
    st.session_state.dm_workbench_db = {
        "doc_k8s": {"text": "Kubernetes pod autoscaling guide.", "dept": "DevOps", "version": 1},
        "doc_pg": {"text": "PostgreSQL connection pooling with PgBouncer.", "dept": "Database", "version": 1}
    }

col_dm_in, col_dm_out = st.columns([1, 1])

with col_dm_in:
    st.subheader("Data Mutation Console")
    action_choice = st.selectbox("Select Operation", ["collection.add() (Strict Insert)", "collection.update() (Strict Update)", "collection.upsert() (Idempotent Upsert)", "collection.delete() (Point Deletion)"])
    
    target_id = st.text_input("Target Record ID", value="doc_k8s")
    target_text = st.text_input("Document Text", value="Kubernetes pod autoscaling guide (Updated v2).")
    target_dept = st.selectbox("Department", ["DevOps", "Database", "Security", "Archived"])
    
    if st.button("🚀 Execute Operation", type="primary"):
        exists = target_id in st.session_state.dm_workbench_db
        
        if "add()" in action_choice:
            if exists:
                st.error(f"❌ **UniqueConstraintError**: Record ID `{target_id}` already exists! Batch rejected.")
            else:
                st.session_state.dm_workbench_db[target_id] = {"text": target_text, "dept": target_dept, "version": 1}
                st.success(f"✅ Record `{target_id}` added successfully!")
                
        elif "update()" in action_choice:
            if not exists:
                st.error(f"❌ **RecordNotFoundError**: Record ID `{target_id}` does not exist in collection!")
            else:
                curr_ver = st.session_state.dm_workbench_db[target_id].get("version", 1)
                st.session_state.dm_workbench_db[target_id] = {"text": target_text, "dept": target_dept, "version": curr_ver + 1}
                st.success(f"✅ Record `{target_id}` updated in-place (New Version: {curr_ver + 1})!")
                
        elif "upsert()" in action_choice:
            curr_ver = st.session_state.dm_workbench_db.get(target_id, {}).get("version", 0)
            st.session_state.dm_workbench_db[target_id] = {"text": target_text, "dept": target_dept, "version": curr_ver + 1}
            st.success(f"✅ Record `{target_id}` upserted idempotently (Existed: {exists} | Version: {curr_ver + 1})!")
            
        else: # delete()
            if exists:
                del st.session_state.dm_workbench_db[target_id]
                st.warning(f"🗑️ Record `{target_id}` deleted successfully from collection!")
            else:
                st.info(f"Record `{target_id}` not found; delete operation was a safe no-op.")

with col_dm_out:
    st.subheader("Collection State Catalog")
    
    state_rows = []
    for rid, rdata in st.session_state.dm_workbench_db.items():
        state_rows.append({
            "Record ID": rid,
            "Document Content": rdata["text"],
            "Department": rdata["dept"],
            "Version": rdata["version"]
        })
        
    if state_rows:
        st.dataframe(pd.DataFrame(state_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Collection is currently empty.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Complete Data Management Suite in Python", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates:
1. `add()`, `update()`, `upsert()`, and `delete()` in ChromaDB.
2. Safe exception handling on duplicate IDs.
3. Mass deletion by metadata query:
""")

code_dm_suite = '''import chromadb
from chromadb.config import Settings

client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
col = client.create_collection("inventory_docs", metadata={"hnsw:space": "cosine"})

# 1. Strict Add Operation
print("=== 1. Testing collection.add() ===")
col.add(
    documents=["Initial inventory policy."],
    metadatas=[{"category": "Logistics", "active": True}],
    ids=["item_01"]
)
print(f"Record Count after add(): {col.count()}")

# Attempting duplicate add (Should raise error)
try:
    col.add(documents=["Duplicate item."], ids=["item_01"])
except Exception as e:
    print(f"Caught Expected Duplicate Exception: {type(e).__name__}")

# 2. Idempotent Upsert Operation
print("\\n=== 2. Testing collection.upsert() ===")
# Item 01 will be UPDATED; Item 02 will be INSERTED
col.upsert(
    documents=[
        "Updated inventory policy v2.",
        "Brand new warehouse safety protocol."
    ],
    metadatas=[
        {"category": "Logistics", "active": True, "version": 2},
        {"category": "Safety", "active": True, "version": 1}
    ],
    ids=["item_01", "item_02"]
)
print(f"Record Count after upsert(): {col.count()}")
print(f"Item 01 Document Content: '{col.get(ids=['item_01'])['documents'][0]}'")

# 3. In-Place Update Operation
print("\\n=== 3. Testing collection.update() ===")
col.update(
    ids=["item_02"],
    metadatas=[{"category": "Safety", "active": False, "version": 1, "status": "reviewed"}]
)
print(f"Item 02 Updated Metadata: {col.get(ids=['item_02'])['metadatas'][0]}")

# 4. Mass Deletion by Metadata Filter
print("\\n=== 4. Mass Deletion by Metadata Query ===")
col.delete(where={"category": "Logistics"})
print(f"Record Count after deleting Logistics: {col.count()}")
print(f"Remaining IDs: {col.get()['ids']}")
'''

st.code(code_dm_suite, language="python")

if st.button("🚀 Execute Data Management Suite Demo", type="primary"):
    exec_scope = {}
    exec(code_dm_suite, globals(), exec_scope)
    
    cnt_final = exec_scope["col"].count()
    rem_ids = exec_scope["col"].get()["ids"]
    
    st.success("Data Management Suite executed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown(f"**Final Collection Record Count:** `{cnt_final}`")
    with col_out2:
        st.markdown(f"**Remaining IDs Post-Mass Delete:** `{rem_ids}` (Logistics records cleanly purged!)")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why do distributed, high-throughput ingestion pipelines standardize on `upsert()` rather than `add()`?"""
a1 = """* **Pipeline Failure Recovery & Idempotency**:
  * In distributed systems, worker pods crash, network timeouts occur, and jobs are retried automatically.
  * If an ingestion worker processes 1,000 documents and crashes on document 800, restarting the job with `add()` will immediately fail on document 1 with a duplicate ID exception!
  * `upsert()` guarantees that the entire pipeline is re-entrant and idempotent: running the ingestion job once, twice, or ten times produces the exact same clean database state without crashing."""

q2 = """2. How do Vector Databases execute mass deletions by metadata query (`delete(where={...})`) efficiently?"""
a2 = """* Naive systems would scan every vector in the database, check the metadata JSON string, and delete matches row by row.
* **Vector DB Optimization**:
  1. The database queries its **Metadata Inverted Index** (Roaring Bitsets) matching the predicate (e.g. `category = 'Logistics'`).
  2. The resulting matching bitset is merged directly into the global deletion tombstone bitset in microseconds.
  3. All matching records become immediately invisible to incoming search queries without a full table scan.
  4. Physical disk reclamation occurs asynchronously during scheduled background segment compaction."""

q3 = """3. What is the difference between Lexical Exact Deduplication and Semantic Near-Deduplication?"""
a3 = """* **Exact Lexical Deduplication**:
  * Compares strings character-for-character using cryptographic hashes (SHA-256).
  * Extreme speed (microseconds), zero AI compute cost.
  * Fails when a user fixes a single typo, changes capitalization, or paraphrases text.
* **Semantic Near-Deduplication**:
  * Generates dense embedding vectors and queries the database for nearest neighbors within a distance threshold (e.g. Cosine Similarity > 0.95).
  * Catches paraphrased policies, cross-scraped articles, and minor textual edits.
  * More computationally expensive: requires running the embedding model prior to deduplication."""

q4 = """4. What happens if you execute a partial `collection.update()` without passing `documents` or `embeddings`?"""
a4 = """* ChromaDB detects that neither `documents` nor `embeddings` was provided in the update payload.
* It performs an **in-place metadata update** inside SQLite, updating only the scalar attributes and inverted filter indexes.
* It completely leaves the HNSW vector index untouched.
* This is a massive performance optimization: updating document permissions or status flags consumes zero GPU inference cycles and causes zero HNSW graph rewiring."""

q5 = """5. What is the danger of running un-batched mass deletions on large production collections?"""
a5 = """* Deleting 500,000 records in a single synchronous operation acquires write locks across SQLite and HNSW segment structures.
* In high-traffic production environments, this can stall concurrent search queries and trigger upstream API gateway timeouts (504 Gateway Timeout).
* **Best Practice**: Batch mass deletions into smaller chunks (e.g. deleting 1,000 to 5,000 records per transaction) to allow concurrent queries to proceed without locking."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Data Management Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Broken Pipeline on Unhandled `add()` Duplicate IDs**  
> Writing a batch ETL script using `collection.add()` against a live database will cause the entire pipeline 
> to crash the moment a duplicate document is encountered. Always use `collection.upsert()` for recurring data pipelines.

> [!WARNING]
> **Trap 2: The Inadvertent Text-Vector Decoupling Bug**  
> Calling `collection.update(ids=["doc_1"], documents=["New policy text"])` without an embedding function or 
> without passing new `embeddings` can result in storing updated text alongside an **OLD, STALE vector**! 
> Always ensure that updating text triggers fresh vector generation.

> [!WARNING]
> **Trap 3: Silent No-Op Deletions on Typoed Metadata Filters**  
> Calling `collection.delete(where={"deparment": "Sales"})` (with a typo in `department`) will silently match 
> zero records and delete nothing without raising an error! Always verify that the targeted metadata key exists 
> before issuing mass deletion commands.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: Why Use `upsert()` Over `add()`?"):
        st.markdown("""
        **`upsert()` Advantage**:  
        Inserts new records and updates existing ones idempotently, making data pipelines crash-proof and re-entrant.
        """)
    with st.expander("🗂️ Flashcard: How Does Mass Delete Work?"):
        st.markdown("""
        **Mass Deletion**:  
        `collection.delete(where={...})` uses metadata bitsets to soft-delete matching records in microseconds without scanning vectors.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Exact vs Semantic Deduplication?"):
        st.markdown("""
        **Deduplication Modes**:  
        Exact uses SHA-256 on CPU (free, fast); Semantic uses cosine proximity > 0.95 (catches paraphrases, requires embedding).
        """)
    with st.expander("🗂️ Flashcard: What Invariant Governs `update()`?"):
        st.markdown("""
        **`update()` Invariant**:  
        Target IDs must already exist in the collection, or the operation fails / skips missing items.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can clearly differentiate the semantics of `add()`, `update()`, `upsert()`, and `delete()`.", value=True)
chk2 = st.checkbox("I understand why `collection.upsert()` is the production standard for pipeline idempotency.")
chk3 = st.checkbox("I know how mass deletions operate using metadata filters (`where={...}`).")
chk4 = st.checkbox("I understand the difference between exact SHA-256 deduplication and semantic near-deduplication.")
chk5 = st.checkbox("I know how to execute in-place metadata updates without triggering vector re-indexing.")
chk6 = st.checkbox("I can implement a complete Data Management suite in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Incredible achievement! You have mastered Data Management and completed Chapter 7 of the Vector Database curriculum!")
