import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import tempfile
import shutil
import os
from typing import List, Dict, Any, Optional

try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(
    page_title="14.3 Persistent Vector Store - Embeddings Curriculum",
    page_icon="💾",
    layout="wide"
)

# Title & Overview
st.title("14.3 Persistent Vector Store: Disk Serialization, CRUD Ops, & Collection Lifecycle")
st.caption("Chapter 14: Embeddings with ChromaDB | Domain: 3. Vector Embeddings")
st.markdown("""
In development, ephemeral in-memory vector stores (`chromadb.EphemeralClient()`) are convenient for quick experimentation.
However, in production applications, embeddings must **survive application restarts, server reboots, and container redeployments**.

In this lesson, you will master the full persistence and operational maintenance lifecycle of ChromaDB:
* **Persistent Storage**: Initializing `chromadb.PersistentClient(path="./chroma_db")` to serialize embeddings and metadata to disk.
* **On-Disk Storage Anatomy**: Inspecting `chroma.sqlite3` database files, UUID segment folders, and HNSW index binaries.
* **Reloading Collections**: Accessing existing collections after app shutdown with `client.get_collection()`.
* **Updating Documents**: Executing in-place document and metadata mutations with `collection.update()` and `collection.upsert()`.
* **Deleting Documents**: Removing stale documents by ID or via metadata filter predicates (`collection.delete(where=...)`).
* **Managing Collections**: Listing active collections, reading telemetry, and safely dropping collections (`client.delete_collection()`).
""")

st.divider()

# Section 1: ChromaDB Persistence Architecture & Disk Anatomy
st.header("1. Persistence Architecture & On-Disk Anatomy", divider="rainbow")

col_disk1, col_disk2 = st.columns([1, 1])

with col_disk1:
    st.markdown("""
    ### How PersistentClient Stores Data
    * **1. `chroma.sqlite3` Database**:
      * The single relational file holding collection catalogs, document text records, metadata dictionaries, and operational write-ahead logs.
      * Handles relational constraints, ACID transactional writes, and segment mapping tables.
    * **2. Segment Directories (UUID Folders)**:
      * For each collection, ChromaDB creates subdirectories named with UUIDs corresponding to index segments.
      * Contains memory-mapped HNSW graph files (`data_level0.bin`, `header.bin`, `link_lists.bin`).
    * **3. Crash Durability & Recovery**:
      * Writes are appended to SQLite and index buffers.
      * When reloading a collection on restart, ChromaDB reads metadata from SQLite and uses `mmap` to load the pre-built HNSW graph instantly without re-embedding documents!
    """)

with col_disk2:
    st.markdown("### Persistence Lifecycle Workflow")
    persist_dot = """
    digraph ChromaPersist {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        AppWrite [label="App Action:\\ncollection.upsert(...)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_disk {
            label="Persistent Directory (./chroma_data)"; style="dashed"; color="#059669";
            SqliteFile [label="chroma.sqlite3\\n(Metadata, IDs, Documents, WAL)", fillcolor="#D1FAE5"];
            HNSWDir [label="Segment UUID Folder\\n(data_level0.bin, hnsw graph)", fillcolor="#A7F3D0"];
        }

        Restart [label="Server Restart / New Session\\nPersistentClient(path='./chroma_data')", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        ReloadCollect [label="client.get_collection(...)\\nInstant mmap reload (0s re-embedding)", fillcolor="#ECFDF5", color="#10B981"];

        AppWrite -> SqliteFile [label="ACID insert"];
        AppWrite -> HNSWDir [label="Index update"];
        SqliteFile -> Restart [style="dotted"];
        HNSWDir -> Restart [style="dotted"];
        Restart -> ReloadCollect [label="Read schema & mount graph"];
    }
    """
    st.graphviz_chart(persist_dot, use_container_width=False)

st.markdown("### PersistentClient Methods vs EphemeralClient Methods")
st.markdown("""
| Operation | Method Signature | Behavior |
| :--- | :--- | :--- |
| **Initialize Disk Client** | `chromadb.PersistentClient(path="./my_data")` | Creates directory if missing; loads existing DB if present. |
| **Get Existing Collection** | `client.get_collection(name="kb")` | Loads collection from disk; throws error if collection not found. |
| **List All Collections** | `client.list_collections()` | Returns list of all available collection objects in the persistent store. |
| **Update Document** | `collection.update(ids=[...], documents=[...])` | Mutates existing documents/metadata; errors if ID not found. |
| **Upsert Document** | `collection.upsert(ids=[...], documents=[...])` | Inserts if new, updates if existing (safe production choice). |
| **Delete by ID / Filter** | `collection.delete(ids=[...], where={...})` | Removes vectors and metadata from SQLite and invalidates HNSW nodes. |
| **Delete Collection** | `client.delete_collection(name="kb")` | Permanently deletes the collection, SQLite records, and index files. |
""")

st.divider()

# Section 2: Production Code: Persistent CRUD Pipeline
st.header("2. Production Code: Complete Persistent CRUD Lifecycle", divider="rainbow")

st.markdown("""
The production utility below demonstrates full lifecycle management: persisting records, reloading collections across process boundaries, updating documents, deleting records, and dropping collections.
""")

st.code(r'''
import chromadb
from typing import List, Dict, Any

def run_persistent_crud_demo(storage_path: str = "./production_chroma_db"):
    # 1. Initialize persistent client pointing to local storage directory
    client = chromadb.PersistentClient(path=storage_path)

    # 2. Create or reload collection
    collection = client.get_or_create_collection(
        name="policy_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Insert initial documents
    collection.upsert(
        ids=["pol_001", "pol_002"],
        documents=[
            "Employees are entitled to 20 days of paid annual leave per calendar year.",
            "Remote work requests must be approved by the department manager 14 days in advance."
        ],
        metadatas=[
            {"category": "hr", "version": 1, "year": 2023},
            {"category": "hr", "version": 1, "year": 2023}
        ]
    )
    print(f"Initial count: {collection.count()}")

    # 4. Update an existing document (e.g. Policy update in 2024)
    collection.update(
        ids=["pol_001"],
        documents=["Employees are entitled to 25 days of paid annual leave per calendar year."],
        metadatas=[{"category": "hr", "version": 2, "year": 2024}]
    )
    print("Updated pol_001 to version 2.")

    # 5. Delete a specific record by ID
    collection.delete(ids=["pol_002"])
    print(f"Count after deletion: {collection.count()}")

    # 6. Verify persistence by creating a fresh client instance pointing to same path
    fresh_client = chromadb.PersistentClient(path=storage_path)
    reloaded_col = fresh_client.get_collection(name="policy_documents")
    print(f"Reloaded count from fresh client: {reloaded_col.count()}")

    # 7. List all collections in client
    all_cols = [c.name for c in fresh_client.list_collections()]
    print(f"Active collections in storage: {all_cols}")

if __name__ == "__main__":
    run_persistent_crud_demo()
''', language="python")

st.divider()

# Section 3: Interactive Persistent Lifecycle & CRUD Studio
st.header("3. Interactive Workbench: Live Persistent Client & CRUD Studio", divider="rainbow")

st.markdown("""
This interactive workbench simulates a live persistent database session. You can create collections, upsert records, delete documents, and trigger a **Simulated Restart** to verify disk reload persistence.
""")

# Setup session state storage directory
if "chroma_temp_dir" not in st.session_state:
    st.session_state["chroma_temp_dir"] = tempfile.mkdtemp(prefix="chroma_demo_")

temp_dir = st.session_state["chroma_temp_dir"]

col_crud1, col_crud2 = st.columns([1, 1])

with col_crud1:
    st.subheader("1. Collection Management")
    st.caption(f"Current Disk Storage Path: `{temp_dir}`")
    
    # Initialize persistent client
    p_client = chromadb.PersistentClient(path=temp_dir)
    existing_cols = [c.name for c in p_client.list_collections()]
    
    new_col_name = st.text_input("New Collection Name:", value="customer_faq")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ Create Collection"):
            try:
                p_client.get_or_create_collection(name=new_col_name, metadata={"hnsw:space": "cosine"})
                st.success(f"Collection '{new_col_name}' created!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                
    with col_btn2:
        if st.button("🗑️ Delete Collection"):
            try:
                p_client.delete_collection(name=new_col_name)
                st.warning(f"Collection '{new_col_name}' deleted!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.write(f"**Existing Collections on Disk ({len(existing_cols)}):** `{existing_cols}`")

    st.subheader("2. Document Mutation (Upsert / Delete)")
    active_col_name = st.selectbox("Select Active Collection:", options=existing_cols if existing_cols else ["customer_faq"])
    
    if active_col_name in existing_cols:
        active_col = p_client.get_collection(name=active_col_name)
        
        doc_id = st.text_input("Document ID:", value="faq_billing_01")
        doc_text = st.text_area("Document Text:", value="Users can cancel their subscription at any time under Account Settings > Billing.")
        doc_cat = st.selectbox("Category:", ["billing", "technical", "account", "security"])
        doc_rev = st.number_input("Revision Version:", min_value=1, max_value=10, value=1)
        
        c_up, c_del = st.columns(2)
        with c_up:
            if st.button("💾 Upsert Document", type="primary"):
                try:
                    active_col.upsert(
                        ids=[doc_id],
                        documents=[doc_text],
                        metadatas=[{"category": doc_cat, "revision": int(doc_rev)}]
                    )
                    st.success(f"Upserted ID '{doc_id}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upsert failed: {e}")
        with c_del:
            if st.button("❌ Delete by ID"):
                try:
                    active_col.delete(ids=[doc_id])
                    st.warning(f"Deleted ID '{doc_id}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

with col_crud2:
    st.subheader("3. Persistent Store Verification & Reload Test")
    
    if active_col_name in existing_cols:
        active_col = p_client.get_collection(name=active_col_name)
        total_items = active_col.count()
        
        st.metric("Total Documents in Active Collection", total_items)
        
        if total_items > 0:
            stored_data = active_col.get(include=["documents", "metadatas"])
            records = []
            for i in range(len(stored_data["ids"])):
                records.append({
                    "ID": stored_data["ids"][i],
                    "Document Snippet": stored_data["documents"][i][:50] + "...",
                    "Metadata": str(stored_data["metadatas"][i])
                })
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.info("Collection is currently empty. Upsert documents using the form on the left.")

    st.markdown("#### Simulate Server Crash & Full Reload")
    st.markdown("Click below to instantiate a completely new `PersistentClient` from disk to verify zero data loss:")
    
    if st.button("🔄 Test Disk Reload Verification"):
        # Re-initialize independent client
        fresh_client = chromadb.PersistentClient(path=temp_dir)
        fresh_collections = [c.name for c in fresh_client.list_collections()]
        
        st.success("✅ Fresh client connected to storage path successfully!")
        st.write(f"Found **{len(fresh_collections)}** persisted collections: `{fresh_collections}`")
        
        if active_col_name in fresh_collections:
            reloaded = fresh_client.get_collection(name=active_col_name)
            st.info(f"Verified collection `{active_col_name}` contains **{reloaded.count()}** records persisted on disk!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Persistence Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does ChromaDB achieve fast cold-start reload times when restarting a persistent server?**
    * **Answer**: ChromaDB stores its vector indexes using HNSWlib memory-mapped (`mmap`) binary files. When a persistent client starts, the OS kernel maps the binary index files directly into the virtual address space without parsing or allocating heap arrays in Python. This allows instantaneous cold-start querying with zero re-embedding overhead.

    **Q2: What is the risk of having multiple Python processes accessing the same `PersistentClient` directory?**
    * **Answer**: ChromaDB uses SQLite under the hood. While SQLite supports concurrent reads, concurrent writes from multiple processes can result in `database is locked` errors or corrupted index segments. For multi-process or distributed workloads, you should deploy ChromaDB in client-server mode (`chroma run --path ./my_data`) and connect via `chromadb.HttpClient(host="...", port=8000)`.

    **Q3: How does deleting a document affect the underlying HNSW index in ChromaDB?**
    * **Answer**: In HNSW graphs, physically removing a node from a multi-layer graph requires complex edge re-wiring. Most ANN engines, including ChromaDB, mark the vector ID as deleted in SQLite and soft-mask it from search results. Periodically, during segment compaction or index optimization, the graph is compacted to reclaim memory.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Running Multiple Concurrent Writers on a Single Persistent Directory**:
      * *Issue*: Spawning 4 Celery worker processes all pointing to `PersistentClient(path="./chroma_db")`.
      * *Consequence*: Crashes with `sqlite3.OperationalError: database is locked`.
      * *Fix*: Deploy ChromaDB as a centralized container (`chromadb.HttpClient`) or serialize writes through a single worker queue.

    * **Trap 2: Accidentally Calling `get_or_create_collection` with Different Distance Metrics**:
      * *Issue*: Initially creating a collection with `cosine`, then later calling `get_or_create_collection(metadata={"hnsw:space": "l2"})`.
      * *Consequence*: The original metadata remains unchanged in SQLite; the new metric is ignored or leads to unexpected distance scoring.
      * *Fix*: Explicitly check `collection.metadata` or create distinct collections for different distance spaces.

    * **Trap 3: Deleting Files Manually Inside the Storage Directory**:
      * *Issue*: Manually deleting UUID segment folders to free up disk space while leaving `chroma.sqlite3` intact.
      * *Consequence*: SQLite metadata becomes out of sync with index segment binaries, causing unrecoverable corrupted index errors on next query.
      * *Fix*: Always manage collections and deletions via the official API (`client.delete_collection()` or `collection.delete()`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `chromadb.PersistentClient(path)`**
      * *Definition*: The ChromaDB client configuration that serializes vectors, SQLite metadata, and HNSW graph files to a persistent directory.
    * **Flashcard 2: `chroma.sqlite3`**
      * *Definition*: The primary relational metadata database file stored within the persistent storage directory.
    * **Flashcard 3: `client.delete_collection(name)`**
      * *Definition*: The method that permanently drops a collection, deleting all documents, metadata, and segment files.
    * **Flashcard 4: Client-Server Mode**
      * *Definition*: Running ChromaDB as a standalone HTTP microservice (`chroma run`) to support multi-process concurrent writes.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Persistent Vector Stores with ChromaDB:

- [ ] I can configure `chromadb.PersistentClient` for disk-backed storage.
- [ ] I understand the dual-file structure of ChromaDB disk storage (`chroma.sqlite3` + segment UUID folders).
- [ ] I know how to reload collections after process restarts using `get_collection()`.
- [ ] I can perform document updates, upserts, and deletions safely.
- [ ] I understand SQLite locking constraints and when to migrate to `HttpClient`.
""")
