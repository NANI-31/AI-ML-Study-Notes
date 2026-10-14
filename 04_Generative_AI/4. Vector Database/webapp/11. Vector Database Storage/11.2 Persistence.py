import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="11.2 Persistence - Vector DB Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("11.2 Persistence: Saving, Loading, Backups, Restoration & Durability")
st.caption("Chapter 11: Vector Database Storage | Domain: 4. Vector Database")
st.markdown("""
In enterprise production environments, vector databases are the authoritative memory of AI agents, customer knowledge bases, 
and RAG retrieval pipelines. 

A vector database that cannot guarantee **strict durability, crash consistency, and point-in-time backup restoration** risks 
destroying weeks of expensive GPU embedding computations and corrupting corporate knowledge repositories.

In ChromaDB 0.4.x+, persistence is native and automated:
* Changes are logged to an ACID Write-Ahead Log (WAL) and stored in `chroma.sqlite3`.
* Serialized HNSW graph segments are flushed to disk as immutable data blocks.
* Explicit manual calls like `client.persist()` from legacy Chroma (0.3.x) have been deprecated in favor of **continuous automatic persistence**.

In this lesson, we explore how vectors are serialized and loaded under the hood, examine enterprise **Hot vs. Cold backup strategies**, 
learn how to recover from corruptions, and walk through an interactive Disaster Recovery Workbench.
""")

st.divider()

# Section 1: The Vector Persistence Lifecycle: Saving & Loading
st.header("1. The Vector Persistence Lifecycle: Saving & Loading", divider="rainbow")

col_life1, col_life2 = col_life1, col_life2 = st.columns([1.1, 0.9])

with col_life1:
    st.markdown("""
    ### How ChromaDB Automatically Saves Data
    When you call `collection.add(...)` or `collection.upsert(...)` using `PersistentClient`:
    1. **SQLite WAL Write**: Record IDs, raw document strings, metadata dictionaries, and embedding float vectors are written as an atomic transaction into SQLite (`chroma.sqlite3-wal`).
    2. **In-Memory HNSW Update**: The in-memory graph index dynamically wires the new vectors into its multi-layer skip graph.
    3. **Background Segment Flush**: Periodically (or upon process exit), the in-memory HNSW index segment is serialized to disk inside subdirectories identified by UUIDs.
    
    ### Legacy vs Modern ChromaDB Persistence
    * **Legacy Chroma (< 0.4.0)**: Required developers to manually invoke `client.persist()` to flush in-memory data to disk. Forgetting this call led to total data loss!
    * **Modern Chroma (>= 0.4.0)**: **Fully Automatic**. Every write is persisted to SQLite immediately. Calling `client.persist()` now raises a deprecation warning.
    """)

with col_life2:
    st.markdown("### Persistence Data Flow Diagram")
    persist_dot = """
    digraph PersistenceFlow {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        WriteReq [label="collection.add()\\nNew Records", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_dual {
            label="Dual-Tier Write Engine"; style="dashed"; color="#059669";
            SQLiteWAL [label="1. SQLite WAL Journal\\nACID Durable Append", fillcolor="#FEF3C7", color="#D97706"];
            RAMGraph [label="2. In-Memory HNSW Graph\\nImmediate Search Availability", fillcolor="#D1FAE5", color="#059669"];
        }

        FlushDaemon [label="Background Compactor\\nSync to Segment Files", fillcolor="#E0E7FF", color="#4338CA"];
        DiskStore [label="Disk Directory (/path)\\nchroma.sqlite3 + UUID dirs", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        WriteReq -> SQLiteWAL;
        WriteReq -> RAMGraph;
        SQLiteWAL -> FlushDaemon;
        RAMGraph -> FlushDaemon;
        FlushDaemon -> DiskStore;
    }
    """
    st.graphviz_chart(persist_dot, use_container_width=False)

st.divider()

# Section 2: Loading Collections & Verifying Integrity
st.header("2. Loading Collections & Verifying Integrity", divider="rainbow")

col_load1, col_load2 = col_load1, col_load2 = st.columns([1, 1])

with col_load1:
    st.markdown("""
    ### Re-Connecting to an Existing Store
    When your application restarts or boots on another machine, pointing `PersistentClient` to the existing directory 
    restores all collections, documents, and index structures:
    
    ```python
    import chromadb

    # Initialize client pointing to existing folder
    client = chromadb.PersistentClient(path="./enterprise_vector_store")

    # 1. Inspect existing collections
    collections = client.list_collections()
    print("Found collections:", [c.name for c in collections])

    # 2. Re-attach to specific collection
    collection = client.get_collection(name="kb_production")

    # 3. Verify record count
    total_docs = collection.count()
    print(f"Loaded {total_docs:,} vectors from disk.")
    ```
    """)

with col_load2:
    st.markdown("""
    ### Defensive Loading: `get_or_create_collection`
    In containerized cloud environments (e.g., Docker, Kubernetes, AWS ECS), containers can restart unexpectedly. 
    Using `client.get_collection(...)` crashes with a `ValueError` if the collection does not yet exist.
    
    **Production Best Practice**: Always use `get_or_create_collection`:
    ```python
    collection = client.get_or_create_collection(
        name="kb_production",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 32,
            "hnsw:construction_ef": 200
        }
    )
    # If the collection exists, it loads seamlessly without modifying metadata.
    # If it is missing, it initializes with verified production hyperparameters.
    ```
    """)

st.divider()

# Section 3: Enterprise Backup & Disaster Recovery Architecture
st.header("3. Enterprise Backup & Disaster Recovery Architecture", divider="rainbow")

st.markdown("""
A robust backup strategy must account for the fact that a vector database directory contains two synchronized components:
1. The **SQLite Relational Metadata Database** (`chroma.sqlite3`).
2. The **HNSW Index Segment Binary Files** (UUID directories holding `.bin` graph adjacency lists).
""")

col_bk1, col_bk2 = col_bk1, col_bk2 = st.columns([1, 1])

with col_bk1:
    st.markdown("""
    ### The "Torn State" Trap in Hot Backups
    * **Danger of Naive File Copy (`cp -r`)**: If a cron script executes `cp -r ./my_chroma_db ./backup` while Chroma is actively writing, SQLite may be in the middle of a WAL checkpoint while an HNSW segment is only half-written. The resulting backup will be **corrupted and un-restorable**!
    * **Safe Hot Backup Protocol**:
      1. **Flush SQLite WAL**: Execute a checkpoint to force dirty WAL pages into `chroma.sqlite3`:
         ```sql
         PRAGMA wal_checkpoint(TRUNCATE);
         ```
      2. **Use SQLite Online Backup API**: Use `VACUUM INTO '/backup/chroma.sqlite3'` to create an atomic, crash-consistent snapshot of the database file.
      3. **Snapshot Index Segments**: Copy the UUID index folders once the checkpoint is complete.
    """)

with col_bk2:
    st.markdown("""
    ### Cold Backups & Cloud Object Storage (S3 / GCS)
    For reliable automated daily backups:
    1. **Pause Ingestion**: Momentarily pause ingestion microservices (read-only queries can continue).
    2. **Archive Directory**: Create a compressed tarball with SHA-256 checksums:
       ```bash
       tar -czvf vector_db_backup_$(date +%F).tar.gz ./chroma_data/
       sha256sum vector_db_backup_*.tar.gz > checksum.txt
       ```
    3. **Upload to Cloud Vault**: Stream the tarball to an AWS S3 bucket with Object Lock (immutability):
       ```bash
       aws s3 cp vector_db_backup_*.tar.gz s3://corp-backups/vectordb/
       ```
    4. **Automated Recovery Test**: Regularly test restoration into a staging container to verify query validity.
    """)

st.divider()

# Section 4: Interactive Workbench: Automated Disaster Recovery Simulator
st.header("4. Interactive Workbench: Disaster Recovery Simulator", divider="rainbow")

st.markdown("""
Simulate an end-to-end enterprise disaster recovery workflow: take an atomic snapshot, simulate filesystem corruption, 
and execute automated rollback and integrity verification.
""")

col_dr_state, col_dr_actions = st.columns([1, 1.2])

with col_dr_state:
    st.subheader("1. Active Production Database Status")
    
    # Mock database state
    live_records_count = st.number_input("Current Live Vector Count:", min_value=1000, max_value=5000000, value=250000, step=50000)
    collection_name = st.text_input("Active Collection Name:", value="customer_service_kb")
    
    st.info(f"""
    **Directory**: `/var/data/chroma_prod`  
    * **Primary Engine**: SQLite 3.42 + WAL  
    * **Active Vectors**: `{live_records_count:,}`  
    * **WAL Journal Size**: `14.2 MB` (uncommitted pages)  
    * **HNSW Index Segments**: `4 segment directories`
    """)

with col_dr_actions:
    st.subheader("2. Disaster Recovery Control Panel")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        trigger_backup = st.button("📸 1. Take Snapshot Backup", use_container_width=True)
    with col_btn2:
        trigger_corrupt = st.button("💥 2. Simulate File Corruption", use_container_width=True)
    with col_btn3:
        trigger_restore = st.button("🔄 3. Restore from Backup", use_container_width=True)

# State Management for Simulator
if "dr_snapshot" not in st.session_state:
    st.session_state.dr_snapshot = None
if "dr_status" not in st.session_state:
    st.session_state.dr_status = "HEALTHY"

if trigger_backup:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    st.session_state.dr_snapshot = {
        "timestamp": timestamp,
        "records": live_records_count,
        "collection": collection_name,
        "archive_size_mb": round((live_records_count * 1536 * 4) / (1024**2) * 0.45, 1),
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"[:16] + "..."
    }
    st.session_state.dr_status = "BACKUP_AVAILABLE"

if trigger_corrupt:
    st.session_state.dr_status = "CORRUPTED"

if trigger_restore:
    if st.session_state.dr_snapshot:
        st.session_state.dr_status = "RESTORED"
    else:
        st.error("No backup snapshot exists to restore from! Take a snapshot first.")

st.divider()

st.subheader("3. System Integrity & Audit Log")

col_log_status, col_log_audit = st.columns([0.8, 1.2])

with col_log_status:
    if st.session_state.dr_status == "HEALTHY":
        st.success("🟢 **System Status**: HEALTHY (Live)")
        st.caption("Active database is serving queries with zero integrity errors.")
    elif st.session_state.dr_status == "BACKUP_AVAILABLE":
        st.success("🟢 **System Status**: HEALTHY + BACKUP READY")
        st.caption("Consistent snapshot verified and archived to vault.")
    elif st.session_state.dr_status == "CORRUPTED":
        st.error("🔴 **System Status**: DISASTER DETECTED!")
        st.markdown("""
        * `chroma.sqlite3`: **Corrupted disk blocks (I/O error)**
        * HNSW Index: **Orphaned edge pointers**
        * Production Traffic: **500 Internal Server Errors**
        """)
    elif st.session_state.dr_status == "RESTORED":
        st.success("🟢 **System Status**: RESTORED & VERIFIED")
        st.caption("Backup tarball unpacked, SQLite PRAGMA check passed, and HNSW index re-validated.")

with col_log_audit:
    if st.session_state.dr_snapshot:
        snap = st.session_state.dr_snapshot
        st.markdown("#### Latest Snapshot Manifest")
        st.code(f"""Snapshot Manifest:
- Created At: {snap['timestamp']}
- Collection: {snap['collection']}
- Captured Records: {snap['records']:,}
- Archive Size: {snap['archive_size_mb']} MB (gzip compressed)
- Checksum: SHA-256:{snap['sha256']}
- Integrity Check: PRAGMA integrity_check == 'ok'
""", language="yaml")
    else:
        st.warning("No snapshot captured yet. Click '1. Take Snapshot Backup' to initiate a backup.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why was `client.persist()` deprecated in ChromaDB 0.4.0?"):
    st.markdown("""
    **Answer**:
    In ChromaDB < 0.4.0, write operations were kept in a volatile Python in-memory state until the user explicitly executed `client.persist()`. 
    This caused widespread production failures:
    1. Unhandled exceptions or container kills before `persist()` caused silent data loss.
    2. Multi-threaded applications had race conditions between write calls and persist calls.
    
    In ChromaDB 0.4.0+, the architecture transitioned to **SQLite as the authoritative write-ahead system of record**. Every `.add()`, `.update()`, or `.delete()` call commits atomically to SQLite and updates index segments immediately, making persistence automatic and continuous.
    """)

with st.expander("Q2: How do you safely back up an embedded ChromaDB database while production writes are occurring?"):
    st.markdown("""
    **Answer**:
    Directly copying files (`cp -r`) risks creating a "torn" backup where the SQLite database and HNSW segment files are out of sync.
    
    To execute a safe hot backup:
    1. **SQLite Checkpoint**: Force WAL flush: `PRAGMA wal_checkpoint(TRUNCATE)`.
    2. **Online Backup API**: Run `VACUUM INTO '/backup/path/chroma.sqlite3'` from a Python script or SQLite shell. This creates an atomic, consistent copy of the database even during active writes.
    3. **Copy Segments**: Copy the immutable index segment directories.
    4. **Checksum Validation**: Compute SHA-256 hashes of all copied files to guarantee transfer fidelity.
    """)

with st.expander("Q3: What happens during startup if ChromaDB detects that SQLite has records that are missing from the HNSW index?"):
    st.markdown("""
    **Answer**:
    SQLite is ChromaDB's **authoritative single source of truth**. 
    During initialization, Chroma checks the recorded index segment commit offsets against the SQLite record tables. 
    If a crash occurred between the SQLite commit and the HNSW segment disk flush, Chroma identifies the missing records in SQLite and **re-indexes them into the HNSW graph** in memory before declaring the collection ready to serve queries.
    """)

with st.expander("Q4: How do you restore a corrupted ChromaDB collection from a backup archive?"):
    st.markdown("""
    **Answer**:
    1. **Halt Application**: Stop the FastAPI / Streamlit backend services to release all file locks on the data directory.
    2. **Isolate Corrupted Directory**: Move the broken folder to a diagnostic quarantine (`mv ./chroma_data ./chroma_data_corrupt_$(date +%s)`).
    3. **Extract Backup**: Unpack the validated backup archive: `tar -xzvf backup.tar.gz -C ./chroma_data/`.
    4. **Integrity Audit**: Verify SQLite health using Python: `sqlite3.connect('chroma.sqlite3').execute('PRAGMA integrity_check')`.
    5. **Smoke Test Query**: Initialize `PersistentClient(path='./chroma_data')`, run a test `.query()`, and verify that `count()` matches the snapshot manifest.
    6. **Resume Service**: Start the web application containers.
    """)

with st.expander("Q5: What is Point-In-Time Recovery (PITR) and how is it implemented for vector databases?"):
    st.markdown("""
    **Answer**:
    Point-In-Time Recovery allows restoring a database to any specific second in the past (e.g. 5 minutes before an accidental bulk deletion). 
    * In self-hosted vector databases, PITR is achieved by archiving **base full snapshots** weekly, combined with **continuous continuous WAL journal streaming** to S3. To restore, the system replays WAL transactions up to the target timestamp.
    * In managed cloud vector databases (Pinecone, Qdrant Cloud), PITR is handled natively by the cloud control plane using distributed log replication (Raft / Kafka).
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Writing `client.persist()` in Chroma 0.4+**  
    Writing code with `client.persist()` in modern Chroma produces deprecation warnings and reveals reliance on outdated documentation. Persistence is continuous and automatic.
    """)
    st.warning("""
    **Trap 2: Backing Up `chroma.sqlite3` Without the Index Folders**  
    Backing up only `chroma.sqlite3` saves the text and metadata, but leaves behind the HNSW graph index. Upon restore, Chroma must rebuild the entire HNSW graph from scratch, taking hours on large collections.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unmonitored WAL Growth (`chroma.sqlite3-wal`)**  
    If long-running read queries prevent SQLite from checkpointing, the `-wal` file can balloon to tens of gigabytes, consuming all host disk space. Monitor WAL file size in production.
    """)
    st.warning("""
    **Trap 4: Never Testing the Restoration Workflow**  
    An un-tested backup is not a backup. Schedule monthly automated disaster recovery drills in staging environments to confirm that restored archives load and query correctly.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Is `client.persist()` needed in Chroma 0.4+?"):
        st.markdown("""
        **No**:  
        Persistence is automatic. Every write commits immediately to SQLite and segment files.
        """)
    with st.expander("🗂️ Flashcard: What is the authoritative source of truth in Chroma?"):
        st.markdown("""
        **Source of Truth**:  
        `chroma.sqlite3`. The HNSW index can be rebuilt from SQLite if damaged.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is a safe hot backup method for SQLite?"):
        st.markdown("""
        **VACUUM INTO**:  
        `VACUUM INTO '/path/backup.sqlite3'` creates an atomic, crash-consistent copy during active writes.
        """)
    with st.expander("🗂️ Flashcard: How to defensively connect to collections?"):
        st.markdown("""
        **Best Practice**:  
        Use `client.get_or_create_collection(name=...)` to avoid crashes on container reboots.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why `client.persist()` is deprecated in modern ChromaDB (0.4+).", value=True)
chk2 = st.checkbox("I know how SQLite WAL transactions guarantee durability for vector metadata and documents.", value=True)
chk3 = st.checkbox("I understand the danger of naive `cp -r` hot backups and how `VACUUM INTO` prevents torn states.", value=True)
chk4 = st.checkbox("I can defensively connect to collections using `get_or_create_collection()`.", value=True)
chk5 = st.checkbox("I know how to structure an automated S3/GCS disaster recovery backup pipeline.", value=True)
chk6 = st.checkbox("I can execute an end-to-end vector database restoration and verify data integrity.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Persistence and are ready for Chapter 11.3: Storage Optimization & Quantization!")
