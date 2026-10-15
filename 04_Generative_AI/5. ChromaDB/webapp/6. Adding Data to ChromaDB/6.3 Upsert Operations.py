import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="6.3 Upsert Operations - ChromaDB Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("6.3 Upsert Operations: The upsert() Primitive, Mutation Semantics & External Sync")
st.caption("Chapter 6: Adding Data to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
In distributed architectures, keeping a vector search index synchronized with an authoritative primary database 
(PostgreSQL, MongoDB, Snowflake, or Notion) is one of the most critical engineering responsibilities.

When an ETL worker reads from a Change Data Capture (CDC) stream or message queue, incoming events 
consist of an unpredictable mix of **brand-new records** and **revisions to existing records**.

Using traditional CRUD primitives creates operational nightmares:
* Calling `collection.add()` on an existing document throws a fatal `DuplicateIDError`.
* Calling `collection.update()` on a newly created document throws an `IDNotFoundError`.

The **`collection.upsert()` Primitive** is the foundational solution for idempotent synchronization:
* **Conditional Branching**: Inserts records that do not exist, and atomically overwrites records that already exist.
* **Idempotency Invariant**: Running an upsert operation once, twice, or a thousand times yields the exact same deterministic database state without raising errors.
* **Zero Race Conditions**: Eliminates the need for expensive client-side pre-flight `get(id)` checks before inserting.

In this lesson, we explore upsert mechanics, compare `add` vs `update` vs `upsert`, build a production CDC synchronization pipeline in Python, 
and test an interactive Live Upsert & Database Sync Simulator.
""")

st.divider()

# Section 1: The Mutation Primitives Compared: add() vs update() vs upsert()
st.header("1. The Mutation Primitives Compared: add() vs update() vs upsert()", divider="rainbow")

col_comp1, col_comp2 = col_comp1, col_comp2 = st.columns([1.1, 0.9])

with col_comp1:
    st.markdown("""
    ### Operational Matrix
    
    * **`collection.add(ids=..., documents=...)`**:
      * *Precondition*: ID **MUST NOT exist** in the collection.
      * *If ID Exists*: Raises `DuplicateIDError`. Entire batch aborts atomically.
      * *Best Used For*: Initial batch ingestion where data is guaranteed to be clean and deduplicated.
    
    * **`collection.update(ids=..., documents=...)`**:
      * *Precondition*: ID **MUST exist** in the collection.
      * *If ID Missing*: Raises `IDNotFoundError`.
      * *Best Used For*: Targeted metadata patching (e.g. changing status to `"archived"`) without re-inserting.
    
    * **`collection.upsert(ids=..., documents=...)`**:
      * *Precondition*: **None**.
      * *If ID Missing*: **Inserts** document, metadata, and embedding.
      * *If ID Exists*: **Overwrites** document, metadata, and recalculates embedding.
      * *Best Used For*: Streaming ETL workers, webhooks, CDC synchronization, retry-safe cron jobs.
    """)

with col_comp2:
    st.markdown("### Upsert Decision Flowchart")
    upsert_dot = """
    digraph UpsertFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InEvent [label="Incoming Record\\nupsert(id='cust_101', ...)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        CheckID [label="Does 'cust_101'\\nexist in SQLite / HNSW?", shape=diamond, fillcolor="#FEF3C7"];

        InsertPath [label="Branch A: INSERT\\nCreate new row in SQLite\\nInsert new node into HNSW", fillcolor="#D1FAE5", color="#059669"];

        UpdatePath [label="Branch B: UPDATE\\nOverwrite SQLite text & metadata\\nRecalculate & re-link HNSW node", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Done [label="Deterministic State Reached\\n(Zero Exceptions Thrown)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InEvent -> CheckID;
        CheckID -> InsertPath [label="ID Missing"];
        CheckID -> UpdatePath [label="ID Exists"];
        InsertPath -> Done;
        UpdatePath -> Done;
    }
    """
    st.graphviz_chart(upsert_dot, use_container_width=False)

st.divider()

# Section 2: Production Change Data Capture (CDC) Architecture
st.header("2. Production Change Data Capture (CDC) Architecture", divider="rainbow")

st.markdown("""
In enterprise software, the primary database (e.g. PostgreSQL) is the single source of truth. 
A vector database is a **derived read-optimized index**. The diagram below demonstrates how Debezium 
and Kafka use `upsert()` to keep ChromaDB continuously in sync:
""")

cdc_arch_dot = """
digraph CDCArchitecture {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Postgres [label="Primary Database\\nPostgreSQL Transactions\\n(WAL Log)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    Debezium [label="Debezium CDC Connector\\nCaptures Row Mutations", fillcolor="#FEF3C7", color="#D97706"];

    Kafka [label="Kafka Topic / Queue\\nStreams JSON Events", fillcolor="#EDE9FE", color="#7C3AED"];

    Worker [label="Python Sync Worker\\nBatches 500 events", fillcolor="#D1FAE5", color="#059669"];

    ChromaDB [label="ChromaDB Collection\\ncollection.upsert()\\nAlways Converges!", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

    Postgres -> Debezium -> Kafka -> Worker -> ChromaDB;
}
"""
st.graphviz_chart(cdc_arch_dot, use_container_width=False)

st.divider()

# Section 3: Clean Python Implementation of CDC Upsert Pipeline
st.header("3. Clean Python Implementation of CDC Upsert Pipeline", divider="rainbow")

st.code("""import chromadb
from typing import List, Dict, Any

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="product_catalog_kb",
    metadata={"hnsw:space": "cosine"}
)

def sync_cdc_batch_to_chroma(
    collection,
    cdc_records: List[Dict[str, Any]]
):
    \"\"\"Idempotently processes a stream of CDC database events.\"\"\"
    upsert_ids = []
    upsert_docs = []
    upsert_metas = []
    
    for record in cdc_records:
        upsert_ids.append(f"prod_{record['product_id']}")
        upsert_docs.append(f"{record['name']}. {record['description']}")
        upsert_metas.append({
            "category": record["category"],
            "price_usd": float(record["price"]),
            "in_stock": bool(record["stock"] > 0),
            "version": int(record["version"])
        })
        
    # Atomic upsert: safely handles both inserts and updates simultaneously
    collection.upsert(
        ids=upsert_ids,
        documents=upsert_docs,
        metadatas=upsert_metas
    )
    print(f"Successfully synced {len(upsert_ids)} records to ChromaDB!")

# Simulated CDC Event Batch (Mixture of new items and existing updates)
event_stream = [
    {"product_id": 101, "name": "Mechanical Keyboard", "description": "Cherry MX Blue switches.", "category": "peripherals", "price": 129.99, "stock": 50, "version": 2},
    {"product_id": 102, "name": "Ultra-Wide Monitor", "description": "34-inch 144Hz curved display.", "category": "monitors", "price": 499.00, "stock": 12, "version": 1}
]

sync_cdc_batch_to_chroma(collection, event_stream)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Upsert Synchronization Arena
st.header("4. Interactive Workbench: Live Upsert Synchronization Arena", divider="rainbow")

st.markdown("""
Simulate an upstream PostgreSQL database synchronizing with ChromaDB. 
Modify records in PostgreSQL and trigger an `upsert()` synchronization to observe atomic updates.
""")

if "primary_sql_db" not in st.session_state:
    st.session_state.primary_sql_db = [
        {"id": "doc_101", "title": "Service Level Agreement", "body": "Guaranteed 99.9% uptime for enterprise customers.", "status": "active", "version": 1},
        {"id": "doc_102", "title": "Security Whitepaper", "body": "Data is encrypted in-transit via TLS 1.3 and at-rest via AES-256.", "status": "active", "version": 1}
    ]

col_sql_view, col_action_view = col_sql_view, col_action_view = st.columns([1.1, 0.9])

with col_sql_view:
    st.subheader("1. Primary Upstream Database State (PostgreSQL)")
    st.dataframe(pd.DataFrame(st.session_state.primary_sql_db), use_container_width=True, hide_index=True)

with col_action_view:
    st.subheader("2. Trigger Upstream Database Mutation")
    mutation_type = st.radio(
        "Choose Upstream Event:",
        [
            "Event 1: Revise Existing Document (doc_101: Upgrade to 99.99%)",
            "Event 2: Insert Brand-New Document (doc_103: Disaster Recovery SLA)"
        ]
    )

    if st.button("Commit Change to Upstream SQL DB", use_container_width=True):
        if "Event 1" in mutation_type:
            st.session_state.primary_sql_db[0]["body"] = "Guaranteed 99.99% high-availability uptime across multiple AWS availability zones."
            st.session_state.primary_sql_db[0]["version"] += 1
            st.success("🟢 Modified doc_101 in PostgreSQL! Version bumped.")
        else:
            new_id = f"doc_10{len(st.session_state.primary_sql_db)+1}"
            st.session_state.primary_sql_db.append({
                "id": new_id,
                "title": "Disaster Recovery SLA",
                "body": "RTO is strictly under 15 minutes with RPO under 1 hour.",
                "status": "active",
                "version": 1
            })
            st.success(f"🟢 Inserted new record {new_id} in PostgreSQL!")

st.divider()

st.subheader("Synchronizing ChromaDB with `collection.upsert()`")

upsert_payload = {
    "ids": [r["id"] for r in st.session_state.primary_sql_db],
    "documents": [f"{r['title']}: {r['body']}" for r in st.session_state.primary_sql_db],
    "metadatas": [{"status": r["status"], "version": r["version"]} for r in st.session_state.primary_sql_db]
}

st.caption("ChromaDB Ingestion Payload generated by sync worker:")
st.json(upsert_payload)

st.success("🟢 **Synchronization Completed Idempotently**: `collection.upsert()` seamlessly updated modified records and inserted new records without raising `DuplicateIDError`!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the fundamental difference between `collection.update()` and `collection.upsert()` in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **`collection.update()`** requires that every ID passed already exists in the collection. If any ID is missing, the operation raises `IDNotFoundError` and fails.
    * **`collection.upsert()`** does not care whether the ID exists. If missing, it creates a new record; if present, it replaces the existing record.
    * `upsert()` is the required primitive for streaming pipelines where the worker cannot guarantee whether an ID has been seen before.
    """)

with st.expander("Q2: How does ChromaDB handle re-indexing when an existing document is upserted with new text?"):
    st.markdown("""
    **Answer**:
    1. ChromaDB retrieves the internal integer index mapped to the string ID.
    2. It passes the revised document text to the bound `EmbeddingFunction` to compute new vector coordinates.
    3. In the HNSW index, the node's vector coordinates are updated, and its nearest-neighbor edges are recomputed.
    4. In SQLite, the raw text and metadata columns are updated within an atomic transaction.
    """)

with st.expander("Q3: Why is idempotency critical in message-queue-driven vector architectures?"):
    st.markdown("""
    **Answer**:
    * Distributed message brokers (Kafka, AWS SQS, RabbitMQ) provide **At-Least-Once Delivery** semantics.
    * This means a network retry can cause the exact same event message to be delivered to a consumer worker multiple times.
    * If the worker uses non-idempotent primitives (`add()`), the second delivery crashes the worker with a duplicate key exception.
    * Because `upsert()` is idempotent, duplicate event deliveries simply overwrite the database with the exact same values, causing zero errors and zero index corruption.
    """)

with st.expander("Q4: Can `upsert()` perform partial metadata updates without overwriting unmentioned keys?"):
    st.markdown("""
    **Answer**:
    **No**. In ChromaDB, `upsert()` replaces the entire metadata dictionary for that record.
    * If an existing document had `{"author": "Alice", "status": "draft"}` and you upsert with `metadatas=[{"status": "published"}]`, the `"author"` key is wiped out!
    * If you need to preserve existing metadata keys, you must read the existing metadata first (`collection.get()`), merge keys in Python, and then call `upsert()`.
    """)

with st.expander("Q5: What happens if an `upsert()` call contains duplicate IDs inside the same batch list?"):
    st.markdown("""
    **Answer**:
    * If `ids=["doc_1", "doc_1"]` appears within a single `upsert()` batch, ChromaDB processes the batch sequentially: the second occurrence overwrites the first.
    * However, best practice is to deduplicate the batch client-side before calling Chroma to eliminate redundant embedding calculations.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Pre-Flight `get(id)` Checks Before Ingestion**  
    Writing code like `if id in collection.get(): update() else: add()` introduces race conditions and doubles network round-trips. Just call `upsert()`.
    """)
    st.warning("""
    **Trap 2: Accidentally Wiping Out Existing Metadata Keys**  
    Forgetting that `upsert()` completely replaces the metadata object causes unmentioned fields (e.g. `created_at`) to be erased. Merge dictionaries upstream.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Re-Embedding Unchanged Text During Metadata Updates**  
    Using `upsert()` with `documents` when only metadata changed forces Chroma to re-calculate vectors on CPU/GPU. Use `collection.update(metadatas=...)` instead.
    """)
    st.warning("""
    **Trap 4: Missing Version Tracking in Distributed Sync**  
    Processing out-of-order Kafka messages without checking a `version_id` or timestamp can cause an older record revision to overwrite a newer one.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does upsert() do?"):
        st.markdown("""
        **Insert or Update**:  
        Inserts record if ID is missing; overwrites if ID already exists.
        """)
    with st.expander("🗂️ Flashcard: Why is upsert() idempotent?"):
        st.markdown("""
        **Deterministic State**:  
        Running the same upsert multiple times leaves the database in the exact same state.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Does upsert merge metadata?"):
        st.markdown("""
        **No**:  
        Upsert completely replaces the metadata object; does not deep merge keys.
        """)
    with st.expander("🗂️ Flashcard: When to use update() over upsert()?"):
        st.markdown("""
        **Metadata-Only Patches**:  
        Use `update()` without documents to patch metadata in <1ms without re-embedding.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the operational differences between `add()`, `update()`, and `upsert()`.", value=True)
chk2 = st.checkbox("I understand why `collection.upsert()` is the required primitive for Change Data Capture (CDC) pipelines.", value=True)
chk3 = st.checkbox("I know why idempotency is essential for handling At-Least-Once delivery in message queues.", value=True)
chk4 = st.checkbox("I understand that `upsert()` replaces the entire metadata dictionary rather than performing deep key merges.", value=True)
chk5 = st.checkbox("I know when to use `update()` without documents to avoid paying re-embedding costs.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 7: Querying ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 6: Adding Data to ChromaDB!")
