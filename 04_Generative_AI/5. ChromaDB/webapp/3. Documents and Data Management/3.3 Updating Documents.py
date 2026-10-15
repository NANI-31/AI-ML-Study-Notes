import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="3.3 Updating Documents - ChromaDB Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("3.3 Updating Documents: Mutation Semantics, Metadata Patches, Upsert Workflows & Duplicate IDs")
st.caption("Chapter 3: Documents and Data Management | Domain: 5. ChromaDB")
st.markdown("""
In real-world production systems, data is rarely static. Product catalogs change prices, engineering runbooks 
get revised, compliance permissions expire, and embedding models get fine-tuned.

Keeping a vector database in sync with real-world changes requires mastering **ChromaDB's Mutation Semantics**:
* **Partial Metadata Patching**: Modifying scalar tags (`status: "archived"`, `views: 1200`) without paying the expensive CPU cost of re-embedding the raw text.
* **Text Mutation & Auto-Re-Embedding**: Updating document text strings, which automatically triggers embedding recalculation and moves the node in HNSW metric space.
* **Atomic Upsert Synchronization**: Using `collection.upsert()` as the engine for Change Data Capture (CDC) to eliminate duplicate ID errors permanently.
* **Duplicate ID Handling**: Understanding why duplicate IDs cause fatal exceptions during `add()` and how to handle them deterministically.

In this lesson, we explore update vs upsert mechanics, build a Change Data Capture sync pipeline in Python, 
analyze HNSW graph node movement, and test an interactive Document Update & State Simulator.
""")

st.divider()

# Section 1: The Three Update Semantics
st.header("1. The Three Update Semantics", divider="rainbow")

col_sem1, col_sem2 = col_sem1, col_sem2 = st.columns([1.1, 0.9])

with col_sem1:
    st.markdown("""
    ### Three Mutation Scenarios
    
    * **1. Metadata-Only Update (Zero Re-Embedding Cost)**:
      * *Call*: `collection.update(ids=["doc_1"], metadatas=[{"status": "archived"}])`
      * *What Happens*: Chroma updates only the SQLite metadata row. The vector embedding and HNSW graph node remain completely untouched.
      * *Performance*: Ultra-fast (<1 millisecond). Zero CPU matrix math!
    
    * **2. Text Mutation with Automatic Re-Embedding**:
      * *Call*: `collection.update(ids=["doc_1"], documents=["New revised text..."])`
      * *What Happens*: Chroma calls its embedding function to generate a new vector. The existing node in the HNSW graph is updated to the new coordinates, and SQLite text is overwritten.
      * *Performance*: Requires ONNX embedding computation (~4ms per chunk).
    
    * **3. Explicit Embedding Replacement**:
      * *Call*: `collection.update(ids=["doc_1"], embeddings=[new_vector])`
      * *What Happens*: Replaces the geometric vector coordinates with external pre-computed floats without modifying the document text.
    """)

with col_sem2:
    st.markdown("### Update Execution Path Flowchart")
    update_dot = """
    digraph UpdatePathways {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InRequest [label="Incoming Update Request\\ncollection.update(id='doc_1', ...)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QType [label="Which fields are\\nbeing updated?", shape=diamond, fillcolor="#FEF3C7"];

        MetaOnly [label="Scenario 1: Metadata Only\\nUpdates SQLite row\\nZero Re-Embedding Cost!", fillcolor="#D1FAE5", color="#059669"];

        DocOnly [label="Scenario 2: Document Text\\nRe-runs Embedding Model\\nMoves HNSW Node Coordinates", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        InRequest -> QType;
        QType -> MetaOnly [label="metadatas provided\\ndocuments omitted"];
        QType -> DocOnly [label="documents provided"];
    }
    """
    st.graphviz_chart(update_dot, use_container_width=False)

st.divider()

# Section 2: Change Data Capture (CDC) with `collection.upsert()`
st.header("2. Change Data Capture (CDC) with collection.upsert()", divider="rainbow")

col_cdc1, col_cdc2 = col_cdc1, col_cdc2 = st.columns([1, 1])

with col_cdc1:
    st.markdown("""
    ### Why `upsert()` is the Backbone of Production CDC
    * In production architectures, an ETL worker syncs records from PostgreSQL or MongoDB into ChromaDB on a recurring schedule.
    * With `add()`, attempting to sync records that already exist throws `DuplicateIDError`, crashing the worker job.
    * With `update()`, attempting to sync newly created records throws `IDNotFoundError`.
    * **`upsert()` Solves Both**:
      * If ID does not exist -> **Inserts as a new record**.
      * If ID exists -> **Overwrites document, metadata, and embedding**.
      * Guarantees that the vector store converges to the exact state of the upstream database without race conditions!
    """)

with col_cdc2:
    st.markdown("### Production CDC Sync Pattern")
    st.code("""import chromadb
from typing import List, Dict, Any

def sync_upstream_records_to_chroma(
    collection,
    upstream_records: List[Dict[str, Any]]
):
    \"\"\"Idempotently syncs external database rows to ChromaDB.\"\"\"
    batch_ids = [r["id"] for r in upstream_records]
    batch_docs = [r["content"] for r in upstream_records]
    batch_metas = [
        {
            "updated_at": r["updated_at"],
            "category": r["category"],
            "version": r["version"]
        }
        for r in upstream_records
    ]

    # upsert handles both newly inserted and modified records seamlessly
    collection.upsert(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_metas
    )
    print(f"Synced {len(batch_ids)} records into Chroma.")
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Mutation Workflows
st.header("3. Clean Python Implementation of Mutation Workflows", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("customer_kb")

# 1. Initial Ingestion
collection.add(
    ids=["cust_42"],
    documents=["Standard customer inquiry regarding product shipping timelines."],
    metadatas=[{"tier": "silver", "status": "open", "assigned_agent": "unassigned"}]
)

# 2. Metadata Patch (Re-assignment): Leaves embedding vector untouched
collection.update(
    ids=["cust_42"],
    metadatas=[{"tier": "gold", "status": "in_progress", "assigned_agent": "agent_sarah"}]
)
print("Updated metadata without re-embedding text.")

# 3. Document Content Revision: Recalculates embedding automatically
collection.update(
    ids=["cust_42"],
    documents=["High-priority enterprise customer requesting express overnight freight delivery."]
)
print("Updated document text and recalculated HNSW vector embedding.")

# 4. Atomic Upsert: Handles newly created cust_43 and updates existing cust_42
collection.upsert(
    ids=["cust_42", "cust_43"],
    documents=[
        "High-priority enterprise customer requesting express overnight freight delivery.",
        "New customer inquiring about billing invoice cycles."
    ],
    metadatas=[
        {"tier": "platinum", "status": "resolved", "assigned_agent": "agent_sarah"},
        {"tier": "bronze", "status": "open", "assigned_agent": "agent_mike"}
    ]
)
print(f"Total documents after upsert: {collection.count()}")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Document Update & Upsert Simulator
st.header("4. Interactive Workbench: Document Update & Upsert Simulator", divider="rainbow")

st.markdown("""
Experience the state transitions of ChromaDB records. Choose an operation to perform on a simulated 
record and observe how metadata, document text, and vector coordinates update.
""")

if "live_sim_record" not in st.session_state:
    st.session_state.live_sim_record = {
        "id": "KB-NET-01",
        "document": "Kubernetes Ingress controllers utilize Envoy proxy for TLS termination.",
        "metadata": {"tier": "internal", "version": 1, "status": "draft"},
        "vector_preview": [0.12, -0.45, 0.88, 0.05]
    }

col_sim_ctrl, col_sim_view = col_sim_ctrl, col_sim_view = st.columns([1, 1.2])

with col_sim_ctrl:
    st.subheader("1. Mutation Action")
    sim_action = st.radio(
        "Choose Action to Apply:",
        [
            "Action A: Patch Metadata Only (Zero Re-Embedding)",
            "Action B: Revise Document Text (Trigger Re-Embedding)",
            "Action C: Upsert with Complete Overwrite"
        ]
    )
    
    if "Action A" in sim_action:
        new_status = st.selectbox("New Status Value:", ["in_review", "approved", "archived"])
        new_version = st.number_input("Bump Version Number:", value=st.session_state.live_sim_record["metadata"]["version"] + 1)
    elif "Action B" in sim_action:
        revised_text = st.text_area("Revised Document Text:", value="Kubernetes Ingress utilizes Envoy proxy for mutual TLS (mTLS) with traffic splitting.")
    else:  # Action C
        upsert_text = st.text_area("Upsert Document Text:", value="Production Envoy proxy architecture with dynamic rate limiting.")
        upsert_status = st.selectbox("Upsert Status:", ["approved", "active"])

with col_sim_view:
    st.subheader("2. Apply Mutation")
    
    if st.button("Apply Mutation to Record", use_container_width=True):
        rec = st.session_state.live_sim_record
        if "Action A" in sim_action:
            rec["metadata"]["status"] = new_status
            rec["metadata"]["version"] = int(new_version)
            st.success("🟢 Metadata patched successfully! Vector embedding remained untouched.")
        elif "Action B" in sim_action:
            rec["document"] = revised_text
            # Simulate new embedding coordinates
            rec["vector_preview"] = [round(v + np.random.uniform(-0.1, 0.1), 4) for v in rec["vector_preview"]]
            st.success("🟢 Document revised! Embedding coordinates recalculated in HNSW graph.")
        else:
            rec["document"] = upsert_text
            rec["metadata"]["status"] = upsert_status
            rec["metadata"]["version"] += 1
            rec["vector_preview"] = [round(v + np.random.uniform(-0.2, 0.2), 4) for v in rec["vector_preview"]]
            st.success("🟢 Record upserted atomically!")

    st.subheader("Current Database Record State")
    st.json(st.session_state.live_sim_record)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the computational cost difference between updating metadata vs updating document text?"):
    st.markdown("""
    **Answer**:
    * **Updating Metadata (`collection.update(ids=..., metadatas=...)`)**:
      * Incurs **zero embedding computation cost**.
      * Executes a simple SQLite `UPDATE` statement on the metadata relational table. Takes sub-millisecond execution time (`<0.5ms`).
    * **Updating Document Text (`collection.update(ids=..., documents=...)`)**:
      * Incurs **high computational cost**.
      * ChromaDB must pass the new text string through the embedding model (ONNX CPU or external GPU/API).
      * The resulting vector must be updated in the HNSW graph, requiring neighbor edge re-evaluation. Takes ~3ms–15ms per chunk.
    """)

with st.expander("Q2: Why does `collection.add()` raise an error on duplicate IDs, while `collection.upsert()` does not?"):
    st.markdown("""
    **Answer**:
    * **`collection.add()`** is designed for strict insertion semantics. It validates that every ID is unique. If an ID collision occurs, it fails fast (`DuplicateIDError`) to prevent accidental overwrites of existing enterprise data.
    * **`collection.upsert()`** implements **idempotent merge semantics**. If an ID already exists, it treats the call as an update; if the ID is missing, it inserts the record. It is mathematically idempotent: running it multiple times leaves the database in the exact same state.
    """)

with st.expander("Q3: How does ChromaDB handle HNSW graph node updates when a vector is replaced?"):
    st.markdown("""
    **Answer**:
    * In standard HNSW graph indexing, directly moving a node's coordinates in high-dimensional space can invalidate existing geometric proximity edges.
    * When an embedding is replaced for an existing ID, ChromaDB:
      1. Marks the old node or removes its links in the HNSW graph.
      2. Inserts the new vector coordinates as an updated node into the graph layers.
      3. Updates the mapping table connecting the record ID to the internal integer node index.
    """)

with st.expander("Q4: Can you update a document's ID directly in ChromaDB?"):
    st.markdown("""
    **Answer**:
    **No**. Primary key IDs are immutable in ChromaDB:
    * There is no `collection.update_id(old_id, new_id)` API.
    * To change an ID, you must fetch the record's content and metadata using `collection.get(ids=[old_id])`, insert the record under `new_id`, and delete the record under `old_id`.
    """)

with st.expander("Q5: What happens if an `upsert()` batch contains duplicate IDs within the same batch list?"):
    st.markdown("""
    **Answer**:
    * If `ids=["doc_1", "doc_1"]` appears inside a single `upsert()` call, the behavior is deterministic: the last item in the batch overwrites the earlier item.
    * However, best practice is to deduplicate batch lists at the client layer before calling Chroma to avoid wasting compute.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Document Strings During Metadata-Only Updates**  
    Passing `documents` when you only meant to change a status tag forces Chroma to re-embed the entire text on CPU/GPU, wasting 95% of compute! Omit `documents`.
    """)
    st.warning("""
    **Trap 2: Using `add()` in Continuous Data Pipelines**  
    Using `add()` in cron jobs causes pipeline crashes whenever an already-indexed document is encountered. Always use `upsert()` for CDC pipelines.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Partial Metadata Replacement (Overwriting vs Merging)**  
    In ChromaDB, passing a new metadata dictionary replaces the entire metadata object for that record! Ensure you preserve existing keys if merging is desired.
    """)
    st.warning("""
    **Trap 4: Forgetting to Update Parent Document References**  
    Updating a child chunk without updating its parent document reference causes desynchronization between vector search hits and parent context hydration.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: When does re-embedding occur?"):
        st.markdown("""
        **Only on Text Update**:  
        Updating `documents` recalculates vectors; updating only `metadatas` skips embedding entirely.
        """)
    with st.expander("🗂️ Flashcard: What makes upsert() essential?"):
        st.markdown("""
        **Idempotent CDC**:  
        Inserts new documents and updates modified ones without crashing on duplicate IDs.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Can IDs be renamed?"):
        st.markdown("""
        **Immutable Primary Key**:  
        IDs cannot be renamed; must re-insert with new ID and delete old ID.
        """)
    with st.expander("🗂️ Flashcard: How to patch metadata cheaply?"):
        st.markdown("""
        **Omit Documents**:  
        Call `collection.update(ids=[...], metadatas=[...])` without passing text to execute in <1ms.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the three update scenarios: metadata-only, text revision, and vector replacement.", value=True)
chk2 = st.checkbox("I know why metadata-only updates incur zero re-embedding cost and execute in <1ms.", value=True)
chk3 = st.checkbox("I understand why `collection.upsert()` is the golden standard for Change Data Capture (CDC) pipelines.", value=True)
chk4 = st.checkbox("I know how duplicate IDs are handled by `add()` vs `upsert()`.", value=True)
chk5 = st.checkbox("I understand that metadata updates replace the entire dictionary rather than performing deep key merges.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 4: Embeddings and Vector Representations in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 3: Documents and Data Management in ChromaDB!")
