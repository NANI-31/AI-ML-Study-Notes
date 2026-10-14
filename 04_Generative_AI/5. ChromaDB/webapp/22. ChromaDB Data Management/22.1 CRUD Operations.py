import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="22.1 CRUD Operations - ChromaDB Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("22.1 CRUD Operations: Create, Read, Update, Delete & Upsert Mutation Semantics")
st.caption("Chapter 22: ChromaDB Data Management | Domain: 5. ChromaDB")
st.markdown("""
Unlike static search indexes, enterprise AI knowledge bases are dynamic living systems. 
Documentation is constantly revised, error codes are updated, security classifications change, 
and user-voted feedback alters document rankings.

In ChromaDB, modifications to existing records are handled via the **`collection.update()`** primitive:
* **The Strict Existence Invariant**: `update()` requires that every specified ID already exists in the collection. If any ID is missing, Chroma raises an error!
* **Targeted In-Place Metadata Updates**: Updating metadata alone mutates SQLite records in sub-millisecond time without triggering costly transformer re-embedding.
* **Text Mutation & Automatic Re-Embedding**: Updating document text automatically runs the collection's embedding function and re-links the vector in the HNSW graph.
* **Direct Vector Replacement**: Updating pre-computed embeddings directly replaces vector coordinates without model inference.

In this lesson, we break down the operational mechanics of `update()`, explore partial versus full mutations, 
implement clean update patterns in Python, and test an interactive Live Mutation Simulator.
""")

st.divider()

# Section 1: Anatomy & Contract of collection.update()
st.header("1. Anatomy & Contract of collection.update()", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### Method Signature & Behavior
    
    ```python
    collection.update(
        ids: List[str],                     # REQUIRED: Must all exist!
        documents: Optional[List[str]],     # Optional: Triggers re-embedding
        metadatas: Optional[List[dict]],    # Optional: Merges / replaces metadata
        embeddings: Optional[List[list]],   # Optional: Replaces vector coords
        images: Optional[List[any]]         # Optional: Multi-modal updates
    )
    ```
    
    * **The Strict ID Contract**:
      * If `ids=["doc_1", "doc_missing"]` is passed and `"doc_missing"` does not exist, the entire update transaction aborts with an exception.
    
    * **Selective Field Mutation**:
      * You only need to pass the fields you wish to change.
      * Passing only `metadatas` leaves `documents` and vector embeddings untouched!
    """)

with col_anat2:
    st.markdown("### Update Execution Path Topology")
    update_topo_dot = """
    digraph UpdatePath {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        CallUpdate [label="collection.update(ids=...)\\nPayload with selective fields", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        CheckBranch [label="Does payload include\\nnew 'documents'?", shape=diamond, fillcolor="#FEF3C7"];

        FastMeta [label="Fast SQLite Update Path\\nMutate metadata columns only\\nLatency: <0.5ms (Zero AI compute)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        FullEmbed [label="Full Embedding & Graph Re-Index\\n1. Invoke Embedding Function\\n2. Re-wire HNSW Graph Nodes\\nLatency: 10ms - 50ms", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        CallUpdate -> CheckBranch;
        CheckBranch -> FastMeta [label="No (Metadata Only)"];
        CheckBranch -> FullEmbed [label="Yes (Text Changed)"];
    }
    """
    st.graphviz_chart(update_topo_dot, use_container_width=False)

st.divider()

# Section 2: Partial Updates vs Complete Replacement
st.header("2. Partial Updates vs Complete Replacement", divider="rainbow")

col_part1, col_part2 = col_part1, col_part2 = st.columns([1, 1])

with col_part1:
    st.markdown("""
    ### Metadata Replacement Mechanics
    * **Important ChromaDB Invariant**:
      * When you pass `metadatas=[{"status": "published"}]` for an ID that previously had `{"status": "draft", "author": "Alice"}`, Chroma **replaces** the metadata dictionary rather than performing a deep key-wise merge!
      * If you want to keep `"author": "Alice"`, your application must read the existing metadata first, merge the keys, and submit the full combined dictionary.
    
    * **Document Text Replacement**:
      * Updating a document's text replaces the stored raw text string in SQLite and recalculates its dense vector representation.
    """)

with col_part2:
    st.markdown("### Clean Python Merge Pattern")
    st.code("""# Recommended Pattern for Safe Metadata Merging
def safe_metadata_patch(collection, doc_id: str, new_fields: dict):
    # 1. Fetch existing metadata via get()
    current = collection.get(ids=[doc_id], include=["metadatas"])
    if not current["ids"]:
        raise ValueError(f"Document {doc_id} does not exist!")
        
    existing_meta = current["metadatas"][0] or {}
    
    # 2. Merge existing fields with new fields
    merged_meta = {**existing_meta, **new_fields}
    
    # 3. Apply atomic update
    collection.update(
        ids=[doc_id],
        metadatas=[merged_meta]
    )
    return merged_meta
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Update Operations
st.header("3. Clean Python Implementation of Update Operations", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="kb_runbooks",
    metadata={"hnsw:space": "cosine"}
)

# 1. Targeted Metadata-Only Update (Sub-millisecond)
# Promote a document to production and increment review counter
collection.update(
    ids=["runbook_pg_01"],
    metadatas=[{
        "env": "production",
        "review_count": 5,
        "is_verified": True
    }]
)
print("Metadata updated in SQLite without re-embedding.")

# 2. Text Update with Automatic Re-Embedding
# Update runbook content with revised CLI commands
collection.update(
    ids=["runbook_pg_01"],
    documents=["PostgreSQL 16 failover procedure using patroni clusterctl failover."]
)
print("Text updated; Chroma automatically generated new embeddings and updated HNSW!")

# 3. Batch Update across multiple documents
collection.update(
    ids=["runbook_pg_01", "runbook_k8s_02"],
    metadatas=[
        {"tier": "tier_1", "last_audited": 2024},
        {"tier": "tier_2", "last_audited": 2024}
    ]
)
print("Batch update complete.")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Mutation Simulator
st.header("4. Interactive Workbench: Live Mutation Simulator", divider="rainbow")

st.markdown("""
Experiment with mutating documents interactively. Choose what fields to update (text, metadata status, review score), 
and inspect how ChromaDB processes the update path and manages latency.
""")

col_wb_s1, col_wb_s2 = col_wb_s2 = st.columns(2)

with col_wb_s1:
    st.subheader("1. Existing Document State")
    doc_id_target = st.selectbox("Target Document ID:", ["runbook_001", "runbook_002", "runbook_NON_EXISTENT"])
    st.caption("Initial Document: 'PostgreSQL 14 replica replication recovery procedures.'")
    st.caption("Initial Metadata: `{'env': 'staging', 'score': 3.5, 'verified': False}`")

with col_wb_s2:
    st.subheader("2. Mutation Fields to Submit")
    update_text = st.checkbox("Mutate Document Text (Triggers Re-Embedding)", value=False)
    new_text_content = ""
    if update_text:
        new_text_content = st.text_area("New Document Text:", value="PostgreSQL 16 patroni high-availability failover guide.")
        
    new_env = st.selectbox("New Environment Metadata:", ["staging", "production"])
    new_score = st.slider("New Quality Score Metadata:", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
    new_verified = st.checkbox("Mark Verified by Human", value=True)

st.divider()

col_mut_res, col_mut_prof = col_mut_res, col_mut_prof = st.columns([1.1, 0.9])

with col_mut_res:
    st.subheader("Update Execution Result")
    
    if doc_id_target == "runbook_NON_EXISTENT":
        st.error("🔴 **UPDATE FAILED (ValueError)**: Document ID 'runbook_NON_EXISTENT' was not found in the collection! In ChromaDB, `update()` strictly requires pre-existing IDs. Use `upsert()` if ID existence is uncertain.")
    else:
        st.success(f"🟢 **UPDATE SUCCESSFUL**: Record '{doc_id_target}' mutated successfully!")
        
        updated_view = {
            "id": doc_id_target,
            "document": new_text_content if update_text else "PostgreSQL 14 replica replication recovery procedures. (Unchanged)",
            "metadata": {
                "env": new_env,
                "score": new_score,
                "verified": new_verified
            }
        }
        st.json(updated_view)

with col_mut_prof:
    st.subheader("Execution Path Profiler")
    if doc_id_target != "runbook_NON_EXISTENT":
        if update_text:
            st.warning("""
            **Heavy Re-Embedding Path**:
            * Invoked Embedding Model (SentenceTransformers / OpenAI API).
            * Calculated 384d / 1536d vector coordinates.
            * Rewired HNSW graph edges.
            * Updated SQLite document text and metadata.
            * **Estimated Latency**: ~28.5 ms
            """)
        else:
            st.info("""
            **High-Speed Metadata-Only Path**:
            * Bypassed Embedding Model entirely.
            * Zero vector math evaluated.
            * Direct SQLite `UPDATE metadatas SET ... WHERE id = ...`
            * **Estimated Latency**: ~0.35 ms
            """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What happens if you call `collection.update()` with an ID that does not exist in the collection?"):
    st.markdown("""
    **Answer**:
    * Unlike `collection.get()` (which silently returns empty lists) or `collection.upsert()` (which creates the missing record), `collection.update()` **raises an exception** (`ValueError` or `KeyError`).
    * The entire batch update fails atomically.
    * If you are unsure whether an ID exists, use **`collection.upsert()`** instead.
    """)

with st.expander("Q2: Does updating metadata in ChromaDB perform a shallow merge or full dictionary replacement?"):
    st.markdown("""
    **Answer**:
    * In standard ChromaDB, providing a metadata dictionary in `update()` **replaces** the existing metadata dictionary for that document with the new one provided.
    * It does NOT perform a deep field-by-field merge. If you provide `{"verified": True}`, any prior keys (like `"author": "Alice"`) will be lost unless you read and merge them prior to calling `update()`.
    """)

with st.expander("Q3: How does ChromaDB handle vector indexing when document text is updated?"):
    st.markdown("""
    **Answer**:
    * When `documents` is passed in `collection.update()`:
      1. ChromaDB runs the text through the collection's configured `embedding_function` to produce a new dense vector.
      2. It updates the raw text record in SQLite.
      3. It updates the vector entry in the underlying `hnswlib` index, adjusting neighborhood graph connectivity for the node.
    """)

with st.expander("Q4: Can you update pre-computed embeddings directly without providing document text?"):
    st.markdown("""
    **Answer**:
    * **Yes**. Call `collection.update(ids=[...], embeddings=[[...]])`.
    * This directly overwrites the vector coordinates in the HNSW index.
    * Useful when embeddings are calculated externally in asynchronous batch workers (e.g. GPU clusters) and streamed into ChromaDB.
    """)

with st.expander("Q5: What is the primary performance optimization when updating document metadata?"):
    st.markdown("""
    **Answer**:
    * **Omit the `documents` argument**.
    * If you only need to change a status, review count, or tag, passing only `ids` and `metadatas` restricts the update to SQLite relational columns.
    * This runs in sub-millisecond time and completely avoids the high CPU/GPU cost of re-embedding text.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Accidental Metadata Key Wiping**  
    Passing `metadatas=[{"status": "published"}]` erases all other metadata keys. Read existing metadata and merge before updating.
    """)
    st.warning("""
    **Trap 2: Using `update()` on Unverified ID Streams**  
    Calling `update()` on a webhook stream crashes if an ID doesn't exist yet. Use `upsert()` for idempotent ingestion.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Re-Passing Document Text During Metadata Edits**  
    Passing text when only updating tags forces Chroma to recalculate embeddings, wasting compute and API quota.
    """)
    st.warning("""
    **Trap 4: Mismatched Batch Array Lengths**  
    Passing 3 IDs but 2 metadatas raises a validation error. Parallel lists must have identical lengths.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What if an ID is missing in update()?"):
        st.markdown("""
        **Raises Error**:  
        `update()` requires pre-existing IDs. Use `upsert()` if uncertain.
        """)
    with st.expander("🗂️ Flashcard: How to update metadata without re-embedding?"):
        st.markdown("""
        **Omit documents**:  
        Pass only `ids` and `metadatas` to execute in <0.5ms in SQLite.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Does metadata do deep merge?"):
        st.markdown("""
        **No, Replacement**:  
        Replaces the dictionary. Merge manually before calling update.
        """)
    with st.expander("🗂️ Flashcard: Can you update raw embeddings?"):
        st.markdown("""
        **Yes**:  
        Pass `embeddings=[[...]]` to overwrite vectors directly.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the strict pre-existing ID contract of `collection.update()`.", value=True)
chk2 = st.checkbox("I know how to execute sub-millisecond metadata-only updates without re-embedding costs.", value=True)
chk3 = st.checkbox("I understand how updating document text automatically triggers re-embedding and HNSW graph re-indexing.", value=True)
chk4 = st.checkbox("I know how to safely merge existing metadata fields prior to updating to prevent accidental key wiping.", value=True)
chk5 = st.checkbox("I understand how to update raw pre-computed embeddings directly.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 22.2: Data Lifecycle in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 22.1: CRUD Operations and are ready for Chapter 22.2: Data Lifecycle!")
