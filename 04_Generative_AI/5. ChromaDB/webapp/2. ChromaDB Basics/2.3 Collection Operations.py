import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="2.3 Collection Operations - ChromaDB Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("2.3 Collection Operations: Adding, Updating, Upserting, Deleting, Counting & Inspection")
st.caption("Chapter 2: ChromaDB Basics | Domain: 5. ChromaDB")
st.markdown("""
Once a collection is created, managing its data lifecycle requires understanding ChromaDB's 
**Core Mutation (CRUD) and Inspection Operations**.

In traditional SQL, developers rely on `INSERT`, `UPDATE`, and `ON CONFLICT DO UPDATE`. 
In ChromaDB, these operations map to four distinct methods:
* **`add()`**: Inserts new records; raises an error if an ID already exists.
* **`update()`**: Updates existing records; raises an error if an ID is missing.
* **`upsert()`**: **The production gold standard**; atomically inserts missing records and updates existing ones.
* **`delete()`**: Removes records by explicit ID lists, metadata predicates, or document text content.

Alongside mutations, Chroma provides inspection primitives: `count()`, `get()`, `peek()`, and `modify()`.

In this lesson, we break down each mutation method, analyze error handling and atomicity, 
inspect collection telemetry, and test an interactive Live Collection CRUD Workbench.
""")

st.divider()

# Section 1: The Core Mutation Operations (add, update, upsert, delete)
st.header("1. The Core Mutation Operations (CRUD)", divider="rainbow")

col_crud1, col_crud2 = col_crud1, col_crud2 = st.columns([1.1, 0.9])

with col_crud1:
    st.markdown("""
    ### Mutation Method Comparison
    
    * **1. `collection.add(ids, documents, ...)`**:
      * *Behavior*: Pure insertion.
      * *Exception*: If any ID in the batch already exists in the collection, the entire call throws `DuplicateIDError`.
      * *Best For*: Initial batch ingestion where duplicate IDs should signal an upstream data pipeline bug.
    
    * **2. `collection.update(ids, documents, ...)`**:
      * *Behavior*: Pure mutation.
      * *Exception*: If any ID does not exist, throws `IDNotFoundError`.
      * *Best For*: Modifying metadata tags or document text for known existing entities.
    
    * **3. `collection.upsert(ids, documents, ...)`**:
      * *Behavior*: Insert or Update (Merge).
      * *Exception*: Idempotent. Never throws duplicate or not-found errors.
      * *Best For*: Continuous CDC (Change Data Capture) pipelines and recurring web scrapers.
    
    * **4. `collection.delete(ids, where, where_document)`**:
      * *Behavior*: Removes nodes from HNSW index and purges rows from SQLite.
      * *Flexibility*: Can delete by ID (`ids=["doc_1"]`) or by metadata predicate (`where={"status": "deprecated"}`).
    """)

with col_crud2:
    st.markdown("### Mutation State Transition Matrix")
    crud_dot = """
    digraph CrudState {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        IDState [label="Incoming Document Batch\\n(e.g., id='doc_42')", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QExists [label="Does 'doc_42'\\nalready exist in DB?", shape=diamond, fillcolor="#FEF3C7"];

        AddSucceed [label="add() SUCCEEDS\\n(Inserted)", fillcolor="#D1FAE5", color="#059669"];
        AddFails [label="add() THROWS ERROR\\n(DuplicateIDError)", fillcolor="#FEE2E2", color="#DC2626"];

        UpdateSucceed [label="update() SUCCEEDS\\n(Modified)", fillcolor="#D1FAE5", color="#059669"];
        UpdateFails [label="update() THROWS ERROR\\n(IDNotFoundError)", fillcolor="#FEE2E2", color="#DC2626"];

        UpsertSucceed [label="upsert() ALWAYS SUCCEEDS!\\n(Inserts or Overwrites)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        IDState -> QExists;
        QExists -> AddSucceed [label="No"];
        QExists -> AddFails [label="Yes"];
        QExists -> UpdateFails [label="No"];
        QExists -> UpdateSucceed [label="Yes"];
        QExists -> UpsertSucceed [label="Both Cases"];
    }
    """
    st.graphviz_chart(crud_dot, use_container_width=False)

st.divider()

# Section 2: Collection Inspection & Telemetry Operations
st.header("2. Collection Inspection & Telemetry Operations", divider="rainbow")

col_insp1, col_insp2 = col_insp1, col_insp2 = st.columns([1, 1])

with col_insp1:
    st.markdown("""
    ### 1. `collection.count()`
    * Returns the total integer count of documents currently stored in the collection.
    * Ultra-fast: reads row count directly from SQLite metadata in **<0.1ms**.
    ```python
    total_docs = collection.count()
    print(f"Total documents indexed: {total_docs}")
    ```
    
    ### 2. `collection.peek(limit=10)`
    * Returns the first N items in the collection.
    * Crucial for debugging during interactive exploration.
    ```python
    preview = collection.peek(limit=5)
    print("Sample IDs:", preview["ids"])
    ```
    """)

with col_insp2:
    st.markdown("""
    ### 3. `collection.get(...)`
    * Retrieves records **without performing vector similarity calculations**.
    * Supports filtering, paging (`limit`, `offset`), and selective inclusion:
    ```python
    # Fetch specific documents by ID
    records = collection.get(
        ids=["doc_101", "doc_102"],
        include=["documents", "metadatas"] # Exclude heavy embeddings
    )

    # Fetch all records matching metadata filter
    finance_docs = collection.get(
        where={"department": "finance"},
        limit=50,
        offset=0
    )
    ```
    
    ### 4. `collection.modify(name, metadata)`
    * Renames a collection or updates its collection-level metadata.
    """)

st.divider()

# Section 3: Clean Python Implementation of Collection Operations
st.header("3. Clean Python Implementation of Collection Operations", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("customer_feedback_kb")

# 1. ADD: Pure insertion
collection.add(
    ids=["fb_01", "fb_02"],
    documents=["The dashboard latency is extremely fast.", "Exporting to PDF failed with error 500."],
    metadatas=[{"category": "praise", "resolved": True}, {"category": "bug", "resolved": False}]
)

# 2. UPDATE: Modifying an existing record
collection.update(
    ids=["fb_02"],
    metadatas=[{"category": "bug", "resolved": True}]  # Bug marked as resolved!
)

# 3. UPSERT: Idempotent merge (Adds fb_03, updates fb_01)
collection.upsert(
    ids=["fb_01", "fb_03"],
    documents=["The dashboard latency is blazingly fast!", "Need dark mode theme support."],
    metadatas=[{"category": "praise", "resolved": True}, {"category": "feature_req", "resolved": False}]
)

# 4. COUNT: Inspect total
print("Total records:", collection.count())

# 5. GET: Retrieve by metadata
unresolved_bugs = collection.get(
    where={"$and": [{"category": "bug"}, {"resolved": False}]}
)

# 6. DELETE: Remove resolved praise
collection.delete(
    where={"category": "praise"}
)
print("Count after deletion:", collection.count())
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Collection CRUD Simulator
st.header("4. Interactive Workbench: Live Collection CRUD Simulator", divider="rainbow")

st.markdown("""
Interact with a simulated in-memory collection. Choose a mutation operation, configure inputs, 
and observe how the collection state updates while testing Chroma's error handling.
""")

# Initialize session state store
if "sim_chroma_store" not in st.session_state:
    st.session_state.sim_chroma_store = {
        "DOC-101": {"text": "Kubernetes Ingress routes traffic via Envoy proxy.", "dept": "Infra", "status": "active"},
        "DOC-102": {"text": "PostgreSQL connection pooling is managed with PgBouncer.", "dept": "Database", "status": "active"},
        "DOC-103": {"text": "Redis cluster acts as semantic cache for sub-millisecond retrieval.", "dept": "Infra", "status": "deprecated"}
    }

col_crud_act, col_crud_inputs = col_crud_act, col_crud_inputs = st.columns([1, 1.2])

with col_crud_act:
    st.subheader("1. Select Mutation Operation")
    op_type = st.radio(
        "Choose Operation:",
        [
            "add() - Insert New Document",
            "update() - Mutate Existing Document",
            "upsert() - Idempotent Insert or Overwrite",
            "delete() - Delete by ID"
        ]
    )

with col_crud_inputs:
    st.subheader("2. Operation Parameters")
    target_id = st.text_input("Target Document ID (`id`):", value="DOC-101")
    
    if "delete" not in op_type:
        in_text = st.text_area("Document Text (`document`):", value="Updated description of cluster architecture.", height=70)
        in_dept = st.selectbox("Department (`dept`):", ["Infra", "Database", "Security", "AI Platforms"])
        in_status = st.selectbox("Status (`status`):", ["active", "deprecated"])

st.divider()

# Execution handler
col_exec_btn, col_table_view = col_exec_btn, col_table_view = st.columns([1, 1.5])

with col_exec_btn:
    st.subheader("Execute Mutation")
    
    if st.button("Run Operation on Collection", use_container_width=True):
        store = st.session_state.sim_chroma_store
        
        if "add()" in op_type:
            if target_id in store:
                st.error(f"🔴 `DuplicateIDError`: Document with ID '{target_id}' already exists! Use `upsert()` or `update()`.")
            else:
                store[target_id] = {"text": in_text, "dept": in_dept, "status": in_status}
                st.success(f"🟢 Successfully added `{target_id}` to collection!")
                
        elif "update()" in op_type:
            if target_id not in store:
                st.error(f"🔴 `IDNotFoundError`: Cannot update '{target_id}' because it does not exist! Use `upsert()`.")
            else:
                store[target_id] = {"text": in_text, "dept": in_dept, "status": in_status}
                st.success(f"🟢 Successfully updated `{target_id}` in collection!")
                
        elif "upsert()" in op_type:
            store[target_id] = {"text": in_text, "dept": in_dept, "status": in_status}
            st.success(f"🟢 Successfully upserted `{target_id}` (atomic merge)!")
            
        elif "delete()" in op_type:
            if target_id in store:
                del store[target_id]
                st.success(f"🗑️ Successfully deleted `{target_id}` from collection!")
            else:
                st.warning(f"ID '{target_id}' was not found in collection. Zero records deleted.")

    st.metric(label="Current Collection Count (`collection.count()`)", value=len(st.session_state.sim_chroma_store))

with col_table_view:
    st.subheader("Live Collection Records (`collection.peek()`)")
    
    if st.session_state.sim_chroma_store:
        table_rows = []
        for d_id, data in st.session_state.sim_chroma_store.items():
            table_rows.append({
                "ID": d_id,
                "Document Text": data["text"],
                "Department": data["dept"],
                "Status": data["status"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("The collection is currently empty.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is `collection.upsert()` preferred over `collection.add()` in production ETL pipelines?"):
    st.markdown("""
    **Answer**:
    * In production data ingestion pipelines (e.g. crawling websites, syncing Notion pages, or polling databases), jobs frequently re-run due to network retries or scheduled cron executions.
    * If you use `add()`, any retry or partial pipeline failure crashes with a fatal `DuplicateIDError`.
    * `upsert()` is **idempotent**: running the pipeline 1 time or 100 times produces the exact same deterministic state without errors.
    * If an existing document's content changed, `upsert()` updates the text and recalculates its vector embedding automatically.
    """)

with st.expander("Q2: What is the performance difference between `collection.query()` and `collection.get()`?"):
    st.markdown("""
    **Answer**:
    * **`collection.query()`**:
      * Executes **vector similarity search**.
      * Generates a query embedding, navigates the multi-layer HNSW graph in memory, computes floating point distance calculations, and ranks results. (Takes 2ms–15ms).
    * **`collection.get()`**:
      * Executes a **direct relational key-value / SQL lookup**.
      * Bypasses the HNSW graph entirely. Fetches records strictly by primary key ID or metadata filter from SQLite. (Takes <0.5ms).
      * Used when you know the exact ID or want to fetch records by attribute without semantic ranking.
    """)

with st.expander("Q3: Why should you exclude embeddings (`include=['documents', 'metadatas']`) when calling `get()` or `query()`?"):
    st.markdown("""
    **Answer**:
    * Floating point vector arrays are heavy: a 1,536-dimensional embedding consists of 1,536 32-bit floats (over 6 KB per vector).
    * When retrieving 100 documents, including raw embeddings serializes over 600 KB of binary floats across the network wire, increasing JSON serialization CPU overhead and network latency.
    * Downstream LLMs only need the **raw text document strings**, not the floating point vectors.
    * Omitting embeddings by specifying `include=["documents", "metadatas", "distances"]` saves 80%+ of response payload size and network bandwidth.
    """)

with st.expander("Q4: Can you delete documents in ChromaDB based on a metadata filter without knowing their IDs?"):
    st.markdown("""
    **Answer**:
    **Yes!** ChromaDB supports predicate-based deletion:
    ```python
    collection.delete(
        where={"tenant_id": "churned_customer_82"}
    )
    ```
    * ChromaDB queries SQLite to identify all IDs matching the `where` predicate, deletes their metadata rows, and unlinks their corresponding nodes in the HNSW index.
    """)

with st.expander("Q5: What happens if you call `collection.update()` with an ID that does not exist?"):
    st.markdown("""
    **Answer**:
    * ChromaDB throws an exception (`ValueError: ID not found in collection`).
    * The operation fails without inserting the record.
    * If you want the system to insert missing records while updating existing ones, always use `collection.upsert()`.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Using `add()` in Scheduled Ingestion Cron Jobs**  
    Running `add()` on nightly cron jobs fails on day 2 because all document IDs already exist. Always use `upsert()` for recurring pipelines.
    """)
    st.warning("""
    **Trap 2: Requesting Embeddings When Hydrating LLM Context**  
    Including `"embeddings"` in `include` parameters bloats response payloads with thousands of raw floats that the LLM ignores.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Updating Metadata Without Providing Document Text**  
    In older Chroma versions, calling `update(ids=[...], metadatas=[...])` without documents could overwrite document strings. Verify SDK version docs.
    """)
    st.warning("""
    **Trap 4: Forgetting Pagination on `collection.get()`**  
    Calling `collection.get()` on a 500,000 document collection without `limit` attempts to load all 500k strings into RAM, causing out-of-memory crashes.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What makes upsert() unique?"):
        st.markdown("""
        **Atomic Merge**:  
        Inserts new records and updates existing ones; idempotent and zero-error in production pipelines.
        """)
    with st.expander("🗂️ Flashcard: How does get() differ from query()?"):
        st.markdown("""
        **SQL vs Vector Search**:  
        `get()` performs fast direct lookups bypassing HNSW; `query()` calculates semantic vector similarity.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What does collection.count() return?"):
        st.markdown("""
        **Total Record Count**:  
        Returns integer count of documents indexed in the collection in sub-millisecond time.
        """)
    with st.expander("🗂️ Flashcard: How to delete by metadata?"):
        st.markdown("""
        **Predicate Deletion**:  
        Call `collection.delete(where={'status': 'expired'})` without needing explicit IDs.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the differences between add(), update(), and upsert() mutations.", value=True)
chk2 = st.checkbox("I know why upsert() is the idempotent gold standard for production data pipelines.", value=True)
chk3 = st.checkbox("I understand how get() and peek() inspect records without vector similarity overhead.", value=True)
chk4 = st.checkbox("I know how to delete records by explicit ID lists or metadata predicates.", value=True)
chk5 = st.checkbox("I know why omitting raw embeddings in `include` parameters optimizes network bandwidth.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 3: Documents and Data Management in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 2: ChromaDB Basics!")
