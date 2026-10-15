import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="8.1 get() Operation - ChromaDB Curriculum",
    page_icon="📥",
    layout="wide"
)

# Title & Overview
st.title("8.1 get() Operation: ID Lookups, Metadata Filtering, Pagination & Field Selection")
st.caption("Chapter 8: Getting and Retrieving Data | Domain: 5. ChromaDB")
st.markdown("""
While `collection.query()` is engineered for geometric approximate nearest-neighbor search, 
ChromaDB also serves as a robust document store through its relational retrieval interface: **`collection.get()`**.

The `get()` operation provides exact deterministic retrieval directly from ChromaDB's underlying SQLite database:
* **Point & Batch ID Lookups (`ids=[...]`)**: Fetching specific records by string identifiers in constant `O(1)` or logarithmic `O(log N)` time.
* **Attribute-Based Selection (`where={...}`)**: Retrieving records purely matching metadata conditions without evaluating vector distances.
* **Full Collection Pagination (`limit` & `offset`)**: Iterating across millions of stored documents page-by-page for maintenance, backups, or audits.
* **Payload Field Selectivity (`include`)**: Selectively fetching documents, metadatas, or raw high-dimensional embeddings.

In this lesson, we break down the operational mechanics of `get()`, explore pagination and filtering patterns in Python, 
and test an interactive Live `get()` Query Builder and Pagination Inspector.
""")

st.divider()

# Section 1: Anatomy of the collection.get() Method
st.header("1. Anatomy of the collection.get() Method", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### Core `get()` Parameters
    
    * **`ids: Optional[List[str]]`**:
      * A list of exact document string identifiers to retrieve. If provided, Chroma executes an indexed primary key lookup in SQLite.
    
    * **`where: Optional[Dict[str, Any]]`**:
      * Metadata filter predicates (e.g. `{"status": "published", "year": {"$gte": 2024}}`).
    
    * **`where_document: Optional[Dict[str, Any]]`**:
      * Full-text search predicate (e.g. `{"$contains": "kubernetes"}`).
    
    * **`limit: Optional[int]`**:
      * Maximum number of documents to return (for pagination or sampling).
    
    * **`offset: Optional[int]`**:
      * Number of matching records to skip before returning results (for page-based browsing).
    
    * **`include: List[str]`**:
      * List of payload keys to hydrate. Defaults to `["documents", "metadatas"]`.
      * Notice: Unlike `query()`, `get()` **never returns `"distances"`** because no geometric search took place!
    """)

with col_anat2:
    st.markdown("### The get() Execution Topology")
    get_topo_dot = """
    digraph GetTopology {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        CallGet [label="collection.get(ids=..., where=...)\\nDeterministic Exact Retrieval", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        SQLiteStore [label="SQLite Storage Subsystem\\n(Primary Key B-Tree & Inverted Index)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        SkipHNSW [label="BYPASS HNSW GRAPH!\\nZero Vector Math / SIMD Required\\nPure SQL Relational Retrieval", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        ResultPayload [label="GetResponse Dictionary\\nFlat 1D Parallel Lists\\n{'ids': [...], 'documents': [...]}", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        CallGet -> SQLiteStore;
        CallGet -> SkipHNSW [style="dotted"];
        SQLiteStore -> ResultPayload;
    }
    """
    st.graphviz_chart(get_topo_dot, use_container_width=False)

st.divider()

# Section 2: Comparing Results: 1D Flat Lists vs 2D Query Lists
st.header("2. Comparing Results: 1D Flat Lists vs 2D Query Lists", divider="rainbow")

col_dim1, col_dim2 = col_dim1, col_dim2 = st.columns([1, 1])

with col_dim1:
    st.markdown("""
    ### Critical Architectural Distinction
    
    * **`collection.query()` Returns 2D Lists**:
      * Structured as `(Q, K)`: `results["documents"][query_idx][rank_idx]`.
      * Accounts for multi-query batching and ranked nearest-neighbor scoring.
      * Includes `distances`.
    
    * **`collection.get()` Returns 1D Flat Lists**:
      * Structured as flat parallel arrays: `results["documents"][idx]`.
      * There is no concept of "queries" or "ranks"; it simply returns a flat set of matching database records.
      * **Never contains `distances`**!
    """)

with col_dim2:
    st.markdown("### Code Comparison")
    st.code("""# 1. QUERY RETURN STRUCTURE (2-Dimensional):
q_res = collection.query(query_texts=["Postgres"], n_results=1)
first_doc = q_res["documents"][0][0]  # Needs double brackets [0][0]
first_dist = q_res["distances"][0][0]

# 2. GET RETURN STRUCTURE (1-Dimensional Flat):
g_res = collection.get(ids=["doc_101"])
first_doc = g_res["documents"][0]     # Single bracket [0]!
# 'distances' is NOT present in g_res
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Advanced get() Operations
st.header("3. Clean Python Implementation of Advanced get() Operations", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("engineering_wiki")

# 1. Point ID Lookup: Retrieve 2 specific documents by ID
id_result = collection.get(
    ids=["doc_k8s_01", "doc_db_02"],
    include=["documents", "metadatas"]
)
print("Point ID Lookup returned:", id_result["ids"])

# 2. Pure Metadata Selection: Find all active production runbooks
meta_result = collection.get(
    where={"$and": [{"env": "production"}, {"status": "active"}]},
    limit=50
)
print(f"Found {len(meta_result['ids'])} active production documents.")

# 3. Paginating through an entire collection (Batch Scrolling)
def iterate_all_documents(collection, page_size: int = 500):
    offset = 0
    total_scanned = 0
    
    while True:
        page = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"]
        )
        batch_ids = page["ids"]
        if not batch_ids:
            break  # End of collection reached
            
        total_scanned += len(batch_ids)
        print(f"Processed page at offset {offset} ({len(batch_ids)} records)")
        offset += len(batch_ids)
        
    print(f"Full collection audit complete! Total documents: {total_scanned}")

iterate_all_documents(collection)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live get() Query Builder & Paginator
st.header("4. Interactive Workbench: Live get() Query Builder & Paginator", divider="rainbow")

st.markdown("""
Test `collection.get()` filtering and pagination interactively against a simulated 
knowledge base. Adjust ID filters, metadata predicates, limit, and offset to inspect the resulting 1D payload.
""")

col_wb_f1, col_wb_f2 = col_wb_f2 = st.columns(2)

with col_wb_f1:
    st.subheader("1. Selection Filters")
    filter_mode = st.radio(
        "Choose Filter Mode:",
        ["Filter by Specific IDs", "Filter by Metadata Predicate", "Scan All with Pagination"]
    )
    
    selected_ids = []
    if "Specific IDs" in filter_mode:
        selected_ids = st.multiselect(
            "Select IDs to Fetch:",
            options=["doc_001", "doc_002", "doc_003", "doc_004", "doc_005", "doc_006"],
            default=["doc_001", "doc_003"]
        )
    elif "Metadata" in filter_mode:
        selected_dept = st.selectbox("Department Filter (`where={'dept': ...}`):", ["engineering", "security", "product"])

with col_wb_f2:
    st.subheader("2. Pagination & Field Selection")
    page_limit = st.slider("Page Limit (`limit`):", min_value=1, max_value=6, value=3)
    page_offset = st.slider("Page Offset (`offset`):", min_value=0, max_value=5, value=0)
    inc_embeddings = st.checkbox("Include Stored Embeddings (`'embeddings'`)", value=False)

# Mock database
SIMULATED_DATA = [
    {"id": "doc_001", "dept": "engineering", "doc": "Kubernetes Ingress routes external traffic to cluster services."},
    {"id": "doc_002", "dept": "engineering", "doc": "PostgreSQL WAL streaming replication disaster recovery guide."},
    {"id": "doc_003", "dept": "security", "doc": "Zero-trust network architecture requires mutual TLS."},
    {"id": "doc_004", "dept": "security", "doc": "OAuth 2.0 PKCE flow eliminates client secret leakage."},
    {"id": "doc_005", "dept": "product", "doc": "Roadmap planning quarterly OKR prioritization rubric."},
    {"id": "doc_006", "dept": "engineering", "doc": "Redis cache eviction algorithms: LRU vs LFU comparison."}
]

# Apply filter
filtered_pool = SIMULATED_DATA
if "Specific IDs" in filter_mode:
    filtered_pool = [d for d in filtered_pool if d["id"] in selected_ids]
elif "Metadata" in filter_mode:
    filtered_pool = [d for d in filtered_pool if d["dept"] == selected_dept]

# Apply pagination
paged_pool = filtered_pool[page_offset : page_offset + page_limit]

st.divider()

col_get_json, col_get_tbl = col_get_json, col_get_tbl = st.columns([1, 1.2])

with col_get_json:
    st.subheader("Raw 1D GetResponse Payload")
    get_payload = {
        "ids": [d["id"] for d in paged_pool],
        "documents": [d["doc"] for d in paged_pool],
        "metadatas": [{"dept": d["dept"]} for d in paged_pool]
    }
    if inc_embeddings:
        get_payload["embeddings"] = [[round(np.random.uniform(-0.5, 0.5), 3) for _ in range(4)] for _ in paged_pool]
    st.json(get_payload)

with col_get_tbl:
    st.subheader("Hydrated Document Table")
    if paged_pool:
        st.dataframe(pd.DataFrame(paged_pool), use_container_width=True, hide_index=True)
        st.success(f"🟢 Returned {len(paged_pool)} documents (Total matching: {len(filtered_pool)}).")
    else:
        st.warning("No records matched the filter and offset parameters.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the mechanical difference between `collection.get()` and `collection.query()`?"):
    st.markdown("""
    **Answer**:
    * **`get()`**: Queries the **SQLite relational database** directly using B-Tree index lookups on primary keys (`ids`) or indexed metadata columns (`where`). Zero vector math, zero embeddings, and zero HNSW graph traversals are involved.
    * **`query()`**: Queries the **HNSW vector index** using geometric distance calculations. It computes Approximate Nearest Neighbors for an input query vector, sorting results by ascending distance.
    """)

with st.expander("Q2: Why does `collection.get()` return 1D lists while `collection.query()` returns 2D lists?"):
    st.markdown("""
    **Answer**:
    * `query()` is designed for **batch searching**: you can pass multiple query texts simultaneously (`query_texts=["Q1", "Q2"]`), requiring an outer list for the query index and an inner list for the Top-K candidates.
    * `get()` is a **flat relational fetch**: you pass an ID list or metadata filter, and it returns a flat set of matching documents from the table. There is no query batch dimension.
    """)

with st.expander("Q3: What happens if you call `collection.get(ids=['missing_id_999'])` with an ID that does not exist?"):
    st.markdown("""
    **Answer**:
    * Unlike `collection.update()`, `collection.get()` **does NOT raise an error** when an ID is not found.
    * It silently returns empty parallel lists: `{"ids": [], "documents": [], "metadatas": []}`.
    * If you request `ids=["doc_1", "missing_999"]` and only `doc_1` exists, Chroma returns only `doc_1` in the lists.
    """)

with st.expander("Q4: How do `limit` and `offset` work in `collection.get()`, and what is their performance limit?"):
    st.markdown("""
    **Answer**:
    * `limit` caps the number of returned rows.
    * `offset` skips that many rows before beginning to return results (standard SQL `LIMIT / OFFSET`).
    * **Performance Trap**: At very large offsets (`offset=1,000,000`), SQLite must still scan through the first million rows to discard them. For massive multi-million document exports, keyset pagination (filtering `where={"id": {"$gt": last_seen_id}}`) is far more efficient.
    """)

with st.expander("Q5: Can you fetch raw embeddings using `collection.get()`?"):
    st.markdown("""
    **Answer**:
    * **Yes**. Pass `include=["documents", "metadatas", "embeddings"]`.
    * This allows exporting pre-computed embeddings for model fine-tuning, visualization (e.g. t-SNE / UMAP), or migrating collections to another vector engine.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Expecting `'distances'` in `get()` Results**  
    Writing `results["distances"]` after calling `get()` raises a `KeyError`. `get()` does not compute vector distances.
    """)
    st.warning("""
    **Trap 2: Accessing `get()` Results with 2D Slicing `[0][0]`**  
    `get()` returns a flat 1D list! Writing `results["documents"][0][0]` extracts the first character of the string instead of the document! Use `[0]`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Full-Collection `get()` Without `limit`**  
    Calling `collection.get()` with no arguments on a 2-million-doc collection pulls every record into RAM, causing an instant Out-of-Memory crash. Always supply `limit`.
    """)
    st.warning("""
    **Trap 4: Slow Pagination with Massive Offsets**  
    Using `offset=500,000` forces SQLite to scan half a million rows on every page. Use keyset ID filtering for high-scale exports.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What engine does get() use?"):
        st.markdown("""
        **SQLite B-Trees**:  
        Executes relational SQL queries; bypasses HNSW vector graph entirely.
        """)
    with st.expander("🗂️ Flashcard: Does get() return distances?"):
        st.markdown("""
        **Never**:  
        No geometric distance math is evaluated during relational lookups.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What if an ID is missing?"):
        st.markdown("""
        **Silently Ignored**:  
        Returns only the IDs that exist; does not throw an error.
        """)
    with st.expander("🗂️ Flashcard: How are get() results shaped?"):
        st.markdown("""
        **1D Flat Lists**:  
        `results['documents'][idx]` (flat array, not nested 2D).
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the operational role of `collection.get()` as a deterministic document retrieval interface.", value=True)
chk2 = st.checkbox("I know why `get()` returns flat 1D lists while `query()` returns nested 2D parallel lists.", value=True)
chk3 = st.checkbox("I understand how to query records by ID, metadata predicates, and full-text keyword filters.", value=True)
chk4 = st.checkbox("I know how to paginate through large collections using `limit` and `offset`.", value=True)
chk5 = st.checkbox("I understand how to use `include` to selectively hydrate or export raw high-dimensional embeddings.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 8.2: Query vs Get in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 8.1: get() Operation and are ready for Chapter 8.2: Query vs Get!")
