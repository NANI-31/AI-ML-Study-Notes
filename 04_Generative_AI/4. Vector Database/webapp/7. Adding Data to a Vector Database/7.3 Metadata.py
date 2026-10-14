import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="7.3 Metadata - Vector DB Curriculum",
    page_icon="🏷️",
    layout="wide"
)

# Title & Overview
st.title("7.3 Metadata: Payloads, Data Types, In-Place Updates & `where` Filtering")
st.caption("Chapter 7: Adding Data to a Vector Database | Domain: 4. Vector Database")
st.markdown("""
In real-world enterprise applications, pure vector similarity search is almost never sufficient on its own. 
A legal query must only search contracts active in `2024`; an internal HR bot must only retrieve benefits documents 
matching the user's `security_clearance`; a SaaS platform must strictly isolate data belonging to `tenant_id = 104`.

**Metadata** is the structured bridge that allows vector databases to execute **hybrid scalar and vector retrieval**. 
In ChromaDB and modern vector databases, metadata is stored as JSON key-value payloads attached to each vector record.

In this lesson, we explore:
1. **Metadata Structure & Schema Conventions**
2. **Supported Scalar Data Types (`str`, `int`, `float`, `bool`)**
3. **Adding & Updating Metadata In-Place** (Zero re-embedding cost!)
4. **Chroma's `where` Query Filter Operators** (`$eq`, `$ne`, `$gt`, `$in`, `$and`, `$or`)
5. **Document Content Filtering (`where_document`)**
""")

st.divider()

# Section 1: Metadata Structure, Types & Schema Conventions
st.header("1. Metadata Structure, Supported Types & Enterprise Schema Design", divider="rainbow")

col_meta1, col_meta2 = col_meta1, col_meta2 = st.columns([1, 1])

with col_meta1:
    st.markdown("""
    ### 1. Supported Primitive Data Types
    ChromaDB metadata dictionaries strictly enforce **flat primitive types**:
    * **`str`**: Categorical tags, titles, authors, URLs (`"dept": "Engineering"`).
    * **`int`**: Timestamps, page numbers, revision versions (`"year": 2024`, `"page": 4`).
    * **`float`**: Prices, confidence scores, geographic coordinates (`"price": 49.99`).
    * **`bool`**: Verification flags, public visibility (`"is_verified": True`).
    * **The Nested Dict Constraint**:
      * ChromaDB currently rejects nested dictionaries: `{"user": {"name": "Alice"}}` will throw a `ValueError`!
      * To store nested objects, flatten them: `{"user_name": "Alice"}` or JSON-stringify them: `{"user_json": json.dumps(...)}`.
    """)

with col_meta2:
    st.markdown("""
    ### 2. Enterprise RAG Metadata Schema Conventions
    In production RAG systems, standard metadata payloads include:
    * **Provenance Attributes**:
      * `source`: Original file path (`"s3://docs/q3_earnings.pdf"`).
      * `page_number`: Exact PDF page (`14`).
      * `chunk_index`: Position within document (`4`).
      * `total_chunks`: Total segments in document (`42`).
    * **Governance & Access Attributes**:
      * `tenant_id`: Multi-tenant customer ID (`"org_8812"`).
      * `access_roles`: Permission tiers (`"finance_admin"`).
      * `published_at`: Unix timestamp for freshness filtering.
    """)

st.markdown("### Metadata Schema Reference Table")
schema_ref_df = pd.DataFrame({
    "Metadata Field": ["tenant_id", "source_file", "page_number", "chunk_index", "category", "created_timestamp", "is_confidential"],
    "Type": ["str", "str", "int", "int", "str", "int", "bool"],
    "Sample Value": ["'org_acme_corp'", "'security_policy.pdf'", "12", "3", "'Compliance'", "1712000000", "True"],
    "Filter Role": [
        "Strict multi-tenant data isolation",
        "Document-level source citation in LLM answer",
        "Deep link navigation for PDF viewers",
        "Parent-child chunk reassembly",
        "Domain-specific topical pre-filtering",
        "Date range queries (`$gte: 1700000000`)",
        "Role-based access control (RBAC)"
    ]
})
st.dataframe(schema_ref_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Adding & Updating Metadata In-Place
st.header("2. Adding & Updating Metadata In-Place (Zero Re-Embedding Cost)", divider="rainbow")

st.markdown("""
A vital architectural feature of vector databases is **in-place metadata updating**. 
Modifying metadata attributes does **NOT** require re-calculating expensive embeddings or rebuilding the HNSW graph!
""")

col_up1, col_up2 = col_up1, col_up2 = st.columns([1, 1])

with col_up1:
    st.markdown("""
    ### Adding Metadata during Ingestion
    Metadata is passed as a list of dictionaries strictly matching the length of `ids`:
    ```python
    collection.add(
        documents=["PostgreSQL performance tuning."],
        metadatas=[{"dept": "Database", "verified": True, "year": 2024}],
        ids=["doc_01"]
    )
    ```
    """)

with col_up2:
    st.markdown("""
    ### Updating Metadata In-Place (`collection.update`)
    If a document's metadata changes (e.g. marking as archived or changing access tier), call `collection.update()` without passing `embeddings` or `documents`:
    ```python
    # In-place metadata update: Vector index is UNTOUCHED!
    collection.update(
        ids=["doc_01"],
        metadatas=[{"dept": "Database", "verified": True, "year": 2024, "status": "archived"}]
    )
    ```
    * **Performance**: Updates the SQLite row in sub-millisecond time.
    * **Zero Cost**: Zero OpenAI API calls; zero GPU cycles.
    """)

st.divider()

# Section 3: Chroma `where` Query Filter Operators
st.header("3. Chroma `where` Query Filter Operators", divider="rainbow")

st.markdown("""
Chroma provides a rich MongoDB-style operator syntax passed via the `where` parameter in `collection.query()`:
""")

col_op1, col_op2 = col_op1, col_op2 = st.columns([1, 1])

with col_op1:
    st.markdown("""
    ### 1. Comparison Operators
    * **Exact Equality (`$eq` or simple dict)**:
      * `where={"dept": "DevOps"}`
      * `where={"dept": {"$eq": "DevOps"}}`
    * **Inequality (`$ne`)**:
      * `where={"status": {"$ne": "archived"}}`
    * **Numerical Range (`$gt`, `$gte`, `$lt`, `$lte`)**:
      * `where={"year": {"$gte": 2023}}`
      * `where={"price": {"$lt": 100.0}}`
    * **Set Membership (`$in`, `$nin`)**:
      * `where={"category": {"$in": ["DevOps", "Database", "SRE"]}}`
      * `where={"tier": {"$nin": ["Free", "Trial"]}}`
    """)

with col_op2:
    st.markdown("""
    ### 2. Logical Operators (`$and`, `$or`)
    To combine multiple conditions, wrap them in `$and` or `$or` lists:
    ```python
    # Logical AND
    where = {
        "$and": [
            {"dept": {"$eq": "Security"}},
            {"year": {"$gte": 2023}}
        ]
    }

    # Logical OR
    where = {
        "$or": [
            {"tier": {"$eq": "Enterprise"}},
            {"is_public": {"$eq": True}}
        ]
    }
    ```
    ### 3. Document Substring Filtering (`where_document`)
    Filter directly on text content:
    * `where_document={"$contains": "Kubernetes"}`
    * `where_document={"$not_contains": "deprecated"}`
    """)

st.divider()

# Section 4: Interactive Metadata Filter Query Builder
st.header("4. Interactive Metadata Filter Query Builder", divider="rainbow")

st.markdown("""
Test complex metadata filter queries against an interactive knowledge base. 
Configure operators, inspect the generated ChromaDB Python query, and observe real-time candidate matching:
""")

# Sample Knowledge Base for Filtering Demo
demo_kb = [
    {"id": "doc_1", "text": "Kubernetes pod autoscaling guide with HPA.", "dept": "DevOps", "year": 2024, "public": True},
    {"id": "doc_2", "text": "PostgreSQL connection pool optimization with PgBouncer.", "dept": "Database", "year": 2023, "public": True},
    {"id": "doc_3", "text": "Internal security incident response runbook.", "dept": "Security", "year": 2024, "public": False},
    {"id": "doc_4", "text": "Legacy MySQL database replication guide.", "dept": "Database", "year": 2020, "public": True},
    {"id": "doc_5", "text": "Corporate financial audit report and balance sheet.", "dept": "Finance", "year": 2024, "public": False}
]

col_qb_in, col_qb_out = st.columns([1, 1])

with col_qb_in:
    st.subheader("Filter Builder Controls")
    f_dept = st.multiselect("Department (`$in`)", ["DevOps", "Database", "Security", "Finance"], default=["DevOps", "Database"])
    f_min_year = st.slider("Minimum Year (`$gte`)", 2020, 2024, 2023)
    f_public_only = st.checkbox("Public Documents Only (`public == True`)", value=True)
    f_contains = st.text_input("Document Must Contain (`$contains`)", value="")

with col_qb_out:
    st.subheader("Generated Chroma Query & Evaluation")
    
    # Build filter dictionary
    and_conditions = []
    if f_dept:
        and_conditions.append({"dept": {"$in": f_dept}})
    and_conditions.append({"year": {"$gte": f_min_year}})
    if f_public_only:
        and_conditions.append({"public": {"$eq": True}})
        
    where_clause = {"$and": and_conditions} if len(and_conditions) > 1 else and_conditions[0]
    
    code_query_preview = f"""results = collection.query(
    query_texts=["performance optimization"],
    n_results=3,
    where={where_clause}{f',\\n    where_document={{"$contains": "{f_contains}"}}' if f_contains else ''}
)"""
    st.code(code_query_preview, language="python")
    
    # Evaluate matches against sample corpus
    matches = []
    for d in demo_kb:
        # Check dept
        if f_dept and d["dept"] not in f_dept:
            continue
        # Check year
        if d["year"] < f_min_year:
            continue
        # Check public
        if f_public_only and not d["public"]:
            continue
        # Check contains
        if f_contains and f_contains.lower() not in d["text"].lower():
            continue
        matches.append(d)
        
    st.markdown(f"**Matching Candidates ({len(matches)} / {len(demo_kb)} records passed):**")
    if matches:
        st.dataframe(pd.DataFrame(matches)[["id", "dept", "year", "public", "text"]], use_container_width=True, hide_index=True)
    else:
        st.warning("Zero records passed this filter combination!")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Complete Metadata Management & Multi-Operator Filtering", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates:
1. Ingesting documents with multi-typed metadata payloads.
2. Executing an **In-Place Metadata Update** without touching embeddings.
3. Querying using **`$in`**, **`$gte`**, **`$and`**, and **`$contains`** operators:
""")

code_meta_demo = '''import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
col = client.create_collection("tech_wiki", metadata={"hnsw:space": "cosine"})

# 1. Ingest documents with structured metadata
print("=== 1. Ingesting Documents with Metadata ===")
col.add(
    documents=[
        "Kubernetes Horizontal Pod Autoscaler deployment.",
        "PostgreSQL database query optimization.",
        "Internal security audit protocols.",
        "Legacy MySQL replication guide."
    ],
    metadatas=[
        {"dept": "DevOps", "year": 2024, "public": True, "tier": 1},
        {"dept": "Database", "year": 2023, "public": True, "tier": 2},
        {"dept": "Security", "year": 2024, "public": False, "tier": 1},
        {"dept": "Database", "year": 2020, "public": True, "tier": 3}
    ],
    ids=["doc_1", "doc_2", "doc_3", "doc_4"]
)
print(f"Total Ingested: {col.count()} documents")

# 2. In-Place Metadata Update (Zero vector re-indexing!)
print("\\n=== 2. Performing In-Place Metadata Update ===")
col.update(
    ids=["doc_4"],
    metadatas=[{"dept": "Database", "year": 2020, "public": False, "tier": 3, "status": "deprecated"}]
)
updated_meta = col.get(ids=["doc_4"])["metadatas"][0]
print(f"Updated doc_4 Metadata: {updated_meta}")

# 3. Complex Multi-Operator Query ($and, $in, $gte)
print("\\n=== 3. Executing Complex Filtered Query ===")
filter_condition = {
    "$and": [
        {"dept": {"$in": ["DevOps", "Database"]}},
        {"year": {"$gte": 2023}},
        {"public": {"$eq": True}}
    ]
}

hits = col.query(
    query_texts=["How to configure autoscaling or tuning?"],
    n_results=5,
    where=filter_condition
)

print(f"Discovered {len(hits['ids'][0])} matching document(s):")
for doc_id, doc_text, meta in zip(hits["ids"][0], hits["documents"][0], hits["metadatas"][0]):
    print(f"  * ID: {doc_id} | Year: {meta['year']} | Dept: {meta['dept']} | Text: '{doc_text}'")
'''

st.code(code_meta_demo, language="python")

if st.button("🚀 Execute Metadata Filtering Demo", type="primary"):
    exec_scope = {}
    exec(code_meta_demo, globals(), exec_scope)
    
    up_m = exec_scope["updated_meta"]
    hit_ids = exec_scope["hits"]["ids"][0]
    
    st.success("Metadata Operations and Filtered Query executed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("**In-Place Updated Payload (`doc_4`):**")
        st.json(up_m)
    with col_out2:
        st.markdown(f"**Filtered Query Matches (`{len(hit_ids)}` hits):**")
        st.json(hit_ids)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. How do Vector Databases evaluate metadata filters during HNSW graph search without causing recall collapse?"""
a1 = """* **The Naive Flaw (Post-Filtering)**:
  * Running pure vector search first, then filtering candidates by metadata.
  * If only 1% of the database matches `tenant_id=42`, the top-100 vector candidates will contain zero matching records, causing complete recall collapse.
* **The Single-Stage Solution (Pre-Filtering with Bitsets)**:
  * The database evaluates the metadata predicate (`where={"dept": "DevOps"}`) against an in-memory inverted index (e.g. Roaring Bitsets) in microseconds, creating a bitmask of valid IDs.
  * During HNSW graph traversal, as the search algorithm hops between nodes, it checks the bitmask.
  * Non-matching nodes can be stepped through as routing bridges, but are excluded from candidate heaps.
  * Delivers 100% filter adherence with sub-5ms ANN search latency."""

q2 = """2. Why does ChromaDB reject nested dictionaries in metadata, and how should developers model hierarchical data?"""
a2 = """* **Why It Rejects Nested Dicts**:
  * ChromaDB stores metadata in SQLite with typed columns and builds inverted indices across flat primitive types (`TEXT`, `INTEGER`, `REAL`).
  * Indexing arbitrary nested JSON trees requires complex document store engines (like MongoDB BSON) and introduces significant indexing overhead.
* **How to Model Hierarchical Data**:
  1. **Flatten Keys**: Replace `{"user": {"role": "admin"}}` with `{"user_role": "admin"}`.
  2. **JSON Stringification**: Serialize complex objects to a string: `{"details": json.dumps(data)}` (useful for hydration, but cannot be queried via `where` comparison operators)."""

q3 = """3. What is the performance difference between `collection.update()` and `collection.upsert()` when only metadata has changed?"""
a3 = """* If you only pass `ids` and `metadatas` to `collection.update()`, Chroma modifies only the SQLite metadata row and updates inverted filter bitsets.
* It completely bypasses vector embedding generation and does not touch the HNSW graph index.
* `collection.upsert()`, however, will check if the record exists; if you omit vectors or documents, it assumes you are updating only the provided fields.
* Always omit `documents` and `embeddings` when performing metadata-only updates to guarantee zero write amplification."""

q4 = """4. What is the difference between `where` and `where_document` in ChromaDB?"""
a4 = """* **`where` (Metadata Filter)**:
  * Operates on structured JSON metadata key-value payloads.
  * Supports equality, inequality, range comparisons, and set membership (`$eq`, `$gt`, `$in`).
* **`where_document` (Content Substring Filter)**:
  * Operates directly on the raw `document` text string.
  * Supports lexical substring matching: `{"$contains": "substring"}` and `{"$not_contains": "substring"}`.
  * Allows enforcing that retrieved chunks must explicitly contain a specific keyword or acronym alongside semantic vector search."""

q5 = """5. What is the High-Cardinality Metadata Trap in vector search engines?"""
a5 = """* If developers attach high-cardinality metadata fields (e.g. unique request IDs, nanosecond timestamps, or random session tokens) to millions of vectors:
  * The database's inverted metadata index must allocate a separate posting list or bitset for every unique value.
  * This causes severe memory bloat and degrades database startup times.
  * Only index metadata fields that will actively be used as query filters (`where`). Store purely informative metadata in an external database."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Metadata Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Nested Dictionary Serialization Exception**  
> Passing nested Python dictionaries like `metadatas=[{"author": {"name": "Alice"}}]` will immediately throw 
> `ValueError: Expected metadata value to be a str, int, float or bool`. 
> Always flatten metadata payloads into simple top-level key-values before ingesting.

> [!WARNING]
> **Trap 2: The 99.9% Hyper-Selective Pre-Filter Stall**  
> If a metadata filter matches only 5 documents out of 10 million in an HNSW graph, navigating the graph can 
> become slower than a brute-force scan because the search wanders through thousands of non-matching nodes 
> before finding an allowed candidate. For hyper-selective filters, configure your vector database to fall back 
> to an indexed brute-force scan.

> [!WARNING]
> **Trap 3: Mixing Data Types on the Same Metadata Key**  
> If Record 1 stores `{"version": 2}` (integer) and Record 2 stores `{"version": "v2"}` (string), 
> range filters like `where={"version": {"$gte": 2}}` will fail or produce inconsistent results due to SQLite 
> type affinity differences. Enforce strict type consistency across your ingestion pipelines.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What Metadata Types Are Allowed?"):
        st.markdown("""
        **Allowed Types**:  
        Chroma metadata values must be flat primitives: `str`, `int`, `float`, or `bool`. Nested dictionaries are rejected.
        """)
    with st.expander("🗂️ Flashcard: How to Update Metadata In-Place?"):
        st.markdown("""
        **In-Place Updates**:  
        Call `collection.update(ids=[...], metadatas=[...])` without passing vectors to update metadata without re-indexing.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What Does `where_document` Do?"):
        st.markdown("""
        **`where_document`**:  
        Enforces lexical substring filtering on raw document text (e.g. `{"$contains": "keyword"}`).
        """)
    with st.expander("🗂️ Flashcard: How Does Single-Stage Filtering Work?"):
        st.markdown("""
        **Bitset Masking**:  
        Metadata filters compile into in-memory bitsets, filtering candidates during HNSW graph hops without recall collapse.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know which primitive data types are supported in ChromaDB metadata.", value=True)
chk2 = st.checkbox("I understand why nested dictionaries are rejected and how to flatten them.")
chk3 = st.checkbox("I know how to perform in-place metadata updates without re-embedding vectors.")
chk4 = st.checkbox("I can construct complex queries using `$in`, `$gte`, `$and`, and `$or` operators.")
chk5 = st.checkbox("I understand the difference between `where` (metadata) and `where_document` (content substring).")
chk6 = st.checkbox("I know how single-stage bitset filtering prevents recall collapse during graph search.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Metadata and are ready for Chapter 7.4: Data Management!")
