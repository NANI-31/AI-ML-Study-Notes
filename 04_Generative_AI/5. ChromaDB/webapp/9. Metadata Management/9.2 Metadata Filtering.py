import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="9.2 Metadata Filtering - ChromaDB Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("9.2 Metadata Filtering: Comparison Operators, Numeric Ranges, $in & $nin Sets")
st.caption("Chapter 9: Metadata Management | Domain: 5. ChromaDB")
st.markdown("""
In enterprise vector search, searching across an entire unpartitioned dataset produces poor precision. 
When an SRE searches for runbooks, they only care about `env: "production"` and `service: "auth-gateway"`.

ChromaDB provides a rich set of **relational metadata filtering operators** executed via the `where` parameter:
* **Equality & Inequality (`$eq`, `$ne`)**: Matching exact categorical attributes or excluding obsolete categories.
* **Numeric Range Filtering (`$gt`, `$gte`, `$lt`, `$lte`)**: Enforcing strict numerical boundaries on dates, versions, scores, or prices.
* **Set Membership (`$in`, `$nin`)**: Filtering against lists of allowed tenants, roles, departments, or tags.
* **Pre-Filtering Execution**: ChromaDB applies metadata filters **before or during HNSW traversal**, ensuring that all returned Top-K candidates are 100% compliant with your business constraints.

In this lesson, we master the full catalog of ChromaDB filter operators, examine pre-filtering mechanics, 
implement practical filtered search pipelines in Python, and test an interactive Visual Filter Builder & Matcher.
""")

st.divider()

# Section 1: The ChromaDB Filter Operator Catalog
st.header("1. The ChromaDB Filter Operator Catalog", divider="rainbow")

col_op1, col_op2 = col_op1, col_op2 = st.columns([1.1, 0.9])

with col_op1:
    st.markdown("""
    ### Operator Reference Guide
    
    * **Equality (`$eq`)**:
      * Shorthand: `where={"category": "runbooks"}`
      * Explicit: `where={"category": {"$eq": "runbooks"}}`
    
    * **Inequality (`$ne`)**:
      * `where={"status": {"$ne": "archived"}}`
      * Excludes all documents with status `"archived"`.
    
    * **Numeric Comparisons (`$gt`, `$gte`, `$lt`, `$lte`)**:
      * Greater Than: `where={"rating": {"$gt": 4.5}}`
      * Greater Than / Equal: `where={"year": {"$gte": 2024}}`
      * Less Than: `where={"latency_ms": {"$lt": 100.0}}`
      * Less Than / Equal: `where={"page_number": {"$lte": 5}}`
    
    * **Set Membership (`$in`, `$nin`)**:
      * Included in List: `where={"region": {"$in": ["us-east-1", "eu-central-1"]}}`
      * Not in List: `where={"tier": {"$nin": ["deprecated", "sandbox"]}}`
    """)

with col_op2:
    st.markdown("### Pre-Filtering vs. Post-Filtering Mechanics")
    filter_mech_dot = """
    digraph FilterMechanics {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="collection.query(where={'year': {'$gte': 2024}})\\nInbound Filtered Search", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        SQLiteFilter [label="Step 1: SQLite Pre-Filter\\nEvaluate B-Trees; generate ID bitmask", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        HNSWTraverse [label="Step 2: Constrained HNSW Traversal\\nOnly explore nodes present in bitmask", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        GuaranteedK [label="Step 3: Guaranteed Top-K\\n100% of returned candidates match criteria!", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InQuery -> SQLiteFilter -> HNSWTraverse -> GuaranteedK;
    }
    """
    st.graphviz_chart(filter_mech_dot, use_container_width=False)

st.divider()

# Section 2: Deep Dive into $in and $nin Set Operators
st.header("2. Deep Dive into $in and $nin Set Operators", divider="rainbow")

col_set1, col_set2 = col_set1, col_set2 = st.columns([1, 1])

with col_set1:
    st.markdown("""
    ### Why Set Operators Transform RAG Architectures
    
    1. **Multi-Role Access Control (RBAC)**:
       * When a user belongs to roles `["viewer", "editor"]`, you cannot query with a single string equality.
       * Using `$in` allows filtering records accessible by either role:
         `where={"allowed_role": {"$in": ["viewer", "editor", "public"]}}`.
    
    2. **Multi-Tenant Cohorts**:
       * Querying data shared across partner organizations:
         `where={"org_id": {"$in": ["org_alpha", "org_beta"]}}`.
    
    3. **Soft-Delete Exclusion**:
       * Filtering out inactive data lifecycle states:
         `where={"lifecycle": {"$nin": ["trashed", "archived", "pending_delete"]}}`.
    """)

with col_set2:
    st.markdown("### Syntactic Constraints")
    st.code("""# ✅ Correct $in usage: Value MUST be a Python list
valid_in = collection.query(
    query_texts=["database disaster recovery"],
    where={"env": {"$in": ["production", "staging"]}}
)

# ❌ Incorrect $in usage: Value cannot be a bare string
# This will raise a ChromaDB SchemaValidationError!
# where={"env": {"$in": "production"}}

# ✅ Correct $nin usage: Exclude multiple statuses
valid_nin = collection.get(
    where={"status": {"$nin": ["draft", "deprecated"]}},
    limit=50
)
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Metadata Filtered Search
st.header("3. Clean Python Implementation of Metadata Filtered Search", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="enterprise_documents",
    metadata={"hnsw:space": "cosine"}
)

# 1. Numeric Range Search: Find recent high-quality technical runbooks
recent_hits = collection.query(
    query_texts=["Kubernetes memory limit OOM killed"],
    n_results=3,
    where={
        "year": {"$gte": 2024}
    }
)

# 2. Set Inclusion: Find articles in allowed regions
region_hits = collection.query(
    query_texts=["Data sovereignty and encryption compliance"],
    n_results=3,
    where={
        "region": {"$in": ["eu-west-1", "eu-central-1"]}
    }
)

# 3. Exclusion: Audit all active non-archived policies
active_policies = collection.get(
    where={
        "status": {"$ne": "archived"}
    },
    limit=25
)
print(f"Retrieved {len(active_policies['ids'])} active non-archived policies.")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Visual Filter Builder & Live Matcher
st.header("4. Interactive Workbench: Visual Filter Builder & Live Matcher", divider="rainbow")

st.markdown("""
Build a metadata filter query interactively and test it against a simulated document collection. 
Inspect the generated ChromaDB `where` syntax and view the matched records in real time.
""")

col_wb_op, col_wb_val = col_wb_op, col_wb_val = st.columns(2)

with col_wb_op:
    st.subheader("1. Operator Configuration")
    target_field = st.selectbox("Select Metadata Field:", ["department", "year", "priority", "status"])
    
    if target_field == "department":
        op_choice = st.selectbox("Comparison Operator:", ["$eq (Equal)", "$ne (Not Equal)", "$in (In Set)", "$nin (Not In Set)"])
    elif target_field == "year":
        op_choice = st.selectbox("Comparison Operator:", ["$gte (>=)", "$gt (>)", "$lte (<=)", "$lt (<)", "$eq (Equal)"])
    elif target_field == "priority":
        op_choice = st.selectbox("Comparison Operator:", ["$lte (<=)", "$gte (>=)", "$eq (Equal)"])
    else:  # status
        op_choice = st.selectbox("Comparison Operator:", ["$eq (Equal)", "$ne (Not Equal)", "$in (In Set)"])

with col_wb_val:
    st.subheader("2. Filter Operand Values")
    if target_field == "department":
        if "$in" in op_choice or "$nin" in op_choice:
            filter_operand = st.multiselect("Select Departments:", ["engineering", "security", "infra", "legal"], default=["engineering", "infra"])
        else:
            filter_operand = st.selectbox("Select Department:", ["engineering", "security", "infra", "legal"])
    elif target_field == "year":
        filter_operand = st.slider("Year Boundary:", min_value=2020, max_value=2026, value=2024)
    elif target_field == "priority":
        filter_operand = st.slider("Priority Level (1=Highest):", min_value=1, max_value=5, value=2)
    else:
        if "$in" in op_choice:
            filter_operand = st.multiselect("Select Statuses:", ["active", "draft", "archived"], default=["active"])
        else:
            filter_operand = st.selectbox("Select Status:", ["active", "draft", "archived"])

# Clean operator key extraction
op_token = op_choice.split()[0]

# Build where clause
if op_token == "$eq":
    generated_where = {target_field: filter_operand}
else:
    generated_where = {target_field: {op_token: filter_operand}}

# Mock Corpus
MOCK_METADATA_DATASET = [
    {"id": "doc_eng_01", "department": "engineering", "year": 2024, "priority": 1, "status": "active", "title": "Microservices gRPC Migration"},
    {"id": "doc_eng_02", "department": "engineering", "year": 2022, "priority": 3, "status": "archived", "title": "Legacy SOAP API Deprecation"},
    {"id": "doc_inf_01", "department": "infra", "year": 2025, "priority": 1, "status": "active", "title": "Terraform AWS Multi-Region Provisioning"},
    {"id": "doc_inf_02", "department": "infra", "year": 2023, "priority": 2, "status": "active", "title": "Kubernetes Ingress Controller Tuning"},
    {"id": "doc_sec_01", "department": "security", "year": 2024, "priority": 1, "status": "active", "title": "Zero-Trust Service Mesh Authentication"},
    {"id": "doc_leg_01", "department": "legal", "year": 2021, "priority": 4, "status": "active", "title": "GDPR Cross-Border Transfer Compliance"}
]

# Match records
matched_docs = []
for doc in MOCK_METADATA_DATASET:
    val = doc[target_field]
    match = False
    if op_token == "$eq":
        match = (val == filter_operand)
    elif op_token == "$ne":
        match = (val != filter_operand)
    elif op_token == "$gt":
        match = (val > filter_operand)
    elif op_token == "$gte":
        match = (val >= filter_operand)
    elif op_token == "$lt":
        match = (val < filter_operand)
    elif op_token == "$lte":
        match = (val <= filter_operand)
    elif op_token == "$in":
        match = (val in filter_operand)
    elif op_token == "$nin":
        match = (val not in filter_operand)
    if match:
        matched_docs.append(doc)

st.divider()

col_where_code, col_where_hits = col_where_code, col_where_hits = st.columns([1, 1.2])

with col_where_code:
    st.subheader("Generated ChromaDB `where` Clause")
    st.json(generated_where)
    st.markdown("""
    **Execution Guarantee**:  
    During search, Chroma applies this condition before returning candidates, ensuring zero irrelevant hits!
    """)

with col_where_hits:
    st.subheader(f"Matching Documents ({len(matched_docs)} / {len(MOCK_METADATA_DATASET)})")
    if matched_docs:
        st.dataframe(pd.DataFrame(matched_docs), use_container_width=True, hide_index=True)
        st.success(f"🟢 Successfully filtered {len(matched_docs)} documents!")
    else:
        st.warning("No documents matched the specified filter criteria.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the difference between Pre-Filtering and Post-Filtering in vector databases, and how does ChromaDB implement it?"):
    st.markdown("""
    **Answer**:
    * **Post-Filtering**: The vector engine retrieves the Top-100 nearest neighbors purely based on distance, and then drops candidates that fail the metadata predicate. If the filter is highly selective (e.g. only 1% of the corpus matches), post-filtering can return zero results!
    * **Pre-Filtering (ChromaDB)**: The engine evaluates the metadata predicate against SQLite relational indexes first to generate a bitmask of valid document IDs. The HNSW graph search then only considers candidate nodes present in the bitmask, **guaranteeing that exactly `n_results` valid matches are returned** if they exist in the collection.
    """)

with st.expander("Q2: How do `$in` and `$nin` operators work in ChromaDB, and what type must their operand be?"):
    st.markdown("""
    **Answer**:
    * `$in` returns documents whose metadata value matches any element in the supplied list.
    * `$nin` returns documents whose metadata value is NOT present in the supplied list.
    * **Strict Type Requirement**: The value supplied to `$in` and `$nin` must always be a **Python list** (`List[Union[str, int, float, bool]]`). Passing a single string or number raises a `SchemaValidationError`.
    """)

with st.expander("Q3: Can you execute numeric range queries (e.g. `$gte`) on string metadata fields?"):
    st.markdown("""
    **Answer**:
    * **No**. In SQLite and ChromaDB, comparing strings using `$gt` or `$gte` performs lexicographical (alphabetical) comparison rather than numerical comparison.
    * For example, lexicographically, `"100"` is less than `"2"`.
    * Always ensure that numerical attributes (timestamps, page numbers, prices) are ingested as true Python `int` or `float` types.
    """)

with st.expander("Q4: What happens if a metadata filter references a key that does not exist in some documents?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does not enforce a uniform metadata schema across all documents.
    * If a query includes `where={"department": "security"}`, any document that completely lacks the `"department"` key evaluates to `False` and is omitted from the search results.
    """)

with st.expander("Q5: Can you use regex or wildcard pattern matching (e.g. `LIKE '%prod%'`) in `where` metadata filters?"):
    st.markdown("""
    **Answer**:
    * **No**. In ChromaDB, `where` metadata filters do not support regex or SQL `LIKE` wildcards.
    * For full-text wildcard or keyword substring searching, use **`where_document={"$contains": "keyword"}`** instead.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing a Bare String to `$in`**  
    Writing `where={"dept": {"$in": "engineering"}}` crashes with a type error. Always pass a list: `["engineering"]`.
    """)
    st.warning("""
    **Trap 2: Storing Numbers as Formatted Strings**  
    Storing `"year": "2024"` causes `$gt` to perform alphabetical sorting where `"2024" < "9"`. Store as numeric `int`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Metadata Filters to Look Up Primary Document IDs**  
    Writing `where={"id": "doc_101"}` fails because primary IDs are stored in internal table columns. Use `get(ids=[...])`.
    """)
    st.warning("""
    **Trap 4: Forgetting that Non-Existent Keys Evaluate to False**  
    Writing `where={"is_draft": {"$ne": True}}` will also discard documents where `"is_draft"` was never set! Backfill defaults.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does Pre-Filtering ensure?"):
        st.markdown("""
        **Guaranteed Candidates**:  
        Filters before graph search, ensuring all returned Top-K hits match criteria.
        """)
    with st.expander("🗂️ Flashcard: What is required operand for $in?"):
        st.markdown("""
        **Python List**:  
        Must be a list of primitives: `{"$in": ["a", "b"]}`.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How does $ne treat missing keys?"):
        st.markdown("""
        **Omitted**:  
        Documents lacking the specified key are excluded.
        """)
    with st.expander("🗂️ Flashcard: Can metadata do wildcard regex?"):
        st.markdown("""
        **No**:  
        Metadata filters do not support regex. Use `where_document` for text search.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the full catalog of ChromaDB metadata comparison operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`).", value=True)
chk2 = st.checkbox("I know how `$in` and `$nin` set membership operators work and that their operands must be Python lists.", value=True)
chk3 = st.checkbox("I understand why pre-filtering during HNSW traversal guarantees that returned candidates meet constraints.", value=True)
chk4 = st.checkbox("I know why numeric fields must be ingested as true `int` or `float` types rather than formatted strings.", value=True)
chk5 = st.checkbox("I understand how missing metadata keys evaluate during filtering operations.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 9.3: Logical Filtering ($and & $or) in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 9.2: Metadata Filtering and are ready for Chapter 9.3: Logical Filtering!")
