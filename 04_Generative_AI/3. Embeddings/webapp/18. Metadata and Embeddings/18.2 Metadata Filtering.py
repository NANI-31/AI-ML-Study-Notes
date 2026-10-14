import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import json
from datetime import datetime
from typing import List, Dict, Any, Callable

st.set_page_config(
    page_title="18.2 Metadata Filtering - Embeddings Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("18.2 Metadata Filtering: Categorical, Temporal, Multi-Tenant, & Composite Filters")
st.caption("Chapter 18: Metadata and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In enterprise AI applications, pure semantic search is almost never sufficient.
Users do not simply ask: *"Find documents about revenue."*
They ask: *"Find documents about revenue in **Q3 2024**, created by the **Finance Department**, for **Acme Corp**, where file format is **PDF**."*

This is the domain of **Metadata Filtering**:
* **Categorical Filtering**: Partitioning search by domain, topic, or department.
* **Temporal Filtering**: Restricting results by timestamp windows (`date >= 2024-01-01`) to eliminate obsolete guidelines.
* **Source & Format Filtering**: Confining queries to verified official knowledge sources (e.g. `source.startswith("s3://official-kb/")`).
* **Multi-Tenant & Security Filtering**: Enforcing strict role-based access control (RBAC) so User A never retrieves documents belonging to Tenant B!
* **Composite Boolean Filters**: Combining multiple predicates using `$and`, `$or`, `$not`, and `$in`.

This lesson explores how filtering algorithms interact with vector search indexes.
""")

st.divider()

# Section 1: Filtering Architecture: Pre, Post, & Single-Stage
st.header("1. Filtering Architecture: Pre-Filter, Post-Filter, & Single-Stage HNSW", divider="rainbow")

col_flt1, col_flt2 = st.columns([1, 1])

with col_flt1:
    st.markdown("""
    ### The Three Filtering Strategies
    * **1. Post-Filtering (The Naive Trap)**:
      * Executes pure vector search first to get top 100 documents, then filters by metadata in Python.
      * **Fatal Flaw**: If all top 100 results belong to other tenants, post-filtering returns **0 results**, even when thousands of valid documents exist deeper in the database!
    * **2. Pre-Filtering**:
      * Executes SQL/relational metadata filter first to extract matching IDs, then runs vector search on that subset.
      * **Flaw**: Bypasses the HNSW vector graph index; forces a slow unindexed linear scan on the subset.
    * **3. Single-Stage Filtered Graph Traversal (Modern Standard)**:
      * The HNSW graph search checks metadata bitmasks **during graph traversal**.
      * It navigates graph edges strictly through nodes that satisfy the metadata filter, preserving sub-millisecond speeds and guaranteeing non-empty top-$K$ results!
    """)

with col_flt2:
    st.markdown("### Composite Boolean Filter Tree")
    tree_dot = """
    digraph FilterTree {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RootAnd [label="Composite Filter ($and)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        TenantNode [label="Tenant Isolation ($eq)\\ntenant_id == 'acme_corp'", fillcolor="#ECFDF5", color="#10B981"];
        DateNode [label="Temporal Filter ($gte)\\ndate >= '2024-01-01'", fillcolor="#FEF3C7", color="#D97706"];
        CatNode [label="Categorical Filter ($in)\\ncategory in ['finance', 'legal']", fillcolor="#EDE9FE", color="#8B5CF6"];

        RootAnd -> TenantNode;
        RootAnd -> DateNode;
        RootAnd -> CatNode;

        BitmaskOut [label="Payload Bitmask Evaluator\\nInstant O(1) Candidate Filtering", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        TenantNode -> BitmaskOut;
        DateNode -> BitmaskOut;
        CatNode -> BitmaskOut;
    }
    """
    st.graphviz_chart(tree_dot, use_container_width=False)

st.markdown("### Vector Database Filter Query Syntax Matrix")
st.markdown("""
| Operation | ChromaDB Syntax | Qdrant Syntax | Pinecone Syntax |
| :--- | :--- | :--- | :--- |
| **Exact Equality** | `where={"dept": "hr"}` | `FieldCondition(key="dept", match=MatchValue(value="hr"))` | `{"dept": {"$eq": "hr"}}` |
| **Set Membership** | `where={"cat": {"$in": ["a", "b"]}}` | `FieldCondition(key="cat", match=MatchAny(any=["a", "b"]))` | `{"cat": {"$in": ["a", "b"]}}` |
| **Numeric Range** | `where={"year": {"$gte": 2023}}` | `FieldCondition(key="year", range=Range(gte=2023))` | `{"year": {"$gte": 2023}}` |
| **Logical AND** | `where={"$and": [cond1, cond2]}` | `Filter(must=[cond1, cond2])` | `{"$and": [cond1, cond2]}` |
| **Logical OR** | `where={"$or": [cond1, cond2]}` | `Filter(should=[cond1, cond2])` | `{"$or": [cond1, cond2]}` |
""")

st.divider()

# Section 2: Production Code: Composite Metadata Filtering Engine
st.header("2. Production Code: Composite Boolean Filter Engine", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to compile and evaluate complex boolean filter expressions (`$and`, `$or`, `$eq`, `$in`, `$gte`) over a document corpus.
""")

st.code(r'''
from typing import List, Dict, Any

class MetadataFilterEvaluator:
    """
    Evaluates nested boolean filter dictionaries over metadata payloads.
    Compatible with ChromaDB and MongoDB-style query filters.
    """
    @classmethod
    def evaluate(cls, metadata: Dict[str, Any], filter_expr: Dict[str, Any]) -> bool:
        if not filter_expr:
            return True

        for key, value in filter_expr.items():
            if key == "$and":
                if not all(cls.evaluate(metadata, sub) for sub in value):
                    return False
            elif key == "$or":
                if not any(cls.evaluate(metadata, sub) for sub in value):
                    return False
            elif isinstance(value, dict):
                # Operator evaluation on field 'key'
                field_val = metadata.get(key)
                for op, target in value.items():
                    if op == "$eq" and field_val != target:
                        return False
                    elif op == "$ne" and field_val == target:
                        return False
                    elif op == "$gt" and not (field_val is not None and field_val > target):
                        return False
                    elif op == "$gte" and not (field_val is not None and field_val >= target):
                        return False
                    elif op == "$lt" and not (field_val is not None and field_val < target):
                        return False
                    elif op == "$lte" and not (field_val is not None and field_val <= target):
                        return False
                    elif op == "$in" and field_val not in target:
                        return False
                    elif op == "$nin" and field_val in target:
                        return False
            else:
                # Direct equality shorthand: {"department": "finance"}
                if metadata.get(key) != value:
                    return False

        return True

    @classmethod
    def filter_corpus(cls, corpus: List[Dict[str, Any]], filter_expr: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [doc for doc in corpus if cls.evaluate(doc.get("metadata", {}), filter_expr)]
''', language="python")

st.divider()

# Section 3: Interactive Multi-Dimension Metadata Filter Studio
st.header("3. Interactive Workbench: Live Multi-Dimension Filter Studio", divider="rainbow")

st.markdown("""
Test how composite filters slice through an enterprise multi-tenant corpus. Adjust tenant boundaries, categories, authors, and dates to see real-time filtering in action.
""")

# Sample enterprise corpus
enterprise_corpus = [
    {"id": "doc_01", "tenant": "acme_corp", "dept": "finance", "category": "tax", "year": 2024, "author": "alice", "text": "Corporate federal tax filings for fiscal year 2023 have been audited."},
    {"id": "doc_02", "tenant": "acme_corp", "dept": "engineering", "category": "architecture", "year": 2024, "author": "bob", "text": "Microservices communicate via gRPC over internal Kubernetes service mesh."},
    {"id": "doc_03", "tenant": "acme_corp", "dept": "hr", "category": "benefits", "year": 2023, "author": "carol", "text": "Comprehensive dental and vision benefits apply to all full-time employees."},
    {"id": "doc_04", "tenant": "globex_inc", "dept": "finance", "category": "tax", "year": 2024, "author": "david", "text": "Globex international VAT consolidation schedule and European remittance."},
    {"id": "doc_05", "tenant": "globex_inc", "dept": "engineering", "category": "security", "year": 2023, "author": "eve", "text": "Automated vulnerability scanning with Snyk integrated into CI/CD pipelines."},
    {"id": "doc_06", "tenant": "globex_inc", "dept": "hr", "category": "benefits", "year": 2022, "author": "frank", "text": "Standard paid time off policy entitles employees to fifteen vacation days."},
    {"id": "doc_07", "tenant": "initech", "dept": "engineering", "category": "architecture", "year": 2024, "author": "grace", "text": "TPS report automation scripts migrated from legacy Perl to Python 3.11."},
    {"id": "doc_08", "tenant": "initech", "dept": "finance", "category": "budget", "year": 2023, "author": "heidi", "text": "Budget approvals exceeding five thousand dollars require vice president sign-off."}
]

col_f1, col_f2 = st.columns([1, 1])

with col_f1:
    st.subheader("1. Multi-Tenant Security & Categorical Filters")
    sel_tenant = st.selectbox("1. Tenant Isolation (`tenant`):", ["all", "acme_corp", "globex_inc", "initech"], index=1)
    sel_dept = st.multiselect("2. Departments (`dept` IN):", ["finance", "engineering", "hr"], default=["finance", "engineering"])
    sel_year = st.slider("3. Minimum Year (`year` >=):", min_value=2022, max_value=2024, value=2023)
    sel_cat = st.multiselect("4. Categories (`category` IN):", ["tax", "architecture", "benefits", "security", "budget"], default=["tax", "architecture"])

with col_f2:
    st.subheader("2. Generated Filter Predicate & Matched Records")

    # Build filter expression
    and_conditions = []
    if sel_tenant != "all":
        and_conditions.append({"tenant": {"$eq": sel_tenant}})
    if sel_dept:
        and_conditions.append({"dept": {"$in": sel_dept}})
    and_conditions.append({"year": {"$gte": sel_year}})
    if sel_cat:
        and_conditions.append({"category": {"$in": sel_cat}})

    filter_dict = {"$and": and_conditions} if len(and_conditions) > 1 else (and_conditions[0] if and_conditions else {})

    st.markdown("**Generated Vector Database Query Filter:**")
    st.json(filter_dict)

    # Execute filtering
    matched = []
    for doc in enterprise_corpus:
        match = True
        if sel_tenant != "all" and doc["tenant"] != sel_tenant:
            match = False
        if sel_dept and doc["dept"] not in sel_dept:
            match = False
        if doc["year"] < sel_year:
            match = False
        if sel_cat and doc["category"] not in sel_cat:
            match = False
        if match:
            matched.append(doc)

    st.metric("Total Matched Documents", f"{len(matched)} of {len(enterprise_corpus)}")

    if matched:
        df_match = pd.DataFrame(matched)
        st.dataframe(df_match[["id", "tenant", "dept", "category", "year", "author"]], use_container_width=True)
        for m in matched:
            st.caption(f"**[{m['id']}]**: {m['text']}")
    else:
        st.warning("Zero documents matched this combination of filter criteria! (High selectivity).")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Filtering Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Post-Filtering catastrophic in a multi-tenant enterprise RAG system?**
    * **Answer**: In a multi-tenant system hosting 1,000 customers, Tenant A's documents might represent 0.1% of the total vector database.
    If you use Post-Filtering:
    1. The vector search engine queries the top $K=50$ global nearest neighbors across all 1,000 tenants.
    2. All 50 returned documents belong to other tenants.
    3. The application post-filters with `tenant_id == 'tenant_a'`, eliminating all 50 items.
    4. The user receives **0 results** and the LLM hallucinates, even though Tenant A has dozens of relevant documents in the database!
    Single-stage filtered search solves this by restricting graph exploration strictly to Tenant A's nodes.

    **Q2: What is the 'Index Disconnection' trap during Single-Stage Filtered HNSW search?**
    * **Answer**: If a metadata filter has very high selectivity (e.g. matching only 10 out of 1,000,000 vectors), the allowed nodes might be isolated islands scattered across the HNSW graph. During greedy graph routing, the search algorithm can hit a dead-end because none of the current node's immediate graph neighbors satisfy the filter. Modern vector databases mitigate this by automatically falling back to an unindexed brute-force scan over the tiny matched payload bitmask when selectivity $< 1\\%$.

    **Q3: How does Qdrant's payload index architecture differ from ChromaDB's SQLite filtering?**
    * **Answer**: ChromaDB delegates metadata filtering to an embedded SQLite engine that executes SQL `WHERE` queries to gather candidate IDs. Qdrant builds dedicated in-memory payload indexes (numeric range trees, keyword bitsets, geo-spatial bounding boxes) tightly integrated into its Rust core, eliminating database IPC overhead.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Filtering on High-Cardinality Unindexed Fields**:
      * *Issue*: Filtering on `user_session_id` or random UUIDs without creating a secondary payload index.
      * *Consequence*: The database must perform a full sequential scan of all metadata records before vector traversal.
      * *Fix*: In databases like Qdrant or Milvus, explicitly define payload schema indexes for heavily filtered fields (`create_payload_index(field_name="user_session_id")`).

    * **Trap 2: Date Comparisons with Incompatible String Formats**:
      * *Issue*: Comparing `date >= "05/12/2024"` against ISO `2024-05-12`.
      * *Consequence*: String lexicographical sorting evaluates `"05"` as earlier than `"10"` (October), producing corrupted temporal filter results.
      * *Fix*: Always store dates in strict ISO-8601 (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`) or Unix epoch integers.

    * **Trap 3: Empty Result Sets Leading to Total Pipeline Silent Failure**:
      * *Issue*: User query passes a date filter so restrictive that zero documents are retrieved, passing empty context to the LLM.
      * *Consequence*: The LLM either hallucinates answers or generates confusing excuses.
      * *Fix*: Implement fallback logic: if a filtered search yields 0 candidates, notify the user that their filter was too strict and suggest broadening it.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Single-Stage Filtered Search**
      * *Definition*: Nearest neighbor search that navigates graph edges strictly through nodes satisfying metadata filter bitmasks.
    * **Flashcard 2: Post-Filtering**
      * *Definition*: The flawed practice of performing unrestricted vector search and then dropping non-matching metadata rows in Python.
    * **Flashcard 3: Filter Selectivity**
      * *Definition*: The percentage of the total corpus that satisfies a given filter predicate (low selectivity = small subset, high selectivity = large subset).
    * **Flashcard 4: Multi-Tenant RBAC**
      * *Definition*: Role-Based Access Control enforced at the vector database level to guarantee cryptographic separation of tenant data.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Metadata Filtering:

- [ ] I can articulate the difference between Pre-Filtering, Post-Filtering, and Single-Stage Filtering.
- [ ] I understand why Post-Filtering causes zero-result failures in multi-tenant systems.
- [ ] I know how to construct composite `$and`, `$or`, `$in`, and range queries across major vector databases.
- [ ] I understand the graph disconnection problem when filter selectivity is extremely high.
- [ ] I know how to format dates and categories to prevent filter bugs.
""")
