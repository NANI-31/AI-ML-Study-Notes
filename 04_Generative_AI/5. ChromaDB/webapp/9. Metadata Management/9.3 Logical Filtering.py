import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="9.3 Logical Filtering - ChromaDB Curriculum",
    page_icon="🔀",
    layout="wide"
)

# Title & Overview
st.title("9.3 Logical Filtering: $and, $or, Nested Expression Trees & Complex Query Logic")
st.caption("Chapter 9: Metadata Management | Domain: 5. ChromaDB")
st.markdown("""
Real-world enterprise retrieval policies rarely depend on a single equality check. 
Consider a multi-tenant corporate knowledge base:
* *A user can access documents if `(tenant == 'org_alpha' AND department == 'engineering') OR (is_public == True AND year >= 2024)`.*

To express sophisticated business rules, ChromaDB provides **logical composition operators**:
* **Conjunction (`$and`)**: All listed condition clauses must evaluate to `True`.
* **Disjunction (`$or`)**: At least one listed condition clause must evaluate to `True`.
* **Nested Abstract Syntax Trees (AST)**: Composing `$and` inside `$or` (and vice-versa) to model arbitrary boolean logic.
* **Strict List-of-Dicts Grammar**: ChromaDB's schema validator strictly requires that `$and` and `$or` receive a **list of single-clause dictionaries**, preventing ambiguous query semantics.

In this lesson, we master the formal grammar of logical filtering, build complex nested boolean queries in Python, 
and test an interactive Visual Boolean Logic Tree Builder & AST Inspector.
""")

st.divider()

# Section 1: The Formal Grammar of $and and $or
st.header("1. The Formal Grammar of $and and $or", divider="rainbow")

col_gram1, col_gram2 = col_gram1, col_gram2 = st.columns([1.1, 0.9])

with col_gram1:
    st.markdown("""
    ### Strict List-of-Dicts Schema Invariant
    
    * In ChromaDB, logical operators **always expect a Python list of condition dictionaries**:
      * `$and: List[Dict[str, Any]]`
      * `$or: List[Dict[str, Any]]`
    
    * **The Common Syntax Trap**:
      * ❌ **WRONG**: `{"$and": {"env": "prod", "year": 2024}}`  
        *(Raises `SchemaValidationError` — passing a dictionary to `$and` is forbidden!)*
      * ✅ **CORRECT**: `{"$and": [{"env": "prod"}, {"year": 2024}]}`  
        *(Each condition must be an individual dictionary element inside the list!)*
    
    * **Minimum Operands**:
      * Both `$and` and `$or` require **at least 2 conditions** in the list. Passing a single condition inside `$and` is redundant and rejected.
    """)

with col_gram2:
    st.markdown("### Boolean Logic Tree Architecture")
    logic_tree_dot = """
    digraph LogicTree {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RootOr [label="$or (Root Disjunction)\\nMatches Branch 1 OR Branch 2", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        Branch1 [label="$and (Branch 1)\\nInternal Engineering Runbooks", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Branch2 [label="$and (Branch 2)\\nPublic Certified Articles", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Cond1 [label="tenant == 'internal'", fillcolor="#D1FAE5", color="#059669"];
        Cond2 [label="dept == 'engineering'", fillcolor="#D1FAE5", color="#059669"];

        Cond3 [label="is_public == True", fillcolor="#A7F3D0", color="#059669"];
        Cond4 [label="year >= 2024", fillcolor="#A7F3D0", color="#059669"];

        RootOr -> Branch1;
        RootOr -> Branch2;
        Branch1 -> Cond1;
        Branch1 -> Cond2;
        Branch2 -> Cond3;
        Branch2 -> Cond4;
    }
    """
    st.graphviz_chart(logic_tree_dot, use_container_width=False)

st.divider()

# Section 2: Constructing Deeply Nested Boolean Expression Trees
st.header("2. Constructing Deeply Nested Boolean Expression Trees", divider="rainbow")

col_nest1, col_nest2 = col_nest1, col_nest2 = st.columns([1, 1])

with col_nest1:
    st.markdown("""
    ### Modeling Complex Real-World Access Control
    * Suppose an enterprise knowledge assistant must enforce:
      1. SRE team members can see anything in `tier: "critical"`.
      2. Regular developers can only see non-critical services within their own department.
    
    * In Boolean Logic:
      `tier == "critical" OR (role == "developer" AND tier != "critical" AND dept == user_dept)`
    
    * **ChromaDB Abstract Syntax Tree (AST)**:
      * Root is an `$or` block.
      * Left branch is a simple equality filter.
      * Right branch is an `$and` block containing three distinct conditions.
    """)

with col_nest2:
    st.markdown("### Clean Python Implementation of Nested AST")
    st.code("""where_clause = {
    "$or": [
        # Branch 1: Critical tier accessible to authorized roles
        {"tier": "critical"},
        
        # Branch 2: General developer access with department isolation
        {
            "$and": [
                {"role": "developer"},
                {"tier": {"$ne": "critical"}},
                {"dept": "backend"}
            ]
        }
    ]
}

results = collection.query(
    query_texts=["database disaster recovery"],
    n_results=5,
    where=where_clause
)
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Complex Logical Queries
st.header("3. Clean Python Implementation of Complex Logical Queries", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("enterprise_kb")

# Example 1: Multi-Condition Filter with $and
# Retrieve high-priority, modern runbooks in production
prod_sre_filter = {
    "$and": [
        {"env": "production"},
        {"priority": {"$lte": 2}},
        {"year": {"$gte": 2024}}
    ]
}

res_prod = collection.query(
    query_texts=["Kubernetes ingress gateway 502 Bad Gateway"],
    n_results=3,
    where=prod_sre_filter
)

# Example 2: Multi-Tenant Cohort with $or and $in
# Match documents belonging to tenant A OR documents in public beta with high review score
cohort_filter = {
    "$or": [
        {"tenant_id": "org_alpha"},
        {
            "$and": [
                {"is_public_beta": True},
                {"review_score": {"$gte": 4.5}}
            ]
        }
    ]
}

res_cohort = collection.get(
    where=cohort_filter,
    limit=50
)
print(f"Retrieved {len(res_cohort['ids'])} compliant cohort documents.")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Visual Boolean Logic Tree Builder & AST Inspector
st.header("4. Interactive Workbench: Visual Boolean Logic Tree Builder & AST Inspector", divider="rainbow")

st.markdown("""
Construct a nested boolean filter interactively. Compose Branch 1 and Branch 2 under a root logical operator, 
inspect the generated ChromaDB JSON AST, and observe which documents match in real time.
""")

col_wb_root, col_wb_b1, col_wb_b2 = st.columns([0.8, 1.1, 1.1])

with col_wb_root:
    st.subheader("Root Logic")
    root_operator = st.radio("Combine Branches With:", ["$or (Union)", "$and (Intersection)"])
    root_op_token = "$or" if "$or" in root_operator else "$and"

with col_wb_b1:
    st.subheader("Branch 1 Conditions ($and)")
    b1_dept = st.selectbox("Branch 1 Dept:", ["engineering", "security", "infra"])
    b1_env = st.selectbox("Branch 1 Env:", ["production", "staging"])

with col_wb_b2:
    st.subheader("Branch 2 Conditions ($and)")
    b2_public = st.checkbox("Branch 2: is_public == True", value=True)
    b2_min_year = st.slider("Branch 2: Year >= ", min_value=2021, max_value=2025, value=2024)

# Build AST
branch1_dict = {
    "$and": [
        {"dept": b1_dept},
        {"env": b1_env}
    ]
}

branch2_dict = {
    "$and": [
        {"is_public": b2_public},
        {"year": {"$gte": b2_min_year}}
    ]
}

full_ast_where = {
    root_op_token: [branch1_dict, branch2_dict]
}

# Mock Dataset
MOCK_LOGIC_DATASET = [
    {"id": "doc_01", "dept": "engineering", "env": "production", "is_public": False, "year": 2023, "title": "Internal Payment Gateway Architecture"},
    {"id": "doc_02", "dept": "engineering", "env": "staging", "is_public": True, "year": 2024, "title": "Public API Sandbox Documentation"},
    {"id": "doc_03", "dept": "security", "env": "production", "is_public": False, "year": 2024, "title": "Zero-Trust Perimeter Firewall Rules"},
    {"id": "doc_04", "dept": "infra", "env": "staging", "is_public": True, "year": 2025, "title": "Open Source Terraform Modules Guide"},
    {"id": "doc_05", "dept": "infra", "env": "production", "is_public": True, "year": 2022, "title": "Historical Kubernetes Post-Mortem"}
]

# Evaluate logic
matched_records = []
for doc in MOCK_LOGIC_DATASET:
    # Eval Branch 1
    m_b1 = (doc["dept"] == b1_dept) and (doc["env"] == b1_env)
    # Eval Branch 2
    m_b2 = (doc["is_public"] == b2_public) and (doc["year"] >= b2_min_year)
    
    if root_op_token == "$or":
        is_hit = m_b1 or m_b2
    else:
        is_hit = m_b1 and m_b2
        
    if is_hit:
        matched_records.append(doc)

st.divider()

col_ast_view, col_match_view = col_ast_view, col_match_view = st.columns([1, 1.2])

with col_ast_view:
    st.subheader("ChromaDB AST `where` Payload")
    st.json(full_ast_where)
    st.caption("Notice how each `$and` and `$or` contains a clean list of condition dictionaries.")

with col_match_view:
    st.subheader(f"Matching Results ({len(matched_records)} / {len(MOCK_LOGIC_DATASET)})")
    if matched_records:
        st.dataframe(pd.DataFrame(matched_records), use_container_width=True, hide_index=True)
        st.success(f"🟢 Successfully resolved complex boolean filter with {len(matched_records)} matching records!")
    else:
        st.warning("No records matched the compound boolean predicate.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the syntax requirement for `$and` and `$or` in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * `$and` and `$or` strictly require a **Python list of dictionaries**: `List[Dict[str, Any]]`.
    * Passing a single dictionary with multiple keys (e.g. `{"$and": {"a": 1, "b": 2}}`) will fail validation.
    * The correct syntax is: `{"$and": [{"a": 1}, {"b": 2}]}`.
    * The list must contain **at least 2 conditions**.
    """)

with st.expander("Q2: Can you nest `$and` inside `$or` in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **Yes**. ChromaDB supports arbitrary nesting of logical operators:
      ```python
      where = {
          "$or": [
              {"$and": [{"dept": "eng"}, {"env": "prod"}]},
              {"$and": [{"dept": "sec"}, {"role": "admin"}]}
          ]
      }
      ```
    * Chroma parses this Abstract Syntax Tree (AST) into SQL boolean expressions (`(A AND B) OR (C AND D)`) evaluated by SQLite.
    """)

with st.expander("Q3: Does ChromaDB have a `$not` logical operator?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does **not** provide a top-level unary `$not` block.
    * Instead, negation is achieved through:
      1. Field-level inequality: `{"status": {"$ne": "archived"}}`.
      2. Set-level exclusion: `{"status": {"$nin": ["archived", "deleted"]}}`.
      3. Document content exclusion: `where_document={"$not_contains": "draft"}`.
    """)

with st.expander("Q4: How do logical metadata filters interact with `where_document` content filters?"):
    st.markdown("""
    **Answer**:
    * `where` (metadata filter) and `where_document` (content filter) are evaluated as an **implicit `$and`**.
    * If both are provided:
      `collection.query(where={"env": "prod"}, where_document={"$contains": "kubernetes"})`
    * A document must satisfy **both** the metadata criteria AND contain the keyword `"kubernetes"` to be returned.
    """)

with st.expander("Q5: What is the performance impact of deeply nested `$or` queries on large collections?"):
    st.markdown("""
    **Answer**:
    * `$or` queries force SQLite to perform a union of multiple index scans or fall back to a full table scan if non-indexed columns are involved.
    * For optimal performance on collections with millions of vectors:
      1. Keep nesting depth under 3 levels.
      2. Use `$in` instead of multiple `$or` equality checks on the same key (`{"status": {"$in": ["a", "b"]}}` is faster than `{"$or": [{"status": "a"}, {"status": "b"}]}`).
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Dicts Instead of Lists to `$and` / `$or`**  
    Writing `{"$and": {"a": 1, "b": 2}}` is the #1 ChromaDB bug. Always use a list of single-key dicts: `[{"a": 1}, {"b": 2}]`.
    """)
    st.warning("""
    **Trap 2: Providing Fewer Than 2 Conditions to `$and`**  
    Writing `{"$and": [{"status": "active"}]}` raises a validation error. If you have only one condition, write `{"status": "active"}` directly.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Chained `$or` Instead of `$in` for Same-Field Checks**  
    Writing `{"$or": [{"dept": "eng"}, {"dept": "sec"}]}` is verbose and slower than writing `{"dept": {"$in": ["eng", "sec"]}}`.
    """)
    st.warning("""
    **Trap 4: Forgetting Case Sensitivity in String Values**  
    ChromaDB metadata string comparisons are strictly case-sensitive. `{"dept": "Engineering"}` will not match `"engineering"`. Standardize case on ingestion.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What structure does $and expect?"):
        st.markdown("""
        **List of Dicts**:  
        `{"$and": [{"key1": val1}, {"key2": val2}]}`.
        """)
    with st.expander("🗂️ Flashcard: Can $and be nested inside $or?"):
        st.markdown("""
        **Yes, Unlimited Nesting**:  
        Forms valid boolean AST expression trees.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Is there a top-level $not?"):
        st.markdown("""
        **No**:  
        Use `$ne`, `$nin`, or `$not_contains` instead.
        """)
    with st.expander("🗂️ Flashcard: How does where relate to where_document?"):
        st.markdown("""
        **Implicit $and**:  
        Both must evaluate to `True` for a candidate to match.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the strict list-of-dicts schema requirement for `$and` and `$or` operators.", value=True)
chk2 = st.checkbox("I know how to construct nested boolean AST expression trees combining `$and` and `$or`.", value=True)
chk3 = st.checkbox("I know how to implement negation using `$ne`, `$nin`, and `$not_contains` without a top-level `$not`.", value=True)
chk4 = st.checkbox("I understand why `$in` is preferred over multiple `$or` clauses on the same attribute.", value=True)
chk5 = st.checkbox("I know that `where` and `where_document` combine via an implicit `$and` conjunction.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 10: Document Filtering in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 9: Metadata Management in ChromaDB!")
