import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import json
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="9.3 Logical Filtering - Vector DB Curriculum",
    page_icon="🔀",
    layout="wide"
)

# Title & Overview
st.title("9.3 Logical Filtering: Combining Conditions with $and, $or, and Nested Logic")
st.caption("Chapter 9: Metadata Filtering | Domain: 4. Vector Database")
st.markdown("""
Real-world enterprise retrieval requires expressing complex business policies:
* *"Retrieve documents from either the Legal OR Compliance department, but ONLY IF their classification is Public AND published after 2022."*
* *"Search across Tenant 42's data WHERE the user has role 'Admin' OR the document is explicitly flagged 'Public'."*

In ChromaDB, expressing these rules requires **Logical Combinators** (`$and`, `$or`). 
Because ChromaDB adheres to a strict Single-Operator Abstract Syntax Tree (AST), understanding how to construct, 
nest, and optimize `$and` and `$or` query blocks is mandatory for enterprise AI engineering.

In this lesson, we master Chroma's logical filtering specification, dissect tree evaluation semantics, explore 
nested boolean hierarchies, and build an interactive Boolean Filter Visualizer.
""")

st.divider()

# Section 1: The Single-Operator Rule and Logical Combinator Grammar
st.header("1. The Single-Operator Rule & Combinator Grammar", divider="rainbow")

col_comb1, col_comb2 = st.columns([1.1, 0.9])

with col_comb1:
    st.markdown("""
    ### The Fundamental ChromaDB Single-Operator Rule
    Unlike MongoDB or raw SQL where multiple conditions can be placed side-by-side in a dictionary, 
    **ChromaDB requires that every `where` dictionary contains exactly ONE top-level key**:

    #### ❌ The Invalid Multi-Key Syntax
    ```python
    # THIS WILL RAISE A RUNTIME ERROR IN CHROMA:
    where = {
        "department": "Engineering",
        "status": "active"
    }
    # ValueError: Expected where to have exactly 1 operator, got 2
    ```

    #### ✅ The Valid Single-Operator `$and` Syntax
    ```python
    # CORRECT CHROMA SYNTAX:
    where = {
        "$and": [
            {"department": "Engineering"},
            {"status": "active"}
        ]
    }
    ```

    ### The Two Logical Operators
    1. **`$and`**: Takes a list of 2 or more condition dictionaries. Evaluates to True only if **all** child conditions evaluate to True.
    2. **`$or`**: Takes a list of 2 or more condition dictionaries. Evaluates to True if **at least one** child condition evaluates to True.
    """)

with col_comb2:
    st.markdown("### Boolean Logic AST Hierarchy")
    tree_dot = """
    digraph BooleanAST {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RootAnd [label="where = {'$and': [...]}\\nTop-Level Conjunction", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        TenantCond [label="tenant_id == 'T-104'\\nExact Equality", fillcolor="#D1FAE5", color="#059669"];
        SubOr [label="{'$or': [...]}\\nDisjunction Sub-Tree", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        AdminCond [label="role == 'admin'", fillcolor="#FCE7F3"];
        PublicCond [label="is_public == True", fillcolor="#FCE7F3"];

        DocPass [label="Passed Candidate Vector\\nEligible for HNSW Search", fillcolor="#C7D2FE", color="#4338CA", style="filled,bold"];

        RootAnd -> TenantCond;
        RootAnd -> SubOr;
        SubOr -> AdminCond;
        SubOr -> PublicCond;

        TenantCond -> DocPass;
        SubOr -> DocPass;
    }
    """
    st.graphviz_chart(tree_dot, use_container_width=False)

st.divider()

# Section 2: Deep Dive into Nested Filtering Hierarchies
st.header("2. Deep Dive into Nested Filtering Hierarchies", divider="rainbow")

st.markdown("""
Real-world authorization and business filtering frequently require nesting `$or` inside `$and`, or `$and` inside `$or`.
""")

col_nest1, col_nest2 = col_nest1, col_nest2 = st.columns([1, 1])

with col_nest1:
    st.markdown("""
    ### Pattern A: Conjunction of Disjunctions (`(A OR B) AND C`)
    * **Use Case**: Security clearance and multi-department collaboration.
    * *"Find documents belonging to either the DevOps OR Security teams, but ONLY IF their classification is internal."*
    * **Structure**:
    ```python
    where = {
        "$and": [
            {
                "$or": [
                    {"department": "DevOps"},
                    {"department": "Security"}
                ]
            },
            {
                "classification": "internal"
            }
        ]
    }
    ```
    * **Equivalent SQL**:  
      `WHERE (department = 'DevOps' OR department = 'Security') AND classification = 'internal'`
    """)

with col_nest2:
    st.markdown("""
    ### Pattern B: Disjunction of Conjunctions (`(A AND B) OR (C AND D)`)
    * **Use Case**: Multi-tier access control matrix.
    * *"Allow access if user belongs to Tenant A with Tier Pro, OR if user belongs to Tenant B with Tier Enterprise."*
    * **Structure**:
    ```python
    where = {
        "$or": [
            {
                "$and": [
                    {"tenant": "acme"},
                    {"tier": "pro"}
                ]
            },
            {
                "$and": [
                    {"tenant": "globex"},
                    {"tier": "enterprise"}
                ]
            }
        ]
    }
    ```
    * **Equivalent SQL**:  
      `WHERE (tenant = 'acme' AND tier = 'pro') OR (tenant = 'globex' AND tier = 'enterprise')`
    """)

st.divider()

# Section 3: Combining Metadata Logic with Document Text Filters
st.header("3. Combining Metadata Logic with Document Text Filters", divider="rainbow")

col_both1, col_both2 = st.columns([1, 1])

with col_both1:
    st.markdown("""
    ### Dual-Predicate Queries: `where` + `where_document`
    ChromaDB allows simultaneously specifying a **logical metadata filter** AND a **document text filter** in the same query.
    
    * **Evaluation Order**: Both filters must pass for a document to be eligible for nearest neighbor ranking:
      ```text
      Eligible(d) = Evaluate(where, d.metadata) AND Evaluate(where_document, d.document_text)
      ```
    * **Production Example**:
    ```python
    results = collection.query(
        query_texts=["API rate limiting configuration"],
        where={
            "$and": [
                {"environment": "production"},
                {"tier": {"$gte": 2}}
            ]
        },
        where_document={
            "$contains": "nginx"
        },
        n_results=5
    )
    ```
    """)

with col_both2:
    st.markdown("""
    ### Document Filter Combinations (`$and` / `$or` in `where_document`)
    Just like metadata, `where_document` also supports logical combinators for text substrings:
    
    ```python
    # Document must contain EITHER 'Redis' OR 'Memcached'
    where_document={
        "$or": [
            {"$contains": "Redis"},
            {"$contains": "Memcached"}
        ]
    }
    
    # Document must contain 'OAuth2' and NOT contain 'v1_deprecated'
    where_document={
        "$and": [
            {"$contains": "OAuth2"},
            {"$not_contains": "v1_deprecated"}
        ]
    }
    ```
    """)

st.divider()

# Section 4: Interactive Workbench: Boolean Filter Visualizer & AST Builder
st.header("4. Interactive Workbench: Boolean Filter Visualizer & AST Builder", divider="rainbow")

st.markdown("""
Construct and test complex nested `$and` / `$or` filters against a live simulated enterprise repository.
""")

# Knowledge Base Records
DOC_REPOSITORY = [
    {
        "id": "SEC-01",
        "title": "Zero Trust Network Access Policy",
        "dept": "Security",
        "classification": "Confidential",
        "year": 2024,
        "region": "US",
        "verified": True,
        "text": "All endpoints must utilize mTLS certificates and biometric hardware authentication."
    },
    {
        "id": "SEC-02",
        "title": "Public Security Whitepaper 2024",
        "dept": "Security",
        "classification": "Public",
        "year": 2024,
        "region": "EU",
        "verified": True,
        "text": "Overview of infrastructure encryption standards and SOC2 Type II audit compliance."
    },
    {
        "id": "ENG-03",
        "title": "Kubernetes Ingress Architecture",
        "dept": "Engineering",
        "classification": "Internal",
        "year": 2023,
        "region": "US",
        "verified": True,
        "text": "Traffic routing via Envoy proxy with automated rate limiting and TLS termination."
    },
    {
        "id": "ENG-04",
        "title": "Engineering Intern Onboarding Notes",
        "dept": "Engineering",
        "classification": "Public",
        "year": 2021,
        "region": "APAC",
        "verified": False,
        "text": "Informal setup steps for local Docker environment and internal Slack channels."
    },
    {
        "id": "HR-05",
        "title": "Global Remote Work Guidelines",
        "dept": "HR",
        "classification": "Public",
        "year": 2024,
        "region": "EU",
        "verified": True,
        "text": "Tax and equipment stipends for full-time employees residing outside headquarters."
    },
    {
        "id": "HR-06",
        "title": "Executive Compensation Structure",
        "dept": "HR",
        "classification": "Confidential",
        "year": 2024,
        "region": "US",
        "verified": True,
        "text": "Bonus tiers and stock option vesting schedules for senior executive leadership."
    }
]

col_cfg_top, col_cfg_sub = st.columns([1, 1.2])

with col_cfg_top:
    st.subheader("1. Configure Outer Root Condition")
    root_operator = st.radio("Top-Level Combinator:", ["$and", "$or"], horizontal=True)
    
    req_year = st.slider("Condition 1: Published Year >= :", min_value=2020, max_value=2024, value=2023)
    req_verified = st.checkbox("Condition 2: Verified Only (`verified == True`)", value=True)

with col_cfg_sub:
    st.subheader("2. Configure Nested Sub-Tree")
    sub_operator = st.radio("Nested Combinator:", ["$or", "$and"], horizontal=True)
    
    col_sub_f1, col_sub_f2 = st.columns(2)
    with col_sub_f1:
        sel_depts = st.multiselect("Allowed Departments:", ["Security", "Engineering", "HR"], default=["Security", "Engineering"])
    with col_sub_f2:
        sel_class = st.multiselect("Allowed Classifications:", ["Public", "Internal", "Confidential"], default=["Public", "Internal"])

# Build Python AST Structure
sub_conditions = []
if sel_depts:
    sub_conditions.append({"dept": {"$in": sel_depts}})
if sel_class:
    sub_conditions.append({"classification": {"$in": sel_class}})

root_conditions = [
    {"year": {"$gte": req_year}}
]
if req_verified:
    root_conditions.append({"verified": True})

if sub_conditions:
    if len(sub_conditions) == 1:
        root_conditions.append(sub_conditions[0])
    else:
        root_conditions.append({sub_operator: sub_conditions})

if len(root_conditions) == 1:
    chroma_where_payload = root_conditions[0]
else:
    chroma_where_payload = {root_operator: root_conditions}

# Evaluate AST on mock data
def evaluate_condition(doc: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    for key, val in cond.items():
        if key == "$and":
            return all(evaluate_condition(doc, c) for c in val)
        elif key == "$or":
            return any(evaluate_condition(doc, c) for c in val)
        elif isinstance(val, dict):
            field_val = doc.get(key)
            for op, op_val in val.items():
                if op == "$eq": return field_val == op_val
                elif op == "$ne": return field_val != op_val
                elif op == "$gte": return field_val is not None and field_val >= op_val
                elif op == "$lte": return field_val is not None and field_val <= op_val
                elif op == "$gt": return field_val is not None and field_val > op_val
                elif op == "$lt": return field_val is not None and field_val < op_val
                elif op == "$in": return field_val in op_val
                elif op == "$nin": return field_val not in op_val
        else:
            # direct equality shorthand
            return doc.get(key) == val
    return True

matching_docs = [d for d in DOC_REPOSITORY if evaluate_condition(d, chroma_where_payload)]

st.divider()

col_disp_tree, col_disp_table = st.columns([1, 1.2])

with col_disp_tree:
    st.subheader("Generated Chroma AST (`where`)")
    st.code(json.dumps(chroma_where_payload, indent=2), language="json")
    
    st.markdown("### Execution Plan")
    st.info(f"""
    **Logic Summary**:  
    Evaluating `{root_operator}` across `{len(root_conditions)}` branch predicates.  
    Nested sub-tree combines `{len(sub_conditions)}` predicates via `{sub_operator}`.
    """)

with col_disp_table:
    st.subheader(f"Matching Results ({len(matching_docs)} of {len(DOC_REPOSITORY)} Passed)")
    if matching_docs:
        df_res = pd.DataFrame([
            {
                "ID": d["id"],
                "Title": d["title"],
                "Dept": d["dept"],
                "Class": d["classification"],
                "Year": d["year"],
                "Verified": "✅" if d["verified"] else "❌"
            }
            for d in matching_docs
        ])
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.warning("Zero documents satisfied the combined boolean logic hierarchy.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why does ChromaDB fail when two keys are passed in a single dictionary without `$and`?"):
    st.markdown("""
    **Answer**:
    In Python, dictionaries have undefined key ordering before Python 3.7 and lack an explicit semantic operator at the dictionary level. 
    To maintain a strict, unambiguous Abstract Syntax Tree (AST) that maps directly to SQL `AND` / `OR` parse trees without ambiguity, ChromaDB enforces that each dictionary contains **exactly one operator**:
    * Either a single field predicate (e.g. `{"field": "val"}` or `{"field": {"$gt": 10}}`),
    * Or an explicit logical combinator (`{"$and": [...]}` or `{"$or": [...]}`).
    
    Allowing multi-key dictionaries would create ambiguous syntax when combining custom operators with field names.
    """)

with st.expander("Q2: How does the database optimize execution order for nested `$and` and `$or` filters?"):
    st.markdown("""
    **Answer**:
    Modern vector query planners optimize boolean execution using **selectivity ordering**:
    1. **Highest Selectivity First**: Conditions that eliminate the largest percentage of candidate records (e.g., `tenant_id == 'T-88'` matching 0.1% of records) are evaluated first in an `$and` block.
    2. **Index-Backed vs Scanned Fields**: Conditions with B-Tree indices or inverted indices are evaluated before non-indexed fields.
    3. **Short-Circuit Evaluation**: In an `$and` chain, evaluation immediately stops on the first `False`. In an `$or` chain, evaluation immediately stops on the first `True`.
    """)

with st.expander("Q3: What happens if an `$or` block contains an empty list in ChromaDB?"):
    st.markdown("""
    **Answer**:
    Passing an empty list (`where={"$or": []}`) or a single-element list (`where={"$or": [{"dept": "HR"}]}`) violates ChromaDB's schema validation and raises a `ValueError: Expected $or to contain a list of at least 2 conditions`. 
    
    In application code, if dynamic UI filters can produce 0 or 1 item, you must normalize the query dictionary before calling `.query()`:
    ```python
    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$or": conditions}
    ```
    """)

with st.expander("Q4: Can you nest an `$and` inside an `$and` (or `$or` inside `$or`)?"):
    st.markdown("""
    **Answer**:
    While valid in AST syntax, nesting identical operators (`{"$and": [A, {"$and": [B, C]}]}`) is redundant because conjunction and disjunction are associative. 
    A well-architected query builder will flatten associative chains into a single flat list: `{"$and": [A, B, C]}`. Flattening simplifies AST parsing and allows database engines to reorder terms for maximum selectivity.
    """)

with st.expander("Q5: How do logical metadata filters interact with ANN graph traversal (HNSW)?"):
    st.markdown("""
    **Answer**:
    When a query includes a complex nested `$and` / `$or` filter:
    1. In **Single-Stage Filtering**, the HNSW graph traversal algorithm checks the boolean AST predicate on candidate nodes as it traverses edges. If a node satisfies the AST, it is added to the nearest-neighbor priority queue. If it does not, it may still be explored as a transit bridge to reach other nodes, but is barred from the final output.
    2. If the filter is extremely selective (<1% match rate), graph traversal may get stuck in disconnected clusters, causing the engine to dynamically fallback to pre-filtering with exact candidate scans.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Fewer Than 2 Items to `$and` / `$or`**  
    `where={"$and": [{"dept": "AI"}]}` crashes with `Expected at least 2 items`.  
    **Fix**: Always dynamically unpack single items into scalar dictionaries.
    """)
    st.warning("""
    **Trap 2: Forgetting That `$or` Expands Search Space**  
    Adding an `$or` branch with low selectivity (e.g., `{"$or": [{"tenant": "acme"}, {"is_public": True}]}`) can cause the candidate pool to balloon from 500 records to 5,000,000 records, degrading query latency.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unescaped Reserved Keywords as Field Names**  
    Naming a metadata column `$and` or `$type` corrupts Chroma's parser because the leading `$` designates operator tokens. Never prefix custom metadata keys with `$`.
    """)
    st.warning("""
    **Trap 4: Deep Recursion in Client-Generated Queries**  
    Allowing end-users to generate arbitrarily nested boolean trees (10+ levels deep) can cause stack overflows in AST parsers and slow down SQL query compilers. Cap nesting depth to 3 levels in the API gateway.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Chroma's Single-Operator Rule?"):
        st.markdown("""
        **Single-Operator Rule**:  
        Every `where` dictionary must contain exactly ONE key (either one field name or one `$and`/`$or` operator).
        """)
    with st.expander("🗂️ Flashcard: Minimum list length for `$or`?"):
        st.markdown("""
        **Min Length**:  
        `$and` and `$or` require at least **2 condition dictionaries** in their list.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to do `(A OR B) AND C`?"):
        st.markdown("""
        **Nested Syntax**:  
        `{"$and": [{"$or": [A, B]}, C]}`.
        """)
    with st.expander("🗂️ Flashcard: What if a dynamic filter has 1 item?"):
        st.markdown("""
        **Single Item Normalization**:  
        Do not wrap in `$and` or `$or`; use the item dictionary directly: `where = items[0]`.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand ChromaDB's Single-Operator AST rule.", value=True)
chk2 = st.checkbox("I can construct valid `$and` and `$or` query blocks containing 2 or more conditions.", value=True)
chk3 = st.checkbox("I know how to nest `$or` blocks inside `$and` to model real-world access control.", value=True)
chk4 = st.checkbox("I know how to combine `where` metadata logic with `where_document` text filters.", value=True)
chk5 = st.checkbox("I know how to dynamically normalize query dictionaries to prevent single-item AST crashes.", value=True)
chk6 = st.checkbox("I can implement production-grade boolean search filters with short-circuit evaluation.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Logical Filtering and are ready for Chapter 9.4: Practical Filtering Use Cases!")
