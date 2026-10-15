import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="9.2 Filter Operators - Vector DB Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("9.2 Filter Operators: Equality, Comparison, Set & Document Predicates")
st.caption("Chapter 9: Metadata Filtering | Domain: 4. Vector Database")
st.markdown("""
To control which vectors are eligible for retrieval, vector databases provide a rich grammar of **filter operators**. 
ChromaDB implements a JSON-based query DSL modeled after MongoDB operator semantics.

In ChromaDB, filter operations are divided into two distinct channels:
1. **Metadata Filters (`where={...}`)**: Evaluate structured scalar fields attached to vectors (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`).
2. **Document Content Filters (`where_document={...}`)**: Evaluate raw text content inside the document itself (`$contains`, `$not_contains`).

In this lesson, we break down each operator's syntax, edge cases, type coercion behavior, and build an interactive 
Filter Operator Simulator.
""")

st.divider()

# Section 1: The Operator Grammar & Syntax Specifications
st.header("1. The Operator Grammar & Syntax Specifications", divider="rainbow")

col_op1, col_op2 = col_op1, col_op2 = st.columns([1.1, 0.9])

with col_op1:
    st.markdown("""
    ### Scalar Comparison Operators (`where`)
    Every operator is formatted as a dictionary mapping a metadata key to an operator object:
    
    | Operator | Syntax | Description | Supported Types |
    | :--- | :--- | :--- | :--- |
    | **`$eq`** | `{"author": {"$eq": "Alice"}}` | Exact equality (or shorthand `{"author": "Alice"}`) | `str`, `int`, `float`, `bool` |
    | **`$ne`** | `{"status": {"$ne": "archived"}}` | Value is not equal to operand | `str`, `int`, `float`, `bool` |
    | **`$gt`** | `{"score": {"$gt": 85}}` | Strictly greater than operand | `int`, `float` |
    | **`$gte`** | `{"created_year": {"$gte": 2023}}` | Greater than or equal to operand | `int`, `float` |
    | **`$lt`** | `{"error_rate": {"$lt": 0.05}}` | Strictly less than operand | `int`, `float` |
    | **`$lte`** | `{"latency_ms": {"$lte": 150}}` | Less than or equal to operand | `int`, `float` |
    | **`$in`** | `{"dept": {"$in": ["AI", "DevOps"]}}` | Metadata value matches any element in list | List of `str`, `int`, `float` |
    | **`$nin`** | `{"env": {"$nin": ["test", "dev"]}}` | Metadata value does not appear in list | List of `str`, `int`, `float` |
    """)

with col_op2:
    st.markdown("### Operator Evaluation Dispatcher")
    op_dot = """
    digraph OperatorDispatch {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        QueryReq [label="collection.query(where={...})", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Parser [label="Chroma Filter AST Parser\\nValidate Types & Schema", fillcolor="#FEF3C7"];
        
        Equality [label="Exact Match\\n$eq / $ne / $in / $nin", fillcolor="#D1FAE5"];
        Range [label="Numeric Comparison\\n$gt / $gte / $lt / $lte", fillcolor="#E0E7FF"];
        DocFilter [label="Document Text\\n$contains / $not_contains", fillcolor="#FCE7F3"];

        CandidateSet [label="Passed Candidate ID Mask\\nFed to HNSW / Brute-Force", fillcolor="#C7D2FE", color="#4338CA", style="filled,bold"];

        QueryReq -> Parser;
        Parser -> Equality [label="scalar / list"];
        Parser -> Range [label="numeric"];
        Parser -> DocFilter [label="where_document"];
        Equality -> CandidateSet;
        Range -> CandidateSet;
        DocFilter -> CandidateSet;
    }
    """
    st.graphviz_chart(op_dot, use_container_width=False)

st.divider()

# Section 2: Comparison vs Set vs Document Text Filtering
st.header("2. Comparison vs Set vs Document Text Filtering", divider="rainbow")

col_comp1, col_comp2 = st.columns([1, 1])

with col_comp1:
    st.markdown("""
    ### Set Operators: `$in` and `$nin`
    * The `$in` operator checks whether a single metadata scalar exists inside a target set.
    * **Crucial Rule**: The metadata field on the stored document must be a **scalar** (`str` or `int`), and the operand in the query must be a **Python list**:
    ```python
    # VALID: Record has metadata {"role": "admin"}
    where={"role": {"$in": ["admin", "editor", "moderator"]}}
    ```
    * **Common Misconception**: You cannot store a list as a metadata value (e.g., `metadata={"roles": ["admin", "editor"]}`) because ChromaDB does not support array metadata values. Storing arrays in metadata raises a validation error.
    
    ### Document Text Filtering (`where_document`)
    Chroma allows filtering based on raw document strings independently of metadata:
    ```python
    # Only return documents containing the exact word 'HIPAA'
    results = collection.query(
        query_texts=["data retention guidelines"],
        where_document={"$contains": "HIPAA"},
        n_results=3
    )
    ```
    """)

with col_comp2:
    st.markdown("""
    ### Operator Behavior on Missing Metadata Keys
    What happens if you query `where={"department": {"$eq": "Finance"}}` when some documents in the collection were inserted with no `department` metadata key?
    
    * **Non-Existence Equals False**: If a document does not possess the queried metadata key, any comparison operator (`$eq`, `$gt`, `$lt`, `$in`) evaluates to **False**. The document is excluded from the results.
    * **The `$ne` Nuance**: In ChromaDB SQLite backend, querying `where={"department": {"$ne": "Finance"}}` will only match documents that **have a department key** whose value is not "Finance". Documents lacking the key entirely are typically excluded because the SQL predicate evaluates against `NULL`.
    
    ### Type Strictness & Coercion Rules
    * Comparing an integer field against a string (e.g., `where={"year": {"$gte": "2023"}}` when `year` was stored as `int(2023)`) triggers a runtime query exception or unexpected empty set.
    * Always ensure Python types in the query payload match the types passed during `collection.add()`.
    """)

st.divider()

# Section 3: Complete Operator Code Reference
st.header("3. Complete ChromaDB Operator Code Reference", divider="rainbow")

st.markdown("""
Review the exact Python snippets for all 8 metadata operators and 2 document operators:
""")

tab_code1, tab_code2, tab_code3 = st.tabs([
    "Equality & Inequality ($eq, $ne)",
    "Numeric Bounds ($gt, $gte, $lt, $lte)",
    "Set & Substring ($in, $nin, $contains)"
])

with tab_code1:
    st.code("""
# 1. Exact Equality Shorthand
res1 = collection.query(
    query_texts=["cloud migration strategy"],
    where={"tier": "enterprise"},
    n_results=5
)

# 2. Explicit $eq Operator
res2 = collection.query(
    query_texts=["cloud migration strategy"],
    where={"tier": {"$eq": "enterprise"}},
    n_results=5
)

# 3. Not-Equal ($ne)
res3 = collection.query(
    query_texts=["incident response playbook"],
    where={"status": {"$ne": "archived"}},
    n_results=5
)
""", language="python")

with tab_code2:
    st.code("""
# 4. Strict Greater Than ($gt)
res4 = collection.query(
    query_texts=["high throughput vector indexing"],
    where={"stars": {"$gt": 1000}},
    n_results=5
)

# 5. Greater Than or Equal ($gte) with Unix Timestamp
res5 = collection.query(
    query_texts=["quarterly earnings report"],
    where={"published_timestamp": {"$gte": 1704067200}},  # Jan 1, 2024
    n_results=5
)

# 6. Less Than ($lt)
res6 = collection.query(
    query_texts=["low latency edge embedding models"],
    where={"param_count_millions": {"$lt": 50}},
    n_results=5
)

# 7. Less Than or Equal ($lte)
res7 = collection.query(
    query_texts=["lightweight docker container configurations"],
    where={"image_size_mb": {"$lte": 250.0}},
    n_results=5
)
""", language="python")

with tab_code3:
    st.code("""
# 8. Set Inclusion ($in)
res8 = collection.query(
    query_texts=["kubernetes pod autoscaling"],
    where={"environment": {"$in": ["production", "staging"]}},
    n_results=5
)

# 9. Set Exclusion ($nin)
res9 = collection.query(
    query_texts=["database connection pool exhaustion"],
    where={"cluster": {"$nin": ["sandbox", "testing-cluster"]}},
    n_results=5
)

# 10. Document Content Substring Match ($contains)
res10 = collection.query(
    query_texts=["encryption at rest protocols"],
    where_document={"$contains": "AES-256"},
    n_results=5
)

# 11. Document Content Negation ($not_contains)
res11 = collection.query(
    query_texts=["API authorization token format"],
    where_document={"$not_contains": "DEPRECATED"},
    n_results=5
)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Filter Operator Playground
st.header("4. Interactive Workbench: Filter Operator Playground", divider="rainbow")

st.markdown("""
Configure individual filter operators against a live simulated enterprise catalog and watch how the candidate set 
and Chroma query payloads update in real time.
""")

# Sample catalog dataset
CATALOG_ITEMS = [
    {
        "id": "ITEM-01",
        "title": "Quantum GPU Server X100",
        "category": "hardware",
        "price": 12500.0,
        "stock": 14,
        "rating": 4.9,
        "status": "active",
        "tags": "datacenter, gpu, tensor, high-performance",
        "doc_text": "High-performance GPU cluster server designed for LLM training with 80GB HBM3 VRAM."
    },
    {
        "id": "ITEM-02",
        "title": "Edge Inference Accelerator Stick",
        "category": "hardware",
        "price": 299.0,
        "stock": 120,
        "rating": 4.2,
        "status": "active",
        "tags": "edge, embedded, usb, low-power",
        "doc_text": "USB low-power tensor processing unit for embedded vision and real-time audio models."
    },
    {
        "id": "ITEM-03",
        "title": "Enterprise Vector DB License 1Y",
        "category": "software",
        "price": 4800.0,
        "stock": 999,
        "rating": 4.7,
        "status": "active",
        "tags": "vector, database, cloud, enterprise",
        "doc_text": "Unlimited QPS enterprise vector database license with multi-tenant encryption and SLA."
    },
    {
        "id": "ITEM-04",
        "title": "Legacy Server Chassis Model B",
        "category": "hardware",
        "price": 850.0,
        "stock": 0,
        "rating": 3.8,
        "status": "discontinued",
        "tags": "rack, chassis, legacy, clearance",
        "doc_text": "Rackmount 4U server chassis. Notice: This model is DEPRECATED and out of warranty."
    },
    {
        "id": "ITEM-05",
        "title": "RAG Pipeline Orchestrator SaaS",
        "category": "software",
        "price": 120.0,
        "stock": 999,
        "rating": 4.5,
        "status": "active",
        "tags": "rag, pipeline, agent, saas",
        "doc_text": "Managed retrieval-augmented generation framework with semantic caching and reranking."
    },
    {
        "id": "ITEM-06",
        "title": "AI Ethics & Audit Consulting",
        "category": "services",
        "price": 7500.0,
        "stock": 5,
        "rating": 4.8,
        "status": "pending",
        "tags": "consulting, compliance, audit, legal",
        "doc_text": "Third-party independent algorithmic bias audit and EU AI Act compliance verification."
    }
]

col_wb_field, col_wb_op, col_wb_val = st.columns([1, 1, 1.2])

with col_wb_field:
    target_field = st.selectbox(
        "1. Select Metadata Field:",
        ["price", "stock", "rating", "category", "status", "Document Content ($contains)"]
    )

with col_wb_op:
    if target_field in ["price", "stock", "rating"]:
        selected_op = st.selectbox("2. Select Comparison Operator:", ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"])
    elif target_field in ["category", "status"]:
        selected_op = st.selectbox("2. Select Operator:", ["$eq", "$ne", "$in", "$nin"])
    else:
        selected_op = st.selectbox("2. Select Document Operator:", ["$contains", "$not_contains"])

with col_wb_val:
    if target_field == "price":
        val_input = st.number_input("3. Value:", min_value=0.0, max_value=20000.0, value=1000.0, step=100.0)
    elif target_field == "stock":
        val_input = st.number_input("3. Value:", min_value=0, max_value=1000, value=10, step=1)
    elif target_field == "rating":
        val_input = st.slider("3. Rating Value:", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    elif target_field == "category":
        if selected_op in ["$in", "$nin"]:
            val_input = st.multiselect("3. Categories:", ["hardware", "software", "services"], default=["hardware", "software"])
        else:
            val_input = st.selectbox("3. Category:", ["hardware", "software", "services"])
    elif target_field == "status":
        if selected_op in ["$in", "$nin"]:
            val_input = st.multiselect("3. Statuses:", ["active", "discontinued", "pending"], default=["active"])
        else:
            val_input = st.selectbox("3. Status:", ["active", "discontinued", "pending"])
    else:
        val_input = st.text_input("3. Target Substring:", value="LLM")

# Evaluate filter logic
filtered_items = []
generated_where_dict = {}
generated_where_doc = None

for item in CATALOG_ITEMS:
    match = False
    if target_field == "Document Content ($contains)":
        if selected_op == "$contains":
            match = str(val_input).lower() in item["doc_text"].lower()
            generated_where_doc = {"$contains": str(val_input)}
        else:
            match = str(val_input).lower() not in item["doc_text"].lower()
            generated_where_doc = {"$not_contains": str(val_input)}
    else:
        field_val = item[target_field]
        generated_where_dict = {target_field: {selected_op: val_input}}
        
        if selected_op == "$eq":
            match = field_val == val_input
        elif selected_op == "$ne":
            match = field_val != val_input
        elif selected_op == "$gt":
            match = field_val > val_input
        elif selected_op == "$gte":
            match = field_val >= val_input
        elif selected_op == "$lt":
            match = field_val < val_input
        elif selected_op == "$lte":
            match = field_val <= val_input
        elif selected_op == "$in":
            match = field_val in val_input
        elif selected_op == "$nin":
            match = field_val not in val_input

    if match:
        filtered_items.append(item)

st.divider()

col_res_tab, col_res_code = st.columns([1.3, 0.7])

with col_res_tab:
    st.subheader(f"Matching Items ({len(filtered_items)} of {len(CATALOG_ITEMS)} passed)")
    if filtered_items:
        df_out = pd.DataFrame([
            {
                "ID": it["id"],
                "Title": it["title"],
                "Category": it["category"],
                "Price ($)": it["price"],
                "Stock": it["stock"],
                "Rating": it["rating"],
                "Status": it["status"]
            }
            for it in filtered_items
        ])
        st.dataframe(df_out, use_container_width=True, hide_index=True)
    else:
        st.warning("No catalog items satisfy the current filter predicate.")

with col_res_code:
    st.subheader("Generated ChromaDB Query")
    if generated_where_doc:
        query_snippet = f"""# Query with document substring filter
results = collection.query(
    query_texts=["high performance accelerators"],
    where_document={generated_where_doc},
    n_results=5
)"""
    else:
        query_snippet = f"""# Query with metadata operator filter
results = collection.query(
    query_texts=["hardware and software solutions"],
    where={generated_where_dict},
    n_results=5
)"""
    st.code(query_snippet, language="python")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the difference between `where` and `where_document` in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **`where`** targets structured key-value metadata fields attached to vectors during insertion (e.g., `{"department": "HR"}`). It supports scalar equality, numeric comparisons (`$gt`, `$lte`), and set operations (`$in`, `$nin`).
    * **`where_document`** targets the raw text string of the document body itself. It only supports substring matching operators: `$contains` (case-sensitive or insensitive substring presence) and `$not_contains` (substring absence).
    """)

with st.expander("Q2: Why cannot the `$in` operator be used to search documents where a metadata key holds a list of tags?"):
    st.markdown("""
    **Answer**:
    ChromaDB does not permit metadata fields to store lists or arrays (e.g., `metadata={"tags": ["python", "ml"]}` is rejected at insert time). Metadata values must be primitive scalars (`str`, `int`, `float`, `bool`).
    
    Therefore, the `$in` operator in ChromaDB works in reverse: the **document metadata value is a single scalar**, and the **query operand is a list of acceptable candidate scalars** (`where={"tag": {"$in": ["python", "ml"]}}`). If a document must have multiple tags, tags should be formatted as comma-separated strings and filtered via `$contains` on document text, or exploded into relational junction tables.
    """)

with st.expander("Q3: How does ChromaDB handle floating-point precision comparisons with `$eq`?"):
    st.markdown("""
    **Answer**:
    Due to IEEE-754 floating-point representation, exact equality (`$eq`) on floats is hazardous in database queries (e.g., `0.1 + 0.2 != 0.3`). 
    
    In production:
    1. Avoid `$eq` on continuous floats; use small epsilon ranges (`{"$gte": 0.299, "$lte": 0.301}`) or integer representation (e.g., storing cents `$12.50` as integer `1250`).
    2. Reserve `$eq` for integers, enum strings, and booleans.
    """)

with st.expander("Q4: What is the performance penalty of `$not_contains` and `$nin` in high-throughput vector queries?"):
    st.markdown("""
    **Answer**:
    Negative operators (`$ne`, `$nin`, `$not_contains`) are inherently un-indexable or exhibit poor selectivity:
    1. **B-Tree Inefficiency**: While `$eq` can jump directly to a B-tree leaf node, `$ne` or `$nin` typically forces a full index scan over all non-matching keys.
    2. **Graph Traversal Obstruction**: In single-stage HNSW filtering, if 98% of nodes match a `$ne` condition, overhead is low. But if the condition excludes the dense central cluster of the graph, navigation hops must detour through peripheral nodes, increasing search latency.
    """)

with st.expander("Q5: Can you combine `$gt` and `$lt` on the same metadata field in a single `where` clause without an `$and` block?"):
    st.markdown("""
    **Answer**:
    In ChromaDB, specifying multiple operator conditions on the same field requires an explicit **`$and`** logical block:
    ```python
    # REQUIRED SYNTAX IN CHROMA:
    where={
        "$and": [
            {"price": {"$gt": 100}},
            {"price": {"$lt": 500}}
        ]
    }
    ```
    Placing both in a single dictionary like `{"price": {"$gt": 100, "$lt": 500}}` is rejected by ChromaDB's strict AST validator.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: The Multi-Key Implicit AND Illusion**  
    Passing `where={"dept": "AI", "status": "active"}` will raise a Chroma `ValueError: Expected where to have exactly 1 operator`.  
    **Fix**: Multiple conditions MUST be wrapped inside an explicit `$and` list: `where={"$and": [{"dept": "AI"}, {"status": "active"}]}`.
    """)
    st.warning("""
    **Trap 2: Case Mismatch in `$contains` Document Filters**  
    Chroma's `$contains` operator performs exact literal substring checks. If the document has "Kubernetes" and you query `{"$contains": "kubernetes"}`, it may fail to match depending on backend SQLite collation.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Passing Empty Lists to `$in`**  
    `where={"category": {"$in": []}}` causes syntax validation errors in ChromaDB. Always guard dynamic UI filters: `if selected_categories: where={"category": {"$in": selected_categories}}`.
    """)
    st.warning("""
    **Trap 4: String Comparisons on Numeric Strings**  
    If numbers are stored as strings (`"price": "95"` and `"price": "120"`), querying `{"$gt": "100"}` does lexicographical comparison where `"95" > "100"` is True! Always store numbers as `int` or `float`.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does `$in` expect in Chroma?"):
        st.markdown("""
        **`$in` Operand**:  
        A Python list of scalar values: `{"dept": {"$in": ["Eng", "Sales"]}}`. The document metadata field must be a scalar.
        """)
    with st.expander("🗂️ Flashcard: What does `where_document` filter?"):
        st.markdown("""
        **Document Text**:  
        Raw string body of the chunk using `$contains` or `$not_contains` substrings.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to do range bounds on one field?"):
        st.markdown("""
        **Range Syntax**:  
        Wrap in `$and`: `{"$and": [{"price": {"$gte": 50}}, {"price": {"$lte": 200}}]}`.
        """)
    with st.expander("🗂️ Flashcard: What happens if a doc lacks the metadata key?"):
        st.markdown("""
        **Missing Keys**:  
        Evaluates to False for all comparison operators; the document is excluded from results.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know all 8 scalar metadata operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.", value=True)
chk2 = st.checkbox("I understand the difference between `where` (metadata) and `where_document` (chunk text).", value=True)
chk3 = st.checkbox("I know why ChromaDB requires explicit `$and` blocks when applying range bounds to a single field.", value=True)
chk4 = st.checkbox("I know how the `$in` operator functions with scalar document metadata values.", value=True)
chk5 = st.checkbox("I understand why negative operators ($ne, $nin, $not_contains) incur performance overhead.", value=True)
chk6 = st.checkbox("I can implement robust, type-safe ChromaDB filter payloads in Python.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Filter Operators and are ready for Chapter 9.3: Logical Filtering ($and, $or)!")
