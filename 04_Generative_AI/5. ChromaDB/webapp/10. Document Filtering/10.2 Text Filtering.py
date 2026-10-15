import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.2 Text Filtering - ChromaDB Curriculum",
    page_icon="📑",
    layout="wide"
)

# Title & Overview
st.title("10.2 Text Filtering: $contains, $not_contains, Case Collation & Compound Text Rules")
st.caption("Chapter 10: Document Filtering | Domain: 5. ChromaDB")
st.markdown("""
While semantic vector embeddings excel at broad conceptual associations, production RAG pipelines 
routinely encounter queries where **exact keyword presence or exclusion is mandatory**:
* **Positive Lexical Pinning**: Ensuring that a returned runbook explicitly mentions a critical technology (e.g. `"Calico CNI"` or `"Mutual TLS"`).
* **Negative Lexical Filtering (`$not_contains`)**: Suppressing documents that contain obsolete or hazardous flags (e.g. `"DEPRECATED"`, `"LEGACY"`, `"DO NOT USE IN PROD"`).
* **Case Collation Nuances**: Understanding how SQLite handles ASCII versus Unicode casing across operating systems.
* **Compound Text Logic**: Combining `$contains` and `$not_contains` using `$and` and `$or` within `where_document`.

In this lesson, we master text filtering operators, explore case sensitivity mechanics, 
implement negative filtering pipelines in Python, and test an interactive Live Text Filter Sandbox.
""")

st.divider()

# Section 1: The Text Filtering Operator Catalog
st.header("1. The Text Filtering Operator Catalog", divider="rainbow")

col_top1, col_top2 = col_top1, col_top2 = st.columns([1.1, 0.9])

with col_top1:
    st.markdown("""
    ### `where_document` Operators
    
    * **1. Substring Containment (`$contains`)**:
      * `where_document={"$contains": "PostgreSQL"}`
      * Evaluates whether the substring `"PostgreSQL"` is present anywhere inside the document string.
    
    * **2. Substring Exclusion (`$not_contains`)**:
      * `where_document={"$not_contains": "deprecated"}`
      * Discards any document that mentions `"deprecated"`. Highly valuable for hallucination suppression!
    
    * **3. Logical Combination (`$and` & `$or`)**:
      * You can combine multiple text conditions within `where_document`:
        ```python
        where_document = {
            "$and": [
                {"$contains": "Kubernetes"},
                {"$not_contains": "Minikube"}
            ]
        }
        ```
      * Operates identically to metadata `$and`, requiring a list of single-condition dictionaries.
    """)

with col_top2:
    st.markdown("### Text Filtering Execution Flow")
    text_flow_dot = """
    digraph TextFilterFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="Inbound Query\\nquery_texts=['Cluster setup']\\nwhere_document={'\\$and': [{'\\$contains': 'EKS'}, {'\\$not_contains': 'v1beta1'}]}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        SQLiteScan [label="SQLite LIKE Substring Scan\\nFilter: docs LIKE '%EKS%' AND docs NOT LIKE '%v1beta1%'", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        GraphPrune [label="Candidate Pool Filtered\\nEliminates obsolete API versions before vector ranking", fillcolor="#EDE9FE", color="#7C3AED"];

        FinalResults [label="Pristine Modern Context\\nGuaranteed production-ready docs for LLM", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        InQuery -> SQLiteScan -> GraphPrune -> FinalResults;
    }
    """
    st.graphviz_chart(text_flow_dot, use_container_width=False)

st.divider()

# Section 2: Case Collation & Production Pitfalls in SQLite
st.header("2. Case Collation & Production Pitfalls in SQLite", divider="rainbow")

col_col1, col_col2 = col_col1, col_col2 = st.columns([1, 1])

with col_col1:
    st.markdown("""
    ### How SQLite Handles Case Collation
    * Under the hood, ChromaDB translates:
      * `{"$contains": "python"}` -> `doc_text LIKE '%python%'`
    * **The ASCII Rule in SQLite**:
      * For ASCII characters (`A-Z`, `a-z`), SQLite's default `LIKE` operator is **case-insensitive**.
      * Searching for `"python"` matches `"Python"`, `"PYTHON"`, and `"pYtHoN"`.
    * **The Non-ASCII Unicode Caveat**:
      * For non-ASCII accented characters (e.g. `é`, `ö`, `ñ`), SQLite's standard `LIKE` operator is **case-sensitive** unless a custom ICU extension is loaded!
    * **Production Recommendation**:
      * If your pipeline ingests multilingual text, normalize text to lower-case before ingestion if you require uniform case-insensitive filtering.
    """)

with col_col2:
    st.markdown("### Clean Python Implementation: Compound Text Filters")
    st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("cloud_kb")

# Compound Text Rule:
# Must contain 'Terraform' AND must NOT contain 'deprecated'
clean_docs = collection.query(
    query_texts=["Provision AWS VPC with multi-AZ subnets"],
    n_results=3,
    where_document={
        "$and": [
            {"$contains": "Terraform"},
            {"$not_contains": "deprecated"}
        ]
    }
)

for rank, (doc_id, text) in enumerate(zip(
    clean_docs["ids"][0],
    clean_docs["documents"][0]
), start=1):
    print(f"Rank #{rank} [{doc_id}]: {text[:50]}...")
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Negative Filtering for Safety
st.header("3. Clean Python Implementation of Negative Filtering for Safety", divider="rainbow")

st.code("""def safe_production_search(
    collection,
    user_query: str,
    required_keywords: List[str] = None,
    forbidden_terms: List[str] = None,
    top_k: int = 3
):
    \"\"\"Guarantees that retrieved chunks satisfy positive and negative safety constraints.\"\"\"
    text_conditions = []
    
    if required_keywords:
        for kw in required_keywords:
            text_conditions.append({"$contains": kw})
            
    if forbidden_terms:
        for term in forbidden_terms:
            text_conditions.append({"$not_contains": term})
            
    where_doc_payload = None
    if len(text_conditions) == 1:
        where_doc_payload = text_conditions[0]
    elif len(text_conditions) > 1:
        where_doc_payload = {"$and": text_conditions}
        
    return collection.query(
        query_texts=[user_query],
        n_results=top_k,
        where_document=where_doc_payload
    )
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Text Filter Sandbox
st.header("4. Interactive Workbench: Live Text Filter Sandbox", divider="rainbow")

st.markdown("""
Test `$contains` and `$not_contains` against a simulated corpus of cloud infrastructure runbooks. 
Observe how negative filtering immediately strips out legacy runbooks and prevents prompt pollution.
""")

col_wb_t1, col_wb_t2 = col_wb_t1, col_wb_t2 = st.columns(2)

with col_wb_t1:
    st.subheader("1. Positive Text Constraint ($contains)")
    pos_term = st.text_input("Must Contain Substring:", value="Terraform")
    apply_pos = st.checkbox("Enable $contains Filter", value=True)

with col_wb_t2:
    st.subheader("2. Negative Text Constraint ($not_contains)")
    neg_term = st.text_input("Must NOT Contain Substring:", value="deprecated")
    apply_neg = st.checkbox("Enable $not_contains Filter", value=True)

# Mock Dataset
TECHNICAL_CORPUS = [
    {
        "id": "runbook_tf_01",
        "doc": "Terraform AWS VPC module v5.0 implementation with public and private NAT gateways.",
        "status": "Production Ready"
    },
    {
        "id": "runbook_tf_02",
        "doc": "Terraform AWS VPC module v1.0 [DEPRECATED - DO NOT USE] legacy single-AZ subnet setup.",
        "status": "Deprecated Legacy"
    },
    {
        "id": "runbook_k8s_01",
        "doc": "Kubernetes Helm chart deployment for ArgoCD gitops pipelines on AWS EKS.",
        "status": "Production Ready"
    },
    {
        "id": "runbook_tf_03",
        "doc": "Terraform Cloud remote state locking and DynamoDB state backend configuration.",
        "status": "Production Ready"
    }
]

# Apply filtering
surviving_docs = []
for item in TECHNICAL_CORPUS:
    text_lower = item["doc"].lower()
    
    pass_pos = True
    if apply_pos and pos_term.strip():
        pass_pos = pos_term.strip().lower() in text_lower
        
    pass_neg = True
    if apply_neg and neg_term.strip():
        pass_neg = neg_term.strip().lower() not in text_lower
        
    if pass_pos and pass_neg:
        surviving_docs.append(item)

st.divider()

col_wb_res_l, col_wb_res_r = col_wb_res_l, col_wb_res_r = st.columns([1, 1.2])

with col_wb_res_l:
    st.subheader("Constructed `where_document` Clause")
    conditions = []
    if apply_pos and pos_term.strip():
        conditions.append({"$contains": pos_term})
    if apply_neg and neg_term.strip():
        conditions.append({"$not_contains": neg_term})
        
    if len(conditions) == 1:
        wb_where_doc = conditions[0]
    elif len(conditions) > 1:
        wb_where_doc = {"$and": conditions}
    else:
        wb_where_doc = {}
        
    st.json(wb_where_doc)
    
    st.markdown("""
    **Safety Mechanism**:  
    Notice how `runbook_tf_02` is dropped instantly because it contains the forbidden keyword `"deprecated"`, protecting the LLM from hallucinating obsolete code!
    """)

with col_wb_res_r:
    st.subheader(f"Surviving Runbooks ({len(surviving_docs)} / {len(TECHNICAL_CORPUS)})")
    if surviving_docs:
        st.dataframe(pd.DataFrame(surviving_docs), use_container_width=True, hide_index=True)
        st.success(f"🟢 All {len(surviving_docs)} documents verified safe for LLM context generation.")
    else:
        st.warning("Zero documents satisfied both the positive and negative text constraints.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the primary use case of `$not_contains` in a production RAG system?"):
    st.markdown("""
    **Answer**:
    * **Negative Lexical Filtering / Hallucination Suppression**:
      * Enterprise documentation frequently contains historical post-mortems or legacy runbooks with banners like `"[DEPRECATED]"`, `"DO NOT USE"`, or `"UNSUPPORTED"`.
      * Semantic vectors often rank these legacy documents very high because their semantic content matches the technology topic.
      * Passing `where_document={"$not_contains": "deprecated"}` deterministically purges dangerous legacy context before the LLM can generate hazardous suggestions.
    """)

with st.expander("Q2: How does ChromaDB evaluate compound `$contains` and `$not_contains` conditions?"):
    st.markdown("""
    **Answer**:
    * By wrapping them inside an `$and` or `$or` block within `where_document`:
      ```python
      where_document = {
          "$and": [
              {"$contains": "PostgreSQL"},
              {"$not_contains": "v9.6"}
          ]
      }
      ```
    * Chroma translates this into SQL boolean logic: `doc_text LIKE '%PostgreSQL%' AND doc_text NOT LIKE '%v9.6%'` evaluated during candidate selection.
    """)

with st.expander("Q3: Can `where_document` be used with `collection.get()` as well as `collection.query()`?"):
    st.markdown("""
    **Answer**:
    * **Yes**. `where_document` is fully supported in both `get()` and `query()`.
    * In `collection.get()`, it functions as a pure relational full-text search:
      `collection.get(where_document={"$contains": "incident-404"})`.
    """)

with st.expander("Q4: What is the computational cost of text filtering compared to metadata filtering?"):
    st.markdown("""
    **Answer**:
    * Metadata filters operate on fixed-width, indexed B-Tree columns in SQLite, running in sub-millisecond logarithmic time `O(log N)`.
    * Document filtering performs substring pattern matching on unbounded text blobs, which requires sequential text scanning `O(N * M)` unless a full-text search (FTS) virtual table is used.
    * Therefore, `where_document` is slightly more CPU-intensive than `where`.
    """)

with st.expander("Q5: Can you combine `where` (metadata) and `where_document` (text) in the same query?"):
    st.markdown("""
    **Answer**:
    * **Yes**. They operate as an implicit logical `$and`:
      ```python
      collection.query(
          query_texts=["deploy cluster"],
          where={"env": "prod"},
          where_document={"$contains": "Terraform"}
      )
      ```
    * A candidate must pass the metadata condition (`env == 'prod'`) AND contain the substring `"Terraform"`.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Document Substring Inside `where` Parameter**  
    Writing `where={"$contains": "foo"}` will crash. `$contains` belongs strictly in `where_document`.
    """)
    st.warning("""
    **Trap 2: Assuming Multi-Word Substrings Match Out-of-Order**  
    `$contains: "AWS Terraform"` only matches that exact consecutive phrase. To match both words anywhere, use `$and: [{"$contains": "AWS"}, {"$contains": "Terraform"}]`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Over-Filtering Resulting in Empty Prompt Context**  
    Chaining 5 restrictive `$not_contains` clauses can eliminate all candidates. Always check for empty results and supply fallbacks.
    """)
    st.warning("""
    **Trap 4: Omitting Text Lowercasing for Non-ASCII Words**  
    SQLite's `LIKE` is case-sensitive for non-ASCII Unicode characters. Normalize text during ingestion if case neutrality is required.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is $not_contains for?"):
        st.markdown("""
        **Negative Filtering**:  
        Suppresses legacy, deprecated, or hazardous document chunks.
        """)
    with st.expander("🗂️ Flashcard: How to require two keywords?"):
        st.markdown("""
        **Compound $and**:  
        `{"$and": [{"$contains": "A"}, {"$contains": "B"}]}`.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Does where_document work in get()?"):
        st.markdown("""
        **Yes**:  
        Works as exact relational text search in both `get()` and `query()`.
        """)
    with st.expander("🗂️ Flashcard: Is ASCII matching case-sensitive?"):
        st.markdown("""
        **Case-Insensitive**:  
        SQLite's ASCII `LIKE` is case-insensitive by default.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the mechanics of `$contains` and `$not_contains` in `where_document`.", value=True)
chk2 = st.checkbox("I know how to construct compound text rules combining `$contains` and `$not_contains` via `$and`.", value=True)
chk3 = st.checkbox("I understand why negative filtering is critical for suppressing deprecated and hazardous documentation.", value=True)
chk4 = st.checkbox("I understand SQLite case collation mechanics for ASCII vs Unicode text.", value=True)
chk5 = st.checkbox("I can implement a safe, production-grade text-filtered retrieval pipeline in Python.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 11: Persistence and Storage in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 10: Document Filtering in ChromaDB!")
