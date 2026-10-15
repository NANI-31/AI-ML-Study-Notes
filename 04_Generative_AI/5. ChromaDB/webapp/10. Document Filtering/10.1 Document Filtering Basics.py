import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.1 Document Filtering Basics - ChromaDB Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("10.1 Document Filtering Basics: where_document, Text Scans vs Metadata & Use Cases")
st.caption("Chapter 10: Document Filtering | Domain: 5. ChromaDB")
st.markdown("""
In many retrieval workflows, semantic vector similarity is not enough on its own. 
A vector embedding can capture the general concept of *"database crash remediation"*, but the user might strictly require 
that the retrieved document contains the exact phrase **`OOMKilled`** or mentions **`error code 137`**.

If you rely solely on vector search, the database might return an article about disk space exhaustion 
because both articles discuss general system failure concepts.

**Document Filtering in ChromaDB** solves this by evaluating criteria directly against the raw text of documents 
using the **`where_document`** parameter:
* **Full-Text Lexical Gating**: Restricting vector search to document chunks that explicitly contain (or do not contain) specific substrings.
* **Separation of Concerns**: `where` filters structured metadata columns in SQLite; `where_document` scans raw stored document text.
* **Hybrid Semantic-Lexical Pinning**: Combining vector proximity with deterministic keyword requirements without running an external Lucene/Elasticsearch cluster.

In this lesson, we break down how `where_document` operates, contrast it with metadata filtering, 
implement filtered search in Python, and test an interactive Dual Filter Playground.
""")

st.divider()

# Section 1: What is Document Filtering?
st.header("1. What is Document Filtering?", divider="rainbow")

col_df1, col_df2 = col_df1, col_df2 = st.columns([1.1, 0.9])

with col_df1:
    st.markdown("""
    ### The Role of `where_document`
    
    * In ChromaDB, every document added with `collection.add(documents=[...])` is stored as raw text in SQLite.
    * When executing `collection.query()` or `collection.get()`, you can pass:
      `where_document={"$contains": "keyword"}`
    * **How it Works**:
      * Chroma inspects the raw text of candidates stored in the SQLite `documents` table.
      * Only documents where the substring is physically present pass the gate into vector distance evaluation.
    
    ### Key Properties:
    * **Zero Extra Schema**: Operates on unstructured text without requiring prior entity extraction or metadata tagging.
    * **Deterministic Presence**: Eliminates vector "hallucinations" where a model returns a semantically similar document that lacks the required keyword.
    """)

with col_df2:
    st.markdown("### Dual Gating Pipeline Architecture")
    doc_filter_dot = """
    digraph DocFilterArch {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="Inbound Query\\nquery_texts=['Database crashes']\\nwhere_document={'$contains': 'OOMKilled'}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        TextGate [label="SQLite Text Filter Gate\\nScan raw document text for 'OOMKilled'", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        VectorRank [label="HNSW Vector Distance Scoring\\nRank only surviving candidates by cosine similarity", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        FinalHit [label="Guaranteed Top Hit\\nContains 'OOMKilled' + Maximum Semantic Match", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        InQuery -> TextGate -> VectorRank -> FinalHit;
    }
    """
    st.graphviz_chart(doc_filter_dot, use_container_width=False)

st.divider()

# Section 2: Document Filtering vs Metadata Filtering Comparison
st.header("2. Document Filtering vs Metadata Filtering Comparison", divider="rainbow")

col_tbl_c, col_diag_c = col_tbl_c, col_diag_c = st.columns([1.1, 0.9])

with col_tbl_c:
    st.markdown("### Comparative Specification Matrix")
    
    COMPARE_DATA = [
        {
            "Architectural Dimension": "Target Parameter",
            "Metadata Filtering (where)": "where={...}",
            "Document Filtering (where_document)": "where_document={...}"
        },
        {
            "Architectural Dimension": "Target Data Source",
            "Metadata Filtering (where)": "Structured Key-Value Dicts (`metadatas`)",
            "Document Filtering (where_document)": "Raw Unstructured Text Strings (`documents`)"
        },
        {
            "Architectural Dimension": "Supported Operators",
            "Metadata Filtering (where)": "$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin",
            "Document Filtering (where_document)": "$contains, $not_contains, $and, $or"
        },
        {
            "Architectural Dimension": "Index Mechanism",
            "Metadata Filtering (where)": "SQLite B-Tree Inverted Indexes (Sub-millisecond)",
            "Document Filtering (where_document)": "SQLite String Pattern Matching / Text Scanning"
        },
        {
            "Architectural Dimension": "Preprocessing Overhead",
            "Metadata Filtering (where)": "Requires extracting and tagging fields at ingest",
            "Document Filtering (where_document)": "Zero preprocessing; works directly on raw text"
        }
    ]
    st.dataframe(pd.DataFrame(COMPARE_DATA), use_container_width=True, hide_index=True)

with col_diag_c:
    st.markdown("### When to Choose Which?")
    st.info("""
    * **Use `where` (Metadata)** when filtering on:
      * Tenant IDs (`tenant_id: "acme"`)
      * Numeric dates, versions, priorities (`year >= 2024`)
      * Categorical tags and departments (`dept: "finance"`)
    * **Use `where_document`** when filtering on:
      * Specific technical jargon, function names, or error codes (`"PostgreSQL_WAL"`)
      * Compliance clauses or legal article codes (`"Article 17 GDPR"`)
      * Unstructured text where metadata tags do not exist.
    """)

st.divider()

# Section 3: Clean Python Implementation of Document Filtered Search
st.header("3. Clean Python Implementation of Document Filtered Search", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="cloud_infrastructure_docs",
    metadata={"hnsw:space": "cosine"}
)

# 1. Standalone Document Text Filter: Require exact keyword 'OOMKilled'
strict_text_hits = collection.query(
    query_texts=["Container unexpected restart remediation"],
    n_results=3,
    where_document={"$contains": "OOMKilled"}
)
print("Hits containing 'OOMKilled':", strict_text_hits["ids"])

# 2. Hybrid Filter: Combine structured metadata AND document text gating
hybrid_hits = collection.query(
    query_texts=["database disaster recovery"],
    n_results=3,
    # Metadata filter: Must belong to production environment
    where={"env": "production"},
    # Document text filter: Must explicitly contain the word 'PostgreSQL'
    where_document={"$contains": "PostgreSQL"}
)

for doc_id, doc_text, dist in zip(
    hybrid_hits["ids"][0],
    hybrid_hits["documents"][0],
    hybrid_hits["distances"][0]
):
    print(f"[{doc_id}] Dist: {dist:.4f} -> {doc_text[:60]}...")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Dual Filter Playground
st.header("4. Interactive Workbench: Dual Filter Playground", divider="rainbow")

st.markdown("""
Experiment with hybrid retrieval interactively. Enter a natural language query, 
apply a structured metadata filter (`where`), apply a document text filter (`where_document`), 
and observe how the candidate pool is filtered and ranked.
""")

col_wb_q1, col_wb_q2 = col_wb_q1, col_wb_q2 = st.columns(2)

with col_wb_q1:
    st.subheader("1. Inbound Query & Text Constraint")
    user_search_q = st.text_input("Semantic Query Text:", value="container memory limit crash")
    text_must_contain = st.text_input("Document Must Contain (`where_document`):", value="OOMKilled")

with col_wb_q2:
    st.subheader("2. Metadata Constraints (`where`)")
    selected_env = st.selectbox("Environment Constraint:", ["Any Environment", "production", "staging"])
    top_k_req = st.slider("Requested Top-K Results:", min_value=1, max_value=4, value=2)

# Mock Dataset
CORPUS = [
    {
        "id": "doc_k8s_01",
        "env": "production",
        "doc": "Kubernetes Pod OOMKilled exit code 137 due to cgroup memory limit exceeded.",
        "base_sim": 0.88
    },
    {
        "id": "doc_k8s_02",
        "env": "staging",
        "doc": "Container CrashLoopBackOff caused by misconfigured environment variable.",
        "base_sim": 0.75
    },
    {
        "id": "doc_db_01",
        "env": "production",
        "doc": "PostgreSQL database memory buffer allocation guide for high-throughput queries.",
        "base_sim": 0.82
    },
    {
        "id": "doc_db_02",
        "env": "staging",
        "doc": "Redis cache node OOMKilled remediation and memory maxmemory-policy configuration.",
        "base_sim": 0.79
    }
]

# Apply filters
filtered_candidates = []
for item in CORPUS:
    # Check where_document
    has_text = True
    if text_must_contain.strip():
        has_text = text_must_contain.strip().lower() in item["doc"].lower()
        
    # Check where metadata
    has_meta = True
    if selected_env != "Any Environment":
        has_meta = (item["env"] == selected_env)
        
    if has_text and has_meta:
        filtered_candidates.append(item)

# Sort by similarity
filtered_candidates = sorted(filtered_candidates, key=lambda x: x["base_sim"], reverse=True)[:top_k_req]

st.divider()

col_bench_res, col_bench_why = col_bench_res, col_bench_why = st.columns([1.2, 0.8])

with col_bench_res:
    st.subheader(f"Retrieved Candidates ({len(filtered_candidates)} Returned)")
    if filtered_candidates:
        rows = []
        for rank, c in enumerate(filtered_candidates, start=1):
            rows.append({
                "Rank": f"#{rank}",
                "ID": c["id"],
                "Env": c["env"],
                "Similarity": f"{c['base_sim'] * 100:.1f}%",
                "Distance": f"{1.0 - c['base_sim']:.4f}",
                "Document Text": c["doc"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success("🟢 Every retrieved candidate satisfies both metadata and exact text keyword constraints!")
    else:
        st.warning("Zero candidates matched both the metadata filter and the exact text requirement.")

with col_bench_why:
    st.subheader("Filter Evaluation Summary")
    st.markdown(f"""
    * **Semantic Concept**: *'{user_search_q}'*
    * **Text Substring Gate**: `"{text_must_contain}"`
    * **Metadata Constraint**: `env = {selected_env}`
    * **Survival Rate**: `{len(filtered_candidates)} / {len(CORPUS)}` chunks.
    """)
    st.info("Notice that `doc_db_01` was eliminated because it did NOT contain the string 'OOMKilled', despite being in production and having high semantic relevance!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is `where_document` in ChromaDB, and how does it differ from `where`?"):
    st.markdown("""
    **Answer**:
    * `where` filters structured **metadata dictionaries** (e.g. `{"env": "prod", "year": 2024}`) using indexed SQLite columns.
    * `where_document` filters the **raw text of the document chunk** (e.g. `{"$contains": "PostgreSQL"}`) using SQLite substring matching.
    * They can be combined in a single query: a candidate must satisfy both criteria to be returned.
    """)

with st.expander("Q2: Why not just extract all keywords into metadata instead of using `where_document`?"):
    st.markdown("""
    **Answer**:
    1. **Preprocessing Cost**: Running Named Entity Recognition (NER) or keyword extraction LLM passes on millions of document chunks is slow and expensive.
    2. **Unpredictable Queries**: Users frequently search for unexpected acronyms, technical function names (e.g. `malloc_trim`), or UUIDs that were not anticipated during metadata schema design.
    3. **Schema Agility**: `where_document` works out-of-the-box on raw text without schema migrations.
    """)

with st.expander("Q3: Does `where_document` support full regular expressions (regex)?"):
    st.markdown("""
    **Answer**:
    * **No**. In standard ChromaDB, `where_document` supports substring containment (`$contains` and `$not_contains`), but does not expose complex regex engines (like PCRE).
    * Substring queries map directly to SQL pattern matching (`LIKE %term%`) in the underlying SQLite engine.
    """)

with st.expander("Q4: Is `where_document` case-sensitive in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * In ChromaDB's SQLite implementation, `$contains` and `$not_contains` behavior depends on SQLite's default `LIKE` collation.
    * By default in SQLite, ASCII text comparisons in `LIKE` are case-insensitive, but unicode characters can be case-sensitive.
    * **Production Best Practice**: Do not rely on case preservation for text filtering; lowercase document text during ingestion if case-agnostic matching is required.
    """)

with st.expander("Q5: What is the performance implication of using `where_document` on very large collections?"):
    st.markdown("""
    **Answer**:
    * Substring matching on large text columns (`LIKE '%term%'`) requires scanning document text rather than traversing a fixed-size numeric B-Tree.
    * On collections with millions of documents, running `where_document` without an accompanying selective metadata filter (`where`) can increase query latency.
    * **Best Practice**: Pair `where_document` with a selective metadata filter (e.g. `tenant_id` or `date_range`) so SQLite only scans text within a small pre-filtered partition!
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Placing Document Substrings Inside `where` Parameter**  
    Writing `where={"$contains": "Postgres"}` fails because `$contains` is a `where_document` operator, not a metadata operator!
    """)
    st.warning("""
    **Trap 2: Scanning Raw Text Across Entire Multi-Million Doc Collections**  
    Using `where_document` alone forces a full-text scan across all documents. Always pair it with a partition filter (`where={"tenant": ...}`).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Relying on Semantic Vectors for Exact Code Identifiers**  
    Vectors blur distinctions between `v1.2.3` and `v1.2.4`. Use `where_document={"$contains": "v1.2.3"}` for exact version pinning.
    """)
    st.warning("""
    **Trap 4: Expecting Regex or Fuzzy Levenshtein Matching**  
    `where_document` does substring containment, not fuzzy typo-tolerant matching. If the user misspells a keyword, `$contains` will match 0 docs.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does where_document filter?"):
        st.markdown("""
        **Raw Text**:  
        Filters the actual text content of documents, not metadata.
        """)
    with st.expander("🗂️ Flashcard: Can where and where_document combine?"):
        st.markdown("""
        **Yes (Implicit $and)**:  
        Candidates must pass both metadata and text filters.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Does where_document need schema?"):
        st.markdown("""
        **Zero Schema**:  
        Operates on unstructured text without prior tagging.
        """)
    with st.expander("🗂️ Flashcard: How to optimize where_document?"):
        st.markdown("""
        **Pair with where**:  
        Pre-filter by tenant/date to avoid full-table text scans.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand what document filtering is and how `where_document` evaluates raw text.", value=True)
chk2 = st.checkbox("I know the architectural differences between metadata filtering (`where`) and document filtering (`where_document`).", value=True)
chk3 = st.checkbox("I know when to use exact lexical gating alongside semantic vector similarity.", value=True)
chk4 = st.checkbox("I understand how combining `where` and `where_document` creates high-precision hybrid retrieval.", value=True)
chk5 = st.checkbox("I understand the performance best practice of pairing text scans with selective partition filters.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 10.2: Text Filtering in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 10.1: Document Filtering Basics and are ready for Chapter 10.2: Text Filtering!")
