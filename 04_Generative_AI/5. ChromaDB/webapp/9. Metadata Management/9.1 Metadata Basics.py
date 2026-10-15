import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="9.1 Metadata Basics - ChromaDB Curriculum",
    page_icon="🏷️",
    layout="wide"
)

# Title & Overview
st.title("9.1 Metadata Basics: Structure, Scalar Types, In-Place Patching & Schema Design")
st.caption("Chapter 9: Metadata Management | Domain: 5. ChromaDB")
st.markdown("""
In enterprise AI architectures, vector embeddings capture **semantic meaning**, but they know nothing about 
**governance, permissions, timestamps, or operational boundaries**.

If you rely solely on dense vector search, an intern's search query might retrieve executive salary spreadsheets 
or draft documentation because the semantic concepts match.

**Metadata in ChromaDB** provides the relational anchor for vector embeddings:
* **The Structured Companion**: Key-value dictionaries stored alongside vectors in SQLite relational tables.
* **Primitive Scalar Invariant**: Values must be flat primitives (`str`, `int`, `float`, `bool`) to support high-speed B-Tree indexing.
* **Zero-Re-Embedding Updates**: Modifying document status (e.g. from `"draft"` to `"published"`) via `collection.update()` executes in sub-millisecond time without re-computing vectors!
* **Enterprise Tenancy & Access Control**: Partitioning search spaces by `tenant_id`, `department`, or `security_classification`.

In this lesson, we break down metadata constraints, design production schemas, implement targeted in-place metadata updates in Python, 
and test an interactive Live Metadata Schema Designer and Patching Tester.
""")

st.divider()

# Section 1: Anatomy & Type Constraints of ChromaDB Metadata
st.header("1. Anatomy & Type Constraints of ChromaDB Metadata", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### The Flat Scalar Requirement
    
    * In ChromaDB, every metadata object is a standard Python dictionary: `Dict[str, Union[str, int, float, bool]]`.
    * **Supported Types**:
      * `str`: Categorical tags, tenant IDs, file paths, author names.
      * `int`: Epoch timestamps, integer priorities, page numbers, chunk indices.
      * `float`: Confidence thresholds, numeric ratings, prices.
      * `bool`: Active flags, verified status, visibility booleans.
    
    * **Prohibited Types (Will Raise `ValueError`)**:
      * ❌ Nested Dictionaries: `{"user": {"id": 101}}` -> **Rejected**.
      * ❌ Lists & Arrays: `{"tags": ["ai", "rag"]}` -> **Rejected**.
      * ❌ Null / None Values: `{"expiry": None}` -> **Rejected**.
      * *Solution*: Flatten nested structures (`"user_id": 101`) and serialize lists to strings (`"tags": "ai,rag"`).
    """)

with col_anat2:
    st.markdown("### Metadata Storage Architecture")
    meta_dot = """
    digraph MetadataStorage {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InPayload [label="Document Ingestion Payload\\nmetadatas=[{'dept': 'eng', 'year': 2024}]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        TypeValidator [label="ChromaDB Type Validator\\nVerify: str, int, float, bool only", fillcolor="#FEF3C7", color="#D97706"];

        SQLiteCols [label="SQLite Inverted Index Table\\nKey-Value B-Trees on primitive columns", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        FastFilter [label="Sub-Millisecond Pre-Filtering\\nExecutes during HNSW graph traversal", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InPayload -> TypeValidator -> SQLiteCols -> FastFilter;
    }
    """
    st.graphviz_chart(meta_dot, use_container_width=False)

st.divider()

# Section 2: Zero-Re-Embedding Metadata Patching
st.header("2. Zero-Re-Embedding Metadata Patching", divider="rainbow")

col_patch1, col_patch2 = col_patch1, col_patch2 = st.columns([1, 1])

with col_patch1:
    st.markdown("""
    ### The Latency Paradox of Updates
    * Computing high-dimensional embeddings on CPU or GPU takes **5ms to 50ms** per document chunk.
    * In enterprise workflows, documents frequently change state without their text changing:
      * An admin marks a runbook as `"verified": True`.
      * An article is tagged with a new department `"dept": "operations"`.
      * Access permissions are revoked: `"role": "admin_only"`.
    * **The Solution**: Calling `collection.update(ids=[...], metadatas=[...])` **without passing `documents`**.
      * Chroma detects that text is unchanged.
      * It updates only the SQLite metadata row in `< 0.5 ms`!
      * Zero transformer inferences, zero API tokens, and zero vector re-indexing required!
    """)

with col_patch2:
    st.markdown("### Clean Python In-Place Patching")
    st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("enterprise_kb")

# High-Speed In-Place Metadata Patch (Zero Re-Embedding!)
collection.update(
    ids=["runbook_042"],
    metadatas=[{
        "status": "approved",
        "verified_by": "lead_architect",
        "last_reviewed_epoch": 1729000000,
        "is_active": True
    }]
)
print("Metadata patched in SQLite in < 1 millisecond!")
""", language="python")

st.divider()

# Section 3: Production Metadata Schema Design Patterns
st.header("3. Production Metadata Schema Design Patterns", divider="rainbow")

st.markdown("""
Below is the standard enterprise schema design for documents ingested into production RAG systems:
""")

SCHEMA_PATTERNS = [
    {
        "Metadata Key": "tenant_id",
        "Data Type": "str",
        "Example Value": "'acme_corp_881'",
        "Purpose & Production Role": "Multi-tenant isolation; ensures strict data segregation between enterprise clients."
    },
    {
        "Metadata Key": "source_path",
        "Data Type": "str",
        "Example Value": "'s3://kb/eng/runbooks/db_failover.pdf'",
        "Purpose & Production Role": "Provenance and source attribution; displayed in UI as citations."
    },
    {
        "Metadata Key": "page_number",
        "Data Type": "int",
        "Example Value": "14",
        "Purpose & Production Role": "Document location pointer for deep linking to PDF pages."
    },
    {
        "Metadata Key": "chunk_index",
        "Data Type": "int",
        "Example Value": "3",
        "Purpose & Production Role": "Sequence ordering; allows reconstructing adjacent paragraphs during generation."
    },
    {
        "Metadata Key": "created_epoch",
        "Data Type": "int",
        "Example Value": "1719801600",
        "Purpose & Production Role": "Temporal filtering; allows filtering for recency (e.g. `$gte`: 2024)."
    },
    {
        "Metadata Key": "access_level",
        "Data Type": "str",
        "Example Value": "'confidential'",
        "Purpose & Production Role": "Role-Based Access Control (RBAC); prevents unauthorized data leakage."
    }
]
st.dataframe(pd.DataFrame(SCHEMA_PATTERNS), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Live Metadata Schema Designer & Validator
st.header("4. Interactive Workbench: Live Metadata Schema Designer & Validator", divider="rainbow")

st.markdown("""
Test metadata key-value configurations interactively. Build a dictionary, test ChromaDB's 
primitive type validation rules, and observe whether the schema is valid for high-speed indexing.
""")

col_wb_l, col_wb_r = col_wb_l, col_wb_r = st.columns(2)

with col_wb_l:
    st.subheader("1. Metadata Field Configurator")
    m_tenant = st.text_input("Tenant Identifier (`tenant_id` - str):", value="tenant_enterprise_01")
    m_page = st.number_input("Page Number (`page_num` - int):", min_value=1, max_value=500, value=12)
    m_conf = st.slider("Quality Score (`quality_score` - float):", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
    m_verified = st.checkbox("Verified by Human Editor (`is_verified` - bool)", value=True)

with col_wb_r:
    st.subheader("2. Type Validation Bug Simulator")
    inject_nested = st.checkbox("Inject Nested Dictionary (Invalid Antipattern)", value=False)
    inject_list = st.checkbox("Inject Array / List Field (Invalid Antipattern)", value=False)
    inject_null = st.checkbox("Inject None / Null Field (Invalid Antipattern)", value=False)

# Build metadata dictionary
sample_metadata = {
    "tenant_id": m_tenant,
    "page_num": int(m_page),
    "quality_score": float(m_conf),
    "is_verified": bool(m_verified)
}

if inject_nested:
    sample_metadata["nested_author"] = {"first": "Jane", "last": "Doe"}
if inject_list:
    sample_metadata["tags_list"] = ["infrastructure", "database", "high-availability"]
if inject_null:
    sample_metadata["deprecated_date"] = None

# Validation check
validation_errors = []
for k, v in sample_metadata.items():
    if v is None:
        validation_errors.append(f"Key '{k}' has None value. ChromaDB does not support null metadata.")
    elif isinstance(v, dict):
        validation_errors.append(f"Key '{k}' is a nested dict. ChromaDB requires flat key-value pairs.")
    elif isinstance(v, (list, tuple, set)):
        validation_errors.append(f"Key '{k}' is a list/array. ChromaDB does not support array metadata values.")
    elif not isinstance(v, (str, int, float, bool)):
        validation_errors.append(f"Key '{k}' has unsupported type '{type(v).__name__}'.")

st.divider()

col_score_view, col_json_view = col_score_view, col_json_view = st.columns([1, 1.2])

with col_score_view:
    st.subheader("ChromaDB Validation Report")
    if not validation_errors:
        st.success("🟢 **VALID METADATA**: All keys contain supported scalar primitives (`str`, `int`, `float`, `bool`)!")
    else:
        st.error(f"🔴 **INVALID METADATA**: ChromaDB will raise `ValueError` on ingestion!")
        for err in validation_errors:
            st.markdown(f"* ⚠️ {err}")
            
    st.markdown("""
    **Remediation Rule of Thumb**:
    * If you need tags, join with delimiters: `"tags": "infra,db,ha"`.
    * If you need nested objects, flatten keys: `"author_name": "Jane Doe"`.
    """)

with col_json_view:
    st.subheader("Constructed Metadata Object")
    st.json(sample_metadata)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What data types are allowed for metadata values in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * ChromaDB strictly permits **primitive scalar types**:
      1. `str` (strings)
      2. `int` (integers)
      3. `float` (floating-point numbers)
      4. `bool` (booleans)
    * Nested dictionaries, lists, sets, and `None` values are completely prohibited and will trigger an immediate `ValueError`.
    """)

with st.expander("Q2: How can you update a document's metadata in ChromaDB without paying re-embedding costs?"):
    st.markdown("""
    **Answer**:
    * Call `collection.update(ids=[target_id], metadatas=[new_metadata])` and **omit the `documents` and `embeddings` parameters**.
    * ChromaDB updates the metadata columns inside SQLite directly.
    * Because no text is passed, the embedding model is never invoked, completing the patch in `< 0.5 ms`.
    """)

with st.expander("Q3: How do you represent a list of tags (e.g., `['finance', 'audit']`) in ChromaDB metadata?"):
    st.markdown("""
    **Answer**:
    * Because lists are rejected, the two standard production patterns are:
      1. **Delimiter-Separated String**: Store `"tags": "finance,audit"`, and query using `$contains` or document content filtering.
      2. **Boolean Flags**: Store `"is_finance": True, "is_audit": True`, and query using direct boolean equality filters.
    """)

with st.expander("Q4: Why does ChromaDB disallow nested metadata dictionaries?"):
    st.markdown("""
    **Answer**:
    * ChromaDB stores metadata in SQLite relational columns with specialized inverted index B-Trees.
    * Supporting arbitrary JSON nesting would require expensive JSON path parsing or document store abstractions, severely degrading the speed of metadata pre-filtering during HNSW graph traversal.
    * Flat primitives ensure sub-millisecond filtering at scale.
    """)

with st.expander("Q5: What is the risk of having missing metadata keys across documents in the same collection?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does not enforce a rigid schema across documents: Document A can have `{"dept": "sales"}` while Document B has `{"dept": "engineering", "year": 2024}`.
    * **Risk**: When running a query with `where={"year": 2024}`, Document A is automatically omitted because the key does not exist.
    * In production, maintain a consistent Pydantic schema for all ingested chunks to ensure queries do not unintentionally filter out relevant records.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Storing Python Lists in Metadata Values**  
    Passing `{"categories": ["ai", "nlp"]}` raises a fatal `ValueError`. Store as a delimited string or separate boolean flags.
    """)
    st.warning("""
    **Trap 2: Passing `None` for Optional Fields**  
    Passing `{"expiry": None}` is rejected. Either omit the key entirely or use a sentinel value like `""` or `-1`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Re-Passing Document Text During Metadata Updates**  
    Passing `documents` when calling `update()` forces Chroma to re-run the transformer embedding model, wasting CPU/GPU compute.
    """)
    st.warning("""
    **Trap 4: Inconsistent Epoch vs String Timestamps**  
    Storing `"created": "2024-10-12"` on some records and `1728700000` on others breaks numeric range comparisons (`$gte`, `$lte`). Standardize on integer epochs.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What metadata types are valid?"):
        st.markdown("""
        **Flat Primitives**:  
        `str`, `int`, `float`, and `bool` only.
        """)
    with st.expander("🗂️ Flashcard: How to patch metadata fast?"):
        st.markdown("""
        **Omit Documents**:  
        `collection.update(ids=..., metadatas=...)` updates SQLite in <1ms without re-embedding.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Are nested dicts supported?"):
        st.markdown("""
        **No**:  
        Nested dictionaries are rejected with a `ValueError`. Flatten keys.
        """)
    with st.expander("🗂️ Flashcard: Are None values allowed?"):
        st.markdown("""
        **No**:  
        `None` / `null` is disallowed. Use sentinel values or omit keys.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the four supported scalar primitive metadata types (`str`, `int`, `float`, `bool`).", value=True)
chk2 = st.checkbox("I know why nested dictionaries, lists, and `None` values are rejected by ChromaDB.", value=True)
chk3 = st.checkbox("I know how to perform zero-re-embedding metadata updates using `collection.update()`.", value=True)
chk4 = st.checkbox("I understand production metadata schema patterns for tenant isolation and provenance.", value=True)
chk5 = st.checkbox("I know how to serialize lists and flatten complex objects before ingesting into ChromaDB.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 9.2: Metadata Filtering in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 9.1: Metadata Basics and are ready for Chapter 9.2: Metadata Filtering!")
