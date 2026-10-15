import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="6.1 Basic Add Operations - ChromaDB Curriculum",
    page_icon="📥",
    layout="wide"
)

# Title & Overview
st.title("6.1 Basic Add Operations: The add() Primitive, Parameters, Metadata & Direct Embeddings")
st.caption("Chapter 6: Adding Data to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
The `collection.add()` method is ChromaDB's foundational data ingestion primitive. 

While conceptually simple, executing `add()` in enterprise environments requires understanding 
its strict operational contracts:
* **The Parallel Array Contract**: IDs, Documents, Metadatas, and Embeddings must be provided as parallel lists of identical length.
* **Strict Uniqueness Invariant**: Every ID must be completely unique within the collection; encountering an existing ID raises an immediate `DuplicateIDError`.
* **Flexible Payload Modes**: You can supply raw documents (triggering automatic embedding), pre-computed vectors (bypassing the embedding function), or both.
* **Metadata Type Guardrails**: Metadata values must be primitive scalars (`str`, `int`, `float`, `bool`) to support high-speed SQLite indexing.

In this lesson, we break down the `add()` parameter matrix, analyze failure modes and atomic rollback behavior, 
write clean Python ingestion pipelines, and test an interactive Live Payload Builder and Schema Validation Tester.
""")

st.divider()

# Section 1: Anatomy of the collection.add() Method
st.header("1. Anatomy of the collection.add() Method", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### The Four Parallel Parameter Lists
    
    * **1. `ids: List[str]` (Mandatory)**:
      * Unique string identifiers. Cannot be empty, null, or duplicated within the collection.
    
    * **2. `documents: Optional[List[str]]`**:
      * The raw text chunks. If provided, ChromaDB passes these strings to the collection's bound `EmbeddingFunction` to compute vectors automatically.
    
    * **3. `metadatas: Optional[List[Dict[str, Union[str, int, float, bool]]]]`**:
      * Key-value dictionaries attached to each document. Used for SQL-like metadata filtering during search queries.
    
    * **4. `embeddings: Optional[List[List[float]]]`**:
      * Pre-computed floating-point coordinate arrays. If provided, Chroma stores these directly, bypassing the local embedding model.
    """)

with col_anat2:
    st.markdown("### Ingestion Parameter Matrix")
    matrix_dot = """
    digraph AddMatrix {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        CallAdd [label="collection.add(...)\\nIngestion Call", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        IDs [label="ids: ['doc_1', 'doc_2']\\n(MANDATORY)", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        Docs [label="documents: ['Text 1', 'Text 2']\\nAuto-generates vectors", fillcolor="#D1FAE5", color="#059669"];

        Metas [label="metadatas: [{'sec': 1}, {'sec': 2}]\\nIndexed for Pre-Filtering", fillcolor="#FEF3C7", color="#D97706"];

        Vecs [label="embeddings: [[0.1, ...], [0.2, ...]]\\nBypasses embedding model", fillcolor="#EDE9FE", color="#7C3AED"];

        CallAdd -> IDs;
        CallAdd -> Docs;
        CallAdd -> Metas;
        CallAdd -> Vecs;
    }
    """
    st.graphviz_chart(matrix_dot, use_container_width=False)

st.divider()

# Section 2: Ingestion Modes & Parallel Array Alignment
st.header("2. Ingestion Modes & Parallel Array Alignment", divider="rainbow")

col_mode1, col_mode2 = col_mode1, col_mode2 = st.columns([1, 1])

with col_mode1:
    st.markdown("""
    ### Three Valid Ingestion Configurations
    
    1. **Mode A: Text-Only (Default Auto-Embedding)**:
       * Supply `ids` and `documents`.
       * Chroma calls its ONNX MiniLM model to compute vectors.
       * *Code*: `collection.add(ids=ids, documents=docs, metadatas=metas)`
    
    2. **Mode B: Pre-Computed Embeddings + Text**:
       * Supply `ids`, `embeddings`, and `documents`.
       * Bypasses the local embedding function, storing external vectors directly while preserving raw text for LLM context hydration.
       * *Code*: `collection.add(ids=ids, embeddings=vecs, documents=docs)`
    
    3. **Mode C: Pure Vector Store (Headless)**:
       * Supply only `ids` and `embeddings`.
       * Text is stored in an external database (e.g. S3 / Postgres).
       * *Code*: `collection.add(ids=ids, embeddings=vecs, metadatas=metas)`
    """)

with col_mode2:
    st.markdown("### The Parallel Alignment Rule")
    st.warning("""
    **All passed lists must share the exact same length!**  
    If `len(ids) == 3`, then `len(documents)` and `len(metadatas)` must also equal `3`.
    """)
    st.code("""# CORRECT: Identical Lengths (len = 2)
collection.add(
    ids=["chunk_1", "chunk_2"],
    documents=["First paragraph.", "Second paragraph."],
    metadatas=[{"idx": 1}, {"idx": 2}]
)

# ERROR: Length Mismatch! Throws ValueError
collection.add(
    ids=["chunk_1", "chunk_2"],
    documents=["Only one document provided!"]  # len=1 != len=2
)
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Basic Add Operations
st.header("3. Clean Python Implementation of Basic Add Operations", divider="rainbow")

st.code("""import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="engineering_kb",
    metadata={"hnsw:space": "cosine"}
)

# 1. Mode A: Basic text ingestion with rich metadata
collection.add(
    ids=["runbook-01", "runbook-02"],
    documents=[
        "Envoy proxy enforces mTLS across all Kubernetes service mesh pods.",
        "PostgreSQL logical replication streams changes to Snowflake data warehouse."
    ],
    metadatas=[
        {"service": "mesh", "env": "prod", "criticality": 1},
        {"service": "database", "env": "prod", "criticality": 2}
    ]
)
print("Ingested documents via automated embedding.")

# 2. Mode B: Direct ingestion with pre-computed external vectors
external_vectors = [
    [0.082, -0.412, 0.991, 0.052],  # Vector 1
    [-0.312, 0.125, 0.441, 0.884]   # Vector 2
]

collection.add(
    ids=["arch-01", "arch-02"],
    embeddings=external_vectors,
    documents=[
        "Kafka brokers partition topic streams for horizontal consumer scaling.",
        "Redis cluster distributes keys across 16,384 hash slots."
    ],
    metadatas=[{"tier": "messaging"}, {"tier": "cache"}]
)
print("Ingested pre-computed external vectors directly!")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Payload Builder & Schema Validator
st.header("4. Interactive Workbench: Live Payload Builder & Schema Validator", divider="rainbow")

st.markdown("""
Construct a ChromaDB ingestion payload interactively. Test parallel array alignments, 
metadata type validation, and inspect the resulting JSON payload structure before execution.
""")

col_wb_l, col_wb_r = col_wb_l, col_wb_r = st.columns(2)

with col_wb_l:
    st.subheader("1. Document Chunks (Line Separated)")
    docs_input = st.text_area(
        "Enter Text Documents (One per line):",
        value="Envoy proxy handles ingress load balancing.\nPostgreSQL manages primary relational transactions.\nRedis cluster serves as low-latency semantic cache.",
        height=110
    )
    raw_lines = [line.strip() for line in docs_input.split("\n") if line.strip()]

with col_wb_r:
    st.subheader("2. Ingestion Settings & Metadata")
    id_prefix = st.text_input("ID Prefix:", value="DOC_INFRA")
    category_tag = st.selectbox("Metadata Category:", ["Infrastructure", "Data Platforms", "Security"])
    simulate_mismatch = st.checkbox("Simulate Array Length Mismatch Bug", value=False)
    add_manual_vecs = st.checkbox("Include Pre-Computed Dense Embeddings", value=False)

# Build arrays
num_docs = len(raw_lines)
generated_ids = [f"{id_prefix}_{idx+1:03d}" for idx in range(num_docs)]
if simulate_mismatch and num_docs > 1:
    generated_ids = generated_ids[:-1]  # Drop one ID to create a mismatch!

generated_metas = [{"category": category_tag, "chunk_num": idx+1, "char_count": len(text)} for idx, text in enumerate(raw_lines)]

# Validate
is_len_match = len(generated_ids) == len(raw_lines) == len(generated_metas)

st.divider()

col_val_box, col_json_box = col_val_box, col_json_box = st.columns([1, 1.2])

with col_val_box:
    st.subheader("Payload Validation Scorecard")
    if is_len_match:
        st.success("🟢 **VALID PAYLOAD**: All parallel arrays have matching lengths!")
    else:
        st.error(f"🔴 **INVALID PAYLOAD**: Length mismatch! `len(ids)={len(generated_ids)}` != `len(documents)={len(raw_lines)}`! Chroma will raise `ValueError`.")
        
    st.markdown(f"""
    * **Document Count**: `{len(raw_lines)}`
    * **ID Count**: `{len(generated_ids)}`
    * **Metadata Count**: `{len(generated_metas)}`
    * **Metadata Types**: All values are primitive scalars (`str`, `int`) ✅
    """)

with col_json_box:
    st.subheader("Generated Ingestion Payload")
    payload_preview = {
        "ids": generated_ids,
        "documents": raw_lines,
        "metadatas": generated_metas
    }
    if add_manual_vecs:
        payload_preview["embeddings"] = [[round(np.random.uniform(-0.5, 0.5), 3) for _ in range(4)] for _ in range(len(raw_lines))]
    st.json(payload_preview)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What happens if `collection.add()` is called with an ID that already exists in the collection?"):
    st.markdown("""
    **Answer**:
    * ChromaDB raises an immediate `DuplicateIDError: ID already exists`.
    * **Atomicity & Rollback**: Because ChromaDB executes mutations within a transactional context, the entire batch call fails.
    * None of the documents in the batch are inserted into SQLite or the HNSW graph.
    * **Remedy**: Use `collection.upsert()` for idempotent insertion, or verify ID uniqueness upstream.
    """)

with st.expander("Q2: Why must metadata values in `collection.add()` be flat primitive types?"):
    st.markdown("""
    **Answer**:
    * ChromaDB persists metadata inside SQLite relational columns with inverted index B-Trees.
    * Supported scalar types are `str`, `int`, `float`, and `bool`.
    * Nested dictionaries (`{"author": {"name": "Bob"}}`) or lists (`{"tags": ["ai", "nlp"]}`) are rejected with a `ValueError`.
    * Flatten dictionaries or serialize lists to delimiter-separated strings before passing them to `add()`.
    """)

with st.expander("Q3: Can you add documents without metadata, or metadata without documents?"):
    st.markdown("""
    **Answer**:
    * **Documents without Metadata**: **Yes**. The `metadatas` parameter is completely optional. If omitted, Chroma stores documents with empty metadata objects.
    * **Metadata without Documents**: **Yes, provided you supply `embeddings`**. If you provide pre-computed vectors, you can omit `documents` and pass only `ids`, `embeddings`, and `metadatas`.
    """)

with st.expander("Q4: What is the computational cost of calling `add()` in a loop vs calling `add()` with a batch?"):
    st.markdown("""
    **Answer**:
    * Calling `add()` 1,000 times for individual single documents triggers **1,000 separate disk I/O syncs, SQLite transactions, and ONNX model inferences**. Throughput: ~25 docs/sec.
    * Calling `add()` once with a batch of 1,000 documents triggers **1 single atomic SQLite transaction, batch vectorized ONNX inference, and 1 disk write**. Throughput: ~2,500 docs/sec (100x faster!).
    """)

with st.expander("Q5: What happens if you supply both `documents` and `embeddings` in `collection.add()`?"):
    st.markdown("""
    **Answer**:
    * When both are supplied, ChromaDB **uses your supplied `embeddings`** and **does NOT call its bound embedding function**.
    * It stores the raw `documents` strings in SQLite, preserving the text so that subsequent search queries can retrieve the text payload for LLM context prompts.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Nested JSON Dictionaries in Metadata**  
    Passing `{"geo": {"lat": 30.2, "lon": -97.7}}` raises an immediate validation exception. Flatten to `{"geo_lat": 30.2, "geo_lon": -97.7}`.
    """)
    st.warning("""
    **Trap 2: Feeding 1-by-1 Documents in Web Request Handlers**  
    Inserting documents individually in FastAPI endpoints creates SQLite database lock contention. Buffer incoming items and flush in batches.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Mismatched Parallel Array Lengths**  
    Passing 100 IDs and 99 documents causes a fatal `ValueError` that crashes the entire ingestion job. Ensure array lengths match before calling `add()`.
    """)
    st.warning("""
    **Trap 4: Using `add()` in Re-Run Ingestion Pipelines**  
    Re-running an ingestion script that uses `add()` fails on the first existing record. Use `collection.upsert()` for reproducible pipelines.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the parallel array rule?"):
        st.markdown("""
        **Identical Lengths**:  
        `ids`, `documents`, `metadatas`, and `embeddings` must have matching lengths.
        """)
    with st.expander("🗂️ Flashcard: What does add() do on existing ID?"):
        st.markdown("""
        **Fatal DuplicateIDError**:  
        Aborts the entire batch; no records are inserted.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Can you bypass the embedding model?"):
        st.markdown("""
        **Yes**:  
        Pass `embeddings=vectors` directly to store pre-computed floats.
        """)
    with st.expander("🗂️ Flashcard: What metadata types are valid?"):
        st.markdown("""
        **Flat Primitives**:  
        Only `str`, `int`, `float`, and `bool` scalars are supported.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the four parallel parameter lists: ids, documents, metadatas, and embeddings.", value=True)
chk2 = st.checkbox("I know why all passed parameter lists must share the exact same length.", value=True)
chk3 = st.checkbox("I understand the three ingestion configurations: text-only, pre-computed vectors, and headless.", value=True)
chk4 = st.checkbox("I know that `add()` fails with DuplicateIDError if an ID already exists in the collection.", value=True)
chk5 = st.checkbox("I understand metadata primitive type constraints and how to flatten structured records.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 6.2: Batch Operations in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 6.1: Basic Add Operations and are ready for Chapter 6.2: Batch Operations!")
