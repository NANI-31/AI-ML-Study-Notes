import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="3.2 Document Structure - ChromaDB Curriculum",
    page_icon="🧬",
    layout="wide"
)

# Title & Overview
st.title("3.2 Document Structure: IDs, Content, Metadata, Vectors & The 1:1 Geometric Mapping")
st.caption("Chapter 3: Documents and Data Management | Domain: 5. ChromaDB")
st.markdown("""
To design scalable RAG knowledge bases, an engineer must look past raw text strings and master 
the **Internal Anatomical Structure of a ChromaDB Record**.

In relational databases like PostgreSQL, a record is a row in a table with strictly typed columns. 
In ChromaDB, a record is a **Quad-Component Composite Object**:
1. **Primary Key Identifier (`id`)**: An immutable, unique string index.
2. **Text Payload (`document`)**: The human-readable string preserved for prompt synthesis.
3. **Structured Context (`metadata`)**: Flat scalar key-value attributes for SQL-style boolean pre-filtering.
4. **Geometric Coordinate (`embedding`)**: A dense floating-point array anchored in HNSW metric space.

Crucially, ChromaDB enforces a strict **1:1 Mapping between IDs and Vectors**: one vector node in the HNSW graph 
corresponds to exactly one document payload.

In this lesson, we dissect the four quadrants of a ChromaDB record, analyze how multi-chunk parent documents 
are modeled using relational metadata pointers, examine Pydantic schema validation, and test an interactive 
Document Structure Anatomy Visualizer.
""")

st.divider()

# Section 1: The Four Quadrants of a ChromaDB Record
st.header("1. The Four Quadrants of a ChromaDB Record", divider="rainbow")

col_quad1, col_quad2 = col_quad1, col_quad2 = st.columns([1.1, 0.9])

with col_quad1:
    st.markdown("""
    ### Record Quadrant Breakdown
    
    * **Quadrant 1: The Primary Identifier (`id: str`)**:
      * Mandatory unique string. Acts as the primary key in SQLite and node identifier in the HNSW index.
    
    * **Quadrant 2: The Text Payload (`document: Optional[str]`)**:
      * The raw UTF-8 textual content. Passed directly to downstream LLMs during RAG context prompt generation.
    
    * **Quadrant 3: The Structured Metadata (`metadata: Optional[Dict[str, Any]]`)**:
      * Flat key-value attributes (`str`, `int`, `float`, `bool`).
      * Used for pre-filtering search candidate pools (`where={"department": "legal", "year": {"$gte": 2023}}`).
    
    * **Quadrant 4: The Geometric Embedding (`embedding: List[float]`)**:
      * Dense array of 32-bit floats (e.g. 384 or 1536 floats).
      * Represents semantic meaning geometrically; used exclusively for distance calculations.
    """)

with col_quad2:
    st.markdown("### The Four-Quadrant Anatomy Topology")
    quad_dot = """
    digraph RecordAnatomy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RecordRoot [label="ChromaDB Composite Record\\n(1:1 Unified Geometric Entity)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QID [label="Quadrant 1: Primary ID\\n'manual:sec_4:chunk_0'\\nUnique String Key", fillcolor="#FEE2E2", color="#DC2626"];

        QDoc [label="Quadrant 2: Document Content\\n'Envoy proxies terminate TLS 1.3...'\\nPreserved for Prompt Context", fillcolor="#FEF3C7", color="#D97706"];

        QMeta [label="Quadrant 3: Metadata\\n{'tier': 'infra', 'clearance': 2}\\nIndexed for Fast Pre-Filtering", fillcolor="#D1FAE5", color="#059669"];

        QVec [label="Quadrant 4: Embedding Vector\\n[0.082, -0.412, 0.991, ...]\\nPlotted in HNSW Metric Space", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        RecordRoot -> QID;
        RecordRoot -> QDoc;
        RecordRoot -> QMeta;
        RecordRoot -> QVec;
    }
    """
    st.graphviz_chart(quad_dot, use_container_width=False)

st.divider()

# Section 2: Modeling 1:Many Relationships: The Parent Document Pattern
st.header("2. Modeling 1:Many Relationships: The Parent Document Pattern", divider="rainbow")

col_rel1, col_rel2 = col_rel1, col_rel2 = st.columns([1, 1])

with col_rel1:
    st.markdown("""
    ### The Fundamental Tension
    * **Small Chunks (100–250 tokens)**: Deliver high retrieval precision because embedding centroids focus closely on specific facts.
    * **Large Contexts (1,000–4,000 tokens)**: Deliver better synthesis quality because the LLM understands surrounding narrative flow.
    * If you embed a 50-page PDF as a single vector, its embedding becomes diluted and retrieval precision plummets.
    
    ### The Parent-Child Chunking Solution
    In ChromaDB, decompose large parent documents into multiple child records, binding them via **relational metadata pointers**:
    ```json
    {
      "id": "contract_42_chunk_03",
      "document": "Section 4.2: Payment shall occur within 30 days...",
      "metadata": {
        "parent_doc_id": "contract_42",
        "parent_title": "Enterprise Master Service Agreement",
        "chunk_index": 3,
        "total_chunks": 18
      }
    }
    ```
    """)

with col_rel2:
    st.markdown("### Parent-Child Retrieval Architecture")
    p_dot = """
    digraph ParentChild {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ParentDoc [label="Large Parent Document (5,000 words)\\n'Master Service Agreement 2024.pdf'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_children {
            label="ChromaDB Records (Fine-Grained Vectors)"; style="dashed"; color="#059669";
            C1 [label="Record: chunk_0\\nEmbeddings: Specific", fillcolor="#D1FAE5"];
            C2 [label="Record: chunk_1\\nEmbeddings: Specific", fillcolor="#D1FAE5"];
            C3 [label="Record: chunk_2\\nEmbeddings: Specific", fillcolor="#D1FAE5"];
        }

        Retriever [label="Vector Search Matches chunk_1\\nExtracts metadata['parent_doc_id']", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        ContextPrompt [label="Hydrates Full Parent Context into LLM!\\nComprehensive Synthesis", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        ParentDoc -> C1;
        ParentDoc -> C2;
        ParentDoc -> C3;
        C2 -> Retriever [label="Top Semantic Match"];
        Retriever -> ContextPrompt [label="Fetch Parent Document"];
    }
    """
    st.graphviz_chart(p_dot, use_container_width=False)

st.divider()

# Section 3: Clean Python Data Modeling with Pydantic
st.header("3. Clean Python Data Modeling with Pydantic", divider="rainbow")

st.markdown("""
To guarantee metadata hygiene and prevent serialization errors before calling ChromaDB, 
use **Pydantic Schemas** to validate documents at the application layer:
""")

st.code("""from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
import chromadb

class ChromaRecordSchema(BaseModel):
    \"\"\"Strict Pydantic schema enforcing ChromaDB record constraints.\"\"\"
    id: str = Field(..., min_length=1, max_length=128)
    document: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("metadata")
    def validate_flat_metadata(cls, v):
        for key, val in v.items():
            if not isinstance(val, (str, int, float, bool)):
                raise ValueError(
                    f"Metadata key '{key}' has invalid type {type(val)}. "
                    "ChromaDB strictly requires flat primitive types (str, int, float, bool)."
                )
        return v

def ingest_validated_records(collection, records: List[ChromaRecordSchema]):
    \"\"\"Ingests a batch of validated records into ChromaDB.\"\"\"
    collection.add(
        ids=[r.id for r in records],
        documents=[r.document for r in records],
        metadatas=[r.metadata for r in records]
    )
    print(f"Ingested {len(records)} validated records.")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Document Structure Anatomy Visualizer
st.header("4. Interactive Workbench: Document Structure Anatomy Visualizer", divider="rainbow")

st.markdown("""
Inspect how a multi-section document is partitioned into four-quadrant ChromaDB records 
with parent-child metadata pointers.
""")

col_p_text, col_p_cfg = col_p_text, col_p_cfg = st.columns([1.1, 0.9])

with col_p_text:
    st.subheader("1. Sample Parent Document Content")
    parent_doc_name = st.text_input("Parent Document Title:", value="Database_Disaster_Recovery_Policy.pdf")
    sample_content = st.text_area(
        "Source Document Sections (Decomposed by paragraph):",
        value="Section 1: Automated nightly backups stream SQLite WAL checkpoints directly to encrypted S3.\n\nSection 2: Disaster Recovery RTO SLA is defined as under 15 minutes with RPO under 1 hour.\n\nSection 3: Kubernetes StatefulSets automatically re-attach persistent NVMe storage on pod eviction.",
        height=100
    )

with col_p_cfg:
    st.subheader("2. Chunking & Modeling Strategy")
    chunk_overlap = st.slider("Chunk Overlap (Words):", min_value=0, max_value=10, value=0)
    add_parent_pointer = st.checkbox("Include `parent_doc_id` Metadata Pointer", value=True)
    add_timestamp = st.checkbox("Include Ingestion Timestamp Metadata", value=True)

# Parse paragraphs into chunks
raw_paragraphs = [p.strip() for p in sample_content.split("\n\n") if p.strip()]
decomposed_records = []

for idx, para in enumerate(raw_paragraphs):
    c_id = f"{parent_doc_name.replace('.pdf', '')}_chunk_{idx:02d}"
    meta = {
        "chunk_index": idx,
        "char_count": len(para)
    }
    if add_parent_pointer:
        meta["parent_doc_id"] = parent_doc_name
    if add_timestamp:
        meta["ingested_at"] = "2024-10-13T05:45:00Z"
        
    decomposed_records.append({
        "Quadrant 1: ID": c_id,
        "Quadrant 2: Document Text": para[:60] + "...",
        "Quadrant 3: Metadata": meta,
        "Quadrant 4: Vector Dimensions": "384 Floats (Built-in ONNX)"
    })

st.divider()

st.subheader(f"Generated ChromaDB Records ({len(decomposed_records)} Structured Records)")
st.caption("Each record maintains a 1:1 relationship between its unique ID, vector node, and metadata payload.")

for idx, rec in enumerate(decomposed_records):
    with st.expander(f"📦 Record {idx+1}: {rec['Quadrant 1: ID']}", expanded=True):
        st.markdown(f"**Document Content**: *{raw_paragraphs[idx]}*")
        st.json({
            "id": rec["Quadrant 1: ID"],
            "metadata": rec["Quadrant 3: Metadata"],
            "vector_embedding": "[0.041, -0.328, 0.771, ... 384 dimensions total]"
        })

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the 1:1 Geometric Invariant in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * In ChromaDB, every record represents an atomic 1:1 binding: **1 Unique ID <-> 1 Vector Node <-> 1 Document Payload**.
    * You cannot store multiple vectors under a single ID, nor can multiple documents share a single vector node in the HNSW index.
    * If a real-world document has multiple sections or images, it must be decomposed into multiple independent ChromaDB records, each with its own unique ID.
    """)

with st.expander("Q2: How does the Parent Document Retriever architecture work with ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **The Problem**: Small chunks (200 tokens) yield accurate vector search matches, but lack the broader context required for rich LLM generation. Large chunks (2,000 tokens) dilute embedding quality.
    * **The Architecture**:
      1. Slices a document into small child chunks for vector indexing in ChromaDB.
      2. Each child chunk carries a metadata pointer: `parent_doc_id: "doc_101"`.
      3. The full original parent document is stored in an external key-value store (e.g. Redis, MongoDB, or local disk).
      4. During search, Chroma finds the top child chunk, and the retriever looks up the **complete parent document** from storage to pass to the LLM!
    """)

with st.expander("Q3: Why can't we store arrays or nested objects in ChromaDB metadata?"):
    st.markdown("""
    **Answer**:
    * ChromaDB persists metadata inside **SQLite relational tables** where metadata keys map to indexed scalar columns.
    * Allowing arbitrary arrays (`tags: ["ai", "nlp"]`) or nested dictionaries (`address: {city: "NYC"}`) would require complex JSON path indexing, destroying SQLite query planning performance.
    * **Best Practice**:
      * Flatten dictionaries: `{"address_city": "NYC"}`.
      * Serialize arrays to comma-separated strings: `{"tags": "ai,nlp"}`.
    """)

with st.expander("Q4: What happens if an ingested document contains 0 characters or only whitespace?"):
    st.markdown("""
    **Answer**:
    * Passing an empty string `""` to Chroma's default embedding function triggers a tokenizer warning or generates a vector representing the start/end token embedding of the language model.
    * These degenerate vectors cluster near the origin or centroid, polluting the HNSW graph and occasionally matching arbitrary unrelated search queries.
    * **Rule**: Always filter out empty or pure whitespace chunks in your preprocessing ETL pipeline before calling `collection.add()`.
    """)

with st.expander("Q5: Can document IDs be changed or renamed after ingestion?"):
    st.markdown("""
    **Answer**:
    **No**. In ChromaDB, document IDs are immutable primary keys:
    * You cannot rename an existing ID directly.
    * To change an ID, you must fetch the record, insert it under the new ID, and delete the record under the old ID.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Lists Inside Metadata Dictionaries**  
    Passing `{"categories": ["sales", "enterprise"]}` raises validation exceptions. Convert lists to delimiter-separated strings (`"sales;enterprise"`).
    """)
    st.warning("""
    **Trap 2: Diluting Embeddings with Giant Document Chunks**  
    Ingesting 3,000-word chapters as single vectors destroys retrieval precision. Chunk into 200–400 word segments with parent metadata pointers.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Inconsistent ID Conventions Across Ingestion Pipelines**  
    Mixing `doc-101` and `DOC_101` causes case-sensitive duplicate accumulation. Normalize all IDs to lowercase before insertion.
    """)
    st.warning("""
    **Trap 4: Missing Parent Traceability Metadata**  
    Storing child chunks without recording their `parent_doc_id` and `chunk_index` prevents reconstructing the original source document for users.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What are the 4 record quadrants?"):
        st.markdown("""
        **Quad-Component Record**:  
        ID (primary key), Document (raw text), Metadata (attributes), and Embedding (dense vector).
        """)
    with st.expander("🗂️ Flashcard: What is the 1:1 Invariant?"):
        st.markdown("""
        **Atomic Mapping**:  
        Exactly 1 ID maps to 1 vector node and 1 document payload in ChromaDB.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Parent Document Pattern?"):
        st.markdown("""
        **Precision + Context**:  
        Index small child vectors in Chroma for precision; fetch full parent document for rich LLM context.
        """)
    with st.expander("🗂️ Flashcard: Can document IDs be renamed?"):
        st.markdown("""
        **Immutable Primary Keys**:  
        IDs cannot be renamed; you must re-insert with new ID and delete the old one.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the four quadrants of a ChromaDB record: ID, Document, Metadata, and Vector.", value=True)
chk2 = st.checkbox("I understand the 1:1 geometric mapping between IDs, HNSW nodes, and SQLite rows.", value=True)
chk3 = st.checkbox("I know how to implement the Parent Document Retriever pattern using metadata pointers.", value=True)
chk4 = st.checkbox("I understand how to validate flat metadata types using Pydantic schemas.", value=True)
chk5 = st.checkbox("I know why empty or pure whitespace chunks must be pruned before embedding.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 3.3: Updating Documents in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 3.2: Document Structure and are ready for Chapter 3.3: Updating Documents!")
