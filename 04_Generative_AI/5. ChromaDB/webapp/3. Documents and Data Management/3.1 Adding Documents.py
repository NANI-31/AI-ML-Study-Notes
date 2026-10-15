import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import uuid
import hashlib
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="3.1 Adding Documents - ChromaDB Curriculum",
    page_icon="📥",
    layout="wide"
)

# Title & Overview
st.title("3.1 Adding Documents: Batch Ingestion, ID Strategies, Metadata & Manual Embeddings")
st.caption("Chapter 3: Documents and Data Management | Domain: 5. ChromaDB")
st.markdown("""
Ingesting documents into ChromaDB is the primary bridge connecting raw unstructured information 
(PDFs, database records, web pages) to the vector space.

While calling `collection.add()` looks deceptively simple, production ingestion requires mastering 
critical engineering patterns:
* **Batch Ingestion & Optimal Chunking**: Sizing batch payloads to maximize throughput without triggering HTTP payload limit errors.
* **Deterministic ID Generation**: Deciding between random UUIDs, content-hash IDs (`sha256`), and hierarchical primary keys.
* **Metadata Schema Hygiene**: Flattening structured records to satisfy ChromaDB's primitive type constraints.
* **Pre-Computed Manual Embeddings**: Supplying external vectors directly (from OpenAI, Cohere, or local PyTorch models) rather than relying on Chroma's default model.

In this lesson, we explore automated vs manual ingestion pathways, evaluate ID generation strategies, 
write clean Python batch ingestion pipelines, and test an interactive Ingestion & ID Strategy Lab.
""")

st.divider()

# Section 1: The Two Ingestion Pathways: Automated vs Manual Embeddings
st.header("1. The Two Ingestion Pathways: Automated vs Manual Embeddings", divider="rainbow")

col_path1, col_path2 = col_path1, col_path2 = st.columns([1.1, 0.9])

with col_path1:
    st.markdown("""
    ### Pathway 1: Automated Ingestion (Default Model)
    * You provide **raw text strings**; ChromaDB automatically calls its configured `EmbeddingFunction` (e.g. `all-MiniLM-L6-v2`) via ONNX Runtime.
    * *Pros*: Simple; zero pre-computation; ideal for rapid prototyping and local development.
    * *Code*: `collection.add(ids=ids, documents=documents, metadatas=metadatas)`
    
    ### Pathway 2: Pre-Computed Manual Embeddings
    * You generate vectors externally using your own embedding pipeline (e.g., OpenAI `text-embedding-3-small`, Cohere `embed-v3`, or Triton GPU) and pass dense float arrays directly.
    * *Pros*: Full control over tokenization, batching, and embedding models; enables multi-modal (image/audio) vectors; 10x faster ingestion.
    * *Code*: `collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)`
    """)

with col_path2:
    st.markdown("### Ingestion Pathway Flowchart")
    path_dot = """
    digraph IngestionPathways {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawDocs [label="Raw Unstructured Documents\\n['Chunk 1', 'Chunk 2']", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QPath [label="Who generates the\\nvector embeddings?", shape=diamond, fillcolor="#FEF3C7"];

        AutoPath [label="Pathway 1: Automated (Chroma)\\nChroma calls internal ONNX MiniLM\\n(384-dim Float Arrays)", fillcolor="#D1FAE5", color="#059669"];

        ManualPath [label="Pathway 2: Pre-Computed (External)\\nClient calls OpenAI / Cohere / Triton\\nPasses embeddings=[[0.08, ...]]", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        HNSWStore [label="HNSW Geometric Index & SQLite\\nInserts Nodes into Graph\\nStores Text & Metadata in Tables", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        RawDocs -> QPath;
        QPath -> AutoPath [label="Omit embeddings"];
        QPath -> ManualPath [label="Pass embeddings"];
        AutoPath -> HNSWStore;
        ManualPath -> HNSWStore;
    }
    """
    st.graphviz_chart(path_dot, use_container_width=False)

st.divider()

# Section 2: ID Assignment Strategies in Production
st.header("2. ID Assignment Strategies in Production", divider="rainbow")

st.markdown("""
Every document in ChromaDB must possess a unique string identifier (`id`). Choosing the right ID strategy 
dictates how your system handles updates, deduplication, and synchronization:
""")

ID_STRATEGIES = [
    {
        "Strategy Name": "1. Content-Hash ID (`sha256`)",
        "Implementation Pattern": "`hashlib.sha256(text.encode()).hexdigest()[:16]`",
        "Primary Advantage": "**Automatic Deduplication**: Identical text passages automatically share the exact same ID, preventing redundant duplicate vectors.",
        "Production Drawback": "If text is updated by one character, a new ID is generated, leaving the old version in the index unless explicitly purged."
    },
    {
        "Strategy Name": "2. Hierarchical Entity Key",
        "Implementation Pattern": "`f'{source_type}:{doc_id}:chunk_{chunk_idx}'` (e.g. `wiki:article_42:chunk_0`)",
        "Primary Advantage": "**Deterministic Traceability**: Allows instant retrieval or deletion of all chunks belonging to a specific parent document.",
        "Production Drawback": "Requires upstream document parser to maintain stable parent and chunk indexing schemes."
    },
    {
        "Strategy Name": "3. Random UUID (`uuid.uuid4()`)",
        "Implementation Pattern": "`str(uuid.uuid4())`",
        "Primary Advantage": "**Guaranteed Global Uniqueness**: Zero possibility of ID collisions across distributed ingestion workers.",
        "Production Drawback": "Complete loss of idempotency. Ingesting the same dataset twice duplicates every vector in the database."
    }
]

st.dataframe(pd.DataFrame(ID_STRATEGIES), use_container_width=True, hide_index=True)

st.divider()

# Section 3: Clean Python Batch Ingestion Implementations
st.header("3. Clean Python Batch Ingestion Implementations", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. Automated Batch Ingestion with Hashing")
    st.code("""import chromadb
import hashlib
from typing import List, Dict

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("automated_kb")

raw_articles = [
    "PostgreSQL WAL archives stream continuously to S3 for disaster recovery.",
    "Kubernetes StatefulSets maintain sticky network identities and volume attachments."
]

# Generate deterministic content-hash IDs
doc_ids = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in raw_articles]
metadata_list = [
    {"source": "disaster_recovery.md", "domain": "database"},
    {"source": "k8s_guide.md", "domain": "infrastructure"}
]

# Automated: Chroma embeds text using default model
collection.add(
    ids=doc_ids,
    documents=raw_articles,
    metadatas=metadata_list
)
print(f"Successfully ingested {collection.count()} documents.")
""", language="python")

with col_code2:
    st.markdown("### 2. Manual Pre-Computed Embeddings")
    st.code("""import chromadb
from typing import List

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="manual_embeddings_kb",
    metadata={"hnsw:space": "cosine"}
)

# Suppose vectors were computed via OpenAI / Cohere API
precomputed_vectors = [
    [0.042, -0.128, 0.884, 0.219],  # Float vector 1
    [-0.312, 0.441, 0.052, 0.771]   # Float vector 2
]

# Manual: Pass precomputed vectors directly
collection.add(
    ids=["ext_01", "ext_02"],
    embeddings=precomputed_vectors,
    documents=[
        "Raw text for document 1 (kept for prompt hydration)",
        "Raw text for document 2 (kept for prompt hydration)"
    ],
    metadatas=[{"source": "external_api"}, {"source": "external_api"}]
)
print("Ingested precomputed vectors without Chroma embedding overhead!")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Ingestion & ID Strategy Lab
st.header("4. Interactive Workbench: Ingestion & ID Strategy Lab", divider="rainbow")

st.markdown("""
Test how different ID generation strategies format document primary keys and inspect the resulting 
parallel ingestion payload structure before it is committed to ChromaDB.
""")

col_wb_src, col_wb_cfg = col_wb_src, col_wb_cfg = st.columns([1.1, 0.9])

with col_wb_src:
    st.subheader("1. Sample Raw Chunks")
    sample_text_1 = st.text_input("Chunk 1 Text:", value="Envoy proxies enforce mutual TLS across internal microservices.")
    sample_text_2 = st.text_input("Chunk 2 Text:", value="Redis cluster caches high-frequency queries to provide sub-millisecond responses.")

with col_wb_cfg:
    st.subheader("2. Ingestion Configuration")
    id_strategy = st.radio(
        "Choose ID Generation Strategy:",
        [
            "Content-Hash (SHA-256 Digest)",
            "Hierarchical Key (source:doc_id:chunk_idx)",
            "Random UUID (UUIDv4)"
        ]
    )
    doc_source = st.text_input("Metadata Source Document:", value="infra_runbook.pdf")
    include_manual_vec = st.checkbox("Simulate Pre-Computed Manual Vectors (Custom Floats)", value=False)

# Compute IDs
if "Content-Hash" in id_strategy:
    id_1 = f"sha_{hashlib.sha256(sample_text_1.encode()).hexdigest()[:12]}"
    id_2 = f"sha_{hashlib.sha256(sample_text_2.encode()).hexdigest()[:12]}"
elif "Hierarchical" in id_strategy:
    clean_src = doc_source.replace(".", "_")
    id_1 = f"{clean_src}:chunk_0"
    id_2 = f"{clean_src}:chunk_1"
else:
    id_1 = f"uuid_{uuid.uuid4().hex[:12]}"
    id_2 = f"uuid_{uuid.uuid4().hex[:12]}"

st.divider()

st.subheader("Constructed Parallel Ingestion Payload (`collection.add()`)")

ingest_payload = {
    "ids": [id_1, id_2],
    "documents": [sample_text_1, sample_text_2],
    "metadatas": [
        {"source": doc_source, "chunk_index": 0, "char_length": len(sample_text_1)},
        {"source": doc_source, "chunk_index": 1, "char_length": len(sample_text_2)}
    ]
}

if include_manual_vec:
    ingest_payload["embeddings"] = [
        [0.125, -0.442, 0.819, 0.052],
        [-0.301, 0.228, 0.612, 0.941]
    ]

st.json(ingest_payload)

if "Content-Hash" in id_strategy:
    st.success("🟢 **Deduplication Active**: If Chunk 1 is ingested again, it will produce the exact same ID, preventing duplicate records!")
elif "Hierarchical" in id_strategy:
    st.info("💡 **Traceable Hierarchy**: Chunks can be bulk-deleted or filtered by source prefix (`source_runbook_pdf:*`).")
else:
    st.warning("⚠️ **Non-Idempotent**: Re-running ingestion with random UUIDs will create duplicate entries in the collection.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What are the risks of using random UUIDs for document IDs in vector ingestion pipelines?"):
    st.markdown("""
    **Answer**:
    * **Loss of Idempotency**: If a nightly batch ingestion pipeline or crawler runs multiple times, random UUIDs assign a brand new identifier to every chunk on every run.
    * This causes **exponential index bloat**: after 10 runs, your collection contains 10 identical copies of every single vector!
    * Downstream search queries will return 5 identical duplicates in the Top-K results.
    * **Production Remedy**: Use **Content-Hash IDs** (`sha256(chunk_text)`) or **Hierarchical Primary Keys** (`f"{doc_id}:chunk_{idx}"`) paired with `collection.upsert()`.
    """)

with st.expander("Q2: What is the optimal batch size when adding documents to ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **Recommended Batch Size**: **500 to 2,000 documents per batch**.
    * **Why Not 1-by-1?**: Adding documents one at a time incurs SQLite transaction lock overhead and network round-trips on every call, dropping throughput to ~20 docs/sec.
    * **Why Not 50,000 at once?**: Huge batches exceed HTTP payload limits (e.g. 100MB body limits on reverse proxies like Nginx/Envoy) and cause out-of-memory spikes during ONNX tensor batching.
    * Slicing large datasets into chunks of 1,000 achieves maximum throughput (2,000–5,000 docs/sec).
    """)

with st.expander("Q3: Can you store raw documents without embeddings, or embeddings without raw documents?"):
    st.markdown("""
    **Answer**:
    * **Embeddings without Documents**: **Yes**. You can call `collection.add(ids=ids, embeddings=vectors)`. This saves storage space if you store raw documents in an external data lake (e.g., S3 or MongoDB) and only need ChromaDB for vector retrieval.
    * **Documents without Embeddings**: **Yes, provided an EmbeddingFunction is configured**. Chroma automatically converts document strings into vectors. If NO embedding function is configured and NO embeddings are supplied, Chroma raises an error.
    """)

with st.expander("Q4: Why does ChromaDB reject nested dictionaries in the `metadatas` parameter?"):
    st.markdown("""
    **Answer**:
    * ChromaDB persists metadata inside **SQLite relational tables** with dedicated scalar columns and inverted index structures.
    * To support fast SQL boolean filtering (`where={"author": "Alice", "year": {"$gte": 2023}}`), columns must be flat primitive scalar types (`str`, `int`, `float`, `bool`).
    * Nested dictionaries require JSON path traversals that cannot easily leverage B-Tree or SQLite inverted indices without substantial latency degradation.
    * **Solution**: Flatten nested structures before ingestion (`{"geo_city": "Austin", "geo_state": "TX"}`).
    """)

with st.expander("Q5: What happens if parallel arrays in `collection.add()` have mismatched lengths?"):
    st.markdown("""
    **Answer**:
    * ChromaDB raises an immediate `ValueError: Number of ids, documents, and metadatas must match`.
    * If `len(ids) == 10` but `len(documents) == 9`, the entire call fails atomically before modifying SQLite or the HNSW graph.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Using `uuid.uuid4()` Without Upsert**  
    Ingesting data with random UUIDs duplicates all vectors on pipeline retries. Use content-hashes or deterministic document chunk IDs.
    """)
    st.warning("""
    **Trap 2: Sending Single-Document Ingestion Calls in a Loop**  
    Calling `collection.add()` inside a 10,000-iteration for-loop creates 10,000 disk sync transactions. Batch documents into chunks of 1,000.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Passing Nested Metadata Dictionaries**  
    Passing `{"author": {"first": "John", "last": "Doe"}}` raises serialization exceptions. Flatten to scalar keys before ingestion.
    """)
    st.warning("""
    **Trap 4: Forgetting to Strip Empty / Whitespace Strings**  
    Passing empty strings `""` or pure whitespace chunks to embedding models wastes index memory and generates uninformative centroid vectors.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Content-Hash ID?"):
        st.markdown("""
        **Automatic Deduplication**:  
        `sha256(text)[:16]` ensures identical chunks share the exact same ID, preventing redundant vectors.
        """)
    with st.expander("🗂️ Flashcard: What is the optimal batch size?"):
        st.markdown("""
        **500 to 2,000 Chunks**:  
        Maximizes throughput without triggering HTTP payload limits or memory spikes.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Can you omit raw documents?"):
        st.markdown("""
        **Yes**:  
        You can pass only `ids` and `embeddings` if raw documents are persisted in an external database.
        """)
    with st.expander("🗂️ Flashcard: Why flatten metadata?"):
        st.markdown("""
        **SQLite Scalar Indexing**:  
        Only flat primitive types (`str`, `int`, `float`, `bool`) can be indexed for high-speed filtering.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the difference between automated embedding ingestion and manual pre-computed vectors.", value=True)
chk2 = st.checkbox("I know how to implement Content-Hash and Hierarchical ID generation strategies.", value=True)
chk3 = st.checkbox("I understand why random UUIDs break idempotency and cause duplicate vector accumulation.", value=True)
chk4 = st.checkbox("I know how to batch document ingestion (500–2,000 docs) for optimal throughput.", value=True)
chk5 = st.checkbox("I understand metadata flattening rules and primitive type constraints in ChromaDB.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 3.2: Document Structure in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 3.1: Adding Documents and are ready for Chapter 3.2: Document Structure!")
