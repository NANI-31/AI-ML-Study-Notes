import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Optional

# Attempt chromadb import with safe fallback
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(
    page_title="14.1 Basic ChromaDB Integration - Embeddings Curriculum",
    page_icon="🧪",
    layout="wide"
)

# Title & Overview
st.title("14.1 Basic ChromaDB Integration: Collections, Metadata, & Ingestion")
st.caption("Chapter 14: Embeddings with ChromaDB | Domain: 3. Vector Embeddings")
st.markdown("""
ChromaDB is an open-source, AI-native embedded vector database designed to make it simple and fast to build LLM applications with semantic search.
Unlike standalone client-server vector databases like Milvus or cloud-hosted engines like Pinecone, ChromaDB can run **completely in-process inside your Python application**, storing metadata in SQLite and vector indexes in HNSW.

In this lesson, you will master the foundational data ingestion pipeline of ChromaDB:
* **Client Initialization**: Creating ephemeral in-memory vs. persistent storage clients.
* **Collections**: The primary organizational unit analogous to a table in SQL or a bucket in S3.
* **Embedding Generation**: ChromaDB's automatic embedding functions vs. manual pre-computed embedding injection.
* **Document Ingestion**: Passing documents, metadata dictionaries, and unique string identifiers into the collection.
* **Storage Schema**: Understanding how ChromaDB handles vector arrays, metadata key-value pairs, and unique IDs.
""")

st.divider()

# Section 1: ChromaDB Core Architecture & Data Model
st.header("1. ChromaDB Core Architecture & Ingestion Lifecycle", divider="rainbow")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    ### Key Architectural Components
    * **1. Client (`chromadb.Client`)**:
      * The primary entry point. Can be ephemeral (`EphemeralClient` / in-memory) or disk-backed (`PersistentClient`).
      * Coordinates collection management and database telemetry.
    * **2. Collection (`Collection`)**:
      * A named container for embeddings, documents, and metadata.
      * Each collection configures an **HNSW index** with a defined distance metric (`cosine`, `l2`, or `ip`).
    * **3. Embedding Function (`EmbeddingFunction`)**:
      * ChromaDB can automatically embed raw text using its built-in default (all-MiniLM-L6-v2 via ONNX runtime) or custom provider wrappers (OpenAI, HuggingFace, Cohere).
      * Alternatively, you can supply your own pre-calculated vectors directly to `embeddings=[...]`.
    * **4. Dual-Store Engine**:
      * **SQLite**: Stores document texts, metadata JSON records, and document IDs with relational integrity.
      * **HNSWlib**: Maintains the hierarchical navigable small-world graph index for sub-millisecond Approximate Nearest Neighbor (ANN) vector lookups.
    """)

with col_arch2:
    st.markdown("### ChromaDB Ingestion Flowchart")
    flow_dot = """
    digraph ChromaIngest {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        DocsIn [label="Raw Text Documents\\n+ Metadata Dicts + String IDs", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        EF [label="Embedding Function\\n(Default ONNX / OpenAI / SentenceTransformers)", fillcolor="#FEF3C7", color="#D97706"];
        Collect [label="ChromaDB Collection\\ncollection.add(...)", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];

        subgraph cluster_dual_store {
            label="Internal Dual-Storage Engine"; style="dashed"; color="#6B7280";
            SQLStore [label="SQLite Store\\nDocument Text, Metadata, & IDs", fillcolor="#F3F4F6"];
            HNSWStore [label="HNSWlib Graph Index\\nNormalized Vector Embeddings", fillcolor="#EDE9FE", color="#8B5CF6"];
        }

        DocsIn -> EF [label="Extract text"];
        EF -> Collect [label="Dense vectors (d=384)"];
        DocsIn -> Collect [label="Metadata & IDs"];
        Collect -> SQLStore [label="Write payload"];
        Collect -> HNSWStore [label="Index vectors"];
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.markdown("### ChromaDB Collection Configuration Parameters")
st.markdown("""
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Yes | Collection name (3-63 chars, alphanumeric, dots, underscores, hyphens). |
| `metadata` | `dict` | No | Collection-level configuration, notably `{"hnsw:space": "cosine"}` or `l2` or `ip`. |
| `embedding_function` | `EmbeddingFunction` | No | Function converting text to vectors. If omitted, uses default ONNX MiniLM. |
| `get_or_create` | `bool` | No | Flag in `get_or_create_collection()` ensuring idempotency without exceptions. |
""")

st.divider()

# Section 2: Production Code: Step-by-Step ChromaDB Ingestion
st.header("2. Production Code: Initializing & Populating Collections", divider="rainbow")

st.markdown("""
The script below demonstrates best-practice production ingestion: configuring an ephemeral client, defining an explicit distance metric (`cosine`), preparing metadata dictionaries, and adding records in batches.
""")

st.code(r'''
import chromadb
from typing import List, Dict, Any

def create_and_populate_collection():
    # 1. Initialize an in-memory ephemeral client
    client = chromadb.EphemeralClient()

    # 2. Define or retrieve the target collection with cosine distance metric
    collection = client.get_or_create_collection(
        name="ai_research_papers",
        metadata={"hnsw:space": "cosine", "description": "AI research corpus"}
    )

    # 3. Prepare documents, metadata, and unique IDs
    documents: List[str] = [
        "Attention Is All You Need introduced the Transformer architecture based entirely on self-attention mechanisms.",
        "BERT utilizes masked language modeling to learn bidirectional context representations from unlabeled text.",
        "Deep Residual Learning for Image Recognition introduced skip connections to train ultra-deep networks.",
        "LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into Transformer layers."
    ]

    metadatas: List[Dict[str, Any]] = [
        {"author": "Vaswani et al.", "year": 2017, "domain": "nlp", "citations": 98000},
        {"author": "Devlin et al.", "year": 2018, "domain": "nlp", "citations": 82000},
        {"author": "He et al.", "year": 2015, "domain": "vision", "citations": 165000},
        {"author": "Hu et al.", "year": 2021, "domain": "peft", "citations": 14000}
    ]

    ids: List[str] = [
        "doc_transformer_01",
        "doc_bert_02",
        "doc_resnet_03",
        "doc_lora_04"
    ]

    # 4. Ingest records (ChromaDB automatically generates embeddings using default MiniLM)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Collection successfully created! Total items: {collection.count()}")
    
    # 5. Inspect the first stored record
    sample = collection.get(ids=["doc_transformer_01"], include=["documents", "metadatas", "embeddings"])
    print(f"Sample Document: {sample['documents'][0]}")
    print(f"Sample Metadata: {sample['metadatas'][0]}")
    print(f"Embedding Dimensions: {len(sample['embeddings'][0]) if sample['embeddings'] is not None else 'Default'}")

if __name__ == "__main__":
    create_and_populate_collection()
''', language="python")

st.divider()

# Section 3: Interactive ChromaDB Ingestion Workbench
st.header("3. Interactive Workbench: Live ChromaDB Collection Builder", divider="rainbow")

st.markdown("""
Test live collection creation, custom document ingestion, metadata assignment, and inspection directly inside this interactive sandbox.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("1. Collection Settings")
    coll_name = st.text_input("Collection Name", value="enterprise_knowledge_base")
    metric_choice = st.selectbox(
        "HNSW Distance Space (`hnsw:space`)",
        options=["cosine", "l2", "ip"],
        index=0,
        help="cosine: Cosine distance (1 - cos). l2: Squared Euclidean distance. ip: Inner product."
    )

    st.subheader("2. Add Sample Documents")
    preset = st.radio(
        "Select Document Dataset:",
        ["Cloud Computing Docs", "Healthcare AI Notes", "Custom Document Entry"],
        horizontal=True
    )

    if preset == "Cloud Computing Docs":
        sample_docs = [
            ("doc_aws_01", "Amazon S3 provides scalable object storage with eleven nines of durability.", "aws", "storage", 2023),
            ("doc_gcp_02", "Google BigQuery is a serverless, highly scalable cloud data warehouse for SQL analytics.", "gcp", "analytics", 2024),
            ("doc_k8s_03", "Kubernetes automates deployment, scaling, and operations of containerized applications.", "k8s", "devops", 2022),
            ("doc_azure_04", "Azure Cosmos DB offers globally distributed, multi-model NoSQL database service.", "azure", "database", 2023)
        ]
    elif preset == "Healthcare AI Notes":
        sample_docs = [
            ("doc_med_01", "Convolutional neural networks detect diabetic retinopathy from retinal fundus images.", "ophthalmology", "vision", 2023),
            ("doc_med_02", "Clinical decision support systems use rule-based engines and LLMs to flag drug interactions.", "pharmacology", "nlp", 2024),
            ("doc_med_03", "Diffusion models synthesize high-resolution MRI scans for rare pathological training data.", "radiology", "generative", 2024),
            ("doc_med_04", "Transformer models summarize longitudinal electronic health records for discharge summaries.", "ehr", "nlp", 2023)
        ]
    else:
        sample_docs = [
            ("custom_01", "Vector databases index embeddings using hierarchical graphs for fast nearest neighbor lookups.", "engineering", "ai", 2024),
            ("custom_02", "Cross-encoders evaluate query-document pairs simultaneously for accurate reranking.", "information_retrieval", "ai", 2024)
        ]

    st.write(f"**Loaded {len(sample_docs)} documents ready for ingestion:**")
    df_preview = pd.DataFrame(sample_docs, columns=["ID", "Document", "Category", "Tag", "Year"])
    st.dataframe(df_preview, use_container_width=True)

with wb_col2:
    st.subheader("3. Execute Ingestion & Inspect Storage")
    
    if st.button("🚀 Ingest into ChromaDB Collection", type="primary"):
        with st.spinner("Initializing ChromaDB client and generating embeddings..."):
            try:
                # Initialize ephemeral client
                client = chromadb.EphemeralClient()
                collection = client.get_or_create_collection(
                    name=coll_name,
                    metadata={"hnsw:space": metric_choice}
                )

                # Prepare payload
                doc_ids = [d[0] for d in sample_docs]
                doc_texts = [d[1] for d in sample_docs]
                doc_metas = [{"category": d[2], "tag": d[3], "year": d[4]} for d in sample_docs]

                # Add to collection
                collection.add(
                    ids=doc_ids,
                    documents=doc_texts,
                    metadatas=doc_metas
                )

                st.success(f"Successfully ingested {collection.count()} documents into collection '{coll_name}'!")

                # Retrieve all stored data with embeddings
                results = collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )

                st.markdown("#### Ingestion Telemetry")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Count", collection.count())
                col_m2.metric("Distance Metric", metric_choice.upper())
                emb_dim = len(results["embeddings"][0]) if results["embeddings"] is not None and len(results["embeddings"]) > 0 else 384
                col_m3.metric("Vector Dimension", emb_dim)

                st.markdown("#### Stored Records Inspection")
                records = []
                for i in range(len(results["ids"])):
                    emb_snippet = results["embeddings"][i][:4] if results["embeddings"] is not None else [0.0]
                    emb_str = "[" + ", ".join([f"{x:.4f}" for x in emb_snippet]) + ", ...]"
                    records.append({
                        "ID": results["ids"][i],
                        "Document Snippet": results["documents"][i][:60] + "...",
                        "Metadata": str(results["metadatas"][i]),
                        "Vector Preview (First 4 Dim)": emb_str
                    })
                
                st.dataframe(pd.DataFrame(records), use_container_width=True)

            except Exception as e:
                st.error(f"Error during ChromaDB execution: {str(e)}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "ChromaDB Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What happens if you add documents to ChromaDB without specifying `embeddings=[...]`?**
    * **Answer**: ChromaDB invokes its built-in default embedding function, which runs `all-MiniLM-L6-v2` packaged via ONNX runtime. It generates 384-dimensional dense vectors locally on your CPU/GPU without needing an external API key.

    **Q2: What is the difference between `collection.add()`, `collection.update()`, and `collection.upsert()` in ChromaDB?**
    * **Answer**:
      * `collection.add()`: Inserts new records. If any provided `id` already exists in the collection, ChromaDB raises an `IDAlreadyExistsError` and rejects the operation.
      * `collection.update()`: Updates existing records matching the provided IDs. If an ID does not exist, ChromaDB raises an error.
      * `collection.upsert()`: Idempotent operation. If an ID exists, it updates documents/metadata/embeddings; if the ID does not exist, it inserts the record. Best suited for data pipelines and synchronization workers.

    **Q3: How does ChromaDB store metadata and what data types are supported?**
    * **Answer**: ChromaDB stores metadata as key-value dictionaries alongside documents in SQLite. Allowed metadata value types are strictly primitive types: `str`, `int`, `float`, and `bool`. Nested dictionaries, lists, and complex objects are rejected and must be serialized to JSON strings or flattened before ingestion.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: ID Collision during `collection.add()`**:
      * *Issue*: Re-running an ingestion script calling `collection.add()` with deterministic document IDs.
      * *Consequence*: The script crashes with `IDAlreadyExistsError` and halts the pipeline mid-batch.
      * *Fix*: Always use `collection.upsert()` in ETL pipelines, or use `collection.delete()` before re-ingesting.

    * **Trap 2: Storing Nested Dictionaries or Lists in Metadata**:
      * *Issue*: Passing `metadatas=[{"tags": ["ai", "nlp"], "author": {"name": "Alice"}}]`.
      * *Consequence*: ChromaDB raises a `ValueError: Expected metadata value to be a str, int, float or bool`.
      * *Fix*: Flatten nested structures (`author_name="Alice"`) or serialize lists to strings (`tags="ai,nlp"`).

    * **Trap 3: Mismatched Embedding Dimensions**:
      * *Issue*: Adding vectors of dimension 384 initially, and later adding 1536-dimensional OpenAI embeddings to the same collection.
      * *Consequence*: ChromaDB rejects the insertion with dimension mismatch errors because an HNSW index has a fixed dimensional topology.
      * *Fix*: Never mix different embedding models in a single collection. Create a separate collection for each model.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `chromadb.EphemeralClient()`**
      * *Definition*: An in-memory ChromaDB instance that stores vectors and metadata purely in RAM, ideal for unit tests and temporary sessions.
    * **Flashcard 2: `hnsw:space`**
      * *Definition*: The collection metadata key defining the vector distance metric: `cosine` (cosine distance), `l2` (squared Euclidean), or `ip` (inner product).
    * **Flashcard 3: `collection.upsert()`**
      * *Definition*: The idempotent method that inserts non-existent IDs and updates existing IDs without throwing collision errors.
    * **Flashcard 4: Primitive Metadata Constraint**
      * *Definition*: ChromaDB's requirement that metadata dictionary values must be only strings, integers, floats, or booleans.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Basic ChromaDB Integration:

- [ ] I can explain the difference between `EphemeralClient` and `PersistentClient`.
- [ ] I know how to configure the distance space (`cosine`, `l2`, `ip`) using collection metadata.
- [ ] I understand how ChromaDB uses its default ONNX embedding function when no vectors are passed.
- [ ] I can distinguish the operational semantics of `add()`, `update()`, and `upsert()`.
- [ ] I know the strict primitive type constraints on ChromaDB metadata dictionaries.
""")
