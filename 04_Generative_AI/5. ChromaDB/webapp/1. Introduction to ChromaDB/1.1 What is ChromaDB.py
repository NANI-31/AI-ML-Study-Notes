import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.1 What is ChromaDB - ChromaDB Curriculum",
    page_icon="🎨",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulation
class MockChromaCollection:
    """Simulates an embedded ChromaDB collection with in-memory cosine distance querying."""
    def __init__(self, name: str):
        self.name = name
        self.records = [
            {"id": "doc_1", "document": "ChromaDB is an AI-native open-source embedding database.", "metadata": {"topic": "overview", "tier": "free"}, "vector": np.array([0.92, 0.15, 0.08, 0.12])},
            {"id": "doc_2", "document": "HNSW graphs enable approximate nearest neighbor search in sub-millisecond time.", "metadata": {"topic": "indexing", "tier": "core"}, "vector": np.array([0.18, 0.89, 0.22, 0.05])},
            {"id": "doc_3", "document": "SQLite persists collection metadata, document IDs, and index configurations.", "metadata": {"topic": "storage", "tier": "core"}, "vector": np.array([0.05, 0.20, 0.95, 0.10])},
            {"id": "doc_4", "document": "Retrieval-Augmented Generation injects private context into LLM prompts.", "metadata": {"topic": "rag", "tier": "prod"}, "vector": np.array([0.12, 0.10, 0.15, 0.94])}
        ]

    def query(self, query_vector: np.ndarray, n_results: int = 2, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        norm_q = np.linalg.norm(query_vector)
        q_norm = query_vector / norm_q if norm_q > 0 else query_vector

        scored = []
        for r in self.records:
            if where:
                match = all(r["metadata"].get(k) == v for k, v in where.items())
                if not match:
                    continue

            v = r["vector"]
            norm_v = np.linalg.norm(v)
            cosine_sim = float(np.dot(q_norm, v) / norm_v) if norm_v > 0 else 0.0
            # Cosine Distance = 1.0 - Cosine Similarity
            cosine_dist = max(0.0, 1.0 - cosine_sim)
            scored.append({
                "id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "distance": round(cosine_dist, 4)
            })

        scored.sort(key=lambda x: x["distance"])
        return scored[:n_results]

# Title & Overview
st.title("1.1 What is ChromaDB? Definition, Architecture & Comparative Analysis")
st.caption("Chapter 1: Introduction to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
In the era of Generative AI, foundation models like GPT-4o, Claude 3.5, and Gemini communicate through 
**continuous, high-dimensional vector embeddings**. While relational databases excel at exact equality lookups 
and transactions, they cannot determine that *"automobile"* and *"car"* refer to the identical underlying concept.

**ChromaDB** is the leading open-source, AI-native vector database engineered specifically for developer simplicity, 
rapid local experimentation, and production Retrieval-Augmented Generation (RAG).

Unlike heavy distributed vector databases that demand complex Kubernetes clusters just to run a test script, 
ChromaDB is **embedded-first**: it runs in-process with a single `import chromadb`, manages indices in memory, 
and persists seamlessly to local disk or remote client-server architectures.

In this lesson, you will master:
1. **The ChromaDB Architecture**: In-memory, persistent, and client-server topologies.
2. **Core Advantages**: Batteries-included embeddings, SQLite persistence, and HNSW indexing.
3. **Comparative Systems Evaluation**: ChromaDB vs traditional databases, SQL, and distributed stores like Milvus and Pinecone.
4. **Production Code Implementation**: Creating collections, upserting documents, and querying with metadata filters.
5. **Production Failure Traps**: Avoiding SQLite concurrency locking, dimension mismatches, and ephemeral data loss.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: ChromaDB Embedded & Client-Server Topology", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChromaDBBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_client {
            label="Tier 1: Chroma Client and Embedding Layer";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            app_code [label="Application Layer (FastAPI / Streamlit / Notebook)\\nCalls chromadb.PersistentClient(path='./chroma_db')", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            ef_model [label="Embedding Function (Default ONNX all-MiniLM-L6-v2)\\nTransforms text strings into 384-d dense vectors", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            app_code -> ef_model [label="Embed text payloads"];
        }

        subgraph cluster_storage {
            label="Tier 2: Collection Management and Storage Engine";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            sqlite_db [label="Metadata and State Store (chroma.sqlite3)\\nCollections, Document IDs, Metadata Dictionaries, WAL", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            hnsw_index [label="Vector Index Segments (hnswlib C++ Engine)\\nHierarchical Proximity Graph (Cosine / L2 / IP)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            parquet_store [label="Payload and Segment Storage\\nImmutable parquet blocks and document text records", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            ef_model -> sqlite_db [label="Upsert IDs & Metadata"];
            ef_model -> hnsw_index [label="Add dense vectors"];
            sqlite_db -> parquet_store [label="Sync segments"];
        }

        subgraph cluster_retrieval {
            label="Tier 3: Query Execution, Distance Metrics, and Filter Engine";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            user_query [label="Incoming Search Query\\ncollection.query(query_texts=[...], where={...}, n_results=5)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            query_filter [label="Metadata Filter Engine ($and, $or, $in)\\nEvaluates SQLite metadata predicates", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            ann_search [label="HNSW Graph Search\\nFinds top-K nearest neighbors by Cosine distance", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            results_payload [label="Final Grounded Results\\nRanked documents + metadata + computed distances", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            user_query -> query_filter;
            user_query -> ann_search;
            sqlite_db -> query_filter;
            hnsw_index -> ann_search;
            query_filter -> results_payload;
            ann_search -> results_payload;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Definition of ChromaDB
st.header("2. Definition of ChromaDB", divider="rainbow")
st.markdown("""
**ChromaDB** is an open-source, AI-native embedding database engineered to make it simple to build LLM applications 
by making knowledge, facts, and skills pluggable for machine learning models.

* **Embedded-First Architecture**: ChromaDB runs directly within your Python process without requiring an external database daemon, server, or cloud account.
* **Batteries-Included Embeddings**: It automatically bundles a local ONNX-accelerated Sentence Transformers model (`all-MiniLM-L6-v2`), so developers can add raw strings directly without calling external embedding APIs.
* **Stateful Persistence**: Automatically writes metadata, document payloads, and HNSW proximity graph files to a local directory (`chroma.sqlite3` and segment folders).
* **Open Source & Extensible**: Fully open-source Apache 2.0 license with native support for Python, JavaScript/TypeScript, and Go.
""")
st.divider()

# Section 3: What is a Vector Database?
st.header("3. What is a Vector Database?", divider="rainbow")
st.markdown("""
A **vector database** is a dedicated database system optimized to store, index, and query high-dimensional vector embeddings:

* **High-Dimensional Geometry**: Standard databases index scalars (numbers, text) in 1 dimension. Vector databases index continuous numerical arrays across 384, 768, or 1536 dimensions.
* **Approximate Nearest Neighbor (ANN)**: Searches through millions of vectors in single-digit milliseconds using graph skip-lists (HNSW) instead of brute-force full table scans.
* **Payload Association**: Associates vector coordinates with raw document strings and JSON metadata attributes, returning complete records upon retrieval.
""")
st.divider()

# Section 4: Why ChromaDB is Important
st.header("4. Why ChromaDB is Important: The Developer Ergonomics Advantage", divider="rainbow")
st.markdown("""
Before ChromaDB, setting up vector search required choosing between two extremes:
1. **Low-Level Math Libraries (FAISS, hnswlib)**: Blazing fast, but completely lack persistence, metadata filtering, document storage, and CRUD operations. Developers had to write custom glue code to map FAISS vector IDs back to relational SQL database rows.
2. **Heavyweight Distributed Vector Clusters (Milvus, Vespa)**: Feature-rich, but require Docker Compose files, Kubernetes operators, MinIO S3 object storage, and etcd clusters just to index a single PDF document.

ChromaDB bridged this gap by delivering **database-grade persistence and filtering with the 3-line simplicity of a lightweight Python library**:
""")

st.code('''
import chromadb

client = chromadb.Client()
collection = client.create_collection("quickstart")
collection.add(documents=["This is a document"], ids=["id1"])
results = collection.query(query_texts=["This is a query"], n_results=1)
''', language="python")
st.divider()

# Section 5: ChromaDB vs Traditional Databases
st.header("5. ChromaDB vs Traditional Databases", divider="rainbow")
st.markdown("""
Comparing ChromaDB against traditional relational (PostgreSQL) and NoSQL (MongoDB) databases reveals divergent architectures:
""")

st.markdown("""
| Dimension | Traditional Relational / NoSQL Database | ChromaDB |
| :--- | :--- | :--- |
| **Search Paradigm** | Exact equality, range matching, regex (`WHERE status = 'PAID'`) | Geometric semantic proximity (`Find top-K nearest concepts`) |
| **Primary Data Structure**| B-Trees, Hash Tables, Document BSON | HNSW Proximity Graphs + SQLite Metadata Index |
| **Query Latency on Vectors**| Degrades to `O(N)` full table scan on high dimensions | `O(log N)` sub-millisecond graph traversal |
| **Handling Synonyms** | Requires manual synonym dictionaries or full-text analyzers | Natively resolves synonyms via latent embedding geometry |
| **ACID Guarantees** | Full ACID transactions with multi-version concurrency (MVCC) | Lightweight SQLite transactional state, optimized for reads |
""")
st.divider()

# Section 6: ChromaDB vs SQL Databases
st.header("6. ChromaDB vs SQL Databases: When to Choose Each", divider="rainbow")
st.markdown("""
When architecting production GenAI systems, developers often evaluate ChromaDB against PostgreSQL with `pgvector`:
""")

st.markdown("""
| Feature | ChromaDB | PostgreSQL + `pgvector` |
| :--- | :--- | :--- |
| **Setup Overhead** | Zero infrastructure (`pip install chromadb`) | Requires PostgreSQL database instance + extension installation |
| **Default Embeddings** | Bundled local ONNX model (automatic embedding) | None (developer must generate and pass vectors manually) |
| **Relational Joins** | Not supported (denormalized metadata payloads) | Full relational SQL joins with existing transactional tables |
| **Multi-Tenancy** | Collection-level and metadata filter isolation | Row-Level Security (RLS) and schema namespaces |
| **Client Interface** | Native Pythonic client (`collection.query()`) | SQL dialect (`SELECT * ORDER BY embedding <-> query LIMIT 5`) |
""")
st.divider()

# Section 7: ChromaDB vs Other Vector Databases
st.header("7. ChromaDB vs Other Vector Databases (Milvus, Qdrant, Pinecone, FAISS)", divider="rainbow")
st.markdown("""
Understanding ChromaDB's placement in the broader vector ecosystem:
""")

st.markdown("""
| Vector Store | Deployment Model | Best Suited Scale | Primary Strength |
| :--- | :--- | :--- | :--- |
| **ChromaDB** | Embedded / In-Process / Local Client-Server | Up to 1–5 million vectors | Developer ergonomics, rapid prototyping, local RAG |
| **FAISS** | In-memory C++ library | 100k to 10M vectors | Raw GPU-accelerated algorithmic indexing (no DB features) |
| **Qdrant** | Rust-native binary / Cloud / Cluster | 10M to billions | Single-stage payload filtering, low memory footprint |
| **Milvus** | Distributed Kubernetes cluster | 100M to billions | Cloud-native horizontal sharding, multi-GPU scaling |
| **Pinecone** | Fully managed proprietary cloud SaaS | Billions of vectors | Zero ops management, enterprise SLA guarantees |
""")
st.divider()

# Section 8: Advantages of ChromaDB
st.header("8. Advantages of ChromaDB: The 7 Core Developer Pillars", divider="rainbow")
st.markdown("""
ChromaDB's widespread adoption across the AI community is anchored on seven foundational pillars:
""")

col_adv1, col_adv2 = st.columns(2)
with col_adv1:
    st.markdown("""
    1. **Easy to Use**: Up and running in less than 60 seconds with minimal boilerplate code.
    2. **Python-Friendly**: Built natively for the Python data science and machine learning ecosystem.
    3. **Local Development**: Completely self-contained execution on developer laptops with zero external dependencies.
    4. **Persistent Storage**: Effortless transition from in-memory prototyping to disk-backed storage via `PersistentClient(path="./chroma_db")`.
    """)
with col_adv2:
    st.markdown("""
    5. **Semantic Search**: Powered by `hnswlib` for blazing-fast approximate nearest neighbor retrieval.
    6. **Metadata Filtering**: Expressive filter expressions (`where={"topic": "rag"}`) evaluated during search.
    7. **Embedding Support**: Pluggable embedding functions supporting OpenAI, Cohere, Hugging Face, Google PaLM, and local ONNX models.
    """)
st.divider()

# Section 9: Real-World Applications
st.header("9. Real-World Applications of ChromaDB", divider="rainbow")
st.markdown("""
ChromaDB serves as the embedded vector engine across diverse enterprise and open-source applications:
* **Retrieval-Augmented Generation (RAG)**: Storing company documentation, wikis, and manuals to supply authoritative context to LLMs.
* **Semantic Code Search**: Indexing function docstrings and code syntax to assist software engineers in exploring large repositories.
* **Customer Support Copilots**: Retrieving relevant historical troubleshooting tickets and resolution guides during active support chats.
* **Personal Desktop Assistants**: Powering local, private on-device knowledge bases (e.g., searching personal notes and PDF archives without sending data to cloud APIs).
* **Recommendation Engines**: Finding analogous items based on user interaction vectors and content embeddings.
""")
st.divider()

# Section 10: Production Implementation
st.header("10. Production Implementation: Full-Lifecycle ChromaDB Knowledge Base Pipeline", divider="rainbow")
st.markdown("""
Below is a complete, production-grade Python script demonstrating how to initialize a persistent ChromaDB client, 
configure custom collection metadata, ingest documents with rich attributes, and execute filtered semantic queries.
""")

st.code('''
import chromadb
from typing import List, Dict, Any

class ChromaProductionManager:
    """Manages collection lifecycle, document ingestion, and filtered querying in ChromaDB."""
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize persistent on-disk client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or retrieve collection configured for Cosine distance
        self.collection = self.client.get_or_create_collection(
            name="enterprise_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Upserts documents, allowing automatic embedding generation via default ONNX function."""
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def semantic_query(self, query_text: str, n_results: int = 3, filter_dict: Dict[str, Any] = None):
        """Executes filtered semantic search and prints results with computed distances."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_dict
        )
        return results

# Production Usage Example:
# manager = ChromaProductionManager()
# manager.ingest_documents(
#     documents=["PostgreSQL connection pool exhausted", "Kubernetes pod evicted due to OOM"],
#     metadatas=[{"category": "database"}, {"category": "devops"}],
#     ids=["rec_01", "rec_02"]
# )
# hits = manager.semantic_query("Why did my container crash?", n_results=1, filter_dict={"category": "devops"})
''', language="python")
st.divider()

# Section 11: Interactive Workbench
st.header("11. Interactive Workbench: ChromaDB Semantic Search & Distance Metric Simulator", divider="rainbow")
st.markdown("""
Test ChromaDB query execution in an interactive simulation. Adjust your query text, metadata filters, 
and inspect how Cosine Distance values rank relevant documents.
""")

chroma_sim = MockChromaCollection("knowledge_base")

col_work1, col_work2 = st.columns([1.2, 0.8])

with col_work1:
    user_query_input = st.text_input("Enter natural language query:", value="How does ChromaDB store its data?")
    
    use_filter = st.checkbox("Apply Metadata Filter (`where={'tier': 'core'}`)", value=False)
    filter_arg = {"tier": "core"} if use_filter else None
    
    # Query vector simulation
    if "data" in user_query_input.lower() or "store" in user_query_input.lower() or "sqlite" in user_query_input.lower():
        sim_vec = np.array([0.05, 0.20, 0.90, 0.10])
    elif "graph" in user_query_input.lower() or "hnsw" in user_query_input.lower() or "speed" in user_query_input.lower():
        sim_vec = np.array([0.15, 0.88, 0.20, 0.05])
    elif "rag" in user_query_input.lower() or "llm" in user_query_input.lower() or "prompt" in user_query_input.lower():
        sim_vec = np.array([0.10, 0.10, 0.15, 0.92])
    else:
        sim_vec = np.array([0.90, 0.15, 0.08, 0.10])

    hits = chroma_sim.query(sim_vec, n_results=2, where=filter_arg)
    
    st.subheader("Query Results (Ranked by Ascending Distance)")
    if hits:
        df_hits = pd.DataFrame(hits)
        st.dataframe(df_hits[["id", "distance", "document", "metadata"]], use_container_width=True)
    else:
        st.warning("No records matched the query and metadata filter criteria.")

with col_work2:
    st.subheader("Distance Metric Dynamics")
    st.info("""
    **Cosine Distance**:
    `Distance = 1.0 - Cosine_Similarity`
    * `0.0`: Identical vector orientation
    * `< 0.3`: High semantic relevance
    * `> 0.8`: Distant conceptual divergence
    """)
    st.caption("Notice that in ChromaDB distance metrics, **smaller values indicate greater similarity**!")
st.divider()

# Section 12: Technical Interview Preparation & Production Traps
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is ChromaDB and why is it preferred for rapid RAG prototyping?", expanded=True):
    st.markdown("""
    **Answer:**
    ChromaDB is an open-source, AI-native vector database designed for developer simplicity and lightweight embedding storage.
    
    * **Embedded-First Design**: Unlike Milvus or Pinecone that require separate server clusters, ChromaDB can run entirely in-process (`chromadb.Client()` or `chromadb.PersistentClient()`), saving local indices to disk with zero Docker or network configuration.
    * **Built-in Embedding Engine**: It automatically bundles a local ONNX-based embedding function (`all-MiniLM-L6-v2`), allowing developers to pass raw text strings directly without manually generating vector embeddings.
    * **Full Metadata Filtering**: Supports expressive SQL-like metadata filtering operators (`$eq`, `$in`, `$and`, `$or`) combined with semantic proximity search.
    """)

with st.expander("Q2: How does ChromaDB handle data persistence across restarts?"):
    st.markdown("""
    **Answer:**
    * When initialized with `chromadb.Client()`, Chroma operates in an **ephemeral in-memory mode**; all collections and vectors are lost when the process terminates.
    * When initialized with `chromadb.PersistentClient(path="./chroma_db")`, Chroma persists metadata, document IDs, and configuration into a local **SQLite database (`chroma.sqlite3`)** and writes HNSW index graph files and segment records to local disk directories. Upon re-initialization, the client reconnects to the existing database seamlessly.
    """)

with st.expander("Q3: What are the three client modes available in ChromaDB, and when should each be used?"):
    st.markdown("""
    **Answer:**
    1. **Ephemeral Client (`chromadb.Client()`)**: In-memory execution for unit tests, CI/CD pipelines, and transient mock workloads.
    2. **Persistent Client (`chromadb.PersistentClient()`)**: Local single-process persistence for desktop apps, data science notebooks, and local developer environments.
    3. **HTTP Client (`chromadb.HttpClient(host, port)`)**: Connects over REST / gRPC to a centralized ChromaDB server container, enabling multi-process web applications (e.g., FastAPI, Streamlit) to share a single concurrent database without SQLite file locking.
    """)

with st.expander("Q4: What metadata operators does ChromaDB support for filtering queries?"):
    st.markdown("""
    **Answer:**
    ChromaDB supports two primary filtering parameters:
    * `where`: Evaluates metadata dictionaries using operators like `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, and logical combinators `$and` and `$or`.
    * `where_document`: Evaluates raw document text payloads using `$contains` and `$not_contains` substring search.
    """)

with st.expander("Q5: What distance metrics are supported in ChromaDB, and what is the default?"):
    st.markdown("""
    **Answer:**
    ChromaDB supports three distance metrics configurable via collection metadata (`metadata={"hnsw:space": ...}`):
    1. `l2`: Squared Euclidean L2 distance (`Sum (u_i - v_i)^2`).
    2. `cosine`: Cosine distance (`1 - Cosine Similarity`), bounded between 0.0 (identical) and 2.0 (opposite). **This is the default metric in ChromaDB**.
    3. `ip`: Inner product distance (`1 - (u . v)` for normalized vectors).
    """)

st.subheader("Production Failure Traps")
trap_col1, trap_col2 = st.columns(2)
with trap_col1:
    st.error("""
    **Trap 1: SQLite Database Locked in Multi-Worker Environments (Fatal)**
    Instantiating `chromadb.PersistentClient()` inside multiple Gunicorn or Celery worker processes accessing the same directory. SQLite does not support concurrent write locks across multiple OS processes, resulting in `sqlite3.OperationalError: database is locked` crashes. Use `chromadb.HttpClient()` against a central Chroma container for multi-worker architectures.
    """)
    st.error("""
    **Trap 2: Collection Distance Space Mismatch (Silent Accuracy Degrade)**
    Defaulting to `cosine` distance when your embedding model was trained for Euclidean distance, or failing to normalize embeddings when using Inner Product (`ip`). This produces scrambled search rankings without throwing runtime errors.
    """)
with trap_col2:
    st.warning("""
    **Trap 3: Mismatched Embedding Dimensions on Collection Upsert (Runtime Failure)**
    Attempting to add 1536-dimensional embeddings (e.g., from OpenAI) into a collection created with the default 384-dimensional ONNX embedding function. ChromaDB raises an `InvalidDimensionException` and aborts the batch.
    """)
    st.warning("""
    **Trap 4: Ephemeral Client Data Loss on Process Termination (Production Outage)**
    Calling `chromadb.Client()` instead of `chromadb.PersistentClient(path=...)` in production scripts, causing all indexed customer documents to evaporate every time the web server restarts.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the default distance metric in ChromaDB?")
        st.markdown("**Back**: Cosine distance (`hnsw:space: cosine`), where 0.0 indicates identical orientation and 2.0 indicates opposite orientation.")
    with st.container(border=True):
        st.markdown("**Front**: Why does `PersistentClient` fail with multiple web workers?")
        st.markdown("**Back**: Multiple processes trying to write concurrently trigger SQLite database lock contention. Use `HttpClient` for multi-worker apps.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What default embedding model does ChromaDB bundle?")
        st.markdown("**Back**: The `all-MiniLM-L6-v2` Sentence Transformer running locally via ONNX Runtime (384-dimensional dense vectors).")
    with st.container(border=True):
        st.markdown("**Front**: How do you filter documents containing a specific word in ChromaDB?")
        st.markdown("**Back**: Using `where_document={\"$contains\": \"search_term\"}` inside `collection.query()` or `collection.get()`.")

st.divider()

# Section 13: Self-Assessment Checklist
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of ChromaDB Fundamentals and Architecture before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand what ChromaDB is and why it excels at local and embedded RAG development.", value=True, key="chk_chroma_1")
    chk2 = st.checkbox("I know the difference between EphemeralClient, PersistentClient, and HttpClient.", value=True, key="chk_chroma_2")
    chk3 = st.checkbox("I understand how ChromaDB stores vectors, IDs, and metadata using SQLite and HNSW segments.", value=True, key="chk_chroma_3")
with c2:
    chk4 = st.checkbox("I can construct metadata filter queries using `$eq`, `$in`, `$and`, and `$or` operators.", value=True, key="chk_chroma_4")
    chk5 = st.checkbox("I know the 3 supported distance spaces (`cosine`, `l2`, `ip`) and how to configure them in collection metadata.", value=True, key="chk_chroma_5")
    chk6 = st.checkbox("I can diagnose production traps including SQLite concurrency lockouts, dimension mismatches, and data loss.", value=True, key="chk_chroma_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **ChromaDB Fundamentals Mastery Demonstrated!** You are ready to advance to Lesson 1.2 in the curriculum.")
