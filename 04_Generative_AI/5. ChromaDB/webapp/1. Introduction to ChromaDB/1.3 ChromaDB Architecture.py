import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.3 ChromaDB Architecture - ChromaDB Curriculum",
    page_icon="🎨",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulation
class ChromaEntityInspector:
    """Simulates internal ChromaDB entity structures and query lifecycle execution."""
    def __init__(self):
        self.entities = {
            "Client": {"role": "Primary API gateway and connection broker.", "backends": ["Ephemeral (In-Memory)", "Persistent (Disk)", "HTTP (Server)"]},
            "Collection": {"role": "Isolated namespace with configurable distance space.", "distance_spaces": ["cosine (default)", "l2", "ip"]},
            "Documents": {"role": "Raw unstructured string payloads.", "schema": "List[str] processed by embedding function"},
            "Embeddings": {"role": "Dense float arrays produced by model.", "default_model": "all-MiniLM-L6-v2 (384-d)"},
            "Metadata": {"role": "Key-value dictionaries indexed for filtering.", "operators": ["$eq", "$ne", "$gt", "$in", "$and", "$or"]},
            "IDs": {"role": "Unique alphanumeric record primary keys.", "constraints": "Non-empty string per document"},
            "Vectors": {"role": "Raw coordinate arrays stored in HNSW segments.", "format": "Normalized numpy float32 arrays"},
            "Query Flow": {"role": "Query text -> Embedding -> HNSW search -> Filter -> Payload return.", "latency": "Sub-5ms"}
        }

    def inspect(self, entity_name: str) -> Dict[str, Any]:
        return self.entities.get(entity_name, {})

# Title & Overview
st.title("1.3 ChromaDB Architecture: Core Primitives, Storage & Execution Lifecycle")
st.caption("Chapter 1: Introduction to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
To design resilient GenAI systems and diagnose production edge cases, developers must look beneath 
ChromaDB's simple client interface and understand its **internal architectural primitives**:
how clients orchestrate collections, how SQLite manages relational metadata state, how hnswlib navigates 
vector graph segments, and how queries execute through the complete retrieval lifecycle.

ChromaDB is architected around a clean separation of concerns: **Relational Metadata Management** is handled 
by embedded SQLite, **Vector Proximity Search** is executed via high-speed C++ HNSW graphs, and **Payload Storage** 
resides in immutable segment blocks.

In this lesson, you will master:
1. **The 8 Core Architectural Primitives**: Client, Collection, Documents, Embeddings, Metadata, IDs, Vectors, and Query Flow.
2. **Client Modes Dissected**: When to use EphemeralClient, PersistentClient, and HttpClient.
3. **Internal Storage Topology**: How `chroma.sqlite3` and segment files coordinate on disk.
4. **End-to-End Query Lifecycle**: Step-by-step tracing from user string input to top-K grounded output.
5. **Production Failure Traps**: Avoiding SQLite concurrency lockouts, dimension mismatches, and payload corruption.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: ChromaDB Storage & Execution Layers", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChromaInternalArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_client_layer {
            label="Layer 1: Client Gateway & API Coordination";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            client_entry [label="Chroma Client Interface\\nEphemeralClient / PersistentClient / HttpClient", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            coll_router [label="Collection Router & Cache\\nValidates collection name, UUID, and distance metric", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            ef_dispatcher [label="Embedding Function Dispatcher\\nExecutes ONNX local or remote OpenAI API embedder", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            client_entry -> coll_router -> ef_dispatcher;
        }

        subgraph cluster_storage_layer {
            label="Layer 2: Dual-Engine Persistence & Storage Subsystems";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            sqlite_engine [label="Relational Engine (chroma.sqlite3)\\nCollections table, Document IDs, Metadata KV, WAL", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            segment_mgr [label="Segment Manager\\nCoordinates index segments and immutable Parquet blocks", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            hnsw_engine [label="Vector Index Engine (hnswlib)\\nMulti-layer proximity graph files (index_metadata.pickle)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            ef_dispatcher -> sqlite_engine [label="Sync IDs & Metadata"];
            ef_dispatcher -> segment_mgr [label="Ingest Vector Batches"];
            segment_mgr -> hnsw_engine [label="Write Graph Segments"];
            sqlite_engine -> segment_mgr [label="Manage Segment State"];
        }

        subgraph cluster_query_layer {
            label="Layer 3: Query Execution, Filtering & Hydration";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            query_ingress [label="Incoming Search Query\\ncollection.query(query_texts=[...], where={...})", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            where_filter [label="Metadata Filter Evaluator\\nQueries SQLite index for valid candidate record IDs", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            ann_search [label="HNSW Graph Traversal\\nEvaluates Cosine distance across allowed node graph", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            hydrator [label="Result Hydrator & Formatter\\nMerges IDs + text chunks + metadata + distances into response dict", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            query_ingress -> where_filter;
            query_ingress -> ann_search;
            sqlite_engine -> where_filter;
            hnsw_engine -> ann_search;
            where_filter -> hydrator;
            ann_search -> hydrator;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Client
st.header("2. Client (Primitive 1): Ephemeral, Persistent, and HTTP Topologies", divider="rainbow")
st.markdown("""
The **Client** is the root object in ChromaDB, acting as the connection broker, configuration container, and collection factory:

* **Ephemeral Client (`chromadb.Client()`)**:
  * Runs completely in volatile memory.
  * Indices, collections, and documents are discarded when the Python process exits.
  * Ideal for automated test suites, CI/CD pipelines, and transient demonstrations.
* **Persistent Client (`chromadb.PersistentClient(path="./chroma_db")`)**:
  * Saves metadata and indices to a designated directory on the local filesystem.
  * Automatically loads pre-existing collections upon re-instantiation.
  * Ideal for desktop tools, developer prototyping, and single-process applications.
* **HTTP Client (`chromadb.HttpClient(host="localhost", port=8000)`)**:
  * Communicates via REST / gRPC with a standalone ChromaDB server container.
  * Solves SQLite multi-process locking by funneling all client requests through a centralized server process.
  * Mandatory for multi-worker web production deployments (FastAPI, Gunicorn, Celery).
""")
st.divider()

# Section 3: Collection
st.header("3. Collection (Primitive 2): Logical Namespaces and Configuration", divider="rainbow")
st.markdown("""
A **Collection** is a logical container for documents, embeddings, and metadata—analogous to a table in SQL or a collection in MongoDB:

* **Collection Naming Constraints**: Names must be between 3 and 63 characters, contain only alphanumeric characters, underscores, or hyphens, start and end with an alphanumeric character, and contain no consecutive periods.
* **Distance Space Configuration**: Configured during collection creation via the `metadata={"hnsw:space": ...}` attribute:
  * `"cosine"`: Cosine distance (`1 - Cosine Similarity`). **Default setting**.
  * `"l2"`: Squared Euclidean distance (`Sum (u_i - v_i)^2`).
  * `"ip"`: Inner product distance (`1 - (u . v)` for normalized vectors).
* **Isolation Guarantee**: Queries are scoped strictly to a single collection; records in Collection A are never evaluated during queries against Collection B.
""")
st.divider()

# Section 4: Documents
st.header("4. Documents (Primitive 3): Unstructured Raw Text Payloads", divider="rainbow")
st.markdown("""
**Documents** represent the raw unstructured text strings stored alongside vector indices:
* **Automatic Embedding Trigger**: When documents are passed to `collection.add(documents=[...])` without specifying `embeddings`, ChromaDB automatically invokes the collection's embedding function to generate vectors.
* **Sub-String Content Filtering**: Documents can be searched using keyword substring matching via the `where_document={"$contains": "keyword"}` query parameter.
* **Hydration on Query**: During query execution, ChromaDB retrieves and returns the full original document string, eliminating the need to query an external secondary database.
""")
st.divider()

# Section 5: Embeddings
st.header("5. Embeddings (Primitive 4): Vector Generation and Embedding Functions", divider="rainbow")
st.markdown("""
The **Embeddings Primitive** represents the dense numerical vectors that power similarity search:
* **Default Embedding Function**: Uses `all-MiniLM-L6-v2` executed locally via ONNX Runtime, generating 384-dimensional dense vectors with zero API token cost.
* **Pluggable Architecture**: ChromaDB supports custom embedding functions:
  ```python
  from chromadb.utils import embedding_functions

  openai_ef = embedding_functions.OpenAIEmbeddingFunction(
      api_key="sk-...",
      model_name="text-embedding-3-small"
  )
  collection = client.create_collection("openai_docs", embedding_function=openai_ef)
  ```
* **Pre-Computed Vectors**: If your ingestion pipeline generates vectors upstream, you can pass them directly via `collection.add(embeddings=[...], ids=[...])`.
""")
st.divider()

# Section 6: Metadata
st.header("6. Metadata (Primitive 5): JSON Payloads and Inverted Filter Indices", divider="rainbow")
st.markdown("""
**Metadata** consists of structured key-value dictionaries attached to each individual record:
* **Supported Value Types**: Strings, integers, floats, and booleans.
* **Filter Expressions**: Evaluated via the `where` parameter using standard operators:
  * Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
  * Inclusion: `$in`, `$nin`
  * Logical Combinators: `$and`, `$or`
* **Internal Indexing**: ChromaDB indexes metadata keys in SQLite to rapidly resolve candidate ID lists before or during vector distance evaluation.
""")
st.divider()

# Section 7: IDs
st.header("7. IDs (Primitive 6): Unique Deterministic and UUID Keys", divider="rainbow")
st.markdown("""
Every document stored in ChromaDB must have a unique **ID**:
* **Requirement**: A non-empty string unique across the collection.
* **Deterministic Hashing vs UUIDs**:
  * **UUIDs (`uuid.uuid4()`)**: Ensure uniqueness across distributed ingestion workers.
  * **Deterministic Hashes (`hashlib.sha256(doc.encode()).hexdigest()`)**: Enable automatic deduplication and safe idempotent upserts.
* **Direct Key Lookups**: Records can be retrieved directly by ID in sub-millisecond time via `collection.get(ids=["doc_123"])`.
""")
st.divider()

# Section 8: Vectors
st.header("8. Vectors (Primitive 7): Dense Floating-Point Coordinate Arrays", divider="rainbow")
st.markdown("""
**Vectors** are the raw coordinate arrays managed within ChromaDB's HNSW graph segment files:
* **Format**: Arrays of 32-bit floating-point numbers (`np.float32`).
* **Dimensional Consistency**: Every vector added to a collection must have the exact same dimensionality as the collection's configured embedding function. Adding a 1536-d vector to a 384-d collection raises an immediate `InvalidDimensionException`.
* **C++ Memory Layout**: Vectors are held in contiguous memory blocks within `hnswlib` for high-throughput SIMD distance calculations.
""")
st.divider()

# Section 9: Query and Retrieval Flow
st.header("9. Query and Retrieval Flow (Primitive 8): End-to-End Execution Trace", divider="rainbow")
st.markdown("""
Tracing what happens when an application calls `collection.query(query_texts=["..."], where={...}, n_results=5)`:
""")

st.markdown("""
1. **Query Embedding**: The embedding function converts the query string into a query vector `q`.
2. **Metadata Filtering**: If a `where` clause is present, the SQLite engine evaluates metadata indices to identify candidate document IDs.
3. **Graph Traversal**: The HNSW engine begins at top-layer entry points and navigates down to Layer 0, computing distances between `q` and graph node vectors.
4. **Candidate Extraction**: A bounded min-heap maintains the top-5 candidates with the lowest distance values.
5. **Payload Hydration**: SQLite fetches document text strings and metadata dictionaries for the 5 winning IDs.
6. **Result Assembly**: The client returns a structured dictionary containing `ids`, `distances`, `metadatas`, and `documents`.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems with multiple concurrent web workers, ChromaDB is deployed as a standalone container 
(`chromadb/chroma`) orchestrated via Docker Compose or Kubernetes. Application processes instantiate `chromadb.HttpClient()` 
over an internal VPC network, preventing SQLite file lock contention and allowing background ingestion workers to update collections 
without disrupting query latency.
""")
st.divider()

# Section 10: Production Implementation
st.header("10. Production Implementation: Full Lifecycle Collection Manager", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating complete collection lifecycle management, 
deterministic ID generation, idempotent upserts, and complex metadata queries.
""")

st.code('''
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

class ChromaLifecycleManager:
    """Manages collection lifecycle, idempotent upserts, and metadata filtering."""
    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="system_runbooks",
            metadata={"hnsw:space": "cosine"}
        )

    def generate_deterministic_id(self, text: str) -> str:
        """Generates a reproducible SHA-256 ID to prevent duplicate document insertions."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add_runbook(self, title: str, text: str, department: str, severity: int):
        """Idempotently adds a runbook document with structured metadata."""
        doc_id = self.generate_deterministic_id(text)
        metadata = {
            "title": title,
            "department": department,
            "severity": severity
        }
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def search_urgent_runbooks(self, query_text: str, department: str, min_severity: int = 3):
        """Queries the collection using combined semantic and metadata filter criteria."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=3,
            where={
                "$and": [
                    {"department": {"$eq": department}},
                    {"severity": {"$gte": min_severity}}
                ]
            }
        )
''', language="python")
st.divider()

# Section 11: Interactive Workbench
st.header("11. Interactive Workbench: Architecture Entity Inspector & Execution Simulator", divider="rainbow")
st.markdown("""
Inspect ChromaDB's internal architectural primitives and explore their configuration rules.
""")

inspector = ChromaEntityInspector()
entity_names = list(inspector.entities.keys())

selected_entity = st.selectbox("Select Architectural Primitive to Inspect:", entity_names)
details = inspector.inspect(selected_entity)

col_ins1, col_ins2 = st.columns([1, 1])
with col_ins1:
    st.subheader(f"Primitive: {selected_entity}")
    st.info(f"**Primary System Role**:\n\n{details.get('role', 'N/A')}")

with col_ins2:
    st.subheader("Architectural Configuration")
    for k, v in details.items():
        if k != "role":
            st.markdown(f"* **{k.capitalize()}**: `{v}`")
st.divider()

# Section 12: Technical Interview Preparation & Production Traps
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: How does ChromaDB separate relational metadata storage from vector graph indexing?", expanded=True):
    st.markdown("""
    **Answer:**
    ChromaDB employs a dual-engine architecture:
    * **Relational Engine**: An embedded SQLite database (`chroma.sqlite3`) manages collection records, document IDs, metadata dictionaries, and operational write-ahead logs.
    * **Vector Index Engine**: The C++ `hnswlib` library maintains vector coordinates and Hierarchical Navigable Small World proximity graphs in dedicated segment files.
    This separation allows vector proximity traversals to execute in contiguous memory without being slowed down by heavy text payloads, while preserving expressive metadata filtering.
    """)

with st.expander("Q2: Why must production web applications avoid using `PersistentClient` across multiple processes?"):
    st.markdown("""
    **Answer:**
    `chromadb.PersistentClient()` opens a direct file handle to the local `chroma.sqlite3` database. SQLite relies on filesystem locks for write serialization. In multi-worker environments (e.g., Gunicorn or Uvicorn with 4 workers), concurrent writes trigger `sqlite3.OperationalError: database is locked` exceptions. Multi-worker production applications must deploy a centralized ChromaDB server container and connect via `chromadb.HttpClient()`.
    """)

with st.expander("Q3: What happens when you pass vectors with differing dimensions into the same collection?"):
    st.markdown("""
    **Answer:**
    ChromaDB collections enforce strict dimensional homogeneity. When a collection is created, its dimensionality is locked to the first vector inserted (or defined by its embedding function). If an application attempts to insert a vector of dimensionality `D2` into a collection initialized for dimensionality `D1`, ChromaDB raises an `InvalidDimensionException` and rejects the batch.
    """)

with st.expander("Q4: How do the three distance spaces (`cosine`, `l2`, `ip`) differ in ChromaDB?"):
    st.markdown("""
    **Answer:**
    * `cosine`: Measures `1 - Cosine Similarity`. Normalized to directional angle, bounded between 0.0 and 2.0. This is the default metric in ChromaDB.
    * `l2`: Measures Squared Euclidean Distance `Sum (u_i - v_i)^2`. Useful for computer vision feature embeddings where magnitude carries information.
    * `ip`: Measures Inner Product Distance `1 - (u . v)` for normalized vectors. When vectors are L2-normalized, `ip` is mathematically equivalent to `cosine` and evaluates faster via SIMD instructions.
    """)

with st.expander("Q5: What is the purpose of idempotent upserts (`collection.upsert`) in ChromaDB pipelines?"):
    st.markdown("""
    **Answer:**
    Calling `collection.add()` with an ID that already exists in the collection throws an `IDAlreadyExistsException` and aborts ingestion. In contrast, `collection.upsert()` checks whether the ID exists: if it exists, it updates the document, vector, and metadata; if it does not exist, it inserts a new record. This ensures that re-running batch ingestion jobs does not fail or create duplicate records.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Multi-Process SQLite Lockout in Web Workers (Fatal 500 Outage)**
    Instantiating `PersistentClient` in multi-worker Gunicorn or Celery applications. Multiple processes competing for write locks on `chroma.sqlite3` crash with `OperationalError: database is locked`. Always use `HttpClient` with a central Chroma server container.
    """)
    st.error("""
    **Trap 2: Mismatched Embedding Dimensions Across Ingestion Batches (Runtime Abort)**
    Adding 1536-d OpenAI vectors to a collection that defaulted to the 384-d local ONNX embedding function. ChromaDB throws an `InvalidDimensionException`, causing downstream pipeline failure.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: EphemeralClient Data Loss During Server Reboots (Data Loss)**
    Using `chromadb.Client()` in production code instead of `PersistentClient(path=...)` or `HttpClient()`. All indexed customer documents evaporate upon process termination.
    """)
    st.warning("""
    **Trap 4: Non-Deterministic ID Collisions Overwriting Records (Data Corruption)**
    Generating sequential integer IDs or non-unique keys across parallel ingestion workers. Documents inadvertently overwrite each other in the collection without warning.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does ChromaDB store metadata vs vector indices?")
        st.markdown("**Back**: Metadata is stored in an embedded SQLite database (`chroma.sqlite3`), while vectors are stored in HNSW segment files via `hnswlib`.")
    with st.container(border=True):
        st.markdown("**Front**: When should you use `chromadb.HttpClient()`?")
        st.markdown("**Back**: In multi-process production web applications (FastAPI/Gunicorn) to avoid SQLite database file lock contention.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is the default distance space in ChromaDB?")
        st.markdown("**Back**: Cosine distance (`hnsw:space: cosine`), where smaller values indicate higher semantic similarity.")
    with st.container(border=True):
        st.markdown("**Front**: Why use `collection.upsert()` instead of `collection.add()`?")
        st.markdown("**Back**: Upsert is idempotent; it updates existing IDs without throwing `IDAlreadyExistsException` crashes during repeated ingestion runs.")

st.divider()

# Section 13: Self-Assessment Checklist
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of ChromaDB Internal Architecture and Primitives before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the 8 core architectural primitives: Client, Collection, Documents, Embeddings, Metadata, IDs, Vectors, and Query Flow.", value=True, key="chk_arch_1")
    chk2 = st.checkbox("I know when to choose between EphemeralClient, PersistentClient, and HttpClient.", value=True, key="chk_arch_2")
    chk3 = st.checkbox("I understand the dual-engine separation between SQLite metadata and HNSW vector segment files.", value=True, key="chk_arch_3")
with c2:
    chk4 = st.checkbox("I can construct idempotent ingestion pipelines using deterministic IDs and collection upserts.", value=True, key="chk_arch_4")
    chk5 = st.checkbox("I understand how query execution coordinates metadata filters and HNSW graph search.", value=True, key="chk_arch_5")
    chk6 = st.checkbox("I can diagnose production traps including SQLite concurrency lockouts, dimension mismatches, and data loss.", value=True, key="chk_arch_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **ChromaDB Architecture Mastery Demonstrated!** You have completed Chapter 1 of the ChromaDB curriculum.")
