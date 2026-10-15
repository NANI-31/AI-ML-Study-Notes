import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.1 Introduction to ChromaDB - Vector DB Curriculum",
    page_icon="🎨",
    layout="wide"
)

# Title & Overview
st.title("6.1 Introduction to ChromaDB: Architecture, Embedded Design & Role in RAG")
st.caption("Chapter 6: ChromaDB Fundamentals | Domain: 4. Vector Database")
st.markdown("""
Among the modern ecosystem of vector databases, **ChromaDB** stands out as the most widely adopted **AI-native, developer-first 
vector database**. 

While enterprise engines like Milvus and Qdrant are engineered for distributed multi-node Kubernetes clusters 
serving billions of vectors, ChromaDB was designed from the ground up for **extreme developer velocity**: 
it runs natively inside your Python process (`pip install chromadb`), requires **zero external infrastructure**, 
includes built-in automatic embedding functions, and scales seamlessly from an ephemeral in-memory prototype 
to persistent on-disk SQLite storage and client-server deployments.

In this lesson, we explore what makes ChromaDB unique, analyze its **dual-storage architecture (SQLite + HNSWlib)**, 
examine its core enterprise use cases, and trace its central role in production **Retrieval-Augmented Generation (RAG)** systems.
""")

st.divider()

# Section 1: What is ChromaDB & Why is it Popular?
st.header("1. What is ChromaDB & Why Developers Choose It", divider="rainbow")

col_what1, col_what2 = col_col1, col_col2 = st.columns([1, 1])

with col_what1:
    st.markdown("""
    ### 1. The ChromaDB Value Proposition
    * **Embedded "Local-First" Design**:
      * Runs directly inside your Python interpreter or Jupyter notebook with zero Docker containers, zero external server processes, and zero cloud API keys required to start.
    * **Integrated Automated Embeddings**:
      * Built-in embedding function abstraction. You can add raw text strings (`collection.add(documents=[...])`), and Chroma automatically vectorizes them using local `all-MiniLM-L6-v2` or your chosen OpenAI/HuggingFace model.
    * **Full Metadata Filtering**:
      * Supports rich MongoDB-style scalar filtering operators (`$eq`, `$ne`, `$gt`, `$in`, `$and`, `$or`) combined directly with vector similarity search.
    * **Progressive Scalability**:
      * Prototype in-memory (`Client()`), persist to disk (`PersistentClient(path="./data")`), or connect to a remote microservice (`HttpClient(host="...")`).
    """)

with col_what2:
    st.markdown("""
    ### 2. When to Choose ChromaDB
    * **Local & Enterprise RAG Applications**:
      * Powering internal company search bots, document Q&A tools, and desktop AI assistants.
    * **Agentic Memory Systems**:
      * Storing conversational history and episodic memories for LangChain, AutoGen, or CrewAI agents.
    * **Rapid Prototyping & Unit Testing**:
      * Testing retrieval pipelines in CI/CD without provisioning expensive cloud database clusters.
    * **Small-to-Medium Vector Workloads**:
      * Highly optimized for collections from **10,000 to 5,000,000 vectors**.
    """)

st.markdown("### Vector Database Architectural Positioning")
position_df = pd.DataFrame({
    "Vector Database": ["ChromaDB", "Qdrant", "Milvus", "Pinecone", "pgvector (PostgreSQL)"],
    "Deployment Model": ["Embedded Python / Local Server", "Self-Hosted Cluster / Cloud", "Distributed Kubernetes / Cloud", "Managed Serverless Cloud", "Relational Database Extension"],
    "Operational Complexity": ["Zero (pip install chromadb)", "Moderate (Docker / Rust binary)", "High (Kube / etcd / MinIO / Pulsar)", "Zero (Managed SaaS API)", "Low (Single PostgreSQL node)"],
    "Ideal Scale": ["10k to 5M vectors", "1M to 100M+ vectors", "100M to Billions of vectors", "1M to Billions of vectors", "100k to 10M vectors"],
    "Primary Superpower": ["Developer speed & zero infra", "Rust speed, advanced filtering", "Massive distributed scale", "Serverless simplicity", "Unified SQL relational joins"]
})
st.dataframe(position_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: ChromaDB Dual-Storage Architecture
st.header("2. ChromaDB Internal Architecture: Dual-Storage Engine", divider="rainbow")

st.markdown("""
How does ChromaDB achieve both relational metadata filtering and microsecond vector retrieval without external dependencies? 
By combining **SQLite** and **HNSWlib**:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    ### The Dual-Engine Subsystems
    * **1. SQLite (Metadata & Document Catalog)**:
      * Stores collection definitions, record UUIDs, raw document text strings, and structured JSON metadata payloads.
      * Provides ACID transactions and crash durability on local disk.
    * **2. HNSWlib (C++ In-Memory Vector Index)**:
      * A high-performance C++ implementation of the Hierarchical Navigable Small World (HNSW) graph.
      * Compiles directly into Python via pybind11, providing hardware-accelerated SIMD vector distance calculations.
    * **3. Embedding Function Abstraction**:
      * Pluggable pipeline: converts incoming text queries and documents into dense float32 arrays before passing them to the HNSW index.
    * **4. Query Coordinator**:
      * Intersects SQLite metadata filters with HNSW nearest neighbor candidate lists, hydrating document payloads for the caller.
    """)

with col_arch2:
    st.markdown("### ChromaDB Internal Architecture Flowchart")
    chroma_dot = """
    digraph ChromaInternalArchitecture {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ClientAPI [label="Python Client API (chromadb)\\ncollection.query(query_texts=['...'])", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_chroma {
            label="ChromaDB Core Engine (In-Process / Server)"; style="dashed"; color="#10B981";
            Embedder [label="Embedding Function Layer\\n(Default MiniLM / OpenAI / BGE)", fillcolor="#D1FAE5"];
            QueryCoord [label="Query & Filter Coordinator\\nEvaluates where={'dept': 'HR'}", fillcolor="#A7F3D0"];

            subgraph cluster_dual_storage {
                label="Dual Storage Layer"; style="dotted"; color="#059669";
                SQLiteStore [label="SQLite Database (Disk/Memory)\\nCollections, UUIDs, Docs, Metadata", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
                HNSWIndex [label="C++ HNSWlib Vector Index\\nMulti-layer Proximity Graph (SIMD)", fillcolor="#34D399", color="#047857", style="filled,bold"];
            }
        }

        ResultOut [label="Hydrated Results Dictionary\\n{ids, distances, documents, metadatas}", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        ClientAPI -> Embedder;
        Embedder -> QueryCoord;
        QueryCoord -> SQLiteStore [label="Filter metadata & get IDs"];
        QueryCoord -> HNSWIndex [label="Traverse vector graph"];
        HNSWIndex -> QueryCoord;
        SQLiteStore -> QueryCoord [label="Hydrate text payloads"];
        QueryCoord -> ResultOut;
    }
    """
    st.graphviz_chart(chroma_dot, use_container_width=False)

st.divider()

# Section 3: ChromaDB in Retrieval-Augmented Generation (RAG)
st.header("3. ChromaDB's Central Role in Production RAG", divider="rainbow")

st.markdown("""
In modern Retrieval-Augmented Generation (RAG), ChromaDB acts as the **external long-term memory** for the LLM:
""")

rag_flow_dot = """
digraph ChromaRAGFlow {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    UserQ [label="1. User Question\\n'How to configure SSO?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
    ChromaQuery [label="2. ChromaDB Retrieval\\ncollection.query()\\nTop-K nearest chunks", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
    RetrievedDocs [label="3. Retrieved Chunks\\nChunk 1 (Score: 0.92)\\nChunk 2 (Score: 0.88)", fillcolor="#A7F3D0"];
    PromptAssembler [label="4. Prompt Context Injection\\nContext: {Chunks}\\nQuestion: {UserQ}", fillcolor="#FEF3C7"];
    LLM [label="5. Foundation LLM\\n(GPT-4o / Claude / Gemini)", fillcolor="#FDE68A", color="#D97706", style="filled,bold"];
    FinalAnswer [label="6. Factual Grounded Answer\\nZero Hallucinations!", fillcolor="#6EE7B7", color="#047857", style="filled,bold"];

    UserQ -> ChromaQuery -> RetrievedDocs -> PromptAssembler -> LLM -> FinalAnswer;
    UserQ -> PromptAssembler;
}
"""
st.graphviz_chart(rag_flow_dot, use_container_width=False)

st.divider()

# Section 4: Interactive ChromaDB RAG Pipeline Simulator
st.header("4. Interactive ChromaDB RAG Pipeline Simulator", divider="rainbow")

st.markdown("""
Simulate an end-to-end ChromaDB RAG retrieval pipeline across a curated knowledge base. 
Observe **automated text vectorization**, **top-k nearest neighbor ranking**, **distance calculation**, 
and **prompt augmentation**:
""")

# Mock ChromaDB Knowledge Base
knowledge_base_chroma = [
    {"id": "doc_sec_01", "text": "Single Sign-On (SSO) requires SAML 2.0 or OIDC protocol configuration with Okta or Azure AD.", "meta": {"dept": "Security", "tier": "Enterprise"}},
    {"id": "doc_sec_02", "text": "Session timeout policies force re-authentication after 15 minutes of inactivity.", "meta": {"dept": "Security", "tier": "All"}},
    {"id": "doc_bill_01", "text": "Annual billing plans receive a 20% discount compared to monthly recurring credit card billing.", "meta": {"dept": "Billing", "tier": "All"}},
    {"id": "doc_infra_01", "text": "Kubernetes pods scaling beyond 10 replicas trigger horizontal pod autoscaler (HPA) alerts.", "meta": {"dept": "Infra", "tier": "Internal"}},
    {"id": "doc_infra_02", "text": "PostgreSQL database connection pool sizing must not exceed max_connections in postgresql.conf.", "meta": {"dept": "Infra", "tier": "Internal"}}
]

col_sim_in, col_sim_out = st.columns([1, 1])

with col_sim_in:
    st.subheader("Query Configuration")
    user_q = st.selectbox(
        "User Search Query",
        options=[
            "How do I set up Single Sign-On with Okta?",
            "What happens when users leave their browser inactive?",
            "How much do I save with annual billing?",
            "What causes database connection errors?"
        ]
    )
    top_k_val = st.slider("Top-K Candidates to Hydrate", 1, 4, 2)
    dept_filter = st.selectbox("Metadata Filter (`where={'dept': ...}`)", ["No Filter", "Security", "Billing", "Infra"])

with col_sim_out:
    st.subheader("ChromaDB Simulated Retrieval Output")
    
    # Calculate simulated cosine scores
    scored_docs = []
    for d in knowledge_base_chroma:
        if dept_filter != "No Filter" and d["meta"]["dept"] != dept_filter:
            continue
            
        # Simulated semantic score
        if "Single Sign-On" in user_q and "SSO" in d["text"]:
            sim = 0.94
        elif "inactive" in user_q and "inactivity" in d["text"]:
            sim = 0.92
        elif "annual" in user_q and "Annual" in d["text"]:
            sim = 0.95
        elif "connection" in user_q and "connection pool" in d["text"]:
            sim = 0.91
        else:
            sim = 0.25 + (abs(hash(d["id"] + user_q)) % 30) / 100.0
            
        dist = 1.0 - sim
        scored_docs.append((d, sim, dist))
        
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_hits = scored_docs[:top_k_val]
    
    if top_hits:
        for idx, (doc, sim, dist) in enumerate(top_hits, 1):
            st.markdown(f"**Hit {idx}:** `ID: {doc['id']}` | **Cosine Distance:** `{dist:.4f}` | **Sim:** `{sim:.4f}`")
            st.markdown(f"*Document:* `{doc['text']}`")
            st.markdown(f"*Metadata:* `{doc['meta']}`")
            st.markdown("---")
            
        st.subheader("Assembled LLM Grounded Prompt")
        context_str = "\n".join([f"- {h[0]['text']}" for h in top_hits])
        prompt_preview = f"""System: Answer the question strictly using the provided context.

Context:
{context_str}

Question: {user_q}
Answer:"""
        st.code(prompt_preview, language="markdown")
    else:
        st.error("Zero records passed the metadata filter!")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Minimal Chroma-Style In-Memory Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Chroma-Style In-Memory Vector Store** demonstrating 
the exact API design: `create_collection`, `add(documents, metadatas, ids)`, and `query(query_texts, n_results, where)`:
""")

code_chroma_impl = '''import numpy as np
from typing import List, Dict, Any, Optional

class SimpleChromaCollection:
    """Mimics ChromaDB collection API with in-memory SQLite-style metadata and vector search."""
    def __init__(self, name: str, dimension: int = 16):
        self.name = name
        self.dimension = dimension
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def _embed(self, text: str) -> np.ndarray:
        """Simulates automated embedding function."""
        np.random.seed(abs(hash(text)) % (2**32))
        v = np.random.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / (norm if norm > 0 else 1.0)

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Adds raw text documents with automated client-side embedding."""
        assert len(documents) == len(ids) == len(metadatas)
        new_vecs = [self._embed(doc) for doc in documents]
        
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        if len(self.vectors) == 0:
            self.vectors = np.array(new_vecs, dtype=np.float32)
        else:
            self.vectors = np.vstack([self.vectors, np.array(new_vecs, dtype=np.float32)])

    def query(
        self,
        query_texts: List[str],
        n_results: int = 2,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Queries collection by text with metadata filtering and returns Chroma-style dict."""
        q_vec = self._embed(query_texts[0])
        
        # 1. Apply metadata pre-filtering (where)
        allowed_indices = []
        for i, meta in enumerate(self.metadatas):
            if where:
                match = all(meta.get(k) == v for k, v in where.items())
                if not match:
                    continue
            allowed_indices.append(i)

        if not allowed_indices:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        # 2. Vector distance calculation
        candidate_matrix = self.vectors[allowed_indices]
        scores = np.dot(candidate_matrix, q_vec)
        # Cosine distance = 1.0 - cosine_similarity
        distances = 1.0 - scores

        # 3. Top-N ranking
        ranked_local = np.argsort(distances)[:n_results]
        
        res_ids = [self.ids[allowed_indices[idx]] for idx in ranked_local]
        res_dists = [float(distances[idx]) for idx in ranked_local]
        res_docs = [self.documents[allowed_indices[idx]] for idx in ranked_local]
        res_metas = [self.metadatas[allowed_indices[idx]] for idx in ranked_local]

        return {
            "ids": [res_ids],
            "distances": [res_dists],
            "documents": [res_docs],
            "metadatas": [res_metas]
        }

# Demonstration
col = SimpleChromaCollection(name="support_kb", dimension=8)

# Ingest documents
col.add(
    documents=[
        "Configure SAML 2.0 Single Sign-On with Okta identity provider.",
        "Reset password via email verification link within 24 hours.",
        "Monthly invoice billing schedule and receipt downloads."
    ],
    metadatas=[
        {"dept": "Security", "public": True},
        {"dept": "Security", "public": True},
        {"dept": "Billing", "public": False}
    ],
    ids=["doc_1", "doc_2", "doc_3"]
)

# Query by raw text with metadata filter
results = col.query(
    query_texts=["How to setup SSO with Okta?"],
    n_results=1,
    where={"dept": "Security"}
)

print("ChromaDB Query Results Format:")
print(results)
'''

st.code(code_chroma_impl, language="python")

if st.button("🚀 Execute Minimal ChromaDB Engine", type="primary"):
    exec_scope = {}
    exec(code_chroma_impl, globals(), exec_scope)
    res_dict = exec_scope["results"]
    
    st.success("Chroma-Style Engine executed successfully!")
    st.json(res_dict)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is ChromaDB's dual-storage architecture, and why does it combine SQLite with HNSWlib?"""
a1 = """* **SQLite Subsystem**:
  * Provides rock-solid ACID relational persistence on disk.
  * Stores collection schemas, record IDs, raw text documents, and JSON metadata payloads.
  * Handles scalar metadata filtering (`where={'status': 'active'}`) using proven B-Tree indexing.
* **HNSWlib Subsystem**:
  * High-performance C++ implementation of the Hierarchical Navigable Small World (HNSW) graph.
  * Compiled directly into Python via pybind11, utilizing hardware SIMD AVX-512 vector instructions for sub-millisecond nearest neighbor search.
* **The Synergy**: SQLite handles durable relational metadata storage; HNSWlib handles high-speed geometric vector search. Chroma orchestrates them into a unified, zero-infrastructure embedded database."""

q2 = """2. What are the trade-offs of ChromaDB's embedded architecture versus a client-server vector database like Qdrant or Milvus?"""
a2 = """* **Embedded ChromaDB Advantages**:
  * Zero deployment overhead: installs with `pip install chromadb` and runs directly inside the Python process.
  * Zero network serialization latency: data transfers between Python and the C++ HNSWlib index occur via zero-copy memory pointers.
  * Ideal for rapid prototyping, unit testing, and desktop/edge AI applications.
* **Limitations (When to migrate to Qdrant/Milvus)**:
  * **Concurrency Contention**: Embedded SQLite has a single-writer lock; high-concurrency write pipelines will experience database lock stalls.
  * **Memory Boundaries**: Embedded Chroma is bound to the single host's physical RAM; it cannot distribute a 500-million-vector HNSW index across a 10-node cluster."""

q3 = """3. How does ChromaDB handle embedding generation automatically, and what is the default model?"""
a3 = """* By default, ChromaDB bundles the `onnxruntime` and uses the **`all-MiniLM-L6-v2`** model from Sentence-Transformers (384 dimensions).
* When a user calls `collection.add(documents=["..."])` without passing vectors, Chroma automatically runs ONNX CPU inference on the text and stores the resulting 384-d vector.
* Users can easily swap the default embedding function to OpenAI (`text-embedding-3-small`), Cohere, Google PaLM, or Hugging Face by passing `embedding_function` during collection initialization."""

q4 = """4. What is the difference between Chroma's Ephemeral Client and Persistent Client?"""
a4 = """* **Ephemeral Client (`chromadb.Client()`)**:
  * Runs entirely in volatile memory (`:memory:`).
  * When the Python script terminates or the Jupyter kernel restarts, all collections and vectors are permanently lost.
  * Used strictly for unit tests and quick scratchpad demonstrations.
* **Persistent Client (`chromadb.PersistentClient(path="./chroma_db")`)**:
  * Persists SQLite databases and HNSW index segment files to a local directory.
  * Data survives process restarts and can be inspected across sessions."""

q5 = """5. Why is ChromaDB considered the foundational retrieval engine for local RAG prototypes?"""
a5 = """* Local RAG requires three core primitives: chunking, embedding, and vector storage.
* Chroma eliminates the need to configure cloud accounts, API keys, or Docker containers just to build a prototype.
* A complete RAG pipeline can be authored in under 20 lines of Python code using ChromaDB and LangChain/LlamaIndex.
* Once the prompt engineering and chunking strategies are validated locally, the code can be ported to enterprise vector databases with identical retrieval logic."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("ChromaDB Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The SQLite Database Lock in High-Concurrency FastAPIs**  
> Deploying embedded ChromaDB inside a multi-worker FastAPI or Gunicorn web application causes frequent 
> `sqlite3.OperationalError: database is locked` exceptions under heavy traffic. 
> Because embedded SQLite only allows one writer at a time, concurrent writes queue up and time out. 
> For multi-worker web services, run ChromaDB in **Client-Server Mode** (`chroma run --path ./db`).

> [!WARNING]
> **Trap 2: Inconsistent Embedding Function on Collection Re-opening**  
> If you create a collection with a custom OpenAI embedding function (`1536-d`), but later re-open the collection 
> using `client.get_collection("name")` without passing the same `embedding_function`, Chroma defaults back 
> to `all-MiniLM-L6-v2` (`384-d`)! Subsequent queries will immediately fail with dimension mismatch errors. 
> Always pass your embedding function when fetching existing collections.

> [!WARNING]
> **Trap 3: Ephemeral Data Loss on Streamlit Script Re-runs**  
> Calling `chromadb.Client()` inside a Streamlit app creates a new ephemeral in-memory database on every user 
> interaction or button click, wiping all indexed documents! Always use `chromadb.PersistentClient(path=...)` 
> or cache the client inside `st.session_state` or `@st.cache_resource`.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is ChromaDB?"):
        st.markdown("""
        **ChromaDB**:  
        An open-source, AI-native vector database designed for simplicity, running embedded inside Python or as a lightweight server.
        """)
    with st.expander("🗂️ Flashcard: What Powers Chroma Under the Hood?"):
        st.markdown("""
        **Dual Storage Engine**:  
        SQLite handles relational metadata, IDs, and raw text persistence; C++ HNSWlib handles fast SIMD vector index search.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Ephemeral vs Persistent Client?"):
        st.markdown("""
        **Client Modes**:  
        `Client()` is volatile in-memory (wiped on exit); `PersistentClient(path=...)` writes SQLite and HNSW files to local disk.
        """)
    with st.expander("🗂️ Flashcard: What is the Default Embedding Model?"):
        st.markdown("""
        **Default Embedder**:  
        Sentence-Transformers `all-MiniLM-L6-v2` (384 dimensions) executed locally via ONNX Runtime CPU.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand what ChromaDB is and why it is the leading embedded vector database.", value=True)
chk2 = st.checkbox("I can explain ChromaDB's dual-storage architecture (SQLite + HNSWlib).")
chk3 = st.checkbox("I know how ChromaDB integrates into production Retrieval-Augmented Generation (RAG).")
chk4 = st.checkbox("I understand the difference between Ephemeral in-memory and Persistent disk clients.")
chk5 = st.checkbox("I know the concurrency limitations of embedded SQLite and when to switch to Client-Server mode.")
chk6 = st.checkbox("I can implement a working Chroma-style collection API in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered ChromaDB Fundamentals and are ready for Chapter 6.2: Installation!")
