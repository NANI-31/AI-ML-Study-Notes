import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="28.4 Vector Databases - LangChain Systems",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vector store telemetry
st.markdown("""
<style>
    .vector-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-vdb {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class VectorDatabaseBenchmarkSimulator:
    """
    Simulates multi-engine vector similarity searches across:
    Pinecone, Chroma, FAISS, Qdrant, Weaviate, and pgvector.
    """
    def __init__(self):
        self.providers = {
            "Pinecone (Cloud Managed)": {
                "type": "Managed Serverless",
                "index_type": "Hierarchical Navigable Small World (HNSW)",
                "filtering": "Metadata Namespaces + Predicates",
                "scale": "Billions of vectors",
                "best_for": "Zero-ops enterprise cloud deployment"
            },
            "Chroma (Embedded/Local)": {
                "type": "Local In-Memory / Docker",
                "index_type": "HNSW via hnswlib",
                "filtering": "Dictionary where-filters",
                "scale": "Millions of vectors",
                "best_for": "Rapid prototyping, testing, local edge agents"
            },
            "FAISS (Meta Research)": {
                "type": "In-Memory C++ / Python",
                "index_type": "IVF-Flat, HNSW, Product Quantization",
                "filtering": "In-memory ID mapping",
                "scale": "Millions on GPU/RAM",
                "best_for": "Raw latency benchmarks and offline batch clustering"
            },
            "Qdrant (Rust Native)": {
                "type": "Distributed Cloud / Self-Hosted",
                "index_type": "HNSW with payload indexing",
                "filtering": "Rich boolean JSON payload filters",
                "scale": "Tens of billions",
                "best_for": "High-throughput production microservices with strict filtering"
            },
            "Weaviate (Multi-Modal)": {
                "type": "Vector Native Engine",
                "index_type": "HNSW + Custom vectorizers",
                "filtering": "GraphQL & hybrid BM25 search",
                "scale": "Billions of objects",
                "best_for": "Multimodal (images, audio, text) and hybrid search"
            },
            "pgvector (PostgreSQL Extension)": {
                "type": "Relational DB Extension",
                "index_type": "HNSW / IVFFlat in Postgres",
                "filtering": "Native SQL WHERE clauses + ACID transactions",
                "scale": "Tens of millions",
                "best_for": "Unifying relational tables and vector embeddings in one DB"
            }
        }

    def simulate_search(self, provider_key: str, query: str) -> Dict[str, Any]:
        info = self.providers.get(provider_key, self.providers["Chroma (Embedded/Local)"])
        return {
            "provider": provider_key,
            "specs": info,
            "results": [
                {"chunk_id": "chk_101", "score": 0.94, "content": "LangChain Runnables provide standard streaming and async contracts across all components.", "metadata": {"source": "architecture.md", "tier": "enterprise"}},
                {"chunk_id": "chk_204", "score": 0.88, "content": "LangGraph manages stateful, multi-actor cyclic agent workflows with persistent checkpointers.", "metadata": {"source": "orchestration.md", "tier": "enterprise"}}
            ]
        }


# --- LESSON HEADER ---
st.title("28.4 Vector Databases: Pinecone, Chroma, FAISS, Qdrant, Weaviate & pgvector")
st.markdown("""
Vector databases form the retrieval backbone of Retrieval-Augmented Generation (RAG) and semantic search.
By converting text into high-dimensional numerical embeddings, vector databases enable nearest-neighbor similarity
searches across millions of unstructured documents. This lesson compares the six premier vector store integrations
in LangChain, covering indexing algorithms, metadata filtering, deployment topologies, and architectural tradeoffs.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Vector Database Ecosystem", divider="rainbow")
st.markdown("""
A vector database pipeline transforms raw documents into searchable embedding vectors.
Documents are loaded, chunked, and embedded via neural models (e.g. `text-embedding-3-small`).
The vectors are indexed using approximate nearest neighbor algorithms (HNSW, IVF-Flat) and queried
via cosine similarity, with metadata filters pruning candidate vectors before or after indexing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VectorDatabases_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_ingest {
            label = "Document Ingestion & Embedding Pipeline";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            RawDocs [label="Enterprise Documents\\n(PDFs, Markdown, Web pages, DB records)", fillcolor="#0F172A", color="#3B82F6"];
            TextSplitter [label="RecursiveCharacterTextSplitter\\n(chunk_size=1000, chunk_overlap=150)", fillcolor="#0F172A", color="#3B82F6"];
            EmbeddingModel [label="Embedding Model (1536d / 3072d)\\n(OpenAI / HuggingFace / Cohere)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_vector_engines {
            label = "LangChain Vector Store Integrations";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            PineconeEngine [label="Pinecone\\n(Managed Serverless Cloud, Namespaces)", fillcolor="#1E293B", color="#60A5FA"];
            ChromaEngine [label="Chroma\\n(Local In-Memory / Ephemeral / Docker)", fillcolor="#1E293B", color="#60A5FA"];
            FAISSEngine [label="FAISS\\n(Meta Research, In-Memory C++, GPU/CPU)", fillcolor="#1E293B", color="#60A5FA"];
            QdrantEngine [label="Qdrant\\n(Rust Native, Advanced Payload Filtering)", fillcolor="#1E293B", color="#60A5FA"];
            WeaviateEngine [label="Weaviate\\n(Multimodal Native, Hybrid Search)", fillcolor="#1E293B", color="#60A5FA"];
            PGVectorEngine [label="pgvector\\n(PostgreSQL Extension, ACID Transactions)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_query_retrieval {
            label = "Approximate Nearest Neighbor (ANN) Retrieval";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            QueryIn [label="User Query Vector\\n(Generated at runtime)", fillcolor="#0369A1", color="#38BDF8"];
            HNSWIndex [label="HNSW / IVF Index Search\\n(Cosine Distance / Dot Product KNN)", fillcolor="#0369A1", color="#38BDF8"];
            MetadataFilter [label="Metadata Pre-Filter\\n(e.g. tenant_id == '101' AND year >= 2026)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_synthesis {
            label = "Downstream Generation";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            TopKDocs [label="Top-K Relevant Chunks\\n(Fed into LLM context window)", fillcolor="#065F46", color="#10B981"];
        }

        RawDocs -> TextSplitter [label="Split"];
        TextSplitter -> EmbeddingModel [label="Embed text"];
        EmbeddingModel -> PineconeEngine [label="Upsert vectors + meta"];
        EmbeddingModel -> ChromaEngine [label="Upsert vectors + meta"];
        EmbeddingModel -> FAISSEngine [label="Upsert vectors + meta"];
        EmbeddingModel -> QdrantEngine [label="Upsert vectors + meta"];
        EmbeddingModel -> WeaviateEngine [label="Upsert vectors + meta"];
        EmbeddingModel -> PGVectorEngine [label="Upsert vectors + meta"];

        QueryIn -> HNSWIndex [label="Search query vector"];
        MetadataFilter -> HNSWIndex [label="Prune search space"];
        HNSWIndex -> TopKDocs [label="Return top-k Document objects"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: PINECONE ---
st.header("2. Pinecone: Managed Cloud Serverless Vector Search & Namespaces", divider="rainbow")
st.markdown("""
Pinecone is a fully managed, cloud-native vector database designed for high availability and zero operational overhead.
It supports serverless auto-scaling and **namespaces**, which provide partition-level multi-tenancy within a single index.
""")

st.code("""
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Pinecone Index with Multi-Tenant Namespace
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="enterprise-knowledge-base",
    embedding=embeddings,
    namespace="tenant_alpha_99"  # Strict tenant isolation
)

# Similarity search with metadata filtering
results = vectorstore.similarity_search(
    "How does authentication work?",
    k=3,
    filter={"tier": "enterprise"}
)
""", language="python")

# --- SECTION 3: CHROMA ---
st.header("3. Chroma: Lightweight Local & Client-Server Embedding Storage", divider="rainbow")
st.markdown("""
Chroma is the default vector database for LangChain development due to its zero-configuration local setup.
It runs embedded in memory, persists to a local directory, or connects to a client-server Docker deployment.
""")

st.code("""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1. Ephemeral In-Memory Vector Store (Ideal for unit tests)
in_memory_store = Chroma.from_documents(docs, embeddings)

# 2. Local Persistent Vector Store
persistent_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="policy_docs"
)
""", language="python")

# --- SECTION 4: FAISS ---
st.header("4. FAISS: High-Performance In-Memory GPU/CPU Vector Similarity by Meta", divider="rainbow")
st.markdown("""
FAISS (Facebook AI Similarity Search) is a C++ library with Python bindings optimized for extremely fast vector search
over dense vectors in RAM or on GPUs. It is unmanaged and does not include a network server, making it ideal
for batch processing and embedded high-speed clustering.
""")

st.code("""
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build FAISS index in memory
faiss_store = FAISS.from_documents(docs, embeddings)

# Save serialized index to disk
faiss_store.save_local("faiss_index_dir")

# Reload serialized index
loaded_store = FAISS.load_local(
    "faiss_index_dir",
    embeddings,
    allow_dangerous_deserialization=True  # Required for pickle safety
)
""", language="python")

# --- SECTION 5: QDRANT ---
st.header("5. Qdrant: Production Rust Vector Search Engine with Advanced Payload Filtering", divider="rainbow")
st.markdown("""
Written in Rust, Qdrant is an enterprise-grade vector search engine built for high throughput.
Its defining advantage is **payload-based filtering**: metadata is indexed alongside vectors, allowing complex boolean
filters (`AND`, `OR`, `NOT`, geo-radius, nested JSON) to execute directly during the HNSW graph traversal.
""")

st.code("""
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")

qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="legal_contracts",
    embedding=embeddings
)

# Rich payload filter during vector similarity search
results = qdrant_store.similarity_search(
    "Liability clause limitations",
    k=4,
    filter=models.Filter(
        must=[
            models.FieldCondition(key="metadata.jurisdiction", match=models.MatchValue(value="US")),
            models.FieldCondition(key="metadata.active", match=models.MatchValue(value=True))
        ]
    )
)
""", language="python")

# --- SECTION 6: WEAVIATE ---
st.header("6. Weaviate: Multi-Modal GraphQL & Native Vector Search Engine", divider="rainbow")
st.markdown("""
Weaviate is an open-source vector search engine featuring built-in vectorization modules, GraphQL APIs,
and multi-modal capabilities (indexing images, video, text, and 3D objects simultaneously).
It supports hybrid search out-of-the-box, combining BM25 keyword matching with dense vectors.
""")

st.code("""
import weaviate
from langchain_weaviate import WeaviateVectorStore

weaviate_client = weaviate.connect_to_local()

weaviate_store = WeaviateVectorStore(
    client=weaviate_client,
    index_name="EnterpriseKnowledge",
    text_key="text",
    embedding=embeddings
)

# Native hybrid search (Dense vector + BM25 keyword)
results = weaviate_store.similarity_search(
    "Q3 sales figures",
    k=3,
    alpha=0.5  # 0.5 balances dense vector and sparse keyword weights equally
)
""", language="python")

# --- SECTION 7: PGVECTOR ---
st.header("7. pgvector: Relational PostgreSQL Hybrid Vector Extensions & ACID Guarantees", divider="rainbow")
st.markdown("""
`pgvector` is an open-source extension that adds vector data types and similarity indexes (HNSW, IVFFlat) to PostgreSQL.
Its primary architectural advantage is **consolidation**: relational tables, user accounts, transactional ledgers,
and vector embeddings reside within the exact same database engine with full ACID transaction guarantees.
""")

st.code("""
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

connection_string = "postgresql+psycopg://postgres:pass@localhost:5432/ai_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Unified relational + vector store
pg_store = PGVector(
    embeddings=embeddings,
    collection_name="employee_policies",
    connection=connection_string,
    use_jsonb=True
)

# Seamless similarity search inside PostgreSQL
results = pg_store.similarity_search("Remote work policy", k=3)
""", language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Multi-Provider Vector Store Adapter", divider="rainbow")
st.markdown("""
Below is an enterprise adapter pattern allowing applications to switch seamlessly between
Chroma (local development), pgvector (hybrid relational), and Pinecone (cloud production)
via configuration without rewriting business logic.
""")

st.code("""
# vector_store_factory.py
from typing import Union
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(env: str, collection_name: str) -> VectorStore:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if env == "local":
            from langchain_community.vectorstores import Chroma
            return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory="./chroma_data")
            
        elif env == "staging":
            from langchain_postgres import PGVector
            return PGVector(
                embeddings=embeddings,
                collection_name=collection_name,
                connection="postgresql+psycopg://user:pass@localhost:5432/db"
            )
            
        elif env == "production":
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore.from_existing_index(
                index_name="prod-cluster",
                embedding=embeddings
            )
            
        raise ValueError(f"Unknown environment: {env}")
""", language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Vector Database Selector & Similarity Search Simulator", divider="rainbow")
st.markdown("""
Compare vector database engines across architectural parameters, index types, metadata filtering capabilities,
and simulate semantic search retrievals.
""")

bench = VectorDatabaseBenchmarkSimulator()

col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    st.subheader("🧭 Vector Store Selector")
    chosen_provider = st.selectbox(
        "Choose Vector Database Provider:",
        list(bench.providers.keys())
    )
    search_prompt = st.text_input("Simulate Retrieval Query:", value="How do LangGraph checkpointers persist agent state?")
    run_search_btn = st.button("🚀 Execute Simulated Vector Retrieval", type="primary")

with col_v2:
    st.subheader("📊 Engine Profile & Retrieved Chunks")
    if run_search_btn:
        res = bench.simulate_search(chosen_provider, search_prompt)
        specs = res["specs"]
        
        st.markdown(
            f"<div class='vector-card'>"
            f"<span class='badge-vdb'>{res['provider']}</span><br><br>"
            f"<b>Architecture:</b> {specs['type']}<br>"
            f"<b>Index Type:</b> {specs['index_type']}<br>"
            f"<b>Filtering:</b> {specs['filtering']}<br>"
            f"<b>Typical Scale:</b> {specs['scale']}<br><br>"
            f"<b>Ideal Workload:</b> {specs['best_for']}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.caption("Top Retrieved Chunks (KNN Cosine Similarity):")
        for chunk in res["results"]:
            with st.container(border=True):
                st.markdown(f"**Chunk ID**: `{chunk['chunk_id']}` | **Score**: `{chunk['score']}`")
                st.write(chunk["content"])
                st.caption(f"Metadata: {json.dumps(chunk['metadata'])}")
    else:
        st.info("Select an engine and run retrieval to inspect architectural tradeoffs.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is the difference between pre-filtering and post-filtering in vector databases?"):
    st.markdown("""
    **Answer**: In **post-filtering**, the index retrieves the top-k nearest vectors first, and then discards any vectors that do not match the metadata filter.
    If the filter is selective (e.g. only 1% of docs match), post-filtering may return zero results. In **pre-filtering** (or single-stage HNSW filtering),
    the database restricts the search space to matching metadata *during* the graph traversal, guaranteeing that top-k results are returned.
    """)

with st.expander("Q2: Why is pgvector popular in enterprise architectures despite standalone vector DBs?"):
    st.markdown("""
    **Answer**: Operational simplicity and ACID transactions. With pgvector, organizations do not need to operate, secure, and synchronize a separate
    distributed vector database. Relational tables (users, orders) and vector embeddings live in the same PostgreSQL database, allowing atomic joins
    between relational data and embeddings in a single SQL query.
    """)

with st.expander("Q3: What is the HNSW algorithm and why is it dominant in vector search?"):
    st.markdown("""
    **Answer**: HNSW (Hierarchical Navigable Small World) is a graph-based Approximate Nearest Neighbor (ANN) algorithm. It builds a multi-layer graph
    where upper layers have long-distance skip links for fast navigation and lower layers have dense short-distance links for high-precision search.
    It provides logarithmic search complexity ($O(\\log N)$), high recall, and rapid query response times.
    """)

with st.expander("Q4: When should FAISS be chosen over Pinecone or Qdrant?"):
    st.markdown("""
    **Answer**: Choose FAISS when you need raw in-memory performance on local machines or GPUs without network latency, for offline batch embedding
    clustering, or when deploying to air-gapped environments without cloud access. Choose Pinecone or Qdrant when you require persistent network services,
    multi-tenancy, dynamic metadata updates, and horizontal auto-scaling.
    """)

with st.expander("Q5: What are namespaces in Pinecone and why are they used?"):
    st.markdown("""
    **Answer**: Namespaces partition an index into discrete, isolated subsets. Queries to a specific namespace only search vectors within that partition.
    They are the standard mechanism for achieving tenant isolation in multi-tenant SaaS applications, preventing cross-tenant data leakage.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Using In-Memory FAISS for Multi-Pod Clusters)**: Storing embeddings in an in-memory FAISS index inside a Kubernetes container. When the container restarts or scales to multiple pods, state is lost or desynchronized.")
    st.warning("**Trap 2 (Mismatched Embedding Dimensions)**: Generating embeddings with a 1536-dimension model and attempting to insert them into a 3072-dimension vector index, causing immediate dimension mismatch errors.")
with col_trap2:
    st.error("**Trap 3 (Relying on Post-Filtering with Selective Metadata)**: Using vector stores that only support post-filtering when querying rare tags, resulting in empty top-k results.")
    st.warning("**Trap 4 (Missing Vector Index Creation in pgvector)**: Adding vectors to PostgreSQL without creating an `HNSW` or `IVFFlat` index. The database falls back to sequential scans ($O(N)$), degrading query speed as tables grow.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary architectural advantage of pgvector?")
        st.caption("Flip for answer:")
        st.info("**Back**: Co-locates relational tables and vector embeddings in PostgreSQL with full ACID transaction guarantees.")
    with st.container(border=True):
        st.markdown("**Front**: What algorithm powers approximate nearest neighbor search in most vector stores?")
        st.caption("Flip for answer:")
        st.info("**Back**: HNSW (Hierarchical Navigable Small World).")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does Pinecone achieve multi-tenant data isolation?")
        st.caption("Flip for answer:")
        st.info("**Back**: Through namespaces, which partition an index into isolated search partitions.")
    with st.container(border=True):
        st.markdown("**Front**: What programming language is Qdrant written in?")
        st.caption("Flip for answer:")
        st.info("**Back**: Rust (providing high-throughput, memory-safe vector search with payload indexing).")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural tradeoffs between managed, embedded, and relational vector stores.", value=True)
c2 = st.checkbox("I know how to implement multi-tenant namespace isolation in Pinecone.", value=True)
c3 = st.checkbox("I can build in-memory and persistent vector stores using Chroma and FAISS.", value=True)
c4 = st.checkbox("I understand Qdrant's payload-based filtering during HNSW graph traversal.", value=True)
c5 = st.checkbox("I can configure pgvector to unify relational data and embeddings inside PostgreSQL.", value=True)
c6 = st.checkbox("I understand why pre-filtering prevents empty results in highly selective metadata searches.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 28: Databases with LangChain.")
