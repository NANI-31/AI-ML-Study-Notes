import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class VectorEngineSimulator:
    """
    Simulates operational benchmarks, architecture profiles, and feature sets
    across the leading vector database engines.
    """
    ENGINES = {
        "FAISS": {
            "type": "Algorithmic Library",
            "hosting": "In-Process / Embedded (C++/Python)",
            "index_support": ["Flat", "IVF", "PQ", "HNSW"],
            "crud_support": "Limited (Static rebuilding typical)",
            "filtering": "Post-filtering or ID-selector masking",
            "p95_latency_ms": 2.1,
            "max_qps_per_node": 4500,
            "sweet_spot": "Offline batch clustering, embedded local fast search, raw GPU acceleration"
        },
        "Chroma": {
            "type": "Embedded / Client-Server DB",
            "hosting": "Local Python / SQLite / DuckDB backend",
            "index_support": ["HNSW"],
            "crud_support": "Full Real-Time ACID CRUD",
            "filtering": "Single-stage metadata filtering via SQLite",
            "p95_latency_ms": 6.8,
            "max_qps_per_node": 1200,
            "sweet_spot": "Local developer prototyping, embedded agent memory, small-to-medium RAG apps"
        },
        "Pinecone": {
            "type": "Fully-Managed Cloud SaaS",
            "hosting": "Serverless & Pod-based Cloud",
            "index_support": ["Proprietary Graph / HNSW"],
            "crud_support": "Full Managed CRUD",
            "filtering": "Single-stage metadata filtering",
            "p95_latency_ms": 8.5,
            "max_qps_per_node": 2500,
            "sweet_spot": "Zero-DevOps production RAG, serverless auto-scaling, enterprise SLA"
        },
        "Weaviate": {
            "type": "Open-Source / Cloud Database",
            "hosting": "Self-hosted Docker/K8s or Managed Cloud",
            "index_support": ["HNSW", "Flat"],
            "crud_support": "Full GraphQL & REST CRUD",
            "filtering": "Inverted index + HNSW hybrid filtering",
            "p95_latency_ms": 7.2,
            "max_qps_per_node": 1800,
            "sweet_spot": "Multi-modal RAG, built-in vectorization modules, GraphQL integrations"
        },
        "Qdrant": {
            "type": "Open-Source / Cloud Database (Rust)",
            "hosting": "Self-hosted Docker/K8s or Managed Cloud",
            "index_support": ["HNSW", "Scalar Quantization", "Product Quantization"],
            "crud_support": "Full Real-time CRUD (gRPC / REST)",
            "filtering": "Custom payload index + ACORN/Filtered HNSW",
            "p95_latency_ms": 4.2,
            "max_qps_per_node": 3800,
            "sweet_spot": "High-throughput production RAG, advanced payload filtering, memory-mapped storage"
        },
        "Milvus": {
            "type": "Distributed Enterprise DB (Go/C++)",
            "hosting": "Self-hosted Kubernetes or Zilliz Cloud",
            "index_support": ["HNSW", "IVF", "SCaNN", "DiskANN", "Knowhere"],
            "crud_support": "Full Distributed CRUD",
            "filtering": "Bitset-based single-stage filtering",
            "p95_latency_ms": 5.1,
            "max_qps_per_node": 4200,
            "sweet_spot": "Billion-scale enterprise datasets, distributed cluster sharding, GPU indexing"
        },
        "Elasticsearch": {
            "type": "Hybrid Search Engine",
            "hosting": "Self-hosted or Elastic Cloud",
            "index_support": ["HNSW (Lucene)"],
            "crud_support": "Full Distributed Document CRUD",
            "filtering": "Lucene inverted index + vector rescoring",
            "p95_latency_ms": 14.5,
            "max_qps_per_node": 950,
            "sweet_spot": "Enterprise hybrid search combining BM25 keyword matching with dense vectors"
        },
        "pgvector": {
            "type": "Relational Extension",
            "hosting": "PostgreSQL extension (Self-hosted / RDS / Supabase)",
            "index_support": ["HNSW", "IVFFlat"],
            "crud_support": "Full ACID SQL Transactions",
            "filtering": "SQL WHERE clause combined with vector distance",
            "p95_latency_ms": 11.2,
            "max_qps_per_node": 1100,
            "sweet_spot": "Teams already operating PostgreSQL who need transactional ACID with vectors"
        }
    }

    @classmethod
    def evaluate_recommendation(
        cls,
        dataset_size: str,
        devops_preference: str,
        hybrid_needed: bool,
        acid_relational: bool
    ) -> List[Dict[str, Any]]:
        rankings = []
        for name, info in cls.ENGINES.items():
            score = 70
            reasons = []

            # Dataset size checks
            if dataset_size == "Billions (100M+)" and name in ["Milvus", "Qdrant", "Pinecone"]:
                score += 20
                reasons.append("Scales horizontally to 100M+ vectors with distributed sharding.")
            elif dataset_size == "Billions (100M+)" and name in ["Chroma", "pgvector"]:
                score -= 30
                reasons.append("Single-node or relational architecture struggles at 100M+ scale without massive compute.")

            if dataset_size == "Small (<100K)" and name in ["Chroma", "FAISS", "pgvector"]:
                score += 15
                reasons.append("Lightweight, low operational overhead, excellent for small collections.")

            # DevOps preference
            if devops_preference == "Serverless / Zero DevOps" and name == "Pinecone":
                score += 25
                reasons.append("Zero server maintenance, managed automated backups and scaling.")
            elif devops_preference == "Self-Hosted / Open-Source" and name in ["Qdrant", "Milvus", "Weaviate"]:
                score += 20
                reasons.append("Full source availability, run on own Kubernetes/Docker infrastructure.")

            # Relational ACID requirement
            if acid_relational and name == "pgvector":
                score += 35
                reasons.append("Native SQL joins and strict ACID transaction compliance.")
            elif acid_relational and name not in ["pgvector", "Elasticsearch"]:
                score -= 15
                reasons.append("Requires maintaining separate database and synchronization pipeline.")

            # Hybrid Search requirement
            if hybrid_needed and name in ["Elasticsearch", "Qdrant", "Weaviate", "Pinecone"]:
                score += 15
                reasons.append("Built-in hybrid search (BM25 + dense vector fusion).")

            rankings.append({
                "Engine": name,
                "Fit Score": min(100, max(10, score)),
                "Architecture": info["type"],
                "Latency (p95)": f"{info['p95_latency_ms']} ms",
                "Max QPS": info["max_qps_per_node"],
                "Primary Advantage": reasons[0] if reasons else info["sweet_spot"]
            })

        rankings.sort(key=lambda x: x["Fit Score"], reverse=True)
        return rankings


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="8.2 Popular Vector Databases - RAG Master Curriculum",
    page_icon="🏢",
    layout="wide"
)

st.title("8.2 Popular Vector Databases: Architectural Comparison, Engine Archetypes, and Selection Frameworks")
st.caption("Chapter 8: Vector Databases | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
The vector database ecosystem has expanded rapidly from in-memory academic research libraries into multi-billion-dollar enterprise platforms. 
Modern RAG systems choose between three architectural archetypes: **algorithmic vector libraries**, **purpose-built specialized vector databases**, 
and **vector-extended traditional data platforms**.

Selecting the correct vector engine requires evaluating tradeoffs across operational overhead, query throughput, 
persistence durability, metadata filtering capabilities, and cost scalability.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: The Modern Vector Database Ecosystem and Taxonomy", divider="rainbow")
st.markdown("""
The diagram below illustrates the three primary architectural archetypes and maps the major industry engines 
according to their deployment topology and underlying persistence engines.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    eco_dot = """
    digraph VectorTaxonomy {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_categories {
            label = "Vector Engine Architectural Archetypes";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            cat_lib [label="1. In-Memory Vector Libraries\\nDirect C++/Python bindings, no networking layer", fillcolor="#0F172A", color="#38BDF8"];
            cat_native [label="2. Purpose-Built Vector Databases\\nNative storage, WAL, distributed sharding, gRPC", fillcolor="#0F172A", color="#10B981"];
            cat_extended [label="3. Vector-Extended Traditional Stores\\nRelational & NoSQL systems with ANN index plugins", fillcolor="#0F172A", color="#F59E0B"];
        }

        subgraph cluster_engines {
            label = "Leading Industry Engines";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            faiss_node [label="FAISS\\n(Meta, C++/CUDA)\\nIn-memory GPU/CPU compute", fillcolor="#334155", color="#38BDF8"];
            chroma_node [label="Chroma\\n(Embedded/Docker)\\nSQLite/DuckDB + HNSW", fillcolor="#334155", color="#38BDF8"];

            pinecone_node [label="Pinecone\\n(Managed SaaS)\\nServerless / Pod architecture", fillcolor="#334155", color="#10B981"];
            qdrant_node [label="Qdrant\\n(Rust Engine)\\nPayload filter + Filtered HNSW", fillcolor="#334155", color="#10B981"];
            weaviate_node [label="Weaviate\\n(Go Core)\\nGraphQL + Vector modules", fillcolor="#334155", color="#10B981"];
            milvus_node [label="Milvus\\n(Go/C++ Distributed)\\nBillion-scale cluster sharding", fillcolor="#334155", color="#10B981"];

            pgvector_node [label="pgvector\\n(PostgreSQL Extension)\\nACID SQL + HNSW/IVF", fillcolor="#334155", color="#F59E0B"];
            elastic_node [label="Elasticsearch / OpenSearch\\n(Lucene HNSW)\\nHybrid BM25 + Vector", fillcolor="#334155", color="#F59E0B"];
        }

        cat_lib -> faiss_node [label="Library Archetype"];
        cat_lib -> chroma_node [label="Embedded Archetype"];

        cat_native -> pinecone_node [label="Managed Cloud"];
        cat_native -> qdrant_node [label="High Performance Rust"];
        cat_native -> weaviate_node [label="Multi-Modal Engine"];
        cat_native -> milvus_node [label="Distributed Cluster"];

        cat_extended -> pgvector_node [label="Relational Extension"];
        cat_extended -> elastic_node [label="Inverted Index Extension"];
    }
    """
    st.graphviz_chart(eco_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In early 2023, developers used **FAISS** or **Chroma** for everything. Today, enterprise architectures split sharply based on team composition:
1. **Existing Postgres Workloads**: Teams already hosting PostgreSQL on AWS RDS or Supabase adopt **pgvector** to avoid adding a secondary data tier.
2. **Dedicated High-Scale RAG**: Teams handling millions of documents or demanding sub-5ms latency choose **Qdrant** or **Pinecone** for their advanced payload indexing and serverless pricing models.
3. **Billion-Scale Enterprise Clusters**: Platforms indexing internet-scale corpora or multimodal images rely on **Milvus** running on Kubernetes.
""")

st.divider()

# ==============================================================================
# SECTION 2: FAISS
# ==============================================================================

st.header("2. FAISS (Facebook AI Similarity Search)", divider="rainbow")
st.markdown("""
Developed by Meta AI Research, **FAISS** is a high-performance library written in C++ with Python bindings, 
specifically engineered for fast dense vector similarity search and clustering.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Algorithmic Engine, Not a Database**:
      FAISS is a calculation library. It has no network layer, no HTTP/gRPC server, no write-ahead log (WAL), and no user authentication.
    * **State-of-the-Art GPU Acceleration**:
      Contains optimized CUDA kernels for GPU-accelerated index construction and batch queries (`faiss.GpuIndexFlatL2`), achieving 10x-50x higher throughput than CPUs.
    * **Rich Quantization Algorithms**:
      Provides reference implementations of Product Quantization (`IndexIVFPQ`), Scalar Quantization, and Inverted File systems.
    * **Memory Footprint**:
      Raw vector data and indices are stored contiguously in memory buffers without document text or metadata payloads.
    """)

with col2:
    st.subheader("When to Use FAISS")
    st.markdown("""
    * **Offline Batch Processing**:
      Clustering tens of millions of embedding vectors overnight for topic discovery or deduplication.
    * **Embedded Single-Node Workloads**:
      Desktop applications or offline edge servers requiring blazing-fast vector math with zero background daemons.
    * **Research & Custom Algorithm Development**:
      Benchmarking novel quantization or metric learning techniques against baseline indices.
    """)

st.divider()

# ==============================================================================
# SECTION 3: CHROMA
# ==============================================================================

st.header("3. Chroma DB", divider="rainbow")
st.markdown("""
**Chroma** is an open-source, AI-native embedding database designed for developer simplicity and rapid prototyping in Python and TypeScript.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Zero-Setup Embedded Mode**:
      Runs directly within the Python application process using SQLite for metadata and a local HNSW implementation for vector proximity.
    * **Native Text & Metadata Storage**:
      Stores original document chunks, IDs, and metadata dictionaries side-by-side with vectors.
    * **Built-in Embedding Integration**:
      Supports automatic text-to-vector transformation using Sentence Transformers, OpenAI, or Cohere through simple collection configurations.
    * **Client-Server Architecture**:
      Can also run as a standalone Docker container with a REST API for multi-service environments.
    """)

with col2:
    st.subheader("When to Use Chroma")
    st.markdown("""
    * **Local Development & RAG Prototyping**:
      Ideal for spinning up a working RAG application in fewer than 10 lines of code without running Docker or cloud infrastructure.
    * **Desktop & Offline Applications**:
      Perfect for local agent memory (e.g., LangChain local scratchpads) where SQLite provides portable file-based persistence.
    * **Dataset Boundaries**:
      Excels up to ~1,000,000 vectors; larger collections require migration to distributed systems.
    """)

st.divider()

# ==============================================================================
# SECTION 4: PINECONE
# ==============================================================================

st.header("4. Pinecone", divider="rainbow")
st.markdown("""
**Pinecone** is a fully-managed, cloud-native vector database offering zero-DevOps operational simplicity, 
low latency, and enterprise-grade reliability.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Serverless Architecture**:
      Decouples storage and compute. Storage is billed per gigabyte on object stores (S3), while search compute dynamically scales on demand.
    * **Zero Infrastructure Management**:
      No nodes to configure, no Kubernetes pods to monitor, and no manual index rebuilding or compaction cycles.
    * **Integrated Single-Stage Metadata Filtering**:
      Enables low-latency metadata filtering via JSON predicates without degrading vector recall.
    * **Namespaces & Multi-Tenancy**:
      Native partition namespaces allow multi-tenant isolation within a single index collection.
    """)

with col2:
    st.subheader("When to Use Pinecone")
    st.markdown("""
    * **Enterprise SaaS Applications**:
      Teams prioritizing time-to-market and low maintenance over running custom vector infrastructure.
    * **Unpredictable Query Traffic**:
      Serverless indexes scale down to zero cost during idle hours and rapidly handle unexpected traffic spikes.
    * **Global Multi-Region Deployments**:
      Seamless multi-region deployment across AWS, GCP, and Azure.
    """)

st.divider()

# ==============================================================================
# SECTION 5: WEAVIATE
# ==============================================================================

st.header("5. Weaviate", divider="rainbow")
st.markdown("""
**Weaviate** is an open-source, cloud-native vector database written in Go that emphasizes modularity, 
multi-modal vectors, and GraphQL query interfaces.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Modular Architecture (`modules`)**:
      Integrates directly with model providers (OpenAI, HuggingFace, Cohere) for inline text, image, and clip vectorization during ingestion.
    * **Hybrid Search First-Class Support**:
      Combines Lucene-based BM25 sparse keyword search and dense HNSW vector search with Reciprocal Rank Fusion (RRF) out-of-the-box.
    * **GraphQL Query Language**:
      Allows clients to query vectors, scalar fields, and graph relations using structured GraphQL schemas.
    * **Object Storage Backup**:
      Supports continuous snapshotting directly to AWS S3 or Google Cloud Storage.
    """)

with col2:
    st.subheader("When to Use Weaviate")
    st.markdown("""
    * **Multimodal Search**:
      Applications searching across text, product images, and audio clips using unified multimodal embeddings.
    * **Complex Hybrid Retrieval**:
      Systems requiring combined keyword matching (serial numbers, names) and semantic conceptual discovery.
    * **Graph-Relational Data**:
      Workloads where documents maintain interconnected relationships (e.g., authors, articles, tags).
    """)

st.divider()

# ==============================================================================
# SECTION 6: QDRANT
# ==============================================================================

st.header("6. Qdrant", divider="rainbow")
st.markdown("""
**Qdrant** is an open-source vector similarity search engine and database written in Rust, 
engineered for high performance, memory safety, and expressive payload filtering.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Engineered in Rust**:
      Zero-cost abstractions, memory safety without garbage-collection pauses, and high query concurrency.
    * **Payload-Based Single-Stage Filtered Search**:
      Uses specialized payload indices alongside HNSW graphs (ACORN-style traversal) to eliminate recall drop on restrictive filters.
    * **MMAP & Quantization by Default**:
      Supports Scalar Quantization (Int8) and Product Quantization with disk-backed memory-mapped (MMAP) storage, reducing RAM costs by 70%-90%.
    * **Rich gRPC & REST Interfaces**:
      Provides high-speed binary gRPC APIs alongside standard JSON REST endpoints.
    """)

with col2:
    st.subheader("When to Use Qdrant")
    st.markdown("""
    * **High-Throughput Production Workloads**:
      Applications requiring thousands of queries per second with deterministic sub-5ms p95 latencies.
    * **Strict Multi-Tenant Filtering**:
      Enterprise RAG systems enforcing strict per-user, per-department, or per-tenant security filtering.
    * **Cost-Conscious Deployments**:
      Projects indexing tens of millions of vectors on commodity VM hardware using quantization and disk storage.
    """)

st.divider()

# ==============================================================================
# SECTION 7: MILVUS & ZILLIZ
# ==============================================================================

st.header("7. Milvus and Zilliz", divider="rainbow")
st.markdown("""
**Milvus** is an open-source, highly distributed vector database created by Zilliz, 
architected for billion-scale vector workloads and cloud-native Kubernetes clustering.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Disaggregated Cloud-Native Architecture**:
      Decouples storage (MinIO/S3), log brokers (Kafka/Pulsar), index nodes, query nodes, and coordinator nodes.
    * **Billion-Scale Vector Capacity**:
      Built to scale horizontally across hundreds of nodes to index billions of vectors.
    * **Comprehensive Index Library (Knowhere)**:
      Supports HNSW, IVF, SCaNN, DiskANN, and GPU-accelerated RAFT index implementations.
    * **Zilliz Cloud Managed Service**:
      Provides enterprise managed Milvus with auto-scaling and multi-region replication.
    """)

with col2:
    st.subheader("When to Use Milvus")
    st.markdown("""
    * **Internet-Scale Datasets**:
      Corpora exceeding 50,000,000 vectors requiring distributed sharding across Kubernetes clusters.
    * **GPU-Accelerated Indexing**:
      Workloads needing real-time high-throughput vector ingestion using GPU clusters.
    * **Enterprise Data Lake Integration**:
      Direct integration with Apache Spark and data warehouse streaming pipelines.
    """)

st.divider()

# ==============================================================================
# SECTION 8: ELASTICSEARCH & OPENSEARCH
# ==============================================================================

st.header("8. Elasticsearch and OpenSearch Vector Search", divider="rainbow")
st.markdown("""
**Elasticsearch** and **OpenSearch** are traditional distributed search and analytics engines that added dense vector capabilities 
via Apache Lucene's native HNSW implementation.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Apache Lucene HNSW Integration**:
      Stores dense vector fields (`dense_vector`) alongside classical text fields using Lucene segment graphs.
    * **Native Lexical BM25 & Dense Fusion**:
      Combines BM25 lexical token matching with dense vector k-NN search using Reciprocal Rank Fusion (RRF).
    * **Rich Enterprise Ecosystem**:
      Battle-tested security (RBAC), Kibana dashboards, log analysis, and established operational tooling.
    * **Resource Overhead**:
      JVM garbage collection and Lucene segment merges create higher memory and CPU demands compared to lightweight C++/Rust engines.
    """)

with col2:
    st.subheader("When to Use Elasticsearch")
    st.markdown("""
    * **Existing Elasticsearch Deployments**:
      Enterprises already operating Elasticsearch or OpenSearch clusters can adopt vector search without procuring new database licenses.
    * **Heavily Keyword-Dependent Workloads**:
      Applications requiring exact term matching (part numbers, legal statutes, product SKUs) combined with semantic conceptual ranking.
    * **E-Commerce Search**:
      Combining facet filters, scalar price ranges, full-text descriptions, and semantic image vectors.
    """)

st.divider()

# ==============================================================================
# SECTION 9: PGVECTOR
# ==============================================================================

st.header("9. pgvector and Relational Vector Extensions", divider="rainbow")
st.markdown("""
**pgvector** is an open-source extension for PostgreSQL that enables vector storage, exact k-NN queries, 
and approximate nearest neighbor indexing (HNSW and IVFFlat) directly within relational SQL tables.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Characteristics")
    st.markdown("""
    * **Full ACID Compliance**:
      Vector inserts, updates, and deletes participate in standard PostgreSQL transactions and write-ahead logging (WAL).
    * **Zero New Infrastructure**:
      Uses existing PostgreSQL backups, replication, role management, and connection pools (e.g., PgBouncer).
    * **Seamless SQL Joins**:
      Query vectors alongside relational data in a single SQL statement:
      `SELECT title, 1 - (embedding <=> $1) FROM documents WHERE user_id = 42 ORDER BY embedding <=> $1 LIMIT 5`.
    * **Index Support**:
      Supports both `HNSW` (fast search, higher build memory) and `IVFFlat` (lower build memory, requires re-indexing on data growth).
    """)

with col2:
    st.subheader("When to Use pgvector")
    st.markdown("""
    * **The Pragmatic Choice for Startups & Small Teams**:
      If your application already uses PostgreSQL, pgvector eliminates the architectural complexity of synchronizing vectors with an external vector database.
    * **Strict Transactional Workloads**:
      Systems requiring atomic updates across relational records and vectors (e.g., financial or healthcare audits).
    * **Scale Limits**:
      Performs exceptionally up to ~5,000,000 vectors; beyond this, specialized vector engines offer superior query throughput and lower memory consumption.
    """)

st.divider()

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("10. Production Implementation: Unified Multi-Engine Vector Store Client Interface", divider="rainbow")
st.markdown("""
Below is an enterprise adapter architecture abstracting multiple vector database backends 
behind a single clean interface, allowing seamless switching between Qdrant, Pinecone, and pgvector.
""")

st.code('''from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseVectorStoreAdapter(ABC):
    """Abstract interface for enterprise vector database clients."""
    
    @abstractmethod
    def upsert_vectors(self, collection_name: str, records: List[Dict[str, Any]]) -> bool:
        """Insert or update vectors with payloads."""
        pass

    @abstractmethod
    def query_similarity(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query top-K similar vectors with optional metadata filtering."""
        pass

class QdrantAdapter(BaseVectorStoreAdapter):
    """Adapter for Qdrant client."""
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        # In production: self.client = QdrantClient(host=host, port=port)

    def upsert_vectors(self, collection_name: str, records: List[Dict[str, Any]]) -> bool:
        # Translates generic records into Qdrant PointStruct objects
        return True

    def query_similarity(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Translates filters to qdrant_client.models.Filter
        return [
            {"id": "doc_101", "score": 0.94, "payload": {"source": "manual.pdf"}}
        ]

class PgVectorAdapter(BaseVectorStoreAdapter):
    """Adapter for PostgreSQL pgvector."""
    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        # In production: self.pool = psycopg2.pool.SimpleConnectionPool(...)

    def upsert_vectors(self, collection_name: str, records: List[Dict[str, Any]]) -> bool:
        # Executes: INSERT INTO table (id, embedding, metadata) VALUES (...) ON CONFLICT DO UPDATE
        return True

    def query_similarity(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Executes: SELECT id, 1 - (embedding <=> %s) AS score FROM table WHERE ... ORDER BY score DESC LIMIT %s
        return [
            {"id": "doc_101", "score": 0.93, "payload": {"source": "manual.pdf"}}
        ]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("11. Interactive Workbench: Vector Engine Decision Matrix and Latency Simulator", divider="rainbow")
st.markdown("""
Use this decision advisor to evaluate which vector database engine best matches your production requirements. 
Adjust dataset scale, operational preferences, and feature needs to inspect suitability scores and performance profiles.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Requirements")
    selected_scale = st.selectbox(
        "Estimated Corpus Scale",
        ["Small (<100K)", "Medium (100K - 5M)", "Large (5M - 50M)", "Billions (100M+)"],
        index=1
    )

    selected_ops = st.selectbox(
        "DevOps & Operational Preference",
        ["Self-Hosted / Open-Source", "Serverless / Zero DevOps", "Embedded In-Process"],
        index=0
    )

    req_hybrid = st.toggle("Requires Built-in Hybrid Search (BM25 + Dense)", value=True)
    req_acid = st.toggle("Requires Native ACID Relational SQL Transactions", value=False)

with col2:
    st.subheader("Engine Recommendation Rankings")
    rankings = VectorEngineSimulator.evaluate_recommendation(
        selected_scale, selected_ops, req_hybrid, req_acid
    )
    rank_df = pd.DataFrame(rankings)
    st.dataframe(rank_df[["Engine", "Fit Score", "Architecture", "Latency (p95)", "Max QPS"]], use_container_width=True, hide_index=True)
    
    top_engine = rankings[0]
    st.success(f"Top Recommendation: **{top_engine['Engine']}** (Fit Score: {top_engine['Fit Score']}%) - {top_engine['Primary Advantage']}")

st.divider()

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: When should an engineering team choose pgvector over a specialized vector database like Qdrant or Pinecone?"):
    st.markdown("""
    **Answer**:
    An engineering team should choose **pgvector** when:
    1. The core application stack already relies heavily on PostgreSQL.
    2. Vector corpus size is under ~5,000,000 vectors.
    3. The application requires strict ACID transactions where vector updates must commit atomically alongside relational tables.
    4. Operational complexity must be minimized (avoiding separate ETL pipelines, backups, and security tiers).
    Specialized engines should be chosen when query throughput exceeds thousands of QPS, dataset size reaches tens of millions, or memory-mapped quantization is mandatory.
    """)

with st.expander("Q2: What is the architectural difference between a vector library (FAISS) and an embedded vector database (Chroma)?"):
    st.markdown("""
    **Answer**:
    **FAISS** is a bare algorithmic C++ library designed for fast vector arithmetic and index construction on static arrays; 
    it has no concept of documents, strings, or scalar metadata. 
    **Chroma** is an embedded database system that wraps HNSW with an SQLite storage backend, persisting raw document text, 
    metadata dictionaries, and unique IDs alongside embedding vectors while offering real-time CRUD APIs.
    """)

with st.expander("Q3: How does Qdrant achieve sub-5ms search latencies on multi-tenant filtered queries?"):
    st.markdown("""
    **Answer**:
    Qdrant achieves this through its native **Payload Index** and **Filtered HNSW (ACORN)** graph traversal. 
    Instead of performing pre-filtering (which can break graph connectivity) or post-filtering (which drops recall), 
    Qdrant inspects metadata conditions directly during each hop of graph traversal. Combined with its memory-efficient 
    Rust implementation, zero garbage-collection overhead, and SIMD vector instructions, it maintains sub-5ms latencies.
    """)

with st.expander("Q4: Why does Milvus decouple its query nodes, index nodes, and storage into separate distributed microservices?"):
    st.markdown("""
    **Answer**:
    Index building is computationally heavy (CPU/GPU-bound), vector querying is memory- and latency-bound, and raw data storage is capacity-bound. 
    By decoupling these tiers, Milvus can independently auto-scale Query Nodes to handle sudden read traffic surges, scale Index Nodes during 
    batch document ingestion, and leverage cost-effective cloud object stores (S3/MinIO) for long-term persistence without wasting high-RAM server resources.
    """)

with st.expander("Q5: What are the main tradeoffs of using Elasticsearch or OpenSearch as a vector database?"):
    st.markdown("""
    **Answer**:
    * **Advantages**: Seamless out-of-the-box hybrid search combining Lucene BM25 sparse keyword scoring with dense HNSW vectors, unified query syntax, and enterprise security.
    * **Tradeoffs**: Higher RAM and CPU consumption due to JVM heap overhead, higher search latency (p95 often 10-20ms compared to 3-5ms in Rust/C++ engines), and slower index build throughput during large-scale ingestion.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Dual-Write Inconsistency Between SQL DB and Standalone Vector DB**  
    Writing relational data to PostgreSQL and vectors to an external vector database without a transactional outbox 
    or CDC pipeline leads to orphan records when one write fails, causing corrupted RAG retrievals.
    """)
    st.error("""
    **Trap 2: Neglecting IVF Index Maintenance in pgvector (IVFFlat vs. HNSW)**  
    Building an `IVFFlat` index on pgvector before all documents are inserted trains cluster centroids on an incomplete distribution. 
    As millions of new vectors arrive, query recall drops drastically unless the index is dropped and rebuilt. (Prefer `HNSW` in pgvector).
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Underestimating Memory Overhead of Unquantized HNSW in Production**  
    Running large HNSW collections on specialized cloud databases without scalar or product quantization 
    causes memory usage to balloon. Budget for at least 1.5x to 2x raw vector size in RAM.
    """)
    st.warning("""
    **Trap 4: Overpaying for Serverless Vector Stores During High Continuous QPS**  
    Serverless platforms charge per read unit. Under continuous high-QPS production workloads, 
    serverless hosting costs can far exceed the monthly cost of self-hosted dedicated instances (e.g., Qdrant on Kubernetes).
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: pgvector Sweet Spot")
        st.caption("When is it the right choice?")
        st.markdown("**Back**: Teams already using PostgreSQL with <5M vectors needing ACID transactions and zero new infrastructure.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Qdrant Key Advantage")
        st.caption("What makes it stand out?")
        st.markdown("**Back**: Rust engine with high throughput, native payload indexing for filtered HNSW, and memory-mapped quantization.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Milvus Scalability")
        st.caption("How does it scale?")
        st.markdown("**Back**: Disaggregated cloud-native microservices architecture on Kubernetes, designed for 100M+ to billion-scale vector collections.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: FAISS vs. Vector DB")
        st.caption("What is missing in FAISS?")
        st.markdown("**Back**: FAISS has no network server, WAL, persistence management, CRUD APIs, or metadata payload storage; it is purely an in-memory math library.")

st.divider()

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Popular Vector Databases:")

c1 = st.checkbox("I can categorize vector engines into algorithmic libraries, specialized databases, and relational extensions.", value=True)
c2 = st.checkbox("I understand the specific strengths and operational limitations of FAISS, Chroma, and Pinecone.", value=True)
c3 = st.checkbox("I know when to choose Qdrant, Weaviate, or Milvus for enterprise-scale self-hosted RAG deployments.", value=True)
c4 = st.checkbox("I understand the architectural tradeoffs of pgvector and when staying within PostgreSQL is the optimal engineering decision.", value=True)
c5 = st.checkbox("I can prevent dual-write inconsistency and memory exhaustion traps in multi-database RAG architectures.", value=True)
c6 = st.checkbox("I can implement an abstract multi-engine client adapter pattern in production Python code.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered the Popular Vector Databases landscape and can evaluate and select the right engine for any production RAG workload.")
