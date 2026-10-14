import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class VectorEngineProfile:
    """Represents the architectural profile and operational metrics of a vector store."""
    def __init__(self, name: str, category: str, language: str, persistence: str,
                 max_scale: str, hybrid_search: bool, typical_latency_ms: float, monthly_cost_est: str):
        self.name = name
        self.category = category
        self.language = language
        self.persistence = persistence
        self.max_scale = max_scale
        self.hybrid_search = hybrid_search
        self.typical_latency_ms = typical_latency_ms
        self.monthly_cost_est = monthly_cost_est

class MockVectorStoreEcosystemSimulator:
    """Simulates multi-engine vector store evaluation and deployment trade-offs."""
    def __init__(self):
        self.engines: Dict[str, VectorEngineProfile] = {
            "FAISS": VectorEngineProfile("FAISS", "In-Memory Library", "C++ / Python", "Ephemeral (Disk serialization)", "10M Vectors", False, 1.8, "0 USD (Local CPU/GPU)"),
            "Chroma": VectorEngineProfile("Chroma", "Embedded Store", "Python / Rust", "Local SQLite / DuckDB / Client-Server", "1M Vectors", False, 4.5, "0 USD (Local)"),
            "Pinecone": VectorEngineProfile("Pinecone", "Managed Cloud SaaS", "Proprietary", "Cloud Persistent (Multi-Region)", "1B+ Vectors", True, 18.0, "70 - 500+ USD/mo"),
            "Weaviate": VectorEngineProfile("Weaviate", "Vector DB & Search Engine", "Go", "Cloud / Self-Hosted (LSM-Tree)", "100M+ Vectors", True, 8.2, "50 - 300 USD/mo"),
            "Qdrant": VectorEngineProfile("Qdrant", "High-Performance Vector DB", "Rust", "Cloud / Self-Hosted (Disk mmap)", "500M+ Vectors", True, 5.1, "40 - 250 USD/mo"),
            "Milvus": VectorEngineProfile("Milvus", "Distributed Cloud-Native", "Go / C++", "Distributed (MinIO + Etcd + Pulsar)", "10B+ Vectors", True, 12.4, "200 - 1500+ USD/mo"),
            "Elasticsearch": VectorEngineProfile("Elasticsearch", "Search Engine Extension", "Java", "Distributed Lucene Segments", "100M+ Vectors", True, 15.0, "100 - 600 USD/mo"),
            "PostgreSQL / pgvector": VectorEngineProfile("PostgreSQL / pgvector", "Relational DB Extension", "C", "Relational Disk (WAL + Tables)", "10M Vectors", True, 9.5, "25 - 150 USD/mo")
        }

    def recommend_engine(self, doc_count: int, deployment_pref: str, need_relational_joins: bool) -> List[Dict[str, Any]]:
        scored = []
        for name, profile in self.engines.items():
            score = 70
            reasons = []

            # Scale evaluation
            if doc_count > 1000000 and name in ["Milvus", "Pinecone", "Qdrant"]:
                score += 20
                reasons.append("Scales effortlessly beyond 1M records.")
            elif doc_count < 50000 and name in ["Chroma", "FAISS"]:
                score += 25
                reasons.append("Zero infrastructure overhead for lightweight corpora.")

            # Deployment alignment
            if deployment_pref == "Embedded (Zero Infra)" and name in ["Chroma", "FAISS"]:
                score += 30
                reasons.append("Runs directly inside Python process memory.")
            elif deployment_pref == "Managed Cloud SaaS" and name in ["Pinecone", "Weaviate", "Qdrant"]:
                score += 25
                reasons.append("Fully managed infrastructure with zero operational burden.")
            elif deployment_pref == "Self-Hosted Enterprise VPC" and name in ["Qdrant", "Milvus", "PostgreSQL / pgvector"]:
                score += 20
                reasons.append("Open-source containerized deployment inside private VPC.")

            # Relational joins
            if need_relational_joins and name == "PostgreSQL / pgvector":
                score += 35
                reasons.append("ACID relational joins across customer tables and vectors.")

            scored.append({
                "engine": name,
                "fit_score": min(100, score),
                "category": profile.category,
                "latency_ms": profile.typical_latency_ms,
                "cost_est": profile.monthly_cost_est,
                "reasons": reasons
            })

        scored.sort(key=lambda x: x["fit_score"], reverse=True)
        return scored


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="14.2 Vector Store Integrations",
    page_icon="🔌",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">14.2 Vector Store Integrations</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Evaluating the production vector store ecosystem: FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, and PostgreSQL/pgvector.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Modern Vector Store Ecosystem & Integration Landscape", divider="rainbow")

st.markdown("""
The vector database ecosystem spans multiple architectural paradigms: from lightweight in-memory C++ libraries (FAISS) 
and embedded developer datastores (Chroma), to managed multi-tenant cloud SaaS (Pinecone), high-performance Rust engines (Qdrant), 
distributed Kubernetes systems (Milvus), and battle-tested relational extensions (PostgreSQL / pgvector).
LangChain abstracts these disparate backends behind an identical unified API.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VectorStoreEcosystem {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_embedded {
            label="Tier 1: Embedded & In-Memory (Zero Infrastructure)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            FAISSNode [label="FAISS (Facebook AI)\\n• C++ core / GPU CUDA\\n• Ephemeral RAM index\\n• Best for: Batch clustering, local experiments", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            ChromaNode [label="Chroma\\n• SQLite / Parquet backend\\n• Local Python process\\n• Best for: Rapid dev, notebooks, prototypes", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_cloud_saas {
            label="Tier 2: Managed Serverless Cloud SaaS";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            PineconeNode [label="Pinecone\\n• Serverless auto-scaling\\n• Zero ops / high SLA\\n• Best for: Fast time-to-market, cloud startups", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_standalone {
            label="Tier 3: Dedicated Open-Source Vector DBs";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            QdrantNode [label="Qdrant (Rust)\\n• Payload filtering & quantization\\n• Low latency / memory mmap\\n• Best for: Self-hosted enterprise VPC", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            WeaviateNode [label="Weaviate (Go)\\n• Multi-modal / GraphQL / gRPC\\n• Native hybrid vector + BM25\\n• Best for: Multi-modal enterprise search", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            MilvusNode [label="Milvus (Go/C++)\\n• Cloud-native distributed K8s\\n• Decoupled storage & compute\\n• Best for: 100M+ to billion-scale vectors", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_extensions {
            label="Tier 4: Enterprise Search & Relational Extensions";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            PGVectorNode [label="PostgreSQL / pgvector\\n• ACID relational joins\\n• Single database stack\\n• Best for: Teams with existing Postgres", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            ESNode [label="Elasticsearch / OpenSearch\\n• Lucene dense_vector fields\\n• Enterprise text search + vectors\\n• Best for: Existing ELK stack users", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
        }

        FAISSNode -> PineconeNode [style="invis"];
        PineconeNode -> QdrantNode [style="invis"];
        QdrantNode -> PGVectorNode [style="invis"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: FAISS ---
st.header("2. FAISS", divider="rainbow")

st.markdown("""
**FAISS (Facebook AI Similarity Search)** is a library written in C++ by Meta AI Research.
- **Key Strength**: Blazing raw GPU and multi-core CPU performance. Supports exact search, IVF, and product quantization with minimal overhead.
- **Architectural Reality**: It is an **in-memory library**, not a database. It does not manage real-time CRUD, network concurrency, or replication.
- **LangChain Integration**: `langchain_community.vectorstores.FAISS` serializes indices directly to `.faiss` and `.pkl` disk files.
""")

st.code("""from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Build FAISS index in-memory
vectorstore = FAISS.from_texts(
    ["Server uptime SLA is 99.99%", "Security policy mandates MFA"],
    embedding=OpenAIEmbeddings()
)

# Persist to local disk
vectorstore.save_local("faiss_index_store")
# Reload from disk
loaded_store = FAISS.load_local("faiss_index_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)""", language="python")

st.divider()

# --- SECTION 3: CHROMA ---
st.header("3. Chroma", divider="rainbow")

st.markdown("""
**Chroma** is the most popular open-source embedding database for AI application prototyping.
- **Key Strength**: Frictionless zero-configuration setup. Runs embedded inside your Python script backed by DuckDB/SQLite or as a lightweight Docker container.
- **Persistence**: Supports on-disk persistence via a local directory (`persist_directory="./chroma_db"`).
""")

st.code("""from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

chroma_store = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_data"
)""", language="python")

st.divider()

# --- SECTION 4: PINECONE ---
st.header("4. Pinecone", divider="rainbow")

st.markdown("""
**Pinecone** is a fully managed cloud-native vector database SaaS.
- **Key Strength**: Zero operational maintenance. Serverless pricing bills only for read/write compute and vector storage.
- **Production Capabilities**: Live index replication across AWS/GCP regions, metadata filtering, and integrated sparse-dense hybrid search.
""")

st.code("""from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

pinecone_store = PineconeVectorStore.from_existing_index(
    index_name="enterprise-docs",
    embedding=OpenAIEmbeddings()
)""", language="python")

st.divider()

# --- SECTION 5: WEAVIATE ---
st.header("5. Weaviate", divider="rainbow")

st.markdown("""
**Weaviate** is an open-source vector search engine written in Go.
- **Key Strength**: Native multi-modal vectorization, native hybrid search (BM25 + Dense Vectors with Reciprocal Rank Fusion), and rich schema validation.
- **Deployment**: Available as a managed cloud service or self-hosted via Docker/Kubernetes.
""")

st.divider()

# --- SECTION 6: QDRANT ---
st.header("6. Qdrant", divider="rainbow")

st.markdown("""
**Qdrant** is an ultra-high-performance vector search engine written in Rust.
- **Key Strength**: Best-in-class payload-based metadata filtering. Supports scalar and product quantization directly on disk mmap vectors, reducing RAM costs by 75% with sub-10ms latency.
""")

st.code("""from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

qdrant_store = QdrantVectorStore.from_documents(
    docs,
    OpenAIEmbeddings(),
    url="http://localhost:6333",
    collection_name="corporate_kb"
)""", language="python")

st.divider()

# --- SECTION 7: MILVUS ---
st.header("7. Milvus", divider="rainbow")

st.markdown("""
**Milvus** (by the LF AI & Data Foundation) is a distributed cloud-native vector database designed for **billion-scale** deployments.
- **Key Strength**: Decouples storage (S3/MinIO), computing worker nodes, and message queues (Kafka/Pulsar).
- **Scale**: Ideal for enterprise organizations indexing hundreds of millions or billions of multi-modal vectors.
""")

st.divider()

# --- SECTION 8: ELASTICSEARCH ---
st.header("8. Elasticsearch", divider="rainbow")

st.markdown("""
**Elasticsearch** (and OpenSearch) supports dense vector embeddings via Lucene's `dense_vector` field mapping.
- **Key Strength**: Combines traditional full-text BM25 search, regex matching, and HNSW vector search within the exact same query.
""")

st.divider()

# --- SECTION 9: POSTGRESQL / PGVECTOR ---
st.header("9. PostgreSQL / pgvector", divider="rainbow")

st.markdown("""
**pgvector** is an open-source extension that adds vector data types and HNSW indexing directly to PostgreSQL.
- **Key Strength**: The simplest operational architecture. If your stack already runs on PostgreSQL (Supabase, AWS RDS, Neon), you can store vectors alongside user accounts, invoices, and audit logs with strict ACID transactional guarantees.
""")

st.code("""from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

connection = "postgresql+psycopg://postgres:secret@localhost:5432/rag_db"
pg_store = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="employee_records",
    connection=connection,
    use_jsonb=True
)""", language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Unified Multi-Engine Vector Store Adapter & Benchmark Suite", divider="rainbow")

st.markdown("""
The following complete Python program implements a factory adapter that unifies vector store initialization, 
query routing, and benchmarking across diverse storage backends.
""")

st.code("""from typing import Dict, List, Any, Optional

class VectorStoreAdapter:
    \"\"\"Production adapter abstracting vector store initialization.\"\"\"
    def __init__(self, engine_type: str, embedding_model, connection_params: Dict[str, Any]):
        self.engine_type = engine_type.lower()
        self.embeddings = embedding_model
        self.params = connection_params
        self.store = self._initialize_store()

    def _initialize_store(self):
        if self.engine_type == "chroma":
            # In production: from langchain_chroma import Chroma
            return {"type": "Chroma", "persist_dir": self.params.get("dir", "./chroma_db")}
        elif self.engine_type == "pinecone":
            # In production: from langchain_pinecone import PineconeVectorStore
            return {"type": "Pinecone", "index": self.params.get("index_name", "prod-kb")}
        elif self.engine_type == "pgvector":
            # In production: from langchain_postgres import PGVector
            return {"type": "PGVector", "conn": self.params.get("conn_string")}
        elif self.engine_type == "qdrant":
            # In production: from langchain_qdrant import QdrantVectorStore
            return {"type": "Qdrant", "url": self.params.get("url", "http://localhost:6333")}
        else:
            raise ValueError(f"Unsupported vector store engine: {self.engine_type}")

    def execute_search(self, query: str, k: int = 3, filter_dict: Optional[dict] = None):
        print(f"Routing query '{query}' to {self.engine_type.upper()} with filter {filter_dict}...")
        return [{"source": f"doc_{i}", "score": 0.92 - (i * 0.05)} for i in range(k)]""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Vector Store Architecture Comparison & Engine Selector", divider="rainbow")

st.markdown("""
Configure your enterprise constraints (document count, deployment preferences, relational database requirements) 
to evaluate which vector store architecture best satisfies your operational SLA.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Deployment Requirements")
    wb_docs = st.select_slider(
        "Estimated Document Corpus Size",
        options=[1000, 10000, 100000, 1000000, 10000000, 50000000],
        value=100000,
        format_func=lambda x: f"{x:,} documents"
    )

    wb_deploy = st.selectbox(
        "Deployment & Hosting Preference",
        options=["Embedded (Zero Infra)", "Self-Hosted Enterprise VPC", "Managed Cloud SaaS"]
    )

    wb_relational = st.toggle("Require ACID Relational Joins with Existing SQL Tables", value=False)

    btn_recom = st.button("🚀 Analyze Architectural Fit", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Engine Suitability Ranking")

    if btn_recom:
        sim = MockVectorStoreEcosystemSimulator()
        rankings = sim.recommend_engine(wb_docs, wb_deploy, wb_relational)

        st.metric("Top Recommended Engine", rankings[0]["engine"], f"{rankings[0]['fit_score']}% Match")

        for idx, r in enumerate(rankings[:4], 1):
            with st.expander(f"Rank #{idx}: {r['engine']} ({r['fit_score']}% Match)", expanded=(idx == 1)):
                c1, c2 = st.columns(2)
                c1.metric("Typical Latency", f"{r['latency_ms']} ms")
                c2.metric("Monthly Cost Est.", r["cost_est"])
                st.write(f"**Architecture**: `{r['category']}`")
                if r["reasons"]:
                    st.markdown("**Key Drivers:**")
                    for reas in r["reasons"]:
                        st.write(f"- {reas}")
    else:
        st.info("Configure your operational constraints and click **Analyze Architectural Fit** for recommendations.")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("When should an engineering team choose PostgreSQL / pgvector over a dedicated vector database?",
     "Choose pgvector if your team already runs PostgreSQL in production, your dataset is under 5-10 million vectors, and your application requires ACID consistency and relational joins between vector chunks and business tables (e.g. users, permissions, orders). Using pgvector eliminates the operational burden of deploying, securing, backing up, and synchronizing a secondary database cluster."),
    ("What are the architectural limitations of FAISS in a production web application?",
     "FAISS is an in-memory C++ similarity library, not a database server. It lacks real-time multi-client concurrency controls, ACID transaction durability, built-in network RPC interfaces, and dynamic document deletion. Modifying an active FAISS index while servicing queries causes race conditions. It is best suited for offline batch clustering or read-only read replicas."),
    ("How does Qdrant's disk mmap quantization reduce vector database hosting costs?",
     "Unquantized 1536-dimensional float32 vectors require ~6.1 GB of RAM per 1 million records. Qdrant implements scalar quantization (converting float32 to int8) and disk memory-mapping (mmap). Vectors are stored on fast NVMe SSDs and paged into RAM on demand, reducing server RAM requirements by up to 75% while keeping query latency under 10 milliseconds."),
    ("What is the difference between Managed Serverless (Pinecone) and Cloud-Native Distributed (Milvus)?",
     "Pinecone is a closed-source, fully managed SaaS that automatically provisions and scales read/write pods with zero DevOps overhead, charging per read/write unit. Milvus is an open-source, highly distributed cloud-native architecture deployed via Kubernetes. Milvus decouples query nodes, index nodes, and object storage (S3/MinIO), making it suited for organizations indexing 100M+ to billions of vectors behind strict VPC firewalls."),
    ("How does Weaviate implement native Hybrid Search?",
     "Weaviate executes traditional BM25 keyword search and dense vector search in parallel on every query. It then applies Reciprocal Rank Fusion (RRF) with an alpha tuning parameter (from 0.0 for pure BM25 to 1.0 for pure vector search) to merge the candidate lists, delivering high relevance across both exact keyword lookups and conceptual semantic queries.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Multi-Worker Race Conditions in SQLite-Backed Chroma. Running multiple gunicorn/uvicorn workers against an embedded local Chroma directory creates SQLite write lock conflicts ('database is locked') under concurrent writes. For multi-process servers, deploy Chroma in Client-Server Docker mode.")
    st.error("**Trap 2 (Breaking Error)**: Unindexed Vector Scans in pgvector. Creating a pgvector column without explicitly executing 'CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)' forces PostgreSQL to execute a sequential table scan on every search, causing query timeouts.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Idle Managed Vector DB Cluster Costs. Provisioning dedicated multi-node vector database clusters for low-traffic internal tools wastes thousands of dollars per month. Use serverless consumption-based tiers or pgvector.")
    st.warning("**Trap 4 (Quality Warning)**: Deserializing Untrusted FAISS Indices. Loading local FAISS pickle files using allow_dangerous_deserialization=True on untrusted inputs exposes the application to arbitrary code execution (RCE). Validate file signatures.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: pgvector"):
        st.markdown("**PostgreSQL extension adding HNSW vector indexing with full relational joins and ACID compliance.**")
    with st.expander("🎴 Flashcard 2: Chroma Embedded"):
        st.markdown("**Lightweight embedded datastore ideal for local development, scripts, and prototyping.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Qdrant Engine"):
        st.markdown("**Rust-based vector DB featuring disk mmap quantization and advanced payload filtering.**")
    with st.expander("🎴 Flashcard 4: Milvus Scale"):
        st.markdown("**Distributed Kubernetes-native architecture built for 100M+ to billion-scale vector workloads.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between FAISS, Chroma, Pinecone, and dedicated vector engines.", value=True)
chk2 = st.checkbox("I know when to deploy PostgreSQL / pgvector instead of a dedicated vector database cluster.", value=True)
chk3 = st.checkbox("I understand the trade-offs of embedded storage vs managed cloud serverless infrastructure.", value=True)
chk4 = st.checkbox("I can articulate how Qdrant and Weaviate implement payload filtering and native hybrid search.", value=True)
chk5 = st.checkbox("I know how to avoid SQLite write lock conflicts when deploying embedded vector stores in production.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 vector store integration interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 14.2 Vector Store Integrations!")
