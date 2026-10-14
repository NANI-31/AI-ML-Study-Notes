import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="13.3 Popular Vector Databases - Embeddings Curriculum",
    page_icon="🏆",
    layout="wide"
)

# Title & Overview
st.title("13.3 Popular Vector Databases: ChromaDB, FAISS, Pinecone, Qdrant, Weaviate, & Milvus")
st.caption("Chapter 13: Embeddings and Vector Databases | Domain: 3. Vector Embeddings")
st.markdown("""
The modern vector database ecosystem spans multiple deployment paradigms: from embedded in-memory libraries running inside a Python script, to distributed cloud-native engines indexing billions of vectors across Kubernetes clusters.

Selecting the right vector database for your application requires balancing **four primary engineering dimensions**:
1. **Deployment Architecture**: In-Process (ChromaDB, FAISS) vs. Self-Hosted Docker Microservices (Qdrant, Weaviate, Milvus) vs. Fully Managed Serverless Cloud (Pinecone, Zilliz).
2. **Dataset Scale**: Thousands of prototypes vs. Tens of millions of enterprise documents vs. Billions of multimodal web vectors.
3. **Filtering Capabilities**: Basic post-filtering vs. Advanced single-stage payload bitmask filtering.
4. **DevOps Overhead & Operational Cost**: Self-managed operational burden vs. Serverless usage-based billing.

This lesson explores the architectural strengths, tradeoffs, and decision criteria across the **Top 6 Vector Databases**.
""")

st.divider()

# Section 1: The Top 6 Vector Database Architectures
st.header("1. The Top 6 Vector Database Architectures", divider="rainbow")

col_vdb1, col_vdb2 = st.columns([1, 1])

with col_vdb1:
    st.markdown("""
    ### Architectural Profiles:
    * **1. ChromaDB (Embedded & In-Process)**:
      * **Core**: Lightweight open-source database running in-process in Python. Backed by SQLite and DuckDB.
      * **Best For**: Rapid local prototyping, developer experiments, and desktop AI apps.
    * **2. Meta FAISS (High-Performance Library)**:
      * **Core**: C++ similarity library developed by Facebook AI Research. Not a full database (lacks dynamic upserts and multi-user networking).
      * **Best For**: Custom algorithmic pipelines, GPU-accelerated batch clustering, and edge inference.
    * **3. Pinecone (Serverless Cloud SaaS)**:
      * **Core**: Proprietary, fully managed cloud database. Zero infrastructure management, live scaling, pay-per-read/write.
      * **Best For**: Cloud-native production teams requiring hands-off operations with instant autoscaling.
    * **4. Qdrant (Rust-Powered Performance)**:
      * **Core**: Open-source database written in Rust. Features advanced single-stage payload filtering, scalar quantization, and distributed clustering.
      * **Best For**: High-throughput microservices, strict low-latency SLAs, and complex metadata filters.
    * **5. Weaviate (Modular Semantic Engine)**:
      * **Core**: Written in Go. Features native multi-modal modules, GraphQL APIs, and built-in hybrid search.
      * **Best For**: Complex multi-modal data pipelines and rich semantic object graphs.
    * **6. Milvus (Hyper-Scale Distributed)**:
      * **Core**: Cloud-native distributed engine with decoupled storage and compute. Scales to billions of vectors across Kubernetes.
      * **Best For**: Massive enterprise datasets (100M+ to Billions) with dedicated platform engineering teams.
    """)

with col_vdb2:
    st.markdown("### Vector Database Deployment Taxonomy")
    tax_dot = """
    digraph VDBTaxonomy {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Root [label="Vector Database Ecosystem", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Embedded [label="Embedded / In-Process\\nChromaDB | FAISS\\nMinimal Setup", fillcolor="#FEF3C7"];
        SelfHosted [label="Self-Hosted Microservices\\nQdrant (Rust) | Weaviate (Go)\\nContainerized Control", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        ManagedCloud [label="Fully Managed Cloud SaaS\\nPinecone | Zilliz Cloud\\nZero DevOps Burden", fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];
        Distributed [label="Hyper-Scale Distributed\\nMilvus (Kubernetes)\\nBillions of Vectors", fillcolor="#FCE7F3", color="#EC4899"];

        Root -> Embedded;
        Root -> SelfHosted;
        Root -> ManagedCloud;
        Root -> Distributed;
    }
    """
    st.graphviz_chart(tax_dot, use_container_width=False)

st.markdown("### Master Decision Matrix: Top 6 Vector Databases")
st.markdown("""
| Vector Database | Language / Engine | Deployment Model | Scaling Ceiling | Metadata Filtering | Standout Advantage | Primary Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ChromaDB** | Python / C++ | Embedded / Client-Server | ~500k vectors | Basic equality & ranges | Zero setup; installs via `pip install chromadb` | Limited horizontal clustering |
| **FAISS** | C++ / CUDA | In-memory library | Millions (GPU-bound) | Manual index partitioning | Rawest GPU throughput; cutting-edge quantization | Library only (no API server or WAL) |
| **Pinecone** | Proprietary | Managed Cloud Serverless | 100M+ vectors | Single-stage metadata | Zero DevOps; serverless consumption pricing | Closed source; vendor lock-in |
| **Qdrant** | **Rust** | Docker / Cloud / Embedded | 100M+ vectors | **Advanced bitmask filters** | Blazing speed; minimal RAM via payload quantization | Moderate self-hosted cluster complexity |
| **Weaviate** | **Go** | Docker / Kubernetes / Cloud | 50M+ vectors | GraphQL & REST filter | Native hybrid search & multimodal modules | High RAM footprint per vector |
| **Milvus** | Go / C++ / Python | Distributed Kubernetes | **Billions of vectors** | Partition keys & expressions | Hyper-scale cloud-native distributed architecture | High infrastructure and operational complexity |
""")

st.divider()

# Section 2: Production Code: Architecture Decision Engine
st.header("2. Production Implementation: Database Selector Decision Engine", divider="rainbow")

st.markdown("""
The production utility below implements programmatic decision logic to recommend the optimal vector database based on scale, hosting preference, latency requirements, and operational constraints.
""")

st.code(r'''
from typing import Dict, Any

class VectorDatabaseAdvisor:
    """
    Evaluates enterprise technical constraints to recommend the optimal
    vector database architecture.
    """
    @staticmethod
    def recommend(
        vector_count: int,
        hosting_preference: str, # "local_embedded", "self_hosted_docker", "managed_cloud"
        needs_gpu: bool = False,
        requires_strict_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Executes decision rules to select database engine.
        """
        if hosting_preference == "local_embedded":
            if needs_gpu or vector_count > 1_000_000:
                return {
                    "primary": "Meta FAISS",
                    "category": "High-Performance In-Memory Library",
                    "rationale": "FAISS provides native GPU acceleration and state-of-the-art IVF-PQ quantization for massive local datasets."
                }
            else:
                return {
                    "primary": "ChromaDB",
                    "category": "Embedded Python Database",
                    "rationale": "ChromaDB offers zero-configuration local persistence with built-in SQLite metadata storage."
                }

        elif hosting_preference == "managed_cloud":
            return {
                "primary": "Pinecone (or Zilliz Cloud)",
                "category": "Serverless Cloud SaaS",
                "rationale": "Pinecone eliminates infrastructure maintenance, delivering automatic sharding and pay-as-you-go serverless billing."
            }

        else: # self_hosted_docker
            if vector_count >= 50_000_000:
                return {
                    "primary": "Milvus",
                    "category": "Distributed Cloud-Native Engine",
                    "rationale": "Milvus's decoupled compute-and-storage architecture scales across Kubernetes clusters to handle billions of records."
                }
            elif requires_strict_metadata:
                return {
                    "primary": "Qdrant",
                    "category": "High-Performance Rust Database",
                    "rationale": "Qdrant provides industry-leading single-stage payload filtering with blazing-fast Rust execution and payload quantization."
                }
            else:
                return {
                    "primary": "Weaviate",
                    "category": "Modular Semantic Search Engine",
                    "rationale": "Weaviate excels at hybrid search (BM25 + vector) and complex semantic graphs with rich GraphQL APIs."
                }
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveAdvisorWorkbench:
    def __init__(self):
        self.profiles = {
            "ChromaDB": {
                "desc": "Lightweight embedded Python database. Perfect for fast iteration, local notebooks, and prototypes.",
                "setup_time": "5 minutes",
                "ops_burden": "None (In-process)",
                "sweet_spot": "1,000 to 250,000 documents"
            },
            "FAISS": {
                "desc": "C++/CUDA similarity library. Unrivaled for custom algorithmic pipelines and GPU batch operations.",
                "setup_time": "30 minutes",
                "ops_burden": "Custom Engineering",
                "sweet_spot": "100,000 to 10,000,000 (GPU batch)"
            },
            "Pinecone": {
                "desc": "Fully managed cloud serverless service. Zero maintenance with instant autoscaling.",
                "setup_time": "10 minutes (API Key)",
                "ops_burden": "Zero (Managed SaaS)",
                "sweet_spot": "100,000 to 100,000,000+"
            },
            "Qdrant": {
                "desc": "Rust-powered high-concurrency database. Exceptional payload filtering and scalar quantization.",
                "setup_time": "15 minutes (Docker)",
                "ops_burden": "Low-Moderate (Single container or K8s)",
                "sweet_spot": "50,000 to 50,000,000"
            },
            "Weaviate": {
                "desc": "Go-based modular engine with native hybrid search, GraphQL, and multi-modal vectorizers.",
                "setup_time": "20 minutes (Docker)",
                "ops_burden": "Moderate (Containerized)",
                "sweet_spot": "50,000 to 25,000,000"
            },
            "Milvus": {
                "desc": "Distributed cloud-native engine for hyper-scale datasets with decoupled storage and compute.",
                "setup_time": "1-2 days (Kubernetes / Helm)",
                "ops_burden": "High (Dedicated Platform Team)",
                "sweet_spot": "50,000,000 to Billions"
            }
        }

    def recommend(self, scale_tier: str, host_mode: str, filter_priority: bool, gpu_req: bool) -> str:
        if host_mode == "Local In-Process (Python Script / Desktop App)":
            if gpu_req or "10M" in scale_tier:
                return "FAISS"
            return "ChromaDB"
        elif host_mode == "Fully Managed Cloud SaaS (Zero DevOps)":
            return "Pinecone"
        else: # Self-Hosted Docker / Kubernetes
            if "50M+" in scale_tier or "Billions" in scale_tier:
                return "Milvus"
            elif filter_priority:
                return "Qdrant"
            else:
                return "Weaviate"

st.divider()

# Section 3: Interactive Workbench: Vector Database Selector & Tradeoff Engine
st.header("3. Interactive Workbench: Vector Database Selector & Tradeoff Engine", divider="rainbow")

st.markdown("""
Configure your application's technical requirements below.
The decision engine evaluates **Scale**, **Hosting Model**, **Filtering Needs**, and **Hardware Acceleration** to recommend the optimal vector database for your production stack.
""")

adv = InteractiveAdvisorWorkbench()

col_adv_l, col_adv_r = st.columns([1, 1])

with col_adv_l:
    st.subheader("1. System Requirements")
    scale_opt = st.selectbox(
        "Estimated Corpus Volume:",
        [
            "Prototyping / Small App (1k - 100k vectors)",
            "Enterprise Production (100k - 10M vectors)",
            "Hyper-Scale Enterprise (50M+ to Billions of vectors)"
        ]
    )

    host_opt = st.selectbox(
        "Hosting & Infrastructure Model:",
        [
            "Local In-Process (Python Script / Desktop App)",
            "Self-Hosted Microservice (Docker / Kubernetes on VPC)",
            "Fully Managed Cloud SaaS (Zero DevOps)"
        ]
    )

    col_chk1, col_chk2 = st.columns(2)
    require_filters = col_chk1.checkbox("Strict Metadata Filtering Required", value=True)
    require_gpu = col_chk2.checkbox("Requires GPU Acceleration", value=False)

    rec_db = adv.recommend(scale_opt, host_opt, require_filters, require_gpu)
    db_profile = adv.profiles[rec_db]

with col_adv_r:
    st.subheader("2. Recommended Architecture")
    st.success(f"🏆 **Primary Recommendation**: **{rec_db}**")
    st.markdown(f"**Profile**: {db_profile['desc']}")

    m1, m2 = st.columns(2)
    m1.metric("Deployment Setup Time", db_profile["setup_time"])
    m2.metric("Operational Burden", db_profile["ops_burden"])

    st.info(f"🎯 **Architectural Sweet Spot**: {db_profile['sweet_spot']}")

    st.markdown("#### Architectural Tradeoffs")
    if rec_db == "ChromaDB":
        st.caption("• **Advantage**: Instant installation; seamless integration with LangChain and LlamaIndex.\n• **Constraint**: Not designed for distributed multi-node clustering.")
    elif rec_db == "Qdrant":
        st.caption("• **Advantage**: High-concurrency Rust engine; best-in-class payload filtering and quantization.\n• **Constraint**: Requires container orchestration (Docker/K8s) if self-hosted.")
    elif rec_db == "Pinecone":
        st.caption("• **Advantage**: Fully managed serverless scalability; zero infrastructure operations.\n• **Constraint**: Cloud-only proprietary service; ongoing consumption-based pricing.")
    elif rec_db == "FAISS":
        st.caption("• **Advantage**: Pure algorithmic speed and cutting-edge GPU quantization.\n• **Constraint**: Raw library lacking dynamic upserts, metadata indexes, or network APIs.")
    elif rec_db == "Milvus":
        st.caption("• **Advantage**: Scales to billions of vectors with decoupled compute and MinIO storage.\n• **Constraint**: Complex Kubernetes deployment and high cluster maintenance.")
    else:
        st.caption("• **Advantage**: Native hybrid search and rich GraphQL ecosystem.\n• **Constraint**: Higher memory footprint per node compared to Rust-based engines.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Database Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Meta FAISS categorized as an 'algorithmic library' rather than a 'full vector database'?**
    * **Answer**:
      * A full database provides: networking APIs (HTTP/gRPC), multi-user authentication, Write-Ahead Logging (WAL) for durability, real-time incremental CRUD operations (Create, Read, Update, Delete), and metadata filtering.
      * **FAISS** is a low-level C++ library that builds and queries vector index structures in memory.
      * FAISS does **not** provide a network server, does not store document text payloads, does not provide ACID crash recovery, and requires building custom wrapper infrastructure to handle incremental updates.
      * Full vector databases (such as ChromaDB, Qdrant, and Milvus) often use FAISS or native HNSW internally as their underlying indexing engine.

    **Q2: What is Product Quantization (PQ) and how does it enable storing millions of vectors in RAM?**
    * **Answer**:
      * Uncompressed 768-d float32 vectors require $768 \\times 4 = 3,072$ bytes per vector.
      * **Product Quantization (PQ)**:
        1. Slices the 768-dimensional vector into $M$ smaller sub-vectors (e.g. $M = 96$ sub-vectors of dimension 8).
        2. Clusters each sub-vector space into 256 centroids using K-Means.
        3. Replaces each sub-vector with an 8-bit centroid index (`uint8`).
      * The vector is compressed from 3,072 bytes down to **96 bytes**—a **32x memory reduction**!
      * This allows indexing 100 million vectors in a single 32 GB RAM server with less than 2% recall degradation.

    **Q3: How do Serverless vector databases (e.g. Pinecone Serverless) decouple compute from storage?**
    * **Answer**:
      * Traditional vector databases keep all vectors and HNSW graph edges in high-cost RAM.
      * **Serverless Vector Architecture**:
        * Stores the master index and raw vectors in inexpensive cloud object storage (Amazon S3 / Google Cloud Storage).
        * Ephemeral compute worker nodes load only the hot index subsets (e.g. DiskANN / compressed navigation clusters) into memory.
        * Customers pay strictly for the queries executed and raw storage stored, eliminating the cost of idle 24/7 GPU/RAM instances.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Deploying ChromaDB for Multi-Node Kubernetes Clusters**:
      * *Issue*: Attempting to scale embedded ChromaDB across multiple Kubernetes pods with shared persistent volumes.
      * *Consequence*: SQLite file lock collisions and corrupted index segments.
      * *Fix*: Deploy client-server vector databases designed for clustering (Qdrant, Milvus, or Pinecone) for multi-pod architectures.

    * **Trap 2: Ignoring Data Sovereignty and Compliance in Cloud SaaS**:
      * *Issue*: Sending sensitive healthcare (HIPAA) or financial vectors to a multi-tenant cloud vector database without evaluating residency.
      * *Consequence*: Potential regulatory violations.
      * *Fix*: Deploy self-hosted open-source databases (Qdrant or Milvus) within your private VPC, or use dedicated single-tenant VPC endpoints.

    * **Trap 3: Underestimating Indexing RAM Requirements for HNSW**:
      * *Issue*: Sizing a vector database node purely for raw vector size without accounting for HNSW graph edges.
      * *Consequence*: HNSW graph pointers (with connectivity $M=32$) require an additional **1.5x to 2x memory** on top of the raw vector floats, triggering out-of-memory crashes during index construction!
      * *Fix*: Allocate at least $2.5 \\times$ the raw vector byte size in server RAM, or enable Scalar Quantization.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Embedded Vector Database**
      * *Definition*: A database (like ChromaDB) that runs directly inside the application process memory without requiring a separate server container.
    * **Flashcard 2: Product Quantization (PQ)**
      * *Definition*: A lossy vector compression technique that decomposes vectors into sub-vectors and maps them to centroid codes, reducing RAM by 8x-32x.
    * **Flashcard 3: Serverless Vector Database**
      * *Definition*: Cloud database architecture decoupling compute from object storage, billing strictly based on queries and storage volume.
    * **Flashcard 4: HNSW Memory Overhead**
      * *Definition*: The extra RAM (typically 1.5x-2x vector size) consumed by bidirectional graph edges to support logarithmic nearest-neighbor search.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Popular Vector Databases:

- [ ] I can describe the architectural profiles and best use cases for ChromaDB, FAISS, Pinecone, Qdrant, Weaviate, and Milvus.
- [ ] I understand why FAISS is classified as an algorithmic library rather than a full vector database.
- [ ] I understand how Product Quantization (PQ) achieves 16x-32x memory compression for large vector indexes.
- [ ] I know how Serverless vector databases decouple storage from compute using cloud object stores.
- [ ] I can calculate the RAM overhead of HNSW graph edges when provisioning production database infrastructure.
""")
