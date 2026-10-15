import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="11.1 Storage Concepts - Vector DB Curriculum",
    page_icon="💾",
    layout="wide"
)

# Title & Overview
st.title("11.1 Vector Database Storage Concepts: Memory, Disk, Embedded & Cloud")
st.caption("Chapter 11: Vector Database Storage | Domain: 4. Vector Database")
st.markdown("""
Where and how vector embeddings are physically stored dictates **system latency, recovery time, infrastructure cost, and data durability**.

Storing high-dimensional float32 vectors differs fundamentally from storing relational rows:
* **Vectors are Massive**: 10 million text embeddings (1536-dim) consume **~61.4 GB of raw binary data**, plus 15-25 GB of graph index pointers.
* **Access Patterns are Random**: HNSW graph navigation jumps unpredictably across high-dimensional space, stressing memory buses and SSD read IOPS.
* **Persistence Matters**: Ephemeral prototypes risk data loss on server restarts, while cloud-native vector engines separate compute from storage.

In this lesson, we explore the core storage hierarchy: **In-Memory vs. Persistent Disk vs. Memory-Mapped Files**, examine the operational 
differences between **Local Embedded Engines** and **Cloud-Managed Distributed Systems**, and experiment with an interactive storage architecture simulator.
""")

st.divider()

# Section 1: The Vector Storage Hierarchy
st.header("1. The Vector Storage Hierarchy", divider="rainbow")

col_hier1, col_hier2 = col_hier1, col_hier2 = st.columns([1.1, 0.9])

with col_hier1:
    st.markdown("""
    ### Three Physical Storage Tiers
    
    1. **In-Memory Storage (RAM / CPU Cache)**:
       * Both the vectors and the HNSW graph indices reside in heap memory (`malloc`).
       * **Latency**: 0.5 - 2 milliseconds (ultra-fast).
       * **Durability**: **Zero (Volatile)**. Any process exit, crash, or server reboot destroys all data unless flushed.
       * **Chroma Equivalent**: `chromadb.EphemeralClient()` or `chromadb.Client()`.
    
    2. **Persistent Local Disk Storage (SQLite + Binary Index Files)**:
       * Documents, IDs, and metadata are written to an embedded ACID relational database (`chroma.sqlite3`).
       * Vector index graphs and binary embedding arrays are serialized to dedicated index segment files.
       * **Latency**: 2 - 8 milliseconds (first queries warm the OS page cache; subsequent queries run near RAM speed).
       * **Chroma Equivalent**: `chromadb.PersistentClient(path="./my_chroma_db")`.
    
    3. **Disk-Native / Memory-Mapped Storage (mmap & NVMe)**:
       * Uses operating system virtual memory mapping (`mmap`). The file stays on high-speed NVMe SSDs while the OS kernel automatically pages hot clusters into RAM.
       * Enables querying datasets 2x-5x larger than physical RAM without crashes.
    """)

with col_hier2:
    st.markdown("### Vector Storage Architecture Blueprint")
    store_dot = """
    digraph StorageHierarchy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        App [label="Application Layer\\nRAG / Search Query", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_mem {
            label="RAM Tier (Active Compute)"; style="dashed"; color="#059669";
            HNSWCache [label="HNSW Graph in RAM\\nUpper Layers + Hot Nodes", fillcolor="#D1FAE5"];
            PageCache [label="OS Page Cache (mmap)\\nHot Vector Chunks", fillcolor="#A7F3D0"];
        }

        subgraph cluster_disk {
            label="Persistent Storage Tier (Disk / SSD)"; style="dashed"; color="#D97706";
            WAL [label="Write-Ahead Log (WAL)\\nCrash Safety & Durability", fillcolor="#FEF3C7"];
            SQLite [label="chroma.sqlite3\\nMetadata, IDs, Documents", fillcolor="#FDE68A"];
            IndexFiles [label="Index Segment Files\\nSerialized Binaries & Edges", fillcolor="#FCD34D"];
        }

        App -> HNSWCache [label="Fast Traversal"];
        HNSWCache -> PageCache;
        App -> WAL [label="Writes / Inserts"];
        WAL -> SQLite;
        WAL -> IndexFiles;
    }
    """
    st.graphviz_chart(store_dot, use_container_width=False)

st.divider()

# Section 2: Local Embedded vs Client-Server vs Cloud-Managed
st.header("2. Local Embedded vs Client-Server vs Cloud-Managed Databases", divider="rainbow")

st.markdown("""
Choosing where the vector database engine executes dictates system scalability and operational overhead:
""")

tab_arch1, tab_arch2, tab_arch3 = st.tabs([
    "1. Local Embedded (ChromaDB / LanceDB)",
    "2. Client-Server (Chroma Server / Milvus / Qdrant)",
    "3. Managed Cloud (Pinecone / Weaviate Cloud / Vertex AI)"
])

with tab_arch1:
    st.markdown("""
    ### Local Embedded Vector Databases
    * **How it Works**: The database engine runs **in-process** inside your Python application (same process ID, direct memory access).
    * **Examples**: ChromaDB (`PersistentClient`), LanceDB, FAISS in-memory.
    * **Advantages**:
      * Zero network latency overhead (function calls rather than HTTP/gRPC round-trips).
      * No server infrastructure, containers, or clusters to manage; stores data in a local directory.
      * Ideal for local development, desktop agents, single-server microservices, and edge computing.
    * **Disadvantages**:
      * Scales vertically only (limited by single host's RAM/disk).
      * Multiple independent Python worker processes cannot easily write concurrently without SQLite locking contention.
    """)

with tab_arch2:
    st.markdown("""
    ### Client-Server Self-Hosted Engines
    * **How it Works**: The vector database runs as a **dedicated standalone service** (Docker container, Kubernetes deployment) exposed via REST or gRPC APIs.
    * **Examples**: Chroma Server (`chroma run --path ...`), Qdrant, Milvus, Weaviate.
    * **Advantages**:
      * Independent scaling: The vector database can be scaled, updated, and provisioned on high-RAM machines independently of web servers.
      * Concurrent multi-client access: Dozens of FastAPI microservices can query and ingest data simultaneously.
    * **Disadvantages**:
      * Adds 1-3ms network serialization latency (JSON/gRPC marshaling).
      * Requires DevOps management: monitoring, disk backups, Kubernetes pod management.
    """)

with tab_arch3:
    st.markdown("""
    ### Managed Cloud Vector Databases
    * **How it Works**: Fully managed serverless SaaS vector infrastructure with automated sharding, replication, and high availability.
    * **Examples**: Pinecone, Weaviate Cloud Services (WCS), Qdrant Cloud, AWS OpenSearch Serverless, Google Vertex AI Vector Search.
    * **Advantages**:
      * Infinite horizontal scalability: Shards billions of vectors across distributed clusters with zero operational burden.
      * High Availability (HA): Multi-region active-active replication, automatic failover, and point-in-time recovery (PITR).
    * **Disadvantages**:
      * Significant operational cost ($0.05 to $0.40+ per GB-month, plus query fees).
      * Data egress latency across the public cloud or VPC peering.
      * Vendor lock-in.
    """)

st.divider()

# Section 3: ChromaDB Storage Initialization Code Reference
st.header("3. ChromaDB Storage Initialization Code Reference", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. In-Memory Ephemeral Storage")
    st.code("""import chromadb

# Ephemeral client lives purely in RAM
client = chromadb.EphemeralClient()

collection = client.create_collection("temporary_session")
collection.add(
    ids=["doc1"],
    documents=["Transient scratchpad text"],
    metadatas=[{"source": "user_session"}]
)

# When the Python script exits, all data is permanently erased.
""", language="python")

with col_code2:
    st.markdown("### 2. Persistent Local Disk Storage")
    st.code("""import chromadb
from chromadb.config import Settings

# Persistent client saves to local directory
client = chromadb.PersistentClient(
    path="./production_data_store",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

collection = client.get_or_create_collection(
    name="persistent_docs",
    metadata={"hnsw:space": "cosine"}
)

# Data survives process restarts and reboots
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Vector Storage Sizing & Cost Estimator
st.header("4. Interactive Workbench: Storage Sizing & Cost Estimator", divider="rainbow")

st.markdown("""
Calculate raw vector disk size, HNSW graph index overhead, SQLite metadata footprint, and compare hosting costs 
across **Local Embedded**, **Self-Hosted Cloud VM**, and **Serverless SaaS**.
""")

col_calc_in1, col_calc_in2 = col_calc_in1, col_calc_in2 = st.columns([1, 1])

with col_calc_in1:
    st.subheader("1. Corpus Volume & Dimensions")
    vec_volume = st.select_slider(
        "Total Number of Embeddings (N):",
        options=[10_000, 100_000, 500_000, 1_000_000, 5_000_000, 20_000_000],
        value=1_000_000,
        format_func=lambda x: f"{x:,} vectors"
    )
    
    vec_dim = st.selectbox(
        "Embedding Model Dimensions (D):",
        [384, 768, 1536, 3072],
        index=2,
        help="384 (MiniLM), 768 (BGE/MPNet), 1536 (OpenAI small), 3072 (OpenAI large)"
    )

with col_calc_in2:
    st.subheader("2. Metadata & Text Chunk Sizing")
    avg_chunk_bytes = st.slider("Average Raw Document Chunk Size (Bytes):", min_value=200, max_value=2000, value=600, step=100)
    avg_meta_bytes = st.slider("Average Metadata JSON Payload Size (Bytes):", min_value=50, max_value=500, value=150, step=50)

# Exact Sizing Math
# Raw vectors: N * D * 4 bytes (float32)
raw_vec_bytes = vec_volume * vec_dim * 4
raw_vec_gb = raw_vec_bytes / (1024**3)

# HNSW Graph: N * M * 2 * 8 bytes (assuming M=32)
hnsw_graph_bytes = vec_volume * (32 * 2 * 8)
hnsw_graph_gb = hnsw_graph_bytes / (1024**3)

# SQLite Metadata + Raw text + Primary keys + B-Trees (~1.4x factor)
sqlite_bytes = vec_volume * (avg_chunk_bytes + avg_meta_bytes + 64) * 1.4
sqlite_gb = sqlite_bytes / (1024**3)

total_disk_gb = raw_vec_gb + hnsw_graph_gb + sqlite_gb
min_ram_rec_gb = raw_vec_gb + hnsw_graph_gb + 2.0  # vectors + graph + OS buffer

st.divider()

col_out_storage, col_out_costs = st.columns([1.2, 0.8])

with col_out_storage:
    st.subheader("Storage Sizing Breakdown")
    df_sizing = pd.DataFrame([
        {"Component": "Raw Float32 Embeddings", "Formula": "N * D * 4 bytes", "Footprint": f"{raw_vec_gb:.2f} GB"},
        {"Component": "HNSW Graph Index (M=32)", "Formula": "N * (M * 2 * 8 bytes)", "Footprint": f"{hnsw_graph_gb:.2f} GB"},
        {"Component": "SQLite Metadata & Documents", "Formula": "N * (Chunk + Meta + BTree)", "Footprint": f"{sqlite_gb:.2f} GB"},
        {"Component": "Total Disk Space Required", "Formula": "Sum of all segments", "Footprint": f"{total_disk_gb:.2f} GB"},
        {"Component": "Minimum Recommended RAM", "Formula": "Vectors + Graph + 2GB buffer", "Footprint": f"{min_ram_rec_gb:.2f} GB"}
    ])
    st.dataframe(df_sizing, use_container_width=True, hide_index=True)

with col_out_costs:
    st.subheader("Monthly Hosting Cost Estimates")
    
    # Cost models
    cost_local = 0.0  # Runs on existing server
    # Self-hosted VM: e.g. AWS r6g.large (16GB RAM) ~$75/mo, r6g.xlarge (32GB) ~$150/mo, etc.
    if min_ram_rec_gb <= 14:
        cost_vm = 75.0
        vm_type = "AWS r6g.large (16 GB RAM)"
    elif min_ram_rec_gb <= 30:
        cost_vm = 150.0
        vm_type = "AWS r6g.xlarge (32 GB RAM)"
    elif min_ram_rec_gb <= 60:
        cost_vm = 300.0
        vm_type = "AWS r6g.2xlarge (64 GB RAM)"
    else:
        cost_vm = 600.0
        vm_type = "AWS r6g.4xlarge (128 GB RAM)"
        
    # SaaS Pinecone: ~$0.096 per GB-hour or standard pod rate (~$70/pod of 25GB)
    pods_needed = max(1, int(np.ceil(total_disk_gb / 25.0)))
    cost_saas = pods_needed * 70.0 + (vec_volume / 1_000_000 * 15.0)

    st.metric(label="Local Embedded (ChromaDB)", value="$0 / month", delta="Zero extra hosting cost")
    st.metric(label=f"Self-Hosted VM ({vm_type})", value=f"${cost_vm:.0f} / month", delta="Predictable fixed cost")
    st.metric(label="Managed Cloud SaaS (Pinecone / WCS)", value=f"~${cost_saas:.0f} / month", delta=f"{pods_needed} pod(s) required")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What happens if an in-memory vector database crashes during a write operation?"):
    st.markdown("""
    **Answer**:
    * In an **In-Memory database** (`EphemeralClient` in ChromaDB), any crash results in **total data loss**; all unpersisted embeddings, indices, and documents in RAM vanish instantly.
    * In a **Persistent database** (`PersistentClient`), crash safety is protected by the **Write-Ahead Log (WAL)**. Changes are appended to an on-disk journal before modifying RAM state. Upon restart, the database replays uncommitted WAL transactions, recovering to the last consistent state.
    """)

with st.expander("Q2: Why does HNSW graph navigation perform poorly on traditional spinning hard drives (HDDs)?"):
    st.markdown("""
    **Answer**:
    HNSW graph traversal relies on **random, non-sequential pointer chasing**:
    * In each hop, the algorithm checks node `v`, reads its `M` neighbor IDs, and fetches vectors at random memory addresses.
    * Spinning hard drives (HDDs) suffer from physical mechanical seek times (5-10ms per seek). Traversing 30 graph hops would take **150ms to 300ms** on an HDD!
    * High-performance vector storage requires **high-speed RAM or NVMe SSDs** with high random read IOPS (500k+ IOPS) to achieve sub-5ms latency.
    """)

with st.expander("Q3: What are the trade-offs of using memory-mapped files (mmap) for vector index persistence?"):
    st.markdown("""
    **Answer**:
    * **Advantages**:
      1. Zero-copy access: The kernel pages binary index data directly from NVMe SSDs into user space without duplicate allocations.
      2. Memory sharing: Multiple worker processes can share the exact same physical RAM pages for read-only queries.
      3. Automatic eviction: The OS kernel dynamically reclaims memory under system pressure without application crashes.
    * **Disadvantages**:
      1. Unpredictable latency: If a query visits cold vectors not present in the page cache, a major page fault triggers disk read I/O, spiking latency from 2ms to 30ms+.
    """)

with st.expander("Q4: When should an enterprise transition from ChromaDB embedded mode to client-server mode?"):
    st.markdown("""
    **Answer**:
    Transition when:
    1. **Multiple Microservices Need Write Access**: Embedded SQLite locks the entire database file during writes, creating bottleneck contention if multiple web workers or Celery tasks insert vectors simultaneously.
    2. **Independent Resource Scaling**: The vector index requires high RAM (e.g. 64GB), while web frontend servers require high CPU but low RAM. Separating them decouples scaling tiers.
    3. **Containerized Deployments**: When web apps scale horizontally across multiple Kubernetes pods, an embedded database would result in each pod having an isolated, out-of-sync copy of the data.
    """)

with st.expander("Q5: How does data durability differ between SQLite metadata storage and HNSW index storage in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **Metadata Storage**: SQLite guarantees full ACID compliance with WAL mode. Once an insert transaction commits, document text, IDs, and metadata are guaranteed to survive power failure.
    * **HNSW Index Storage**: Serializing graph edges is computationally expensive. Chroma periodically writes immutable index segments to disk. In the event of an abrupt power kill, Chroma re-syncs the HNSW graph against the authoritative SQLite record table during startup.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Deploying EphemeralClient to Production**  
    Using `chromadb.Client()` in production means any container restart, deployment rollout, or crash wipes the entire vector store clean. Always use `PersistentClient(path=...)`.
    """)
    st.warning("""
    **Trap 2: Placing SQLite Vector Storage on Network Filesystems (NFS)**  
    Hosting ChromaDB persistent directories on NFS, SMB, or AWS EFS causes severe SQLite locking errors and corrupted WAL journals due to broken file locking semantics. Always use local NVMe or EBS volumes.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Under-Sizing RAM by Ignoring Graph Pointers**  
    Calculating server RAM based only on vector float bytes (`N * D * 4`) leads to Out-Of-Memory (OOM) crashes because HNSW graph pointers add 25% to 50% more memory.
    """)
    st.warning("""
    **Trap 4: Multi-Process Writes to Embedded SQLite**  
    Running Gunicorn with 8 workers all instantiating `PersistentClient` on the same directory will trigger `sqlite3.OperationalError: database is locked`. Use Chroma Client-Server mode.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is EphemeralClient?"):
        st.markdown("""
        **Ephemeral Storage**:  
        In-memory only; fast for scratchpads and testing, but completely destroyed upon process exit.
        """)
    with st.expander("🗂️ Flashcard: How does PersistentClient store data?"):
        st.markdown("""
        **Persistent Storage**:  
        Combines an embedded SQLite database (`chroma.sqlite3`) for metadata with serialized binary HNSW index segment files.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why are NVMe SSDs required for vector DBs?"):
        st.markdown("""
        **Random Read IOPS**:  
        HNSW graph traversal chases pointers randomly across space; mechanical HDD seek times (10ms) are far too slow.
        """)
    with st.expander("🗂️ Flashcard: What is the main limitation of embedded mode?"):
        st.markdown("""
        **Concurrency**:  
        SQLite file locking prevents high-concurrency multi-process writes. Client-server architecture is required.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the physical trade-offs between In-Memory, Persistent Disk, and Memory-Mapped storage.", value=True)
chk2 = st.checkbox("I know how to initialize In-Memory (`EphemeralClient`) and Persistent (`PersistentClient`) storage in Chroma.", value=True)
chk3 = st.checkbox("I can accurately calculate storage and RAM requirements for millions of vectors and graph edges.", value=True)
chk4 = st.checkbox("I understand why vector graph traversal requires high random read IOPS (NVMe SSDs).", value=True)
chk5 = st.checkbox("I know when to transition from embedded single-process storage to client-server or managed cloud.", value=True)
chk6 = st.checkbox("I understand the ACID durability mechanisms of SQLite WAL and HNSW index segments.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Storage Concepts and are ready for Chapter 11.2: Persistence & Data Durability!")
