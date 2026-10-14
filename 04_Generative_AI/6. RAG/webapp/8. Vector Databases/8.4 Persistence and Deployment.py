import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class PersistenceRecoverySimulator:
    """
    Simulates crash recovery, cold-start index reconstruction,
    snapshot exports, and WAL replay across storage modalities.
    """
    MODALITIES = {
        "In-Memory Only": {
            "wal_enabled": False,
            "cold_start_sec": 0.1,
            "data_loss_risk": "100% on Process Crash",
            "ram_overhead": "100% in RAM",
            "throughput_qps": 8500
        },
        "WAL + Periodic Snapshots": {
            "wal_enabled": True,
            "cold_start_sec": 4.5,
            "data_loss_risk": "Near Zero (Last un-flushed transaction only)",
            "ram_overhead": "100% in RAM + Disk WAL",
            "throughput_qps": 6200
        },
        "Memory-Mapped (MMAP) on NVMe": {
            "wal_enabled": True,
            "cold_start_sec": 0.8,
            "data_loss_risk": "Zero (OS Page Cache sync)",
            "ram_overhead": "Partial (Pages cached on-demand)",
            "throughput_qps": 4800
        },
        "Cloud Distributed Cluster": {
            "wal_enabled": True,
            "cold_start_sec": 12.0,
            "data_loss_risk": "Zero (Multi-AZ Replicated S3)",
            "ram_overhead": "Tiered (S3 Storage + Query Node RAM)",
            "throughput_qps": 15000
        }
    }

    @classmethod
    def simulate_recovery(cls, modality: str, vector_count: int) -> Dict[str, Any]:
        info = cls.MODALITIES[modality]
        # Scaling cold-start time with vector count
        scale_factor = vector_count / 100000.0
        cold_start = round(info["cold_start_sec"] * (1.0 + scale_factor), 2)
        
        return {
            "Modality": modality,
            "Estimated Cold-Start Recovery Time": f"{cold_start} seconds",
            "Durability Guarantee": info["data_loss_risk"],
            "RAM Strategy": info["ram_overhead"],
            "Peak Throughput": f"{info['throughput_qps']} QPS",
            "WAL Status": "Active (Synchronous Append)" if info["wal_enabled"] else "Disabled"
        }


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="8.4 Persistence & Deployment - RAG Master Curriculum",
    page_icon="💾",
    layout="wide"
)

st.title("8.4 Persistence and Deployment: In-Memory vs. Persistent Stores, Cloud Topologies, and Disaster Recovery")
st.caption("Chapter 8: Vector Databases | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Deploying a vector database in an enterprise production environment introduces architectural challenges 
beyond mathematical nearest-neighbor search. Systems engineers must decide where vector coordinates and index 
structures reside across **volatile RAM**, **memory-mapped NVMe storage**, and **cloud object stores**.

Furthermore, production systems demand clear disaster recovery protocols: write-ahead logging (WAL), 
point-in-time snapshots, multi-zone replication, and deterministic cold-start re-indexing.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Vector Persistence, WAL, and Deployment Topologies", divider="rainbow")
st.markdown("""
The blueprint below illustrates the storage hierarchy of a modern persistent vector database, 
tracing mutations from client calls through the Write-Ahead Log (WAL) to disk segments and backup snapshots.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    persist_bp_dot = """
    digraph PersistenceTopology {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_tier_memory {
            label = "1. Active Volatile Memory Tier (RAM)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            client_write [label="Incoming Write Operations\\n(Upsert / Delete / Update)", fillcolor="#0F172A", color="#38BDF8"];
            mem_buffer [label="Active MemTable Buffer\\nMutable In-Memory Vectors", fillcolor="#0F172A", color="#10B981"];
            hnsw_ram [label="Live In-Memory HNSW Graph\\nFast Low-Latency Traversal", fillcolor="#0F172A", color="#8B5CF6"];

            client_write -> mem_buffer;
            mem_buffer -> hnsw_ram [label="Link Nodes"];
        }

        subgraph cluster_tier_disk {
            label = "2. Persistent Local Storage Tier (NVMe / MMAP)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            wal_disk [label="Write-Ahead Log (WAL)\\nSequential Append-Only Log", fillcolor="#334155", color="#F59E0B"];
            sealed_seg [label="Sealed Disk Segments\\nImmutable Vectors + Payloads", fillcolor="#334155", color="#10B981"];
            mmap_cache [label="Memory-Mapped (MMAP) Files\\nKernel Managed Page Cache", fillcolor="#334155", color="#06B6D4"];

            client_write -> wal_disk [label="Sync fsync()"];
            mem_buffer -> sealed_seg [label="Periodic Flush"];
            sealed_seg -> mmap_cache [label="Virtual Memory Page"];
        }

        subgraph cluster_tier_cloud {
            label = "3. Cold Storage & Disaster Recovery Tier (S3 / GCS)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            snapshot_mgr [label="Snapshot Backup Worker\\nPeriodic Tarball / Chunked Sync", fillcolor="#0F172A", color="#EC4899"];
            cloud_backup [label="Cloud Object Storage (S3 / GCS)\\nPoint-in-Time Recovery (PITR)", fillcolor="#0F172A", color="#10B981"];

            sealed_seg -> snapshot_mgr [label="Archive"];
            snapshot_mgr -> cloud_backup [label="Upload Snapshot"];
        }
    }
    """
    st.graphviz_chart(persist_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production engines like **Qdrant** and **Chroma Persistent** store sealed segments on local NVMe drives using **MMAP (Memory Mapping)**. 
Instead of loading 100 GB of vectors into RAM on startup (which could cause a 10-minute server downtime during rolling deployments), 
MMAP lets the Linux kernel page in vector chunks on-demand as queries touch specific sectors. 
Meanwhile, cloud-native services like **Pinecone Serverless** continuously stream sealed index blocks to AWS S3, enabling instant pod restarts and zero-maintenance storage.
""")

st.divider()

# ==============================================================================
# SECTION 2: IN-MEMORY VECTOR STORES
# ==============================================================================

st.header("2. In-Memory Vector Stores", divider="rainbow")
st.markdown("""
In-memory vector stores hold all vector coordinates, graph edges, and document metadata entirely in RAM.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Advantages")
    st.markdown("""
    * **Zero Disk I/O Bottlenecks**:
      Every graph hop and vector distance calculation runs directly in processor cache and system RAM, yielding sub-2ms query latencies.
    * **Maximum Query Throughput (QPS)**:
      Can sustain 5,000 to 20,000 queries per second per host.
    * **Simplicity**:
      No file handles, disk serialization formats, or background compactor lock contention.
    """)

with col2:
    st.subheader("Critical Limitations")
    st.markdown("""
    * **Total Volatility (Data Loss)**:
      A process restart, node crash, or OOM kill completely erases all indexed data unless an external mechanism re-ingests the corpus.
    * **Extreme Hardware Cost**:
      10 million 1536-dimensional vectors with HNSW require ~120 GB to 180 GB of high-frequency RAM, which is significantly more expensive than NVMe storage.
    * **Slow Cold Start**:
      Reconstructing the in-memory graph from raw documents on system reboot can take hours.
    """)

st.divider()

# ==============================================================================
# SECTION 3: PERSISTENT VECTOR STORES
# ==============================================================================

st.header("3. Persistent Vector Stores: WAL, Disk-Backed Segments, and MMAP", divider="rainbow")
st.markdown("""
Persistent vector databases write changes to non-volatile storage, guaranteeing durability and fast cold restarts.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Core Persistence Mechanisms")
    st.markdown("""
    1. **Write-Ahead Logging (WAL)**:
       Every insert, delete, or update is appended to a sequential disk log before acknowledging success to the client. Upon a server crash, replaying the WAL restores state.
    2. **Segmented Architecture**:
       The database maintains an active mutable memory buffer. When the buffer reaches a threshold (e.g., 256 MB), it is sealed, transformed into an immutable disk segment with precomputed HNSW graphs, and flushed.
    3. **Memory-Mapped Files (MMAP)**:
       Uses the OS `mmap()` system call to map disk segment files into the virtual address space.
       The OS pages vectors into RAM on demand, allowing datasets larger than physical RAM without application-level buffer management.
    """)

with col2:
    st.subheader("Storage Hierarchy Comparison")
    tier_df = pd.DataFrame([
        {"Storage Layer", "Latency", "Cost / GB", "Durability"},
        {"System RAM", "0.5 - 2 ms", "High (~$5.00/GB/mo)", "Volatile (Lost on reboot)"},
        {"NVMe SSD (MMAP)", "2 - 6 ms", "Moderate (~$0.25/GB/mo)", "Persistent (Survivable)"},
        {"Cloud Object (S3)", "50 - 150 ms", "Low (~$0.02/GB/mo)", "Durable (99.999999999%)"}
    ])
    st.dataframe(tier_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 4: CLOUD VECTOR DATABASES
# ==============================================================================

st.header("4. Cloud Vector Databases: Managed Distributed Architectures", divider="rainbow")
st.markdown("""
Cloud vector databases (Pinecone, Zilliz Cloud, Qdrant Cloud, Weaviate Cloud) provide fully managed, distributed, multi-tenant vector infrastructure.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Key Distributed Features")
    st.markdown("""
    * **Horizontal Sharding**:
      Collections are automatically divided across multiple worker nodes (shards). Queries execute in parallel across shards and are merged by coordinator nodes.
    * **Multi-Zone High Availability (HA)**:
      Vector replicas are distributed across multiple cloud availability zones (AZs) to withstand datacenter outages.
    * **Serverless Auto-Scaling**:
      Dynamically allocates query capacity based on incoming QPS, scaling down during low-traffic periods to minimize cost.
    * **Managed Security & Compliance**:
      Automated encryption at rest (KMS), SOC2 compliance, role-based access control (RBAC), and private VPC peering.
    """)

with col2:
    st.subheader("Cloud vs. Self-Hosted Decision Framework")
    st.markdown("""
    * **Choose Cloud Managed SaaS When**:
      * Team lacks dedicated Kubernetes database administrators (DBAs).
      * Rapid time-to-market is the primary business metric.
      * Traffic is bursty and requires automatic scaling.
    * **Choose Self-Hosted (On-Premises / VPC) When**:
      * Strict data sovereignty or regulatory compliance forbids external SaaS transmission.
      * Continuous high QPS makes dedicated compute hardware significantly cheaper than serverless pay-per-read billing.
    """)

st.divider()

# ==============================================================================
# SECTION 5: LOCAL VECTOR DATABASES
# ==============================================================================

st.header("5. Local Vector Databases: Embedded & Edge Deployments", divider="rainbow")
st.markdown("""
Local vector stores run in-process or on edge hardware (mobile phones, local workstations, IoT devices).
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Profile")
    st.markdown("""
    * **Embedded Execution**:
      Runs directly within the application process (e.g., Chroma embedded mode, DuckDB with VSS, SQLite with sqlite-vec). No external service dependencies.
    * **Single-File Portability**:
      Entire collections, vectors, and metadata are stored in a single file on disk, making them easy to distribute, clone, or embed in desktop tools.
    * **Zero Network Overhead**:
      Eliminates HTTP/gRPC serialization, achieving minimal latency for single-threaded queries.
    """)

with col2:
    st.subheader("Production Applications")
    st.markdown("""
    * **AI Agents & Local Scratchpads**:
      Storing episodic conversation history for local desktop agents.
    * **Edge Computing & Air-Gapped Environments**:
      Industrial or military environments operating without internet connectivity.
    * **Local Developer Prototyping**:
      Building and testing RAG pipelines locally before deploying to cloud clusters.
    """)

st.divider()

# ==============================================================================
# SECTION 6: BACKUP AND RECOVERY
# ==============================================================================

st.header("6. Backup and Disaster Recovery: Snapshots, Re-indexing, and PITR", divider="rainbow")
st.markdown("""
Maintaining enterprise data resilience requires proactive disaster recovery plans for vector indices.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Disaster Recovery Strategies")
    st.markdown("""
    1. **Storage Snapshots**:
       Freezes immutable disk segments and exports a point-in-time archive to cloud object storage (S3/GCS). Fast recovery without re-embedding.
    2. **Raw Corpus Re-Indexing**:
       Re-embedding all original documents from the primary SQL data lake into a brand-new vector collection. Guaranteed clean state, but incurs substantial embedding API costs and takes hours.
    3. **Continuous WAL Archival (Point-in-Time Recovery - PITR)**:
       Periodically ships sealed WAL segments to cold storage. Enables restoring the database to any exact minute before corruption or accidental deletion.
    """)

with col2:
    st.subheader("Recovery Time Objective (RTO) vs. Cost")
    dr_df = pd.DataFrame([
        {"Recovery Strategy", "Recovery Time (RTO)", "Compute & API Cost", "Recovery Point (RPO)"},
        {"Point-in-Time Snapshot", "< 5 minutes", "Near Zero (Download & Unpack)", "Last scheduled snapshot (e.g. 1 hour)"},
        {"Re-Index from Primary DB", "2 - 12 hours", "High (Full LLM embedding fees)", "Zero (Current primary state)"},
        {"Multi-AZ Live Replica", "< 1 second (Failover)", "High (2x - 3x running infrastructure)", "Zero (Real-time sync)"}
    ])
    st.dataframe(dr_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Persistent Segmented Vector Store with Snapshot Exporter", divider="rainbow")
st.markdown("""
Below is an enterprise persistence manager illustrating how mutable memory buffers, 
simulated disk segments, WAL logging, and snapshot backups operate in production Python.
""")

st.code('''import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

class PersistentVectorEngine:
    """
    Demonstrates write-ahead log (WAL) persistence, segment flushing,
    and snapshot backup creation.
    """
    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension
        self.wal_log: List[Dict[str, Any]] = []
        self.active_memory: Dict[str, np.ndarray] = {}
        self.active_payloads: Dict[str, Dict[str, Any]] = {}
        self.sealed_segments: List[Dict[str, Any]] = []

    def insert_with_wal(self, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Appends mutation to WAL before updating active memory buffer."""
        wal_entry = {
            "op": "UPSERT",
            "id": doc_id,
            "vector": vector,
            "payload": payload,
            "timestamp": time.time()
        }
        # In production: Append synchronously to disk file using fsync()
        self.wal_log.append(wal_entry)

        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        self.active_memory[doc_id] = v / norm if norm > 0 else v
        self.active_payloads[doc_id] = payload

    def flush_segment(self) -> str:
        """Seals the active memory buffer into an immutable persistent segment."""
        if not self.active_memory:
            return "NO_DATA_TO_FLUSH"

        segment_id = f"seg_{len(self.sealed_segments) + 1}_{int(time.time())}"
        segment_data = {
            "segment_id": segment_id,
            "vectors": {k: v.tolist() for k, v in self.active_memory.items()},
            "payloads": dict(self.active_payloads)
        }
        self.sealed_segments.append(segment_data)
        
        # Clear mutable buffer and truncate WAL
        self.active_memory.clear()
        self.active_payloads.clear()
        self.wal_log.clear()
        return segment_id

    def export_snapshot(self) -> str:
        """Packages sealed segments and active state into a persistent archive JSON."""
        snapshot = {
            "collection": self.collection_name,
            "dimension": self.dimension,
            "exported_at": time.time(),
            "sealed_segments": self.sealed_segments,
            "unflushed_wal_entries": len(self.wal_log)
        }
        return json.dumps(snapshot, indent=2)

    def recover_from_crash(self) -> int:
        """Replays WAL entries to reconstruct state following an abrupt process failure."""
        replayed = 0
        for entry in self.wal_log:
            if entry["op"] == "UPSERT":
                v = np.array(entry["vector"], dtype=np.float32)
                norm = np.linalg.norm(v)
                self.active_memory[entry["id"]] = v / norm if norm > 0 else v
                self.active_payloads[entry["id"]] = entry["payload"]
                replayed += 1
        return replayed
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Persistence Failure, Recovery, and Cold-Start Simulator", divider="rainbow")
st.markdown("""
Evaluate how different storage modalities handle process crashes, cold starts, and backup recovery. 
Adjust vector scale and deployment topologies to inspect expected recovery times and durability guarantees.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Deployment & Crash Configuration")
    selected_modality = st.selectbox(
        "Storage Modality",
        list(PersistenceRecoverySimulator.MODALITIES.keys())
    )

    vector_count_val = st.select_slider(
        "Corpus Size (Number of Vectors)",
        options=[10000, 50000, 100000, 500000, 1000000, 5000000],
        value=500000
    )

    simulate_crash = st.button("Simulate Sudden Server Crash & Restart")

with col2:
    st.subheader("Disaster Recovery Telemetry")
    telemetry = PersistenceRecoverySimulator.simulate_recovery(selected_modality, vector_count_val)
    
    t_df = pd.DataFrame([
        {"Parameter": k, "Value": str(v)} for k, v in telemetry.items()
    ])
    st.dataframe(t_df, use_container_width=True, hide_index=True)

    if simulate_crash:
        if selected_modality == "In-Memory Only":
            st.error("CRASH DETECTED: Process terminated! 100% data loss occurred. Zero vectors recovered.")
        else:
            st.success(f"CRASH RECOVERED: WAL replayed and segments mapped in {telemetry['Estimated Cold-Start Recovery Time']}. Zero data loss!")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: How does Memory Mapping (MMAP) allow vector databases to index datasets larger than available physical RAM?"):
    st.markdown("""
    **Answer**:
    Memory Mapping (`mmap`) instructs the operating system kernel to bind disk-backed segment files into the virtual memory address space. 
    Rather than allocating RAM explicitly and loading entire collections into memory, the application accesses vectors as if they were in RAM. 
    The OS page cache dynamically loads required 4 KB pages from NVMe storage into available physical memory and evicts cold pages when memory pressure rises. 
    This allows querying collections spanning hundreds of gigabytes on machines with only 32 GB of RAM.
    """)

with st.expander("Q2: Why is Write-Ahead Logging (WAL) essential in persistent vector databases?"):
    st.markdown("""
    **Answer**:
    Building high-dimensional vector indices (like HNSW) is compute-intensive and operates primarily in volatile memory buffers. 
    If a power loss or kernel panic occurs before in-memory graphs are sealed to disk, all pending inserts would be lost. 
    The WAL records mutations sequentially and synchronously to disk before acknowledging success to the caller. 
    On reboot, the database replays un-flushed WAL transactions to reconstruct memory state without data loss.
    """)

with st.expander("Q3: What are the tradeoffs between cold-storage snapshot restore vs. full corpus re-indexing from the primary SQL store?"):
    st.markdown("""
    **Answer**:
    * **Snapshot Restore**: Downloads pre-built index segment archives from cloud object storage (S3/GCS). Restores in minutes with zero LLM API costs, but restores data only up to the timestamp of the last snapshot.
    * **Full Re-Indexing**: Re-fetches raw text from PostgreSQL and calls embedding APIs for every chunk. Guarantees 100% fresh data, but takes hours or days for large corpora and incurs significant embedding API costs.
    """)

with st.expander("Q4: How does a disaggregated cloud-native vector architecture (like Milvus or Pinecone Serverless) handle sudden query spikes?"):
    st.markdown("""
    **Answer**:
    By decoupling storage from compute, coordinator nodes can dynamically spin up additional stateless Query Nodes on Kubernetes 
    or cloud VMs to distribute query traffic across multiple replicas. Because persistent data resides in cloud object storage (S3), 
    new query nodes can pull immutable index segments or stream them into local page caches without re-indexing or disturbing active write nodes.
    """)

with st.expander("Q5: What is the primary difference between Recovery Time Objective (RTO) and Recovery Point Objective (RPO) in vector stores?"):
    st.markdown("""
    **Answer**:
    * **RTO (Recovery Time Objective)**: The maximum acceptable duration of downtime required to restore the vector database to operational status following a disaster.
    * **RPO (Recovery Point Objective)**: The maximum acceptable age of data that can be permanently lost when a failure occurs (e.g., losing the last 5 minutes of writes vs. 24 hours of writes).
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying In-Memory Vector Stores in Containerized Kubernetes Deployments**  
    Running in-memory stores in Kubernetes pods without persistent volume claims (PVCs) causes complete data loss 
    every time a pod is rescheduled, restarted, or OOM-killed.
    """)
    st.error("""
    **Trap 2: Neglecting Snapshot Storage Lifecycle Policies**  
    Taking daily 50 GB index snapshots and storing them indefinitely in S3 without lifecycle expiration policies 
    generates massive cloud storage bills. Always enforce 30-day retention or Glacier tiering.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Slow Cold Starts Blocking Health Checks During Rolling Deployments**  
    Loading a 100 GB HNSW index into RAM during container startup can take 10+ minutes. 
    Kubernetes readiness probes will time out and continuously kill the container in a crash loop. Use MMAP or adjust probe timeouts.
    """)
    st.warning("""
    **Trap 4: WAL Disk Space Exhaustion Due to Stalled Flush Cycles**  
    If background segment flushes hang due to disk I/O bottlenecks, the WAL continues growing until it fills the host disk, 
    crashing the entire server. Monitor WAL size alerts closely.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: MMAP in Vector DBs")
        st.caption("How does it work?")
        st.markdown("**Back**: Maps disk-backed vector segment files directly into virtual memory. Kernel pages in data on-demand, allowing indexes larger than RAM.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Write-Ahead Log (WAL)")
        st.caption("Why is it necessary?")
        st.markdown("**Back**: Synchronously logs mutations to disk before updating in-memory graphs, preventing data loss during sudden server crashes.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Snapshot vs. Re-Index")
        st.caption("Which to pick for disaster recovery?")
        st.markdown("**Back**: Snapshots restore in minutes with zero API cost. Full re-indexing takes hours and incurs significant LLM embedding fees.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: RTO vs. RPO")
        st.caption("What do they measure?")
        st.markdown("**Back**: RTO is acceptable downtime duration until recovery; RPO is acceptable volume of lost data measured in elapsed time.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Database Persistence and Deployment:")

c1 = st.checkbox("I understand the performance and data loss tradeoffs of in-memory vector stores.", value=True)
c2 = st.checkbox("I understand how the Write-Ahead Log (WAL) guarantees durability for in-memory graph modifications.", value=True)
c3 = st.checkbox("I know how Memory-Mapped (MMAP) storage enables vector collections larger than physical RAM.", value=True)
c4 = st.checkbox("I can evaluate when to adopt cloud managed serverless stores vs. self-hosted on-premises deployments.", value=True)
c5 = st.checkbox("I know how to design disaster recovery protocols combining storage snapshots and WAL archival.", value=True)
c6 = st.checkbox("I can prevent Kubernetes cold-start crash loops and WAL disk bloat traps in production.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Persistence and Deployment and can architect resilient, enterprise-grade vector storage infrastructure.")
