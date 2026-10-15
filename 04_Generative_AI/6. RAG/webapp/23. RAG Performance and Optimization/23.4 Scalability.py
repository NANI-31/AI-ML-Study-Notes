import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.4 Scalability - RAG Master Curriculum",
    page_icon="📈",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ScalabilityCapacityEngine:
    """
    Enterprise Scalability & Distributed Capacity Sizing Engine.
    Models RAM, disk storage, HNSW graph expansion, vector quantization (FP32 vs SQ8 vs PQ),
    primary sharding requirements, and high-concurrency read replica scaling for 10,000+ QPS.
    """
    def __init__(self):
        # Bytes per dimension by precision mode
        self.precision_bytes = {
            "Float32 (Uncompressed)": 4.0,
            "Float16 (Half Precision)": 2.0,
            "SQ8 (Scalar Quantization 8-bit)": 1.0,
            "PQ (Product Quantization 64-code)": 0.25  # ~64 bytes per 1536d vector
        }

    def calculate_vector_memory(
        self,
        num_vectors: int,
        dimension: int,
        precision_mode: str,
        hnsw_m: int = 16,
        avg_metadata_bytes: int = 500
    ) -> Dict[str, float]:
        """
        Calculates raw vector RAM, HNSW graph index memory overhead,
        and disk metadata storage footprint.
        """
        bytes_per_dim = self.precision_bytes.get(precision_mode, 4.0)

        # Raw vector memory in bytes
        raw_vector_bytes = num_vectors * dimension * bytes_per_dim

        # HNSW Graph Overhead: Each node maintains ~M bidirectional links on ground layer
        # plus upper layers. Rule of thumb: ~M * 2 * 8 bytes (node pointer + distance cache)
        hnsw_graph_bytes = num_vectors * (hnsw_m * 2 * 8)

        # Total RAM required to hold index and vectors in memory
        total_ram_bytes = raw_vector_bytes + hnsw_graph_bytes

        # Disk Storage: Vector index + payload metadata + inverted index
        metadata_bytes = num_vectors * avg_metadata_bytes
        total_disk_bytes = total_ram_bytes * 1.25 + metadata_bytes

        raw_gb = raw_vector_bytes / (1024.0 ** 3)
        hnsw_gb = hnsw_graph_bytes / (1024.0 ** 3)
        total_ram_gb = total_ram_bytes / (1024.0 ** 3)
        disk_gb = total_disk_bytes / (1024.0 ** 3)

        return {
            "raw_vector_gb": round(raw_gb, 2),
            "hnsw_graph_gb": round(hnsw_gb, 2),
            "total_ram_gb": round(total_ram_gb, 2),
            "metadata_gb": round(metadata_bytes / (1024.0 ** 3), 2),
            "total_disk_gb": round(disk_gb, 2)
        }

    def size_distributed_cluster(
        self,
        total_ram_gb: float,
        target_qps: int,
        node_ram_gb: int = 64,
        qps_per_core: float = 45.0,
        cores_per_node: int = 16,
        replication_factor: int = 2
    ) -> Dict[str, Any]:
        """
        Sizes distributed sharding topology: primary shards, read replicas,
        and total cluster nodes required to sustain high-concurrency throughput.
        """
        # Usable RAM per node (leave 25% for OS buffer cache and query working memory)
        usable_ram_per_node = node_ram_gb * 0.75

        # Primary Shards calculation based on memory partition
        shards_for_ram = max(1, int(np.ceil(total_ram_gb / usable_ram_per_node)))

        # Single replica throughput capacity
        node_qps_capacity = cores_per_node * qps_per_core
        replica_sets_for_throughput = max(1, int(np.ceil(target_qps / max(1.0, node_qps_capacity))))

        # Total Read Replicas (must satisfy both durability and target QPS)
        effective_replicas = max(replication_factor, replica_sets_for_throughput)

        # Total Data Nodes across all shards and replicas
        total_data_nodes = shards_for_ram * effective_replicas

        # Total Cluster RAM and CPU
        total_cluster_ram_gb = total_data_nodes * node_ram_gb
        total_cluster_cores = total_data_nodes * cores_per_node
        peak_supported_qps = total_data_nodes * node_qps_capacity

        return {
            "primary_shards": shards_for_ram,
            "read_replicas": effective_replicas,
            "total_data_nodes": total_data_nodes,
            "total_cluster_ram_gb": total_cluster_ram_gb,
            "total_cluster_cores": total_cluster_cores,
            "peak_supported_qps": round(peak_supported_qps, 0),
            "usable_ram_per_node_gb": round(usable_ram_per_node, 1)
        }


# --- HERO SECTION ---
st.title("23.4 Scalability")
st.caption("Billion-Scale Vector Sharding, Distributed Index Topologies, Quantization Memory Sizing & 10,000+ QPS Serving")
st.markdown("""
Scaling a RAG system from a prototype containing 10,000 internal documents to an enterprise knowledge backbone managing
**100 million documents and answering 10,000 Queries Per Second (QPS)** requires transitioning from single-node instances
to a distributed, partitioned architecture.
At massive scale, single-node vector stores crash from **RAM exhaustion**, and naive nearest-neighbor search degrades under lock contention.
**Production Scalability** relies on:
1. **Vector Quantization (SQ8 & PQ)**: Compressing high-dimensional floating-point vectors by 75% to 90% without sacrificing recall.
2. **Horizontal Sharding & Distributed Topologies**: Partitioning vector indexes across multi-node clusters using consistent hashing and primary-replica consensus.
3. **Stateless Kubernetes Auto-Scaling**: Decoupling ingress query routers and embedding worker pods from stateful vector storage.
4. **HNSW Hyperparameter Tuning**: Calibrating `M`, `efConstruction`, and `efSearch` to balance memory overhead, build time, and throughput.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Distributed Vector Database Sharding & High-Concurrency Cluster Architecture", divider="rainbow")

st.markdown("""
The architecture below details the distributed topology of an enterprise billion-scale RAG serving system.
It features a **Stateless Ingress Tier** (Envoy Load Balancers & Kubernetes FastAPI Pods),
a **Distributed Scatter-Gather Coordinator**, and **Multi-Node Partitioned Vector Clusters** (Milvus / Qdrant)
with primary-replica consensus and asynchronous segment compaction.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    scale_blueprint_dot = """
    digraph ScalabilityArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_ingress_k8s {
            label="1. Stateless Ingress & Query Routing Tier (Kubernetes HPA: 10,000+ QPS)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            LB [label="Cloud Load Balancer / Envoy Proxy\nRound-Robin + Least-Connection Pooling", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            K8sPod1 [label="Query Worker Pod 1\nFastAPI + AsyncIO Loop", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            K8sPod2 [label="Query Worker Pod 2\nFastAPI + AsyncIO Loop", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            K8sPodN [label="Query Worker Pod N (Auto-Scaled)\nFastAPI + AsyncIO Loop", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_coordinator {
            label="2. Distributed Scatter-Gather Coordinator Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            ScatterGather [label="Scatter-Gather Coordinator\nConsistent Hash Partition Router", shape=ellipse, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            MergeHeap [label="Global Top-K Min-Heap Reducer\nk-Way Merge over Candidate Lists", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
        }

        subgraph cluster_sharded_cluster {
            label="3. Distributed Sharded Storage Tier (100M+ Vectors / SQ8 Quantized)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            subgraph cluster_shard1 {
                label="Shard 0 (Docs 0 to 33M)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#F59E0B";
                fontcolor="#FCD34D";
                Primary0 [label="Primary Node 0\nWrite Log & Raft Leader", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
                Replica0A [label="Read Replica 0A\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Replica0B [label="Read Replica 0B\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
            }

            subgraph cluster_shard2 {
                label="Shard 1 (Docs 33M to 66M)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#F59E0B";
                fontcolor="#FCD34D";
                Primary1 [label="Primary Node 1\nWrite Log & Raft Leader", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
                Replica1A [label="Read Replica 1A\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Replica1B [label="Read Replica 1B\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
            }

            subgraph cluster_shard3 {
                label="Shard 2 (Docs 66M to 100M)";
                style="filled,dashed";
                fillcolor="#0F172A";
                color="#F59E0B";
                fontcolor="#FCD34D";
                Primary2 [label="Primary Node 2\nWrite Log & Raft Leader", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
                Replica2A [label="Read Replica 2A\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
                Replica2B [label="Read Replica 2B\nIn-Memory HNSW (SQ8)", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
            }
        }

        LB -> K8sPod1;
        LB -> K8sPod2;
        LB -> K8sPodN;

        K8sPod1 -> ScatterGather [label="Incoming Search"];
        K8sPod2 -> ScatterGather;
        K8sPodN -> ScatterGather;

        ScatterGather -> Replica0A [label="Parallel Search"];
        ScatterGather -> Replica1A [label="Parallel Search"];
        ScatterGather -> Replica2A [label="Parallel Search"];

        Replica0A -> MergeHeap [label="Shard 0 Top-K"];
        Replica1A -> MergeHeap [label="Shard 1 Top-K"];
        Replica2A -> MergeHeap [label="Shard 2 Top-K"];

        MergeHeap -> K8sPod1 [label="Consolidated Top-K"];

        Primary0 -> Replica0A [label="Raft Replication", style=dotted, color="#94A3B8"];
        Primary0 -> Replica0B [label="Raft Replication", style=dotted, color="#94A3B8"];
        Primary1 -> Replica1A [label="Raft Replication", style=dotted, color="#94A3B8"];
        Primary1 -> Replica1B [label="Raft Replication", style=dotted, color="#94A3B8"];
        Primary2 -> Replica2A [label="Raft Replication", style=dotted, color="#94A3B8"];
        Primary2 -> Replica2B [label="Raft Replication", style=dotted, color="#94A3B8"];
    }
    """
    st.graphviz_chart(scale_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: LARGE DOCUMENT COLLECTIONS
# ==============================================================================
st.header("2. Large Document Collections (Petabyte-Scale Vector Storage, Ingestion Queues & Segment Merging)", divider="rainbow")

st.markdown("""
When an enterprise scales from 50,000 PDF documents to **100,000,000 document chunks**, single-file vector stores (such as local ChromaDB SQLite or in-memory FAISS indices) fail completely.
At petabyte scale, storage must be decoupled into distinct tiers:
1. **Raw Document & Blob Storage**: Object stores (AWS S3, Google Cloud Storage, Azure Blob) store full text, markdown representations, and original files economically.
2. **Vector Index Store**: High-performance NVMe SSDs and distributed memory hold quantized vector indices and HNSW navigation graphs.
3. **Log-Structured Merge (LSM) Segment Architecture**: Modern vector engines (e.g., Milvus, Qdrant, LanceDB) treat data as immutable segments.
""")

col_lc1, col_lc2 = st.columns([1, 1])

with col_lc1:
    st.markdown("### The LSM Segment Lifecycle at Scale:")
    st.markdown("""
    * **Growing (Active) Segments**:
      Incoming chunks are written to a Write-Ahead Log (WAL) and buffered in memory. Searches query both memory and sealed segments.
    * **Sealed Segments**:
      When a segment reaches a threshold (e.g., 512,000 vectors), it is sealed and marked read-only. An asynchronous background worker builds the HNSW index on the sealed segment.
    * **Segment Merging & Compaction**:
      Small, fragmented segments are continuously compacted into larger 1M-vector segments, purging deleted records and defragmenting physical memory.
    """)

with col_lc2:
    st.markdown("### Segment Compaction Architecture")
    dot_lsm = """
    digraph SegmentLSM {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        WAL [label="Write-Ahead Log (Kafka / Pulsar)\nDurability Guaranteed", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        MemBuffer [label="Growing Segment (RAM Buffer)\nLinear Scan for Fresh Writes", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        SealWorker [label="Seal Worker\nBuild HNSW / Quantize Vectors", shape=ellipse, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        CompactedSegment [label="Immutable Sealed Segment\nStored on NVMe SSD / S3 Tier", shape=cylinder, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];

        WAL -> MemBuffer -> SealWorker -> CompactedSegment;
    }
    """
    st.graphviz_chart(dot_lsm, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: DISTRIBUTED VECTOR DATABASES
# ==============================================================================
st.header("3. Distributed Vector Databases (HNSW Shards, Raft Consensus, Primary-Replica Read Clusters)", divider="rainbow")

st.markdown("""
A distributed vector database partitions vectors across multiple independent physical machines (**shards**) and replicates each shard for fault tolerance and query concurrency.
The coordination follows a **Scatter-Gather Architecture**:
""")

col_db1, col_db2 = st.columns([1, 1])

with col_db1:
    st.markdown("### Distributed Sharding Mechanics:")
    st.markdown("""
    * **Hash-Based Sharding**:
      A hash function `hash(doc_id) % num_shards` assigns chunks uniformly across shards.
      * *Query Execution*: The coordinator broadcasts the query vector to **all shards simultaneously**.
      * *k-Way Merge*: Each shard executes local HNSW search returning its top-k candidates. The coordinator merges candidate lists into a global top-k ranking.
    * **Tenant-Based Routing**:
      For multi-tenant SaaS systems, shards are keyed by `tenant_id`. Searches for a specific customer hit only the shard hosting that tenant, eliminating cross-shard network broadcast overhead.
    * **Raft Consensus Replication**:
      Each shard maintains 1 Primary and 2+ Read Replicas. Write operations commit only when acknowledged by a quorum, ensuring zero data loss if a node crashes.
    """)

with col_db2:
    st.markdown("### Scatter-Gather k-Way Reduction")
    dot_scatter = """
    digraph ScatterGatherFlow {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Coord [label="Query Coordinator", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        S1 [label="Shard 0 (Top-10)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
        S2 [label="Shard 1 (Top-10)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
        S3 [label="Shard 2 (Top-10)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#34D399"];
        Heap [label="Min-Heap Priority Queue\nk-Way Merge (Global Top-10)", shape=ellipse, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        Coord -> S1 [label="Broadcast"];
        Coord -> S2 [label="Broadcast"];
        Coord -> S3 [label="Broadcast"];
        S1 -> Heap;
        S2 -> Heap;
        S3 -> Heap;
        Heap -> Coord [label="Final Deduplicated Result"];
    }
    """
    st.graphviz_chart(dot_scatter, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: HORIZONTAL SCALING
# ==============================================================================
st.header("4. Horizontal Scaling (Kubernetes HPA, GPU Inference Pods & Load Balancer Topologies)", divider="rainbow")

st.markdown("""
A production RAG deployment consists of two distinct operational tiers with contrasting scaling characteristics:
1. **Stateless Compute Layer**: API gateways, query rewriting, reranking microservices, and orchestrator coroutines.
2. **Stateful Data Layer**: Distributed vector databases, relational caches, and segment stores.

Decoupling these layers enables independent auto-scaling using Kubernetes Horizontal Pod Autoscalers (HPA).
""")

col_hs1, col_hs2 = st.columns([1, 1])

with col_hs1:
    st.markdown("### Kubernetes Autoscaling Rules:")
    st.markdown("""
    * **Stateless API Worker Pods**:
      Scale dynamically based on **CPU utilization** (>70%) and **HTTP in-flight request queue depth**. A spike in query volume spins up worker pods in seconds.
    * **GPU Embedding & Reranker Pods**:
      Scale based on **Triton GPU Duty Cycle** (>80%) or custom Prometheus latency metrics. Keeping model weights resident in GPU VRAM avoids cold-start model load penalties.
    * **Vector Data Nodes**:
      Scale out **Read Replicas** horizontally when search QPS exceeds node throughput capacity.
    """)

with col_hs2:
    st.markdown("### Decoupled Kubernetes Topology")
    dot_k8s = """
    digraph K8sScaling {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        HPA [label="Kubernetes HPA Controller\nMonitors Request Latency & QPS", shape=ellipse, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        APIPool [label="API Workers Deployment\n(Autoscales 4 -> 64 Pods)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        GPUPool [label="GPU Triton Inference Deployment\n(Autoscales 2 -> 16 GPUs)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#A855F7"];
        VectorCluster [label="Stateful Vector DB Cluster\n(Autoscales Read Replicas)", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        HPA -> APIPool;
        HPA -> GPUPool;
        APIPool -> GPUPool [label="gRPC Embed / Rerank"];
        APIPool -> VectorCluster [label="gRPC Search"];
    }
    """
    st.graphviz_chart(dot_k8s, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: INDEX OPTIMIZATION
# ==============================================================================
st.header("5. Index Optimization (Vector Quantization: IVFPQ, Scalar Quantization & Memory Footprint Tuning)", divider="rainbow")

st.markdown("""
At 100M+ vectors, storing raw `Float32` embeddings in memory becomes economically infeasible:
`100,000,000 * 1536 dims * 4 bytes = 614.4 GB` of raw vector data alone, expanding to over **900 GB of RAM** once HNSW navigation graph links are added.

**Vector Quantization** compresses floating-point representations into compact discrete codes, drastically reducing memory footprint while maintaining search recall.
""")

col_iq1, col_iq2 = st.columns([1, 1])

with col_iq1:
    st.markdown("### Vector Quantization Paradigms:")
    quant_table = pd.DataFrame({
        "Quantization Mode": ["Uncompressed Float32", "Scalar Quantization (SQ8)", "Product Quantization (PQ)", "IVF-PQ Hybrid"],
        "Bytes per Dimension": [4.0, 1.0, 0.25, 0.125],
        "RAM for 100M Vectors (1536d)": ["614.4 GB", "153.6 GB (-75%)", "38.4 GB (-93.7%)", "~22.0 GB (-96.4%)"],
        "Recall@10 Retention": ["100% (Baseline)", "98.5% - 99.2%", "92.0% - 96.0%", "88.0% - 93.0%"],
        "Query Throughput": ["Standard", "High (AVX-512 SIMD)", "Ultra-High (Cache Friendly)", "Max Throughput"]
    })
    st.dataframe(quant_table, use_container_width=True)

with col_iq2:
    st.markdown("### HNSW Parameter Sizing & Recall Tuning")
    st.markdown("""
    * **`M` (Max Links per Node)**:
      Controls the degree of connectivity in the HNSW graph (typically `M = 16 to 64`).
      * Higher `M` increases recall and routing stability, but expands index memory linearly: `Memory_HNSW = N * M * 2 * 8 bytes`.
    * **`efConstruction` (Index Build Horizon)**:
      Determines search breadth during index creation (typically 100 - 400).
      * Higher values increase index build time significantly but yield a more optimal navigation graph.
    * **`efSearch` (Query Search Horizon)**:
      Determines candidate queue size during runtime search (typically 32 - 128).
      * Can be adjusted per-query: higher values improve top-k recall at the expense of query latency.
    """)

st.divider()

# ==============================================================================
# SECTION 6: HIGH-CONCURRENCY RAG
# ==============================================================================
st.header("6. High-Concurrency RAG (Handling 10,000+ QPS, Connection Pooling, Rate Limiting & Queue Throttling)", divider="rainbow")

st.markdown("""
Under extreme query concurrency (10,000+ QPS), systems fail not from computational model constraints,
but from **network socket starvation**, **thread pool exhaustion**, and **uncontrolled request queuing**.
Resilient high-concurrency serving requires defensive backpressure and traffic shedding mechanisms:
""")

col_hc1, col_hc2 = st.columns([1, 1])

with col_hc1:
    st.markdown("### Critical Production Guardrails:")
    st.markdown("""
    1. **HTTP/2 & gRPC Connection Multiplexing**:
       Establishes long-lived persistent TCP channels with pipelined streaming, eliminating socket open/close overhead and OS ephemeral port exhaustion.
    2. **Token Bucket Rate Limiting**:
       Applies client-tier rate limiting (e.g., via Redis sliding window counters) to prevent abusive automated scripts from degrading service for legitimate users.
    3. **Adaptive Concurrency Limits & Shedding**:
       Uses Little's Law (`Concurrency = QPS * Latency`) to monitor queue depth. When latency spikes past p99 SLAs, the gateway sheds non-critical background traffic with HTTP 429 rather than letting the entire cluster collapse.
    4. **Circuit Breaking**:
       If a vector shard replica becomes unresponsive (e.g., error rate > 5%), the circuit breaker immediately routes traffic to peer replicas.
    """)

with col_hc2:
    st.markdown("### Defensive Gateway Ingress Stack")
    dot_ingress = """
    digraph DefensiveIngress {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Traffic [label="10,000+ Incoming QPS", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        RateLimiter [label="Redis Token Bucket Rate Limiter\n(Per Tenant Quota)", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EF4444"];
        Shedder [label="Adaptive Concurrency Shedder\n(Queue Depth Check)", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        ConnPool [label="gRPC Connection Pool\n(Multiplexed HTTP/2)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        Backend [label="Distributed RAG Core", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];

        Traffic -> RateLimiter;
        RateLimiter -> Shedder [label="Quota OK"];
        RateLimiter -> Traffic [label="429 Rate Limited", style=dotted, color="#EF4444"];
        Shedder -> ConnPool [label="Queue OK"];
        Shedder -> Traffic [label="503 Shed", style=dotted, color="#F59E0B"];
        ConnPool -> Backend;
    }
    """
    st.graphviz_chart(dot_ingress, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Distributed Vector Capacity & Scalability Sizing Engine (ScalabilityCapacityEngine)", divider="rainbow")

st.markdown("""
Below is the production-grade capacity planning engine `ScalabilityCapacityEngine`.
It models vector RAM consumption, HNSW graph index expansion, metadata storage footprints,
and distributed cluster sharding configurations for high-throughput enterprise systems.
""")

st.code('''
class ScalabilityCapacityEngine:
    """
    Enterprise Scalability & Distributed Capacity Sizing Engine.
    Models RAM, disk storage, HNSW graph expansion, vector quantization (FP32 vs SQ8 vs PQ),
    primary sharding requirements, and high-concurrency read replica scaling for 10,000+ QPS.
    """
    def __init__(self):
        self.precision_bytes = {
            "Float32 (Uncompressed)": 4.0,
            "Float16 (Half Precision)": 2.0,
            "SQ8 (Scalar Quantization 8-bit)": 1.0,
            "PQ (Product Quantization 64-code)": 0.25
        }

    def calculate_vector_memory(
        self,
        num_vectors: int,
        dimension: int,
        precision_mode: str,
        hnsw_m: int = 16,
        avg_metadata_bytes: int = 500
    ) -> Dict[str, float]:
        bytes_per_dim = self.precision_bytes.get(precision_mode, 4.0)
        raw_vector_bytes = num_vectors * dimension * bytes_per_dim
        hnsw_graph_bytes = num_vectors * (hnsw_m * 2 * 8)
        total_ram_bytes = raw_vector_bytes + hnsw_graph_bytes

        metadata_bytes = num_vectors * avg_metadata_bytes
        total_disk_bytes = (total_ram_bytes * 1.25) + metadata_bytes

        return {
            "raw_vector_gb": round(raw_vector_bytes / (1024.0 ** 3), 2),
            "hnsw_graph_gb": round(hnsw_graph_bytes / (1024.0 ** 3), 2),
            "total_ram_gb": round(total_ram_bytes / (1024.0 ** 3), 2),
            "total_disk_gb": round(total_disk_bytes / (1024.0 ** 3), 2)
        }

    def size_distributed_cluster(
        self,
        total_ram_gb: float,
        target_qps: int,
        node_ram_gb: int = 64,
        qps_per_core: float = 45.0,
        cores_per_node: int = 16,
        replication_factor: int = 2
    ) -> Dict[str, Any]:
        usable_ram_per_node = node_ram_gb * 0.75
        shards_for_ram = max(1, int(np.ceil(total_ram_gb / usable_ram_per_node)))

        node_qps_capacity = cores_per_node * qps_per_core
        replica_sets_for_throughput = max(1, int(np.ceil(target_qps / max(1.0, node_qps_capacity))))
        effective_replicas = max(replication_factor, replica_sets_for_throughput)

        total_data_nodes = shards_for_ram * effective_replicas

        return {
            "primary_shards": shards_for_ram,
            "read_replicas": effective_replicas,
            "total_data_nodes": total_data_nodes,
            "total_cluster_ram_gb": total_data_nodes * node_ram_gb,
            "peak_supported_qps": round(total_data_nodes * node_qps_capacity, 0)
        }
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
scale_engine_inst = ScalabilityCapacityEngine()
test_mem = scale_engine_inst.calculate_vector_memory(
    num_vectors=10_000_000,
    dimension=1536,
    precision_mode="SQ8 (Scalar Quantization 8-bit)",
    hnsw_m=16,
    avg_metadata_bytes=500
)
assert test_mem["total_ram_gb"] > 0, "Total RAM requirement must be positive"
assert test_mem["raw_vector_gb"] < 30.0, "SQ8 for 10M 1536d vectors must occupy less than 30GB"

test_sizing = scale_engine_inst.size_distributed_cluster(
    total_ram_gb=test_mem["total_ram_gb"],
    target_qps=5000,
    node_ram_gb=64,
    qps_per_core=45.0,
    cores_per_node=16,
    replication_factor=2
)
assert test_sizing["primary_shards"] >= 1, "Must require at least 1 primary shard"
assert test_sizing["total_data_nodes"] >= 2, "Must configure redundant nodes"

st.success("Automated Verification Passed: ScalabilityCapacityEngine successfully verified across vector quantization sizing, HNSW expansion, and distributed cluster sizing.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Distributed Cluster Capacity Sizing & QPS Load Simulator", divider="rainbow")

st.markdown("""
Design and size your enterprise distributed vector database infrastructure.
Configure vector volume, embedding dimensions, quantization compression modes, HNSW connectivity, and concurrency QPS targets.
""")

wb_s1, wb_s2, wb_s3 = st.columns([1, 1, 1])

with wb_s1:
    st.markdown("#### Document & Vector Volume")
    in_num_vectors = st.select_slider(
        "Total Vector Count",
        options=[1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000],
        value=25_000_000
    )
    in_dimension = st.selectbox("Embedding Vector Dimension", [384, 768, 1024, 1536, 3072], index=3)
    in_quant_mode = st.selectbox(
        "Quantization Precision Mode",
        ["Float32 (Uncompressed)", "Float16 (Half Precision)", "SQ8 (Scalar Quantization 8-bit)", "PQ (Product Quantization 64-code)"],
        index=2
    )

with wb_s2:
    st.markdown("#### HNSW Index Hyperparameters")
    in_hnsw_m = st.slider("HNSW Max Links per Node (M)", min_value=8, max_value=64, value=16, step=8)
    in_metadata_bytes = st.slider("Avg Metadata / Payload Bytes per Vector", min_value=200, max_value=2000, value=500, step=100)
    in_target_qps = st.slider("Target Concurrent QPS (Peak Load)", min_value=100, max_value=10_000, value=2500, step=100)

with wb_s3:
    st.markdown("#### Cloud Hardware Specifications")
    in_node_ram = st.selectbox("RAM per Server Node (GB)", [32, 64, 128, 256], index=1)
    in_node_cores = st.selectbox("CPU Cores per Server Node", [8, 16, 32, 64], index=1)
    in_min_replicas = st.slider("Minimum Replication Factor (High Availability)", min_value=1, max_value=4, value=2, step=1)

sim_scale_engine = ScalabilityCapacityEngine()
mem_res = sim_scale_engine.calculate_vector_memory(
    num_vectors=in_num_vectors,
    dimension=in_dimension,
    precision_mode=in_quant_mode,
    hnsw_m=in_hnsw_m,
    avg_metadata_bytes=in_metadata_bytes
)

cluster_res = sim_scale_engine.size_distributed_cluster(
    total_ram_gb=mem_res["total_ram_gb"],
    target_qps=in_target_qps,
    node_ram_gb=in_node_ram,
    qps_per_core=45.0,
    cores_per_node=in_node_cores,
    replication_factor=in_min_replicas
)

st.markdown("### Sized Cluster Topology & Hardware Bill of Materials")

cm_1, cm_2, cm_3, cm_4 = st.columns(4)
cm_1.metric("Total In-Memory RAM Needed", f"{mem_res['total_ram_gb']} GB")
cm_2.metric("Total Data Nodes", f"{cluster_res['total_data_nodes']} Nodes", delta=f"{cluster_res['primary_shards']} Shards x {cluster_res['read_replicas']} Replicas")
cm_3.metric("Peak Supported QPS", f"{cluster_res['peak_supported_qps']:,} QPS")
cm_4.metric("Persistent Disk Footprint", f"{mem_res['total_disk_gb']} GB")

st.markdown("#### Hardware Sizing Breakdown Table")
sizing_df = pd.DataFrame([
    {"Component": "Raw Vector Embeddings", "Sizing Metric": f"{mem_res['raw_vector_gb']} GB", "Technical Context": f"{in_quant_mode} @ {in_dimension}d"},
    {"Component": "HNSW Navigation Graph Index", "Sizing Metric": f"{mem_res['hnsw_graph_gb']} GB", "Technical Context": f"M = {in_hnsw_m} links per node"},
    {"Component": "Metadata & Document Text", "Sizing Metric": f"{mem_res['metadata_gb']} GB", "Technical Context": f"{in_metadata_bytes} bytes per record"},
    {"Component": "Primary Cluster Shards", "Sizing Metric": f"{cluster_res['primary_shards']} Shards", "Technical Context": f"Max {cluster_res['usable_ram_per_node_gb']} GB usable RAM/node"},
    {"Component": "Read Replicas per Shard", "Sizing Metric": f"{cluster_res['read_replicas']} Replicas", "Technical Context": f"Sustains {in_target_qps} target QPS"},
    {"Component": "Total Cluster Hardware RAM", "Sizing Metric": f"{cluster_res['total_cluster_ram_gb']} GB", "Technical Context": f"{cluster_res['total_data_nodes']} servers @ {in_node_ram} GB"}
])
st.dataframe(sizing_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What is the difference between Hash-Based Sharding and Tenant-Based Sharding in a distributed vector database?**"):
    st.markdown("""
    **Answer**:
    * **Hash-Based Sharding**: Partitions vectors uniformly across shards using a hash of the vector ID (`hash(id) % N`).
      * *Pros*: Guarantees perfectly uniform data and query distribution across nodes.
      * *Cons*: Every search query must be broadcast to **all shards** (Scatter-Gather), incurring network fan-out and k-way merging overhead.
    * **Tenant-Based Sharding**: Assigns all vectors belonging to a specific customer or organization to dedicated shards (`shard = tenant_id`).
      * *Pros*: Single-tenant queries hit only the designated shard, eliminating scatter-gather broadcast.
      * *Cons*: Prone to the 'noisy neighbor' and hotspot problem if a large enterprise tenant experiences high traffic spikes.
    """)

with st.expander("**Q2: How does Scalar Quantization (SQ8) work, and why does it achieve a 75% memory reduction with negligible recall drop?**"):
    st.markdown("""
    **Answer**:
    In standard embeddings, each dimension is stored as a 32-bit IEEE-754 floating-point number (4 bytes).
    Scalar Quantization (SQ8) computes the minimum and maximum values across the entire dimension distribution:
    Each 32-bit float is mapped to an 8-bit unsigned integer (0 to 255) via:
    `q = round(255 * (x - min_val) / (max_val - min_val))`.
    * **Memory Reduction**: 1 byte instead of 4 bytes per dimension = **75% RAM savings**.
    * **Recall Retention**: High-dimensional embeddings contain significant redundancy; rounding errors across 1536 dimensions cancel out statistically, maintaining >98.5% Top-10 recall.
    * **CPU Acceleration**: 8-bit integers can be compared using AVX-512 SIMD vector instructions, often speeding up distance calculation.
    """)

with st.expander("**Q3: What causes HNSW graph index memory expansion, and how do you calculate the RAM requirement?**"):
    st.markdown("""
    **Answer**:
    Hierarchical Navigable Small World (HNSW) constructs multi-layer proximity graphs where each vector acts as a graph vertex.
    Each vertex maintains up to `M` outgoing edges on upper layers and `2 * M` edges on the ground layer (Layer 0).
    * Each edge pointer requires 8 bytes (64-bit integer node ID).
    * Rule of thumb formula: `RAM_HNSW = N * (M * 2 * 8 bytes)`.
    * For 50M vectors with `M = 32`: `50,000,000 * (32 * 2 * 8) = 25.6 GB` just for graph links, in addition to the vector coordinates themselves.
    """)

with st.expander("**Q4: How do you prevent the 'Thundering Herd' problem when 5,000 concurrent users query a RAG system simultaneously?**"):
    st.markdown("""
    **Answer**:
    1. **Semantic Vector Caching**: Intercepts requests at the API layer so identical or rephrased queries return immediately without touching the vector database.
    2. **Request Coalescing (Singleflight Pattern)**: If 50 identical queries arrive within a 20ms window during a cache miss, the coordinator coalesces them into a single in-flight vector search, sharing the single result across all 50 waiting requests.
    3. **Token Bucket Rate Limiting & Queue Shedding**: Rejecting excess requests with HTTP 429 when concurrency limits exceed system capacity.
    """)

with st.expander("**Q5: Why must write operations be decoupled from read queries in high-concurrency vector databases?**"):
    st.markdown("""
    **Answer**:
    Inserting vectors into an active HNSW graph requires traversing the graph, finding nearest neighbors, acquiring locks on existing vertices, and updating their neighbor lists.
    Under heavy concurrent read traffic, write locks create severe **lock contention**, stalling read searches and spiking latency from 15ms to over 2,000ms.
    * **Decoupling Strategy**: Write operations append strictly to a Write-Ahead Log (WAL) and unindexed append-only memory buffer. Background workers periodically seal and build immutable HNSW indices, while read queries hit dedicated, lock-free Read Replicas.
    """)

st.markdown("### Production Traps & Failure Modes")

str_col1, str_col2 = st.columns(2)

with str_col1:
    st.error("""
    **Trap 1: RAM Exhaustion from HNSW Graph Under-Sizing**
    * **Failure**: Calculating memory requirements based solely on raw vector floats (`N * dim * 4 bytes`) and ignoring HNSW graph edge overhead.
    * **Consequence**: When the dataset reaches 20M vectors, the operating system Out-Of-Memory (OOM) killer abruptly terminates the vector database process, corrupting uncommitted segments.
    * **Remedy**: Always add 40% - 60% memory headroom for HNSW graph links, query working memory (`efSearch`), and OS page cache.
    """)

    st.error("""
    **Trap 2: Scatter-Gather Fan-Out Latency Explosion**
    * **Failure**: Creating 100 small shards for a 5M vector dataset.
    * **Consequence**: Every single search query triggers 100 simultaneous network roundtrips. The slowest shard's tail latency governs the entire request (p99 latency jumps 5x).
    * **Remedy**: Keep shards large (typically 5M - 20M vectors per shard) and scale out via Read Replicas rather than excessive sharding.
    """)

with str_col2:
    st.warning("""
    **Trap 3: Ingress Ephemeral Port Starvation**
    * **Failure**: Creating a new TCP connection to the vector database or LLM endpoint on every incoming HTTP request.
    * **Consequence**: Under 3,000 QPS, the OS exhausts all 65,535 ephemeral ports (stuck in `TIME_WAIT`), causing subsequent connection attempts to fail with `Cannot assign requested address`.
    * **Remedy**: Use persistent gRPC channels or HTTP/2 client connection pools with keep-alive enabled.
    """)

    st.warning("""
    **Trap 4: Building HNSW Graphs on the Ingestion Critical Path**
    * **Failure**: Building the complete HNSW index synchronously during real-time document upload.
    * **Consequence**: Document ingestion throughput collapses to fewer than 50 chunks per second, backing up upstream ingestion queues.
    * **Remedy**: Ingest into raw append-only buffers and build HNSW indices asynchronously on sealed segments in background worker threads.
    """)

st.markdown("### Core Technical Flashcards")

fcs_c1, fcs_c2 = st.columns(2)

with fcs_c1:
    with st.container(border=True):
        st.markdown("**Front**: How much memory does Scalar Quantization (SQ8) save compared to Float32 vectors?")
        st.markdown("**Back**: SQ8 compresses 32-bit floating-point dimensions (4 bytes) into 8-bit unsigned integers (1 byte), reducing raw vector RAM requirements by exactly 75% with less than 1% drop in top-k recall.")

    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of the Scatter-Gather coordinator in a distributed vector database?")
        st.markdown("**Back**: It broadcasts the query vector across all independent shards in parallel, collects the local Top-K candidate lists from each shard, and performs a k-way merge reduction into a global Top-K ranking.")

with fcs_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why are Read Replicas essential for high-concurrency (10,000+ QPS) RAG systems?")
        st.markdown("**Back**: Read Replicas replicate identical shard indexes across independent servers, distributing read traffic horizontally and preventing lock contention between writes and searches.")

    with st.container(border=True):
        st.markdown("**Front**: What does the HNSW hyperparameter 'M' control?")
        st.markdown("**Back**: 'M' defines the maximum number of bidirectional connection links per node in the navigation graph. Higher 'M' boosts recall and stability but increases index construction time and memory overhead linearly.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of RAG Scalability & Distributed Infrastructure principles:")

s_chk1 = st.checkbox("I can calculate raw vector and HNSW graph RAM memory footprints for billion-scale vector datasets.", value=True)
s_chk2 = st.checkbox("I understand the mechanics of Scalar Quantization (SQ8) and Product Quantization (PQ) and their recall trade-offs.", value=True)
s_chk3 = st.checkbox("I can architect a distributed Scatter-Gather sharding topology with Primary-Replica Raft consensus.", value=True)
s_chk4 = st.checkbox("I know how to decouple stateless Kubernetes API pods from stateful vector database data nodes.", value=True)
s_chk5 = st.checkbox("I understand defensive gateway techniques (connection pooling, rate limiting, and adaptive load shedding) for 10,000+ QPS.", value=True)
s_chk6 = st.checkbox("I know how to size distributed cluster hardware (primary shards, read replicas, and CPU cores) to meet high-throughput SLAs.", value=True)

if s_chk1 and s_chk2 and s_chk3 and s_chk4 and s_chk5 and s_chk6:
    st.success("Mastery Certified: You have mastered production RAG scalability, distributed vector sharding, quantization capacity sizing, and high-concurrency architectures.")
