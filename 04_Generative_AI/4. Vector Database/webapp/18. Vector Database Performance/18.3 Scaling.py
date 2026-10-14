import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="18.3 Scaling Vector Databases - Vector DB Curriculum",
    page_icon="🌐",
    layout="wide"
)

# Title & Overview
st.title("18.3 Scaling Vector Databases: Vertical vs Horizontal, Sharding, Replication & Distributed Search")
st.caption("Chapter 18: Vector Database Performance | Domain: 4. Vector Database")
st.markdown("""
When a vector database expands from 50,000 internal documents to **hundreds of millions of customer vectors 
serving thousands of queries per second**, single-node architectures reach a hard physical wall.

High-dimensional vector indexes like HNSW cannot be simply paginated like SQL tables: 
they require **contiguous in-memory graph traversal**, meaning that RAM limits dictate total capacity, 
while CPU core contention dictates maximum query throughput.

To scale beyond the limits of a single machine, enterprise vector architectures apply two complementary dimensions:
1. **Replication (Scaling Read Throughput - QPS)**: Multiplying read-only copies of the index across nodes to divide search traffic.
2. **Sharding (Scaling Corpus Volume - RAM)**: Partitioning the dataset into disjoint subsets distributed across independent nodes.
3. **Distributed Scatter-Gather Search**: Broadcasting a query vector to all shards in parallel and fusing the results with a global min-heap.

In this lesson, we break down the mechanics of horizontal scaling, analyze the **Scatter-Gather** query protocol, 
examine distributed failure modes, and test an interactive Cluster Sizing & Topology Simulator.
""")

st.divider()

# Section 1: Vertical vs Horizontal Scaling Compared
st.header("1. Vertical vs Horizontal Scaling Compared", divider="rainbow")

col_scal1, col_scal2 = col_scal1, col_scal2 = st.columns([1.1, 0.9])

with col_scal1:
    st.markdown("""
    ### Vertical Scaling (Scale-Up)
    * **Mechanism**: Deploying on progressively larger instances (e.g. upgrading from 16 GB to 512 GB RAM, 64 vCPUs).
    * **Pros**: Simple operational model; zero network serialization overhead; ACID consistency remains local.
    * **The Ceiling**: Becomes astronomically expensive. Cloud instances max out around 1 TB RAM, and memory-bus contention plateaus search throughput around ~2,500 QPS.
    
    ### Horizontal Scaling (Scale-Out)
    * **Mechanism**: Distributing data and query traffic across a cluster of commodity nodes using **Sharding** and **Replication**.
    * **Pros**: Unlimited capacity scaling (can host billions of vectors); resilient against node failures; cost-effective.
    * **Engineering Challenges**: Network partition handling, coordinator scatter-gather latency, and the **Straggler Problem** (query speed is bounded by the slowest shard).
    """)

with col_scal2:
    st.markdown("### The Dual Dimensions of Horizontal Scaling")
    scale_dot = """
    digraph ScalingDimensions {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ScalingCore [label="Horizontal Scaling Engine\\n(Distributed Cluster)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_sharding {
            label="Dimension 1: Sharding"; style="dashed"; color="#059669";
            ShardA [label="Shard 1 (Vectors 0 - 33M)\\nLocal HNSW Graph", fillcolor="#D1FAE5"];
            ShardB [label="Shard 2 (Vectors 33M - 66M)\\nLocal HNSW Graph", fillcolor="#D1FAE5"];
            ShardC [label="Shard 3 (Vectors 66M - 100M)\\nLocal HNSW Graph", fillcolor="#D1FAE5"];
        }

        subgraph cluster_rep {
            label="Dimension 2: Replication"; style="dashed"; color="#D97706";
            Replica1 [label="Replica A (Read Only)\\nServes Traffic Pool 1", fillcolor="#FEF3C7"];
            Replica2 [label="Replica B (Read Only)\\nServes Traffic Pool 2", fillcolor="#FEF3C7"];
        }

        ScalingCore -> ShardA [label="Scales Total Corpus Volume"];
        ScalingCore -> ShardB;
        ScalingCore -> ShardC;
        ScalingCore -> Replica1 [label="Scales Search QPS & HA"];
        ScalingCore -> Replica2;
    }
    """
    st.graphviz_chart(scale_dot, use_container_width=False)

st.divider()

# Section 2: Sharding, Replication & The Scatter-Gather Query Protocol
st.header("2. Sharding, Replication & The Scatter-Gather Protocol", divider="rainbow")

col_sg1, col_sg2 = col_sg1, col_sg2 = st.columns([1, 1])

with col_sg1:
    st.markdown("""
    ### How Distributed Search Works (Scatter-Gather)
    When an indexed dataset is split across `M` shards, no single shard holds all candidate vectors. To execute `Top-K` search:
    
    1. **Query Ingestion**: Client sends `query_vector` to the **Coordinator Node**.
    2. **Scatter (Parallel Broadcast)**: The Coordinator dispatches the query asynchronously to **every active shard** in the cluster.
    3. **Local Search**: Each shard traverses its local HNSW index, finding its **local Top-K candidates** (e.g. K=5).
    4. **Gather (Network Collection)**: Shards send their local top-K candidates back to the Coordinator (total candidates collected = `M * K`).
    5. **Global Merge**: The Coordinator merges the `M * K` candidates using a global min-heap priority queue, selecting the **definitive Top-K** to return to the user.
    """)

with col_sg2:
    st.markdown("### Distributed Scatter-Gather Flowchart")
    sg_dot = """
    digraph ScatterGather {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Client [label="Client Search Request\\nTop-K = 3", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Coord [label="Query Coordinator Node\\nBroadcasts Vector to Shards", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        S1 [label="Shard 1 (Node A)\\nFinds Local Top-3", fillcolor="#D1FAE5"];
        S2 [label="Shard 2 (Node B)\\nFinds Local Top-3", fillcolor="#D1FAE5"];
        S3 [label="Shard 3 (Node C)\\nFinds Local Top-3", fillcolor="#D1FAE5"];

        GlobalMerge [label="Global Priority Queue Merge\\nFuses 3 x 3 = 9 candidates -> Top 3", fillcolor="#FDE68A", color="#D97706", style="filled,bold"];

        Client -> Coord;
        Coord -> S1 [label="Scatter (Parallel)", color="#059669", style="bold"];
        Coord -> S2 [label="Scatter (Parallel)", color="#059669", style="bold"];
        Coord -> S3 [label="Scatter (Parallel)", color="#059669", style="bold"];
        S1 -> GlobalMerge [label="Local Hits", color="#D97706"];
        S2 -> GlobalMerge [label="Local Hits", color="#D97706"];
        S3 -> GlobalMerge [label="Local Hits", color="#D97706"];
        GlobalMerge -> Client [label="Final Grounded Top-3", style="bold"];
    }
    """
    st.graphviz_chart(sg_dot, use_container_width=False)

st.divider()

# Section 3: Distributed System Challenges in Vector DBs
st.header("3. Distributed System Challenges in Vector DBs", divider="rainbow")

st.markdown("""
Operating distributed vector databases introduces unique distributed systems complexities:
""")

CHALLENGES = [
    {
        "Distributed Challenge": "The Straggler Problem",
        "Mechanics": "Because the Coordinator must wait for results from all M shards before global merging, total search latency equals the latency of the single slowest shard: T_query = max(T_shard_1 ... T_shard_M).",
        "Mitigation Strategy": "Hedged requests (canceling and re-dispatching slow requests to alternate read replicas) and strict network timeouts."
    },
    {
        "Distributed Challenge": "Index Imbalance / Skew",
        "Mechanics": "If data is partitioned naively by tenant or date, 80% of vectors land on Shard 1, exhausting its RAM while Shard 2 and 3 sit idle.",
        "Mitigation Strategy": "Consistent hashing on Document UUID (e.g. MurmurHash3 or Ketama) to guarantee uniform distribution."
    },
    {
        "Distributed Challenge": "Eventual Consistency in HNSW Replicas",
        "Mechanics": "Replicating graph updates across nodes takes time. A document inserted on the Leader may not be immediately searchable on Follower replicas.",
        "Mitigation Strategy": "Use Raft consensus protocols (as in Qdrant and Milvus) to ensure linearizable index versioning."
    },
    {
        "Distributed Challenge": "Cross-Shard Metadata Filtering",
        "Mechanics": "If a query filters by a rare tag (where={'status': 'archived'}), some shards may find 0 matching vectors while other shards find 20.",
        "Mitigation Strategy": "Dynamic candidate buffer sizing: shards return up to 2 * K candidates if matching volume is sparse."
    }
]

st.dataframe(pd.DataFrame(CHALLENGES), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Distributed Cluster Topology & Scaling Simulator
st.header("4. Interactive Workbench: Cluster Topology & Sizing Simulator", divider="rainbow")

st.markdown("""
Input your enterprise scale requirements (total vector volume and desired search QPS). 
The simulator calculates the exact number of **Shards** and **Replicas** needed, projects RAM distribution, 
and charts the cluster topology.
""")

col_sim_req, col_sim_hw = col_sim_req, col_sim_hw = st.columns(2)

with col_sim_req:
    st.subheader("1. Scale Requirements")
    target_vectors_m = st.slider("Total Dataset Volume (Million Vectors):", min_value=1.0, max_value=200.0, value=30.0, step=5.0)
    target_qps = st.slider("Target Query Throughput (QPS):", min_value=100, max_value=5000, value=1200, step=100)

with col_sim_hw:
    st.subheader("2. Node Hardware Profile")
    node_ram_gb = st.selectbox("RAM per Worker Node:", [32, 64, 128], index=1)
    qps_per_node = st.selectbox("Max QPS Supported per Node Core:", [300, 500, 800], index=1)

# Calculations
# 1M 1536-dim vectors needs ~8 GB RAM
total_dataset_ram_gb = target_vectors_m * 8.0
usable_ram_per_node = node_ram_gb * 0.75  # 75% usable for vectors, 25% OS/buffers

# Shards needed to hold data
required_shards = int(np.ceil(total_dataset_ram_gb / usable_ram_per_node))
# Replicas needed to handle QPS
qps_capacity_single_set = qps_per_node
required_replicas = int(np.ceil(target_qps / qps_capacity_single_set))

total_worker_nodes = required_shards * required_replicas
total_cluster_ram_gb = total_worker_nodes * node_ram_gb

st.divider()

col_topo_metrics, col_topo_diag = col_topo_metrics, col_topo_diag = st.columns([1, 1.2])

with col_topo_metrics:
    st.subheader("Cluster Capacity Blueprint")
    
    st.metric(label="Total Worker Nodes Required", value=f"{total_worker_nodes} Nodes", delta=f"{required_shards} Shards x {required_replicas} Replicas")
    st.metric(label="Total Cluster RAM Provisioned", value=f"{total_cluster_ram_gb} GB RAM", delta=f"{total_dataset_ram_gb:.1f} GB Dataset Active")
    st.metric(label="Estimated Scatter-Gather Latency", value="4.2 ms (p95)", delta="Network Overlap Budget")
    
    st.success(f"""
    **Architecture Summary**:
    * **Data Sharding**: Partitioned into **{required_shards} Shards** (each hosting ~{target_vectors_m/required_shards:.1f}M vectors).
    * **Read Replication**: Replicated **{required_replicas}x** across independent availability zones to satisfy {target_qps} QPS.
    * **High Availability**: Tolerates failure of up to {required_replicas - 1} node(s) per shard without downtime.
    """)

with col_topo_diag:
    st.subheader("Generated Cluster Topology")
    
    # Generate dynamic graphviz
    nodes_code = ""
    for s in range(1, min(required_shards + 1, 4)):
        for r in range(1, min(required_replicas + 1, 3)):
            nodes_code += f'Shard_{s}_Rep_{r} [label="Shard {s} (Rep {r})\\nNode {s}-{r}", fillcolor="#D1FAE5"];\n'
            nodes_code += f'Coord -> Shard_{s}_Rep_{r};\n'
            
    sim_dot = f"""
    digraph DynamicCluster {{
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Coord [label="Load Balancer / Coordinator\\n(Scatter-Gather Engine)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        {nodes_code}
    }}
    """
    st.graphviz_chart(sim_dot, use_container_width=False)
    if required_shards > 3 or required_replicas > 2:
        st.caption("*(Topology preview truncated to first 3 shards and 2 replicas for visual clarity)*")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the architectural difference between Sharding and Replication in vector databases?"):
    st.markdown("""
    **Answer**:
    * **Sharding (Horizontal Partitioning)**: Slices the dataset into disjoint subsets distributed across multiple machines.
      * *Purpose*: Solves the **RAM storage ceiling**. If 50 million vectors require 400 GB RAM, but nodes only have 128 GB RAM, the data is split across 4 shards.
      * *Query Impact*: Requires **Scatter-Gather** (queries must visit all shards).
    * **Replication**: Maintains complete duplicate copies of an index.
      * *Purpose*: Solves **Search Throughput (QPS) and High Availability (HA)**. If 1 node handles 500 QPS, deploying 4 read replicas allows serving 2,000 QPS.
    """)

with st.expander("Q2: How does the 'Straggler Problem' degrade latency in distributed Scatter-Gather vector search?"):
    st.markdown("""
    **Answer**:
    In a cluster with 10 shards:
    * The Coordinator must receive candidates from **all 10 shards** before running the global priority queue merge.
    * If 9 shards complete in 2ms, but 1 shard encounters a momentary GC pause, lock contention, or network packet drop and takes 45ms:
    * **Total user query latency is 45ms!**
    * As the number of shards `M` increases, the probability that *at least one* shard experiences a latency spike approaches 100%.
    * **Mitigations**:
      1. Deploy Hedged Requests: If a shard doesn't respond within p95 time, re-issue the query to an alternate replica.
      2. Keep shard count moderate (prefer larger worker nodes with 128GB RAM over hundreds of tiny nodes).
    """)

with st.expander("Q3: Why can't a coordinator simply query 1 random shard instead of broadcasting to all shards?"):
    st.markdown("""
    **Answer**:
    Because high-dimensional vector similarity is an **unindexed global geometric optimization**:
    * Unlike a SQL database where records are partitioned by primary key (`WHERE user_id = 42`), a vector query does not know which shard holds the nearest neighbor prior to calculating distances.
    * The true closest vector could reside on Shard 1, Shard 2, or Shard 10.
    * To guarantee correctness and high recall, the coordinator **must query all shards** unless domain-specific routing (e.g. tenant-based partitioning) restricts the search space.
    """)

with st.expander("Q4: How does Tenant-Based Sharding optimize multi-tenant enterprise vector architectures?"):
    st.markdown("""
    **Answer**:
    In SaaS enterprise applications, users only query documents belonging to their organization (`tenant_id = 'org_42'`):
    * **Tenant-Based Routing**: Rather than random hash partitioning, all vectors for `org_42` are routed to Shard 3.
    * **Targeted Query Routing**: When a search arrives with `tenant_id = 'org_42'`, the coordinator sends the query **only to Shard 3**, bypassing the other 9 shards completely!
    * **Benefit**: Completely eliminates Scatter-Gather overhead and eliminates the Straggler Problem for multi-tenant workloads.
    """)

with st.expander("Q5: What consistency models are used for distributed vector index replication?"):
    st.markdown("""
    **Answer**:
    * **Eventual Consistency**: Primary node builds HNSW updates and asynchronously ships WAL segments to follower replicas. Replicas may serve queries missing updates for a few seconds. Highest read throughput.
    * **Raft Consensus (Strong/Linearizable Consistency)**: Systems like Qdrant and Milvus use Raft consensus to commit writes to a quorum of nodes before acknowledging success. Guarantees read-your-own-writes consistency at the cost of slight write latency.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Sharding When Data Fits in a Single Node's RAM**  
    Introducing 4 shards for a dataset of 500,000 vectors adds network RPC overhead and scatter-gather latency without any benefit. Only shard when RAM capacity forces it.
    """)
    st.warning("""
    **Trap 2: Forgetting to Provision Headroom for Ingestion Buffers**  
    Sizing worker node RAM at 100% of vector coordinates causes OOM crashes when new vectors are inserted during online indexing. Reserve 25% RAM for OS page cache and write buffers.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Hash Partitioning in Multi-Tenant Environments**  
    Spreading each customer's data across all 20 shards forces every query to scatter-gather across the entire cluster. Partition by tenant ID to isolate queries to single nodes.
    """)
    st.warning("""
    **Trap 4: Missing Read Replica Health Checks**  
    Routing traffic to a replica that is actively rebuilding its HNSW index causes severe latency spikes. Remove lagging replicas from the load balancer until indexing completes.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What does Sharding scale?"):
        st.markdown("""
        **RAM & Volume**:  
        Partitions the dataset across multiple nodes when total vectors exceed single-node memory.
        """)
    with st.expander("🗂️ Flashcard: What does Replication scale?"):
        st.markdown("""
        **QPS & Availability**:  
        Duplicates indexes across read-only nodes to multiply query throughput and provide failover.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How does Scatter-Gather work?"):
        st.markdown("""
        **Broadcast & Merge**:  
        Coordinator broadcasts vector to all shards, gathers local top-K hits, and merges via global min-heap.
        """)
    with st.expander("🗂️ Flashcard: What is the Straggler Problem?"):
        st.markdown("""
        **Slowest Node Bottleneck**:  
        Total scatter-gather search latency is dictated by whichever single shard takes the longest to respond.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural difference between Sharding (Volume) and Replication (QPS).", value=True)
chk2 = st.checkbox("I understand the distributed Scatter-Gather query protocol and global priority queue merging.", value=True)
chk3 = st.checkbox("I know how the Straggler Problem degrades p99 latency as shard counts increase.", value=True)
chk4 = st.checkbox("I can calculate required node counts based on dataset volume, vector dimensions, and QPS targets.", value=True)
chk5 = st.checkbox("I understand how Tenant-Based Routing bypasses scatter-gather broadcasts in multi-tenant architectures.", value=True)
chk6 = st.checkbox("I can architect a resilient, highly-available distributed vector search cluster in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completely mastered Chapter 18: Vector Database Performance and are ready for Chapter 19: Vector Database Evaluation!")
