import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.4 When to Use Which - Vector DB Curriculum",
    page_icon="🧭",
    layout="wide"
)

# Title & Overview
st.title("4.4 When to Use Which: Relational, Vector & Hybrid Polyglot Architectures")
st.caption("Chapter 4: Vector Database vs Traditional Databases | Domain: 4. Vector Database")
st.markdown("""
In enterprise software engineering, the question is almost never *"Will Vector Databases replace Relational Databases?"*—the answer is an emphatic **NO**.

Relational databases remain the unassailable foundation for **transactional systems of record**, **financial ledgers**, 
and **relational business entities**. Vector databases excel at **unstructured semantic retrieval**, **RAG context hydration**, 
and **multimodal search**.

The true engineering challenge lies in **workload classification** and **system integration**: knowing when to use 
an RDBMS, when to use a dedicated Vector DB, when to consolidate into a single database with `pgvector`, 
and when to implement a **decoupled Polyglot Persistence architecture** using Change Data Capture (CDC).
""")

st.divider()

# Section 1: Workload Classification Framework
st.header("1. Enterprise Workload Classification: RDBMS vs. Vector DB", divider="rainbow")

col_work1, col_work2 = col_work1, col_work2 = st.columns([1, 1])

with col_work1:
    st.markdown("""
    ### 1. When to Use a Traditional Relational Database (RDBMS)
    * **Strict ACID Financial Transactions**:
      * Banking ledgers, double-entry accounting, stock trading, and payment processing where **zero data loss** and absolute serializability are legally required.
    * **Complex Relational Entities**:
      * Data models with heavy parent-child relationships requiring multi-table `JOIN` operations: `Customers -> Orders -> LineItems -> Invoices`.
    * **Deterministic Exact Filtering**:
      * Filtering by exact scalar values: `WHERE status = 'PAID' AND created_at >= '2024-01-01' AND total_amount > 500`.
    * **Structured Analytics & Aggregations**:
      * Grouping and aggregation queries across millions of rows: `SELECT country, SUM(revenue) GROUP BY country`.
    """)

with col_work2:
    st.markdown("""
    ### 2. When to Use a Dedicated Vector Database
    * **Retrieval-Augmented Generation (RAG)**:
      * Hydrating relevant knowledge chunks for Large Language Models (LLMs) to eliminate hallucinations.
    * **Semantic & Intent-Based Search**:
      * Customer service ticket deflection, documentation portals, and knowledge bases where users search with concepts rather than exact keywords.
    * **Multimodal Visual & Audio Search**:
      * Reverse image search, visual retail catalog matching, audio acoustic fingerprinting.
    * **High-Throughput Recommender Systems**:
      * Sub-10ms candidate generation over 100M+ items using Two-Tower user/item embedding dot products.
    * **Autonomous AI Agent Memory**:
      * Long-term episodic memory buffers for conversational AI agents.
    """)

st.markdown("### Master Workload Decision Matrix")
decision_matrix_df = pd.DataFrame({
    "Workload Characteristic": [
        "Data Nature",
        "Primary Query Style",
        "Consistency Requirement",
        "Query Matching",
        "Scale Dynamics",
        "Recommended Database"
    ],
    "Relational Workload (OLTP)": [
        "Structured tables, normalized schemas",
        "Declarative SQL (`JOIN`, `GROUP BY`)",
        "Strict ACID (Immediate Consistency)",
        "100% Deterministic exact boolean",
        "Vertical scaling / Read replicas",
        "PostgreSQL / MySQL / Oracle"
    ],
    "AI / Semantic Workload": [
        "Unstructured text, images, audio",
        "Proximity search (`Top-K nearest`)",
        "Eventual Consistency (Acceptable)",
        "Probabilistic Approximate (ANN)",
        "Horizontal sharding across segments",
        "Qdrant / Milvus / Pinecone"
    ],
    "Hybrid Enterprise Workload": [
        "Structured business metadata + Raw text",
        "Hybrid scalar filter + Vector search",
        "ACID on business data, fast ANN on search",
        "Single-stage filtered ANN retrieval",
        "Decoupled Polyglot Architecture",
        "PostgreSQL + CDC + Vector DB"
    ]
})
st.dataframe(decision_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Hybrid Polyglot Persistence Architectures
st.header("2. Modern Hybrid Architecture: Decoupled Polyglot Persistence", divider="rainbow")

st.markdown("""
In production enterprise architectures, the gold standard is **Polyglot Persistence**: 
using each database for what it does best.

* **PostgreSQL** serves as the **System of Record (Source of Truth)**, handling user authentication, payments, and transactional updates with strict ACID guarantees.
* **Change Data Capture (CDC via Debezium / Kafka)** captures insert and update events in real time.
* An **Asynchronous Embedding Worker** vectorizes text chunks.
* **Qdrant / Milvus** serves as the **Fast Search Index**, answering semantic queries in single-digit milliseconds.
""")

poly_dot = """
digraph PolyglotArchitecture {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Client [label="Enterprise Web App\\nUsers & Business Logic", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_write_path {
        label="Write Path (System of Record)"; style="dashed"; color="#2563EB";
        PostgresDB [label="PostgreSQL (RDBMS)\\nPrimary System of Record\\nACID Transactions", fillcolor="#DBEAFE", color="#1D4ED8", style="filled,bold"];
        WALLog [label="Postgres WAL Engine\\nLogical Replication Stream", fillcolor="#BFDBFE"];
        KafkaBus [label="Debezium CDC + Kafka\\nEvent-Driven Message Bus", fillcolor="#93C5FD"];
        EmbedWorker [label="Async Embedding Worker\\n(Batch Vectorization)", fillcolor="#BFDBFE"];
    }

    subgraph cluster_read_path {
        label="Read & Search Path (Vector Index)"; style="dashed"; color="#059669";
        VectorDB [label="Vector DB (Qdrant / Milvus)\\nDedicated HNSW ANN Index\\nSub-5ms Semantic Search", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        RAGService [label="GenAI / RAG Service\\nLLM Context Hydration", fillcolor="#A7F3D0"];
    }

    Client -> PostgresDB [label="1. Write Transaction (ACID)"];
    PostgresDB -> WALLog -> KafkaBus -> EmbedWorker;
    EmbedWorker -> VectorDB [label="2. Async Upsert Vector"];
    
    Client -> RAGService [label="3. Semantic Search Query"];
    RAGService -> VectorDB [label="4. Top-K Vector Search"];
    VectorDB -> RAGService [label="5. Return Chunks"];
    RAGService -> Client [label="6. Answer to User"];
}
"""
st.graphviz_chart(poly_dot, use_container_width=False)

st.divider()

# Section 3: The Architectural Trade-Off Spectrum
st.header("3. Architectural Trade-Off Spectrum: Single DB vs. Polyglot", divider="rainbow")

st.markdown("""
Should your organization maintain a single database using an extension like `pgvector`, 
or deploy a dedicated decoupled vector database cluster?
""")

col_opt1, col_opt2 = col_opt1, col_opt2 = st.columns([1, 1])

with col_opt1:
    st.markdown("""
    ### Option A: Consolidated Single DB (`PostgreSQL + pgvector`)
    * **Best For**:
      * Small to medium engineering teams.
      * Collection sizes under **5 to 10 million vectors**.
      * Applications where query traffic is under **100 QPS**.
      * Systems requiring atomic transactions joining business tables and vector embeddings.
    * **Pros**:
      * Zero distributed infrastructure overhead; single backup and migration pipeline; familiar SQL toolchain.
    * **Cons**:
      * Memory competition: In-memory HNSW graphs compete directly with PostgreSQL buffer cache.
      * Vector search compute spikes can stall business transactions.
    """)

with col_opt2:
    st.markdown("""
    ### Option B: Decoupled Polyglot Architecture (`Postgres + Vector DB`)
    * **Best For**:
      * Large scale: **10M to billions of vectors**.
      * High query throughput: **500 to 10,000+ QPS**.
      * Dedicated GenAI teams operating independently from core transactional backend teams.
    * **Pros**:
      * Compute isolation: Heavy SIMD/GPU vector queries never degrade primary business checkouts.
      * Cloud-native scaling: Scale search nodes horizontally independently of storage.
    * **Cons**:
      * Dual-system operational overhead; eventual consistency lag between RDBMS write and Vector DB index.
    """)

st.divider()

# Section 4: Interactive Architecture Selector & Blueprint Recommender
st.header("4. Interactive Architecture Selector & Blueprint Recommender", divider="rainbow")

st.markdown("""
Configure your application requirements to receive an architectural recommendation, 
complete with operational complexity ratings, estimated infrastructure cost, and implementation blueprint:
""")

col_sel_in, col_sel_out = st.columns([1, 1])

with col_sel_in:
    st.subheader("Application Requirements")
    sel_vector_count = st.select_slider(
        "Anticipated Vector Collection Size",
        options=[10000, 100000, 1000000, 5000000, 20000000, 100000000],
        value=1000000,
        format_func=lambda x: f"{x:,} vectors"
    )
    sel_qps = st.select_slider(
        "Peak Search Query Throughput (QPS)",
        options=[5, 25, 100, 500, 2500, 10000],
        value=100
    )
    sel_strict_acid = st.checkbox("Requires Immediate ACID Transactions on Business Data", value=True)
    sel_team_size = st.radio("Engineering & DevOps Team Size", ["Solo / Small Team (1-4 devs)", "Mid-Size Team (5-15 devs)", "Enterprise / Platform Team (15+ devs)"], index=1)

with col_sel_out:
    st.subheader("Recommended Architectural Blueprint")
    
    if sel_vector_count <= 5000000 and sel_qps <= 100 and "Solo" in sel_team_size:
        rec_title = "Consolidated Single Database: PostgreSQL + pgvector"
        rec_desc = """
        * **Architecture**: Deploy a single PostgreSQL instance with the `pgvector` extension enabled.
        * **Operational Complexity**: **Low (1/5)** — Single database to backup, monitor, and deploy.
        * **Estimated Infra Cost**: **$50 - $150 / month** on managed RDS or Supabase.
        * **Trade-off**: Simplicity wins. At under 5M vectors and 100 QPS, managing two separate databases is unwarranted over-engineering.
        """
        rec_color = "success"
    elif sel_vector_count >= 20000000 or sel_qps >= 500:
        rec_title = "Decoupled Polyglot Architecture: PostgreSQL + Debezium CDC + Qdrant / Milvus"
        rec_desc = """
        * **Architecture**: PostgreSQL as ACID System of Record -> Debezium CDC -> Kafka -> Dedicated Vector DB Cluster.
        * **Operational Complexity**: **High (4/5)** — Requires distributed message broker and dual cluster monitoring.
        * **Estimated Infra Cost**: **$600 - $2,500 / month** (Dedicated compute nodes for HNSW in RAM).
        * **Trade-off**: Mandatory at scale. Compute isolation guarantees that 10k QPS search spikes never bring down core payment transactions.
        """
        rec_color = "warning"
    else:
        rec_title = "Managed Hybrid Pattern: PostgreSQL + Managed Pinecone / Qdrant Cloud"
        rec_desc = """
        * **Architecture**: PostgreSQL for relational transactional tables + Managed Serverless Vector DB for semantic search.
        * **Operational Complexity**: **Moderate (2.5/5)** — Managed cloud vector service eliminates Kubernetes cluster maintenance.
        * **Estimated Infra Cost**: **$200 - $600 / month**.
        * **Trade-off**: Balanced approach. Offloads vector compute and memory sizing to managed SaaS while keeping relational data clean.
        """
        rec_color = "info"
        
    st.markdown(f"### {rec_title}")
    st.markdown(rec_desc)

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Polyglot Relational + Vector Synchronization Simulator", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Polyglot Persistence System** demonstrating how 
a transactional relational table writes to an append-only transaction log and triggers **Change Data Capture (CDC)** 
to asynchronously synchronize an Approximate Nearest Neighbor vector search index:
""")

code_polyglot = '''import numpy as np
import time
from typing import Dict, Any, List

class RelationalSystemOfRecord:
    """Primary transactional RDBMS storing business entities."""
    def __init__(self):
        self.table: Dict[int, Dict[str, Any]] = {}
        self.cdc_stream: List[Dict[str, Any]] = []

    def commit_transaction(self, record_id: int, title: str, category: str, body: str):
        """Executes an ACID transactional write and publishes a CDC event."""
        # 1. Write to transactional table
        record = {
            "id": record_id,
            "title": title,
            "category": category,
            "body": body,
            "updated_at": time.time()
        }
        self.table[record_id] = record

        # 2. Append event to CDC stream (Simulating Debezium / Kafka event)
        cdc_event = {
            "op": "UPSERT",
            "id": record_id,
            "payload": record
        }
        self.cdc_stream.append(cdc_event)
        return record

class VectorSearchIndex:
    """Dedicated secondary search index."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[int, np.ndarray] = {}
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def sync_from_cdc(self, cdc_event: Dict[str, Any]):
        """Consumes CDC events and updates the vector index."""
        rec_id = cdc_event["id"]
        payload = cdc_event["payload"]
        
        # Simulating vector embedding generation
        np.random.seed(abs(hash(payload["body"])) % (2**32))
        vec = np.random.randn(self.dimension)
        norm = np.linalg.norm(vec)
        unit_vec = vec / (norm if norm > 0 else 1.0)

        # Index vector and attach searchable metadata
        self.vectors[rec_id] = unit_vec
        self.metadata[rec_id] = {
            "title": payload["title"],
            "category": payload["category"]
        }

    def search(self, query_vec: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """Executes ANN Cosine Search."""
        q_norm = query_vec / np.linalg.norm(query_vec)
        scores = {}
        for rid, vec in self.vectors.items():
            scores[rid] = float(np.dot(vec, q_norm))
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": rid, "score": sc, "meta": self.metadata[rid]} for rid, sc in ranked]

# Demonstration of Polyglot Synchronization Flow
rdb = RelationalSystemOfRecord()
vdb = VectorSearchIndex(dimension=32)

print("1. Executing Transaction in PostgreSQL System of Record...")
rdb.commit_transaction(
    record_id=101,
    title="Kubernetes Ingress Controller Setup",
    category="DevOps",
    body="Configure NGINX ingress with TLS cert-manager on EKS."
)

print(f"PostgreSQL Committed Rows: {len(rdb.table)}")
print(f"CDC Stream Event Count: {len(rdb.cdc_stream)}")

print("\\n2. Change Data Capture (CDC) Worker Synchronizing Vector DB...")
while rdb.cdc_stream:
    event = rdb.cdc_stream.pop(0)
    vdb.sync_from_cdc(event)

print(f"Vector Index Synchronized! Total Indexed Vectors: {len(vdb.vectors)}")

# Query Search Index
dummy_q = np.random.randn(32)
hits = vdb.search(dummy_q, top_k=1)
print(f"\\n3. Vector DB Search Hit: ID={hits[0]['id']} | Title='{hits[0]['meta']['title']}'")
'''

st.code(code_polyglot, language="python")

if st.button("🚀 Execute Polyglot CDC Synchronization Simulator", type="primary"):
    exec_scope = {}
    exec(code_polyglot, globals(), exec_scope)
    
    r_table = exec_scope["rdb"].table
    v_meta = exec_scope["vdb"].metadata
    hits_out = exec_scope["hits"]
    
    st.success("Polyglot Pipeline synchronized and verified successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("**PostgreSQL System of Record:**")
        st.json(r_table)
    with col_out2:
        st.markdown("**Synchronized Vector DB Metadata & Hit:**")
        st.json(hits_out)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the 'Dual-Write Problem' when synchronizing an RDBMS with a Vector Database, and how does CDC solve it?"""
a1 = """* **The Dual-Write Problem**:
  * An application writes to both PostgreSQL and a Vector DB sequentially:
    `db.save(user_doc); vector_db.upsert(doc_vector);`
  * If the network fails or the vector DB crashes after the PostgreSQL write, the two systems fall permanently out of sync.
  * If wrapped in a single transaction, the external vector DB call holds the database transaction open, exhausting connection pools.
* **How Change Data Capture (CDC) Solves It**:
  * The application writes strictly to PostgreSQL (guaranteeing atomic ACID durability).
  * A CDC agent (e.g. Debezium) tails PostgreSQL's Write-Ahead Log (WAL) and publishes events to a durable Kafka queue.
  * An asynchronous consumer reads Kafka and upserts vectors with at-least-once delivery and exponential retries, guaranteeing eventual consistency without dual-write race conditions."""

q2 = """2. When is it architectural over-engineering to deploy a dedicated vector database instead of `pgvector`?"""
a2 = """* It is over-engineering when:
  1. Your dataset is under **1 million vectors** (which easily fits into modest PostgreSQL RAM).
  2. Your team consists of only 1 to 3 engineers without dedicated platform/SRE staff.
  3. Your query rate is low to moderate (< 50 QPS).
  4. Your application requires strict transactional consistency between relational rows and vectors (e.g. deleting a user account must immediately delete their embedding).
* In these scenarios, `pgvector` inside PostgreSQL provides 100% of the needed functionality with zero added operational overhead."""

q3 = """3. How does Eventual Consistency impact user experience in a decoupled Polyglot vector search architecture?"""
a3 = """* In a decoupled architecture (Postgres -> Kafka -> Vector DB), there is a replication lag (typically 200ms to 2 seconds).
* **The "Read-Your-Own-Writes" Hazard**:
  * A user uploads a new contract or writes a support ticket, and immediately searches for it in the search bar.
  * If the vector embedding job has not finished, the search query returns zero results, causing the user to believe their upload failed!
* **Mitigation**:
  * Return optimistic UI confirmations.
  * Query the primary transactional database directly for recently created entities (`created_at < 5 minutes`), merging them with vector search results."""

q4 = """4. What is Reciprocal Rank Fusion (RRF), and how does it combine results from hybrid search engines?"""
a4 = """* Different search engines output scores on completely incompatible scales (BM25 outputs unbound positive scores like `14.2`, while Cosine Similarity outputs `[0.0, 1.0]`).
* **Reciprocal Rank Fusion (RRF)** combines them using rank position rather than raw score:
  `RRF_Score(d) = sum(1.0 / (k_const + Rank_i(d)))` across all search engines (typically `k_const = 60`).
* Documents that appear near the top of both the lexical search engine (BM25) and the vector database receive the highest combined score, providing the ultimate hybrid retrieval precision."""

q5 = """5. Why does high-volume vector search consume disproportionately more CPU and RAM than standard OLTP database queries?"""
a5 = """* **OLTP Relational Queries**: Typically traverse 3 to 4 B-Tree pointer pages in RAM (a few microseconds of CPU time) and return a discrete row.
* **Vector ANN Search**: Evaluates floating-point dot products across hundreds of 1536-dimensional vectors per query.
* A single query requires hundreds of thousands of SIMD multiply-accumulate operations and random memory lookups across graph edges.
* At 1,000 QPS, this demands dozens of dedicated CPU cores running at 100% utilization. Isolating this workload in a dedicated vector cluster prevents it from starving transactional business systems."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Architecture Decision Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Direct Application Dual-Write Trap**  
> Writing `db.insert(); vector_db.insert()` directly in your REST API handler will inevitably lead to database desynchronization. 
> When the vector database experiences a momentary network timeout, your primary database will contain documents that 
> are completely missing from search. Always use transactional outbox patterns or Change Data Capture (CDC).

> [!WARNING]
> **Trap 2: Premature Microservices with Dedicated Vector Clusters**  
> Small startups often deploy Milvus on a multi-node Kubernetes cluster for 50,000 document chunks, spending 
> $800/month on idle infrastructure and 20 hours/week debugging pod networking. 
> Start with PostgreSQL and `pgvector` or managed serverless vector databases; decouple only when scale demands it.

> [!WARNING]
> **Trap 3: Ignoring Metadata Synchronization Overhead**  
> If document permissions (`access_roles: ["finance_admin"]`) change in PostgreSQL, but are not synchronized 
> immediately to the vector database metadata payload, users may access unauthorized context chunks in RAG responses! 
> Enforce permission check filtering at the application layer or ensure CDC syncs permission updates in real time.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: When Use RDBMS vs Vector DB?"):
        st.markdown("""
        **System Boundary**:  
        Use RDBMS for transactional systems of record, financial ledgers, and exact relational joins; use Vector DB for semantic search, RAG, and multimodal retrieval.
        """)
    with st.expander("🗂️ Flashcard: What is Polyglot Persistence?"):
        st.markdown("""
        **Polyglot Persistence**:  
        Using multiple specialized database technologies in a single architecture (e.g., PostgreSQL for transactions, Qdrant for semantic search).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How Does CDC Prevent Dual-Write Bugs?"):
        st.markdown("""
        **Change Data Capture (CDC)**:  
        Tails the primary database Write-Ahead Log (WAL) to asynchronously replicate updates to search indexes, eliminating dual-write failures.
        """)
    with st.expander("🗂️ Flashcard: What is Reciprocal Rank Fusion (RRF)?"):
        st.markdown("""
        **RRF Fusion**:  
        A rank-based scoring algorithm that seamlessly blends results from lexical search engines (BM25) and vector databases without score normalization.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can clearly define when to use an RDBMS, a Vector DB, or a Hybrid architecture.", value=True)
chk2 = st.checkbox("I understand the Dual-Write Problem and how Change Data Capture (CDC) solves it.")
chk3 = st.checkbox("I know when to choose `pgvector` inside PostgreSQL versus deploying a dedicated vector database.")
chk4 = st.checkbox("I understand why Polyglot Persistence isolates heavy vector SIMD compute from transactional systems.")
chk5 = st.checkbox("I know how Reciprocal Rank Fusion (RRF) merges hybrid lexical and vector search results.")
chk6 = st.checkbox("I can implement a simulated CDC pipeline synchronizing an RDBMS with a vector index in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered When to Use Which and completed Chapter 4 of the Vector Database curriculum!")
