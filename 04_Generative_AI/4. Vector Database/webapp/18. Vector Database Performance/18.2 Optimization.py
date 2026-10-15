import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="18.2 Vector DB Optimization - Vector DB Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("18.2 Vector Database Optimization: Batching, Index Tuning, Filtering & Semantic Caching")
st.caption("Chapter 18: Vector Database Performance | Domain: 4. Vector Database")
st.markdown("""
Deploying a default, out-of-the-box vector database is sufficient for small proof-of-concept prototypes. 
However, when scaling to enterprise workloads with millions of documents and high concurrency, unoptimized systems 
suffer from **slow ingestion bottlenecks, high search tail latencies, and bloated cloud bills**.

Production optimization requires tuning the complete data and inference path across **six core engineering pillars**:
1. **Batch Insertion Tuning**: Coalescing single writes into optimal 500–1,000 vector chunks to maximize I/O throughput.
2. **Batch Querying**: Slicing multiple queries into parallel SIMD tensor operations in a single database round-trip.
3. **HNSW Parameter Tuning**: Finding the sweet spot between `M`, `efConstruction`, and `efSearch`.
4. **Embedding Optimization**: Leveraging Matryoshka Representation Learning (MRL) and local inference runtimes.
5. **Metadata Index Pruning**: Eliminating post-filtering overhead by indexing high-cardinality metadata tags.
6. **Semantic Caching**: Intercepting incoming queries with a vector cache to return verified answers in **0.5 milliseconds**, bypassing LLM generation completely.

In this lesson, we master these six optimization strategies, implement an enterprise Semantic Cache in Python, 
and test an interactive Live Semantic Caching & Throughput Simulator.
""")

st.divider()

# Section 1: The Six Pillars of Vector Database Optimization
st.header("1. The Six Pillars of Vector Database Optimization", divider="rainbow")

col_pil1, col_pil2 = col_pil1, col_pil2 = st.columns([1.1, 0.9])

with col_pil1:
    st.markdown("""
    ### Deep Dive into the 6 Pillars
    
    * **1. Batch Insertion**:
      * Individual `.add()` calls trigger separate disk syncs (WAL commits).
      * Batching 500–1,000 records together increases ingestion throughput by **30x to 50x** (from 60 VPS to 3,500+ VPS).
    
    * **2. Batch Querying**:
      * Passing `query_texts=["Q1", "Q2", "Q3"]` executes parallel matrix multiplication on CPU/GPU, eliminating network round-trip overhead.
    
    * **3. HNSW Parameter Tuning**:
      * Higher `efConstruction` (e.g. 128) builds higher-quality graph connectivity offline, allowing runtime `efSearch` to remain lean (e.g. 48) for sub-2ms query times.
    
    * **4. Embedding Optimization**:
      * Matryoshka Representation Learning (MRL) shrinks vectors from 1,536 to 512 dimensions with <1% loss in accuracy, slashing RAM by 66%.
    
    * **5. Payload / Metadata Indexing**:
      * Ensuring fields used in `where` filters (e.g. `tenant_id`, `department`) have dedicated B-Tree indices to prevent table scans.
    
    * **6. Semantic Caching**:
      * Storing previous `(Query_Vector, Generated_Answer)` pairs. If a new query has `Cosine_Distance < 0.08` to a cached query, return the cached answer immediately!
    """)

with col_pil2:
    st.markdown("### Semantic Cache Interception Architecture")
    cache_dot = """
    digraph SemanticCache {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="Incoming User Query\\n'How to setup Envoy mTLS?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        SemCache [label="1. Semantic Vector Cache\\nNearest neighbor search in Cache Index\\n(Threshold: Cosine Distance < 0.08)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        CacheHit [label="CACHE HIT! (< 1ms)\\nReturn cached response\\nCost: $0.00 | Latency: 0.8ms", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        subgraph cluster_full {
            label="Cache Miss Fallback Path"; style="dashed"; color="#DC2626";
            VDB [label="2. ChromaDB Main Store\\nRetrieve Top-K Chunks (~2ms)", fillcolor="#FEE2E2"];
            LLM [label="3. Synthesize via LLM\\nStream Answer (~750ms)", fillcolor="#FEE2E2"];
            StoreCache [label="4. Store (Query, Answer) in Cache", fillcolor="#FEE2E2"];
            VDB -> LLM -> StoreCache;
        }

        UserQ -> SemCache;
        SemCache -> CacheHit [label="Distance < 0.08", color="#059669", style="bold"];
        SemCache -> VDB [label="Distance >= 0.08 (Miss)", color="#DC2626"];
        StoreCache -> SemCache [style="dotted"];
    }
    """
    st.graphviz_chart(cache_dot, use_container_width=False)

st.divider()

# Section 2: Building an Enterprise Semantic Cache in Python
st.header("2. Building an Enterprise Semantic Cache in Python", divider="rainbow")

st.markdown("""
Here is a complete, production-grade **Semantic Cache** implementation using ChromaDB:
""")

st.code("""import chromadb
import uuid
from typing import Optional, Tuple

class SemanticQueryCache:
    \"\"\"Ultra-fast semantic cache for RAG systems using vector proximity.\"\"\"
    
    def __init__(self, cache_distance_threshold: float = 0.08):
        self.client = chromadb.PersistentClient(path="./semantic_cache_store")
        # Initialize specialized cache collection
        self.cache_collection = self.client.get_or_create_collection(
            name="query_cache",
            metadata={"hnsw:space": "cosine"}
        )
        self.threshold = cache_distance_threshold

    def lookup(self, query: str) -> Optional[Tuple[str, float]]:
        \"\"\"Checks if a semantically equivalent query exists in cache.\"\"\"
        if self.cache_collection.count() == 0:
            return None
            
        results = self.cache_collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results["distances"][0]:
            dist = results["distances"][0][0]
            if dist <= self.threshold:
                cached_answer = results["metadatas"][0][0]["cached_answer"]
                return cached_answer, dist
        return None

    def store(self, query: str, answer: str):
        \"\"\"Persists a new query-answer pair in the semantic cache.\"\"\"
        self.cache_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[query],
            metadatas=[{"cached_answer": answer}]
        )
""", language="python")

st.divider()

# Section 3: Ingestion Throughput Tuning Matrix
st.header("3. Ingestion Throughput Tuning Matrix", divider="rainbow")

st.markdown("""
Choosing the right batch size during data ingestion fundamentally dictates migration duration:
""")

INGESTION_BENCHMARKS = [
    {
        "Batch Size (Chunks)": "1 (Single Insert)",
        "Ingestion Throughput": "55 Vectors / Sec",
        "1M Vector Duration": "5.05 Hours",
        "Disk I/O Character": "Massive SQLite fsync bottleneck; disk head thrashing.",
        "Production Status": "❌ Forbidden in production."
    },
    {
        "Batch Size (Chunks)": "100",
        "Ingestion Throughput": "1,420 Vectors / Sec",
        "1M Vector Duration": "11.7 Minutes",
        "Disk I/O Character": "Efficient sequential WAL writes.",
        "Production Status": "🟡 Acceptable for small updates."
    },
    {
        "Batch Size (Chunks)": "500 - 1,000",
        "Ingestion Throughput": "4,250 Vectors / Sec",
        "1M Vector Duration": "3.9 Minutes",
        "Disk I/O Character": "Optimal NVMe bus saturation; balanced RAM buffers.",
        "Production Status": "🏆 Production Gold Standard."
    },
    {
        "Batch Size (Chunks)": "10,000+",
        "Ingestion Throughput": "2,100 Vectors / Sec",
        "1M Vector Duration": "7.9 Minutes",
        "Disk I/O Character": "Process memory spikes; triggers Python GC pauses.",
        "Production Status": "⚠️ Diminishing returns (OOM risk)."
    }
]

st.dataframe(pd.DataFrame(INGESTION_BENCHMARKS), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Live Semantic Caching & Optimization Lab
st.header("4. Interactive Workbench: Live Semantic Caching Lab", divider="rainbow")

st.markdown("""
Test how a Semantic Cache intercepts queries. Submit an exact query, a paraphrased variation, 
or a completely new question to observe cache hit dynamics, latency differences, and financial savings.
""")

col_cache_q, col_cache_cfg = col_cache_q, col_cache_cfg = st.columns([1.1, 0.9])

with col_cache_q:
    st.subheader("1. Test Search Inquiry")
    query_scenario = st.selectbox(
        "Choose Query Submission:",
        [
            "Existing Query: 'How do I configure mutual TLS in Envoy proxy?'",
            "Paraphrased Variation: 'What are the steps to enable Envoy mTLS authentication?'",
            "Slightly Different: 'How do I disable mTLS client cert validation?'",
            "Unrelated Question: 'What is the company policy for travel meal expense?'"
        ]
    )

with col_cache_cfg:
    st.subheader("2. Semantic Cache Knobs")
    cache_threshold = st.slider(
        "Cache Distance Hit Threshold (`hnsw:space: cosine`):",
        min_value=0.02,
        max_value=0.25,
        value=0.08,
        step=0.01,
        help="Queries with distance smaller than this threshold trigger an instant Cache Hit."
    )

# Simulated Semantic Cache Entry
CACHED_ENTRY = {
    "cached_query": "How do I configure mutual TLS in Envoy proxy?",
    "cached_answer": "To configure mTLS in Envoy, mount certificates in /etc/envoy/certs and set require_client_certificate: true on downstream listeners.",
    "created_at": "10 mins ago",
    "hits_count": 42
}

# Distance calculations
if "Existing Query" in query_scenario:
    sim_dist = 0.000
elif "Paraphrased" in query_scenario:
    sim_dist = 0.048
elif "Slightly Different" in query_scenario:
    sim_dist = 0.165
else:
    sim_dist = 0.812

is_cache_hit = sim_dist <= cache_threshold

st.divider()

col_exec_view, col_exec_stats = col_exec_view, col_exec_stats = st.columns([1.2, 0.8])

with col_exec_view:
    st.subheader("Execution Trace & Response")
    
    if is_cache_hit:
        st.success(f"🟢 **CACHE HIT!** (Cosine Distance: `{sim_dist:.4f}` <= `{cache_threshold:.2f}`)")
        st.markdown(f"**Matched Cached Question**: *\"{CACHED_ENTRY['cached_query']}\"*")
        st.info(f"**Returned Verified Answer**:\n\n{CACHED_ENTRY['cached_answer']}")
        st.caption("Execution Path: **Bypassed ChromaDB Main Store and Bypassed LLM Generation entirely.**")
    else:
        st.error(f"🔴 **CACHE MISS** (Cosine Distance: `{sim_dist:.4f}` > `{cache_threshold:.2f}`)")
        st.markdown("Query is sufficiently distinct from existing cached entries. Executing full pipeline:")
        st.markdown("""
        1. Query ChromaDB Vector Index (Retrieved 20 candidates in 2.1 ms)
        2. Rerank via Cross-Encoder (Selected top 3 in 18.4 ms)
        3. Stream Generation from LLM (Generated 140 tokens in 680 ms)
        4. Stored new `(Query, Answer)` pair into Semantic Cache.
        """)

with col_exec_stats:
    st.subheader("Latency & Financial Audit")
    
    lat_val = "0.8 ms" if is_cache_hit else "700.5 ms"
    cost_val = "$0.0000" if is_cache_hit else "$0.0065"
    
    st.metric(label="Total Query Latency", value=lat_val, delta="-99.8% Latency Reduction" if is_cache_hit else "Full Generation")
    st.metric(label="Query Cost", value=cost_val, delta="100% Cost Savings" if is_cache_hit else "Incurred API Fee")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is a Semantic Cache, and how does it differ from a standard Redis exact-key cache?"):
    st.markdown("""
    **Answer**:
    * **Exact-Key Cache (Standard Redis)**: Hashes the exact query string (`SHA256("how to setup mTLS")`). If a user asks *"steps to configure mTLS"*, the string hashes do not match, causing a cache miss.
    * **Semantic Cache (Vector Cache)**: Embeds the user query into a high-dimensional vector and searches a dedicated vector index of past queries. If the **cosine distance is below a calibrated threshold** (e.g. `distance < 0.08`), it recognizes that the user is asking the identical conceptual question, returning the cached answer in **sub-millisecond time**.
    * **Impact**: In enterprise customer support, 30% to 60% of user queries are semantically repetitive. A semantic cache slashes LLM API bills by 40%+ while delivering instantaneous responses.
    """)

with st.expander("Q2: Why does batch insertion (`batch_size = 500`) achieve 50x higher throughput than single-item insertion in ChromaDB?"):
    st.markdown("""
    **Answer**:
    ChromaDB persists metadata and collection state in SQLite using Write-Ahead Logging (WAL):
    * Calling `.add()` with 1 document opens a SQLite transaction, executes an `fsync()` system call to flush bytes to physical storage, and updates the HNSW graph. Disk disk I/O and kernel sync overhead dominate runtime (~15ms per write).
    * Calling `.add()` with 500 documents wraps all 500 vector writes and metadata updates into a **single atomic transaction**, executing only one `fsync()` for the entire batch.
    * Furthermore, the C++ HNSW graph insertion engine can utilize multi-threaded vectorized SIMD instructions to update graph edges in parallel.
    """)

with st.expander("Q3: What are the risks of setting the Semantic Cache distance threshold too loose (e.g. threshold = 0.20)?"):
    st.markdown("""
    **Answer**:
    Setting the cache hit threshold too loose creates **Semantic False Positive Collisions**:
    * A user asks: *"How do I configure mutual TLS in Envoy?"* (Cached answer: set `require_client_certificate: true`).
    * Later, another user asks: *"How do I disable mutual TLS in Envoy?"*
    * At `threshold = 0.20`, the cache triggers a false positive hit, serving the instructions for *enabling* mTLS to a user who asked to *disable* it!
    * **Best Practice**: Keep semantic cache thresholds extremely tight (**`0.05 to 0.08`** in cosine distance), and include an automated cache invalidation policy (TTL or document update triggers).
    """)

with st.expander("Q4: What is Matryoshka Representation Learning (MRL), and how does it optimize vector database memory?"):
    st.markdown("""
    **Answer**:
    Historically, truncating a 1,536-dimensional vector to 512 dimensions destroyed its semantic meaning.
    * **Matryoshka Representation Learning (MRL)** (used in OpenAI `text-embedding-3` and Nomic Embed) trains models so that the **most critical information is packed into the first `D` dimensions** like Russian nesting dolls.
    * Developers can slice vectors to 512 or 256 dimensions simply by truncating the array: `truncated_vec = full_vec[:512]`.
    * **Benefit**: Reduces vector storage and RAM by **66% to 83%**, accelerates distance calculation SIMD throughput by 3x, while retaining 98%+ of original retrieval recall!
    """)

with st.expander("Q5: What is the difference between Pre-Filtering and Single-Stage (Iterative) Filtering in vector search?"):
    st.markdown("""
    **Answer**:
    * **Pre-Filtering**: Executes the SQL/metadata filter first to find qualifying document IDs, then executes vector search only on those IDs. If the filter matches only 5 documents out of 1,000,000, graph search fails because the HNSW graph is disconnected across the 5 points.
    * **Single-Stage / In-Graph Filtering**: Traverses the standard HNSW graph, but during neighbor exploration, any node failing the metadata predicate is ignored as a candidate while still being traversed as a routing step. This preserves graph connectivity and guarantees sub-millisecond filtered search.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Loosely Calibrated Semantic Cache Thresholds**  
    Setting cache hit distance to 0.20 causes the cache to return answers to questions with opposite intent (e.g. enable vs disable). Keep thresholds strictly under 0.08.
    """)
    st.warning("""
    **Trap 2: Stale Semantic Cache Invalidation**  
    When an underlying technical runbook is updated in the vector database, forgetting to flush the semantic cache causes users to receive outdated, deprecated answers. Flush cache on document upserts.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Single-Item Ingestion in Web Handlers**  
    Inserting documents one-by-one synchronously locks SQLite and caps throughput at 50 VPS. Always buffer and insert in batches of 500.
    """)
    st.warning("""
    **Trap 4: Unindexed High-Cardinality Metadata**  
    Filtering by a user ID or tenant ID without an index forces the database to scan entire metadata tables. Ensure filter keys are indexed.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Semantic Cache?"):
        st.markdown("""
        **Vector Cache**:  
        Intercepts incoming queries using vector proximity; returns verified answers in <1ms if cosine distance < 0.08.
        """)
    with st.expander("🗂️ Flashcard: What is the optimal batch ingestion size?"):
        st.markdown("""
        **500 to 1,000 Vectors**:  
        Achieves optimal NVMe disk I/O throughput (3,500+ VPS) without memory exhaustion.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Matryoshka Embeddings (MRL)?"):
        st.markdown("""
        **Nesting Compression**:  
        Packs primary semantics into early dimensions, allowing slicing 1536 -> 512 dims to slash RAM by 66%.
        """)
    with st.expander("🗂️ Flashcard: What is Single-Stage Filtering?"):
        st.markdown("""
        **In-Graph Traversal**:  
        Navigates HNSW graph while evaluating metadata predicates at each node, preserving graph connectivity.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the six pillars of vector database performance optimization.", value=True)
chk2 = st.checkbox("I know why batch insertion (500-1000 chunks) delivers 50x higher throughput than single writes.", value=True)
chk3 = st.checkbox("I can implement an enterprise Semantic Cache in Python using ChromaDB vector search.", value=True)
chk4 = st.checkbox("I understand the risks of loose semantic cache thresholds and how to prevent false positive hits.", value=True)
chk5 = st.checkbox("I know how Matryoshka Representation Learning (MRL) slashes RAM by 66% with negligible recall loss.", value=True)
chk6 = st.checkbox("I can architect an end-to-end optimized vector pipeline that balances caching, batching, and latency.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Vector Database Optimization and are ready for Chapter 18.3: Scaling Vector Databases!")
