import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.2 Vector Database Differences - Vector DB Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("4.2 Vector Database Differences: Unstructured Data, Approximate Matching & Semantic Retrieval")
st.caption("Chapter 4: Vector Database vs Traditional Databases | Domain: 4. Vector Database")
st.markdown("""
A **Vector Database** is not simply a traditional database with an extra column for arrays; it represents 
a **fundamental paradigm shift** across data models, query semantics, indexing structures, and computational guarantees.

Where traditional databases operate on **structured tabular rows** and enforce **deterministic, exact boolean matches** (`WHERE id = 42`), 
vector databases operate on **unstructured high-dimensional vectors** and execute **probabilistic Approximate Nearest Neighbor (ANN)** 
searches across continuous metric spaces.

In this lesson, we dissect the 5 foundational differences that set vector databases apart:
1. Handling Unstructured Data at Enterprise Scale
2. Managing High-Dimensional Coordinate Manifolds (d = 768 to 3072)
3. The Mechanics of Approximate Matching (Trading 1% Recall for 1000x Speed)
4. Continuous Similarity Metrics vs. Discrete Boolean Predicates
5. Semantic Intent Retrieval vs. Lexical Token Overlap
""")

st.divider()

# Section 1: The 5 Core Architectural Differences
st.header("1. The 5 Core Dimensional & Algorithmic Shifts", divider="rainbow")

col_d1, col_d2 = col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.markdown("""
    ### 1. From Structured Tables to Unstructured Payloads
    * **Traditional RDBMS**:
      * Data must fit rigid rectangular schemas with typed columns: `INT`, `VARCHAR`, `TIMESTAMP`.
      * Fails on raw PDFs, audio waveforms, video frames, customer calls, or software codebases.
    * **Vector Database**:
      * Ingests high-dimensional vector embeddings generated from unstructured assets.
      * Retains flexible, schema-free JSON **metadata payloads** (`{"author": "Alice", "tags": ["nlp", "rag"], "version": 2}`) attached directly to each vector.
    
    ### 2. High-Dimensional Continuous Vector Spaces
    * **Traditional RDBMS**:
      * Indexes scalars in 1-dimensional order (`a < b`).
    * **Vector Database**:
      * Operates in 768-d, 1536-d, or 3072-d continuous vector spaces.
      * Proximity is determined by geometric angles or distances (Cosine, Dot Product, Euclidean L2).
    """)

with col_d2:
    st.markdown("""
    ### 3. Approximate Matching vs. Exact Boolean Matching
    * **Traditional RDBMS (Exact Matching)**:
      * A row either matches the SQL predicate (100% True) or it does not (0% False).
      * Deterministic, zero-variance results.
    * **Vector Database (Approximate Nearest Neighbor - ANN)**:
      * Exact nearest neighbor search over 10M vectors takes hundreds of milliseconds (`O(N * d)`).
      * Vector databases trade a negligible amount of accuracy (typically **95% to 99% Recall@K**) to answer queries in **2 to 5 milliseconds** (`O(log N)`).
    
    ### 4. Semantic Intent Retrieval
    * **Traditional Search**:
      * Fails completely when words differ (Vocabulary Mismatch Problem).
    * **Vector Retrieval**:
      * Matches concepts, intent, context, and cross-lingual translations automatically.
    """)

st.markdown("### The Paradigm Shift Matrix")
paradigm_matrix_df = pd.DataFrame({
    "System Dimension": ["Primary Data Asset", "Query Philosophy", "Result Certainty", "Index Architecture", "Search Complexity", "Hardware Acceleration"],
    "Traditional Database (RDBMS)": ["Structured tabular rows & columns", "Deterministic exact equality & range (`=`, `<`)", "100% Exact & Deterministic", "B+ Trees, B-Trees, Hash Tables", "O(log N) point lookup / O(N) scan", "Standard CPU Cache & Memory Bus"],
    "Vector Database": ["High-dimensional dense vectors + JSON", "Geometric proximity (Min distance / Max cosine)", "Probabilistic (Approximate: 95-99% recall)", "HNSW Proximity Graphs, IVF-PQ Centroids", "O(log N) ANN graph traversal", "SIMD AVX-512, GPU Tensor Cores (FMA)"]
})
st.dataframe(paradigm_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Complete Architectural Paradigm Comparison Flowchart
st.header("2. Complete Architectural Paradigm Comparison", divider="rainbow")

st.markdown("""
The diagram below illustrates the fundamental divergence between an RDBMS query engine and a Vector DB query engine:
""")

shift_dot = """
digraph ParadigmDivergence {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    UserIntent [label="User Query\\n'How to troubleshoot memory leaks?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_rdbms {
        label="Traditional Relational Engine (Deterministic)"; style="dashed"; color="#DC2626";
        SQLGen [label="SQL Exact Filter\\nWHERE title LIKE '%memory leak%'", fillcolor="#FEE2E2"];
        BTreeSearch [label="B+ Tree / Sequential Scan\\nExact string matching", fillcolor="#FECACA"];
        RDBMSOut [label="0 Rows Returned!\\n(Docs use phrase 'OOM heap exhaustion')", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];
    }

    subgraph cluster_vectordb {
        label="Vector Database Engine (Probabilistic / Semantic)"; style="dashed"; color="#059669";
        Embedder [label="Neural Embedding Model\\nq in R^1536", fillcolor="#D1FAE5"];
        HNSWTraverse [label="HNSW Graph Traversal\\nSub-5ms Geometric Proximity", fillcolor="#A7F3D0"];
        VDBOut [label="Top Chunks Retained!\\nMatches: 'OOM heap exhaustion'\\n(Cosine Sim: 0.91)", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
    }

    UserIntent -> SQLGen -> BTreeSearch -> RDBMSOut;
    UserIntent -> Embedder -> HNSWTraverse -> VDBOut;
}
"""
st.graphviz_chart(shift_dot, use_container_width=False)

st.divider()

# Section 3: The Mechanics of Approximate Matching (Recall vs. Latency)
st.header("3. The Mechanics of Approximate Matching & The Recall@K Trade-Off", divider="rainbow")

st.markdown("""
Why do vector databases accept **approximation**? 
Because in semantic retrieval, achieving **100% ground-truth exactness** is mathematically wasteful!
""")

col_ann1, col_ann2 = col_ann1, col_ann2 = st.columns([1, 1])

with col_ann1:
    st.markdown("""
    ### 1. Mathematical Definition of Recall@K
    * Let `True_K` be the exact set of top-k closest vectors calculated via brute-force linear scan.
    * Let `Retrieved_K` be the set of top-k vectors returned by an ANN algorithm (e.g. HNSW).
    * **ANN Recall@K** is defined as the overlap fraction:
      `Recall@K = |True_K intersect Retrieved_K| / K`.
    * *Example*: If brute-force finds items `[A, B, C, D, E]` and HNSW returns `[A, B, C, D, X]`, 
      the recall is `4 / 5 = 80.0%`.
    """)

with col_ann2:
    st.markdown("""
    ### 2. The Speed vs. Recall Trade-Off Curve
    * **Tuning Parameter (`efSearch` in HNSW / `nprobe` in IVF)**:
      * **Low `efSearch` (e.g. 16)**: Visits very few graph nodes. Latency: **1.2 ms**. Recall: **88%**.
      * **Balanced `efSearch` (e.g. 64)**: Production sweet spot. Latency: **3.8 ms**. Recall: **98.5%**.
      * **High `efSearch` (e.g. 512)**: Visits extensive graph neighborhoods. Latency: **35 ms**. Recall: **99.9%**.
      * **Brute Force (Exact)**: Scans all N items. Latency: **450 ms**. Recall: **100%**.
    * **The Insight**: Spending 450ms for 100% recall vs 4ms for 98.5% recall is a **112x speedup** for a 1.5% accuracy difference that human users and LLMs cannot even detect!
    """)

st.divider()

# Section 4: Interactive Paradigm & ANN Parameter Simulator
st.header("4. Interactive ANN Recall vs. Latency Simulator", divider="rainbow")

st.markdown("""
Simulate tuning the search parameters of an HNSW index across a collection of 500,000 vectors. 
Observe how the candidate queue size (`efSearch`) directly influences **search recall**, **graph hops**, and **query latency**:
""")

col_ann_cfg, col_ann_res = st.columns([1, 1])

with col_ann_cfg:
    st.subheader("HNSW Search Configuration")
    sim_ef_search = st.select_slider(
        "Candidate Queue Size (efSearch)",
        options=[8, 16, 32, 64, 128, 256, 512, 1024],
        value=64,
        help="Higher efSearch visits more neighbor links during greedy graph exploration"
    )
    sim_k = st.slider("Target Top-K Results", 1, 20, 5)
    sim_n_vectors = st.selectbox("Collection Vector Count (N)", [50000, 200000, 500000, 2000000], index=2)

with col_ann_res:
    st.subheader("Simulated Performance Metrics")
    
    # Mathematical approximation of HNSW behavior
    base_hops = np.log2(sim_n_vectors) * 1.5
    eff_hops = base_hops + (sim_ef_search * 0.45)
    est_latency_ms = 0.5 + (eff_hops * 0.04)
    
    # Recall curve modeled as sigmoid approaching 1.0
    recall_val = min(0.999, 1.0 - (1.0 / (1.0 + (sim_ef_search / 18.0) ** 1.8)))
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Estimated Recall@K", f"{recall_val * 100:.2f}%", f"Top-{sim_k} overlap")
        st.metric("Query Latency SLA", f"{est_latency_ms:.2f} ms", f"efSearch: {sim_ef_search}")
    with col_r2:
        st.metric("Graph Nodes Evaluated", f"{int(eff_hops):,} hops", f"Out of {sim_n_vectors:,} vectors")
        speedup_vs_flat = (sim_n_vectors * 0.001) / est_latency_ms
        st.metric("Speedup vs. Exact Flat Scan", f"{speedup_vs_flat:.0f}x Faster")
        
    if recall_val >= 0.98:
        st.success("**Optimal Production Zone**: High recall (>98%) with sub-5ms query response time!")
    elif recall_val >= 0.90:
        st.warning("**Fast Exploration Zone**: Low latency, but may drop 1 out of 10 relevant nearest neighbors.")
    else:
        st.error("**Undersized efSearch**: Substantial recall degradation. Increase efSearch to at least 32.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Comparing Exact SQL vs. Vector Semantic Search", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates the contrast between **Exact Relational SQL-style search** 
and **Vector Semantic Cosine Search** on a real set of technical documents:
""")

code_comp_impl = '''import numpy as np
from typing import List, Dict, Any

class SearchComparisonBenchmark:
    """Demonstrates why Vector Semantic Search outperforms Relational Exact Search on text."""
    def __init__(self):
        self.corpus = [
            {"id": 1, "text": "PostgreSQL connection pool exhausted. Check pg_stat_activity.", "tags": ["db", "postgres"]},
            {"id": 2, "text": "Kubernetes node out of memory error. Pod was terminated by OOMKiller.", "tags": ["k8s", "infra"]},
            {"id": 3, "text": "Redis server memory full. Eviction policy volatile-lru triggered.", "tags": ["cache", "redis"]},
            {"id": 4, "text": "Network socket handshake timeout during TLS negotiation.", "tags": ["network", "tls"]}
        ]
        # Synthetic dense embeddings (768-d) capturing semantic topics
        np.random.seed(42)
        dim = 64
        # Generate topic vectors
        topic_db = np.random.randn(dim)
        topic_k8s = np.random.randn(dim)
        
        self.vectors = [
            topic_db + np.random.randn(dim) * 0.1,    # Doc 1 (Postgres)
            topic_k8s + np.random.randn(dim) * 0.1,   # Doc 2 (K8s OOM)
            topic_db + np.random.randn(dim) * 0.25,   # Doc 3 (Redis)
            np.random.randn(dim)                       # Doc 4 (Network)
        ]
        # Normalize
        self.vectors = [v / np.linalg.norm(v) for v in self.vectors]
        self.topic_k8s = topic_k8s / np.linalg.norm(topic_k8s)

    def relational_exact_search(self, query_phrase: str) -> List[Dict[str, Any]]:
        """Simulates SQL: SELECT * FROM corpus WHERE text LIKE '%query_phrase%'"""
        matches = []
        for doc in self.corpus:
            if query_phrase.lower() in doc["text"].lower():
                matches.append(doc)
        return matches

    def vector_semantic_search(self, query_vector: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """Simulates Vector DB ANN Cosine Search."""
        scores = [float(np.dot(query_vector, v)) for v in self.vectors]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.corpus[i], "score": scores[i]} for i in ranked_idx]

# Benchmark Demonstration
bench = SearchComparisonBenchmark()

# User searches for "server ran out of RAM"
# Notice that none of the documents contain the word "RAM"!
user_query_text = "server ran out of RAM"
query_vec = bench.topic_k8s + np.random.randn(64) * 0.12
query_vec = query_vec / np.linalg.norm(query_vec)

print(f"User Query: '{user_query_text}'\\n")

# 1. Relational SQL Search
sql_hits = bench.relational_exact_search("out of RAM")
print("1. Relational SQL Exact Search Results:")
if not sql_hits:
    print("   -> 0 Matches! (Relational query failed because 'RAM' is not in text)")
else:
    for h in sql_hits:
        print(f"   -> Found: {h['text']}")

# 2. Vector Semantic Search
vdb_hits = bench.vector_semantic_search(query_vec, top_k=1)
print("\\n2. Vector Semantic Search Results:")
for h in vdb_hits:
    print(f"   -> Top Match (Score: {h['score']:.4f}): {h['doc']['text']}")
'''

st.code(code_comp_impl, language="python")

if st.button("🚀 Execute Search Benchmark in Sandbox", type="primary"):
    exec_scope = {}
    exec(code_comp_impl, globals(), exec_scope)
    
    b = exec_scope["bench"]
    sql_res = exec_scope["sql_hits"]
    vdb_res = exec_scope["vdb_hits"]
    
    st.success("Benchmark completed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("**Relational SQL Exact Match**:")
        if not sql_res:
            st.error("0 Rows Returned (Query: 'out of RAM' had zero literal character matches).")
        else:
            st.json(sql_res)
    with col_out2:
        st.markdown("**Vector DB Semantic Retrieval**:")
        st.success(f"Matched Doc ID {vdb_res[0]['doc']['id']} with Cosine Score `{vdb_res[0]['score']:.4f}`!")
        st.markdown(f"*{vdb_res[0]['doc']['text']}*")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why is an Approximate Nearest Neighbor (ANN) index fundamentally probabilistic rather than deterministic?"""
a1 = """* **Deterministic Search (Brute Force)**: Calculates distances to all N vectors without skipping any data. The result is mathematically guaranteed to be the exact nearest neighbor every single time.
* **Probabilistic Search (ANN)**:
  * Uses heuristic graph traversal (HNSW) or quantized clustering (IVF) to navigate high-dimensional space without visiting every node.
  * In rare cases (e.g., local minima or boundary edge crossings), the greedy search may bypass the true absolute nearest neighbor and return the 2nd or 3rd closest neighbor instead.
* **Why This is Acceptable in AI**:
  * Semantic embeddings are already continuous, noisy statistical representations of human language.
  * Returning the 2nd nearest neighbor (cosine similarity 0.912) instead of the 1st (cosine similarity 0.915) makes virtually zero difference to downstream LLM response quality."""

q2 = """2. What is Recall@K, and how do you systematically measure it in a production vector database?"""
a2 = """* **Recall@K Formula**:
  `Recall@K = |Retrieved_ANN_Set intersect True_Exact_Set| / K`.
* **Measurement Methodology**:
  1. Sample a golden validation set of 1,000 realistic queries.
  2. Compute exact ground-truth nearest neighbors using brute-force flat scan (`IndexFlatL2` or `IndexFlatIP`).
  3. Query the production ANN index (e.g. HNSW or IVF) with the same 1,000 queries.
  4. Compare returned IDs: if K = 10 and 9 of the 10 IDs match the ground truth across queries, the system achieves **Recall@10 = 90.0%**."""

q3 = """3. Why does increasing the `efSearch` parameter in HNSW improve recall while increasing latency?"""
a3 = """* In HNSW, `efSearch` defines the capacity of the internal dynamic candidate priority queue maintained during greedy graph exploration.
* **When `efSearch` is Small**: The search hops quickly between neighbors and terminates early when candidates stop improving, visiting few nodes.
* **When `efSearch` is Large**: The search maintains a deeper list of candidate nodes, allowing it to backtrack out of local minima and explore alternative graph branches.
* This dramatically increases the probability of discovering the true nearest neighbor (higher recall), but requires evaluating more SIMD distance calculations (higher latency)."""

q4 = """4. How does unstructured data handling in vector databases differ from document stores like MongoDB?"""
a4 = """* **Document Stores (MongoDB, Couchbase)**:
  * Store unstructured/semi-structured JSON trees.
  * Queries rely on scalar path expressions: `db.users.find({"profile.age": {"$gt": 25}})`.
  * Indexes are standard B-Trees on JSON keys or text search via inverted indexes.
* **Vector Databases (Qdrant, Milvus, Pinecone)**:
  * Do not query JSON keys for primary retrieval.
  * Treat the raw document as an embedding vector, using proximity in mathematical space as the primary retrieval mechanism.
  * The JSON payload is used secondarily for hybrid Boolean filtering during graph traversal."""

q5 = """5. Can traditional relational databases like PostgreSQL be transformed into vector databases?"""
a5 = """* **Yes, via Extensions like `pgvector`**:
  * Adds a native `vector` data type and implements ANN index algorithms (HNSW and IVF-Flat) inside PostgreSQL.
* **Trade-offs**:
  * **Pros**: Keep relational data and vectors in the same ACID-compliant database; execute native SQL JOINs between relational tables and vector embeddings.
  * **Cons**: Single-node memory constraints; PostgreSQL's buffer pool and MVCC vacuuming overhead are not optimized for multi-gigabyte in-memory HNSW graphs at 100M+ vector scale compared to dedicated distributed engines (Milvus, Qdrant)."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Vector Database Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: Expecting 100% Deterministic Reproducibility from Concurrent ANN**  
> In distributed vector databases with dynamic insertions, two consecutive identical queries may return slightly 
> different results at rank 9 and 10 due to background segment compaction, HNSW graph re-wiring, or concurrency. 
> Never write unit tests that assert exact deterministic ID order without tolerance thresholds.

> [!WARNING]
> **Trap 2: Tuning `efSearch` Too Low to Cheat Latency Benchmarks**  
> Teams often set `efSearch = 16` in staging to boast about "1.5ms query latency" to management. 
> In production, users complain that searches miss obvious documents because Recall@K dropped to 82%! 
> Maintain `efSearch >= 64` to guarantee at least 95%+ recall in enterprise RAG systems.

> [!WARNING]
> **Trap 3: Synchronous Graph Building on Every Single Insert**  
> Calling `insert()` one vector at a time on an active HNSW index locks graph adjacency lists, destroying 
> write throughput. Always batch your writes, append to the Write-Ahead Log (WAL), and allow background 
> threads to update the ANN index asynchronously.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Approximate Nearest Neighbor (ANN)?"):
        st.markdown("""
        **ANN Search**:  
        Sub-linear search algorithms (HNSW, IVF) that trade 1-2% accuracy for 100x-1000x latency speedups over exact brute-force scans.
        """)
    with st.expander("🗂️ Flashcard: What is Recall@K?"):
        st.markdown("""
        **Recall@K**:  
        The fraction of true nearest neighbors returned by an ANN search compared to exact brute-force ground truth.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What Does `efSearch` Control?"):
        st.markdown("""
        **`efSearch` in HNSW**:  
        Controls the size of the dynamic candidate priority queue during query traversal. Higher `efSearch` yields higher recall at the cost of higher latency.
        """)
    with st.expander("🗂️ Flashcard: What is Semantic Retrieval?"):
        st.markdown("""
        **Semantic Retrieval**:  
        Finding information based on conceptual meaning and intent in vector space rather than literal keyword or character matches.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can articulate the 5 core differences between relational databases and vector databases.", value=True)
chk2 = st.checkbox("I understand why vector databases use Approximate Nearest Neighbor (ANN) search instead of exact scans.")
chk3 = st.checkbox("I can define the mathematical formula for Recall@K and explain how to evaluate it.")
chk4 = st.checkbox("I know how the `efSearch` parameter tunes the latency-vs-recall trade-off in HNSW.")
chk5 = st.checkbox("I understand why relational keyword queries fail on synonyms (Vocabulary Mismatch Problem).")
chk6 = st.checkbox("I know the trade-offs of using `pgvector` inside PostgreSQL versus a dedicated vector database.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Vector Database Differences and are ready for Chapter 4.3: Feature Comparison!")
