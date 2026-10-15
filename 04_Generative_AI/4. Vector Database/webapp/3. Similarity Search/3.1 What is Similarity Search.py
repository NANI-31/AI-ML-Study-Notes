import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import heapq
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.1 What is Similarity Search - Vector DB Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("3.1 What is Similarity Search: Search Paradigms, Nearest Neighbor & k-NN Retrieval")
st.caption("Chapter 3: Similarity Search | Domain: 4. Vector Database")
st.markdown("""
For decades, computing was built on **deterministic boolean retrieval**: does record A match criteria B? 
`WHERE id = 42` or `SELECT * FROM users WHERE email = 'alice@example.com'`. 
Later, search engines introduced **lexical keyword search** (BM25, TF-IDF), ranking documents by term frequency and token overlap.

However, artificial intelligence introduced a completely different paradigm: **Similarity Search**. 
Instead of testing whether two items share identical bits or characters, similarity search calculates how close 
two items are in a continuous geometric coordinate space.

In this lesson, we trace the full evolution of search paradigms, formulate the **Nearest Neighbor Search (NNS)** 
problem, dissect **k-Nearest Neighbor (k-NN)** algorithms, and establish the computational necessity 
for Approximate Nearest Neighbor (ANN) vector database engines.
""")

st.divider()

# Section 1: The 4 Paradigms of Search Compared
st.header("1. The Evolution of Search Paradigms", divider="rainbow")

col_par1, col_par2 = col_par1, col_par2 = st.columns([1, 1])

with col_par1:
    st.markdown("""
    ### 1. Exact Search (Relational & Key-Value)
    * **Mechanism**: Binary equality test (`A == B`).
    * **Indexing**: B-Trees, B+ Trees, Hash Maps.
    * **Complexity**: O(1) for Hash, O(log N) for B-Tree.
    * **Limitation**: Zero tolerance for variance. A single character typo (`'PostgreSQ'` vs `'PostgreSQL'`) returns 0 results.
    
    ### 2. Keyword Search (Lexical Inverted Index)
    * **Mechanism**: Tokenized inverted index with BM25 / TF-IDF scoring.
    * **Indexing**: Posting lists (`term -> [doc_1, doc_4, doc_9]`).
    * **Strengths**: Lightning-fast keyword overlap; handles word frequencies and document lengths.
    * **Limitation**: The **Vocabulary Mismatch Problem**. Fails when user searches for *"automobile repair"* but documentation uses *"car maintenance"*.
    """)

with col_par2:
    st.markdown("""
    ### 3. Semantic Search (Intent-Aware)
    * **Mechanism**: Neural language models map queries and documents into dense vectors where angle captures concept intent.
    * **Strengths**: Matches synonyms, cross-lingual equivalents, conceptual paraphrases, and context.
    * **Limitation**: Can overlook rare product part numbers or exact string acronyms without hybrid assistance.
    
    ### 4. Similarity Search (General Mathematical Paradigm)
    * **Mechanism**: Given a dataset S and query q, retrieve items that minimize a distance function `dist(q, x)` or maximize similarity `sim(q, x)`.
    * **Universality**: Powers image visual search, audio fingerprinting, customer profile clustering, molecular drug design, and LLM context retrieval.
    """)

st.markdown("### The 4 Search Paradigms Side-by-Side")
paradigms_df = pd.DataFrame({
    "Search Paradigm": ["1. Exact Search", "2. Keyword Search (Lexical)", "3. Semantic Search", "4. Geometric Similarity Search"],
    "Underlying Technology": ["SQL / B-Tree / Hash Tables", "Elasticsearch / Lucene / BM25", "Transformer Bi-Encoders + Vector DB", "ANN Graphs / Metric Spaces"],
    "Query Example": ["WHERE product_id = 9182", "laptop battery overheating", "How to stop my computer from getting hot?", "Query embedding vector q in R^d"],
    "Matching Logic": ["Exact bitwise equality", "Token overlap & inverse doc freq", "Latent conceptual alignment", "Spatial proximity: min dist(q, x)"],
    "Handling of Synonyms": ["0% (Fails)", "Requires manual synonym dictionary", "Automatic (Deep embeddings)", "Automatic (Learned metric)"],
    "Algorithmic Complexity": ["O(1) to O(log N)", "O(posting_length)", "O(log N) via ANN", "O(N * d) brute / O(log N) ANN"]
})
st.dataframe(paradigms_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Nearest Neighbor Search & k-NN Foundations
st.header("2. Nearest Neighbor Search (NNS) & k-NN Retrieval", divider="rainbow")

st.markdown("""
At the mathematical core of all vector databases lies the **k-Nearest Neighbor (k-NN)** problem:
""")

col_knn1, col_knn2 = col_knn1, col_knn2 = st.columns([1, 1])

with col_knn1:
    st.markdown("""
    ### Mathematical Formulation
    * **Given**:
      * A collection of N vectors `S = {x_1, x_2, ..., x_N}` where each `x_i in R^d`.
      * A query vector `q in R^d`.
      * A distance metric `D(u, v)` (e.g. Euclidean L2 or Cosine distance).
    * **Nearest Neighbor Search (1-NN)**:
      * Find the single vector `x* in S` such that:
        `D(q, x*) <= D(q, x_i)` for all `i = 1, ..., N`.
    * **k-Nearest Neighbor Search (k-NN)**:
      * Find the subset of k vectors `S_k subset S` of cardinality k such that for any `u in S_k` and `v in S \\ S_k`:
        `D(q, u) <= D(q, v)`.
    * **Exact k-NN Complexity (Brute Force / Flat)**:
      * Calculating distance to all N vectors: `O(N * d)` operations.
      * Maintaining top-k using a bounded Max-Heap of size k: `O(N * log k)`.
      * Total Brute Force Complexity: `O(N * d + N * log k)`.
    """)

with col_knn2:
    st.markdown("### Exact k-NN Search vs Approximate Nearest Neighbor (ANN)")
    knn_dot = """
    digraph KNNSearch {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQuery [label="User Query Vector (q)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_exact {
            label="Exact k-NN (Flat Scan)"; style="dashed"; color="#DC2626";
            FlatScan [label="Compute dist(q, x_i) for ALL N items\\n(10 Million dot products)", fillcolor="#FEE2E2"];
            MaxHeap [label="Bounded Max-Heap (size k)\\nO(N log k) priority queue", fillcolor="#FECACA"];
            ExactOut [label="100% Deterministic Ground Truth\\nLatency: 850 ms (Too slow for SLA!)", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];
        }

        subgraph cluster_ann {
            label="Approximate Nearest Neighbor (ANN)"; style="dashed"; color="#059669";
            HNSWTraverse [label="Navigate Proximity Graph (HNSW)\\nVisit only ~500 nodes (O(log N))", fillcolor="#D1FAE5"];
            FastHeap [label="Candidate Heap of size efSearch\\nSub-linear pruning", fillcolor="#A7F3D0"];
            ANNOut [label="98%+ Recall at 4 ms Latency!\\n(Production Enterprise Standard)", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
        }

        UserQuery -> FlatScan -> MaxHeap -> ExactOut;
        UserQuery -> HNSWTraverse -> FastHeap -> ANNOut;
    }
    """
    st.graphviz_chart(knn_dot, use_container_width=False)

st.divider()

# Section 3: The 4 Search Paradigms Interactive Demonstration
st.header("3. Interactive Search Paradigm Simulator", divider="rainbow")

st.markdown("""
Test how **Exact SQL Matching**, **BM25 Keyword Matching**, and **Semantic k-NN Similarity Search** 
respond to varying search queries across a realistic technical knowledge base:
""")

# Sample corpus
knowledge_base = [
    {"id": "DOC-101", "text": "PostgreSQL connection pool exhausted. Increase max_connections or use PgBouncer.", "keywords": ["postgresql", "connection", "pool", "pgbouncer", "max_connections"]},
    {"id": "DOC-102", "text": "Kubernetes pod evicted due to node memory pressure OOMKilled.", "keywords": ["kubernetes", "pod", "evicted", "memory", "oomkilled", "node"]},
    {"id": "DOC-103", "text": "Database handshake timeout occurred during network socket initialization.", "keywords": ["database", "handshake", "timeout", "network", "socket"]},
    {"id": "DOC-104", "text": "Redis cache eviction policy volatile-lru drops expired keys under RAM pressure.", "keywords": ["redis", "cache", "eviction", "lru", "ram"]},
    {"id": "DOC-105", "text": "How to bake a sourdough bread loaf with crispy golden crust.", "keywords": ["bake", "sourdough", "bread", "crust"]}
]

col_sim_in, col_sim_out = st.columns([1, 1])

with col_sim_in:
    st.subheader("Query Input")
    sim_query = st.selectbox(
        "Select a Test Query Scenario",
        options=[
            "Scenario A (Exact ID): DOC-101",
            "Scenario B (Keyword Match): database handshake timeout",
            "Scenario C (Semantic Synonym - No exact words): server out of memory killed",
            "Scenario D (Paraphrase): My relational database ran out of client connections"
        ],
        index=3
    )
    
    clean_query = sim_query.split(": ")[-1].strip()
    st.info(f"**Active Query**: `{clean_query}`")

with col_sim_out:
    st.subheader("Retrieval Results Across Paradigms")
    
    # 1. Exact Match
    exact_matches = [doc["id"] for doc in knowledge_base if doc["id"].lower() == clean_query.lower() or doc["text"].lower() == clean_query.lower()]
    
    # 2. Keyword Match (Token overlap)
    q_words = set(clean_query.lower().split())
    keyword_hits = []
    for doc in knowledge_base:
        overlap = len(q_words & set(doc["keywords"]))
        if overlap > 0:
            keyword_hits.append((doc["id"], doc["text"][:50] + "...", overlap))
    keyword_hits.sort(key=lambda x: x[2], reverse=True)
    
    # 3. Semantic k-NN (Simulated semantic affinity)
    semantic_scores = {
        "DOC-101": 0.94 if "connections" in clean_query or "DOC-101" in clean_query else 0.42,
        "DOC-102": 0.91 if "memory" in clean_query or "DOC-102" in clean_query else 0.25,
        "DOC-103": 0.88 if "timeout" in clean_query or "handshake" in clean_query or "DOC-103" in clean_query else 0.55,
        "DOC-104": 0.65 if "memory" in clean_query or "DOC-104" in clean_query else 0.30,
        "DOC-105": 0.05
    }
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("#### 1. Exact Match")
        if exact_matches:
            st.success(f"Found: `{exact_matches[0]}`")
        else:
            st.error("0 Matches (Exact match requires 100% identical string)")
            
    with col_p2:
        st.markdown("#### 2. Keyword Match (BM25)")
        if keyword_hits:
            for hid, htxt, hscore in keyword_hits[:2]:
                st.markdown(f"**{hid}** (Overlap: {hscore})\n`{htxt}`")
        else:
            st.warning("0 Matches (No matching keywords found in document posting lists!)")
            
    with col_p3:
        st.markdown("#### 3. Semantic k-NN")
        sorted_sem = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        for sid, sscore in sorted_sem[:2]:
            st.markdown(f"**{sid}** | Cosine Sim: `{sscore:.2f}`\n`{next(d['text'][:45] for d in knowledge_base if d['id'] == sid)}...`")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Exact k-NN Search with Max-Heap in Python", divider="rainbow")

st.markdown("""
The following self-contained Python program implements **Exact k-Nearest Neighbor Search** using 
vectorized Euclidean distance and a bounded priority queue (Max-Heap) to maintain Top-K candidates:
""")

code_knn = '''import numpy as np
import heapq
from typing import List, Tuple, Dict, Any

class ExactKNNSearcher:
    """Exact brute-force k-NN search engine using a bounded Max-Heap."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_vectors(self, ids: List[str], vectors: np.ndarray):
        """Appends vectors to in-memory matrix."""
        assert vectors.shape[1] == self.dimension
        self.doc_ids.extend(ids)
        if len(self.vectors) == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search_top_k(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Exact k-NN search using Euclidean distance and Max-Heap priority queue."""
        assert len(query_vector) == self.dimension
        N = len(self.doc_ids)
        if N == 0:
            return []

        # Vectorized Euclidean distance calculation: ||x_i - q||
        diff = self.vectors - query_vector
        distances = np.linalg.norm(diff, axis=1)

        # Python heapq is a MIN-HEAP. To maintain Top-K SMALLEST distances,
        # we push (-distance, doc_id) into a heap of size k.
        heap = []
        for i in range(N):
            dist = float(distances[i])
            doc_id = self.doc_ids[i]

            if len(heap) < k:
                heapq.heappush(heap, (-dist, doc_id))
            else:
                # If current item is closer than the farthest item in our top-k heap
                if -dist > heap[0][0]:
                    heapq.heapreplace(heap, (-dist, doc_id))

        # Extract and sort results in ascending order of distance
        results = []
        while heap:
            neg_dist, doc_id = heapq.heappop(heap)
            results.append({"id": doc_id, "euclidean_distance": -neg_dist})

        return results[::-1] # Nearest first

# Demonstration
dim = 4
searcher = ExactKNNSearcher(dimension=dim)

# Ingest 5 vectors
ids = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
matrix = np.array([
    [0.10, 0.20, 0.30, 0.40],
    [0.12, 0.22, 0.31, 0.38], # Very close to A
    [0.90, 0.85, 0.10, 0.05],
    [0.15, 0.18, 0.28, 0.42], # Close to A
    [0.01, 0.99, 0.50, 0.20]
], dtype=np.float32)

searcher.add_vectors(ids, matrix)

query = np.array([0.11, 0.21, 0.30, 0.39], dtype=np.float32)
top_hits = searcher.search_top_k(query, k=3)

print("Exact k-NN Results for Query:")
for rank, hit in enumerate(top_hits, 1):
    print(f"Rank {rank}: ID={hit['id']}, Euclidean Distance={hit['euclidean_distance']:.4f}")
'''

st.code(code_knn, language="python")

if st.button("🚀 Execute Exact k-NN Search Engine", type="primary"):
    exec_scope = {}
    exec(code_knn, globals(), exec_scope)
    top_hits_res = exec_scope["top_hits"]
    
    st.success("Exact k-NN Engine executed successfully!")
    for rank, hit in enumerate(top_hits_res, 1):
        st.markdown(f"**Rank {rank}** | `ID: {hit['id']}` | **Distance:** `{hit['euclidean_distance']:.4f}`")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the fundamental difference between Exact Search, Keyword Search, and Similarity Search?"""
a1 = """* **Exact Search**: Deterministic binary match (`WHERE col = 'value'`). Backed by B-Trees and Hash Maps. Zero tolerance for variations or typos.
* **Keyword Search (Lexical)**: Inverted index ranking documents by term frequency and token overlap (BM25). Tolerates document length variance but suffers from the Vocabulary Mismatch Problem (fails on synonyms).
* **Similarity Search (Geometric)**: Maps queries and documents into continuous vector space, retrieving items that minimize distance or maximize angular cosine similarity. Captures concepts, intent, and semantics rather than literal characters."""

q2 = """2. What is the computational complexity of Exact k-NN Search, and why does it fail to scale to production?"""
a2 = """* **Complexity**: Computing distance across N vectors in d dimensions requires `O(N * d)` floating-point operations. Maintaining the top-k results via a bounded heap adds `O(N * log k)`.
* **The Scaling Cliff**:
  * For 10 million vectors of 1536 dimensions, a single query requires `10,000,000 * 1536 = 15.36 Billion` floating-point operations!
  * Even on multi-core SIMD hardware, this requires several seconds per query.
  * At 100 QPS (queries per second), the system demands hundreds of expensive servers just to scan vectors.
* **Why ANN is Required**: Approximate Nearest Neighbor (ANN) indexes (e.g. HNSW, IVF-PQ) trade 1-2% recall for a 100x-1000x latency reduction, answering queries in 2-5 milliseconds."""

q3 = """3. How does a Bounded Max-Heap maintain Top-K smallest distances efficiently during a linear scan?"""
a3 = """* In finding the k smallest distances, a naive approach would sort all N distances at cost `O(N log N)`.
* Instead, a Bounded Max-Heap of capacity k maintains the k best candidates seen so far. The root of the Max-Heap is always the **largest (farthest)** distance among the current k candidates.
* For each new vector distance `D`:
  * If the heap contains fewer than k elements, push `D`.
  * If `D` is smaller than the heap's root (`D < Max`), replace the root with `D` in `O(log k)` time.
  * If `D >= Max`, immediately discard `D` in `O(1)` time.
* Total sorting overhead is reduced from `O(N log N)` down to `O(N log k)`."""

q4 = """4. What is the difference between 1-NN (Nearest Neighbor) and k-NN (k-Nearest Neighbors) in retrieval systems?"""
a4 = """* **1-NN**: Retrieves strictly the single closest vector. Used in nearest-centroid classification (e.g., Voronoi cell routing in IVF indexes).
* **k-NN**: Retrieves the top-k closest vectors (typically k = 3 to 20).
* **Why k-NN is Mandatory in RAG**:
  * A single chunk rarely contains the complete context needed to answer a multi-faceted user query.
  * Downstream LLMs require multiple retrieved chunks representing different perspectives, background facts, or supporting evidence to synthesize a factual response."""

q5 = """5. What is the Vocabulary Mismatch Problem, and how does semantic similarity search overcome it?"""
a5 = """* **The Problem**: In natural language, the same idea can be expressed with completely different vocabulary:
  * User: `"automobile diagnostic software"` vs Doc: `"car repair tool"`.
  * Inverted index search engines like Elasticsearch assign a score of 0.0 because there is zero token overlap.
* **The Solution**:
  * Deep neural embedding models map both phrases to proximate coordinates in vector space because their pre-training objectives taught them that `"automobile"` and `"car"` share identical semantic contexts.
  * Similarity search identifies their proximity in single-digit milliseconds regardless of zero character overlap."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Similarity Search Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: Defaulting to Flat Exact Scan in Production Collections**  
> Many vector databases default to `IndexType.FLAT` if an index is not explicitly built after data ingestion. 
> A flat index works fine in unit tests with 500 documents, but the moment the database scales to 500,000 documents, 
> query latency spikes from 2ms to 650ms, triggering cascading upstream timeouts. Always build HNSW or IVF indexes before deploying.

> [!WARNING]
> **Trap 2: Setting k Too Large in Downstream RAG Pipelines**  
> Setting `k = 50` or `k = 100` to "ensure high recall" injects low-similarity, noisy chunks into the LLM context window. 
> This triggers the "Lost in the Middle" phenomenon, increasing hallucination rates and inflating LLM API token costs. 
> Keep initial retrieval at `k = 10 - 20`, pass candidates through a Cross-Encoder Reranker, and hydrate the top 3-5 chunks.

> [!WARNING]
> **Trap 3: Heap Re-allocation Overhead in High-Throughput Scenarios**  
> Allocating a new heap data structure per query inside high-frequency search loops creates significant memory 
> garbage collection overhead. Production C++ and Rust engines (Faiss, Qdrant) pre-allocate thread-local fixed-size 
> candidate heaps to maintain zero-allocation query loops.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Similarity Search?"):
        st.markdown("""
        **Similarity Search**:  
        Retrieving items from a dataset that minimize a distance function or maximize a similarity metric relative to a query vector.
        """)
    with st.expander("🗂️ Flashcard: What is Exact k-NN Complexity?"):
        st.markdown("""
        **Exact k-NN Complexity**:  
        `O(N * d + N * log k)`. For large N (e.g. 10M vectors), this brute-force scan is too slow for real-time web applications.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Vocabulary Mismatch Problem?"):
        st.markdown("""
        **Vocabulary Mismatch**:  
        The failure of keyword search engines when queries and documents express identical concepts using different words (synonyms).
        """)
    with st.expander("🗂️ Flashcard: Why Use a Bounded Max-Heap?"):
        st.markdown("""
        **Bounded Max-Heap**:  
        Maintains the top-k nearest items during a linear scan in `O(N log k)` time, avoiding expensive `O(N log N)` full-array sorting.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can clearly contrast Exact Search, Keyword Search, Semantic Search, and Geometric Similarity Search.", value=True)
chk2 = st.checkbox("I understand the mathematical formulation of Nearest Neighbor Search (1-NN and k-NN).")
chk3 = st.checkbox("I know the computational complexity of brute force k-NN and why it fails at scale.")
chk4 = st.checkbox("I understand how a bounded Max-Heap maintains Top-K candidates in O(N log k) time.")
chk5 = st.checkbox("I can explain the Vocabulary Mismatch Problem and how dense vector search overcomes it.")
chk6 = st.checkbox("I can implement an exact k-NN search algorithm in Python using NumPy and heapq.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Similarity Search foundations and are ready for Chapter 3.2: Distance Metrics!")
