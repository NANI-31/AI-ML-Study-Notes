import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="7.1 Basic Queries - ChromaDB Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("7.1 Basic Queries: The query() Primitive, Text vs Embeddings, Batching & n_results")
st.caption("Chapter 7: Querying ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
The `collection.query()` method is the primary retrieval interface of ChromaDB. 

Whether powering an enterprise RAG assistant, a recommendation engine, or an automated support triage system, 
executing vector search efficiently requires mastering ChromaDB's query semantics:
* **Natural Language Querying (`query_texts`)**: Chroma automatically vectorizes inbound human questions using the collection's bound `EmbeddingFunction`.
* **Direct Vector Querying (`query_embeddings`)**: Directly searching with pre-computed float arrays to bypass local model execution.
* **Batch Query Processing**: Passing multiple queries simultaneously (`query_texts=["Q1", "Q2"]`) to leverage vectorized SIMD batch search.
* **Top-K Retrieval Sizing (`n_results`)**: Controlling the candidate beam width to balance context completeness against prompt bloat.

In this lesson, we explore the query execution lifecycle, contrast single vs multi-query batching, 
implement clean retrieval pipelines in Python, and test an interactive Multi-Query Search Simulator.
""")

st.divider()

# Section 1: Anatomy of the collection.query() Method
st.header("1. Anatomy of the collection.query() Method", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### Core Query Parameters
    
    * **`query_texts: Optional[List[str]]`**:
      * A list of natural language strings. Chroma embeds each string using the collection's bound model before traversing the HNSW graph.
    
    * **`query_embeddings: Optional[List[List[float]]]`**:
      * A list of pre-computed float vectors. Bypasses text embedding entirely.
    
    * **`n_results: int = 10`**:
      * The number of closest nearest-neighbor documents to return for **each** query in the batch.
    
    * **`where: Optional[Dict[str, Any]]`**:
      * Metadata filter predicates (e.g., `{"department": "engineering"}`). Evaluated as pre-filters during graph traversal.
    
    * **`where_document: Optional[Dict[str, Any]]`**:
      * Full-text keyword filter predicates (e.g., `{"$contains": "kubernetes"}`).
    
    * **`include: List[str]`**:
      * Controls which payload fields to hydrate and return. Defaults to `["documents", "metadatas", "distances"]`.
    """)

with col_anat2:
    st.markdown("### The Query Execution Lifecycle")
    q_life_dot = """
    digraph QueryLifecycle {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="collection.query(query_texts=['...'])\\nInbound User Question", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        BoundEF [label="Bound EmbeddingFunction\\nVectorizes query text to float array", fillcolor="#EDE9FE", color="#7C3AED"];

        HNSWGraph [label="HNSW Graph Traversal\\nO(log N) Logarithmic Skip Search", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        WhereFilter [label="Evaluate 'where' Metadata Filter\\nBitmask candidate nodes", fillcolor="#FEF3C7", color="#D97706"];

        SortSlice [label="Sort by Ascending Distance\\nSlice Top-K (n_results)", fillcolor="#FDE68A", color="#D97706"];

        HydratePayload [label="Hydrate from SQLite\\nAttach Documents & Metadatas", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InQuery -> BoundEF -> HNSWGraph;
        HNSWGraph -> WhereFilter -> SortSlice -> HydratePayload;
    }
    """
    st.graphviz_chart(q_life_dot, use_container_width=False)

st.divider()

# Section 2: Single-Query vs. Multi-Query Batch Execution
st.header("2. Single-Query vs. Multi-Query Batch Execution", divider="rainbow")

col_batch1, col_batch2 = col_batch1, col_batch2 = st.columns([1, 1])

with col_batch1:
    st.markdown("""
    ### Why Multi-Query Batching is Critical
    * Notice that both `query_texts` and `query_embeddings` **require lists of items**, not raw single strings!
      * ❌ `query(query_texts="What is Chroma?")` -> Throws `ValueError`!
      * ✅ `query(query_texts=["What is Chroma?"])` -> Correct!
    * **The Power of Batching**:
      * Passing `query_texts=["Query A", "Query B", "Query C"]` triggers vectorized embedding calculation in a single forward pass.
      * The HNSW search runs in parallel across multiple query vectors.
      * Chroma returns a **2-Dimensional nested results structure** where index `[0]` corresponds to Query A, index `[1]` to Query B, etc.
    """)

with col_batch2:
    st.markdown("### The 2D Parallel Results Structure")
    st.code("""results = collection.query(
    query_texts=["Postgres replication", "Kubernetes ingress"],
    n_results=2
)

# Accessing Results for Query #0 ("Postgres replication"):
top_doc_q0 = results["documents"][0][0]
top_dist_q0 = results["distances"][0][0]
top_id_q0 = results["ids"][0][0]

# Accessing Results for Query #1 ("Kubernetes ingress"):
top_doc_q1 = results["documents"][1][0]
top_dist_q1 = results["distances"][1][0]
top_id_q1 = results["ids"][1][0]
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Search Queries
st.header("3. Clean Python Implementation of Search Queries", divider="rainbow")

st.code("""import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="enterprise_wiki",
    metadata={"hnsw:space": "cosine"}
)

# 1. Querying with Natural Language Text
text_results = collection.query(
    query_texts=["How do we configure multi-region database failover?"],
    n_results=3,
    where={"department": "sre"},
    include=["documents", "metadatas", "distances"]
)

# Safely unpack 2D list for single query
for rank, (doc_id, text, dist) in enumerate(zip(
    text_results["ids"][0],
    text_results["documents"][0],
    text_results["distances"][0]
), start=1):
    sim = max(0.0, 1.0 - dist) * 100
    print(f"Rank {rank}: [{doc_id}] Dist={dist:.4f} (Sim={sim:.1f}%) | {text[:60]}...")

# 2. Multi-Query Batching: Searching 2 questions simultaneously
batch_results = collection.query(
    query_texts=[
        "Kubernetes container crash loop troubleshooting",
        "OAuth2 refresh token expiry rotation policy"
    ],
    n_results=2
)

for q_idx, q_text in enumerate(["Kubernetes Query", "Security Query"]):
    print(f"\\nResults for {q_text}:")
    for hit_id, hit_dist in zip(batch_results["ids"][q_idx], batch_results["distances"][q_idx]):
        print(f"  -> Match ID: {hit_id} (Dist: {hit_dist:.4f})")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Multi-Query Search Simulator
st.header("4. Interactive Workbench: Multi-Query Search Simulator", divider="rainbow")

st.markdown("""
Simulate executing single or multi-query searches against a simulated ChromaDB index. 
Adjust `n_results` and observe how the 2D results structure unpacks for each query.
""")

col_sim_q1, col_sim_q2 = col_sim_q1, col_sim_q2 = st.columns(2)

with col_sim_q1:
    st.subheader("1. Inbound Search Queries")
    q1_input = st.text_input("Query #0 (`query_texts[0]`):", value="How to configure database read replicas?")
    multi_q_toggle = st.checkbox("Enable Multi-Query Batching", value=True)
    q2_input = ""
    if multi_q_toggle:
        q2_input = st.text_input("Query #1 (`query_texts[1]`):", value="Kubernetes pod autoscaling using Prometheus metrics.")

with col_sim_q2:
    st.subheader("2. Query Parameters")
    n_results_val = st.slider("Top-K Results (`n_results`):", min_value=1, max_value=5, value=3)
    filter_dept = st.selectbox("Metadata Pre-Filter (`where={'dept': ...}`):", ["All Departments", "Infrastructure", "Databases", "Security"])

# Mock database records
MOCK_CORPUS = [
    {"id": "doc_pg_01", "dept": "Databases", "text": "PostgreSQL read replicas distribute SELECT queries across streaming standbys.", "dist_q1": 0.082, "dist_q2": 0.781},
    {"id": "doc_pg_02", "dept": "Databases", "text": "MySQL asynchronous replication binlog configuration guide.", "dist_q1": 0.145, "dist_q2": 0.812},
    {"id": "doc_k8s_01", "dept": "Infrastructure", "text": "Kubernetes Horizontal Pod Autoscaler (HPA) scales pods via Prometheus adapter.", "dist_q1": 0.650, "dist_q2": 0.055},
    {"id": "doc_k8s_02", "dept": "Infrastructure", "text": "Cluster Autoscaler provisions extra cloud worker nodes on pending pod schedule.", "dist_q1": 0.590, "dist_q2": 0.180},
    {"id": "doc_sec_01", "dept": "Security", "text": "OAuth 2.0 PKCE authorization flow for single-page applications.", "dist_q1": 0.880, "dist_q2": 0.910}
]

def run_mock_query(dist_key: str, dept_filter: str, k: int):
    candidates = MOCK_CORPUS
    if dept_filter != "All Departments":
        candidates = [c for c in candidates if c["dept"] == dept_filter]
    sorted_hits = sorted(candidates, key=lambda x: x[dist_key])[:k]
    return sorted_hits

hits_q0 = run_mock_query("dist_q1", filter_dept, n_results_val)
hits_q1 = run_mock_query("dist_q2", filter_dept, n_results_val) if multi_q_toggle else []

st.divider()

st.subheader("Retrieved 2D Query Results Preview")

col_out_q0, col_out_q1 = col_out_q0, col_out_q1 = st.columns(2)

with col_out_q0:
    st.markdown(f"### Results for Query #0: *'{q1_input}'*")
    if hits_q0:
        for r, hit in enumerate(hits_q0, start=1):
            sim = max(0.0, 1.0 - hit['dist_q1']) * 100
            st.markdown(f"**Rank #{r}: [{hit['id']}]** (Dist: `{hit['dist_q1']:.3f}` | Sim: `{sim:.1f}%`)")
            st.caption(f"Dept: {hit['dept']} — {hit['text']}")
    else:
        st.warning("No records matched the metadata filter.")

with col_out_q1:
    if multi_q_toggle:
        st.markdown(f"### Results for Query #1: *'{q2_input}'*")
        if hits_q1:
            for r, hit in enumerate(hits_q1, start=1):
                sim = max(0.0, 1.0 - hit['dist_q2']) * 100
                st.markdown(f"**Rank #{r}: [{hit['id']}]** (Dist: `{hit['dist_q2']:.3f}` | Sim: `{sim:.1f}%`)")
                st.caption(f"Dept: {hit['dept']} — {hit['text']}")
        else:
            st.warning("No records matched the metadata filter.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why does `collection.query(query_texts='How to backup?')` raise an exception?"):
    st.markdown("""
    **Answer**:
    * ChromaDB's query API is explicitly designed for **vectorized batch processing**.
    * The `query_texts` parameter strictly expects a **list of strings** (`List[str]`), not a single bare string.
    * Passing a bare string causes Chroma to treat the string as an iterable of single characters or raises an argument type validation error.
    * **Fix**: Always wrap single queries in brackets: `query_texts=["How to backup?"]`.
    """)

with st.expander("Q2: How does the 2D list return format work in ChromaDB?"):
    st.markdown("""
    **Answer**:
    * Because `query_texts` accepts a list of $Q$ queries, and each query retrieves $K$ documents:
    * `results['ids']` is structured as `List[List[str]]` with dimensions `(Q, K)`.
    * To access the top hit for the first query: `results['ids'][0][0]`.
    * To access the top hit for the second query: `results['ids'][1][0]`.
    * Even if you pass only a single query string (`Q=1`), the outer list of length 1 is still present.
    """)

with st.expander("Q3: When should you use `query_embeddings` instead of `query_texts`?"):
    st.markdown("""
    **Answer**:
    * Use **`query_embeddings`** when:
      1. Your application pre-computes embeddings on a dedicated GPU microservice or Triton cluster before sending queries to Chroma.
      2. You are executing multi-modal searches (e.g. searching images with a CLIP image embedding vector).
      3. You are performing query caching: cached query vectors can be sent directly to Chroma, avoiding redundant embedding inference.
    """)

with st.expander("Q4: What happens if `n_results` is greater than the total number of documents in the collection?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does **not throw an error** or raise an index out-of-bounds exception.
    * It safely returns all available documents in the collection, sorted by ascending distance.
    * For example, if the collection contains 4 documents and you request `n_results=10`, the returned list contains exactly 4 items.
    """)

with st.expander("Q5: How does metadata pre-filtering interact with `n_results`?"):
    st.markdown("""
    **Answer**:
    * In ChromaDB, `where` filters are evaluated as **pre-filters during HNSW graph traversal**.
    * Only candidate vectors whose metadata matches the predicate are considered during the search.
    * The query returns up to `n_results` matching documents. If only 2 documents match the predicate, exactly 2 documents are returned regardless of collection size.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Bare Strings Instead of Lists**  
    Writing `collection.query(query_texts="my query")` fails immediately. Always wrap queries in a list: `query_texts=["my query"]`.
    """)
    st.warning("""
    **Trap 2: Accessing 1D Slices from 2D Result Dictionaries**  
    Writing `results["documents"][0]` yields the list of all Top-K documents for Query 0, not a single document! Use `results["documents"][0][0]`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Requesting Excessive `n_results` (>20)**  
    Passing 20+ chunks into an LLM context creates prompt bloat and degrades recall via 'Lost in the Middle'. Use `n_results=3 to 5`.
    """)
    st.warning("""
    **Trap 4: Forgetting Query Vectorization Model Alignment**  
    Passing pre-computed `query_embeddings` from Model A against a collection built with Model B produces nonsensical random rankings.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is query_texts input type?"):
        st.markdown("""
        **List[str]**:  
        Strictly requires a list of strings, even for single queries (`["..."]`).
        """)
    with st.expander("🗂️ Flashcard: How is results structured?"):
        st.markdown("""
        **2D Nested Lists**:  
        `results['documents'][query_idx][rank_idx]`.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What if n_results > doc count?"):
        st.markdown("""
        **Graceful Degradation**:  
        Returns all available documents without padding or throwing errors.
        """)
    with st.expander("🗂️ Flashcard: Why use query_embeddings?"):
        st.markdown("""
        **Bypass Model**:  
        Directly search using pre-computed float arrays for speed or caching.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the parameters of `collection.query()` including query_texts and query_embeddings.", value=True)
chk2 = st.checkbox("I know why `query_texts` strictly requires a list of strings (`List[str]`).", value=True)
chk3 = st.checkbox("I understand how multi-query batching executes in parallel and returns a 2D nested results structure.", value=True)
chk4 = st.checkbox("I know how `n_results` controls Top-K retrieval and how Chroma handles counts exceeding dataset size.", value=True)
chk5 = st.checkbox("I understand how to safely unpack 2D parallel lists into ranked search results.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 7.2: Query Results Structure in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 7.1: Basic Queries and are ready for Chapter 7.2: Query Results!")
