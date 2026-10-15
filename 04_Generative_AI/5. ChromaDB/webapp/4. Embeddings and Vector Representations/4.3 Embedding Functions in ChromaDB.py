import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="4.3 Embedding Functions in ChromaDB - ChromaDB Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("4.3 Embedding Functions in ChromaDB: Default Behavior, Custom Configuration & Query Vectorization")
st.caption("Chapter 4: Embeddings and Vector Representations | Domain: 5. ChromaDB")
st.markdown("""
One of ChromaDB's greatest architectural strengths is its **Automated Embedding Pipeline**. 

Unlike other vector databases where developers must write manual glue code to call an external embedding API, 
serialize the resulting JSON, extract float arrays, and pass raw vectors into the database, 
ChromaDB handles the entire lifecycle seamlessly via its **`EmbeddingFunction` interface**.

When an embedding function is bound to a collection:
* Ingesting documents (`collection.add(documents=[...])`) automatically vectorizes the text and builds the HNSW graph.
* Querying documents (`collection.query(query_texts=[...])`) automatically vectorizes the user's natural language question and conducts ANN search using the exact same model!

In this lesson, we explore ChromaDB's default embedding behavior, analyze how embedding functions are configured 
and persisted, write custom vectorized pipelines, and test an interactive Dual Ingestion & Query Vectorization Inspector.
""")

st.divider()

# Section 1: The Automated Embedding Pipeline & Default Behavior
st.header("1. The Automated Embedding Pipeline & Default Behavior", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1.1, 0.9])

with col_pipe1:
    st.markdown("""
    ### Default Embedding Behavior (`DefaultEmbeddingFunction`)
    * When no embedding function is explicitly specified, ChromaDB instantiates `DefaultEmbeddingFunction`.
    * **Under the Hood**:
      * Model: **`all-MiniLM-L6-v2`** from Sentence-Transformers.
      * Architecture: 6-layer MiniLM transformer producing **384-dimensional dense vectors**.
      * Engine: **ONNX Runtime on CPU**, executing optimized C++ kernels with AVX2 vectorization.
      * Model Weights: Downloaded once automatically on first invocation into `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/` (~80 MB) and cached locally forever.
    * **Zero GPU Requirement**: Runs blazingly fast on standard developer laptops and minimal cloud containers.
    """)

with col_pipe2:
    st.markdown("### The Dual Vectorization Flowchart")
    dual_dot = """
    digraph DualVectorization {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InDoc [label="1. Ingestion: Raw Documents\\ncollection.add(documents=['...'])", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        InQuery [label="2. Retrieval: Search Query\\ncollection.query(query_texts=['...'])", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        BoundEF [label="Bound EmbeddingFunction\\n(e.g., OpenAI / MiniLM / BGE)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        DocVec [label="Document Embeddings\\n(Stored in HNSW Index)", fillcolor="#D1FAE5", color="#059669"];

        QueryVec [label="Query Embedding\\n(Used for ANN Traversal)", fillcolor="#FDE68A", color="#D97706"];

        ANN [label="HNSW Cosine Search\\nReturns Top-K Document Matches", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        InDoc -> BoundEF [label="Auto-Vectorize"];
        InQuery -> BoundEF [label="Auto-Vectorize"];
        BoundEF -> DocVec;
        BoundEF -> QueryVec;
        DocVec -> ANN;
        QueryVec -> ANN;
    }
    """
    st.graphviz_chart(dual_dot, use_container_width=False)

st.divider()

# Section 2: Binding & Re-Connecting Embedding Functions
st.header("2. Binding & Re-Connecting Embedding Functions", divider="rainbow")

col_bind1, col_bind2 = col_bind1, col_bind2 = st.columns([1, 1])

with col_bind1:
    st.markdown("""
    ### Critical Architectural Rule: Client Binding
    * ChromaDB persists document text, IDs, and vector floats to disk or the remote server.
    * **However, ChromaDB does NOT serialize Python function objects into the database!**
    * When you restart your application and call `client.get_collection("my_kb")`:
      * If you do not supply `embedding_function`, Chroma reverts to `DefaultEmbeddingFunction`!
      * If your collection was built with OpenAI (1536d) and you call `query(query_texts=...)` without passing `embedding_function=openai_ef`, Chroma attempts to embed the query with MiniLM (384d), causing a **Fatal Dimensionality Mismatch Error**!
    
    ### The Production Rule: Always Pass the Function
    ```python
    # Correct Production Pattern:
    collection = client.get_or_create_collection(
        name="my_openai_kb",
        embedding_function=openai_ef
    )
    ```
    """)

with col_bind2:
    st.markdown("### Clean Implementation Patterns")
    st.code("""import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_vault")

# 1. Instantiate chosen embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"  # 768-dim model
)

# 2. Bind to collection during creation or retrieval
collection = client.get_or_create_collection(
    name="enterprise_mpnet_kb",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# 3. Add raw text: Auto-vectorized via mpnet-base-v2
collection.add(
    ids=["doc_1"],
    documents=["High-availability PostgreSQL active-standby replication."]
)

# 4. Query raw text: Auto-vectorized via the same mpnet model!
results = collection.query(
    query_texts=["How do we configure database standby replicas?"],
    n_results=1
)
print("Retrieved:", results["documents"][0][0])
""", language="python")

st.divider()

# Section 3: Bypassing Embedding Functions with Pre-Computed Vectors
st.header("3. Bypassing Embedding Functions with Pre-Computed Vectors", divider="rainbow")

st.markdown("""
In distributed architectures where embeddings are generated by an upstream streaming worker 
or GPU cluster, you can bypass the collection's embedding function completely:
""")

st.code("""# Bypass during Ingestion: Pass embeddings directly
collection.add(
    ids=["ext_42"],
    embeddings=[[0.052, -0.128, 0.991, -0.042]],  # Raw pre-computed float array
    documents=["Document text preserved for context synthesis."]
)

# Bypass during Search: Pass query vector directly
collection.query(
    query_embeddings=[[0.050, -0.125, 0.985, -0.040]],  # Raw pre-computed query vector
    n_results=1
)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Dual Vectorization & Search Inspector
st.header("4. Interactive Workbench: Dual Vectorization & Search Inspector", divider="rainbow")

st.markdown("""
Simulate how a bound `EmbeddingFunction` vectorizes both ingested documents and inbound queries. 
Enter sample documents and a query to watch the automatic vectorization and similarity calculation.
""")

col_sim_docs, col_sim_q = col_sim_docs, col_sim_q = st.columns([1.1, 0.9])

with col_sim_docs:
    st.subheader("1. Indexed Knowledge Base Documents")
    doc_a = st.text_input("Document A:", value="PostgreSQL WAL archives stream to encrypted S3 for disaster recovery.")
    doc_b = st.text_input("Document B:", value="Kubernetes Ingress controllers utilize Envoy for TLS termination.")
    doc_c = st.text_input("Document C:", value="Redis cluster caches high-frequency queries to achieve sub-millisecond latency.")

with col_sim_q:
    st.subheader("2. Inbound User Search Query")
    query_text = st.text_input("Query (`query_texts`):", value="How are database backups archived to cloud storage?")
    bound_model = st.selectbox(
        "Bound EmbeddingFunction:",
        ["all-MiniLM-L6-v2 (384d / ONNX)", "text-embedding-3-small (1536d / OpenAI)", "bge-base-en (768d / Local)"]
    )

# Simulated semantic scoring
def mock_semantic_score(query: str, doc: str) -> float:
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    overlap = len(q_words.intersection(d_words))
    
    if "backup" in query.lower() and "wal" in doc.lower():
        dist = 0.048
    elif "tls" in query.lower() and "envoy" in doc.lower():
        dist = 0.055
    elif "cache" in query.lower() and "redis" in doc.lower():
        dist = 0.062
    else:
        dist = max(0.04, 0.85 - (overlap * 0.12))
    return round(dist, 4)

score_a = mock_semantic_score(query_text, doc_a)
score_b = mock_semantic_score(query_text, doc_b)
score_c = mock_semantic_score(query_text, doc_c)

ranking = sorted([
    ("Doc A", doc_a, score_a),
    ("Doc B", doc_b, score_b),
    ("Doc C", doc_c, score_c)
], key=lambda x: x[2])

st.divider()

st.subheader("Dual Pipeline Retrieval Results")

res_table = [
    {
        "Rank": f"#{idx}",
        "Document ID": item[0],
        "Cosine Distance": item[2],
        "Similarity Match": f"{round((1.0 - item[2]) * 100, 1)}%",
        "Document Text Preview": item[1][:70] + "..."
    }
    for idx, item in enumerate(ranking, start=1)
]

st.dataframe(pd.DataFrame(res_table), use_container_width=True, hide_index=True)
top_doc = ranking[0]
st.success(f"🏆 **Top Match**: `{top_doc[0]}` with Cosine Distance `{top_doc[2]}`! Both document and query were vectorized using `{bound_model}`.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What happens if you instantiate `client.get_collection()` without supplying the `embedding_function`?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does not store the Python function definition in the database.
    * If you call `get_collection("my_collection")` without specifying `embedding_function`, Chroma defaults to `DefaultEmbeddingFunction` (MiniLM, 384 dimensions).
    * If the collection was originally created using OpenAI (1536 dimensions), calling `collection.query(query_texts=["..."])` causes Chroma to embed the query into 384 dimensions using MiniLM.
    * The HNSW index then crashes with:
      `ValueError: Query embedding has 384 dimensions, but index has 1536 dimensions`.
    * **Production Remedy**: Always pass the configured `embedding_function` to `get_collection()` or `get_or_create_collection()`.
    """)

with st.expander("Q2: How does ChromaDB cache ONNX embedding model weights to avoid re-downloading on every run?"):
    st.markdown("""
    **Answer**:
    * On the first call to `DefaultEmbeddingFunction`, Chroma downloads the quantized ONNX model file from Hugging Face / Chroma CDN.
    * It saves the model files into the local user cache directory: `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`.
    * Subsequent executions check for the presence of the model file on disk. If present, it loads the model into RAM immediately, requiring zero internet connectivity.
    """)

with st.expander("Q3: Can a collection have different embedding functions for different documents?"):
    st.markdown("""
    **Answer**:
    **No**. A collection represents a **single, unified metric space**:
    * All vectors inside a collection must be generated by the exact same embedding model.
    * If you need to use multiple embedding models (e.g. MiniLM for English and a multilingual model for Spanish), you must create separate collections for each model.
    """)

with st.expander("Q4: When should you use `query_embeddings` instead of `query_texts`?"):
    st.markdown("""
    **Answer**:
    * Use **`query_texts`** when you want ChromaDB to handle the text-to-vector embedding step automatically.
    * Use **`query_embeddings`** when:
      1. Embeddings are generated by an external GPU service or API gateway before reaching ChromaDB.
      2. You are implementing multi-modal search (e.g. querying text against image vectors using CLIP).
      3. You are benchmarking retrieval speed and want to exclude embedding inference latency from database query timings.
    """)

with st.expander("Q5: What is the signature of ChromaDB's `EmbeddingFunction` protocol?"):
    st.markdown("""
    **Answer**:
    ```python
    from chromadb import EmbeddingFunction
    from chromadb.api.types import Documents, Embeddings

    class MyCustomEF(EmbeddingFunction[Documents]):
        def __call__(self, input: Documents) -> Embeddings:
            # input is List[str]
            # Must return List[List[float]]
            ...
    ```
    The input is always a list of strings (`Documents`), and the return type is a list of float arrays (`Embeddings`).
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Forgetting to Re-Pass the Embedding Function on `get_collection()`**  
    Calling `get_collection()` without passing the custom embedding function reverts to the default 384d model, causing runtime dimension mismatch crashes.
    """)
    st.warning("""
    **Trap 2: Internet Dependency on Container Cold-Start**  
    Running Chroma in an air-gapped production Docker container without pre-baking ONNX weights in the Dockerfile causes crash-loops when the container cannot reach Hugging Face.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Mixing Query Texts and Query Embeddings in a Single Call**  
    Supplying both `query_texts` and `query_embeddings` to `collection.query()` creates ambiguity and raises an argument validation exception. Provide one or the other.
    """)
    st.warning("""
    **Trap 4: Slow Single-String Embedding in Custom Functions**  
    Implementing a custom embedding function that calls an API one document at a time in a loop destroys throughput. Always implement vectorized batching.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the default embedding model?"):
        st.markdown("""
        **all-MiniLM-L6-v2**:  
        384 dimensions; executed via CPU ONNX Runtime with zero external API calls.
        """)
    with st.expander("🗂️ Flashcard: Are embedding functions saved in DB?"):
        st.markdown("""
        **No**:  
        Chroma persists data, not Python code; must pass `embedding_function` on collection retrieval.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What does query_texts do?"):
        st.markdown("""
        **Auto-Vectorization**:  
        Automatically embeds user question using the bound function before searching HNSW.
        """)
    with st.expander("🗂️ Flashcard: What is EmbeddingFunction signature?"):
        st.markdown("""
        **`__call__(Documents) -> Embeddings`**:  
        Accepts `List[str]` and returns `List[List[float]]`.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand ChromaDB's default embedding behavior with ONNX all-MiniLM-L6-v2.", value=True)
chk2 = st.checkbox("I understand how bound embedding functions automatically vectorize both documents and query texts.", value=True)
chk3 = st.checkbox("I know why `embedding_function` must be explicitly passed when retrieving collections with `get_collection()`.", value=True)
chk4 = st.checkbox("I know how to bypass embedding functions using pre-computed `embeddings` and `query_embeddings`.", value=True)
chk5 = st.checkbox("I can implement a custom `EmbeddingFunction` adhering to ChromaDB's callable protocol.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 5: Vector Database Fundamentals in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 4: Embeddings and Vector Representations in ChromaDB!")
