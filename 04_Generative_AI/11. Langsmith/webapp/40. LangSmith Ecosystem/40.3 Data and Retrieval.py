import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="40.3 Data and Retrieval - LangSmith Ecosystem",
    page_icon="📚",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .data-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-sim {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class DataAndRetrievalEngine:
    """
    Simulates core data manipulation, dense vector similarity,
    and retriever telemetry in LangSmith ecosystems.
    """
    def __init__(self):
        self.doc_corpus = [
            {"id": "doc-01", "title": "LangSmith Tracing Architecture", "content": "LangSmith uses non-blocking background threads to export hierarchical trace spans asynchronously.", "category": "architecture"},
            {"id": "doc-02", "title": "VectorDB Indexing with Qdrant", "content": "Qdrant stores dense vector embeddings and supports payload filtering with HNSW graphs.", "category": "database"},
            {"id": "doc-03", "title": "Embedding Dimensionality and Cosine Distance", "content": "Cosine similarity computes the dot product of normalized vectors to measure semantic proximity.", "category": "math"},
            {"id": "doc-04", "title": "Pandas Metric Aggregation", "content": "Pandas DataFrame structures allow grouping test experiment feedback runs by prompt version.", "category": "data"}
        ]

    def compute_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(round(np.dot(a, b) / (norm_a * norm_b), 4))

    def simulate_retrieval(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        results = []
        q_words = set(query.lower().split())
        for doc in self.doc_corpus:
            doc_words = set(doc["content"].lower().split())
            overlap = len(q_words.intersection(doc_words)) / max(len(q_words), 1)
            # Simulated dense similarity score
            sim_score = min(round(0.40 + overlap * 0.55, 3), 0.99)
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "score": sim_score,
                "content": doc["content"]
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Instantiate engine
data_engine = DataAndRetrievalEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Data Ingestion, Vector Indexing, & Retrieval Telemetry Pipeline", divider="rainbow")

st.markdown("""
High-accuracy generative AI applications depend fundamentally on **data quality and retrieval precision**. 
Before an LLM receives a prompt, source documents must be ingested by document loaders, processed and cleaned via Pandas, 
chunked and vectorized into dense embeddings with NumPy, and indexed into vector databases. 
LangSmith provides complete observability across this pre-generation pipeline.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Data_Retrieval_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingest {
            label = "Document Ingestion & Preparation Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            doc_loaders [label="Document Loaders (PDF, HTML, MD)\\n(Extracts raw text + metadata dictionary)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            text_splitters [label="Recursive Text Splitters\\n(Splits into 500-token chunks with 50-token overlap)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            pandas_cleaner [label="Pandas Metadata Cleaning\\n(Deduplication, filtering, schema normalization)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_vector {
            label = "Vectorization & Embedding Storage Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            embed_model [label="Embedding Models (OpenAI, Voyage, BGE)\\n(Converts text to dense float arrays)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            numpy_matrix [label="NumPy Vector Array\\n(Dense float32 matrix operations & L2 normalization)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            vector_db [label="Vector Databases (Qdrant, Chroma, Pinecone)\\n(HNSW graph indexing & approximate nearest neighbor)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_observability {
            label = "LangSmith Retriever Telemetry Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            retriever_span [label="@traceable(run_type='retriever')\\n(Captures search query, latency, top_k documents)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            context_precision [label="Evaluation Quality Gate\\n(Measures Context Precision & Hit Rate)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        doc_loaders -> text_splitters;
        text_splitters -> pandas_cleaner;
        pandas_cleaner -> embed_model;

        embed_model -> numpy_matrix;
        numpy_matrix -> vector_db;

        vector_db -> retriever_span [label="dispatches similarity search"];
        retriever_span -> context_precision [label="records retrieved chunks to trace"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Retriever Span Standard:</strong> In LangSmith, every retrieval step should be marked with 
    <code>run_type="retriever"</code>. This surfaces retrieved chunks directly in the UI trace tree, 
    allowing engineers to immediately verify whether hallucinated answers were caused by retrieval misses.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: NUMPY
# ==============================================================================
st.header("2. NumPy: Dense Vector Computations, Cosine Similarity Matrices, and Precision Optimization", divider="rainbow")

st.markdown("""
At the mathematical foundation of modern vector retrieval lies **NumPy**. 
Embedding models transform text into high-dimensional arrays (e.g., 1,536 dimensions for `text-embedding-3-small`, 
3,072 dimensions for `text-embedding-3-large`). NumPy provides vectorized C-level performance for 
distance computations and matrix multiplications.
""")

st.markdown("""
### Essential NumPy Operations in Retrieval Pipelines
- **L2 Vector Normalization**: Normalizing embedding vectors to unit length transforms Euclidean distance into a simple dot product, dramatically speeding up similarity search.
- **Batched Cosine Similarity**: Computing query vector alignment against hundreds of candidate document vectors in a single vectorized matrix dot product (`np.dot(query_vec, doc_matrix.T)`).
- **Float Precision Trade-offs**: Storing vectors in `float16` or `bfloat16` rather than `float32` slashes memory consumption by 50% with negligible loss in semantic ranking precision.
""")

st.code('''import numpy as np

# Simulating 3 document embeddings and 1 query embedding in 4-dimensional space
docs_embeddings = np.array([
    [0.12, 0.85, 0.33, 0.41],
    [0.91, 0.22, 0.15, 0.05],
    [0.05, 0.78, 0.45, 0.42]
], dtype=np.float32)

query_embedding = np.array([0.08, 0.81, 0.38, 0.44], dtype=np.float32)

# Vectorized Cosine Similarity calculation
doc_norms = np.linalg.norm(docs_embeddings, axis=1, keepdims=True)
query_norm = np.linalg.norm(query_embedding)

normalized_docs = docs_embeddings / doc_norms
normalized_query = query_embedding / query_norm

similarity_scores = np.dot(normalized_docs, normalized_query)
print("Cosine Similarities:", similarity_scores)
''', language="python")

# ==============================================================================
# SECTION 3: PANDAS
# ==============================================================================
st.header("3. Pandas: Dataset Manipulation, Evaluation Metric Aggregations, and Error Auditing", divider="rainbow")

st.markdown("""
**Pandas** is the workhorse of LLMOps data engineering. In LangSmith workflows, Pandas DataFrames 
are utilized for three mission-critical operations:
""")

c_p1, c_p2 = st.columns(2)
with c_p1:
    st.markdown("""
    ### 1. Golden Dataset Preparation & Sanitization
    - Ingesting raw customer chat logs or support tickets from CSV/Parquet.
    - Filtering out corrupted records, deduplicating identical questions, and masking PII.
    - Formatting raw tables into LangSmith's required `[{"inputs": {...}, "outputs": {...}}]` JSON schema.
    """)

with c_p2:
    st.markdown("""
    ### 2. Experiment Analysis & Regression Audits
    - Exporting LangSmith test runs into DataFrames for statistical analysis.
    - Aggregating feedback metrics grouped by prompt version (`df.groupby('prompt_version')['accuracy'].mean()`).
    - Generating diagnostic percentile distributions for latency and token spend.
    """)

st.code('''import pandas as pd

# Analyzing LangSmith experiment export
experiments_data = [
    {"experiment": "v1-zero-shot", "score": 0.82, "latency_ms": 640, "cost": 0.0021},
    {"experiment": "v1-zero-shot", "score": 0.79, "latency_ms": 580, "cost": 0.0019},
    {"experiment": "v2-few-shot", "score": 0.94, "latency_ms": 1120, "cost": 0.0042},
    {"experiment": "v2-few-shot", "score": 0.96, "latency_ms": 1050, "cost": 0.0039}
]

df = pd.DataFrame(experiments_data)
summary = df.groupby("experiment").agg(
    avg_score=("score", "mean"),
    p95_latency=("latency_ms", lambda x: np.percentile(x, 95)),
    avg_cost=("cost", "mean")
)
print("Aggregated Benchmark Report:")
print(summary)
''', language="python")

# ==============================================================================
# SECTION 4: VECTOR DATABASES
# ==============================================================================
st.header("4. Vector Databases: Chroma, Qdrant, Pinecone, and Tracing Retrieval Latency & Recall", divider="rainbow")

st.markdown("""
**Vector databases** are purpose-built storage engines optimized for **Approximate Nearest Neighbor (ANN)** 
search across millions of high-dimensional vectors. Major engines include **Chroma** (embedded prototyping), 
**Qdrant** (high-performance Rust-backed vector search with payload filtering), and **Pinecone** (managed serverless).
""")

vdb_df = pd.DataFrame([
    {"Engine": "ChromaDB", "Architecture": "Embedded / In-Process SQLite", "Best For": "Local development, rapid prototyping, notebook testing", "LangSmith Tracing": "Captures top_k documents and query vectors via langchain-chroma"},
    {"Engine": "Qdrant", "Architecture": "Distributed Rust engine with HNSW", "Best For": "High-throughput production, complex payload JSON filtering", "LangSmith Tracing": "Logs exact query filter predicates and segment search latency"},
    {"Engine": "Pinecone", "Architecture": "Managed Serverless Cloud", "Best For": "Zero-maintenance multi-tenant enterprise deployments", "LangSmith Tracing": "Tracks network round-trip overhead and index namespace queries"}
])
st.dataframe(vdb_df, use_container_width=True)

# ==============================================================================
# SECTION 5: EMBEDDING MODELS
# ==============================================================================
st.header("5. Embedding Models: Dense Representations, Dimensionality, Embedding Drifts, and Latency Spans", divider="rainbow")

st.markdown("""
Embedding models map unstructured text into dense vector space where semantic proximity corresponds 
to mathematical distance. When selecting an embedding model, engineers must balance dimensionality, 
model latency, cost, and retrieval quality.
""")

st.markdown("""
### Critical Embedding Telemetry Factors
1. **Embedding Latency Spans**: Generating embeddings requires an API network call or GPU inference pass. Tracing embedding models in LangSmith identifies when latency bottlenecks originate in vectorization rather than generation.
2. **Matryoshka Representation Learning (MRL)**: Modern models (e.g., OpenAI `text-embedding-3`) allow truncating 1,536-dimensional embeddings to 512 dimensions with minimal accuracy loss, slashing storage and RAM costs by 66%.
3. **Embedding Drift**: Upgrading an embedding model from `text-embedding-ada-002` to `text-embedding-3-small` invalidates the entire existing vector index. Both old and new embeddings cannot be compared.
""")

# ==============================================================================
# SECTION 6: DOCUMENT LOADERS
# ==============================================================================
st.header("6. Document Loaders: Unstructured Ingestion, Recursive Chunking, and Metadata Preservation", divider="rainbow")

st.markdown("""
Garbage in, garbage out. If document loaders discard headers, mangle tables, or strip source URLs, 
downstream vector search and generative synthesis will inevitably fail.
""")

st.code('''from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Document Ingestion
# Preserves metadata: {"source": "architecture_guide.md"}
raw_documents = [
    {"page_content": "LangSmith tracing enables deep observability into multi-agent workflows.", "metadata": {"source": "guide.md", "page": 1}}
]

# 2. Semantic Chunking
# Recursive splitter tries paragraphs, then sentences, then words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", " ", ""]
)

# Splitting preserves parent metadata across all child chunks!
chunks = splitter.create_documents(
    texts=[d["page_content"] for d in raw_documents],
    metadatas=[d["metadata"] for d in raw_documents]
)
print(f"Generated {len(chunks)} chunk with source: {chunks[0].metadata['source']}")
''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Traced Ingestion & Retrieval Pipeline with LangSmith", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating a complete 
traced retrieval pipeline instrumented with `@traceable(run_type="retriever")`.
""")

st.code('''import time
from typing import List, Dict, Any
from langsmith import Client, traceable

client = Client()

# In-memory document index with metadata
CORPUS = [
    {"doc_id": "doc-101", "text": "LangSmith batch export operations timeout after 300 seconds.", "source": "ops_guide.md"},
    {"doc_id": "doc-102", "text": "Qdrant utilizes HNSW graphs to perform approximate nearest neighbor search.", "source": "vectordb_manual.md"},
    {"doc_id": "doc-103", "text": "Embeddings convert arbitrary text strings into dense float arrays.", "source": "ml_glossary.md"}
]

@traceable(
    run_type="retriever",
    name="production_in_memory_retriever",
    tags=["vector-search", "knowledge-base"]
)
def retrieve_documents(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Traced retriever span: records search query, latency, and returned document payload.
    """
    time.sleep(0.06) # Simulates vector search latency
    query_words = set(query.lower().split())
    
    scored_docs = []
    for d in CORPUS:
        words = set(d["text"].lower().split())
        overlap = len(query_words.intersection(words))
        scored_docs.append({"document": d, "overlap": overlap})
        
    scored_docs.sort(key=lambda x: x["overlap"], reverse=True)
    return [s["document"] for s in scored_docs[:top_k]]

@traceable(run_type="chain", name="end_to_end_rag_chain")
def run_rag_chain(user_query: str) -> Dict[str, Any]:
    """Root chain calling traced retriever and synthesizer."""
    docs = retrieve_documents(user_query)
    context_str = " ".join([d["text"] for d in docs])
    
    # Synthesize response
    answer = f"Synthesized answer based on: {context_str[:60]}..."
    return {
        "query": user_query,
        "answer": answer,
        "retrieved_documents": docs
    }

# Execute Traced Pipeline
res = run_rag_chain("What is the timeout for LangSmith batch export?")
print("Chain completed. Retrieved Docs Count:", len(res["retrieved_documents"]))
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Vector Similarity Calculator & Document Retrieval Inspector", divider="rainbow")

st.markdown("""
Experiment with vector mathematics and simulated retrieval. 
Inspect how token overlap and dense embeddings influence document retrieval rankings and trace metadata.
""")

wb_view = st.selectbox(
    "Select Workbench Tool:",
    ["1. Cosine Similarity Vector Calculator", "2. Simulated VectorDB Retrieval Inspector", "3. Document Ingestion Chunking Tester"]
)

if wb_view == "1. Cosine Similarity Vector Calculator":
    st.subheader("Interactive Cosine Similarity Calculator")
    st.markdown("Adjust the 3D vector coordinates to observe how angle and orientation dictate cosine similarity:")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Vector A (Query Embedding)")
        a_x = st.slider("A_x:", -1.0, 1.0, 0.70, 0.05)
        a_y = st.slider("A_y:", -1.0, 1.0, 0.50, 0.05)
        a_z = st.slider("A_z:", -1.0, 1.0, 0.20, 0.05)
    with col_v2:
        st.markdown("#### Vector B (Document Embedding)")
        b_x = st.slider("B_x:", -1.0, 1.0, 0.65, 0.05)
        b_y = st.slider("B_y:", -1.0, 1.0, 0.55, 0.05)
        b_z = st.slider("B_z:", -1.0, 1.0, 0.15, 0.05)

    sim = data_engine.compute_cosine_similarity([a_x, a_y, a_z], [b_x, b_y, b_z])
    st.metric("Cosine Similarity Score", f"{sim}")
    if sim >= 0.85:
        st.success("High Semantic Match (Strong alignment between query and document vectors)")
    elif sim >= 0.50:
        st.info("Moderate Semantic Match (Broad topical overlap)")
    else:
        st.error("Low Semantic Match (Orthogonal or divergent vectors)")

elif wb_view == "2. Simulated VectorDB Retrieval Inspector":
    st.subheader("VectorDB Query & Span Simulator")
    test_q = st.text_input("Enter Search Query:", "How does LangSmith trace vector embeddings?")
    k_val = st.slider("Top K Retrieved Chunks:", 1, 4, 2)

    retrieved = data_engine.simulate_retrieval(test_q, k_val)
    st.markdown(f"**Retrieved {len(retrieved)} Documents:**")

    for doc in retrieved:
        with st.expander(f"[{doc['id']}] {doc['title']} — Score: {doc['score']}"):
            st.write(doc["content"])

else:
    st.subheader("Document Ingestion & Chunking Sandbox")
    sample_raw_text = st.text_area(
        "Raw Document Content:",
        "LangSmith is designed for production observability. It provides non-blocking tracing.\n\n"
        "Vector databases like Qdrant and Chroma store dense representations. Embeddings map words to high-dimensional space.\n\n"
        "Pandas dataframes structure evaluation results for regression analysis."
    )
    chunk_sz = st.slider("Target Chunk Size (Characters):", 50, 200, 100, 10)
    overlap_sz = st.slider("Chunk Overlap (Characters):", 0, 50, 20, 5)

    # Simplified chunking display
    chunks_sim = [sample_raw_text[i:i+chunk_sz] for i in range(0, len(sample_raw_text), max(chunk_sz - overlap_sz, 1))]
    st.markdown(f"**Generated {len(chunks_sim)} Chunks:**")
    for idx, c in enumerate(chunks_sim, 1):
        st.caption(f"Chunk {idx} (Length: {len(c)} chars):")
        st.code(c, language="markdown")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why is cosine similarity preferred over Euclidean distance for high-dimensional text embeddings?"):
    st.markdown("""
    **Answer:**  
    In high-dimensional embedding spaces, document length variance can cause two semantically identical texts 
    to have a large Euclidean distance simply because one is longer. Cosine similarity measures the angle 
    between vectors rather than magnitude, ensuring length-invariant semantic matching.
    """)

with st.expander("Q2: How does marking retriever functions with run_type='retriever' benefit LangSmith debugging?"):
    st.markdown("""
    **Answer:**  
    Marking a span as `run_type="retriever"` signals to LangSmith that the outputs represent retrieved context. 
    LangSmith formats these outputs as expandable document cards in the web UI, displaying source metadata, 
    page numbers, and relevance scores, enabling rapid identification of retrieval failures.
    """)

with st.expander("Q3: What is the risk of using an embedding model with different dimensionality during database migrations?"):
    st.markdown("""
    **Answer:**  
    Embeddings are coordinate representations within a specific model's latent manifold. 
    If you change embedding models (or dimensionalities), the geometric coordinates are completely incomparable. 
    Attempting to query a Chroma or Qdrant collection with embeddings from a different model produces meaningless random noise.
    """)

with st.expander("Q4: How do Pandas DataFrames facilitate automated regression benchmarking with LangSmith?"):
    st.markdown("""
    **Answer:**  
    LangSmith client methods allow exporting test run feedback scores as structured records. 
    Converting these into Pandas DataFrames allows data scientists to compute aggregate performance metrics, 
    pivot tables by model variant, and perform statistical significance testing across commits.
    """)

with st.expander("Q5: Why should document chunking include an overlap buffer (e.g., 50 tokens)?"):
    st.markdown("""
    **Answer:**  
    Sentences and concepts frequently span across arbitrary chunk boundaries. 
    Without an overlap buffer, critical facts get split in half (e.g., a subject in Chunk 1 and its predicate in Chunk 2), 
    rendering both chunks useless for vector retrieval and grounded generation.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Stripping Source Metadata During Document Loading**  
Discarding document source URLs, page numbers, or timestamps during ingestion prevents the generator 
from providing inline citations and makes it impossible to trace the origin of hallucinated facts.
""")

st.error("""
**Trap 2: Normalizing Vectors in Python Iteration Loops Rather than NumPy C-Arrays**  
Using pure Python loops to calculate vector magnitudes over 100,000 documents causes severe CPU bottlenecks. 
Always use vectorized NumPy routines (`np.linalg.norm(matrix, axis=1)`).
""")

st.warning("""
**Trap 3: Ingesting Massive Chunks Without Sub-Chunk Filtering**  
Creating 4,000-token chunks injects enormous amounts of irrelevant text into the prompt context window, 
increasing latency and token costs while degrading generation accuracy.
""")

st.warning("""
**Trap 4: Forgetting to Track Vector Database Query Latencies**  
Assuming LLM generation is the only slow component can hide unindexed vector search queries 
or slow network round-trips to managed cloud vector stores.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Cosine Similarity</h4>
        <p>A metric measuring the cosine of the angle between two vectors, ranging from -1.0 to 1.0, invariant to document length.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Retriever Span</h4>
        <p>A LangSmith execution span with run_type='retriever' displaying search queries, top_k documents, and metadata cards.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>HNSW Graph</h4>
        <p>Hierarchical Navigable Small World graph, an efficient data structure for fast approximate nearest neighbor search in VectorDBs.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Chunk Overlap</h4>
        <p>A contiguous segment of duplicated tokens shared between adjacent chunks to prevent loss of semantic context across boundaries.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the role of NumPy in vectorized embedding normalization and cosine distance calculations.", value=True)
chk_2 = st.checkbox("I know how to manipulate and aggregate LangSmith experiment feedback runs using Pandas DataFrames.", value=True)
chk_3 = st.checkbox("I can differentiate vector database architectures across Chroma, Qdrant, and Pinecone.", value=True)
chk_4 = st.checkbox("I understand embedding dimensionality trade-offs and the risk of embedding drift during index migrations.", value=True)
chk_5 = st.checkbox("I know how to structure document loaders with recursive text splitting and metadata preservation.", value=True)
chk_6 = st.checkbox("I can instrument retrieval functions with @traceable(run_type='retriever') for transparent LangSmith audits.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Data and Retrieval ecosystem mastery verified! Ready for Application Infrastructure.")
