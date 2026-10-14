import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="27.2 Vector Databases - LangGraph Curriculum",
    page_icon="🧬",
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
    .badge-vdb {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class VectorDBComparisonSimulator:
    """
    Simulates vector database selection tradeoffs:
    - Index types (HNSW vs IVFFlat vs Flat)
    - Latency vs recall benchmarks
    - Metadata filtering capabilities
    """
    def __init__(self):
        self.engines = {
            "FAISS": {"type": "Library (In-Memory / GPU)", "deployment": "Embedded", "latency_ms": 2.1, "max_scale": "Millions", "filtering": "Basic Post-Filter"},
            "Chroma": {"type": "Embedded / Server", "deployment": "Open Source", "latency_ms": 8.4, "max_scale": "Millions", "filtering": "Metadata Dictionary"},
            "Pinecone": {"type": "Managed Cloud", "deployment": "Serverless SaaS", "latency_ms": 24.0, "max_scale": "Billions", "filtering": "Rich Metadata Filters"},
            "Weaviate": {"type": "Hybrid Engine", "deployment": "Cloud / Self-Hosted", "latency_ms": 14.5, "max_scale": "Billions", "filtering": "GraphQL / Filter Vectors"},
            "Qdrant": {"type": "Rust Vector Engine", "deployment": "Cloud / Self-Hosted", "latency_ms": 7.2, "max_scale": "Billions", "filtering": "Deep Payload JSON Filters"},
            "Milvus": {"type": "Distributed Cluster", "deployment": "Kubernetes / Cloud", "latency_ms": 11.0, "max_scale": "Tens of Billions", "filtering": "Boolean Scalar Expressions"},
            "pgvector": {"type": "Postgres Extension", "deployment": "Existing PostgreSQL", "latency_ms": 16.0, "max_scale": "Tens of Millions", "filtering": "Full SQL WHERE Clauses"}
        }

    def get_comparison_table(self) -> List[Dict[str, Any]]:
        rows = []
        for name, data in self.engines.items():
            r = {"Engine": name}
            r.update(data)
            rows.append(r)
        return rows

vdb_sim = VectorDBComparisonSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Vector Database Ecosystem, ANN Indexing, and RAG Integration Topology", divider="rainbow")

st.markdown("""
Vector databases serve as the semantic retrieval engine for Retrieval-Augmented Generation (RAG) within LangGraph. 
By converting unstructured textual data into high-dimensional dense embeddings (e.g., 1536-dimensional or 3072-dimensional vectors), 
vector databases enable agents to perform Approximate Nearest Neighbor (ANN) search to identify relevant context, 
documentation, and historical cases in milliseconds.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VectorBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_ingestion {
            label="1. Document Embedding Ingestion Pipeline (Offline / Asynchronous)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            docs_in [label="Enterprise Knowledge Docs\\n(PDFs, Markdown, Web Crawls)", fillcolor="#0F172A"];
            chunker [label="Recursive Semantic Chunker\\n512 Tokens with 50 Token Overlap", fillcolor="#1E293B"];
            embedder [label="Embedding Model (text-embedding-3-small)\\nProduces 1536-d Vectors", fillcolor="#312E81"];
        }

        subgraph cluster_vdb_tier {
            label="2. Vector Database Engine (HNSW Index + Metadata Store)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            vdb_cluster [label="Vector Store (Qdrant / Pinecone / pgvector)\\nHierarchical Navigable Small World (HNSW) Index\\nCosine Distance Metric", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_graph_runtime {
            label="3. LangGraph Execution & Retrieval Flow";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            user_query [label="User Query Input", fillcolor="#0F172A"];
            retriever_node [label="Retriever Node\\nPerforms ANN Vector Search + Hybrid BM25", fillcolor="#1E293B"];
            synthesizer [label="Agent Reasoning Node\\nSynthesizes Grounded Answer", fillcolor="#0F172A"];
        }

        docs_in -> chunker;
        chunker -> embedder;
        embedder -> vdb_cluster [label="Upsert Vectors & Payloads"];

        user_query -> retriever_node;
        retriever_node -> vdb_cluster [label="Query Vector (Top K=5)"];
        vdb_cluster -> retriever_node [label="Ranked Document Chunks"];
        retriever_node -> synthesizer;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: FAISS
# ==============================================================================
st.header("2. FAISS: In-Memory High-Speed Dense Vector Search and Local Index Prototyping", divider="rainbow")

st.markdown("""
Developed by Meta AI, **FAISS** (Facebook AI Similarity Search) is a C++ library with Python bindings optimized for 
efficient similarity search of dense vectors on CPU and GPU.

### Key Characteristics:
- **Zero Server Overhead**: Embedded directly in Python processes without managing a database service.
- **Blazing Fast**: Microsecond-level lookups using optimized BLAS linear algebra and GPU acceleration.
- **Limitations**: In-memory only (ephemeral unless serialized to disk), lacks integrated metadata filtering or distributed scalability.
""")

# ==============================================================================
# SECTION 3: Chroma
# ==============================================================================
st.header("3. Chroma: Lightweight Embedded and Client-Server Open-Source Vector Store", divider="rainbow")

st.markdown("""
**Chroma** is an open-source, AI-native vector database designed for developer ergonomics and local prototyping:
- **Embedded Mode**: Runs in-memory or persists to local SQLite/DuckDB files (`persist_directory="./chroma_db"`).
- **Client-Server Mode**: Deployable as a lightweight Docker container for small team microservices.
- **Ideal For**: Local testing, CI/CD integration tests, and desktop applications.
""")

# ==============================================================================
# SECTION 4: Pinecone
# ==============================================================================
st.header("4. Pinecone: Serverless Cloud Vector Database with Managed Metadata Filtering", divider="rainbow")

st.markdown("""
**Pinecone** is a fully managed, cloud-native vector database built for hands-off enterprise scalability:
- **Serverless Architecture**: Pay strictly per read/write request with zero compute cluster provisioning.
- **Real-Time Indexing**: Immediate vector upserts and low-latency p95 query responses.
- **Metadata Filtering**: First-class support for combining vector similarity with complex JSON attribute filters.
""")

# ==============================================================================
# SECTION 5: Weaviate
# ==============================================================================
st.header("5. Weaviate: Multi-Modal GraphQL Vector Search Engine with Hybrid Inverted Indexes", divider="rainbow")

st.markdown("""
**Weaviate** is an open-source, cloud-native vector database that combines vector search with structured BM25 keyword search:
- **Native Hybrid Search**: Merges dense vector embeddings with sparse keyword inverted indexes using Reciprocal Rank Fusion (RRF).
- **Multi-Modal Support**: Indexes text, images, and audio in a shared embedding space.
- **GraphQL & REST APIs**: Flexible query syntax for filtering objects and traversing graph relations.
""")

# ==============================================================================
# SECTION 6: Qdrant
# ==============================================================================
st.header("6. Qdrant: High-Throughput Rust-Powered Vector Store with Extended Payload Filtering", divider="rainbow")

st.markdown("""
**Qdrant** is a high-performance vector search engine written in Rust:
- **Deep Payload Filtering**: Evaluates complex metadata conditions (arrays, nested objects, geo-coordinates) 
  during vector graph traversal, preventing recall degradation.
- **Quantization Support**: Supports scalar and product quantization (PQ) to reduce RAM footprint by up to 4x.
- **Enterprise Proven**: Available as an open-source container and fully managed cloud cluster.
""")

# ==============================================================================
# SECTION 7: Milvus
# ==============================================================================
st.header("7. Milvus: Distributed Enterprise-Scale Vector Cluster for Billion-Scale Vector Collections", divider="rainbow")

st.markdown("""
**Milvus** is a cloud-native, distributed vector database engineered for massive scale:
- **Billion-Scale Collections**: Decouples storage and compute using object storage (S3/MinIO) and distributed query nodes.
- **Multiple Index Types**: Supports HNSW, IVF-FLAT, SCaNN, and GPU-accelerated indexes.
- **Best Suited For**: Global enterprise deployments managing billions of document chunks across multi-region clusters.
""")

# ==============================================================================
# SECTION 8: pgvector
# ==============================================================================
st.header("8. pgvector: Extending PostgreSQL Relational Databases with HNSW and IVFFlat Vector Indexes", divider="rainbow")

st.markdown("""
**`pgvector`** is an open-source extension for PostgreSQL that allows storing dense vectors directly inside relational tables:

### Why pgvector Dominates Enterprise Architectures:
1. **Single Database Simplicity**: Store user profiles, billing records, LangGraph checkpoints, and semantic vector embeddings 
   inside the exact same PostgreSQL database.
2. **ACID Transactions**: Join vector search results directly with relational business tables in a single SQL query:
""")

st.code('''-- Performing vector search joined with relational tenant filtering in PostgreSQL
SELECT 
    d.id,
    d.title,
    d.content,
    1 - (d.embedding <=> :query_vector) AS cosine_similarity
FROM enterprise_documents d
JOIN tenant_permissions p ON d.tenant_id = p.tenant_id
WHERE p.user_id = :current_user_id
  AND d.category = 'financial_reports'
ORDER BY d.embedding <=> :query_vector
LIMIT 5;
''', language="sql")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Engine Vector Retrieval Pipeline with Metadata Filters", divider="rainbow")

st.markdown("""
The following complete module implements an enterprise vector retriever node compatible with pgvector and Qdrant, 
supporting tenant isolation and minimum similarity score thresholds.
""")

st.code('''from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage

class RAGAgentState(TypedDict):
    query: str
    tenant_id: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str

class ProductionVectorRetriever:
    """
    Simulates production vector database query client
    with tenant-level metadata filtering and threshold gating.
    """
    def __init__(self, min_similarity_threshold: float = 0.75):
        self.min_similarity_threshold = min_similarity_threshold

    def search(self, query: str, tenant_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # In production: vector_client.search(collection="docs", query_vector=vec, filter={"tenant": tenant_id})
        mock_chunks = [
            {"id": "doc_1", "text": "Q3 Revenue reached 42M USD, exceeding projections by 8%.", "similarity": 0.89, "tenant": tenant_id},
            {"id": "doc_2", "text": "Operating expenses decreased by 3.2% year-over-year.", "similarity": 0.82, "tenant": tenant_id},
            {"id": "doc_3", "text": "Global headcount expanded by 140 engineers in Q3.", "similarity": 0.68, "tenant": tenant_id}
        ]
        # Gate by similarity threshold
        return [c for c in mock_chunks if c["similarity"] >= self.min_similarity_threshold]

retriever_service = ProductionVectorRetriever(min_similarity_threshold=0.75)

def retriever_node(state: RAGAgentState) -> dict:
    docs = retriever_service.search(state["query"], state["tenant_id"])
    return {"retrieved_docs": docs}

def synthesizer_node(state: RAGAgentState) -> dict:
    context_str = "\\n".join([f"- {d['text']}" for d in state["retrieved_docs"]])
    ans = f"Based on retrieved documents:\\n{context_str}"
    return {"final_answer": ans}
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Vector Database Selector & Similarity Metric Comparator", divider="rainbow")

st.markdown("""
Compare vector database architectures, evaluation latencies, and metadata filtering capabilities across the ecosystem.
""")

comp_data = vdb_sim.get_comparison_table()
df_comp = pd.DataFrame(comp_data)
st.dataframe(df_comp, use_container_width=True)

col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    st.subheader("Interactive Vector DB Advisor")
    scale_req = st.selectbox("Expected Vector Scale", ["Local Testing (< 50K)", "Production (1M - 50M)", "Billion-Scale Enterprise (> 1B)"])
    deploy_pref = st.selectbox("Deployment Preference", ["Existing PostgreSQL", "Self-Hosted Container", "Managed Serverless Cloud"])

with col_ctrl2:
    st.subheader("Recommended Architecture")
    if scale_req == "Local Testing (< 50K)":
        st.success("Recommended: **Chroma** or **FAISS** (Zero ops, in-memory/embedded speed).")
    elif deploy_pref == "Existing PostgreSQL":
        st.success("Recommended: **pgvector** with HNSW indexing (Unified database, joins, zero extra infrastructure).")
    elif deploy_pref == "Managed Serverless Cloud":
        st.success("Recommended: **Pinecone** or **Qdrant Cloud** (Managed scaling, low ops, sub-25ms latency).")
    else:
        st.success("Recommended: **Qdrant** or **Milvus** (High throughput, deep metadata filtering, cloud-native).")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What is the difference between HNSW and IVFFlat index algorithms in vector databases?"):
    st.markdown("""
    **Answer:**  
    - **HNSW (Hierarchical Navigable Small World)**: Constructs a multi-layer graph of vectors. Delivers superior recall and ultra-fast query latency, but consumes significantly more RAM to maintain graph structures.  
    - **IVFFlat (Inverted File Flat)**: Clusters vectors into Voronoi cells. Requires less RAM and builds faster, but delivers lower recall on high-dimensional vectors and requires periodic retraining as data changes.
    """)

with st.expander("Q2: Why is pgvector often preferred over dedicated vector stores in enterprise architectures?"):
    st.markdown("""
    **Answer:**  
    `pgvector` avoids the "split-brain" architecture where relational data lives in PostgreSQL and vector data lives in a separate database. 
    It enables transactional consistency, standard database backups, unified role-based permissions, and joining relational tables 
    with vector searches in a single SQL query.
    """)

with st.expander("Q3: What causes recall degradation during metadata pre-filtering in vector databases?"):
    st.markdown("""
    **Answer:**  
    In naive graph-based ANN algorithms, if an aggressive metadata filter excludes 99% of nodes, the search graph can become disconnected, 
    preventing the traversal from locating the nearest matching neighbors. Advanced vector databases (like Qdrant) implement 
    payload-aware graph exploration or fallback to iterative payload scanning to preserve recall.
    """)

with st.expander("Q4: How does Reciprocal Rank Fusion (RRF) work in hybrid search engines like Weaviate?"):
    st.markdown("""
    **Answer:**  
    RRF combines ranked lists from dense vector search and sparse BM25 keyword search:  
    `RRF_score(doc) = sum(1 / (60 + rank_i(doc)))`  
    This prioritizes documents that score high in both semantic meaning and exact keyword matches without requiring score normalization.
    """)

with st.expander("Q5: What is vector quantization and why is it essential for billion-scale deployments?"):
    st.markdown("""
    **Answer:**  
    Quantization (scalar or product quantization) compresses floating-point vector dimensions (e.g., float32 to int8). 
    This reduces RAM consumption by 4x-8x with only marginal losses in search recall, making billion-scale vector indexes financially feasible.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Dimension Mismatch Between Embedding Model and Vector Index**  
Creating an index configured for 1536 dimensions and attempting to query it with a 3072-dimensional embedding model 
causes runtime dimension mismatch errors. Always bind index creation parameters to model dimensions.
""")

st.error("""
**Trap 2: Ignoring Distance Metric Alignment**  
Using Euclidean distance (L2) on normalized vectors trained for Cosine similarity degrades retrieval accuracy. 
Ensure the vector index distance metric matches the embedding provider's training objective.
""")

st.warning("""
**Trap 3: Full Table Vector Scans Due to Missing HNSW Indexes in pgvector**  
Querying `pgvector` columns without creating an HNSW or IVFFlat index triggers sequential full-table scans, 
causing queries to slow down from milliseconds to tens of seconds as collections grow.
""")

st.warning("""
**Trap 4: Storing Entire Document Payloads in RAM-Bound Vector Stores**  
Storing multi-megabyte raw document texts directly inside in-memory vector stores bloats RAM costs. 
Store vector embeddings and small metadata payloads in the vector DB, keeping raw text in object storage (S3/GCS).
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>HNSW Index</h4>
        <p>A multi-layered graph indexing algorithm providing industry-leading Approximate Nearest Neighbor search speed and recall.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>pgvector</h4>
        <p>PostgreSQL extension enabling dense vector storage, Cosine distance indexing, and unified SQL relational queries.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Hybrid Search (RRF)</h4>
        <p>Combining dense semantic embeddings with sparse keyword BM25 retrieval to maximize precision and contextual relevance.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Payload Filtering</h4>
        <p>Filtering vector candidates by structured metadata attributes (tenant ID, department, date) during vector traversal.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architectural differences between FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus, and pgvector.", value=True)
chk_2 = st.checkbox("I know how to index vectors using HNSW and IVFFlat algorithms.", value=True)
chk_3 = st.checkbox("I understand why pgvector is often preferred for unified enterprise data and relational joins.", value=True)
chk_4 = st.checkbox("I can implement tenant-scoped metadata filtering to enforce strict data isolation in RAG.", value=True)
chk_5 = st.checkbox("I know how to combine vector search with BM25 keyword retrieval using Reciprocal Rank Fusion.", value=True)
chk_6 = st.checkbox("I understand how vector quantization reduces RAM consumption for massive collections.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Vector database engineering mastery confirmed! Ready for External API integration.")
