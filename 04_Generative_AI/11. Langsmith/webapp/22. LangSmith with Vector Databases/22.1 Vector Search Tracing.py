import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.1 Vector Search Tracing - LangSmith Master Curriculum",
    page_icon="🔍",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class VectorSearchTracingEngine:
    """
    Production Vector Search Tracing & Run Hierarchy Engine.
    Emulates LangSmith distributed trace collection across query embedding,
    HNSW vector index search, document chunk retrieval, and latency waterfall profiling.
    """
    def __init__(self, project_name: str = "rag-vector-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 100

        # Seed knowledge base documents for retrieval tracing
        self.corpus: List[Dict[str, Any]] = [
            {
                "doc_id": "doc_cloud_01",
                "content": "Kubernetes worker nodes require at least 4 vCPU and 16GB RAM for high-throughput serving.",
                "vector": np.array([0.82, 0.15, 0.45, 0.10]),
                "metadata": {"source": "infra_handbook.md", "tier": "Production", "page": 12}
            },
            {
                "doc_id": "doc_cloud_02",
                "content": "VPC peering connection establishes private routing between multi-region AWS clusters.",
                "vector": np.array([0.18, 0.88, 0.22, 0.35]),
                "metadata": {"source": "networking_guide.pdf", "tier": "Internal", "page": 44}
            },
            {
                "doc_id": "doc_cloud_03",
                "content": "SSL certificates automatically renew through Let's Encrypt ingress cert-manager controllers.",
                "vector": np.array([0.35, 0.25, 0.89, 0.12]),
                "metadata": {"source": "security_runbook.md", "tier": "Production", "page": 5}
            }
        ]

    def trace_vector_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 2,
        distance_metric: str = "cosine",
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a vector search and constructs a full LangSmith trace hierarchy
        including Root Span, Query Embedding Child Span, and VectorDB Search Child Span.
        """
        self._span_counter += 1
        trace_id = f"trace_vs_{self._span_counter}"
        root_run_id = f"run_retriever_{self._span_counter}"
        embed_run_id = f"run_embed_{self._span_counter}"
        vdb_run_id = f"run_vdb_search_{self._span_counter}"

        # 1. Simulated Embedding Latency
        embed_latency_ms = 18.5 + (len(query_text) * 0.05)

        # 2. Vector Similarity Search
        sim_scores = []
        norm_q = np.linalg.norm(query_vector)

        for item in self.corpus:
            # Metadata filter check
            if metadata_filter:
                match = all(item["metadata"].get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue

            v = item["vector"]
            norm_v = np.linalg.norm(v)

            if distance_metric == "cosine":
                score = float(np.dot(query_vector, v) / (norm_q * norm_v)) if norm_q * norm_v > 0 else 0.0
            else:
                # Euclidean distance inverted
                dist = float(np.linalg.norm(query_vector - v))
                score = 1.0 / (1.0 + dist)

            sim_scores.append((score, item))

        # Sort descending
        sim_scores.sort(key=lambda x: x[0], reverse=True)
        retrieved_items = sim_scores[:top_k]

        # 3. Vector DB Search Latency
        vdb_latency_ms = 12.0 + (len(self.corpus) * 0.4)
        total_latency_ms = embed_latency_ms + vdb_latency_ms

        retrieved_docs_payload = [
            {
                "doc_id": doc["doc_id"],
                "score": round(score, 4),
                "content_preview": doc["content"][:60] + "...",
                "metadata": doc["metadata"]
            }
            for score, doc in retrieved_items
        ]

        # Construct LangSmith Trace Representation
        trace_record = {
            "trace_id": trace_id,
            "project_name": self.project_name,
            "root_run": {
                "run_id": root_run_id,
                "name": "VectorRetrieverChain",
                "run_type": "retriever",
                "inputs": {"query": query_text, "top_k": top_k, "filter": metadata_filter},
                "outputs": {"documents": retrieved_docs_payload},
                "total_latency_ms": round(total_latency_ms, 2)
            },
            "child_runs": [
                {
                    "run_id": embed_run_id,
                    "parent_run_id": root_run_id,
                    "name": "TextEmbeddingBiEncoder",
                    "run_type": "embedding",
                    "inputs": {"text": query_text},
                    "outputs": {"dimensions": len(query_vector), "sample": list(np.round(query_vector[:2], 3))},
                    "latency_ms": round(embed_latency_ms, 2),
                    "metadata": {"model": "text-embedding-3-small", "batch_size": 1}
                },
                {
                    "run_id": vdb_run_id,
                    "parent_run_id": root_run_id,
                    "name": "MilvusHNSWVectorIndexSearch",
                    "run_type": "tool",
                    "inputs": {"metric": distance_metric, "top_k": top_k, "index": "hnsw_cosine"},
                    "outputs": {"hits_count": len(retrieved_docs_payload)},
                    "latency_ms": round(vdb_latency_ms, 2),
                    "metadata": {"ef_search": 64, "cluster_node": "data-worker-03"}
                }
            ]
        }
        self.active_traces.append(trace_record)
        return trace_record


# --- HERO SECTION ---
st.title("22.1 Vector Search Tracing")
st.caption("Chapter 22: LangSmith with Vector Databases | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
In production Retrieval-Augmented Generation (RAG) architectures, the retrieval layer is the single greatest determinant of output quality.
When an LLM produces a hallucinated or irrelevant answer, traditional application monitoring merely logs an opaque *"500 Bad Request"* or a successful HTTP 200 with wrong text.
**Vector Search Tracing** in LangSmith brings full distributed observability into the vector database pipeline:
1. **Embedding Generation Spans**: Measuring vectorization duration, token batching efficiency, and bi-encoder model metadata.
2. **Query Vectorization Telemetry**: Tracking real-time query transformations and embedding vector dimensions.
3. **Similarity Search Execution Trees**: Visualizing HNSW graph exploration, distance metrics (Cosine vs L2), and cluster index hops.
4. **Retrieved Document Payloads**: Capturing exact document chunks, similarity scores, and metadata attributes passed to generators.
5. **Latency Waterfall Decomposition**: Isolating network I/O, serialization overhead, and index search bottlenecks to guarantee strict sub-100ms retrieval SLAs.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Distributed Vector Search Tracing & Run Tree Telemetry", divider="rainbow")

st.markdown("""
The architecture below details how LangSmith automatically correlates vector database retrieval into a unified distributed trace.
The **Root Retriever Run** branches into an **Embedding Bi-Encoder Span** and a **Vector Store HNSW Search Span**,
capturing input text, dense vector parameters, distance scores, and exact chunk payloads.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    vs_blueprint_dot = """
    digraph VectorSearchTrace {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_ingress {
            label="1. Ingress Application Tier (LangChain / FastAPI)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            UserQuery [label="User Query: 'How to configure multi-region VPC?'\nFastAPI Endpoint", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            RootRetriever [label="Root Run: VectorRetrieverChain\nrun_id: run_retriever_101 (Parent Span)\nTotal Latency: 32.4ms", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_embedding_span {
            label="2. Child Run A: Query Embedding Generation Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            EmbedSpan [label="Child Span A: TextEmbeddingBiEncoder\nrun_type: embedding\nInputs: query_text | Outputs: [0.18, 0.88, ...]\nModel: text-embedding-3-small (1536d)\nLatency: 18.5ms", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_vdb_span {
            label="3. Child Run B: Vector Store Similarity Search Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            VectorDBSpan [label="Child Span B: Milvus / Qdrant HNSW Search\nrun_type: tool\nMetric: Cosine | ef_search: 64\nMetadata Filter: {tier: 'Production'}\nLatency: 13.9ms", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
            ChunkPayload [label="Retrieved Chunk Payloads\nDOC_02: Score 0.8912 | Page 44\nDOC_01: Score 0.7421 | Page 12", shape=box, fillcolor="#451A03", fontcolor="#FED7AA", color="#F97316"];
        }

        subgraph cluster_langsmith_collector {
            label="4. LangSmith Centralized Observability Platform";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#EC4899";
            fontcolor="#F472B6";
            fontsize=11;

            TraceCollector [label="LangSmith Async Trace Collector\nPOST /api/v1/runs (Zero User Overhead)", shape=component, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            TraceTree [label="Interactive Trace Explorer\nVisual Run Hierarchy & Latency Waterfall", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        UserQuery -> RootRetriever [label="Dispatches Request"];
        RootRetriever -> EmbedSpan [label="1. Invokes Embedder"];
        EmbedSpan -> VectorDBSpan [label="2. Passes Query Vector"];
        VectorDBSpan -> ChunkPayload [label="3. Top-K Candidates"];
        ChunkPayload -> RootRetriever [label="4. Yields Formatted Documents"];

        RootRetriever -> TraceCollector [label="Async Telemetry Event", style=dotted, color="#EC4899"];
        EmbedSpan -> TraceCollector [label="Child Span A Telemetry", style=dotted, color="#EC4899"];
        VectorDBSpan -> TraceCollector [label="Child Span B Telemetry", style=dotted, color="#EC4899"];
        TraceCollector -> TraceTree [label="Correlated Trace Hierarchy"];
    }
    """
    st.graphviz_chart(vs_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: EMBEDDING GENERATION
# ==============================================================================
st.header("2. Embedding Generation (Document Ingestion Spans, Token Batches & Trace Metadata)", divider="rainbow")

st.markdown("""
In production systems, embedding generation occurs in two distinct environments:
1. **Offline Ingestion Pipelines**: Vectorizing millions of documents into thousands of token chunks.
2. **Real-Time Online Ingress**: Vectorizing the dynamic user prompt before database querying.

Without tracing, an ingestion failure leaves broken gaps in the vector database that are impossible to diagnose until users report missing data.
""")

col_eg1, col_eg2 = st.columns([1, 1])

with col_eg1:
    st.markdown("""
    ### Critical Embedding Span Metadata:
    * **`model_name`**: Identifies the exact model checkpoint (e.g., `text-embedding-3-small`, `bge-large-en-v1.5`).
    * **`dimensions`**: Number of vector dimensions (e.g., 1536 or 384). A mismatch between ingest model (1536d) and query model (768d) throws unlogged cosine dimension mismatch exceptions.
    * **`token_count`**: Total tokens in the batch. Tracing token counts prevents hitting API Rate Limits (TPM limits) and provides granular unit cost accounting.
    * **`batch_size`**: Number of chunks processed concurrently.
    """)

with col_eg2:
    st.markdown("""
    ### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
    Modern enterprise stacks instrument their embedding clients using the `@traceable` decorator from `langsmith`:
    ```python
    from langsmith import traceable

    @traceable(run_type="embedding", name="OpenAI_Embeddings")
    def embed_chunks(chunks: List[str]) -> List[List[float]]:
        response = client.embeddings.create(
            input=chunks, 
            model="text-embedding-3-small"
        )
        return [e.embedding for e in response.data]
    ```
    Every batch call automatically streams latency, token count, and model parameters to LangSmith without blocking your main application loop.
    """)

st.divider()

# ==============================================================================
# SECTION 3: QUERY EMBEDDING
# ==============================================================================
st.header("3. Query Embedding (Query Transformation, Bi-Encoder Latency & Model Attributes)", divider="rainbow")

st.markdown("""
In advanced RAG architectures, user queries undergo transformation prior to vectorization:
**Hypothetical Document Embeddings (HyDE)**, multi-query expansion, or semantic step-back prompting.
Tracing the query embedding step captures the exact transformed text passed to the bi-encoder,
revealing whether query rewriting improved semantic alignment or introduced hallucinated noise.
""")

col_qe1, col_qe2 = st.columns([1, 1])

with col_qe1:
    st.markdown("### Query Transformation Trace Lineage")
    query_lineage_df = pd.DataFrame([
        {"Pipeline Step": "1. Raw User Input", "Payload": "VPC private connection", "Latency (ms)": 0.0},
        {"Pipeline Step": "2. HyDE Expansion", "Payload": "A Virtual Private Cloud (VPC) peering connection enables routing between...", "Latency (ms)": 380.0},
        {"Pipeline Step": "3. Query Vectorization", "Payload": "1536-dimensional float vector", "Latency (ms)": 18.5},
        {"Pipeline Step": "Total Query Preparation", "Payload": "Transformed Vector ready for HNSW lookup", "Latency (ms)": 398.5}
    ])
    st.dataframe(query_lineage_df, use_container_width=True)

with col_qe2:
    st.markdown("### Query Transformation Run Span Hierarchy")
    dot_query_span = """
    digraph QueryTransformationSpan {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Parent [label="Parent: QueryPreProcessorChain\nLatency: 398.5ms", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        HyDE [label="Child 1: HyDE_Generator\nLLM Prompt Expansion (380.0ms)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        Embed [label="Child 2: BiEncoder_Embed\nVectorization (18.5ms)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        Parent -> HyDE;
        Parent -> Embed;
    }
    """
    st.graphviz_chart(dot_query_span, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: SIMILARITY SEARCH
# ==============================================================================
st.header("4. Similarity Search (Vector DB Spans, Distance Metrics & Top-K Retrieval Hierarchy)", divider="rainbow")

st.markdown("""
The similarity search span captures the core database interaction:
querying indices (HNSW, IVF-PQ, DiskANN) within vector databases like Milvus, Qdrant, Pinecone, or pgvector.
A single retriever query often executes multiple searches in parallel (e.g., dense vector search + sparse BM25 search).
LangSmith records the **exact search parameters**, enabling engineers to optimize indexing without deploying code changes blindly.
""")

col_ss1, col_ss2 = st.columns([1, 1])

with col_ss1:
    st.markdown("### Essential Search Parameters Captured in Spans:")
    st.markdown("""
    * **`metric_type`**: Cosine, DotProduct (IP), or Euclidean (L2). Misconfiguring this metric in code leads to inverted candidate rankings.
    * **`top_k`**: The number of nearest neighbors requested (typically 3 to 20).
    * **`ef_search` (HNSW)**: The search queue capacity. Higher `ef_search` improves recall but spikes search latency.
    * **`metadata_filter`**: SQL or boolean expression applied to filter chunks by tenant, role, or date.
    """)

with col_ss2:
    st.markdown("""
    ### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
    When using LangChain's vectorstore retrievers with LangSmith enabled, the similarity search span is automatically tagged as a `retriever` run type:
    ```python
    # Automatically logged to LangSmith with zero manual spans
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.80, "k": 5}
    )
    docs = retriever.invoke("How to setup VPC peering?")
    ```
    The resulting LangSmith trace includes the raw vector query, the index configuration, and the similarity scores of all candidate chunks.
    """)

st.divider()

# ==============================================================================
# SECTION 5: RETRIEVED DOCUMENTS
# ==============================================================================
st.header("5. Retrieved Documents (Document Payloads, Relevance Scores & Chunk Metadata Capture)", divider="rainbow")

st.markdown("""
The output of a vector search is a collection of document chunks.
A critical failure mode in RAG occurs when the retriever successfully returns 5 documents, but all 5 documents are **irrelevant, redundant, or outdated**.
LangSmith preserves the exact payload of every retrieved chunk:
its text snippet, similarity score, source URI, page number, and tenant ID.
""")

col_rd1, col_rd2 = st.columns([1, 1])

with col_rd1:
    st.markdown("### Trace Output Inspector: Chunk Payloads")
    st.code('''
{
  "run_id": "run_vdb_search_101",
  "outputs": {
    "documents": [
      {
        "doc_id": "doc_cloud_02",
        "score": 0.8912,
        "content": "VPC peering connection establishes private routing between multi-region AWS clusters.",
        "metadata": {
          "source": "networking_guide.pdf",
          "page": 44,
          "last_updated": "2026-08-15"
        }
      }
    ]
  }
}
''', language="json")

with col_rd2:
    st.markdown("### Why Chunk Payload Capture is Vital:")
    st.markdown("""
    1. **Debugging Hallucinations**:
       When the LLM hallucinates, you can inspect the exact chunks the model read. If the fact was absent from the retrieved chunks, it is a **retrieval failure**; if the fact was present but ignored, it is a **generator failure**.
    2. **Tracking Document Staleness**:
       Inspecting the `last_updated` metadata field allows identifying outdated policy files that were never purged from the vector index.
    3. **Detecting Chunk Boundary Fractures**:
       If a key sentence was severed across two chunks during ingestion, reading the chunk text directly inside the trace immediately reveals the chunking defect.
    """)

st.divider()

# ==============================================================================
# SECTION 6: SEARCH LATENCY
# ==============================================================================
st.header("6. Search Latency (p50/p95/p99 Profiling, Network I/O Waterfall & Index Bottlenecks)", divider="rainbow")

st.markdown("""
RAG latency is characterized by a multi-stage **Latency Waterfall**:
Total Retriever Latency consists of:
`T_total = t_query_rewrite + t_bi_encoder_embed + t_network_roundtrip + t_hnsw_index_search + t_payload_deserialization`.
LangSmith breaks down each phase into microsecond-level spans, allowing teams to isolate whether latency spikes originate from GPU bi-encoder queues or remote vector database socket connections.
""")

col_sl1, col_sl2 = st.columns([1, 1])

with col_sl1:
    st.markdown("### Retrieval Stage Latency Breakdown (Typical P95 Budget):")
    latency_breakdown_df = pd.DataFrame([
        {"Stage": "1. Query Embedding (Bi-Encoder)", "Typical P95 (ms)": "15ms - 25ms", "Bottleneck Root Cause": "GPU Queue depth, token length"},
        {"Stage": "2. Network Transmission (gRPC/HTTP)", "Typical P95 (ms)": "8ms - 15ms", "Bottleneck Root Cause": "Cross-region latency, TLS handshake"},
        {"Stage": "3. HNSW Vector Index Traversal", "Typical P95 (ms)": "12ms - 35ms", "Bottleneck Root Cause": "Index size, ef_search setting, RAM paging"},
        {"Stage": "4. Metadata Filtering & Payloads", "Typical P95 (ms)": "5ms - 10ms", "Bottleneck Root Cause": "Unindexed payload attributes, large JSON bloat"},
        {"Stage": "Total Retrieval Phase", "Typical P95 (ms)": "40ms - 85ms", "Bottleneck Root Cause": "Target SLA: Must complete under 100ms"}
    ])
    st.dataframe(latency_breakdown_df, use_container_width=True)

with col_sl2:
    st.markdown("### Latency Waterfall Span Timeline")
    dot_waterfall = """
    digraph LatencyWaterfall {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        T0 [label="0.0ms: Ingress Request Received", shape=ellipse, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        T1 [label="18.5ms: Query Vector Generated (TextEmbedding)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        T2 [label="24.0ms: gRPC Sent to Milvus Node", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        T3 [label="36.5ms: HNSW Nearest Neighbors Identified", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        T4 [label="42.0ms: Payloads Deserialized & Returned", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        T0 -> T1 -> T2 -> T3 -> T4;
    }
    """
    st.graphviz_chart(dot_waterfall, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Vector Search Tracer (VectorSearchTracingEngine)", divider="rainbow")

st.markdown("""
Below is the verified production engine `VectorSearchTracingEngine`.
It implements complete LangSmith run hierarchy generation, multi-stage latency waterfall tracking,
vector similarity scoring, and structured document payload capture in pure Python.
""")

st.code('''
class VectorSearchTracingEngine:
    """
    Production Vector Search Tracing & Run Hierarchy Engine.
    Emulates LangSmith distributed trace collection across query embedding,
    HNSW vector index search, document chunk retrieval, and latency waterfall profiling.
    """
    def __init__(self, project_name: str = "rag-vector-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 100

        self.corpus: List[Dict[str, Any]] = [
            {
                "doc_id": "doc_cloud_01",
                "content": "Kubernetes worker nodes require at least 4 vCPU and 16GB RAM for high-throughput serving.",
                "vector": np.array([0.82, 0.15, 0.45, 0.10]),
                "metadata": {"source": "infra_handbook.md", "tier": "Production", "page": 12}
            },
            {
                "doc_id": "doc_cloud_02",
                "content": "VPC peering connection establishes private routing between multi-region AWS clusters.",
                "vector": np.array([0.18, 0.88, 0.22, 0.35]),
                "metadata": {"source": "networking_guide.pdf", "tier": "Internal", "page": 44}
            },
            {
                "doc_id": "doc_cloud_03",
                "content": "SSL certificates automatically renew through Let's Encrypt ingress cert-manager controllers.",
                "vector": np.array([0.35, 0.25, 0.89, 0.12]),
                "metadata": {"source": "security_runbook.md", "tier": "Production", "page": 5}
            }
        ]

    def trace_vector_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 2,
        distance_metric: str = "cosine",
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self._span_counter += 1
        trace_id = f"trace_vs_{self._span_counter}"
        root_run_id = f"run_retriever_{self._span_counter}"
        embed_run_id = f"run_embed_{self._span_counter}"
        vdb_run_id = f"run_vdb_search_{self._span_counter}"

        embed_latency_ms = 18.5 + (len(query_text) * 0.05)
        norm_q = np.linalg.norm(query_vector)
        sim_scores = []

        for item in self.corpus:
            if metadata_filter:
                match = all(item["metadata"].get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue

            v = item["vector"]
            norm_v = np.linalg.norm(v)
            if distance_metric == "cosine":
                score = float(np.dot(query_vector, v) / (norm_q * norm_v)) if norm_q * norm_v > 0 else 0.0
            else:
                dist = float(np.linalg.norm(query_vector - v))
                score = 1.0 / (1.0 + dist)
            sim_scores.append((score, item))

        sim_scores.sort(key=lambda x: x[0], reverse=True)
        retrieved_items = sim_scores[:top_k]

        vdb_latency_ms = 12.0 + (len(self.corpus) * 0.4)
        total_latency_ms = embed_latency_ms + vdb_latency_ms

        retrieved_docs_payload = [
            {
                "doc_id": doc["doc_id"],
                "score": round(score, 4),
                "content_preview": doc["content"][:60] + "...",
                "metadata": doc["metadata"]
            }
            for score, doc in retrieved_items
        ]

        trace_record = {
            "trace_id": trace_id,
            "project_name": self.project_name,
            "root_run": {
                "run_id": root_run_id,
                "name": "VectorRetrieverChain",
                "run_type": "retriever",
                "inputs": {"query": query_text, "top_k": top_k, "filter": metadata_filter},
                "outputs": {"documents": retrieved_docs_payload},
                "total_latency_ms": round(total_latency_ms, 2)
            },
            "child_runs": [
                {
                    "run_id": embed_run_id,
                    "parent_run_id": root_run_id,
                    "name": "TextEmbeddingBiEncoder",
                    "run_type": "embedding",
                    "latency_ms": round(embed_latency_ms, 2)
                },
                {
                    "run_id": vdb_run_id,
                    "parent_run_id": root_run_id,
                    "name": "MilvusHNSWVectorIndexSearch",
                    "run_type": "tool",
                    "latency_ms": round(vdb_latency_ms, 2)
                }
            ]
        }
        self.active_traces.append(trace_record)
        return trace_record
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
tracer_engine_inst = VectorSearchTracingEngine(project_name="test-verification-project")

test_trace = tracer_engine_inst.trace_vector_search(
    query_text="How to set up VPC peering across AWS clusters?",
    query_vector=np.array([0.20, 0.85, 0.25, 0.30]),
    top_k=2,
    distance_metric="cosine"
)

assert test_trace["trace_id"].startswith("trace_vs_"), "Trace ID must follow standard naming convention"
assert len(test_trace["child_runs"]) == 2, "Must generate exactly 2 child spans (embedding and vdb search)"
assert test_trace["root_run"]["total_latency_ms"] > 0, "Total latency must be positive"
assert len(test_trace["root_run"]["outputs"]["documents"]) == 2, "Must return Top-2 document payloads"

st.success("Automated Verification Passed: VectorSearchTracingEngine successfully verified across trace tree construction, latency recording, and document payload capture.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Live Vector Run Tree Inspector & Latency Waterfall Simulator", divider="rainbow")

st.markdown("""
Simulate and inspect a live vector search trace in LangSmith.
Adjust query parameters, toggle distance metrics, apply metadata filters, and observe the resulting distributed run hierarchy.
""")

wb_v1, wb_v2 = st.columns([1, 1])

with wb_v1:
    st.markdown("#### Retrieval Query Configuration")
    user_q_preset = st.selectbox(
        "Select User Query Preset:",
        [
            "How to configure VPC peering private connection? (Networking)",
            "What are the Kubernetes worker node sizing limits? (Infra)",
            "How do SSL certificates automatically renew? (Security)"
        ]
    )
    opt_metric = st.selectbox("Vector Distance Metric", ["cosine", "euclidean"])
    opt_top_k = st.slider("Top-K Candidates", min_value=1, max_value=3, value=2)
    opt_apply_filter = st.toggle("Apply Metadata Filter: {tier: 'Production'}", value=False)

with wb_v2:
    st.markdown("#### Live Simulated Vector Query")
    if "VPC" in user_q_preset:
        sim_vec = np.array([0.19, 0.86, 0.23, 0.32])
    elif "Kubernetes" in user_q_preset:
        sim_vec = np.array([0.80, 0.16, 0.44, 0.12])
    else:
        sim_vec = np.array([0.34, 0.26, 0.87, 0.14])

    meta_filter_dict = {"tier": "Production"} if opt_apply_filter else None

    live_trace = tracer_engine_inst.trace_vector_search(
        query_text=user_q_preset,
        query_vector=sim_vec,
        top_k=opt_top_k,
        distance_metric=opt_metric,
        metadata_filter=meta_filter_dict
    )

    st.metric("Total Retrieval Latency", f"{live_trace['root_run']['total_latency_ms']} ms", delta=f"{len(live_trace['root_run']['outputs']['documents'])} Chunks Retrieved")

st.markdown("### LangSmith Trace Explorer: Run Tree Hierarchy")

run_tree_records = [
    {
        "Run ID": live_trace["root_run"]["run_id"],
        "Name": live_trace["root_run"]["name"],
        "Type": live_trace["root_run"]["run_type"],
        "Parent Run ID": "ROOT (None)",
        "Duration (ms)": live_trace["root_run"]["total_latency_ms"]
    }
]
for child in live_trace["child_runs"]:
    run_tree_records.append({
        "Run ID": child["run_id"],
        "Name": child["name"],
        "Type": child["run_type"],
        "Parent Run ID": child["parent_run_id"],
        "Duration (ms)": child["latency_ms"]
    })

st.dataframe(pd.DataFrame(run_tree_records), use_container_width=True)

st.markdown("#### Retrieved Documents Payload Inspector")
docs_table = []
for d in live_trace["root_run"]["outputs"]["documents"]:
    docs_table.append({
        "Document ID": d["doc_id"],
        "Similarity Score": d["score"],
        "Source File": d["metadata"]["source"],
        "Tier": d["metadata"]["tier"],
        "Content Preview": d["content_preview"]
    })
st.dataframe(pd.DataFrame(docs_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What is the difference between a Run and a Trace in LangSmith, and how are vector searches represented?**"):
    st.markdown("""
    **Answer**:
    * A **Run** represents a single unit of execution—such as an LLM call, an embedding vectorization step, a tool execution, or a retriever lookup. Every run has an explicit `run_id`, inputs, outputs, start/end timestamps, and optional `parent_run_id`.
    * A **Trace** is the complete directed acyclic graph (DAG) or tree of runs that starts with a Root Run and encompasses all nested child runs triggered by an end-to-end request.
    * A vector search is represented as a **Retriever Run** (often the parent span) containing two child runs: an **Embedding Run** (vectorizing the query) and a **Tool/VectorDB Run** (performing index search in Milvus/Qdrant).
    """)

with st.expander("**Q2: How does tracing document chunk payloads help resolve 'hallucination vs retrieval' blame attribution?**"):
    st.markdown("""
    **Answer**:
    When a customer complains that the RAG bot answered incorrectly, engineers must immediately determine which subsystem failed:
    1. **Retrieval Failure**: If the retrieved document chunks in the LangSmith trace did not contain the factual answer (or contained contradictory facts), the issue is in the chunking, embedding, or ranking pipeline.
    2. **Generation Failure**: If the retrieved document chunks contained the exact correct fact, but the LLM outputted false information, the failure is prompt following, context window overflow, or model hallucination.
    Without payload tracing, resolving blame attribution requires reproducing the exact query state manually.
    """)

with st.expander("**Q3: What causes high p99 latency spikes during similarity search, and how does LangSmith isolate the bottleneck?**"):
    st.markdown("""
    **Answer**:
    High p99 retrieval spikes typically originate from:
    1. **Bi-Encoder Thread Starvation**: Heavy CPU/GPU contention during embedding generation.
    2. **HNSW RAM Paging**: The vector index exceeds available RAM and the OS pages vectors from disk.
    3. **Unindexed Metadata Filtering**: Searching with restrictive metadata filters without pre-indexed scalar fields, forcing the database into an exhaustive brute-force scan.
    LangSmith isolates the root cause by breaking total latency into child spans: if the `embedding` span is 15ms but the `vdb_search` span is 950ms, the bottleneck is verified to be database index saturation.
    """)

with st.expander("**Q4: Why should vector database tracing be asynchronous and non-blocking in production?**"):
    st.markdown("""
    **Answer**:
    Sending telemetry events synchronously over HTTP to an observability platform would add 30ms - 80ms of latency to every user query.
    LangSmith client libraries solve this by using an in-memory queue and background thread pool worker.
    Trace events are buffered locally and flushed in compressed background batches over HTTP/2, adding zero measurable latency overhead to user-facing retrieval requests.
    """)

with st.expander("**Q5: How do metadata filters in vector search traces help enforce multi-tenant compliance?**"):
    st.markdown("""
    **Answer**:
    In B2B SaaS, an enterprise must prove to auditors that Tenant A never retrieves Tenant B's data.
    By inspecting LangSmith retriever traces, security auditors can inspect the `inputs.filter` field of every similarity search run.
    The presence of `{"tenant_id": "TENANT_ACME"}` on 100% of runs cryptographically demonstrates that the application strictly enforced pre-retrieval tenant isolation.
    """)

st.markdown("### Production Traps & Failure Modes")

vtr_col1, vtr_col2 = st.columns(2)

with vtr_col1:
    st.error("""
    **Trap 1: Omitting Similarity Scores from Retriever Span Outputs**
    * **Failure**: Logging retrieved document text into the trace while discarding the cosine similarity scores returned by the vector store.
    * **Consequence**: Engineers cannot diagnose whether low-quality answers stem from poor retriever confidence (score 0.45) or poor LLM synthesis on high-confidence chunks (score 0.92).
    * **Remedy**: Always serialize both the chunk text and its numerical similarity score into the run outputs.
    """)

    st.error("""
    **Trap 2: Dimension Mismatch Failures Failing Silently**
    * **Failure**: Updating the embedding model from 1536d to 3072d without updating the query embedding span metadata.
    * **Consequence**: The vector database throws dimensional incompatibility errors, causing queries to fail with uninformative 500 errors.
    * **Remedy**: Log the embedding vector dimension explicitly in the embedding child span metadata for instant diagnosis.
    """)

with vtr_col2:
    st.warning("""
    **Trap 3: Tracing Sensitive PII in Raw Document Payloads**
    * **Failure**: Emitting unredacted customer Social Security numbers or credit card numbers inside document chunk trace payloads.
    * **Consequence**: The observability platform becomes a data privacy compliance violation under GDPR and HIPAA.
    * **Remedy**: Apply client-side DLP masking to chunk text before emitting traces, or enable LangSmith PII scrubbing rules.
    """)

    st.warning("""
    **Trap 4: Unbounded Trace Payload Bloat from High Top-K**
    * **Failure**: Requesting Top-K = 50 chunks of 1,000 tokens each and logging all 50 chunks into every run output.
    * **Consequence**: Trace payload sizes explode to hundreds of kilobytes per request, consuming significant client network bandwidth.
    * **Remedy**: Log truncated previews of chunks or limit detailed payload tracing to the top 5 candidates.
    """)

st.markdown("### Core Technical Flashcards")

fcv_c1, fcv_c2 = st.columns(2)

with fcv_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the two essential child spans of a VectorRetriever run in LangSmith?")
        st.markdown("**Back**: 1. Query Embedding Span (vectorizing input text via a bi-encoder), and 2. Vector DB Search Span (executing HNSW index exploration and returning chunk payloads).")

    with st.container(border=True):
        st.markdown("**Front**: How does LangSmith trace data isolate retrieval failures from LLM generation failures?")
        st.markdown("**Back**: By inspecting the exact document payloads and similarity scores inside the retriever span outputs. If the factual answer was absent from the chunks, retrieval failed; if present, the generator failed.")

with fcv_c2:
    with st.container(border=True):
        st.markdown("**Front**: What are the four core latency components of a vector search waterfall?")
        st.markdown("**Back**: 1. Query bi-encoder embedding latency, 2. Network transmission latency, 3. HNSW index traversal latency, and 4. Document payload retrieval and deserialization latency.")

    with st.container(border=True):
        st.markdown("**Front**: Why is non-blocking background telemetry essential for vector search tracing?")
        st.markdown("**Back**: To ensure that sending telemetry payloads to the observability server does not add network latency to the user's synchronous retrieval request.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Vector Search Tracing principles:")

v_chk1 = st.checkbox("I can architect a LangSmith run tree decomposing vector retrieval into embedding and search spans.", value=True)
v_chk2 = st.checkbox("I understand how embedding span metadata (model, dimensions, token count) diagnoses dimension mismatches.", value=True)
v_chk3 = st.checkbox("I know how to capture document chunk payloads, similarity scores, and source metadata in run outputs.", value=True)
v_chk4 = st.checkbox("I can profile the 4 stages of the retrieval latency waterfall to isolate database vs network bottlenecks.", value=True)
v_chk5 = st.checkbox("I understand how to attribute hallucination blame between the retrieval layer and the generator model.", value=True)
v_chk6 = st.checkbox("I know how to sanitize PII and bound trace payload size to maintain GDPR compliance and low overhead.", value=True)

if v_chk1 and v_chk2 and v_chk3 and v_chk4 and v_chk5 and v_chk6:
    st.success("Mastery Certified: You have mastered vector search tracing, run hierarchy instrumentation, and latency waterfall profiling.")
