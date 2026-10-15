import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="21.2 Production Architecture - Vector DB Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# Title & Overview
st.title("21.2 Production Vector Architecture: Microservices, API Layer, Caching, Monitoring & Tracing")
st.caption("Chapter 21: Production Vector Database Architecture | Domain: 4. Vector Database")
st.markdown("""
Moving from a single-file prototype to an enterprise production system serving millions of users requires 
decoupling monolithic scripts into a **Resilient, Scalable Microservices Architecture**.

In production, putting embedding generation, vector search, and LLM calls into a single synchronous thread 
causes catastrophic request queuing, memory leaks, and cascading failures under heavy traffic.

An enterprise production architecture orchestrates **seven decoupled tiers**:
1. **API Gateway & Ingress**: Enforcing TLS 1.3, rate limiting, and JWT authentication.
2. **Semantic Cache Tier (Redis / Vector Cache)**: Intercepting repetitive queries in **<1ms**.
3. **Dedicated Embedding Microservice (TEI / Triton)**: Serving high-throughput batched tensor inference on GPUs.
4. **Distributed Vector Database Cluster**: Autoscaling read replicas for low-latency HNSW graph traversal.
5. **Stage-2 Cross-Encoder Reranker**: Slicing coarse candidate pools down to top precision context.
6. **LLM Gateway & Circuit Breakers**: Managing fallback routing and token streaming.
7. **Full-Stack Observability**: OpenTelemetry distributed traces, Prometheus metrics, and LangSmith evaluation.

In this lesson, we explore the production microservices topology, build an asynchronous FastAPI orchestration pattern, 
examine production telemetry signals, and test an interactive Distributed Trace & Topology Simulator.
""")

st.divider()

# Section 1: The Enterprise Microservices Topology
st.header("1. The Enterprise Microservices Topology", divider="rainbow")

col_top1, col_top2 = col_top1, col_top2 = st.columns([1.1, 0.9])

with col_top1:
    st.markdown("""
    ### The Seven Specialized Tiers
    
    * **Tier 1: API Gateway (Envoy / Kong)**:
      * Validates JWT identity tokens, injects `tenant_id` headers, and applies token-bucket rate limiting.
    
    * **Tier 2: Semantic Cache (Redis Vector Store)**:
      * Evaluates vector proximity against previous queries. Returns cached responses on distance < 0.08, saving 98% latency.
    
    * **Tier 3: Embedding Service (HuggingFace TEI / Triton)**:
      * Decoupled from the web server. Batches vector calculations on GPU with dynamic padding.
    
    * **Tier 4: Vector Database (ChromaDB / Qdrant / Milvus)**:
      * Clustered HNSW storage with sharded memory and replicated query workers.
    
    * **Tier 5: Cross-Encoder Reranker (vLLM / Cohere)**:
      * Reranks top 25 candidates down to top 3–5 dense context chunks.
    
    * **Tier 6: LLM Gateway (LiteLLM / vLLM)**:
      * Manages token streaming, multi-provider fallbacks (OpenAI -> Anthropic), and circuit breaking.
    
    * **Tier 7: Observability Engine (OpenTelemetry + Prometheus)**:
      * Correlates every query with a unique `trace_id` tracking hop latencies and token usage.
    """)

with col_top2:
    st.markdown("### Production Service Mesh Flowchart")
    prod_dot = """
    digraph ProductionMesh {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Client [label="Client App (Web / Mobile)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        APIGW [label="Tier 1: API Gateway (Envoy)\\nAuth + Rate Limiter + WAF", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        AppCore [label="RAG Orchestrator Service\\n(Async Python / FastAPI)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        SemCache [label="Tier 2: Semantic Cache\\n(Redis Vector DB)", fillcolor="#D1FAE5"];
        EmbSvc [label="Tier 3: Embedding Svc\\n(Triton GPU Batch)", fillcolor="#D1FAE5"];
        VecDB [label="Tier 4: Vector DB Cluster\\n(ChromaDB Shards)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        RerankSvc [label="Tier 5: Reranker Microservice\\n(Cross-Encoder)", fillcolor="#D1FAE5"];
        LLMGW [label="Tier 6: LLM Gateway\\n(Streaming Inference)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Client -> APIGW -> AppCore;
        AppCore -> SemCache [label="1. Check Cache (<1ms)"];
        AppCore -> EmbSvc [label="2. Embed Query (4ms)"];
        AppCore -> VecDB [label="3. Search (2ms)"];
        AppCore -> RerankSvc [label="4. Rerank (18ms)"];
        AppCore -> LLMGW [label="5. Stream (600ms)"];
    }
    """
    st.graphviz_chart(prod_dot, use_container_width=False)

st.divider()

# Section 2: Production Asynchronous FastAPI Orchestrator Pattern
st.header("2. Production Asynchronous FastAPI Orchestrator Pattern", divider="rainbow")

st.markdown("""
Review the modular, non-blocking asynchronous Python pattern orchestrating the microservices pipeline:
""")

st.code("""from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import uuid
import time
from typing import Dict, Any

app = FastAPI(title="Enterprise Production RAG Gateway")
security = HTTPBearer()

# Async HTTP Client with Connection Pooling
http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_keepalive_connections=50))

EMBEDDING_SVC_URL = "http://embedding-service.internal:8080/embed"
VECTOR_DB_URL = "http://chroma-cluster.internal:8000/api/v1/collections/kb/query"
RERANKER_SVC_URL = "http://reranker.internal:8000/rerank"
LLM_GATEWAY_URL = "http://llm-gateway.internal:8080/v1/chat/completions"

@app.post("/v1/search_and_generate")
async def execute_rag(
    request: Dict[str, Any],
    auth: HTTPAuthorizationCredentials = Security(security)
):
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    user_query = request["query"]

    # 1. Identity & Tenant Verification
    token = auth.credentials
    tenant_id = verify_jwt_and_extract_tenant(token)

    # 2. Asynchronous Vector Embedding Generation
    emb_res = await http_client.post(
        EMBEDDING_SVC_URL,
        json={"texts": [user_query]},
        headers={"X-Trace-ID": trace_id}
    )
    query_vector = emb_res.json()["embeddings"][0]

    # 3. Vector Database Traversal with Enforced Tenant Filter
    vdb_res = await http_client.post(
        VECTOR_DB_URL,
        json={
            "query_embeddings": [query_vector],
            "n_results": 20,
            "where": {"tenant_id": tenant_id}
        },
        headers={"X-Trace-ID": trace_id}
    )
    candidates = vdb_res.json()["documents"][0]

    # 4. Stage-2 Cross-Encoder Reranking
    rerank_res = await http_client.post(
        RERANKER_SVC_URL,
        json={"query": user_query, "candidates": candidates, "top_n": 3},
        headers={"X-Trace-ID": trace_id}
    )
    top_chunks = rerank_res.json()["ranked_documents"]

    # 5. LLM Synthesis via Streaming Gateway
    # Returns streaming response with trace context
    return {
        "trace_id": trace_id,
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "context_chunks": top_chunks
    }
""", language="python")

st.divider()

# Section 3: Observability & Key Performance Telemetry Signals
st.header("3. Observability & Key Performance Telemetry Signals", divider="rainbow")

st.markdown("""
In production, you cannot fix what you cannot measure. Enterprise SRE teams monitor 
the **RED Method (Rate, Errors, Duration)** across all vector components:
""")

TELEMETRY_SIGNALS = [
    {
        "Component": "Semantic Cache Tier",
        "Key Metric": "`cache_hit_ratio`",
        "Production Target": ">= 30% to 50%",
        "Alert Condition": "Ratio < 10% (Indicates bad distance hit threshold or high query drift)."
    },
    {
        "Component": "Embedding Microservice",
        "Key Metric": "`embedding_latency_p99`",
        "Production Target": "< 10 ms (GPU batched)",
        "Alert Condition": "p99 > 30 ms (Indicates GPU saturation or queue backup; trigger auto-scale)."
    },
    {
        "Component": "Vector DB (ChromaDB)",
        "Key Metric": "`hnsw_query_duration_p95`",
        "Production Target": "< 5 ms",
        "Alert Condition": "p95 > 25 ms (Indicates RAM pressure or OS disk page swapping)."
    },
    {
        "Component": "Cross-Encoder Reranker",
        "Key Metric": "`rerank_duration_p95`",
        "Production Target": "< 25 ms",
        "Alert Condition": "p95 > 60 ms (Candidate pool size too large or CPU worker bottleneck)."
    },
    {
        "Component": "LLM Synthesis Gateway",
        "Key Metric": "`time_to_first_token (TTFT)`",
        "Production Target": "< 400 ms",
        "Alert Condition": "TTFT > 1,500 ms (Upstream provider congestion or prompt context too bloated)."
    }
]

st.dataframe(pd.DataFrame(TELEMETRY_SIGNALS), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Live Enterprise Distributed Trace Simulator
st.header("4. Interactive Workbench: Live Enterprise Distributed Trace Simulator", divider="rainbow")

st.markdown("""
Submit a simulated enterprise search request and watch it propagate across all microservices. 
Inspect the **OpenTelemetry Waterfall Trace**, hop latencies, and circuit breaker health statuses.
""")

col_trace_btn, col_trace_cfg = col_trace_btn, col_trace_cfg = st.columns([1, 1.2])

with col_trace_btn:
    st.subheader("1. Inbound Search Request")
    user_req = st.text_input("User Search Inquiry:", value="What is our disaster recovery RTO and RPO SLA?")
    force_cache_hit = st.checkbox("Simulate Semantic Cache Hit (Returns immediately)", value=False)
    simulate_slow_rerank = st.checkbox("Simulate High Reranker Load (Latency Spike)", value=False)

# Latency modeling
if force_cache_hit:
    t_gw = 0.4
    t_cache = 0.8
    t_embed = 0.0
    t_vdb = 0.0
    t_rerank = 0.0
    t_llm = 0.0
    total_ms = t_gw + t_cache
    path_desc = "Cache Hit -> Bypassed full downstream retrieval"
else:
    t_gw = 0.5
    t_cache = 1.1
    t_embed = 4.2
    t_vdb = 2.1
    t_rerank = 45.0 if simulate_slow_rerank else 18.2
    t_llm = 580.0
    total_ms = t_gw + t_cache + t_embed + t_vdb + t_rerank + t_llm
    path_desc = "Full 6-Tier Distributed Retrieval & LLM Generation"

with col_trace_cfg:
    st.subheader("2. Real-Time Telemetry Summary")
    st.metric(label="End-to-End Latency", value=f"{total_ms:.1f} ms", delta=path_desc)
    st.metric(label="Distributed Trace ID", value="tr-8f90-a2b1-998c", delta="OpenTelemetry / Jaeger")

st.divider()

st.subheader("OpenTelemetry Distributed Trace Waterfall")

trace_data = [
    {"Span Name": "1. API Gateway Ingress (Envoy)", "Service": "api-gateway", "Duration (ms)": t_gw, "Status": "OK (200)"},
    {"Span Name": "2. Semantic Cache Lookup", "Service": "redis-cache", "Duration (ms)": t_cache, "Status": "HIT" if force_cache_hit else "MISS"}
]

if not force_cache_hit:
    trace_data.extend([
        {"Span Name": "3. Embedding Tensor Inference", "Service": "tei-embedding-gpu", "Duration (ms)": t_embed, "Status": "OK (200)"},
        {"Span Name": "4. ChromaDB HNSW Traversal", "Service": "chroma-cluster-vdb", "Duration (ms)": t_vdb, "Status": "OK (200)"},
        {"Span Name": "5. Cross-Encoder Reranking", "Service": "reranker-svc", "Duration (ms)": t_rerank, "Status": "DEGRADED" if simulate_slow_rerank else "OK (200)"},
        {"Span Name": "6. LLM Stream Synthesis", "Service": "llm-gateway", "Duration (ms)": t_llm, "Status": "OK (200)"}
    ])

df_trace = pd.DataFrame(trace_data)
st.dataframe(df_trace, use_container_width=True, hide_index=True)

if simulate_slow_rerank and not force_cache_hit:
    st.warning("⚠️ Telemetry Alert: `rerank_duration_p95` exceeded 40ms threshold! Triggering Kubernetes Horizontal Pod Autoscaler (HPA) to scale reranker pods.")
elif force_cache_hit:
    st.success("🎉 Sub-Millisecond Response: Semantic cache resolved query in 1.2ms without consuming GPU or LLM tokens!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why should the Embedding Generation service be decoupled from the core application server?"):
    st.markdown("""
    **Answer**:
    1. **Independent Resource Profiles**: Web application servers (FastAPI/Node.js) are I/O-bound and require high CPU concurrency. Embedding models (e.g. BGE-large) are matrix-multiplication compute-bound and require **GPU acceleration or multi-core AVX-512 SIMD**.
    2. **Dynamic Tensor Batching**: A dedicated microservice (like HuggingFace Text Embeddings Inference or Triton) can queue incoming requests from 100 web pods and batch them into a single GPU forward pass, multiplying inference throughput by 10x.
    3. **Independent Autoscaling**: If traffic spikes, you can scale the embedding microservice horizontally without replicating web application state.
    """)

with st.expander("Q2: What is a Circuit Breaker, and how is it used in production LLM gateways?"):
    st.markdown("""
    **Answer**:
    A Circuit Breaker (e.g. Resilience4j / pybreaker) protects upstream applications when a downstream cloud dependency (like an LLM API) starts failing:
    * **Closed State**: Normal operation; requests flow through.
    * **Open State**: If the failure rate exceeds 20% (or latency exceeds 5 seconds), the circuit trips OPEN. All incoming requests immediately trigger a local fallback without waiting for timeouts.
    * **Half-Open State**: After a cooldown period (e.g. 30 seconds), a few trial requests are permitted. If they succeed, the circuit resets to CLOSED.
    * **Production Utility**: Prevents application worker threads from hanging indefinitely when an LLM provider suffers an outage.
    """)

with st.expander("Q3: How do you implement OpenTelemetry distributed tracing across a vector RAG pipeline?"):
    st.markdown("""
    **Answer**:
    1. **Context Propagation**: The API Gateway generates a unique `X-Trace-ID` and `traceparent` header (W3C standard) for every inbound request.
    2. **Span Creation**: Each microservice creates child spans for its specific phase:
       ```python
       with tracer.start_as_current_span("chromadb_query") as span:
           span.set_attribute("tenant_id", tenant_id)
           span.set_attribute("k_candidates", 20)
           results = collection.query(...)
       ```
    3. **Centralized Ingestion**: Spans are exported to an OTel Collector and visualized in Jaeger or Datadog, providing an exact millisecond breakdown of each hop.
    """)

with st.expander("Q4: Why should read queries and write ingestion pipelines be routed to separate vector database nodes?"):
    st.markdown("""
    **Answer**:
    * **Resource Contention**: Ingesting new documents requires building HNSW graph edges, acquiring SQLite write locks, and computing neighbor updates, consuming 100% of available CPU cores.
    * If read search queries share the same node, incoming user searches will experience massive p99 latency spikes and queue timeouts.
    * **Architecture**: Deploy a **Leader-Follower cluster**: The Leader node exclusively handles writes and builds indexes; Read Follower replicas exclusively serve search queries.
    """)

with st.expander("Q5: What are the primary failure modes of a distributed semantic cache?"):
    st.markdown("""
    **Answer**:
    1. **Cache Drift / Stale Answers**: Knowledge base documents are updated, but the cache continues serving deprecated information. (Mitigation: Invalidate cache entries when source documents change).
    2. **Loose Distance Hit Collisions**: Cache returns answers to questions with opposite intent. (Mitigation: Restrict hit distance strictly to `< 0.08`).
    3. **Memory Eviction Pressure**: Cache fills RAM, dropping frequently used entries. (Mitigation: Enforce LRU - Least Recently Used eviction policy).
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Synchronous Blocking I/O in Async Web Handlers**  
    Running synchronous `collection.query()` calls inside an async FastAPI handler blocks the event loop, dropping server throughput to single digits. Use async thread pools or async HTTP clients.
    """)
    st.warning("""
    **Trap 2: Missing Timeout Budgets on Downstream Services**  
    Setting no timeout on the LLM API call allows hanging requests to exhaust connection pools, crashing the entire gateway. Enforce strict timeouts (e.g. 5-10s).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Logging Full Prompt Context to Plaintext Logs**  
    Writing raw user prompts and retrieved context to Splunk or Datadog logs exposes sensitive PII and violates compliance regulations. Mask PII before logging.
    """)
    st.warning("""
    **Trap 4: Missing Correlation IDs (Trace IDs)**  
    Troubleshooting an error across 6 microservices without a shared `trace_id` makes root cause analysis impossible. Propagate W3C trace headers everywhere.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is an LLM Circuit Breaker?"):
        st.markdown("""
        **Failure Guard**:  
        Trips open when LLM APIs fail or hang, immediately executing fallback routing to prevent service deadlock.
        """)
    with st.expander("🗂️ Flashcard: Why decouple the Embedding Service?"):
        st.markdown("""
        **Hardware Specialization**:  
        Allows GPU batching and independent autoscaling without overloading CPU web application pods.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What does OpenTelemetry do?"):
        st.markdown("""
        **Distributed Tracing**:  
        Tracks a single user request across all microservices with millisecond span breakdowns.
        """)
    with st.expander("🗂️ Flashcard: Why segregate Reads and Writes?"):
        st.markdown("""
        **Zero Contention**:  
        Prevents heavy HNSW graph construction from locking CPU cores needed for low-latency search queries.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the seven decoupled tiers of enterprise production RAG architecture.", value=True)
chk2 = st.checkbox("I can implement an asynchronous, non-blocking FastAPI orchestration gateway in Python.", value=True)
chk3 = st.checkbox("I understand why the Embedding service must be decoupled from the web application tier.", value=True)
chk4 = st.checkbox("I know how to implement Circuit Breakers and fallback routing for cloud LLM dependencies.", value=True)
chk5 = st.checkbox("I can monitor RED metrics and configure OpenTelemetry distributed tracing across vector services.", value=True)
chk6 = st.checkbox("I can architect a highly-available, production-grade vector database microservices stack.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Production Architecture and are ready for Chapter 21.3: Multi-User Systems!")
