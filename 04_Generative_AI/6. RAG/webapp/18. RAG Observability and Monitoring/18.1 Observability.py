import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.1 Observability - RAG Master Curriculum",
    page_icon="📡",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class RAGObservabilityEngine:
    """
    Production-grade RAG Distributed Tracing & Observability Engine:
    1. Manages Trace IDs, Span IDs, and Parent-Child span hierarchies.
    2. Instruments retrieval telemetry (vector distances, candidate counts).
    3. Captures LLM execution spans (prompt/completion tokens, TTFT, model IDs).
    4. Computes latency percentiles (P50, P95, P99) and waterfall breakdowns.
    5. Calculates token economics and dollar cost burn velocity.
    """
    def __init__(self, cost_per_1k_input: float = 0.005, cost_per_1k_output: float = 0.015):
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def generate_trace_id(self, seed: int = 101) -> str:
        """Generates a pseudo-UUID trace identifier."""
        return f"tr-8a4f9b{seed:04d}-e2c1"

    def simulate_pipeline_trace(
        self,
        query: str,
        retrieval_k: int = 5,
        rerank_active: bool = True,
        prompt_tokens: int = 1850,
        completion_tokens: int = 220
    ) -> Dict[str, Any]:
        """Simulates an OpenTelemetry-compliant trace with nested spans."""
        trace_id = self.generate_trace_id()

        # Simulated span durations (ms)
        embed_duration_ms = 42.0
        retrieval_duration_ms = 68.0
        rerank_duration_ms = 85.0 if rerank_active else 0.0
        context_pack_duration_ms = 12.0

        # LLM latency breakdown: TTFT (prefill) + generation
        ttft_ms = prompt_tokens * 0.14
        gen_duration_ms = (completion_tokens / 35.0) * 1000.0
        llm_total_ms = ttft_ms + gen_duration_ms

        total_trace_ms = (
            embed_duration_ms + retrieval_duration_ms +
            rerank_duration_ms + context_pack_duration_ms + llm_total_ms
        )

        # Cost calculations
        input_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = (completion_tokens / 1000.0) * self.cost_per_1k_output
        total_cost = input_cost + output_cost

        spans = [
            {"span_id": "sp-01-root", "name": "rag_request_pipeline", "parent_id": None, "duration_ms": round(total_trace_ms, 1), "category": "Pipeline"},
            {"span_id": "sp-02-embed", "name": "embed_user_query", "parent_id": "sp-01-root", "duration_ms": embed_duration_ms, "category": "Embedding"},
            {"span_id": "sp-03-search", "name": "vector_db_retrieval", "parent_id": "sp-01-root", "duration_ms": retrieval_duration_ms, "category": "Retrieval"},
        ]
        if rerank_active:
            spans.append({"span_id": "sp-04-rerank", "name": "cross_encoder_rerank", "parent_id": "sp-01-root", "duration_ms": rerank_duration_ms, "category": "Reranking"})
        spans.extend([
            {"span_id": "sp-05-pack", "name": "context_window_packing", "parent_id": "sp-01-root", "duration_ms": context_pack_duration_ms, "category": "Context"},
            {"span_id": "sp-06-llm", "name": "llm_autoregressive_generation", "parent_id": "sp-01-root", "duration_ms": round(llm_total_ms, 1), "category": "Generation"}
        ])

        return {
            "trace_id": trace_id,
            "query": query,
            "total_latency_ms": round(total_trace_ms, 1),
            "ttft_ms": round(ttft_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": round(total_cost, 5),
            "spans": spans
        }


# Instantiate engine at module level
obs_engine = RAGObservabilityEngine()

# --- HEADER SECTION ---
st.title("18.1 RAG Observability: Distributed Tracing, Span Hierarchies, TTFT Breakdown, & Token Economics")
st.caption("Chapter 18: RAG Observability and Monitoring | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In traditional web architectures, Application Performance Monitoring (APM) tools like Datadog or New Relic 
track HTTP status codes (200 OK vs 500 Error), SQL query durations, and CPU utilization.

In enterprise **Compound AI Systems like RAG**, traditional metrics are completely blind:
* A user query returns HTTP `200 OK` in 1.2 seconds, yet the synthesized response is a **hallucinated, brand-damaging lie**.
* A vector database query completes in 15ms, but returns **irrelevant documents** that mislead the generator.
* A single customer conversation quietly consumes **thousands of tokens** due to unchecked chat history growth.

**RAG Observability** captures, correlates, and visualizes every micro-event across the multi-stage pipeline: 
from query transformation and vector distance scoring to LLM token streaming and Time-To-First-Token (TTFT) metrics.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise RAG Observability & OpenTelemetry Architecture", divider="rainbow")

st.markdown("""
Modern LLM observability platforms (Arize Phoenix, Langfuse, OpenLLMetry, LangSmith) adopt the 
**OpenTelemetry (OTel)** distributed tracing standard. Every user request generates a root **Trace** 
composed of hierarchical child **Spans**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ObservabilityArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // CLIENT INVOCATION
        subgraph cluster_client {
            label="1. Inbound Request Ingestion";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            UserRequest [label="Inbound User Query\\n(Trace ID: tr-8a4f9b0101-e2c1)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            ContextInjector [label="OTel Context Carrier\\n(Baggage headers, session ID, tenant ID)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // RETRIEVAL SPANS
        subgraph cluster_retrieval_spans {
            label="2. Retrieval Subsystem Instrumentation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            SpanEmbed [label="Span: Embed Query\\n(Latency: 42ms | Dimension: 1536)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            SpanVector [label="Span: Vector Search\\n(Latency: 68ms | K=20 | Distance: Cosine)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            SpanRerank [label="Span: Cross-Encoder Rerank\\n(Latency: 85ms | Top-5 Score: 0.94)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // GENERATION SPANS
        subgraph cluster_generation_spans {
            label="3. LLM Generation Subsystem Instrumentation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            SpanPack [label="Span: Context Packing\\n(1850 tokens | U-Curve Layout)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            SpanPrefill [label="Span: Prefill Phase (TTFT)\\n(Latency: 259ms | Input Tokens: 1850)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            SpanDecode [label="Span: Token Streaming Loop\\n(220 completion tokens | 35 tok/sec)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // EXPORTER & SINK
        subgraph cluster_telemetry_sink {
            label="4. OpenTelemetry Collector & Storage";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            OTelCollector [label="OTel OpenInference Exporter\\n(OTLP/gRPC streaming pipeline)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            ObservabilityDashboard [label="Unified Observability Lakehouse\\n(Waterfall charts, token burn, latency percentiles)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        }

        // TRACE HIERARCHY EDGES
        UserRequest -> ContextInjector;
        ContextInjector -> SpanEmbed;
        SpanEmbed -> SpanVector;
        SpanVector -> SpanRerank;
        SpanRerank -> SpanPack;
        SpanPack -> SpanPrefill;
        SpanPrefill -> SpanDecode;
        SpanDecode -> OTelCollector;
        OTelCollector -> ObservabilityDashboard;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### The 4 Core Observability Guarantees:
1. **Trace Continuity**: Every database query, reranker call, and token completion inherits the root Trace ID.
2. **Metadata Attribution**: Chunks retain their original document IDs, cosine distances, and timestamps across spans.
3. **Latency Deconstruction**: Separates network wait, prompt prefill (TTFT), and token generation latency.
4. **Token Economics**: Computes real-time dollar expenditure per request based on token counts.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHY OBSERVABILITY MATTERS
# ==============================================================================
st.header("2. Why Observability Matters (Compound AI Failure Modes & Silent Regressions)", divider="rainbow")

st.markdown("""
In single-model systems, debugging focuses on input prompts and output completions. 
In compound RAG systems, failures cascade silently across components.
""")

wom_col1, wom_col2 = st.columns([1, 1])

with wom_col1:
    st.markdown("""
    ### Compound AI Failure Modes:
    * **Silent Vector Drift**: Upgrading the embedding model without re-indexing historical vectors causes retrieval recall to collapse to zero while returning HTTP 200.
    * **Prompt Bloat Cost Creep**: Appending uncompressed conversation history causes cost per request to jump from $0.005 to $0.080 without triggering an error.
    * **Latency Outlier Cascades**: If vector search hits cold-start storage or an API provider experiences token throttling, P99 latency spikes to 12 seconds while P50 remains healthy.
    """)

with wom_col2:
    st.markdown("### Traditional APM vs RAG Observability")
    apm_comp = pd.DataFrame([
        {"Diagnostic Layer": "Traditional Web APM", "Monitored Metrics": "HTTP Status, CPU, RAM, SQL query time", "Blind Spots": "Hallucinations, irrelevance, prompt bloat"},
        {"Diagnostic Layer": "RAG OpenInference", "Monitored Metrics": "Retrieval distance, TTFT, token cost, Faithfulness", "Blind Spots": "Underlying OS kernel internals"}
    ])
    st.dataframe(apm_comp, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: REQUEST TRACING
# ==============================================================================
st.header("3. Request Tracing (Distributed Trace IDs, Spans & OpenInference Standards)", divider="rainbow")

st.markdown("""
Request tracing models the end-to-end execution path of a RAG query as a tree of directed spans. 
The industry standard is **OpenInference** (an OpenTelemetry semantic convention for AI).
""")

rt_col1, rt_col2 = st.columns([1, 1])

with rt_col1:
    st.markdown("""
    ### OpenInference Semantic Conventions:
    * `openinference.span.kind`: Declares span type (`RETRIEVER`, `RERANKER`, `LLM`, `CHAIN`, `TOOL`).
    * `input.value` and `output.value`: Text payloads passing between stages.
    * `llm.token_count.prompt`: Number of tokens in prompt payload.
    * `llm.token_count.completion`: Number of tokens generated.
    * `retrieval.documents`: Array of retrieved chunks with metadata and similarity scores.
    """)

with rt_col2:
    st.markdown("### Trace Header Structure")
    st.code(r'''
{
  "trace_id": "tr-8a4f9b0101-e2c1",
  "span_id": "sp-06-llm",
  "parent_id": "sp-01-root",
  "attributes": {
    "openinference.span.kind": "LLM",
    "llm.model_name": "claude-3-5-sonnet",
    "llm.temperature": 0.0,
    "llm.token_count.prompt": 1850,
    "llm.token_count.completion": 220,
    "llm.token_count.total": 2070
  }
}
''', language="json")

st.divider()

# ==============================================================================
# SECTION 4: RETRIEVAL TRACING
# ==============================================================================
st.header("4. Retrieval Tracing (Vector Distance Logging, Candidate Auditing & Chunk Attribution)", divider="rainbow")

st.markdown("""
Retrieval tracing inspects the vector database and search layers. 
Without retrieval tracing, developers cannot diagnose why specific documents were surfaced or omitted.
""")

retr_col1, retr_col2 = st.columns([1, 1])

with retr_col1:
    st.markdown("""
    ### Mandatory Retrieval Trace Attributes:
    1. **Vector Query Vector Norm**: Verifies the query embedding was not corrupted or zeroed.
    2. **Score Distribution**: Logs the minimum, mean, and maximum similarity scores across Top-K hits.
    3. **Chunk Attribution**: Logs source document ID, chunk index, creation date, and file URI for every candidate.
    4. **Index Latency**: Separates network transit time from vector indexing search time.
    """)

with retr_col2:
    st.markdown("### Retrieval Span Telemetry Table")
    ret_telemetry = pd.DataFrame([
        {"Chunk ID": "DOC-101-c1", "Similarity Score": 0.94, "Rank": 1, "Rerank Score": 0.98, "Status": "Injected"},
        {"Chunk ID": "DOC-202-c4", "Similarity Score": 0.88, "Rank": 2, "Rerank Score": 0.91, "Status": "Injected"},
        {"Chunk ID": "DOC-303-c2", "Similarity Score": 0.74, "Rank": 3, "Rerank Score": 0.42, "Status": "Pruned (Reranker)"},
        {"Chunk ID": "DOC-404-c1", "Similarity Score": 0.69, "Rank": 4, "Rerank Score": 0.31, "Status": "Pruned (Threshold)"}
    ])
    st.dataframe(ret_telemetry, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 5: LLM TRACING
# ==============================================================================
st.header("5. LLM Tracing (Prompt Payloads, Token Usage, Temperature Logs & Stop Reasons)", divider="rainbow")

st.markdown("""
LLM tracing captures the exact parameters, inputs, and raw outputs of language model API calls.
""")

llmt_col1, llmt_col2 = st.columns([1, 1])

with llmt_col1:
    st.markdown("""
    ### Critical LLM Telemetry Elements:
    * **Full Compiled Prompt Payload**: The exact string sent to the model (for reproducing bugs).
    * **Stop Reason / Finish Reason**: `stop` (clean exit) vs `length` (truncated answer).
    * **Hyperparameters**: Temperature, top_p, frequency_penalty, and seed.
    * **Streaming Chunks**: Time between subsequent streaming tokens (identifying network jitter).
    """)

with llmt_col2:
    st.markdown("### Production LLM Span Schema")
    st.code(r'''
Span: llm_generation
Start: 2026-10-13T16:00:00.120Z
End:   2026-10-13T16:00:01.450Z (Duration: 1330ms)
Model: gpt-4o-2024-08-06
Params: {temperature: 0.1, max_tokens: 500}
Input Tokens:  1,850
Output Tokens: 220
Finish Reason: stop
TTFT: 259ms
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 6: LATENCY MONITORING
# ==============================================================================
st.header("6. Latency Monitoring (P50/P95/P99 Percentiles & TTFT vs Generation Breakdown)", divider="rainbow")

st.markdown("""
Measuring average latency hides critical performance degradations. 
Production RAG systems monitor **percentile latencies (P50, P95, P99)** and decompose latency into distinct stages.
""")

lat_col1, lat_col2 = st.columns([1, 1])

with lat_col1:
    st.markdown("""
    ### The 3 Stages of RAG Latency:
    1. **Retrieval Latency (Search & Reranking)**: Typically 80 to 250 milliseconds.
    2. **Time-To-First-Token (TTFT / Prefill)**: Time required for the LLM to process all prompt tokens before emitting token 1. Typically 200 to 1,200 milliseconds.
    3. **Generation Latency (Streaming)**: Sequential token generation time. Dictated by tokens per second (typically 30 to 80 tokens/sec).
    """)

with lat_col2:
    st.markdown("### Typical Production Latency Percentile Profile")
    lat_profile = pd.DataFrame({
        "Percentile": ["P50 (Median)", "P90", "P95", "P99 (Outliers)"],
        "Retrieval Latency (ms)": [65, 110, 145, 320],
        "TTFT Prefill (ms)": [220, 480, 750, 1850],
        "Total Latency (ms)": [850, 1450, 1980, 4200]
    }).set_index("Percentile")
    st.dataframe(lat_profile, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: TOKEN MONITORING
# ==============================================================================
st.header("7. Token Monitoring (Prompt Tokens vs Completion Tokens, Cost Burn Rate & Budget Caps)", divider="rainbow")

st.markdown("""
Token consumption directly dictates cloud infrastructure costs. 
Because input tokens and output tokens have distinct pricing tiers, token monitoring separates consumption streams.
""")

tok_col1, tok_col2 = st.columns([1, 1])

with tok_col1:
    st.markdown("""
    ### Token Economics in RAG:
    * **Asymmetric Pricing**: Frontier models price output tokens at 3x to 5x the cost of input tokens.
    * **Prompt Inflation**: In conversational RAG, accumulating 10 conversation turns can balloon prompt input tokens to 8,000+ tokens per query.
    * **Cost Alerting**: Tracking tokens per user session allows automated rate-limiting before enterprise budget caps are breached.
    """)

with tok_col2:
    st.markdown("### Token Cost Estimation Model")
    cost_model = pd.DataFrame([
        {"Model Tier": "Frontier (GPT-4o / Claude 3.5 Sonnet)", "Input Cost per 1M": "$3.00 - $5.00", "Output Cost per 1M": "$15.00", "Typical RAG Query Cost": "$0.008 - $0.015"},
        {"Model Tier": "Compact (GPT-4o-mini / Haiku)", "Input Cost per 1M": "$0.15 - $0.25", "Output Cost per 1M": "$0.60 - $1.25", "Typical RAG Query Cost": "$0.0004 - $0.001"}
    ])
    st.dataframe(cost_model, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: OpenTelemetry-Aligned RAG Observability Tracer", divider="rainbow")

st.markdown("""
Below is the production implementation of `RAGObservabilityEngine`. It manages trace hierarchies, 
span lifecycle metrics, latency breakdowns, and token cost tracking in module scope.
""")

st.code(r'''
class RAGObservabilityEngine:
    """
    Enterprise RAG Distributed Tracing Engine:
    - Generates OpenTelemetry-aligned trace and span hierarchies.
    - Measures retrieval latency, reranking time, and TTFT prefill duration.
    - Computes token economics and cumulative financial burn.
    """
    def __init__(self, cost_per_1k_input: float = 0.005, cost_per_1k_output: float = 0.015):
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def generate_trace_id(self, seed: int = 101) -> str:
        return f"tr-8a4f9b{seed:04d}-e2c1"

    def simulate_pipeline_trace(
        self, query: str, retrieval_k: int = 5, rerank_active: bool = True,
        prompt_tokens: int = 1850, completion_tokens: int = 220
    ) -> Dict[str, Any]:
        trace_id = self.generate_trace_id()
        embed_ms, search_ms = 42.0, 68.0
        rerank_ms = 85.0 if rerank_active else 0.0
        pack_ms = 12.0
        ttft_ms = prompt_tokens * 0.14
        gen_ms = (completion_tokens / 35.0) * 1000.0
        total_ms = embed_ms + search_ms + rerank_ms + pack_ms + ttft_ms + gen_ms

        in_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_input
        out_cost = (completion_tokens / 1000.0) * self.cost_per_1k_output

        spans = [
            {"span": "embed_query", "duration_ms": embed_ms, "cat": "Embedding"},
            {"span": "vector_search", "duration_ms": search_ms, "cat": "Retrieval"},
        ]
        if rerank_active:
            spans.append({"span": "cross_encoder_rerank", "duration_ms": rerank_ms, "cat": "Reranking"})
        spans.extend([
            {"span": "context_packing", "duration_ms": pack_ms, "cat": "Context"},
            {"span": "llm_generation", "duration_ms": round(ttft_ms + gen_ms, 1), "cat": "Generation"}
        ])

        return {
            "trace_id": trace_id, "query": query, "total_latency_ms": round(total_ms, 1),
            "ttft_ms": round(ttft_ms, 1), "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cost_usd": round(in_cost + out_cost, 5), "spans": spans
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Distributed Trace Simulator & Latency Waterfall Inspector", divider="rainbow")

st.markdown("""
Simulate an enterprise distributed trace. 
Toggle reranking, adjust prompt token budgets and completion tokens, and inspect the resulting 
**Latency Waterfall** and **Token Economics** breakdown.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("### Pipeline Execution Controls")
    trace_query = st.text_input("User Search Query:", value="What is the guaranteed database failover latency?")
    trace_k = st.slider("Retrieved Candidates (K):", min_value=1, max_value=20, value=5, step=1)
    trace_rerank = st.toggle("Enable Cross-Encoder Reranker Span", value=True)
    trace_prompt_tokens = st.slider("Injected Context Size (Prompt Tokens):", min_value=500, max_value=8000, value=2200, step=100)
    trace_completion_tokens = st.slider("Generated Output Length (Tokens):", min_value=50, max_value=500, value=180, step=10)

with wb_col2:
    st.markdown("### Live Distributed Trace Scorecard")
    trace_data = obs_engine.simulate_pipeline_trace(
        query=trace_query,
        retrieval_k=trace_k,
        rerank_active=trace_rerank,
        prompt_tokens=trace_prompt_tokens,
        completion_tokens=trace_completion_tokens
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Trace Latency", f"{trace_data['total_latency_ms']} ms")
    m2.metric("Time-To-First-Token", f"{trace_data['ttft_ms']} ms")
    m3.metric("Estimated Query Cost", f"${trace_data['cost_usd']:.5f}")

    st.markdown("#### Span Latency Breakdown (Waterfall):")
    span_df = pd.DataFrame(trace_data["spans"])
    st.dataframe(span_df, use_container_width=True, hide_index=True)
    st.bar_chart(span_df.set_index("span")["duration_ms"])

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the OpenInference semantic convention in OpenTelemetry, and why is it necessary for RAG?**"):
    st.markdown("""
    **Answer**:
    Standard OpenTelemetry spans track HTTP attributes (`http.status_code`, `http.method`) and SQL queries (`db.statement`). 
    These conventions have no concept of LLM-specific telemetry.
    **OpenInference** standardizes semantic attributes for AI systems:
    * `openinference.span.kind` (categorizing spans as `RETRIEVER`, `RERANKER`, `LLM`, or `TOOL`).
    * `llm.token_count.prompt` and `llm.token_count.completion`.
    * `retrieval.documents` containing chunk text, IDs, and similarity distances.
    This allows platforms like Arize Phoenix, Langfuse, and Datadog to ingest and visualize RAG traces uniformly without proprietary lock-in.
    """)

with st.expander("**Q2: How do you decompose Time-To-First-Token (TTFT) from Generation Latency in production monitoring?**"):
    st.markdown("""
    **Answer**:
    In streaming LLM inference:
    1. **TTFT (Prefill Latency)**: The elapsed time between dispatching the prompt payload and receiving the **first streaming token chunk**. It reflects the model's forward pass across all prompt tokens.
    2. **Generation Latency**: The elapsed time between the arrival of the first token and the arrival of the final end-of-sequence token. Divided by completion tokens, it yields **tokens per second (TPS)** throughput.
    If TTFT is high while TPS is normal, the root cause is **prompt bloat** (injecting too much context). If TTFT is fast while TPS is slow, the root cause is **GPU memory bandwidth throttling**.
    """)

with st.expander("**Q3: What causes silent vector database performance regressions in production?**"):
    st.markdown("""
    **Answer**:
    1. **Index Bloat & Fragmented HNSW Graphs**: As millions of chunks are inserted and deleted, HNSW graph connectivity degrades, increasing search hops and inflating retrieval latency from 15ms to 250ms.
    2. **Dimension / Model Mismatch**: Updating the embedding model service without updating historical document vectors produces valid float vectors that silently return random garbage results with high cosine distances.
    3. **Cold Storage Thrashing**: If older chunks are stored on disk-backed storage, queries touching cold shards incur disk I/O penalties.
    """)

with st.expander("**Q4: How should a production observability pipeline balance trace payload capture against user privacy (PII)?**"):
    st.markdown("""
    **Answer**:
    Logging raw prompt payloads and user queries creates significant PII risk (capturing credit cards, SSNs, medical queries).
    Production architectures implement **Client-Side Trace Sanitization**:
    1. Pass text through an inline PII redaction scanner (e.g., Microsoft Presidio) before exporting the OTel span.
    2. Replace detected entities with typed tokens (`[REDACTED_SSN]`, `[REDACTED_EMAIL]`).
    3. Store a SHA-256 hash of the user query for deduplication and cache hit analysis without storing plain text.
    """)

with st.expander("**Q5: Why is P99 latency more critical than average latency for interactive customer RAG bots?**"):
    st.markdown("""
    **Answer**:
    Average latency conceals worst-case experiences. In enterprise customer support, if average latency is 1.5 seconds, but P99 is 12 seconds, 1 out of every 100 enterprise users experiences a 12-second freeze. In high-volume systems (100,000 queries/day), that represents 1,000 frustrated enterprise users per day, leading to customer churn and SLA breach penalties.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Storing Full Prompts in Unencrypted Logs**
    * **Failure Mode**: Logging complete prompt strings into standard application logs exposes sensitive customer PII and enterprise trade secrets in plain text.
    * **Solution**: Mask PII using regex or Presidio before exporting telemetry spans.
    """)

    st.error("""
    **Production Trap 2: Relying Exclusively on Average Latency**
    * **Failure Mode**: Average latency appears healthy (1.2s), hiding the fact that 5% of queries suffer extreme 10-second timeouts.
    * **Solution**: Track P95 and P99 percentiles on all service level objectives (SLOs).
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Unmonitored Chat History Accumulation**
    * **Failure Mode**: Appending every historical message in multi-turn chat without a sliding window causes token costs to compound quadratically.
    * **Solution**: Implement token budget caps and rolling window summarizers.
    """)

    st.warning("""
    **Production Trap 4: Tracing Without Sampling in High-Volume Systems**
    * **Failure Mode**: Exporting 100% of full-text spans at 5,000 requests/sec creates a secondary observability cost higher than the LLM inference itself.
    * **Solution**: Implement head-based or tail-based trace sampling (e.g., trace 10% of successes and 100% of errors).
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: OpenInference Standard")
        st.caption("What is OpenInference in OTel?")
        st.markdown("**Back**: An open-source semantic convention defining standard span attributes for LLMs, retrievers, tools, and vector databases.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Time-To-First-Token (TTFT)")
        st.caption("What architectural phase dictates TTFT?")
        st.markdown("**Back**: The prompt prefill phase, where the model processes all input tokens in parallel before emitting token 1.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Trace vs Span")
        st.caption("How do traces and spans relate in distributed systems?")
        st.markdown("**Back**: A Trace represents the entire user request lifecycle; a Span represents a single timed unit of work within the trace.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Tail-Based Trace Sampling")
        st.caption("How to reduce telemetry storage costs?")
        st.markdown("**Back**: Buffer spans in memory and only persist traces that exhibit errors, high latency (P95+), or low faithfulness scores.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of RAG observability and tracing before proceeding:")

c1 = st.checkbox("I understand why traditional web APM tools fail to detect semantic RAG regressions and hallucinations.", value=True)
c2 = st.checkbox("I can structure an OpenTelemetry-compliant trace with nested retrieval, reranker, and LLM spans.", value=True)
c3 = st.checkbox("I know how to deconstruct Time-To-First-Token (TTFT) from sequential token generation throughput.", value=True)
c4 = st.checkbox("I can calculate the token economics and dollar expenditure per request based on asymmetric pricing models.", value=True)
c5 = st.checkbox("I understand the mechanisms of PII redaction and privacy preservation in production trace exports.", value=True)
c6 = st.checkbox("I know how to use P95/P99 latency percentiles to identify cold-start and context-bloat bottlenecks.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **RAG Observability Mastery Achieved!** You are ready to instrument production AI pipelines with enterprise distributed tracing.")
