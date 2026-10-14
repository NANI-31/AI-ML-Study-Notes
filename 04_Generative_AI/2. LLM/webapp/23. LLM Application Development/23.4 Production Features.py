import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="23.4 Production Features - LLM Engineering",
    page_icon="🚀",
    layout="wide"
)

# Custom Theme CSS
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
    .badge-ok {
        background-color: #059669;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-fallback {
        background-color: #D97706;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-error {
        background-color: #DC2626;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE CLASSES FOR PRODUCTION GATEWAY & WORKBENCH
# ==============================================================================
class ProviderEndpoint:
    """Represents a candidate inference provider endpoint."""
    def __init__(self, name: str, cost_per_m_in: float, cost_per_m_out: float, base_latency_ms: int):
        self.name = name
        self.cost_per_m_in = cost_per_m_in
        self.cost_per_m_out = cost_per_m_out
        self.base_latency_ms = base_latency_ms

class ProductionTelemetryPipeline:
    """
    Enterprise production gateway pipeline coordinator.
    Handles exact-hash caching, jittered retries, fallback cascading, and cost attribution.
    """
    def __init__(self):
        self.cache_store: Dict[str, str] = {
            "what is our corporate refund policy?": "Our policy allows full refunds within 30 days of purchase upon receipt of product."
        }
        self.primary_provider = ProviderEndpoint("OpenAI (GPT-4o)", cost_per_m_in=2.50, cost_per_m_out=10.00, base_latency_ms=450)
        self.secondary_provider = ProviderEndpoint("Anthropic (Claude 3.5 Sonnet)", cost_per_m_in=3.00, cost_per_m_out=15.00, base_latency_ms=480)
        self.tertiary_provider = ProviderEndpoint("Self-Hosted vLLM (Llama 3.3 70B)", cost_per_m_in=0.60, cost_per_m_out=0.60, base_latency_ms=280)

    def compute_cost(self, provider: ProviderEndpoint, prompt_tokens: int, completion_tokens: int) -> float:
        in_cost = (prompt_tokens / 1_000_000.0) * provider.cost_per_m_in
        out_cost = (completion_tokens / 1_000_000.0) * provider.cost_per_m_out
        return in_cost + out_cost

    def execute_request(
        self,
        query: str,
        enable_cache: bool,
        primary_fails: bool,
        secondary_fails: bool,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Executes an inference request through the gateway lifecycle with fault simulation.
        """
        normalized_query = query.strip().lower()
        trace_id = f"trc-{np.random.randint(100000, 999999)}"
        log_steps: List[str] = []
        
        # 1. Exact Cache Check
        if enable_cache and normalized_query in self.cache_store:
            cached_text = self.cache_store[normalized_query]
            log_steps.append("Exact Cache Hit: Returned response from Redis in 2ms without calling LLM.")
            return {
                "trace_id": trace_id,
                "status": "CACHE_HIT",
                "provider": "Redis Exact Cache",
                "response": cached_text,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0000,
                "total_latency_ms": 2,
                "retries": 0,
                "log_steps": log_steps
            }

        log_steps.append("Cache Miss: Forwarding request to Primary Provider Gateway.")
        prompt_tokens = max(10, len(query) // 4 + 15)
        completion_tokens = 45

        # 2. Primary Provider Execution with Retries
        attempts = 0
        current_latency = 0
        primary_exhausted = False

        if primary_fails:
            while attempts < max_retries:
                attempts += 1
                backoff_ms = int(50 * (2 ** (attempts - 1)) + np.random.uniform(10, 40))
                current_latency += backoff_ms
                log_steps.append(f"Primary ({self.primary_provider.name}) failed (HTTP 503). Attempt {attempts} backoff wait: {backoff_ms}ms.")
            primary_exhausted = True
            log_steps.append(f"Primary ({self.primary_provider.name}) retry budget exhausted. Triggering Secondary Fallback.")
        else:
            current_latency += self.primary_provider.base_latency_ms + np.random.randint(20, 60)
            cost = self.compute_cost(self.primary_provider, prompt_tokens, completion_tokens)
            log_steps.append(f"Primary ({self.primary_provider.name}) returned HTTP 200 OK successfully.")
            return {
                "trace_id": trace_id,
                "status": "PRIMARY_SUCCESS",
                "provider": self.primary_provider.name,
                "response": f"Generated answer for '{query}' via primary model pipeline.",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_cost_usd": cost,
                "total_latency_ms": current_latency,
                "retries": 0,
                "log_steps": log_steps
            }

        # 3. Secondary Provider Fallback
        if primary_exhausted:
            if secondary_fails:
                log_steps.append(f"Secondary ({self.secondary_provider.name}) returned HTTP 429 Rate Limit. Triggering Tertiary vLLM fallback.")
                current_latency += 150
                # Tertiary Fallback
                current_latency += self.tertiary_provider.base_latency_ms + np.random.randint(10, 30)
                cost = self.compute_cost(self.tertiary_provider, prompt_tokens, completion_tokens)
                log_steps.append(f"Tertiary ({self.tertiary_provider.name}) returned HTTP 200 OK. Graceful degradation successful.")
                return {
                    "trace_id": trace_id,
                    "status": "TERTIARY_FALLBACK",
                    "provider": self.tertiary_provider.name,
                    "response": f"Generated answer for '{query}' via self-hosted tertiary failover cluster.",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_cost_usd": cost,
                    "total_latency_ms": current_latency,
                    "retries": attempts,
                    "log_steps": log_steps
                }
            else:
                current_latency += self.secondary_provider.base_latency_ms + np.random.randint(20, 50)
                cost = self.compute_cost(self.secondary_provider, prompt_tokens, completion_tokens)
                log_steps.append(f"Secondary ({self.secondary_provider.name}) returned HTTP 200 OK. Fallback successful.")
                return {
                    "trace_id": trace_id,
                    "status": "SECONDARY_FALLBACK",
                    "provider": self.secondary_provider.name,
                    "response": f"Generated answer for '{query}' via secondary commercial provider.",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_cost_usd": cost,
                    "total_latency_ms": current_latency,
                    "retries": attempts,
                    "log_steps": log_steps
                }

        return {"status": "UNKNOWN"}


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("23.4 Production Features: Streaming, Retries, Fallbacks, Caching, & Telemetry")
st.caption("Chapter 23: LLM Application Development | Domain: 2. Large Language Models")

st.markdown("""
Building a proof-of-concept with Large Language Models is deceptively simple: a developer writes ten lines of Python 
and calls an API endpoint. However, deploying an LLM application to **millions of production enterprise users** reveals 
a harsh operational reality: upstream cloud APIs frequently experience rate limits (HTTP 429), regional outages (HTTP 503), 
and stochastic latency spikes. Furthermore, unmonitored token usage can bankrupt unit economics overnight.

Production-grade LLM engineering bridges this gap with seven foundational production pillars:
**Server-Sent Events (SSE) Streaming**, **Decorrelated Jittered Retries**, **Multi-Provider Fallback Routing**, 
**Multi-Tier Semantic & KV Caching**, **Privacy-Masked Structured JSON Logging**, **OpenTelemetry Distributed Tracing**, 
and **Granular Cost Attribution**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise High-Reliability LLM Production Telemetry & Gateway Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates the path of a production request through an enterprise AI Gateway. 
Every request passes through exact and semantic caches, applies rate-limiting buckets, evaluates model cascades, 
protects availability with circuit breakers and fallback providers, streams SSE tokens back to the user, and records 
structured OpenTelemetry spans.
""")

blueprint_dot = """
digraph ProductionGateway {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_ingress {
        label = "Tier 1: Request Ingress & Auth Guard";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        ClientApp [label="Client Application\\n(Web UI / Mobile App / API Client)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Gateway [label="Enterprise LLM Gateway Proxy\\n(Token Bucket Rate Limiter + Tenant Quotas)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_caching {
        label = "Tier 2: Dual Caching Hierarchy";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        ExactCache [label="Exact Hash Cache (Redis)\\n(SHA-256 Prompt Hash | 2ms Latency | $0 Cost)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        SemanticCache [label="Semantic Vector Cache (Qdrant)\\n(Embedding Distance < 0.08 | 15ms Latency)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_resilience {
        label = "Tier 3: Resilience & Model Routing";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        ModelRouter [label="Intelligent Complexity Router\\n(Classifies Query -> Cheap vs Frontier Tier)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        CircuitBreaker [label="Circuit Breaker & Retry Engine\\n(Full Jitter Backoff + Idempotency Keys)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        PrimaryProvider [label="Primary Provider: OpenAI (GPT-4o)\\n(Active Endpoint)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        SecondaryFallback [label="Secondary Fallback: Anthropic (Claude 3.5)\\n(Automatic Failover on HTTP 429/503)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        TertiaryFallback [label="Tertiary Fallback: Self-Hosted vLLM\\n(On-Prem Llama 3.3 70B Cluster)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_observability {
        label = "Tier 4: Streaming & Telemetry Pipeline";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        SSEStreamer [label="SSE Streaming Gateway\\n(Flushes Delimited Chunks | TTFT < 350ms)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        TelemetryCollector [label="Telemetry & Cost Collector\\n(OpenTelemetry Traces + Token Attribution)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        DataDogLog [label="Observability APM (Langfuse / Datadog)\\n(PII-Scrubbed Structured JSON Spans)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    ClientApp -> Gateway [label="1. Ingress Request"];
    Gateway -> ExactCache [label="2. Check Exact Hash"];
    ExactCache -> ClientApp [label="Hit: Return Cached (2ms)", style="dashed", color="#10B981"];
    ExactCache -> SemanticCache [label="Miss"];
    SemanticCache -> ClientApp [label="Hit: Return Cached (15ms)", style="dashed", color="#10B981"];
    SemanticCache -> ModelRouter [label="Miss: Invoke Model"];

    ModelRouter -> CircuitBreaker;
    CircuitBreaker -> PrimaryProvider [label="Attempt 1"];
    PrimaryProvider -> SecondaryFallback [label="On 503/429 Failover", style="dashed", color="#F59E0B"];
    SecondaryFallback -> TertiaryFallback [label="On Secondary Outage", style="dashed", color="#F59E0B"];

    PrimaryProvider -> SSEStreamer [label="HTTP 200 Stream"];
    SecondaryFallback -> SSEStreamer [label="HTTP 200 Stream"];
    TertiaryFallback -> SSEStreamer [label="HTTP 200 Stream"];

    SSEStreamer -> ClientApp [label="3. Stream Tokens"];
    SSEStreamer -> TelemetryCollector [label="4. Token Telemetry"];
    TelemetryCollector -> DataDogLog [label="5. Export Spans & Costs"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - STREAMING
# ==============================================================================
st.header("2. Streaming: Server-Sent Events (SSE), Chunk Parsers, and Time-to-First-Token (TTFT)", divider="rainbow")

st.markdown("""
When generating a long 800-token response, an autoregressive language model takes 5 to 15 seconds to finish complete execution. 
If an application waits for the complete response before returning data (blocking request-response), the user stares at a frozen 
loading spinner, creating a perception of extreme sluggishness. 

**Streaming** over HTTP eliminates this bottleneck by progressively delivering tokens the microsecond they are sampled 
from the GPU matrix multiplications.
""")

col_st1, col_st2 = st.columns([1, 1])

with col_st1:
    st.markdown("""
    ### Server-Sent Events (SSE) Protocol
    * Streaming uses standard HTTP/1.1 or HTTP/2 persistent connections with response header:
      `Content-Type: text/event-stream`.
    * The server sends newline-delimited chunks:
    """)
    st.code("""
data: {"choices": [{"delta": {"content": "Hello"}}]}

data: {"choices": [{"delta": {"content": " world"}}]}

data: {"choices": [{"delta": {"content": "!"}}]}

data: [DONE]
    """, language="http")

with col_st2:
    st.markdown("""
    ### Key Production Latency Metrics
    * **Time-to-First-Token (TTFT)**: The wall-clock elapsed time between the client sending the request and the arrival of token #1. Reflects server queue time, prompt token prefill computation, and network latency. Production target: **< 400ms**.
    * **Inter-Token Latency (ITL)**: The duration between consecutive tokens during generation (e.g., 25ms per token = 40 tokens/sec). Dictated by memory bandwidth and model parameter size.
    * **Total Generation Time (TGT)**: Total wall-clock execution time (`TTFT + (Tokens * ITL)`).
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - RETRIES
# ==============================================================================
st.header("3. Retries: Exponential Backoff, Jitter Algorithms, and Idempotency Keys", divider="rainbow")

st.markdown("""
Commercial LLM APIs fail frequently due to transient server overload, rate limits (HTTP 429), gateway timeouts (HTTP 504), 
and GPU node reboots (HTTP 503). Blindly retrying requests instantly or at fixed intervals triggers a **Thundering Herd** 
disaster that crashes upstream systems.
""")

col_rt1, col_rt2 = st.columns([1, 1])

with col_rt1:
    st.markdown("""
    ### Decorrelated Full Jitter Backoff
    Production systems implement exponential backoff with full randomized jitter to scatter retry spikes evenly:
    * Standard Backoff: `t = min(t_max, t_base * 2^attempt)`
    * **Full Jitter Formula**:
      `Sleep_Time = Uniform(0, min(t_max, t_base * 2^attempt))`
    * Spreading retry attempts randomly across the time domain reduces upstream queue contention by over 80%!
    """)

with col_rt2:
    st.markdown("""
    ### Idempotency Keys
    If a client request drops connection midway through an inference call, how does the client retry without causing double-billing 
    or executing side-effecting agent tool calls twice (such as charging a credit card)?
    * **Solution**: Clients attach a unique `Idempotency-Key: UUIDv4` HTTP header.
    * The gateway records the idempotency key in Redis. If a duplicate key arrives within 24 hours, the gateway returns the existing result instead of re-executing.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - FALLBACK MODELS
# ==============================================================================
st.header("4. Fallback Models: Cascade Tiering, Graceful Degradation, and Model Routing", divider="rainbow")

st.markdown("""
Relying on a single AI provider is a single-point-of-failure (SPOF) that violates enterprise availability SLAs. 
Furthermore, sending 100% of user queries to expensive frontier models is financially irresponsible.
""")

col_fb1, col_fb2 = st.columns([1, 1])

with col_fb1:
    st.markdown("""
    ### Multi-Provider Fallback Topology
    When primary commercial endpoints return HTTP 500, 503, or persistent 429, the gateway automatically switches to hot standby endpoints:
    1. **Primary**: OpenAI `gpt-4o` (Default Frontier).
    2. **Secondary Fallback**: Anthropic `claude-3-5-sonnet` (Cross-Cloud Redundancy).
    3. **Tertiary Fallback**: Self-hosted vLLM `llama-3.3-70b` (On-Premises High-SLA Safety Net).
    * Switches occur completely transparently to the end-user in under 200 milliseconds.
    """)

with col_fb2:
    st.markdown("""
    ### Model Cascading (Complexity Routing)
    * In enterprise production, **80%+ of incoming queries are simple tasks** (classification, simple extraction, polite greetings) that do not require frontier reasoning.
    * A lightweight rule or small model (e.g. `gpt-4o-mini` at $0.15/1M tokens) handles easy queries.
    * Only queries tagged with complex reasoning or coding are routed to frontier models ($5.00/1M tokens).
    * Blended monthly infrastructure savings: **70% to 85%**.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - CACHING
# ==============================================================================
st.header("5. Caching: Exact Hash Hits, Semantic Vector Caching, and Prefix KV Reuse", divider="rainbow")

st.markdown("""
The cheapest and fastest LLM query is the one that is never executed on a GPU. Multi-tier caching intercepts repetitive 
queries at the gateway before they consume API budget.
""")

col_cc1, col_cc2 = st.columns([1, 1])

with col_cc1:
    st.markdown("""
    ### Multi-Tier Caching Hierarchy
    1. **Exact Hash Cache (Redis / Memcached)**:
       * Computes SHA-256 of `(model_name + system_prompt + user_prompt)`.
       * Latency: **sub-2ms**. Cost: **$0.00**.
       * Perfect for identical FAQ queries and automated unit test runs.
    2. **Semantic Vector Cache (Qdrant / Redis Vector)**:
       * Encodes user query into a dense embedding vector.
       * Searches cached queries using cosine distance. If `cosine_distance < 0.08`, returns the cached answer.
       * Bridges natural language variations (*'How to reset password?'* vs *'Steps for password reset'*).
    """)

with col_cc2:
    st.markdown("""
    ### Foundation Model Prefix KV-Caching
    * Cloud providers (Anthropic Prompt Caching, OpenAI Prompt Caching) and local engines (vLLM RadixAttention) cache the computed Key-Value activations of identical prompt prefixes in GPU VRAM.
    * By placing static instructions, schemas, and large RAG context blocks at the *exact beginning* of the prompt, subsequent calls reuse precomputed tensors:
      * **Prefill Latency**: Reduced by up to **80%**.
      * **Cloud Provider Token Cost**: Discounted by **50% to 90%**!
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 6 - LOGGING
# ==============================================================================
st.header("6. Logging: Structured JSON Telemetry, Privacy Masking, and Prompt Scrubbing", divider="rainbow")

st.markdown("""
Standard application logging (e.g., `print("User asked: " + query)`) fails disastrously in LLM production. 
Unstructured text is unsearchable at scale and risks committing severe regulatory violations (GDPR, HIPAA) 
if Personally Identifiable Information (PII) is written to centralized log collectors.
""")

col_lg1, col_lg2 = st.columns([1, 1])

with col_lg1:
    st.markdown("""
    ### Canonical Structured Telemetry Schema
    Production systems format every event as structured JSON:
    """)
    st.code("""
{
  "timestamp": "2026-10-13T16:40:00Z",
  "trace_id": "trc-894102-a4",
  "tenant_id": "tenant-acme-corp",
  "user_id": "usr-8812",
  "model_requested": "gpt-4o",
  "model_routed": "claude-3-5-sonnet",
  "fallback_triggered": true,
  "cache_hit": false,
  "tokens": {
    "prompt": 450,
    "completion": 120,
    "total": 570
  },
  "cost_usd": 0.00315,
  "latency": {
    "ttft_ms": 320,
    "total_ms": 840
  },
  "status_code": 200
}
    """, language="json")

with col_lg2:
    st.markdown("""
    ### Privacy Scrubbing & Sanitization
    * **Presidio / Regex Scrubbers**: Before persisting prompt payloads to disk, all Social Security Numbers, credit card numbers, phone numbers, and email addresses are replaced with anonymized tokens (e.g., `<EMAIL_REDACTED>`).
    * **Zero-Retention Tiers**: For sensitive financial or healthcare enterprise clients, configure loggers with `log_payloads = False`, preserving only numerical telemetry (tokens, latencies, error codes).
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 7 - MONITORING
# ==============================================================================
st.header("7. Monitoring: Distributed Tracing, Drift Detection, and APM Metrics", divider="rainbow")

st.markdown("""
Observing complex LLM pipelines requires Application Performance Monitoring (APM) adapted specifically to 
non-deterministic AI workflows.
""")

col_mn1, col_mn2 = st.columns([1, 1])

with col_mn1:
    st.markdown("""
    ### The Four Golden Signals of LLM APM
    1. **Time-to-First-Token (TTFT) P95 / P99**: Detects upstream cloud throttling and GPU prefill queue saturation.
    2. **Tokens Per Second (TPS) Throughput**: Monitors generation velocity across active streaming connections.
    3. **Error & Fallback Rate**: Spikes in HTTP 429/503 indicate provider outages, triggering automated alerts.
    4. **Safety & Refusal Rate**: Tracks the percentage of prompts rejected by guardrails or model safety filters.
    """)

with col_mn2:
    st.markdown("""
    ### Distributed Tracing (OpenTelemetry)
    In a RAG or multi-agent application, a single user click triggers multiple distributed sub-operations:
    * `Span 1`: Ingress Auth Check (5ms)
    * `Span 2`: Redis Semantic Cache Lookup (12ms)
    * `Span 3`: Dense Vector Retrieval in Qdrant (45ms)
    * `Span 4`: Cross-Encoder Reranking (60ms)
    * `Span 5`: LLM Generation via Anthropic (650ms)
    OpenTelemetry links these spans under a unified `trace_id`, pinpointing the exact microsecond bottleneck.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 8 - COST TRACKING
# ==============================================================================
st.header("8. Cost Tracking: Token Unit Economics, Budget Caps, and Attribution Quotas", divider="rainbow")

st.markdown("""
Without disciplined cost engineering, Generative AI applications can rapidly burn six-figure monthly cloud budgets. 
Enterprise platforms enforce real-time cost accounting at the token level.
""")

col_ct1, col_ct2 = st.columns([1, 1])

with col_ct1:
    st.markdown("""
    ### Unit Economics Formula
    * Commercial APIs charge separate rates for input (prompt) and output (completion) tokens:
      `Cost = (Tokens_Prompt / 10^6) * Rate_Prompt + (Tokens_Completion / 10^6) * Rate_Completion`
    * Output tokens are typically **3x to 5x more expensive** than input tokens due to autoregressive decoding constraints.
    * **Granular Attribution**: Every token spent is tagged with `tenant_id`, `department`, and `feature_flag` to calculate exact gross margins per customer.
    """)

with col_ct2:
    st.markdown("""
    ### Hard Budget Caps & Circuit Breakers
    * **Soft Warning**: At 80% of monthly department quota, dispatch webhook alert to Slack / PagerDuty.
    * **Hard Stop**: At 100% quota, gateway returns HTTP 429 with custom body: *"Monthly Generative AI budget cap reached for this organization."*
    * Prevents runaway infinite loops in autonomous agent architectures from racking up immense cloud bills.
    """)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Resilient Enterprise LLM Gateway & Telemetry Pipeline", divider="rainbow")

st.markdown("""
Below is the complete, self-contained Python coordinator implementing the core production capabilities: 
cache inspection, exponential backoff retries, multi-provider fallback cascading, and token telemetry tracking.
""")

prod_code = """
import numpy as np
from typing import Dict, List, Any, Optional

class EnterpriseGatewayPipeline:
    \"\"\"
    Production-grade LLM Gateway Coordinator.
    Handles exact-hash caching, jittered retries, and cross-provider failover.
    \"\"\"
    def __init__(self):
        self.exact_cache: Dict[str, str] = {}
        self.provider_rates = {
            "OpenAI (GPT-4o)": {"in": 2.50, "out": 10.00, "base_latency": 420},
            "Anthropic (Claude 3.5 Sonnet)": {"in": 3.00, "out": 15.00, "base_latency": 450},
            "Self-Hosted (Llama 3.3 70B)": {"in": 0.60, "out": 0.60, "base_latency": 260}
        }

    def _full_jitter_backoff(self, attempt: int, base_ms: int = 50, max_ms: int = 1000) -> int:
        ceiling = min(max_ms, base_ms * (2 ** attempt))
        return int(np.random.uniform(0, ceiling))

    def dispatch(
        self,
        query: str,
        allow_cache: bool = True,
        simulate_primary_down: bool = False
    ) -> Dict[str, Any]:
        trace_id = f"trace-{np.random.randint(10000, 99999)}"
        key = query.strip().lower()

        # 1. Cache Check
        if allow_cache and key in self.exact_cache:
            return {
                "trace_id": trace_id,
                "status": "CACHE_HIT",
                "provider": "Redis Exact Cache",
                "response": self.exact_cache[key],
                "cost_usd": 0.0,
                "latency_ms": 2
            }

        # 2. Primary Provider Attempt
        if not simulate_primary_down:
            rates = self.provider_rates["OpenAI (GPT-4o)"]
            cost = (20 / 1e6) * rates["in"] + (50 / 1e6) * rates["out"]
            return {
                "trace_id": trace_id,
                "status": "PRIMARY_SUCCESS",
                "provider": "OpenAI (GPT-4o)",
                "response": f"Response generated by Primary for: {query}",
                "cost_usd": cost,
                "latency_ms": rates["base_latency"] + 30
            }

        # 3. Fallback to Secondary Provider
        rates = self.provider_rates["Anthropic (Claude 3.5 Sonnet)"]
        backoff_delay = self._full_jitter_backoff(attempt=1)
        cost = (20 / 1e6) * rates["in"] + (50 / 1e6) * rates["out"]
        return {
            "trace_id": trace_id,
            "status": "FALLBACK_SUCCESS",
            "provider": "Anthropic (Claude 3.5 Sonnet)",
            "response": f"Response generated by Fallback for: {query}",
            "cost_usd": cost,
            "latency_ms": rates["base_latency"] + backoff_delay
        }
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Enterprise Production Gateway & Fault Injection Simulator", divider="rainbow")

st.markdown("""
Test resilience engineering in practice. Inject artificial API outages, rate limits, and network latency into 
the production gateway and observe automated cache hits, jittered retry backoffs, and fallback cascades.
""")

pipeline = ProductionTelemetryPipeline()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Gateway Configuration & Fault Injection")
    
    user_query = st.text_input(
        "User Request",
        value="What is our corporate refund policy?",
        help="Try typing 'What is our corporate refund policy?' to trigger an instant exact cache hit!"
    )
    
    st.markdown("#### Resilience Toggles")
    enable_cache = st.checkbox("Enable Multi-Tier Caching (Redis Exact Hit)", value=True)
    
    st.markdown("#### Fault Injection Controls")
    sim_primary_down = st.checkbox("Inject Primary Provider Outage (HTTP 503 Service Unavailable)", value=True)
    sim_secondary_down = st.checkbox("Inject Secondary Fallback Outage (HTTP 429 Rate Limit)", value=False)
    
    max_retries = st.slider("Max Jittered Retry Attempts for Primary", min_value=1, max_value=4, value=2)

    exec_button = st.button("Dispatch Request Through Gateway 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Gateway Telemetry & Routing Decision")
    
    if exec_button and user_query.strip():
        result = pipeline.execute_request(
            query=user_query,
            enable_cache=enable_cache,
            primary_fails=sim_primary_down,
            secondary_fails=sim_secondary_down,
            max_retries=max_retries
        )

        status = result["status"]
        if status == "CACHE_HIT":
            badge_html = "<span class='badge-ok'>CACHE HIT (2ms)</span>"
        elif status == "PRIMARY_SUCCESS":
            badge_html = "<span class='badge-ok'>PRIMARY SUCCESS (HTTP 200)</span>"
        elif status in ["SECONDARY_FALLBACK", "TERTIARY_FALLBACK"]:
            badge_html = f"<span class='badge-fallback'>{status} (RECOVERED)</span>"
        else:
            badge_html = "<span class='badge-error'>FAILED</span>"

        st.markdown(f"**Execution Status**: {badge_html}", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Provider", result["provider"])
        m2.metric("Latency", f"{result['total_latency_ms']} ms")
        m3.metric("Token Cost", f"${result['total_cost_usd']:.6f}")
        m4.metric("Retry Count", result["retries"])

        st.markdown("#### Response Content")
        st.info(result["response"])

        st.markdown("#### Gateway Execution Trace Logs")
        with st.container(height=200):
            for step in result["log_steps"]:
                st.markdown(f"`{result['trace_id']}` — {step}")
    else:
        st.info("Click 'Dispatch Request Through Gateway' above to simulate request routing and failover.")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering production LLM features requires answering tough operational questions and understanding critical 
architectural antipatterns.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: How do you design an enterprise LLM gateway to handle a total OpenAI outage with zero downtime?"):
    st.markdown("""
    **Answer**:
    1. **Unified Schema Layer**: Standardize all internal requests to an OpenAI-compatible JSON format (using LiteLLM, Portkey, or custom middleware).
    2. **Health Checking & Circuit Breakers**: The gateway monitors error rates on the primary endpoint. If 5 consecutive requests return HTTP 500/503 or timeout > 5 seconds, the circuit breaker opens for 30 seconds.
    3. **Automated Cross-Provider Failover**: The router immediately shifts traffic to Anthropic Claude or AWS Bedrock endpoints with translated prompt payloads.
    4. **Client Invariance**: The client application continues to receive standard Server-Sent Events without noticing provider downtime.
    """)

with st.expander("Q2: What is the mathematical advantage of Decorrelated Full Jitter over standard Exponential Backoff?"):
    st.markdown("""
    **Answer**: In standard exponential backoff without jitter, every client experiencing a transient rate limit at time $t=0$ retries at exactly $t=1$, $t=2$, $t=4$, and $t=8$. 
    This creates synchronized periodic spikes that repeatedly overwhelm the upstream provider (the **Thundering Herd** problem).
    
    **Full Jitter** selects a random uniform sleep time between $0$ and the exponential ceiling:
    `Sleep = Uniform(0, min(Max_Wait, Base * 2^attempt))`
    
    This continuously smears client retries across the time domain, maximizing throughput and allowing the upstream service to recover.
    """)

with st.expander("Q3: How does prompt KV-cache reuse fundamentally alter LLM application cost structures and architecture?"):
    st.markdown("""
    **Answer**:
    1. **Structural Cost Inversion**: Historically, developers aggressively minimized prompt length to save costs. With KV prompt caching, long static system prompts (20,000 tokens of reference documentation) cost up to 90% less on subsequent requests.
    2. **Architectural Layout Requirement**: Systems must structure prompts strictly with static invariant content at the *exact beginning* of the prompt, and dynamic variable user inputs placed exclusively at the *very end*. Inserting a timestamp at the top of a prompt destroys the cache prefix on every single turn!
    """)

with st.expander("Q4: What metrics constitute the 'Golden Signals' of LLM Application Performance Monitoring (APM)?"):
    st.markdown("""
    **Answer**:
    1. **Time-to-First-Token (TTFT)**: Measures prompt prefill speed and network latency.
    2. **Tokens Per Second (TPS)**: Measures streaming generation speed.
    3. **Inter-Token Latency (ITL)**: Detects GPU memory thrashing or streaming jitter.
    4. **Provider Error & Failover Rates**: Tracks HTTP 429 rate limits and 5xx server faults.
    5. **Cost Attribution per Tenant**: Tracks real-time dollar consumption against organizational budgets.
    """)

with st.expander("Q5: How do you implement PII scrubbing in LLM production logging without losing debugging utility?"):
    st.markdown("""
    **Answer**:
    * Implement an asynchronous redaction pipeline (such as Microsoft Presidio) that detects entities (EMAIL, PHONE, CREDIT_CARD, IP_ADDRESS) and replaces them with cryptographic salted hashes:
      `john.doe@example.com -> <EMAIL_HASH:9f8a2b>`
    * This prevents raw PII from reaching log lakes (Datadog, Elastic) while still allowing engineers to correlate identical users across error traces.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Retrying Non-Idempotent Tool Calls Causing Duplicate Mutations (CRITICAL)**
    * *Error*: Retrying an agent request that timed out after executing a database modification or financial charge tool.
    * *Fix*: Require idempotent transaction IDs for all external stateful tools, and store execution statuses in a distributed lock manager.
    """)

    st.error("""
    **Trap 2: Unbounded Concurrency Overwhelming Upstream Rate Limits (CRITICAL)**
    * *Error*: Spawning 100 parallel asynchronous `asyncio.gather()` tasks directly against commercial cloud APIs.
    * *Fix*: Enforce token-bucket client rate limiters (e.g., Leaky Bucket in Redis) to throttle requests before they trigger upstream HTTP 429 blocks.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Overly Permissive Semantic Cache Similarity Returning Incorrect Answers (WARNING)**
    * *Warning*: Setting cosine similarity threshold too loosely (e.g. distance < 0.20) causes the cache to return answers for subtly different queries (*'How do I cancel my order?'* vs *'How do I return my order?'*).
    * *Fix*: Tune semantic cache thresholds conservatively (distance < 0.05 - 0.08) and validate with golden evaluation test sets.
    """)

    st.warning("""
    **Trap 4: Blindly Logging Unsanitized User Prompts Violating Compliance (WARNING)**
    * *Warning*: Storing full prompt text in general APM logs allows sensitive user data or API keys entered in chat to leak into developer logs.
    * *Fix*: Enforce zero-retention logging rules or mandatory automated PII masking in gateway middleware.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: TTFT vs ITL</strong><br><br>
        <em>Front:</em> What is the difference between TTFT and ITL?<br><br>
        <em>Back:</em> TTFT is time until the 1st token arrives (prefill); ITL is latency between subsequent tokens (decoding).
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Full Jitter Backoff</strong><br><br>
        <em>Front:</em> Why is Full Jitter preferred over pure Exponential Backoff?<br><br>
        <em>Back:</em> Full Jitter spreads retry attempts uniformly across time, preventing synchronized thundering herd spikes.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Circuit Breaker</strong><br><br>
        <em>Front:</em> What does an Open Circuit Breaker do?<br><br>
        <em>Back:</em> Immediately fails or redirects traffic to a fallback model without wasting time attempting the dead provider.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Prefix KV-Cache</strong><br><br>
        <em>Front:</em> What is the cardinal rule of prompt prefix caching?<br><br>
        <em>Back:</em> Put static invariant context at the beginning; place dynamic user text strictly at the very end.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical mastery of production LLM features and infrastructure:")

chk1 = st.checkbox("I understand Server-Sent Events (SSE) streaming and the difference between TTFT and Inter-Token Latency.", value=True)
chk2 = st.checkbox("I can implement Decorrelated Full Jitter backoff algorithms to prevent upstream thundering herd problems.", value=True)
chk3 = st.checkbox("I know how to design multi-provider fallback routing (OpenAI -> Anthropic -> vLLM) for 99.95% availability SLAs.", value=True)
chk4 = st.checkbox("I understand how multi-tier caching (Exact Hash, Semantic Vector, and Prefix KV reuse) slashes cloud inference bills.", value=True)
chk5 = st.checkbox("I can architect structured OpenTelemetry distributed tracing and PII-redacted logging pipelines.", value=True)
chk6 = st.checkbox("I know how to implement granular cost attribution and hard circuit breaker budget caps for enterprise tenants.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Production LLM Application Features! You have completed Chapter 23.")
