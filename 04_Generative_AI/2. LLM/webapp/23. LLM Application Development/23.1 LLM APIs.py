import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="23.1 LLM APIs - Generative AI Curriculum",
    page_icon="🔌",
    layout="wide"
)

# Title & Overview Block
st.title("23.1 LLM APIs: Fundamentals, Authentication, Chat Completion, Streaming, & Error Handling")
st.caption("Chapter 23: LLM Application Development | Domain: 2. Large Language Models")

st.markdown("""
Integrating Large Language Models into enterprise software architectures requires mastering commercial and open-source **LLM APIs**. 
Modern inference providers (OpenAI, Anthropic, Google Vertex AI, AWS Bedrock, and self-hosted vLLM/Ollama engines) expose standardized 
HTTP interfaces governing model interaction.

Unlike traditional CRUD REST endpoints that return small, synchronous database payloads in tens of milliseconds, 
LLM APIs involve distinct architectural constraints: long-running autoregressive token generation, streaming transport over 
**Server-Sent Events (SSE)**, dual-dimension **Token-per-Minute (TPM) and Request-per-Minute (RPM)** quotas, asymmetric pricing structures, 
and frequent HTTP 429 rate-limit exceptions requiring exponential jittered retries and circuit breaker patterns.
""")

st.markdown("""
In this lesson, you will master:
1. The structural anatomy of modern LLM API specifications (OpenAI chat completion schema vs Anthropic Messages API).
2. Enterprise authentication protocols: Bearer tokens, AWS SigV4, vault key injection, and IAM service credentials.
3. Chat Completion payload mechanics: role hierarchies (system, user, assistant, tool) and deterministic seed control.
4. Streaming protocols: Server-Sent Events (SSE), event chunks (`data: {...}`), and Time-to-First-Token (TTFT) latency optimization.
5. Production telemetry: token accounting, leaky bucket rate-limiting algorithms, and resilient exponential backoff strategies.
""")

st.divider()

# ==============================================================================
# SECTION 1: BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Enterprise LLM Client-Server Ingestion & Streaming Architecture", divider="rainbow")

st.markdown("""
An enterprise LLM client microservice manages connection pooling, rate limit throttling, token budgeting, and SSE stream parsing:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    api_dot = """
    digraph LLMAPIArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_client {
            label="Tier 1: Enterprise Client Application & Request Gate";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            AppClient [label="Frontend Client / User Query\\n'Summarize quarterly sales'", fillcolor="#1E293B", color="#38BDF8"];
            TokenBudgeter [label="Context Window & Token Budgeter\\ntiktoken calculation: Prompt Tokens <= 8,192", fillcolor="#334155", color="#64748B"];
            RateLimiter [label="Client-Side Rate Limiter\\nToken Bucket Algorithm (TPM / RPM Guard)", fillcolor="#0F766E", color="#14B8A6"];
            VaultAuth [label="Credential Vault Injector\\nInjects ephemeral Bearer key from Secrets Manager", fillcolor="#4338CA", color="#818CF8"];

            AppClient -> TokenBudgeter -> RateLimiter -> VaultAuth;
        }

        subgraph cluster_gateway {
            label="Tier 2: Network Transport & Retry Circuit Breaker";
            style="rounded";
            bgcolor="#151E32";
            color="#8B5CF6";
            fontcolor="#8B5CF6";
            fontsize=12;

            HTTPConn [label="HTTP/2 Connection Pool\\nKeep-Alive sockets, TCP TLS 1.3 handshake", fillcolor="#6D28D9", color="#A78BFA"];
            CircuitBreaker [label="Circuit Breaker & Retry Handler\\nExponential backoff with full jitter on HTTP 429/503", fillcolor="#0369A1", color="#38BDF8"];

            VaultAuth -> HTTPConn -> CircuitBreaker;
        }

        subgraph cluster_provider {
            label="Tier 3: Cloud Inference Provider & SSE Streaming Engine";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            ModelCluster [label="LLM Provider Cluster (vLLM / Bedrock / OpenAI)\\nContinuous Batching & PagedAttention Engine", fillcolor="#1E293B", color="#38BDF8", shape=cylinder];
            SSEStreamer [label="Server-Sent Events (SSE) Egress Stream\\nHTTP chunked transfer: 'data: {delta: token}'", fillcolor="#047857", color="#10B981"];
            ClientParser [label="Streaming Chunk Aggregator\\nDecodes tokens in real-time -> User UI stream", fillcolor="#065F46", color="#34D399"];

            CircuitBreaker -> ModelCluster [label="POST /v1/chat/completions", color="#10B981"];
            ModelCluster -> SSEStreamer -> ClientParser;
        }

        ClientParser -> AppClient [label="Incremental Rendered Tokens", style="dashed", color="#38BDF8"];
    }
    """
    st.graphviz_chart(api_dot, use_container_width=True)

st.markdown("""
### The Protocol Dimension Matrix: REST vs Streaming SSE vs WebSockets
The table below contrasts the transport protocols utilized across foundation model interfaces:
""")

protocol_table_df = pd.DataFrame({
    "Protocol Pattern": ["Standard Blocking REST (HTTP POST)", "Server-Sent Events (SSE Streaming)", "Bi-directional WebSockets", "gRPC (HTTP/2 Binary Protocol)"],
    "Underlying Transport": ["Standard TCP / HTTP/1.1 or HTTP/2", "HTTP/2 text/event-stream chunks", "Persistent full-duplex TCP socket", "HTTP/2 Protocol Buffers (protobuf)"],
    "Time-to-First-Token (TTFT)": ["Poor (User waits for 100% completion generation, 5s - 30s)", "Optimal (First token displayed in 200ms - 800ms)", "Optimal (Sub-200ms for continuous conversation)", "Optimal (Minimal serialization overhead)"],
    "Infrastructure Complexity": ["Minimal (Standard HTTP clients, curl)", "Moderate (Requires streaming connection maintenance)", "High (Stateful proxy routing, load balancer sticky sessions)", "High (Requires compiled protobuf stubs and gRPC load balancers)"],
    "Primary Use Case": ["Batch extraction, background processing, JSON tool calls", "Interactive chat applications, documentation assistants", "Real-time speech-to-speech agents, audio streaming", "High-throughput internal microservice-to-microservice inference"]
})
st.dataframe(protocol_table_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - API FUNDAMENTALS
# ==============================================================================
st.header("2. API Fundamentals: REST Endpoints, JSON-RPC, Model Namespaces, and Specifications", divider="rainbow")

st.markdown("""
Commercial LLM APIs have largely standardized around the **OpenAI Chat Completion Specification** as an industry de-facto standard. 
Self-hosted engines (vLLM, Ollama, TGI, LocalAI) implement this identical schema, allowing enterprise clients to swap model backends seamlessly.
""")

col_api1, col_api2 = st.columns([1, 1])
with col_api1:
    st.markdown("""
    ### Canonical Chat Completion Endpoint
    * **HTTP Method**: `POST`
    * **Standard Path**: `/v1/chat/completions`
    * **Core Headers**:
      * `Content-Type: application/json`
      * `Authorization: Bearer <API_KEY>`
    * **Model Namespacing**:
      * Explicit string identifier designating checkpoint and context capacity:
        `"model": "gpt-4o-2024-08-06"`, `"model": "meta-llama/Llama-3.1-70B-Instruct"`
    """)

with col_api2:
    st.markdown("""
    ### Core Hyperparameters in Request Body
    * `temperature`: Controls randomness in softmax sampling (0.0 = deterministic greedy decoding, 1.0 = creative sampling).
    * `top_p`: Nucleus sampling cutoff (cumulative probability threshold).
    * `max_tokens`: Hard upper bound on generated completion tokens.
    * `stop`: String or list of tokens terminating generation immediately (e.g., `["\n\n", "User:"]`).
    * `stream`: Boolean flag toggling Server-Sent Events streaming.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - AUTHENTICATION
# ==============================================================================
st.header("3. Authentication: Bearer Tokens, Ephemeral Key Vaults, and IAM Service Principals", divider="rainbow")

st.markdown("""
Securing LLM API endpoints requires isolating credentials from client-side code and maintaining 
fine-grained access policies to prevent account takeover and unauthorized quota consumption.
""")

st.markdown("""
### Authentication Patterns Across Providers
1. **Bearer Token Authentication (OpenAI / Anthropic / Groq)**:
   * Client transmits static secret in HTTP header: `Authorization: Bearer sk-proj-...`
   * *Enterprise Hazard*: If embedded in frontend JavaScript or committed to git repositories, attackers drain account credits.
   * *Mitigation*: Backend API reverse proxies (LiteLLM, Cloudflare AI Gateway) inject the bearer token server-side.
2. **AWS SigV4 (Amazon Bedrock)**:
   * Requires cryptographically signing HTTP request headers with temporary AWS STS credentials (access key, secret key, session token). Zero static API keys transmitted over the wire.
3. **Google Cloud OAuth2 / Service Accounts (Vertex AI)**:
   * Short-lived Google OAuth bearer tokens refreshed automatically via service account JSON keys.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CHAT COMPLETION
# ==============================================================================
st.header("4. Chat Completion: Messages Payload Structure, System/User/Assistant Roles, and Seed Reproducibility", divider="rainbow")

st.markdown("""
Modern LLMs are conversational foundation models expecting a structured array of **Messages**, 
where each item defines a specific conversational participant role.
""")

col_cc1, col_cc2 = st.columns([1, 1])
with col_cc1:
    st.markdown("""
    ### The Four Core Participant Roles
    1. **`system`**: Sets the global persona, behavioral rules, constraints, and operational boundaries.
    2. **`user`**: The end-user prompt, question, or external data to be processed.
    3. **`assistant`**: Model-generated completions from earlier conversation turns (used for few-shot prompting or multi-turn history).
    4. **`tool` / `function`**: Structured execution results returned by external client tools back to the model.
    """)

with col_cc2:
    st.markdown("""
    ### Deterministic Seed Control (`seed`)
    * Autoregressive GPU inference is inherently non-deterministic due to floating-point atomic addition races across CUDA thread blocks.
    * Supplying an integer parameter:
      `"seed": 42`
      instructs the inference engine to make a best-effort attempt to sample deterministically, returning the exact same completion tokens for automated regression testing.
    * The response header includes `system_fingerprint` tracking underlying server backend hardware changes.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - STREAMING RESPONSES
# ==============================================================================
st.header("5. Streaming Responses: Server-Sent Events (SSE), Chunk Delimiters, and Time-to-First-Token (TTFT)", divider="rainbow")

st.markdown("""
Generating a 500-token completion at 30 tokens/second requires ~17 seconds of inference time. 
If an application waits for the entire completion before responding, the user experiences an unbearable 17-second freeze. 
**Streaming** transmits tokens incrementally over **Server-Sent Events (SSE)**.
""")

st.markdown("""
### Anatomy of an SSE Stream
When `"stream": true` is passed, the server responds with HTTP header:
`Content-Type: text/event-stream`

The server pushes individual chunks separated by double newlines (`\n\n`):
```text
data: {"choices":[{"delta":{"content":"The"},"index":0,"finish_reason":null}]}

data: {"choices":[{"delta":{"content":" capital"},"index":0,"finish_reason":null}]}

data: {"choices":[{"delta":{"content":" of"},"index":0,"finish_reason":null}]}

data: [DONE]
```
* **Time-to-First-Token (TTFT)** drops from 17,000ms to **300ms**, dramatically improving perceived user responsiveness.
* The terminal sentinel string `data: [DONE]` signals the client parser to close the HTTP socket cleanly.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - TOKEN USAGE
# ==============================================================================
st.header("6. Token Usage: Prompt Tokens, Completion Tokens, Context Budgeting, and Cost Calculation", divider="rainbow")

st.markdown("""
Commercial LLM providers bill exclusively by **Tokens consumed**, not HTTP requests or server CPU seconds. 
Furthermore, **Prompt Tokens** and **Completion Tokens** are billed at asymmetric price rates.
""")

col_tu1, col_tu2 = st.columns([1, 1])
with col_tu1:
    st.markdown("""
    ### Asymmetric Pricing Rationale
    * **Prompt (Input) Tokens**: Processed in parallel during the **Prefill Phase** using high-throughput matrix multiplications (`GEMM`). Incur lower compute cost (typically $2.50 to $5.00 per 1M tokens).
    * **Completion (Output) Tokens**: Generated sequentially one token at a time during the **Decode Phase** (`GEMV`), memory-bandwidth bound. Incur 3x to 4x higher cost (typically $10.00 to $15.00 per 1M tokens).
    """)

with col_tu2:
    st.markdown("""
    ### Mathematical Cost Formula
    The cost of an inference interaction is:
    `Total_Cost = (N_prompt * Cost_prompt_per_token) + (N_completion * Cost_completion_per_token)`

    The completion response payload provides exact token counters:
    ```json
    "usage": {
        "prompt_tokens": 1420,
        "completion_tokens": 185,
        "total_tokens": 1605
    }
    ```
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - RATE LIMITS
# ==============================================================================
st.header("7. Rate Limits: Token-per-Minute (TPM), Request-per-Minute (RPM), and Leaky Bucket Algorithms", divider="rainbow")

st.markdown("""
Cloud providers enforce strict multi-dimensional rate limits to prevent GPU cluster saturation and manage multi-tenant fairness:
1. **RPM (Requests per Minute)**: Maximum count of discrete HTTP API calls permitted in a rolling 60-second window.
2. **TPM (Tokens per Minute)**: Maximum cumulative sum of prompt + completion tokens processed in a rolling 60-second window.
""")

st.markdown("""
### Rate Limiting Algorithms
* **Token Bucket Algorithm**: A bucket fills with rate tokens at a continuous rate $R$. Each incoming request claims $N$ tokens. If the bucket lacks sufficient tokens, the request is rejected with `HTTP 429 Too Many Requests`.
* **Leaky Bucket Algorithm**: Requests enter a queue that drains at a steady, fixed rate, smoothing out spiky traffic bursts.
* **Provider Response Headers**:
  * `x-ratelimit-remaining-requests`: Remaining RPM quota.
  * `x-ratelimit-remaining-tokens`: Remaining TPM quota.
  * `x-ratelimit-reset-tokens`: Milliseconds until the token bucket resets.
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - ERROR HANDLING
# ==============================================================================
st.header("8. Error Handling: HTTP 429 Backoff, Context Window Exceeded (400), Server Overload (503), & Circuit Breakers", divider="rainbow")

st.markdown("""
In distributed AI architectures, failures are not exceptions; they are inevitable operational events. 
Production applications must handle specific error categories deterministically:
""")

error_taxonomy_df = pd.DataFrame({
    "HTTP Status Code": ["HTTP 429 (Too Many Requests)", "HTTP 400 (Context Length Exceeded)", "HTTP 500 / 503 (Server Overload)", "HTTP 401 (Unauthorized)"],
    "Root Cause": ["Exceeded allocated RPM or TPM tier quotas", "Prompt tokens + max_tokens exceeds model context window limit (e.g. > 128k)", "Cloud GPU cluster out of capacity or internal inference engine crash", "Expired, revoked, or malformed API token"],
    "Deterministic Resolution": ["Exponential backoff with full randomized jitter; queue request in client buffer", "Truncate conversation history, summarize earlier context, or upgrade to long-context model", "Failover to secondary provider (e.g. failover from OpenAI to Anthropic or Azure)", "Alert security administrator immediately; do not retry"]
})
st.dataframe(error_taxonomy_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION CODE IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Resilient Enterprise LLM API Client Engine", divider="rainbow")

st.markdown("""
The production class below demonstrates a resilient LLM API client implementing 
token budgeting, streaming SSE chunk parsing, simulated rate-limit handling, and exponential backoff retry logic.
""")

class ResilientLLMClient:
    """
    Enterprise-grade LLM API Client implementing token budgeting,
    rate limit backoff simulation, and streaming SSE decoding.
    """
    def __init__(self, api_key: str = "mock-sk-live-9921", tpm_quota: int = 10000):
        self.api_key = api_key
        self.tpm_quota = tpm_quota
        self.used_tokens_window = 0
        self.prompt_cost_per_k = 0.0025  # $2.50 per 1M tokens ($0.0025 per 1k)
        self.completion_cost_per_k = 0.010 # $10.00 per 1M tokens ($0.010 per 1k)

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates total dollar cost for the API transaction."""
        cost = (prompt_tokens / 1000.0 * self.prompt_cost_per_k) + \
               (completion_tokens / 1000.0 * self.completion_cost_per_k)
        return round(cost, 6)

    def simulate_chat_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> Dict[str, Any]:
        """
        Executes a simulated chat completion with rate-limiting verification 
        and token usage accounting.
        """
        full_prompt = " ".join([m["content"] for m in messages])
        prompt_tokens = self.estimate_tokens(full_prompt)

        # Check TPM limit
        if self.used_tokens_window + prompt_tokens > self.tpm_quota:
            return {
                "success": False,
                "error_code": 429,
                "error_message": "Rate limit exceeded (TPM Quota Exhausted). Retrying with exponential backoff...",
                "tokens": 0,
                "cost": 0.0
            }

        # Simulated response generation
        canned_response = "Acme Corp Q3 summary: Revenue reached $4.2B with steady margin expansion. Enterprise renewals remain strong."
        completion_tokens = self.estimate_tokens(canned_response)
        total_tokens = prompt_tokens + completion_tokens
        
        self.used_tokens_window += total_tokens
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        # Generate simulated SSE chunks
        chunks = canned_response.split()
        sse_events = [f'data: {{"delta": "{chunk} "}}\n\n' for chunk in chunks]
        sse_events.append('data: [DONE]\n\n')

        return {
            "success": True,
            "response_text": canned_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "sse_chunks": sse_events if stream else []
        }

st.code('''
class ResilientLLMClient:
    def estimate_tokens(self, text: str) -> int:
        # Computes token approximation (~4 chars/token)
        ...

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Computes dollar cost based on asymmetric pricing
        ...

    def simulate_chat_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> Dict[str, Any]:
        # Evaluates rate limits, executes completion, and emits SSE stream chunks
        ...
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: LLM API Client & Rate-Limit Simulator", divider="rainbow")

st.markdown("""
Test how chat completion payloads are validated, stream tokens via **Server-Sent Events (SSE)**, 
and observe how **TPM Rate Limits** and **Cost Accounting** behave in production.
""")

client = ResilientLLMClient(tpm_quota=1500)

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Chat Completion Request Payload")
    sys_prompt = st.text_input("System Role Directive:", value="You are Acme Financial Analyst assistant. Be concise.")
    user_prompt = st.text_area("User Role Message:", value="Summarize the key revenue highlights from our Q3 quarterly earnings report.", height=100)
    
    c1, c2 = st.columns(2)
    with c1:
        param_temp = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    with c2:
        param_stream = st.checkbox("Enable SSE Streaming", value=True)

    sim_calls_count = st.slider("Simulate Burst Calls (Test TPM Quota):", min_value=1, max_value=5, value=1)
    send_api_btn = st.button("🚀 Dispatch Chat Completion API Request", type="primary")

with col_wb2:
    st.subheader("2. API Telemetry & Streaming Stream")
    if send_api_btn or user_prompt:
        messages_payload = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        last_result = None
        for i in range(sim_calls_count):
            last_result = client.simulate_chat_completion(messages_payload, stream=param_stream)

        if last_result["success"]:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Tokens", last_result["total_tokens"], delta=f"{last_result['prompt_tokens']} In / {last_result['completion_tokens']} Out")
            with m2:
                st.metric("Estimated Cost", f"${last_result['estimated_cost_usd']:.6f}")
            with m3:
                st.metric("TPM Window Used", f"{client.used_tokens_window} / {client.tpm_quota}")

            st.markdown("#### Real-Time Streaming Output (SSE Simulation)")
            st.info(last_result["response_text"])

            if param_stream and last_result["sse_chunks"]:
                with st.expander("🔍 View Raw Server-Sent Events (SSE) Wire Chunks", expanded=False):
                    st.code("".join(last_result["sse_chunks"][:6]) + "...\ndata: [DONE]\n\n", language="text")
        else:
            st.error(f"HTTP 429 ERROR: {last_result['error_message']}")
            st.warning("Client Action: Activating exponential backoff retry with full jitter (Wait: 1.84s)...")

st.divider()

# ==============================================================================
# SECTION 11: INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Engineering Interview Questions")

with st.expander("Q1: Explain the architectural mechanics of Server-Sent Events (SSE) in streaming LLM completions."):
    st.markdown("""
    **Answer**: Server-Sent Events (SSE) is an HTTP standard (`text/event-stream`) allowing a server to push incremental unbuffered data over a single persistent TCP connection. 
    In streaming LLM APIs:
    1. The client initiates an HTTP POST request with `"stream": true`.
    2. The server responds with `Content-Type: text/event-stream` and `Transfer-Encoding: chunked`.
    3. As the GPU inference engine generates each new token during the autoregressive decode phase, it packages the token delta into a JSON payload formatted as `data: {"delta": {"content": "..."}}\n\n`.
    4. The client's EventSource or HTTP reader parses each line immediately upon receipt, updating the UI.
    5. The server signals stream completion by sending the sentinel token `data: [DONE]\n\n` and closing the connection.
    """)

with st.expander("Q2: Why is exponential backoff with 'Full Jitter' preferred over simple exponential backoff for HTTP 429 errors?"):
    st.markdown("""
    **Answer**: When thousands of distributed worker threads hit a rate limit simultaneously, standard exponential backoff causes all workers to wait the exact same duration:
    `Wait = 2^attempt * Base`
    Consequently, all workers retry at the exact same millisecond, creating a synchronized thundering herd problem that repeatedly crashes the rate limiter. 
    **Full Jitter** randomizes the sleep duration uniformly:
    `Wait_Jitter = Random(0, 2^attempt * Base)`
    This desynchronizes worker retries, smoothing out traffic spikes and maximizing throughput.
    """)

with st.expander("Q3: What causes the asymmetric pricing between Prompt (Input) Tokens and Completion (Output) Tokens?"):
    st.markdown("""
    **Answer**: The asymmetry stems from GPU computational characteristics in transformer inference:
    * **Prompt Processing (Prefill Phase)**: All input tokens are known simultaneously. The attention and MLP operations execute in parallel across tensor cores as large matrix-matrix multiplications (`GEMM`), maximizing GPU compute utilization.
    * **Completion Generation (Decode Phase)**: Each token depends on the previous token. Generation is strictly sequential (`O(1)` token per forward pass) and executes as memory-bandwidth bound matrix-vector multiplications (`GEMV`). Serving sequential decode passes requires orders of magnitude more GPU memory-bandwidth time per token, justifying 3x to 4x higher pricing.
    """)

with st.expander("Q4: How does a Circuit Breaker pattern protect enterprise backend systems from cascading LLM API outages?"):
    st.markdown("""
    **Answer**: A Circuit Breaker monitors the health of external LLM API endpoints across three states:
    1. **CLOSED**: Normal operation; all requests pass to the primary provider.
    2. **OPEN**: If error rates (HTTP 500, 503, timeouts) exceed a threshold (e.g., > 20% over 30s), the breaker trips OPEN. All subsequent requests immediately fail fast or route to a fallback provider (e.g., Azure or local vLLM) without hitting the struggling primary endpoint.
    3. **HALF-OPEN**: After a cooldown period (e.g., 60s), the breaker permits a small sample of probe requests. If they succeed, it resets to CLOSED; if they fail, it re-opens.
    """)

with st.expander("Q5: What is the purpose of the `seed` parameter and `system_fingerprint` in OpenAI chat completions?"):
    st.markdown("""
    **Answer**: The `seed` parameter requests deterministic sampling for regression testing and debugging. 
    However, hardware architecture changes, CUDA driver upgrades, or server-side quantization shifts can alter floating-point summation orders. 
    The `system_fingerprint` response header represents the specific backend software/hardware configuration used to service the request. 
    If two identical requests return different outputs despite having the same `seed`, checking whether the `system_fingerprint` changed reveals whether the backend provider modified its underlying inference environment.
    """)

st.subheader("Critical Production Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Hardcoding Static API Secrets in Frontend Applications**
    * *Error*: Placing raw API keys in client-side mobile apps or single-page React apps.
    * *Impact*: Keys are extracted via browser inspection within seconds, resulting in thousands of dollars in unauthorized usage.
    * *Fix*: Always route requests through an authenticated backend proxy or API gateway.
    """)
    st.error("""
    **Trap 2: Blocking Entire Workers on Long Non-Streaming Completions**
    * *Error*: Synchronously awaiting a 20-second LLM completion in a web thread pool.
    * *Impact*: Gunicorn/Uvicorn worker starvation causing HTTP 504 gateway timeouts across the entire platform.
    * *Fix*: Use asynchronous I/O (`httpx.AsyncClient`) and stream tokens via Server-Sent Events.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Retrying HTTP 400 Context Length Exceeded Errors**
    * *Warning*: Treating all API errors as transient retriable network glitches.
    * *Impact*: Repeating a prompt that exceeds the context window wastes resources and never succeeds.
    * *Fix*: Inspect error codes; only retry transient errors (429, 500, 503). For 400 errors, truncate the context.
    """)
    st.warning("""
    **Trap 4: Neglecting Token Budgeting Before Forward Dispatch**
    * *Warning*: Dispatching large documents without verifying prompt token counts via local tokenizers (`tiktoken`).
    * *Impact*: Incurring unexpected rate limits and unpredictable cloud bills.
    * *Fix*: Measure prompt token counts locally and reject or summarize inputs exceeding context quotas.
    """)

st.subheader("Concept Flashcards")
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("🃏 Flashcard: TTFT vs ITL"):
        st.markdown("**TTFT (Time-to-First-Token)**: Latency from request dispatch until the first token appears on screen.\\n\\n**ITL (Inter-Token Latency)**: Duration between successive streaming tokens during generation (e.g., 30ms/token).")
with fc2:
    with st.expander("🃏 Flashcard: Full Jitter"):
        st.markdown("**Formula**: `Sleep = Random(0, 2^attempt * Base)`\\n\\n**Purpose**: Eliminates synchronized thundering herd retries by distributing worker traffic evenly across time.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LLM API engineering, streaming protocols, and resilience architecture:")

chk1 = st.checkbox("I understand the structure of the standardized Chat Completion API schema (system, user, assistant, tool roles).", value=True)
chk2 = st.checkbox("I can implement Server-Sent Events (SSE) streaming and handle the `data: [DONE]` terminal sentinel.", value=True)
chk3 = st.checkbox("I understand why Prompt and Completion tokens have asymmetric pricing due to Prefill vs Decode GPU characteristics.", value=True)
chk4 = st.checkbox("I can differentiate between TPM and RPM rate limits and implement exponential backoff with full jitter.", value=True)
chk5 = st.checkbox("I know how to protect API credentials using backend reverse proxies and ephemeral secret vaults.", value=True)
chk6 = st.checkbox("I can design Circuit Breakers to gracefully failover between cloud inference providers during outages.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered LLM API Fundamentals! Proceed to Lesson 23.2: Application Architecture.")
