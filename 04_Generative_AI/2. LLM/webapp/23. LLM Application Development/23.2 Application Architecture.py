import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="23.2 Application Architecture - Generative AI Curriculum",
    page_icon="🏛️",
    layout="wide"
)

# Title & Overview Block
st.title("23.2 Application Architecture: Frontend, Backend, LLM Service, Databases, Caching, & Auth")
st.caption("Chapter 23: LLM Application Development | Domain: 2. Large Language Models")

st.markdown("""
Building production-grade Generative AI software requires transcending single-file Python scripts and Jupyter notebooks. 
An enterprise LLM application is a distributed, high-concurrency systems engineering architecture: coupling 
**Reactive Frontends** (handling streaming token deltas and LaTeX rendering), **Asynchronous Backends** (FastAPI / Celery orchestration), 
**Decoupled LLM Gateway Services** (routing queries across multi-provider cloud clusters), **Relational Databases** (session persistence), 
**Vector Databases** (semantic RAG embeddings), **Multi-Tier Caching Layers** (semantic Redis caches and prompt KV-cache reuse), 
and **Identity Access Management** (OAuth2/JWT tenant isolation).

Architecting enterprise LLM applications demands deep systems design: resolving the impedance mismatch between millisecond-latency database 
queries and multi-second autoregressive generation streams while maintaining strict high availability, cost efficiency, and horizontal scalability.
""")

st.markdown("""
In this lesson, you will master:
1. The seven architectural tiers of modern enterprise Generative AI systems.
2. Frontend streaming UI patterns: Server-Sent Events rendering, optimistic updates, and markdown sanitization.
3. Asynchronous backend orchestration: non-blocking event loops, WebSocket proxies, and job queues.
4. The LLM Service Gateway: multi-model routing, automatic failover, load balancing, and prompt management.
5. The storage tri-factor: Relational SQL stores, distributed Vector databases (HNSW), and Exact/Semantic in-memory caching (Redis/GPTCache).
""")

st.divider()

# ==============================================================================
# SECTION 1: BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Full-Stack Distributed Enterprise LLM Application Topology", divider="rainbow")

st.markdown("""
The architecture blueprint below illustrates the decoupled tiers of a production enterprise Generative AI platform:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    app_arch_dot = """
    digraph EnterpriseLLMTopology {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_presentation {
            label="Tier 1: Presentation & Client Ingress Tier";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            WebClient [label="Single Page Application (Next.js / React / Streamlit)\\nReal-time SSE markdown renderer & client-side state", fillcolor="#1E293B", color="#38BDF8"];
            APIGateway [label="Cloud Reverse Proxy & Ingress Load Balancer\\n(Nginx / Cloudflare / Envoy) TLS 1.3 Termination", fillcolor="#334155", color="#64748B"];

            WebClient -> APIGateway;
        }

        subgraph cluster_backend {
            label="Tier 2: Asynchronous Backend & Orchestration Tier";
            style="rounded";
            bgcolor="#151E32";
            color="#8B5CF6";
            fontcolor="#8B5CF6";
            fontsize=12;

            AuthService [label="Identity & Access Management (OAuth2 / Okta / JWT)\\nVerifies tenant_id, user_role, rate quotas", fillcolor="#4338CA", color="#818CF8"];
            AppServer [label="Asynchronous Application Server (FastAPI / Python asyncio)\\nConversation lifecycle manager & prompt assembler", fillcolor="#6D28D9", color="#A78BFA"];
            TaskQueue [label="Distributed Background Worker Pool (Celery / Redis Queue)\\nLong-running document ingestion, evaluations, batch jobs", fillcolor="#0F766E", color="#14B8A6"];

            APIGateway -> AuthService -> AppServer;
            AppServer -> TaskQueue;
        }

        subgraph cluster_data {
            label="Tier 3: Persistence, Vector Index, & Caching Tier";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            SemanticCache [label="In-Memory Semantic Cache (Redis / GPTCache)\\nExact prompt hash + Cosine vector similarity match (L2 distance)", fillcolor="#991B1B", color="#EF4444", shape=cylinder];
            RelationalDB [label="Relational Store (PostgreSQL / Supabase)\\nChat session history, user profiles, audit logs, billing records", fillcolor="#1E293B", color="#38BDF8", shape=cylinder];
            VectorDB [label="Enterprise Hybrid Vector Database (Qdrant / Pinecone / pgvector)\\nDense embedding HNSW index + Sparse BM25 index", fillcolor="#15803D", color="#22C55E", shape=cylinder];

            AppServer -> SemanticCache;
            AppServer -> RelationalDB;
            AppServer -> VectorDB;
        }

        subgraph cluster_llm_service {
            label="Tier 4: Enterprise LLM Service & Multi-Cloud Provider Layer";
            style="rounded";
            bgcolor="#151E32";
            color="#F59E0B";
            fontcolor="#F59E0B";
            fontsize=12;

            LLMGateway [label="Model Gateway & Router (LiteLLM / Portkey)\\nDynamic load balancing, cost optimizer, circuit breaker", fillcolor="#334155", color="#64748B"];
            CloudLLM1 [label="Primary Frontier Provider (OpenAI / Claude 3.5)\\nHigh-complexity reasoning & coding tasks", fillcolor="#047857", color="#10B981"];
            CloudLLM2 [label="Secondary / Fallback Provider (Azure OpenAI / Bedrock)\\nAutomated failover during 503 provider outages", fillcolor="#0369A1", color="#38BDF8"];
            SelfHosted [label="Self-Hosted Inference Engine (vLLM / TensorRT-LLM)\\nLow-latency open weights (Llama 3.1 8B) on private GPUs", fillcolor="#4338CA", color="#818CF8"];

            AppServer -> LLMGateway [label="Cache Miss -> Route to LLM"];
            LLMGateway -> CloudLLM1;
            LLMGateway -> CloudLLM2 [label="Failover", style="dotted"];
            LLMGateway -> SelfHosted [label="Low-Cost Tasks"];
        }

        CloudLLM1 -> AppServer [label="SSE Streaming Tokens", style="dashed", color="#10B981"];
        AppServer -> WebClient [label="Egress Stream to Browser", style="dashed", color="#38BDF8"];
    }
    """
    st.graphviz_chart(app_arch_dot, use_container_width=True)

st.markdown("""
### Enterprise Architecture Component Matrix
The table below specifies the technologies and latency expectations for each architecture tier:
""")

arch_matrix_df = pd.DataFrame({
    "Tier": ["1. Presentation / Frontend", "2. Backend Orchestrator", "3. Semantic Caching", "4. Relational Storage", "5. Vector Storage", "6. LLM Gateway & Router"],
    "Technology Stack": ["Next.js (React), Tailwind, Vercel AI SDK, Streamlit", "FastAPI (Python), Node.js, LangGraph, Temporal", "Redis, DragonflyDB, GPTCache", "PostgreSQL (with JSONB), Supabase, Prisma", "Qdrant, Pinecone, Weaviate, pgvector", "LiteLLM, Portkey, Cloudflare AI Gateway, Kong"],
    "Typical Latency": ["Client-side render (16ms / 60fps)", "5ms - 15ms internal routing", "1ms (Hash) to 8ms (Semantic vector match)", "2ms - 10ms ACID transactional queries", "15ms - 40ms ANN HNSW vector search", "100ms - 800ms Time-to-First-Token (TTFT)"],
    "Primary Scaling Challenge": ["Handling hundreds of concurrent streaming SSE sockets", "Managing non-blocking Python asyncio event loop worker limits", "Cache invalidation on document updates; vector threshold tuning", "Partitioning chat history tables containing millions of messages", "HNSW memory consumption (RAM-heavy vector graphs)", "Managing provider rate quotas (TPM/RPM) and distributed failover"]
})
st.dataframe(arch_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - FRONTEND
# ==============================================================================
st.header("2. Frontend: Reactive Streaming UIs, Markdown/Code Rendering, and Optimistic Updates", divider="rainbow")

st.markdown("""
The **Frontend Layer** is the primary interface between human end-users and the probabilistic language model. 
Traditional web applications render static JSON payloads; generative AI frontends must manage **real-time streaming token deltas**.
""")

col_fe1, col_fe2 = st.columns([1, 1])
with col_fe1:
    st.markdown("""
    ### Streaming UI Requirements
    1. **Streaming Token Reassembly**: Incrementally appending incoming SSE text deltas to the UI DOM without causing full component re-renders.
    2. **Streaming Markdown & Code Parser**: Incrementally syntax-highlighting code blocks (````python...````) before the closing triple backticks arrive over the wire.
    3. **Optimistic Updates**: Displaying the user's message immediately upon submission before the network request returns.
    """)

with col_fe2:
    st.markdown("""
    ### Latency Perception & Scroll Anchoring
    * **Auto-Scroll Anchoring**: Keeping the viewport locked to the bottom of the streaming text stream while permitting the user to scroll up without jarring jumps.
    * **Cancel Stream Capability**: Providing an immediate "Stop Generating" button that aborts the client `AbortController` signal, terminating the upstream HTTP socket.
    * **Error Resilience**: Gracefully displaying partial completions if an SSE connection disconnects prematurely.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - BACKEND
# ==============================================================================
st.header("3. Backend: Asynchronous Task Queues, WebSocket/SSE Gateways, and Frameworks", divider="rainbow")

st.markdown("""
The **Backend Layer** acts as the central conductor: authenticating requests, retrieving user history, assembling prompts, 
and managing streaming responses without exhausting server worker threads.
""")

st.markdown("""
### The Asynchronous Concurrency Imperative
* In synchronous frameworks (e.g., standard Flask / Django WSGI), each web worker thread blocks until the HTTP response completes. If an LLM takes 15 seconds to generate tokens, **100 concurrent users will exhaust a 100-worker cluster completely**, crashing the API.
* Modern GenAI backends mandate **Asynchronous I/O (FastAPI / ASGI / Node.js)**:
  ```python
  @app.post("/chat")
  async def chat_endpoint(request: ChatRequest):
      async def token_generator():
          async for chunk in llm_client.astream(request.prompt):
              yield f"data: {chunk}\n\n"
      return StreamingResponse(token_generator(), media_type="text/event-stream")
  ```
* Under asyncio, a single server process can manage **10,000+ idle streaming connections concurrently** while waiting for GPU decode passes.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - LLM SERVICE
# ==============================================================================
st.header("4. LLM Service Layer: Model Routing, Gateway Proxies, Continuous Batching, and Fallbacks", divider="rainbow")

st.markdown("""
Coupling application code directly to a single vendor SDK (`import openai`) creates vendor lock-in and single-point-of-failure vulnerabilities. 
The **LLM Service Gateway** abstracts multiple providers behind a unified proxy.
""")

col_gw1, col_gw2 = st.columns([1, 1])
with col_gw1:
    st.markdown("""
    ### Core Capabilities of an LLM Gateway
    * **Dynamic Model Routing**: Directing simple classification tasks to fast, low-cost models (Llama 3.1 8B, GPT-4o-mini) and complex reasoning tasks to frontier models (Claude 3.5 Sonnet, o1).
    * **Automated Provider Failover**: If OpenAI returns HTTP 503 or 429, the gateway automatically switches to Azure OpenAI or AWS Bedrock within 50ms without client awareness.
    * **Load Balancing**: Distributing high-volume traffic across multiple API keys or multiple cloud regions to bypass TPM/RPM quotas.
    """)

with col_gw2:
    st.markdown("""
    ### Continuous Batching in Self-Hosted Engines
    * For self-hosted instances (vLLM, TensorRT-LLM):
      * Traditional static batching waits for $B$ requests to finish before starting new ones.
      * **Continuous (Iteration-Level) Batching**: Ingests new incoming requests into the GPU batch immediately after any existing request finishes generating a single token.
      * Increases GPU throughput by **4x to 10x** compared to naive inference servers.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - DATABASE
# ==============================================================================
st.header("5. Database: Relational & Document Stores for Persistent Chat Threads, User Profiles, & Audit Logs", divider="rainbow")

st.markdown("""
While vector databases store non-parametric knowledge, the **Relational / Document Store** (PostgreSQL, MySQL, MongoDB) 
remains the operational backbone for transactional enterprise data.
""")

st.markdown("""
### Relational Schema Design for Generative Conversations
A robust conversation schema captures multi-turn history, token usage telemetry, and human feedback:
```sql
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    title VARCHAR(255)
);

CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL, -- 'system', 'user', 'assistant', 'tool'
    content TEXT NOT NULL,
    prompt_tokens INT,
    completion_tokens INT,
    model_name VARCHAR(64),
    feedback_rating INT, -- +1 for thumbs up, -1 for thumbs down
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - VECTOR DATABASE
# ==============================================================================
st.header("6. Vector Database: Semantic Search Stores, ANN Indices (HNSW), Hybrid BM25, and Sharding", divider="rainbow")

st.markdown("""
The **Vector Database** stores high-dimensional dense embeddings (e.g., 1536-dimensional vectors from `text-embedding-3-small`) 
alongside document text and metadata, enabling sub-50ms Approximate Nearest Neighbor (ANN) semantic search.
""")

col_vdb1, col_vdb2 = st.columns([1, 1])
with col_vdb1:
    st.markdown("""
    ### Approximate Nearest Neighbor (HNSW)
    * Exact k-Nearest Neighbors (k-NN) requires computing cosine distance against all N vectors (O(N * d) complexity), which is unscalable for millions of documents.
    * **HNSW (Hierarchical Navigable Small World)**: A multi-layer graph index achieving O(log N) search complexity by traversing coarse-to-fine graph layers.
    * Provides **95%+ recall** while searching 1,000,000 vectors in under 10 milliseconds.
    """)

with col_vdb2:
    st.markdown("""
    ### Hybrid Search Architecture
    * Pure dense semantic search struggles with exact keywords (part numbers like `SKU-8821B`, legal statute numbers, acronyms).
    * Production vector databases combine:
      `Final_Score = alpha * Dense_Cosine_Score + (1 - alpha) * Sparse_BM25_Score`
    * Using Reciprocal Rank Fusion (RRF) to merge the best of dense semantic nuance and sparse lexical precision.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CACHE
# ==============================================================================
st.header("7. Multi-Tier Caching: Exact Semantic Caching, KV-Cache Reuse, and Latency Optimization", divider="rainbow")

st.markdown("""
LLM calls are expensive ($0.01 per query) and slow (1,000ms+). 
**Caching** is the single most effective architectural optimization for slashing enterprise AI costs and latencies.
""")

cache_strategy_df = pd.DataFrame({
    "Caching Tier": ["1. Exact Prompt Hash Cache", "2. Semantic Vector Cache (GPTCache)", "3. Prompt KV-Cache Reuse (vLLM / Anthropic)"],
    "Storage Engine": ["In-Memory Redis / Key-Value Store", "Redis / Vector DB + Distance Threshold", "GPU High-Bandwidth Memory (HBM) / VRAM"],
    "Hit Condition": ["Input prompt string matches previous query byte-for-byte (`SHA256(prompt)`)", "Input embedding vector is within cosine distance threshold (`distance < 0.08`)", "New prompt shares identical prefix tokens with previously processed prompt"],
    "Latency on Hit": ["< 2ms (Zero GPU compute required)", "< 10ms (Embedding calculation + vector lookup)", "10x faster prefill phase (Bypasses matrix multiplication)"],
    "Cost Reduction": ["100% cost reduction for repeated identical queries", "100% cost reduction for semantically equivalent queries", "50% to 90% discount on prompt token billing (Prompt Caching API)"]
})
st.dataframe(cache_strategy_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - AUTHENTICATION
# ==============================================================================
st.header("8. Authentication & Authorization: JWT Delegation, OAuth2, and Tenant-Bound RBAC", divider="rainbow")

st.markdown("""
In enterprise applications, **Identity and Access Management (IAM)** must govern both user identity and 
downstream model/data permissions.
""")

st.markdown("""
### Enterprise Auth Propagation Architecture
1. **User Authentication**: Client authenticates via OAuth2 / OIDC (Okta, Auth0, Azure AD) and receives a cryptographically signed **JSON Web Token (JWT)**.
2. **Gateway Verification**: The API Gateway validates the JWT signature, extracting `user_id`, `tenant_id`, and `roles` (e.g., `["finance_analyst", "employee"]`).
3. **Contextual Token Injection**: The backend binds the `tenant_id` to all database and vector search queries:
   `SELECT * FROM chunks WHERE tenant_id = jwt.tenant_id AND security_clearance IN (jwt.roles)`
   Ensuring that an authenticated user can never view or retrieve unauthorized cross-tenant corporate documents.
""")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION CODE IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full-Stack Enterprise Application Coordinator", divider="rainbow")

st.markdown("""
The production class below demonstrates an end-to-end enterprise application coordinator. 
It verifies user JWT tokens, queries an in-memory semantic cache, executes RAG retrieval, 
and coordinates simulated LLM streaming completion with session persistence.
""")

class EnterpriseAppCoordinator:
    """
    Coordinates full-stack application lifecycle: auth verification,
    exact/semantic caching, mock vector retrieval, and LLM dispatch.
    """
    def __init__(self):
        # Exact prompt cache: SHA256 -> response
        self.exact_cache = {}
        # Simulated chat message database
        self.conversation_db = []

    def authenticate_jwt(self, auth_header: str) -> Optional[Dict[str, Any]]:
        """Validates Bearer token and extracts user claims."""
        if not auth_header.startswith("Bearer ") or len(auth_header) < 15:
            return None
        # Mock decoded claims
        return {
            "user_id": "usr_9912",
            "tenant_id": "tenant_enterprise_a",
            "roles": ["analyst", "general_user"]
        }

    def check_cache(self, prompt: str) -> Optional[str]:
        """Checks exact cache for previously answered queries."""
        import hashlib
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        return self.exact_cache.get(prompt_hash)

    def store_cache(self, prompt: str, response: str):
        """Stores query response in exact cache."""
        import hashlib
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        self.exact_cache[prompt_hash] = response

    def process_chat_interaction(self, auth_header: str, user_prompt: str) -> Dict[str, Any]:
        """Orchestrates end-to-end chat request lifecycle."""
        # 1. Auth Gate
        user_claims = self.authenticate_jwt(auth_header)
        if not user_claims:
            return {"status": "ERROR_401", "error": "Invalid or missing Bearer token"}

        # 2. Cache Check
        cached_res = self.check_cache(user_prompt)
        if cached_res:
            return {
                "status": "CACHE_HIT",
                "response": cached_res,
                "latency_ms": 2,
                "cost_usd": 0.00,
                "user": user_claims["user_id"]
            }

        # 3. Cache Miss: Execute LLM Dispatch
        simulated_llm_response = f"Grounded response for '{user_prompt[:30]}...': Enterprise operational metrics indicate positive trajectory."
        self.store_cache(user_prompt, simulated_llm_response)
        
        # 4. Persist to Database
        self.conversation_db.append({
            "tenant": user_claims["tenant_id"],
            "user": user_claims["user_id"],
            "prompt": user_prompt,
            "response": simulated_llm_response
        })

        return {
            "status": "CACHE_MISS_GENERATED",
            "response": simulated_llm_response,
            "latency_ms": 380,
            "cost_usd": 0.0042,
            "user": user_claims["user_id"]
        }

st.code('''
class EnterpriseAppCoordinator:
    def authenticate_jwt(self, auth_header: str) -> Optional[Dict[str, Any]]:
        # Verifies JWT Bearer token and extracts tenant_id & roles
        ...

    def check_cache(self, prompt: str) -> Optional[str]:
        # Evaluates exact SHA-256 prompt cache for 0ms, $0 responses
        ...

    def process_chat_interaction(self, auth_header: str, user_prompt: str) -> Dict[str, Any]:
        # Full orchestration: Auth -> Cache Check -> Vector Retrieval -> LLM -> DB Save
        ...
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Architecture Topology Simulator & Cache Analyzer", divider="rainbow")

st.markdown("""
Simulate enterprise full-stack request lifecycles. Observe how **JWT Authentication Gates**, 
**In-Memory Caching**, and **Database Persistence** coordinate across repetitive and unique queries.
""")

coord = EnterpriseAppCoordinator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Ingress Request Simulation")
    bearer_token = st.text_input("Authorization Header (JWT):", value="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.mock-session-active")
    
    query_presets = [
        "What is our corporate policy on remote work?",
        "Summarize our latest enterprise Q3 revenue results.",
        "What is our corporate policy on remote work?", # Duplicate for cache hit test
        "Explain our cloud infrastructure backup frequency."
    ]
    selected_query = st.selectbox("Select or Test Scenario:", query_presets)
    custom_query = st.text_input("Or Enter Custom Query:", value=selected_query)
    
    dispatch_btn = st.button("🏛️ Execute Full-Stack Request Cycle", type="primary")

with col_wb2:
    st.subheader("2. Architecture Telemetry & Execution State")
    if dispatch_btn or custom_query:
        result = coord.process_chat_interaction(bearer_token, custom_query)
        
        if result["status"] == "ERROR_401":
            st.error(f"HTTP 401 Unauthorized: {result['error']}")
        else:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Cache Status", result["status"], delta="0ms / $0" if result["status"] == "CACHE_HIT" else "Generated")
            with m2:
                st.metric("Latency", f"{result['latency_ms']} ms")
            with m3:
                st.metric("Incurred Cost", f"${result['cost_usd']:.4f}")

            st.markdown("#### Response Payload")
            st.info(result["response"])

            st.markdown(f"**Database History:** Logged {len(coord.conversation_db)} message transactions under tenant.")

st.divider()

# ==============================================================================
# SECTION 11: INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Engineering Interview Questions")

with st.expander("Q1: Why is asynchronous I/O (ASGI/asyncio) mandatory over synchronous WSGI for LLM backend servers?"):
    st.markdown("""
    **Answer**: In traditional synchronous WSGI architectures (e.g., standard Flask or synchronous Django), each incoming HTTP request ties up 
    a dedicated operating system worker thread until the response completes. 
    Because LLM generation streams take anywhere from 2 to 30 seconds per request, a standard 32-worker server pool can only service 32 concurrent users 
    before all new requests queue and trigger HTTP 504 Gateway Timeouts. 
    In asynchronous ASGI architectures (FastAPI, Uvicorn, Node.js), workers release their thread while awaiting network I/O or token stream deltas, 
    allowing a single server process to maintain tens of thousands of active streaming SSE connections concurrently.
    """)

with st.expander("Q2: How does Semantic Caching differ from Exact Hash Caching in production LLM applications?"):
    st.markdown("""
    **Answer**: 
    * **Exact Hash Caching** computes a deterministic hash (e.g., SHA-256) of the prompt text. It only hits if the user's input matches previous queries byte-for-byte. Minor variations ('What is our return policy?' vs 'What is your return policy?') cause complete cache misses.
    * **Semantic Caching** embeds the incoming query into a dense vector and performs a fast nearest-neighbor search against previous cached queries in an in-memory vector index (Redis / GPTCache). If the cosine distance is below an empirical threshold (e.g., distance < 0.08), the cached response is returned immediately. This achieves **20% to 40% cache hit rates** on real customer traffic, eliminating millions in API costs.
    """)

with st.expander("Q3: What is the HNSW algorithm and why is it preferred over flat vector search in production Vector DBs?"):
    st.markdown("""
    **Answer**: Flat vector search calculates the exact cosine similarity against every vector in the database, requiring O(N * d) computations. 
    For 10 million vectors, flat search requires seconds of CPU/GPU time per query. 
    **HNSW (Hierarchical Navigable Small World)** builds a multi-layer graph where top layers have long-range skip edges and bottom layers have dense local connectivity. 
    Query traversal jumps across coarse clusters at high layers before zooming into the nearest neighborhood at layer 0, reducing search complexity from O(N) to O(log N) 
    and returning Top-K candidates in sub-15ms latencies.
    """)

with st.expander("Q4: How does Prompt KV-Cache reuse work in modern commercial LLM APIs (e.g., Anthropic Prompt Caching)?"):
    st.markdown("""
    **Answer**: Foundation models compute Key and Value tensor activations for each prompt token during the prefill phase. 
    If multiple customer requests share a large identical prefix (e.g., a 20,000-token enterprise system prompt or code repository context), 
    the cloud inference engine (using Radix trees in vLLM or internal cloud caching) **stores the precomputed KV-cache in GPU memory**. 
    Subsequent requests reuse the precomputed KV tensors directly, bypassing the matrix multiplications of the prefill phase, cutting Time-to-First-Token (TTFT) by up to 80% 
    and qualifying for a 50% to 90% cloud token cost discount.
    """)

with st.expander("Q5: What is the role of an LLM Gateway proxy (such as LiteLLM or Portkey) in enterprise architecture?"):
    st.markdown("""
    **Answer**: An LLM Gateway serves as a centralized reverse proxy decoupling application code from third-party model providers. 
    Key enterprise benefits include:
    1. **Unified Schema**: Normalizing disparate provider schemas (OpenAI, Anthropic, Bedrock, Gemini) into a single API format.
    2. **Automated Failover & Load Balancing**: Rerouting queries to secondary providers when primary endpoints experience rate limits (HTTP 429) or outages (HTTP 503).
    3. **Centralized Cost & Rate Quota Tracking**: Enforcing department-level budget caps before dispatching requests to external vendors.
    4. **Audit Logging & PII Redaction**: Scrubbing corporate data centrally before tokens leave the corporate VPC.
    """)

st.subheader("Critical Production Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Synchronous Blocking I/O on LLM API Calls**
    * *Error*: Calling blocking functions like `requests.post()` inside web endpoints.
    * *Impact*: Web server thread pools starve rapidly, causing HTTP 504 timeouts across all users.
    * *Fix*: Always use non-blocking asynchronous clients (`httpx.AsyncClient`) with `async/await`.
    """)
    st.error("""
    **Trap 2: Direct Client Coupling to a Single Cloud Provider SDK**
    * *Error*: Importing `from openai import OpenAI` throughout frontend and microservice codebases.
    * *Impact*: Total platform downtime when the sole provider experiences an outage; expensive refactors to adopt new models.
    * *Fix*: Place an LLM gateway (LiteLLM) in front of model calls with automated fallback routing.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Overly Permissive Semantic Cache Similarity Thresholds**
    * *Warning*: Setting the semantic cache distance threshold too loose (e.g., cosine distance > 0.15).
    * *Impact*: The cache returns answers for completely different questions, producing hallucinated responses.
    * *Fix*: Calibrate distance thresholds strictly (< 0.08) and invalidate cache on data source updates.
    """)
    st.warning("""
    **Trap 4: Neglecting Relational Database Indexing on Conversation IDs**
    * *Warning*: Querying chat history tables via `SELECT * FROM messages WHERE conversation_id = ?` without a B-tree index.
    * *Impact*: As message volume grows to millions of rows, database latency spikes from 2ms to 5,000ms.
    * *Fix*: Create composite indices on `(conversation_id, created_at)`.
    """)

st.subheader("Concept Flashcards")
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("🃏 Flashcard: Continuous Batching"):
        st.markdown("**Concept**: Iteration-level batching in inference engines (vLLM).\\n\\n**Benefit**: Ingests new requests into GPU forward passes immediately without waiting for existing completions to terminate.")
with fc2:
    with st.expander("🃏 Flashcard: Exact vs Semantic Caching"):
        st.markdown("**Exact Cache**: Byte-for-byte SHA256 prompt match ($0ms, $0 cost).\\n\\n**Semantic Cache**: Dense vector cosine proximity match; captures linguistic phrasing variations.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of distributed enterprise LLM application architecture:")

chk1 = st.checkbox("I understand the seven architectural tiers of an enterprise Generative AI application platform.", value=True)
chk2 = st.checkbox("I can explain why asynchronous ASGI backends are mandatory over synchronous WSGI for streaming generation.", value=True)
chk3 = st.checkbox("I understand the role of an LLM Service Gateway in multi-model routing, load balancing, and automated failovers.", value=True)
chk4 = st.checkbox("I can design a relational database schema for multi-turn conversations with token and feedback telemetry.", value=True)
chk5 = st.checkbox("I understand how HNSW graph indices enable O(log N) approximate nearest neighbor search in vector databases.", value=True)
chk6 = st.checkbox("I know how to implement multi-tier caching (Exact Hash, Semantic Vector Cache, and Prompt KV-Cache reuse).", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered LLM Application Architecture! Proceed to Lesson 23.3: Conversation Systems.")
