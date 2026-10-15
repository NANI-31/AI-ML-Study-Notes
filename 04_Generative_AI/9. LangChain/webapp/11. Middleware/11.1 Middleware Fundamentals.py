import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Callable

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockMiddlewareLayer:
    """Represents a discrete middleware interceptor in the execution stack."""
    def __init__(self, name: str, enabled: bool = True, early_return: bool = False, latency_ms: float = 12.0):
        self.name = name
        self.enabled = enabled
        self.early_return = early_return
        self.latency_ms = latency_ms

    def before_call(self, state: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        if not self.enabled:
            return state
        trace.append(f"🟢 [INGRESS] {self.name}: Validated state payload (latency: {self.latency_ms:.1f}ms)")
        if self.name == "Auth & Tenant Guard":
            state["tenant_id"] = state.get("tenant_id", "tenant-alpha")
            state["authenticated"] = True
        elif self.name == "PII & Masking Filter":
            prompt = state.get("prompt", "")
            state["prompt"] = prompt.replace("123-45-6789", "[REDACTED-SSN]").replace("user@corp.com", "[REDACTED-EMAIL]")
        elif self.name == "Semantic Cache":
            if self.early_return:
                state["cached_response"] = "Cached Response: Enterprise Policy Document v4.2"
                trace.append(f"⚡ [SHORT-CIRCUIT] {self.name}: Exact semantic match found. Aborting downstream LLM call.")
        return state

    def after_call(self, state: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        if not self.enabled or self.early_return:
            return state
        trace.append(f"🔵 [EGRESS] {self.name}: Post-processed response (latency: {self.latency_ms:.1f}ms)")
        if self.name == "Audit & Metric Logger":
            state["logged_to_siem"] = True
            state["trace_id"] = "tr-8942-uuid"
        elif self.name == "Safety Guardrails":
            output = state.get("response", "")
            if "malicious" in output.lower():
                state["response"] = "[REDACTED: Output flagged by Safety Guardrail]"
        return state


class MockMiddlewarePipelineSimulator:
    """Simulates an enterprise Onion-architecture middleware engine."""
    def __init__(self, layers: List[MockMiddlewareLayer]):
        self.layers = layers

    def execute(self, user_prompt: str, tenant_id: str) -> Dict[str, Any]:
        trace: List[str] = []
        state: Dict[str, Any] = {
            "prompt": user_prompt,
            "tenant_id": tenant_id,
            "tokens_in": len(user_prompt.split()) * 4,
            "early_returned": False
        }

        trace.append("🚀 Dispatching request into outer middleware onion layer...")
        short_circuited = False

        # Ingress (FIFO)
        for layer in self.layers:
            state = layer.before_call(state, trace)
            if layer.early_return and layer.name == "Semantic Cache":
                short_circuited = True
                state["early_returned"] = True
                state["response"] = state["cached_response"]
                break

        # Core Execution (simulated LLM / Agent)
        if not short_circuited:
            trace.append("🧠 Core Agent Loop / Model Invocation: Synthesizing plan with dynamic tools...")
            state["response"] = f"Agent Answer for '{state['prompt']}': Operation executed successfully under {state.get('tenant_id')}."
            state["tokens_out"] = 48
            state["total_tokens"] = state["tokens_in"] + state["tokens_out"]

            # Egress (LIFO - reverse order of middleware)
            trace.append("🔄 Response exiting core agent loop. Entering LIFO egress middleware chain...")
            for layer in reversed(self.layers):
                state = layer.after_call(state, trace)
        else:
            state["total_tokens"] = 0
            trace.append("⏭️ Skipped core LLM inference and egress chain due to early return.")

        total_latency = sum(l.latency_ms for l in self.layers if l.enabled)
        if not short_circuited:
            total_latency += 180.0  # simulated model latency
        state["total_latency_ms"] = total_latency
        state["trace_log"] = trace
        return state


# ==============================================================================
# STREAMLIT APPLICATION PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="11.1 Middleware Fundamentals",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for polished layout
st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-pill { background: #0284C7; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">11.1 Middleware Fundamentals</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cross-cutting interception, the Onion execution model, lifecycle hooks, and composable agent governance.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Agent Middleware Onion", divider="rainbow")

st.markdown("""
In enterprise agent engineering, business capabilities (prompts, tool logic, cognitive reasoning) must be strictly decoupled 
from infrastructure requirements (rate limiting, credential validation, audit logging, PII sanitization, semantic caching, and error trapping).
**Middleware** provides an extensible, non-invasive interception pipeline that surrounds the agent core in concentric, onion-like execution layers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MiddlewareArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_ingress {
            label="Inbound Ingress Pipeline (Outer to Inner)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            ClientReq [label="Client Inbound Request\\n(Prompt, Headers, Tenant)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            AuthLayer [label="Layer 1: Auth & Tenant Validation\\n(Verify JWT, inject tenant_id)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            RateLimit [label="Layer 2: Token Bucket Rate Limiting\\n(Throttling, concurrency limits)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            PIIFilter [label="Layer 3: PII & Prompt Sanitizer\\n(Regex/NER redaction, prompt defense)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            CacheLayer [label="Layer 4: Semantic Cache\\n(Vector similarity lookup on prompt)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        }

        subgraph cluster_core {
            label="Agent Core & Execution Runtime";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            AgentLoop [label="Agent ReAct Loop / Model Node\\n(LLM Prompt Assembly & Tool Invocation)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
        }

        subgraph cluster_egress {
            label="Outbound Egress Pipeline (Inner to Outer - LIFO)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            Guardrails [label="Layer 4: Safety & Toxicity Guardrails\\n(Hallucination & content inspection)", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            CostTracker [label="Layer 3: Cost & Token Tracker\\n(Usage accounting, billing attribution)", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            AuditLogger [label="Layer 2: Audit & SIEM Logger\\n(Immutable trace emission, telemetry)", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            ClientResp [label="Client Outbound Response\\n(Sanitized Answer, Metrics Header)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#34D399"];
        }

        ClientReq -> AuthLayer [label="1. Ingress Start"];
        AuthLayer -> RateLimit [label="2. Authorized"];
        RateLimit -> PIIFilter [label="3. Quota OK"];
        PIIFilter -> CacheLayer [label="4. Clean State"];
        CacheLayer -> AgentLoop [label="5. Cache Miss"];
        CacheLayer -> Guardrails [label="⚡ Cache Hit (Early Return)", style="dashed", color="#F59E0B", fontcolor="#F59E0B"];
        AgentLoop -> Guardrails [label="6. Model Output"];
        Guardrails -> CostTracker [label="7. Content Safe"];
        CostTracker -> AuditLogger [label="8. Billed"];
        AuditLogger -> ClientResp [label="9. Egress Delivery"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT IS MIDDLEWARE? ---
st.header("2. What is Middleware?", divider="rainbow")

st.markdown("""
In conversational AI and autonomous agent frameworks, **Middleware** is a software design pattern where independent, 
pluggable components intercept and transform requests and responses before and after they reach the core reasoning agent.

Rather than cluttering agent prompt templates or tool definitions with authentication logic, rate limiting, token counters, 
or safety scanners, middleware wraps around the execution chain like an **onion**.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("#### ❌ Without Middleware (Monolithic & Tangled)")
    st.code("""# Anti-Pattern: Agent prompt builder doing infrastructure chores
def run_agent(user_query, tenant_id, api_token):
    # 1. Auth mixed into agent
    if not verify_token(api_token): raise AuthError()
    # 2. Rate limit check inside agent
    if not check_rate_limit(tenant_id): raise QuotaError()
    # 3. PII masking inside prompt assembly
    sanitized = mask_ssn(user_query)
    # 4. Cache check inside agent
    cached = redis_cache.get(sanitized)
    if cached: return cached
    # 5. Model call
    response = llm.invoke(sanitized)
    # 6. Audit logging mixed in
    siem_log(user_query, response, tenant_id)
    return response""", language="python")

with col_m2:
    st.markdown("#### ✅ With Middleware (Decoupled & Composable)")
    st.code("""# Production Standard: Clean Agent with Composable Middleware
agent = create_agent(model=chat_model, tools=system_tools)

# Middleware stack declared declaratively
governed_agent = MiddlewarePipeline([
    AuthMiddleware(verifier=jwt_verifier),
    RateLimitMiddleware(rate=50, per_second=1),
    PIIMaskingMiddleware(patterns=[SSN_REGEX, EMAIL_REGEX]),
    SemanticCacheMiddleware(vector_store=redis_cache),
    SafetyGuardrailMiddleware(safety_checker=llama_guard),
    AuditLoggingMiddleware(emitter=datadog_tracer)
]).wrap(agent)

# Clean, singular invocation interface
result = governed_agent.invoke({"prompt": user_query})""", language="python")

st.markdown("""
### Middleware vs Callbacks vs LangGraph Subgraphs
""")

comp_df = pd.DataFrame([
    {
        "Dimension": "Primary Objective",
        "Middleware": "Active interception, bidirectional state transformation, short-circuiting",
        "LangChain Callbacks": "Passive observation, event-driven telemetry, and metric streaming",
        "LangGraph Subgraphs": "Complex cognitive branching, hierarchical multi-agent state machines"
    },
    {
        "Dimension": "Can Alter Inputs/Outputs?",
        "Middleware": "Yes (can mutate prompts, inject context, and scrub model outputs)",
        "LangChain Callbacks": "No (read-only observer pattern; cannot mutate payloads)",
        "LangGraph Subgraphs": "Yes (via explicit state channel reducers and node transitions)"
    },
    {
        "Dimension": "Can Abort / Short-Circuit?",
        "Middleware": "Yes (can return cached responses or raise quota exceptions early)",
        "LangChain Callbacks": "No (raising errors crashes the execution unexpectedly)",
        "LangGraph Subgraphs": "Yes (via conditional edges and router functions)"
    },
    {
        "Dimension": "Execution Model",
        "Middleware": "Concentric Onion (FIFO ingress, LIFO egress)",
        "LangChain Callbacks": "Broadcast event handlers (`on_llm_start`, `on_tool_end`)",
        "LangGraph Subgraphs": "Cyclic directed graphs with checkpointed channel states"
    }
])
st.dataframe(comp_df, use_container_width=True)

st.divider()

# --- SECTION 3: WHY MIDDLEWARE IS NEEDED ---
st.header("3. Why Middleware is Needed", divider="rainbow")

st.markdown("""
Modern enterprise AI systems must meet stringent compliance, security, and economic service-level objectives (SLOs). 
Relying on prompting alone to enforce enterprise constraints is unreliable because large language models are non-deterministic.
Middleware provides **deterministic software guardrails** around non-deterministic LLMs.
""")

why_c1, why_c2, why_c3 = st.columns(3)

with why_c1:
    st.subheader("🛡️ Enterprise Defense")
    st.markdown("""
    - **Prompt Injection Neutralization**: Strip malicious delimiters and delimiter escape sequences before the model evaluates them.
    - **PII / Secret Masking**: Guarantee that Social Security Numbers, API keys, and patient identifiers never leave your VPC boundary.
    - **Multi-Tenant Isolation**: Enforce tenant ID verification at the perimeter.
    """)

with why_c2:
    st.subheader("💰 Economic Governance")
    st.markdown("""
    - **Semantic Caching**: Cut API costs by 30-60% by serving recurring queries directly from vector caches without model inference.
    - **Token Rate Limiting**: Throttling runaway conversational loops before monthly billing quotas are exhausted.
    - **Dynamic Fallbacks**: Gracefully downgrade to smaller models when high-tier API budgets are saturated.
    """)

with why_c3:
    st.subheader("⚙️ Observability & Telemetry")
    st.markdown("""
    - **Zero-Touch Distributed Tracing**: Auto-inject W3C trace contexts and span IDs across all LLM and tool calls.
    - **Latency Accounting**: Pinpoint whether latency bottlenecks stem from network DNS, vector retrieval, or model token generation.
    - **Compliance Audit Logs**: Maintain tamper-proof audit trails for SOC2, HIPAA, and GDPR compliance.
    """)

st.divider()

# --- SECTION 4: MIDDLEWARE LIFECYCLE ---
st.header("4. Middleware Lifecycle", divider="rainbow")

st.markdown("""
The middleware lifecycle governs how an input payload travels from the network boundary through intermediate transformations, 
into the agent runtime, and back out to the consumer.

The lifecycle consists of four standardized phases:
1. **Pre-Execution (Ingress)**: Validating headers, parsing authentication tokens, decrypting payloads, and checking caches.
2. **Execution Boundary**: Core model inference or tool dispatching within the agent environment.
3. **Post-Execution (Egress)**: Toxicity filtering, PII unmasking, token usage counting, and output formatting.
4. **Exception Handling**: Intercepting network timeouts, rate limit 429s, or tool execution errors to trigger retries or fallbacks.
""")

lifecycle_dot = """
digraph LifecyclePhases {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4, fontname="Segoe UI"];
    node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
    edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

    Phase1 [label="Phase 1: Ingress (Pre-Execution)\\n• Authenticate User\\n• Check Token Quotas\\n• Sanitize Prompt Input\\n• Check Semantic Cache", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
    Phase2 [label="Phase 2: Execution Boundary\\n• Model Prompt Assembly\\n• LLM Generation\\n• Tool Call Dispatching", fillcolor="#581C87", fontcolor="#FFFFFF", color="#C084FC"];
    Phase3 [label="Phase 3: Egress (Post-Execution)\\n• Content Safety Evaluation\\n• Telemetry & Token Attribution\\n• Output Transformation", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
    Phase4 [label="Phase 4: Fault / Exception Boundary\\n• Catch RateLimit / Timeout Exceptions\\n• Trigger Exponential Backoff\\n• Emit Error Metrics to SIEM", fillcolor="#7F1D1D", fontcolor="#FFFFFF", color="#F87171"];

    Phase1 -> Phase2 [label="Normal Cache Miss"];
    Phase1 -> Phase3 [label="Short-Circuit on Cache Hit", style="dashed", color="#F59E0B"];
    Phase2 -> Phase3 [label="Successful Inference"];
    Phase2 -> Phase4 [label="Model / Tool Failure", color="#EF4444"];
    Phase4 -> Phase2 [label="Retry with Backoff", color="#F59E0B", style="dashed"];
    Phase4 -> Phase3 [label="Fallback Response Generated", color="#94A3B8"];
}
"""
st.graphviz_chart(lifecycle_dot, use_container_width=True)

st.divider()

# --- SECTION 5: MIDDLEWARE EXECUTION ORDER ---
st.header("5. Middleware Execution Order", divider="rainbow")

st.markdown("""
Middleware operates according to the classic **Onion Architecture** (also known as the Russian Matryoshka Doll model).
The execution order follows strict symmetry:
- **Ingress Hook Execution**: Sequential order from outermost layer to innermost layer (First-In, First-Evaluated).
- **Egress Hook Execution**: Reverse order from innermost layer to outermost layer (Last-In, First-Evaluated - LIFO).
""")

order_df = pd.DataFrame([
    {"Layer": "1 (Outer)", "Middleware": "Authentication & Tenant Guard", "Ingress Order": "Step 1 (First)", "Egress Order": "Step 8 (Last)", "Critical Reason": "Must verify credentials before spending compute; must finalize security headers last."},
    {"Layer": "2", "Middleware": "Rate Limiter & Concurrency Throttler", "Ingress Order": "Step 2", "Egress Order": "Step 7", "Critical Reason": "Drop unauthorized requests before running regex PII scanners or vector lookups."},
    {"Layer": "3", "Middleware": "PII Detection & Prompt Sanitizer", "Ingress Order": "Step 3", "Egress Order": "Step 6", "Critical Reason": "Scrub sensitive data before semantic cache vectorization to prevent cache poisoning."},
    {"Layer": "4 (Inner)", "Middleware": "Semantic Vector Cache", "Ingress Order": "Step 4 (Last Ingress)", "Egress Order": "Step 5 (First Egress)", "Critical Reason": "Closest to model: if cache hits, immediately short-circuit without entering model loop."}
])
st.dataframe(order_df, use_container_width=True)

st.warning("""
**Architectural Rule of Thumb**: Security and authentication layers must ALWAYS reside on the outer rim of the onion. 
Placing PII sanitization or caching ahead of authentication introduces security vulnerabilities where unauthenticated actors 
can exploit expensive vector searches or populate caches.
""")

st.divider()

# --- SECTION 6: MIDDLEWARE COMPOSITION ---
st.header("6. Middleware Composition", divider="rainbow")

st.markdown("""
Middleware components should follow the Single Responsibility Principle: each module manages exactly one concern.
Complex governance systems are composed by assembling lightweight middleware classes into a linear pipeline.
""")

st.code("""from typing import List, Callable, Any, Dict

class BaseMiddleware:
    def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

class MiddlewareStack:
    def __init__(self, middlewares: List[BaseMiddleware]):
        self.middlewares = middlewares

    def __call__(self, agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            # 1. Ingress traversal (Outer to Inner)
            for mw in self.middlewares:
                state = mw.before(state)
                # Short-circuit check
                if state.get("short_circuit"):
                    return state

            # 2. Core Agent Invocations
            state = agent_fn(state)

            # 3. Egress traversal (Inner to Outer - LIFO)
            for mw in reversed(self.middlewares):
                state = mw.after(state)

            return state
        return wrapped""", language="python")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Full Lifecycle Pipeline & Execution Flow", divider="rainbow")

st.markdown("""
Below is an enterprise-grade, clean Python implementation of a composable agent middleware engine. 
It supports state mutation, early returns (short-circuiting), error recovery, and bi-directional timing instrumentation.
""")

st.code("""import time
from typing import Dict, List, Any, Optional, Callable

class RequestState:
    \"\"\"Immutable-friendly request state carried through the middleware stack.\"\"\"
    def __init__(self, prompt: str, tenant_id: str):
        self.prompt = prompt
        self.tenant_id = tenant_id
        self.metadata: Dict[str, Any] = {}
        self.response: Optional[str] = None
        self.short_circuited: bool = False
        self.execution_spans: List[Dict[str, Any]] = []

class BaseAgentMiddleware:
    \"\"\"Base contract for all enterprise agent middleware.\"\"\"
    def on_enter(self, state: RequestState) -> RequestState:
        return state

    def on_exit(self, state: RequestState) -> RequestState:
        return state

    def on_error(self, state: RequestState, exc: Exception) -> Optional[RequestState]:
        raise exc

class AuthenticationMiddleware(BaseAgentMiddleware):
    def on_enter(self, state: RequestState) -> RequestState:
        t0 = time.perf_counter()
        if not state.tenant_id or state.tenant_id.strip() == "":
            raise PermissionError("Access Denied: Missing or empty tenant_id header.")
        state.metadata["auth_verified"] = True
        state.execution_spans.append({"layer": "Auth", "duration_ms": (time.perf_counter() - t0) * 1000})
        return state

class PromptSanitizerMiddleware(BaseAgentMiddleware):
    FORBIDDEN_PATTERNS = ["DROP TABLE", "IGNORE PREVIOUS INSTRUCTIONS"]

    def on_enter(self, state: RequestState) -> RequestState:
        t0 = time.perf_counter()
        clean_prompt = state.prompt
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in clean_prompt.upper():
                clean_prompt = clean_prompt.replace(pattern, "[BLOCKED_INJECTION]")
        state.prompt = clean_prompt
        state.execution_spans.append({"layer": "Sanitizer", "duration_ms": (time.perf_counter() - t0) * 1000})
        return state

class SemanticCacheMiddleware(BaseAgentMiddleware):
    def __init__(self, cache_store: Dict[str, str]):
        self.cache_store = cache_store

    def on_enter(self, state: RequestState) -> RequestState:
        t0 = time.perf_counter()
        if state.prompt in self.cache_store:
            state.response = self.cache_store[state.prompt]
            state.short_circuited = True
            state.metadata["cache_hit"] = True
        state.execution_spans.append({"layer": "Cache", "duration_ms": (time.perf_counter() - t0) * 1000})
        return state

class EnterpriseMiddlewareEngine:
    \"\"\"Manages symmetric onion execution across multiple middleware components.\"\"\"
    def __init__(self, layers: List[BaseAgentMiddleware]):
        self.layers = layers

    def run(self, state: RequestState, core_agent_fn: Callable[[RequestState], RequestState]) -> RequestState:
        # Ingress (FIFO)
        for layer in self.layers:
            state = layer.on_enter(state)
            if state.short_circuited:
                return state  # Skip core model call and egress

        # Core Agent Loop
        t_model = time.perf_counter()
        state = core_agent_fn(state)
        state.execution_spans.append({"layer": "CoreModel", "duration_ms": (time.perf_counter() - t_model) * 1000})

        # Egress (LIFO)
        for layer in reversed(self.layers):
            state = layer.on_exit(state)

        return state""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Middleware Onion & Lifecycle Simulator", divider="rainbow")

st.markdown("""
Interact with the live simulation engine below to observe how different middleware layers process prompts, 
sanitize security threats, intercept requests for cache hits, and execute in symmetric Onion order.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Configure Middleware Pipeline")
    wb_tenant = st.text_input("Tenant Identification ID", value="enterprise-corp-01")
    wb_prompt = st.text_area(
        "Simulated User Input Prompt",
        value="Please summarize account history for user@corp.com with SSN 123-45-6789."
    )

    st.markdown("#### Active Onion Layers:")
    layer_auth = st.toggle("Layer 1: Auth & Tenant Guard", value=True)
    layer_pii = st.toggle("Layer 2: PII & Masking Filter", value=True)
    layer_cache = st.toggle("Layer 3: Semantic Cache (Simulate Cache Hit)", value=False)
    layer_safety = st.toggle("Layer 4: Safety Guardrails", value=True)
    layer_audit = st.toggle("Layer 5: Audit & Metric Logger", value=True)

with col_wb2:
    st.subheader("📊 Pipeline Execution Status")
    st.info(f"""
    **Current Configuration:**
    - **Tenant**: `{wb_tenant}`
    - **Cache Mode**: `{'⚡ Cache Hit (Will Short-Circuit)' if layer_cache else 'Inference Mode (Cache Miss)'}`
    - **Active Interceptors**: `{sum([layer_auth, layer_pii, layer_cache, layer_safety, layer_audit])} / 5`
    """)

    btn_run = st.button("🚀 Execute Middleware Pipeline", type="primary", use_container_width=True)

if btn_run:
    # Build active layers
    sim_layers = [
        MockMiddlewareLayer("Auth & Tenant Guard", enabled=layer_auth, latency_ms=8.0),
        MockMiddlewareLayer("PII & Masking Filter", enabled=layer_pii, latency_ms=14.0),
        MockMiddlewareLayer("Semantic Cache", enabled=True, early_return=layer_cache, latency_ms=6.0),
        MockMiddlewareLayer("Safety Guardrails", enabled=layer_safety, latency_ms=22.0),
        MockMiddlewareLayer("Audit & Metric Logger", enabled=layer_audit, latency_ms=9.0)
    ]

    pipeline = MockMiddlewarePipelineSimulator(sim_layers)
    result = pipeline.execute(wb_prompt, wb_tenant)

    st.markdown("### 🎯 Final Execution Output:")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Latency", f"{result['total_latency_ms']:.1f} ms")
    m2.metric("Tokens Consumed", result.get("total_tokens", 0))
    m3.metric("Execution Mode", "⚡ Short-Circuited" if result["early_returned"] else "Full Model Loop")

    st.markdown("#### 💬 Synthesized Response Payload:")
    st.success(result["response"])

    st.markdown("#### 📜 Granular Onion Trace Log (Chronological):")
    for step in result["trace_log"]:
        st.write(f"- `{step}`")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the architectural difference between LangChain Callbacks and Middleware?",
     "LangChain Callbacks follow the Observer pattern: they are passive, read-only event handlers (e.g. on_llm_start, on_tool_end) designed primarily for logging and telemetry. Callbacks cannot mutate input prompts or model outputs. In contrast, Middleware follows the Interceptor/Decorator pattern: it can mutate payloads bidirectionally, halt execution, return early, and handle retries."),
    ("Explain the Onion Architecture execution order in middleware.",
     "The Onion Architecture executes in symmetric reverse order. On ingress (request inbound), layers execute from outer to inner (FIFO: Layer 1, Layer 2, Layer 3 -> Model). On egress (response outbound), layers execute in reverse from inner to outer (LIFO: Model -> Layer 3, Layer 2, Layer 1). This ensures that outer concerns like authentication wrap and protect all inner logic."),
    ("How does short-circuiting work in agent middleware, and when should you use it?",
     "Short-circuiting occurs when an ingress middleware layer satisfies the request or detects an invalid state before reaching the core model. Examples include semantic vector cache hits, rate-limit quota exhaustion, or prompt injection blocks. When triggered, the middleware immediately returns a response or raises an exception, skipping downstream layers and eliminating costly model inference."),
    ("Why should authentication middleware execute before PII masking or vector caching?",
     "If caching or PII masking executes before authentication, unauthenticated attackers could exploit the system to populate the cache with malicious data (cache poisoning) or exhaust CPU and vector database compute without valid credentials. Security perimeters must always be outermost."),
    ("How do you handle asynchronous streaming when using egress middleware?",
     "When an agent streams tokens via an async generator, traditional egress middleware that expects a complete string will break the stream. Streaming-aware middleware must wrap the token generator itself, inspecting chunks in sliding windows or buffering until safety thresholds are validated before yielding tokens to the client.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Inverting Layer Order (Placing Cache Ahead of Auth). Exposing semantic vector cache lookups prior to JWT token validation allows unauthenticated users to fish for cached corporate secrets. Always enforce Auth as Layer 1.")
    st.error("**Trap 2 (Breaking Error)**: Mutating State In-Place Without Deep Copies. When concurrent async tasks run through middleware that mutates shared dictionary keys directly, cross-request state contamination occurs. Always return modified copies or scoped request contexts.")
with trap_c2:
    st.warning("**Trap 3 (Latency Warning)**: Heavy Synchronous Regex / LLM Scanners in Every Middleware Layer. Chaining multiple auxiliary LLM calls (e.g. one for safety, one for PII, one for intent) adds 1.5+ seconds of latency to every interaction. Consolidate safety checks or use compiled fast-path regex.")
    st.warning("**Trap 4 (Cost Warning)**: Unbounded Cache Growth Without TTL. Storing every prompt-response pair in a semantic cache indefinitely causes storage costs to explode and serves stale information. Enforce TTL eviction and cosine similarity thresholds.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Middleware Pattern"):
        st.markdown("**Pluggable interceptor pipeline surrounding the agent core for cross-cutting governance.**")
    with st.expander("🎴 Flashcard 2: Onion Execution"):
        st.markdown("**FIFO ingress (outer to inner) followed by LIFO egress (inner to outer).**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Short-Circuiting"):
        st.markdown("**Early return from ingress (e.g. cache hit or rate limit) skipping core model inference.**")
    with st.expander("🎴 Flashcard 4: Callbacks vs Middleware"):
        st.markdown("**Callbacks are passive read-only observers; Middleware is active bi-directional mutator.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural definition and purpose of the Agent Middleware Onion.", value=True)
chk2 = st.checkbox("I can articulate the key differences between Callbacks, Middleware, and LangGraph Subgraphs.", value=True)
chk3 = st.checkbox("I understand the four phases of the middleware lifecycle and how state flows through them.", value=True)
chk4 = st.checkbox("I know how to design symmetric FIFO ingress and LIFO egress execution orders.", value=True)
chk5 = st.checkbox("I can implement short-circuiting mechanisms for semantic caching and rate limiting.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 middleware fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 11.1 Middleware Fundamentals!")
