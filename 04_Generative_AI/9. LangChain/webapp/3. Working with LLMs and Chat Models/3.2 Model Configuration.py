import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="3.2 Model Configuration - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("3.2 Model Configuration: Hyperparameters, Timeouts, Streaming & Dynamic Selection")
st.caption("Chapter 3: Working with LLMs and Chat Models | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Configuring Chat Models for production requires much more than specifying a model name and temperature. 
Enterprise applications demand rigorous controls over **timeout deadlines**, **exponential backoff retries**, 
**token ceilings**, **streaming chunk dynamics**, and **runtime parameter mutability**.

Through LangChain v0.3+, developers configure models statically during initialization, override them 
dynamically using `.bind()`, or reconfigure them per-request using `RunnableConfig` and `configurable_fields`.

In this lesson, you will master:
- **Core Hyperparameters**: Model checkpoint targeting, temperature sampling, and token generation limits.
- **Latency & Resilience Guardrails**: Client-side timeout deadlines and exponential retry policies.
- **Progressive Delivery**: Token streaming mechanics and Time-To-First-Token (TTFT) optimization.
- **Transport & Security Controls**: API keys, custom base URLs, proxy routing, and TLS certificates.
- **Dynamic Model Selection**: Switching models and parameters at runtime without modifying application code.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockConfiguredChatModel:
    """Simulates runtime model configuration, latency timeouts, and streaming token generation in module scope."""
    def __init__(self, model_name: str, temperature: float, max_tokens: int, timeout: float, max_retries: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def execute_with_telemetry(self, prompt_text: str, simulate_delay: float = 0.5) -> Dict[str, Any]:
        """Simulates an invocation with timeout detection and retry accounting."""
        # Check for simulated timeout
        timed_out = simulate_delay > self.timeout
        retries_attempted = 0
        success = not timed_out
        
        if timed_out and self.max_retries > 0:
            # Simulate retry recovery if max_retries is enabled
            retries_attempted = min(self.max_retries, 2)
            success = True
            effective_latency = int((self.timeout * 1000) + (simulate_delay * 1000))
        elif timed_out:
            effective_latency = int(self.timeout * 1000)
        else:
            effective_latency = int(simulate_delay * 1000)

        # Truncate content based on max_tokens
        simulated_response = (
            f"Configured execution for '{prompt_text}'. "
            f"Model '{self.model_name}' generated answer with temperature {self.temperature:.2f}. "
            f"Parameters: max_tokens={self.max_tokens}, timeout={self.timeout}s, max_retries={self.max_retries}."
        )
        words = simulated_response.split()
        est_tokens = len(words) * 2
        truncated = False
        if est_tokens > self.max_tokens:
            words = words[:max(2, self.max_tokens // 2)]
            simulated_response = " ".join(words) + " [TRUNCATED_MAX_TOKENS]"
            truncated = True

        return {
            "success": success,
            "timed_out": timed_out,
            "retries_attempted": retries_attempted,
            "effective_latency_ms": effective_latency,
            "truncated": truncated,
            "finish_reason": "length" if truncated else ("stop" if success else "timeout_failure"),
            "content": simulated_response if success else "Error: Request exceeded configured timeout threshold.",
            "usage": {
                "input_tokens": len(prompt_text.split()) * 2,
                "output_tokens": len(words) * 2,
                "total_tokens": (len(prompt_text.split()) * 2) + (len(words) * 2)
            }
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Chat Model Configuration Topology & Parameter Lifecycle", divider="rainbow")

st.markdown("""
The diagram below illustrates how configuration flows through a Chat Model: from static initialization 
through dynamic runtime binding, transport resilience policies, and progressive streaming emission.
""")

dot_blueprint = """
digraph ModelConfigTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_init {
        label="1. Model Initialization Configuration";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_static [label="ChatOpenAI(model='gpt-4o', temperature=0.2, ...)\nStatic baseline client", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_runtime {
        label="2. Runtime Dynamic Overrides";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_bind [label="model.bind(temperature=0.0, stop=['DONE'])\nCreates immutable RunnableBinding", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_config [label="RunnableConfig (per-request)\nconfig={'configurable': {'model': 'claude-3-5'}}", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_static -> node_bind [label=".bind()", color="#38BDF8"];
        node_static -> node_config [label="configurable_fields()", color="#38BDF8"];
    }

    subgraph cluster_transport {
        label="3. Network & Resilience Guardrails";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_timeout [label="Client Timeout (e.g., timeout=12.0s)\nPrevents hanging threads", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_retries [label="Exponential Backoff (max_retries=3)\nAutomatic recovery on HTTP 429/503", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_transport [label="Transport Layer (HTTPX Client)\nBase URL, Proxy, SSL & Custom Headers", fillcolor="#1E293B", fontcolor="#F59E0B"];

        node_bind -> node_timeout;
        node_config -> node_timeout;
        node_timeout -> node_retries;
        node_retries -> node_transport;
    }

    subgraph cluster_emission {
        label="4. Generation & Streaming Emission";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_chunks [label="AIMessageChunk Generator\nProgressive token output stream", fillcolor="#1E293B", fontcolor="#34D399"];
        node_aimsg [label="Final Standardized AIMessage\nusage_metadata + response_metadata", fillcolor="#1E293B", fontcolor="#34D399"];

        node_transport -> node_chunks [label="stream=True", color="#34D399"];
        node_transport -> node_aimsg [label="stream=False", color="#34D399"];
    }
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: LangChain uses **immutable Runnables**. When you call `.bind()` or attach 
> timeouts and stop sequences, you are not mutating the underlying client in place; you are returning 
> a new, thread-safe `RunnableBinding`. This allows a single model connection pool to be safely shared 
> across thousands of concurrent asynchronous requests.
""")

# --- SECTION 2: MODEL NAME ---
st.header("2. Model Name: Checkpoint Targeting, Context Dimensions & Alias Resolution", divider="rainbow")

st.markdown("""
The `model` (or `model_name`) parameter specifies the exact foundation model checkpoint hosted by the provider. 
In enterprise systems, checkpoint selection governs intelligence tier, token context horizons, and unit economics.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("""
    #### Pinned Checkpoints vs Aliases
    - **Dynamic Aliases** (e.g., `gpt-4o`, `claude-3-5-sonnet-latest`): Convenient for development, but risky in production because the provider may update the model behavior without notice.
    - **Pinned Checkpoints** (e.g., `gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`): Critical for production regression testing, deterministic evaluations, and change control.
    """)
with col_m2:
    st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Pinned production checkpoints for auditability
openai_prod = ChatOpenAI(model="gpt-4o-2024-08-06")
anthropic_prod = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Inspecting configured model
print(openai_prod.model_name)
''', language="python")

# --- SECTION 3: TEMPERATURE ---
st.header("3. Temperature: Stochastic Variance, Top-P Nucleus Sampling & Hallucination Guardrails", divider="rainbow")

st.markdown("""
The `temperature` parameter scales token logits prior to Softmax probability normalization.
In production systems, choosing temperature is an architectural decision based on the task:

1. **`temperature = 0.0` (Deterministic / Greedy Decoding)**:
   - Always selects the argmax token.
   - Essential for **SQL generation**, **JSON schema extraction**, **math**, and **classification**.
2. **`temperature = 0.2 to 0.4` (Focused Reasoning)**:
   - Highly consistent while avoiding repetitive token loops.
   - Recommended for **RAG synthesis** and **summarization**.
3. **`temperature = 0.7 to 1.0` (Exploratory / Creative)**:
   - Encourages vocabulary diversity.
   - Recommended for **brainstorming** and **creative writing**.
""")

st.code('''from langchain_openai import ChatOpenAI

# Deterministic configuration for structured extraction
extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Exploratory configuration for synthetic scenario generation
brainstorm_llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
''', language="python")

# --- SECTION 4: TOKEN LIMITS ---
st.header("4. Token Limits: max_tokens, max_completion_tokens, Context Ceilings & Cost Caps", divider="rainbow")

st.markdown("""
Controlling token generation boundaries is the primary defense against run-away costs and infinite generation loops.

In LangChain:
- **`max_tokens`**: The standard parameter across all providers specifying the upper bound of generated tokens.
- **`max_completion_tokens`**: OpenAI's newer parameter (specifically required for `o1` and `o3-mini` models to account for reasoning tokens).
""")

st.code('''from langchain_openai import ChatOpenAI

# Enforcing strict 500-token completion ceiling
bounded_model = ChatOpenAI(
    model="gpt-4o",
    max_tokens=500,
    temperature=0.2
)

# For OpenAI o1 / o3-mini reasoning models:
reasoning_bounded = ChatOpenAI(
    model="o3-mini",
    max_completion_tokens=1500  # Covers both internal reasoning + visible response
)
''', language="python")

# --- SECTION 5: STREAMING ---
st.header("5. Streaming: Progressive Token Iteration, astream, Chunk Arithmetic & TTFT Optimization", divider="rainbow")

st.markdown("""
Foundation model responses can take several seconds to generate in full. Rather than forcing end-users 
to stare at a blank screen, **streaming** pushes token chunks over HTTP chunked transfer encoding (or Server-Sent Events) 
as they are synthesized.

LangChain provides `.stream()` (sync) and `.astream()` (async), yielding `AIMessageChunk` objects that 
support progressive token concatenation.
""")

st.code('''from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 1. Synchronous Streaming
for chunk in model.stream("Outline 3 benefits of event-driven architectures:"):
    print(chunk.content, end="", flush=True)

# 2. Asynchronous Streaming for FastAPI / WebSocket endpoints:
async def stream_to_client(query: str):
    async for chunk in model.astream(query):
        yield chunk.content
''', language="python")

# --- SECTION 6: TIMEOUTS ---
st.header("6. Timeouts: Client-Side Timeout Deadlines, Gateway Latency & Hanging Request Mitigation", divider="rainbow")

st.markdown("""
By default, standard HTTP clients may wait up to 60 or 120 seconds before aborting a stalled network connection. 
In high-throughput enterprise microservices, an unhandled network stall exhausts thread pools and crashes web servers.

LangChain allows configuring **`timeout`** (in seconds) directly on model initialization or per-call via `RunnableConfig`.
""")

st.code('''from langchain_openai import ChatOpenAI

# Setting strict 10.0 second client-side deadline
fast_timeout_model = ChatOpenAI(
    model="gpt-4o",
    timeout=10.0  # Raises httpx.TimeoutException if provider does not respond within 10s
)

# Overriding timeout per invocation via config:
# response = fast_timeout_model.invoke("Quick query", config={"timeout": 5.0})
''', language="python")

# --- SECTION 7: RETRIES ---
st.header("7. Retries: Exponential Backoff, Jitter, HTTP 429 Interception & max_retries", divider="rainbow")

st.markdown("""
Frontier APIs regularly throw transient errors: HTTP 429 (Rate Limit Exceeded), HTTP 500 (Internal Server Error), 
and HTTP 503 (Service Unavailable). 

LangChain's provider integrations automatically implement **exponential backoff with randomized jitter** 
controlled by **`max_retries`**.
""")

dot_retry_flow = """
digraph RetryBackoffFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_req [label="Model Invocation Attempt 1", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_err [label="HTTP 429 Rate Limit Intercepted", fillcolor="#1E293B", fontcolor="#EF4444"];
    node_wait1 [label="Exponential Backoff Sleep (e.g., 1.5s + jitter)", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_req2 [label="Model Invocation Attempt 2", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_success [label="HTTP 200 Success: AIMessage returned", fillcolor="#1E293B", fontcolor="#34D399"];

    node_req -> node_err [label="transient error"];
    node_err -> node_wait1;
    node_wait1 -> node_req2 [label="retry 1 of max_retries"];
    node_req2 -> node_success;
}
"""

with st.expander("🔍 Contextual Diagram: Exponential Backoff & Retry Interception", expanded=True):
    st.graphviz_chart(dot_retry_flow, use_container_width=True)

st.code('''from langchain_openai import ChatOpenAI

# Setting max retries for resilient autonomous background jobs
reliable_model = ChatOpenAI(
    model="gpt-4o",
    max_retries=3,  # Automatically retries up to 3 times on transient 429/500 errors
    timeout=15.0
)
''', language="python")

# --- SECTION 8: API CONFIGURATION ---
st.header("8. API Configuration: Custom Base URLs, Proxies, Authentication Headers & TLS Settings", divider="rainbow")

st.markdown("""
Enterprise networks often require routing all outbound HTTP traffic through corporate proxy servers, 
internal API gateways (e.g., LiteLLM, Kong, or Portkey), or custom endpoints.

LangChain models provide direct arguments to configure underlying HTTP transports:
- **`base_url`** / `openai_api_base`: Redirect traffic to internal proxy or local vLLM server.
- **`http_client`**: Pass custom `httpx.Client` with custom TLS certificates or corporate proxies.
- **`api_key`**: Explicitly supply keys instead of relying on ambient environment variables.
""")

st.code('''from langchain_openai import ChatOpenAI
import httpx

# Custom internal gateway routing with corporate proxy
custom_client = httpx.Client(
    proxy="http://corp-proxy.internal:8080",
    verify=True,
    timeout=20.0
)

gateway_model = ChatOpenAI(
    model="gpt-4o",
    base_url="https://ai-gateway.company.internal/v1",  # Corporate gateway
    api_key="internal-gateway-token",
    http_client=custom_client
)
''', language="python")

# --- SECTION 9: PROVIDER CONFIGURATION ---
st.header("9. Provider Configuration: model_kwargs, Backend Flags & Vendor-Specific Extensions", divider="rainbow")

st.markdown("""
When providers release brand-new flags that have not yet been standardized into LangChain Core, 
you pass them directly into the underlying API call using **`model_kwargs`**.
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("""
    #### Common `model_kwargs` Use Cases
    - **OpenAI**: `seed` (deterministic RNG seed), `logprobs` (token probabilities), `user` (end-user tracking string).
    - **Anthropic**: `extra_headers` (beta feature flags like prompt caching).
    - **Google Gemini**: `safety_settings`, `top_k`.
    """)
with col_p2:
    st.code('''from langchain_openai import ChatOpenAI

# Passing provider-specific parameters
specialized_model = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={
        "seed": 1337,
        "logprobs": True,
        "top_logprobs": 2,
        "user": "session-user-9821"
    }
)
''', language="python")

# --- SECTION 10: DYNAMIC MODEL SELECTION ---
st.header("10. Dynamic Model Selection: configurable_fields, Runtime Overrides & Cost-Based Routing", divider="rainbow")

st.markdown("""
In multi-tenant or cost-conscious applications, you do not want to hardcode model choices. 
LangChain provides **`configurable_fields`**, allowing pipelines to dynamically swap models, temperatures, 
or endpoints per request using `RunnableConfig`.
""")

st.code('''from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Create configurable chat model
configurable_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).configurable_fields(
    model_name=ConfigurableField(
        id="llm_model",
        name="Model Checkpoint",
        description="The active foundation model checkpoint"
    ),
    temperature=ConfigurableField(
        id="llm_temperature",
        name="Temperature",
        description="Sampling stochasticity"
    )
)

# Runtime invocation overriding model and temperature on the fly:
res = configurable_model.invoke(
    "Perform enterprise system architecture review",
    config={
        "configurable": {
            "llm_model": "gpt-4o",
            "llm_temperature": 0.0
        }
    }
)
''', language="python")

# --- SECTION 11: PRODUCTION IMPLEMENTATION ---
st.header("11. Production Implementation: Enterprise Resilient Client Configuration Pipeline", divider="rainbow")

st.markdown("""
Below is an enterprise-grade production client factory that enforces standardized configurations: 
pinned model checkpoints, conservative timeout ceilings, exponential backoff retries, and dynamic runtime binding.
""")

st.code('''from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI

class EnterpriseModelConfigFactory:
    """Production factory for instantiating hardened, auditable ChatModel pipelines."""

    @staticmethod
    def create_hardened_client(
        api_key: str,
        pinned_model: str = "gpt-4o-2024-08-06",
        timeout_seconds: float = 12.0,
        max_retries: int = 3,
        base_url: Optional[str] = None
    ) -> ChatOpenAI:
        """Constructs a thread-safe ChatOpenAI instance with enterprise guardrails."""
        return ChatOpenAI(
            model=pinned_model,
            temperature=0.1,
            max_tokens=2048,
            timeout=timeout_seconds,
            max_retries=max_retries,
            base_url=base_url,
            api_key=api_key,
            model_kwargs={"seed": 42}
        )

    @staticmethod
    def bind_request_parameters(
        client: ChatOpenAI,
        temperature: float,
        stop_sequences: Optional[List[str]] = None
    ) -> RunnableBinding:
        """Dynamically binds request parameters without mutating base client."""
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        return client.bind(**kwargs)

# Usage Example:
# client = EnterpriseModelConfigFactory.create_hardened_client(api_key="sk-...")
# bound_runnable = EnterpriseModelConfigFactory.bind_request_parameters(client, temperature=0.0, stop_sequences=["\\n\\n"])
# response = bound_runnable.invoke([HumanMessage(content="Audit system architecture")])
''', language="python")

# --- SECTION 12: INTERACTIVE WORKBENCH ---
st.header("12. Interactive Workbench: Interactive Hyperparameter Simulator & Dynamic Model Switcher", divider="rainbow")

st.markdown("""
Test hyperparameter configurations in the live simulator below. Adjust **Timeout ceilings**, 
**Max Retries**, **Max Completion Tokens**, and observe how the client handles simulated network delays, 
retry recovery, and token truncations.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    model_choice = st.selectbox(
        "Model Checkpoint",
        ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "claude-3-5-sonnet-20241022"],
        index=0
    )
    temp_val = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.1, step=0.05)
    max_tok_val = st.slider("Max Tokens Ceiling", min_value=10, max_value=120, value=40, step=5)

with col_wb2:
    timeout_val = st.slider("Client Timeout (seconds)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    retries_val = st.slider("Max Retries", min_value=0, max_value=3, value=2, step=1)
    sim_delay = st.slider("Simulate Network Latency Delay (seconds)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
    test_prompt = st.text_input("Test Prompt", value="Define idempotent API design in distributed systems.")

if st.button("🚀 Execute Configured Model Invocator", type="primary"):
    sim_model = MockConfiguredChatModel(
        model_name=model_choice,
        temperature=temp_val,
        max_tokens=max_tok_val,
        timeout=timeout_val,
        max_retries=retries_val
    )
    
    result = sim_model.execute_with_telemetry(test_prompt, simulate_delay=sim_delay)
    
    st.subheader("📊 Execution Diagnostics & Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", "SUCCESS" if result["success"] else "FAILED")
    m2.metric("Latency", f"{result['effective_latency_ms']} ms")
    m3.metric("Retries Used", result["retries_attempted"])
    m4.metric("Finish Reason", result["finish_reason"])

    if result["timed_out"] and result["retries_attempted"] > 0:
        st.warning(f"⚠️ Initial request exceeded {timeout_val}s timeout, but recovered via retry mechanism ({result['retries_attempted']} retries).")
    elif result["timed_out"]:
        st.error(f"❌ Execution failed: Network delay ({sim_delay}s) exceeded configured client timeout ({timeout_val}s) with 0 retries remaining.")

    if result["truncated"]:
        st.info(f"ℹ️ Output was bounded by max_tokens={max_tok_val} ceiling.")

    st.write("### Model Response Payload")
    st.code(result["content"], language="markdown")

# --- SECTION 13: TECHNICAL INTERVIEW PREPARATION ---
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is model.bind() preferred over mutating client instance attributes in production?",
     "model.bind() is a purely functional operation that creates a new immutable RunnableBinding without altering the shared base client. In multithreaded or asynchronous web frameworks (like FastAPI), mutating shared instance attributes leads to catastrophic race conditions where one request's parameters leak into another."),
    ("What is the difference between max_tokens and max_completion_tokens?",
     "max_tokens is the standard parameter across legacy and current providers for limiting output tokens. Newer reasoning models (such as OpenAI o1 and o3-mini) use max_completion_tokens, which governs the sum of both invisible chain-of-thought reasoning tokens and visible completion tokens."),
    ("How does LangChain handle transient HTTP 429 errors under the hood?",
     "LangChain's provider integrations use exponential backoff with randomized jitter. When an HTTP 429 (Rate Limit) or 503 response is intercepted, the client sleeps for 2^attempt + jitter seconds before retrying, up to the configured max_retries ceiling."),
    ("Why should production systems pin exact model checkpoints instead of using alias names?",
     "Aliases like 'gpt-4o' or 'claude-3-5-sonnet' are dynamically updated by model providers when new model revisions are rolled out. Unpinned aliases can silently alter output formats, degrade prompt adherence, or break structured JSON parsers. Pinned checkpoints (e.g., 'gpt-4o-2024-08-06') guarantee behavioral immutability."),
    ("How do configurable_fields enable multi-tenant personalization in LangChain applications?",
     "configurable_fields attach metadata IDs to model parameters, allowing downstream callers to override attributes like model checkpoint, temperature, or system prompt on a per-request basis via RunnableConfig dictionaries without rebuilding the pipeline DAG.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Race Conditions from Shared Client Mutation. Directly altering `client.temperature = 0.8` on a global singleton client mutates state across all concurrent async requests. Always use `client.bind(temperature=0.8)`.")
    st.error("**Trap 2 (Breaking Error)**: Unbounded Timeouts in Microservices. Leaving default client timeouts (60–120 seconds) in an upstream API gateway will exhaust thread pools during downstream provider outages. Always set `timeout=10.0` or `15.0`.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Low max_tokens Truncating Pydantic JSON. Setting `max_tokens=100` on a model bound to `.with_structured_output()` truncates the closing JSON brackets, causing Pydantic validation exceptions. Ensure token ceilings comfortably accommodate schema length.")
    st.warning("**Trap 4 (Quality Warning)**: High Temperature on Structured Output Chains. Setting `temperature > 0.3` for JSON extraction increases syntax failure and schema violation rates. Always set `temperature=0.0` for structured extraction.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Immutable Parameter Binding"):
        st.markdown("**model.bind(**kwargs) returns an immutable RunnableBinding, ensuring thread safety.**")
    with st.expander("🎴 Flashcard 2: Timeout Configuration"):
        st.markdown("**timeout=12.0 raises an httpx.TimeoutException if the provider fails to respond in time.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Retry Backoff Policy"):
        st.markdown("**max_retries=3 triggers exponential backoff with jitter on transient 429/500 errors.**")
    with st.expander("🎴 Flashcard 4: Dynamic Reconfiguration"):
        st.markdown("**configurable_fields() enables per-request parameter overrides via RunnableConfig.**")

st.divider()

# --- SECTION 14: SELF-ASSESSMENT CHECKLIST ---
st.header("14. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why pinning model checkpoints is mandatory for enterprise production.", value=True)
chk2 = st.checkbox("I know how to select the right temperature for deterministic vs creative tasks.", value=True)
chk3 = st.checkbox("I understand the difference between max_tokens and max_completion_tokens.", value=True)
chk4 = st.checkbox("I can implement timeouts and exponential retry policies to harden microservices.", value=True)
chk5 = st.checkbox("I know how to use .bind() and configurable_fields for dynamic model reconfiguration.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 3.2 Model Configuration. Ready for 3.3 Model Capabilities!")
