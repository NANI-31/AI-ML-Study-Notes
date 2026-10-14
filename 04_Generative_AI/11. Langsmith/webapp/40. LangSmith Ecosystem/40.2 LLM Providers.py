import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="40.2 LLM Providers - LangSmith Ecosystem",
    page_icon="🤖",
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
    .provider-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-cost {
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
class LLMProvidersEngine:
    """
    Simulates multi-provider telemetry tracking in LangSmith:
    OpenAI, Anthropic, Google Gemini, Open-Source (vLLM/Groq), and Local (Ollama).
    """
    def __init__(self):
        self.providers_catalog = {
            "OpenAI (GPT-4o)": {
                "category": "Commercial Cloud",
                "context_window": 128000,
                "prompt_cost_per_m": 2.50,
                "completion_cost_per_m": 10.00,
                "avg_ttft_ms": 320,
                "strengths": "Structured output JSON schemas, multi-modal vision, function calling accuracy."
            },
            "Anthropic (Claude 3.5 Sonnet)": {
                "category": "Commercial Cloud",
                "context_window": 200000,
                "prompt_cost_per_m": 3.00,
                "completion_cost_per_m": 15.00,
                "avg_ttft_ms": 380,
                "strengths": "Advanced coding synthesis, nuanced reasoning, prompt caching (90% discount on cache hits)."
            },
            "Google (Gemini 1.5 Pro)": {
                "category": "Commercial Cloud",
                "context_window": 2000000,
                "prompt_cost_per_m": 1.25,
                "completion_cost_per_m": 5.00,
                "avg_ttft_ms": 490,
                "strengths": "Massive 2M token context, native audio/video multimodal ingestion, cost-efficiency."
            },
            "Open-Source (Groq / Llama-3.1-70B)": {
                "category": "Hosted Open Source",
                "context_window": 128000,
                "prompt_cost_per_m": 0.59,
                "completion_cost_per_m": 0.79,
                "avg_ttft_ms": 110,
                "strengths": "Ultra-low latency inference (300+ tokens/sec), open weights, zero vendor lock-in."
            },
            "Local (Ollama / Llama-3.2-3B)": {
                "category": "Local / Air-Gapped",
                "context_window": 32000,
                "prompt_cost_per_m": 0.00,
                "completion_cost_per_m": 0.00,
                "avg_ttft_ms": 180,
                "strengths": "Zero data egress, air-gapped compliance, local workstation inference."
            }
        }

    def simulate_provider_run(self, provider_key: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        info = self.providers_catalog.get(provider_key, self.providers_catalog["OpenAI (GPT-4o)"])
        p_cost = (prompt_tokens / 1_000_000) * info["prompt_cost_per_m"]
        c_cost = (completion_tokens / 1_000_000) * info["completion_cost_per_m"]
        total_cost = round(p_cost + c_cost, 6)
        
        simulated_latency = info["avg_ttft_ms"] + int(completion_tokens * 8.5)
        return {
            "provider": provider_key,
            "category": info["category"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": total_cost,
            "latency_ms": simulated_latency,
            "ttft_ms": info["avg_ttft_ms"]
        }

# Instantiate engine
providers_engine = LLMProvidersEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Multi-Provider LLM Orchestration, Tracing Adapters, & Fallback Routing", divider="rainbow")

st.markdown("""
Enterprise AI architectures rarely depend on a single model provider. Production systems utilize 
**multi-provider ecosystems** to balance cost, latency, context length, and compliance. 
LangSmith acts as the universal telemetry adapter, normalizing token accounting, cost metrics, and error payloads 
across commercial providers, hosted inference APIs, and self-hosted local models.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LLM_Providers_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_router {
            label = "Application Router & Fallback Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            app_client [label="Application Request\\n(User Query + Metadata)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            model_router [label="Dynamic Model Router\\n(Routes by cost, task complexity, rate limits)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            fallback_handler [label="Provider Fallback Circuit Breaker\\n(OpenAI -> Anthropic -> Groq on HTTP 429)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_providers {
            label = "Diverse LLM Execution Endpoints";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            p_openai [label="OpenAI API\\n(GPT-4o, GPT-4o-mini)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            p_anthropic [label="Anthropic Claude API\\n(Claude 3.5 Sonnet / Haiku)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            p_gemini [label="Google Gemini API\\n(Gemini 1.5 Pro / Flash)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            p_opensource [label="Hosted OSS Endpoints\\n(vLLM, Groq, Together, DeepSeek)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
            p_local [label="Local / Edge Inference\\n(Ollama, Llama.cpp, Air-gapped)", fillcolor="#0F172A", fontcolor="#CBD5E1"];
        }

        subgraph cluster_langsmith {
            label = "LangSmith Normalization & Telemetry Engine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            token_normalizer [label="Token & Cost Normalizer\\n(Maps provider-specific usage to dollar catalog)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            trace_dag [label="Unified Trace & Error Log\\n(Captures TTFT, latencies, provider rate limits)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        app_client -> model_router;
        model_router -> fallback_handler;

        fallback_handler -> p_openai [label="primary route"];
        fallback_handler -> p_anthropic [label="failover route"];
        fallback_handler -> p_gemini [label="long-context route"];
        fallback_handler -> p_opensource [label="low-latency route"];
        fallback_handler -> p_local [label="air-gapped route"];

        p_openai -> token_normalizer [label="usage metadata"];
        p_anthropic -> token_normalizer [label="usage metadata"];
        p_gemini -> token_normalizer [label="usage metadata"];
        p_opensource -> token_normalizer [label="usage metadata"];
        p_local -> token_normalizer [label="@traceable span"];

        token_normalizer -> trace_dag [label="records uniform telemetry"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Universal Telemetry Standard:</strong> Regardless of whether an inference call goes to OpenAI, Anthropic, 
    Google Gemini, or a self-hosted Ollama server on localhost, LangSmith normalizes prompt tokens, completion tokens, 
    time to first token, and error payloads into a single uniform trace schema.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: OPENAI
# ==============================================================================
st.header("2. OpenAI: GPT-4o, Structured Outputs, and LangSmith Token/Cost Telemetry", divider="rainbow")

st.markdown("""
**OpenAI** remains a cornerstone of enterprise generative AI, offering high-throughput models like `gpt-4o` 
and cost-efficient variants like `gpt-4o-mini`. LangSmith natively integrates with `langchain-openai` and 
the raw OpenAI Python SDK.
""")

st.markdown("""
### Key Features & Observability Integration
1. **Strict Structured Outputs**:
   OpenAI guarantees 100% adherence to Pydantic JSON schemas via constrained decoding (`response_format={"type": "json_schema"}`). 
   In LangSmith, schema parsing errors are captured directly in the LLM run span.
2. **Native Tool Calling**:
   Tool invocation arguments and function names are parsed into distinct child spans in the trace tree.
3. **Usage Metadata Extraction**:
   LangSmith extracts `prompt_tokens`, `completion_tokens`, and `total_tokens` directly from the raw completion response 
   and calculates real-time dollar cost using its pricing catalog.
""")

st.code('''from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class AnalysisReport(BaseModel):
    summary: str = Field(description="Concise summary of user inquiry")
    urgency: int = Field(description="Urgency rating from 1 to 5")

# Initializing OpenAI model with structured output
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_llm = llm.with_structured_output(AnalysisReport)

# When invoked, LangSmith records the exact prompt, token count, cost, and parsed JSON
response = structured_llm.invoke("Our production database went down 5 minutes ago!")
print("Structured Output:", response)
''', language="python")

# ==============================================================================
# SECTION 3: ANTHROPIC
# ==============================================================================
st.header("3. Anthropic: Claude 3.5 Sonnet, Artifacts, Extended Thinking, and Prompt Caching Tracing", divider="rainbow")

st.markdown("""
**Anthropic's Claude 3.5 Sonnet** has established market leadership in complex reasoning, software engineering, 
and nuanced human interaction. Anthropic introduces unique architectural features that require dedicated observability.
""")

col_a1, col_a2 = st.columns(2)
with col_a1:
    st.markdown("""
    ### Architectural Strengths
    - **Prompt Caching**:
      Allows caching static system instructions and few-shot examples for up to 5 minutes, reducing prompt cost by up to 90% and latency by up to 80%.
    - **Extended Thinking**:
      Allows Claude models to output detailed internal reasoning blocks before synthesizing the final response.
    - **Nuanced Instruction Following**:
      Superior adherence to complex, multi-page system rubrics.
    """)

with col_a2:
    st.markdown("""
    ### LangSmith Telemetry Capture
    - **Cache Hit / Cache Miss Auditing**:
      LangSmith captures `cache_read_input_tokens` and `cache_creation_input_tokens`, allowing teams to measure prompt cache hit ratios.
    - **Thinking Block Spans**:
      Internal reasoning tokens are captured and displayed in expandable trace views without polluting the user-facing output string.
    """)

st.code('''from langchain_anthropic import ChatAnthropic

# Initializing Claude 3.5 Sonnet
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.0,
    max_tokens=4096
)

# Traced invocation
result = claude.invoke("Write a Python unit test verifying LangSmith client initialization.")
print(result.content[:100])
''', language="python")

# ==============================================================================
# SECTION 4: GOOGLE GEMINI
# ==============================================================================
st.header("4. Google Gemini: Gemini 1.5 Flash/Pro, Multi-Modal Ingestion, and Million-Token Context Observability", divider="rainbow")

st.markdown("""
**Google Gemini 1.5 Pro and Flash** feature an unprecedented **2,000,000 token context window**, 
enabling applications that ingest entire video files, hours of audio, and multiple large codebases into a single prompt.
""")

st.markdown("""
### Observability Challenges in Million-Token Regimes
1. **Context Window Inflation**: Ingesting 500,000 tokens per prompt creates significant latency (several seconds for prefill) and high token bills if unmonitored. LangSmith alerts identify runaway context bloat.
2. **Multi-Modal Payload Inspection**: When images, PDF pages, or audio segments are passed to Gemini, LangSmith trace spans record the multimodal data blobs or references for auditability.
3. **Price Disparity Monitoring**: In Gemini, queries exceeding 128k tokens transition to a higher pricing tier; LangSmith dynamically maps this price tier transition.
""")

st.code('''from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.0
)

# Long-context ingestion
query = "Analyze the entire codebase architecture from the provided documentation context."
response = gemini.invoke(query)
print(response.content[:100])
''', language="python")

# ==============================================================================
# SECTION 5: OPEN-SOURCE LLMS
# ==============================================================================
st.header("5. Open-Source LLMs: HuggingFace, vLLM, DeepSeek, and Hosted Endpoints (Groq, Together)", divider="rainbow")

st.markdown("""
Open-source models (Meta Llama 3.1, Mistral, DeepSeek-Coder, Qwen 2.5) offer enterprise data sovereignty, 
zero vendor lock-in, and aggressive price competition. Production deployments typically use **vLLM** for self-hosting 
or specialized ultra-fast inference APIs such as **Groq**, **Together AI**, or **Fireworks AI**.
""")

st.markdown("""
### Tracing OpenAI-Compatible Hosted Providers in LangSmith
Most open-source inference servers expose an OpenAI-compatible REST API (`/v1/chat/completions`). 
You can instrument them using `ChatOpenAI` simply by overriding `base_url` and `api_key`:
""")

st.code('''from langchain_openai import ChatOpenAI

# 1. Ultra-Low-Latency Inference via Groq (Llama-3.1-70B)
groq_llm = ChatOpenAI(
    model="llama-3.1-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key="gsk_your_groq_api_key",
    temperature=0.0
)

# 2. Self-Hosted vLLM Instance on Internal Kubernetes
vllm_llm = ChatOpenAI(
    model="deepseek-ai/deepseek-coder-33b-instruct",
    openai_api_base="http://vllm-service.internal:8000/v1",
    openai_api_key="none", # Internal cluster token
    temperature=0.0
)

# LangSmith treats both as standard chat models, capturing TTFT and tokens automatically
res = groq_llm.invoke("What is Reciprocal Rank Fusion?")
print(res.content[:100])
''', language="python")

# ==============================================================================
# SECTION 6: LOCAL LLMS
# ==============================================================================
st.header("6. Local LLMs: Ollama, Llama.cpp, Air-Gapped Deployments, and Local Tracing Proxies", divider="rainbow")

st.markdown("""
For sensitive data environments (healthcare, defense, finance) or developer workstations without internet access, 
**local LLMs** running via **Ollama** or **Llama.cpp** run completely on-device.
""")

st.markdown("""
### Tracing Local LLMs with LangSmith
Even when models run locally, LangSmith can record traces to the cloud or to a self-hosted LangSmith instance 
deployed within the air-gapped network:
""")

st.code('''from langchain_community.chat_models import ChatOllama
from langsmith import traceable

# Connect to local Ollama daemon (localhost:11434)
local_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0
)

@traceable(run_type="chain", name="air_gapped_analysis_chain")
def analyze_sensitive_document(doc_text: str) -> str:
    """Traced local chain: telemetry pushes to LangSmith while inference stays on localhost."""
    response = local_llm.invoke(f"Extract confidential entities: {doc_text}")
    return response.content

# Executes locally, traces to LangSmith
print("Local Model Loaded Successfully.")
''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Provider Unified Tracing & Dynamic Fallback Router", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a **multi-provider fallback router**. 
If the primary provider (OpenAI) experiences rate limits (HTTP 429) or network timeouts, the router 
automatically fails over to Anthropic or Groq while logging the failover event to LangSmith.
""")

st.code('''import time
from typing import Dict, Any
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

client = Client()

@traceable(run_type="llm", name="openai_provider_span")
def call_openai_mock(prompt: str) -> str:
    # Simulates potential provider rate limit failure
    time.sleep(0.15)
    if "simulate_fail" in prompt:
        raise ConnectionError("HTTP 429 Too Many Requests: Rate limit exceeded")
    return f"OpenAI Response for: {prompt[:30]}"

@traceable(run_type="llm", name="anthropic_fallback_span")
def call_anthropic_mock(prompt: str) -> str:
    # Fallback secondary provider
    time.sleep(0.20)
    return f"Anthropic Fallback Response for: {prompt[:30]}"

@traceable(run_type="chain", name="resilient_multi_provider_router")
def multi_provider_router(user_query: str) -> Dict[str, Any]:
    """
    Attempts primary provider; catches exceptions and transparently fails over to secondary.
    Records failover status in LangSmith run metadata.
    """
    run_tree = get_current_run_tree()
    provider_used = "OpenAI"
    
    try:
        response_text = call_openai_mock(user_query)
    except Exception as e:
        # Failover trigger
        provider_used = "Anthropic (Fallback)"
        if run_tree:
            run_tree.metadata["failover_triggered"] = True
            run_tree.metadata["primary_error"] = str(e)
            run_tree.tags.append("failover-recovered")
        response_text = call_anthropic_mock(user_query)
        
    return {
        "query": user_query,
        "answer": response_text,
        "provider_resolved": provider_used
    }

# Test 1: Normal execution (OpenAI succeeds)
res1 = multi_provider_router("Standard customer inquiry")
print("Run 1 Result:", res1)

# Test 2: Triggering failover (OpenAI fails, Anthropic recovers)
res2 = multi_provider_router("simulate_fail: high load query")
print("Run 2 Result:", res2)
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Multi-Provider Latency, Cost, and Token Comparator", divider="rainbow")

st.markdown("""
Compare operational metrics across major LLM providers. Simulate model invocations with varying 
token volumes to evaluate cost differentials and latency profiles.
""")

wb_mode = st.selectbox(
    "Select Workbench Mode:",
    ["1. Provider Cost & Latency Simulator", "2. Comprehensive Provider Catalog Matrix", "3. Fallback Resilience Tester"]
)

if wb_mode == "1. Provider Cost & Latency Simulator":
    st.subheader("Model Invocations Cost & Latency Calculator")
    sel_prov = st.selectbox("Select Model:", list(providers_engine.providers_catalog.keys()))
    
    c1, c2 = st.columns(2)
    with c1:
        p_tok = st.slider("Prompt Tokens:", 100, 50000, 2000, 500)
    with c2:
        c_tok = st.slider("Completion Tokens:", 50, 4000, 400, 50)
        
    sim_data = providers_engine.simulate_provider_run(sel_prov, p_tok, c_tok)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Run Cost", f"${sim_data['total_cost_usd']}")
    with m2:
        st.metric("Simulated Total Latency", f"{sim_data['latency_ms']}ms")
    with m3:
        st.metric("Time to First Token (TTFT)", f"{sim_data['ttft_ms']}ms")
        
    st.info(f"**Provider Category:** `{sim_data['category']}` | **Total Tokens:** `{sim_data['total_tokens']}`")

elif wb_mode == "2. Comprehensive Provider Catalog Matrix":
    st.subheader("Enterprise LLM Provider Catalog")
    cat_df = pd.DataFrame([
        {
            "Provider Model": k,
            "Category": v["category"],
            "Context Window": f"{v['context_window']:,}",
            "Prompt ($/1M)": f"${v['prompt_cost_per_m']}",
            "Completion ($/1M)": f"${v['completion_cost_per_m']}",
            "Avg TTFT": f"{v['avg_ttft_ms']}ms",
            "Key Strengths": v["strengths"]
        }
        for k, v in providers_engine.providers_catalog.items()
    ])
    st.dataframe(cat_df, use_container_width=True)

else:
    st.subheader("Multi-Provider Fallback Resilience Tester")
    st.markdown("Simulate what happens when primary provider APIs experience rate limits (HTTP 429):")
    
    trigger_fail = st.toggle("Simulate Primary Provider Outage (HTTP 429)", value=True)
    
    if st.button("Dispatch Request via Router"):
        if trigger_fail:
            st.warning("Primary Provider (OpenAI) returned HTTP 429 Rate Limit!")
            st.success("Fallback Activated: Anthropic Claude 3.5 Sonnet answered successfully in 410ms.")
            st.info("LangSmith Telemetry: Trace flagged with tag 'failover-recovered' and primary_error recorded.")
        else:
            st.success("Primary Provider (OpenAI) answered successfully in 320ms.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do you normalize token accounting and cost tracking when using multiple LLM providers?"):
    st.markdown("""
    **Answer:**  
    Different providers format usage payloads differently (e.g., OpenAI returns `prompt_tokens`, Anthropic returns 
    `input_tokens`, Google returns `prompt_token_count`). LangSmith standardizes these into a uniform `UsageMetadata` 
    schema and references an internal pricing dictionary to calculate exact dollar cost across all providers automatically.
    """)

with st.expander("Q2: What is the architectural advantage of Anthropic's prompt caching in high-volume RAG systems?"):
    st.markdown("""
    **Answer:**  
    In RAG or few-shot pipelines where the system prompt and static documentation context remain constant across queries, 
    Anthropic prompt caching allows reusing the pre-computed key-value cache. This reduces input token costs by up to 90% 
    and slashes Time to First Token (TTFT) by up to 80% for cache hits.
    """)

with st.expander("Q3: How do you trace self-hosted open-source models running on vLLM with LangSmith?"):
    st.markdown("""
    **Answer:**  
    Since vLLM exposes an OpenAI-compatible API, initialize `ChatOpenAI(openai_api_base="http://vllm-host:8000/v1")`. 
    When LangSmith tracing is active, all requests and streaming responses are captured exactly like standard OpenAI 
    calls without requiring custom SDK instrumentation.
    """)

with st.expander("Q4: What makes Google Gemini's 2-million-token context window both powerful and risky for production?"):
    st.markdown("""
    **Answer:**  
    It allows whole-repository and multimodal video ingestion without chunking. However, it introduces significant 
    risks: 'lost in the middle' retrieval degradation, increased latency during prefill, and high cost per query 
    if irrelevant documents are carelessly appended to the prompt.
    """)

with st.expander("Q5: How do you design an automated provider fallback mechanism using LangSmith tags for alerting?"):
    st.markdown("""
    **Answer:**  
    Wrap provider invocations in a try-except block. In the exception handler, catch `RateLimitError` or `APIConnectionError`, 
    append a tag (e.g., `tags=["failover-recovered"]`) and metadata (`metadata={"primary_error": str(e)}`) to the 
    current LangSmith run tree, and dispatch the secondary provider. In LangSmith, configure an alert rule to notify 
    Slack when the `failover-recovered` tag exceeds 5 occurrences per minute.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Hardcoding Provider-Specific Enums in Business Logic**  
Writing code directly coupled to `openai.OpenAI()` makes it impossible to switch to Claude or vLLM during outages. 
Always use standardized LangChain abstractions or unified wrapper interfaces.
""")

st.error("""
**Trap 2: Ignoring Provider Rate Limits (HTTP 429) Without Exponential Backoff and Fallback**  
Surges in user traffic can quickly exhaust provider TPM (Tokens Per Minute) quotas. 
Failing to implement fallback routes causes immediate hard errors for end users.
""")

st.warning("""
**Trap 3: Blindly Passing Multi-Megabyte Documents to Long-Context Models**  
Just because a model accepts 2M tokens does not mean it should be used as a brute-force database. 
Dense RAG pre-filtering almost always yields faster response times and 90% lower operational costs.
""")

st.warning("""
**Trap 4: Forgetting to Track Streaming TTFT Separately from Total Duration**  
A 4,000-token completion may take 15 seconds to finish, but if the Time to First Token is 300ms, the user perceives 
instant responsiveness. Monitoring only total duration misrepresents perceived latency.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Prompt Caching</h4>
        <p>Anthropic mechanism that stores prompt prefixes in memory for up to 5 minutes, yielding a 90% cost reduction on cache hits.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>vLLM Endpoint</h4>
        <p>A high-throughput, memory-efficient open-source LLM inference engine supporting OpenAI-compatible REST APIs.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Provider Failover</h4>
        <p>An architectural circuit breaker that routes traffic to an alternate LLM provider upon detecting HTTP 429 rate limits or timeouts.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>UsageMetadata</h4>
        <p>The standardized LangChain and LangSmith schema normalizing token usage across disparate AI model APIs.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand how LangSmith normalizes token accounting and cost tracking across diverse model providers.", value=True)
chk_2 = st.checkbox("I know how to instrument OpenAI structured outputs and trace schema validation errors.", value=True)
chk_3 = st.checkbox("I understand Anthropic's prompt caching mechanics and how cache hits are audited in LangSmith.", value=True)
chk_4 = st.checkbox("I can articulate the observability trade-offs of Google Gemini's 2M token context window.", value=True)
chk_5 = st.checkbox("I know how to trace open-source models deployed via vLLM or Groq using OpenAI-compatible endpoints.", value=True)
chk_6 = st.checkbox("I can implement a resilient multi-provider router with automated fallback and LangSmith failover tagging.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LLM Providers ecosystem mastery verified! Ready for Data and Retrieval.")
