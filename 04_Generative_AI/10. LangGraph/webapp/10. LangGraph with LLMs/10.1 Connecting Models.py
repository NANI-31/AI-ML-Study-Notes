import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="10.1 Connecting Models - LangGraph Curriculum",
    page_icon="🔌",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE MODEL CONNECTOR REGISTRY FOR WORKBENCH & PIPELINE
# ==============================================================================
class ModelConnectorRegistry:
    """
    Catalogues and simulates the multi-provider LLM ecosystem in LangGraph:
    OpenAI, Anthropic, Google Gemini, Ollama, Hugging Face, and Custom BaseChatModel.
    """
    def __init__(self):
        self.providers = {
            "OpenAI": {
                "flagship": "gpt-4o",
                "package": "langchain-openai",
                "tool_calling": "Native json_schema",
                "context_window": "128k tokens",
                "latency_profile": "~450 ms TTFT"
            },
            "Anthropic": {
                "flagship": "claude-3-5-sonnet-20241022",
                "package": "langchain-anthropic",
                "tool_calling": "Native xml/json tools",
                "context_window": "200k tokens",
                "latency_profile": "~520 ms TTFT"
            },
            "Google Gemini": {
                "flagship": "gemini-1.5-pro",
                "package": "langchain-google-genai",
                "tool_calling": "Native functionDeclarations",
                "context_window": "2M tokens",
                "latency_profile": "~600 ms TTFT"
            },
            "Ollama (Local)": {
                "flagship": "llama3.3:70b",
                "package": "langchain-ollama",
                "tool_calling": "Native tool calling supported",
                "context_window": "8k-128k tokens",
                "latency_profile": "Local GPU dependent"
            },
            "Hugging Face": {
                "flagship": "Qwen/Qwen2.5-72B-Instruct",
                "package": "langchain-huggingface",
                "tool_calling": "Prompt or TGI native",
                "context_window": "32k-128k tokens",
                "latency_profile": "Endpoint dependent"
            }
        }

    def simulate_provider_call(self, provider: str, prompt: str) -> Dict[str, Any]:
        meta = self.providers.get(provider, self.providers["OpenAI"])
        return {
            "provider": provider,
            "model_used": meta["flagship"],
            "tool_calling_protocol": meta["tool_calling"],
            "simulated_response": f"[{provider} - {meta['flagship']}]: Generated response grounded in prompt '{prompt[:35]}...'",
            "estimated_cost_per_m": "2.50 USD / 10.00 USD" if provider in ["OpenAI", "Anthropic"] else "0.00 USD (Self-hosted)"
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Heterogeneous Model Provider Integration in LangGraph", divider="rainbow")

st.markdown("""
LangGraph is **model-agnostic**. It does not lock applications into a single LLM vendor.
Because LangGraph accepts any callable that adheres to the standard LangChain `BaseChatModel` interface,
enterprises can seamlessly orchestrate **heterogeneous model topologies**:
using fast open-source models for routing, frontier models for complex reasoning, and local Ollama models for air-gapped PII scrubbing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    models_blueprint_dot = """
    digraph ModelProviderArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_graph_runtime {
            label="1. LangGraph State Machine Runtime";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START", fillcolor="#1E293B", shape=ellipse];
            Node_Triage [label="Triage Node (Fast / Low Cost)\\nModel: Ollama llama3.2 (Local)", fillcolor="#0284C7"];
            Node_Reason [label="Reasoning Node (Frontier)\\nModel: Claude 3.5 Sonnet / GPT-4o", fillcolor="#4338CA"];
            Node_Embed [label="Summarizer Node (High Context)\\nModel: Gemini 1.5 Pro (2M Context)", fillcolor="#059669"];
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        subgraph cluster_provider_adapters {
            label="2. LangChain Model Adapters (BaseChatModel)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            P_Ollama [label="langchain-ollama\\nChatOllama", fillcolor="#334155"];
            P_OpenAI [label="langchain-openai\\nChatOpenAI", fillcolor="#334155"];
            P_Anthropic [label="langchain-anthropic\\nChatAnthropic", fillcolor="#334155"];
            P_Google [label="langchain-google-genai\\nChatGoogleGenerativeAI", fillcolor="#334155"];
        }

        START -> Node_Triage;
        Node_Triage -> P_Ollama [label="Local inference"];
        Node_Triage -> Node_Reason [label="Complex task"];
        Node_Reason -> P_Anthropic [label="Frontier reasoning"];
        Node_Reason -> P_OpenAI [label="Fallback provider"];
        Node_Reason -> Node_Embed;
        Node_Embed -> P_Google [label="Massive context ingestion"];
        Node_Embed -> END;
    }
    """
    st.graphviz_chart(models_blueprint_dot, use_container_width=True)

st.markdown("""
### Model Portability Standard: BaseChatModel Protocol
Every model provider in LangGraph conforms to the `BaseChatModel` interface:
- `.invoke(messages)` -> `AIMessage`
- `.stream(messages)` -> Token generator
- `.bind_tools(tools)` -> Native tool schema binding
- `.with_structured_output(Schema)` -> Pydantic guaranteed output
""")

# ==============================================================================
# SECTION 2: OPENAI MODELS
# ==============================================================================
st.header("2. OpenAI Models: GPT-4o, GPT-4o-mini, and Structured Outputs", divider="rainbow")

st.markdown("""
OpenAI models connect via the `langchain-openai` package.
They offer strong function calling reliability and native structured JSON outputs.
""")

code_openai = """from langchain_openai import ChatOpenAI

# 1. Initialize Model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=4096,
    api_key="..." # Or export OPENAI_API_KEY
)

# 2. Node Function
def call_openai_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
"""
st.code(code_openai, language="python")

# ==============================================================================
# SECTION 3: ANTHROPIC MODELS
# ==============================================================================
st.header("3. Anthropic Models: Claude 3.5 Sonnet, Tool Calling, and Extended Thinking", divider="rainbow")

st.markdown("""
Anthropic models connect via `langchain-anthropic`.
Claude 3.5 Sonnet is considered the premier frontier model for complex coding, agentic tool workflows,
and nuanced multi-step system prompt adherence.
""")

code_anthropic = """from langchain_anthropic import ChatAnthropic

claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.0,
    max_tokens=8192
)

def call_claude_node(state: AgentState) -> dict:
    return {"messages": [claude.invoke(state["messages"])]}
"""
st.code(code_anthropic, language="python")

# ==============================================================================
# SECTION 4: GOOGLE GEMINI
# ==============================================================================
st.header("4. Google Gemini: Gemini 1.5 Pro, Flash, and Long-Context Grounding", divider="rainbow")

st.markdown("""
Google Gemini models connect via `langchain-google-genai`.
Gemini 1.5 Pro provides a massive **2-million token context window**, making it ideal for ingesting
entire codebase repositories, video footage, or multi-hour audio recordings directly inside graph nodes.
""")

code_gemini = """from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1
)
"""
st.code(code_gemini, language="python")

# ==============================================================================
# SECTION 5: OPEN-SOURCE MODELS
# ==============================================================================
st.header("5. Open-Source Models: vLLM, TGI, and Local Host Serving", divider="rainbow")

st.markdown("""
Enterprises running open-source models (Llama 3, Qwen 2.5, DeepSeek) on private Kubernetes clusters
typically serve them via **vLLM** or **Text Generation Inference (TGI)** exposing an OpenAI-compatible endpoint.
""")

code_vllm = """from langchain_openai import ChatOpenAI

# Point ChatOpenAI to your private vLLM cluster!
private_llm = ChatOpenAI(
    base_url="http://vllm-cluster.internal:8000/v1",
    api_key="none",
    model="meta-llama/Llama-3.3-70B-Instruct"
)
"""
st.code(code_vllm, language="python")

# ==============================================================================
# SECTION 6: OLLAMA
# ==============================================================================
st.header("6. Ollama: Local Desktop Offline Model Execution", divider="rainbow")

st.markdown("""
**Ollama** allows developers to run models locally on Mac, Windows, or Linux with zero cloud API keys.
Connects via `langchain-ollama`.
""")

code_ollama = """from langchain_ollama import ChatOllama

local_model = ChatOllama(
    model="llama3.2:latest",
    temperature=0.0
)

def local_privacy_node(state: AgentState) -> dict:
    # Runs entirely on local machine; zero data leaves company hardware!
    return {"messages": [local_model.invoke(state["messages"])]}
"""
st.code(code_ollama, language="python")

# ==============================================================================
# SECTION 7: HUGGING FACE MODELS
# ==============================================================================
st.header("7. Hugging Face Models: Inference Endpoints, Pipelines, and Serverless APIs", divider="rainbow")

st.markdown("""
Connects via `langchain-huggingface`. Supports Hugging Face Serverless Inference APIs
or dedicated Inference Endpoints on AWS/GCP.
""")

code_hf = """from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
hf_chat = ChatHuggingFace(llm=endpoint)
"""
st.code(code_hf, language="python")

# ==============================================================================
# SECTION 8: OTHER LANGCHAIN-COMPATIBLE MODELS
# ==============================================================================
st.header("8. Other LangChain-Compatible Models: BaseChatModel Interface and Custom Wrappers", divider="rainbow")

st.markdown("""
Any custom or proprietary internal model can be plugged into LangGraph by subclassing `BaseChatModel`
from `langchain_core.language_models.chat_models`.
""")

code_custom = """from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

class ProprietaryCompanyModel(BaseChatModel):
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        # Call proprietary internal HTTP API
        output_text = "Custom corporate response"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output_text))])
        
    @property
    def _llm_type(self) -> str:
        return "proprietary_company_model"
"""
st.code(code_custom, language="python")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Multi-Provider Chat Engine with Graceful Model Fallback", divider="rainbow")

st.markdown("""
Below is a complete, runnable script implementing a multi-provider fallback pattern.
The graph attempts inference using the primary model provider; if it encounters a rate limit or timeout,
it seamlessly cascades to the secondary provider without failing the user request.
""")

prod_models_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: State Contract
class MultiProviderState(TypedDict):
    prompt: str
    active_provider: str
    response: Optional[str]
    has_failed: bool
    audit_trace: Annotated[List[str], operator.add]

# Nodes
def primary_provider_node(state: MultiProviderState) -> Dict[str, Any]:
    # Simulate primary frontier provider (e.g. OpenAI / Anthropic)
    prompt = state["prompt"]
    
    # Simulate rare outage trigger
    if "simulate_outage" in prompt.lower():
        return {
            "has_failed": True,
            "audit_trace": ["Primary Provider (Frontier) failed with HTTP 503 Outage."]
        }
        
    return {
        "response": f"Primary Provider generated response for: '{prompt}'",
        "active_provider": "Frontier-Primary",
        "has_failed": False,
        "audit_trace": ["Primary Provider completed inference successfully."]
    }

def fallback_provider_node(state: MultiProviderState) -> Dict[str, Any]:
    # Secondary fallback provider (e.g. Gemini / Local vLLM)
    prompt = state["prompt"]
    return {
        "response": f"Fallback Provider generated response for: '{prompt}'",
        "active_provider": "Fallback-Secondary",
        "has_failed": False,
        "audit_trace": ["Fallback Provider successfully rescued failed turn."]
    }

# Router Predicate
def check_provider_health(state: MultiProviderState) -> str:
    if state["has_failed"]:
        return "fallback"
    return "finalize"

# Step 2: Build Graph Topology
def build_multi_provider_graph():
    builder = StateGraph(MultiProviderState)
    
    builder.add_node("primary_model", primary_provider_node)
    builder.add_node("fallback_model", fallback_provider_node)
    
    builder.add_edge(START, "primary_model")
    builder.add_conditional_edges(
        "primary_model",
        check_provider_health,
        {"fallback": "fallback_model", "finalize": END}
    )
    builder.add_edge("fallback_model", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_multi_provider_graph()
    
    # Run 1: Normal primary execution
    out1 = app.invoke({
        "prompt": "Explain LangGraph state channels",
        "active_provider": "None",
        "response": None,
        "has_failed": False,
        "audit_trace": []
    })
    print("Run 1 Provider:", out1["active_provider"])
    
    # Run 2: Fallback triggered on simulated outage
    out2 = app.invoke({
        "prompt": "simulate_outage: check fallback mechanics",
        "active_provider": "None",
        "response": None,
        "has_failed": False,
        "audit_trace": []
    })
    print("Run 2 Provider:", out2["active_provider"])
    print("Run 2 Audit:", out2["audit_trace"])
'''
st.code(prod_models_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Model Provider Benchmark and Compatibility Inspector", divider="rainbow")

st.markdown("""
Compare model provider capabilities across latency, tool calling protocols, and context windows.
Select a provider and observe its simulated invocation profile.
""")

reg = ModelConnectorRegistry()

col_mw_l, col_mw_r = st.columns([1, 1.2])

with col_mw_l:
    st.subheader("Select Model Provider")
    chosen_prov = st.selectbox("Provider Family", list(reg.providers.keys()))
    meta = reg.providers[chosen_prov]
    st.caption(f"**Flagship Model**: `{meta['flagship']}`")
    st.caption(f"**LangChain Package**: `{meta['package']}`")
    st.caption(f"**Context Window**: `{meta['context_window']}`")

    test_prompt = st.text_input("Test Prompt", value="Construct resilient multi-agent architecture in LangGraph")

    if st.button("🔌 Simulate Provider Connection", use_container_width=True):
        st.session_state.prov_sim = reg.simulate_provider_call(chosen_prov, test_prompt)
        st.success(f"Connected to {chosen_prov}!")

with col_mw_r:
    st.subheader("Provider Invocation Telemetry")
    if "prov_sim" in st.session_state:
        st.json(st.session_state.prov_sim)
    else:
        st.info("Select a provider and click 'Simulate Provider Connection'.")

st.markdown("### Model Ecosystem Comparison Matrix")
df_prov = pd.DataFrame([
    {"Provider": k, **v} for k, v in reg.providers.items()
])
st.dataframe(df_prov, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does LangGraph ensure unified message handling across different model providers?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph relies on LangChain's normalized message schema (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`).
    Regardless of whether an OpenAI, Anthropic, or Gemini model is invoked, the provider package normalizes the API's idiosyncratic JSON response into a standard `AIMessage(content=..., tool_calls=[...])`. This allows downstream nodes and routers to evaluate messages agnostically.
    """)

with st.expander("Q2: Why should API keys never be stored inside the LangGraph state dictionary?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph state is serialized to checkpointer storage (Postgres, SQLite, Redis) at every superstep.
    Storing API keys in state persists sensitive secrets in database dumps and exposes them in LangSmith traces.
    API keys should always be managed via environment variables (`.env`) or injected at runtime via `config["configurable"]`.
    """)

with st.expander("Q3: How does with_fallbacks() interact with LangGraph nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    LangChain models support `.with_fallbacks([fallback_model])`.
    You can bind this inside a single node function:
    ```python
    resilient_model = primary_llm.with_fallbacks([backup_llm])
    response = resilient_model.invoke(messages)
    ```
    This handles provider-level HTTP timeouts and rate limits instantly without requiring separate graph nodes.
    """)

with st.expander("Q4: When should an enterprise deploy local Ollama/vLLM nodes instead of cloud frontier APIs?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Data Sovereignty & PII**: Medical or classified environments where zero data may leave on-premise hardware.
    2. **Cost at High Volume**: Ingestion pipelines processing millions of tokens per minute where API costs would be astronomical.
    3. **Zero Latency Jitter**: Dedicated on-premise GPUs eliminate internet routing jitter and third-party rate limits.
    """)

with st.expander("Q5: What is the primary difference in tool calling format between OpenAI and Anthropic?", expanded=False):
    st.markdown("""
    **Answer**:
    - **OpenAI**: Emits `tool_calls` with JSON strings inside `arguments`.
    - **Anthropic**: Natively models tools using XML tags or structured JSON blocks.
    LangChain packages (`ChatOpenAI`, `ChatAnthropic`) abstract this difference, converting both into standardized `tool_calls = [{"name": "...", "args": {...}, "id": "..."}]` dictionaries on `AIMessage`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Hardcoded Cloud Model Names in Air-Gapped Environments")
    st.markdown("""
    *Mechanism*: Deploying code referencing `ChatOpenAI(model="gpt-4o")` on an isolated network without internet egress.  
    *Consequence*: Nodes crash with `httpx.ConnectError` or DNS resolution timeouts.  
    *Fix*: Parameterize model instantiation via environment variables to toggle between cloud APIs and local vLLM endpoints.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Unaligned Tool Call IDs Across Heterogeneous Model Handoffs")
    st.markdown("""
    *Mechanism*: Model A emits a tool call, but a fallback Model B attempts to process the conversation without propagating `tool_call_id`.  
    *Consequence*: Cloud APIs reject subsequent turns with `400 Bad Request: Invalid tool_call_id`.  
    *Fix*: Maintain exact `tool_call_id` linkage in `ToolMessage` instances across provider handoffs.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Allocating Frontier Models for Trivial Routing")
    st.markdown("""
    *Mechanism*: Calling GPT-4o or Claude 3.5 Sonnet for simple binary if/else classification triage nodes.  
    *Consequence*: Increases system latency by 600ms and inflates billing unnecessarily.  
    *Fix*: Use small fast models (e.g. `gpt-4o-mini`, `gemini-1.5-flash`, or `llama3.2`) for classification nodes.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Token Limit Truncation in Local Ollama Nodes")
    st.markdown("""
    *Mechanism*: Passing 30k tokens of conversation history to an Ollama model configured with default `num_ctx=2048`.  
    *Consequence*: Silent truncation of early prompt messages, causing severe agent amnesia.  
    *Fix*: Explicitly configure `num_ctx=32768` in `ChatOllama` parameters.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**BaseChatModel**\\nThe standard LangChain interface implemented by all model providers (OpenAI, Claude, Gemini, Ollama).")
with col_fc2:
    st.info("**with_fallbacks()**\\nA LangChain model wrapper automatically cascading to secondary providers on rate-limits.")
with col_fc3:
    st.info("**Heterogeneous Graph**\\nA graph topology mixing diverse model families (e.g. local Ollama triage + Claude reasoning).")
with col_fc4:
    st.info("**vLLM Endpoint**\\nAn open-source high-throughput serving engine exposing an OpenAI-compatible API for private clusters.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to connect OpenAI, Anthropic, Gemini, and Ollama models in LangGraph.", value=True)
c2 = st.checkbox("I understand the BaseChatModel protocol enabling provider-agnostic node execution.", value=True)
c3 = st.checkbox("I can implement multi-provider fallback mechanisms to withstand cloud outages.", value=True)
c4 = st.checkbox("I know how to connect private vLLM clusters using OpenAI-compatible base URLs.", value=True)
c5 = st.checkbox("I understand the cost and latency benefits of heterogeneous model graphs.", value=True)
c6 = st.checkbox("I know how to prevent API key leakage into persistent state checkpointers.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 10.1 'Connecting Models' with full architectural and practical comprehension!")
