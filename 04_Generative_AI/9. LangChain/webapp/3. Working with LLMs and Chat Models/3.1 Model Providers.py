import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="3.1 Model Providers - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("3.1 Model Providers: Unified Multi-Cloud Chat Model Ecosystem")
st.caption("Chapter 3: Working with LLMs and Chat Models | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In the enterprise Generative AI landscape, relying on a single foundation model vendor creates critical 
vendor lock-in, exposes applications to provider outages, and prevents optimizing for cost versus intelligence. 
Modern architectures require a **multi-provider strategy**: routing high-volume simple queries to ultra-fast 
or local engines, while reserving frontier reasoning models for complex planning and code synthesis.

**LangChain (v0.3+)** decouples your application logic from underlying model vendors. Through specialized 
partner integration packages (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-groq`, etc.) 
and the unified `BaseChatModel` interface, you interact with every major commercial, cloud hyperscaler, and open-weight 
provider through an identical API contract.

In this lesson, you will master:
- **Commercial Frontier Providers**: OpenAI, Anthropic, Google Gemini, and Cohere.
- **Enterprise Cloud Hyperscalers**: Azure OpenAI and AWS Bedrock with private networking and IAM authentication.
- **Ultra-Low Latency & Open Engines**: Groq LPU inference, Mistral AI, and local air-gapped Ollama instances.
- **Open-Source Hubs**: Hugging Face endpoints and community providers (Together, Fireworks, DeepInfra).
- **Architectural Portability**: Designing zero-downtime multi-provider fallback gateways.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockProviderBenchmarkRecord:
    """Represents a simulated model benchmark profile in module scope."""
    def __init__(self, provider: str, default_model: str, p95_latency_ms: int, 
                 input_cost_per_m: float, output_cost_per_m: float, 
                 context_window_k: int, native_tools: bool, vision_support: bool):
        self.provider = provider
        self.default_model = default_model
        self.p95_latency_ms = p95_latency_ms
        self.input_cost_per_m = input_cost_per_m
        self.output_cost_per_m = output_cost_per_m
        self.context_window_k = context_window_k
        self.native_tools = native_tools
        self.vision_support = vision_support

class MultiProviderBenchmarkEngine:
    """Simulates multi-cloud provider comparisons, latency profiles, and cost modeling in module scope."""
    def __init__(self):
        self.profiles = [
            MockProviderBenchmarkRecord("OpenAI", "gpt-4o", 420, 2.50, 10.00, 128, True, True),
            MockProviderBenchmarkRecord("Anthropic", "claude-3-5-sonnet", 480, 3.00, 15.00, 200, True, True),
            MockProviderBenchmarkRecord("Google Gemini", "gemini-1.5-flash", 290, 0.075, 0.30, 1000, True, True),
            MockProviderBenchmarkRecord("Azure OpenAI", "gpt-4o (WestUS)", 450, 2.50, 10.00, 128, True, True),
            MockProviderBenchmarkRecord("AWS Bedrock", "anthropic.claude-v3-5", 510, 3.00, 15.00, 200, True, True),
            MockProviderBenchmarkRecord("Cohere", "command-r-plus", 550, 2.50, 10.00, 128, True, False),
            MockProviderBenchmarkRecord("Mistral AI", "mistral-large", 410, 2.00, 6.00, 128, True, False),
            MockProviderBenchmarkRecord("Groq", "llama-3.3-70b-versatile", 110, 0.59, 0.79, 128, True, False),
            MockProviderBenchmarkRecord("Ollama (Local)", "llama-3.1-8b", 85, 0.00, 0.00, 128, True, False),
            MockProviderBenchmarkRecord("Hugging Face", "deepseek-ai/v2-chat", 380, 0.80, 1.20, 64, True, False)
        ]

    def get_dataframe(self) -> pd.DataFrame:
        data = []
        for p in self.profiles:
            data.append({
                "Provider": p.provider,
                "Flagship Model": p.default_model,
                "p95 Latency (ms)": p.p95_latency_ms,
                "Input $/1M Tokens": f"${p.input_cost_per_m:.3f}",
                "Output $/1M Tokens": f"${p.output_cost_per_m:.2f}",
                "Context Window": f"{p.context_window_k}k tokens",
                "Tool Calling": "✅ Native" if p.native_tools else "⚠️ Emulated",
                "Multimodal Vision": "✅ Yes" if p.vision_support else "❌ No"
            })
        return pd.DataFrame(data)

    def simulate_invocation(self, provider_name: str, prompt_text: str) -> Dict[str, Any]:
        match = next((p for p in self.profiles if p.provider == provider_name), self.profiles[0])
        simulated_tokens = len(prompt_text.split()) * 2 + 65
        est_cost = (match.input_cost_per_m * (len(prompt_text.split()) * 2) / 1000000) + (match.output_cost_per_m * 65 / 1000000)
        
        return {
            "provider": match.provider,
            "model": match.default_model,
            "latency_ms": match.p95_latency_ms + np.random.randint(-20, 35),
            "tokens_processed": simulated_tokens,
            "estimated_cost_usd": f"${est_cost:.6f}",
            "response_text": (
                f"Generated response from {match.provider} ({match.default_model}): "
                f"Successfully processed query '{prompt_text[:35]}...' under unified BaseChatModel contract."
            )
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Multi-Provider Chat Model Ecosystem", divider="rainbow")

st.markdown("""
The diagram below illustrates the unified architecture of LangChain v0.3+: applications code strictly 
against `BaseChatModel` or the `init_chat_model` factory, while decoupled provider partner packages translate 
requests into vendor-specific wire protocols.
""")

dot_blueprint = """
digraph MultiProviderEcosystem {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_app {
        label="1. Application Layer (Vendor-Agnostic)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_app [label="Your Production Application\n(LCEL Chains, LangGraph Agents, RAG Pipelines)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_factory [label="init_chat_model(model_name, model_provider=...)\nOr BaseChatModel Interface", fillcolor="#1E293B", fontcolor="#34D399"];

        node_app -> node_factory [label="invokes via LCEL", color="#38BDF8"];
    }

    subgraph cluster_commercial {
        label="2. Commercial Frontier Providers";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_openai [label="langchain-openai\nChatOpenAI (gpt-4o, o1, o3-mini)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_anthropic [label="langchain-anthropic\nChatAnthropic (claude-3-5-sonnet)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_gemini [label="langchain-google-genai\nChatGoogleGenerativeAI (gemini-1.5-pro)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_cohere [label="langchain-cohere\nChatCohere (command-r-plus)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_enterprise {
        label="3. Enterprise Cloud Hyperscalers";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_azure [label="AzureChatOpenAI\nManaged VNet & Entra ID Auth", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_bedrock [label="langchain-aws\nChatBedrock (IAM Roles & VPC Endpoints)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_specialized {
        label="4. Ultra-Fast, Open-Weight & Local Inference";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_groq [label="langchain-groq\nChatGroq (LPU Speed > 500 T/s)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_mistral [label="langchain-mistralai\nChatMistralAI (mistral-large)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_ollama [label="langchain-ollama\nChatOllama (Local Private Air-Gapped)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_hf [label="langchain-huggingface\nChatHuggingFace / TGI Endpoints", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    // Wiring
    node_factory -> node_openai [color="#38BDF8"];
    node_factory -> node_anthropic [color="#38BDF8"];
    node_factory -> node_gemini [color="#38BDF8"];
    node_factory -> node_cohere [color="#38BDF8"];
    node_factory -> node_azure [color="#38BDF8"];
    node_factory -> node_bedrock [color="#38BDF8"];
    node_factory -> node_groq [color="#38BDF8"];
    node_factory -> node_mistral [color="#38BDF8"];
    node_factory -> node_ollama [color="#38BDF8"];
    node_factory -> node_hf [color="#38BDF8"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Notice how every partner integration lives in its own dedicated package 
> (`langchain-openai`, `langchain-anthropic`, etc.). This modular packaging prevents dependency conflicts, 
> drastically reduces Docker image sizes, and allows individual provider SDK updates without redeploying `langchain-core`.
""")

# --- SECTION 2: OPENAI ---
st.header("2. OpenAI: ChatOpenAI, GPT-4o, Reasoning Models & Structured Extraction", divider="rainbow")

st.markdown("""
OpenAI remains the flagship provider in enterprise GenAI, offering models ranging from general-purpose frontier 
multimodal engines (`gpt-4o`, `gpt-4o-mini`) to specialized reasoning chains (`o1`, `o3-mini`).

LangChain's `ChatOpenAI` supports native tool calling, deterministic seed binding, logprobs, and Pydantic structured output.
""")

st.code('''from langchain_openai import ChatOpenAI

# Standard production ChatOpenAI initialization
model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=2048,
    timeout=15.0,
    max_retries=2
)

# Reasoning-optimized model (OpenAI o1 / o3-mini)
# Note: reasoning models manage internal chain-of-thought tokens
reasoning_model = ChatOpenAI(
    model="o3-mini",
    model_kwargs={"reasoning_effort": "medium"}
)
''', language="python")

# --- SECTION 3: ANTHROPIC ---
st.header("3. Anthropic: ChatAnthropic, Claude 3.5 Sonnet & Native Prompt Caching", divider="rainbow")

st.markdown("""
Anthropic's Claude series (`claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-opus`) is renowned for superior coding ability, 
nuanced system prompt adherence, and exceptional reasoning over large contexts (200k tokens).

LangChain's `ChatAnthropic` natively integrates with Claude's **Prompt Caching** API, allowing applications 
to cache repetitive system prompts and long document chunks for up to 90% cost savings and 80% latency reduction.
""")

st.code('''from langchain_anthropic import ChatAnthropic

# Instantiating Claude 3.5 Sonnet with prompt caching headers
model_claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_tokens=4096,
    timeout=20.0,
    model_kwargs={
        "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}
    }
)
''', language="python")

# --- SECTION 4: GOOGLE GEMINI ---
st.header("4. Google Gemini: ChatGoogleGenerativeAI & Million-Token Context Horizons", divider="rainbow")

st.markdown("""
Google's Gemini ecosystem (`gemini-1.5-pro`, `gemini-1.5-flash`) offers industry-leading context horizons 
(1,000,000 to 2,000,000 tokens) and native multimodal understanding (text, images, audio, video) at high performance.

LangChain's `ChatGoogleGenerativeAI` supports safety settings tuning, structured outputs, and direct Google Cloud Vertex AI integrations.
""")

st.code('''from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

# Initializing Gemini 1.5 Flash for cost-effective large-context RAG
model_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
)
''', language="python")

# --- SECTION 5: AZURE OPENAI ---
st.header("5. Azure OpenAI: AzureChatOpenAI, Enterprise Compliance & Private Endpoints", divider="rainbow")

st.markdown("""
For enterprise organizations requiring HIPAA, SOC2, or strict EU data sovereignty compliance, 
OpenAI models are frequently consumed through Microsoft Azure. 

`AzureChatOpenAI` uses **deployment names** instead of raw model names, and authenticates via Azure API keys 
or Microsoft Entra ID (Active Directory) tokens.
""")

st.code('''from langchain_openai import AzureChatOpenAI

# Enterprise Azure OpenAI configuration with private endpoint
model_azure = AzureChatOpenAI(
    azure_deployment="prod-gpt4o-deployment",   # Custom deployment name configured in Azure Portal
    azure_endpoint="https://corp-ai-prod.openai.azure.com/",
    api_version="2024-08-01-preview",
    temperature=0.1
)
''', language="python")

# --- SECTION 6: AWS BEDROCK ---
st.header("6. AWS Bedrock: ChatBedrock, IAM Authentication & Cross-Region Inference", divider="rainbow")

st.markdown("""
AWS Bedrock provides serverless access to foundation models from Anthropic (Claude), Meta (Llama 3), 
Mistral, and Amazon (Nova) governed by AWS IAM credentials, VPC private links, and KMS encryption.

LangChain's `ChatBedrock` connects using `boto3` session credentials, avoiding API keys in environment variables.
""")

st.code('''from langchain_aws import ChatBedrock

# AWS Bedrock invoking Claude 3.5 Sonnet using ambient IAM role
model_bedrock = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.2, "max_tokens": 2048}
)
''', language="python")

# --- SECTION 7: COHERE ---
st.header("7. Cohere: ChatCohere, Command R+ & Native Citation Grounding", divider="rainbow")

st.markdown("""
Cohere specializes in enterprise RAG and multilingual business search. Models like `command-r-plus` 
are specifically tuned for multi-step tool use, enterprise connectors, and **verifiable citation generation**.

LangChain's `ChatCohere` seamlessly integrates with Cohere's citation endpoints, returning grounded evidence references in the response payload.
""")

st.code('''from langchain_cohere import ChatCohere

# Initializing Cohere Command R+
model_cohere = ChatCohere(
    model="command-r-plus",
    temperature=0.3
)
''', language="python")

# --- SECTION 8: MISTRAL ---
st.header("8. Mistral: ChatMistralAI, Mistral Large & Specialized Code Engines", divider="rainbow")

st.markdown("""
Mistral AI provides state-of-the-art open-weight and hosted frontier models, including `mistral-large`, 
`mistral-small`, and `codestral` (specialized for code generation).

`ChatMistralAI` features native function calling, JSON output modes, and excellent multi-lingual performance across European languages.
""")

st.code('''from langchain_mistralai import ChatMistralAI

# Instantiating Mistral Large
model_mistral = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.1
)
''', language="python")

# --- SECTION 9: GROQ ---
st.header("9. Groq: ChatGroq & Ultra-Low-Latency LPU Real-Time Inference", divider="rainbow")

st.markdown("""
Groq uses custom **Language Processing Units (LPUs)** to deliver astonishing inference speeds (frequently 
exceeding 300 to 700 tokens per second for Llama 3 models). 

This ultra-low latency makes `ChatGroq` the premier choice for voice agents, real-time typing feedback, 
and low-latency agentic loops.
""")

st.code('''from langchain_groq import ChatGroq

# Ultra-fast Llama 3.3 70B running on Groq LPUs
model_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024
)
''', language="python")

# --- SECTION 10: OLLAMA ---
st.header("10. Ollama: ChatOllama & Private Local Air-Gapped Deployments", divider="rainbow")

st.markdown("""
For sensitive data, local development, or air-gapped secure enclaves, **Ollama** runs foundation models 
locally on macOS, Linux, and Windows hardware with zero external network connectivity and zero API costs.

LangChain's `ChatOllama` allows your exact same LCEL pipeline to run completely locally.
""")

st.code('''from langchain_ollama import ChatOllama

# Local air-gapped model running on localhost:11434
model_ollama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,
    base_url="http://localhost:11434"
)
''', language="python")

# --- SECTION 11: HUGGING FACE ---
st.header("11. Hugging Face: Open-Source Ecosystem & Inference Endpoints", divider="rainbow")

st.markdown("""
Hugging Face hosts tens of thousands of open-weight models. LangChain integrates with Hugging Face via 
`HuggingFaceEndpoint` (for serverless inference or dedicated Hugging Face Dedicated Endpoints running vLLM or TGI) 
and wraps them in `ChatHuggingFace` to ensure full chat message compatibility.
""")

st.code('''from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Connecting to a Hugging Face Text Generation Inference (TGI) endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512
)
model_hf = ChatHuggingFace(llm=llm_endpoint)
''', language="python")

# --- SECTION 12: OTHER COMMUNITY INTEGRATIONS ---
st.header("12. Other Community Integrations: Together AI, Fireworks, Perplexity & DeepInfra", divider="rainbow")

st.markdown("""
Beyond the top cloud platforms, the LangChain community maintains partner adapters for high-performance 
model aggregators:
- **Together AI** (`langchain-together`): Deeply discounted open-source model hosting.
- **Fireworks AI** (`langchain-fireworks`): Ultra-low latency fine-tuned model serving.
- **Perplexity** (`langchain-community.chat_models.ChatPerplexity`): Online web-grounded search models.
- **DeepInfra**: Serverless, pay-per-token open-weight hosting.

Because all of these implement `BaseChatModel`, swapping from OpenAI to Together AI requires changing only 
a single line of configuration.
""")

st.code('''# Example: Swapping models dynamically via unified factory
from langchain.chat_models import init_chat_model

# Seamless switch across providers with zero pipeline refactoring:
# llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
# llm = init_chat_model("accounts/fireworks/models/llama-v3-70b-instruct", model_provider="fireworks")
''', language="python")

# --- SECTION 13: PRODUCTION IMPLEMENTATION ---
st.header("13. Production Implementation: Unified Multi-Provider Gateway with Automated Fallbacks", divider="rainbow")

st.markdown("""
Below is an enterprise-grade multi-provider gateway that initializes primary and fallback models, 
handles dynamic routing based on query complexity, and guarantees zero-downtime failover.
""")

st.code('''from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

class ProductionMultiCloudGateway:
    """Enterprise multi-cloud model gateway with automatic failover and tiered routing."""

    def __init__(self, openai_api_key: str, anthropic_api_key: str, groq_api_key: str):
        # Tier 1 (Ultra-Fast / Low Cost): Groq LPU
        self.fast_tier = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            timeout=8.0,
            api_key=groq_api_key
        )

        # Tier 2 (Primary High-Intelligence): OpenAI GPT-4o
        self.frontier_primary = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=12.0,
            max_retries=1,
            api_key=openai_api_key
        )

        # Tier 3 (Frontier Fallback): Anthropic Claude 3.5 Sonnet
        self.frontier_fallback = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            timeout=12.0,
            max_retries=2,
            api_key=anthropic_api_key
        )

        # Resilient frontier chain with zero-downtime fallback
        self.resilient_frontier: RunnableWithFallbacks = self.frontier_primary.with_fallbacks(
            [self.frontier_fallback]
        )

    def route_and_execute(self, user_query: str, require_high_reasoning: bool = False) -> Dict[str, Any]:
        """Routes query based on reasoning requirements and failover boundaries."""
        messages = [
            SystemMessage(content="You are an enterprise technical advisory assistant."),
            HumanMessage(content=user_query)
        ]

        if not require_high_reasoning:
            # Use ultra-fast LPU tier for latency-sensitive tasks
            ai_msg = self.fast_tier.invoke(messages)
            tier_used = "Ultra-Fast (Groq LPU)"
        else:
            # Use resilient frontier tier with automated fallback
            ai_msg = self.resilient_frontier.invoke(messages)
            tier_used = "Frontier Resilient (OpenAI / Anthropic)"

        resp_meta = ai_msg.response_metadata or {}
        usage = ai_msg.usage_metadata or {}

        return {
            "tier_selected": tier_used,
            "resolved_model": resp_meta.get("model_name", "model_active"),
            "content": str(ai_msg.content),
            "tokens": {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0)
            }
        }
''', language="python")

# --- SECTION 14: INTERACTIVE WORKBENCH ---
st.header("14. Interactive Workbench: Multi-Provider Latency, Cost & Feature Matrix Evaluator", divider="rainbow")

st.markdown("""
Explore the interactive multi-provider evaluation lab below. Compare latency, pricing profiles, 
context windows, and multimodal capabilities across all 10 major providers, and test simulated 
multi-cloud routing in real-time.
""")

bench_engine = MultiProviderBenchmarkEngine()

st.subheader("📊 Cross-Provider Empirical Decision Matrix")
matrix_df = bench_engine.get_dataframe()
st.dataframe(matrix_df, use_container_width=True)

st.divider()

st.subheader("⚡ Live Multi-Provider Routing Simulation")
col_w1, col_w2 = st.columns(2)
with col_w1:
    selected_prov = st.selectbox(
        "Select Target Provider Endpoint",
        [p.provider for p in bench_engine.profiles],
        index=0
    )
    prompt_input = st.text_area(
        "User Prompt",
        value="Explain why adopting a multi-provider strategy mitigates cloud provider outage risks."
    )

with col_w2:
    st.markdown("**Enterprise Architecture Tip:**")
    st.info("""
    - For high-volume summarization and classification: Use **Groq** or **Gemini Flash**.
    - For complex code synthesis and tool orchestration: Use **OpenAI GPT-4o** or **Claude 3.5 Sonnet**.
    - For HIPAA/GDPR private VPC isolation: Use **Azure OpenAI** or **AWS Bedrock**.
    - For air-gapped local development: Use **Ollama**.
    """)

if st.button("🚀 Dispatch Request to Provider", type="primary"):
    sim_result = bench_engine.simulate_invocation(selected_prov, prompt_input)
    
    st.write("### Invocation Execution Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Provider", sim_result["provider"])
    m2.metric("Flagship Model", sim_result["model"])
    m3.metric("Simulated Latency", f"{sim_result['latency_ms']} ms")
    m4.metric("Estimated Cost", sim_result["estimated_cost_usd"])

    st.write("### Standardized Output Response (`AIMessage.content`)")
    st.code(sim_result["response_text"], language="markdown")

# --- SECTION 15: TECHNICAL INTERVIEW PREPARATION ---
st.header("15. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why did LangChain decouple providers into standalone packages like langchain-openai and langchain-anthropic?",
     "Decoupling into partner packages solves critical dependency bloat. In legacy monolithic architectures, installing LangChain pulled conflicting versions of pydantic, httpx, and vendor SDKs. Standalone packages keep core dependencies minimal, reduce Docker image footprint, and allow teams to upgrade specific provider SDKs independently."),
    ("How does AzureChatOpenAI differ from standard ChatOpenAI in initialization and routing?",
     "ChatOpenAI talks directly to api.openai.com using standard model names (e.g., 'gpt-4o'). AzureChatOpenAI talks to an Azure-hosted private endpoint and requires an azure_deployment parameter (the customer's deployed model alias in Azure Portal) and an api_version string, supporting Azure Entra ID token auth."),
    ("What makes Groq LPUs fundamentally faster than standard GPU-based inference endpoints?",
     "Groq uses Language Processing Units (LPUs) with deterministic Tensor Streaming Architecture. Unlike GPUs that suffer from memory bandwidth bottlenecks during autoregressive token generation, LPUs store model parameters in ultra-high-speed on-chip SRAM, achieving 300 to 700+ tokens per second for real-time applications."),
    ("What are the security and compliance benefits of deploying ChatOllama vs commercial APIs?",
     "ChatOllama executes models completely on local CPU/GPU hardware. No prompts, embeddings, or proprietary corporate data ever traverse the public internet or external third-party servers. This satisfies strict air-gapped regulatory regimes (defense, banking, healthcare) with zero ongoing API licensing costs."),
    ("How does ChatAnthropic implement prompt caching to decrease inference cost and latency?",
     "Anthropic allows developers to flag system prompts or documents with a cache control breakpoint. The server saves the computed KV-cache of those tokens for 5 minutes. Subsequent queries reusing that prompt segment achieve up to 90% cost discounts and drastically lower time-to-first-token.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Model Name Instead of Deployment Name in Azure. Passing `model='gpt-4o'` to `AzureChatOpenAI` instead of `azure_deployment='your-deployment-name'` will result in an immediate 404 Resource Not Found error from Azure API gateways.")
    st.error("**Trap 2 (Breaking Error)**: Unhandled Timeout Latency Cascades. Chaining three providers in `.with_fallbacks()` without setting aggressive `timeout` values (e.g., leaving default 60s) causes client HTTP requests to hang for over 3 minutes during an outage.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Silent Cloud Egress Costs with Bedrock/Azure. Deploying your LangChain backend on AWS while querying Azure OpenAI incurs substantial cross-cloud egress bandwidth charges. Colocate your application containers in the same cloud region as your model endpoints.")
    st.warning("**Trap 4 (Feature Warning)**: Assuming Universal Tool Calling Support. Not all open-weight models hosted on community endpoints support JSON tool calling reliably. Verify that the chosen model has native function calling checkpoints (e.g., Llama-3-Instruct or Mistral-Large).")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Azure Model Parameter"):
        st.markdown("**AzureChatOpenAI requires azure_deployment and azure_endpoint instead of model.**")
    with st.expander("🎴 Flashcard 2: AWS Bedrock Authentication"):
        st.markdown("**ChatBedrock uses ambient AWS IAM credentials and boto3 sessions without API keys.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Ultra-Fast Inference Provider"):
        st.markdown("**ChatGroq uses LPUs for sub-100ms time-to-first-token and 500+ tokens/second throughput.**")
    with st.expander("🎴 Flashcard 4: Air-Gapped Local Inference"):
        st.markdown("**ChatOllama runs locally on localhost:11434 with zero external network access.**")

st.divider()

# --- SECTION 16: SELF-ASSESSMENT CHECKLIST ---
st.header("16. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural purpose of LangChain's decoupled partner packages.", value=True)
chk2 = st.checkbox("I can configure AzureChatOpenAI using deployment names and private endpoints.", value=True)
chk3 = st.checkbox("I know how to authenticate ChatBedrock using AWS IAM credentials.", value=True)
chk4 = st.checkbox("I understand when to deploy Groq for latency vs Claude/GPT-4o for complex reasoning.", value=True)
chk5 = st.checkbox("I can construct a multi-provider fallback gateway using .with_fallbacks().", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 3.1 Model Providers. Ready for 3.2 Model Configuration!")
