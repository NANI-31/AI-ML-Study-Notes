import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.1 What is LangChain? - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.1 What is LangChain? Foundations, Ecosystem & LCEL")
st.caption("Chapter 1: Introduction to LangChain | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
A foundation Large Language Model (LLM) is an extraordinary probabilistic reasoning engine, but in isolation 
it operates in a vacuum: it cannot execute real-time web searches, query internal SQL databases, produce guaranteed 
strongly-typed JSON schemas, or chain multi-step logic without fragile boilerplate.

**LangChain (v0.3+)** is the industry-standard software engineering framework designed to bridge foundation models 
with production applications. Through the **LangChain Expression Language (LCEL)**, unified `Runnable` protocols, 
and seamless integrations across model providers, LangChain transforms raw LLMs into modular, observable, and 
fault-tolerant cognitive pipelines.

In this lesson, you will master:
- **Core Orchestration Primitives**: Why declarative composition replaces brittle imperative glue code.
- **The Modern Ecosystem**: How LangChain Core, LangGraph, LangSmith, and Deep Agents interact.
- **Architectural Trade-offs**: Direct LLM SDKs versus unified abstraction layers.
- **Production Decision Matrix**: Exactly when to adopt LangChain and when to stick to direct API calls.
- **Systems Interview Readiness**: High-frequency interview Q&As, production failure modes, and active recall.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockChatModel:
    """Simulates multi-provider ChatModel behavior in module scope."""
    def __init__(self, provider: str, model_name: str, temperature: float):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, prompt_text: str) -> Dict[str, Any]:
        simulated_latencies = {
            "OpenAI (gpt-4o)": 420,
            "Anthropic (claude-3-5-sonnet)": 480,
            "Google (gemini-1.5-pro)": 390,
            "Meta (llama-3.1-70b-instruct)": 260
        }
        latency = simulated_latencies.get(f"{self.provider} ({self.model_name})", 350)
        
        # Deterministic mock response based on input
        category = "Strategic Advisory" if "strategy" in prompt_text.lower() else "Operational Execution"
        confidence = round(0.92 - (self.temperature * 0.15), 2)
        
        return {
            "provider": self.provider,
            "model": self.model_name,
            "latency_ms": latency,
            "token_usage": {"input_tokens": 128, "output_tokens": 84, "total_tokens": 212},
            "output_payload": {
                "category": category,
                "confidence_score": confidence,
                "summary": f"Synthesized analysis for query: '{prompt_text[:40]}...'",
                "actionable_next_step": "Dispatch tool execution to verified data store."
            }
        }

class LCELWorkbenchPipeline:
    """Mock LCEL RunnableSequence simulator executing in real module scope."""
    def __init__(self, model: MockChatModel):
        self.model = model

    def run_pipeline(self, user_query: str, system_role: str) -> Dict[str, Any]:
        # Step 1: Prompt formatting
        formatted_prompt = f"Role: {system_role}\nQuery: {user_query}"
        # Step 2: Model invocation
        raw_result = self.model.invoke(formatted_prompt)
        # Step 3: Output parser extraction
        parsed_data = raw_result["output_payload"]
        
        return {
            "pipeline_status": "SUCCESS",
            "provider_active": raw_result["provider"],
            "model_active": raw_result["model"],
            "latency_ms": raw_result["latency_ms"],
            "tokens": raw_result["token_usage"],
            "parsed_contract": parsed_data
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern LangChain v0.3+ Composition Ecosystem", divider="rainbow")

dot_blueprint = """
digraph LangChainEcosystem {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_ingress {
        label="1. Ingress & Context Assembly Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        user_input [label="Client Application / Event Trigger\\n{'user_id': 1024, 'query': 'Analyze audit logs'}", fillcolor="#0F172A"];
        prompt_template [label="ChatPromptTemplate\\nSystemMessagePrompt + HumanMessagePrompt", fillcolor="#1E293B"];
    }

    subgraph cluster_lcel_core {
        label="2. LCEL Composition Core (Runnable Protocol |)";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        runnable_seq [label="RunnableSequence (| Pipe Operator)\\nSynchronous (.invoke), Streaming (.stream), Async (.ainvoke)", fillcolor="#1E293B"];
        unified_model [label="BaseChatModel Abstraction\\n(OpenAI / Anthropic / Google / Groq / Ollama)", fillcolor="#1E293B"];
        tool_binding [label="Tool Calling Interface (.bind_tools)\\nJSON Schema Extraction & Function Registry", fillcolor="#1E293B"];
        output_parser [label="Structured Output Parser / with_structured_output\\nPydantic v2 Type Enforcement & Validation", fillcolor="#1E293B"];
    }

    subgraph cluster_orchestration {
        label="3. Extended Orchestration & Observability Tier";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        langgraph [label="LangGraph State Engine\\nCyclic Multi-Agent Graphs, Persistence & Human-in-the-Loop", fillcolor="#064E3B"];
        langsmith [label="LangSmith Telemetry Platform\\nDistributed Tracing, Token Accounting & Latency Profiling", fillcolor="#064E3B"];
    }

    user_input -> prompt_template [label="Params Dict"];
    prompt_template -> runnable_seq [label="PromptValue"];
    runnable_seq -> unified_model [label="Dispatches Messages"];
    unified_model -> tool_binding [label="Tool Call Request", dir=both];
    unified_model -> output_parser [label="AIMessage"];
    output_parser -> langgraph [label="State Updates"];
    runnable_seq -> langsmith [label="Async Tracing Spans", style=dotted, color="#F59E0B"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTION 2: DEFINITION OF LANGCHAIN ---
st.header("2. Definition of LangChain: Orchestration Framework for Foundation Models", divider="rainbow")

st.markdown("""
**LangChain** is an open-source development framework engineered to simplify the creation of applications 
powered by Large Language Models (LLMs). Rather than treating LLMs as standalone REST endpoints returning raw strings, 
LangChain models them as composable components within a broader computational architecture.

The core philosophy of modern LangChain (v0.3+) is centered on the **`Runnable` abstraction** and the **LangChain Expression 
Language (LCEL)**. Every component in LangChain—whether a prompt template, a chat model, an output parser, an external tool, 
or a retriever—implements a standardized interface with unified methods:
- `.invoke()`: Synchronous execution on a single input.
- `.ainvoke()`: Asynchronous non-blocking execution.
- `.stream()`: Streaming token-by-token output chunks to minimize perceived latency.
- `.astream_events()`: Real-time event streaming exposing intermediate steps, tool calls, and final tokens.
- `.batch()` / `.abatch()`: Parallelized concurrent execution over batches of inputs.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In legacy LangChain (v0.0.x / v0.1.x), chains were built using rigid, poorly typed classes such as `LLMChain`, 
`SequentialChain`, and `TransformChain`. These classes suffered from high debugging friction, black-box state mutability, 
and sluggish execution. 

Modern production stacks strictly mandate **LCEL composition (`prompt | model | parser`)** and **`init_chat_model()`** 
from `langchain.chat_models`. This provides full type annotations, native streaming support, and transparent tracing 
without opaque wrapper classes.
""")

df_def_compare = pd.DataFrame({
    "Dimension": ["Legacy LangChain (v0.0 - v0.1)", "Modern LangChain (v0.3+)", "Direct Vendor SDK"],
    "Architecture": ["Heavy monolithic classes (LLMChain)", "Modular primitives via LCEL (|)", "Raw REST API wrappers"],
    "Execution Protocol": ["Synchronous, opaque black-box", "Standardized Runnable interface", "Custom per-vendor async code"],
    "Streaming Support": ["Fragile callback handlers", "Native .stream() on every Runnable", "Vendor-specific event loops"],
    "Type Safety": ["Loose Dict[str, Any] outputs", "Pydantic v2 structured schemas", "TypedDict / JSON responses"]
})
st.table(df_def_compare)

# --- SECTION 3: WHY LANGCHAIN WAS CREATED ---
st.header("3. Why LangChain Was Created: Overcoming the Limitations of Raw Foundation Model APIs", divider="rainbow")

st.markdown("""
Foundation models like GPT-4, Claude 3.5, and Gemini are stateless, parametric text-completion engines. When developers 
initially began building applications using direct HTTP requests, severe recurring friction points emerged:

1. **Vendor API Incompatibility**: OpenAI used one schema for message formatting (`role: "developer"` / `"system"`), 
   Anthropic used another, and Google Gemini used a third (`Content(parts=[...])`). Migrating between providers required 
   rewriting complete application layers.
2. **Context Assembly Friction**: Injecting external database rows, system constraints, few-shot examples, and user variables 
   required messy Python f-string interpolation that frequently caused prompt injection vulnerabilities and escaping errors.
3. **Fragile Output Parsing**: Models returned unstructured Markdown text. Extracting a specific integer or boolean required 
   brittle regular expressions (`re.search()`), which broke whenever the model slightly altered its prose formatting.
4. **Tool Actuation Overhead**: Connecting an LLM to external APIs required manually parsing function calling arguments, 
   dispatching tool execution, capturing stderr/stdout, formatting observation messages, and re-invoking the model.

LangChain was created to establish a **unified, standardized abstraction layer** across all these primitives, allowing 
engineers to write model-agnostic application logic once and deploy it across any foundation model provider.
""")

# --- SECTION 4: WHAT PROBLEMS LANGCHAIN SOLVES ---
st.header("4. What Problems LangChain Solves: Context, Tool Actuation, and Standardized Interfaces", divider="rainbow")

st.markdown("""
LangChain solves four fundamental problems encountered when transitioning from a prototype script to an enterprise GenAI service:
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("1. Component Decoupling & Portability")
    st.markdown("""
    With LangChain, swapping your model from `gpt-4o` to `claude-3-5-sonnet` requires changing exactly **one line of code**:
    ```python
    model = init_chat_model("claude-3-5-sonnet", model_provider="anthropic")
    ```
    Your prompt templates, tool bindings, parsers, and error-handling chains remain completely unchanged.
    """)
    
    st.subheader("2. Strongly-Typed Structured Outputs")
    st.markdown("""
    LangChain guarantees that model responses conform to Pydantic v2 schemas using the `.with_structured_output()` interface. 
    It automatically builds the JSON schema, binds it to the provider's native tool calling API, and parses the returned JSON 
    into a typed Python object with validation error recovery.
    """)

with col_p2:
    st.subheader("3. Composable Tool Calling")
    st.markdown("""
    Defining tools in LangChain is as simple as adding the `@tool` decorator to a Python function. LangChain inspects type 
    hints and docstrings to generate OpenAPI-compliant JSON Schemas that the LLM understands natively.
    """)
    
    st.subheader("4. Production Observability Integration")
    st.markdown("""
    Through zero-code environment variable configuration (`LANGCHAIN_TRACING_V2="true"`), every pipeline execution, 
    token count, latency duration, and tool call payload is automatically published to LangSmith for distributed debugging.
    """)

# --- SECTION 5: LANGCHAIN VS DIRECT LLM APIS ---
st.header("5. LangChain vs Direct LLM APIs: Abstraction Layer vs Vendor Lock-In", divider="rainbow")

st.markdown("""
A common debate in AI engineering is whether to use an orchestration framework or write direct calls against vendor SDKs 
(`openai`, `anthropic`, `google-genai`). The decision depends on the architectural complexity of the target system.
""")

with st.expander("🔍 View Decision Flowchart: Direct API vs LangChain", expanded=False):
    dot_api = """
    digraph DecisionTree {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#334155"];
        edge [color="#38BDF8"];
        q1 [label="Is your app a single, static prompt\\nwith one provider?"];
        ans_direct [label="Use Direct Vendor SDK\\n(Zero dependencies, minimal overhead)", fillcolor="#064E3B"];
        q2 [label="Do you need multi-provider fallbacks,\\nretrieval pipelines, or tool calling?"];
        ans_langchain [label="Use Modern LangChain (LCEL)\\n(Unified Runnable API, model portability)", fillcolor="#78350F"];
        q1 -> ans_direct [label="Yes"];
        q1 -> q2 [label="No"];
        q2 -> ans_langchain [label="Yes"];
    }
    """
    st.graphviz_chart(dot_api, use_container_width=True)

df_api_compare = pd.DataFrame({
    "Criterion": ["Vendor Portability", "Boilerplate for Retries/Fallbacks", "Streaming Implementation", "Tool Calling Schema", "Dependency Footprint"],
    "Direct Vendor SDK": ["Zero (Hard vendor lock-in)", "Manual try/except loops per vendor", "Vendor-specific iterator handling", "Manual dict schema construction", "Lightweight (1 vendor client)"],
    "LangChain v0.3+": ["100% Portable across 50+ providers", "Native .with_fallbacks() and retries", "Standard .stream() across all models", "Automatic via @tool docstrings", "Modular (langchain-core + provider)"]
})
st.table(df_api_compare)

# --- SECTION 6: LANGCHAIN VS TRADITIONAL PYTHON APPLICATIONS ---
st.header("6. LangChain vs Traditional Python Applications: Non-Deterministic Pipelines vs Deterministic Code", divider="rainbow")

st.markdown("""
Traditional Python web applications (e.g. Django, FastAPI with PostgreSQL) operate on **deterministic computation**:
given input `X`, code path `Y` produces identical output `Z` every single time. Compilers and linters verify types, 
and database foreign keys guarantee data integrity.

In contrast, applications integrating LLMs are inherently **probabilistic and non-deterministic**:
- An identical prompt can yield differing phrasing, tone, or formatting across invocations.
- The execution time of an LLM call varies wildly based on output token length and server load.
- Output strings may hallucinate non-existent JSON keys or invalid dates.

LangChain provides the engineering guardrails necessary to interface probabilistic reasoning engines with deterministic 
software architectures through **output validation loops, schema enforcement, fallback chains, and timeout management**.
""")

# --- SECTION 7: THE COMPLETE LANGCHAIN ECOSYSTEM ---
st.header("7. The Complete LangChain Ecosystem: LangChain Core, LangGraph, LangSmith, and Deep Agents", divider="rainbow")

st.markdown("""
The modern LangChain ecosystem is partitioned into four distinct, decoupled architectural tiers:
""")

with st.expander("🔍 View Ecosystem Architecture Diagram (Top-to-Bottom Flow)", expanded=True):
    dot_eco = """
    digraph EcosystemLayers {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
        edge [color="#38BDF8", penwidth=1.5];

        subgraph cluster_layer1 {
            label="1. Foundation & Abstraction Layer";
            fontcolor="#38BDF8";
            style=dashed;
            color="#334155";
            core [label="langchain-core\\nBaseChatModel, BasePromptTemplate, Runnable, OutputParser\\n(Ultra-lightweight, zero dependency bloat)", fillcolor="#0F172A"];
            partner [label="Partner Packages (langchain-openai, langchain-anthropic)\\nHardware-optimized native provider integrations", fillcolor="#1E293B"];
        }

        subgraph cluster_layer2 {
            label="2. Composition & Assembly Layer";
            fontcolor="#F59E0B";
            style=dashed;
            color="#334155";
            lcel_pipe [label="LangChain (LCEL Pipelines)\\nDeclarative Chains, Retrievers, Text Splitters, Tool Wrappers", fillcolor="#1E293B"];
        }

        subgraph cluster_layer3 {
            label="3. Stateful Multi-Agent Runtime Layer";
            fontcolor="#10B981";
            style=dashed;
            color="#334155";
            langgraph_node [label="LangGraph Engine\\nCyclic State Machines, Branching, Memory Checkpointers, HITL", fillcolor="#064E3B"];
            deep_agents [label="Deep Agents / Swarms\\nAutonomous Goal Decomposition & Multi-Agent Teams", fillcolor="#064E3B"];
        }

        subgraph cluster_layer4 {
            label="4. Enterprise Observability & Evaluation Layer";
            fontcolor="#A855F7";
            style=dashed;
            color="#334155";
            langsmith_node [label="LangSmith Platform\\nTracing, Evaluation, LLM Unit Tests, Dataset Management", fillcolor="#4C1D95"];
        }

        core -> partner;
        partner -> lcel_pipe;
        lcel_pipe -> langgraph_node;
        langgraph_node -> deep_agents;
        lcel_pipe -> langsmith_node [style=dotted, color="#A855F7", label="Telemetry Spans"];
        langgraph_node -> langsmith_node [style=dotted, color="#A855F7", label="State Snapshots"];
    }
    """
    st.graphviz_chart(dot_eco, use_container_width=True)

st.markdown("""
1. **`langchain-core`**: The minimal foundational package containing the core interfaces (`Runnable`, `BaseChatModel`, `Messages`). It has zero heavy dependencies and installs in milliseconds.
2. **Partner Packages (`langchain-openai`, `langchain-anthropic`, etc.)**: Separate packages maintained directly with model providers to prevent version conflicts.
3. **`langgraph`**: The runtime for cyclic, multi-actor applications and state machines. When an application requires loops, memory checkpoints, or human approvals, developers use LangGraph.
4. **`langsmith`**: The DevOps control plane that records every invocation, tool call, token cost, and latency metric across your pipelines.
""")

# --- SECTION 8: KEY PRODUCTION ADVANTAGES OF LANGCHAIN ---
st.header("8. Key Production Advantages of LangChain: Model Abstraction, Structured Output, and Tooling", divider="rainbow")

st.markdown("""
When evaluated by enterprise engineering teams, LangChain provides six concrete operational advantages:
""")

adv_cols = st.columns(3)
with adv_cols[0]:
    st.info("**1. Zero-Cost Model Switching**\n\nAbstracts vendor-specific differences in message roles, token limits, and function calling syntax.")
    st.info("**2. Native Tool Calling Engine**\n\nTransforms Python functions into validated JSON Schema tools automatically via the `@tool` decorator.")
with adv_cols[1]:
    st.info("**3. Pydantic Structured Outputs**\n\nEliminates regex parsing by enforcing strict schema contracts directly at the model invocation layer.")
    st.info("**4. Declarative Streaming**\n\nEvery LCEL pipeline streams output tokens natively, drastically improving frontend user experience.")
with adv_cols[2]:
    st.info("**5. Production-Ready Fallbacks**\n\nSeamlessly fallback from a primary high-cost model to a low-cost backup on rate limits using `.with_fallbacks()`.")
    st.info("**6. Turnkey Tracing**\n\nInstant zero-instrumentation integration with LangSmith for distributed debugging and latency profiling.")

# --- SECTION 9: ARCHITECTURAL LIMITATIONS OF LANGCHAIN ---
st.header("9. Architectural Limitations of LangChain: Abstraction Overhead and Debugging Complexity", divider="rainbow")

st.markdown("""
Despite its industry dominance, LangChain is not a silver bullet. Senior architects must understand its real-world limitations:

1. **Abstraction Overhead (Leaky Abstractions)**: When a provider API changes or encounters a novel error code, LangChain's 
   abstraction layer may mask the underlying exception, making low-level HTTP debugging more challenging than using the raw SDK.
2. **Deep Stack Traces**: Because LCEL uses functional composition and extensive `Runnable` wrapping, Python stack traces 
   can span 30+ frames when an exception occurs inside a nested pipeline.
3. **Rapid Package Evolution**: LangChain evolved rapidly from v0.1 to v0.3. Tutorials online frequently reference deprecated 
   patterns (e.g. `langchain.chains.LLMChain`), leading to confusion for beginners who copy legacy code.
4. **Over-Engineering Risk**: For trivial single-turn tasks (e.g. summarizing a static paragraph with one fixed API key), 
   importing LangChain introduces unnecessary dependency weight compared to a 10-line direct client call.
""")

# --- SECTION 10: WHEN TO USE LANGCHAIN ---
st.header("10. When to Use LangChain: Ideal Architectures and Workloads", divider="rainbow")

st.markdown("""
Adopt LangChain (v0.3+) when your application architecture satisfies one or more of the following criteria:

- ✅ **Multi-Provider Strategy**: Your enterprise mandates vendor redundancy to avoid downtime when OpenAI or Anthropic suffers an outage.
- ✅ **Dynamic Tool Calling**: Your application connects LLMs to enterprise databases, web search, or calculators.
- ✅ **RAG & Context Injection**: You are building multi-stage retrieval pipelines combining retrievers, rerankers, and custom prompt templates.
- ✅ **Structured Data Extraction**: You are extracting structured entities (invoices, resumes, contracts) directly into Pydantic models.
- ✅ **Streaming User Interfaces**: Your application serves chat webapps requiring real-time token streaming with intermediate status indicators.
""")

# --- SECTION 11: WHEN NOT TO USE LANGCHAIN ---
st.header("11. When NOT to Use LangChain: Anti-Patterns and Over-Engineering Traps", divider="rainbow")

st.markdown("""
Avoid LangChain and stick to direct vendor SDKs or simple standard library Python under the following circumstances:

- ❌ **Trivial Single-Turn Prompts**: A microservice that takes text and calls a single hardcoded LLM model with no tools and no retrieval.
- ❌ **Severely Resource-Constrained Environments**: Edge devices or AWS Lambda micro-functions where cold-start latency and minimal wheel package size are paramount.
- ❌ **Custom Non-Standard Model Protocols**: Proprietary internal inference engines that do not conform to OpenAI-compatible or standard ChatModel protocols.
- ❌ **Simple Bash / CLI Scripts**: Quick one-off scripts where standard `urllib` or `requests` is sufficient.
""")

# --- SECTION 12: PRODUCTION IMPLEMENTATION ---
st.header("12. Production Implementation: End-to-End LCEL Runnable Pipeline", divider="rainbow")

st.markdown("""
The following complete, self-contained Python script demonstrates a production-grade LCEL composition pipeline using 
modern `langchain-core` syntax with structured fallback handling:
""")

st.code("""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# 1. Define Target Output Schema via Pydantic v2
class SecurityAuditReport(BaseModel):
    threat_level: str = Field(description="Severity: LOW, MEDIUM, HIGH, CRITICAL")
    vulnerability_detected: bool = Field(description="True if exploit or risk discovered")
    findings_summary: str = Field(description="Executive summary of security assessment")
    mitigation_action: str = Field(description="Prescribed remediation step")

# 2. Production LCEL Pipeline Architecture (Pseudo-code implementation)
'''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model

# Initialize model via unified factory function
primary_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.1)
backup_model = init_chat_model("claude-3-5-sonnet", model_provider="anthropic", temperature=0.1)

# Configure resilient fallback chain
resilient_model = primary_model.with_fallbacks([backup_model])

# Structured output binding
structured_llm = resilient_model.with_structured_output(SecurityAuditReport)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Principal Cloud Security Engineer. Audit server logs for security anomalies."),
    ("human", "Inspect following audit log segment:\\n{log_content}")
])

# Assemble declarative LCEL RunnableSequence
audit_pipeline = prompt | structured_llm

# Execute synchronous invocation
result: SecurityAuditReport = audit_pipeline.invoke({"log_content": "Failed root SSH login from 192.168.1.50"})
print(result.threat_level)
'''
""", language="python")

# --- SECTION 13: INTERACTIVE WORKBENCH ---
st.header("13. Interactive Workbench: LCEL Pipeline Simulator and Model Abstraction Inspector", divider="rainbow")

st.markdown("""
Experiment with the interactive LCEL simulator below. Toggle model providers, adjust generation temperature, 
and observe how the unified `Runnable` interface maintains a standard output contract across diverse model backends.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    selected_provider = st.selectbox(
        "Select Model Provider",
        ["OpenAI", "Anthropic", "Google", "Meta"],
        index=0
    )
    provider_models = {
        "OpenAI": "gpt-4o",
        "Anthropic": "claude-3-5-sonnet",
        "Google": "gemini-1.5-pro",
        "Meta": "llama-3.1-70b-instruct"
    }
    model_choice = provider_models[selected_provider]
    st.caption(f"Active Provider Model: `{model_choice}`")
    
    temp_val = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

with col_wb2:
    sys_role_input = st.text_input("System Role", value="Principal Enterprise Architect")
    user_query_input = st.text_area("User Query", value="Design a multi-region fallback strategy for enterprise LLM deployments.")

if st.button("🚀 Invoke LCEL Pipeline (.invoke)", type="primary"):
    # Instantiate module-scope mock model and pipeline
    mock_model = MockChatModel(selected_provider, model_choice, temp_val)
    pipeline = LCELWorkbenchPipeline(mock_model)
    
    execution_result = pipeline.run_pipeline(user_query_input, sys_role_input)
    
    st.write("### Pipeline Execution Telemetry")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Provider", execution_result["provider_active"])
    metric_cols[1].metric("Model", execution_result["model_active"])
    metric_cols[2].metric("Latency", f"{execution_result['latency_ms']} ms")
    metric_cols[3].metric("Total Tokens", execution_result["tokens"]["total_tokens"])
    
    st.write("### Strongly-Typed Output Contract")
    st.json(execution_result["parsed_contract"])

# --- SECTION 14: TECHNICAL INTERVIEW PREPARATION ---
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is LCEL and why did LangChain introduce it in place of LLMChain?",
     "LCEL (LangChain Expression Language) is a declarative syntax for chaining Runnables using the pipe operator (|). It replaced legacy LLMChain because it natively supports synchronous, asynchronous, batch, and streaming execution, exposes transparent intermediate state, provides type annotations, and eliminates opaque wrapper classes."),
    ("How does LangChain achieve vendor portability across OpenAI, Anthropic, and Gemini?",
     "LangChain implements the BaseChatModel interface. It standardizes message representations (SystemMessage, HumanMessage, AIMessage, ToolMessage), unifies tool calling formats into standard JSON Schema dictionaries, and normalizes generation responses into standard AIMessage objects with usage metadata."),
    ("What is the difference between LangChain and LangGraph?",
     "LangChain (via LCEL) is optimized for linear Directed Acyclic Graphs (DAGs) where execution flows sequentially or in parallel branches without cycles. LangGraph is built on top of LangChain to support cyclic state machines, multi-agent loops, long-term memory checkpointing, and Human-in-the-Loop interruptions."),
    ("How does `.with_structured_output()` enforce Pydantic schemas in production?",
     "It translates the Pydantic model into a JSON Schema, binds it to the provider's native tool calling / structured output endpoint, receives the model's raw JSON argument string, validates it against the Pydantic validator, and returns an instantiated Pydantic model. If validation fails, it can trigger retry logic."),
    ("What is the performance overhead of using LangChain compared to raw HTTP calls?",
     "In modern LangChain (v0.3+), core execution overhead is negligible (<2–5 milliseconds), as langchain-core consists of lightweight Python wrappers. The dominant latency in any pipeline is the network round-trip and LLM token generation time (typically 300ms to 2000ms).")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Copying Deprecated v0.0.x Tutorials. Importing `from langchain.chains import LLMChain` or using `predict()` pulls deprecated code with broken type hints. Always use `langchain_core` and LCEL.")
    st.error("**Trap 2 (Breaking Error)**: Unpinned Provider Dependencies. Upgrading `langchain-openai` without testing can introduce breaking parameter changes. Always pin exact versions in `requirements.txt`.")
with trap_c2:
    st.warning("**Trap 3 (Performance Warning)**: Blocking Async Loops. Calling `.invoke()` inside an `async def` FastAPI route blocks the entire Python event loop. Always use `.ainvoke()` or `.astream()` in async web servers.")
    st.warning("**Trap 4 (Cost Warning)**: Silent Retry Loops. Misconfiguring `.with_fallbacks()` or agent retry loops without exponential backoff and max retry caps can exhaust API rate limits and multiply token bills.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Runnable Interface Methods"):
        st.markdown("**.invoke() (Sync), .ainvoke() (Async), .stream() (Streaming), .batch() (Concurrent)**")
    with st.expander("🎴 Flashcard 2: LCEL Pipe Operator (|)"):
        st.markdown("**Composes Runnables into a RunnableSequence where output of component A becomes input of component B.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Unified Factory Method"):
        st.markdown("**init_chat_model('model_name', model_provider='provider') initializes any supported ChatModel.**")
    with st.expander("🎴 Flashcard 4: LangSmith Activation"):
        st.markdown("**Set LANGCHAIN_TRACING_V2='true' and LANGCHAIN_API_KEY in environment to enable distributed traces.**")

st.divider()

# --- SECTION 15: SELF-ASSESSMENT CHECKLIST ---
st.header("15. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can define LangChain and articulate why it was created.", value=True)
chk2 = st.checkbox("I understand the difference between legacy chains and modern LCEL Runnables.", value=True)
chk3 = st.checkbox("I understand the 4 pillars of the LangChain ecosystem (Core, LCEL, LangGraph, LangSmith).", value=True)
chk4 = st.checkbox("I know when to use LangChain versus when to write direct vendor SDK calls.", value=True)
chk5 = st.checkbox("I can construct a resilient LCEL pipeline with structured outputs and fallbacks.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 1.1 What is LangChain. Ready for 1.2 LangChain Architecture!")
