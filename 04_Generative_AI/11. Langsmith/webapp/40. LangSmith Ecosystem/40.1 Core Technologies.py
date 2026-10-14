import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="40.1 Core Technologies - LangSmith Ecosystem",
    page_icon="🧩",
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
    .tech-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-status {
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
class CoreTechnologiesEcosystemEngine:
    """
    Simulates the interaction between LangChain, LangGraph, LangSmith,
    and ecosystem partner integrations.
    """
    def __init__(self):
        self.technologies = {
            "LangSmith": {
                "role": "Observability, Evaluation & Telemetry Control Plane",
                "core_primitives": ["Run", "Trace", "Dataset", "Experiment", "Feedback"],
                "best_for": "Production monitoring, LLM-as-a-judge regression tests, latency/cost SLIs.",
                "lifecycle_phase": "Day 2 Operations & Continuous Evaluation"
            },
            "LangChain": {
                "role": "Composable LCEL Pipeline & Retrieval Abstraction Framework",
                "core_primitives": ["RunnableSequence", "ChatPromptTemplate", "BaseRetriever", "StrOutputParser"],
                "best_for": "Linear RAG chains, document transformation, prompt templates, structured output parsing.",
                "lifecycle_phase": "Day 0 / Day 1 Prototyping & Pipeline Construction"
            },
            "LangGraph": {
                "role": "Cyclic State Machine & Multi-Agent Orchestration Engine",
                "core_primitives": ["StateGraph", "add_messages reducer", "MemorySaver checkpointer", "interrupt()"],
                "best_for": "Multi-turn agents, human-in-the-loop workflows, error recovery loops, complex multi-agent teams.",
                "lifecycle_phase": "Day 1 Advanced Agent Architecture & Multi-Agent State"
            },
            "LangChain Integrations": {
                "role": "Standardized Connectors to 700+ AI Ecosystem Providers",
                "core_primitives": ["langchain-openai", "langchain-anthropic", "langchain-community", "langchain-chroma"],
                "best_for": "Vendor-agnostic switching between models, vector stores, and external APIs.",
                "lifecycle_phase": "Infrastructure Integration & Cross-Cloud Interoperability"
            }
        }

    def simulate_invocation_flow(self, framework_mode: str, query: str) -> Dict[str, Any]:
        latency_map = {
            "LangChain LCEL Linear Chain": 580,
            "LangGraph Cyclic Agent": 1820,
            "Full Stack: LangChain + LangGraph + LangSmith": 1890
        }
        spans_map = {
            "LangChain LCEL Linear Chain": ["ChatPromptTemplate", "ChatOpenAI", "StrOutputParser"],
            "LangGraph Cyclic Agent": ["StateGraph:planner", "tool:search_docs", "StateGraph:synthesizer"],
            "Full Stack: LangChain + LangGraph + LangSmith": [
                "LangSmith:RootTrace", "LangGraph:StateGraph", "LangChain:Prompt",
                "LangChain:ChatModel", "Tool:SearchAPI", "LangSmith:RunEvaluator"
            ]
        }
        return {
            "mode": framework_mode,
            "query": query,
            "latency_ms": latency_map.get(framework_mode, 800),
            "trace_spans": spans_map.get(framework_mode, ["run"]),
            "status": "SUCCESS"
        }

# Instantiate engine
tech_engine = CoreTechnologiesEcosystemEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Modern LangChain, LangGraph, & LangSmith Triad Ecosystem", divider="rainbow")

st.markdown("""
The modern generative AI engineering stack has converged around three complementary layers:
**LangChain** provides composable building blocks and standardized interfaces; **LangGraph** provides 
stateful cyclic orchestration for multi-agent workflows; and **LangSmith** acts as the observability, 
evaluation, and control plane that wraps both runtime layers with transparent telemetry.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Core_Technologies_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_orchestration {
            label = "Agentic & Pipeline Execution Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            langchain_lcel [label="LangChain v0.3+ (LCEL)\\n(Runnables, Prompts, Output Parsers, Document Loaders)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            langgraph_sm [label="LangGraph State Engine\\n(StateGraph, Cyclic Transitions, Reducers, Checkpointers)", fillcolor="#1E1B4B", fontcolor="#A78BFA", color="#818CF8"];
        }

        subgraph cluster_integrations {
            label = "Ecosystem Integrations Layer (700+ Partner Packages)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            partner_models [label="Model Providers\\n(OpenAI, Anthropic, Google, Ollama, vLLM)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            partner_vector [label="Data & Storage Providers\\n(Chroma, Pinecone, Qdrant, PostgreSQL, SQLite)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_observability {
            label = "LangSmith Observability & Governance Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            langsmith_tracing [label="LangSmith Tracing Engine\\n(Runs, Traces, Token Accounting, Latency Auditing)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            langsmith_eval [label="LangSmith Evaluation & Datasets\\n(Golden Benchmarks, LLM-as-a-Judge, CI/CD Gates)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            langsmith_hub [label="LangSmith Prompt Hub\\n(Versioned Prompts, Collaborative Playgrounds)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        langgraph_sm -> langchain_lcel [label="invokes runnable tools"];
        langchain_lcel -> partner_models [label="bind_tools / invoke"];
        langchain_lcel -> partner_vector [label="vector similarity search"];

        langgraph_sm -> langsmith_tracing [label="auto-emits node & state spans"];
        langchain_lcel -> langsmith_tracing [label="auto-emits LCEL run tree"];
        partner_models -> langsmith_tracing [label="token usage & TTFT metadata"];

        langsmith_eval -> langchain_lcel [label="benchmarks candidate chains"];
        langsmith_eval -> langgraph_sm [label="audits agent trajectories"];
        langsmith_hub -> langchain_lcel [label="pulls prompt templates"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Architectural Separation of Concerns:</strong> 
    <strong>LangChain</strong> is the library for pipeline composition; 
    <strong>LangGraph</strong> is the engine for stateful, cyclic multi-agent coordination; and 
    <strong>LangSmith</strong> is the standalone platform that monitors, tests, and debugs them both in production.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: LANGSMITH
# ==============================================================================
st.header("2. LangSmith: The Observability, Evaluation, and Telemetry Control Plane", divider="rainbow")

st.markdown("""
**LangSmith** is an enterprise-grade platform specifically engineered for the LLM application lifecycle. 
While traditional APMs (Datadog, New Relic) monitor CPU, memory, and generic HTTP routes, LangSmith understands 
the internal semantics of LLMs: token expenditures, prompt versions, intermediate tool reasoning, and non-deterministic text outputs.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Primary LangSmith Pillars
    1. **Full-Fidelity Tracing**:
       Captures every run, prompt template input, model completion, and tool execution in a hierarchical trace DAG without blocking the application runtime.
    2. **Versioned Golden Datasets**:
       Allows engineering teams to curate, tag, and version benchmark datasets for regression testing, fine-tuning, and few-shot curation.
    3. **Automated Evaluation Harness**:
       Executes reproducible experiments using deterministic metrics, semantic embedding distances, and structured LLM-as-a-judge rubrics.
    """)

with col2:
    st.markdown("""
    ### Enterprise Capabilities
    - **Prompt Hub**: Collaborative repository to version, fork, and test prompts with team members.
    - **Multi-Tenant Cost Attribution**: Dynamic model pricing catalogs that map token volumes to exact dollar expenditures grouped by customer tenant.
    - **Active Learning Feedback Loop**: Automatically captures production failures (user thumbs down or tool crashes) and pipes them into curation queues.
    - **Zero-Code Instrumentation**: Native callback handlers instrument LangChain and LangGraph processes automatically via environment variables.
    """)

# ==============================================================================
# SECTION 3: LANGCHAIN
# ==============================================================================
st.header("3. LangChain: The Composable LCEL Component and Retrieval Framework", divider="rainbow")

st.markdown("""
**LangChain** has evolved from an experimental collection of monolithic wrappers into a high-performance, 
type-safe composition engine built upon **LangChain Expression Language (LCEL)** and the `Runnable` protocol.
""")

st.markdown("""
### The LCEL Architecture & Standard Primitives
In modern LangChain (v0.3+), pipelines are composed using the pipe operator (`|`), creating declarative 
chains that support synchronous execution, asynchronous concurrency, real-time token streaming, and batching:
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Declarative LCEL Composition: Prompt | Model | Parser
prompt = ChatPromptTemplate.from_template("Summarize the following architecture: {input_text}")
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# The chain is a RunnableSequence conforming to standard Runnable interface
chain = prompt | model | parser

# Native support for .invoke(), .ainvoke(), .stream(), .astream(), and .batch()
result = chain.invoke({"input_text": "LangChain provides standard interfaces for LLM pipelines."})
print(result)
''', language="python")

st.markdown("""
### Key Functional Domains of LangChain
- **Document Loaders & Text Splitters**: Ingestion connectors for 100+ formats (PDF, Markdown, HTML, CSV) and semantic chunkers (RecursiveCharacterTextSplitter, SemanticChunker).
- **Embeddings & Vector Stores**: Standard abstractions for generating dense vectors and performing similarity queries across vector stores.
- **Retrievers**: Advanced retrieval logic including MultiQueryRetriever, EnsembleRetriever (hybrid BM25 + dense search), and ParentDocumentRetriever.
""")

# ==============================================================================
# SECTION 4: LANGGRAPH
# ==============================================================================
st.header("4. LangGraph: The Cyclic State Machine and Multi-Agent Orchestration Engine", divider="rainbow")

st.markdown("""
While LangChain LCEL excels at linear Directed Acyclic Graphs (DAGs), real-world AI agents require 
**cycles, persistent multi-turn memory, and human-in-the-loop validation**. 
**LangGraph** extends the ecosystem to support stateful multi-agent systems modeled as cyclic graphs.
""")

col_lg1, col_lg2 = st.columns(2)
with col_lg1:
    st.markdown("""
    ### Core Architectural Primitives
    - **StateGraph**: The graph definition parameterized by a central state schema (e.g., Pydantic model or TypedDict).
    - **Nodes**: Plain Python functions or Runnables that receive current state and return state deltas.
    - **Edges & Conditional Edges**: Routing logic that directs execution to the next node based on state properties (e.g., whether a tool call is requested).
    - **Channel Reducers**: Functions (e.g., `add_messages`) that control how node outputs are merged into state rather than overwriting it.
    """)

with col_lg2:
    st.markdown("""
    ### Enterprise Agent Capabilities
    - **Durable Persistence (Checkpointers)**: Saves graph state after every step to SQLite, PostgreSQL, or memory, enabling seamless thread resumption.
    - **Human-in-the-Loop (`interrupt()`)**: Pauses execution before sensitive actions (e.g., wire transfers, database writes) to await manual human approval.
    - **Time Travel & Rewind**: Allows inspecting past state snapshots, modifying state variables, and replaying execution from any historical checkpoint.
    """)

lg_dot = """
digraph LangGraphCyclic {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4, fontname="Helvetica"];
    node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10];

    entry [label="Entry Point", fillcolor="#0F172A", fontcolor="#F8FAFC"];
    agent_node [label="Agent Reasoner Node\\n(Calls LLM with bind_tools)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
    cond_router [label="Conditional Router\\n(Check: Tool calls requested?)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
    tool_node [label="Tool Executor Node\\n(Executes search, SQL, or API)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
    end_node [label="END (Goal Achieved)", fillcolor="#0F172A", fontcolor="#F8FAFC"];

    entry -> agent_node;
    agent_node -> cond_router;
    cond_router -> tool_node [label="tools present (CYCLE)"];
    cond_router -> end_node [label="no tools / response ready"];
    tool_node -> agent_node [label="appends ToolMessage to state"];
}
"""
st.graphviz_chart(lg_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: LANGCHAIN INTEGRATIONS
# ==============================================================================
st.header("5. LangChain Integrations: 700+ Ecosystem Providers, Tool Calling, and Standardized Interfaces", divider="rainbow")

st.markdown("""
To prevent a bloated monolithic package, modern LangChain split into dedicated, independently-versioned 
partner packages under the `@langchain` umbrella. This modular architecture allows fast updates to provider 
SDKs without breaking core dependencies.
""")

integrations_df = pd.DataFrame([
    {"Package": "langchain-core", "Role": "Base Interfaces & Types", "Key Exports": "Runnable, BaseMessage, ChatPromptTemplate, StrOutputParser", "Dependencies": "Zero heavy dependencies (pure Python & Pydantic)"},
    {"Package": "langchain", "Role": "Chains & Retrieval Workflows", "Key Exports": "create_retrieval_chain, create_history_aware_retriever", "Dependencies": "Lightweight composition layer"},
    {"Package": "langchain-community", "Role": "Third-Party Community Connectors", "Key Exports": "Hundreds of document loaders, vector stores, and custom tools", "Dependencies": "Optional extras per integration"},
    {"Package": "langchain-openai", "Role": "Dedicated OpenAI Partner Package", "Key Exports": "ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI", "Dependencies": "openai>=1.0"},
    {"Package": "langchain-anthropic", "Role": "Dedicated Anthropic Partner Package", "Key Exports": "ChatAnthropic", "Dependencies": "anthropic>=0.18"},
    {"Package": "langchain-google-genai", "Role": "Google Gemini Partner Package", "Key Exports": "ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings", "Dependencies": "google-generativeai"}
])
st.dataframe(integrations_df, use_container_width=True)

# ==============================================================================
# SECTION 6: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("6. Production Implementation: Full Multi-Technology Integration Pipeline", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating how **LangChain**, **LangGraph**, 
and **LangSmith** work together in a unified, traced pipeline.
""")

st.code('''import os
from typing import TypedDict, Annotated, List
from langsmith import Client, traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Environment & Tracing Activation
# LANGSMITH_TRACING="true" automatically hooks into LangChain & LangGraph
client = Client()

# 2. LangChain LCEL Component Definition
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise AI architect. Provide precise, technical summaries."),
    ("human", "{request}")
])

# Simulated model response generator to ensure offline execution resilience
@traceable(run_type="llm", name="mock_enterprise_chat_model")
def mock_chat_model(formatted_prompt: str) -> str:
    return "LangSmith, LangChain, and LangGraph form the unified production GenAI operating stack."

# 3. LangGraph Cyclic State Engine
class AgentState(TypedDict):
    request: str
    intermediate_summary: str
    status: str

@traceable(run_type="chain", name="graph_summarizer_node")
def summarizer_node(state: AgentState) -> dict:
    """Executes LangChain LCEL step inside LangGraph node."""
    formatted = prompt.format(request=state["request"])
    summary = mock_chat_model(formatted)
    return {"intermediate_summary": summary, "status": "SUMMARIZED"}

@traceable(run_type="chain", name="graph_validator_node")
def validator_node(state: AgentState) -> dict:
    """Audits summary quality and prepares final output."""
    return {"status": "COMPLETED"}

# 4. Orchestrating Multi-Technology Request
@traceable(run_type="chain", name="enterprise_triad_orchestrator")
def run_triad_pipeline(user_request: str) -> AgentState:
    state: AgentState = {"request": user_request, "intermediate_summary": "", "status": "INIT"}
    
    # Step 1: Execute summarizer
    s1 = summarizer_node(state)
    state.update(s1)
    
    # Step 2: Execute validator
    s2 = validator_node(state)
    state.update(s2)
    
    return state

# Execute Traced Pipeline
final_result = run_triad_pipeline("How do the LangChain core technologies fit together?")
print("Execution Result:", final_result)
''', language="python")

# ==============================================================================
# SECTION 7: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("7. Interactive Workbench: Core Technology Ecosystem Simulator & Component Visualizer", divider="rainbow")

st.markdown("""
Explore how the different technologies in the LangSmith ecosystem interact, 
compare their functional roles, and simulate execution traces across pipeline configurations.
""")

wb_view = st.selectbox(
    "Select Ecosystem Tool:",
    ["1. Technology Role & Primitives Comparator", "2. End-to-End Invocation Trace Simulator", "3. Architectural Stack Advisor"]
)

if wb_view == "1. Technology Role & Primitives Comparator":
    st.subheader("Core Technology Comparison")
    selected_tech = st.selectbox("Choose Technology:", list(tech_engine.technologies.keys()))
    t_info = tech_engine.technologies[selected_tech]

    st.markdown(f"### {selected_tech}")
    st.info(f"**Primary Role:** {t_info['role']}")
    st.markdown(f"**Best For:** {t_info['best_for']}")
    st.markdown(f"**Lifecycle Phase:** `{t_info['lifecycle_phase']}`")

    st.markdown("#### Foundational Primitives:")
    for p in t_info["core_primitives"]:
        st.markdown(f"- `{p}`")

elif wb_view == "2. End-to-End Invocation Trace Simulator":
    st.subheader("Trace Span Generation Simulator")
    sim_mode = st.selectbox("Pipeline Architecture:", [
        "LangChain LCEL Linear Chain",
        "LangGraph Cyclic Agent",
        "Full Stack: LangChain + LangGraph + LangSmith"
    ])
    user_q = st.text_input("Test Query:", "Explain how checkpointers persist state in LangGraph.")

    if st.button("Simulate Pipeline Execution"):
        trace_data = tech_engine.simulate_invocation_flow(sim_mode, user_q)
        st.markdown(f"**Architecture Mode:** `{trace_data['mode']}`")
        st.markdown(f"**Execution Latency:** `{trace_data['latency_ms']}ms` | **Status:** `{trace_data['status']}`")

        st.markdown("#### Generated Trace Spans:")
        for idx, span in enumerate(trace_data["trace_spans"], 1):
            st.markdown(f"`{idx}. {span}`")
        st.success("All spans successfully recorded into LangSmith background queue.")

else:
    st.subheader("Architectural Stack Advisor")
    st.markdown("Select your application requirements to determine the optimal stack combination:")

    req_cycles = st.selectbox("Does your system require cyclic execution or self-correcting tool loops?", ["No (Linear execution)", "Yes (Cyclic agent)"])
    req_memory = st.selectbox("Do you need persistent multi-turn thread memory with human checkpoints?", ["No (Stateless API)", "Yes (Durable persistence)"])
    req_tracing = st.selectbox("Do you require production latency/cost tracking and regression testing?", ["Yes (Mandatory)", "No (Toy script)"])

    st.markdown("#### Recommended Architecture:")
    if req_cycles == "Yes (Cyclic agent)" or req_memory == "Yes (Durable persistence)":
        st.success("Recommended Stack: **LangGraph + LangSmith + langchain-core**")
        st.caption("Use LangGraph for the cyclic state machine and checkpointer, with LangSmith providing trace visibility into every node transition.")
    else:
        st.info("Recommended Stack: **LangChain LCEL + LangSmith**")
        st.caption("A lean LCEL chain (`prompt | model | parser`) is optimal for linear pipelines, avoiding state machine overhead while retaining full LangSmith observability.")

# ==============================================================================
# SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do LangChain, LangGraph, and LangSmith differ in their architectural responsibilities?"):
    st.markdown("""
    **Answer:**  
    - **LangChain** is a library for composing linear LLM pipelines (LCEL), abstracting prompt templates, model wrappers, output parsers, and retrieval connectors.
    - **LangGraph** is a multi-agent state orchestration framework that models applications as cyclic state machines with durable checkpointers and human-in-the-loop controls.
    - **LangSmith** is an observability, testing, and evaluation platform that tracks, debugs, and benchmarks both LangChain and LangGraph applications.
    """)

with st.expander("Q2: Why was modern LangChain refactored to separate langchain-core, langchain, and partner packages?"):
    st.markdown("""
    **Answer:**  
    Monolithic LangChain previously suffered from heavy dependency conflicts and slow release cycles. 
    By separating `langchain-core` (lightweight, zero-heavy dependencies), `langchain` (chain workflows), 
    and dedicated partner packages (`langchain-openai`, `langchain-anthropic`), teams can install only what they need, 
    reducing Docker image sizes and eliminating dependency version locks.
    """)

with st.expander("Q3: How does LangSmith automatically trace LangGraph workflows without manual instrumentation?"):
    st.markdown("""
    **Answer:**  
    LangGraph inherits from LangChain's core `Runnable` interface. When `LANGSMITH_TRACING=true` is set, 
    the LangSmith callback handler is injected into the execution context. Every time a graph node executes, 
    a conditional edge resolves, or a state reducer runs, LangGraph triggers the callback, emitting child 
    run spans to LangSmith with state snapshots attached as run metadata.
    """)

with st.expander("Q4: Can LangSmith be used without LangChain or LangGraph?"):
    st.markdown("""
    **Answer:**  
    Yes. LangSmith is completely framework-agnostic. Non-LangChain applications (e.g., raw OpenAI SDK, Anthropic SDK, 
    LlamaIndex, custom PyTorch models, or plain Python scripts) can be fully traced using the `@traceable` decorator 
    or the low-level `RunTree` API.
    """)

with st.expander("Q5: What is the purpose of channel reducers in LangGraph state schemas?"):
    st.markdown("""
    **Answer:**  
    By default, updating a key in LangGraph overwrites its value with the node's output. 
    Channel reducers (such as `Annotated[List[BaseMessage], add_messages]`) define custom merge operations. 
    The `add_messages` reducer appends new messages to conversation history while intelligently updating existing 
    messages with matching IDs.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Installing the Monolithic 'langchain' Package in Lean Microservices**  
Installing `pip install langchain` pulls hundreds of optional dependencies. 
In production microservices, install only `langchain-core` and specific partner packages (`langchain-openai`).
""")

st.error("""
**Trap 2: Forcing Linear LCEL Chains into Cyclic Agent Workflows**  
Attempting to build multi-turn error recovery loops using recursive `RunnableSequence` chains leads to unmaintainable 
spaghetti code. Use LangGraph whenever cycles, state retention, or branching are required.
""")

st.warning("""
**Trap 3: Mixing Multiple Model Provider SDKs Without Unified LangChain Abstractions**  
Calling the raw OpenAI client in one module and raw Anthropic client in another creates inconsistent output schemas, 
fragmented token counters, and divergent error handling. Standardize on LangChain chat model interfaces.
""")

st.warning("""
**Trap 4: Neglecting to Version Prompts via LangSmith Prompt Hub**  
Hardcoding prompt strings into application code makes it impossible to update prompts without redeploying code. 
Use LangSmith Prompt Hub to manage prompt templates collaboratively.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>LCEL (LangChain Expression Language)</h4>
        <p>A declarative composition syntax using the pipe operator (|) to construct streaming, async-ready Runnable pipelines.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>StateGraph</h4>
        <p>The LangGraph cyclic state machine primitive that coordinates nodes, conditional edges, and state reducers.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>langchain-core</h4>
        <p>The minimal, high-performance foundation package providing the Runnable protocol, messages, and prompt templates.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Durable Checkpointer</h4>
        <p>A LangGraph persistence mechanism that serializes graph state at each step to disk or database for fault tolerance.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("9. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can articulate the architectural boundary separating LangChain, LangGraph, and LangSmith.", value=True)
chk_2 = st.checkbox("I understand modern LCEL composition using the pipe operator (|) and Runnable interfaces.", value=True)
chk_3 = st.checkbox("I know how LangGraph models cyclic agents using StateGraph, reducers, and checkpointers.", value=True)
chk_4 = st.checkbox("I understand the modular package structure of langchain-core, langchain, and partner packages.", value=True)
chk_5 = st.checkbox("I know how LangSmith transparently instruments both LangChain and LangGraph via callback hooks.", value=True)
chk_6 = st.checkbox("I can explain when to choose a simple LCEL chain versus a cyclic LangGraph state machine.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangSmith Core Technologies ecosystem mastery verified! Ready for LLM Providers.")
