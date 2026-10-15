import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="1.3 LangGraph Ecosystem - LangGraph Curriculum",
    page_icon="🌐",
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
    .badge-eco {
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
# MODULE-SCOPE ECOSYSTEM CONFIGURATOR CLASS FOR WORKBENCH & PIPELINE
# ==============================================================================
class EcosystemStackConfigurator:
    """
    Evaluates enterprise stack compatibility and generates architectural topologies
    across the LangChain and LangGraph ecosystem.
    """
    def __init__(self):
        self.providers = {
            "OpenAI (GPT-4o)": {"in": 2.50, "out": 10.00, "tool_support": "Native Strict Tool Calling"},
            "Anthropic (Claude 3.5 Sonnet)": {"in": 3.00, "out": 15.00, "tool_support": "Prompt Caching & Computer Use"},
            "Ollama (Self-Hosted Llama 3.3)": {"in": 0.00, "out": 0.00, "tool_support": "Local Ollama JSON Schema"}
        }
        self.vectordbs = {
            "Qdrant": {"latency_ms": 12, "type": "Hybrid HNSW + BM25", "filter_efficiency": "Sub-millisecond payload filtering"},
            "Pinecone": {"latency_ms": 18, "type": "Managed Serverless", "filter_efficiency": "Metadata namespace isolation"},
            "ChromaDB": {"latency_ms": 8, "type": "In-Memory / Embedded", "filter_efficiency": "Local rapid prototyping"}
        }

    def evaluate_stack(self, provider: str, vectordb: str, enable_tracing: bool, deployment_mode: str) -> Dict[str, Any]:
        p_info = self.providers.get(provider, {})
        v_info = self.vectordbs.get(vectordb, {})
        
        latency = v_info.get("latency_ms", 15) + (350 if "Self-Hosted" not in provider else 180)
        observability = "LangSmith Full Trace Active" if enable_tracing else "Local Console Logging Only"
        
        return {
            "selected_provider": provider,
            "provider_tool_tier": p_info.get("tool_support", "Standard"),
            "selected_vectordb": vectordb,
            "vectordb_type": v_info.get("type", "Standard"),
            "estimated_roundtrip_ms": latency,
            "observability": observability,
            "deployment": deployment_mode,
            "production_readiness": "Enterprise Ready" if enable_tracing and deployment_mode != "Local Python Script" else "Prototype Stage"
        }


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("1.3 LangGraph Ecosystem: Runtimes, Platforms, Studio, & Enterprise Tooling")
st.caption("Chapter 1: Introduction to LangGraph | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
LangGraph does not exist in isolation—it sits at the epicenter of a comprehensive, production-grade enterprise AI ecosystem. 
While **LangGraph** provides the low-level cyclic state machine runtime, surrounding developer tools, deployment platforms, 
and partner integrations turn raw graphs into observable, scalable enterprise products.

The modern LangGraph ecosystem spans:
* **The Core Stack**: LangGraph (cyclic runtime) + LangChain (component library) + LangSmith (observability & evals).
* **Developer Tooling**: LangGraph Studio (visual desktop IDE for debugging graphs).
* **Deployment Platforms**: LangGraph Cloud, Self-Hosted Docker/Kubernetes runtime, and serverless scaling.
* **Integrations**: Model providers (OpenAI, Anthropic, Bedrock), Vector databases (Qdrant, Pinecone), Tool protocols (MCP), and Frontend SDKs (CopilotKit, Assistant UI).
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Complete LangGraph Enterprise Production Ecosystem", divider="rainbow")

st.markdown("""
The architecture below illustrates how the LangGraph ecosystem components interoperate across the application lifecycle: 
from local design in LangGraph Studio, to runtime orchestration, to deployment on LangGraph Platform, and deep observability in LangSmith.
""")

blueprint_dot = """
digraph EcosystemArchitecture {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_dev {
        label = "1. Development & Visual Debugging Tier";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        LangGraphStudio [label="LangGraph Studio (Desktop IDE)\\n(Visual Graph Canvas, State Inspector, Time-Travel)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        LangChainCore [label="LangChain Primitives\\n(PromptTemplates, OutputParsers, Pydantic Schemas)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_runtime {
        label = "2. Orchestration & State Machine Runtime";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        StateGraphEngine [label="LangGraph Core (StateGraph)\\n(Pregel Supersteps, Channel Reducers, Cyclic Edges)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        CheckpointerStore [label="Durable Checkpointers\\n(PostgresSaver / SqliteSaver / MemorySaver)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_integrations {
        label = "3. Enterprise Integration Perimeters";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        Models [label="Model Providers\\n(OpenAI / Anthropic / Bedrock / Ollama)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        Tools [label="Tools & MCP Protocols\\n(Model Context Protocol, SQL, Webhook APIs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        VectorDBs [label="Vector Stores (Long-Term Memory)\\n(Qdrant / Pinecone / ChromaDB)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_deploy_obs {
        label = "4. Deployment & Observability Operations";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        LangGraphPlatform [label="LangGraph Platform (Deploy)\\n(Docker / Cloud Serverless / Task Queues / Webhooks)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        LangSmith [label="LangSmith Enterprise APM\\n(Distributed Tracing, Latency, Token Costs, Regression Evals)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        FrontendUI [label="Frontend Clients\\n(CopilotKit / Assistant UI / Next.js React SDK)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    LangGraphStudio -> StateGraphEngine [label="Visual Code Generation"];
    LangChainCore -> StateGraphEngine [label="Node Implementation Logic"];

    StateGraphEngine -> CheckpointerStore [label="Superstep State Snapshots"];
    StateGraphEngine -> Models [label="Inference Invocations"];
    StateGraphEngine -> Tools [label="Tool Calls"];
    StateGraphEngine -> VectorDBs [label="Episodic Memory Recall"];

    StateGraphEngine -> LangGraphPlatform [label="Production Build"];
    LangGraphPlatform -> FrontendUI [label="SSE Streaming Protocol"];
    StateGraphEngine -> LangSmith [label="OTel Spans & Run Logs", style="dashed", color="#EC4899"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - LANGGRAPH
# ==============================================================================
st.header("2. LangGraph: The Core Cyclic Multi-Agent Orchestration Runtime", divider="rainbow")

st.markdown("""
At the core of the ecosystem sits **LangGraph** itself (`langgraph`). 
Unlike high-level wrappers that hide prompt mechanics behind pre-configured agent templates, LangGraph is intentionally 
a **low-level systems library**:
* It makes zero assumptions about your prompts, agent architectures, or model providers.
* It provides three foundational primitives: **`StateGraph`** (the graph builder), **`Pregel`** (the runtime executor), and **`Checkpointers`** (the persistence layer).
* It can be used completely independently of the broader LangChain ecosystem if developers prefer raw Python functions and direct OpenAI SDK calls!
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - LANGCHAIN
# ==============================================================================
st.header("3. LangChain: Component Primitives, Prompt Templates, & Model Wrappers", divider="rainbow")

st.markdown("""
While LangGraph controls the **graph topology and state flow**, **LangChain** (`langchain-core`) provides the **content 
inside individual nodes**:
""")

col_lc1, col_lc2 = st.columns([1, 1])

with col_lc1:
    st.markdown("""
    ### Why Pair LangChain with LangGraph?
    * **Standardized Message Classes**: `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` standardize schemas across OpenAI, Anthropic, and Gemini.
    * **Prompt Templates**: Format complex multi-variable system prompts with strict input validation.
    * **Structured Output Parsers**: `model.with_structured_output(PydanticSchema)` guarantees valid JSON extraction inside nodes.
    """)

with col_lc2:
    st.markdown("""
    ### Modern Separation of Concerns
    * **Inside Node Function**: Use LangChain Expression Language (`prompt | model | parser`) to run an isolated reasoning step.
    * **Between Node Functions**: Use LangGraph to decide which node runs next, route conditional edges, and persist checkpoints.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - LANGSMITH
# ==============================================================================
st.header("4. LangSmith: Full-Stack Observability, Tracing, & Evaluation Suite", divider="rainbow")

st.markdown("""
Debugging a multi-agent system without observability is like debugging a microservice architecture without distributed logs. 
**LangSmith** is the dedicated enterprise observability platform for LangGraph applications:
""")

col_ls1, col_ls2 = st.columns([1, 1])

with col_ls1:
    st.markdown("""
    ### Core LangSmith Capabilities
    1. **Hierarchical Run Trees**: Visualizes the exact execution hierarchy: Parent Graph -> Node -> LLM Call -> Tool Execution.
    2. **Latencies and Token Economics**: Measures exact TTFT, generation latency, and cumulative dollar costs per run.
    3. **Dataset Curation & Golden Sets**: Convert failed production runs into benchmark test cases with a single click.
    4. **Automated LLM-as-a-Judge Evals**: Run regression tests on agent task completion prior to production deployment.
    """)

with col_ls2:
    st.markdown("""
    ### Zero-Code Auto-Instrumentation
    Instrumenting an entire LangGraph application requires only environment variables:
    """)
    st.code("""
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT="enterprise-customer-agent"
    """, language="bash")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - LANGGRAPH PLATFORM
# ==============================================================================
st.header("5. LangGraph Platform: Cloud & Self-Hosted Scalable Deployment Infrastructure", divider="rainbow")

st.markdown("""
Running a LangGraph application in production requires enterprise deployment infrastructure: handling HTTP webhooks, 
managing background task queues, scaling worker pods, and providing thread persistence. 

**LangGraph Platform** is the specialized deployment backend:
""")

col_lp1, col_lp2 = st.columns([1, 1])

with col_lp1:
    st.markdown("""
    ### Deployment Flavors
    * **LangGraph Cloud**: Fully managed serverless infrastructure hosted by LangChain, scaling automatically with usage.
    * **LangGraph Self-Hosted (Enterprise)**: Deployable via Docker Compose or Kubernetes Helm charts inside private enterprise VPCs (AWS, Azure, GCP).
    """)

with col_lp2:
    st.markdown("""
    ### Built-in Platform Features
    * **Task Queues & Cron Triggers**: Run background agents on scheduled intervals.
    * **Webhook Event Egress**: Emits events to external enterprise systems upon state transitions.
    * **Horizontal Auto-Scaling**: Worker containers scale dynamically based on active thread volume.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - LANGGRAPH STUDIO
# ==============================================================================
st.header("6. LangGraph Studio: Desktop Visual IDE for Graph Inspection & Time-Travel Debugging", divider="rainbow")

st.markdown("""
**LangGraph Studio** is the specialized local desktop visual development environment engineered specifically for LangGraph.
""")

col_st1, col_st2 = st.columns([1, 1])

with col_st1:
    st.markdown("""
    ### Visual Graph Canvas
    * Renders live interactive diagrams of your compiled `StateGraph`.
    * Highlights the active node in real time as execution proceeds through supersteps.
    * Displays token counts and execution times directly on node badges.
    """)

with col_st2:
    st.markdown("""
    ### Time-Travel Debugging & State Forking
    * Inspect the exact JSON state snapshot at any historical superstep.
    * Click 'Edit State' to manually alter a variable (e.g. rewrite an agent's drafted email) and click 'Re-run from here' to fork execution down an alternative branch!
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - LANGCHAIN INTEGRATIONS
# ==============================================================================
st.header("7. LangChain Integrations: Ecosystem Modularity, Community Packages, and Partners", divider="rainbow")

st.markdown("""
Historically, LangChain was a monolithic Python package containing hundreds of unmaintained third-party integrations. 
In modern LangChain (v0.3+), the architecture is strictly modularized into isolated partner packages:
* **`langchain-core`**: Base abstractions, runnables, and message schemas with minimal dependencies.
* **`langchain`**: Standard chains, retrieval workflows, and agent algorithms.
* **Partner Packages (`langchain-openai`, `langchain-anthropic`, `langchain-qdrant`)**: Maintained directly by technology partners with pinned, conflict-free dependency trees.
* **`langchain-community`**: Community-contributed tools, document loaders, and experimental connectors.
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - MODEL PROVIDERS
# ==============================================================================
st.header("8. Model Providers: OpenAI, Anthropic, Google Gemini, Ollama, & AWS Bedrock", divider="rainbow")

st.markdown("""
LangGraph is model-agnostic. A single state graph can seamlessly coordinate different model providers within different nodes:
""")

prov_data = [
    {
        "Provider": "OpenAI (`langchain-openai`)",
        "Role in LangGraph": "Frontier Reasoning & Planner Nodes",
        "Key Feature": "Strict JSON Schema tool calling, deterministic outputs"
    },
    {
        "Provider": "Anthropic (`langchain-anthropic`)",
        "Role in LangGraph": "Complex Analysis & Long Context Nodes",
        "Key Feature": "Prompt KV-Caching, Computer Use API"
    },
    {
        "Provider": "Ollama (`langchain-ollama`)",
        "Role in LangGraph": "Local / On-Premise Worker Nodes",
        "Key Feature": "Zero API cost, data privacy, runs Llama 3.3 locally"
    },
    {
        "Provider": "AWS Bedrock (`langchain-aws`)",
        "Role in LangGraph": "Enterprise VPC Compliance Nodes",
        "Key Feature": "IAM role authentication, HIPAA/SOC2 compliance"
    }
]
st.dataframe(pd.DataFrame(prov_data), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - TOOL INTEGRATIONS
# ==============================================================================
st.header("9. Tool Integrations: Prebuilt Toolkits, OpenAPI Specs, and MCP Protocols", divider="rainbow")

st.markdown("""
Agents derive real-world value from their ability to execute external tools. 
LangGraph integrates with tools across three primary paradigms:
1. **Python Decorated Tools (`@tool`)**: Standard Python functions wrapped with type hints and docstrings.
2. **OpenAPI / REST Toolkits**: Automatically converting existing corporate Swagger/OpenAPI specifications into agent-callable tools.
3. **Model Context Protocol (MCP)**: The emerging Anthropic standard for connecting agents to secure local desktop and enterprise microservice servers.
""")

st.divider()

# ==============================================================================
# SECTION 10: TOPIC 9 - VECTOR DATABASES
# ==============================================================================
st.header("10. Vector Databases: Semantic Long-Term Memory with Pinecone, Qdrant, & Chroma", divider="rainbow")

st.markdown("""
While LangGraph's checkpointers manage short-term thread state, **Vector Databases** power **Cross-Thread Long-Term Memory**:
* Nodes extract facts from past conversations and embed them into vector stores (`qdrant.add_documents(...)`).
* When the user initiates a new session weeks later, retrieval nodes query the vector index via cosine similarity to recall historical context.
""")

st.divider()

# ==============================================================================
# SECTION 11: TOPIC 10 - FRONTEND INTEGRATIONS
# ==============================================================================
st.header("11. Frontend Integrations: CopilotKit, Assistant UI, & Next.js Streaming SDKs", divider="rainbow")

st.markdown("""
Connecting a backend LangGraph service to a modern React / Next.js frontend historically required writing complex WebSocket 
and SSE event decoders. Today, dedicated frontend SDKs provide drop-in integration:
* **CopilotKit**: Prebuilt React UI components that bind directly to LangGraph endpoints, rendering agent thoughts, tool progress, and human approval modals out of the box.
* **Assistant UI**: Highly customizable, headless conversational UI library for streaming LangGraph responses.
* **Vercel AI SDK**: Seamlessly consumes LangGraph Server-Sent Events in Next.js Server Components.
""")

st.divider()

# ==============================================================================
# SECTION 12: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("12. Production Implementation: End-to-End Enterprise Ecosystem Pipeline", divider="rainbow")

st.markdown("""
Below is a complete, production-grade template demonstrating how modern LangGraph applications configure ecosystem 
components: typed state, LangChain model wrappers with structured outputs, checkpointer persistence, and LangSmith tracing.
""")

prod_code = """
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Structured Output Schema using Pydantic
class TaskExtraction(BaseModel):
    task_name: str = Field(description="Name of the task")
    estimated_hours: int = Field(description="Estimated duration in hours")
    assigned_role: str = Field(description="Role required to complete task")

# 2. Centralized Graph State
class ProjectPlanState(TypedDict):
    raw_requirements: str
    extracted_tasks: List[Dict[str, Any]]
    total_hours: int
    plan_approved: bool

# 3. Node Functions
def extractor_node(state: ProjectPlanState) -> Dict[str, Any]:
    # In production: model.with_structured_output(TaskExtraction).invoke(...)
    mock_tasks = [
        {"task_name": "API Design", "estimated_hours": 8, "assigned_role": "Backend Engineer"},
        {"task_name": "UI Wireframes", "estimated_hours": 12, "assigned_role": "Product Designer"}
    ]
    hours = sum(t["estimated_hours"] for t in mock_tasks)
    return {"extracted_tasks": mock_tasks, "total_hours": hours}

def review_node(state: ProjectPlanState) -> Dict[str, Any]:
    # Auto-approve if under 40 hours
    approved = state["total_hours"] <= 40
    return {"plan_approved": approved}

# 4. Topology Builder with Checkpoint Persistence
builder = StateGraph(ProjectPlanState)
builder.add_node("extract", extractor_node)
builder.add_node("review", review_node)

builder.add_edge(START, "extract")
builder.add_edge("extract", "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 13: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("13. Interactive Workbench: Ecosystem Integration & Architecture Configurator", divider="rainbow")

st.markdown("""
Configure and evaluate your enterprise AI stack across the LangGraph ecosystem. Select model providers, vector stores, 
and deployment targets to assess production readiness, estimated latencies, and observability tiering.
""")

configurator = EcosystemStackConfigurator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Configure Enterprise Architecture Stack")
    
    sel_provider = st.selectbox("Foundation Model Provider", list(configurator.providers.keys()), index=0)
    sel_vdb = st.selectbox("Vector Database (Long-Term Memory)", list(configurator.vectordbs.keys()), index=0)
    sel_deploy = st.selectbox("Deployment Infrastructure", ["LangGraph Cloud", "Self-Hosted Kubernetes", "Local Python Script"], index=0)
    
    enable_tracing = st.checkbox("Enable LangSmith Observability & Deep Tracing", value=True)

    eval_btn = st.button("Evaluate Architecture Topology 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Architecture Audit & Stack Evaluation")
    
    eval_result = configurator.evaluate_stack(sel_provider, sel_vdb, enable_tracing, sel_deploy)
    
    m1, m2 = st.columns(2)
    m1.metric("Est. Roundtrip Latency", f"~{eval_result['estimated_roundtrip_ms']} ms")
    m2.metric("Production Tier", eval_result["production_readiness"])

    st.markdown("#### Topology Breakdown")
    st.markdown(f"* **Inference Engine**: `<span class='badge-eco'>{eval_result['selected_provider']}</span>` ({eval_result['provider_tool_tier']})", unsafe_allow_html=True)
    st.markdown(f"* **Vector Memory**: `<span class='badge-eco'>{eval_result['selected_vectordb']}</span>` ({eval_result['vectordb_type']})", unsafe_allow_html=True)
    st.markdown(f"* **APM Tracing**: `{eval_result['observability']}`")
    st.markdown(f"* **Runtime Target**: `{eval_result['deployment']}`")

    if eval_result["production_readiness"] == "Enterprise Ready":
        st.success("✅ Fully Compliant: Stack features durable deployment and comprehensive distributed observability.")
    else:
        st.warning("⚠️ Warning: Stack lacks distributed tracing or relies on local script execution.")

st.divider()

# ==============================================================================
# SECTION 14: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Architectural interviews evaluate knowledge of the broader LangChain and LangGraph ecosystem, 
vendor lock-in risks, and deployment tradeoffs.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What are the distinct responsibilities of LangGraph, LangChain, and LangSmith?"):
    st.markdown("""
    **Answer**:
    * **LangGraph** is the **execution runtime**: it coordinates cyclic control flow, handles state transitions, executes supersteps, and manages thread checkpointing.
    * **LangChain** is the **component library**: it provides prompt templates, output parsers, model wrappers, and standardized message schemas (`HumanMessage`, `AIMessage`).
    * **LangSmith** is the **observability platform**: it logs execution traces, measures token costs and latencies, tracks regression evaluations, and monitors production health.
    """)

with st.expander("Q2: Why did LangChain move to modular partner packages (e.g. `langchain-openai`)?"):
    st.markdown("""
    **Answer**: Early LangChain suffered from dependency bloat: installing the library pulled in hundreds of conflicting dependencies, 
    making enterprise deployments fragile. The modular architecture decouples `langchain-core` (lightweight, zero heavy dependencies) 
    from partner packages (`langchain-openai`, `langchain-anthropic`). This allows enterprise teams to install only the specific model 
    SDKs they need, reducing image size, speeding CI/CD builds, and preventing library version conflicts.
    """)

with st.expander("Q3: What is LangGraph Studio and what problems does it solve during development?"):
    st.markdown("""
    **Answer**: LangGraph Studio is a specialized desktop visual development IDE. Debugging multi-agent graphs purely in terminal tracebacks 
    is extraordinarily difficult because execution branches non-linearly across cycles. Studio renders the live graph topology, visualizes 
    supersteps as they fire, inspects state dictionaries at every checkpoint, and allows developers to manually edit state variables and 
    replay runs from any historical point (Time-Travel).
    """)

with st.expander("Q4: How does LangGraph Platform deploy stateful graphs compared to deploying standard FastAPI endpoints?"):
    st.markdown("""
    **Answer**: Deploying a stateful graph on raw FastAPI requires manually engineering background worker queues (Celery/Temporal), 
    building database checkpoint commit handlers, managing SSE streaming chunkers, and writing thread-locking logic to prevent race conditions. 
    LangGraph Platform bundles these production requirements out of the box: it provides an optimized API server with automatic thread locking, 
    built-in PostgreSQL checkpointing, background task queues, and streaming event protocols.
    """)

with st.expander("Q5: What is the Model Context Protocol (MCP) and how does it fit into LangGraph?"):
    st.markdown("""
    **Answer**: MCP (Model Context Protocol) is an open standard introduced by Anthropic that standardizes how applications expose tools, 
    prompts, and resources to AI agents. In LangGraph, instead of writing custom bespoke Python wrappers for every corporate API, agents 
    can connect to standardized MCP servers (e.g., GitHub MCP, Postgres MCP, Slack MCP), providing a secure, uniform protocol for tool execution.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Hardcoding Model Provider Classes Directly in Graph Nodes (CRITICAL)**
    * *Error*: Instantiating `ChatOpenAI()` directly inside node function bodies rather than passing configurable parameters.
    * *Impact*: Prevents multi-provider fallback, destroys unit test mockability, and causes vendor lock-in.
    * *Fix*: Inject models via graph configurations: `config['configurable']['model']`.
    """)

    st.error("""
    **Trap 2: Leaking LangSmith API Keys in Client-Side Frontends (CRITICAL)**
    * *Error*: Exposing `LANGSMITH_API_KEY` in React/Next.js frontend bundles.
    * *Impact*: Allows malicious actors to hijack your LangSmith workspace, read private customer conversation traces, or deplete quotas.
    * *Fix*: Keep all LangSmith credentials strictly on backend proxy servers.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Using Monolithic `langchain` Package Instead of Partner Packages (WARNING)**
    * *Warning*: Importing from `langchain.chat_models import ChatOpenAI` instead of `langchain_openai import ChatOpenAI`.
    * *Impact*: Triggers deprecation warnings and pulls legacy dependencies that degrade cold start latency.
    * *Fix*: Always import directly from dedicated partner packages (`langchain-openai`, `langchain-anthropic`).
    """)

    st.warning("""
    **Trap 4: Neglecting Token Budgeting Across Multi-Agent Swarms (WARNING)**
    * *Warning*: Allowing 5 parallel agents to loop freely without shared token usage tracking.
    * *Impact*: A single customer query can trigger 50 recursive calls, racking up massive cloud API bills.
    * *Fix*: Enforce global recursion limits and aggregate token attribution in centralized state.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: LangSmith</strong><br><br>
        <em>Front:</em> What is the primary operational role of LangSmith?<br><br>
        <em>Back:</em> Distributed observability, run tree tracing, token economics monitoring, and LLM-as-a-judge evaluation.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Studio</strong><br><br>
        <em>Front:</em> What is LangGraph Studio?<br><br>
        <em>Back:</em> A local desktop visual IDE for inspecting graph topology, watching live supersteps, and time-travel debugging.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Partner Packages</strong><br><br>
        <em>Front:</em> Why use <code>langchain-openai</code> over <code>langchain</code>?<br><br>
        <em>Back:</em> Modular, conflict-free dependency trees with faster cold-start import latencies.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: MCP</strong><br><br>
        <em>Front:</em> What does Model Context Protocol standardize?<br><br>
        <em>Back:</em> A uniform protocol for exposing tools, prompts, and data resources from servers to AI agents.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 15: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("15. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of the LangGraph ecosystem:")

chk1 = st.checkbox("I understand the distinct roles of LangGraph (runtime), LangChain (components), and LangSmith (observability).", value=True)
chk2 = st.checkbox("I know how to instrument LangGraph applications with LangSmith using environment variables.", value=True)
chk3 = st.checkbox("I can explain how LangGraph Studio enables visual graph inspection and time-travel debugging.", value=True)
chk4 = st.checkbox("I understand the benefits of modular partner packages (`langchain-openai`, `langchain-anthropic`).", value=True)
chk5 = st.checkbox("I know how vector databases provide cross-thread long-term memory alongside checkpointers.", value=True)
chk6 = st.checkbox("I understand how modern frontend SDKs (CopilotKit, Assistant UI) stream LangGraph state to React UIs.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 1.3: LangGraph Ecosystem! You have completed Chapter 1.")
