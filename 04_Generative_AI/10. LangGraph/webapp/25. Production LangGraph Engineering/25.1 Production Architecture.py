import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="25.1 Production Architecture - LangGraph Curriculum",
    page_icon="🏗️",
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
    .badge-arch {
        background-color: #172554;
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class ProductionArchitectureSimulator:
    """
    Simulates enterprise service topology health, throughput, and latencies
    across API layer, checkpointer, databases, model gateways, and tool workers.
    """
    def __init__(self):
        self.services = {
            "api_gateway": {"name": "FastAPI / LangGraph Server", "tier": "Ingress", "status": "HEALTHY", "p95_ms": 14, "rps": 420},
            "task_broker": {"name": "Redis Queue / Celery Workers", "tier": "Async Workers", "status": "HEALTHY", "p95_ms": 8, "rps": 380},
            "postgres_app": {"name": "PostgreSQL (User & App DB)", "tier": "Relational Storage", "status": "HEALTHY", "p95_ms": 22, "rps": 310},
            "postgres_chk": {"name": "PostgresSaver (Checkpointer)", "tier": "Persistence", "status": "HEALTHY", "p95_ms": 18, "rps": 850},
            "vector_store": {"name": "Qdrant / pgvector Cluster", "tier": "Vector DB", "status": "HEALTHY", "p95_ms": 45, "rps": 190},
            "llm_gateway": {"name": "LiteLLM / Model Gateway", "tier": "Inference Gateway", "status": "HEALTHY", "p95_ms": 820, "rps": 125},
            "tool_sandbox": {"name": "Secure gRPC Tool Microservices", "tier": "Tool Sandbox", "status": "HEALTHY", "p95_ms": 95, "rps": 240}
        }

    def get_service_metrics(self) -> List[Dict[str, Any]]:
        return list(self.services.values())

    def simulate_failure(self, service_key: str, status: str = "DEGRADED") -> None:
        if service_key in self.services:
            self.services[service_key]["status"] = status
            if status == "DEGRADED":
                self.services[service_key]["p95_ms"] *= 4
            elif status == "OUTAGE":
                self.services[service_key]["p95_ms"] = 0
                self.services[service_key]["rps"] = 0

    def reset_status(self) -> None:
        for s in self.services.values():
            s["status"] = "HEALTHY"

arch_sim = ProductionArchitectureSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Multi-Tier Production Architecture for LangGraph Systems", divider="rainbow")

st.markdown("""
Building production-grade LangGraph systems requires transitioning from isolated script executions to an 
enterprise multi-tier service topology. In production, LangGraph does not run in a vacuum; it operates as an 
orchestration runtime embedded between an API Gateway, an async task broker, a persistent checkpoint store, 
an enterprise knowledge base, a resilient LLM gateway, and secure tool execution sandboxes.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ProductionArchitectureBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_tier_ingress {
            label="1. Ingress & Client Tier (Web, Mobile, External Webhooks)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            clients [label="Client Applications\\nWeb Frontend (useStream), REST Clients, Webhooks", fillcolor="#0F172A"];
            api_gw [label="API Layer (FastAPI / LangGraph Server)\\nAuth, SSE Streaming, Rate Limiting, CORS", fillcolor="#1E293B"];
        }

        subgraph cluster_tier_orchestration {
            label="2. LangGraph Execution & Worker Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            graph_runtime [label="LangGraph State Engine\\nCompiled StateGraph / Functional Entrypoint\\nAsync Superstep Dispatcher", fillcolor="#312E81"];
            worker_pool [label="Background Task Workers (Celery / Temporal)\\nLong-running Agent Workflows & Scheduled Jobs", fillcolor="#1E293B"];
        }

        subgraph cluster_tier_storage {
            label="3. Persistence & Data Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chk_store [label="Checkpoint Storage (PostgresSaver)\\nThread State, Snapshot History, Branch Forks", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            app_db [label="Application Relational DB (PostgreSQL)\\nUser Accounts, Auth Roles, Billing, Audit Logs", fillcolor="#0F172A"];
            vector_db [label="Vector Store (pgvector / Qdrant)\\nSemantic Documents, Chunk Embeddings, RAG Index", fillcolor="#0F172A"];
        }

        subgraph cluster_tier_external {
            label="4. Inference & Tool Integration Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            llm_gateway [label="Model Provider Gateway (LiteLLM / Portkey)\\nFallback Routing, Caching, Token Quotas, Retries", fillcolor="#831843", fontcolor="#FBCFE8"];
            tool_services [label="Isolated Tool Sandbox (gRPC / Container Microservices)\\nSQL Runner, Code Interpreter, Enterprise APIs", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        clients -> api_gw [label="HTTPS / WSS / SSE"];
        api_gw -> graph_runtime [label="Dispatch Thread"];
        api_gw -> worker_pool [label="Enqueue Job"];
        worker_pool -> graph_runtime [label="Executes Workflow"];

        graph_runtime -> chk_store [label="Read/Write Checkpoints"];
        graph_runtime -> app_db [label="User State / Profiles"];
        graph_runtime -> vector_db [label="Similarity Search"];
        graph_runtime -> llm_gateway [label="Model Inferences"];
        graph_runtime -> tool_services [label="Execute Tool Actions"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Designing Production Graphs
# ==============================================================================
st.header("2. Designing Production Graphs: Decoupled Topologies, State Contracts, and Service Boundaries", divider="rainbow")

st.markdown("""
Designing graphs for enterprise production demands strict software engineering principles:
1. **Explicit State Contracts**: Use type-safe schemas (via `TypedDict` or `Pydantic`) with validated field types and well-defined default factories.
2. **Decoupled Nodes**: Each node function must represent a single responsibility. Pure business logic should be separated from framework plumbing.
3. **Graceful Degradation Topologies**: If an optional tool or enrichment node fails, conditional edges must redirect to fallback nodes rather than aborting the pipeline.
4. **Deterministic Exit Paths**: Every cyclic loop must have an enforced exit condition (such as `recursion_limit` or max retry counters) to prevent unbounded loops.
""")

# ==============================================================================
# SECTION 3: API Layer
# ==============================================================================
st.header("3. API Layer: FastAPI, LangGraph Server, SSE Streaming Endpoints, and Async Event Gateways", divider="rainbow")

st.markdown("""
The API Layer serves as the boundary between end users and the LangGraph runtime. 
Production systems expose endpoints for graph initialization, message appending, state inspection, and Server-Sent Events (SSE) streaming.

### Key Endpoint Patterns:
- `POST /threads/{thread_id}/runs`: Initiates a new superstep run with input payload.
- `GET /threads/{thread_id}/stream`: Opens an SSE stream returning real-time node state deltas (`values`, `updates`, or `messages`).
- `POST /threads/{thread_id}/resume`: Submits external human approval to resume an interrupted thread.
- `GET /threads/{thread_id}/history`: Returns checkpoint history for auditing and UI timeline rendering.
""")

st.code('''from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import json

api = FastAPI(title="LangGraph Enterprise Gateway")

@api.post("/threads/{thread_id}/runs")
async def start_run(thread_id: str, payload: dict):
    config = {"configurable": {"thread_id": thread_id}}
    # Trigger graph execution asynchronously or stream response
    async def event_generator():
        async for chunk in compiled_app.astream(payload, config, stream_mode="messages"):
            # Stream tokens to client via SSE format
            yield f"data: {json.dumps({'content': chunk[0].content})}\\n\\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
''', language="python")

# ==============================================================================
# SECTION 4: Backend Services
# ==============================================================================
st.header("4. Backend Services: Microservice Orchestration, Task Workers, and Background Message Brokers", divider="rainbow")

st.markdown("""
For long-running agents (such as web research, data extraction, or code analysis taking minutes or hours), 
graph execution cannot be held inside a synchronous HTTP request. 

### Worker Architecture:
- **Message Brokers**: Use Redis, RabbitMQ, or Apache Kafka to queue incoming agent tasks.
- **Durable Workers**: Celery, Temporal, or AWS SQS workers pull tasks and invoke the compiled graph.
- **Heartbeat & Resumption**: Checkpointers persist intermediate supersteps to Postgres, allowing interrupted 
  worker pods to resume without loss of progress.
""")

# ==============================================================================
# SECTION 5: Database
# ==============================================================================
st.header("5. Database: Relational Store for Application State, User Profiles, and Billing Data", divider="rainbow")

st.markdown("""
A common architectural pitfall is storing generic enterprise application data inside the LangGraph checkpoint state. 
In production, a clean boundary separates **Graph State** from **Application Relational Data**:

- **Checkpoint State (LangGraph)**: Ephemeral and runtime-scoped state (conversation context, active plan, current node counter, temporary scratchpad).
- **Application Database (PostgreSQL / MySQL)**: Persistent business entities (user identities, enterprise org accounts, subscription tiers, payment records, role-based access control rules).
""")

# ==============================================================================
# SECTION 6: Checkpoint Storage
# ==============================================================================
st.header("6. Checkpoint Storage: PostgreSQL, Redis, and Partitioned Checkpoint Namespaces", divider="rainbow")

st.markdown("""
Checkpoint storage is the foundation of durable execution, HITL interrupts, and time travel. 
Choosing the correct checkpointer depends on latency and persistence requirements:

| Persistence Engine | Latency | Durability | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`MemorySaver`** | Sub-millisecond | None (In-memory) | Unit tests, local prototyping, ephemeral sessions |
| **`RedisSaver`** | 1 - 5 ms | In-memory with RDB/AOF | Fast conversational sessions with TTL eviction |
| **`PostgresSaver`** | 10 - 25 ms | ACID Guaranteed | Enterprise production, audit history, HITL approvals |
""")

st.code('''from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Production connection pool for Postgres checkpointer
DB_URI = "postgresql://agent_admin:secret@pg-cluster.internal:5432/agent_checkpoints"

pool = ConnectionPool(conninfo=DB_URI, max_size=20)
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensures tables and indexes exist

# Compile graph with enterprise checkpointer
app = workflow.compile(checkpointer=checkpointer)
''', language="python")

# ==============================================================================
# SECTION 7: Vector Database
# ==============================================================================
st.header("7. Vector Database: Semantic Retrieval, Chunk Store, and RAG Integration Architecture", divider="rainbow")

st.markdown("""
When agents require enterprise knowledge retrieval, vector databases (such as Qdrant, Milvus, Chroma, or `pgvector`) 
are integrated as specialized tool services or retriever nodes.

### Production Vector Architecture:
1. **Hybrid Indexing**: Combine dense semantic embeddings with sparse BM25 keyword indexes for high precision retrieval.
2. **Metadata Filtering**: Filter documents by tenant ID, department clearance, and document age at the database layer.
3. **Decoupled Embedding Pipelines**: Keep heavy embedding ingestion separate from the real-time agent execution loop.
""")

# ==============================================================================
# SECTION 8: Model Provider
# ==============================================================================
st.header("8. Model Provider: LLM Gateway, Fallback Routing, Load Balancing, and Quota Management", divider="rainbow")

st.markdown("""
Directly calling provider SDKs (such as OpenAI or Anthropic directly) in production graphs creates vendor lock-in 
and vulnerability to API outages. 

### Production LLM Gateway Features:
- **Automatic Fallback Routing**: If Claude 3.5 Sonnet returns a 503 or 429 error, the gateway automatically redirects to GPT-4o.
- **Latency Load Balancing**: Distribute traffic across multiple regional API keys and endpoints.
- **Centralized Quota & Token Budgets**: Enforce per-user and per-organization spending limits before requests reach the model provider.
""")

# ==============================================================================
# SECTION 9: Tool Services
# ==============================================================================
st.header("9. Tool Services: Isolated Sandbox Execution, Secure gRPC/REST APIs, and Dynamic Registries", divider="rainbow")

st.markdown("""
Allowing agents to execute arbitrary Python code, bash scripts, or database updates inside the primary API container 
presents severe security risks.

### Sandboxed Tool Execution:
- **Container Sandboxing**: Run untrusted code in short-lived, gVisor or Firecracker-isolated containers (e.g., Modal, E2B).
- **Network Isolation**: Strip external internet access from code interpreter containers to prevent data exfiltration.
- **Least-Privilege Database Tools**: Database query tools must connect via read-only credentials with enforced `STATEMENT_TIMEOUT` limits.
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Multi-Tier Production Architecture Integration Pipeline", divider="rainbow")

st.markdown("""
The following complete module illustrates a production factory configuring a compiled graph with connection pooling, 
PostgreSQL checkpointing, and model gateway routing.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class ProductionAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tenant_id: str
    user_id: str
    execution_status: str

class ProductionGraphFactory:
    """
    Enterprise factory that assembles the compiled StateGraph
    with checkpointers, model gateways, and secure tools.
    """
    def __init__(self, checkpointer, model_client, tool_registry):
        self.checkpointer = checkpointer
        self.model = model_client
        self.tools = tool_registry

    def agent_node(self, state: ProductionAgentState) -> dict:
        # Calls model gateway with tenant-scoped context
        response = self.model.invoke(state["messages"])
        return {"messages": [response], "execution_status": "in_progress"}

    def tool_node(self, state: ProductionAgentState) -> dict:
        # Executes tools in isolated sandbox
        last_msg = state["messages"][-1]
        results = []
        for call in getattr(last_msg, "tool_calls", []):
            res = self.tools.execute(call["name"], call["args"], tenant_id=state["tenant_id"])
            results.append(res)
        return {"messages": results, "execution_status": "tools_executed"}

    def build_app(self):
        workflow = StateGraph(ProductionAgentState)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=self.checkpointer)
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Production Architecture Topology Simulator & Service Health Dashboard", divider="rainbow")

st.markdown("""
Simulate enterprise microservice latency, throughput, and failover scenarios across the multi-tier production architecture.
""")

col_srv1, col_srv2 = st.columns([1, 2])

with col_srv1:
    st.subheader("Service Health Controls")
    target_service = st.selectbox(
        "Select Service to Chaos Test",
        list(arch_sim.services.keys()),
        format_func=lambda k: arch_sim.services[k]["name"]
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Simulate Degradation", type="primary"):
            arch_sim.simulate_failure(target_service, "DEGRADED")
            st.rerun()
    with col_btn2:
        if st.button("Reset All Services"):
            arch_sim.reset_status()
            st.rerun()

with col_srv2:
    st.subheader("Live Topology Health Dashboard")
    metrics = arch_sim.get_service_metrics()
    
    df_metrics = pd.DataFrame(metrics)
    st.dataframe(df_metrics, use_container_width=True)
    
    # Check overall architecture status
    has_degraded = any(m["status"] == "DEGRADED" for m in metrics)
    if has_degraded:
        st.warning("Warning: One or more architectural tiers are degraded. Automatic fallback routing engaged.")
    else:
        st.success("All production architecture tiers operating at nominal latency and capacity.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: Why should checkpoint storage and application relational data be stored in separate schemas or databases?"):
    st.markdown("""
    **Answer:**  
    Checkpoint tables have fundamentally different access patterns than business entity tables:
    1. **High Write Churn**: Checkpoints append large JSON state snapshots at every superstep, resulting in high write IOPS and rapid disk growth.
    2. **Retention & Partitioning**: Checkpoints require automated time-based partitioning and TTL retention policies.
    3. **Schema Stability**: Separating concerns ensures application user and billing schemas remain decoupled from graph state migrations.
    """)

with st.expander("Q2: How do you scale LangGraph worker pods horizontally without state synchronization conflicts?"):
    st.markdown("""
    **Answer:**  
    By using an external persistent checkpointer (e.g., PostgreSQL with row-level locks on `thread_id`). Worker pods remain completely stateless. 
    Any worker pod can pull a job for `thread-42`, load the latest checkpoint from Postgres, execute the superstep, and commit the new snapshot.
    """)

with st.expander("Q3: What is the recommended method for handling model provider outages in production?"):
    st.markdown("""
    **Answer:**  
    Implement an LLM gateway (such as LiteLLM or Portkey) configured with automated fallback cascades (e.g., Primary: Claude 3.5 Sonnet -> Fallback: GPT-4o -> Secondary: Llama 3.3 70B), 
    paired with circuit breakers and exponential backoff retry policies.
    """)

with st.expander("Q4: How do you prevent untrusted tool code execution from compromising host infrastructure?"):
    st.markdown("""
    **Answer:**  
    Isolate tool execution in ephemeral sandboxed microservices running inside containerized hypervisors (such as Firecracker or gVisor) 
    with dropped Linux capabilities, read-only root filesystems, strict CPU/memory cgroups, and disabled outbound internet access.
    """)

with st.expander("Q5: What is the role of Server-Sent Events (SSE) compared to WebSockets for LangGraph frontend streaming?"):
    st.markdown("""
    **Answer:**  
    SSE is unidirectional, simpler to configure over standard HTTP/2 and corporate firewalls, and natively supported by web clients via `EventSource`. 
    For agent interactions where the client sends an initial prompt and streams back step-by-step progress, SSE is preferred. WebSockets are reserved 
    for continuous bidirectional voice or low-latency canvas collaboration.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Synchronous In-Process Long-Running Graphs**  
Executing multi-minute agent graphs directly within synchronous HTTP request threads blocks ASGI/WSGI workers, 
exhausts connection pools, and causes HTTP 504 Gateway Timeouts at the reverse proxy.
""")

st.error("""
**Trap 2: Storing Unbounded Binary Data in Checkpoint Channels**  
Placing large PDF documents, images, or raw audio files into state channels rapidly exhausts Postgres checkpointer memory 
and balloons backup storage. Always store binary files in S3/GCS and store signed URIs in state channels.
""")

st.warning("""
**Trap 3: Unpooled Database Checkpointer Connections**  
Instantiating a new database connection on every `app.invoke()` call without connection pooling exhausts Postgres `max_connections` 
during traffic spikes, taking down the persistence layer.
""")

st.warning("""
**Trap 4: Bypassing Rate Limits on Tool Services**  
Allowing autonomous agent loops to invoke external SaaS tools without rate limiters can lead to API quota exhaustion, 
account suspension, or massive unanticipated cloud bills.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Stateless Worker Pods</h4>
        <p>Worker instances that store zero local state, relying entirely on PostgresSaver to load and commit state snapshots per superstep.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>LLM Gateway</h4>
        <p>A middleware layer managing model fallback routing, load balancing, caching, and rate limit quotas across multiple providers.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>PostgresSaver Connection Pool</h4>
        <p>A persistent pool of database connections (via psycopg_pool) ensuring efficient high-concurrency checkpointer operations.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Tool Isolation Sandbox</h4>
        <p>A secure microservice environment executing code and database tools in isolated containers away from the primary API runtime.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the multi-tier production architecture (API, workers, persistence, models, tools).", value=True)
chk_2 = st.checkbox("I know how to decouple ephemeral graph checkpoint state from enterprise relational business data.", value=True)
chk_3 = st.checkbox("I can implement asynchronous SSE streaming endpoints for LangGraph execution in FastAPI.", value=True)
chk_4 = st.checkbox("I understand how to configure PostgresSaver with connection pooling in high-concurrency environments.", value=True)
chk_5 = st.checkbox("I know how to integrate an LLM gateway with fallback routing to protect against model outages.", value=True)
chk_6 = st.checkbox("I can design isolated, secure sandbox microservices for executing untrusted tool operations.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Production architecture mastery confirmed! Ready for performance optimization engineering.")
