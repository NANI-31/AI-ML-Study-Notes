import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="13.4 Production Agent - LangGraph Curriculum",
    page_icon="🏭",
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
    .badge-prod {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-stream {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #9333EA;
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
class ProductionAgentTelemetryEngine:
    """
    Simulates production runtime telemetry:
    - State schema validation
    - Checkpoint snapshot generation
    - Streaming token events
    - Error recovery audit
    - Observability tracing
    """
    def __init__(self):
        self.checkpoints = []

    def simulate_production_run(
        self,
        thread_id: str,
        query: str,
        enable_streaming: bool,
        simulate_error: bool
    ) -> Dict[str, Any]:
        trace_events = []
        tokens_emitted = []
        
        trace_events.append({"event": "ThreadInit", "thread_id": thread_id, "status": "200_OK"})
        trace_events.append({"event": "StateCheckpointed", "step": 1, "snapshot_size_bytes": 1420})

        if simulate_error:
            trace_events.append({"event": "ToolExecutionError", "detail": "HTTP 503 from backend payment gateway", "severity": "HIGH"})
            trace_events.append({"event": "FallbackInvoked", "detail": "Switched to secondary cache provider", "status": "RECOVERED"})
            final_text = "System recovered from payment gateway timeout via secondary cache. Operational status is normal."
        else:
            final_text = f"Production query ('{query}') processed across 3 supersteps with full telemetry tracing."

        if enable_streaming:
            tokens_emitted = final_text.split(" ")

        self.checkpoints.append({
            "thread_id": thread_id,
            "step": len(self.checkpoints) + 1,
            "status": "COMMITTED"
        })

        return {
            "thread_id": thread_id,
            "final_text": final_text,
            "trace_events": trace_events,
            "streamed_tokens": tokens_emitted,
            "total_checkpoints": len(self.checkpoints)
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production Enterprise Agent Architecture", divider="rainbow")

st.markdown("""
Moving an agent from prototype to production requires hardening across **6 core pillars**:
1. **State**: Explicit, type-safe TypedDict schemas with formal channel reducers.
2. **Memory**: Short-term thread state coupled with long-term semantic profiles.
3. **Persistence**: Production checkpointers (PostgreSQL / SQLite) enabling fault recovery and audit trails.
4. **Error Handling**: Non-blocking tool failure handling and multi-provider fallbacks.
5. **Streaming**: Real-time token streaming (`astream_events`) for responsive user interfaces.
6. **Observability**: Distributed tracing, latency telemetry, and token accounting via LangSmith.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    prod_blueprint_dot = """
    digraph ProductionAgentArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CLIENT [label="Production Client API / UI\\n(Streaming Ingestion + thread_id)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_runtime {
            label = "Production Agent Runtime";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            STATE_LAYER [label="1. State & Channel Layer\\nType-safe schema + add_messages reducer", fillcolor="#064E3B", color="#34D399"];
            REASON_LAYER [label="2. Resilient Model Node\\nChatOpenAI with fallback to Claude", fillcolor="#0F172A", color="#38BDF8"];
            TOOL_LAYER [label="3. Hardened ToolNode\\nSandboxed tools with handle_tool_errors=True", fillcolor="#431407", color="#FB923C"];

            STATE_LAYER -> REASON_LAYER;
            REASON_LAYER -> TOOL_LAYER [label="tool_calls"];
            TOOL_LAYER -> REASON_LAYER [label="ToolMessage (Loop)"];
        }

        subgraph cluster_persistence {
            label = "Persistence & Memory Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            CHECKPOINTER [label="PostgresSaver Checkpointer\\nACID snapshot committed per superstep", fillcolor="#0F172A", color="#F59E0B"];
        }

        subgraph cluster_telemetry {
            label = "Observability & Streaming Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            STREAM_BUS [label="Event Streamer: astream_events(version='v2')\\nYields token deltas in real-time to UI", fillcolor="#2E1065", color="#C084FC"];
            TRACER [label="LangSmith Observability Collector\\nCaptures latency, tokens, error stacks", fillcolor="#0F172A", color="#38BDF8"];
        }

        CLIENT -> STATE_LAYER;
        REASON_LAYER -> CHECKPOINTER [label="Commit state"];
        REASON_LAYER -> STREAM_BUS [label="Emit tokens"];
        REASON_LAYER -> TRACER [label="Log traces"];
    }
    """
    st.graphviz_chart(prod_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Production Philosophy:</b> A prototype runs in memory; a production system survives node restarts,
network blips, and cloud outages without dropping conversation state or hanging customer requests.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: STATE
# ==============================================================================
st.header("2. State: Schema Rigor, Channel Typing, and Reducer Invariants", divider="rainbow")

st.markdown("""
In production, state must be defined using explicit `TypedDict` or Pydantic classes with
annotated channels:
""")

st.code("""
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class ProductionEnterpriseState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    tenant_id: str
    security_clearance: str
    audit_trail: Annotated[List[Dict[str, Any]], operator.add]
""", language="python")

# ==============================================================================
# SECTION 3: MEMORY
# ==============================================================================
st.header("3. Memory: Short-Term Thread State vs Long-Term Semantic Memory", divider="rainbow")

st.markdown("""
Production agents differentiate between two tiers of memory:
- **Short-Term Memory (Thread State)**: Stored inside the graph's `MessagesState`, scoped to the active `thread_id`.
- **Long-Term Memory (Cross-Thread Knowledge)**: Stored in external stores (LangGraph Store / VectorDB), persisting user preferences across weeks and months.
""")

# ==============================================================================
# SECTION 4: PERSISTENCE
# ==============================================================================
st.header("4. Persistence: Database Checkpointers and Time Travel", divider="rainbow")

st.markdown("""
Never deploy `MemorySaver` in production: all state vanishes when the container restarts.
Use production database checkpointers:
- **PostgresSaver**: Scalable ACID persistence for cloud deployments.
- **SqliteSaver**: Lightweight on-disk persistence for edge or local deployments.
""")

st.code("""
from langgraph.checkpoint.postgres import PostgresSaver

# Production Postgres connection pool
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
with PostgresSaver.from_conn_string("postgresql://user:pass@db:5432/agents", **connection_kwargs) as checkpointer:
    checkpointer.setup()  # Auto-creates checkpoint tables
    production_app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 5: ERROR HANDLING
# ==============================================================================
st.header("5. Error Handling: Tool Exceptions, Model Fallbacks, and Dead-Letter Queues", divider="rainbow")

st.markdown("""
Errors must be contained at the component level:
1. **Tool Errors**: `ToolNode(tools=..., handle_tool_errors=True)` catches exceptions and returns them as `ToolMessage`.
2. **Provider Outages**: `primary_model.with_fallbacks([backup_model])` seamlessly handles cloud outages.
3. **Dead-Letter Recovery**: Exception edges route unrecoverable failures to alert supervisors.
""")

# ==============================================================================
# SECTION 6: STREAMING
# ==============================================================================
st.header("6. Streaming: Real-Time Token Streaming and UI Delivery", divider="rainbow")

st.markdown("""
Waiting 10 seconds for an agent to finish multiple tool calls creates a poor user experience.
LangGraph supports **real-time streaming** via `astream_events`:
""")

st.code("""
# Real-time event streaming in FastAPI / Streamlit
async for event in app.astream_events(input_data, config=config, version="v2"):
    kind = event["event"]
    
    # Stream token chunks directly to client
    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            yield chunk.content
            
    # Notify client of tool execution
    elif kind == "on_tool_start":
        yield f"\\n[Executing: {event['name']}]...\\n"
""", language="python")

# ==============================================================================
# SECTION 7: OBSERVABILITY
# ==============================================================================
st.header("7. Observability: LangSmith Tracing, Step Telemetry, and Latency Auditing", divider="rainbow")

st.markdown("""
Production agents require end-to-end distributed tracing.
Enabling LangSmith traces every superstep, LLM call, and tool execution without modifying your code:
""")

st.code("""
# Environment configuration for automated LangSmith tracing:
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=lsv2_pt_...
# export LANGCHAIN_PROJECT=production-enterprise-agent
""", language="bash")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Enterprise Production Agent Pipeline with Persistence and Streaming", divider="rainbow")

st.markdown("""
Below is the complete production agent pipeline combining type-safe state, resilient model fallbacks,
error-handled tools, and checkpointer persistence:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class ProductionAgentState(MessagesState):
    tenant_id: str
    user_id: str

# 2. Resilient Models
primary_model = ChatOpenAI(model="gpt-4o", timeout=6.0)
fallback_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=6.0)
resilient_model = primary_model.with_fallbacks([fallback_model])

# 3. Production Tools
@tool
def fetch_system_health(service: str) -> str:
    \"\"\"Inspect live microservice health telemetry.\"\"\"
    return f"Service {service}: HEALTHY, p99_latency=18ms, error_rate=0.01%"

tools = [fetch_system_health]
model_with_tools = resilient_model.bind_tools(tools)

# 4. Nodes
def agent_node(state: ProductionAgentState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools, handle_tool_errors=True)

# 5. Graph Assembly
builder = StateGraph(ProductionAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# Compile with checkpointer
checkpointer = MemorySaver()
production_agent = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Production Agent Telemetry, Streaming & Checkpoint Inspector", divider="rainbow")

st.markdown("""
Simulate enterprise runtime telemetry: inspect state snapshots, simulate token streaming,
and observe automated recovery from backend service errors.
""")

telemetry_engine = ProductionAgentTelemetryEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Deployment Controls")
    thread_key = st.text_input("Active Thread ID:", value="session_prod_0912")
    user_request = st.text_input("Incoming User Request:", value="Check system health for billing-api and report anomalies.")
    stream_toggle = st.toggle("Enable Real-Time Token Streaming Simulation", value=True)
    error_sim = st.checkbox("🔥 Simulate Transient Backend Tool Exception (503)", value=False)
    
    deploy_btn = st.button("🚀 Invoke Production Agent", type="primary")

with wb_col2:
    st.subheader("📊 Telemetry & Observability Bus")
    if deploy_btn:
        res = telemetry_engine.simulate_production_run(
            thread_id=thread_key,
            query=user_request,
            enable_streaming=stream_toggle,
            simulate_error=error_sim
        )
        
        st.success("Execution completed and committed to checkpointer!")
        st.markdown(f"**Final Answer**: {res['final_text']}")
        
        if stream_toggle:
            st.markdown("#### Streamed Tokens (`astream_events`)")
            st.code(" ".join(res["streamed_tokens"]))

        st.markdown("#### Runtime Event Log")
        for evt in res["trace_events"]:
            st.json(evt)
    else:
        st.info("Configure parameters and click 'Invoke Production Agent' to inspect runtime telemetry.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: Why should MemorySaver never be deployed in a production microservice?", expanded=False):
    st.markdown("""
    **Answer**:
    `MemorySaver` stores state snapshots exclusively in Python process RAM.
    When containers scale down, restart, or crash, all in-flight conversation state is permanently lost.
    Production systems require a persistent database backend like `PostgresSaver` to ensure durability across restarts.
    """)

with st.expander("Q2: How does astream_events differ from standard astream in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    - `astream`: Streams complete node output updates as each node finishes its superstep.
    - `astream_events(version="v2")`: Streams fine-grained sub-node events in real time (individual LLM token chunks, tool starts, tool ends), enabling responsive token-by-token typing in web interfaces.
    """)

with st.expander("Q3: How do model fallbacks (with_fallbacks) operate inside production agent nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    When the primary model raises an exception (such as HTTP 429 Rate Limit or 500 Server Error),
    the `with_fallbacks()` runnable catches the exception and immediately invokes the backup model with identical messages,
    preventing user-facing outages without requiring separate graph nodes.
    """)

with st.expander("Q4: What is Time Travel in LangGraph and how is it used in production debugging?", expanded=False):
    st.markdown("""
    **Answer**:
    Because database checkpointers store every superstep snapshot indexed by `thread_id` and checkpoint ID,
    developers can fetch any historical state snapshot via `app.get_state_history()`, inspect the exact messages,
    and even fork execution from that point to test fixes.
    """)

with st.expander("Q5: What telemetry metrics are most critical to monitor for production LangGraph agents?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Superstep Latency**: Time spent in LLM reasoning vs. tool execution.
    2. **Token Consumption**: Input tokens vs. completion tokens per turn.
    3. **Recursion Depth**: Number of loop iterations before reaching END.
    4. **Tool Error Rate**: Frequency of tool exceptions and retries.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing Database Indexing on Thread Checkpoints")
    st.markdown("""
    *Mechanism*: Running custom Postgres checkpointers without index on `thread_id`.  
    *Consequence*: As the checkpoint table reaches millions of rows, state loading queries slow down from 5ms to 8 seconds.  
    *Fix*: Use official `PostgresSaver.setup()` which builds optimized indexes.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Non-Serializable Custom State Objects")
    st.markdown("""
    *Mechanism*: Storing live SQLAlchemy sessions or thread locks inside the state TypedDict.  
    *Consequence*: Checkpointers crash with `TypeError: Object is not JSON serializable`.  
    *Fix*: Only store JSON-compliant primitives in graph state channels.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Unbounded Context Windows on Long Threads")
    st.markdown("""
    *Mechanism*: Retaining 50+ conversation turns without applying `trim_messages()`.  
    *Consequence*: Linear increase in latency and exponential growth in cloud API bills.  
    *Fix*: Integrate automated message trimming into the agent node.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Silently Swallowing Unrecoverable Tool Errors")
    st.markdown("""
    *Mechanism*: Tools failing silently and returning generic success messages.  
    *Consequence*: Corrupted business logic and false reporting to customers.  
    *Fix*: Always return explicit error descriptions and monitor error rates in LangSmith.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**PostgresSaver**\\nProduction-grade PostgreSQL checkpointer ensuring ACID persistence across node restarts.")
with col_fc2:
    st.info("**astream_events**\\nThe streaming protocol yielding real-time token deltas and tool lifecycle events to client UIs.")
with col_fc3:
    st.info("**with_fallbacks**\\nMethod cascading to secondary model providers on rate limits or cloud outages.")
with col_fc4:
    st.info("**LangSmith**\\nObservability platform providing distributed tracing and latency auditing for LangGraph applications.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the 6 core pillars of production agents (State, Memory, Persistence, Errors, Streaming, Observability).", value=True)
c2 = st.checkbox("I understand why PostgresSaver is required over MemorySaver in production.", value=True)
c3 = st.checkbox("I can implement resilient model fallbacks using with_fallbacks().", value=True)
c4 = st.checkbox("I know how to stream real-time tokens to user interfaces via astream_events.", value=True)
c5 = st.checkbox("I understand how to leverage Time Travel to debug historical agent states.", value=True)
c6 = st.checkbox("I can monitor superstep latency, token consumption, and tool errors with LangSmith.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 13.4 'Production Agent' with full architectural and practical comprehension!")
