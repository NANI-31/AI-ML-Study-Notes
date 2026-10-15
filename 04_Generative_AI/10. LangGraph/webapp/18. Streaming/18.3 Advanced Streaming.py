import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="18.3 Advanced Streaming - LangGraph Curriculum",
    page_icon="🌊",
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
    .badge-subgraph {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #7E22CE;
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
class AdvancedStreamingSimulator:
    """
    Simulates advanced streaming topologies in LangGraph:
    - Nested subgraph event streaming with namespace prefixes
    - Tool execution streaming (arguments -> status -> execution -> result)
    - SSE (Server-Sent Events) formatted payloads for frontend transport
    """
    def __init__(self):
        pass

    def simulate_nested_stream(self, parent_query: str) -> List[Dict[str, Any]]:
        events = []
        # Parent node 1
        events.append({
            "namespace": "parent",
            "node": "parent_orchestrator",
            "event_type": "node_update",
            "payload": {"status": f"Dispatched child research for query: '{parent_query}'"},
            "elapsed_ms": 50
        })
        # Subgraph node 1
        events.append({
            "namespace": "parent:research_subgraph",
            "node": "search_arxiv",
            "event_type": "tool_start",
            "payload": {"tool": "arxiv_api", "args": {"query": parent_query}},
            "elapsed_ms": 180
        })
        # Subgraph node 2
        events.append({
            "namespace": "parent:research_subgraph",
            "node": "search_arxiv",
            "event_type": "tool_complete",
            "payload": {"papers_found": 3, "top_paper": "Deep LangGraph Architectures"},
            "elapsed_ms": 420
        })
        # Subgraph node 3
        events.append({
            "namespace": "parent:research_subgraph",
            "node": "synthesize_research",
            "event_type": "node_update",
            "payload": {"subgraph_output": "Research synthesis complete."},
            "elapsed_ms": 650
        })
        # Parent node 2
        events.append({
            "namespace": "parent",
            "node": "final_assembler",
            "event_type": "node_update",
            "payload": {"final_response": "Consolidated report generated and ready."},
            "elapsed_ms": 820
        })
        return events

    def format_sse_chunk(self, event_id: int, event_name: str, data: Dict[str, Any]) -> str:
        import json
        return f"id: {event_id}\nevent: {event_name}\ndata: {json.dumps(data)}\n\n"


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Streaming Topologies, Subgraphs, and Real-Time UI Pipelines", divider="rainbow")

st.markdown("""
In enterprise-grade AI applications, streaming goes far beyond simply piping tokens from an LLM into a console.
Modern production architectures demand:
- **Nested Subgraph Streaming**: Forwarding and demultiplexing events from deep hierarchical graphs to the root client.
- **Tool Lifecycle Streaming**: Emitting events when tools are invoked, running, and finished with intermediate payloads.
- **Frontend Protocol Bridging**: Transporting multi-modal event streams over Server-Sent Events (SSE) or WebSockets.
- **State Reconciliation**: Enabling optimistic UI updates and live progress bars without race conditions.

The blueprint below illustrates the full multi-tier streaming pipeline from deep subgraphs through web gateways to browser interfaces.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedStreamingTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_parent_graph {
            label="1. Parent Graph Orchestrator";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            parent_node [label="Supervisor / Orchestrator Node\\nNamespace: 'root'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            subgraph_invocation [label="Subgraph Invoker Node\\nsubgraphs=True configuration", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_subgraph {
            label="2. Nested Child Subgraph";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            child_node_a [label="Researcher Agent Node\\nNamespace: 'root:research_subgraph'", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            tool_execution [label="ToolNode Sandbox\\nEmits Tool Call & Progress Chunks", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            child_node_b [label="Summarizer Node\\nEmits Subgraph Tokens", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_gateway {
            label="3. Streaming Web Gateway / API Layer";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            sse_formatter [label="SSE / WebSocket Formatter\\nWraps Events with IDs and Namespaces", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            http_stream [label="HTTP Transfer-Encoding: chunked\\nX-Accel-Buffering: no", fillcolor="#18181B", fontcolor="#E4E4E7", color="#71717A"];
        }

        subgraph cluster_client_ui {
            label="4. Real-Time Frontend (React / Vue / Mobile)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            ui_nested_tree [label="Nested Step Visualizer\\n(Collapsible Subgraph Breadcrumb)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_tool_card [label="Live Tool Inspection Card\\n(Raw Args & JSON Preview)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_typewriter [label="Typewriter Chat Assistant\\nSub-second TTFT Rendering", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        parent_node -> subgraph_invocation;
        subgraph_invocation -> child_node_a [label="Dispatches Nested Superstep"];
        child_node_a -> tool_execution [label="Invokes External Tool"];
        tool_execution -> child_node_b [label="Tool Output Passed"];
        
        subgraph_invocation -> sse_formatter [label="Parent Events Forwarded"];
        child_node_a -> sse_formatter [label="Child Events (Namespaced)"];
        tool_execution -> sse_formatter [label="Tool Lifecycle Events"];
        child_node_b -> sse_formatter [label="Nested Token Stream"];

        sse_formatter -> http_stream;
        http_stream -> ui_nested_tree [label="Routes by Namespace"];
        http_stream -> ui_tool_card [label="Updates Tool UI"];
        http_stream -> ui_typewriter [label="Appends Token Stream"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: STREAMING NESTED GRAPHS
# ==============================================================================
st.header("2. Streaming Nested Graphs: Subgraph Event Propagation and Namespace Demarcation", divider="rainbow")

st.markdown("""
When a parent graph invokes a child subgraph as a node, default streaming in older architectures swallowed child events, only emitting the subgraph's final output dictionary.
In modern LangGraph, you can stream events from **both the parent graph and all nested child graphs** simultaneously using `subgraphs=True`!
""")

st.code('''# Streaming events across the entire graph hierarchy
for namespace, chunk in app.stream(
    {"query": "Research quantum error correction"},
    subgraphs=True, # Enable deep recursive streaming from all nested subgraphs!
    stream_mode="updates"
):
    print(f"Namespace: {namespace}") # e.g. () for root, ('research_subgraph',) for child
    print(f"Emitted Payload: {chunk}\\n")''', language="python")

st.markdown("""
- **Empty Namespace `()`**: Indicates the event originated from the root parent graph.
- **Nested Tuple Namespace `('research_subgraph',)`**: Identifies the exact child subgraph that generated the event.
- **Deep Hierarchies `('team_lead', 'coder_subgraph')`**: Chains hierarchically for multi-tier organizational agents!
""")

# ==============================================================================
# SECTION 3: STREAMING TOOL EXECUTION
# ==============================================================================
st.header("3. Streaming Tool Execution: Intermediate Tool Status, Arguments, and Partial Artifacts", divider="rainbow")

st.markdown("""
Users shouldn't be left wondering what an agent is doing when it calls an external API.
Streaming tool execution allows the UI to show:
1. **Tool Selected**: The tool name and parameters chosen by the LLM before execution begins.
2. **Tool Execution In-Flight**: Live status spinner (`Calling PostgreSQL...`).
3. **Tool Complete**: The raw result or preview card rendered for user inspection.
""")

st.code('''# Consuming tool call events from messages stream
for chunk, metadata in app.stream(inputs, stream_mode="messages"):
    if chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            print(f"Tool in progress: {tool_chunk.get('name')}")
            print(f"Arguments buffer: {tool_chunk.get('args')}")''', language="python")

# ==============================================================================
# SECTION 4: STREAMING AGENT PROGRESS
# ==============================================================================
st.header("4. Streaming Agent Progress: Multi-Step Milestone Indicators and Execution Telemetry", divider="rainbow")

st.markdown("""
Complex reasoning loops (like Plan-and-Solve or Deep Research) can involve dozens of steps.
Tracking milestone indicators keeps the user informed and engaged.
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("#### Progress Event Schema")
    st.code('''# Standardized progress event contract
{
    "stage": "retrieval",
    "step_index": 2,
    "total_steps": 5,
    "description": "Searching 10-K SEC filings",
    "status": "IN_PROGRESS",
    "duration_ms": 420
}''', language="json")

with col_p2:
    st.markdown("#### Client Rendering Pattern")
    st.markdown("""
    - **Step 1: Planning**: Marked completed in 0.3s.
    - **Step 2: Retrieval**: Progress bar increments from 0% to 100%.
    - **Step 3: Synthesis**: Typewriter chat window begins rendering text.
    - **Step 4: Quality Review**: Green checkmark badge rendered upon completion.
    """)

# ==============================================================================
# SECTION 5: FRONTEND STREAMING
# ==============================================================================
st.header("5. Frontend Streaming: Server-Sent Events (SSE), WebSockets, and Protocol Bridging", divider="rainbow")

st.markdown("""
To deliver real-time streams to browsers, applications bridge LangGraph's Python async generators to HTTP protocols.
The industry standard is **Server-Sent Events (SSE)** due to its simplicity, automatic browser reconnection, and pure HTTP compatibility.
""")

st.code('''# FastAPI SSE Endpoint Bridge
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/api/v1/agent/stream")
async def sse_agent_stream(payload: dict):
    async def sse_generator():
        step_id = 0
        async for namespace, chunk in langgraph_app.astream(
            {"messages": [("user", payload["prompt"])]},
            subgraphs=True,
            stream_mode="updates"
        ):
            step_id += 1
            data = json.dumps({"namespace": namespace, "data": chunk})
            yield f"id: {step_id}\\nevent: graph_update\\ndata: {data}\\n\\n"
            
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )''', language="python")

# ==============================================================================
# SECTION 6: REAL-TIME UI UPDATES
# ==============================================================================
st.header("6. Real-Time UI Updates: Typewriter Effects, Optimistic UI, and State Reconciliation", divider="rainbow")

st.markdown("""
On the frontend (React, Vue, iOS, Android), consuming an event stream requires careful state management:
1. **Typewriter Effect**: Append token chunks to the active assistant message without re-rendering the entire chat history.
2. **Optimistic UI**: Immediately show the user's message in the thread before the backend acknowledges receipt.
3. **State Reconciliation**: When the superstep completes, reconcile the local optimistic state with the server's committed state snapshot to ensure zero divergence.
""")

st.markdown("""
> **LangGraph Frontend SDKs**: Tools such as `@langchain/langgraph-sdk` provide hooks like `useStream` for React, which handle subgraphs, namespaces, and token reassembly out-of-the-box.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Tier Streaming Agent with Nested Graph Forwarding", divider="rainbow")

st.markdown("""
Here is a complete, copy-pasteable production pattern constructing a parent graph with a compiled child subgraph, streaming events recursively with namespace tracking.
""")

st.code('''from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define Child Subgraph
class SubState(TypedDict):
    sub_query: str
    sub_result: str

def worker_search_node(state: SubState) -> dict:
    return {"sub_result": f"Found 3 articles for '{state['sub_query']}'"}

child_builder = StateGraph(SubState)
child_builder.add_node("search_worker", worker_search_node)
child_builder.add_edge(START, "search_worker")
child_builder.add_edge("search_worker", END)
child_graph = child_builder.compile()

# 2. Define Parent Graph
class ParentState(TypedDict):
    user_topic: str
    final_report: str

def dispatch_node(state: ParentState) -> dict:
    return {"user_topic": state["user_topic"].upper()}

def run_child_node(state: ParentState) -> dict:
    # Invoke child graph directly as a node function
    child_res = child_graph.invoke({"sub_query": state["user_topic"]})
    return {"final_report": f"Report: {child_res['sub_result']}"}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("dispatch", dispatch_node)
parent_builder.add_node("subgraph_runner", run_child_node)

parent_builder.add_edge(START, "dispatch")
parent_builder.add_edge("dispatch", "subgraph_runner")
parent_builder.add_edge("subgraph_runner", END)

parent_app = parent_builder.compile(checkpointer=MemorySaver())

print("Parent and Child Graphs Successfully Built with Full Streaming Capabilities.")''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Real-Time Subgraph Streaming & SSE Event Simulator", divider="rainbow")

st.markdown("""
Test real-time recursive subgraph streaming and inspect SSE-formatted message payloads.
""")

sim = AdvancedStreamingSimulator()

col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    st.subheader("🌲 Simulation 1: Recursive Nested Subgraph Stream")
    topic_input = st.text_input("Parent Topic", value="Autonomous Drone Navigation")

    if st.button("🚀 Stream Hierarchical Workflow"):
        nested_events = sim.simulate_nested_stream(topic_input)
        
        for ev in nested_events:
            with st.container():
                st.markdown(f"**[{ev['elapsed_ms']}ms] Namespace**: `{ev['namespace']}` | Node: `{ev['node']}`")
                st.json(ev["payload"])
        st.success("🎉 Hierarchical Subgraph Execution Stream Finished!")

with col_adv2:
    st.subheader("📡 Simulation 2: Server-Sent Events (SSE) Wire Format")
    st.markdown("Inspect raw HTTP SSE text frames emitted to browser clients.")
    
    sample_events = [
        (1, "node_start", {"node": "planner", "thread_id": "th_100"}),
        (2, "token", {"content": "Planning", "node": "planner"}),
        (3, "token", {"content": " completed.", "node": "planner"}),
        (4, "node_complete", {"node": "planner", "status": "OK"})
    ]

    sse_text = ""
    for eid, ename, edata in sample_events:
        sse_text += sim.format_sse_chunk(eid, ename, edata)
        
    st.text_area("Live SSE Wire Protocol Buffer", value=sse_text, height=260)
    st.info("Browser JavaScript clients parse this using: `new EventSource('/api/v1/agent/stream')`.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: How do you stream events from nested subgraphs in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    By passing `subgraphs=True` to `app.stream()` or `app.astream()`.
    When enabled, the stream yields 2-tuples containing `(namespace, chunk)`, where the namespace tuple indicates whether the event originated from the root graph or a child subgraph (e.g. `('research_agent',)`).
    """)

with st.expander("Q2: Why is Server-Sent Events (SSE) generally preferred over WebSockets for agent streaming?", expanded=False):
    st.markdown("""
    **Answer**:
    SSE is unidirectional (server-to-client), operates over standard HTTP/HTTPS (port 443), works through enterprise firewalls without special upgrades, and includes automatic client reconnection.
    WebSockets are only necessary when bidirectional real-time audio or low-latency human intervention is needed.
    """)

with st.expander("Q3: How do you capture partial tool call arguments as they stream from an LLM?", expanded=False):
    st.markdown("""
    **Answer**:
    Using `stream_mode='messages'`, inspect the `tool_call_chunks` attribute of the emitted `AIMessageChunk`.
    LangGraph streams the tool name and incremental JSON string fragments before the model finishes and invokes the tool.
    """)

with st.expander("Q4: What causes race conditions in frontend chat UIs during multi-node streaming?", expanded=False):
    st.markdown("""
    **Answer**:
    When multiple nodes execute in parallel and stream tokens concurrently, their chunks interleave.
    If the frontend appends all incoming tokens to a single global text buffer, the messages scramble.
    The frontend must key buffers by `metadata['langgraph_node']` or `metadata['run_id']`.
    """)

with st.expander("Q5: What is the purpose of the X-Accel-Buffering: no header in streaming responses?", expanded=False):
    st.markdown("""
    **Answer**:
    It instructs reverse proxies like Nginx to disable response buffering.
    Without this header, Nginx buffers output until the stream finishes, causing the client to receive everything all at once at the end instead of real-time tokens.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting subgraphs=True When Debugging Child Graphs")
    st.markdown("""
    *Mechanism*: Invoking `app.stream(inputs)` on a graph containing subgraphs without specifying `subgraphs=True`.  
    *Consequence*: Child graph nodes execute invisibly, and frontend UI displays a frozen spinner during child execution.  
    *Fix*: Always pass `subgraphs=True` whenever hierarchical subgraphs are present.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Emitting Non-JSON Serializable Objects in Custom Streams")
    st.markdown("""
    *Mechanism*: Passing complex Python objects (e.g., custom database connection handles) into `writer({...})`.  
    *Consequence*: JSON serialization fails when writing to the SSE HTTP stream, abruptly closing the connection.  
    *Fix*: Ensure all custom stream events contain only standard primitive Python types (dicts, strings, numbers).
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Buffer Bloat on Slow Client Connections")
    st.markdown("""
    *Mechanism*: Fast LLM token generation (100 tokens/sec) streaming to a mobile client on a slow 3G cellular network.  
    *Consequence*: Server memory accumulates unacknowledged TCP buffers.  
    *Fix*: Implement client backpressure or chunk aggregation for bandwidth-constrained clients.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (UI Layout Warning): Re-rendering the Entire DOM on Every Token")
    st.markdown("""
    *Mechanism*: Binding incoming token chunks to top-level React state that re-renders all messages.  
    *Consequence*: Severe browser UI lag and frame drops during fast token generation.  
    *Fix*: Localize streaming text state to the active message component only.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**subgraphs=True**\\nEnables recursive event streaming across all nested child subgraphs.")
with col_fc2:
    st.info("**Namespace Tuple**\\nThe hierarchy path (e.g. ('agent_a', 'subgraph_b')) identifying event origins.")
with col_fc3:
    st.info("**SSE (Server-Sent Events)**\\nThe standard HTTP protocol for unidirectional server-to-browser streaming.")
with col_fc4:
    st.info("**tool_call_chunks**\\nPartial tool name and argument fragments streamed by LLMs before execution.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to stream nested subgraphs using the subgraphs=True flag.", value=True)
c2 = st.checkbox("I understand how namespace tuples identify the exact origin of streaming events.", value=True)
c3 = st.checkbox("I know how to capture partial tool call chunks as they are generated by an LLM.", value=True)
c4 = st.checkbox("I understand the Server-Sent Events (SSE) wire protocol format for browsers.", value=True)
c5 = st.checkbox("I know why X-Accel-Buffering: no is required when deploying behind Nginx.", value=True)
c6 = st.checkbox("I understand how to prevent UI race conditions when streaming parallel nodes.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 18.3 'Advanced Streaming' with full architectural and practical comprehension!")
