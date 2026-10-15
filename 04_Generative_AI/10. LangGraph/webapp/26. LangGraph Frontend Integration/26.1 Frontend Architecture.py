import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="26.1 Frontend Architecture - LangGraph Curriculum",
    page_icon="🖥️",
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
    .badge-frontend {
        background-color: #0F2D37;
        color: #38BDF8;
        border: 1px solid #0284C7;
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
class FrontendArchitectureSimulator:
    """
    Simulates full-stack client-server event streams,
    state synchronization deltas, and reconnection mechanics.
    """
    def __init__(self):
        self.stream_events = [
            {"event": "metadata", "data": {"run_id": "run-001", "thread_id": "thread-109", "graph": "CustomerAssistant"}},
            {"event": "on_chain_start", "data": {"name": "START", "input": {"query": "Check invoice status"}}},
            {"event": "on_node_start", "data": {"node": "planner", "superstep": 1}},
            {"event": "on_chat_model_stream", "data": {"token": "Looking", "node": "planner"}},
            {"event": "on_chat_model_stream", "data": {"token": " up", "node": "planner"}},
            {"event": "on_chat_model_stream", "data": {"token": " invoice...", "node": "planner"}},
            {"event": "on_node_end", "data": {"node": "planner", "output": {"plan": "query_database"}}},
            {"event": "on_tool_start", "data": {"tool": "sql_query", "params": {"id": "INV-402"}}},
            {"event": "on_tool_end", "data": {"tool": "sql_query", "result": "Paid on 2026-09-12"}},
            {"event": "on_chat_model_stream", "data": {"token": "Invoice", "node": "assistant"}},
            {"event": "on_chat_model_stream", "data": {"token": " INV-402", "node": "assistant"}},
            {"event": "on_chat_model_stream", "data": {"token": " was paid.", "node": "assistant"}},
            {"event": "on_chain_end", "data": {"status": "SUCCESS", "final_step": 3}}
        ]

    def get_events(self) -> List[Dict[str, Any]]:
        return self.stream_events

    def simulate_packet_loss_reconnect(self, dropped_idx: int) -> Dict[str, Any]:
        delivered = self.stream_events[:dropped_idx]
        missed = self.stream_events[dropped_idx:]
        return {
            "delivered_count": len(delivered),
            "missed_count": len(missed),
            "last_event_id": f"evt-{dropped_idx}",
            "resumed_from": f"evt-{dropped_idx + 1}"
        }

fe_sim = FrontendArchitectureSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Full-Stack Frontend-Backend Streaming and State Synchronization Topology", divider="rainbow")

st.markdown("""
Connecting a modern web frontend (React, Next.js, Vue) to a LangGraph agent requires an asynchronous, 
event-driven architecture. Unlike traditional request-response REST APIs, agent interfaces demand real-time feedback: 
streaming individual tokens, rendering intermediate tool invocations, showing reasoning steps, and reflecting 
checkpoint-persisted state changes. 

The architecture consists of three interconnected layers:
1. **Frontend Client**: React / Next.js UI leveraging the `@langchain/langgraph-sdk` or `useStream` hooks.
2. **API Layer / Gateway**: FastAPI or LangGraph Server translating HTTP/2 and Server-Sent Events (SSE).
3. **Backend Graph Engine**: Compiled `StateGraph` backed by durable PostgreSQL or Redis checkpointers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FrontendArchitectureBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_frontend {
            label="1. Frontend Client Layer (React / Next.js / TypeScript)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            ui_components [label="Chat UI & State Visualizer\\nMessage Feed, Thought Logs, Interrupt Banners", fillcolor="#0F172A"];
            sdk_hook [label="useStream / LangGraph SDK Client\\nManages SSE Connection, Optimistic Updates & Cache", fillcolor="#0F2D37", fontcolor="#38BDF8"];
            client_store [label="Client State Store (Zustand / Redux)\\nLocal Messages, Pending Writes, Active Thread ID", fillcolor="#1E293B"];
        }

        subgraph cluster_api {
            label="2. Ingress & API Transport Layer";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            api_gateway [label="FastAPI Gateway / LangGraph Server\\nAuth JWT, Rate Limiting, CORS, HTTP/2", fillcolor="#1E293B"];
            sse_pipe [label="Server-Sent Events (SSE) Stream\\ntext/event-stream Protocol with Last-Event-ID", fillcolor="#312E81"];
        }

        subgraph cluster_backend {
            label="3. Backend Graph & Persistence Layer";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            graph_engine [label="LangGraph State Engine\\nastream_events(v2) Dispatcher", fillcolor="#1E293B"];
            checkpointer [label="PostgresSaver / Checkpoint Store\\nImmutable Thread Snapshots & Lineage", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        ui_components -> sdk_hook [label="User Input Action"];
        sdk_hook -> client_store [label="Optimistic Append"];
        sdk_hook -> api_gateway [label="POST /threads/{id}/runs/stream"];

        api_gateway -> graph_engine [label="Dispatch Execution"];
        graph_engine -> checkpointer [label="Commit Superstep"];
        graph_engine -> sse_pipe [label="Emit Events (Tokens/Nodes)"];

        sse_pipe -> sdk_hook [label="SSE Event Stream"];
        sdk_hook -> client_store [label="Reconcile Channel Delta"];
        client_store -> ui_components [label="Reactive Re-render"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Backend Graph
# ==============================================================================
st.header("2. Backend Graph: Server-Side StateGraph Compilation, Persistent Threads, and Event Embers", divider="rainbow")

st.markdown("""
On the server side, the agent graph is defined and compiled with an enterprise checkpointer (`PostgresSaver`). 
To serve web clients efficiently, the graph must support asynchronous execution and fine-grained event streaming.

### Key Server-Side Requirements:
- **Asynchronous Execution**: Graph nodes and tools must use `async/await` to prevent blocking the web server event loop.
- **Event Streaming**: Graphs must be invoked via `astream()` or `astream_events(version="v2")` to emit real-time telemetry 
  as nodes start, stream LLM tokens, and complete supersteps.
- **Thread Scoping**: Every incoming client session is identified by a unique `thread_id` to maintain conversation continuity.
""")

st.code('''from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# Server-side asynchronous compilation
async def get_compiled_graph(pool: AsyncConnectionPool):
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_tools)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer)
''', language="python")

# ==============================================================================
# SECTION 3: API Layer
# ==============================================================================
st.header("3. API Layer: FastAPI Gateway, LangGraph Server Protocol, and REST / SSE Route Specifications", divider="rainbow")

st.markdown("""
The API Layer bridges the web frontend and the compiled graph. Production architectures standardize on 
the **LangGraph Server REST/SSE API protocol**.

### Core API Route Specifications:
1. `POST /threads`: Creates a new persistent conversation thread and returns `{ "thread_id": "uuid" }`.
2. `POST /threads/{thread_id}/runs/stream`: Accepts user message input, initiates graph execution, 
   and returns an open `text/event-stream` Server-Sent Events response.
3. `GET /threads/{thread_id}/state`: Retrieves the latest snapshot (`get_state`) for initial page loads.
4. `POST /threads/{thread_id}/runs/{run_id}/resume`: Submits human approval decisions to resume interrupted graphs.
""")

st.code('''from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/threads/{thread_id}/runs/stream")
async def run_agent_stream(thread_id: str, request: Request):
    body = await request.json()
    user_input = body.get("input", {})
    
    config = {"configurable": {"thread_id": thread_id}}
    
    async def sse_event_publisher():
        # Stream events over Server-Sent Events protocol
        async for event in agent_app.astream_events(user_input, config, version="v2"):
            kind = event["event"]
            payload = json.dumps(event)
            yield f"event: {kind}\\ndata: {payload}\\n\\n"

    return StreamingResponse(sse_event_publisher(), media_type="text/event-stream")
''', language="python")

# ==============================================================================
# SECTION 4: Frontend Client
# ==============================================================================
st.header("4. Frontend Client: React / Next.js SDK, State Management Stores, and EventSource Transport", divider="rainbow")

st.markdown("""
On the client side, modern web applications consume the SSE stream using `@langchain/langgraph-sdk` or native React hooks. 
The client manages three distinct categories of state:

- **Optimistic State**: Immediately displaying the user's message before the server acknowledges receipt.
- **Streaming State**: Incrementally accumulating streamed tokens in the active assistant message card.
- **Server-Committed State**: Authoritative snapshot returned at the superstep barrier upon graph completion.
""")

st.code('''// React / Next.js Client Hook using LangGraph Client SDK
import { Client } from "@langchain/langgraph-sdk";
import { useState } from "react";

const client = new Client({ apiUrl: "https://api.myagent.enterprise.com" });

export function useAgentChat(threadId: string) {
  const [messages, setMessages] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = async (userText: string) => {
    // 1. Optimistic append
    const userMsg = { role: "user", content: userText, id: crypto.randomUUID() };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);

    // 2. Open stream
    const stream = client.runs.stream(threadId, "my_agent", {
      input: { messages: [userMsg] },
      streamMode: "messages"
    });

    for await (const chunk of stream) {
      if (chunk.event === "messages/partial") {
        // Incrementally update streaming assistant bubble
        updateAssistantMessage(chunk.data);
      }
    }
    setIsStreaming(false);
  };

  return { messages, sendMessage, isStreaming };
}
''', language="typescript")

# ==============================================================================
# SECTION 5: Streaming Connection
# ==============================================================================
st.header("5. Streaming Connection: Server-Sent Events Protocol, Chunk Multiplexing, and Reconnection Logic", divider="rainbow")

st.markdown("""
Server-Sent Events (SSE) is the industry standard for LLM and agent streaming. Unlike WebSockets, SSE operates 
over standard HTTP/2, supports native browser automatic reconnection, and bypasses aggressive enterprise firewall proxies.

### SSE Protocol Invariants:
- **`event`**: Discriminates the payload type (e.g., `metadata`, `on_chat_model_stream`, `on_node_end`, `error`).
- **`data`**: JSON string payload containing token fragments or node output deltas.
- **`id`**: Monotonically increasing event ID allowing clients to resume interrupted connections via the `Last-Event-ID` header.
""")

# ==============================================================================
# SECTION 6: State Synchronization
# ==============================================================================
st.header("6. State Synchronization: Bi-directional Optimistic Updates, Channel Delta Merging, and Conflict Resolution", divider="rainbow")

st.markdown("""
Because agents generate non-deterministic outputs across multiple supersteps, the frontend client and backend checkpointer 
must synchronize state without race conditions.

### State Synchronization Flow:
1. **Client Dispatches Mutation**: Client appends optimistic message and disables submit button.
2. **Superstep Barriers Emit Snapshots**: At the end of each superstep, the backend checkpointer commits the state 
   and emits an `on_state_update` event with the authoritative channel values.
3. **Reconciliation**: The client reconciles local optimistic data with the server snapshot, replacing temporary client IDs 
   with permanent database message IDs.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full-Stack End-to-End SSE Streaming and State Synchronization Architecture", divider="rainbow")

st.markdown("""
The following complete FastAPI module implements production SSE streaming with client reconnection handling, 
event serialization, and error recovery.
""")

st.code('''from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio

app = FastAPI(title="LangGraph SSE Production Bridge")

async def generate_sse_stream(thread_id: str, prompt: str) -> AsyncGenerator[str, None]:
    # 1. Emit Initial Metadata
    yield f"event: metadata\\ndata: {json.dumps({'thread_id': thread_id, 'status': 'connected'})}\\n\\n"
    
    try:
        # Simulate agent execution steps
        steps = [
            ("node_start", {"node": "planner", "superstep": 1}),
            ("token", {"content": "Analyzing"}),
            ("token", {"content": " customer"}),
            ("token", {"content": " records..."}),
            ("node_end", {"node": "planner", "status": "completed"}),
            ("state_snapshot", {"thread_id": thread_id, "messages_count": 2, "status": "idle"})
        ]
        
        for idx, (event_type, payload) in enumerate(steps):
            await asyncio.sleep(0.08)  # Simulated latency
            payload_str = json.dumps(payload)
            yield f"id: evt-{idx}\\nevent: {event_type}\\ndata: {payload_str}\\n\\n"
            
    except asyncio.CancelledError:
        # Client disconnected prematurely
        print(f"Client disconnected from thread {thread_id}")
    except Exception as e:
        yield f"event: error\\ndata: {json.dumps({'error': str(e)})}\\n\\n"

@app.post("/api/v1/threads/{thread_id}/stream")
async def stream_agent(thread_id: str, payload: dict):
    prompt = payload.get("message", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing message prompt")
        
    return StreamingResponse(
        generate_sse_stream(thread_id, prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disables Nginx reverse proxy buffering
        }
    )
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Frontend-Backend Network Inspector & Event Stream Simulator", divider="rainbow")

st.markdown("""
Simulate real-time Server-Sent Events streaming between a LangGraph backend and a web frontend.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("Stream Controls")
    simulate_drop = st.toggle("Simulate Packet Drop & Reconnection", value=False)
    drop_point = st.slider("Drop at Event Index", min_value=2, max_value=10, value=6)
    
    if st.button("Trigger Live Stream Simulation", type="primary"):
        st.session_state["stream_active"] = True

with col_wb2:
    st.subheader("Live SSE Packet Inspector")
    events = fe_sim.get_events()
    
    if simulate_drop:
        res = fe_sim.simulate_packet_loss_reconnect(drop_point)
        st.warning(f"Simulated network drop at event #{drop_point}. Client requested reconnection using Last-Event-ID: `{res['last_event_id']}`")
        st.info(f"Delivered: {res['delivered_count']} packets | Resumed: {res['missed_count']} packets")
    else:
        st.success("Network connection healthy: HTTP/2 SSE stream active.")

    display_events = events[:drop_point] if simulate_drop else events
    df_stream = pd.DataFrame([
        {
            "Index": idx,
            "Event Type": e["event"].upper(),
            "Payload": str(e["data"])
        }
        for idx, e in enumerate(display_events)
    ])
    st.dataframe(df_stream, use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: Why is Server-Sent Events (SSE) preferred over WebSockets for LangGraph frontend chat interfaces?"):
    st.markdown("""
    **Answer:**  
    1. **Simpler Protocol**: SSE runs over standard HTTP/2, simplifying load balancer and API gateway routing.  
    2. **Automatic Reconnection**: Browsers natively reconnect via `EventSource` and transmit `Last-Event-ID` to resume streams.  
    3. **Unidirectional Match**: Agent chat interactions are predominantly unidirectional (user sends single prompt, server streams multi-step execution).
    """)

with st.expander("Q2: How does the client reconcile local optimistic messages with backend checkpoint state?"):
    st.markdown("""
    **Answer:**  
    The client generates a temporary UUID for the optimistic message. When the backend checkpointer commits the state at the superstep barrier, 
    it returns the committed message array with database IDs and metadata. The client store performs a deduplication merge, replacing the temporary 
    optimistic record with the authoritative server record.
    """)

with st.expander("Q3: What causes buffering issues in Nginx when streaming SSE events to web clients?"):
    st.markdown("""
    **Answer:**  
    By default, Nginx buffers HTTP response bodies to optimize bandwidth. This buffers tokens until the buffer fills, completely breaking real-time token streaming. 
    Fix this by configuring the HTTP response header `X-Accel-Buffering: no` or setting `proxy_buffering off;` in Nginx configuration.
    """)

with st.expander("Q4: How does LangGraph handle concurrent user inputs while a thread is already running?"):
    st.markdown("""
    **Answer:**  
    LangGraph Server enforces thread-level queueing or locking. If a user submits a message while a run is active on that thread, 
    the API can either reject the request with `409 Conflict`, queue the run to execute immediately after the active run completes, 
    or interrupt the current run depending on concurrency settings.
    """)

with st.expander("Q5: What is the role of astream_events(v2) in full-stack frontend integration?"):
    st.markdown("""
    **Answer:**  
    `astream_events(version="v2")` emits granular, typed telemetry events for every node start/end, tool call, chain transition, 
    and LLM token chunk. This enables the frontend to render rich interactive visualizers (thought timelines, tool pills, progress bars) 
    rather than just raw final text.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Proxy Buffering Stalling Token Delivery**  
Failing to send `X-Accel-Buffering: no` on SSE endpoints causes reverse proxies (Nginx, AWS ALB, Cloudflare) to buffer streamed tokens, 
delivering the entire agent output in a single batch after 5 seconds instead of token-by-token.
""")

st.error("""
**Trap 2: State Desynchronization During Silent SSE Connection Drops**  
If the client's WiFi drops mid-stream, the browser may close the socket while the backend graph continues running. 
When the client reconnects, it must query `GET /threads/{id}/state` to synchronize with committed checkpoints rather than assuming the run failed.
""")

st.warning("""
**Trap 3: Memory Leaks from Unbounded SSE Event Listeners**  
In React components, failing to close the `EventSource` connection or unsubscribe from SDK streams in `useEffect` cleanup return functions 
causes lingering background sockets and browser memory leaks.
""")

st.warning("""
**Trap 4: Missing CORS Headers on SSE Endpoints**  
SSE streaming requires explicit CORS headers (`Access-Control-Allow-Origin`, `Access-Control-Allow-Headers`) allowing `EventSource` requests. 
Missing CORS blocks browser streaming entirely.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Server-Sent Events (SSE)</h4>
        <p>A unidirectional HTTP streaming protocol delivering real-time agent token chunks and node execution events to web clients.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Optimistic Updates</h4>
        <p>Frontend pattern rendering user messages immediately in the UI before receiving backend server acknowledgement.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>X-Accel-Buffering: no</h4>
        <p>A critical HTTP response header instructing reverse proxies like Nginx to disable buffering and deliver SSE chunks instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>astream_events(v2)</h4>
        <p>LangGraph async streaming method emitting fine-grained event payloads across nodes, tools, and model inference spans.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the full-stack architecture connecting React clients to LangGraph backend engines.", value=True)
chk_2 = st.checkbox("I know how to implement FastAPI SSE endpoints with text/event-stream media types.", value=True)
chk_3 = st.checkbox("I understand how to disable reverse proxy buffering using X-Accel-Buffering: no.", value=True)
chk_4 = st.checkbox("I can manage client state synchronization between optimistic updates and server checkpoints.", value=True)
chk_5 = st.checkbox("I know how to handle client reconnection and stream resumption using event IDs.", value=True)
chk_6 = st.checkbox("I understand how to stream granular telemetry using astream_events(version='v2').", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Frontend architecture mastery confirmed! Ready for real-time interface engineering.")
