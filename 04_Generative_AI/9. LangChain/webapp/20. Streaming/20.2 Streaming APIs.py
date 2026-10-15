import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.2 Streaming APIs - LangChain Architecture",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and events
st.markdown("""
<style>
    .api-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-event {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-node {
        background-color: #701A75;
        color: #F472B6;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockStreamingAPIsEngine:
    """
    Simulation engine showcasing LangChain and LangGraph streaming APIs:
    astream_events(v2), graph.stream(stream_mode=...), and partial tool call streaming.
    """
    def __init__(self):
        self.event_stream_database = [
            {"event": "on_chain_start", "name": "ConversationalAgent", "run_id": "run_01", "parent_id": None, "data": {"input": "Check flight UA-402 and weather."}},
            {"event": "on_chat_model_start", "name": "ChatOpenAI", "run_id": "run_02", "parent_id": "run_01", "data": {"model": "gpt-4o"}},
            {"event": "on_chat_model_stream", "name": "ChatOpenAI", "run_id": "run_02", "parent_id": "run_01", "data": {"chunk": "I will check the flight "}},
            {"event": "on_chat_model_stream", "name": "ChatOpenAI", "run_id": "run_02", "parent_id": "run_01", "data": {"chunk": "status and destination weather."}},
            {"event": "on_chat_model_end", "name": "ChatOpenAI", "run_id": "run_02", "parent_id": "run_01", "data": {"tool_calls": [{"name": "check_flight", "id": "tc_1"}]}},
            {"event": "on_tool_start", "name": "check_flight", "run_id": "run_03", "parent_id": "run_01", "data": {"args": {"flight_no": "UA-402"}}},
            {"event": "on_tool_end", "name": "check_flight", "run_id": "run_03", "parent_id": "run_01", "data": {"output": {"status": "On-Time", "dest": "SFO"}}},
            {"event": "on_chat_model_start", "name": "ChatOpenAI", "run_id": "run_04", "parent_id": "run_01", "data": {"model": "gpt-4o"}},
            {"event": "on_chat_model_stream", "name": "ChatOpenAI", "run_id": "run_04", "parent_id": "run_01", "data": {"chunk": "Flight UA-402 is on-time to SFO."}},
            {"event": "on_chat_model_end", "name": "ChatOpenAI", "run_id": "run_04", "parent_id": "run_01", "data": {"finish_reason": "stop"}},
            {"event": "on_chain_end", "name": "ConversationalAgent", "run_id": "run_01", "parent_id": None, "data": {"output": "Flight UA-402 is on-time to SFO."}}
        ]

    def filter_events(self, event_type_filter: str = "ALL", name_filter: str = "") -> List[Dict[str, Any]]:
        results = []
        for ev in self.event_stream_database:
            if event_type_filter != "ALL" and ev["event"] != event_type_filter:
                continue
            if name_filter and name_filter.lower() not in ev["name"].lower():
                continue
            results.append(ev)
        return results

# Main Application Title & Objective
st.title("🌊 20.2 Streaming APIs")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
LangChain provides a comprehensive hierarchy of streaming APIs tailored to different consumption patterns: 
from basic synchronous `stream()` and asynchronous `astream()`, to fine-grained event telemetry with `astream_events()`, 
and stateful graph streaming modes (`values`, `updates`, `messages`, and `custom`).
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: LangChain Streaming APIs & Pregel Event Dispatch Engine", divider="rainbow")
st.markdown("""
The architecture below delineates how execution calls propagate through LangChain's Runnables and LangGraph's Pregel engine. 
Standard invocations funnel through the internal callback and event bus, allowing clients to stream high-level state updates 
or tap directly into low-level LLM and tool lifecycle events.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    api_dot = graphviz.Digraph(comment="Streaming APIs")
    api_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    api_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    api_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    api_dot.node("client", "Client Request Invocation\n[Frontend WebApp / API Consumer]", fillcolor="#0F172A", shape="ellipse")
    api_dot.node("api_router", "API Interface Selection\n[stream / astream / astream_events / graph.stream]", shape="diamond", fillcolor="#334155")
    
    # API Branches
    api_dot.node("stream_sync", "stream()\n[Sync Iterator: Yields Output Chunks]", fillcolor="#1E3A8A")
    api_dot.node("astream_async", "astream()\n[Async Iterator: Async Event Loop Safe]", fillcolor="#0E7490")
    api_dot.node("events_engine", "astream_events(version='v2')\n[Event Bus: Model, Tool, Chain Lifecycle]", fillcolor="#701A75")
    api_dot.node("pregel_modes", "graph.stream(stream_mode=...)\n['values', 'updates', 'messages', 'custom']", fillcolor="#047857")

    api_dot.node("event_bus", "Unified Async Callback Bus\n[Standardized RunTree Event Dispatcher]", fillcolor="#4338CA")
    api_dot.node("sse_gateway", "Server-Sent Events (SSE) Stream\n[Pushed to Client Frontend]", fillcolor="#065F46", shape="ellipse")

    api_dot.edge("client", "api_router")
    api_dot.edge("api_router", "stream_sync", label=" Simple CLI / Sync")
    api_dot.edge("api_router", "astream_async", label=" FastAPI Async Endpoint")
    api_dot.edge("api_router", "events_engine", label=" Deep Observability")
    api_dot.edge("api_router", "pregel_modes", label=" LangGraph Workflows")

    api_dot.edge("stream_sync", "event_bus")
    api_dot.edge("astream_async", "event_bus")
    api_dot.edge("events_engine", "event_bus")
    api_dot.edge("pregel_modes", "event_bus")
    api_dot.edge("event_bus", "sse_gateway")

    st.graphviz_chart(api_dot, use_container_width=True)

# --- SECTION 2: STREAM() ---
st.header("2. Synchronous Streaming with stream()", divider="rainbow")
st.markdown("""
The synchronous `stream()` method is the most direct streaming interface available on any `Runnable`. 
It executes on the current Python thread and returns an `Iterator` that yields chunks sequentially as they become available.
""")

col_st1, col_st2 = st.columns(2)
with col_st1:
    st.subheader("Key Characteristics of `stream()`")
    st.markdown("""
    * **Blocking I/O**: Blocks the caller thread while waiting between network chunk arrivals.
    * **Ideal Use Cases**: CLI utilities, standalone batch processing scripts, or simple local Jupyter testing.
    * **Composition Support**: In an LCEL pipeline `chain = prompt | model | parser`, `chain.stream()` yields 
      streamed parser deltas automatically if the parser implements streaming support.
    """)
with col_st2:
    st.subheader("Synchronous LCEL Pipeline Streaming")
    st.code("""
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    PromptTemplate.from_template("Write a haiku about {topic}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# Returns a standard Python generator:
for chunk in chain.stream({"topic": "cybersecurity"}):
    print(chunk, end="", flush=True)
    """, language="python")

# --- SECTION 3: ASTREAM() ---
st.header("3. Asynchronous Streaming with astream()", divider="rainbow")
st.markdown("""
For modern web frameworks (FastAPI, Starlette, Quart, or Tornado), synchronous streaming is strictly prohibited because 
blocking threads destroys server concurrency. `astream()` operates as an **`AsyncIterator`**, yielding control back 
to the `asyncio` event loop between token emissions.
""")

col_ast1, col_ast2 = st.columns(2)
with col_ast1:
    st.subheader("FastAPI Integration Pattern")
    st.markdown("""
    * **High Concurrency**: Handles thousands of concurrent users per pod without thread starvation.
    * **Event-Driven**: Non-blocking network socket reads via `await`.
    * **Streaming Response**: Directly compatible with FastAPI's `StreamingResponse(generator(), media_type='text/event-stream')`.
    """)
with col_ast2:
    st.subheader("FastAPI Streaming Endpoint Example")
    st.code("""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def event_generator(query: str):
    async for chunk in chain.astream({"topic": query}):
        yield f"data: {chunk}\\n\\n"

@app.get("/stream")
async def stream_endpoint(query: str):
    return StreamingResponse(
        event_generator(query),
        media_type="text/event-stream"
    )
    """, language="python")

# --- SECTION 4: STREAMING EVENTS ---
st.header("4. Streaming Fine-Grained Events with astream_events()", divider="rainbow")
st.markdown("""
While `astream()` only emits the final output chunk of the entire chain, complex production architectures need to observe 
everything occurring under the hood: when an LLM starts, which tools are called, intermediate token streams, and child run lifecycles. 
**`astream_events(version="v2")`** is LangChain's comprehensive event streaming API.
""")

col_ev1, col_ev2 = st.columns(2)
with col_ev1:
    st.subheader("Standard Lifecycle Events in `v2`")
    st.markdown("""
    * `on_chat_model_start`: Triggered when an LLM call begins.
    * `on_chat_model_stream`: Emitted for every single token chunk produced by any model in the chain!
    * `on_chat_model_end`: Triggered when model generation terminates.
    * `on_tool_start`: Emitted when an agent calls an external tool with input parameters.
    * `on_tool_end`: Emitted when tool execution completes with return output.
    * `on_chain_start` / `on_chain_end`: Lifecycle hooks for intermediate chains and subgraphs.
    """)
with col_ev2:
    st.subheader("Event Filtering with `astream_events`")
    st.code("""
# Stream events and filter specifically for LLM tokens:
async for event in agent.astream_events(input_data, version="v2"):
    kind = event["event"]
    
    if kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
    elif kind == "on_tool_start":
        print(f"\\n[TOOL CALL]: {event['name']} with args {event['data'].get('input')}")
    """, language="python")

# --- SECTION 5: STREAMING AGENT STATES ---
st.header("5. Streaming Graph & Agent States (stream_mode)", divider="rainbow")
st.markdown("""
In LangGraph, stateful agents possess a shared state dictionary. When executing a graph, developers can specify 
exactly how that state is streamed back to the client using the **`stream_mode`** parameter.
""")

st.markdown("""
### LangGraph `stream_mode` Specifications
| `stream_mode` | Payload Format | Best Suited For |
| :--- | :--- | :--- |
| **`"values"`** | Full updated state dict after every node | Rendering complete chat window without frontend state management |
| **`"updates"`** | Dict keyed by node name: `{node: {key: delta}}` | Granular progress animations and node step indicators |
| **`"messages"`** | Tuple: `(AIMessageChunk, metadata)` | Direct LLM token streaming across internal graph nodes |
| **`"custom"`** | Arbitrary dict emitted by `writer()` | Custom progress bars and domain telemetry |
| **`["updates", "messages"]`** | Multiple modes combined simultaneously | Production apps needing both step tracking and token streaming |
""")

st.code("""
# Combining multiple streaming modes in LangGraph:
config = {"configurable": {"thread_id": "thread-1"}}

for mode, payload in graph.stream(
    {"messages": [("user", "Summarize report")]},
    config,
    stream_mode=["updates", "messages"]
):
    if mode == "messages":
        chunk, metadata = payload
        print(chunk.content, end="", flush=True)
    elif mode == "updates":
        print(f"\\nNode Finished: {list(payload.keys())[0]}")
""", language="python")

# --- SECTION 6: STREAMING TOOL CALLS ---
st.header("6. Streaming Tool Calls & Partial Argument Deserialization", divider="rainbow")
st.markdown("""
Modern tool calling models (OpenAI, Anthropic, Gemini) stream tool invocations as fragmented JSON chunks. 
LangChain's `AIMessageChunk.tool_call_chunks` captures these partial payloads incrementally before the complete tool call resolves.
""")

col_tc1, col_tc2 = st.columns(2)
with col_tc1:
    st.subheader("Tool Call Fragment Evolution")
    st.markdown("""
    * **Chunk 1**: `name='fetch_weather', id='call_1', args='{"cit'`
    * **Chunk 2**: `args='y": "Lon'`
    * **Chunk 3**: `args='don"}'`
    * **Client-Side Assembly**: Chunks accumulate until valid JSON syntax is detected.
    * **Predictive UI**: Frontends can display an early spinner (e.g. *Querying London weather...*) before the arguments even finish streaming!
    """)
with col_tc2:
    st.subheader("Inspecting Tool Call Chunks")
    st.code("""
async for chunk in model_with_tools.astream("What is the weather in London?"):
    if chunk.tool_call_chunks:
        for tc_chunk in chunk.tool_call_chunks:
            print(f"Tool Index: {tc_chunk['index']}")
            print(f"Arg Fragment: {tc_chunk['args']}")
    """, language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: High-Throughput Async astream_events Handler with Custom Filter Predicates", divider="rainbow")
st.markdown("""
The following complete production pattern implements an enterprise event multiplexer that consumes `astream_events(version="v2")`, 
applies custom filtering predicates, and yields clean, typed SSE-ready payloads.
""")

st.code('''
import json
from typing import AsyncIterator, Dict, Any, Callable
from langchain_core.runnables import Runnable

async def production_event_multiplexer(
    pipeline: Runnable,
    input_payload: dict,
    event_predicate: Callable[[dict], bool] = lambda e: True
) -> AsyncIterator[str]:
    """
    High-throughput event streaming handler that transforms fine-grained
    LangChain lifecycle events into formatted Server-Sent Event (SSE) strings.
    """
    # Stream events using version v2 API
    async for event in pipeline.astream_events(input_payload, version="v2"):
        # Apply filter predicate (e.g., discard noisy intermediate chain events)
        if not event_predicate(event):
            continue

        kind = event["event"]
        sse_event_type = "unknown"
        payload = {}

        if kind == "on_chat_model_stream":
            sse_event_type = "token"
            chunk = event["data"]["chunk"]
            payload = {"delta": chunk.content}

        elif kind == "on_tool_start":
            sse_event_type = "tool_start"
            payload = {
                "tool_name": event["name"],
                "arguments": event["data"].get("input")
            }

        elif kind == "on_tool_end":
            sse_event_type = "tool_end"
            payload = {
                "tool_name": event["name"],
                "output": event["data"].get("output")
            }

        elif kind == "on_chain_end" and event["parent_ids"] == []:
            sse_event_type = "completed"
            payload = {"final_output": event["data"].get("output")}

        # Format as standard Server-Sent Event (SSE)
        formatted_sse = f"event: {sse_event_type}\\ndata: {json.dumps(payload)}\\n\\n"
        yield formatted_sse
''', language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: astream_events Visualizer & Event Filter Sandbox", divider="rainbow")
st.markdown("""
Explore the fine-grained event hierarchy generated by LangChain's `astream_events(version="v2")`. 
Filter by event type or component name to inspect run IDs, parent-child hierarchies, and streaming deltas.
""")

# Instantiate Module Scope Simulator
if "api_sim" not in st.session_state:
    st.session_state.api_sim = MockStreamingAPIsEngine()

sim_api = st.session_state.api_sim

col_f1, col_f2 = st.columns([1, 1])

with col_f1:
    st.subheader("1. Event Stream Filters")
    type_options = ["ALL", "on_chat_model_stream", "on_tool_start", "on_tool_end", "on_chain_start", "on_chain_end"]
    selected_event_type = st.selectbox("Filter by Event Type:", options=type_options)
    name_filter = st.text_input("Filter by Component Name (Substring):", value="")

    filtered_events = sim_api.filter_events(selected_event_type, name_filter)
    st.markdown(f"**Filtered Events Count**: `{len(filtered_events)}`")

with col_f2:
    st.subheader("2. Event Stream Feed")
    for ev in filtered_events:
        st.markdown(f"""
        <div class="api-card">
            <span class="badge-event">{ev['event']}</span> 
            <span class="badge-node">{ev['name']}</span>
            <p><strong>Run ID:</strong> <code>{ev['run_id']}</code> | <strong>Parent:</strong> <code>{ev['parent_id']}</code></p>
            <p><strong>Payload Data:</strong> <code>{ev['data']}</code></p>
        </div>
        """, unsafe_allow_html=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: What are the key architectural improvements introduced in `astream_events(version='v2')` over v1?"):
    st.markdown("""
    `astream_events(v2)` provides standardized, normalized event schemas across all component types, guarantees consistent 
    parent-child `parent_ids` lineage tracking, handles edge cases around nested tool streams without duplicate events, 
    and significantly improves JSON serialization efficiency for high-throughput async workloads.
    """)

with st.expander("Q2: Why should `stream_mode=['updates', 'messages']` be used in modern production agent UIs?"):
    st.markdown("""
    Using both modes simultaneously provides the best of both worlds:
    * `messages`: Delivers real-time next-token streaming directly from the internal LLM call to achieve sub-second TTFT.
    * `updates`: Delivers deterministic node completion events so the UI can advance step progress bars and display tool call badges.
    """)

with st.expander("Q3: What causes an `AsyncIterator` memory leak in FastAPI when streaming LLM responses?"):
    st.markdown("""
    If a client abruptly disconnects (closes tab or loses Wi-Fi) while an `AsyncIterator` is awaiting a chunk, the generator 
    may hang in an un-closed state if not wrapped in proper cancellation exception handling. In FastAPI, using `asyncio.shield` 
    or listening for the `request.is_disconnected()` event ensures resources are freed immediately.
    """)

with st.expander("Q4: How do tool call arguments stream across multiple chunks, and when can they be safely executed?"):
    st.markdown("""
    Tool call arguments arrive as fragmented string deltas inside `chunk.tool_call_chunks`. They must never be executed immediately. 
    The application must accumulate the fragments until `chunk.tool_calls` appears or the model emits an `on_chat_model_end` event, 
    signifying that the JSON payload has closed and validated.
    """)

with st.expander("Q5: Can you stream intermediate results from inside a custom LangGraph node without returning early?"):
    st.markdown("""
    Yes. By accepting the `writer: StreamWriter` argument in the node signature and configuring `stream_mode='custom'`, 
    the node can call `writer(custom_data)` at any point during its execution to push live data to the client stream 
    while continuing its internal computation to completion.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_at1, col_at2 = st.columns(2)
with col_at1:
    st.error("""
    **Production Trap 1 (Fatal Error): Mixing Sync `stream()` into Async Event Loops**  
    Calling `for chunk in chain.stream():` inside a FastAPI async route handler blocks the central `asyncio` event loop thread, 
    freezing all other concurrent requests arriving at that server worker. Always use `async for chunk in chain.astream():`.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Unhandled Empty Chunks in SSE Streams**  
    Certain intermediate events emit empty content chunks (e.g. metadata-only chunks). 
    Failing to guard against empty strings causes the frontend to receive blank deltas, increasing unnecessary DOM render cycles.
    """)
with col_at2:
    st.error("""
    **Production Trap 3 (Fatal Error): Omission of Version Specifier in `astream_events`**  
    Invoking `astream_events()` without explicitly declaring `version="v2"`. 
    Omitting this parameter defaults to legacy `v1` schemas which exhibit inconsistent parent run IDs and will raise deprecation errors in future releases.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): High-Frequency SSE Network Throttling**  
    Emitting single-character token events over cellular mobile networks can overwhelm mobile browser networking layers. 
    In production, implement a micro-batcher that groups tokens arriving within a 15ms window into a single SSE frame.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fcd1, col_fcd2 = st.columns(2)
with col_fcd1:
    st.markdown("""
    <div class="api-card">
        <h4>Flashcard 1: astream_events</h4>
        <p><strong>API:</strong> <code>chain.astream_events(version="v2")</code></p>
        <p><strong>Definition:</strong> Comprehensive async API that emits fine-grained lifecycle events for models, tools, and chains across the execution tree.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="api-card">
        <h4>Flashcard 2: stream_mode='updates'</h4>
        <p><strong>Mode:</strong> LangGraph State Streaming</p>
        <p><strong>Definition:</strong> Emits only the state changes returned by the most recently executed node, optimizing payload bandwidth.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fcd2:
    st.markdown("""
    <div class="api-card">
        <h4>Flashcard 3: StreamWriter</h4>
        <p><strong>Primitive:</strong> Custom Telemetry Emitter</p>
        <p><strong>Definition:</strong> A callable passed into graph nodes allowing arbitrary custom events to stream to the client under <code>stream_mode='custom'</code>.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="api-card">
        <h4>Flashcard 4: tool_call_chunks</h4>
        <p><strong>Property:</strong> Partial Argument Deltas</p>
        <p><strong>Definition:</strong> The attribute on <code>AIMessageChunk</code> containing incremental fragments of structured tool names and arguments.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of LangChain streaming APIs and event multiplexing patterns:")

ch1 = st.checkbox("1. I understand when to use synchronous `stream()` versus asynchronous `astream()` in web services.")
ch2 = st.checkbox("2. I know how to use `astream_events(version='v2')` to observe fine-grained model and tool lifecycles.")
ch3 = st.checkbox("3. I understand the distinct behaviors of LangGraph's `values`, `updates`, and `messages` streaming modes.")
ch4 = st.checkbox("4. I know how to handle partial tool call argument streaming without corrupting JSON payloads.")
ch5 = st.checkbox("5. I understand how `StreamWriter` enables custom progress emission from within long-running graph nodes.")
ch6 = st.checkbox("6. I understand why mixing synchronous streaming into an async event loop causes fatal worker thread starvation.")

if ch1 and ch2 and ch3 and ch4 and ch5 and ch6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Streaming APIs in LangChain & LangGraph.")
