import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="18.1 Streaming Basics - LangGraph Curriculum",
    page_icon="⚡",
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
    .badge-stream {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-token {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class StreamingBasicsSimulator:
    """
    Simulates real-time streaming lifecycles in LangGraph:
    - Synchronous stream() vs Async astream() execution
    - Superstep node deltas
    - Token-by-token emission latency
    - State snapshot progression
    """
    def __init__(self):
        self.tokens_sample = [
            "Based", " on", " the", " query", ",", " the", " verified", " answer",
            " is", " 42", ".", " Let", " me", " double", " check", " the",
            " calculation", " with", " tool", " output", "."
        ]

    def simulate_node_stream(self, query: str) -> List[Dict[str, Any]]:
        events = []
        # Superstep 1: Planning / Retrieval
        events.append({
            "superstep": 1,
            "node": "retrieve_context",
            "delta": {"context_docs": [f"Knowledge snippet matching '{query}'"]},
            "timestamp_ms": 120
        })
        # Superstep 2: LLM Reasoning
        events.append({
            "superstep": 2,
            "node": "agent_llm",
            "delta": {"draft_answer": "Draft response synthesized."},
            "timestamp_ms": 480
        })
        # Superstep 3: Quality Check
        events.append({
            "superstep": 3,
            "node": "verify_quality",
            "delta": {"final_score": 0.98, "status": "APPROVED"},
            "timestamp_ms": 610
        })
        return events

    def simulate_token_stream(self, token_delay_ms: int) -> List[Dict[str, Any]]:
        emissions = []
        cumulative = ""
        current_time = 0
        for idx, tok in enumerate(self.tokens_sample):
            current_time += token_delay_ms + int(np.random.uniform(-5, 10))
            current_time = max(current_time, 0)
            cumulative += tok
            emissions.append({
                "token_index": idx + 1,
                "token": tok,
                "cumulative_text": cumulative,
                "elapsed_ms": current_time
            })
        return emissions


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Streaming Lifecycles and Event Distribution in LangGraph", divider="rainbow")

st.markdown("""
In complex multi-node agent applications, an end-to-end execution can take anywhere from 5 to 60+ seconds.
Waiting for the final result without real-time feedback leads to poor user experience, high perceived latency, and timeouts.

LangGraph is built from the ground up on an **asynchronous event-driven architecture**.
Instead of blocking until the entire state graph reaches `END`, LangGraph exposes real-time streaming interfaces that emit:
- **Graph State Updates**: Emitted after each superstep barrier completes.
- **Node-by-Node Deltas**: Specific channel additions yielded by individual worker nodes.
- **LLM Tokens**: Sub-word tokens emitted on-the-fly as language models generate output text.

The blueprint below contrasts traditional blocking execution against LangGraph's streaming pipeline.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StreamingArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_input {
            label="1. Ingress & Invocation Channel";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_client [label="Client / Web UI\\n(FastAPI / Next.js / Streamlit)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            stream_entry [label="app.stream() / app.astream()\\n(Initial State + Config)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_pregel_engine {
            label="2. Pregel Engine Supersteps & Event Generation";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            step1 [label="Superstep 1: Planner Node\\nEmits: {'plan': '...'}", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            step2 [label="Superstep 2: LLM Tool Node\\nEmits: Live LLM Tokens (a-stream)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            step3 [label="Superstep 3: Formatter Node\\nEmits: Final Verified Response", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_event_bus {
            label="3. Event Stream Demultiplexer";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            mode_updates [label="stream_mode='updates'\\nNode-Level State Deltas", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            mode_values [label="stream_mode='values'\\nFull State Snapshots", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            mode_messages [label="stream_mode='messages'\\nToken-Level Chat Chunks", fillcolor="#701A75", fontcolor="#F5D0FE", color="#E879F9"];
        }

        subgraph cluster_client_rendering {
            label="4. Real-Time UI Rendering Engine";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            ui_progress [label="Agent Status Badges\\n('Searching docs...', 'Verifying')", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_typewriter [label="Typewriter Output\\nSub-second Time to First Token", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        user_client -> stream_entry;
        stream_entry -> step1 [label="Starts Step 1"];
        step1 -> mode_updates [label="Emits Node Delta"];
        step1 -> step2 [label="Starts Step 2"];
        step2 -> mode_messages [label="Emits Token Stream"];
        step2 -> step3 [label="Starts Step 3"];
        step3 -> mode_values [label="Emits Final Snapshot"];

        mode_updates -> ui_progress [label="Renders Node Progress"];
        mode_messages -> ui_typewriter [label="Renders Token Text"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: WHY STREAMING?
# ==============================================================================
st.header("2. Why Streaming?: Latency Perception, User Feedback, and Real-Time Responsiveness", divider="rainbow")

st.markdown("""
In conversational and multi-agent systems, **Time to First Token (TTFT)** is the single most important user experience metric.
Consider an agent that analyzes financial reports:
- **Blocking Invocation (`invoke`)**: The user stares at a spinner for 18 seconds before seeing any text. The user assumes the application is frozen.
- **Streaming Invocation (`stream`)**: Within 400 milliseconds, the user sees `Searching SEC EDGAR database...`, followed by token-by-token synthesis. The perceived latency is near zero.
""")

col_ws1, col_ws2 = st.columns(2)
with col_ws1:
    st.markdown("#### Perceived Latency Comparison")
    df_lat = pd.DataFrame({
        "Metric": ["Time to First Visual Feedback", "Perceived Wait Time", "Long-Polling HTTP Risk", "Debugging Visibility"],
        "app.invoke() (Blocking)": ["15 - 30 seconds", "Frustrating / Abandonment", "High (Gateway 504 Timeouts)", "Blind box until completion"],
        "app.stream() (Streaming)": ["300 - 600 milliseconds", "Instantaneous / Engaging", "Zero (SSE / WebSocket keepalive)", "Real-time superstep audit"]
    })
    st.dataframe(df_lat, hide_index=True, use_container_width=True)

with col_ws2:
    st.markdown("#### Primary Streaming Benefits")
    st.markdown("""
    1. **Eliminate Gateway Timeouts**: Reverse proxies like Nginx or AWS ALB terminate connections after 60 seconds if no bytes are transferred. Streaming keeps HTTP connections warm via Server-Sent Events (SSE).
    2. **Agent Observability**: Users and operators watch sub-tasks complete in real-time.
    3. **Early Cancellation**: Users can interrupt runaway or misdirected reasoning before expensive model runs finish.
    """)

# ==============================================================================
# SECTION 3: THE stream() SYNCHRONOUS GENERATOR
# ==============================================================================
st.header("3. The stream() Synchronous Generator: Chunk-by-Chunk State Iteration", divider="rainbow")

st.markdown("""
The `stream()` method executes the graph synchronously and returns a standard Python generator.
Each iteration yields an event as soon as the corresponding superstep or token completes.
""")

st.code('''from langgraph.graph import StateGraph

# Synchronous streaming iteration
for chunk in app.stream({"messages": [("user", "Summarize Q3 earnings")]}, config={"configurable": {"thread_id": "t1"}}):
    # By default, chunk is a dict where keys are node names and values are their channel updates
    for node_name, node_update in chunk.items():
        print(f"--> Node '{node_name}' finished superstep:")
        print(f"    Payload: {node_update}\\n")''', language="python")

st.markdown("""
> **Generator Behavior**: `stream()` executes locally and blocks between supersteps, yielding objects sequentially. It is ideal for command-line interfaces, background workers, and synchronous Streamlit scripts.
""")

# ==============================================================================
# SECTION 4: THE astream() ASYNC GENERATOR
# ==============================================================================
st.header("4. The astream() Async Generator: Non-Blocking Web Server Concurrency", divider="rainbow")

st.markdown("""
In asynchronous web frameworks like FastAPI, Quart, or Starlette, synchronous generators block the event loop.
LangGraph provides `astream()`, an **asynchronous generator** that yields chunks cooperatively using `async for`.
""")

st.code('''import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

fastapi_app = FastAPI()

@fastapi_app.post("/chat/stream")
async def chat_stream_endpoint(request_data: dict):
    async def event_generator():
        # astream() yields control back to asyncio event loop between steps
        async for chunk in app.astream(
            {"messages": [("user", request_data["query"])]},
            config={"configurable": {"thread_id": request_data["thread_id"]}},
            stream_mode="updates"
        ):
            # Format as Server-Sent Events (SSE) standard
            yield f"data: {chunk}\\n\\n"
            await asyncio.sleep(0)  # cooperative yield
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")''', language="python")

# ==============================================================================
# SECTION 5: STREAMING GRAPH STATE
# ==============================================================================
st.header("5. Streaming Graph State: Observing Full-State Snapshots Across Supersteps", divider="rainbow")

st.markdown("""
When configuring `stream_mode="values"`, LangGraph yields the **entire current state** after each superstep completes.
""")

st.markdown("""
- **Initial Yield**: Emits the initial input state passed to `stream()`.
- **Intermediate Yields**: Emits the complete updated state dictionary after each node runs and updates channels.
- **Final Yield**: Emits the terminal state snapshot right before graph completion.
""")

st.code('''# Streaming complete state progression
for state_snapshot in app.stream(initial_input, stream_mode="values"):
    print("Messages count:", len(state_snapshot["messages"]))
    print("Current accumulator:", state_snapshot.get("summary", ""))''', language="python")

# ==============================================================================
# SECTION 6: STREAMING LLM TOKENS
# ==============================================================================
st.header("6. Streaming LLM Tokens: Token-by-Token Streaming from AI Messages", divider="rainbow")

st.markdown("""
While streaming graph states tells you when *nodes* finish, conversational applications need **sub-word token streaming** directly from the underlying LLM.
LangGraph achieves token streaming using `stream_mode="messages"`.
""")

st.code('''# Token-level streaming chunk by chunk
for message_chunk, metadata in app.stream(
    {"messages": [("user", "Explain quantum computing in one sentence.")]},
    stream_mode="messages"
):
    # Only stream tokens from the assistant AI node
    if metadata.get("langgraph_node") == "assistant":
        if hasattr(message_chunk, "content") and message_chunk.content:
            print(message_chunk.content, end="", flush=True)''', language="python")

st.markdown("""
Each emitted item in `"messages"` mode is a 2-tuple:
1. `message_chunk`: An `AIMessageChunk` containing newly generated text fragments or partial tool call definitions.
2. `metadata`: Dictionary containing the producing `langgraph_node`, step index, and run ID.
""")

# ==============================================================================
# SECTION 7: STREAMING NODE UPDATES
# ==============================================================================
st.header("7. Streaming Node Updates: Granular Superstep Delta Tracking", divider="rainbow")

st.markdown("""
The default mode in LangGraph is `stream_mode="updates"`.
In this mode, LangGraph yields a dictionary where:
- Keys are the names of the nodes that executed in that superstep.
- Values are the *exact updates* returned by those nodes (the channel deltas).
""")

st.code('''# Observing granular node deltas
for update in app.stream(inputs, stream_mode="updates"):
    for node_name, delta in update.items():
        print(f"Node '{node_name}' modified channels: {list(delta.keys())}")''', language="python")

st.info("💡 **Key Architectural Distinction**: `values` mode yields the whole accumulated state snapshot; `updates` mode yields only what the node just returned in that superstep.")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Streaming Graph with Token & Node Event Handlers", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern demonstrating an agent with streaming support, checkpointer configuration, and node delta parsing.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class StreamAgentState(TypedDict):
    query: str
    intermediate_steps: List[str]
    final_output: str

def planner_node(state: StreamAgentState) -> dict:
    return {
        "intermediate_steps": ["Plan created: 1. Search docs, 2. Synthesize"]
    }

def retrieval_node(state: StreamAgentState) -> dict:
    return {
        "intermediate_steps": state.get("intermediate_steps", []) + ["Retrieved 3 financial filings"]
    }

def synthesis_node(state: StreamAgentState) -> dict:
    return {
        "final_output": f"Comprehensive analysis for '{state['query']}' successfully generated."
    }

# Build graph
builder = StateGraph(StreamAgentState)
builder.add_node("planner", planner_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("synthesis", synthesis_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "retrieval")
builder.add_edge("retrieval", "synthesis")
builder.add_edge("synthesis", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# Production Consumer Function
def run_streaming_consumer(user_query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"--- Starting Stream for Thread: {thread_id} ---")
    
    for event in app.stream({"query": user_query}, config=config, stream_mode="updates"):
        for node_name, delta in event.items():
            print(f"[EVENT] Node '{node_name}' completed. Delta: {delta}")

run_streaming_consumer("Evaluate Apple Q3 Cash Flow", "stream_thread_01")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Real-Time Stream Simulator and Token Latency Analyzer", divider="rainbow")

st.markdown("""
Simulate and inspect how LangGraph streams node updates and individual LLM tokens in real time.
""")

sim = StreamingBasicsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("📡 Simulation 1: Superstep Node Stream (stream_mode='updates')")
    user_query_input = st.text_input("Simulated User Prompt", value="Calculate Q4 gross margin")
    
    if st.button("🚀 Stream Supersteps"):
        events = sim.simulate_node_stream(user_query_input)
        progress_bar = st.progress(0)
        
        for idx, ev in enumerate(events):
            st.markdown(f"**Superstep {ev['superstep']}**: Node `{ev['node']}` completed in `{ev['timestamp_ms']}ms`")
            st.json(ev["delta"])
            progress_bar.progress((idx + 1) / len(events))
        st.success("🎉 Graph Execution Stream Complete!")

with col_sim2:
    st.subheader("🔤 Simulation 2: Sub-Word Token Emission (stream_mode='messages')")
    delay_input = st.slider("Inter-Token Delay (ms)", min_value=10, max_value=200, value=50, step=10)

    if st.button("⚡ Simulate Live Token Stream"):
        emissions = sim.simulate_token_stream(token_delay_ms=delay_input)
        placeholder = st.empty()
        
        for em in emissions:
            placeholder.markdown(f"**Streaming Output**: `{em['cumulative_text']}` ▌")
        
        placeholder.markdown(f"**Final Output**: `{emissions[-1]['cumulative_text']}`")
        st.info(f"Total Tokens: {len(emissions)} | Total Elapsed Time: {emissions[-1]['elapsed_ms']} ms")
        
        df_tokens = pd.DataFrame(emissions)[["token_index", "token", "elapsed_ms"]]
        st.dataframe(df_tokens.head(8), use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the fundamental difference between app.stream() and app.astream()?", expanded=False):
    st.markdown("""
    **Answer**:
    `app.stream()` is a synchronous generator that blocks between supersteps, intended for CLI tools and synchronous Python environments.
    `app.astream()` is an asynchronous generator (`async for`) designed for async web servers (FastAPI, aiohttp) that yields execution back to the event loop, enabling high concurrency without blocking server worker threads.
    """)

with st.expander("Q2: How do stream_mode='values' and stream_mode='updates' differ in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    - `stream_mode='values'` yields the complete accumulated state dictionary after each superstep barrier.
    - `stream_mode='updates'` yields only the dictionary returned by the active node(s) during that specific superstep (the state deltas).
    """)

with st.expander("Q3: How do you stream sub-word LLM tokens instead of whole graph state snapshots?", expanded=False):
    st.markdown("""
    **Answer**:
    By configuring `stream_mode='messages'`.
    In this mode, LangGraph yields `(message_chunk, metadata)` tuples, where `message_chunk` is an `AIMessageChunk` containing streaming text pieces emitted directly by the LLM runnable.
    """)

with st.expander("Q4: Can you stream both node updates and LLM tokens simultaneously in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! In modern LangGraph releases, you can pass a list of stream modes:
    `app.stream(inputs, stream_mode=['updates', 'messages'])`.
    Each emitted item is a 2-tuple `(stream_mode, chunk)` allowing the client to handle UI progress bars and text streaming on a single connection.
    """)

with st.expander("Q5: Why does streaming prevent HTTP gateway timeouts in production microservices?", expanded=False):
    st.markdown("""
    **Answer**:
    Standard HTTP requests block until the entire response is available, triggering gateway timeouts (HTTP 504) if agent execution takes longer than 30–60 seconds.
    Streaming sends bytes immediately (via Server-Sent Events or WebSockets), keeping the TCP connection active.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Calling Synchronous app.stream() Inside Async FastAPI Endpoints")
    st.markdown("""
    *Mechanism*: Iterating over `app.stream()` in an `async def` FastAPI route.  
    *Consequence*: Blocks the asyncio event loop entirely, preventing the web server from serving concurrent requests.  
    *Fix*: Always use `async for chunk in app.astream(...)` inside async route handlers.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Forgetting to Flush Buffers in Reverse Proxies")
    st.markdown("""
    *Mechanism*: Deploying streaming behind Nginx without configuring `proxy_buffering off;`.  
    *Consequence*: Nginx buffers all streaming chunks until the response completes, destroying real-time streaming for the end-user.  
    *Fix*: Set HTTP header `X-Accel-Buffering: no` or disable Nginx proxy buffering for streaming endpoints.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Attempting to Stream Without Model Streaming Support")
    st.markdown("""
    *Mechanism*: Using an LLM wrapper initialized with `streaming=False` and expecting `messages` mode to yield tokens.  
    *Consequence*: The model yields the entire response at once at the end of the node, defeating token streaming.  
    *Fix*: Ensure the underlying LangChain chat model has streaming enabled (default in modern versions).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Reliability Warning): Consuming Streams Without Handling Client Disconnects")
    st.markdown("""
    *Mechanism*: The client closes their browser tab, but the async generator continues running expensive LLM calls to completion.  
    *Consequence*: Unnecessary cloud and LLM token costs.  
    *Fix*: Monitor `request.is_disconnected()` in FastAPI and cancel the underlying task immediately.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**stream()**\\nSynchronous generator yielding graph events chunk by chunk.")
with col_fc2:
    st.info("**astream()**\\nAsynchronous generator yielding chunks cooperatively for async web apps.")
with col_fc3:
    st.info("**stream_mode='updates'**\\nEmits only the channel deltas produced by nodes in each superstep.")
with col_fc4:
    st.info("**stream_mode='messages'**\\nEmits sub-word token chunks directly from LLM completions.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know why streaming is essential for perceived latency and preventing HTTP timeouts.", value=True)
c2 = st.checkbox("I understand the difference between stream() and astream().", value=True)
c3 = st.checkbox("I know what stream_mode='values' emits compared to stream_mode='updates'.", value=True)
c4 = st.checkbox("I can stream sub-word tokens using stream_mode='messages'.", value=True)
c5 = st.checkbox("I know how to consume stream chunks inside an async FastAPI Server-Sent Events endpoint.", value=True)
c6 = st.checkbox("I understand why proxy_buffering off is necessary in reverse proxies like Nginx.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 18.1 'Streaming Basics' with full architectural and practical comprehension!")
