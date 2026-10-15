import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.1 Streaming Basics - LangChain Architecture",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and chunk visualization
st.markdown("""
<style>
    .stream-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .token-chip {
        display: inline-block;
        background-color: #0F172A;
        border: 1px solid #38BDF8;
        color: #38BDF8;
        padding: 2px 6px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .badge-ttft {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockStreamingBasicsSimulator:
    """
    Deterministic multi-mode streaming engine simulating token, message,
    agent thought, and tool output chunk emissions.
    Demonstrates TTFT (Time-to-First-Token) advantages and chunk aggregation.
    """
    def __init__(self):
        self.scenarios = {
            "token_stream": {
                "name": "LLM Token Stream",
                "chunks": [
                    {"type": "token", "content": "Lang", "latency_ms": 120},
                    {"type": "token", "content": "Chain ", "latency_ms": 40},
                    {"type": "token", "content": "provides ", "latency_ms": 35},
                    {"type": "token", "content": "first-class ", "latency_ms": 45},
                    {"type": "token", "content": "streaming ", "latency_ms": 38},
                    {"type": "token", "content": "primitives ", "latency_ms": 42},
                    {"type": "token", "content": "for ", "latency_ms": 30},
                    {"type": "token", "content": "reactive ", "latency_ms": 35},
                    {"type": "token", "content": "UI ", "latency_ms": 40},
                    {"type": "token", "content": "architectures.", "latency_ms": 50}
                ]
            },
            "agent_stream": {
                "name": "ReAct Agent Trajectory Stream",
                "chunks": [
                    {"type": "thought", "content": "[THOUGHT] User requests account balance query.", "latency_ms": 200},
                    {"type": "tool_call", "content": "[ACTION] query_db(account_id='AC-992')", "latency_ms": 150},
                    {"type": "tool_output", "content": "[OBSERVATION] Balance: 14,250.00 USD (Clear)", "latency_ms": 310},
                    {"type": "final_token", "content": "[RESPONSE] Your ", "latency_ms": 80},
                    {"type": "final_token", "content": "current ", "latency_ms": 40},
                    {"type": "final_token", "content": "balance ", "latency_ms": 35},
                    {"type": "final_token", "content": "is ", "latency_ms": 30},
                    {"type": "final_token", "content": "14,250.00 USD.", "latency_ms": 45}
                ]
            },
            "tool_stream": {
                "name": "Tool Chunk Payload Stream",
                "chunks": [
                    {"type": "chunk", "content": '{"status": "searching", ', "latency_ms": 180},
                    {"type": "chunk", "content": '"progress": 45, ', "latency_ms": 120},
                    {"type": "chunk", "content": '"records_found": 1200, ', "latency_ms": 140},
                    {"type": "chunk", "content": '"completed": true}', "latency_ms": 90}
                ]
            }
        }

    def emit_stream(self, scenario_key: str) -> List[Dict[str, Any]]:
        return self.scenarios.get(scenario_key, {}).get("chunks", [])

# Main Application Title & Objective
st.title("⚡ 20.1 Streaming Basics")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Streaming is the foundational technique for reducing perceived user latency from multi-second blocking waits down to 
sub-second visual feedback. In modern LLM engineering, streaming spans multiple abstraction layers: **raw token streaming**, 
**chat message delta streaming**, **agent thought trajectory streaming**, and **tool chunk emission**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Multi-Layer Streaming Pipeline & Chunk Aggregators", divider="rainbow")
st.markdown("""
The architecture below contrasts the legacy **Blocking Batch Pipeline** with the modern **Multi-Layer Reactive Streaming Pipeline**. 
Instead of waiting for the entire response to buffer before emitting, the model stream yields incremental chunks that are dynamically 
accumulated by client-side reducers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    pipe_dot = graphviz.Digraph(comment="Streaming Pipeline")
    pipe_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    pipe_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    pipe_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    pipe_dot.node("user_req", "User Query / Prompt Request\n[Client Initiates Stream]", fillcolor="#0F172A", shape="ellipse")
    pipe_dot.node("llm_engine", "LLM Inference Engine\n[Autoregressive Next-Token Generation]", fillcolor="#1E293B")
    
    # Streaming Layer Tiers
    pipe_dot.node("token_emitter", "1. Raw Token Stream\n[HTTP Server-Sent Events / SSE Chunks]", fillcolor="#0284C7")
    pipe_dot.node("msg_chunk", "2. AIMessageChunk Reducer\n[add_messages / Chunk Addition Operator (+)]", fillcolor="#4338CA")
    pipe_dot.node("agent_events", "3. Agent Trajectory Stream\n[Thoughts, Tool Invocations & Observables]", fillcolor="#701A75")
    pipe_dot.node("tool_emitter", "4. Tool Data Stream\n[Live Progress & Partial DB/API Payloads]", fillcolor="#047857")

    pipe_dot.node("ui_consumer", "Client Frontend UI\n[Instant Sub-Second TTFT Visual Feedback]", fillcolor="#065F46", shape="ellipse")

    pipe_dot.edge("user_req", "llm_engine")
    pipe_dot.edge("llm_engine", "token_emitter", label=" Chunk 0 (TTFT ~ 200ms)")
    pipe_dot.edge("token_emitter", "msg_chunk", label=" Chunk Concat")
    pipe_dot.edge("llm_engine", "agent_events", label=" Intermediate Steps")
    pipe_dot.edge("agent_events", "tool_emitter", label=" Tool Execution")
    pipe_dot.edge("msg_chunk", "ui_consumer", label=" Continuous Yield")
    pipe_dot.edge("tool_emitter", "ui_consumer", label=" Progress Events")

    st.graphviz_chart(pipe_dot, use_container_width=True)

# --- SECTION 2: WHAT IS STREAMING? ---
st.header("2. What is Streaming?", divider="rainbow")
st.markdown("""
In traditional synchronous computing, a function call blocks until the entire computation terminates: `response = llm.invoke(prompt)`. 
For a 1,000-token generation, the user might stare at a spinning indicator for 8 to 15 seconds.
""")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.subheader("Blocking vs. Streaming Paradigms")
    st.markdown("""
    * **Blocking (`invoke`)**:
      * Total Latency = Model processing time + Network roundtrip.
      * Memory Buffering: Entire generation stored in server RAM before transmission.
      * High drop-off rate: Users perceive the system as sluggish or unresponsive.
    * **Streaming (`stream` / `astream`)**:
      * Emits an `Iterator` or `AsyncIterator` over incremental deltas.
      * As each token is sampled by the transformer's softmax head, it is pushed over the wire immediately.
      * Reduces perceived latency from seconds to milliseconds.
    """)
with col_s2:
    st.subheader("Latency Metrics Comparison")
    st.markdown("""
    | Operational Metric | Synchronous (invoke) | Streaming (stream) |
    | :--- | :--- | :--- |
    | **Time-to-First-Token (TTFT)** | 6,500 ms (Entire output) | 220 ms (Instant delta) |
    | **Total Generation Time** | 6,500 ms | 6,550 ms (Near identical) |
    | **Perceived Responsiveness** | Poor (Freeze / Stall) | Immediate (Interactive) |
    | **Memory Overhead** | Full buffer retained | Minimal per-chunk frame |
    | **Network Protocol** | Standard HTTP POST | HTTP 1.1 Chunked / SSE |
    """)

# --- SECTION 3: TOKEN STREAMING ---
st.header("3. Token Streaming & Time-to-First-Token (TTFT)", divider="rainbow")
st.markdown("""
**Token streaming** is the process of yielding text tokens individually or in micro-batches as they are emitted by the autoregressive decoder. 
The critical North Star metric in GenAI user experience is **TTFT (Time-to-First-Token)**.
""")

col_tk1, col_tk2 = st.columns(2)
with col_tk1:
    st.subheader("The Mechanics of TTFT")
    st.markdown("""
    1. **Prefill Phase (Prompt Evaluation)**: The model processes the input prompt in parallel across GPU tensor cores. 
       This phase accounts for almost 100% of the delay before TTFT.
    2. **Decode Phase (Autoregressive Generation)**: Tokens are generated sequentially one by one. 
       Each decode step takes roughly 20 to 50 milliseconds depending on model parameter size and quantization.
    3. **Streaming Advantage**: By streaming, the user starts reading after the prefill phase finishes, completely masking the subsequent decode duration!
    """)
with col_tk2:
    st.subheader("LangChain Token Streaming Code")
    st.code("""
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", streaming=True)

# stream() returns a generator of AIMessageChunk objects
for chunk in model.stream("Explain quantum superposition in 2 sentences."):
    print(chunk.content, end="", flush=True)
    """, language="python")

# --- SECTION 4: MESSAGE STREAMING ---
st.header("4. Message Streaming & Chat Generation Chunks", divider="rainbow")
st.markdown("""
In LangChain, streamed tokens do not arrive as raw strings; they are structured as **`AIMessageChunk`** objects. 
`AIMessageChunk` supports the mathematical addition operator (`+`), allowing chunks to be aggregated deterministically.
""")

col_msg1, col_msg2 = st.columns(2)
with col_msg1:
    st.subheader("`AIMessageChunk` Capabilities")
    st.markdown("""
    * **Content Accumulation**: `chunk1 + chunk2` concatenates the underlying string content.
    * **Tool Call Assembly**: Partial tool call arguments (e.g. fragments of a JSON string) are merged across chunks.
    * **Metadata Propagation**: Response metadata (finish reasons, token usage counts) is populated on the terminal chunk.
    * **Role Preservation**: Maintains the `ai` role metadata throughout the pipeline.
    """)
with col_msg2:
    st.subheader("Chunk Addition Demonstration")
    st.code("""
from langchain_core.messages import AIMessageChunk

chunk1 = AIMessageChunk(content="Lang")
chunk2 = AIMessageChunk(content="Chain")

# The addition operator merges chunks
full_message = chunk1 + chunk2
print(full_message.content)  # "LangChain"
print(type(full_message))    # AIMessageChunk
    """, language="python")

# --- SECTION 5: AGENT STREAMING ---
st.header("5. Agent Streaming & Intermediate Thought Propagation", divider="rainbow")
st.markdown("""
Complex agents do not merely generate text; they formulate **thoughts**, call **tools**, evaluate **observations**, 
and synthesize **answers**. Streaming an agent requires surfacing these intermediate reasoning steps in real-time.
""")

col_ag1, col_ag2 = st.columns(2)
with col_ag1:
    st.subheader("Agent Streaming Modes in LangGraph")
    st.markdown("""
    LangGraph provides distinct streaming modes via `graph.stream(input, stream_mode=...)`:
    * `stream_mode="values"`: Emits the entire graph state after each node runs.
    * `stream_mode="updates"`: Emits only the state delta (updates) produced by the currently executing node.
    * `stream_mode="messages"`: Emits LLM message chunks as they are generated by internal model calls.
    * `stream_mode="custom"`: Emits custom telemetry emitted explicitly via `StreamWriter`.
    """)
with col_ag2:
    st.subheader("LangGraph Agent Stream Example")
    st.code("""
# Stream node updates in real-time:
config = {"configurable": {"thread_id": "T-101"}}

for event in agent_graph.stream(
    {"messages": [("user", "What is the weather in Tokyo?")]}, 
    config, 
    stream_mode="updates"
):
    for node_name, state_update in event.items():
        print(f"Node Executed: {node_name}")
        print(f"Update: {state_update}")
    """, language="python")

# In-section diagram: Agent Streaming Breakdown
with st.expander("🔍 View Agent Trajectory Streaming Stages", expanded=False):
    agent_dot = graphviz.Digraph(comment="Agent Trajectory")
    agent_dot.attr(bgcolor="#0B1120", rankdir="TB")
    agent_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    agent_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    agent_dot.node("user", "User Message Received", fillcolor="#0F172A")
    agent_dot.node("thought_stream", "Thought Stream Emitted\n[LLM Reasoning Delta Chunk]", fillcolor="#1E3A8A")
    agent_dot.node("tool_dispatch", "Tool Invocation Stream\n[Tool Name + Parsed Args]", fillcolor="#701A75")
    agent_dot.node("obs_stream", "Tool Observation Stream\n[Raw JSON Return Delta]", fillcolor="#0E7490")
    agent_dot.node("final_stream", "Final Response Token Stream\n[Synthesis to User]", fillcolor="#047857")

    agent_dot.edge("user", "thought_stream")
    agent_dot.edge("thought_stream", "tool_dispatch")
    agent_dot.edge("tool_dispatch", "obs_stream")
    agent_dot.edge("obs_stream", "final_stream")

    st.graphviz_chart(agent_dot, use_container_width=True)

# --- SECTION 6: TOOL STREAMING ---
st.header("6. Tool Streaming & External Payload Emitters", divider="rainbow")
st.markdown("""
When an agent invokes long-running tools (e.g. querying a 50GB database, performing vector similarity over 100,000 documents, 
or crawling external websites), waiting for the entire tool output blocks the stream. **Tool Streaming** enables tools to emit 
real-time progress updates or partial data records back into the stream.
""")

col_tl1, col_tl2 = st.columns(2)
with col_tl1:
    st.subheader("Custom Tool Streaming Mechanisms")
    st.markdown("""
    * **Async Generators inside Tools**: Tools define `async def func(): yield chunk`.
    * **StreamWriter in LangGraph**: Custom writers allow tools to push progress events without polluting the official state schema.
    * **Progress Indicators**: Informing users of long-running operations (e.g. *Found 40/100 documents...*) prevents premature browser disconnects.
    """)
with col_tl2:
    st.subheader("StreamWriter Tool Pattern")
    st.code("""
from langgraph.types import StreamWriter

def custom_search_tool(query: str, writer: StreamWriter) -> str:
    writer({"status": "Connecting to search index..."})
    # Intermediate progress
    writer({"status": "Retrieved 15 relevant articles. Ranking..."})
    return "Top Result: LangChain v0.3 Release Notes"
    """, language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Unified Multi-Mode Streaming Pipeline with Token Chunk Aggregation", divider="rainbow")
st.markdown("""
The following complete production pattern illustrates how to build a unified async streaming pipeline that consumes 
both token deltas and node state updates while accumulating a verified terminal message.
""")

st.code('''
import asyncio
from typing import AsyncIterator, Dict, Any
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

async def stream_with_aggregation(prompt_text: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Unified streaming handler that yields real-time token chunks
    while aggregating the complete message in memory.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI systems architect. Answer concisely."),
        ("human", "{question}")
    ])
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    chain = prompt | model

    accumulated_message: AIMessageChunk = AIMessageChunk(content="")
    token_count = 0

    # Execute async stream over LCEL chain
    async for chunk in chain.astream({"question": prompt_text}):
        token_count += 1
        # Mathematical addition merges chunk content & metadata
        accumulated_message += chunk
        
        # Yield delta to downstream consumer (e.g., FastAPI SSE endpoint)
        yield {
            "event": "token_delta",
            "delta": chunk.content,
            "token_index": token_count
        }

    # Yield terminal summary event
    yield {
        "event": "stream_complete",
        "full_text": accumulated_message.content,
        "total_tokens": token_count
    }
''', language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Modal Stream Emitter & Chunk Inspector", divider="rainbow")
st.markdown("""
Inspect how different streaming modalities emit data over time. Select a streaming scenario below and click 
**Simulate Live Stream** to observe real-time chunk emissions, latency timings, and payload aggregation.
""")

# Instantiate Module Scope Simulator
if "stream_sim" not in st.session_state:
    st.session_state.stream_sim = MockStreamingBasicsSimulator()

sim_stream = st.session_state.stream_sim

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.subheader("1. Stream Configuration")
    scenario_choice = st.selectbox(
        "Choose Streaming Modality:",
        options=list(sim_stream.scenarios.keys()),
        format_func=lambda x: sim_stream.scenarios[x]["name"]
    )
    chunks_to_emit = sim_stream.emit_stream(scenario_choice)
    st.markdown(f"**Total Chunks in Stream**: `{len(chunks_to_emit)}`")

    if st.button("🚀 Simulate Live Stream Emission", use_container_width=True):
        st.session_state.run_sim = True

with col_sb2:
    st.subheader("2. Live Stream Receiver & Chunk Inspector")
    if st.session_state.get("run_sim"):
        accumulated_text = ""
        total_time = 0
        chunk_logs = []

        status_placeholder = st.empty()
        text_placeholder = st.empty()

        for idx, ch in enumerate(chunks_to_emit):
            accumulated_text += ch["content"]
            total_time += ch["latency_ms"]
            chunk_logs.append({
                "Chunk #": idx + 1,
                "Type": ch["type"],
                "Content": ch["content"],
                "Latency (ms)": ch["latency_ms"],
                "Cumulative Time (ms)": total_time
            })
            
            status_placeholder.markdown(f"**Emitting Chunk {idx+1}/{len(chunks_to_emit)}** | TTFT: `{chunks_to_emit[0]['latency_ms']} ms` | Elapsed: `{total_time} ms`")
            text_placeholder.info(accumulated_text)

        st.success(f"Stream Complete! Emitted {len(chunks_to_emit)} chunks in {total_time} ms.")
        st.dataframe(pd.DataFrame(chunk_logs), use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: What is Time-to-First-Token (TTFT), and why is it more critical than overall generation speed?"):
    st.markdown("""
    TTFT measures the time elapsed from when a user submits a query to when the first token renders in the UI. 
    Human cognition perceives delays under 250 milliseconds as near-instantaneous. While overall generation time depends 
    on response length, a fast TTFT engages the user immediately, eliminating perceived latency and drop-off.
    """)

with st.expander("Q2: How does the addition operator (`+`) on `AIMessageChunk` work under the hood?"):
    st.markdown("""
    In LangChain, `AIMessageChunk.__add__` implements mathematical monoid addition. It concatenates the `content` string, 
    merges dictionary fields in `response_metadata`, combines fragments of structured `tool_calls` by matching indices, 
    and returns a new aggregated `AIMessageChunk`.
    """)

with st.expander("Q3: What are the differences between `stream_mode='values'` and `stream_mode='updates'` in LangGraph?"):
    st.markdown("""
    * **`stream_mode='values'`**: Emits the complete accumulated state dictionary after each graph step. 
      Useful when the UI needs to render the entire conversation history without local state management.
    * **`stream_mode='updates'`**: Emits only the partial dictionary returned by the specific node that just executed. 
      More bandwidth-efficient and ideal for triggering node-specific animations or status indicators.
    """)

with st.expander("Q4: Why does streaming structured output (JSON) require specialized parser handling?"):
    st.markdown("""
    Standard JSON parsers (e.g., `json.loads`) fail when evaluating partial strings (e.g. `{"name": "Al`). 
    Streaming structured output requires an incremental or partial JSON parser that can yield valid partial objects 
    or wait until matching key-value pairs close before emitting to the UI.
    """)

with st.expander("Q5: How does HTTP chunked transfer encoding differ from Server-Sent Events (SSE) for LLM streaming?"):
    st.markdown("""
    * **HTTP Chunked Encoding**: Raw byte transmission without standardized message framing. 
      Clients must manually buffer and parse delimiters.
    * **Server-Sent Events (SSE)**: Standardized text/event-stream format with explicit event typing (`event: token\\ndata: {...}\\n\\n`) 
      and automatic browser reconnection handling via `EventSource`.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_strp1, col_strp2 = st.columns(2)
with col_strp1:
    st.error("""
    **Production Trap 1 (Fatal Error): Synchronous Network Blocking in Streaming Loops**  
    Running CPU-heavy synchronous parsing or un-awaited network calls inside `for chunk in stream:` blocks the Python event loop, 
    destroying the high-frequency emission rate and causing stuttering token delivery on the frontend.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Tool Call Fragment Corruption**  
    Attempting to parse tool arguments from a single intermediate chunk instead of aggregating chunks. 
    Tool call arguments arrive fragmented across dozens of chunks; parsing must only occur after the tool call closes.
    """)
with col_strp2:
    st.error("""
    **Production Trap 3 (Fatal Error): Proxy Buffering Destroying Stream Delivery**  
    Deploying streaming endpoints behind reverse proxies (e.g. Nginx, Cloudflare, AWS ALB) without disabling proxy buffering 
    (`X-Accel-Buffering: no` or `proxy_buffering off;`). The proxy buffers the entire response and sends it as one giant block.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Premature Stream Aborts on Browser Tab Close**  
    Failing to handle client disconnects cleanly. If a user closes their browser tab, unhandled server-side generator exceptions 
    can leave database connections or GPU worker threads locked. Always wrap streams in `try ... finally` blocks.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    st.markdown("""
    <div class="stream-card">
        <h4>Flashcard 1: TTFT (Time-to-First-Token)</h4>
        <p><strong>Metric:</strong> Initial Latency Milestone</p>
        <p><strong>Definition:</strong> The elapsed time from request dispatch to receipt of the initial token chunk, defining user-perceived responsiveness.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="stream-card">
        <h4>Flashcard 2: AIMessageChunk</h4>
        <p><strong>Class:</strong> <code>langchain_core.messages.AIMessageChunk</code></p>
        <p><strong>Definition:</strong> The data container for streamed message deltas, supporting native mathematical concatenation via the <code>+</code> operator.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc2:
    st.markdown("""
    <div class="stream-card">
        <h4>Flashcard 3: StreamWriter</h4>
        <p><strong>Primitive:</strong> LangGraph Custom Emitter</p>
        <p><strong>Definition:</strong> A dedicated callable passed to graph nodes allowing custom progress events and partial payloads to stream without modifying graph state.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="stream-card">
        <h4>Flashcard 4: Server-Sent Events (SSE)</h4>
        <p><strong>Protocol:</strong> <code>text/event-stream</code></p>
        <p><strong>Definition:</strong> The standard HTTP unidirectional protocol used to push real-time LLM token deltas and agent events to client web applications.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of streaming basics and chunk aggregation patterns:")

chk_s1 = st.checkbox("1. I understand the difference between blocking `invoke()` and streaming `stream()` / `astream()`.")
chk_s2 = st.checkbox("2. I know why Time-to-First-Token (TTFT) is the primary driver of perceived user experience.")
chk_s3 = st.checkbox("3. I understand how `AIMessageChunk` objects are concatenated using the `+` operator.")
chk_s4 = st.checkbox("4. I know how to configure `stream_mode` in LangGraph to emit values, updates, or messages.")
chk_s5 = st.checkbox("5. I understand how long-running tools can emit intermediate progress via `StreamWriter`.")
chk_s6 = st.checkbox("6. I understand how reverse proxy buffering (e.g. Nginx) can accidentally break streaming in production.")

if chk_s1 and chk_s2 and chk_s3 and chk_s4 and chk_s5 and chk_s6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Streaming Basics in LangChain & LangGraph.")
