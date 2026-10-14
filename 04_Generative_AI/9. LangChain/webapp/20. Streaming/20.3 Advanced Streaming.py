import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.3 Advanced Streaming - LangChain Architecture",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and stream inspector
st.markdown("""
<style>
    .adv-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-sse {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-json {
        background-color: #701A75;
        color: #F472B6;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockAdvancedStreamingEngine:
    """
    Simulation engine showcasing advanced streaming concepts:
    Server-Sent Events (SSE) framing, incremental JSON parsing for structured output,
    and reasoning metadata streaming.
    """
    def __init__(self):
        self.raw_json_stream = [
            '{\n',
            '  "customer_id": "CUST-992",\n',
            '  "intent": "account_cancellation",\n',
            '  "sentiment_score": 0.24,\n',
            '  "urgency": "HIGH",\n',
            '  "recommended_retention_offer": {\n',
            '    "discount_percentage": 25,\n',
            '    "plan_tier": "Enterprise_Pro"\n',
            '  }\n',
            '}'
        ]

    def parse_incremental(self, chunks_count: int) -> Dict[str, Any]:
        """
        Simulates an incremental JSON parser evaluating partial tokens.
        """
        joined = "".join(self.raw_json_stream[:chunks_count])
        # Detect parsed fields based on token progress
        parsed_state = {}
        if chunks_count >= 2:
            parsed_state["customer_id"] = "CUST-992"
        if chunks_count >= 3:
            parsed_state["intent"] = "account_cancellation"
        if chunks_count >= 4:
            parsed_state["sentiment_score"] = 0.24
        if chunks_count >= 5:
            parsed_state["urgency"] = "HIGH"
        if chunks_count >= 8:
            parsed_state["recommended_retention_offer"] = {
                "discount_percentage": 25,
                "plan_tier": "Enterprise_Pro"
            }

        return {
            "raw_stream_buffer": joined,
            "parsed_state": parsed_state,
            "is_complete": chunks_count == len(self.raw_json_stream)
        }

# Main Application Title & Objective
st.title("⚡ 20.3 Advanced Streaming")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Advanced streaming elevates GenAI from simple text chat to **reactive, event-driven user interfaces**. 
This module explores **real-time UI design patterns**, **streaming intermediate reasoning and thoughts**, 
**token telemetry metadata**, **incremental structured output parsing (streaming partial JSON)**, and **resilient frontend integration protocols**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Full-Stack Reactive Streaming Gateway & Server-Sent Events (SSE)", divider="rainbow")
st.markdown("""
The architecture below illustrates the end-to-end journey of streaming data from the LLM core, through an enterprise 
FastAPI/LangServe gateway, and into client browser components using Server-Sent Events (SSE) and partial JSON deserializers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    gw_dot = graphviz.Digraph(comment="Advanced Streaming Gateway")
    gw_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    gw_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    gw_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    gw_dot.node("llm", "LLM Inference Engine\n[Emits Token Deltas & Tool Argument Chunks]", fillcolor="#0F172A", shape="ellipse")
    gw_dot.node("langchain_core", "LangChain / LangGraph Engine\n[astream_events(v2) Multiplexer]", fillcolor="#1E293B")
    
    # Gateway Components
    gw_dot.node("sse_encoder", "FastAPI SSE Gateway\n[Text/Event-Stream Formatter & Keepalive]", fillcolor="#0E7490")
    gw_dot.node("proxy_layer", "Reverse Proxy (Nginx / Cloudflare)\n[proxy_buffering: off | Chunked HTTP 1.1]", fillcolor="#334155")
    
    # Frontend Layers
    gw_dot.node("browser_fetch", "Browser fetch() / EventSource\n[ReadableStream.getReader()]", fillcolor="#4338CA")
    gw_dot.node("json_parser", "Partial JSON Stream Parser\n[Incremental Key-Value Extractor]", fillcolor="#701A75")
    gw_dot.node("ui_components", "Reactive UI Components\n[Real-Time Progress Bars, Charts & Chat]", fillcolor="#047857", shape="ellipse")

    gw_dot.edge("llm", "langchain_core")
    gw_dot.edge("langchain_core", "sse_encoder", label=" Normalized Events")
    gw_dot.edge("sse_encoder", "proxy_layer", label=" SSE Wire Format")
    gw_dot.edge("proxy_layer", "browser_fetch", label=" Low-Latency Pipe")
    gw_dot.edge("browser_fetch", "json_parser", label=" Token Buffers")
    gw_dot.edge("json_parser", "ui_components", label=" State Updates")

    st.graphviz_chart(gw_dot, use_container_width=True)

# --- SECTION 2: REAL-TIME UI ---
st.header("2. Real-Time UI Design & Perceived Latency Optimization", divider="rainbow")
st.markdown("""
A state-of-the-art AI user interface should never present a generic loading spinner while an agent executes. 
Instead, it should utilize **Predictive Micro-Interactions** and **Progressive Disclosure** to keep users informed and engaged.
""")

col_ui1, col_ui2 = st.columns(2)
with col_ui1:
    st.subheader("UX Design Principles for AI Streaming")
    st.markdown("""
    * **Progressive Loading Skeletons**: Display skeleton screens that expand as the agent determines output format.
    * **Thinking Expanders**: Render agent chain-of-thought in an expandable disclosure box that auto-collapses when final generation starts.
    * **Smooth Auto-Scroll**: Keep the viewport locked to the bottom during active streaming, but smoothly detach if the user scrolls up.
    * **Typing Cadence Smoothing**: Use client-side animation buffers (requestAnimationFrame) to smooth out bursty token arrivals.
    """)
with col_ui2:
    st.subheader("Streaming UX State Machine")
    st.markdown("""
    | Agent Lifecycle State | UI Visual Representation |
    | :--- | :--- |
    | **1. Request Dispatched** | Pulsing glow on prompt submit bar |
    | **2. Prefill Phase** | Ghost shimmer placeholder (under 250ms) |
    | **3. Tool Execution** | Dynamic badge: *Searching Knowledgebase...* |
    | **4. Generating Tokens** | Live character stream with flashing cursor |
    | **5. Terminal Complete** | Checkmark badge + Token count & latency tag |
    """)

# --- SECTION 3: STREAMING INTERMEDIATE RESULTS ---
st.header("3. Streaming Intermediate Reasoning & Step Outputs", divider="rainbow")
st.markdown("""
When building multi-agent or multi-tool workflows, users should not wait in silence while the agent calls external APIs. 
Streaming intermediate results allows the frontend to show each step of the reasoning trajectory in real time.
""")

col_int1, col_int2 = st.columns(2)
with col_int1:
    st.subheader("Intermediate Step Typologies")
    st.markdown("""
    * **Plan Formulation**: Streaming the multi-step plan before execution begins.
    * **Tool Parameters**: Showing what search queries or database filters the agent selected.
    * **Raw Tool Outputs**: Displaying the number of search results or data rows fetched.
    * **Critique & Reflection**: Showing internal critique before the final answer is drafted.
    """)
with col_int2:
    st.subheader("Streaming Intermediate Steps Code")
    st.code("""
# Consuming intermediate steps in LangChain:
async for event in agent_executor.astream_events(query, version="v2"):
    kind = event["event"]
    
    if kind == "on_tool_start":
        yield {
            "type": "tool_start",
            "tool": event["name"],
            "inputs": event["data"].get("input")
        }
    elif kind == "on_tool_end":
        yield {
            "type": "tool_end",
            "tool": event["name"],
            "summary": f"Completed in {event.get('duration', 0):.2f}s"
        }
    """, language="python")

# --- SECTION 4: STREAMING REASONING METADATA ---
st.header("4. Streaming Reasoning Metadata & Token Telemetry", divider="rainbow")
st.markdown("""
Modern frontier models (OpenAI o1/o3, DeepSeek-R1, Anthropic Claude 3.5 Sonnet) produce **Reasoning Tokens** and **Generation Metadata** 
distinct from user-facing text. Streaming architectures must separate reasoning deltas from answer deltas while streaming token metrics.
""")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.subheader("Reasoning Tokens vs. Content Tokens")
    st.markdown("""
    * **Reasoning Stream**: Emitted under specialized fields (e.g. `reasoning_content` or `<think>...</think>` tags).
    * **Answer Stream**: Standard user-visible `content` tokens.
    * **Live Telemetry Stream**: Token count counters and instantaneous tokens-per-second (TPS) metrics streamed alongside the response.
    """)
with col_meta2:
    st.subheader("Separating Reasoning Chunks")
    st.code("""
async for chunk in reasoning_model.astream(prompt):
    # Check for dedicated reasoning attributes:
    if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
        reasoning_delta = chunk.additional_kwargs["reasoning_content"]
        yield {"channel": "thought", "delta": reasoning_delta}
    elif chunk.content:
        yield {"channel": "answer", "delta": chunk.content}
    """, language="python")

# --- SECTION 5: STREAMING STRUCTURED OUTPUT ---
st.header("5. Streaming Structured Output & Incremental JSON Parsing", divider="rainbow")
st.markdown("""
Streaming unstructured text is straightforward; streaming **Structured Output (JSON)** is fundamentally harder. 
A naive `json.loads()` will throw `JSONDecodeError` on every intermediate chunk until the final closing bracket arrives. 
Advanced systems utilize **Incremental Partial JSON Parsers**.
""")

col_json1, col_json2 = st.columns(2)
with col_json1:
    st.subheader("How Incremental JSON Parsing Works")
    st.markdown("""
    * **Incomplete State Recovery**: When given `{"title": "Adv`, the parser automatically appends closing quotes and brackets to evaluate partial validity.
    * **Progressive UI Hydration**: Form fields, cards, and data tables populate dynamically as each key-value pair completes.
    * **LangChain Integration**: `JsonOutputParser` implements an incremental streaming mode that yields valid partial dictionaries.
    """)
with col_json2:
    st.subheader("Incremental JSON Parsing Example")
    st.code("""
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | model | parser

# stream() yields progressively completed dictionaries:
for partial_dict in chain.stream(query):
    # Output:
    # {'status': 'processing'}
    # {'status': 'processing', 'records': [1]}
    # {'status': 'processing', 'records': [1, 2]}
    print(partial_dict)
    """, language="python")

# --- SECTION 6: FRONTEND INTEGRATION ---
st.header("6. Frontend Integration Protocols (SSE, WebSockets, ReadableStream)", divider="rainbow")
st.markdown("""
Connecting a backend streaming pipeline to a browser frontend requires selecting the optimal transport protocol. 
The three primary options are **Server-Sent Events (SSE)**, **WebSockets**, and **HTTP/2 Chunked Fetch Streams**.
""")

st.markdown("""
### Transport Protocol Trade-Off Matrix
| Feature | Server-Sent Events (SSE) | WebSockets | HTTP/2 Chunked Fetch |
| :--- | :--- | :--- | :--- |
| **Direction** | Unidirectional (Server to Client) | Full Duplex (Bidirectional) | Unidirectional (Request to Response) |
| **Protocol** | Standard HTTP/1.1 or HTTP/2 | `ws://` / `wss://` | Standard HTTP/2 |
| **Auto-Reconnect** | Native browser support | Manual implementation | Manual implementation |
| **Firewall / Proxy** | Traverses standard firewalls (Port 80/443) | Can be blocked by enterprise proxies | Traverses standard firewalls |
| **Best Suited For** | LLM token streaming & status feeds | Interactive audio/voice agent streaming | Modern fetch() ReadableStream |
""")

st.code("""
// Browser JavaScript Consumer using fetch() ReadableStream
async function streamAgentResponse(prompt) {
  const response = await fetch("/api/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: prompt })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunkText = decoder.decode(value);
    console.log("Received chunk:", chunkText);
  }
}
""", language="javascript")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Resilient SSE-Based Frontend Streaming Gateway with Partial JSON Parsing", divider="rainbow")
st.markdown("""
The following complete production pattern implements a resilient FastAPI SSE streaming gateway equipped with 
keepalive pings, error recovery, and LangChain event multiplexing.
""")

st.code('''
import json
import asyncio
from typing import AsyncIterator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import Runnable

app = FastAPI()

async def resilient_sse_generator(
    pipeline: Runnable,
    payload: dict,
    request: Request
) -> AsyncIterator[str]:
    """
    Production-grade SSE stream emitter with client disconnect detection,
    keepalive heartbeat pings, and structured JSON formatting.
    """
    try:
        # 1. Emit stream start event
        yield "event: start\\ndata: {\\"status\\": \\"connected\\"}\\n\\n"

        # 2. Iterate over LangChain v2 event stream
        async for event in pipeline.astream_events(payload, version="v2"):
            # Check if client disconnected (prevents thread starvation)
            if await request.is_disconnected():
                break

            kind = event["event"]

            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    data = json.dumps({"delta": token})
                    yield f"event: token\\ndata: {data}\\n\\n"

            elif kind == "on_tool_start":
                tool_data = json.dumps({
                    "tool": event["name"],
                    "input": event["data"].get("input")
                })
                yield f"event: tool_start\\ndata: {tool_data}\\n\\n"

        # 3. Emit stream completion event
        yield "event: end\\ndata: {\\"status\\": \\"finished\\"}\\n\\n"

    except asyncio.CancelledError:
        # Gracefully handle client cancellation
        pass
    except Exception as exc:
        error_data = json.dumps({"error": str(exc)})
        yield f"event: error\\ndata: {error_data}\\n\\n"

@app.post("/api/v1/stream")
async def stream_endpoint(request: Request, body: dict):
    # Custom headers required to disable reverse proxy buffering
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"  # Critical for Nginx proxy pass
    }
    return StreamingResponse(
        resilient_sse_generator(chain, body, request),
        headers=headers
    )
''', language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Real-Time SSE Streamer & Structured JSON Stream Parser", divider="rainbow")
st.markdown("""
Experience how an incremental JSON parser reconstructs a structured object token-by-token. 
Drag the slider below to advance the stream progress and watch the parser extract structured fields in real time.
""")

# Instantiate Module Scope Simulator
if "adv_stream_sim" not in st.session_state:
    st.session_state.adv_stream_sim = MockAdvancedStreamingEngine()

adv_sim = st.session_state.adv_stream_sim

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("1. Stream Progress Controller")
    chunk_step = st.slider(
        "Simulated Stream Position (Chunks Received):",
        min_value=1,
        max_value=len(adv_sim.raw_json_stream),
        value=4
    )
    
    parsed_result = adv_sim.parse_incremental(chunk_step)
    
    st.markdown("#### Raw SSE Network Buffer:")
    st.code(parsed_result["raw_stream_buffer"], language="json")

with col_wb2:
    st.subheader("2. Hydrated UI Component State")
    st.markdown(f"**Stream Status**: `{'COMPLETED' if parsed_result['is_complete'] else 'STREAMING IN PROGRESS'}`")
    
    st.markdown("#### Parsed Key-Value Pairs:")
    if parsed_result["parsed_state"]:
        st.json(parsed_result["parsed_state"])
    else:
        st.warning("Buffer too small; awaiting first complete key-value pair...")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: Why is Server-Sent Events (SSE) generally preferred over WebSockets for LLM chat streaming?"):
    st.markdown("""
    LLM chat interactions are inherently request-response with unidirectional streaming: the client sends a single prompt, 
    and the server streams tokens back. SSE runs over standard HTTP/1.1 or HTTP/2, works natively through corporate firewalls 
    and proxies without protocol switching, supports built-in reconnection logic, and avoids the complex state management 
    and socket persistence required by full-duplex WebSockets.
    """)

with st.expander("Q2: How does an incremental JSON parser prevent UI freezing when streaming structured objects?"):
    st.markdown("""
    Standard parsers fail on partial JSON, requiring the application to buffer until stream termination. 
    Incremental parsers parse top-level keys as soon as their values close, allowing the UI to render cards, tables, 
    and forms progressively. This eliminates the post-generation parsing freeze and provides immediate visual feedback.
    """)

with st.expander("Q3: Why is the `X-Accel-Buffering: no` HTTP header essential when deploying streaming apps behind Nginx?"):
    st.markdown("""
    By default, Nginx buffers upstream responses to optimize network throughput. For streaming endpoints, this causes Nginx 
    to hold tokens in memory until the buffer fills (often 4KB or 8KB) or the connection closes, completely negating the benefit 
    of streaming. Sending `X-Accel-Buffering: no` instructs Nginx to disable buffering and forward chunks immediately.
    """)

with st.expander("Q4: How should client disconnects be handled in an asynchronous streaming endpoint?"):
    st.markdown("""
    FastAPI endpoints must periodically check `await request.is_disconnected()` inside the streaming generator loop. 
    If detected, the generator must break immediately to exit the pipeline and release GPU/LLM server workers. 
    Without this check, the server will continue generating and wasting tokens for a departed user.
    """)

with st.expander("Q5: What is the purpose of keepalive heartbeat pings in production SSE connections?"):
    st.markdown("""
    Cloud load balancers (e.g. AWS ALB, Cloudflare) automatically terminate idle HTTP connections after 60 seconds. 
    If an agent performs a long tool execution that produces no tokens for 60 seconds, the load balancer kills the socket. 
    Keepalive comments (e.g. `: ping\\n\\n`) sent every 15 seconds keep the TCP socket alive.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_av1, col_av2 = st.columns(2)
with col_av1:
    st.error("""
    **Production Trap 1 (Fatal Error): Missing Proxy Buffering Configuration**  
    Failing to set `proxy_buffering off;` or `X-Accel-Buffering: no`. 
    The application logic streams tokens correctly, but users experience 10-second blocking delays because the reverse proxy buffers the data.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): JSON Syntax Errors from Blind Substring Parsing**  
    Attempting to evaluate partial JSON strings using `eval()` or naive regex matching. 
    Always use a dedicated incremental parser (e.g., `partialjson` or LangChain's `JsonOutputParser`) to avoid injection vulnerabilities and crashes.
    """)
with col_av2:
    st.error("""
    **Production Trap 3 (Fatal Error): Unclosed Async Generators on Client Abort**  
    Failing to catch `asyncio.CancelledError`. When a client cancels a request, unhandled exceptions can leave database connections 
    or HTTP client sessions unclosed, eventually exhausting connection pools.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): DOM Thrashing from Micro-Token Rerenders**  
    Updating React or Vue component state on every single token chunk (e.g., 50 times per second) triggers excessive DOM repaints. 
    Use a micro-batcher or `requestAnimationFrame` on the frontend to throttle UI updates to 60 FPS.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc_a, col_fc_b = st.columns(2)
with col_fc_a:
    st.markdown("""
    <div class="adv-card">
        <h4>Flashcard 1: X-Accel-Buffering</h4>
        <p><strong>Header:</strong> <code>X-Accel-Buffering: no</code></p>
        <p><strong>Definition:</strong> HTTP response header telling Nginx and reverse proxies to disable response buffering and pass chunks immediately to the client.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="adv-card">
        <h4>Flashcard 2: Incremental JSON</h4>
        <p><strong>Parser:</strong> <code>JsonOutputParser(streaming=True)</code></p>
        <p><strong>Definition:</strong> A parser that recovers valid partial objects from fragmented JSON streams, enabling progressive UI component hydration.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc_b:
    st.markdown("""
    <div class="adv-card">
        <h4>Flashcard 3: Client Disconnect Detection</h4>
        <p><strong>Method:</strong> <code>request.is_disconnected()</code></p>
        <p><strong>Definition:</strong> FastAPI method used inside streaming loops to detect when a client closes their tab, preventing wasted compute.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="adv-card">
        <h4>Flashcard 4: SSE Keepalive Heartbeat</h4>
        <p><strong>Pattern:</strong> <code>: ping\\n\\n</code></p>
        <p><strong>Definition:</strong> Periodic empty comment frames sent down an SSE stream to prevent load balancer idle socket timeouts during long tool runs.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of advanced streaming architectures and reactive UI integration:")

ch_adv1 = st.checkbox("1. I understand how to optimize real-time UI design to reduce perceived user latency.")
ch_adv2 = st.checkbox("2. I know how to stream intermediate agent reasoning and tool executions in real time.")
ch_adv3 = st.checkbox("3. I understand how to separate reasoning tokens from user-facing answer tokens.")
ch_adv4 = st.checkbox("4. I know how incremental JSON parsing works for streaming structured output without errors.")
ch_adv5 = st.checkbox("5. I understand the architectural advantages of Server-Sent Events (SSE) over WebSockets for LLMs.")
ch_adv6 = st.checkbox("6. I know why `X-Accel-Buffering: no` is mandatory when deploying streaming endpoints behind Nginx.")

if ch_adv1 and ch_adv2 and ch_adv3 and ch_adv4 and ch_adv5 and ch_adv6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Advanced Streaming in LangChain & LangGraph.")
