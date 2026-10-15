import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="3.3 Graph Execution - LangGraph Curriculum",
    page_icon="⚡",
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
    .badge-stream {
        background-color: #1E293B;
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE EXECUTION RUNNER CLASS FOR WORKBENCH & PIPELINE
# ==============================================================================
class ExecutionRunnerSimulator:
    """
    Simulates LangGraph execution protocols:
    invoke (blocking), stream values, stream updates, and token events.
    """
    def __init__(self):
        self.pipeline_steps = [
            ("triage_node", {"intent": "TECHNICAL_SUPPORT", "priority": "HIGH"}),
            ("retrieve_docs", {"docs_found": ["DOC-102", "DOC-881"], "retrieval_score": 0.92}),
            ("synthesize_reply", {"reply": "Identified resolution in DOC-102: Restart service daemon.", "tokens": 42})
        ]

    def execute_mode(self, mode: str, query: str) -> Dict[str, Any]:
        accumulated_state = {"query": query, "step": 0}
        events = []

        if mode == "invoke() (Blocking Full Return)":
            for name, delta in self.pipeline_steps:
                accumulated_state.update(delta)
                accumulated_state["step"] += 1
            return {
                "mode": mode,
                "type": "Single Final State Dictionary",
                "output": accumulated_state,
                "events_count": 1,
                "stream_events": [{"event": "final_state", "data": accumulated_state}]
            }

        elif mode == "stream(mode='values')":
            # Emits full state snapshot after each superstep
            for name, delta in self.pipeline_steps:
                accumulated_state.update(delta)
                accumulated_state["step"] += 1
                events.append({"event": f"superstep_{accumulated_state['step']}_values", "snapshot": dict(accumulated_state)})
            return {
                "mode": mode,
                "type": "Stream of Full State Dictionaries per Superstep",
                "output": accumulated_state,
                "events_count": len(events),
                "stream_events": events
            }

        elif mode == "stream(mode='updates')":
            # Emits only the delta dictionary from completed node
            for name, delta in self.pipeline_steps:
                accumulated_state["step"] += 1
                events.append({"event": f"node_{name}_update", "delta": delta})
                accumulated_state.update(delta)
            return {
                "mode": mode,
                "type": "Stream of Delta Dictionaries only",
                "output": accumulated_state,
                "events_count": len(events),
                "stream_events": events
            }

        else: # astream_events (tokens)
            tokens = ["Identified ", "resolution ", "in ", "DOC-102: ", "Restart ", "service ", "daemon."]
            for t in tokens:
                events.append({"event": "on_chat_model_stream", "token": t})
            accumulated_state.update(self.pipeline_steps[-1][1])
            return {
                "mode": mode,
                "type": "Fine-Grained Token Streaming",
                "output": accumulated_state,
                "events_count": len(events),
                "stream_events": events
            }


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("3.3 Graph Execution: Compilation, Invocation, Streaming, & Async Protocols")
st.caption("Chapter 3: LangGraph Core Concepts | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
Once a state graph's nodes and edges are wired together, the graph must be compiled into an executable runtime engine. 
LangGraph provides a rich suite of execution protocols designed for both simple command-line scripts and high-throughput, 
asynchronous enterprise web services.

In this lesson, you will master:
* **`compile()`**: Locking the graph architecture, attaching checkpointers, and initializing the Pregel runner.
* **`invoke()` vs `ainvoke()`**: Synchronous blocking execution versus asynchronous non-blocking web gateways.
* **`stream()` and `astream()`**: Streaming full state values versus node delta updates.
* **Fine-Grained Token Streaming**: Consuming `on_chat_model_stream` events via `astream_events` (v2).
* Building enterprise FastAPI and WebSocket streaming servers with zero blocking threads.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Mode Graph Execution Engine & Streaming Protocols", divider="rainbow")

st.markdown("""
The architecture below outlines the execution pathways available in LangGraph: from compilation, to blocking invocations, 
to superstep-level streaming and fine-grained token-level Server-Sent Events (SSE).
""")

blueprint_dot = """
digraph ExecutionProtocols {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_compile {
        label = "1. Compilation Gate";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        StateGraphObj [label="StateGraph Definition\\n(Nodes + Edges + Schema)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        CompiledApp [label="CompiledStateGraph (app)\\n(Pregel Runnable Instance)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_protocols {
        label = "2. Invocation & Execution Modes";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        SyncInvoke [label="app.invoke(input)\\n(Synchronous blocking; returns final state)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        AsyncInvoke [label="await app.ainvoke(input)\\n(Asynchronous non-blocking; returns final state)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        StreamValues [label="app.stream(mode='values')\\n(Emits full state snapshot after each superstep)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        StreamUpdates [label="app.stream(mode='updates')\\n(Emits only node delta updates)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        StreamEvents [label="app.astream_events(v2)\\n(Emits raw LLM tokens & tool events)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    subgraph cluster_egress {
        label = "3. Client Consumer Channels";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#8B5CF6"; fontsize = 11;
        BatchConsumer [label="Batch Script / CLI Process\\n(Waits for complete execution)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
        SSEClient [label="Frontend Web Chat / SSE Stream\\n(Tokens stream with TTFT < 350ms)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
    }

    StateGraphObj -> CompiledApp [label="builder.compile()"];
    CompiledApp -> SyncInvoke;
    CompiledApp -> AsyncInvoke;
    CompiledApp -> StreamValues;
    CompiledApp -> StreamUpdates;
    CompiledApp -> StreamEvents;

    SyncInvoke -> BatchConsumer [label="Final Dict"];
    AsyncInvoke -> BatchConsumer [label="Final Dict"];
    StreamEvents -> SSEClient [label="Real-Time Token Stream", color="#EC4899", penwidth=1.8];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - COMPILE()
# ==============================================================================
st.header("2. compile(): Locking Graph Topology, State Channels, and Checkpointers", divider="rainbow")

st.markdown("""
The **`builder.compile()`** method transforms a mutable graph builder into an immutable, executable **`CompiledStateGraph`**.
""")

col_cp1, col_cp2 = st.columns([1, 1])

with col_cp1:
    st.markdown("""
    ### Arguments Passed to `compile()`
    * **`checkpointer`**: Attaches persistent storage (e.g., `MemorySaver()`, `PostgresSaver(pool)`).
    * **`interrupt_before`**: List of node names where the engine should halt execution *before* running.
    * **`interrupt_after`**: List of node names where the engine should halt *after* execution.
    * **`store`**: Attaches cross-thread long-term memory store (`BaseStore`).
    """)

with col_cp2:
    st.markdown("""
    ### Code Example
    """)
    st.code("""
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review_node"]
)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - INVOKE()
# ==============================================================================
st.header("3. invoke(): Synchronous Blocking Execution and State Extraction", divider="rainbow")

st.markdown("""
The **`invoke()`** method is the primary synchronous execution interface:
`final_state = app.invoke(initial_state, config=config)`

* **Blocking Behavior**: Halts Python execution until all supersteps have concluded and control reaches `END` (or an interrupt).
* **Return Value**: Returns the final, reconciled state dictionary.
* **Best Used For**: Command-line tools, synchronous unit testing, and batch offline data transformation jobs.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - STREAM()
# ==============================================================================
st.header("4. stream(): Superstep Iteration Streaming, Values Mode, and Updates Mode", divider="rainbow")

st.markdown("""
The **`stream()`** method yields execution progress iteratively after each superstep completes:
""")

col_sm1, col_sm2 = st.columns([1, 1])

with col_sm1:
    st.markdown("""
    ### Mode 1: `mode="values"`
    Yields the **complete state snapshot** at every step:
    """)
    st.code("""
for state_snapshot in app.stream(input_data, mode="values"):
    print("Active Step Messages:", len(state_snapshot["messages"]))
    """, language="python")

with col_sm2:
    st.markdown("""
    ### Mode 2: `mode="updates"`
    Yields **only the delta update** produced by the specific node:
    """)
    st.code("""
for node_delta in app.stream(input_data, mode="updates"):
    for node_name, updates in node_delta.items():
        print(f"Node {node_name} emitted: {updates}")
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - AINVOKE()
# ==============================================================================
st.header("5. ainvoke(): Non-Blocking Asynchronous Graph Invocation for High-Concurrency Backends", divider="rainbow")

st.markdown("""
In modern production backends (FastAPI, Quart, Tornado), calling synchronous `invoke()` blocks the Python thread 
and starves the event loop. 

**`ainvoke()`** is the native asynchronous execution method:
""")

st.code("""
from fastapi import FastAPI
app_api = FastAPI()

@app_api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Non-blocking async graph execution!
    result = await app.ainvoke(
        {"query": request.message},
        config={"configurable": {"thread_id": request.session_id}}
    )
    return {"reply": result["reply"]}
""", language="python")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - ASTREAM()
# ==============================================================================
st.header("6. astream(): Real-Time Async Streaming and Token Generation Events (v2)", divider="rainbow")

st.markdown("""
For conversational web applications, users expect tokens to stream across the wire in real time. 
LangGraph's **`astream_events(version="v2")`** yields fine-grained internal events:
""")

col_ev1, col_ev2 = st.columns([1, 1])

with col_ev1:
    st.markdown("""
    ### Token Streaming Syntax
    """)
    st.code("""
async for event in app.astream_events(input_data, version="v2"):
    kind = event["event"]
    # Check for streaming tokens from Chat Models
    if kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        if token:
            yield f"data: {token}\\n\\n"
    """, language="python")

with col_ev2:
    st.markdown("""
    ### Other Event Types Emitted
    * **`on_chain_start` / `on_chain_end`**: Marks node entry and exit.
    * **`on_tool_start` / `on_tool_end`**: Emits tool name and inputs.
    * Allows the frontend to render "Searching database..." cards in real time!
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - SYNCHRONOUS EXECUTION
# ==============================================================================
st.header("7. Synchronous Execution: CLI Batch Pipelines, Simple Workflows, and Threading Bounds", divider="rainbow")

st.markdown("""
Synchronous execution (`invoke`) is ideal for:
1. **Offline Evaluation Scripts**: Running golden dataset regression tests in pytest.
2. **Local Prototyping**: Developing and testing node logic in standalone Python scripts.
3. **Data Ingestion Batches**: ETL workflows processing 10,000 PDF documents in sequential worker jobs.
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - ASYNCHRONOUS EXECUTION
# ==============================================================================
st.header("8. Asynchronous Execution: FastAPI ASGI Servers, Event Loops, and WebSockets", divider="rainbow")

st.markdown("""
In enterprise deployments, asynchronous execution (`ainvoke`, `astream`, `astream_events`) is **strictly mandatory**:
* A single FastAPI worker process can handle **thousands of concurrent connections** because network waits (calling OpenAI, querying vector databases) yield control back to the asyncio event loop.
* Eliminates thread starvation and drastically cuts cloud infrastructure hosting costs.
""")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Multi-Protocol Graph Execution Suite", divider="rainbow")

st.markdown("""
Below is a complete implementation demonstrating the execution lifecycle across synchronous, 
asynchronous, and streaming modes:
""")

prod_code = """
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State Schema
class TaskState(TypedDict):
    input_text: str
    processed_count: int
    summary: str

# 2. Nodes
def count_words(state: TaskState) -> dict:
    words = len(state["input_text"].split())
    return {"processed_count": words}

def summarize(state: TaskState) -> dict:
    return {"summary": f"Document has {state['processed_count']} words."}

# 3. Assemble & Compile
builder = StateGraph(TaskState)
builder.add_node("count", count_words)
builder.add_node("summarize", summarize)

builder.add_edge(START, "count")
builder.add_edge("count", "summarize")
builder.add_edge("summarize", END)

app = builder.compile()

# Mode A: Synchronous Blocking
print("--- [Mode A] Synchronous invoke() ---")
res_sync = app.invoke({"input_text": "LangGraph executes multi-agent states."})
print("Result:", res_sync)

# Mode B: Streaming Superstep Updates
print("--- [Mode B] Superstep stream(updates) ---")
for update in app.stream({"input_text": "Streaming execution pipeline."}, mode="updates"):
    print("Step update:", update)

# Mode C: Asynchronous Execution (AsyncIO)
async def run_async():
    print("--- [Mode C] Async ainvoke() ---")
    res_async = await app.ainvoke({"input_text": "Async execution in event loop."})
    print("Async Result:", res_async)

# Run async in event loop
# asyncio.run(run_async())
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Live Graph Invocation & Streaming Mode Inspector", divider="rainbow")

st.markdown("""
Compare LangGraph's execution protocols interactively. Select an execution mode and watch how the runtime emits 
single blocking payloads, superstep snapshots (`values`), node deltas (`updates`), or real-time token events.
""")

runner = ExecutionRunnerSimulator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Select Execution Mode & Ingress Payload")
    
    sel_mode = st.selectbox(
        "Execution Protocol",
        [
            "invoke() (Blocking Full Return)",
            "stream(mode='values')",
            "stream(mode='updates')",
            "astream_events(v2) (Token Delimited)"
        ],
        index=0
    )
    
    sample_query = st.text_input("User Prompt Query", value="How do I restart the production Kubernetes daemon?")

    exec_btn = st.button("Execute Protocol 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Emission Stream Inspector")
    
    if exec_btn:
        exec_data = runner.execute_mode(sel_mode, sample_query)
        
        m1, m2 = st.columns(2)
        m1.metric("Emitted Chunks", exec_data["events_count"])
        m2.metric("Protocol Tier", "Streaming" if "stream" in sel_mode else "Blocking")

        st.markdown(f"**Emission Type**: `<span class='badge-stream'>{exec_data['type']}</span>`", unsafe_allow_html=True)

        st.markdown("#### Stream Events Log")
        with st.container(height=260):
            for idx, ev in enumerate(exec_data["stream_events"]):
                st.markdown(f"`Event #{idx+1}`: `{ev['event']}`")
                st.json(ev)
                st.divider()

        st.success("Execution completed successfully!")
    else:
        st.info("Select a mode and click 'Execute Protocol' above.")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Questions covering compilation guarantees, streaming edge cases, and asynchronous performance:
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What is the primary difference between `stream(mode='values')` and `stream(mode='updates')`?"):
    st.markdown("""
    **Answer**:
    * **`mode='values'`**: Yields the entire state dictionary after every completed superstep. This provides full visibility into the cumulative state of the application.
    * **`mode='updates'`**: Yields only the specific delta dictionary returned by the node(s) that finished in that superstep (e.g. `{"node_name": {"new_key": "val"}}`). This is far more bandwidth-efficient for backend services that only care about incremental changes.
    """)

with st.expander("Q2: Why is `astream_events` (version 2) preferred over raw `stream()` for interactive chatbots?"):
    st.markdown("""
    **Answer**: Raw `stream()` only emits updates *after an entire node completes*. If an LLM node takes 12 seconds to generate a long 600-word response, `stream()` remains completely silent for 12 seconds before emitting the full message. 
    In contrast, **`astream_events(version="v2")`** listens to internal model callbacks and emits individual token chunks (`on_chat_model_stream`) the microsecond they are sampled on the GPU, lowering perceived Time-to-First-Token (TTFT) to under 350 milliseconds.
    """)

with st.expander("Q3: What happens if you call `invoke()` on a graph that was compiled with `interrupt_before`?"):
    st.markdown("""
    **Answer**: Execution proceeds through the preceding supersteps and halts immediately before entering the interrupted node. 
    The state is saved to the checkpointer database, and `invoke()` returns the state snapshot *up to that point*. 
    To resume execution across the breakpoint, you must call `invoke(None, config=config)` (or pass updated state values) on the same thread.
    """)

with st.expander("Q4: Can you pass runtime configuration overrides to `.invoke()`?"):
    st.markdown("""
    **Answer**: Yes! You can pass a `config` dictionary: `app.invoke(input, config={"configurable": {"thread_id": "...", "recursion_limit": 50}})`. 
    This allows you to dynamically tune recursion limits, inject user JWTs, or alter model temperature per invocation.
    """)

with st.expander("Q5: What is the risk of calling synchronous `invoke()` inside a FastAPI async route?"):
    st.markdown("""
    **Answer**: Calling a blocking synchronous function inside an async route blocks the entire Python asyncio event loop. 
    While that graph runs (which could take 10-30 seconds), the FastAPI server cannot accept or process any incoming HTTP requests 
    from any other users. Production async endpoints must always call `await app.ainvoke()`!
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Blocking the Async Event Loop with Synchronous Nodes (CRITICAL)**
    * *Error*: Defining synchronous node functions (`def my_node(state):`) that perform blocking `requests.get()` inside async graphs.
    * *Impact*: Starves the asyncio loop, destroying web server throughput.
    * *Fix*: Use `async def` node functions with `httpx` or `aiohttp` for all external I/O.
    """)

    st.error("""
    **Trap 2: Forgetting `version="v2"` in `astream_events` (CRITICAL)**
    * *Error*: Calling `app.astream_events(..., version="v1")`.
    * *Impact*: Version 1 is deprecated and emits a legacy schema with irregular event names.
    * *Fix*: Always specify `version="v2"` for standard OTel-compliant event dictionaries.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Not Consuming the Async Generator Completely (WARNING)**
    * *Warning*: Breaking early from an `astream` loop before execution terminates.
    * *Impact*: May leave in-flight nodes or database connections hanging without committing final checkpoints.
    * *Fix*: Consume streams to completion or use async context managers.
    """)

    st.warning("""
    **Trap 4: Missing `thread_id` When Resuming Interrupted Graphs (WARNING)**
    * *Warning*: Attempting to resume an interrupted thread without passing the identical `thread_id` in `config`.
    * *Impact*: Checkpointer cannot find historical state and starts a new blank execution thread.
    * *Fix*: Retain and pass the same `thread_id` across the entire multi-turn interaction.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: invoke vs ainvoke</strong><br><br>
        <em>Front:</em> When must ainvoke be used?<br><br>
        <em>Back:</em> Inside async web servers (FastAPI) to avoid blocking the asyncio event loop.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: stream(updates)</strong><br><br>
        <em>Front:</em> What does mode='updates' emit?<br><br>
        <em>Back:</em> Only the specific delta dictionary produced by the completed node.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: astream_events</strong><br><br>
        <em>Front:</em> What event name yields token chunks?<br><br>
        <em>Back:</em> <code>on_chat_model_stream</code>.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: interrupt_before</strong><br><br>
        <em>Front:</em> What does interrupt_before=['node'] do?<br><br>
        <em>Back:</em> Automatically halts execution and checkpoints state before running that node.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LangGraph execution protocols:")

chk1 = st.checkbox("I know how to compile graphs with checkpointers and interrupt breakpoints.", value=True)
chk2 = st.checkbox("I understand the synchronous blocking behavior of `invoke()`.", value=True)
chk3 = st.checkbox("I can differentiate between `stream(mode='values')` and `stream(mode='updates')`.", value=True)
chk4 = st.checkbox("I know why `ainvoke()` is mandatory for production FastAPI services.", value=True)
chk5 = st.checkbox("I understand how `astream_events(v2)` enables real-time token streaming.", value=True)
chk6 = st.checkbox("I know how to pass runtime configuration overrides via `RunnableConfig`.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 3.3: Graph Execution! You have completed Chapter 3.")
