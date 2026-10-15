import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="18.2 Streaming Modes - LangGraph Curriculum",
    page_icon="🎛️",
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
    .badge-mode {
        background-color: #1E293B;
        color: #FBBF24;
        border: 1px solid #D97706;
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
class StreamingModesSimulator:
    """
    Simulates and compares all 5 LangGraph streaming modes:
    - 'values': Full state snapshots per superstep
    - 'updates': Granular node channel deltas
    - 'messages': LLM token chunks with metadata
    - 'custom': Custom progress events emitted via StreamWriter
    - 'debug': Full Pregel debug traces including task lifecycles
    """
    def __init__(self):
        pass

    def run_mode_simulation(self, mode: str, query: str) -> List[Dict[str, Any]]:
        events = []
        if mode == "values":
            # Initial state
            events.append({
                "mode": "values",
                "step": 0,
                "data": {"query": query, "plan": None, "tokens": 0, "status": "INIT"}
            })
            # Step 1: Planner
            events.append({
                "mode": "values",
                "step": 1,
                "data": {"query": query, "plan": "1. Analyze, 2. Synthesize", "tokens": 25, "status": "PLANNED"}
            })
            # Step 2: Synthesis
            events.append({
                "mode": "values",
                "step": 2,
                "data": {"query": query, "plan": "1. Analyze, 2. Synthesize", "answer": "Analysis complete.", "tokens": 140, "status": "DONE"}
            })

        elif mode == "updates":
            # Step 1 delta
            events.append({
                "mode": "updates",
                "step": 1,
                "data": {"planner_node": {"plan": "1. Analyze, 2. Synthesize", "tokens": 25}}
            })
            # Step 2 delta
            events.append({
                "mode": "updates",
                "step": 2,
                "data": {"synthesis_node": {"answer": "Analysis complete.", "status": "DONE"}}
            })

        elif mode == "messages":
            tokens = ["Hello", "!", " I", " have", " evaluated", " '", query, "'", " successfully", "."]
            for i, tok in enumerate(tokens):
                events.append({
                    "mode": "messages",
                    "step": i + 1,
                    "data": {
                        "chunk": tok,
                        "node": "assistant_llm",
                        "type": "AIMessageChunk"
                    }
                })

        elif mode == "custom":
            events.append({
                "mode": "custom",
                "step": 1,
                "data": {"type": "progress_bar", "percent": 25, "message": "Connecting to document vector store..."}
            })
            events.append({
                "mode": "custom",
                "step": 2,
                "data": {"type": "progress_bar", "percent": 70, "message": "Applying cross-encoder reranker..."}
            })
            events.append({
                "mode": "custom",
                "step": 3,
                "data": {"type": "progress_bar", "percent": 100, "message": "Synthesis complete."}
            })

        elif mode == "debug":
            events.append({
                "mode": "debug",
                "step": 1,
                "data": {
                    "type": "task",
                    "name": "planner_node",
                    "step": 1,
                    "writes": [("plan", "1. Analyze")],
                    "triggers": ["__start__"]
                }
            })
            events.append({
                "mode": "debug",
                "step": 2,
                "data": {
                    "type": "checkpoint",
                    "checkpoint_id": "1efb-88a2-9011",
                    "parent_checkpoint_id": "1efb-88a2-9000",
                    "versions_seen": {"messages": 2}
                }
            })
        return events


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Demultiplexing Streaming Modes in the LangGraph Pregel Engine", divider="rainbow")

st.markdown("""
LangGraph provides a unified streaming architecture with **five specialized stream modes**:
1. `values`: Full accumulated state snapshots after each superstep.
2. `updates`: Node-level state deltas (channel additions only).
3. `messages`: Sub-word tokens and tool call chunks emitted by LLM models.
4. `custom`: Custom user-defined telemetry emitted from within nodes via `StreamWriter`.
5. `debug`: Comprehensive low-level Pregel lifecycle traces (tasks, checkpoint writes, and triggers).

The blueprint below illustrates how the LangGraph event stream demultiplexer dispatches each mode to specialized frontend consumers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StreamingModesArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_invocation {
            label="1. Stream Ingress";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            app_stream [label="app.stream(inputs, stream_mode=[...])\\nUnified Stream Demultiplexer", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_modes {
            label="2. The Five Streaming Modes";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            m_values [label="stream_mode='values'\\nFull State Snapshots", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            m_updates [label="stream_mode='updates'\\nNode Channel Deltas", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            m_messages [label="stream_mode='messages'\\nToken & Tool Call Chunks", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            m_custom [label="stream_mode='custom'\\nStreamWriter Telemetry", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            m_debug [label="stream_mode='debug'\\nTask & Checkpoint Traces", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
        }

        subgraph cluster_consumers {
            label="3. Specialized Client Consumers";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            ui_state_viewer [label="State Inspector & History\\n(Full Tree Visualizer)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_node_progress [label="Node Step Indicator\\n('Planner Complete', 'DB Synced')", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_chat_stream [label="Typewriter Chat Box\\n(Live LLM Generation)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_custom_progress [label="Custom Progress Bars\\n(Vector DB Search 75%)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            ui_debugger [label="Developer Console / Datadog\\n(Pregel Superstep Metrics)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        app_stream -> m_values;
        app_stream -> m_updates;
        app_stream -> m_messages;
        app_stream -> m_custom;
        app_stream -> m_debug;

        m_values -> ui_state_viewer;
        m_updates -> ui_node_progress;
        m_messages -> ui_chat_stream;
        m_custom -> ui_custom_progress;
        m_debug -> ui_debugger;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: VALUES STREAM MODE
# ==============================================================================
st.header("2. Values Stream Mode: Continuous Full-State Inspection Across Supersteps", divider="rainbow")

st.markdown("""
In `stream_mode="values"`, LangGraph yields the **entire accumulated state dictionary** at every superstep boundary.
""")

st.markdown("""
- **Yield Frequency**: Once before the first node runs (initial state), and once after every superstep barrier completes.
- **Data Payload**: The full state object containing all channels.
- **Primary Use Case**: Synchronizing a client-side state store (e.g. Redux or Zustand) where the frontend wants to replace its local state tree with the latest server state snapshot.
""")

st.code('''for state in app.stream({"messages": [("user", "Hello")]}, stream_mode="values"):
    print("Messages count:", len(state["messages"]))
    print("All keys present:", list(state.keys()))''', language="python")

# ==============================================================================
# SECTION 3: UPDATES STREAM MODE
# ==============================================================================
st.header("3. Updates Stream Mode: Discrete Superstep Deltas and Multi-Node Dictionaries", divider="rainbow")

st.markdown("""
In `stream_mode="updates"` (the default in LangGraph), the stream yields only the **channel deltas** produced by nodes in that superstep.
""")

st.markdown("""
- **Yield Frequency**: After each superstep barrier completes.
- **Data Payload**: A dictionary mapping node names to their output dictionaries: `{node_name: {channel: delta_value}}`.
- **Parallel Execution**: If two nodes execute concurrently in the same superstep, the dictionary contains keys for both nodes:
  `{"node_a": {...}, "node_b": {...}}`
- **Primary Use Case**: Triggering toast notifications or status indicators when specific nodes finish without redrawing the whole UI.
""")

st.code('''for chunk in app.stream(inputs, stream_mode="updates"):
    for node_name, updates in chunk.items():
        print(f"Node '{node_name}' finished. Output: {updates}")''', language="python")

# ==============================================================================
# SECTION 4: MESSAGES STREAM MODE
# ==============================================================================
st.header("4. Messages Stream Mode: Real-Time Token Generation and Tool Call Chunking", divider="rainbow")

st.markdown("""
In `stream_mode="messages"`, LangGraph taps directly into LLM runnables inside nodes, emitting tokens and tool call fragments as they are generated.
""")

st.markdown("""
- **Yield Frequency**: On every token emitted by any LLM inside any node.
- **Data Payload**: A 2-tuple: `(message_chunk, metadata)`.
  - `message_chunk`: An `AIMessageChunk` with `.content` (text) or `.tool_call_chunks` (partial JSON tool arguments).
  - `metadata`: Contains `langgraph_node`, `langgraph_step`, and `ls_provider`.
""")

st.code('''for message_chunk, metadata in app.stream(inputs, stream_mode="messages"):
    if metadata.get("langgraph_node") == "assistant":
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)''', language="python")

# ==============================================================================
# SECTION 5: CUSTOM STREAMING
# ==============================================================================
st.header("5. Custom Streaming: Emitting Arbitrary Progress Events via StreamWriter", divider="rainbow")

st.markdown("""
Sometimes a node performs a long-running operation (e.g., downloading a 500MB PDF or querying a vector index) that is neither an LLM token nor a complete superstep update.
LangGraph provides `StreamWriter` to emit **custom streaming events** directly from inside node code!
""")

st.code('''from langgraph.types import StreamWriter

def custom_search_node(state: State, writer: StreamWriter) -> dict:
    # Emit custom progress events to the client stream
    writer({"type": "status", "message": "Connecting to Pinecone..."})
    time.sleep(1)
    
    writer({"type": "progress", "percent": 50, "docs_found": 12})
    time.sleep(1)
    
    writer({"type": "status", "message": "Reranking documents..."})
    return {"documents": ["Doc 1", "Doc 2"]}

# Consuming custom stream events
for custom_event in app.stream(inputs, stream_mode="custom"):
    print("Received custom event:", custom_event)''', language="python")

# ==============================================================================
# SECTION 6: DEBUG STREAMING
# ==============================================================================
st.header("6. Debug Streaming: Deep Tracing of Tasks, State Transitions, and Checkpoint Writes", divider="rainbow")

st.markdown("""
For distributed systems debugging, `stream_mode="debug"` provides comprehensive low-level execution telemetry.
""")

st.markdown("""
- **Task Traces**: Shows when a task is scheduled, started, and completed, along with input arguments.
- **Checkpoint Traces**: Displays the exact checkpoint ID generated, parent checkpoint ID, and pending writes.
- **Trigger Traces**: Identifies which channel update triggered which node.
""")

st.code('''for debug_payload in app.stream(inputs, stream_mode="debug"):
    print(f"Debug Type: {debug_payload.get('type')}")
    print(f"Payload: {debug_payload}\\n")''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Mode Event Multiplexer and UI Dispatcher", divider="rainbow")

st.markdown("""
In modern production applications, frontends consume **multiple stream modes simultaneously**.
LangGraph allows passing a list of modes: `stream_mode=["updates", "messages", "custom"]`.
""")

st.code('''from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import MemorySaver

class MultiModeState(TypedDict):
    query: str
    summary: str

def analysis_node(state: MultiModeState, writer: StreamWriter) -> dict:
    writer({"progress": 33, "status": "Reading dataset"})
    writer({"progress": 66, "status": "Calculating metrics"})
    return {"summary": "Analysis of data completed."}

builder = StateGraph(MultiModeState)
builder.add_node("analysis", analysis_node)
builder.add_edge(START, "analysis")
builder.add_edge("analysis", END)

app = builder.compile(checkpointer=MemorySaver())

# Multi-mode consumer loop
def consume_multi_mode(query: str):
    print("=== Multi-Mode Stream Dispatcher Started ===")
    # When multiple modes are requested, items are emitted as (mode, data) tuples
    for mode, payload in app.stream(
        {"query": query},
        config={"configurable": {"thread_id": "mm_01"}},
        stream_mode=["updates", "custom"]
    ):
        if mode == "custom":
            print(f"[UI PROGRESS BAR] {payload['status']} ({payload['progress']}%)")
        elif mode == "updates":
            print(f"[SUPERSTEP COMPLETE] Node update: {payload}")

consume_multi_mode("Evaluate Portfolio Risk")''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Streaming Mode Comparison Matrix and Live Event Inspector", divider="rainbow")

st.markdown("""
Select any of the 5 streaming modes and observe the exact shape and frequency of emitted payloads.
""")

sim = StreamingModesSimulator()

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    st.subheader("⚙️ Stream Mode Selector")
    selected_mode = st.radio(
        "Choose Streaming Mode",
        ["values", "updates", "messages", "custom", "debug"],
        index=1
    )
    test_query = st.text_input("Test Query", value="Summarize annual report")
    trigger_stream = st.button("🚀 Stream Events")

with col_m2:
    st.subheader(f"📡 Emitted Event Stream (`stream_mode='{selected_mode}'`)")
    if trigger_stream:
        events = sim.run_mode_simulation(selected_mode, test_query)
        st.info(f"Emitted {len(events)} event(s) for mode: `{selected_mode}`")
        for ev in events:
            with st.container():
                st.markdown(f"**Step {ev['step']}** | Mode: `{ev['mode']}`")
                st.json(ev["data"])

st.subheader("📊 Comparative Mode Selection Matrix")
df_matrix = pd.DataFrame({
    "Stream Mode": ["values", "updates", "messages", "custom", "debug"],
    "Payload Type": ["Full State Dict", "Node Delta Dict", "(AIMessageChunk, meta)", "Arbitrary User Dict", "Pregel Internal Event"],
    "Emission Frequency": ["Per Superstep", "Per Superstep", "Per LLM Token", "On writer() call", "On Every Pregel Action"],
    "Best For": ["Redux/Zustand Store Sync", "Step Progress Indicators", "Typewriter Chat UIs", "Task Progress & Metrics", "Engine Tracing & Telemetry"]
})
st.dataframe(df_matrix, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the primary difference between 'values' and 'updates' streaming modes?", expanded=False):
    st.markdown("""
    **Answer**:
    - `stream_mode='values'` emits the entire accumulated state dictionary after each superstep barrier.
    - `stream_mode='updates'` emits only the state delta returned by the active node(s) during that superstep, mapped under `{node_name: delta}`.
    """)

with st.expander("Q2: How does StreamWriter work and when should you use 'custom' stream mode?", expanded=False):
    st.markdown("""
    **Answer**:
    `StreamWriter` is a callable injected into node functions by adding a `writer: StreamWriter` parameter.
    Calling `writer({...})` emits custom telemetry to the stream in real-time, ideal for progress bars, intermediate calculations, or external API status updates before the node finishes.
    """)

with st.expander("Q3: What data format is emitted when requesting multiple stream modes simultaneously?", expanded=False):
    st.markdown("""
    **Answer**:
    When passing a list of modes (e.g. `stream_mode=['updates', 'messages']`), LangGraph emits a 2-tuple:
    `(mode_name, payload)`, allowing clients to demultiplex events and route them to different UI components.
    """)

with st.expander("Q4: What information does 'debug' streaming mode provide that others do not?", expanded=False):
    st.markdown("""
    **Answer**:
    `debug` mode emits internal Pregel engine events, including task initialization, task inputs, checkpoint generation, parent checkpoint IDs, triggers, and pending writes.
    """)

with st.expander("Q5: Can stream_mode='messages' emit tool call arguments before the tool executes?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! When the LLM decides to invoke a tool, `stream_mode='messages'` emits partial tool call chunks containing fragments of the tool name and JSON arguments as they are being generated.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Parsing Stream Tuples Without Mode Matching")
    st.markdown("""
    *Mechanism*: Assuming `for chunk in app.stream(...)` always yields a dict when multiple stream modes are requested.  
    *Consequence*: In multi-mode streaming, chunks are `(mode, data)` tuples; treating them as dictionaries raises `AttributeError: tuple has no attribute get`.  
    *Fix*: Always unpack `mode, payload = chunk` when multiple modes are enabled.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Passing StreamWriter Without Parameter Type Annotation")
    st.markdown("""
    *Mechanism*: Defining `def my_node(state, writer):` without the `StreamWriter` type hint.  
    *Consequence*: LangGraph does not recognize the parameter as a StreamWriter and fails to inject it at runtime.  
    *Fix*: Always annotate with `writer: StreamWriter` from `langgraph.types`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Overusing 'values' Mode with Large State Schemas")
    st.markdown("""
    *Mechanism*: Streaming `values` mode over a graph where state contains 10MB of indexed documents or large embeddings.  
    *Consequence*: Re-serializes and transfers 10MB across the wire on every single superstep, causing network congestion.  
    *Fix*: Use `stream_mode='updates'` to transmit only tiny deltas.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (UI Glitch Warning): Mixing Token Chunks from Multiple Parallel Nodes")
    st.markdown("""
    *Mechanism*: In a graph with two parallel LLM nodes running concurrently, streaming tokens directly to a single chat box.  
    *Consequence*: Tokens from both models interleave and create gibberish text in the UI.  
    *Fix*: Filter tokens using `metadata['langgraph_node']` to route them to separate UI buffers.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**stream_mode='values'**\\nEmits the full state dictionary after every completed superstep barrier.")
with col_fc2:
    st.info("**stream_mode='updates'**\\nEmits a dict of {node_name: delta} for nodes completed in that superstep.")
with col_fc3:
    st.info("**StreamWriter**\\nNode callable parameter for emitting arbitrary custom telemetry events.")
with col_fc4:
    st.info("**stream_mode='debug'**\\nEmits low-level Pregel engine events, task allocations, and checkpoint IDs.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know all 5 streaming modes in LangGraph and their primary use cases.", value=True)
c2 = st.checkbox("I understand the difference between 'values' and 'updates' payloads.", value=True)
c3 = st.checkbox("I know how to emit custom telemetry from inside a node using StreamWriter.", value=True)
c4 = st.checkbox("I know how to consume multiple stream modes concurrently using a list.", value=True)
c5 = st.checkbox("I understand what information is surfaced by 'debug' mode for Pregel engine tracing.", value=True)
c6 = st.checkbox("I know how to filter message chunks by node name using event metadata.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 18.2 'Streaming Modes' with full architectural and practical comprehension!")
