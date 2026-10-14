import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="18.4 Type-Safe Streaming - LangGraph Curriculum",
    page_icon="🛡️",
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
    .badge-v2 {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-v1 {
        background-color: #382B14;
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
class TypeSafeStreamingSimulator:
    """
    Simulates LangGraph version="v2" type-safe streaming:
    - StreamPart discriminated union event payloads
    - Type safety validation (event tag matching)
    - Protocol parsing comparison (v1 ad-hoc tuples vs v2 StreamPart objects)
    """
    def __init__(self):
        pass

    def generate_v2_stream(self, query: str) -> List[Dict[str, Any]]:
        stream_parts = []
        # Event 1: Updates
        stream_parts.append({
            "event": "updates",
            "data": {"planner": {"query": query, "plan": "1. Analyze financial metrics"}},
            "tags": ["langgraph:step:1"],
            "metadata": {"langgraph_node": "planner", "run_id": "run-001"}
        })
        # Event 2: Messages (Token)
        stream_parts.append({
            "event": "messages",
            "data": ("AIMessageChunk", "Beginning", {"id": "msg-01"}),
            "tags": ["llm:token"],
            "metadata": {"langgraph_node": "assistant", "run_id": "run-002"}
        })
        stream_parts.append({
            "event": "messages",
            "data": ("AIMessageChunk", " financial analysis...", {"id": "msg-01"}),
            "tags": ["llm:token"],
            "metadata": {"langgraph_node": "assistant", "run_id": "run-002"}
        })
        # Event 3: Custom
        stream_parts.append({
            "event": "custom",
            "data": {"metric": "latency_ms", "value": 142},
            "tags": ["telemetry"],
            "metadata": {"langgraph_node": "assistant", "run_id": "run-002"}
        })
        # Event 4: Values (Final state)
        stream_parts.append({
            "event": "values",
            "data": {"query": query, "result": "Verified financial report ready.", "status": "COMPLETE"},
            "tags": ["langgraph:step:2"],
            "metadata": {"langgraph_node": "__end__", "run_id": "run-003"}
        })
        return stream_parts


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The v2 Type-Safe Streaming Protocol and StreamPart Unified Schema", divider="rainbow")

st.markdown("""
In earlier versions of LangGraph, streaming returned varying ad-hoc shapes depending on which flags were passed:
sometimes a raw dictionary, sometimes a 2-tuple, and sometimes a 3-tuple when subgraphs were enabled.
This inconsistency made it difficult for TypeScript frontends, microservices, and static type checkers to safely handle event streams without fragile type assertions (`isinstance` checks).

LangGraph introduced **type-safe streaming with `version="v2"`**.
Under `version="v2"`, every single emitted event conforms to a standardized, discriminated union envelope named **`StreamPart`**.

The blueprint below contrasts the legacy unstructured streaming formats against the unified `v2` `StreamPart` standard.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph TypeSafeStreamingTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_invocation {
            label="1. Invocation Configuration";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            app_stream [label="app.stream(inputs, version='v2')\\nType-Safe Protocol Flag Activated", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_streampart_envelope {
            label="2. The Standardized StreamPart Envelope";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            streampart_def [label="StreamPart Object\\n(Unified Discriminated Union)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            part_event [label="part['event']: Literal['updates', 'values', 'messages', 'custom']", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            part_data [label="part['data']: Strongly Typed Payload", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            part_meta [label="part['metadata']: Node, Step, & Run Context", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_pattern_matcher {
            label="3. Exhaustive Pattern Matching / Consumer Dispatch";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            match_event [label="match part['event']:\\n(Python 3.10+ / TypeScript Switch)", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            case_updates [label="case 'updates':\\nRender Node Step Badges", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            case_messages [label="case 'messages':\\nAppend LLM Tokens to Active Bubble", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            case_custom [label="case 'custom':\\nUpdate Progress Bars & Sliders", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            case_values [label="case 'values':\\nSync Redux / Zustand State Tree", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        app_stream -> streampart_def;
        streampart_def -> part_event;
        streampart_def -> part_data;
        streampart_def -> part_meta;

        part_event -> match_event;
        match_event -> case_updates [label="event == 'updates'"];
        match_event -> case_messages [label="event == 'messages'"];
        match_event -> case_custom [label="event == 'custom'"];
        match_event -> case_values [label="event == 'values'"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: STREAMING VERSIONS
# ==============================================================================
st.header("2. Streaming Versions: Evolution from Legacy Ad-Hoc Tuples to Structured Protocols", divider="rainbow")

st.markdown("""
Understanding why LangGraph introduced versions requires examining the historical evolution of streaming in the framework:
""")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("#### Legacy `version='v1'` (Default)")
    st.markdown("""
    - Yields raw dictionaries `{"node": {...}}` if a single mode is used.
    - Yields `(mode, data)` 2-tuples if multiple modes are requested.
    - Yields `(namespace, data)` or `(namespace, mode, data)` if `subgraphs=True` is enabled!
    - **Downside**: Writing generic event parsers requires multiple nested `isinstance` branches and checks for tuple lengths.
    """)

with col_v2:
    st.markdown("#### Modern `version='v2'` (Recommended)")
    st.markdown("""
    - **Always** yields a standardized dictionary structure (`StreamPart`).
    - The top-level key `event` clearly discriminates the payload type.
    - Uniform schema whether single-mode, multi-mode, or nested subgraphs are executing.
    - Compatible with TypeScript discriminated unions and Python `match/case` blocks.
    """)

# ==============================================================================
# SECTION 3: version="v2" ENGINE CONFIGURATION
# ==============================================================================
st.header("3. version=\"v2\" Engine Configuration: Activating Structured Stream Serialization", divider="rainbow")

st.markdown("""
To enable `version="v2"`, pass `version="v2"` into `app.stream()` or `app.astream()`.
""")

st.code('''# Activating version="v2" in synchronous or asynchronous streams
for part in app.stream(
    {"messages": [("user", "Analyze recent transactions")]},
    config={"configurable": {"thread_id": "thread_v2_001"}},
    stream_mode=["updates", "messages", "custom"],
    version="v2" # Activates structured StreamPart events!
):
    print("Event Type:", part["event"])
    print("Payload:", part["data"])
    print("Metadata:", part.get("metadata", {}))''', language="python")

# ==============================================================================
# SECTION 4: StreamPart
# ==============================================================================
st.header("4. StreamPart: The Unified Discriminated Union Event Envelope", divider="rainbow")

st.markdown("""
The formal typing definition of a `StreamPart` object in LangGraph is defined as a TypedDict with the following canonical structure:
""")

st.code('''from typing import TypedDict, Literal, Any, Dict, List, Optional

class StreamPart(TypedDict):
    event: Literal["updates", "values", "messages", "custom", "debug"]
    data: Any
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]''', language="python")

st.markdown("""
**Fields Breakdown**:
- `event`: The discriminator string identifying which streaming mode produced this chunk.
- `data`: The concrete payload for that mode (e.g. node delta dict for `updates`, message chunk for `messages`).
- `tags`: List of execution tags associated with the producing node or runnable.
- `metadata`: Execution context containing `langgraph_node`, `langgraph_step`, and `run_id`.
""")

# ==============================================================================
# SECTION 5: TYPED STREAMING OUTPUT
# ==============================================================================
st.header("5. Typed Streaming Output: Type-Safe Consumption, Pydantic Models, and IDE Autocompletion", divider="rainbow")

st.markdown("""
By combining `version="v2"` with Python 3.10+ structural pattern matching (`match/case`), you can write clean, compiler-validated, exhaustive consumer handlers:
""")

st.code('''# Exhaustive, type-safe stream event processing
def handle_v2_stream_part(part: dict):
    event_type = part["event"]
    data = part["data"]
    meta = part.get("metadata", {})
    
    match event_type:
        case "updates":
            for node_name, delta in data.items():
                print(f"[STEP COMPLETE] Node '{node_name}' finished with delta: {delta}")
                
        case "messages":
            message_chunk, message_meta = data
            if hasattr(message_chunk, "content") and message_chunk.content:
                print(message_chunk.content, end="", flush=True)
                
        case "custom":
            print(f"[CUSTOM TELEMETRY] {data}")
            
        case "values":
            print(f"[STATE SNAPSHOT] Current state: {data}")
            
        case _:
            print(f"[UNKNOWN EVENT] {event_type}")''', language="python")

# ==============================================================================
# SECTION 6: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("6. Production Implementation: Full v2 Type-Safe Streaming Pipeline with Pattern Matching", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable implementation demonstrating an agent configured with `version="v2"` streaming, custom StreamWriter telemetry, and structured event routing.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import MemorySaver

class AnalysisState(TypedDict):
    topic: str
    result: str

def research_node(state: AnalysisState, writer: StreamWriter) -> dict:
    writer({"stage": "fetch_papers", "progress": 50})
    return {"result": f"Synthesized whitepaper findings on {state['topic']}."}

builder = StateGraph(AnalysisState)
builder.add_node("researcher", research_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", END)

app = builder.compile(checkpointer=MemorySaver())

# Run Type-Safe v2 Stream Consumer
def run_v2_consumer():
    print("--- Executing v2 Stream Consumer ---")
    stream = app.stream(
        {"topic": "Quantum Encryption"},
        config={"configurable": {"thread_id": "v2_demo"}},
        stream_mode=["updates", "custom"],
        version="v2"
    )
    
    for part in stream:
        event = part["event"]
        if event == "custom":
            print(f"--> [TELEMETRY]: Stage={part['data']['stage']} ({part['data']['progress']}%)")
        elif event == "updates":
            print(f"--> [NODE DELTA]: {part['data']}")

run_v2_consumer()''', language="python")

# ==============================================================================
# SECTION 7: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("7. Interactive Workbench: StreamPart Event Demuxer and Protocol Inspector", divider="rainbow")

st.markdown("""
Inspect how `version="v2"` encapsulates diverse events into uniform, predictable `StreamPart` dictionaries.
""")

sim = TypeSafeStreamingSimulator()

col_v2_1, col_v2_2 = st.columns([1, 2])

with col_v2_1:
    st.subheader("⚙️ Stream Configuration")
    user_query = st.text_input("User Prompt", value="Evaluate company solvency")
    stream_version = st.selectbox("Protocol Version", ["version='v2' (Recommended)", "version='v1' (Legacy)"])
    execute_button = st.button("🚀 Emit Stream")

with col_v2_2:
    st.subheader("📦 Received Stream Envelope Inspect")
    if execute_button:
        parts = sim.generate_v2_stream(user_query)
        st.success(f"Emitted {len(parts)} StreamPart events with version='v2'!")
        
        for idx, p in enumerate(parts):
            with st.container():
                st.markdown(f"**Part {idx + 1}**: Event discriminator = `{p['event']}`")
                st.json(p)

st.subheader("📋 Version Protocol Comparison")
df_ver = pd.DataFrame({
    "Feature": ["Event Envelope", "Discriminator Field", "Subgraph Hierarchy Format", "Client Type Safety"],
    "version='v1'": ["Ad-hoc tuple or dict", "Inferred by length or key names", "Variable length tuples", "Requires runtime duck-typing"],
    "version='v2'": ["StreamPart object", "part['event'] Literal", "Preserved in metadata", "Guaranteed discriminated union"]
})
st.dataframe(df_ver, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 8: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("8. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What problem does version='v2' solve in LangGraph streaming?", expanded=False):
    st.markdown("""
    **Answer**:
    In `version='v1'`, stream outputs had inconsistent shapes depending on which modes were active and whether subgraphs were enabled (sometimes dicts, sometimes 2-tuples, sometimes 3-tuples).
    `version='v2'` standardizes all stream emissions into a uniform `StreamPart` dictionary with an explicit `event` discriminator field.
    """)

with st.expander("Q2: What is a StreamPart in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    `StreamPart` is the standardized dictionary envelope emitted by `version='v2'` streaming.
    It contains four core keys: `event` (the mode name), `data` (the mode payload), `tags` (execution tags), and `metadata` (node, step, and run identifiers).
    """)

with st.expander("Q3: How do TypeScript frontends benefit from version='v2'?", expanded=False):
    st.markdown("""
    **Answer**:
    TypeScript can treat `StreamPart` as a **discriminated union**.
    Switching on `part.event` allows TypeScript's compiler to automatically narrow the type of `part.data` without manual casting, eliminating runtime type errors.
    """)

with st.expander("Q4: Can you use version='v2' with single stream_mode='updates'?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Even with a single mode, passing `version='v2'` wraps the output in `{'event': 'updates', 'data': {...}}`, providing future-proof consistency if additional stream modes are added later.
    """)

with st.expander("Q5: What happens to subgraphs when using version='v2'?", expanded=False):
    st.markdown("""
    **Answer**:
    With `version='v2'` and `subgraphs=True`, the emitted items retain the uniform `StreamPart` envelope, and the subgraph namespace path is cleanly encapsulated inside the `metadata` dictionary instead of altering the outer tuple shape.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Accessing Chunks as Tuples When version='v2' is Active")
    st.markdown("""
    *Mechanism*: Writing `for mode, data in app.stream(..., version='v2')`.  
    *Consequence*: Raises `ValueError: too many values to unpack (expected 2)` because `part` is a 4-key dictionary, not a 2-tuple.  
    *Fix*: Unpack as `for part in app.stream(..., version='v2'): event = part['event']`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing Default Branch in match/case Handlers")
    st.markdown("""
    *Mechanism*: Matching only `case 'updates'` and `case 'messages'` in a frontend consumer.  
    *Consequence*: When LangGraph introduces a new event tag or custom event, the consumer raises unhandled exceptions.  
    *Fix*: Always provide a wildcard `case _:` fallback branch.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Assuming Metadata is Always Populated")
    st.markdown("""
    *Mechanism*: Directly referencing `part['metadata']['langgraph_node']` without `get()` or null checking.  
    *Consequence*: Certain root-level lifecycle events may have empty metadata dictionaries.  
    *Fix*: Use `part.get('metadata', {}).get('langgraph_node')`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Compatibility Warning): Upgrading Backend to v2 Without Updating Frontend Clients")
    st.markdown("""
    *Mechanism*: Setting `version='v2'` in the Python backend while React clients still expect v1 ad-hoc tuples.  
    *Consequence*: Immediate frontend crash across all active client sessions.  
    *Fix*: Coordinate version migration across API gateways and client SDKs.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**version='v2'**\\nThe LangGraph parameter that enforces structured, type-safe streaming events.")
with col_fc2:
    st.info("**StreamPart**\\nThe standardized dictionary envelope: {'event', 'data', 'tags', 'metadata'}.")
with col_fc3:
    st.info("**Discriminated Union**\\nA type pattern where the 'event' field dictates the exact shape of 'data'.")
with col_fc4:
    st.info("**Structural Pattern Matching**\\nUsing Python match/case to dispatch typed handlers across StreamParts.")

# ==============================================================================
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("9. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know why version='v2' was introduced to replace v1 ad-hoc streaming shapes.", value=True)
c2 = st.checkbox("I understand the four core keys of the StreamPart dictionary envelope.", value=True)
c3 = st.checkbox("I know how to consume v2 streams using Python 3.10+ match/case pattern matching.", value=True)
c4 = st.checkbox("I understand how TypeScript treats StreamPart as a discriminated union for safe UI rendering.", value=True)
c5 = st.checkbox("I know where subgraph hierarchy information resides under version='v2'.", value=True)
c6 = st.checkbox("I can migrate a legacy streaming consumer to the type-safe v2 protocol.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 18.4 'Type-Safe Streaming' with full architectural and practical comprehension!")
