import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="28.2 Advanced State - LangGraph Curriculum",
    page_icon="📦",
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
    .badge-state {
        background-color: #022C22;
        color: #6EE7B7;
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
class AdvancedStateReducerSimulator:
    """
    Simulates channel reducer behavior under concurrent updates:
    - Default Overwrite (raises InvalidUpdateError if multiple concurrent writes)
    - List Append Reducer
    - Custom Dictionary Deep Merge
    - Numeric Counter Reducer
    """
    def __init__(self):
        pass

    def simulate_concurrent_merge(self, strategy: str, update_a: Any, update_b: Any) -> Dict[str, Any]:
        if strategy == "Default (No Reducer)":
            return {
                "success": False,
                "error": "InvalidUpdateError: Multiple concurrent writes to unreduced channel 'output_key' in the same superstep!",
                "result": None
            }
        elif strategy == "List Append (operator.add)":
            base = ["initial_item"]
            merged = base + [update_a, update_b]
            return {"success": True, "result": merged, "explanation": "Appended both updates sequentially into the list channel."}
        elif strategy == "Numeric Counter":
            base = 10
            merged = base + int(update_a) + int(update_b)
            return {"success": True, "result": merged, "explanation": f"Summed both updates into base 10: 10 + {update_a} + {update_b} = {merged}"}
        else:  # Custom Dict Merge
            base = {"version": "v1.0", "author": "system"}
            base.update(update_a)
            base.update(update_b)
            return {"success": True, "result": base, "explanation": "Coalesced dictionary keys sequentially."}

state_sim = AdvancedStateReducerSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Channel Reducers, Concurrent Writes, and State Merging Topology", divider="rainbow")

st.markdown("""
State management in LangGraph is governed by **Channels** and **Reducers**. Unlike conventional programming frameworks 
where state variables are directly mutated in-place, LangGraph channels are immutable communication pipelines. 
When one or more nodes complete their execution within a superstep, their emitted updates are submitted to the channel's 
reducer function. 

Understanding advanced channel mechanics is essential for building concurrent multi-agent architectures, 
preventing race conditions during parallel fan-out, and ensuring deterministic checkpoint commits.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedStateBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_parallel_nodes {
            label="1. Concurrent Node Execution (Superstep Fan-Out)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            node_a [label="Worker Node A\\nEmits: {'messages': [MsgA], 'counter': 1}", fillcolor="#0F172A"];
            node_b [label="Worker Node B\\nEmits: {'messages': [MsgB], 'counter': 2}", fillcolor="#0F172A"];
        }

        subgraph cluster_superstep_barrier {
            label="2. Superstep Barrier & Channel Reducers";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            barrier [label="Superstep Barrier Synchronization\\nBuffers all emitted deltas simultaneously", fillcolor="#312E81"];
            reducer_msg [label="Channel: 'messages'\\nReducer: add_messages\\nAppends & Deduplicates IDs", fillcolor="#022C22", fontcolor="#6EE7B7"];
            reducer_counter [label="Channel: 'counter'\\nReducer: operator.add\\nSums 1 + 2 = 3", fillcolor="#022C22", fontcolor="#6EE7B7"];
            unreduced_chan [label="Channel: 'status' (No Reducer)\\nThrows InvalidUpdateError if collided", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        subgraph cluster_committed_state {
            label="3. Committed Checkpoint State";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            final_state [label="StateSnapshot Committed\\nChannels updated deterministically\\nPersisted to PostgresSaver", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        node_a -> barrier;
        node_b -> barrier;

        barrier -> reducer_msg;
        barrier -> reducer_counter;
        barrier -> unreduced_chan;

        reducer_msg -> final_state;
        reducer_counter -> final_state;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Reducers
# ==============================================================================
st.header("2. Reducers: Custom Accumulator Functions, Delta Transformation, and Mathematical Monoids", divider="rainbow")

st.markdown("""
A **Reducer** is a binary function accepting two arguments: the current channel value and an incoming update value:
`reducer(current_value, new_update) -> updated_value`

### Formal Reducer Properties:
1. **Mathematical Closure**: The returned value must conform to the channel's declared type.
2. **Associativity**: For parallel fan-in where updates may be merged in variable order, associative operations 
   (such as set unions or list concatenations) guarantee deterministic outcomes.
3. **Pure Execution**: Reducers must be pure Python functions with zero I/O or side effects, as checkpointers 
   may re-evaluate reducers during state replay.
""")

st.code('''from typing import List, Dict, Any

# Custom dictionary deep-merge reducer
def merge_dict_channel(current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    if not current:
        return dict(update)
    merged = dict(current)
    merged.update(update)
    return merged

# Custom set-union reducer for deduplicating URLs
def union_urls(current: set, update: set) -> set:
    if not current:
        return set(update)
    return current.union(update)
''', language="python")

# ==============================================================================
# SECTION 3: Annotated State
# ==============================================================================
st.header("3. Annotated State: Using typing.Annotated, Channel Type Metadata, and Default Factories", divider="rainbow")

st.markdown("""
LangGraph binds reducers to state channels using Python's `typing.Annotated` type hinting mechanism. 

### Syntax Breakdown:
```python
channel_name: Annotated[Type, ReducerFunction]
```

If a channel is declared without `Annotated` (e.g., `query: str`), LangGraph treats it as an **unreduced channel**:
- Updates overwrite the previous value.
- If two nodes attempt to update the channel in the same superstep, LangGraph raises `InvalidUpdateError`.
""")

st.code('''import operator
from typing import TypedDict, Annotated, List, Set
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ComplexAgentState(TypedDict):
    # Reduced list appending & message ID deduplication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Reduced integer summation across parallel workers
    total_tokens_consumed: Annotated[int, operator.add]
    
    # Reduced unique URL set union
    visited_urls: Annotated[Set[str], lambda cur, upd: cur.union(upd)]
    
    # Unreduced channel: Overwritten on update; single writer per superstep
    current_status: str
''', language="python")

# ==============================================================================
# SECTION 4: State Merging
# ==============================================================================
st.header("4. State Merging: Superstep Barrier Aggregation, Order Determinism, and Key Coalescence", divider="rainbow")

st.markdown("""
During execution, multiple nodes can complete their tasks at slightly different times. LangGraph synchronizes them 
at the **Superstep Barrier**.

### How Merging Executes:
1. All node return dictionaries are buffered.
2. State keys are partitioned by channel name.
3. For each channel, the corresponding reducer is invoked sequentially with each node's update.
4. The resulting state snapshot is written to storage atomically.
""")

# ==============================================================================
# SECTION 5: Concurrent Updates
# ==============================================================================
st.header("5. Concurrent Updates: Parallel Node State Writes, Fan-In Collisions, and Superstep Resolution", divider="rainbow")

st.markdown("""
When scaling graphs with fan-out patterns (e.g., Map-Reduce or parallel research sub-agents), concurrent writes 
to the same state keys are standard practice.

### Safe Concurrent Channel Patterns:
- **`add_messages`**: Safe. Parallel nodes can emit `ToolMessage` or `AIMessage` concurrently; both will be appended.
- **`operator.add`**: Safe for counters and lists.
- **Unreduced Channels**: Unsafe. Attempting concurrent writes to an unreduced string or integer raises a fatal runtime exception.
""")

# ==============================================================================
# SECTION 6: Conflict Resolution
# ==============================================================================
st.header("6. Conflict Resolution: Last-Write-Wins vs Custom Reducer Merging vs Error Gating", divider="rainbow")

st.markdown("""
When two parallel nodes emit conflicting data (e.g., Node A says `approved: True`, Node B says `approved: False`), 
organizations must choose a conflict resolution strategy:

| Strategy | Implementation | Tradeoff |
| :--- | :--- | :--- |
| **Error Gating** | No reducer (Default) | Fails loudly; prevents silent bugs |
| **Last-Write-Wins** | `lambda cur, upd: upd` | Non-deterministic in parallel execution |
| **Strict Logical AND** | `operator.and_` | Conservative; both nodes must approve |
| **Priority Coalescence**| Custom Reducer | Inspects node metadata or confidence score |
""")

# ==============================================================================
# SECTION 7: State Isolation
# ==============================================================================
st.header("7. State Isolation: Thread Scoping, Checkpoint Namespaces, and Ephemeral Channel Scrubbing", divider="rainbow")

st.markdown("""
Maintaining clean state boundaries prevents state leakage between tenants, sessions, and child subgraphs:

- **Thread Scoping**: Ensures distinct users operate on strictly isolated checkpoints.
- **Checkpoint Namespaces**: Subgraphs run in nested namespace partitions.
- **Ephemeral Scratchpad Scrubbing**: Channels used for temporary calculations can define reducers that reset 
  the channel value to `None` upon reaching `END`.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Channel Advanced State Pipeline with Custom Reducers", divider="rainbow")

st.markdown("""
The following complete module implements custom reducers for parallel worker task tracking, confidence score averaging, 
and deduplicated URL logging.
""")

st.code('''from typing import TypedDict, Annotated, List, Set, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END

# Custom Reducer: Moving Average of Quality Scores
def average_score_reducer(current: float, update: float) -> float:
    if current == 0.0:
        return update
    return round((current + update) / 2.0, 3)

# Custom Reducer: Set Union
def set_union_reducer(current: Set[str], update: Set[str]) -> Set[str]:
    if not current:
        return set(update)
    return current.union(update)

class AdvancedState(TypedDict):
    task: str
    worker_notes: Annotated[List[str], operator.add]
    overall_confidence: Annotated[float, average_score_reducer]
    visited_sources: Annotated[Set[str], set_union_reducer]

# Parallel Worker 1
def worker_1_node(state: AdvancedState) -> dict:
    return {
        "worker_notes": ["Worker 1 completed SEC 10-K analysis"],
        "overall_confidence": 0.92,
        "visited_sources": {"https://sec.gov/filing-10k"}
    }

# Parallel Worker 2
def worker_2_node(state: AdvancedState) -> dict:
    return {
        "worker_notes": ["Worker 2 completed Earnings Call transcript check"],
        "overall_confidence": 0.88,
        "visited_sources": {"https://earnings.com/transcript", "https://sec.gov/filing-10k"}
    }

builder = StateGraph(AdvancedState)
builder.add_node("worker_1", worker_1_node)
builder.add_node("worker_2", worker_2_node)

builder.add_edge(START, "worker_1")
builder.add_edge(START, "worker_2")
builder.add_edge("worker_1", END)
builder.add_edge("worker_2", END)

parallel_merge_graph = builder.compile()
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: State Reducer Simulator & Concurrent Write Conflict Sandbox", divider="rainbow")

st.markdown("""
Test different channel reducer strategies when two parallel nodes write to the same state channel simultaneously.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Concurrent Channel Write Simulator")
    selected_strategy = st.selectbox(
        "Select Channel Reducer Strategy",
        ["Default (No Reducer)", "List Append (operator.add)", "Numeric Counter", "Custom Dict Merge"]
    )
    
    val_a = st.text_input("Worker Node A Update", value="15")
    val_b = st.text_input("Worker Node B Update", value="25")
    
    if st.button("Simulate Superstep Merge", type="primary"):
        st.session_state["merge_simulated"] = True

with col_wb2:
    st.subheader("Superstep Barrier Merge Verdict")
    if selected_strategy == "Custom Dict Merge":
        res = state_sim.simulate_concurrent_merge(selected_strategy, {"node_a": val_a}, {"node_b": val_b})
    else:
        res = state_sim.simulate_concurrent_merge(selected_strategy, val_a, val_b)
        
    if not res["success"]:
        st.error(res["error"])
        st.warning("Recommendation: Annotate the channel using Annotated[T, reducer_func] to allow concurrent writes.")
    else:
        st.success("State Merge Successful! Superstep barrier resolved without collision.")
        st.markdown(f"**Explanation:** {res['explanation']}")
        st.json(res["result"])

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What exactly triggers an InvalidUpdateError in LangGraph?"):
    st.markdown("""
    **Answer:**  
    An `InvalidUpdateError` is triggered when two or more nodes executing concurrently in the same superstep return updates 
    for the exact same state key, and that state key was declared without a channel reducer in the `TypedDict` schema. 
    LangGraph cannot determine which update should take precedence, so it fails loudly.
    """)

with st.expander("Q2: How does add_messages handle message deduplication during state merging?"):
    st.markdown("""
    **Answer:**  
    `add_messages` inspects the `id` property of incoming messages. If an incoming message ID already exists in the channel list, 
    the incoming message overwrites the existing message in-place rather than appending a duplicate. If the ID is new, it appends to the list.
    """)

with st.expander("Q3: Why must custom reducer functions remain side-effect free?"):
    st.markdown("""
    **Answer:**  
    Reducers are executed by the LangGraph engine during superstep barriers and during state replay/inspection traversals. 
    If a reducer performs network requests, database writes, or modifies mutable objects in-place, state inspection and time travel 
    will trigger unintended side effects and corrupted data.
    """)

with st.expander("Q4: Can a channel reducer return a completely different data type than its input arguments?"):
    st.markdown("""
    **Answer:**  
    No. The reducer must return an object matching the channel's declared type annotation. For example, if a channel is declared 
    as `Annotated[List[str], reducer]`, the reducer must return `List[str]`.
    """)

with st.expander("Q5: How can you reset or clear a reduced channel value back to empty?"):
    st.markdown("""
    **Answer:**  
    Define a reducer that supports a sentinel reset value (e.g., `REMOVE` or `None`). When an update contains the sentinel, 
    the reducer returns an empty list or dictionary rather than appending.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Mutating Mutable State Objects In-Place Inside Node Functions**  
Mutating lists or dictionaries in-place (e.g., `state["messages"].append(msg)`) bypasses channel reducers, corrupts checkpointer 
snapshot immutability, and breaks time-travel rollbacks. Always return new delta dictionaries: `return {"messages": [msg]}`.
""")

st.error("""
**Trap 2: Forgetting to Annotate Parallel Fan-Out State Channels**  
Branching execution to parallel workers that write to unreduced channels immediately crashes with `InvalidUpdateError` 
on the very first superstep barrier.
""")

st.warning("""
**Trap 3: Non-Commutative Reducers in Distributed Parallelism**  
If a reducer depends strictly on the arrival order of parallel updates (e.g., string concatenation `a + b`), variations in network 
latency can cause non-deterministic state across runs. Use associative, commutative reducers where possible.
""")

st.warning("""
**Trap 4: Excessive Memory Bloat in Unbounded Reducer Lists**  
Using `operator.add` on raw lists without pruning or deduplication causes state arrays to grow indefinitely, 
inflating checkpoint storage and serialization latencies.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Channel Reducer</h4>
        <p>A binary pure function specifying how incoming node updates are merged into an existing state channel at superstep barriers.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>InvalidUpdateError</h4>
        <p>The exception thrown when multiple parallel nodes attempt to write to an unreduced state channel simultaneously.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>typing.Annotated</h4>
        <p>The Python typing construct used in LangGraph TypedDict schemas to attach custom reducer functions to channel types.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>add_messages Reducer</h4>
        <p>The built-in LangGraph reducer managing message list accumulation, ID-based replacements, and deduplication.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to define custom channel reducers using typing.Annotated.", value=True)
chk_2 = st.checkbox("I understand why unreduced channels throw InvalidUpdateError during concurrent writes.", value=True)
chk_3 = st.checkbox("I can implement conflict resolution strategies (set union, list append, mathematical accumulation).", value=True)
chk_4 = st.checkbox("I understand how add_messages deduplicates and updates existing messages by ID.", value=True)
chk_5 = st.checkbox("I know how to maintain state isolation across threads and checkpoint namespaces.", value=True)
chk_6 = st.checkbox("I understand why node functions must return state deltas rather than mutating state in-place.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced state channels and reducer mastery confirmed! Ready for Advanced Execution.")
