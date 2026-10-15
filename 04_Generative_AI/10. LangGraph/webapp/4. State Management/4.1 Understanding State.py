import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="4.1 Understanding State - LangGraph Curriculum",
    page_icon="🧠",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE STATE ENGINE & REDUCER SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class StateLifecycleEngine:
    """
    Simulates LangGraph state lifecycle across supersteps.
    Demonstrates snapshot isolation, partial state updates, and reducer merges.
    """
    def __init__(self, initial_query: str, session_id: str):
        self.session_id = session_id
        self.current_state: Dict[str, Any] = {
            "query": initial_query,
            "messages": [f"User: {initial_query}"],
            "iteration_count": 0,
            "context_docs": [],
            "status": "INITIALIZED"
        }
        self.history: List[Dict[str, Any]] = [
            {
                "superstep": 0,
                "node": "START",
                "delta": dict(self.current_state),
                "snapshot": dict(self.current_state),
                "reducer_action": "Initial Ingestion"
            }
        ]

    def execute_node_step(self, node_name: str, delta: Dict[str, Any], use_append_reducer: bool = True) -> Dict[str, Any]:
        """
        Executes a superstep update.
        Simulates channel reducers:
        - messages: append reducer (operator.add style) if use_append_reducer else overwrite
        - iteration_count: numeric overwrite / increment
        - status: overwrite
        - context_docs: list extend
        """
        superstep_idx = len(self.history)
        new_state = dict(self.current_state)

        for key, val in delta.items():
            if key == "messages":
                if use_append_reducer:
                    if isinstance(val, list):
                        new_state["messages"] = self.current_state["messages"] + val
                    else:
                        new_state["messages"] = self.current_state["messages"] + [val]
                else:
                    new_state["messages"] = val if isinstance(val, list) else [val]
            elif key == "context_docs":
                new_state["context_docs"] = self.current_state["context_docs"] + (val if isinstance(val, list) else [val])
            elif key == "iteration_count":
                new_state["iteration_count"] = self.current_state["iteration_count"] + val
            else:
                new_state[key] = val

        self.current_state = new_state
        step_record = {
            "superstep": superstep_idx,
            "node": node_name,
            "delta": delta,
            "snapshot": dict(self.current_state),
            "reducer_action": "Reducer Merged (Append + Overwrite Channels)" if use_append_reducer else "Direct Key Overwrite"
        }
        self.history.append(step_record)
        return step_record

    def get_audit_trail(self) -> pd.DataFrame:
        rows = []
        for h in self.history:
            rows.append({
                "Superstep": h["superstep"],
                "Node": h["node"],
                "Delta Keys": ", ".join(h["delta"].keys()),
                "Messages Count": len(h["snapshot"].get("messages", [])),
                "Iteration": h["snapshot"].get("iteration_count", 0),
                "Status": h["snapshot"].get("status", "N/A"),
                "Reducer Mode": h["reducer_action"]
            })
        return pd.DataFrame(rows)


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: State Lifecycle and Memory Topology in LangGraph", divider="rainbow")

st.markdown("""
In LangGraph, **State** is the centralized, immutable-per-superstep data structure that serves as the single source
of truth across the entire execution graph. Unlike traditional LangChain chains or linear DAGs where each node receives
only the output of its predecessor, LangGraph nodes read from a unified state snapshot and emit **partial dictionary updates**
that are merged back into state via deterministic channel reducers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StateArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_input {
            label="1. Ingestion Boundary";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nUser Input / Initial Payload", fillcolor="#1E293B", shape=ellipse];
        }

        subgraph cluster_bsp_superstep_1 {
            label="2. Superstep 1: Read and Compute Phase";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            
            State_Snapshot_0 [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#1E293B"><b>State Snapshot (t=0)</b></td></tr>
                <tr><td>messages</td><td>[HumanMessage('Query')]</td></tr>
                <tr><td>iteration</td><td>0</td></tr>
                <tr><td>status</td><td>'INITIALIZED'</td></tr>
            </table>>, shape=plaintext];
            
            Node_A [label="Node A (Analyze)\\nReads Snapshot (t=0)\\nReturns delta: {'status': 'ANALYZING'}", fillcolor="#334155"];
            Node_B [label="Node B (Retrieve)\\nReads Snapshot (t=0)\\nReturns delta: {'context': ['Doc1']}", fillcolor="#334155"];
        }

        subgraph cluster_reducer_channel {
            label="3. Channel Reducer and Barrier Synchronization";
            style="filled";
            color="#1E293B";
            fontcolor="#A855F7";
            
            Reducer_Hub [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#4C1D95"><b>LangGraph Channel Reducers</b></td></tr>
                <tr><td>Overwrite Channel</td><td>status: 'ANALYZING' (Last write wins)</td></tr>
                <tr><td>Annotated Append Channel</td><td>context: [] + ['Doc1'] = ['Doc1']</td></tr>
                <tr><td>Checkpointer Persist</td><td>Write state snapshot to Checkpointer</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_bsp_superstep_2 {
            label="4. Superstep 2: Downstream Execution";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            
            State_Snapshot_1 [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#1E293B"><b>State Snapshot (t=1)</b></td></tr>
                <tr><td>messages</td><td>[HumanMessage('Query')]</td></tr>
                <tr><td>context</td><td>['Doc1']</td></tr>
                <tr><td>iteration</td><td>1</td></tr>
                <tr><td>status</td><td>'ANALYZING'</td></tr>
            </table>>, shape=plaintext];
            
            Node_Generate [label="Node C (Synthesize Answer)\\nReads Consolidated State (t=1)", fillcolor="#334155"];
        }

        subgraph cluster_output {
            label="5. Terminal State";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            END [label="END\\nFinal State Exported", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> State_Snapshot_0 [label="StateGraph Schema Ingestion"];
        State_Snapshot_0 -> Node_A [label="Pass State (Frozen View)"];
        State_Snapshot_0 -> Node_B [label="Pass State (Frozen View)"];
        Node_A -> Reducer_Hub [label="Emit Partial Delta A"];
        Node_B -> Reducer_Hub [label="Emit Partial Delta B"];
        Reducer_Hub -> State_Snapshot_1 [label="Barrier Resolve & Checkpoint"];
        State_Snapshot_1 -> Node_Generate [label="Next Superstep Input"];
        Node_Generate -> END [label="Terminal Transition"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Modern Engineering Perspective: Actor Model & Bulk Synchronous Parallelism (BSP)
LangGraph adopts the **Bulk Synchronous Parallel (BSP)** computational paradigm. In BSP:
1. **Parallel Computation**: All active nodes in a superstep execute concurrently, evaluating solely against a frozen snapshot of state at time `t`.
2. **Barrier Synchronization**: All nodes must finish their compute cycle before state is mutated.
3. **Communication & Reducers**: Node outputs (deltas) are gathered and processed by predefined channel reducer functions to construct the state snapshot for superstep `t+1`.
""")

# ==============================================================================
# SECTION 2: WHAT IS GRAPH STATE?
# ==============================================================================
st.header("2. What is Graph State?", divider="rainbow")

st.markdown("""
In LangGraph, **Graph State** represents the shared schema and runtime storage allocated to a single graph execution thread.
It defines the complete universe of data accessible to nodes, routers, and tools during graph evaluation.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    #### Key Characteristics of Graph State
    - **Centralized Memory**: Every node accesses the exact same schema. There is no hidden node-local private memory unless explicitly nested.
    - **Strongly Typed**: Defined via Python `TypedDict`, `@dataclass`, or Pydantic `BaseModel`.
    - **Channel-Driven**: Every field in state behaves as a communication channel with its own update rule.
    - **Checkpointable**: State can be serialized to disk, Redis, or SQLite at every superstep, allowing seamless pause/resume and human-in-the-loop inspection.
    """)
with col2:
    st.markdown("""
    #### State vs Legacy LangChain Memory
    | Capability | Legacy LangChain Memory | LangGraph Graph State |
    | :--- | :--- | :--- |
    | **Topology** | Hidden string buffer attached to chains | Explicit, structured schema for entire graph |
    | **State Access** | Automatic prompt formatting | Direct, typed attribute access in node code |
    | **Concurrency** | Unsafe; prone to race conditions | BSP barrier isolation with reducer merge |
    | **Persistence** | Ad-hoc memory classes | First-class `BaseCheckpointSaver` threads |
    """)

# ==============================================================================
# SECTION 3: SHARED STATE ACROSS NODES
# ==============================================================================
st.header("3. Shared State Across Nodes", divider="rainbow")

st.markdown("""
Because state is shared across all nodes in the graph, parallel nodes can collaborate without directly communicating with each other.
When a graph branches into parallel paths (fan-out), each branch receives the exact same state snapshot.
""")

# Contextual diagram for shared state
shared_state_dot = """
digraph SharedStateTopology {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Central_State [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#1E293B">
        <tr><td colspan="2" bgcolor="#0F172A"><b>Central State Store</b></td></tr>
        <tr><td>query</td><td>'Analyze Q3 Revenue'</td></tr>
        <tr><td>messages</td><td>[HumanMessage(...)]</td></tr>
        <tr><td>financial_data</td><td>None</td></tr>
        <tr><td>sentiment_score</td><td>None</td></tr>
    </table>>, shape=plaintext];

    Node_Finance [label="Finance Extractor Node\\nReads: query\\nWrites: {'financial_data': 1420000}", fillcolor="#334155"];
    Node_Sentiment [label="Sentiment Analyzer Node\\nReads: query\\nWrites: {'sentiment_score': 0.88}", fillcolor="#334155"];
    Node_Aggregator [label="Synthesis Aggregator Node\\nReads: financial_data and sentiment_score", fillcolor="#1E3A8A"];

    Central_State -> Node_Finance [label="Reads snapshot"];
    Central_State -> Node_Sentiment [label="Reads snapshot"];
    Node_Finance -> Central_State [label="Emits delta"];
    Node_Sentiment -> Central_State [label="Emits delta"];
    Central_State -> Node_Aggregator [label="Reads consolidated t+1 state"];
}
"""
with st.expander("📊 View Shared State Topology Diagram", expanded=True):
    st.graphviz_chart(shared_state_dot, use_container_width=True)

st.markdown("""
> **Key Architectural Insight**: Notice that `Node_Finance` and `Node_Sentiment` do not pass variables to each other.
> They are completely decoupled. They write their partial updates to `Central_State`, and downstream nodes consume the combined result.
""")

# ==============================================================================
# SECTION 4: STATE SCHEMAS AND STRUCTURAL CONTRACTS
# ==============================================================================
st.header("4. State Schemas and Structural Contracts", divider="rainbow")

st.markdown("""
A state schema defines the keys, value types, and update semantics for every piece of information stored in the graph.
In LangGraph, schemas are specified using Python's typing ecosystem.
""")

code_schema_example = """from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

# Define the State Schema Contract
class AgentState(TypedDict):
    # Channel 1: Overwrite channel (Default behavior: incoming value replaces previous value)
    query: str
    iteration: int
    status: str

    # Channel 2: Reducer channel (Annotated with a binary reducer function)
    # operator.add ensures that returned lists are appended, not overwritten!
    messages: Annotated[List[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[dict], operator.add]
"""
st.code(code_schema_example, language="python")

st.markdown("""
#### Channel Update Semantics
1. **Default Channels (No Reducer)**: If no reducer is specified, LangGraph defaults to **Overwrite** semantics. The new value emitted by the node completely replaces the existing key in state.
2. **Annotated Channels (With Reducer)**: When annotated using `Annotated[Type, ReducerFunction]`, LangGraph executes `ReducerFunction(old_value, delta_value)` to calculate the new state value.
""")

# ==============================================================================
# SECTION 5: STATE UPDATES AND IMMUTABILITY SEMANTICS
# ==============================================================================
st.header("5. State Updates and Immutability Semantics", divider="rainbow")

st.markdown("""
A foundational design rule of LangGraph is that **nodes must never mutate the incoming state dictionary in place**.
Nodes should treat incoming state as read-only, perform their computation, and return a dictionary of changes.
""")

col_anti, col_best = st.columns(2)
with col_anti:
    st.error("❌ Anti-Pattern: In-Place Mutation")
    st.code("""def bad_node(state: AgentState) -> dict:
    # MUTATING STATE IN-PLACE BREAKS BSP AND CHECKPOINTING!
    state["messages"].append("New message")
    state["iteration"] += 1
    return state  # Returns mutated instance
""", language="python")
    st.caption("Mutating in-place causes data corruption in parallel branches and destroys historical checkpoint validity.")

with col_best:
    st.success("✅ Recommended: Partial Delta Return")
    st.code("""def good_node(state: AgentState) -> dict:
    # Return ONLY the keys that were modified
    new_message = "New message"
    return {
        "messages": [new_message], # Merged via reducer
        "iteration": state["iteration"] + 1
    }
""", language="python")
    st.caption("LangGraph applies channel reducers cleanly to calculate the next superstep state snapshot.")

# ==============================================================================
# SECTION 6: STATE LIFECYCLE ACROSS SUPERSTEPS
# ==============================================================================
st.header("6. State Lifecycle Across Supersteps", divider="rainbow")

st.markdown("""
The state lifecycle in LangGraph follows a rigorous three-phase execution rhythm per superstep:
""")

lifecycle_dot = """
digraph StateLifecycle {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    P1 [label="Phase 1: Read Phase\\nRuntime creates frozen snapshot S_t.\\nAll active nodes in superstep read S_t.", fillcolor="#1E293B"];
    P2 [label="Phase 2: Compute Phase\\nNodes run business logic in isolation.\\nNo state mutations are visible to other nodes.", fillcolor="#334155"];
    P3 [label="Phase 3: Barrier & Reducer Phase\\nNodes emit delta dictionaries.\\nReducers merge deltas into S_{t+1}.\\nCheckpointer persists snapshot.", fillcolor="#1E3A8A"];
    P4 [label="Next Superstep or END\\nIf graph has active nodes -> repeat Phase 1\\nElse return final state to caller.", fillcolor="#0F172A"];

    P1 -> P2 [label="Dispatch nodes"];
    P2 -> P3 [label="Collect deltas"];
    P3 -> P4 [label="State t+1 ready"];
}
"""
with st.expander("🔄 View State Lifecycle Three-Phase Flow", expanded=True):
    st.graphviz_chart(lifecycle_dot, use_container_width=True)

st.markdown("""
### Why Superstep Snapshots Matter
- **Deterministic Replay**: You can reproduce the exact inputs and outputs of any superstep.
- **Fault Tolerance**: If node `N` crashes during superstep `t`, LangGraph can restart from the `t` snapshot without rerunning earlier steps.
- **Time-Travel Debugging**: Developers can inspect, edit, and fork execution from any prior superstep.
""")

# ==============================================================================
# SECTION 7: READING STATE IN NODE FUNCTIONS
# ==============================================================================
st.header("7. Reading State in Node Functions", divider="rainbow")

st.markdown("""
Node functions in LangGraph follow a clean, functional signature:
```python
def node_name(state: StateType, config: RunnableConfig = None) -> Dict[str, Any]:
    ...
```
""")

st.markdown("""
#### Best Practices for Reading State
1. **Defensive Key Access**: Use `.get("key", default)` if fields might be omitted in early supersteps.
2. **Schema Type Annotations**: Always annotate the `state` parameter with your custom `TypedDict` or Pydantic class to enable full IDE autocomplete and type checking.
3. **Reading Metadata via Config**: Use `config.get("configurable", {})` to access session IDs, user credentials, or model overrides without polluting the core state schema.
""")

code_read_demo = """from typing import TypedDict, Optional
from langchain_core.runnables import RunnableConfig

class ResearchState(TypedDict):
    topic: str
    documents: list[str]
    max_loops: int
    loop_count: int

def evaluator_node(state: ResearchState, config: RunnableConfig) -> dict:
    # 1. Clean typed reading
    current_docs = state["documents"]
    current_loop = state["loop_count"]
    limit = state.get("max_loops", 3)

    # 2. Extract configuration metadata
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 3. Decision logic
    if len(current_docs) >= 5 or current_loop >= limit:
        return {"loop_count": current_loop + 1}
    
    return {"loop_count": current_loop + 1}
"""
st.code(code_read_demo, language="python")

# ==============================================================================
# SECTION 8: UPDATING STATE VIA PARTIAL DICTIONARIES
# ==============================================================================
st.header("8. Updating State via Partial Dictionaries", divider="rainbow")

st.markdown("""
LangGraph is designed around **partial state emission**. A node only needs to return the specific keys it intends to change.
Keys that are not included in the returned dictionary retain their previous values automatically.
""")

col_u1, col_u2 = st.columns(2)
with col_u1:
    st.markdown("""
    #### Partial Return Behaviors
    - **Empty Return `{}`**: The node executed side effects (e.g. logging, analytics) but makes zero state modifications. State passes through unchanged.
    - **Single Key Return `{"status": "DONE"}`**: Only `status` is updated. All other state channels (`messages`, `context`, etc.) remain identical.
    - **Reducer Key Return `{"messages": [new_msg]}`**: The list containing `new_msg` is passed to the reducer function (e.g. `add_messages`), which merges it into the existing list.
    """)
with col_u2:
    st.markdown("""
    #### Common State Update Errors
    - **Returning Non-Dict**: Returning a raw string, integer, or `None` will trigger a runtime `ValueError`. Nodes must return a dictionary or `Command`.
    - **Returning Deleted Keys**: Setting `{"key": None}` does not delete the key from state; it sets the value to `None`.
    - **Conflicting Parallel Writes**: If two parallel nodes return different values for an overwrite key without a reducer, an exception is raised unless ordered.
    """)

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: End-to-End Typed Graph State Workflow", divider="rainbow")

st.markdown("""
Below is a complete, production-grade implementation of a Multi-Channel StateGraph.
It demonstrates typed schemas, reducer channels (`operator.add`), defensive state reading, and partial delta returns.
""")

prod_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Step 1: Define the Structured State Contract
class ProductionAgentState(TypedDict):
    # Channel 1: Overwrite channels
    session_id: str
    user_query: str
    intent: str
    confidence: float
    current_step: int
    
    # Channel 2: Append Reducer channels
    event_log: Annotated[List[str], operator.add]
    retrieved_chunks: Annotated[List[Dict[str, Any]], operator.add]

# Step 2: Implement Pure Node Functions Returning Partial Deltas
def classify_intent_node(state: ProductionAgentState) -> Dict[str, Any]:
    query = state["user_query"]
    # Compute intent classification
    detected_intent = "TECHNICAL_INQUIRY" if "code" in query.lower() else "GENERAL_FAQ"
    
    return {
        "intent": detected_intent,
        "confidence": 0.95,
        "current_step": state["current_step"] + 1,
        "event_log": [f"Intent classified as {detected_intent} (conf=0.95)"]
    }

def retrieval_node(state: ProductionAgentState) -> Dict[str, Any]:
    intent = state["intent"]
    mock_chunks = [
        {"id": "doc_101", "content": f"Verified documentation for {intent}", "score": 0.89}
    ]
    return {
        "retrieved_chunks": mock_chunks,
        "current_step": state["current_step"] + 1,
        "event_log": [f"Retrieved {len(mock_chunks)} knowledge chunks"]
    }

def synthesis_node(state: ProductionAgentState) -> Dict[str, Any]:
    chunks = state["retrieved_chunks"]
    summary = f"Synthesized answer based on {len(chunks)} sources."
    return {
        "current_step": state["current_step"] + 1,
        "event_log": [f"Synthesis complete: {summary}"]
    }

# Step 3: Build and Compile StateGraph
def build_production_graph():
    builder = StateGraph(ProductionAgentState)
    
    # Register Nodes
    builder.add_node("classify", classify_intent_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("synthesize", synthesis_node)
    
    # Define Transitions
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()

# Example Invocation
if __name__ == "__main__":
    app = build_production_graph()
    initial_input = {
        "session_id": "sess_001",
        "user_query": "How do I optimize code for LangGraph state?",
        "intent": "UNKNOWN",
        "confidence": 0.0,
        "current_step": 0,
        "event_log": ["Graph execution started."],
        "retrieved_chunks": []
    }
    result = app.invoke(initial_input)
    print("Final State Event Log:", result["event_log"])
'''
st.code(prod_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: State Mutation and Superstep Visualizer", divider="rainbow")

st.markdown("""
Test how state evolves across supersteps in real-time. Configure your initial state payload, choose node operations,
and observe how partial deltas are merged via channel reducers.
""")

if "state_sim" not in st.session_state:
    st.session_state.state_sim = StateLifecycleEngine("Explain State Management in LangGraph", "session_alpha_1")

col_wb_ctrl, col_wb_view = st.columns([1, 1.2])

with col_wb_ctrl:
    st.subheader("Superstep Action Dispatcher")
    selected_node = st.selectbox(
        "Select Node to Execute",
        ["Node_A_Analyzer", "Node_B_Retriever", "Node_C_Synthesizer", "Custom_Node"]
    )
    
    delta_msg = st.text_input("New Message Delta", value="Generated intermediate analysis summary.")
    delta_iter = st.number_input("Iteration Increment Delta", min_value=1, max_value=5, value=1)
    delta_status = st.selectbox("Status Channel Delta", ["ANALYZING", "FETCHING", "COMPLETED", "ERROR"])
    use_append = st.checkbox("Enable Append Reducer for Messages Channel", value=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Execute Superstep", use_container_width=True):
            delta = {
                "messages": [f"[{selected_node}] {delta_msg}"],
                "iteration_count": delta_iter,
                "status": delta_status
            }
            st.session_state.state_sim.execute_node_step(selected_node, delta, use_append_reducer=use_append)
            st.success(f"Executed {selected_node} successfully!")
            
    with col_btn2:
        if st.button("🔄 Reset State Engine", use_container_width=True):
            st.session_state.state_sim = StateLifecycleEngine("Explain State Management in LangGraph", "session_alpha_1")
            st.info("State reset to Superstep 0.")

with col_wb_view:
    st.subheader("Current State Snapshot (Active Thread)")
    curr = st.session_state.state_sim.current_state
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Superstep", len(st.session_state.state_sim.history) - 1)
    c_m2.metric("Iteration Count", curr.get("iteration_count", 0))
    c_m3.metric("Status", curr.get("status", "N/A"))
    
    st.markdown("**Messages Channel (`Annotated[list, operator.add]`):**")
    for idx, msg in enumerate(curr.get("messages", [])):
        st.markdown(f"- `{idx}`: {msg}")
        
    st.markdown("**Raw State Dictionary:**")
    st.json(curr)

st.markdown("### Superstep Audit Trail & Channel Reducer Telemetry")
df_audit = st.session_state.state_sim.get_audit_trail()
st.dataframe(df_audit, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does LangGraph handle concurrent node updates to the same state key?", expanded=False):
    st.markdown("""
    **Answer**:
    In LangGraph's Bulk Synchronous Parallel (BSP) model, when two or more parallel nodes return updates to the same key:
    1. If the key is defined with an **Annotated Reducer function** (such as `Annotated[list, operator.add]`), LangGraph invokes the reducer sequentially on the incoming deltas in a deterministic order, accumulating the results.
    2. If the key is a **Default Overwrite channel** without a reducer, LangGraph will raise an `InvalidUpdateError` due to a conflicting write, preventing non-deterministic race conditions from silently corrupting your agent state.
    """)

with st.expander("Q2: Why is in-place mutation of lists in state considered a critical anti-pattern in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    In-place mutation (e.g. `state['messages'].append(msg)`) modifies the underlying Python object directly in memory.
    This creates three major bugs:
    1. **Bypasses Reducers**: The channel reducer is never invoked because the graph cannot compute the delta between the old and new object references.
    2. **Corrupts Parallel Branches**: Parallel sibling nodes reading the snapshot will see partial mutations made by another thread mid-superstep.
    3. **Destroys Checkpoint Time-Travel**: The checkpointer stores references to the mutated object, corrupting historical state snapshots.
    """)

with st.expander("Q3: What is the semantic difference between returning an empty dictionary {} vs returning None in a node?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Returning `{}`**: Explicitly signals to LangGraph that the node has successfully finished its compute cycle and has produced **zero state mutations**. State channels pass through to the next superstep unmodified.
    - **Returning `None`**: In modern LangGraph versions, returning `None` or failing to return a dictionary raises a runtime `ValueError: Expected dict, got NoneType`. LangGraph requires an explicit mapping of state deltas or an orchestration `Command`.
    """)

with st.expander("Q4: How does LangGraph checkpoint state between supersteps, and what serialization format is used?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph utilizes checkpointer classes implementing `BaseCheckpointSaver` (e.g. `MemorySaver`, `SqliteSaver`, `AsyncPostgresSaver`).
    At the conclusion of each superstep's reducer phase:
    1. The consolidated state dictionary is captured alongside superstep metadata, channel versions, and parent checkpoint IDs.
    2. It is serialized using `SerializerProtocol` (typically JSON or msgpack with custom serializers for LangChain `BaseMessage` and Pydantic objects).
    3. The checkpoint is keyed by `thread_id` and `checkpoint_id`, enabling state restoration and human-in-the-loop branch forks.
    """)

with st.expander("Q5: What is the role of typing.Annotated in state schemas, and how does it differ from standard Python type annotations?", expanded=False):
    st.markdown("""
    **Answer**:
    Standard Python type annotations (e.g. `messages: list[str]`) are static type hints used solely by linters (mypy) and IDEs.
    In LangGraph, `Annotated[Type, Metadata/Reducer]` leverages Python 3.9+ type metadata to attach runtime behaviors to state channels:
    ```python
    messages: Annotated[list[BaseMessage], operator.add]
    ```
    LangGraph inspects the second argument of `Annotated` during graph compilation. If a callable function is provided, LangGraph registers it as the channel's merge reducer.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("🚨 Trap 1 (Fatal Error): Mutating State In-Place Leading to Silent Reducer Bypasses")
    st.markdown("""
    *Mechanism*: Directly calling `.append()`, `.extend()`, or `dict.update()` on incoming state objects inside a node function.  
    *Consequence*: Reducers fail to trigger, parallel nodes experience dirty reads, and time-travel rollbacks fail.  
    *Fix*: Always construct new collections or return partial deltas `{"messages": [new_item]}`.
    """)

with col_trap2:
    st.error("🚨 Trap 2 (Fatal Error): Conflicting Parallel Overwrites on Un-Reduced Channels")
    st.markdown("""
    *Mechanism*: Branching into parallel nodes where both nodes return updates to an unannotated scalar channel (e.g. `{"status": "DONE"}`).  
    *Consequence*: LangGraph aborts with `InvalidUpdateError` due to ambiguous concurrent assignment.  
    *Fix*: Use custom reducer functions or design separate keys for each parallel branch (e.g. `branch_a_status`, `branch_b_status`).
    """)

col_trap3, col_trap4 = st.columns(2)
with col_trap3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-allocating Large Binary Payloads in State")
    st.markdown("""
    *Mechanism*: Storing multi-megabyte PDF texts, raw image bytearrays, or large embedding matrices in the central state dictionary.  
    *Consequence*: Every superstep serializes the entire state to the checkpointer, causing exponential memory growth and database latency spikes.  
    *Fix*: Store large payloads in object storage (S3/GCS) and store lightweight URIs or pointers in graph state.
    """)

with col_trap4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Relying on Python Dict Ordering in Multithreaded Async Runtimes")
    st.markdown("""
    *Mechanism*: Assuming sequential execution order when multiple independent nodes are triggered in parallel supersteps.  
    *Consequence*: Flaky race conditions where non-associative reducers yield different final states depending on thread scheduling.  
    *Fix*: Ensure channel reducers are strictly associative and commutative, or enforce sequential edges.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Graph State**\\nThe central typed data structure representing the current execution context and memory across all nodes.")
with col_fc2:
    st.info("**Superstep**\\nA discrete execution epoch where active nodes read a frozen snapshot, compute deltas, and barrier-sync.")
with col_fc3:
    st.info("**Channel Reducer**\\nA binary function `f(current, delta) -> updated` that defines how partial updates are merged into state.")
with col_fc4:
    st.info("**Checkpointer**\\nA persistence adapter that saves state snapshots at every superstep keyed by thread ID.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand that LangGraph uses the Bulk Synchronous Parallel (BSP) model for state evaluation.", value=True)
c2 = st.checkbox("I know that nodes must never mutate state in-place and must return partial delta dictionaries.", value=True)
c3 = st.checkbox("I can differentiate between default overwrite channels and Annotated reducer channels.", value=True)
c4 = st.checkbox("I understand how checkpointers persist state at superstep boundaries for time-travel debugging.", value=True)
c5 = st.checkbox("I know how to structure strongly-typed schemas using TypedDict, Dataclasses, or Pydantic.", value=True)
c6 = st.checkbox("I can diagnose and prevent concurrent write collisions on un-reduced state channels.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 4.1 'Understanding State' with full architectural and practical comprehension!")
