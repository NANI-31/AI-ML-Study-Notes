import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="6.1 Normal Edges - LangGraph Curriculum",
    page_icon="➡️",
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
# MODULE-SCOPE STATIC EDGE TOPOLOGY ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class StaticEdgeTopologyEngine:
    """
    Simulates normal edge control flow:
    Sequential transitions, parallel fan-out (multiple outgoing),
    and barrier fan-in (multiple incoming edges).
    """
    def __init__(self):
        self.nodes = ["START", "ingest", "branch_a", "branch_b", "aggregate", "END"]
        self.edges = [
            ("START", "ingest"),
            ("ingest", "branch_a"),
            ("ingest", "branch_b"), # Fan-out
            ("branch_a", "aggregate"), # Fan-in
            ("branch_b", "aggregate"), # Fan-in
            ("aggregate", "END")
        ]

    def trace_execution(self, initial_query: str) -> List[Dict[str, Any]]:
        """
        Traces step-by-step supersteps according to static edge definitions.
        """
        steps = []
        # Superstep 0: START -> ingest
        steps.append({
            "superstep": 1,
            "phase": "Ingestion",
            "active_nodes": ["ingest"],
            "state_delta": {"raw_query": initial_query, "status": "INGESTED"},
            "edge_transition": "START -> ingest"
        })
        # Superstep 1: Fan-out (branch_a and branch_b run in parallel)
        steps.append({
            "superstep": 2,
            "phase": "Parallel Fan-Out",
            "active_nodes": ["branch_a (Sentiment)", "branch_b (Entities)"],
            "state_delta": {"sentiment": "POSITIVE", "entities": ["LangGraph", "Edges"]},
            "edge_transition": "ingest -> [branch_a, branch_b]"
        })
        # Superstep 2: Fan-in (aggregate waits for both branches)
        steps.append({
            "superstep": 3,
            "phase": "Barrier Fan-In",
            "active_nodes": ["aggregate"],
            "state_delta": {"summary": "Positive sentiment for entities [LangGraph, Edges]"},
            "edge_transition": "[branch_a, branch_b] -> aggregate"
        })
        # Superstep 3: aggregate -> END
        steps.append({
            "superstep": 4,
            "phase": "Completion",
            "active_nodes": ["END"],
            "state_delta": {"status": "COMPLETED"},
            "edge_transition": "aggregate -> END"
        })
        return steps


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Directed Graph Control Flow and Static Edge Topology", divider="rainbow")

st.markdown("""
In LangGraph, **Normal (Static) Edges** define fixed, deterministic transitions between nodes.
Unlike dynamic routers that inspect state values at runtime, static edges declare the invariant structural
skeleton of your application: which node executes first, how tasks fan out into parallel branches,
and how parallel threads join together before termination.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    normal_edge_blueprint_dot = """
    digraph NormalEdgeArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress Transition";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nVirtual Ingress Marker", fillcolor="#1E293B", shape=ellipse];
            Node_Ingest [label="Ingest Node\\nNormal edge from START", fillcolor="#0284C7"];
        }

        subgraph cluster_parallel_fanout {
            label="2. Parallel Fan-Out (Multiple Outgoing Edges)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Node_Branch1 [label="Worker Node Alpha\\nRuns in parallel (BSP superstep)", fillcolor="#334155"];
            Node_Branch2 [label="Worker Node Beta\\nRuns in parallel (BSP superstep)", fillcolor="#334155"];
        }

        subgraph cluster_barrier_fanin {
            label="3. Barrier Fan-In (Multiple Incoming Edges)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Node_Join [label="Aggregator Node\\nWaits for Alpha and Beta to complete", fillcolor="#059669"];
        }

        subgraph cluster_egress {
            label="4. Egress Transition";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            END [label="END\\nVirtual Egress Marker", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Node_Ingest [label="builder.add_edge(START, 'ingest')"];
        Node_Ingest -> Node_Branch1 [label="builder.add_edge('ingest', 'alpha')"];
        Node_Ingest -> Node_Branch2 [label="builder.add_edge('ingest', 'beta')"];
        Node_Branch1 -> Node_Join [label="builder.add_edge('alpha', 'join')"];
        Node_Branch2 -> Node_Join [label="builder.add_edge('beta', 'join')"];
        Node_Join -> END [label="builder.add_edge('join', END)"];
    }
    """
    st.graphviz_chart(normal_edge_blueprint_dot, use_container_width=True)

st.markdown("""
### Static Determinism in Multi-Agent Graphs
Normal edges are registered with `builder.add_edge(source, destination)`.
Because they do not evaluate runtime conditions, LangGraph can statically validate graph topology during compilation,
detecting unreachable orphan nodes, cycles without exit paths, and missing START/END transitions at compile time.
""")

# ==============================================================================
# SECTION 2: NODE-TO-NODE EDGES
# ==============================================================================
st.header("2. Node-to-Node Edges: Deterministic Transitions Between Vertices", divider="rainbow")

st.markdown("""
A node-to-node edge specifies that when node `A` completes its compute superstep, node `B` must execute in the subsequent superstep.
""")

code_n2n = """from langgraph.graph import StateGraph

builder = StateGraph(MyState)
builder.add_node("step_a", step_a_function)
builder.add_node("step_b", step_b_function)

# Deterministic transition from step_a to step_b
builder.add_edge("step_a", "step_b")
"""
st.code(code_n2n, language="python")

st.markdown("""
#### Invariants of Node-to-Node Edges:
- **Source Node Must Exist**: Calling `add_edge("unknown", "b")` raises `ValueError: Node 'unknown' not found`.
- **Destination Node Must Exist**: Calling `add_edge("a", "unknown")` raises `ValueError: Node 'unknown' not found`.
- **Zero Runtime Evaluation**: The edge is always traversed unconditionally whenever the source node completes.
""")

# ==============================================================================
# SECTION 3: SEQUENTIAL WORKFLOWS
# ==============================================================================
st.header("3. Sequential Workflows: Linear Chain Execution Pipelines", divider="rainbow")

st.markdown("""
When multiple normal edges connect nodes in a single line, they create a **Sequential Pipeline**.
This replicates classical LangChain `RunnableSequence` workflows (`prompt | model | parser`) while unlocking
state persistence, checkpoint rollbacks, and interactive streaming.
""")

col_seq1, col_seq2 = st.columns(2)
with col_seq1:
    st.markdown("""
    #### Architectural Benefits of LangGraph Sequences
    - **Step-by-Step Checkpointing**: If node 4 fails, you can restart from node 3 without re-running nodes 1 and 2.
    - **Granular Tracing**: Every individual node invocation appears as an isolated span in LangSmith.
    - **Streaming Updates**: Stream intermediate outputs using `app.stream(..., stream_mode="updates")`.
    """)
with col_seq2:
    code_linear_chain = """# Sequential pipeline: A -> B -> C
builder.add_edge(START, "load_document")
builder.add_edge("load_document", "extract_entities")
builder.add_edge("extract_entities", "generate_summary")
builder.add_edge("generate_summary", END)
"""
    st.code(code_linear_chain, language="python")

# ==============================================================================
# SECTION 4: START EDGES
# ==============================================================================
st.header("4. START Edges: Ingress Control and Graph Entry Points", divider="rainbow")

st.markdown("""
The `START` constant (`from langgraph.graph import START`) represents the virtual entry boundary of the graph.
Every graph must have at least one outgoing edge originating from `START`.
""")

col_st1, col_st2 = st.columns(2)
with col_st1:
    st.markdown("""
    #### Single Entry vs Fan-Out Entry
    - **Single Entry**: Directs all incoming graph calls to a designated ingress router or validation node.
    - **Fan-Out Entry**: Calling `add_edge(START, "worker_1")` and `add_edge(START, "worker_2")` causes both workers to execute immediately in parallel upon graph invocation!
    """)
with col_st2:
    st.code("""from langgraph.graph import StateGraph, START

# 1. Single entry point
builder.add_edge(START, "input_sanitizer")

# 2. Parallel entry fan-out
builder.add_edge(START, "audit_logger")
builder.add_edge(START, "primary_pipeline")
""", language="python")

# ==============================================================================
# SECTION 5: END EDGES
# ==============================================================================
st.header("5. END Edges: Terminal Boundaries and Graph Completion", divider="rainbow")

st.markdown("""
The `END` constant (`from langgraph.graph import END`) signifies the virtual terminal exit boundary of the graph.
When all active paths reach `END`, graph execution finishes, the final state snapshot is projected through `output_schema`,
and execution control returns to the caller.
""")

st.markdown("""
#### Crucial Compilation Rule:
Every reachable path in a graph must eventually terminate at `END` (or an interrupt).
If a node has no outgoing edges and does not connect to `END`, LangGraph raises a compile-time warning or validation error.
```python
builder.add_edge("final_formatter", END)
```
""")

# ==============================================================================
# SECTION 6: MULTIPLE OUTGOING EDGES
# ==============================================================================
st.header("6. Multiple Outgoing Edges: Parallel Fan-Out Execution", divider="rainbow")

st.markdown("""
When a single node has multiple outgoing normal edges, LangGraph executes a **Parallel Fan-Out**.
In the subsequent superstep, all destination nodes are dispatched concurrently in the Bulk Synchronous Parallel (BSP) model.
""")

fanout_dot = """
digraph FanOut {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Dispatcher [label="Document Dispatcher Node", fillcolor="#1E293B"];
    Worker1 [label="Worker 1: Extract Dates", fillcolor="#334155"];
    Worker2 [label="Worker 2: Extract Legal Entities", fillcolor="#334155"];
    Worker3 [label="Worker 3: Calculate Sentiment", fillcolor="#334155"];

    Dispatcher -> Worker1 [label="Edge 1"];
    Dispatcher -> Worker2 [label="Edge 2"];
    Dispatcher -> Worker3 [label="Edge 3"];
}
"""
with st.expander("⚡ View Parallel Fan-Out Topology", expanded=True):
    st.graphviz_chart(fanout_dot, use_container_width=True)

st.markdown("""
```python
# Dispatches Worker 1, Worker 2, and Worker 3 in parallel!
builder.add_edge("dispatcher", "worker_1")
builder.add_edge("dispatcher", "worker_2")
builder.add_edge("dispatcher", "worker_3")
```
""")

# ==============================================================================
# SECTION 7: MULTIPLE INCOMING EDGES
# ==============================================================================
st.header("7. Multiple Incoming Edges: Parallel Fan-In and Barrier Synchronization", divider="rainbow")

st.markdown("""
When multiple nodes have outgoing edges pointing to the same downstream node, LangGraph forms a **Parallel Fan-In** (Join).
Under LangGraph's BSP model, the downstream node only executes after **all** incoming active predecessor branches have completed!
""")

col_fi1, col_fi2 = st.columns(2)
with col_fi1:
    st.markdown("""
    #### Barrier Synchronization Mechanics
    1. **Wait Barrier**: LangGraph runtime holds `Aggregator Node` until all parallel branches emit their deltas.
    2. **Reducer Consolidation**: The channel reducers merge all deltas from the parallel branches.
    3. **Consolidated Ingestion**: `Aggregator Node` receives the unified state containing updates from all branches.
    """)
with col_fi2:
    st.code("""# Parallel branches join at 'aggregator'
builder.add_edge("worker_1", "aggregator")
builder.add_edge("worker_2", "aggregator")
builder.add_edge("worker_3", "aggregator")
builder.add_edge("aggregator", END)
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Branch Static Graph Topology", divider="rainbow")

st.markdown("""
Below is a complete, runnable script demonstrating static normal edges with sequential ingress,
parallel fan-out, barrier fan-in synchronization, and termination.
""")

prod_static_edges_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Step 1: Define Multi-Branch State
class FanOutState(TypedDict):
    input_text: str
    cleaned_text: str
    sentiment: str
    entities: List[str]
    audit_trace: Annotated[List[str], operator.add]
    final_report: str

# Node Functions
def sanitize_node(state: FanOutState) -> Dict[str, Any]:
    cleaned = state["input_text"].strip()
    return {
        "cleaned_text": cleaned,
        "audit_trace": ["Sanitized incoming text."]
    }

def sentiment_worker_node(state: FanOutState) -> Dict[str, Any]:
    # Runs in parallel with entity_worker_node
    is_pos = "great" in state["cleaned_text"].lower() or "good" in state["cleaned_text"].lower()
    return {
        "sentiment": "POSITIVE" if is_pos else "NEUTRAL",
        "audit_trace": ["Sentiment analysis completed."]
    }

def entity_worker_node(state: FanOutState) -> Dict[str, Any]:
    # Runs in parallel with sentiment_worker_node
    words = [w for w in state["cleaned_text"].split() if w.istitle()]
    return {
        "entities": words,
        "audit_trace": [f"Extracted {len(words)} entities."]
    }

def aggregator_node(state: FanOutState) -> Dict[str, Any]:
    # Barrier: Executes ONLY after both workers finish!
    report = f"Report: Sentiment={state['sentiment']} | Entities={state['entities']}"
    return {
        "final_report": report,
        "audit_trace": ["Consolidated final report."]
    }

# Step 2: Build Static Edge Topology
def build_fanout_graph():
    builder = StateGraph(FanOutState)
    
    # Register Nodes
    builder.add_node("sanitize", sanitize_node)
    builder.add_node("sentiment_worker", sentiment_worker_node)
    builder.add_node("entity_worker", entity_worker_node)
    builder.add_node("aggregator", aggregator_node)
    
    # 1. Sequential Ingress Edge
    builder.add_edge(START, "sanitize")
    
    # 2. Parallel Fan-Out (Multiple Outgoing Edges from sanitize)
    builder.add_edge("sanitize", "sentiment_worker")
    builder.add_edge("sanitize", "entity_worker")
    
    # 3. Parallel Fan-In (Multiple Incoming Edges to aggregator)
    builder.add_edge("sentiment_worker", "aggregator")
    builder.add_edge("entity_worker", "aggregator")
    
    # 4. Terminal Egress Edge
    builder.add_edge("aggregator", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_fanout_graph()
    initial_payload = {
        "input_text": "LangGraph is a great orchestration framework from LangChain.",
        "cleaned_text": "",
        "sentiment": "",
        "entities": [],
        "audit_trace": ["Pipeline initialized."],
        "final_report": ""
    }
    result = app.invoke(initial_payload)
    print("Final Report:", result["final_report"])
    print("Audit Trail:", result["audit_trace"])
'''
st.code(prod_static_edges_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Topology Builder and Static Edge Simulator", divider="rainbow")

st.markdown("""
Trace how normal edges coordinate execution step-by-step. Enter an input query and watch
the simulator step through sequential ingress, parallel fan-out, barrier fan-in, and termination.
""")

edge_engine = StaticEdgeTopologyEngine()

col_ew_l, col_ew_r = st.columns([1, 1.2])

with col_ew_l:
    st.subheader("Graph Invocation Parameters")
    user_q = st.text_input("Initial Query String", value="LangGraph provides excellent multi-agent edge topologies!")
    
    if st.button("🚀 Trace Graph Execution Across Supersteps", use_container_width=True):
        st.session_state.edge_trace = edge_engine.trace_execution(user_q)
        st.success("Execution trace completed!")

with col_ew_r:
    st.subheader("Superstep Execution Trace")
    if "edge_trace" in st.session_state:
        for step in st.session_state.edge_trace:
            with st.expander(f"Superstep {step['superstep']}: {step['phase']} ({step['edge_transition']})", expanded=True):
                st.markdown(f"**Active Nodes**: `{step['active_nodes']}`")
                st.markdown(f"**State Delta Emitted**:")
                st.json(step["state_delta"])
    else:
        st.info("Click 'Trace Graph Execution' to observe how edges dispatch nodes across supersteps.")

st.markdown("### Topology Edge Definitions")
df_edges = pd.DataFrame([
    {"Source": src, "Destination": dst, "Type": "Normal Edge"} for src, dst in edge_engine.edges
])
st.dataframe(df_edges, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What happens if two parallel fan-out nodes attempt to mutate the same state key?", expanded=False):
    st.markdown("""
    **Answer**:
    In LangGraph, if two parallel nodes write to the same key:
    1. If the key has an `Annotated` reducer (such as `Annotated[list, operator.add]`), LangGraph executes the reducer function in deterministic order, merging the values safely.
    2. If the key is an unreduced overwrite channel, LangGraph raises an `InvalidUpdateError` due to ambiguous concurrent assignment.
    Parallel branches must always write to distinct keys or use reducer channels.
    """)

with st.expander("Q2: How does LangGraph detect and prevent unreachable nodes at compilation time?", expanded=False):
    st.markdown("""
    **Answer**:
    During `builder.compile()`, LangGraph analyzes the graph's adjacency matrix:
    1. It traverses all paths starting from `START`. Any node not reachable from `START` is flagged as unreachable.
    2. It checks for nodes that have no incoming edges or no outgoing edges.
    3. If any node cannot reach `END` through any valid path, compilation warns or fails to prevent runtime deadlocks.
    """)

with st.expander("Q3: Can a normal edge connect a node to itself to create an infinite loop?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes, `builder.add_edge('node_a', 'node_a')` creates a direct self-loop.
    However, with a static normal edge, there is no exit condition! The loop will run until `recursion_limit` is exceeded (default: 25 steps), raising `GraphRecursionError`.
    Loops should always use **conditional edges** so that a predicate function can terminate the loop when a condition is met.
    """)

with st.expander("Q4: Can START have multiple outgoing normal edges?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. Calling `builder.add_edge(START, 'worker_a')` and `builder.add_edge(START, 'worker_b')` is completely valid.
    When the graph is invoked, LangGraph dispatches both `worker_a` and `worker_b` concurrently in superstep 1.
    """)

with st.expander("Q5: What is the computational difference between sequential execution and fan-out/fan-in in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Sequential Execution**: Node B waits for Node A to finish; total latency is `latency(A) + latency(B)`.
    - **Fan-Out / Fan-In**: Nodes Alpha and Beta run concurrently in the same superstep; total latency is `max(latency(Alpha), latency(Beta)) + latency(Join)`.
    Fan-out reduces wall-clock time significantly when nodes perform independent LLM or tool calls.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Conflicting Overwrites in Parallel Fan-Out Branches")
    st.markdown("""
    *Mechanism*: Branching into two parallel nodes where both return `{"status": "IN_PROGRESS"}` without a reducer.  
    *Consequence*: Crash with `InvalidUpdateError: Conflicting updates for channel 'status'`.  
    *Fix*: Use separate channel names (`status_alpha`, `status_beta`) or an append reducer.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Orphan Nodes Missing Connections to END")
    st.markdown("""
    *Mechanism*: Adding a node `builder.add_node("audit", fn)` without an outgoing edge to `END` or another node.  
    *Consequence*: Compilation error or silent execution leaks where state transitions never resolve cleanly.  
    *Fix*: Connect every terminal branch explicitly to `END`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Overusing Sequential Edges for Independent Tasks")
    st.markdown("""
    *Mechanism*: Running 4 independent vector search nodes sequentially instead of using parallel fan-out.  
    *Consequence*: Latency increases linearly from 200ms to 800ms.  
    *Fix*: Use fan-out edges from START or a dispatcher node to parallelize independent I/O.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Self-Referencing Normal Edge Causing Immediate Recursion Limit Error")
    st.markdown("""
    *Mechanism*: Adding `builder.add_edge("node", "node")` without an exit condition.  
    *Consequence*: Immediate `GraphRecursionError` after 25 steps, failing the request.  
    *Fix*: Use `builder.add_conditional_edges` with a loop counter check for cyclic flows.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Normal Edge**\\nA fixed, deterministic transition between two nodes registered via `add_edge(source, target)`.")
with col_fc2:
    st.info("**START**\\nThe virtual ingress sentinel representing the initial entry point of a graph.")
with col_fc3:
    st.info("**END**\\nThe virtual egress sentinel representing the terminal completion boundary of a graph.")
with col_fc4:
    st.info("**Fan-In Barrier**\\nThe synchronization mechanism where a node with multiple incoming edges waits for all predecessors.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand how normal edges create deterministic transitions between nodes.", value=True)
c2 = st.checkbox("I know how to connect START and END virtual boundaries to govern ingress and egress.", value=True)
c3 = st.checkbox("I can implement parallel fan-out using multiple outgoing edges from a single node.", value=True)
c4 = st.checkbox("I understand the Bulk Synchronous Parallel barrier synchronization in fan-in joins.", value=True)
c5 = st.checkbox("I can prevent concurrent write collisions on shared channels during parallel execution.", value=True)
c6 = st.checkbox("I understand why static cycles without conditional edges lead to GraphRecursionError.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 6.1 'Normal Edges' with full architectural and practical comprehension!")
