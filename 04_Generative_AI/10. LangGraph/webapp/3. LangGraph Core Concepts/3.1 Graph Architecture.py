import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="3.1 Graph Architecture - LangGraph Curriculum",
    page_icon="🕸️",
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
# MODULE-SCOPE GRAPH ARCHITECTURE ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class ArchitectureGraphEngine:
    """
    Simulates the core Graph Architecture lifecycle:
    Vertices (V), Directed Edges (E), Centralized State (S),
    Compilation validation, and Superstep execution.
    """
    def __init__(self):
        self.registered_nodes: Dict[str, str] = {
            "ingest": "Validates and sanitizes input data",
            "transform": "Applies business transformations and ML reasoning",
            "review": "Validates transformation quality or routes for corrections"
        }
        self.registered_edges: List[Tuple[str, str]] = [
            ("START", "ingest"),
            ("ingest", "transform"),
            ("transform", "review"),
            ("review", "END")
        ]

    def compile_graph(self) -> Tuple[bool, List[str]]:
        errors = []
        # Invariant 1: START must have at least one outgoing edge
        has_start = any(src == "START" for src, _ in self.registered_edges)
        if not has_start:
            errors.append("Validation Error: Missing entry edge from START.")

        # Invariant 2: At least one edge must reach END
        has_end = any(dst == "END" for _, dst in self.registered_edges)
        if not has_end:
            errors.append("Validation Error: Missing terminal edge to END.")

        return len(errors) == 0, errors

    def execute_supersteps(self, initial_payload: str) -> List[Dict[str, Any]]:
        trace = []
        curr_state = {"payload": initial_payload, "step_count": 0, "status": "INITIALIZED"}
        
        # Step 1: Ingest
        curr_state["step_count"] += 1
        curr_state["status"] = "SANITIZED"
        trace.append({
            "superstep": 1,
            "node": "ingest",
            "state_snapshot": dict(curr_state),
            "edge_transition": "ingest -> transform"
        })

        # Step 2: Transform
        curr_state["step_count"] += 1
        curr_state["payload"] = curr_state["payload"].title() + " [Processed]"
        curr_state["status"] = "TRANSFORMED"
        trace.append({
            "superstep": 2,
            "node": "transform",
            "state_snapshot": dict(curr_state),
            "edge_transition": "transform -> review"
        })

        # Step 3: Review
        curr_state["step_count"] += 1
        curr_state["status"] = "APPROVED"
        trace.append({
            "superstep": 3,
            "node": "review",
            "state_snapshot": dict(curr_state),
            "edge_transition": "review -> END"
        })

        return trace


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("3.1 Graph Architecture: Vertices, Edges, State, & Compilation Invariants")
st.caption("Chapter 3: LangGraph Core Concepts | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
At its core, **LangGraph** models artificial intelligence workflows using classical **Graph Theory**. 
Rather than chaining function calls sequentially in opaque procedural scripts, LangGraph formalizes execution 
as a formal tuple: `G = (V, E, S)`, where `V` represents computational **Nodes** (vertices), `E` represents 
directed **Edges** (control flow transitions), and `S` represents a centralized, typed **State** schema.

In this lesson, you will master:
* The mathematical and architectural definition of an AI Execution Graph.
* How Nodes transform state and why pure functions prevent concurrency corruption.
* Connecting Nodes using Directed Edges, Conditional Routers, and Entry/Exit markers (`START`, `END`).
* What occurs under the hood during Graph Compilation (`builder.compile()`).
* The step-by-step lifecycle of Graph Invocation and Superstep execution.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Core Graph Architecture Topology & Pregel Runtime", divider="rainbow")

st.markdown("""
The architecture below illustrates the anatomy of a compiled LangGraph state machine: from graph definition, 
compilation invariant verification, through to the Pregel superstep execution loop.
""")

blueprint_dot = """
digraph GraphArchitecture {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_definition {
        label = "1. Graph Definition Phase (StateGraph Builder)";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        StateSchema [label="Typed State Schema (S)\\n(TypedDict / Pydantic Base Model)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        NodesVertices [label="Nodes / Vertices (V)\\n(add_node('agent', fn), add_node('tools', fn))", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        EdgesTransitions [label="Edges / Control Flow (E)\\n(add_edge, add_conditional_edges)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_compilation {
        label = "2. Compilation Phase (builder.compile())";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        ValidationCheck [label="Topology Validation Check\\n(Cycles valid? Entry START bound? End reachable?)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        CompiledGraph [label="CompiledStateGraph Instance\\n(Pregel Runner + Checkpointer Attached)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_execution {
        label = "3. Invocation & Superstep Execution Loop";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        StartEntry [label="START\\n(Ingress State Injected)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        SuperstepLoop [label="Superstep Execution Loop\\n(Read State -> Run Nodes -> Reduce Delts -> Checkpoint)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        EndTerminal [label="END\\n(Egress Terminal State Harvested)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    StateSchema -> NodesVertices;
    NodesVertices -> EdgesTransitions;
    EdgesTransitions -> ValidationCheck [label="Invoke .compile()"];
    ValidationCheck -> CompiledGraph [label="Invariants Satisfied"];

    CompiledGraph -> StartEntry [label="app.invoke(input)"];
    StartEntry -> SuperstepLoop [label="Superstep 1"];
    SuperstepLoop -> SuperstepLoop [label="Cyclic Transitions", color="#38BDF8"];
    SuperstepLoop -> EndTerminal [label="Termination Condition Met", color="#EC4899", penwidth=1.8];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - WHAT IS A GRAPH?
# ==============================================================================
st.header("2. What is a Graph? Mathematical Graph Theory Applied to AI Control Flow", divider="rainbow")

st.markdown("""
In computer science, a **Graph** is a non-linear data structure comprising a set of **Vertices (Nodes)** connected by **Edges (Links)**. 

Historically, machine learning pipelines were strictly modeled as **DAGs (Directed Acyclic Graphs)**. In a DAG, data flows 
exclusively forward; backtracking or looping is mathematically illegal. 

However, **reasoning is inherently non-linear**:
* A software engineering agent writes code, runs unit tests, catches a syntax error, and must loop back to fix the script.
* A research agent searches Google, discovers insufficient data, and loops back with an expanded query.

LangGraph formalizes execution as a **Cyclic Directed Graph**, turning loops, retries, and reflections into first-class citizens.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - NODES
# ==============================================================================
st.header("3. Nodes: Discrete Computation Units, State Transformers, and Pure Functions", divider="rainbow")

st.markdown("""
A **Node** represents a single discrete unit of work within the graph. In Python, a node is simply a function:
""")

col_nd1, col_nd2 = st.columns([1, 1])

with col_nd1:
    st.markdown("""
    ### Node Contract
    * **Input**: Receives the current centralized `State` snapshot (and optional `RunnableConfig`).
    * **Processing**: Performs arbitrary Python logic: calls foundation LLMs, queries vector databases, or runs APIs.
    * **Output**: Returns a dictionary representing the *state update* (delta) to be merged.
    """)

with col_nd2:
    st.markdown("""
    ### Code Structure
    """)
    st.code("""
def model_node(state: AgentState) -> dict:
    prompt = state["user_query"]
    response = llm.invoke(prompt)
    # Return ONLY the keys to update
    return {"messages": [response]}

builder.add_node("call_model", model_node)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - EDGES
# ==============================================================================
st.header("4. Edges: Directed Transitions, Control Flow Pathways, and Routing Protocols", divider="rainbow")

st.markdown("""
An **Edge** defines how control flow transitions between nodes. LangGraph provides two distinct edge types:
""")

col_ed1, col_ed2 = st.columns([1, 1])

with col_ed1:
    st.markdown("""
    ### 1. Fixed Normal Edges (`add_edge`)
    Unconditional deterministic transitions. Whenever Node A finishes, Node B always executes next:
    `builder.add_edge("node_a", "node_b")`
    """)

with col_ed2:
    st.markdown("""
    ### 2. Conditional Edges (`add_conditional_edges`)
    Dynamic routing based on runtime state evaluation. A router function inspects the state and returns a destination string:
    """)
    st.code("""
def route_next_step(state: AgentState) -> str:
    if state["has_errors"]:
        return "retry_node"
    return "finalize_node"

builder.add_conditional_edges(
    "evaluate_step",
    route_next_step,
    {"retry_node": "retry_node", "finalize_node": "finalize_node"}
)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - STATE
# ==============================================================================
st.header("5. State: Centralized Typed Memory Channel Shared Across Graph Vertices", divider="rainbow")

st.markdown("""
In LangGraph, there is no global application variable or hidden thread memory. All data passing between nodes 
is explicitly governed by the **State Schema**:
* The state is declared using Python `TypedDict`, dataclasses, or Pydantic models.
* Nodes do not mutate state directly; they emit dictionary updates.
* **Channel Reducers** specify how updates are merged (e.g., overriding a scalar value vs appending to a list).
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - START NODE
# ==============================================================================
st.header("6. Start Node: Ingress Validation, Thread Initialization, and Entry Transitions", divider="rainbow")

st.markdown("""
The **`START`** node is a reserved architectural constant in LangGraph representing the entrypoint of the state machine.
""")

col_st1, col_st2 = st.columns([1, 1])

with col_st1:
    st.markdown("""
    ### Why `START` Exists
    * Historically, developers called `builder.set_entry_point("node_name")`. 
    * In modern LangGraph, `START` is a first-class virtual node. 
    * This allows developers to use standard edge wiring: `builder.add_edge(START, "first_node")`.
    """)

with col_st2:
    st.markdown("""
    ### Conditional Graph Ingress
    Using `START`, you can even route traffic conditionally upon ingress before running any nodes!
    """)
    st.code("""
# Dynamically route on graph entry based on user query
builder.add_conditional_edges(
    START,
    classify_intent,
    {"billing": "billing_agent", "tech": "tech_agent"}
)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - END NODE
# ==============================================================================
st.header("7. End Node: Egress Termination, State Finalization, and Output Harvesting", divider="rainbow")

st.markdown("""
The **`END`** node is a reserved architectural constant representing the termination of graph execution for the current invocation.
* When execution reaches an edge pointing to `END`, the Pregel superstep loop terminates immediately.
* The final state dictionary is committed to the checkpointer and returned to the caller.
* Any active streaming connection flushes its final payload and closes the HTTP stream.
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - GRAPH EXECUTION
# ==============================================================================
st.header("8. Graph Execution: The Pregel Superstep Iteration & Synchronization Loop", divider="rainbow")

st.markdown("""
LangGraph's execution engine is built on **Pregel**, Google's bulk-synchronous parallel graph processing model:
1. **Superstep Initialization**: Identify which nodes are currently scheduled to run.
2. **Parallel Step Execution**: Run all active nodes concurrently.
3. **Channel Reductions**: Apply channel reducers to merge all node updates deterministically into the unified state.
4. **Edge Traversal**: Evaluate outgoing fixed and conditional edges to determine the next batch of nodes.
5. **Persistence Commit**: If a checkpointer is attached, serialize the state snapshot to disk.
""")

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - GRAPH COMPILATION
# ==============================================================================
st.header("9. Graph Compilation: Topology Validation, Checkpoint Attachment, and Invariant Checks", divider="rainbow")

st.markdown("""
Before a graph can be invoked, you must call **`builder.compile()`**. 
Compilation is not a no-op—it performs strict structural analysis:
""")

comp_invariants = [
    {"Invariant Check": "Start Reachability", "Verification": "Confirms that an edge exists from `START` to at least one valid node."},
    {"Invariant Check": "End Reachability", "Verification": "Ensures that at least one control flow path can terminate at `END`."},
    {"Invariant Check": "Orphan Node Detection", "Verification": "Flags any node that has no incoming edges (unreachable code)."},
    {"Invariant Check": "Reducer Type Safety", "Verification": "Validates that channel reducers are compatible with state schema types."},
    {"Invariant Check": "Checkpointer Binding", "Verification": "Attaches durable checkpointer storage (`MemorySaver`, `PostgresSaver`)."}
]
st.dataframe(pd.DataFrame(comp_invariants), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 10: TOPIC 9 - GRAPH INVOCATION
# ==============================================================================
st.header("10. Graph Invocation: Dispatching Initial Payloads and Thread Context", divider="rainbow")

st.markdown("""
Once compiled, the graph instance is a standard LangChain `Runnable`:
""")

col_inv1, col_inv2 = st.columns([1, 1])

with col_inv1:
    st.markdown("""
    ### Invocation Protocols
    * **`app.invoke(input, config)`**: Synchronous blocking execution until `END` is reached.
    * **`app.ainvoke(input, config)`**: Asynchronous non-blocking execution for web gateways.
    * **`app.stream(input, config)`**: Yields state updates after each superstep.
    """)

with col_inv2:
    st.markdown("""
    ### Thread Scoping
    Always pass `thread_id` in `config` when using persistent checkpointers:
    """)
    st.code("""
config = {"configurable": {"thread_id": "session-8821"}}
final_state = app.invoke({"query": "Analyze financial report"}, config=config)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 11: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("11. Production Implementation: Full-Featured Graph Architecture Coordinator", divider="rainbow")

st.markdown("""
Below is a complete, self-contained Python implementation showing explicit state declaration, 
nodes, fixed edges, conditional routing, compilation, and invocation:
""")

prod_code = """
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END

# 1. State Definition
class PipelineState(TypedDict):
    raw_data: str
    validated: bool
    result: str
    iteration: int

# 2. Node Functions
def ingress_node(state: PipelineState) -> Dict[str, Any]:
    print(f"Ingesting: {state['raw_data']}")
    return {"validated": len(state["raw_data"]) > 3, "iteration": 1}

def process_node(state: PipelineState) -> Dict[str, Any]:
    return {"result": f"Processed '{state['raw_data']}' successfully."}

def error_node(state: PipelineState) -> Dict[str, Any]:
    return {"result": "Data failed validation check."}

# 3. Router Function
def routing_condition(state: PipelineState) -> str:
    if state["validated"]:
        return "process"
    return "error"

# 4. Assemble Graph Architecture
builder = StateGraph(PipelineState)
builder.add_node("ingress", ingress_node)
builder.add_node("process", process_node)
builder.add_node("error", error_node)

# Connect Edges
builder.add_edge(START, "ingress")
builder.add_conditional_edges(
    "ingress",
    routing_condition,
    {"process": "process", "error": "error"}
)
builder.add_edge("process", END)
builder.add_edge("error", END)

# 5. Compile & Invoke
app = builder.compile()
output = app.invoke({"raw_data": "Production Enterprise Transaction"})
print("Execution Result:", output)
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 12: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("12. Interactive Workbench: Live Graph Topology Builder & Execution Tracer", divider="rainbow")

st.markdown("""
Simulate the graph execution lifecycle. Input sample data and watch the architecture engine validate graph invariants, 
execute nodes step-by-step, and transition state to `END`.
""")

engine = ArchitectureGraphEngine()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Ingress Payload & Compilation Check")
    
    test_payload = st.text_input("Ingress Payload", value="autonomous customer order #9921")
    
    is_compiled, comp_errors = engine.compile_graph()
    
    if is_compiled:
        st.success("✅ Graph Compilation Status: Invariants Satisfied (START, Nodes, END reachable).")
    else:
        st.error(f"Compilation Failed: {comp_errors}")

    run_btn = st.button("Invoke Graph Pipeline 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Superstep Execution Trace")
    
    if run_btn:
        trace = engine.execute_supersteps(test_payload)
        
        m1, m2 = st.columns(2)
        m1.metric("Supersteps Completed", len(trace))
        m2.metric("Final Status", trace[-1]["state_snapshot"]["status"])

        st.markdown("#### Transition Log")
        for step in trace:
            with st.container():
                st.markdown(f"**Superstep {step['superstep']}** | Active Node: `<span class='badge-node'>{step['node']}</span>`", unsafe_allow_html=True)
                st.caption(f"Transition: `{step['edge_transition']}`")
                st.json(step["state_snapshot"])
                st.divider()

        st.success("Graph execution reached END successfully!")
    else:
        st.info("Click 'Invoke Graph Pipeline' above to simulate execution.")

st.divider()

# ==============================================================================
# SECTION 13: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Questions on graph topology, compilation guarantees, and Pregel superstep lifecycles:
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: How does LangGraph represent a graph mathematically compared to traditional DAG engines?"):
    st.markdown("""
    **Answer**: In traditional workflow engines (Airflow/Prefect), graphs are strictly **Directed Acyclic Graphs (DAGs)** where 
    topological sorting guarantees each node executes exactly once without cycles. In LangGraph, the execution structure is a 
    **Cyclic Directed Graph** governed by a centralized state channel. Cycles are permitted and expected, allowing agents to loop, 
    self-correct, and retry steps until a dynamic condition evaluates to `END`.
    """)

with st.expander("Q2: What happens if a node throws an unhandled Python exception during a superstep?"):
    st.markdown("""
    **Answer**: If a node throws an unhandled exception, the current superstep aborts immediately. The state update emitted by 
    the failing node is discarded, and no checkpoint is committed for that superstep. If durable checkpointers (`PostgresSaver`) 
    are active, the state remains preserved at the *preceding* completed superstep, allowing external retry queues or worker pods 
    to re-attempt execution from that exact historical state.
    """)

with st.expander("Q3: What is the architectural role of `START` and `END`?"):
    st.markdown("""
    **Answer**: `START` and `END` are reserved virtual node identifiers. `START` acts as the ingress boundary, mapping initial 
    user-provided input into the state channels. `END` acts as the terminal egress boundary, halting the Pregel iteration loop 
    and returning the final state dictionary to the caller.
    """)

with st.expander("Q4: Can a single node have multiple outgoing fixed edges?"):
    st.markdown("""
    **Answer**: Yes! If Node A has outgoing fixed edges to both Node B and Node C (`builder.add_edge("A", "B")` and 
    `builder.add_edge("A", "C")`), LangGraph schedules both Node B and Node C to run concurrently in parallel during the next superstep.
    """)

with st.expander("Q5: Why is `builder.compile()` required before calling `.invoke()`?"):
    st.markdown("""
    **Answer**: `StateGraph` is a builder pattern object for configuring vertices and edges. `.compile()` validates the topology 
    (detects orphan nodes, ensures entry/exit reachability), binds checkpointers and interrupt hooks, and constructs the optimized 
    `Pregel` runtime runner that executes the superstep loop.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Defining Orphan Nodes Disconnected from the Graph (CRITICAL)**
    * *Error*: Registering `builder.add_node("helper", fn)` without any incoming edges.
    * *Impact*: The node is completely unreachable and will never execute.
    * *Fix*: Always wire nodes using `add_edge` or `add_conditional_edges`.
    """)

    st.error("""
    **Trap 2: Returning Invalid State Keys from Node Functions (CRITICAL)**
    * *Error*: Returning a dictionary key not declared in the state `TypedDict` schema.
    * *Impact*: In strict mode, causes runtime validation errors or pollutes state channels.
    * *Fix*: Ensure all returned keys strictly match your `State` schema definition.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Forgetting Terminal Edges to `END` Causing Infinite Execution (WARNING)**
    * *Warning*: Wiring all conditional edges to loop without any path reaching `END`.
    * *Impact*: The graph loops continuously until it crashes with `GraphRecursionError`.
    * *Fix*: Always guarantee at least one conditional branch points to `END`.
    """)

    st.warning("""
    **Trap 4: Mutating Shared Global Objects Inside Nodes (WARNING)**
    * *Warning*: Modifying global variables or external singletons inside node functions.
    * *Impact*: Breaks thread isolation in multi-tenant environments.
    * *Fix*: Keep node functions pure; pass all data through `State` and `RunnableConfig`.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: Graph Tuple</strong><br><br>
        <em>Front:</em> What is the formal tuple of a LangGraph system?<br><br>
        <em>Back:</em> G = (V, E, S) where V=Nodes, E=Edges, S=Centralized State.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Pregel</strong><br><br>
        <em>Front:</em> What execution model powers LangGraph?<br><br>
        <em>Back:</em> The Pregel bulk-synchronous parallel superstep model.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Node Return</strong><br><br>
        <em>Front:</em> What does a node return?<br><br>
        <em>Back:</em> A dictionary delta of state updates to be merged by reducers.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: compile()</strong><br><br>
        <em>Front:</em> What invariant does compile() check on START?<br><br>
        <em>Back:</em> Verifies that START connects to at least one valid node.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 14: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("14. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of Graph Architecture fundamentals:")

chk1 = st.checkbox("I understand the definition of a graph as Vertices, Edges, and State.", value=True)
chk2 = st.checkbox("I know why pure node functions prevent concurrency bugs in LangGraph.", value=True)
chk3 = st.checkbox("I can differentiate between fixed edges and dynamic conditional edges.", value=True)
chk4 = st.checkbox("I understand the roles of `START` and `END` virtual node markers.", value=True)
chk5 = st.checkbox("I know what topology invariants are verified during `builder.compile()`.", value=True)
chk6 = st.checkbox("I understand the Pregel superstep execution and synchronization model.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 3.1: Graph Architecture! Proceed to Lesson 3.2: Core Building Blocks.")
