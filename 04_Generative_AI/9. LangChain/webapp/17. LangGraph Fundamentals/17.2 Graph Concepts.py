import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class GraphConceptNodeTrace:
    """Represents the execution trace of an individual graph node."""
    def __init__(self, node_name: str, input_state: Dict[str, Any], output_delta: Dict[str, Any], next_nodes: List[str]):
        self.node_name = node_name
        self.input_state = input_state
        self.output_delta = output_delta
        self.next_nodes = next_nodes

class MockGraphConceptsStudio:
    """
    Simulates core LangGraph primitives:
    START -> Node Validation -> Conditional Branching -> Parallel Fan-Out -> Join Node -> END.
    """
    def __init__(self):
        pass

    def run_graph_simulation(
        self,
        task_type: str,
        enable_parallel_fanout: bool
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "task_type": task_type,
            "status": "initialized",
            "findings": [],
            "score": 0.0,
            "completed": False
        }
        traces: List[GraphConceptNodeTrace] = []

        # 1. START Edge -> validate_input
        traces.append(GraphConceptNodeTrace(
            "START",
            {},
            {"task_type": task_type, "status": "validated"},
            ["validate_input"]
        ))
        state["status"] = "validated"

        # 2. validate_input Node
        traces.append(GraphConceptNodeTrace(
            "validate_input",
            dict(state),
            {"status": "in_progress"},
            ["router_conditional_edge"]
        ))
        state["status"] = "in_progress"

        # 3. Conditional Edge Evaluation
        if task_type == "Security Audit":
            next_step = "security_scanner"
        elif task_type == "Performance Benchmark":
            next_step = "latency_analyzer"
        else: # General Triage
            next_step = "triage_handler"

        traces.append(GraphConceptNodeTrace(
            "router_conditional_edge",
            dict(state),
            {"route_decision": next_step},
            [next_step]
        ))

        # 4. Processing Branches (Simulating Parallel Fan-Out if enabled)
        if enable_parallel_fanout:
            traces.append(GraphConceptNodeTrace(
                f"{next_step} (Primary Branch)",
                dict(state),
                {"findings": [f"{task_type}: Primary validation passed."], "score": 0.88},
                ["join_aggregator"]
            ))
            traces.append(GraphConceptNodeTrace(
                "compliance_sentinel (Parallel Fan-Out Branch)",
                dict(state),
                {"findings": ["Compliance: Zero PII violations detected."]},
                ["join_aggregator"]
            ))
            state["findings"].append(f"{task_type}: Primary validation passed.")
            state["findings"].append("Compliance: Zero PII violations detected.")
            state["score"] = 0.94
        else:
            traces.append(GraphConceptNodeTrace(
                next_step,
                dict(state),
                {"findings": [f"{task_type}: Standard execution completed."], "score": 0.85},
                ["join_aggregator"]
            ))
            state["findings"].append(f"{task_type}: Standard execution completed.")
            state["score"] = 0.85

        # 5. Join Node -> END
        traces.append(GraphConceptNodeTrace(
            "join_aggregator",
            dict(state),
            {"status": "completed", "completed": True},
            ["END"]
        ))
        state["status"] = "completed"
        state["completed"] = True

        traces.append(GraphConceptNodeTrace(
            "END",
            dict(state),
            {"final_payload": True},
            []
        ))

        return {
            "final_state": state,
            "traces": traces
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="17.2 Graph Concepts — LangChain Course",
    page_icon="🧩",
    layout="wide"
)

st.title("17.2 Graph Concepts")
st.caption("Deep-Dive into LangGraph Primitives: State Schemas, Nodes, Edges, Conditional Edges, START, END, State Updates, and Compilation")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: Core Graph Topologies, Primitives & Lifecycle", divider="rainbow")
st.markdown("""
A **LangGraph** application is a mathematical graph G = (V, E) where vertices V represent computational nodes 
and directed edges E represent state transitions. Execution is governed by a centralized state channel that passes 
immutable snapshots between supersteps.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GraphConceptsArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_entry {
            label = "Primitive 1: Graph Entrypoint";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            start_prim [label="START\\nVirtual node receiving initial invoke() payload", fillcolor="#1E293B", color="#0284C7"];
        }

        subgraph cluster_nodes {
            label = "Primitive 2 & 3: Nodes & State Transformation";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            node_a [label="Node A: Python Function / Runnable\\nInput: StateSnapshot\\nOutput: Dict[key, delta]", fillcolor="#0F172A", color="#38BDF8"];
            cond_prim [label="Primitive 4: Conditional Edge\\nRouting function returning next node key", fillcolor="#D97706", color="#F59E0B"];
        }

        subgraph cluster_fanout {
            label = "Primitive 5: Parallel Fan-Out & Join";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            worker_1 [label="Node B (Worker 1)", fillcolor="#1E293B", color="#64748B"];
            worker_2 [label="Node C (Worker 2)", fillcolor="#1E293B", color="#64748B"];
            join_node [label="Node D (Join / Aggregator Node)\\nMerges parallel channel updates", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_exit {
            label = "Primitive 6: Terminal Exit";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            end_prim [label="END\\nVirtual terminal node emitting final state", fillcolor="#064E3B", color="#10B981"];
        }

        start_prim -> node_a [label="Direct Edge"];
        node_a -> cond_prim;
        
        cond_prim -> worker_1 [label="path == 'parallel'"];
        cond_prim -> worker_2 [label="path == 'parallel'"];
        cond_prim -> join_node [label="path == 'direct'"];

        worker_1 -> join_node [label="Direct Edge"];
        worker_2 -> join_node [label="Direct Edge"];
        
        join_node -> end_prim [label="Direct Edge"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: STATE ---
st.header("2. State Schemas & State Channels", divider="rainbow")
st.markdown("""
**State** is the central data contract of a LangGraph application. It represents the shared memory passed to every node.
In Python, State is typically defined using either `typing.TypedDict` or `pydantic.BaseModel`.

### Core Rules of State Design:
1. **Type Safety**: Explicit type annotations guarantee static linting and runtime schema validation.
2. **Channel Reducers**: Attributes wrapped with `Annotated[T, reducer_fn]` define how concurrent updates are combined.
3. **Immutability Invariant**: Nodes must treat the incoming state dictionary as read-only. Modifying existing state objects 
   in place breaks time-travel checkpointers and parallel execution.
""")

st.code("""
from typing import Annotated, List
from typing_extensions import TypedDict
import operator

def append_records(existing: List[str], new_records: List[str]) -> List[str]:
    return existing + new_records

class EnterpriseState(TypedDict):
    task_id: str
    # Uses operator.add as a channel reducer to concatenate lists
    audit_logs: Annotated[List[str], operator.add]
    overall_score: float
""", language="python")

st.divider()

# --- SECTION 3: NODES ---
st.header("3. Nodes: Computational Units", divider="rainbow")
st.markdown("""
A **Node** is a computational unit in the graph. Nodes can be standard Python functions, asynchronous coroutines, 
or pre-existing LCEL Runnables.

### Node Signatures:
A node accepts the current state (and optionally an execution config) and returns a **dictionary containing state updates**:
`def my_node(state: MyState, config: RunnableConfig) -> Dict[str, Any]:`
""")

st.code("""
def analyze_compliance(state: EnterpriseState) -> Dict[str, Any]:
    # Read current state attributes
    task = state["task_id"]
    
    # Perform business logic / model inference
    result = f"Task {task}: Compliant with SOC2 Type II standards."
    
    # Return update dictionary (merged into state by channel reducers)
    return {
        "audit_logs": [result],
        "overall_score": 0.98
    }
""", language="python")

st.divider()

# --- SECTION 4: EDGES ---
st.header("4. Direct Edges & Execution Flow", divider="rainbow")
st.markdown("""
A **Direct Edge** defines an unconditional transition from a source node to a target node. 
Direct edges are registered using `builder.add_edge(source, target)`.
""")

col_ed1, col_ed2 = st.columns(2)
with col_ed1:
    st.info("""
    **Sequential Direct Edges**:
    `builder.add_edge("node_a", "node_b")`
    `builder.add_edge("node_b", "node_c")`
    Execution runs strictly in sequence: Node A completes, then Node B executes, then Node C.
    """)
with col_ed2:
    st.success("""
    **Parallel Fan-Out Edges**:
    `builder.add_edge("source", "worker_1")`
    `builder.add_edge("source", "worker_2")`
    A single node emits edges to multiple targets, triggering parallel execution in the next superstep!
    """)

st.divider()

# --- SECTION 5: CONDITIONAL EDGES ---
st.header("5. Conditional Edges & Dynamic Routing", divider="rainbow")
st.markdown("""
A **Conditional Edge** introduces dynamic routing logic based on the state emitted by the preceding node. 
It requires a routing function and an optional path map.
""")

st.code("""
from typing import Literal

def route_triage(state: EnterpriseState) -> Literal["escalate_incident", "standard_resolution"]:
    if state["overall_score"] < 0.70:
        return "escalate_incident"
    return "standard_resolution"

# Register conditional edge
builder.add_conditional_edges(
    "analyze_compliance",
    route_triage,
    ["escalate_incident", "standard_resolution"]  # Explicit target node destinations
)
""", language="python")

st.divider()

# --- SECTION 6: START ---
st.header("6. The START Node", divider="rainbow")
st.markdown("""
`START` is a special virtual node exported by `langgraph.graph`. It represents the entry point of the graph.
When you call `graph.invoke(initial_state)`, execution begins at `START` and flows immediately to whichever node(s) 
are connected via `builder.add_edge(START, "first_node")`.
""")

st.code("""
from langgraph.graph import StateGraph, START

builder = StateGraph(EnterpriseState)
# Connect START to the initial node
builder.add_edge(START, "analyze_compliance")
""", language="python")

st.divider()

# --- SECTION 7: THE END NODE ---
st.header("7. The END Node", divider="rainbow")
st.markdown("""
`END` is the virtual terminal node exported by `langgraph.graph`. When an edge transitions to `END`, graph execution 
terminates, and the final state dictionary is returned to the caller.
""")

st.code("""
from langgraph.graph import END

# Transitions to END terminate graph execution
builder.add_edge("standard_resolution", END)
builder.add_edge("escalate_incident", END)
""", language="python")

st.divider()

# --- SECTION 8: STATE UPDATES ---
st.header("8. State Updates & Superstep Resolution", divider="rainbow")
st.markdown("""
LangGraph operates on the **Pregel superstep execution model**:
1. In Superstep N, all scheduled nodes execute concurrently in worker threads.
2. Each node returns a partial update dictionary (e.g. `{"audit_logs": [entry]}`).
3. At the end of the superstep, LangGraph runs each attribute's channel reducer to merge the partial updates into the central state.
4. Downstream edges are evaluated to schedule the nodes that will execute in Superstep N+1.
""")

st.divider()

# --- SECTION 9: GRAPH COMPILATION ---
st.header("9. Graph Compilation (`compile()`)", divider="rainbow")
st.markdown("""
Before a `StateGraph` can be invoked, it must be validated and compiled via `.compile()`. Compilation:
1. **Validates Graph Invariants**: Detects orphaned nodes, unreachable cycles, or missing edges to `START` and `END`.
2. **Attaches Checkpointers**: Registers persistent memory stores (`MemorySaver`, `SqliteSaver`, `PostgresSaver`).
3. **Injects Interrupts**: Configures human-in-the-loop pause points (`interrupt_before=["publish_report"]`).
4. **Returns a CompiledStateGraph**: The compiled object implements the full LCEL Runnable interface (`invoke`, `stream`, `batch`).
""")

st.code("""
from langgraph.checkpoint.memory import MemorySaver

# Compilation with checkpointer and breakpoints
checkpointer = MemorySaver()
compiled_app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["escalate_incident"]  # Pauses for human approval before escalation!
)
""", language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Multi-Branch StateGraph with Parallel Fan-Out & Join", divider="rainbow")
st.markdown("""
Below is a full production-ready StateGraph demonstrating dynamic conditional routing, parallel worker fan-out, 
and a join aggregator node.
""")

st.code("""
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END

# 1. State Definition
class PipelineState(TypedDict):
    ticket_id: str
    urgency: str
    logs: Annotated[List[str], operator.add]
    metrics: Dict[str, float]

# 2. Node Functions
def intake_node(state: PipelineState):
    return {"logs": ["Intake complete. Analyzing severity."]}

def route_urgency(state: PipelineState) -> Literal["parallel_fanout_a", "standard_handler"]:
    if state["urgency"] == "critical":
        return "parallel_fanout_a"
    return "standard_handler"

def worker_sec(state: PipelineState):
    return {"logs": ["Security scanner: Pass."], "metrics": {"sec_score": 1.0}}

def worker_perf(state: PipelineState):
    return {"logs": ["Perf scanner: Latency optimal."], "metrics": {"perf_score": 0.95}}

def standard_handler(state: PipelineState):
    return {"logs": ["Standard triage complete."], "metrics": {"triage_score": 0.9}}

def join_node(state: PipelineState):
    return {"logs": ["Aggregation finished. Ticket resolved."]}

# 3. Graph Assembly
builder = StateGraph(PipelineState)
builder.add_node("intake", intake_node)
builder.add_node("worker_sec", worker_sec)
builder.add_node("worker_perf", worker_perf)
builder.add_node("standard_handler", standard_handler)
builder.add_node("join_node", join_node)

# 4. Connecting Edges
builder.add_edge(START, "intake")
builder.add_conditional_edges(
    "intake",
    route_urgency,
    {
        "parallel_fanout_a": ["worker_sec", "worker_perf"],  # Parallel fan-out!
        "standard_handler": "standard_handler"
    }
)

builder.add_edge("worker_sec", "join_node")
builder.add_edge("worker_perf", "join_node")
builder.add_edge("standard_handler", "join_node")
builder.add_edge("join_node", END)

# 5. Compile
app = builder.compile()
""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Graph Concepts Topology Studio", divider="rainbow")
st.markdown("""
Experiment with graph execution topologies: select a task type to observe **conditional routing**, and toggle 
**parallel fan-out** to see how multiple worker nodes run simultaneously and merge into a join aggregator.
""")

# Instantiate module-scope studio
concepts_studio = MockGraphConceptsStudio()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    task_choice = st.selectbox(
        "Select Pipeline Task Type:",
        ["Security Audit", "Performance Benchmark", "General Triage"]
    )

    parallel_fanout_toggle = st.toggle("Enable Parallel Fan-Out (2 Concurrent Workers)", value=True)
    run_sim_btn = st.button("🚀 Execute Graph Topology", type="primary", use_container_width=True)

with wb_c2:
    st.subheader("Graph Execution Trace")

    if run_sim_btn or task_choice:
        sim_output = concepts_studio.run_graph_simulation(task_choice, parallel_fanout_toggle)

        st.markdown("### 📊 Final Resolved State")
        st.json(sim_output["final_state"])

        st.markdown("### 🗺️ Node Execution Step Trace")
        for trace in sim_output["traces"]:
            with st.expander(f"Node: [{trace.node_name}] -> Outgoing: {trace.next_nodes}", expanded=True):
                st.write(f"**State Delta Emitted**: `{trace.output_delta}`")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the difference between a channel and a state key in LangGraph?",
     "In LangGraph, state is composed of named channels. A standard state key without a reducer is an 'overwrite channel': whenever a node returns a value for that key, the previous value is completely overwritten. A channel with a reducer function (like `Annotated[list, add_messages]` or `Annotated[list, operator.add]`) merges new data with existing data. Channels manage synchronization, concurrency, and conflict resolution during superstep execution."),
    ("Can a conditional edge route to multiple nodes in parallel?",
     "Yes. A conditional edge routing function can return a list of node names rather than a single string (e.g. `return ['worker_a', 'worker_b']`). When a list is returned, LangGraph executes all specified nodes concurrently in parallel during the next superstep, enabling dynamic fan-out workflows."),
    ("What happens if two parallel nodes return updates to the same state key without a reducer?",
     "If two nodes execute concurrently in the same superstep and both attempt to write to an un-reduced state channel (overwrite channel), LangGraph raises an `InvalidUpdateError` exception due to conflicting non-deterministic state mutations. To allow parallel nodes to write to the same channel, you must attach a reducer function (e.g. `operator.add`)."),
    ("Why is `START` necessary instead of simply calling the first node directly?",
     "The virtual `START` node serves as the formal boundary where input validation occurs. When `app.invoke(input_data)` is called, the input data is validated against the State schema at `START` before any user node executes. It also allows multiple entry edges (e.g. fan-out directly from START to multiple initialization nodes)."),
    ("What are the advantages of using Pydantic BaseModel vs TypedDict for State?",
     "TypedDict provides zero runtime validation overhead, fast dictionary access, and seamless JSON serialization. Pydantic BaseModel provides strict runtime type coercion, field constraints (e.g. `Field(gt=0)`), and custom validators. TypedDict is preferred for lightweight high-throughput agents; Pydantic is preferred when input states originate from untrusted external APIs.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Writing to Un-reduced State Keys from Parallel Nodes. If parallel worker nodes write to `state['status']` simultaneously without a channel reducer, LangGraph throws an InvalidUpdateError. Always use list reducers or separate dedicated output keys.")
    st.error("**Trap 2 (Breaking Error)**: Returning Undefined Node Names from Conditional Edges. If a conditional edge routing function returns a node name that was not registered with `builder.add_node()`, graph execution will fail with a KeyNotFound exception at runtime.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Forgetting Loop Termination in Cyclic Graphs. If a conditional edge never evaluates to END due to a logic bug, the graph loops until `recursion_limit` is hit, burning unnecessary model tokens.")
    st.warning("**Trap 4 (Quality Warning)**: Re-compiling the Graph on Every HTTP Request. Calling `builder.compile()` is an expensive operation that validates graph schemas and initializes checkpointers. Always compile the graph once at application startup as a global singleton.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: StateGraph"):
        st.markdown("**Graph builder compiling nodes, direct edges, and conditional edges into a runnable state machine.**")
    with st.expander("🎴 Flashcard 2: Conditional Edge"):
        st.markdown("**Dynamic routing function inspecting current state to select the next node or nodes.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Channel Reducer"):
        st.markdown("**Aggregation function (e.g. operator.add) defining how concurrent updates are merged into state.**")
    with st.expander("🎴 Flashcard 4: Parallel Fan-Out"):
        st.markdown("**Routing execution to multiple worker nodes concurrently within a single Pregel superstep.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural roles of State, Nodes, Edges, and Conditional Edges.", value=True)
chk2 = st.checkbox("I know how the START and END virtual nodes govern graph lifecycle boundaries.", value=True)
chk3 = st.checkbox("I can implement channel reducers using operator.add to prevent update collisions.", value=True)
chk4 = st.checkbox("I understand how parallel fan-out and join nodes execute across supersteps.", value=True)
chk5 = st.checkbox("I understand why graphs should be compiled once at application startup.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 graph concepts technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 17.2 Graph Concepts!")
