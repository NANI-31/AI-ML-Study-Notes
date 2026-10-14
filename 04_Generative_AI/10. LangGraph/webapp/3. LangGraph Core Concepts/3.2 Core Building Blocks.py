import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="3.2 Core Building Blocks - LangGraph Curriculum",
    page_icon="🧱",
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
    .badge-block {
        background-color: #1E293B;
        color: #10B981;
        border: 1px solid #10B981;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE BUILDING BLOCKS SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class BuildingBlocksSimulator:
    """
    Simulates the core building blocks of LangGraph:
    StateGraph builder, START/END nodes, fixed edges, conditional edges,
    and partial state delta updates.
    """
    def __init__(self):
        self.blocks = [
            "StateGraph(Schema)",
            "START Virtual Node",
            "END Virtual Node",
            "add_node('name', fn)",
            "add_edge('A', 'B')",
            "add_conditional_edges('A', router, map)",
            "Channel Reducer (State Updates)"
        ]

    def simulate_pipeline(self, user_score: int, user_category: str) -> Dict[str, Any]:
        state = {
            "score": user_score,
            "category": user_category,
            "decision": "",
            "audit_trail": []
        }

        # Step 1: START -> evaluate_node
        state["audit_trail"].append("1. START -> evaluate_node")
        
        # Step 2: Router conditional edge
        if state["score"] >= 70:
            branch = "fast_track"
            state["audit_trail"].append("2. evaluate_node -> fast_track (Score >= 70)")
            state["decision"] = "FAST_TRACK_APPROVED"
        else:
            branch = "manual_review"
            state["audit_trail"].append("3. evaluate_node -> manual_review (Score < 70)")
            state["decision"] = "PENDING_MANUAL_REVIEW"

        # Step 3: Branch -> END
        state["audit_trail"].append(f"4. {branch} -> END")
        
        return state


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("3.2 Core Building Blocks: StateGraph, START, END, Nodes, Edges, & Updates")
st.caption("Chapter 3: LangGraph Core Concepts | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
Building robust multi-agent systems requires mastering the seven foundational **Building Blocks** of LangGraph. 
Rather than wrestling with high-level opaque agent wrappers, LangGraph provides explicit, transparent components 
that give developers 100% deterministic control over every decision boundary and state mutation.

In this lesson, you will master:
* **`StateGraph`**: The central blueprint builder coordinating schemas and vertices.
* **`START` and `END`**: The reserved virtual boundaries of graph execution.
* **Nodes**: Functional workers transforming state.
* **Fixed vs Conditional Edges**: Deterministic pipelines versus dynamic runtime routers.
* **State Updates**: Partial dictionary returns and channel reducer mechanics.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Core Building Blocks & State Channel Composition", divider="rainbow")

st.markdown("""
The architecture below illustrates how the seven building blocks interlock to construct a complete agent state machine. 
Every element plays a precise architectural role: from schema binding to dynamic branching.
""")

blueprint_dot = """
digraph BuildingBlocks {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_builder {
        label = "Block 1: Blueprint Foundation";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        StateGraphBuilder [label="StateGraph(AgentState)\\n(Defines state channels & schema types)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_vertices {
        label = "Blocks 2, 3, 4: Boundaries & Vertices";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        StartNode [label="Block 2: START\\n(Ingress Entrypoint)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        ComputeNode [label="Block 4: Node (add_node)\\n(Executable worker: fn(state) -> delta)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        EndNode [label="Block 3: END\\n(Egress Terminal Boundary)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    subgraph cluster_edges {
        label = "Blocks 5, 6, 7: Transitions & Reducers";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        FixedEdge [label="Block 5: Fixed Edge (add_edge)\\n(Unconditional transition)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        CondEdge [label="Block 6: Conditional Edge (add_conditional_edges)\\n(Dynamic routing based on state values)", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        StateReducer [label="Block 7: State Update & Reducer\\n(Reconciles partial dictionary delta into State)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
    }

    StateGraphBuilder -> StartNode [label="Registers Schema"];
    StartNode -> FixedEdge [label="Initial Wiring"];
    FixedEdge -> ComputeNode;
    ComputeNode -> StateReducer [label="Emits Delta {'result': ...}"];
    StateReducer -> CondEdge [label="Evaluates Updated State"];
    CondEdge -> ComputeNode [label="Cyclic Branch", color="#38BDF8"];
    CondEdge -> EndNode [label="Terminal Branch", color="#EC4899", penwidth=1.8];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - STATEGRAPH
# ==============================================================================
st.header("2. StateGraph: The Core State Machine Builder & Schema Registry", divider="rainbow")

st.markdown("""
**`StateGraph`** is the master class used to construct applications in LangGraph. 
When instantiating `StateGraph(StateSchema)`, you establish the **State Contract** that binds all nodes and edges:
""")

col_sg1, col_sg2 = st.columns([1, 1])

with col_sg1:
    st.markdown("""
    ### Architectural Responsibilities
    * **Channel Allocation**: Inspects the schema type annotations to set up communication channels.
    * **Reducer Registration**: Identifies any channels with custom reducers (`Annotated[list, add_messages]`).
    * **Registry of Vertices & Edges**: Stores all registered node functions and adjacency lists prior to compilation.
    """)

with col_sg2:
    st.markdown("""
    ### Instantiation Syntax
    """)
    st.code("""
from typing import TypedDict
from langgraph.graph import StateGraph

class MyState(TypedDict):
    query: str
    response: str

# Instantiate the builder
builder = StateGraph(MyState)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - START
# ==============================================================================
st.header("3. START Node: Entry Gateway, Ingress Channel Binding, and Graph Initialization", divider="rainbow")

st.markdown("""
The **`START`** node marks the ingress point of the graph. 
* It is imported directly from `langgraph.graph import START`.
* In modern LangGraph, every graph must have at least one edge originating at `START`.
* You can connect `START` unconditionally: `builder.add_edge(START, "first_node")` 
  or conditionally: `builder.add_conditional_edges(START, route_on_entry)`.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - END
# ==============================================================================
st.header("4. END Node: Egress Boundary, Execution Termination, and Terminal State Commit", divider="rainbow")

st.markdown("""
The **`END`** node signals that execution has concluded for the active thread invocation:
* When control flow transitions to `END`, LangGraph halts the Pregel execution loop immediately.
* Any uncommitted state is flushed to the checkpointer database.
* The current state dictionary is returned directly to the caller.
* Edge syntax: `builder.add_edge("final_node", END)`.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - NODES
# ==============================================================================
st.header("5. Nodes: Registering Computational Vertices, Tool Nodes, and Async Callables", divider="rainbow")

st.markdown("""
Nodes are registered using **`builder.add_node(name, callable)`**:
""")

col_no1, col_no2 = st.columns([1, 1])

with col_no1:
    st.markdown("""
    ### Node Types
    * **Standard Python Functions**: `def my_node(state): ...`
    * **Asynchronous Coroutines**: `async def my_async_node(state): ...`
    * **Prebuilt Tool Nodes**: `ToolNode(tools)` from `langgraph.prebuilt`.
    * **LangChain Runnables**: Any LCEL runnable (`prompt | model`).
    """)

with col_no2:
    st.markdown("""
    ### Best Practice: Explicit Naming
    * Use descriptive action verbs for node names (`"generate_plan"`, `"execute_query"`, `"review_draft"`) 
      to make LangSmith traces readable.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - EDGES
# ==============================================================================
st.header("6. Edges: Fixed Deterministic Transitions and Unconditional Sequence Wiring", divider="rainbow")

st.markdown("""
A **Fixed Edge** represents an unconditional deterministic transition:
`builder.add_edge("source_node", "destination_node")`

* When `source_node` completes its superstep, `destination_node` is scheduled to execute in the next superstep.
* **Parallel Execution (Fan-Out)**: You can add multiple fixed edges from the same source:
  `builder.add_edge("planner", "worker_a")` and `builder.add_edge("planner", "worker_b")`. 
  LangGraph will execute `worker_a` and `worker_b` concurrently in parallel!
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CONDITIONAL EDGES
# ==============================================================================
st.header("7. Conditional Edges: Dynamic Branching, Router Functions, and Route Dictionaries", divider="rainbow")

st.markdown("""
A **Conditional Edge** allows the graph to branch dynamically based on the state emitted by the preceding node:
""")

col_ce1, col_ce2 = st.columns([1, 1])

with col_ce1:
    st.markdown("""
    ### Three Components of Conditional Edges
    1. **Source Node**: Where the edge begins (`"evaluator"`).
    2. **Router Callable**: A function `fn(state) -> str` that returns a route key.
    3. **Destination Map**: A dictionary mapping the return key to destination node names:
       `{"retry": "coder", "approved": END}`.
    """)

with col_ce2:
    st.markdown("""
    ### Code Example
    """)
    st.code("""
def route_quality(state: CodeState) -> str:
    if state["tests_passed"]:
        return "approved"
    return "retry"

builder.add_conditional_edges(
    "evaluator",
    route_quality,
    {
        "approved": END,
        "retry": "coder"  # Cyclic edge!
    }
)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - STATE UPDATES
# ==============================================================================
st.header("8. State Updates: Channel Reducers, Delta Dictionaries, and Partial Reconciliations", divider="rainbow")

st.markdown("""
A node never replaces the entire state object. It returns a **Partial Dictionary Delta**:
* If `State` has 10 keys, and Node A only modifies `user_email`, it returns `{"user_email": "new@example.com"}`.
* LangGraph applies the update according to the channel reducer defined in the state schema:
  * **Default (No Reducer)**: Overwrites the key with the new value.
  * **`add_messages` Reducer**: Appends messages while updating existing messages that share the same ID.
  * **Custom Reducers (`Annotated[list, lambda a, b: a + b]`)**: Appends or merges custom lists.
""")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Multi-Branch StateGraph Building Blocks Pipeline", divider="rainbow")

st.markdown("""
Below is an end-to-end production implementation demonstrating all seven building blocks working in harmony:
""")

prod_code = """
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Block 1: State Definition with Channel Reducer
class ApprovalState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    claim_amount: float
    risk_score: int
    decision: str

# Block 4: Node Functions
def triage_node(state: ApprovalState) -> Dict[str, Any]:
    # Returns partial delta
    risk = 85 if state["claim_amount"] > 10000 else 25
    return {"risk_score": risk}

def auto_approve_node(state: ApprovalState) -> Dict[str, Any]:
    return {"decision": "AUTO_APPROVED"}

def manual_review_node(state: ApprovalState) -> Dict[str, Any]:
    return {"decision": "ESCALATED_TO_HUMAN"}

# Block 6: Router Function for Conditional Edge
def route_risk(state: ApprovalState) -> str:
    if state["risk_score"] > 50:
        return "escalate"
    return "approve"

# Block 1: StateGraph Builder
builder = StateGraph(ApprovalState)

# Block 4: Register Nodes
builder.add_node("triage", triage_node)
builder.add_node("auto_approve", auto_approve_node)
builder.add_node("manual_review", manual_review_node)

# Block 2 & 5: Fixed Edge from START
builder.add_edge(START, "triage")

# Block 6: Conditional Edge with Destination Mapping
builder.add_conditional_edges(
    "triage",
    route_risk,
    {
        "approve": "auto_approve",
        "escalate": "manual_review"
    }
)

# Block 3 & 5: Fixed Edges to END
builder.add_edge("auto_approve", END)
builder.add_edge("manual_review", END)

# Compilation
app = builder.compile()

# Invocation
result = app.invoke({"claim_amount": 15000.0, "messages": []})
print("Decision Result:", result["decision"])
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Building Blocks Assembler & Branching Simulator", divider="rainbow")

st.markdown("""
Simulate how the seven building blocks execute in a live decision engine. 
Adjust user scores and observe how conditional edges evaluate state deltas and route to distinct terminal outcomes.
""")

sim = BuildingBlocksSimulator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Ingress Parameters")
    
    score_slider = st.slider("User Credit / Qualification Score", min_value=0, max_value=100, value=75)
    category_select = st.selectbox("Application Category", ["Corporate Loan", "Retail Mortgage", "Personal Credit"], index=0)

    sim_btn = st.button("Execute Building Blocks Pipeline 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. State Transition & Edge Route Inspector")
    
    if sim_btn:
        exec_result = sim.simulate_pipeline(score_slider, category_select)
        
        m1, m2 = st.columns(2)
        m1.metric("Final Decision", exec_result["decision"])
        m2.metric("Supersteps", len(exec_result["audit_trail"]))

        st.markdown("#### Transition Trail (Edges & Nodes)")
        for step in exec_result["audit_trail"]:
            st.markdown(f"• `<span class='badge-block'>{step}</span>`", unsafe_allow_html=True)

        st.markdown("#### Final Output State")
        st.json(exec_result)
    else:
        st.info("Click 'Execute Building Blocks Pipeline' to simulate routing.")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Interview questions focused on core primitives, conditional routing mechanics, and edge cases:
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What is the difference between a fixed edge and a conditional edge in LangGraph?"):
    st.markdown("""
    **Answer**: A fixed edge (`add_edge`) represents an unconditional transition: whenever the source node completes, 
    the destination node is scheduled deterministically. A conditional edge (`add_conditional_edges`) introduces dynamic 
    branching: it executes a Python router callable that inspects the updated state dictionary and returns a string key, 
    which is resolved against a destination mapping dictionary to determine the next node (or `END`).
    """)

with st.expander("Q2: How do you implement parallel execution (fan-out) using normal edges?"):
    st.markdown("""
    **Answer**: To implement fan-out, add multiple outgoing fixed edges from the same source node:
    `builder.add_edge("supervisor", "researcher_a")` and `builder.add_edge("supervisor", "researcher_b")`. 
    During the next Pregel superstep, LangGraph schedules both `researcher_a` and `researcher_b` to run concurrently.
    """)

with st.expander("Q3: What happens if a conditional router function returns a key not present in the destination map?"):
    st.markdown("""
    **Answer**: If a router returns a key not declared in the mapping dictionary, LangGraph raises a runtime `KeyError` 
    or invalid edge error during execution. Best practice is to provide a fallback or ensure the router function strictly 
    returns validated enum members.
    """)

with st.expander("Q4: Why does a node return a partial dictionary instead of the full state?"):
    st.markdown("""
    **Answer**: Returning partial dictionaries enforces functional purity and enables concurrent node execution. 
    If two parallel worker nodes returned full state objects, they would overwrite each other's changes. 
    By returning only the keys they modified, LangGraph's channel reducers merge both partial updates cleanly without data loss.
    """)

with st.expander("Q5: Can you wire an edge from `START` directly to `END`?"):
    st.markdown("""
    **Answer**: Yes! While unusual, `builder.add_edge(START, END)` is syntactically valid and creates an immediate no-op graph. 
    More practically, conditional edges from `START` can route directly to `END` if ingress input fails validation:
    `builder.add_conditional_edges(START, validate_input, {"valid": "process", "invalid": END})`.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Omitting the Destination Mapping in Conditional Edges (CRITICAL)**
    * *Error*: Calling `builder.add_conditional_edges("node", router_fn)` without the destination dict.
    * *Impact*: In older versions, LangGraph attempted to match the string directly to node names; if the router returned a status instead of a node name, execution crashed.
    * *Fix*: Always pass an explicit mapping dictionary: `{"success": "next_node", "fail": "retry_node"}`.
    """)

    st.error("""
    **Trap 2: Creating Unreachable Dead-End Nodes (CRITICAL)**
    * *Error*: Defining a node that has incoming edges but no outgoing edge to another node or `END`.
    * *Impact*: When execution reaches this node, the graph halts without reaching `END`, leaving state unfinalized.
    * *Fix*: Ensure every registered node has an outgoing edge or routes to `END`.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Modifying State Keys Without Declaring Them in the Schema (WARNING)**
    * *Warning*: Returning `{"untracked_key": 123}` when `untracked_key` is not in the `TypedDict`.
    * *Impact*: May be discarded or cause serialization errors in strict Pydantic checkpointers.
    * *Fix*: Always declare all state keys in the schema.
    """)

    st.warning("""
    **Trap 4: Heavy In-Memory Computations Inside Router Callables (WARNING)**
    * *Warning*: Running slow LLM calls or network requests inside the conditional routing function.
    * *Impact*: Router functions should be lightweight, synchronous routing logic. Heavy work belongs inside dedicated Nodes!
    * *Fix*: Execute the LLM inside a Node; have the router simply inspect the node's output string.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: StateGraph</strong><br><br>
        <em>Front:</em> What does StateGraph take as its first argument?<br><br>
        <em>Back:</em> The State Schema (e.g. <code>TypedDict</code> or Pydantic model).
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: START Node</strong><br><br>
        <em>Front:</em> What is START in LangGraph?<br><br>
        <em>Back:</em> The reserved entrypoint constant indicating where execution begins.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Conditional Edge</strong><br><br>
        <em>Front:</em> What does a conditional router function return?<br><br>
        <em>Back:</em> A string key that is mapped to a destination node or END.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Partial Update</strong><br><br>
        <em>Front:</em> What must a node function return?<br><br>
        <em>Back:</em> A dictionary containing only the keys that were modified (partial delta).
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LangGraph building blocks:")

chk1 = st.checkbox("I know how to instantiate `StateGraph` with a typed state schema.", value=True)
chk2 = st.checkbox("I understand the roles of `START` and `END` virtual node constants.", value=True)
chk3 = st.checkbox("I know how to register node functions using `builder.add_node()`.", value=True)
chk4 = st.checkbox("I can differentiate between fixed edges and conditional edges.", value=True)
chk5 = st.checkbox("I know how to write router functions with destination mapping dictionaries.", value=True)
chk6 = st.checkbox("I understand how partial dictionary updates are merged by channel reducers.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 3.2: Core Building Blocks! Proceed to Lesson 3.3: Graph Execution.")
