import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="1.1 What is LangGraph? - LangGraph Curriculum",
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
    .badge-edge {
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
# MODULE-SCOPE STATEGRAPH ENGINE FOR WORKBENCH & PRODUCTION PIPELINE
# ==============================================================================
class MockStateGraph:
    """
    Module-scope simulator of LangGraph cyclic state machine runtime.
    Implements nodes, conditional routing, state reduction, and execution loop tracking.
    """
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Any] = {}
        
    def add_node(self, name: str, func: Any):
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, condition_func: Any, mapping: Dict[str, str]):
        self.conditional_edges[from_node] = (condition_func, mapping)

    def run(self, initial_state: Dict[str, Any], max_iterations: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        current_state = dict(initial_state)
        current_node = "agent"
        execution_trace = []
        iteration = 0

        while current_node != "END" and iteration < max_iterations:
            iteration += 1
            node_fn = self.nodes.get(current_node)
            state_update = node_fn(current_state) if node_fn else {}
            current_state.update(state_update)

            step_record = {
                "iteration": iteration,
                "node": current_node,
                "state_snapshot": dict(current_state),
                "next_action": ""
            }

            if current_node in self.conditional_edges:
                cond_fn, mapping = self.conditional_edges[current_node]
                decision = cond_fn(current_state)
                next_node = mapping.get(decision, "END")
                step_record["next_action"] = f"Branch condition evaluated to '{decision}' -> Next: {next_node}"
                current_node = next_node
            elif current_node in self.edges:
                next_node = self.edges[current_node]
                step_record["next_action"] = f"Fixed edge -> Next: {next_node}"
                current_node = next_node
            else:
                step_record["next_action"] = "Terminated (END)"
                current_node = "END"

            execution_trace.append(step_record)

        return current_state, execution_trace


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("1.1 What is LangGraph? Definition, Graph Runtimes, & Agent Orchestration")
st.caption("Chapter 1: Introduction to LangGraph | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
Modern Artificial Intelligence applications have evolved past simple prompt-response chains and rigid Directed 
Acyclic Graphs (DAGs). Real-world autonomous agents require **cyclical reasoning**: generating an action, inspecting 
tool output, diagnosing execution errors, reflecting on partial progress, and looping backward to try an improved approach.

**LangGraph** is a low-level, high-control orchestration runtime built specifically for stateful, multi-actor LLM applications. 
By modeling agent computation as a **Graph of Nodes and Edges** governing a centralized, typed **State Schema**, LangGraph 
provides the durable execution primitives missing in traditional workflow engines: **Cyclic Loops**, **Message Channel Reducers**, 
**Thread Persistence**, **Deterministic Human-in-the-Loop Interrupts**, and **Multi-Agent Hierarchies**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Cyclic StateGraph Agent Execution Runtime", divider="rainbow")

st.markdown("""
The architecture below illustrates how LangGraph executes an autonomous agent cycle. Unlike linear DAGs, the control flow 
can route dynamically back to previous nodes based on runtime evaluation, tool results, or human approval.
""")

blueprint_dot = """
digraph LangGraphRuntime {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_ingress {
        label = "1. Graph Ingress & Thread Initialization";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        StartNode [label="START\\n(Thread Configuration & Initial State)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Checkpointer [label="State Checkpointer (Memory / Postgres)\\n(Persists snapshot at thread_id checkpoint)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_reasoning {
        label = "2. Agent Reasoning & Tool Invocation Cycle";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        AgentNode [label="Agent Reasoning Node\\n(LLM evaluates query & chooses tools)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        RouterCondition [label="Conditional Routing Edge\\n(tools_called? True / False)", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        ToolNode [label="Tool Execution Node\\n(Executes SQL, Web Search, or APIs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        HumanInterrupt [label="Human-in-the-Loop Node (interrupt)\\n(Pauses graph for external supervisor sign-off)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    subgraph cluster_egress {
        label = "3. Channel Reduction & Termination";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#8B5CF6"; fontsize = 11;
        Reducer [label="State Channel Reducer (add_messages)\\n(Appends tool output to message history)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
        EndNode [label="END\\n(Returns Final Synthesized Answer)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
    }

    StartNode -> Checkpointer [label="1. Init Snapshot"];
    Checkpointer -> AgentNode [label="2. Dispatch State"];
    AgentNode -> RouterCondition [label="3. Inspect Delta"];

    RouterCondition -> ToolNode [label="Tools Required", color="#10B981"];
    RouterCondition -> EndNode [label="Final Response Generated", color="#8B5CF6"];
    RouterCondition -> HumanInterrupt [label="High-Risk Action", color="#EC4899", style="dashed"];

    HumanInterrupt -> ToolNode [label="Approved by User", color="#10B981"];
    ToolNode -> Reducer [label="4. Emit Tool Message"];
    Reducer -> Checkpointer [label="5. Checkpoint Commit"];
    Checkpointer -> AgentNode [label="6. Cyclic Feedback Loop", color="#38BDF8", penwidth=2.0];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - DEFINITION OF LANGGRAPH
# ==============================================================================
st.header("2. Definition of LangGraph: Cyclical Computation, Nodes, & Shared State", divider="rainbow")

st.markdown("""
**LangGraph** is an open-source orchestration library created by Harrison Chase and the LangChain team designed to build 
stateful, multi-actor applications with Large Language Models. 

At its mathematical and architectural foundation, LangGraph represents an application as a **Directed Graph**:
* **State ($S$)**: A centralized data structure (schema) shared by all components in the graph. Every node receives this state and returns incremental updates.
* **Nodes ($V$)**: Individual computational units (Python functions, API calls, or LLM prompts) that process state and return modifications.
* **Edges ($E$)**: Control flow paths defining which node executes next. Edges can be fixed (direct transitions) or conditional (dynamic routing functions).
""")

col_def1, col_def2 = st.columns([1, 1])

with col_def1:
    st.markdown("""
    ### Fundamental Graph Invariants
    1. **Single Source of Truth**: All data passing between agents or tools lives in the typed `State`.
    2. **Functional Purity**: Nodes do not mutate the state in place; they return state dictionaries or messages that channel reducers merge.
    3. **First-Class Cyclicity**: Unlike Apache Airflow or standard workflow engines, a node can point back to itself or an earlier node without throwing recursion exceptions.
    """)

with col_def2:
    st.markdown("""
    ### Minimal LangGraph Structure
    """)
    st.code("""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", execute_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # Cyclic loop back!
graph = builder.compile()
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - WHY LANGGRAPH WAS CREATED
# ==============================================================================
st.header("3. Why LangGraph Was Created: The Limitations of Linear Chains & Black-Box Agents", divider="rainbow")

st.markdown("""
In 2023, early Generative AI applications relied heavily on linear chain abstractions (like LangChain's `LLMChain` and 
`SimpleSequentialChain`) or monolithic agent abstractions (like `AgentExecutor`). As enterprise engineering teams attempted 
to move proof-of-concepts into production, these abstractions broke down completely.
""")

col_wh1, col_wh2 = st.columns([1, 1])

with col_wh1:
    st.markdown("""
    ### Failure Modes of Early Abstractions
    * **The Linear Fallacy**: Real business workflows are never linear. They require fallbacks, iterative retries, human sign-off, and loops.
    * **The `AgentExecutor` Black Box**: `AgentExecutor` was an unconfigurable `while True` loop driven by regex parsing. Developers had zero control over step-by-step state modification, checkpointing, or mid-stream rollbacks.
    * **No Thread Memory**: Pausing an agent to wait 24 hours for a manager's email approval required keeping a Python thread running in RAM—an unscalable impossibility.
    """)

with col_wh2:
    st.markdown("""
    ### The LangGraph Breakthrough
    LangGraph extracted the execution engine out of black-box loops and re-architected it around **Pregel** (Google's graph computing model):
    * **Explicit Control Flow**: Every edge and condition is written in native Python.
    * **Durable Persistence**: State is serialized to disk/DB at every superstep.
    * **First-Class Interrupts**: The runtime can suspend execution, cleanly save thread state, and resume hours later.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - LANGGRAPH VS LANGCHAIN
# ==============================================================================
st.header("4. LangGraph vs LangChain: Orchestration Runtime vs Component Abstraction Library", divider="rainbow")

st.markdown("""
A ubiquitous source of confusion in modern enterprise AI engineering is the distinction between LangChain and LangGraph. 
They are not mutually exclusive competitors; they are complementary tiers in the modern GenAI stack.
""")

comp_data = {
    "Architectural Dimension": [
        "Primary Purpose",
        "Control Flow Philosophy",
        "State Management",
        "Persistence & Time-Travel",
        "Human-in-the-Loop",
        "Primary Building Block"
    ],
    "LangChain (LCEL)": [
        "Component interfaces & linear pipelines",
        "Directed Acyclic Graph (DAG) pipelines",
        "Transient dictionary inputs and outputs",
        "In-memory session memory",
        "Not natively supported in pipeline flow",
        "Runnables (`prompt | model | parser`)"
    ],
    "LangGraph": [
        "Cyclic agent runtime & multi-agent coordinator",
        "Cyclic State Machines with dynamic branching",
        "Centralized, durable `TypedDict` / Pydantic state",
        "First-class checkpointers (`MemorySaver`, `SqliteSaver`)",
        "Native `interrupt()` breakpoints & human approval",
        "StateGraph (`Nodes`, `Edges`, `Reducers`)"
    ]
}
st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern enterprise architectures (2025–2026), developers use **LangChain for individual nodes** (e.g., standardizing 
prompt templates, parsing Pydantic outputs, and binding tool schemas to models) and **LangGraph for the overall application runtime** 
(orchestrating the loop between the LLM, the database, the tools, and human review).
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - LANGGRAPH VS TRADITIONAL WORKFLOWS
# ==============================================================================
st.header("5. LangGraph vs Traditional Workflows: Determinism vs Probabilistic Non-Determinism", divider="rainbow")

st.markdown("""
Traditional business process management (BPM) tools (Camunda, Temporal, Zapier) were engineered for **deterministic software**: 
if input is $A$, execute function $B$, and return result $C$. 

LLMs introduce fundamental **probabilistic non-determinism**:
* An LLM may choose Tool A, Tool B, or hallucinate a nonexistent tool.
* A tool might fail due to database syntax errors, requiring the LLM to inspect the traceback and rewrite the SQL query.
* A user query might require 2 reasoning steps or 20 reasoning steps.
""")

col_tw1, col_tw2 = st.columns([1, 1])

with col_tw1:
    st.markdown("""
    ### Why Temporal / Airflow Struggle with LLMs
    * **Rigid Static Graphs**: Traditional workflow schedulers require all task steps to be compiled before execution. They cannot dynamically generate new nodes on the fly.
    * **Heavyweight Overhead**: Spinning up tasks across distributed workers introduces hundreds of milliseconds of latency per hop, destroying conversational real-time responsiveness.
    * **No LLM Context Primitives**: Traditional engines lack concepts of token budgets, message history arrays, and tool call payload parsing.
    """)

with col_tw2:
    st.markdown("""
    ### The LangGraph Hybrid Solution
    LangGraph combines the **durable execution and checkpointing** of engines like Temporal with the **micro-hop speed and cyclic flexibility** demanded by autoregressive models. 
    State transitions execute in sub-millisecond Python memory while persisting durable snapshots to relational storage.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - LANGGRAPH VS STATE MACHINES
# ==============================================================================
st.header("6. LangGraph vs State Machines: Classical FSMs vs Channel-Reduced Agent State", divider="rainbow")

st.markdown("""
Classical **Finite State Machines (FSMs)** possess a fixed set of named states (e.g., `IDLE`, `PROCESSING`, `SUCCESS`, `ERROR`) 
and transition tables. While LangGraph is conceptually a state machine, it differs in critical ways:
""")

col_sm1, col_sm2 = st.columns([1, 1])

with col_sm1:
    st.markdown("""
    ### Classical FSMs
    * **State is an Enum**: The system is exclusively in one state label at a time.
    * **Hardcoded Transitions**: Transitions are fixed rules (e.g., event `BUTTON_CLICK` transitions `OFF` to `ON`).
    * **Zero Internal Memory**: Traditional state machines have no payload memory; all history must be encoded into separate state names.
    """)

with col_sm2:
    st.markdown("""
    ### LangGraph Extended State Machines
    * **State is an Arbitrary Schema**: The state is an expressive data structure containing message arrays, context variables, and document lists.
    * **Dynamic LLM Transitions**: Edges evaluate complex LLM outputs to decide the next path.
    * **Parallel Multi-Node Execution**: Multiple nodes can execute concurrently, broadcasting updates to separate state channels that merge via custom reducers.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - LANGGRAPH VS DAG FRAMEWORKS
# ==============================================================================
st.header("7. LangGraph vs DAG Frameworks: Escaping the Acyclic Constraint", divider="rainbow")

st.markdown("""
A **DAG (Directed Acyclic Graph)** is mathematically forbidden from containing cycles ($A \\to B \\to C \\to A$ is illegal). 
Every major data pipeline engine (Apache Airflow, Prefect, Spark, dbt) is strictly a DAG engine.
""")

dag_cycle_dot = """
digraph DAGvsCycle {
    graph [rankdir=LR, bgcolor="#0B1120", pad="0.2", nodesep="0.3", ranksep="0.4"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
    edge [fontname="Helvetica", fontsize=8, color="#64748B"];

    subgraph cluster_dag {
        label = "Classical DAG (Strictly Linear Forward)";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EF4444";
        D1 [label="Extract", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EF4444"];
        D2 [label="Transform", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EF4444"];
        D3 [label="Load", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EF4444"];
        D1 -> D2 -> D3;
    }

    subgraph cluster_langgraph {
        label = "LangGraph Cyclic State Machine";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981";
        L1 [label="Agent Generate", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        L2 [label="Execute Code / Tool", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        L3 [label="Evaluate Test / Linter", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        L1 -> L2 -> L3;
        L3 -> L1 [label="Error: Self-Correct Loop", color="#38BDF8", penwidth=2.0];
    }
}
"""
st.graphviz_chart(dag_cycle_dot, use_container_width=True)

st.markdown("""
In AI agent engineering, **feedback loops are the cornerstone of autonomous reasoning**. 
When an agent generates Python code that fails unit tests, it must loop back to `Agent Generate`, passing the compiler error 
into its state to write a patch. Attempting to model this in a DAG requires artificially unrolling loops into infinite 
nodes ($A_1 \\to B_1 \\to A_2 \\to B_2 \\dots$), which blows up graph complexity and fails when the loop count is unknown upfront.
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - AGENT ORCHESTRATION RUNTIME
# ==============================================================================
st.header("8. LangGraph as an Agent Orchestration Runtime: Channels, Supersteps, & Pregel", divider="rainbow")

st.markdown("""
LangGraph is built upon the **Actor / Pregel Graph Model** popularized by Google for distributed computing:
""")

col_or1, col_or2 = st.columns([1, 1])

with col_or1:
    st.markdown("""
    ### The Superstep Execution Model
    1. **Step Start**: The runtime inspects all active nodes scheduled for execution.
    2. **Parallel Node Execution**: Scheduled nodes run concurrently in isolated threads.
    3. **State Channel Reductions**: Nodes return state updates. The runtime applies channel reducers (e.g., `add_messages`) to safely merge updates into the unified state.
    4. **Edge Evaluation**: Conditional edges evaluate the new state to schedule the next batch of nodes.
    5. **Checkpoint Commit**: The full state snapshot is flushed to storage with a sequence increment.
    """)

with col_or2:
    st.markdown("""
    ### Built-in Fault Tolerance
    * **Step-Level Retries**: If an API call fails inside Node B, LangGraph re-runs only Node B from its last checkpoint without restarting the entire graph.
    * **Time-Travel Debugging**: Because state snapshots are saved at every superstep, developers can replay, rewind, or fork execution threads from any historical step in LangSmith or LangGraph Studio.
    """)

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - LANGGRAPH AND LLM APPLICATIONS
# ==============================================================================
st.header("9. LangGraph and LLM Applications: From Conversational Bots to Stateful Engines", divider="rainbow")

st.markdown("""
LangGraph elevates basic LLM applications into enterprise-grade conversational systems. 
Consider a customer support assistant for an airline:
""")

col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("""
    ### State Primitives for LLM Apps
    * **Thread Isolation**: Every customer conversation is isolated by a unique `thread_id`.
    * **Session Persistence**: Customers can disconnect their mobile browser, reconnect 3 hours later, and resume the exact same conversation thread with full memory.
    * **Context Pruning**: Reducer channels can automatically truncate older conversational turns or run background summarization nodes to stay within token limits.
    """)

with col_app2:
    st.markdown("""
    ### Human Escapes (Human-in-the-Loop)
    * If a customer demands a refund exceeding $500, the graph halts at a conditional edge.
    * The node calls `interrupt({"reason": "Refund requires human approval"})`.
    * The system alerts a human agent via webhook. Once the agent clicks 'Approve' in their admin dashboard, the graph resumes seamlessly and completes the transaction.
    """)

st.divider()

# ==============================================================================
# SECTION 10: TOPIC 9 - LANGGRAPH AND AI AGENTS
# ==============================================================================
st.header("10. LangGraph and AI Agents: Multi-Agent Hierarchies, Supervisors, & Swarms", divider="rainbow")

st.markdown("""
Single-agent architectures collapse when tasks become complex: a single prompt cannot be an expert in Python coding, 
SQL database querying, and legal compliance simultaneously. 

LangGraph is the industry standard for **Multi-Agent Architectures**:
""")

col_ag1, col_ag2 = st.columns([1, 1])

with col_ag1:
    st.markdown("""
    ### Canonical Multi-Agent Topologies
    1. **Supervisor Architecture**: A central routing agent inspects user goals and delegates sub-tasks to specialized worker agents (e.g., Coder Agent, Researcher Agent).
    2. **Collaborative Peer Swarm**: Agents pass conversational control directly to each other based on mutual agreement (e.g., Writer -> Editor -> Critic).
    3. **Hierarchical Subgraphs**: A node inside a parent graph can itself be a compiled `StateGraph` sub-agent with its own internal state, loops, and checkpointers.
    """)

with col_ag2:
    st.markdown("""
    ### Shared vs Private State
    LangGraph supports **Isolated Subgraph State**:
    * Workers can maintain high-volume temporary scratchpads that never pollute the parent graph's primary dialogue history.
    * Only finalized synthesis artifacts are returned to the root thread.
    """)

st.divider()

# ==============================================================================
# SECTION 11: TOPIC 10 - REAL-WORLD APPLICATIONS
# ==============================================================================
st.header("11. Real-World Applications: Production Deployments across Enterprise Verticals", divider="rainbow")

st.markdown("""
LangGraph powers mission-critical applications across global engineering teams:
""")

app_matrix = [
    {
        "Vertical": "Autonomous Software Engineering",
        "Architecture Pattern": "Code-Test-Fix Cyclic Loop",
        "LangGraph Mechanism": "Planner agent -> Code generator node -> Linter execution -> Pytest node. If tests fail, loops back to Coder with traceback."
    },
    {
        "Vertical": "Financial Due Diligence",
        "Architecture Pattern": "Multi-Document Parallel Fan-out",
        "LangGraph Mechanism": "Supervisor spawns 10 parallel subgraphs (one per 10-K report) via Send API; aggregates metrics into summary table."
    },
    {
        "Vertical": "Clinical Healthcare Assistance",
        "Architecture Pattern": "Human-in-the-Loop Medical Validation",
        "LangGraph Mechanism": "Retrieves medical literature, synthesizes clinical draft, hits `interrupt()` for physician sign-off before EHR transmission."
    },
    {
        "Vertical": "Enterprise Customer Support",
        "Architecture Pattern": "Durable Session Multi-Tool Routing",
        "LangGraph Mechanism": "PostgreSQL checkpointer persists customer conversation state; routes between CRM lookups, ticketing tools, and live chat handoff."
    }
]
st.dataframe(pd.DataFrame(app_matrix), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 12: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("12. Production Implementation: End-to-End Cyclic StateGraph Pipeline", divider="rainbow")

st.markdown("""
Below is an end-to-end, runnable implementation demonstrating the core architectural patterns of LangGraph: 
a typed state schema, nodes returning partial state deltas, a conditional routing edge, and a cyclic self-correction loop.
""")

prod_code = """
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# 1. Define Centralized Type-Safe State
class ResearchState(TypedDict):
    query: str
    search_results: List[str]
    draft_summary: str
    review_status: str
    iteration_count: int

# 2. Define Node Functions (Return partial updates)
def search_node(state: ResearchState) -> Dict[str, Any]:
    print(f"Searching web for: {state['query']}")
    return {
        "search_results": [
            "LangGraph is a library for building stateful multi-actor applications.",
            "Supports cyclic loops, checkpointers, and human interrupts."
        ]
    }

def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    count = state.get("iteration_count", 0) + 1
    draft = f"Draft {count}: LangGraph manages agent state across cycles."
    # Simulate critique
    status = "APPROVED" if count >= 2 else "NEEDS_REVISION"
    return {
        "draft_summary": draft,
        "review_status": status,
        "iteration_count": count
    }

# 3. Define Conditional Routing Function
def check_draft_quality(state: ResearchState) -> str:
    if state["review_status"] == "APPROVED":
        return "approved"
    return "revise"

# 4. Assemble Graph Topology
builder = StateGraph(ResearchState)
builder.add_node("search", search_node)
builder.add_node("synthesize", synthesize_node)

builder.add_edge(START, "search")
builder.add_edge("search", "synthesize")

# Conditional Edge creating the cyclic self-correction loop!
builder.add_conditional_edges(
    "synthesize",
    check_draft_quality,
    {
        "approved": END,
        "revise": "synthesize"  # Cyclic self-loop to rewrite draft!
    }
)

compiled_graph = builder.compile()
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 13: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("13. Interactive Workbench: StateGraph Cyclic Loop & Branching Simulator", divider="rainbow")

st.markdown("""
Interact with the live execution engine simulator below. Watch how state transitions through nodes, 
triggers conditional branches, loops cyclically upon error detection, and terminates cleanly when criteria are met.
""")

# Setup mock graph
sim_graph = MockStateGraph()

def mock_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    iters = state.get("iterations", 0) + 1
    needs_tools = state.get("simulate_tool_requirement", True) and (iters < 2)
    return {
        "iterations": iters,
        "agent_thought": f"Iteration {iters}: Evaluating query '{query}'",
        "has_tools_to_run": needs_tools
    }

def mock_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tool_output": "Executed SQL query: Found 3 relevant customer records.",
        "simulate_tool_requirement": False
    }

def mock_router(state: Dict[str, Any]) -> str:
    return "run_tools" if state.get("has_tools_to_run", False) else "finalize"

sim_graph.add_node("agent", mock_agent_node)
sim_graph.add_node("tools", mock_tool_node)
sim_graph.add_conditional_edges("agent", mock_router, {"run_tools": "tools", "finalize": "END"})
sim_graph.add_edge("tools", "agent")  # Cyclic feedback edge

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Configure Initial Thread State")
    
    user_query = st.text_input("User Prompt", value="Look up customer 8812 and generate billing summary.")
    require_tools = st.checkbox("Simulate Tool Invocation Required", value=True)
    max_iter_slider = st.slider("Max Recursion Limit (Safety Guard)", min_value=1, max_value=6, value=4)

    run_sim_btn = st.button("Execute StateGraph Run 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Live Execution Trace & State Inspector")
    
    if run_sim_btn and user_query:
        init_state = {
            "query": user_query,
            "simulate_tool_requirement": require_tools,
            "iterations": 0
        }
        
        final_state, trace = sim_graph.run(init_state, max_iterations=max_iter_slider)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Supersteps Executed", len(trace))
        m2.metric("Final Node", trace[-1]["node"] if trace else "START")
        m3.metric("Iterations Count", final_state.get("iterations", 0))

        st.markdown("#### Superstep Transition Log")
        for step in trace:
            with st.container():
                st.markdown(f"**Superstep {step['iteration']}** — Node: `<span class='badge-node'>{step['node']}</span>`", unsafe_allow_html=True)
                st.caption(f"Action: {step['next_action']}")
                st.json(step["state_snapshot"])
                st.divider()

        st.success("StateGraph execution finished successfully!")
    else:
        st.info("Click 'Execute StateGraph Run' to simulate the cyclic loop.")

st.divider()

# ==============================================================================
# SECTION 14: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Senior AI Engineer and LLM Architect interviewers frequently evaluate candidates on LangGraph's core primitives, 
superstep execution models, and edge failure modes.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What is the architectural difference between a DAG and a LangGraph cyclic state machine?"):
    st.markdown("""
    **Answer**: A Directed Acyclic Graph (DAG) strictly forbids cycles: every task executes once in topological order. 
    In contrast, LangGraph is a cyclic state machine where nodes can route control flow backward to previous nodes 
    (or to themselves). This enables autonomous agents to iteratively critique, self-correct, and retry actions based on tool outputs, 
    which is mathematically impossible in a standard DAG without unrolling infinite duplicate nodes.
    """)

with st.expander("Q2: How does LangGraph handle state synchronization across parallel branches (Pregel model)?"):
    st.markdown("""
    **Answer**: LangGraph executes on the **Pregel superstep computational model**. When multiple nodes execute concurrently, 
    they receive read-only copies of the current state snapshot. Instead of overwriting global state variables, nodes return 
    partial state dictionaries. During the superstep join phase, LangGraph applies **Channel Reducers** (such as `add_messages` 
    or custom aggregation functions) to merge concurrent updates deterministically into the unified thread state.
    """)

with st.expander("Q3: What is a LangGraph Checkpointer and why is it essential for production deployments?"):
    st.markdown("""
    **Answer**: A Checkpointer (e.g., `MemorySaver`, `SqliteSaver`, `PostgresSaver`) serializes the full state dictionary 
    to a persistent database at the completion of every superstep, indexed by a `thread_id` and checkpoint sequence. 
    This is essential because:
    1. It enables long-running agents to pause indefinitely for human approval without holding active memory threads.
    2. It guarantees durability: if a worker pod crashes mid-execution, the agent resumes from its exact last checkpoint.
    3. It enables 'Time-Travel' debugging: users can rewind state to turn 3 and re-execute a different branch.
    """)

with st.expander("Q4: How do you prevent infinite loops in cyclic agent architectures?"):
    st.markdown("""
    **Answer**: Unbounded cyclic loops can burn thousands of dollars in LLM API calls. LangGraph guards against this using:
    1. **`recursion_limit` Configuration**: A global compile setting (defaulting to 25 supersteps) that raises `GraphRecursionError` if exceeded.
    2. **State Iteration Counters**: Explicitly incrementing an integer counter in state and routing to an `emergency_exit` node if iterations exceed a business threshold.
    3. **Deterministic Fallbacks**: Forcing termination to an escalation node if repeated tool errors match specific error signatures.
    """)

with st.expander("Q5: What is the difference between LangGraph and LangChain Expression Language (LCEL)?"):
    st.markdown("""
    **Answer**: LCEL is an abstraction for composing linear chains (`prompt | model | parser`) where data flows unidirectionally. 
    LangGraph is an execution runtime for cyclic multi-agent coordination. LCEL chains typically serve as the computational logic 
    *inside individual LangGraph nodes*, while LangGraph orchestrates the stateful transitions, persistence, and loops *between* those nodes.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Mutating State In-Place Inside Node Functions (CRITICAL)**
    * *Error*: Directly modifying `state["messages"].append(...)` inside a node instead of returning `{"messages": [new_msg]}`.
    * *Impact*: Bypasses channel reducers, corrupts checkpointer state diffing, and triggers nondeterministic concurrency bugs.
    * *Fix*: Always treat node state as immutable; return only the delta dictionary to be merged.
    """)

    st.error("""
    **Trap 2: Omission of Thread Config IDs in Multi-Tenant Deployments (CRITICAL)**
    * *Error*: Calling `graph.invoke(input)` without specifying `{"configurable": {"thread_id": "user_123"}}`.
    * *Impact*: In persistent checkpointers, all user requests collide on a global default thread, leaking private chat memory across tenants.
    * *Fix*: Mandate `thread_id` injection in gateway middleware for every single invocation.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Unhandled Recursion Limits Crashing Production Servers (WARNING)**
    * *Warning*: Relying solely on default `recursion_limit=25` without catching `GraphRecursionError`.
    * *Impact*: Uncaught exceptions return HTTP 500 crashes to end-users instead of graceful fallback messages.
    * *Fix*: Wrap execution in try-except blocks and return a fallback recovery node output.
    """)

    st.warning("""
    **Trap 4: Over-Stuffing Centralized State with Ephemeral Scratchpads (WARNING)**
    * *Warning*: Storing multi-megabyte intermediate raw HTML dumps or vector arrays directly in the main message state.
    * *Impact*: Massive serialization overhead in PostgreSQL checkpointers and rapid context window exhaustion.
    * *Fix*: Use isolated subgraphs or persist large binary payloads in S3/Redis, passing only lightweight IDs in state.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: Superstep</strong><br><br>
        <em>Front:</em> What is a superstep in LangGraph execution?<br><br>
        <em>Back:</em> A single atomic execution cycle where active nodes run concurrently, merge state via reducers, and checkpoint.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Reducer</strong><br><br>
        <em>Front:</em> What does the <code>add_messages</code> channel reducer do?<br><br>
        <em>Back:</em> Appends new messages while updating existing messages that share the same <code>id</code>.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Conditional Edge</strong><br><br>
        <em>Front:</em> How does a conditional edge differ from a standard edge?<br><br>
        <em>Back:</em> A conditional edge runs a Python routing function on state to dynamically choose the destination node.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Interrupt</strong><br><br>
        <em>Front:</em> What happens when a node calls <code>interrupt()</code>?<br><br>
        <em>Back:</em> Execution pauses cleanly, state is checkpointed, and the graph waits for an external resume command.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 15: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("15. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LangGraph fundamentals:")

chk1 = st.checkbox("I understand the architectural definition of LangGraph as a cyclic state machine runtime.", value=True)
chk2 = st.checkbox("I can explain why traditional DAG frameworks (Airflow) fail when modeling autonomous agent loops.", value=True)
chk3 = st.checkbox("I understand the difference between LangChain (component library) and LangGraph (orchestration runtime).", value=True)
chk4 = st.checkbox("I know how the Pregel superstep execution model synchronizes state across parallel nodes.", value=True)
chk5 = st.checkbox("I understand how persistent checkpointers enable human-in-the-loop interrupts and time-travel debugging.", value=True)
chk6 = st.checkbox("I know how to prevent infinite recursion loops using recursion limits and iteration counters.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 1.1: What is LangGraph! Proceed to Lesson 1.2: Why LangGraph is Important.")
