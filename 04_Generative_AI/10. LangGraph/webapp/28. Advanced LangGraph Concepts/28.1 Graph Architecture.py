import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="28.1 Graph Architecture - LangGraph Curriculum",
    page_icon="🕸️",
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
    .badge-topo {
        background-color: #312E81;
        color: #C7D2FE;
        border: 1px solid #6366F1;
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
class AdvancedGraphTopologySimulator:
    """
    Simulates advanced graph architectures:
    - Dynamic routing via Command(goto=...)
    - Nested subgraphs with namespace isolation
    - Cyclic self-correction recursion
    """
    def __init__(self):
        self.superstep_trace = []

    def run_cyclic_simulation(self, quality_threshold: float, simulate_recursion_limit: bool) -> List[Dict[str, Any]]:
        trace = []
        current_score = 0.55
        
        # Step 0: Initial Draft
        trace.append({"step": 1, "node": "generator", "score": current_score, "decision": "Generate initial draft"})
        
        step_idx = 2
        max_steps = 4 if simulate_recursion_limit else 6
        
        while current_score < quality_threshold and step_idx <= max_steps:
            # Critic Node evaluates
            trace.append({"step": step_idx, "node": "evaluator", "score": current_score, "decision": f"Score {current_score:.2f} < Threshold {quality_threshold:.2f} -> Loop back"})
            step_idx += 1
            
            if simulate_recursion_limit and step_idx >= max_steps:
                trace.append({"step": step_idx, "node": "__system__", "score": current_score, "decision": "GraphRecursionError: recursion_limit exceeded!"})
                break
                
            # Generator refines
            current_score = min(current_score + 0.18, 0.98)
            trace.append({"step": step_idx, "node": "generator", "score": current_score, "decision": f"Refined draft with score {current_score:.2f}"})
            step_idx += 1
            
        if current_score >= quality_threshold:
            trace.append({"step": step_idx, "node": "evaluator", "score": current_score, "decision": f"Score {current_score:.2f} >= {quality_threshold:.2f} -> Route to END"})
            
        return trace

topo_sim = AdvancedGraphTopologySimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Graph Topologies, Cyclic State Machines, and Hierarchical Composition", divider="rainbow")

st.markdown("""
Advanced LangGraph engineering shifts from simple linear directed acyclic graphs (DAGs) to **hierarchical, cyclic state machines**. 
Modern enterprise agents require dynamic runtime routing, nested subgraphs with isolated state namespaces, 
and cyclic self-correction loops. LangGraph's architecture natively models cycles without infinite execution deadlocks, 
enforcing deterministic superstep barriers and configurable recursion boundaries.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedArchitectureBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_parent {
            label="1. Parent Orchestrator Graph (Enterprise Supervisor)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            parent_start [label="START\\nUser Request Ingress", fillcolor="#0F172A"];
            supervisor [label="Supervisor Router\\nDynamic Routing via Command(goto=...)", fillcolor="#312E81"];
            parent_end [label="END\\nFinal Solution Delivery", fillcolor="#0F172A"];
        }

        subgraph cluster_subgraph {
            label="2. Encapsulated Nested Subgraph (Research Team)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            sub_entry [label="Subgraph START\\nChild Namespace: ('researcher')", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            sub_fetch [label="Web Retriever Node", fillcolor="#1E293B"];
            sub_summarize [label="Document Synthesizer Node", fillcolor="#1E293B"];
            sub_exit [label="Subgraph END\\nReturns Channel Delta to Parent", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_cycle {
            label="3. Cyclic Self-Correction Sub-Topology";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            generator [label="Draft Generator Node", fillcolor="#1E293B"];
            critic [label="Critic Evaluator Node\\nScore < Threshold: Loop\\nScore >= Threshold: Complete", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        parent_start -> supervisor;
        supervisor -> sub_entry [label="Dispatch to Subgraph"];
        sub_entry -> sub_fetch -> sub_summarize -> sub_exit;
        sub_exit -> generator [label="Research Results"];

        generator -> critic [label="Evaluate Draft"];
        critic -> generator [label="Cycle: Score < 0.85"];
        critic -> parent_end [label="Pass: Score >= 0.85"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Dynamic Graphs
# ==============================================================================
st.header("2. Dynamic Graphs: Runtime Edge Resolution, Command(goto=...), and Adaptive Topologies", divider="rainbow")

st.markdown("""
In traditional workflows, graph topologies are statically compiled. LangGraph's modern **Command API** introduces 
**Dynamic Graphs**, where a node can dynamically decide its destination node at runtime without pre-declaring 
every possible conditional edge.

### Core Dynamic Mechanics:
- **`Command(goto="node_name", update={...})`**: Simultaneously mutates state channels and routes execution to any named node.
- **Dynamic Multi-Dispatch**: `goto` accepts a list of node names (e.g., `Command(goto=["worker_a", "worker_b"])`), 
  spawning dynamic parallel fan-out on-the-fly.
""")

st.code('''from langgraph.types import Command

def dynamic_router_node(state: AgentState) -> Command:
    complexity = state.get("complexity", "low")
    
    if complexity == "high":
        # Dynamically spawn two parallel research workers
        return Command(
            goto=["deep_web_researcher", "database_analyst"],
            update={"status": "parallel_research_active"}
        )
    else:
        # Route directly to simple synthesizer
        return Command(
            goto="simple_synthesizer",
            update={"status": "fast_path"}
        )
''', language="python")

# ==============================================================================
# SECTION 3: Dynamic Routing
# ==============================================================================
st.header("3. Dynamic Routing: Multi-Branch Conditional Edges, State-Aware Dispatch, and Router Invariants", divider="rainbow")

st.markdown("""
Dynamic routing enables the graph to adapt to probabilistic model outputs:
- **Exhaustive Destination Mapping**: When defining `add_conditional_edges(source, router_func, path_map)`, 
  `path_map` must map all possible return strings to valid destination nodes.
- **Pure Callable Functions**: Conditional routing functions should remain pure functions of state without side effects, 
  ensuring deterministic replay during time travel.
""")

# ==============================================================================
# SECTION 4: Nested Graphs
# ==============================================================================
st.header("4. Nested Graphs: Subgraph Isolation, Parent-Child Boundaries, and Encapsulated Workflows", divider="rainbow")

st.markdown("""
As agent systems scale, monolithic state graphs become unmaintainable. LangGraph enables **Nested Graphs (Subgraphs)**, 
where a compiled `StateGraph` is added directly as a node inside a parent graph.

### Subgraph Architecture:
- **Independent State Schemas**: A subgraph can define its own unique state schema (`ResearchState`), completely isolated 
  from the parent state (`OrchestratorState`).
- **Namespace Isolation**: Checkpoints for subgraphs are stored under partitioned namespaces: 
  `checkpoint_ns = ("parent", "subgraph_node_name")`.
- **Encapsulated Compilation**: Subgraphs compile independently with their own checkpointers or inherit the parent checkpointer.
""")

st.code('''# Define Child Subgraph
child_builder = StateGraph(ChildState)
child_builder.add_node("step_a", step_a_node)
child_builder.add_edge(START, "step_a")
child_builder.add_edge("step_a", END)
compiled_child = child_builder.compile()

# Embed into Parent Graph
parent_builder = StateGraph(ParentState)
parent_builder.add_node("research_team", compiled_child)  # Subgraph added as regular node
parent_builder.add_edge(START, "research_team")
parent_builder.add_edge("research_team", END)
compiled_parent = parent_builder.compile()
''', language="python")

# ==============================================================================
# SECTION 5: Graph Composition
# ==============================================================================
st.header("5. Graph Composition: Combining Independent Graphs, Modular Reusability, and Namespace Federation", divider="rainbow")

st.markdown("""
Graph composition allows organizations to build modular libraries of reusable agent sub-topologies:
- **Verification Graph**: Reusable across code generation, legal drafting, and financial analysis.
- **RAG Subgraph**: Standardized retrieval pipeline shared across multiple customer-facing assistants.
- **Federated Governance**: Distinct engineering teams can own and test their respective subgraphs independently.
""")

# ==============================================================================
# SECTION 6: Graph Recursion
# ==============================================================================
st.header("6. Graph Recursion: Controlling Superstep Depth, recursion_limit Thresholds, and Infinite Loop Prevention", divider="rainbow")

st.markdown("""
Because LangGraph supports cyclic edges, runaway loops must be bounded:
- **`recursion_limit` Configuration**: Configured in `RunnableConfig` (e.g., `config={"recursion_limit": 50}`). 
  Defaults to 25 supersteps.
- **`GraphRecursionError`**: Raised automatically when superstep count reaches the threshold, terminating execution 
  before burning unbounded tokens.
""")

# ==============================================================================
# SECTION 7: Cyclic Workflows
# ==============================================================================
st.header("7. Cyclic Workflows: Self-Correction Loops, Critic-Refiner Patterns, and Convergence Criteria", divider="rainbow")

st.markdown("""
Cyclic workflows underpin the most capable reasoning architectures:
- **Generator -> Critic -> Generator Cycle**: A generator creates an initial artifact; a critic evaluates 
  correctness against rubrics. If flaws are found, the state loops back with critique feedback until convergence is achieved.
""")

with st.expander("🔍 View Cyclic Self-Correction State Machine Flow", expanded=True):
    cycle_dot = """
    digraph SelfCorrectionCycle {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        draft_in [label="Generator Node\\nProduces Draft Code / Plan", fillcolor="#0F172A"];
        critic_node [label="Critic Evaluator Node\\nRuns Tests & Computes Quality Score", fillcolor="#312E81"];
        eval_gate [label="Quality Gate Decision\\nScore >= 0.85?", fillcolor="#831843", fontcolor="#FBCFE8"];
        pass_end [label="END\\nVerified Deliverable", fillcolor="#064E3B", fontcolor="#A7F3D0"];

        draft_in -> critic_node;
        critic_node -> eval_gate;
        eval_gate -> draft_in [label="Loop: Score < 0.85\\n(Injects Critique Feedback)"];
        eval_gate -> pass_end [label="Exit: Score >= 0.85"];
    }
    """
    st.graphviz_chart(cycle_dot, use_container_width=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Hierarchical Composite Graph Pipeline with Subgraphs and Cycles", divider="rainbow")

st.markdown("""
The following complete module implements a hierarchical composite graph featuring an isolated research subgraph 
and a cyclic generator-critic loop.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class SubgraphState(TypedDict):
    query: str
    raw_notes: List[str]

class ParentState(TypedDict):
    task: str
    research_summary: str
    draft: str
    critique_score: float

# Subgraph Implementation
def research_fetch_node(state: SubgraphState) -> dict:
    return {"raw_notes": [f"Note A for {state['query']}", f"Note B for {state['query']}"]}

sub_builder = StateGraph(SubgraphState)
sub_builder.add_node("fetcher", research_fetch_node)
sub_builder.add_edge(START, "fetcher")
sub_builder.add_edge("fetcher", END)
compiled_research_subgraph = sub_builder.compile()

# Parent Cyclic Workflow
def generator_node(state: ParentState) -> dict:
    score = state.get("critique_score", 0.0)
    return {"draft": f"Draft iteration (score was {score:.2f})"}

def critic_node(state: ParentState) -> dict:
    prev = state.get("critique_score", 0.0)
    new_score = min(prev + 0.35, 0.95)
    return {"critique_score": new_score}

def evaluate_draft_route(state: ParentState) -> str:
    return END if state["critique_score"] >= 0.80 else "generator"

parent_builder = StateGraph(ParentState)
parent_builder.add_node("generator", generator_node)
parent_builder.add_node("critic", critic_node)

parent_builder.add_edge(START, "generator")
parent_builder.add_edge("generator", "critic")
parent_builder.add_conditional_edges("critic", evaluate_draft_route)

production_composite_graph = parent_builder.compile(checkpointer=MemorySaver())
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Dynamic Graph Topology Simulator & Subgraph Invocation Tracer", divider="rainbow")

st.markdown("""
Simulate cyclic self-correction loops, observe convergence behavior, and test `recursion_limit` threshold triggers.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Cyclic Loop Simulation Controls")
    threshold = st.slider("Quality Threshold Required to Exit", min_value=0.60, max_value=0.95, value=0.85, step=0.05)
    test_limit = st.toggle("Simulate Exceeding recursion_limit", value=False)
    
    if st.button("Execute Cyclic Graph Simulation", type="primary"):
        st.session_state["cycle_ran"] = True

with col_wb2:
    st.subheader("Superstep Execution Trace")
    trace = topo_sim.run_cyclic_simulation(threshold, test_limit)
    
    df_trace = pd.DataFrame(trace)
    st.dataframe(df_trace, use_container_width=True)
    
    last_event = trace[-1]
    if "GraphRecursionError" in last_event["decision"]:
        st.error(f"Recursion Aborted: {last_event['decision']}")
    else:
        st.success(f"Convergence Achieved in {len(trace)} supersteps! Final score: {last_event['score']:.2f}")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How do subgraphs maintain state isolation from parent graphs in LangGraph?"):
    st.markdown("""
    **Answer:**  
    Subgraphs can define independent `TypedDict` state schemas. When a parent node invokes a compiled subgraph, 
    it maps parent channels to the subgraph's input channels. The child runs inside its own isolated namespace 
    `checkpoint_ns = ("parent", "child_node")`, preventing child intermediate scratchpad variables from polluting the parent state.
    """)

with st.expander("Q2: What is the primary difference between Command(goto=...) and traditional conditional edges?"):
    st.markdown("""
    **Answer:**  
    Traditional conditional edges require pre-defining a router function and static mapping dictionary on the graph builder. 
    `Command(goto=...)` allows the node function itself to determine the next destination node dynamically at runtime while simultaneously 
    returning state updates, enabling dynamic multi-dispatch and adaptive topologies.
    """)

with st.expander("Q3: How does LangGraph detect and prevent infinite loops in cyclic graphs?"):
    st.markdown("""
    **Answer:**  
    LangGraph tracks the superstep execution counter per invocation. If the counter reaches the configured `recursion_limit` 
    (default 25) without reaching `END` or an interrupt, the runtime raises a `GraphRecursionError`, halting execution.
    """)

with st.expander("Q4: How do checkpointers handle subgraphs during time travel and rollbacks?"):
    st.markdown("""
    **Answer:**  
    Checkpointers index snapshots using a compound key: `(thread_id, checkpoint_ns, checkpoint_id)`. 
    When rolling back a parent graph to an earlier checkpoint, LangGraph automatically rolls back child subgraph namespaces 
    to their corresponding parent superstep timestamps.
    """)

with st.expander("Q5: Can a node in a subgraph directly yield to an interrupt() paused for human approval?"):
    st.markdown("""
    **Answer:**  
    Yes. An `interrupt()` invoked inside a nested subgraph bubbles up to the parent graph runtime, pausing the entire thread 
    at that superstep. When resumed with `Command(resume=...)`, the input payload is routed back into the nested subgraph node.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Omitting Strict Convergence Criteria in Generator-Critic Loops**  
Designing a critic that demands perfection without relaxing criteria or incrementing a hard loop counter causes the graph 
to cycle until `GraphRecursionError` aborts the run, wasting tokens without delivering an answer.
""")

st.error("""
**Trap 2: State Channel Name Collisions Across Unisolated Subgraphs**  
Using the exact same state schema for parent and child graphs without custom mapping can cause child nodes to inadvertently 
overwrite parent channels during superstep barrier merges.
""")

st.warning("""
**Trap 3: Under-sizing recursion_limit for Legitimate Multi-Turn Agents**  
Setting `recursion_limit` too low (e.g., 5) on complex research agents with valid multi-step workflows causes premature terminations 
during legitimate deep investigations.
""")

st.warning("""
**Trap 4: Stateful Side Effects Inside Dynamic Router Callables**  
Performing database writes or external API mutations inside conditional routing functions causes duplicate mutations 
if the router is re-evaluated during state history traversals or testing.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Command(goto=...)</h4>
        <p>A primitive allowing node functions to dynamically specify destination nodes and state updates simultaneously at runtime.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Nested Subgraphs</h4>
        <p>Compiled StateGraphs embedded as standard nodes within parent graphs with isolated state schemas and checkpoint namespaces.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Cyclic State Machine</h4>
        <p>A graph topology featuring feedback loops (such as generator-critic pairs) where nodes iteratively refine artifacts until convergence.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>recursion_limit</h4>
        <p>Configurable boundary defining the maximum superstep count permitted before execution terminates with GraphRecursionError.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to implement dynamic graph topologies using Command(goto=...).", value=True)
chk_2 = st.checkbox("I understand how to build dynamic multi-branch routing edges with path mappings.", value=True)
chk_3 = st.checkbox("I can embed compiled child subgraphs into parent graphs with isolated state schemas.", value=True)
chk_4 = st.checkbox("I understand how checkpoint namespaces isolate parent and child state persistence.", value=True)
chk_5 = st.checkbox("I know how to configure and tune recursion_limit to prevent runaway loops.", value=True)
chk_6 = st.checkbox("I can design cyclic self-correction loops with robust convergence criteria.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced graph architecture and cyclic topology mastery confirmed! Ready for Advanced State.")
