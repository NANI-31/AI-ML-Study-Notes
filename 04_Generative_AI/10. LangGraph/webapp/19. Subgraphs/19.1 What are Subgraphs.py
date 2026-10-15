import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="19.1 What are Subgraphs - LangGraph Curriculum",
    page_icon="🧩",
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
    .badge-parent {
        background-color: #1E3A8A;
        color: #93C5FD;
        border: 1px solid #3B82F6;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-child {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #7E22CE;
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
class SubgraphsSimulator:
    """
    Simulates parent-child subgraph interaction lifecycles in LangGraph:
    - Task delegation from parent to specialized child
    - Independent child superstep cycling
    - Output return & channel merge back into parent state
    """
    def __init__(self):
        pass

    def execute_hierarchical_workflow(
        self,
        query: str,
        child_cycles: int
    ) -> Dict[str, Any]:
        parent_trace = []
        child_trace = []

        # Parent Step 1
        parent_trace.append({
            "level": "Parent",
            "node": "supervisor_router",
            "action": f"Analyzed prompt: '{query}'. Delegated research to 'ResearchSubgraph'.",
            "step": 1
        })

        # Child Subgraph Execution
        for c in range(child_cycles):
            child_trace.append({
                "level": "Child (ResearchSubgraph)",
                "node": f"child_cycle_{c + 1}",
                "action": f"Cycle {c + 1}/{child_cycles}: Gathered source evidence and refined findings.",
                "step": c + 1
            })

        # Parent Step 2
        parent_trace.append({
            "level": "Parent",
            "node": "final_editor",
            "action": "Received child output. Consolidated final report for user.",
            "step": 2
        })

        return {
            "parent_trace": parent_trace,
            "child_trace": child_trace,
            "final_answer": f"Verified research response for '{query}' completed after {child_cycles} internal child iterations."
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Subgraph Hierarchies, Encapsulation, and Nested Graph Topologies", divider="rainbow")

st.markdown("""
As agent systems scale from single prompts to enterprise software, assembling all logic into a single monolithic graph becomes unmaintainable.
Monolithic graphs suffer from:
- State channel explosion (dozens of conflicting keys).
- Fragile coupling between unrelated business domains.
- Inability for independent engineering teams to build, test, and deploy specialized agent modules.

LangGraph solves this through **Subgraphs**.
A subgraph is simply a standard compiled `StateGraph` embedded inside another graph as an ordinary node!

The architectural blueprint below illustrates the hierarchical boundary between a Parent Orchestrator and specialized Child Subgraphs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SubgraphArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_parent {
            label="1. Parent Orchestration Graph (ParentState)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            p_start [label="START (Parent)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            p_supervisor [label="Supervisor / Task Router\\nInspects High-Level Goals", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            p_subgraph_node [label="Research Node\\n(Embedded Child Subgraph)", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            p_aggregator [label="Executive Summary Node\\nConsolidates All Deliverables", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            p_end [label="END (Parent)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_child {
            label="2. Specialized Child Subgraph (ResearchState)";
            color="#4C1D95";
            fillcolor="#2E1065";
            style="filled,rounded";
            c_start [label="START (Child)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#A855F7"];
            c_search [label="Academic Search Node\\n(ArXiv & Google Scholar)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            c_evaluate [label="Evidence Evaluator\\nNeeds More Data?", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            c_synthesize [label="Citation Synthesizer\\nBuilds Markdown Doc", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            c_end [label="END (Child)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#A855F7"];
        }

        p_start -> p_supervisor;
        p_supervisor -> p_subgraph_node [label="Delegates Topic"];
        
        # Subgraph embedding connection
        p_subgraph_node -> c_start [label="Invokes Subgraph", color="#F97316", style=dashed];
        c_start -> c_search;
        c_search -> c_evaluate;
        c_evaluate -> c_search [label="Cycle: Loop if Incomplete", color="#FBBF24"];
        c_evaluate -> c_synthesize [label="Threshold Met", color="#34D399"];
        c_synthesize -> c_end;
        c_end -> p_subgraph_node [label="Returns Research Dict", color="#34D399", style=dashed];

        p_subgraph_node -> p_aggregator [label="Passes Research to Parent"];
        p_aggregator -> p_end;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: DEFINITION OF SUBGRAPHS
# ==============================================================================
st.header("2. Definition of Subgraphs: Self-Contained Graphs Compiled as Node Callables", divider="rainbow")

st.markdown("""
In LangGraph, a **subgraph** is not a separate special class or runtime engine.
It is literally any standard `StateGraph` that has been compiled via `.compile()`.
Because a compiled graph is a **Runnable**, it can be:
1. Registered directly as a node in another graph: `builder.add_node("research_team", compiled_child_graph)`.
2. Called imperatively inside a normal Python node function: `child_result = child_graph.invoke(...)`.

When LangGraph executes, it treats the compiled child graph as a single execution step within the parent superstep, while internally executing multiple Pregel supersteps inside the child!
""")

col_def1, col_def2 = st.columns(2)
with col_def1:
    st.markdown("#### Standalone Child Definition")
    st.code('''# Define Child Graph
class ChildState(TypedDict):
    sub_topic: str
    summary: str

child_builder = StateGraph(ChildState)
child_builder.add_node("summarize", child_summary_fn)
child_builder.add_edge(START, "summarize")
child_builder.add_edge("summarize", END)

# Compiled child graph is a Runnable!
child_graph = child_builder.compile()''', language="python")

with col_def2:
    st.markdown("#### Embedding Inside Parent")
    st.code('''# Embed into Parent Graph directly as a node
class ParentState(TypedDict):
    main_prompt: str
    child_output: str

parent_builder = StateGraph(ParentState)
# Directly add the compiled graph as a node!
parent_builder.add_node("child_node", child_graph)
parent_builder.add_edge(START, "child_node")
parent_builder.add_edge("child_node", END)

parent_app = parent_builder.compile()''', language="python")

# ==============================================================================
# SECTION 3: WHY USE SUBGRAPHS?
# ==============================================================================
st.header("3. Why Use Subgraphs?: Modularity, Team Ownership, Reusability, and Context Isolation", divider="rainbow")

st.markdown("""
Subgraphs are the fundamental building block for production AI systems built by large engineering organizations.
""")

col_why1, col_why2 = st.columns(2)
with col_why1:
    st.markdown("#### 1. Context & Channel Isolation")
    st.markdown("""
    In a monolithic graph, temporary scratchpad variables (e.g. `raw_search_html`, `retry_index`, `vector_embeddings`) pollute the main state dictionary.
    Subgraphs confine temporary data to their internal child schema, returning only clean final deliverables to the parent.
    """)

    st.markdown("#### 2. Reusability Across Applications")
    st.markdown("""
    A well-architected `CodeReviewSubgraph` or `SqlValidatorSubgraph` can be packaged as an internal Python library and imported into 10 different agent workflows across the enterprise without code duplication.
    """)

with col_why2:
    st.markdown("#### 3. Independent Team Ownership")
    st.markdown("""
    Team A (Data Engineering) maintains the `SQLQuerySubgraph`.  
    Team B (Risk & Compliance) maintains the `GuardrailSubgraph`.  
    Team C (Application Core) maintains the `ParentOrchestrator`.  
    Teams can test and release updates independently without merge conflicts.
    """)

    st.markdown("#### 4. Isolated Failure & Retry Boundaries")
    st.markdown("""
    A child graph can cycle 5 times trying to fix an invalid SQL query. The parent graph only sees a single clean execution step that succeeds or fails as a cohesive unit.
    """)

# ==============================================================================
# SECTION 4: PARENT GRAPHS
# ==============================================================================
st.header("4. Parent Graphs: Orchestration, Delegation, and High-Level Boundary Routing", divider="rainbow")

st.markdown("""
The **parent graph** acts as the high-level executive or supervisor.
Its responsibilities are strictly macro-level:
- Ingesting user input and authenticating permissions.
- Breaking complex requests into sub-tasks.
- Routing tasks to appropriate specialized child graphs.
- Aggregating outputs and formatting final user communications.
""")

st.info("💡 **Design Rule**: The parent graph should not contain low-level tool calls or micro-reasoning loops. Delegate all domain-specific tools into specialized child graphs.")

# ==============================================================================
# SECTION 5: CHILD GRAPHS
# ==============================================================================
st.header("5. Child Graphs: Dedicated Specialization, Autonomous Cycles, and Internal States", divider="rainbow")

st.markdown("""
The **child graph** is an autonomous specialist.
It owns:
- **Private State Schema**: Fields that only the child needs during its operation.
- **Internal Micro-Cycles**: Loops where an agent reasons, calls tools, critiques results, and retries.
- **Domain Tools**: Specific API keys, database connections, and specialized prompts.
""")

# ==============================================================================
# SECTION 6: NESTED WORKFLOWS
# ==============================================================================
st.header("6. Nested Workflows: Multi-Tier Agent Trees and Distributed Task Execution", divider="rainbow")

st.markdown("""
Subgraphs can be nested to arbitrary depths, creating hierarchical agent trees:
- **Level 1 (Root Orchestrator)**: Enterprise Workflow Router
  - **Level 2 (Department Head)**: Software Development Subgraph
    - **Level 3 (Specialist)**: Python Syntax Validator Subgraph
    - **Level 3 (Specialist)**: Unit Test Generator Subgraph

Each level maintains its own encapsulation boundary, making complex hierarchical reasoning manageable.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Level Parent-Child Graph Architecture", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a Parent Orchestrator invoking a Child Research Subgraph.
""")

st.code('''from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. CHILD SUBGRAPH DEFINITION
# ==========================================
class ChildResearchState(TypedDict):
    topic: str
    findings: List[str]
    report: str

def gather_sources_node(state: ChildResearchState) -> dict:
    return {"findings": [f"Source A on {state['topic']}", f"Source B on {state['topic']}"]}

def compile_child_report_node(state: ChildResearchState) -> dict:
    summary = " | ".join(state["findings"])
    return {"report": f"Research Synthesis: {summary}"}

child_builder = StateGraph(ChildResearchState)
child_builder.add_node("gather_sources", gather_sources_node)
child_builder.add_node("compile_child_report", compile_child_report_node)
child_builder.add_edge(START, "gather_sources")
child_builder.add_edge("gather_sources", "compile_child_report")
child_builder.add_edge("compile_child_report", END)

child_research_graph = child_builder.compile()

# ==========================================
# 2. PARENT GRAPH DEFINITION
# ==========================================
class ParentOrchestratorState(TypedDict):
    user_goal: str
    delegated_task: str
    child_summary: str
    executive_response: str

def plan_and_delegate_node(state: ParentOrchestratorState) -> dict:
    return {"delegated_task": state["user_goal"]}

def invoke_subgraph_wrapper_node(state: ParentOrchestratorState) -> dict:
    # Transform parent state to child input
    child_input = {"topic": state["delegated_task"]}
    # Execute child graph
    child_output = child_research_graph.invoke(child_input)
    # Map child output back into parent state
    return {"child_summary": child_output["report"]}

def finalize_executive_output_node(state: ParentOrchestratorState) -> dict:
    return {
        "executive_response": f"EXECUTIVE BRIEF: {state['child_summary']} [Reviewed & Approved]"
    }

parent_builder = StateGraph(ParentOrchestratorState)
parent_builder.add_node("plan_delegate", plan_and_delegate_node)
parent_builder.add_node("execute_research", invoke_subgraph_wrapper_node)
parent_builder.add_node("finalize_output", finalize_executive_output_node)

parent_builder.add_edge(START, "plan_delegate")
parent_builder.add_edge("plan_delegate", "execute_research")
parent_builder.add_edge("execute_research", "finalize_output")
parent_builder.add_edge("finalize_output", END)

parent_app = parent_builder.compile(checkpointer=MemorySaver())

# Test Execution
result = parent_app.invoke(
    {"user_goal": "Next-Gen Solid State Batteries"},
    config={"configurable": {"thread_id": "subgraph_demo_01"}}
)
print("Final Output:", result["executive_response"])''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Hierarchical Subgraph Invocation & Execution Visualizer", divider="rainbow")

st.markdown("""
Simulate how task requests flow from the parent orchestrator down into specialized child subgraphs and return.
""")

sim = SubgraphsSimulator()

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.subheader("⚙️ Hierarchy Parameters")
    test_query = st.text_input("Parent Goal / Prompt", value="Evaluate Quantum Key Distribution viability")
    num_child_cycles = st.slider("Internal Child Subgraph Cycles", min_value=1, max_value=5, value=3)

    if st.button("🚀 Execute Hierarchical Workflow"):
        res = sim.execute_hierarchical_workflow(test_query, num_child_cycles)
        st.session_state["subgraph_sim_res"] = res

with col_sb2:
    st.subheader("📊 Execution Flow Trace")
    if "subgraph_sim_res" in st.session_state:
        res = st.session_state["subgraph_sim_res"]
        
        st.markdown("#### 1. Parent Execution Trace")
        st.dataframe(pd.DataFrame(res["parent_trace"]), use_container_width=True)
        
        st.markdown("#### 2. Child Subgraph Execution Trace")
        st.dataframe(pd.DataFrame(res["child_trace"]), use_container_width=True)

        st.success(f" {res['final_answer']}")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is a subgraph in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    A subgraph is a compiled `StateGraph` instance that is embedded inside another graph.
    It can be added directly as a node (`builder.add_node('child', child_app)`) or called inside a regular node function.
    """)

with st.expander("Q2: Why should an architecture use subgraphs instead of one large StateGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Subgraphs prevent state schema explosion, isolate temporary intermediate variables, enable independent team ownership and testing, and allow specialized sub-graphs to be reused across different applications.
    """)

with st.expander("Q3: Does a subgraph have access to the parent graph's state channels by default?", expanded=False):
    st.markdown("""
    **Answer**:
    If the child graph defines the exact same state schema or if nodes are added directly and share channel keys, yes.
    However, the recommended pattern is **isolated state schemas** where a node function explicitly maps parent state inputs into the child state and maps child outputs back to the parent state.
    """)

with st.expander("Q4: Can a subgraph execute cyclic loops while the parent graph is a strictly linear DAG?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! The parent graph can be a simple linear 3-node sequence, while one of those nodes is a child subgraph containing complex multi-turn reasoning loops, self-correction cycles, and tool retries.
    """)

with st.expander("Q5: How does streaming work when subgraphs are executed?", expanded=False):
    st.markdown("""
    **Answer**:
    By default, `app.stream()` emits only parent events.
    Passing `subgraphs=True` instructs LangGraph to recursively stream events from all child subgraphs, prefixing each event with a namespace tuple identifying the child.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Passing Incompatible State Keys to Unwrapped Subgraphs")
    st.markdown("""
    *Mechanism*: Directly calling `parent_builder.add_node('child', child_graph)` when `ParentState` and `ChildState` have different keys with no common channels.  
    *Consequence*: Raises `KeyError` or schema validation error because child nodes look for keys that do not exist in the parent state.  
    *Fix*: Either share common keys or use a wrapper node function to map inputs and outputs.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Infinite Recursion Between Nested Graphs")
    st.markdown("""
    *Mechanism*: Graph A invokes Subgraph B, and Subgraph B conditionally routes back to Graph A on the same thread without a recursion limit.  
    *Consequence*: Infinite recursion exhausts memory or triggers `GraphRecursionError`.  
    *Fix*: Enforce strict directional boundaries and set `recursion_limit` in execution config.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Monolithic Subgraphs Defeating Encapsulation")
    st.markdown("""
    *Mechanism*: Passing the parent's entire 50-field state schema into every child subgraph.  
    *Consequence*: Reintroduces tight coupling and prevents independent testing of the child.  
    *Fix*: Design minimal, dedicated state schemas for each child subgraph.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Observability Warning): Missing Tracing Spans in Nested Invocations")
    st.markdown("""
    *Mechanism*: Invoking child subgraphs via custom thread pools or subprocesses without forwarding the parent config.  
    *Consequence*: LangSmith traces detach, making it impossible to debug child steps from parent dashboard views.  
    *Fix*: Always pass `config` into child invocations: `child_graph.invoke(sub_input, config=config)`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Subgraph**\\nA compiled StateGraph embedded inside another graph as an ordinary node.")
with col_fc2:
    st.info("**Encapsulation**\\nConfining internal scratchpad variables and micro-cycles to child state schemas.")
with col_fc3:
    st.info("**Wrapper Node**\\nA Python function that translates parent state to child input and maps child results back.")
with col_fc4:
    st.info("**subgraphs=True**\\nThe flag enabling recursive streaming and inspection of nested child graph lifecycles.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the formal definition of a subgraph in LangGraph.", value=True)
c2 = st.checkbox("I understand why enterprise systems use subgraphs for modularity and team ownership.", value=True)
c3 = st.checkbox("I understand the difference between parent orchestration and child specialization.", value=True)
c4 = st.checkbox("I know how to wrap a child subgraph to map state inputs and outputs cleanly.", value=True)
c5 = st.checkbox("I understand how child subgraphs can execute autonomous internal cycles.", value=True)
c6 = st.checkbox("I know why forwarding the execution config to child invocations is mandatory for tracing.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 19.1 'What are Subgraphs' with full architectural and practical comprehension!")
