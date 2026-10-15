import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="9.1 Basic Patterns - LangGraph Curriculum",
    page_icon="📐",
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
# MODULE-SCOPE PATTERN SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class BasicPatternsEngine:
    """
    Simulates the 7 core structural graph patterns:
    Linear, Branching, Fan-out, Fan-in, Loop, Conditional Loop, and Retry.
    """
    def __init__(self):
        self.pattern_metadata = {
            "Linear Pipeline": {"nodes": 3, "edges": 3, "cyclical": False, "use": "Sequential multi-stage ETL"},
            "Branching Pipeline": {"nodes": 4, "edges": 4, "cyclical": False, "use": "Categorical query triage"},
            "Fan-Out": {"nodes": 4, "edges": 3, "cyclical": False, "use": "Parallel task dispatch"},
            "Fan-In": {"nodes": 4, "edges": 3, "cyclical": False, "use": "Barrier result consolidation"},
            "Loop": {"nodes": 2, "edges": 2, "cyclical": True, "use": "Fixed iterative refinement"},
            "Conditional Loop": {"nodes": 3, "edges": 3, "cyclical": True, "use": "Quality-gated self-correction"},
            "Retry Workflow": {"nodes": 3, "edges": 4, "cyclical": True, "use": "Fault recovery with exponential backoff"}
        }

    def generate_pattern_trace(self, pattern_name: str, payload: str) -> List[Dict[str, Any]]:
        """
        Produces an execution trace for the selected pattern.
        """
        trace = []
        if pattern_name == "Linear Pipeline":
            trace = [
                {"step": 1, "node": "Ingest", "desc": f"Sanitized payload: '{payload}'"},
                {"step": 2, "node": "Process", "desc": "Applied domain business logic transformation"},
                {"step": 3, "node": "Format", "desc": "Emitted clean customer-facing response"}
            ]
        elif pattern_name == "Fan-Out / Fan-In":
            trace = [
                {"step": 1, "node": "Dispatcher", "desc": "Partitioned task into 2 worker streams"},
                {"step": 2, "node": "Worker_Alpha & Worker_Beta", "desc": "Parallel concurrent BSP execution"},
                {"step": 3, "node": "Aggregator", "desc": "Synchronized worker results into single report"}
            ]
        elif pattern_name == "Conditional Loop":
            trace = [
                {"step": 1, "node": "Generator", "desc": "Drafted candidate solution"},
                {"step": 2, "node": "Evaluator", "desc": "Assessed quality: Score 0.65 (Target >= 0.85)"},
                {"step": 3, "node": "Refiner", "desc": "Patched defects; loop count incremented to 1"},
                {"step": 4, "node": "Evaluator", "desc": "Assessed quality: Score 0.90 -> Routing to END"}
            ]
        else: # Retry Workflow
            trace = [
                {"step": 1, "node": "API_Caller", "desc": "Attempt 1 failed with transient 503 error"},
                {"step": 2, "node": "Backoff_Timer", "desc": "Applied 1.0s exponential delay"},
                {"step": 3, "node": "API_Caller", "desc": "Attempt 2 succeeded (200 OK) -> Routing to END"}
            ]
        return trace


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Fundamental Graph Topologies and Primitive Control Flow Patterns", divider="rainbow")

st.markdown("""
All complex multi-agent architectures are constructed from combinations of **seven primitive graph topologies**:
**Linear Pipeline**, **Branching Pipeline**, **Fan-Out**, **Fan-In**, **Fixed Loop**, **Conditional Loop**, and **Retry Workflow**.
Understanding these primitives allows engineers to assemble modular, failure-tolerant agent topologies.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    patterns_blueprint_dot = """
    digraph PrimitiveTopologies {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_linear {
            label="1. Linear Pipeline (DAG)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            L1 [label="Step 1", fillcolor="#334155"];
            L2 [label="Step 2", fillcolor="#334155"];
            L3 [label="Step 3", fillcolor="#334155"];
            L1 -> L2 -> L3;
        }

        subgraph cluster_branching {
            label="2. Branching & Fan-Out";
            style="filled";
            color="#1E293B";
            fontcolor="#A855F7";
            B_Root [label="Router / Dispatcher", fillcolor="#4338CA"];
            B_OptA [label="Worker A", fillcolor="#334155"];
            B_OptB [label="Worker B", fillcolor="#334155"];
            B_Root -> B_OptA;
            B_Root -> B_OptB;
        }

        subgraph cluster_fanin {
            label="3. Fan-In (Join Barrier)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            F_Aggregator [label="Join Aggregator", fillcolor="#059669"];
            B_OptA -> F_Aggregator;
            B_OptB -> F_Aggregator;
        }

        subgraph cluster_loop {
            label="4. Cyclic Loop with Guard";
            style="filled";
            color="#1E293B";
            fontcolor="#EAB308";
            C_Gen [label="Generator", fillcolor="#334155"];
            C_Eval [label="Evaluator Gate", shape=diamond, fillcolor="#854D0E"];
            C_Gen -> C_Eval;
            C_Eval -> C_Gen [label="Refine (score < limit)"];
            C_Exit [label="END", fillcolor="#1E293B", shape=doublecircle];
            C_Eval -> C_Exit [label="Pass (score >= limit)"];
        }

        L3 -> B_Root [style=invis];
    }
    """
    st.graphviz_chart(patterns_blueprint_dot, use_container_width=True)

st.markdown("""
### Composable Primitives
Just as LEGO bricks assemble into intricate structures, advanced systems like Self-RAG, Plan-and-Solve,
and multi-agent debate are simply compositions of these fundamental primitive topologies.
""")

# ==============================================================================
# SECTION 2: LINEAR PIPELINE
# ==============================================================================
st.header("2. Linear Pipeline: Deterministic Step Chaining and Sequential Execution", divider="rainbow")

st.markdown("""
A **Linear Pipeline** is a sequence of normal edges where each node connects directly to exactly one successor.
Execution flows deterministically from `START` to `END` with zero branching or looping.
""")

code_linear = """builder.add_edge(START, "load_data")
builder.add_edge("load_data", "clean_data")
builder.add_edge("clean_data", "score_data")
builder.add_edge("score_data", END)
"""
st.code(code_linear, language="python")

# ==============================================================================
# SECTION 3: BRANCHING PIPELINE
# ==============================================================================
st.header("3. Branching Pipeline: Path Divergence and Categorical Routing", divider="rainbow")

st.markdown("""
A **Branching Pipeline** uses a conditional edge or `Command(goto=...)` to direct execution to **one**
out of several mutually exclusive paths based on input characteristics.
""")

code_branching = """def triage_router(state: IngressState) -> str:
    if state["is_vip_customer"]:
        return "priority_queue"
    return "standard_queue"

builder.add_conditional_edges("ingress", triage_router)
"""
st.code(code_branching, language="python")

# ==============================================================================
# SECTION 4: FAN-OUT
# ==============================================================================
st.header("4. Fan-Out: Parallel Worker Dispatch and Concurrent Workflows", divider="rainbow")

st.markdown("""
**Fan-Out** triggers multiple downstream nodes to execute concurrently in parallel.
Can be achieved statically via multiple `add_edge` calls or dynamically via the `Send` API.
""")

code_fanout = """# Static Fan-out: Node 'dispatcher' triggers both workers
builder.add_edge("dispatcher", "worker_finance")
builder.add_edge("dispatcher", "worker_marketing")
"""
st.code(code_fanout, language="python")

# ==============================================================================
# SECTION 5: FAN-IN
# ==============================================================================
st.header("5. Fan-In: Barrier Synchronization and Reducer Aggregations", divider="rainbow")

st.markdown("""
**Fan-In** is the inverse of Fan-Out: multiple parallel branches converge on a single downstream node.
The aggregator node executes only after all parallel incoming branches have concluded.
""")

code_fanin = """# Both workers converge at 'aggregator'
builder.add_edge("worker_finance", "aggregator")
builder.add_edge("worker_marketing", "aggregator")
builder.add_edge("aggregator", END)
"""
st.code(code_fanin, language="python")

# ==============================================================================
# SECTION 6: LOOP
# ==============================================================================
st.header("6. Loop: Controlled Cycles and Monotonic Counter Progression", divider="rainbow")

st.markdown("""
A **Loop** creates a cycle back to an earlier node.
To ensure safety, the state schema must track a monotonic counter (`iteration += 1`),
and the loop must terminate when a maximum iteration bound is reached.
""")

# ==============================================================================
# SECTION 7: CONDITIONAL LOOP
# ==============================================================================
st.header("7. Conditional Loop: Dynamic Termination Gates and Quality-Driven Cycling", divider="rainbow")

st.markdown("""
In a **Conditional Loop**, termination is driven by a qualitative evaluation (e.g. `score >= 0.85` or `tests_passed == True`).
The loop iterates until the quality gate passes or the iteration ceiling is hit.
""")

code_cond_loop = """def quality_gate(state: AgentState) -> str:
    if state["accuracy"] >= 0.90 or state["iteration"] >= 3:
        return END # Exit loop!
    return "refiner" # Cycle back!

builder.add_conditional_edges("evaluator", quality_gate)
builder.add_edge("refiner", "evaluator")
"""
st.code(code_cond_loop, language="python")

# ==============================================================================
# SECTION 8: RETRY WORKFLOW
# ==============================================================================
st.header("8. Retry Workflow: Resilient Transient Fault Cycling and Fallback Guards", divider="rainbow")

st.markdown("""
A **Retry Workflow** loops back to a node when transient errors occur, incorporating delay backoff or auth refreshes.
Can be handled natively via `builder.add_node("name", fn, retry=RetryPolicy(...))` or explicitly via graph edges.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Pattern Composite Orchestration Graph", divider="rainbow")

st.markdown("""
Below is a complete, runnable script assembling a composite graph featuring:
Linear ingress -> Parallel Fan-out -> Fan-in join -> Conditional refinement loop -> Clean termination.
""")

prod_composite_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: Define Composite State
class CompositeWorkflowState(TypedDict):
    input_prompt: str
    cleaned_prompt: str
    branch_a_output: str
    branch_b_output: str
    combined_draft: str
    iteration: int
    quality_score: float
    audit_trace: Annotated[List[str], operator.add]

# Nodes
def sanitize_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Linear Stage
    return {
        "cleaned_prompt": state["input_prompt"].strip().title(),
        "audit_trace": ["Sanitized input prompt."]
    }

def worker_alpha_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Parallel Fan-Out
    return {
        "branch_a_output": "Alpha: Verified architectural requirements.",
        "audit_trace": ["Worker Alpha completed."]
    }

def worker_beta_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Parallel Fan-Out
    return {
        "branch_b_output": "Beta: Verified security credentials.",
        "audit_trace": ["Worker Beta completed."]
    }

def aggregator_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Fan-In Join
    draft = f"{state['branch_a_output']} | {state['branch_b_output']}"
    return {
        "combined_draft": draft,
        "audit_trace": ["Aggregated parallel outputs into initial draft."]
    }

def evaluator_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Conditional Loop Node
    it = state["iteration"] + 1
    # Simulated improvement per iteration
    score = 0.60 + (it * 0.15)
    return {
        "iteration": it,
        "quality_score": score,
        "audit_trace": [f"Evaluation loop {it}: score={score:.2f}"]
    }

def refiner_node(state: CompositeWorkflowState) -> Dict[str, Any]:
    # Refinement cycle
    return {
        "combined_draft": state["combined_draft"] + f" [Refined v{state['iteration']}]",
        "audit_trace": ["Applied critique patches."]
    }

# Loop Router
def evaluate_quality_gate(state: CompositeWorkflowState) -> str:
    if state["quality_score"] >= 0.85 or state["iteration"] >= 3:
        return "approved"
    return "refine"

# Step 2: Build Graph Topology
def build_composite_graph():
    builder = StateGraph(CompositeWorkflowState)
    
    builder.add_node("sanitize", sanitize_node)
    builder.add_node("worker_alpha", worker_alpha_node)
    builder.add_node("worker_beta", worker_beta_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("refiner", refiner_node)
    
    # 1. Linear edge
    builder.add_edge(START, "sanitize")
    
    # 2. Parallel fan-out
    builder.add_edge("sanitize", "worker_alpha")
    builder.add_edge("sanitize", "worker_beta")
    
    # 3. Fan-in join
    builder.add_edge("worker_alpha", "aggregator")
    builder.add_edge("worker_beta", "aggregator")
    
    # 4. Ingress to cyclic loop
    builder.add_edge("aggregator", "evaluator")
    
    # 5. Conditional loop edge
    builder.add_conditional_edges(
        "evaluator",
        evaluate_quality_gate,
        {"approved": END, "refine": "refiner"}
    )
    builder.add_edge("refiner", "evaluator") # Cycle back!
    
    return builder.compile()

if __name__ == "__main__":
    app = build_composite_graph()
    initial_input = {
        "input_prompt": "deploy autonomous microservices graph",
        "cleaned_prompt": "",
        "branch_a_output": "",
        "branch_b_output": "",
        "combined_draft": "",
        "iteration": 0,
        "quality_score": 0.0,
        "audit_trace": ["Pipeline initialized."]
    }
    result = app.invoke(initial_input)
    print("Final Quality Score:", result["quality_score"])
    print("Final Output Draft:", result["combined_draft"])
    print("Total Trace Entries:", len(result["audit_trace"]))
'''
st.code(prod_composite_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Structural Pattern Configurator and Topology Inspector", divider="rainbow")

st.markdown("""
Inspect how the 7 fundamental patterns operate under the hood.
Select a pattern, enter a test payload, and review its step-by-step execution trace.
""")

engine = BasicPatternsEngine()

col_pw_l, col_pw_r = st.columns([1, 1.2])

with col_pw_l:
    st.subheader("Select Topology Pattern")
    selected_pat = st.selectbox("Graph Pattern Topology", list(engine.pattern_metadata.keys()))
    meta = engine.pattern_metadata[selected_pat]
    st.caption(f"**Nodes**: {meta['nodes']} | **Edges**: {meta['edges']} | **Cyclical**: `{meta['cyclical']}`")
    st.caption(f"**Primary Use Case**: {meta['use']}")

    test_input = st.text_input("Test Input Payload", value="LangGraph distributed orchestrator")

    if st.button("🚀 Trace Topology Execution", use_container_width=True):
        st.session_state.pat_trace = engine.generate_pattern_trace(selected_pat, test_input)
        st.success(f"Generated trace for '{selected_pat}'!")

with col_pw_r:
    st.subheader("Simulated Step-by-Step Trace")
    if "pat_trace" in st.session_state:
        for t in st.session_state.pat_trace:
            st.markdown(f"**Step {t['step']} — Node `{t['node']}`**")
            st.info(t["desc"])
    else:
        st.info("Click 'Trace Topology Execution' to view the trace.")

st.markdown("### Primitive Patterns Matrix")
df_pats = pd.DataFrame([
    {"Pattern": k, "Nodes": v["nodes"], "Edges": v["edges"], "Cyclical": v["cyclical"], "Use Case": v["use"]}
    for k, v in engine.pattern_metadata.items()
])
st.dataframe(df_pats, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What are the three core synchronization guarantees of Fan-In in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Barrier Hold**: The aggregator node will not start until every parallel predecessor node emits its output.
    2. **Reducer Resolution**: Deltas from all parallel branches are consolidated via channel reducers before the aggregator receives state.
    3. **Snapshot Consistency**: The aggregator reads a clean, unified state snapshot reflecting all completed branches.
    """)

with st.expander("Q2: How do you prevent an Evaluator-Generator loop from causing an infinite recursion?", expanded=False):
    st.markdown("""
    **Answer**:
    Always enforce a dual-condition exit in the router predicate:
    ```python
    if quality_score >= threshold or iteration_count >= max_limit:
        return END
    return "refiner"
    ```
    This ensures that even if the generator fails to achieve the target quality score, the loop exits predictably when the recursion budget is reached.
    """)

with st.expander("Q3: When should an engineer use an explicit retry graph loop versus node-level RetryPolicy?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Node-level `RetryPolicy`**: Best for fast, stateless retries of transient network timeouts (HTTP 503, connection drops) where no prompt changes are needed.
    - **Graph-level Retry Loop**: Best when retry requires changing state: rotating API keys, updating prompts with error feedback, or switching from OpenAI to Anthropic.
    """)

with st.expander("Q4: Can a single graph combine both DAG and cyclic sub-topologies?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. In LangGraph, DAG structures and cyclic structures coexist naturally. A common enterprise topology features a linear DAG for ingestion, a fan-out for parallel tool calls, and a cyclic reflection loop for draft refinement before exiting to `END`.
    """)

with st.expander("Q5: What is the risk of having a Branching Pipeline without a default fallback edge?", expanded=False):
    st.markdown("""
    **Answer**:
    If a user query does not match any of the branch conditions and the router returns `None` or an unmapped string, LangGraph raises a runtime `KeyError`, halting execution. Always provide a fallback branch to a general FAQ or human escalation node.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing Hard Limit on Conditional Self-Correction Loops")
    st.markdown("""
    *Mechanism*: Routing back to the generator purely on `score >= 0.9` without tracking iteration counts.  
    *Consequence*: Hits LangGraph's default `recursion_limit` (25 steps) and crashes with `GraphRecursionError`.  
    *Fix*: Always track `iteration += 1` in state and exit on `iteration >= max_loops`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Conflicting State Writes in Parallel Fan-Out Branches")
    st.markdown("""
    *Mechanism*: Having two parallel branches write to an unannotated scalar channel like `{"status": "DONE"}`.  
    *Consequence*: Crash with `InvalidUpdateError` due to ambiguous concurrent assignment.  
    *Fix*: Use separate channel names or an `Annotated[list, operator.add]` append reducer.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Overusing Complex Cycles for Trivial Queries")
    st.markdown("""
    *Mechanism*: Running 3 reflection loops on a simple query like "What is the capital of France?".  
    *Consequence*: Triples latency and token costs with zero improvement in accuracy.  
    *Fix*: Use an upfront complexity classifier to bypass reflection for simple factual questions.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Silent Deadlocks in Unbalanced Fan-In Joins")
    st.markdown("""
    *Mechanism*: Aggregator node expects inputs from branches A and B, but a conditional edge routed to branch A only.  
    *Consequence*: Aggregator hangs or behaves unpredictably waiting for inactive branch B.  
    *Fix*: Use conditional fan-in routing or ensure branches converge consistently.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Linear Pipeline**\\nA sequential DAG where each node connects to exactly one successor without branching.")
with col_fc2:
    st.info("**Fan-Out**\\nA topological pattern that dispatches tasks to multiple parallel worker nodes.")
with col_fc3:
    st.info("**Fan-In**\\nA synchronization barrier where an aggregator waits for all parallel branches to finish.")
with col_fc4:
    st.info("**Conditional Loop**\\nA stateful cyclic pattern that iterates until a quality threshold or recursion guard is met.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the seven primitive graph topologies in LangGraph.", value=True)
c2 = st.checkbox("I know how to implement parallel fan-out and fan-in barrier synchronization.", value=True)
c3 = st.checkbox("I can implement conditional loops with hard recursion limit guards.", value=True)
c4 = st.checkbox("I understand the difference between node-level RetryPolicy and graph-level retry loops.", value=True)
c5 = st.checkbox("I know how to prevent conflicting state updates during parallel fan-out.", value=True)
c6 = st.checkbox("I can combine multiple primitive patterns into a robust composite graph.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 9.1 'Basic Patterns' with full architectural and practical comprehension!")
