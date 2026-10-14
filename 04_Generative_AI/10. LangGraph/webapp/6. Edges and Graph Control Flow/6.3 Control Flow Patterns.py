import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="6.3 Control Flow Patterns - LangGraph Curriculum",
    page_icon="🔄",
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
# MODULE-SCOPE AGENTIC LOOP SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class AgenticLoopSimulator:
    """
    Simulates cyclical agentic control patterns:
    Generator -> Critic -> Refiner loop with recursion budget guards.
    """
    def __init__(self):
        self.pattern_catalog = {
            "Sequential": "Linear DAG without cycles (A -> B -> C).",
            "Branch & Merge": "Fan-out to parallel workers followed by fan-in join.",
            "Retry Loop": "Re-invokes failed node until success or max retries.",
            "Reflection Loop": "Evaluates draft against rubric and critiques reasoning.",
            "Self-Correction Loop": "Executes unit tests against code and patches bugs iteratively."
        }

    def simulate_reflection_cycle(self, initial_draft: str, target_score: float, max_loops: int) -> List[Dict[str, Any]]:
        """
        Simulates an iterative reflection loop until target_score is met or max_loops is reached.
        """
        history = []
        curr_draft = initial_draft
        curr_score = 0.50
        iteration = 0

        while iteration < max_loops:
            iteration += 1
            # Step 1: Critic
            critic_feedback = f"Iteration {iteration}: Lacks specific technical details and edge-case handling."
            curr_score += min(0.40, (1.0 - curr_score) * 0.5)

            # Record state
            history.append({
                "loop_index": iteration,
                "draft": curr_draft,
                "score": round(curr_score, 2),
                "critique": critic_feedback,
                "decision": "APPROVED" if curr_score >= target_score else "REFINE_AGAIN"
            })

            if curr_score >= target_score:
                break

            # Step 2: Refiner
            curr_draft = f"{curr_draft} [Refined with enhanced technical specifications in loop {iteration}]"

        return history


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Control Flow Patterns and Agentic Loops", divider="rainbow")

st.markdown("""
Unlike traditional workflow engines that are strictly limited to Directed Acyclic Graphs (DAGs),
**LangGraph is built from the ground up for cyclic state machines**.
By combining directed edges and conditional routing, developers can construct advanced agentic topologies:
**Reflection Loops**, **Evaluator-Optimizer Cycles**, and **Self-Correction Engines**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    patterns_blueprint_dot = """
    digraph ControlFlowPatterns {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress Phase";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START", fillcolor="#1E293B", shape=ellipse];
            Node_Generate [label="Generator Node\\nProduces initial draft or code", fillcolor="#0284C7"];
        }

        subgraph cluster_cyclic_core {
            label="2. Cyclic Feedback Core (Reflection Loop)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Node_Evaluator [label="Evaluator / Critic Node\\nTests output & assigns score", fillcolor="#4338CA"];
            Router_Decision [label="Decision Gate (Predicate)\\nscore >= 0.85 OR loops >= max?", shape=diamond, fillcolor="#6D28D9"];
            Node_Refiner [label="Refiner Node\\nIncorporates critique feedback", fillcolor="#0D9488"];
        }

        subgraph cluster_egress {
            label="3. Egress Phase";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            Node_Finalize [label="Finalize Node\\nFormats approved response", fillcolor="#059669"];
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Node_Generate;
        Node_Generate -> Node_Evaluator;
        Node_Evaluator -> Router_Decision;
        Router_Decision -> Node_Refiner [label="Refine (score < target)"];
        Node_Refiner -> Node_Evaluator [label="Cyclic Edge (Feedback)"];
        Router_Decision -> Node_Finalize [label="Pass (score >= target)"];
        Node_Finalize -> END;
    }
    """
    st.graphviz_chart(patterns_blueprint_dot, use_container_width=True)

st.markdown("""
### The Power of Cycles in Generative AI
A standard LLM completion is often flawed on the first pass.
Cyclic control flow allows an agent to **inspect its own output, execute unit tests, critique shortcomings, and iterate**
until quality thresholds are satisfied.
""")

# ==============================================================================
# SECTION 2: SEQUENTIAL EXECUTION
# ==============================================================================
st.header("2. Sequential Execution: Deterministic Pipelining and Step Chaining", divider="rainbow")

st.markdown("""
Sequential execution is the simplest control flow pattern. Nodes execute one after another in a fixed linear order.
It is best suited for deterministic, multi-stage pipelines where later stages strictly depend on earlier outputs.
""")

col_sq1, col_sq2 = st.columns(2)
with col_sq1:
    st.markdown("""
    #### Use Cases for Sequential Execution:
    - Text ingestion -> Tokenization -> PII scrubbing
    - Retrieval -> Context injection -> Answer generation
    - Document translation -> Spellcheck -> Formatting
    """)
with col_sq2:
    st.code("""# Sequential pipeline: A -> B -> C -> END
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "clean")
builder.add_edge("clean", "synthesize")
builder.add_edge("synthesize", END)
""", language="python")

# ==============================================================================
# SECTION 3: BRANCHING
# ==============================================================================
st.header("3. Branching: Dynamic Path Divergence and Multi-Way Triage", divider="rainbow")

st.markdown("""
Branching uses conditional edges to divide execution into alternative pathways based on state attributes.
Only the chosen branch is activated, conserving tokens and compute.
""")

code_branch = """def route_urgency(state: TicketState) -> str:
    if state["priority"] == "CRITICAL":
        return "pager_duty_escalation"
    elif state["priority"] == "HIGH":
        return "senior_engineer_queue"
    return "standard_support_queue"

builder.add_conditional_edges("triage", route_urgency)
"""
st.code(code_branch, language="python")

# ==============================================================================
# SECTION 4: MERGING
# ==============================================================================
st.header("4. Merging: Fan-In Joins and Multi-Branch Synchronization", divider="rainbow")

st.markdown("""
Merging (or Fan-In) occurs when multiple parallel paths converge back into a single shared node.
The merging node serves as a **synchronization barrier**, waiting until all active parallel branches complete.
""")

code_merge = """# Both workers converge at 'synthesizer'
builder.add_edge("web_search_worker", "synthesizer")
builder.add_edge("database_worker", "synthesizer")
builder.add_edge("synthesizer", END)
"""
st.code(code_merge, language="python")

# ==============================================================================
# SECTION 5: LOOPS
# ==============================================================================
st.header("5. Loops: Controlled Iteration and Termination Invariants", divider="rainbow")

st.markdown("""
A **Loop** is a cycle that returns to a previously executed node.
Every loop in LangGraph MUST satisfy two invariants:
1. **Progress Invariant**: Each iteration must mutate state towards termination (e.g. incrementing `loop_count += 1`).
2. **Termination Invariant**: A conditional router must guarantee an exit to `END` when limits are reached.
""")

# ==============================================================================
# SECTION 6: CYCLES
# ==============================================================================
st.header("6. Cycles: Feedback Topologies and State-Driven Cyclic State Machines", divider="rainbow")

st.markdown("""
Cycles distinguish LangGraph from DAG workflow engines (like Airflow or Prefect).
In LangGraph, state persists across cycles, enabling stateful state-machine behaviors like game loops,
conversational multi-turn agents, and autonomous coding assistants.
""")

# ==============================================================================
# SECTION 7: RETRY LOOPS
# ==============================================================================
st.header("7. Retry Loops: Transient Fault Recovery and Exponential Delay Cycling", divider="rainbow")

st.markdown("""
A **Retry Loop** routes execution back to the failing node when a transient error occurs.
Unlike simple local retries inside a function, a graph-level retry loop can incorporate intermediate diagnostic steps:
refreshing auth tokens, selecting an alternative model provider, or adjusting prompt temperature before retrying.
""")

code_retry_loop = """def should_retry(state: APIState) -> str:
    if state.get("is_success", False):
        return "proceed"
    elif state["retry_count"] < 3:
        return "refresh_token_and_retry"
    return "fail_and_alert"

builder.add_conditional_edges("call_api", should_retry, {
    "proceed": "process_data",
    "refresh_token_and_retry": "refresh_auth",
    "fail_and_alert": "alert_ops"
})
builder.add_edge("refresh_auth", "call_api") # Cycle back!
"""
st.code(code_retry_loop, language="python")

# ==============================================================================
# SECTION 8: REFLECTION LOOPS
# ==============================================================================
st.header("8. Reflection Loops: Self-Critique and Evaluator-Optimizer Workflows", divider="rainbow")

st.markdown("""
The **Reflection Pattern** pairs a **Generator** node with a **Critic/Evaluator** node:
1. Generator produces an initial draft.
2. Critic evaluates the draft against explicit quality rubrics and emits critique notes.
3. Generator refines the draft using the critique notes.
4. Process repeats until quality targets are met.
""")

# ==============================================================================
# SECTION 9: SELF-CORRECTION LOOPS
# ==============================================================================
st.header("9. Self-Correction Loops: Automated Bug Fixing and Test-Driven Refinement", divider="rainbow")

st.markdown("""
In coding and technical workflows, self-correction uses **external test harnesses** rather than subjective LLM critiques:
1. LLM generates Python code.
2. Sandboxed Test Runner executes `pytest`.
3. If tests pass -> proceed to `END`.
4. If tests fail -> pass error traceback back to LLM to debug and fix its own code!
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Self-Correction and Reflection Agent Loop", divider="rainbow")

st.markdown("""
Below is a complete, production-grade implementation of a Reflection & Self-Correction Agent Loop.
It includes an explicit loop counter guard to prevent infinite recursion.
""")

prod_loop_code = '''from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END

# Step 1: Define Cyclic State
class CodeRefinementState(TypedDict):
    task: str
    code_draft: str
    test_passed: bool
    feedback: Optional[str]
    iteration: int
    max_iterations: int

# Nodes
def code_generator_node(state: CodeRefinementState) -> Dict[str, Any]:
    it = state["iteration"] + 1
    feedback = state.get("feedback")
    
    if feedback:
        new_code = f"def solution(): return 42 # Patched based on: {feedback}"
    else:
        new_code = "def solution(): return 0 # Initial flawed draft"
        
    return {
        "code_draft": new_code,
        "iteration": it
    }

def test_runner_node(state: CodeRefinementState) -> Dict[str, Any]:
    # Simulate running unit test
    code = state["code_draft"]
    is_correct = "42" in code
    
    return {
        "test_passed": is_correct,
        "feedback": None if is_correct else "AssertionError: expected 42, got 0"
    }

# Router Predicate with Loop Guard
def evaluate_test_results(state: CodeRefinementState) -> str:
    if state["test_passed"]:
        return "success"
    elif state["iteration"] >= state["max_iterations"]:
        return "max_loops_exceeded"
    return "refine"

# Step 2: Assemble Cyclic Graph
def build_self_correction_graph():
    builder = StateGraph(CodeRefinementState)
    
    builder.add_node("generator", code_generator_node)
    builder.add_node("tester", test_runner_node)
    
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "tester")
    
    builder.add_conditional_edges(
        "tester",
        evaluate_test_results,
        {
            "success": END,
            "max_loops_exceeded": END,
            "refine": "generator" # Cyclic Edge!
        }
    )
    
    return builder.compile()

if __name__ == "__main__":
    app = build_self_correction_graph()
    initial_task = {
        "task": "Write a function returning the universal answer",
        "code_draft": "",
        "test_passed": False,
        "feedback": None,
        "iteration": 0,
        "max_iterations": 3
    }
    final_res = app.invoke(initial_task)
    print("Test Passed:", final_res["test_passed"])
    print("Iterations Run:", final_res["iteration"])
    print("Final Code:", final_res["code_draft"])
'''
st.code(prod_loop_code, language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Agentic Loop Simulator and Recursion Budget Inspector", divider="rainbow")

st.markdown("""
Observe how cyclic state machines iterate towards perfection. Configure an initial draft,
set the quality threshold target, and trace the iterative refinement cycle.
""")

sim = AgenticLoopSimulator()

col_pw_l, col_pw_r = st.columns([1, 1.2])

with col_pw_l:
    st.subheader("Reflection Engine Parameters")
    init_draft = st.text_input("Initial User Draft", value="Antigravity provides tools for AI pair programming.")
    target_quality = st.slider("Target Quality Score (Exit Threshold)", 0.5, 0.95, 0.85)
    max_loops_budget = st.number_input("Maximum Allowed Iterations (Guard Limit)", min_value=1, max_value=6, value=4)

    if st.button("🔄 Execute Reflection Cycle", use_container_width=True):
        st.session_state.loop_history = sim.simulate_reflection_cycle(
            init_draft,
            target_quality,
            int(max_loops_budget)
        )
        st.success("Cycle simulation completed!")

with col_pw_r:
    st.subheader("Iterative Loop History")
    if "loop_history" in st.session_state:
        df_hist = pd.DataFrame(st.session_state.loop_history)
        st.dataframe(df_hist[["loop_index", "score", "decision", "critique"]], use_container_width=True)
        
        last = st.session_state.loop_history[-1]
        st.markdown(f"**Final Status**: `{last['decision']}` | **Final Score**: `{last['score']}`")
        st.markdown("**Final Refined Draft:**")
        st.info(last["draft"])
    else:
        st.info("Click 'Execute Reflection Cycle' to observe iterative feedback.")

st.markdown("### Control Flow Pattern Taxonomy")
df_patterns = pd.DataFrame([
    {"Pattern": k, "Description": v} for k, v in sim.pattern_catalog.items()
])
st.dataframe(df_patterns, use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What is GraphRecursionError in LangGraph and what causes it?", expanded=False):
    st.markdown("""
    **Answer**:
    `GraphRecursionError` occurs when the total number of superstep transitions in a single invocation exceeds the configured `recursion_limit` (default: 25).
    It is caused by:
    1. Cyclic loops that never meet their termination condition.
    2. Missing loop counter guards (`iteration += 1`).
    3. Flaky router predicates that oscillate indefinitely between two nodes.
    You can increase the limit using `app.invoke(..., config={"recursion_limit": 50})`, but you should always fix the underlying loop guard first.
    """)

with st.expander("Q2: How does the Evaluator-Optimizer pattern differ from standard zero-shot prompting?", expanded=False):
    st.markdown("""
    **Answer**:
    In zero-shot prompting, the model receives one chance to produce the answer.
    In the Evaluator-Optimizer pattern, one LLM generates a candidate response, a second LLM evaluates it against a detailed grading rubric, and the generator revises the response based on targeted critique. This mirrors human drafting and editing, dramatically improving factual accuracy and adherence to constraints.
    """)

with st.expander("Q3: Why are cyclic graphs superior to linear chains for software engineering tasks?", expanded=False):
    st.markdown("""
    **Answer**:
    Software development is inherently iterative. Code generated on the first try frequently contains syntax errors, missing imports, or failing unit tests.
    A cyclic graph connects the code generation node to a linter/unit-test node and loops back with the error traceback, enabling the agent to autonomously fix its own bugs before returning the result.
    """)

with st.expander("Q4: How do you prevent state bloat in multi-turn reflection loops?", expanded=False):
    st.markdown("""
    **Answer**:
    1. Overwrite the `draft_text` channel instead of appending every intermediate draft to a list.
    2. Store only the latest critique and a numerical `iteration_count`.
    3. Prune historical reasoning tokens using `RemoveMessage` if message channels are used.
    """)

with st.expander("Q5: What is the difference between a retry loop and a reflection loop?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Retry Loop**: Retries an identical operation in response to transient infrastructure errors (HTTP 503, timeouts) without altering prompt instructions.
    - **Reflection Loop**: Intentionally refines output based on qualitative critique notes and rubric evaluations, changing the prompt context on each turn.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Loops from Missing Recursion Limit Loop Guards")
    st.markdown("""
    *Mechanism*: Creating a cyclic edge without an `iteration >= max_loops` check in the router predicate.  
    *Consequence*: If the model fails to satisfy the rubric, the graph loops endlessly until `GraphRecursionError` crashes the thread.  
    *Fix*: Always enforce a hard maximum iteration guard: `if loop_count >= 3: return END`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Oscillating Router Cycles Between Sibling Nodes")
    st.markdown("""
    *Mechanism*: Node A changes state such that router directs to Node B, and Node B changes state such that router directs back to Node A indefinitely.  
    *Consequence*: Rapid token consumption and catastrophic quota depletion.  
    *Fix*: Use monotonic state variables (e.g. counters that only increase) to break cycles.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Compounding Latency in Deep Reflection Cycles")
    st.markdown("""
    *Mechanism*: Running 5 reflection loops where each loop takes 4s (total 20s latency) for trivial user queries.  
    *Consequence*: Unacceptable user response times and user abandonment.  
    *Fix*: Dynamically bypass reflection for simple queries using an upfront complexity classifier.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Destructive Overwrites of Historical Best Drafts")
    st.markdown("""
    *Mechanism*: An agent generates a great draft in loop 1, but in loop 2 over-optimizes and degrades quality.  
    *Consequence*: The final returned draft is worse than the initial draft.  
    *Fix*: Track `best_score` and `best_draft_so_far` in state and return the best version upon completion.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Reflection Loop**\\nA cyclic pattern where an evaluator critiques drafts and feeds feedback back to the generator.")
with col_fc2:
    st.info("**Self-Correction**\\nAn iterative loop using automated tests or linters to automatically debug and patch code.")
with col_fc3:
    st.info("**recursion_limit**\\nA runtime safety threshold (default: 25) that stops infinite cycles from consuming infinite quota.")
with col_fc4:
    st.info("**Loop Guard**\\nA mandatory state invariant (e.g. `iteration >= limit`) guaranteeing an exit path to END.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand why cyclic state machines are essential for advanced AI agent behaviors.", value=True)
c2 = st.checkbox("I know how to design reflection loops pairing a generator with an evaluator.", value=True)
c3 = st.checkbox("I can implement automated self-correction loops using unit test feedback.", value=True)
c4 = st.checkbox("I understand GraphRecursionError and always implement hard loop counter guards.", value=True)
c5 = st.checkbox("I know how to avoid oscillating state cycles between sibling nodes.", value=True)
c6 = st.checkbox("I understand strategies for tracking the best draft across multiple reflection iterations.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 6.3 'Control Flow Patterns' with full architectural and practical comprehension!")
