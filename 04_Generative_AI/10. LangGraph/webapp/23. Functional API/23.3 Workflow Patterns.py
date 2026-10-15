import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="23.3 Workflow Patterns - LangGraph Curriculum",
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
    .badge-pattern {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class FunctionalWorkflowPatternsSimulator:
    """
    Simulates native Python workflow patterns in the Functional API:
    - Native for-loop fan-out
    - While loop convergence
    - Human-in-the-loop interrupt inside an entrypoint
    """
    def __init__(self):
        pass

    def simulate_workflow_pattern(
        self,
        pattern: str,
        item_count: int,
        require_approval: bool
    ) -> Dict[str, Any]:
        trace = []
        if pattern == "Dynamic Fan-Out (Python for)":
            trace.append({"step": 1, "action": f"Dispatched {item_count} parallel task futures using a standard Python for-loop."})
            trace.append({"step": 2, "action": "Awaited task.result() across all items concurrently."})
            trace.append({"step": 3, "action": f"Aggregated {item_count} results into a final deliverable dictionary."})
            status = "COMPLETED"

        elif pattern == "Iterative Convergence (Python while)":
            trace.append({"step": 1, "action": "Started iterative optimization loop."})
            for i in range(1, 4):
                trace.append({"step": i + 1, "action": f"Iteration {i}: Computed loss metric. Updated candidate parameters."})
            trace.append({"step": 5, "action": "Convergence threshold met: While loop terminated naturally."})
            status = "COMPLETED"

        else: # Human Interrupt
            trace.append({"step": 1, "action": "Calculated risk threshold for high-value financial transaction."})
            if require_approval:
                trace.append({"step": 2, "action": "interrupt() called: Workflow halted! State saved to checkpointer database."})
                trace.append({"step": 3, "action": "Human review pending: Ready to resume via Command(resume=True)."})
                status = "PAUSED_FOR_APPROVAL"
            else:
                trace.append({"step": 2, "action": "Transaction within auto-approval limits. Processed without interrupt."})
                status = "COMPLETED"

        return {
            "pattern": pattern,
            "status": status,
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Native Python Workflow Topologies in the Functional API", divider="rainbow")

st.markdown("""
The true power of the LangGraph Functional API is **idiomatic Python expressiveness**.
Instead of learning a specialized graph domain-specific language (DSL) to model control flow, developers use:
- **`if` / `else`**: For dynamic conditional routing.
- **`for item in items`**: For dynamic task fan-out and parallel batch processing.
- **`while not converged`**: For cyclic iterative refinement and reflection.
- **`interrupt()`**: For pausing execution at any arbitrary line to collect human input.

The blueprint below illustrates how native Python control structures translate into resilient Pregel task graphs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FunctionalPatternsTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_python_structures {
            label="1. Native Python Control Structures";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            py_for [label="for item in items:\\n(Dynamic Task Fan-Out)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            py_if [label="if risk_score > 0.8:\\n(Native Branching)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            py_while [label="while score < target:\\n(Iterative Convergence Loop)", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_pregel_mapping {
            label="2. Automatic Pregel Graph Translation";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_futures [label="Concurrent Task Futures\\nPregel Parallel Superstep", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            interrupt_call [label="interrupt('Approve wire?')\\nPauses Thread in DB Checkpoint", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
        }

        subgraph cluster_completion {
            label="3. Durable State Resolution";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            resume_cmd [label="Command(resume=True)\\nResumes from Exact Line of Code", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            final_return [label="return deliverable\\nCommitted to Checkpointer", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        py_for -> task_futures;
        py_if -> interrupt_call [label="Risk Exceeded"];
        py_while -> py_while [label="Repeat until threshold", color="#FBBF24"];
        
        interrupt_call -> resume_cmd [label="External Webhook / Human UI"];
        resume_cmd -> final_return;
        task_futures -> final_return;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: PYTHON IF
# ==============================================================================
st.header("2. Python if: Native Conditional Branching Without Conditional Edge Boilerplate", divider="rainbow")

st.markdown("""
In the Graph API, routing between nodes requires defining a router function, registering conditional edges, and returning string target names.
In the Functional API, you use standard Python **`if` / `else`** statements:
""")

st.code('''@entrypoint(checkpointer=checkpointer)
def customer_service_flow(user_query: str) -> str:
    category = classify_intent(user_query).result()
    
    # Native Python if-statement!
    if category == "billing":
        return process_billing(user_query).result()
    elif category == "technical":
        return process_tech_support(user_query).result()
    else:
        return fallback_faq(user_query).result()''', language="python")

# ==============================================================================
# SECTION 3: PYTHON FOR
# ==============================================================================
st.header("3. Python for: Dynamic Task Fan-Out and Batch List Processing", divider="rainbow")

st.markdown("""
Instead of configuring the `Send` API or map-reduce patterns, dynamic task fan-out is written with a standard **`for` loop**:
""")

st.code('''@entrypoint(checkpointer=checkpointer)
def batch_document_analyzer(documents: list) -> list:
    # 1. Dispatch tasks concurrently in a loop
    futures = [analyze_single_doc(doc) for doc in documents]
    
    # 2. Gather results
    return [f.result() for f in futures]''', language="python")

# ==============================================================================
# SECTION 4: PYTHON LOOPS
# ==============================================================================
st.header("4. Python loops: Cyclic Execution, While Loops, and Convergence Tracking", divider="rainbow")

st.markdown("""
Cyclic reasoning loops (like reflection or iterative optimization) use a native **`while` loop**:
""")

st.code('''@entrypoint(checkpointer=checkpointer)
def iterative_refiner(initial_code: str) -> str:
    code = initial_code
    attempts = 0
    
    while attempts < 3:
        test_passed = run_tests(code).result()
        if test_passed:
            break
        code = fix_bugs(code).result()
        attempts += 1
        
    return code''', language="python")

# ==============================================================================
# SECTION 5: FUNCTION COMPOSITION
# ==============================================================================
st.header("5. Function Composition: Hierarchical Modular Pipelines and Nested Entrypoints", divider="rainbow")

st.markdown("""
An `@entrypoint` function can call another `@entrypoint` function.
LangGraph automatically preserves the checkpoint namespace hierarchy, allowing teams to compose modular durable functions into larger enterprise workflows.
""")

# ==============================================================================
# SECTION 6: LONG-RUNNING FUNCTIONS
# ==============================================================================
st.header("6. Long-Running Functions: Multi-Hour Workflows and Spot-Instance Resilience", divider="rainbow")

st.markdown("""
Because every completed `@task` result is committed to the checkpointer, functions can run for hours or days across Kubernetes spot-instance evictions.
If the pod hosting the function terminates on Step 4 of a 10-step process, the replacement pod resumes from Step 4 without repeating Steps 1 through 3.
""")

# ==============================================================================
# SECTION 7: INTERRUPTS
# ==============================================================================
st.header("7. Interrupts: Pausing Functions for Human-in-the-Loop Reviews with interrupt()", divider="rainbow")

st.markdown("""
You can pause a functional workflow anywhere by invoking **`interrupt()`**:
""")

st.code('''from langgraph.types import interrupt

@entrypoint(checkpointer=checkpointer)
def wire_transfer_flow(amount: float) -> str:
    if amount > 10000.0:
        # Pauses execution right here!
        approved = interrupt(f"High-value wire transfer of ${amount:,.2f}. Approve?")
        if not approved:
            return "TRANSFER_REJECTED"
            
    return execute_transfer(amount).result()''', language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Functional Workflow with Loops, Tasks, and Interrupts", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a functional workflow combining loops, tasks, and human interrupts.
""")

st.code('''from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

@task
def scrape_article(url: str) -> str:
    return f"Content extracted from {url}"

@task
def summarize_text(content: str) -> str:
    return f"Summary: {content[:30]}..."

@entrypoint(checkpointer=MemorySaver())
def research_digest_workflow(urls: list) -> dict:
    # 1. Parallel Task Fan-Out with Python for-loop
    scrape_futures = [scrape_article(u) for u in urls]
    contents = [f.result() for f in scrape_futures]
    
    # 2. Summarize all
    summary_futures = [summarize_text(c) for c in contents]
    summaries = [f.result() for f in summary_futures]
    
    # 3. Human Approval Interrupt
    review = interrupt({
        "question": f"Review {len(summaries)} compiled summaries before publishing.",
        "summaries": summaries
    })
    
    return {
        "status": "PUBLISHED" if review else "REJECTED_BY_HUMAN",
        "digest": summaries
    }

# Invocation: Runs up to interrupt
config = {"configurable": {"thread_id": "digest_001"}}
urls = ["https://news.com/a", "https://tech.org/b"]
# state = research_digest_workflow.invoke(urls, config=config)
# Resume: research_digest_workflow.invoke(Command(resume=True), config=config)
print("Functional Workflow with Fan-Out and Interrupt Ready.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Functional Control Flow & Interrupt Simulator", divider="rainbow")

st.markdown("""
Test how native Python control patterns (for loops, while loops, interrupts) execute within the Functional API.
""")

sim = FunctionalWorkflowPatternsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Workflow Pattern Controls")
    pattern_choice = st.selectbox(
        "Choose Functional Pattern",
        [
            "Dynamic Fan-Out (Python for)",
            "Iterative Convergence (Python while)",
            "Human Approval Interrupt (interrupt())"
        ]
    )
    num_items = st.slider("Fan-Out Item Count", min_value=1, max_value=8, value=4)
    require_appr = st.toggle("Trigger Human Approval Gate", value=True)

    if st.button("🚀 Run Functional Pattern Simulation"):
        p_res = sim.simulate_workflow_pattern(pattern_choice, num_items, require_appr)
        st.session_state["func_pattern_res"] = p_res

with col_sim2:
    st.subheader("📋 Execution Step Audit")
    if "func_pattern_res" in st.session_state:
        res = st.session_state["func_pattern_res"]
        
        if res["status"] == "PAUSED_FOR_APPROVAL":
            st.warning("⚠️ **INTERRUPT HIT**: Execution paused at `interrupt()` statement!")
        else:
            st.success(f" **Status**: `{res['status']}`")

        st.markdown("#### Lifecycle Steps:")
        for ev in res["trace"]:
            st.markdown(f"- **Step {ev['step']}**: {ev['action']}")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: How does LangGraph resume an @entrypoint function after an interrupt()?", expanded=False):
    st.markdown("""
    **Answer**:
    When `app.invoke(Command(resume=payload))` is called on a paused thread, LangGraph re-runs the entrypoint function.
    Any `@task` functions completed before the interrupt return their cached results immediately from the checkpointer without re-executing.
    When execution reaches the `interrupt()` line, it resolves to the payload and continues execution forward.
    """)

with st.expander("Q2: Why are standard Python loops better than graph cycles for simple tasks?", expanded=False):
    st.markdown("""
    **Answer**:
    Python loops provide readable, intuitive control flow with standard break conditions without requiring developers to define multiple edges, conditional routers, and state dictionaries.
    """)

with st.expander("Q3: How do you achieve parallel task fan-out in the Functional API?", expanded=False):
    st.markdown("""
    **Answer**:
    By instantiating a list of task futures in a list comprehension (`futures = [my_task(x) for x in items]`) and then resolving them (`[f.result() for f in futures]`).
    LangGraph's Pregel scheduler executes all scheduled tasks concurrently.
    """)

with st.expander("Q4: Can an @entrypoint function call another @entrypoint function?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Sub-entrypoints operate as sub-workflows, and LangGraph automatically prefixes their checkpoints with a nested namespace.
    """)

with st.expander("Q5: What happens if code outside an @task is non-deterministic during replay?", expanded=False):
    st.markdown("""
    **Answer**:
    Because the entrypoint function re-runs from the top upon resumption, any non-deterministic code outside an `@task` (e.g. `random.randint()` or `datetime.now()`) will compute a different value, causing logical divergence.
    Non-deterministic or external operations must always be isolated inside `@task` functions.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Non-Deterministic Code Outside @task During Replay")
    st.markdown("""
    *Mechanism*: Writing `ts = time.time()` in an `@entrypoint` and using it in an `if` branch.  
    *Consequence*: When the workflow resumes from an interrupt, `time.time()` changes, causing the function to take a different branch and break.  
    *Fix*: Put non-deterministic operations inside `@task` functions so they are cached.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Infinite While Loops Without Counter Guards")
    st.markdown("""
    *Mechanism*: Running a `while not converged:` loop that never reaches convergence threshold.  
    *Consequence*: Loops indefinitely, burning API credits and hanging worker containers.  
    *Fix*: Always enforce a `max_turns <= 5` guard in while loop conditions.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Resolving Futures Inside the For-Loop")
    st.markdown("""
    *Mechanism*: Writing `[my_task(x).result() for x in items]`.  
    *Consequence*: Resolves each task sequentially, completely disabling concurrency.  
    *Fix*: Create all futures first: `futures = [my_task(x) for x in items]; results = [f.result() for f in futures]`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (HITL Warning): Resuming an Interrupt with Bare Input Rather than Command")
    st.markdown("""
    *Mechanism*: Calling `workflow.invoke({"approved": True})` to resume an interrupt.  
    *Consequence*: Treated as a new input rather than delivering the resume payload to the paused `interrupt()` line.  
    *Fix*: Always pass `workflow.invoke(Command(resume=True), config=config)`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Native Python if**\\nConditional routing using standard Python syntax instead of conditional edges.")
with col_fc2:
    st.info("**Parallel Fan-Out**\\nDispatching multiple task futures in a list comprehension for concurrent execution.")
with col_fc3:
    st.info("**interrupt() in Entrypoint**\\nPausing a procedural function at an exact line to await human review.")
with col_fc4:
    st.info("**Command(resume=...)**\\nThe mechanism used to deliver human decisions directly to paused functional interrupts.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand how to write conditional routing using native Python if-statements.", value=True)
c2 = st.checkbox("I know how to implement concurrent task fan-out using Python for-loops.", value=True)
c3 = st.checkbox("I understand how while loops implement cyclic reasoning in the Functional API.", value=True)
c4 = st.checkbox("I can use interrupt() inside an @entrypoint function for human approval.", value=True)
c5 = st.checkbox("I know how to resume a paused functional workflow using Command(resume=...).", value=True)
c6 = st.checkbox("I understand why non-deterministic logic must be enclosed inside @task decorators.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 23.3 'Workflow Patterns' with full architectural and practical comprehension!")
