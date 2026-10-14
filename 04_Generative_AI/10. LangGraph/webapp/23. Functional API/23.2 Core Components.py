import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="23.2 Core Components - LangGraph Curriculum",
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
    .badge-comp {
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
class FunctionalComponentsSimulator:
    """
    Simulates core Functional API component mechanics:
    - @entrypoint initialization
    - @task future resolution and memoization
    - State snapshot extraction and checkpoint commits
    """
    def __init__(self):
        self.memoized_tasks = {}

    def run_task_simulation(self, task_name: str, task_args: str, force_cache_hit: bool) -> Dict[str, Any]:
        cache_key = f"{task_name}:{task_args}"
        
        if force_cache_hit and cache_key in self.memoized_tasks:
            return {
                "task": task_name,
                "status": "CACHE_HIT (SKIPPED_REEXECUTION)",
                "result": self.memoized_tasks[cache_key],
                "execution_ms": 0.4
            }

        # Fresh execution
        result = f"Computed output for {task_name}({task_args})"
        self.memoized_tasks[cache_key] = result
        return {
            "task": task_name,
            "status": "FRESH_EXECUTION (CHECKPOINTED)",
            "result": result,
            "execution_ms": 142.0
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Core Component Hierarchy in the Functional API", divider="rainbow")

st.markdown("""
The Functional API is built upon **five foundational components**:
1. **`@entrypoint`**: The outer boundary managing workflow lifecycle, thread configuration, and top-level checkpointer commits.
2. **`@task`**: Independent units of computation whose outputs are memoized and cached into checkpoints.
3. **Function Calls**: Standard Python argument passing that establishes the dynamic task execution graph.
4. **State Handling**: Automatic local variable scope management without requiring explicit TypedDict declarations.
5. **Persistence**: Under-the-hood integration with relational checkpointers (e.g. `PostgresSaver`) for crash recovery.

The architectural blueprint below illustrates the component interaction hierarchy.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FunctionalComponentsHierarchy {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_entrypoint {
            label="1. Top-Level Workflow Entrypoint (@entrypoint)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            entry_dec [label="@entrypoint(checkpointer=PostgresSaver)\\ndef my_application(input_data, *, previous=None):\\n(Owns Thread ID & Session Context)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_tasks {
            label="2. Atomic Checkpointed Tasks (@task)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_1 [label="@task\\ndef fetch_db(user_id):\\n(Returns Future -> .result())", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            task_2 [label="@task\\ndef compute_tax(amount):\\n(Deterministic Pure Task)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            task_cache [label="Task Result Cache\\n(Skips on Resume if Args Match)", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_persistence {
            label="3. Durable State & Persistence Storage";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            local_scope [label="Local Variable Scope\\n(Variables persist naturally in Python frame)", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            postgres_ckpt [label="PostgresSaver Database\\nStores Completed Task Checkpoints", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            client_output [label="Workflow Return Value", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        entry_dec -> task_1 [label="Calls Task 1"];
        task_1 -> task_cache [label="Checks Cache"];
        task_cache -> task_2 [label="Passes Result"];
        
        task_1 -> local_scope [label="Stored as Local Var"];
        task_2 -> local_scope [label="Stored as Local Var"];
        local_scope -> postgres_ckpt [label="Committed on Superstep"];
        entry_dec -> client_output;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: THE @entrypoint DECORATOR
# ==============================================================================
st.header("2. The @entrypoint Decorator: Entry Boundaries, Configuration Context, and Runtime Initialization", divider="rainbow")

st.markdown("""
The **`@entrypoint`** decorator designates the primary entry function of the application:
- **Checkpointer Binding**: Configured via `@entrypoint(checkpointer=my_checkpointer)`.
- **Previous State Injection**: An optional keyword argument `previous=None` receives the state from the prior execution on the same thread.
- **Runnable Surface**: An `@entrypoint` function is a compiled Runnable offering `.invoke()`, `.stream()`, and `.ainvoke()`.
""")

st.code('''from langgraph.func import entrypoint

@entrypoint()
def conversational_flow(user_message: str, *, previous: str = None) -> str:
    # 'previous' contains the output from the last invocation on this thread_id!
    if previous:
        return f"Continuing from: {previous}. New query: {user_message}"
    return f"First turn: {user_message}"''', language="python")

# ==============================================================================
# SECTION 3: TASKS (@task)
# ==============================================================================
st.header("3. Tasks (@task): Task Futures, Execution Granularity, and Memoization", divider="rainbow")

st.markdown("""
Functions decorated with **`@task`** become checkpointable units of work:
- **Asynchronous Futures**: Invoking `my_task(args)` does not run immediately; it schedules the task in the Pregel engine and returns a Future.
- **Resolution**: Call `.result()` to wait for and retrieve the value.
- **Automatic Memoization**: If an `@task` with identical arguments was completed in a previous step before a crash, LangGraph fetches the cached result from the checkpointer instead of re-running!
""")

# ==============================================================================
# SECTION 4: FUNCTION CALLS
# ==============================================================================
st.header("4. Function Calls: Inter-Task Invocation, Parameter Passing, and Dependency Resolution", divider="rainbow")

st.markdown("""
In the Graph API, passing data between nodes requires channel updates and state dictionaries.
In the Functional API, **parameter passing is standard Python**:
`result = task_b(task_a().result()).result()`
The dependencies between tasks are inferred automatically from Python call graphs.
""")

# ==============================================================================
# SECTION 5: STATE HANDLING
# ==============================================================================
st.header("5. State Handling: Local Variable Scope vs Thread-Level Channel Persistence", divider="rainbow")

st.markdown("""
In the Functional API:
- Intermediate state lives in normal Python local variables: `x = 10`, `summary = summarize(text).result()`.
- You do not need to define `MessagesState` or complex TypedDict schemas.
- When an `interrupt()` or superstep occurs, LangGraph automatically serializes active task results to the checkpointer.
""")

# ==============================================================================
# SECTION 6: PERSISTENCE
# ==============================================================================
st.header("6. Persistence: Seamless Checkpointing with PostgresSaver and MemorySaver", divider="rainbow")

st.markdown("""
Attaching a checkpointer to an `@entrypoint` enables durable execution across crashes:
""")

st.code('''from langgraph.checkpoint.postgres import PostgresSaver

# In production backed by PostgreSQL connection pool
# checkpointer = PostgresSaver(conn_pool)

# @entrypoint(checkpointer=checkpointer)
# def durable_workflow(data: dict) -> dict:
#     ...''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Task Functional Workflow with Database Checkpointing", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a multi-task functional workflow with previous state access and persistence.
""")

st.code('''from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

@task
def query_knowledge_base(topic: str) -> str:
    return f"Knowledge base records for {topic}: Verified active status."

@task
def calculate_risk_score(topic: str) -> float:
    return 0.15 # Low risk score

@entrypoint(checkpointer=MemorySaver())
def financial_audit_pipeline(customer_name: str, *, previous: dict = None) -> dict:
    # 1. Fetch records
    kb_future = query_knowledge_base(customer_name)
    risk_future = calculate_risk_score(customer_name)
    
    # 2. Resolve concurrent tasks
    kb_data = kb_future.result()
    risk_score = risk_future.result()
    
    # 3. Native Python business logic
    decision = "APPROVED" if risk_score < 0.25 else "MANUAL_REVIEW"
    
    # Check if this customer was audited previously on this thread
    audit_count = (previous.get("audit_count", 0) + 1) if previous else 1
    
    return {
        "customer": customer_name,
        "decision": decision,
        "risk_score": risk_score,
        "kb_summary": kb_data,
        "audit_count": audit_count
    }

config = {"configurable": {"thread_id": "cust_acme_corp"}}
turn_1 = financial_audit_pipeline.invoke("Acme Corp", config=config)
print("Turn 1:", turn_1)

# Second invocation on the same thread receives 'previous' naturally!
turn_2 = financial_audit_pipeline.invoke("Acme Corp", config=config)
print("Turn 2 Audit Count:", turn_2["audit_count"]) # Prints 2''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Functional State & Task Memoization Laboratory", divider="rainbow")

st.markdown("""
Observe how the `@task` execution engine memoizes results to avoid duplicate re-execution on repeated turns.
""")

sim = FunctionalComponentsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Task Execution Controls")
    target_task = st.selectbox("Select Task Function", ["fetch_user_profile", "calculate_vat", "query_pinecone"])
    task_param = st.text_input("Function Argument", value="user_90210")
    force_cache = st.toggle("Simulate Recovery / Cache Hit", value=True)

    if st.button("🚀 Invoke @task Function"):
        t_res = sim.run_task_simulation(target_task, task_param, force_cache)
        st.session_state["task_comp_res"] = t_res

with col_sim2:
    st.subheader("📋 Component Execution Telemetry")
    if "task_comp_res" in st.session_state:
        r = st.session_state["task_comp_res"]
        
        if "CACHE_HIT" in r["status"]:
            st.success(f" {r['status']}")
            st.info(f"Execution Latency: **{r['execution_ms']}ms** (Instant memoization reuse!)")
        else:
            st.warning(f"⚡ {r['status']}")
            st.info(f"Execution Latency: **{r['execution_ms']}ms** (Network & compute consumed)")

        st.markdown(f"**Task Output**: `{r['result']}`")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the purpose of the 'previous' keyword argument in an @entrypoint function?", expanded=False):
    st.markdown("""
    **Answer**:
    The `previous` parameter automatically receives the return value of the previous execution on the same thread.
    This enables conversational memory and state continuity across multi-turn interactions without defining explicit state channels.
    """)

with st.expander("Q2: How does @task memoization work across worker restarts?", expanded=False):
    st.markdown("""
    **Answer**:
    When a task finishes, its return value is stored in the thread checkpoint keyed by the task name and serialized argument hash.
    If execution restarts, LangGraph identifies that the task has already run and immediately resolves `.result()` from the database checkpoint.
    """)

with st.expander("Q3: Can an @entrypoint return a complex object or only primitive types?", expanded=False):
    st.markdown("""
    **Answer**:
    It can return any JSON-serializable Python data structure: dictionaries, lists, strings, numbers, or Pydantic models (if serialized).
    """)

with st.expander("Q4: What happens if an @task throws an unhandled exception?", expanded=False):
    st.markdown("""
    **Answer**:
    The exception bubbles up to the `@entrypoint`, causing execution to pause in an error state.
    Prior completed tasks remain safely committed in the checkpointer.
    """)

with st.expander("Q5: Can you stream outputs from an @entrypoint workflow?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Calling `workflow.stream(...)` emits intermediate task results and values just like a compiled `StateGraph`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Passing Non-Serializable Arguments to @task")
    st.markdown("""
    *Mechanism*: Passing raw database connection handles or active sockets as arguments to an `@task`.  
    *Consequence*: Checkpointer serialization fails with `PickleError` or `TypeError`, crashing the workflow.  
    *Fix*: Pass only primitive data (IDs, strings, dicts) to `@task` functions.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Forgetting Keyword-Only Syntax for 'previous'")
    st.markdown("""
    *Mechanism*: Defining `def my_entry(user_input, previous=None):` without the `*` separator.  
    *Consequence*: LangGraph does not inject the previous state properly or treats it as a positional argument.  
    *Fix*: Always declare as keyword-only: `def my_entry(user_input, *, previous=None):`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Large File Bloat in Task Return Values")
    st.markdown("""
    *Mechanism*: Returning a 200MB binary file directly from an `@task`.  
    *Consequence*: The binary is written to the checkpointer database, causing severe I/O lag and storage bloat.  
    *Fix*: Save large files to S3/GCS and return the object URI string.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Not Using Tasks for Expensive Operations")
    st.markdown("""
    *Mechanism*: Making heavy external LLM calls directly inside `@entrypoint` without `@task`.  
    *Consequence*: If the workflow crashes later, the expensive LLM call is re-run because it wasn't cached in an `@task`.  
    *Fix*: Wrap every costly external call in an `@task`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**@entrypoint**\\nThe root decorator establishing thread sessions, persistence, and previous state.")
with col_fc2:
    st.info("**@task**\\nAn atomic unit of work whose outputs are cached in the checkpoint store.")
with col_fc3:
    st.info("**previous=None**\\nThe keyword argument injecting the last run's output for conversational continuity.")
with col_fc4:
    st.info("**Task Memoization**\\nReusing previously computed task results upon workflow resumption.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the 5 core components of the LangGraph Functional API.", value=True)
c2 = st.checkbox("I know how the previous argument injects prior state across thread turns.", value=True)
c3 = st.checkbox("I understand how @task memoization avoids redundant computation during recovery.", value=True)
c4 = st.checkbox("I know how to resolve task futures using the .result() method call.", value=True)
c5 = st.checkbox("I can attach persistent checkpointers to @entrypoint functions.", value=True)
c6 = st.checkbox("I know why non-serializable objects must never be passed as @task arguments.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 23.2 'Core Components' with full architectural and practical comprehension!")
