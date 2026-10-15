import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="23.1 Functional API Basics - LangGraph Curriculum",
    page_icon="⚡",
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
    .badge-func {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-graph {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
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
class FunctionalApiSimulator:
    """
    Simulates Functional API vs Graph API execution:
    - @entrypoint and @task decorator lifecycles
    - Implicit Pregel state creation
    - Checkpoint persistence in functional Python workflows
    """
    def __init__(self):
        pass

    def simulate_comparison(self, paradigm: str, user_prompt: str) -> Dict[str, Any]:
        trace = []
        if paradigm == "Functional API (@entrypoint + @task)":
            trace.append({"step": 1, "component": "@entrypoint(checkpointer=checkpointer)", "action": f"Ingested prompt: '{user_prompt}'. Created implicit task context."})
            trace.append({"step": 2, "component": "@task extract_entities()", "action": "Extracted key nouns and numbers asynchronously."})
            trace.append({"step": 3, "component": "Standard Python if-statement", "action": "Evaluated conditional logic using native Python 'if len(entities) > 0:'."})
            trace.append({"step": 4, "component": "@task synthesize_answer()", "action": "Synthesized final deliverable and saved snapshot to checkpointer."})
            lines_of_code = 18
            boilerplate_score = "Low (Zero GraphBuilder, Zero State TypedDict)"
        else: # Graph API (StateGraph)
            trace.append({"step": 1, "component": "StateGraph(StateSchema)", "action": "Explicit State TypedDict initialized."})
            trace.append({"step": 2, "component": "builder.add_node('extract', ...)", "action": "Registered extract node in Pregel DAG."})
            trace.append({"step": 3, "component": "builder.add_conditional_edges()", "action": "Configured formal edge routing function."})
            trace.append({"step": 4, "component": "app.compile()", "action": "Compiled graph against checkpointer."})
            lines_of_code = 42
            boilerplate_score = "Higher (Explicit Edges & Reducers Required)"

        return {
            "paradigm": paradigm,
            "lines_of_code": lines_of_code,
            "boilerplate_score": boilerplate_score,
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Functional API Execution Model and Pregel Integration", divider="rainbow")

st.markdown("""
For many engineers, expressing every AI workflow as an explicit graph with `StateGraph`, `builder.add_node()`, and `builder.add_conditional_edges()` introduces excessive structural boilerplate.
Python developers already know how to write control flow using standard functions, `if`/`else` branches, and `for` loops.

LangGraph introduced the **Functional API** (`@entrypoint` and `@task`).
The Functional API allows developers to write standard Python functions while gaining the full enterprise superpowers of LangGraph:
- **Durable Execution & Checkpointing**: Functions can be paused, resumed, and inspected across crashes.
- **Human-in-the-Loop**: Functions can call `interrupt()` anywhere in their execution.
- **Streaming**: Native streaming of intermediate task results and LLM tokens.
- **Zero Graph Boilerplate**: No need to define a dedicated `TypedDict` state schema or register nodes and edges manually!

The blueprint below illustrates how `@entrypoint` and `@task` map directly into the underlying LangGraph Pregel execution engine.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FunctionalApiTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_entrypoint {
            label="1. Top-Level Workflow Boundary (@entrypoint)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_invoke [label="app.invoke(input, config={'thread_id': 'th_101'})\\nInvokes @entrypoint Function", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            entry_func [label="@entrypoint(checkpointer=checkpointer)\\ndef workflow_entrypoint(query):\\n(Owns Thread Session & Checkpoint Writes)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_tasks {
            label="2. Discrete Checkpointed Units of Work (@task)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_a [label="@task\\ndef fetch_records(query):\\n(Result Cached in Checkpoint)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            task_b [label="@task\\ndef run_llm_analysis(records):\\n(Cached; Skips on Resume)", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            py_control [label="Native Python Control Flow\\nif len(records) > 0: ... else: ...", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_pregel_backend {
            label="3. Underlying LangGraph Pregel Runtime";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            pregel_tasks [label="Pregel Task Scheduler\\nTracks Task Results & Deduplicates Calls", fillcolor="#18181B", fontcolor="#E4E4E7", color="#71717A"];
            db_saver [label="Persistent Store (PostgresSaver / MemorySaver)\\nState Persisted Automatically", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            client_res [label="Returned Return Value\\nStandard Python Result", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        user_invoke -> entry_func;
        entry_func -> task_a [label="Calls Task A"];
        task_a -> py_control [label="Returns Records"];
        py_control -> task_b [label="Conditional Branch"];
        
        task_a -> pregel_tasks [label="Result Checkpointed", style=dashed];
        task_b -> pregel_tasks [label="Result Checkpointed", style=dashed];
        pregel_tasks -> db_saver;

        task_b -> entry_func [label="Returns Analysis"];
        entry_func -> client_res;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: WHAT IS THE FUNCTIONAL API?
# ==============================================================================
st.header("2. What is the Functional API?: Decorator-Driven Workflows with Native Persistence", divider="rainbow")

st.markdown("""
The **Functional API** in LangGraph provides a lightweight, functional alternative to building explicit graph DAGs:
- Decorate your top-level workflow function with **`@entrypoint`**.
- Decorate individual asynchronous or synchronous sub-tasks with **`@task`**.
- Return standard Python data structures directly.

Whenever an `@entrypoint` executes, LangGraph wraps it in a Pregel execution graph behind the scenes.
Any `@task` called inside the entrypoint is automatically tracked, checkpointed, and cached!
""")

st.code('''from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

@task
def step_one(text: str) -> str:
    return text.upper()

@task
def step_two(text: str) -> str:
    return f"Processed: {text}"

# Top-level entrypoint with persistence
@entrypoint(checkpointer=MemorySaver())
def my_workflow(input_str: str) -> str:
    # Native Python execution flow!
    res1 = step_one(input_str).result()
    res2 = step_two(res1).result()
    return res2

# Invocable just like a compiled StateGraph!
result = my_workflow.invoke("hello langgraph", config={"configurable": {"thread_id": "func_01"}})
print(result) # "Processed: HELLO LANGGRAPH"''', language="python")

# ==============================================================================
# SECTION 3: GRAPH API VS FUNCTIONAL API
# ==============================================================================
st.header("3. Graph API vs Functional API: Explicit Graphs vs Idiomatic Python Procedures", divider="rainbow")

st.markdown("""
Choosing between the Graph API (`StateGraph`) and the Functional API (`@entrypoint`) depends on team preferences and architectural complexity:
""")

col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("#### Graph API (`StateGraph`)")
    st.markdown("""
    - **Paradigm**: Explicit Declarative Graph (Nodes, Edges, State Schema).
    - **Best For**: Complex multi-agent topologies, cyclical feedback loops (reflection, debate), dynamic map-reduce with the Send API.
    - **Pros**: Explicit visual graph diagram, channel reducers (`add_messages`).
    - **Cons**: More boilerplate code for linear or simple procedural flows.
    """)

with col_g2:
    st.markdown("#### Functional API (`@entrypoint` / `@task`)")
    st.markdown("""
    - **Paradigm**: Idiomatic Imperative Python.
    - **Best For**: Adding persistence and HITL to existing Python code, simple sequential pipelines, procedural data workflows.
    - **Pros**: Zero state schema boilerplate, native `if`/`for` loops, familiar function syntax.
    - **Cons**: Less intuitive for dense cyclic multi-agent meshes.
    """)

# ==============================================================================
# SECTION 4: WHEN TO USE THE FUNCTIONAL API
# ==============================================================================
st.header("4. When to Use the Functional API: Decision Matrix and Architectural Heuristics", divider="rainbow")

st.markdown("""
Use this heuristic matrix to decide whether to use `@entrypoint` or `StateGraph`:
""")

df_matrix = pd.DataFrame({
    "Criterion": [
        "Control Flow Style",
        "State Management",
        "Loops / Cycles",
        "Migration Path",
        "Parallel Tasks",
        "Human-in-the-Loop",
        "Streaming Support"
    ],
    "Use Functional API (@entrypoint)": [
        "Procedural Python (if, for, while)",
        "Standard function arguments & returns",
        "Small deterministic loops",
        "Wrapping existing legacy Python services",
        "Supported via multiple task.result() awaits",
        "Supported via interrupt()",
        "Fully supported (stream and astream)"
    ],
    "Use Graph API (StateGraph)": [
        "Complex State Machine (Edges & Routers)",
        "Global Blackboard with Channel Reducers",
        "Heavy multi-turn agentic cycles (ReAct, Debate)",
        "Building new greenfield agent systems",
        "Supported via Send API and parallel branches",
        "Supported via interrupt() and checkpointers",
        "Fully supported (stream and astream)"
    ]
})
st.dataframe(df_matrix, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 5: EXISTING PYTHON WORKFLOWS
# ==============================================================================
st.header("5. Existing Python Workflows: Incrementally Adding Durability and Memory to Legacy Code", divider="rainbow")

st.markdown("""
A common challenge in production engineering is retrofitting persistence and fault tolerance into an existing 500-line Python script.
Rewriting a functional script into a `StateGraph` requires restructuring everything into node dictionaries and edge functions.

With the **Functional API**, migration is non-invasive:
1. Add `@entrypoint(checkpointer=...)` above the main execution function.
2. Add `@task` above the 2 or 3 expensive external API calls (LLM calls, database queries).
3. Call `.result()` to resolve task outcomes.
4. The legacy script now has full crash-resilience and time-travel inspection without altering business logic!
""")

# ==============================================================================
# SECTION 6: FUNCTION-BASED WORKFLOWS
# ==============================================================================
st.header("6. Function-Based Workflows: Composition, Typing, and Native Control Flow", divider="rainbow")

st.markdown("""
Because `@task` functions return a **Task Future**, you can compose them using standard Python concurrency patterns:
""")

st.code('''# Parallel Task Execution in Functional API
@entrypoint(checkpointer=MemorySaver())
def parallel_analyst(topic: str) -> dict:
    # Dispatch two tasks concurrently
    future_a = task_fetch_news(topic)
    future_b = task_fetch_financials(topic)
    
    # Await results (Pregel executes them in parallel!)
    news_res = future_a.result()
    fin_res = future_b.result()
    
    return {"news": news_res, "financials": fin_res}''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Functional API Application with Checkpointing", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a Functional API workflow with `@entrypoint`, `@task`, and persistence.
""")

st.code('''from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

# 1. Define atomic tasks
@task
def clean_input(text: str) -> str:
    return text.strip().lower()

@task
def generate_summary(cleaned_text: str) -> str:
    return f"Summary of: {cleaned_text}"

@task
def audit_summary(summary: str) -> bool:
    return len(summary) > 10

# 2. Define main workflow entrypoint
@entrypoint(checkpointer=MemorySaver())
def process_document_workflow(raw_input: str) -> dict:
    # Step 1: Clean
    cleaned = clean_input(raw_input).result()
    
    # Step 2: Summarize
    summary = generate_summary(cleaned).result()
    
    # Step 3: Native Python conditional logic!
    is_valid = audit_summary(summary).result()
    
    if is_valid:
        status = "APPROVED"
    else:
        status = "REJECTED_TOO_SHORT"
        
    return {
        "status": status,
        "summary": summary
    }

# 3. Invocation with thread persistence
config = {"configurable": {"thread_id": "functional_prod_01"}}
result = process_document_workflow.invoke("  Enterprise AI Infrastructure Report  ", config=config)
print("Workflow Result:", result)''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Graph API vs Functional API Paradigm Explorer", divider="rainbow")

st.markdown("""
Compare the execution traces, boilerplate overhead, and code ergonomics between the Functional API and the Graph API.
""")

sim = FunctionalApiSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Paradigm Selector")
    paradigm_choice = st.radio(
        "Select LangGraph API Paradigm",
        ["Functional API (@entrypoint + @task)", "Graph API (StateGraph)"]
    )
    user_prompt_in = st.text_input("Workflow Input", value="Audit Q4 Compliance Logs")

    if st.button("🚀 Simulate Execution Flow"):
        res = sim.simulate_comparison(paradigm_choice, user_prompt_in)
        st.session_state["func_api_res"] = res

with col_sim2:
    st.subheader("📋 Architectural Trace")
    if "func_api_res" in st.session_state:
        r = st.session_state["func_api_res"]
        st.info(f"Paradigm: **{r['paradigm']}**")
        st.markdown(f"**Estimated Lines of Code**: `{r['lines_of_code']}`")
        st.markdown(f"**Boilerplate Rating**: `{r['boilerplate_score']}`")
        st.markdown("#### Execution Steps:")
        st.dataframe(pd.DataFrame(r["trace"]), use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the LangGraph Functional API?", expanded=False):
    st.markdown("""
    **Answer**:
    The Functional API is a decorator-based programming model (`@entrypoint` and `@task`) that brings LangGraph's durability, checkpointing, streaming, and human-in-the-loop capabilities to idiomatic Python functions without requiring explicit `StateGraph` definitions.
    """)

with st.expander("Q2: How does @task handle caching during workflow recovery?", expanded=False):
    st.markdown("""
    **Answer**:
    When a workflow is resumed from a checkpoint, the Pregel engine checks if an `@task` has already completed for the given arguments.
    If already completed, LangGraph reuses the cached result without re-executing the task function, saving time and API costs.
    """)

with st.expander("Q3: Why must you call .result() on an @task function?", expanded=False):
    st.markdown("""
    **Answer**:
    Calling an `@task` returns a Task Future representing an asynchronous unit of work tracked by the Pregel scheduler.
    Calling `.result()` resolves the future, returning the actual computed value.
    """)

with st.expander("Q4: Can you use interrupts with the Functional API?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! You can call `interrupt(...)` anywhere inside an `@entrypoint` function.
    Execution pauses at that line, state is committed to the checkpointer, and calling `app.invoke(Command(resume=...))` resumes from that line.
    """)

with st.expander("Q5: When should an engineer still prefer the Graph API over the Functional API?", expanded=False):
    st.markdown("""
    **Answer**:
    When building multi-agent systems with complex cyclic feedback loops, blackboard state architectures with channel reducers (`add_messages`), or dynamic multi-node parallel map-reduce topologies.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting .result() on @task Futures")
    st.markdown("""
    *Mechanism*: Writing `data = my_task(query)` and passing `data` into a string formatter.  
    *Consequence*: `data` is a Future object, not the string, resulting in `TypeError` or mangled output.  
    *Fix*: Always call `my_task(query).result()`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Mutating Global Non-Deterministic State Inside @task")
    st.markdown("""
    *Mechanism*: Relying on random numbers or non-persisted global dictionaries inside an `@task`.  
    *Consequence*: Breaks deterministic replay and recovery when the workflow is resumed from a crash.  
    *Fix*: Ensure `@task` functions are pure functions dependent only on their input arguments.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Calling @task Outside of an @entrypoint")
    st.markdown("""
    *Mechanism*: Invoking an `@task` from a normal undecorated Python function.  
    *Consequence*: Raises runtime error because the Pregel execution context is not active.  
    *Fix*: Only call `@task` from within an `@entrypoint` or another `@task`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Awaiting Tasks Sequentially Instead of Fan-Out")
    st.markdown("""
    *Mechanism*: Calling `task_a().result()` and then `task_b().result()` when they are independent.  
    *Consequence*: Runs them sequentially, doubling execution latency.  
    *Fix*: Dispatch both futures first (`f1 = task_a()`, `f2 = task_b()`), then resolve (`f1.result()`, `f2.result()`).
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**@entrypoint**\\nDecorator defining the top-level durable workflow boundary.")
with col_fc2:
    st.info("**@task**\\nDecorator defining a discrete, checkpointable, cached unit of execution.")
with col_fc3:
    st.info("**.result()**\\nThe method call used to resolve a Task Future into its computed value.")
with col_fc4:
    st.info("**Idiomatic Control Flow**\\nUsing native Python if/for statements rather than explicit graph edges.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the purpose and ergonomics of the LangGraph Functional API.", value=True)
c2 = st.checkbox("I know how @entrypoint and @task map into the underlying Pregel engine.", value=True)
c3 = st.checkbox("I understand the architectural trade-offs between Functional API and Graph API.", value=True)
c4 = st.checkbox("I know why .result() must be called on task futures to retrieve values.", value=True)
c5 = st.checkbox("I know how to retrofit persistence into existing Python scripts without full graph rewrites.", value=True)
c6 = st.checkbox("I understand how to dispatch parallel tasks concurrently using the Functional API.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 23.1 'Functional API Basics' with full architectural and practical comprehension!")
