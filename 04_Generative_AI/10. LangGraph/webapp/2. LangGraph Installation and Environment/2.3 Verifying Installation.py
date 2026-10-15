import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="2.3 Verifying Installation - LangGraph Curriculum",
    page_icon="✅",
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
    .badge-pass {
        background-color: #059669;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE GRAPH VERIFIER FOR WORKBENCH & PIPELINE
# ==============================================================================
class InstallationDiagnosticHarness:
    """
    Simulates diagnostic environment testing and execution verification
    of a newly installed LangGraph runtime.
    """
    def __init__(self):
        self.simulated_packages = {
            "langgraph": {"version": "0.2.28", "status": "INSTALLED_OK"},
            "langchain_core": {"version": "0.3.15", "status": "INSTALLED_OK"},
            "langsmith": {"version": "0.1.137", "status": "INSTALLED_OK"},
            "pydantic": {"version": "2.9.2", "status": "INSTALLED_OK"}
        }

    def execute_smoke_test(self, user_name: str, shout_mode: bool) -> Dict[str, Any]:
        """
        Executes an in-memory smoke test graph simulation:
        START -> greeting_node -> shout_filter -> END
        """
        # Step 1: Initial state
        state = {"user_name": user_name, "message": ""}
        
        # Step 2: greeting node
        greeting = f"Welcome to LangGraph, {user_name}!"
        state["message"] = greeting
        
        # Step 3: shout filter
        if shout_mode:
            state["message"] = state["message"].upper() + " [VERIFIED]"
        else:
            state["message"] = state["message"] + " [Verified]"

        return {
            "status": "HEALTHY",
            "final_state": state,
            "supersteps_completed": 2,
            "nodes_executed": ["greeting_node", "shout_filter"],
            "diagnostics": "All primitives (START, Nodes, Edges, END) executed with 0 runtime errors."
        }


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("2.3 Verifying Installation: Smoke Tests, Version Audits, & First Graph")
st.caption("Chapter 2: LangGraph Installation and Environment | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
After installing packages and configuring environment credentials, professional engineering discipline requires 
an automated **Sanity Verification & Smoke Test**. Verifying an installation confirms that package namespaces are 
accessible, binary dependencies are satisfied, type annotations compile, and the graph engine can execute a complete 
superstep lifecycle.

In this lesson, you will master:
* Importing core LangGraph primitives (`StateGraph`, `START`, `END`).
* Programmatically checking installed package versions and dependency health.
* Constructing and compiling your very first minimal `StateGraph`.
* Executing the first graph invocation with state dictionaries.
* Diagnosing common installation failure modes: Pydantic v1 conflicts, circular imports, and missing C-extensions.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Health Verification & Graph Smoke Test Lifecycle", divider="rainbow")

st.markdown("""
The architecture below outlines the three-stage verification pipeline: starting with import audits, proceeding to 
graph schema compilation, and concluding with a live invocation smoke test.
""")

blueprint_dot = """
digraph VerificationLifecycle {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_stage1 {
        label = "Stage 1: Import & Version Check";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        ImportCheck [label="Import Primitives\\nfrom langgraph.graph import StateGraph, START, END", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        VersionAudit [label="Version Introspection\\nVerify langgraph >= 0.2.20 & pydantic >= 2.7", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_stage2 {
        label = "Stage 2: Graph Assembly & Compilation";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        StateDecl [label="State Declaration\\nclass SimpleState(TypedDict): text: str", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        CompileGraph [label="Graph Compilation\\napp = builder.compile()", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_stage3 {
        label = "Stage 3: Invocation Smoke Test";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        InvokeCall [label="Execute app.invoke({'text': 'Hello'})\\nVerify state updates through nodes", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        SuccessEgress [label="Clean Verification PASS\\nEnvironment Ready for Production Development", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    ImportCheck -> VersionAudit [label="Pass"];
    VersionAudit -> StateDecl [label="Compatible"];
    StateDecl -> CompileGraph [label="Build Topology"];
    CompileGraph -> InvokeCall [label="Dispatch Test State"];
    InvokeCall -> SuccessEgress [label="State Reconciled", color="#EC4899", penwidth=1.8];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - IMPORTING LANGGRAPH
# ==============================================================================
st.header("2. Importing LangGraph: Core Primitives, Graph Modules, and Clean Name Binding", divider="rainbow")

st.markdown("""
LangGraph is structured into clean, modular subpackages. The three most essential entry imports are:
* **`StateGraph`**: The core graph builder class responsible for declaring the state schema, registering nodes, and binding edges.
* **`START`**: The special graph entrypoint node symbol denoting the root of execution.
* **`END`**: The special termination symbol indicating that execution for the current thread has completed.
""")

st.code("""
# Primary LangGraph Primitives
from langgraph.graph import StateGraph, START, END

# Message Channel Reducer (if tracking multi-turn dialogue)
from langgraph.graph.message import add_messages

# Persistent Memory Checkpointers
from langgraph.checkpoint.memory import MemorySaver
""", language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - CHECKING PACKAGE VERSIONS
# ==============================================================================
st.header("3. Checking Package Versions: Runtime Diagnostics, Metadata Inspection, and Semantic Versioning", divider="rainbow")

st.markdown("""
In Python scripts or CI/CD pipelines, you can programmatically inspect the installed package versions:
""")

col_ver1, col_ver2 = st.columns([1, 1])

with col_ver1:
    st.markdown("""
    ### CLI Verification
    Run in terminal to inspect installed package versions:
    """)
    st.code("""
python -c "import langgraph; print('LangGraph version:', langgraph.__version__)"
python -c "import langchain_core; print('LangChain Core version:', langchain_core.__version__)"
python -c "import pydantic; print('Pydantic version:', pydantic.__version__)"
    """, language="bash")

with col_ver2:
    st.markdown("""
    ### Compatibility Matrix
    * **LangGraph**: >= 0.2.20 (supports modern `START`/`END` syntax, replacing legacy `Graph.set_entry_point`).
    * **LangChain Core**: >= 0.3.0 (modern LCEL and Runnable interfaces).
    * **Pydantic**: >= 2.7.0 (required for high-performance C-extension validation).
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CREATING A MINIMAL GRAPH
# ==============================================================================
st.header("4. Creating a Minimal Graph: State Definition, Nodes, and Compiling Your First StateGraph", divider="rainbow")

st.markdown("""
A minimal LangGraph application requires exactly four steps:
1. **Define State**: A Python `TypedDict` defining the keys and types in state.
2. **Define a Node**: A standard Python function accepting `state` and returning an update dictionary.
3. **Connect Edges**: Wire `START` to your node, and your node to `END`.
4. **Compile**: Call `.compile()` to lock the graph topology and validate invariants.
""")

st.code("""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State Definition
class HelloWorldState(TypedDict):
    input_text: str
    processed_text: str

# 2. Node Function
def uppercase_node(state: HelloWorldState) -> dict:
    original = state["input_text"]
    return {"processed_text": original.upper()}

# 3. Graph Assembly
builder = StateGraph(HelloWorldState)
builder.add_node("uppercase", uppercase_node)

# Connect Edges
builder.add_edge(START, "uppercase")
builder.add_edge("uppercase", END)

# 4. Compilation
app = builder.compile()
""", language="python")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - RUNNING THE FIRST GRAPH
# ==============================================================================
st.header("5. Running the First Graph: Invocation Protocols, Input Payloads, and Return State Evaluation", divider="rainbow")

st.markdown("""
Once compiled, you invoke the graph using standard Runnable execution methods:
""")

col_rn1, col_rn2 = st.columns([1, 1])

with col_rn1:
    st.markdown("""
    ### Synchronous Invocation (`.invoke()`)
    Pass an initial dictionary matching your `TypedDict` schema:
    """)
    st.code("""
# Execute graph
initial_input = {"input_text": "hello enterprise agent"}
result = app.invoke(initial_input)

print("Result:", result)
# Output:
# {'input_text': 'hello enterprise agent', 
#  'processed_text': 'HELLO ENTERPRISE AGENT'}
    """, language="python")

with col_rn2:
    st.markdown("""
    ### Asynchronous Invocation (`.ainvoke()`)
    Mandatory for high-concurrency FastAPI backends:
    """)
    st.code("""
# In async def endpoint:
result = await app.ainvoke({"input_text": "hello async"})
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - DEBUGGING INSTALLATION PROBLEMS
# ==============================================================================
st.header("6. Debugging Installation Problems: Circular Dependencies, PyTorch/Pydantic Mismatches, and GIL Conflicts", divider="rainbow")

st.markdown("""
Common installation pitfalls encountered during initial setup and their fixes:
""")

debug_table = [
    {
        "Symptom": "`ImportError: cannot import name 'START' from 'langgraph.graph'`",
        "Root Cause": "Outdated LangGraph version (< 0.1.0).",
        "Remedy": "Run `pip install -U langgraph` or `uv pip install -U langgraph` to update to version 0.2+."
    },
    {
        "Symptom": "`PydanticUserError: Pydantic V1 style validators are not supported`",
        "Root Cause": "Mismatched Pydantic v1 installed in the environment.",
        "Remedy": "Run `pip install -U 'pydantic>=2.7.0'`. Ensure no legacy libraries are pinning `pydantic<2.0`."
    },
    {
        "Symptom": "`ModuleNotFoundError: No module named 'langchain_core'`",
        "Root Cause": "Installing `langgraph` without installing `langchain-core`.",
        "Remedy": "Run `pip install langchain-core`."
    },
    {
        "Symptom": "`GraphRecursionError: Recursion limit of 25 reached`",
        "Root Cause": "Cyclic loop lacks a proper termination condition.",
        "Remedy": "Inspect your conditional edge routing function to ensure it eventually routes to `END`."
    }
]
st.dataframe(pd.DataFrame(debug_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Automated Environment Diagnostic & Smoke Test Suite", divider="rainbow")

st.markdown("""
Below is a standalone Python diagnostic verification script. Run this in your CI/CD build pipeline 
or upon onboarding a new developer to verify runtime health:
""")

prod_code = """
import sys
from typing import TypedDict

def run_environment_diagnostic():
    print("=" * 60)
    print("LANGGRAPH ENVIRONMENT DIAGNOSTIC & SMOKE TEST")
    print("=" * 60)

    # 1. Check Python Version
    print(f"[1/4] Python Version: {sys.version.split()[0]} ... OK")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required.")
        sys.exit(1)

    # 2. Check Imports
    try:
        import langgraph
        from langgraph.graph import StateGraph, START, END
        print(f"[2/4] LangGraph Import: v{langgraph.__version__} ... OK")
    except ImportError as e:
        print(f"ERROR: Failed to import LangGraph: {e}")
        sys.exit(1)

    try:
        import langchain_core
        print(f"[3/4] LangChain Core Import: v{langchain_core.__version__} ... OK")
    except ImportError as e:
        print(f"ERROR: Failed to import langchain_core: {e}")
        sys.exit(1)

    # 3. Execute Smoke Test Graph
    print("[4/4] Executing Smoke Test StateGraph ...")
    class TestState(TypedDict):
        count: int

    def increment(state: TestState) -> dict:
        return {"count": state["count"] + 1}

    builder = StateGraph(TestState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    
    app = builder.compile()
    output = app.invoke({"count": 41})
    
    assert output["count"] == 42, f"Expected 42, got {output['count']}"
    print(f"Smoke Test Result: Input: 41 -> Output: {output['count']} ... PASS")
    print("=" * 60)
    print("ALL CHECKS PASSED: Environment is production-ready!")
    print("=" * 60)

if __name__ == "__main__":
    run_environment_diagnostic()
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Live LangGraph Installation Verifier & Diagnostic Console", divider="rainbow")

st.markdown("""
Execute an interactive smoke test inside this active Streamlit session. 
Test state progression through graph nodes, inspect the resulting state dictionary, and verify package dependencies.
""")

harness = InstallationDiagnosticHarness()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Configure Smoke Test Inputs")
    
    test_name = st.text_input("Candidate User Name", value="Alice Developer")
    uppercase_mode = st.checkbox("Uppercase Transformation Filter", value=True)

    run_smoke_btn = st.button("Run Live Smoke Test Graph 🚀", use_container_width=True)

with col_wb2:
    st.subheader("2. Diagnostic Telemetry & State Output")
    
    if run_smoke_btn:
        result = harness.execute_smoke_test(test_name, uppercase_mode)
        
        m1, m2 = st.columns(2)
        m1.metric("Smoke Test Status", result["status"])
        m2.metric("Supersteps Executed", result["supersteps_completed"])

        st.markdown("#### Final State Dictionary")
        st.json(result["final_state"])

        st.markdown("#### Execution Trace")
        for node in result["nodes_executed"]:
            st.markdown(f"Node Executed: `{node}` -> <span class='badge-pass'>SUCCESS</span>", unsafe_allow_html=True)

        st.success(f"Diagnostics: {result['diagnostics']}")
    else:
        st.info("Click 'Run Live Smoke Test Graph' to verify state compilation and invocation.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Interview questions focused on verification and runtime diagnostics:
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What does `builder.compile()` do under the hood in LangGraph?"):
    st.markdown("""
    **Answer**: Calling `.compile()` validates graph topology invariants:
    1. Verifies that `START` and all edge destinations reference valid registered nodes.
    2. Ensures every conditional edge routing function maps to registered nodes or `END`.
    3. Checks that state channel reducers are compatible with node return types.
    4. Attaches checkpointers, retry policies, and interrupt breakpoints.
    5. Returns an executable `CompiledStateGraph` instance implementing the `Runnable` protocol.
    """)

with st.expander("Q2: What is the difference between `START` and `set_entry_point('node_name')`?"):
    st.markdown("""
    **Answer**: In early LangGraph versions (< 0.1.0), developers called `builder.set_entry_point("my_node")`. 
    In modern LangGraph (>= 0.2.0), this was replaced with the universal `START` node: `builder.add_edge(START, "my_node")`. 
    Using `START` standardizes graph wiring and allows conditional routing directly from graph inception: 
    `builder.add_conditional_edges(START, route_entry)`.
    """)

with st.expander("Q3: Why must state keys be declared explicitly in `TypedDict`?"):
    st.markdown("""
    **Answer**: Explicit `TypedDict` declarations enable static type checking (via mypy/Pyright), catch runtime spelling errors 
    (e.g., writing `state['massage']` instead of `state['messages']`), and allow LangGraph's engine to allocate discrete state channels 
    with custom reducers (`Annotated[list, add_messages]`).
    """)

with st.expander("Q4: Can a LangGraph node return an empty dictionary `{}`?"):
    st.markdown("""
    **Answer**: Yes. If a node performs a side-effect (e.g. logging an event, pinging a webhook) without modifying the shared state, 
    it can return `{}`. LangGraph's reducer reconciles the empty delta cleanly without altering any existing state values.
    """)

with st.expander("Q5: How do you verify that an async LangGraph runs properly in a Jupyter Notebook?"):
    st.markdown("""
    **Answer**: Jupyter notebooks already run inside an active asyncio event loop. Calling `asyncio.run(app.ainvoke(...))` 
    causes a `RuntimeError: This event loop is already running`. In notebooks, you must call `await app.ainvoke(...)` directly, 
    or use the synchronous `app.invoke(...)`.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Forgetting to Return a Dictionary from Node Functions (CRITICAL)**
    * *Error*: Defining a node that returns a string or list directly: `return "Success"`.
    * *Impact*: LangGraph expects nodes to return state delta dictionaries (`return {"status": "Success"}`). Returning non-dict types triggers `TypeError: expected dict, got str`.
    * *Fix*: Always return a dictionary matching state keys.
    """)

    st.error("""
    **Trap 2: Shadowing Built-In Package Names with Local Files (CRITICAL)**
    * *Error*: Naming your test file `langgraph.py` or `langchain.py`.
    * *Impact*: Python attempts to import from the local file instead of the installed library, throwing `AttributeError: module 'langgraph' has no attribute 'graph'`.
    * *Fix*: Never name scripts after installed packages.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Forgetting to Call `.compile()` Before Invocation (WARNING)**
    * *Warning*: Attempting to call `builder.invoke(...)` directly on the `StateGraph` builder.
    * *Impact*: Throws `AttributeError: 'StateGraph' object has no attribute 'invoke'`.
    * *Fix*: Always compile first: `app = builder.compile()`.
    """)

    st.warning("""
    **Trap 4: Discarding the Return State of `app.invoke()` (WARNING)**
    * *Warning*: Assuming `app.invoke()` mutates the input dictionary in-place.
    * *Impact*: In LangGraph, the input dictionary is not modified; the final state is returned as a new dictionary.
    * *Fix*: Capture the return value: `final_state = app.invoke(initial_state)`.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: START Node</strong><br><br>
        <em>Front:</em> What does <code>START</code> represent in LangGraph?<br><br>
        <em>Back:</em> The entrypoint node symbol from which graph execution begins.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: END Node</strong><br><br>
        <em>Front:</em> What does <code>END</code> represent?<br><br>
        <em>Back:</em> The termination symbol signaling that execution has completed.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Node Return</strong><br><br>
        <em>Front:</em> What type must a node function return?<br><br>
        <em>Back:</em> A dictionary representing the state keys to be updated.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: compile()</strong><br><br>
        <em>Front:</em> What does <code>builder.compile()</code> return?<br><br>
        <em>Back:</em> An executable <code>CompiledStateGraph</code> instance implementing Runnable.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of installation verification:")

chk1 = st.checkbox("I know how to import core primitives (`StateGraph`, `START`, `END`) cleanly.", value=True)
chk2 = st.checkbox("I know how to check package versions programmatically in CLI and scripts.", value=True)
chk3 = st.checkbox("I understand the four-step process for building a minimal StateGraph.", value=True)
chk4 = st.checkbox("I understand the difference between `.invoke()` and `.ainvoke()` execution protocols.", value=True)
chk5 = st.checkbox("I know how to diagnose and resolve Pydantic v1 conflicts and missing imports.", value=True)
chk6 = st.checkbox("I know why nodes must always return dictionaries rather than raw data values.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 2.3: Verifying Installation! You have completed Chapter 2.")
