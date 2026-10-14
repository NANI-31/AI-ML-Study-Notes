import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="24.1 Debugging LangGraph - LangGraph Curriculum",
    page_icon="🔍",
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
    .badge-debug {
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
class DebuggingSimulator:
    """
    Simulates graph debugging lifecycles:
    - Node output inspection
    - State update verification
    - Conditional edge routing traces
    - Loop recursion limit warnings
    """
    def __init__(self):
        pass

    def run_debug_trace(
        self,
        query: str,
        simulate_loop_bug: bool
    ) -> Dict[str, Any]:
        trace = []
        # Step 1: Input
        trace.append({
            "step": 1,
            "superstep_barrier": "START -> agent_node",
            "node": "agent_node",
            "input_state": {"query": query, "messages_count": 1},
            "output_delta": {"draft": "Proposed solution", "tool_needed": simulate_loop_bug},
            "status": "OK"
        })

        if simulate_loop_bug:
            # Bug: tool returns data, but agent keeps requesting tool again
            for turn in range(2, 5):
                trace.append({
                    "step": turn,
                    "superstep_barrier": "tool_node -> agent_node",
                    "node": "agent_node",
                    "input_state": {"tool_output": f"Result {turn}", "loop_count": turn},
                    "output_delta": {"tool_needed": True},
                    "status": "RECURSION_RISK" if turn >= 4 else "LOOPING"
                })
            diagnosis = "Infinite Loop Detected: Agent node continually generates tool_needed=True without reaching termination criteria."
        else:
            trace.append({
                "step": 2,
                "superstep_barrier": "agent_node -> validator_node",
                "node": "validator_node",
                "input_state": {"draft": "Proposed solution"},
                "output_delta": {"approved": True, "final_answer": "Validated deliverable"},
                "status": "OK"
            })
            trace.append({
                "step": 3,
                "superstep_barrier": "validator_node -> END",
                "node": "__end__",
                "input_state": {"approved": True},
                "output_delta": {},
                "status": "TERMINATED_CLEANLY"
            })
            diagnosis = "Normal Execution: State transitions and routing resolved cleanly."

        return {
            "query": query,
            "has_bug": simulate_loop_bug,
            "diagnosis": diagnosis,
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Debugging and Diagnostic Lifecycle in LangGraph", divider="rainbow")

st.markdown("""
Debugging stateful, multi-agent graphs is fundamentally different from debugging traditional sequential Python code.
In LangGraph:
- Execution proceeds in discrete **atomic supersteps**.
- Parallel nodes run concurrently and merge updates through **channel reducers**.
- Control flow diverges dynamically across **conditional edges**.
- Checkpoints commit state snapshots at every barrier synchronization.

Effective debugging requires visibility into every tier of this runtime:
1. **Node Boundary Tier**: Inspecting the exact input dictionary received and delta returned.
2. **Channel Reducer Tier**: Verifying how channel updates are merged into state.
3. **Routing Tier**: Auditing the return values of conditional edge functions.
4. **Lifecycle Tier**: Monitoring loop iterations against `recursion_limit`.

The blueprint below illustrates this comprehensive diagnostic topology.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DebuggingTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_node_inspection {
            label="1. Node-Level Output & Delta Inspection";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            node_exec [label="Node Execution: research_node\\nReturns: {'findings': ['Fact 1']}", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            delta_eval [label="Delta Verifier\\nValidates Schema & Reducer Types", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_routing_inspection {
            label="2. Conditional Edge & Routing Diagnostics";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            router_func [label="Router Function: evaluate_readiness(state)\\nEvaluates: is_ready == True?", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            route_audit [label="Routing Target Tracer\\nSelected Target: 'writer_node'", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_superstep_barrier {
            label="3. Superstep Barrier & Reducer Commit";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            pregel_barrier [label="Pregel Barrier Synchronization\\nAtomic Commit of All Parallel Updates", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            checkpoint_write [label="Checkpoint Written to Store\\n(thread_id, checkpoint_id)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_loop_guard {
            label="4. Loop Guardrail & Recursion Limits";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            recursion_checker [label="Recursion Limit Monitor\\nStep 14/25: Warn if Approaching Limit", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
            stream_updates [label="Live Stream Consumer: stream_mode='updates'\\nPrints Live Deltas to Terminal / Console", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        node_exec -> delta_eval;
        delta_eval -> router_func;
        router_func -> route_audit;
        route_audit -> pregel_barrier;
        pregel_barrier -> checkpoint_write;
        checkpoint_write -> recursion_checker;
        recursion_checker -> stream_updates;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: PRINTING STATE
# ==============================================================================
st.header("2. Printing State: Strategic Logging, State Dumps, and Reducer Auditing", divider="rainbow")

st.markdown("""
The simplest and most direct debugging technique is inspecting state snapshots across supersteps.
Rather than sprinkling arbitrary `print()` statements inside nodes, the standard pattern in LangGraph is streaming with `stream_mode="updates"`:
""")

st.code('''# Clean superstep update printer
for step_idx, event in enumerate(app.stream(inputs, stream_mode="updates")):
    for node_name, delta in event.items():
        print(f"[{step_idx}] Node Completed: {node_name}")
        print(f"    Modified Keys: {list(delta.keys())}")
        print(f"    Delta: {delta}\\n")''', language="python")

# ==============================================================================
# SECTION 3: INSPECTING NODE OUTPUTS
# ==============================================================================
st.header("3. Inspecting Node Outputs: Channel Deltas, Return Dictionaries, and Schema Validation", divider="rainbow")

st.markdown("""
A common bug in LangGraph occurs when a node returns a dictionary containing keys that are **not defined in the State TypedDict**:
- If a node returns `{"unrecognized_key": 42}`, LangGraph may discard it or raise a validation error.
- Always inspect node outputs by writing isolated unit tests that invoke the node function as an ordinary Python callable:
  `output = my_node({"messages": [...]})`
""")

# ==============================================================================
# SECTION 4: INSPECTING GRAPH EXECUTION
# ==============================================================================
st.header("4. Inspecting Graph Execution: Superstep Barriers, Task Scheduling, and Step Dictionaries", divider="rainbow")

st.markdown("""
To observe low-level Pregel engine operations, use `stream_mode="debug"`.
This surfaces internal scheduling tasks, task input arguments, and checkpoint write operations:
""")

st.code('''for debug_event in app.stream(inputs, stream_mode="debug"):
    print("Event Type:", debug_event["type"])
    print("Payload:", debug_event)''', language="python")

# ==============================================================================
# SECTION 5: DEBUGGING ROUTING
# ==============================================================================
st.header("5. Debugging Routing: Tracing Conditional Edge Outputs and Decision Branches", divider="rainbow")

st.markdown("""
Conditional edges determine the next node by returning a string key matching the target node mapping.
Common routing bugs include:
- The routing function returns an uppercase string `"RESEARCH"` while the node is named `"research"`.
- The routing function returns `None` or an unmapped string, causing `InvalidUpdateError`.
""")

st.code('''# Defensively instrumenting conditional edge functions
def safe_router(state: State) -> str:
    target = determine_next_agent(state)
    valid_targets = ["researcher", "coder", END]
    
    if target not in valid_targets:
        raise ValueError(f"Router error: target '{target}' not in {valid_targets}!")
        
    print(f"[ROUTER TRACE] Routing execution to: '{target}'")
    return target''', language="python")

# ==============================================================================
# SECTION 6: DEBUGGING STATE UPDATES
# ==============================================================================
st.header("6. Debugging State Updates: Resolving Reducer Conflicts and Invalid Channel Writes", divider="rainbow")

st.markdown("""
When two parallel nodes write to the same channel without an `Annotated` reducer, LangGraph raises `InvalidUpdateError`.
- **Debugging Tip**: Check whether the state field is declared with `Annotated[list, operator.add]` or `Annotated[list, add_messages]`.
- Channels without reducers use **Last-Write-Wins** or disallow concurrent updates.
""")

# ==============================================================================
# SECTION 7: DEBUGGING LOOPS
# ==============================================================================
st.header("7. Debugging Loops: Detecting Runaway Cycles, Recursion Limits, and Convergence Stalls", divider="rainbow")

st.markdown("""
When an agent graph contains cycles (e.g. ReAct loops or reflection), a logic error can cause the graph to cycle endlessly.
LangGraph protects against infinite loops via **`recursion_limit`**:
- The default limit is **25 supersteps**.
- If exceeded, LangGraph raises **`GraphRecursionError`**.
- To debug: pass a lower limit (`config={"recursion_limit": 5}`) to catch loops quickly during testing!
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Instrumentable Graph with Debug Callbacks and Hooks", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a fully instrumented graph with custom logging hooks and defensive routing validation.
""")

st.code('''from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DebuggableState(TypedDict):
    input_text: str
    step_history: List[str]
    next_node: str

def ingress_node(state: DebuggableState) -> dict:
    print("[DEBUG HOOK] Ingress Node Executing...")
    return {
        "step_history": state.get("step_history", []) + ["ingress_completed"],
        "next_node": "process"
    }

def process_node(state: DebuggableState) -> dict:
    print("[DEBUG HOOK] Process Node Executing...")
    return {
        "step_history": state.get("step_history", []) + ["process_completed"],
        "next_node": "FINISH"
    }

def defensive_router(state: DebuggableState) -> str:
    target = state.get("next_node")
    print(f"[DEBUG HOOK] Evaluating router... target='{target}'")
    if target == "FINISH":
        return END
    return target

builder = StateGraph(DebuggableState)
builder.add_node("ingress", ingress_node)
builder.add_node("process", process_node)

builder.add_edge(START, "ingress")
builder.add_conditional_edges("ingress", defensive_router, {"process": "process", END: END})
builder.add_conditional_edges("process", defensive_router, {END: END})

app = builder.compile(checkpointer=MemorySaver())

# Run with custom recursion limit for safe debugging
config = {"configurable": {"thread_id": "debug_th_01"}, "recursion_limit": 10}
res = app.invoke({"input_text": "Audit sample"}, config=config)
print("Execution Completed Cleanly! Steps:", res["step_history"])''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Graph Debugger & State Transition Tracer", divider="rainbow")

st.markdown("""
Simulate how the LangGraph debugging suite catches state updates, conditional edge routes, and infinite loop bugs.
""")

sim = DebuggingSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Debugger Simulation Settings")
    test_query = st.text_input("User Prompt", value="Calculate Q3 cloud expenditure")
    inject_loop = st.toggle("Simulate Infinite Loop Bug (Agent Never Terminates)", value=False)

    if st.button("🚀 Run Diagnostic Trace"):
        diag_res = sim.run_debug_trace(test_query, inject_loop)
        st.session_state["diag_sim_res"] = diag_res

with col_sim2:
    st.subheader("📋 Diagnostic Trace Output")
    if "diag_sim_res" in st.session_state:
        res = st.session_state["diag_sim_res"]
        
        if res["has_bug"]:
            st.error(f"⚠️ **DIAGNOSIS**: {res['diagnosis']}")
        else:
            st.success(f" **DIAGNOSIS**: {res['diagnosis']}")

        st.markdown("#### Superstep Transition Table:")
        st.dataframe(pd.DataFrame(res["trace"]), use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is a GraphRecursionError and how do you resolve it?", expanded=False):
    st.markdown("""
    **Answer**:
    `GraphRecursionError` occurs when execution supersteps exceed `recursion_limit` (default 25).
    To resolve: inspect the graph's cyclic conditional edges for missing exit conditions, or increase `config={'recursion_limit': 100}` if the workflow legitimately requires more turns.
    """)

with st.expander("Q2: Why is stream_mode='updates' preferred for debugging over print() statements inside nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    `stream_mode='updates'` emits a structured dictionary mapping node names to their exact returned deltas after every superstep barrier, providing centralized, standardized observability without cluttering node source code.
    """)

with st.expander("Q3: How do you catch InvalidUpdateError exceptions?", expanded=False):
    st.markdown("""
    **Answer**:
    `InvalidUpdateError` occurs when parallel nodes return updates to the same channel without an `Annotated` reducer.
    Resolve it by adding a reducer (e.g. `Annotated[list, operator.add]` or `add_messages`) to the channel definition.
    """)

with st.expander("Q4: Can you unit-test a LangGraph node without compiling the StateGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Nodes are standard Python functions. You can invoke a node directly:
    `output_dict = my_node({"messages": [HumanMessage(content="Test")]})`
    and assert expected return values in pytest.
    """)

with st.expander("Q5: What diagnostic information does stream_mode='debug' provide?", expanded=False):
    st.markdown("""
    **Answer**:
    `debug` mode surfaces internal Pregel engine actions: scheduled tasks, input payloads, checkpoint commits, and trigger events across all nodes.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Recursion on Failing Conditional Edges")
    st.markdown("""
    *Mechanism*: A conditional edge routing back to the previous node without an exit condition.  
    *Consequence*: Raises `GraphRecursionError` and crashes the application thread.  
    *Fix*: Always enforce a `loop_count <= max_loops` check in router functions.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Case-Sensitive Node Name Mismatch in Routers")
    st.markdown("""
    *Mechanism*: Router returns `"Writer"` while the node is registered as `"writer"`.  
    *Consequence*: Raises `KeyError` or routing fault immediately.  
    *Fix*: Use Python Enums or `Literal` types for node names.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Setting Recursion Limit to 1000 in Production")
    st.markdown("""
    *Mechanism*: Setting an enormous recursion limit to bypass `GraphRecursionError`.  
    *Consequence*: Masked infinite loops burn thousands of dollars in LLM API calls before terminating.  
    *Fix*: Keep recursion limits tight (25-50) and fix the underlying routing logic.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Noise Warning): Leaving stream_mode='debug' Active in Production")
    st.markdown("""
    *Mechanism*: Streaming debug payloads over production API gateways.  
    *Consequence*: Floods network bandwidth and log aggregation systems with internal Pregel traces.  
    *Fix*: Use `stream_mode='updates'` or `stream_mode='messages'` in production.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**stream_mode='updates'**\\nEmits discrete node deltas per superstep for clean debugging.")
with col_fc2:
    st.info("**recursion_limit**\\nThe maximum allowed superstep barrier count before GraphRecursionError.")
with col_fc3:
    st.info("**InvalidUpdateError**\\nCollision error caused by parallel nodes updating a non-reduced channel.")
with col_fc4:
    st.info("**stream_mode='debug'**\\nLow-level Pregel engine action stream for deep diagnostic inspection.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to debug state transitions using stream_mode='updates'.", value=True)
c2 = st.checkbox("I understand what causes GraphRecursionError and how to configure recursion_limit.", value=True)
c3 = st.checkbox("I can unit-test node functions independently as pure Python callables.", value=True)
c4 = st.checkbox("I know how to debug conditional edge routing functions defensively.", value=True)
c5 = st.checkbox("I understand what causes InvalidUpdateError and how channel reducers resolve it.", value=True)
c6 = st.checkbox("I know what information is exposed by stream_mode='debug'.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 24.1 'Debugging LangGraph' with full architectural and practical comprehension!")
