import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="7.1 Understanding Command - LangGraph Curriculum",
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
# MODULE-SCOPE COMMAND DISPATCHER FOR WORKBENCH & PIPELINE
# ==============================================================================
class CommandExecutionEngine:
    """
    Simulates LangGraph's Command primitive:
    Co-locating state updates with dynamic destination navigation (goto).
    """
    def __init__(self):
        self.node_registry = {
            "supervisor": "Analyzes task and returns Command(update=..., goto=...)",
            "coder_agent": "Generates code and returns Command(update=..., goto='tester')",
            "tester_agent": "Runs tests and returns Command(goto='END' or goto='coder')",
            "human_review": "Surfaces questions to user and resumes execution"
        }

    def dispatch_command(self, active_node: str, state_payload: Dict[str, Any], goto_target: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a simulated Command return.
        """
        merged_state = dict(state_payload)
        for k, v in updates.items():
            if k == "audit_log":
                merged_state["audit_log"] = state_payload.get("audit_log", []) + (v if isinstance(v, list) else [v])
            else:
                merged_state[k] = v

        return {
            "dispatched_from": active_node,
            "goto_destination": goto_target,
            "emitted_update": updates,
            "resulting_state": merged_state
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Unified State Mutation and Control Flow with the Command API", divider="rainbow")

st.markdown("""
In earlier versions of LangGraph, state updates and edge routing were strictly separated:
nodes returned dictionaries to mutate state, while external router functions in conditional edges decided where to go next.
Introduced in modern LangGraph (v0.2+), the **Command** primitive unifies these operations:
a node can simultaneously **update state** (`update={...}`) and **direct control flow** (`goto="..."`).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    command_blueprint_dot = """
    digraph CommandArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_legacy_pattern {
            label="Legacy Decoupled Pattern (Node + Separate Conditional Edge)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            L_Node [label="Node A\\nReturns: {'status': 'REVIEW'}", fillcolor="#334155"];
            L_Router [label="Router Predicate\\nInspects state['status']", shape=diamond, fillcolor="#475569"];
            L_Target [label="Node B", fillcolor="#1E293B"];
            L_Node -> L_Router [label="State Delta"];
            L_Router -> L_Target [label="add_conditional_edges()"];
        }

        subgraph cluster_command_pattern {
            label="Modern Command Primitive (Unified Update + Goto)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            C_Node [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#0284C7"><b>Node Function returning Command</b></td></tr>
                <tr><td>update</td><td>{'status': 'REVIEW', 'score': 0.92}</td></tr>
                <tr><td>goto</td><td>'node_b' (Direct Target Transition)</td></tr>
            </table>>, shape=plaintext];
            
            C_Engine [label="LangGraph Command Dispatcher\\n1. Merges 'update' through Reducers\\n2. Bypasses router; routes directly to 'goto'", fillcolor="#4338CA"];
            C_Target [label="Node B\\nExecutes next superstep", fillcolor="#059669"];
            
            C_Node -> C_Engine [label="return Command(...)"];
            C_Engine -> C_Target [label="Zero-Hop Dynamic Dispatch"];
        }

        L_Target -> C_Node [style=invis];
    }
    """
    st.graphviz_chart(command_blueprint_dot, use_container_width=True)

st.markdown("""
### Paradigm Shift: Dynamic Agent Choreography
With `Command`, nodes become autonomous actors. A supervisor agent does not need an external router sitting between it
and worker nodes; the supervisor directly returns `Command(goto="researcher", update={"task": "Find market cap"})`.
""")

# ==============================================================================
# SECTION 2: WHAT IS COMMAND?
# ==============================================================================
st.header("2. What is Command? The Unified Orchestration Primitive", divider="rainbow")

st.markdown("""
`Command` (`from langgraph.types import Command`) is a specialized Python data structure designed to represent
an atomic instruction directed at the LangGraph runtime engine.
""")

col_wc1, col_wc2 = st.columns(2)
with col_wc1:
    st.markdown("""
    #### Core Attributes of Command:
    - **`update`**: A dictionary representing the state delta to apply to the current superstep.
    - **`goto`**: A target node identifier (e.g. `"worker_node"`, `END`), or a list of node identifiers for parallel fan-out.
    - **`graph`**: (Optional) Destination graph scope for navigating between parents and subgraphs.
    - **`resume`**: (Optional) Payload passed to resume execution after an `interrupt()`.
    """)
with col_wc2:
    st.code("""from langgraph.types import Command

def my_node(state: MyState) -> Command:
    return Command(
        update={"counter": state["counter"] + 1},
        goto="next_step"
    )
""", language="python")

# ==============================================================================
# SECTION 3: WHY USE COMMAND?
# ==============================================================================
st.header("3. Why Use Command? Eradicating Redundant Routers and Decoupling Topology", divider="rainbow")

st.markdown("""
In complex multi-agent architectures, using traditional conditional edges results in boilerplate explosion.
Every multi-way decision required defining:
1. A node function to compute the decision.
2. An update dictionary to write the decision into state.
3. A separate router function to read the decision from state.
4. An `add_conditional_edges` call with a `path_map`.
""")

col_cmp1, col_cmp2 = st.columns(2)
with col_cmp1:
    st.error("Traditional Conditional Edge (High Boilerplate)")
    st.code("""# 1. Node writes to state
def decide_node(state):
    return {"choice": "specialist"}

# 2. Separate router reads state
def router(state):
    return state["choice"]

# 3. Graph connects both
builder.add_conditional_edges("decide", router)
""", language="python")

with col_cmp2:
    st.success("Modern Command API (Clean & Co-Located)")
    st.code("""from langgraph.types import Command

# One atomic function does both!
def decide_node(state):
    return Command(
        update={"reason": "requires code"},
        goto="specialist"
    )
# Zero add_conditional_edges needed!
""", language="python")

# ==============================================================================
# SECTION 4: COMMAND(UPDATE=...)
# ==============================================================================
st.header("4. Command(update=...): Atomic State Delta Specification", divider="rainbow")

st.markdown("""
The `update` parameter in `Command(update=...)` behaves identically to returning a standard dictionary from a node.
LangGraph passes the dictionary to its channel reducers at the superstep boundary.
""")

st.markdown("""
#### Invariants of `Command(update=...)`:
- **Optional**: You can emit `Command(goto="next")` without `update` if no state changes occurred.
- **Reducer Compliant**: Channels annotated with `operator.add` append lists cleanly.
- **Type Checked**: Validated against the graph's `StateGraph` schema.
""")

# ==============================================================================
# SECTION 5: COMMAND(GOTO=...)
# ==============================================================================
st.header("5. Command(goto=...): Dynamic Edge Traversal and Destination Targeting", divider="rainbow")

st.markdown("""
The `goto` argument specifies where control flow transitions immediately after the superstep updates are committed.
""")

col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("""
    #### Valid Values for `goto`:
    - **Single Node Name**: `goto="analyzer"`
    - **Terminal Marker**: `goto=END`
    - **Parallel Fan-Out**: `goto=["worker_a", "worker_b"]` (Dispatches both concurrently!)
    - **Self-Loop (Cycle)**: `goto="current_node"`
    """)
with col_g2:
    st.code("""from langgraph.graph import END
from langgraph.types import Command

# 1. Route to single node
Command(goto="validator")

# 2. Parallel fan-out
Command(goto=["search_web", "query_sql"])

# 3. Direct termination
Command(goto=END)
""", language="python")

# ==============================================================================
# SECTION 6: COMBINING STATE UPDATES AND ROUTING
# ==============================================================================
st.header("6. Combining State Updates and Routing: Co-Located Logic and Control", divider="rainbow")

st.markdown("""
By combining `update` and `goto`, node functions achieve transactional clarity:
the state change and the destination path are expressed simultaneously in a single line of code.
""")

code_combined = """def supervisor_agent(state: TeamState) -> Command:
    last_message = state["messages"][-1].content
    
    if "python" in last_message:
        return Command(
            update={"active_agent": "python_coder", "logs": ["Assigned to Python coder"]},
            goto="python_coder"
        )
    elif "sql" in last_message:
        return Command(
            update={"active_agent": "db_analyst", "logs": ["Assigned to Database analyst"]},
            goto="db_analyst"
        )
    else:
        return Command(
            update={"active_agent": "general_faq", "logs": ["Assigned to FAQ agent"]},
            goto="general_faq"
        )
"""
st.code(code_combined, language="python")

# ==============================================================================
# SECTION 7: RETURNING COMMAND FROM NODES
# ==============================================================================
st.header("7. Returning Command from Nodes: Signatures, Type Annotations, and Compiler Expectations", divider="rainbow")

st.markdown("""
To ensure full static typing and IDE validation, node functions that return `Command` should annotate their return type:
""")

code_typing_cmd = """from typing import Literal
from langgraph.types import Command

# Annotate possible destinations using Literal
def typed_router_node(state: MyState) -> Command[Literal["billing", "tech", "__end__"]]:
    if state["is_billing"]:
        return Command(goto="billing")
    return Command(goto="tech")
"""
st.code(code_typing_cmd, language="python")

st.markdown("""
#### Graph Compilation Behavior with Command:
When a node returns `Command(goto="target")`, you do **not** need to call `builder.add_edge("source", "target")`.
LangGraph registers possible destinations automatically during compile validation.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Turn Workflow Powered by Command API", divider="rainbow")

st.markdown("""
Below is a complete, runnable script showing a Multi-Agent Supervisor built entirely with the `Command` API,
featuring dynamic routing, parallel task dispatch, and clean termination.
""")

prod_command_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

# Step 1: Define Multi-Agent State
class TeamState(TypedDict):
    task_description: str
    code_artifact: Optional[str]
    audit_trail: Annotated[List[str], operator.add]
    final_status: str

# Node 1: Supervisor Node returning Command
def supervisor_node(state: TeamState) -> Command:
    task = state["task_description"].lower()
    
    if "code" in task or "script" in task:
        # Direct routing to developer
        return Command(
            update={"audit_trail": ["Supervisor: Dispatched task to Developer."]},
            goto="developer"
        )
    else:
        # Direct termination with update
        return Command(
            update={
                "audit_trail": ["Supervisor: Non-coding task. Resolved immediately."],
                "final_status": "RESOLVED_GENERAL"
            },
            goto=END
        )

# Node 2: Developer Node returning Command
def developer_node(state: TeamState) -> Command:
    generated_code = "def calculate_orbit(): return '384,400 km'"
    return Command(
        update={
            "code_artifact": generated_code,
            "audit_trail": ["Developer: Code implementation complete."]
        },
        goto="qa_tester"
    )

# Node 3: QA Tester Node returning Command
def qa_tester_node(state: TeamState) -> Command:
    artifact = state.get("code_artifact", "")
    is_valid = "calculate_orbit" in artifact
    
    return Command(
        update={
            "final_status": "VERIFIED_SUCCESS" if is_valid else "FAILED_QA",
            "audit_trail": [f"QA: Test verification passed={is_valid}."]
        },
        goto=END
    )

# Step 2: Build Graph with Zero Conditional Edge Boilerplate
def build_command_graph():
    builder = StateGraph(TeamState)
    
    # Register Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("developer", developer_node)
    builder.add_node("qa_tester", qa_tester_node)
    
    # Connect START to supervisor (all other transitions governed by Command!)
    builder.add_edge(START, "supervisor")
    
    return builder.compile()

if __name__ == "__main__":
    app = build_command_graph()
    sample_job = {
        "task_description": "Write a python script to calculate moon orbit distance",
        "code_artifact": None,
        "audit_trail": ["Graph run started."],
        "final_status": "PENDING"
    }
    result = app.invoke(sample_job)
    print("Final Status:", result["final_status"])
    print("Generated Artifact:", result["code_artifact"])
    print("Audit Trail:", result["audit_trail"])
'''
st.code(prod_command_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Command Object Inspector and Execution Simulator", divider="rainbow")

st.markdown("""
Test how the `Command` primitive dispatches updates and dynamic transitions in real-time.
Configure updates, pick a `goto` destination, and inspect the resulting state mutation.
""")

cmd_engine = CommandExecutionEngine()

if "cmd_state" not in st.session_state:
    st.session_state.cmd_state = {
        "task": "Build scalable telemetry microservice",
        "assigned_agent": "unassigned",
        "status": "QUEUED",
        "audit_log": ["State initialized."]
    }

col_cmw_l, col_cmw_r = st.columns([1, 1.2])

with col_cmw_l:
    st.subheader("Configure Command Emission")
    origin_node = st.selectbox("Dispatching Node", list(cmd_engine.node_registry.keys()))
    target_goto = st.selectbox("Target `goto` Destination", ["coder_agent", "tester_agent", "human_review", "END"])
    
    new_agent_name = st.text_input("State update: `assigned_agent`", value="specialist_alpha")
    new_status = st.selectbox("State update: `status`", ["IN_PROGRESS", "REVIEW_REQUIRED", "COMPLETED"])
    log_msg = st.text_input("State update: `audit_log` entry", value="Dispatched via Command(goto=...)")

    if st.button("⚡ Dispatch Command Primitive", use_container_width=True):
        updates = {
            "assigned_agent": new_agent_name,
            "status": new_status,
            "audit_log": [f"[{origin_node} -> {target_goto}]: {log_msg}"]
        }
        dispatch_record = cmd_engine.dispatch_command(
            origin_node,
            st.session_state.cmd_state,
            target_goto,
            updates
        )
        st.session_state.cmd_state = dispatch_record["resulting_state"]
        st.session_state.last_cmd_record = dispatch_record
        st.success(f"Dispatched Command to `{target_goto}`!")

with col_cmw_r:
    st.subheader("Command Packet & Resulting State")
    if "last_cmd_record" in st.session_state:
        st.markdown("**Simulated Command Packet:**")
        st.code(f"""Command(
    update={st.session_state.last_cmd_record['emitted_update']},
    goto='{st.session_state.last_cmd_record['goto_destination']}'
)""", language="python")
    else:
        st.info("Configure and dispatch a Command to view packet inspection.")

    st.markdown("**Consolidated Active State:**")
    st.json(st.session_state.cmd_state)

st.markdown("### Command Navigation Registry")
df_nodes = pd.DataFrame([
    {"Node Name": k, "Role": v} for k, v in cmd_engine.node_registry.items()
])
st.dataframe(df_nodes, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What are the primary advantages of the Command API over add_conditional_edges?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Eliminates Intermediate State Pollution**: You don't need to write dummy routing keys (`"route_choice": "agent_b"`) into state just for an edge to read it.
    2. **Reduces Boilerplate**: Eliminates separate router functions and `add_conditional_edges` mappings.
    3. **Dynamic Topology**: Allows nodes to decide their destination dynamically based on localized private variables without exposing them to global state.
    """)

with st.expander("Q2: Can a node return Command without specifying the update parameter?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. `Command(goto="next_node")` is completely valid. It directs control flow to `next_node` without mutating any state channels.
    Similarly, `Command(update={"key": "val"})` without `goto` mutates state while following standard graph edges.
    """)

with st.expander("Q3: Can Command(goto=...) dispatch multiple parallel nodes simultaneously?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. Passing a list of strings: `Command(goto=["worker_alpha", "worker_beta"])` triggers a parallel fan-out.
    LangGraph executes both nodes concurrently in the next superstep in accordance with BSP semantics.
    """)

with st.expander("Q4: How does LangGraph validate that a Command(goto=...) destination actually exists?", expanded=False):
    st.markdown("""
    **Answer**:
    If a node returns `Command(goto="nonexistent_node")` at runtime, LangGraph raises a runtime `ValueError: Node 'nonexistent_node' not found in graph`.
    To catch this during development, use type hints: `Command[Literal["node_a", "node_b", "__end__"]]` to leverage static linter validation.
    """)

with st.expander("Q5: What happens if a node has a normal outgoing edge defined in the graph but returns Command(goto='different_node')?", expanded=False):
    st.markdown("""
    **Answer**:
    `Command(goto=...)` takes **absolute precedence** over static edges!
    If `builder.add_edge("node_a", "node_b")` exists, but `node_a` returns `Command(goto="node_c")`, LangGraph will route execution to `node_c`, overriding the static edge.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Typo in goto String Causing Runtime Crash")
    st.markdown("""
    *Mechanism*: Returning `Command(goto='develper')` (misspelled `developer`).  
    *Consequence*: Crash with `ValueError: Node 'develper' not found`.  
    *Fix*: Use Python string constants or `Literal` type annotations to guarantee exact spelling.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Circular Command Transitions Without Termination Guards")
    st.markdown("""
    *Mechanism*: Node A returns `Command(goto="node_b")` and Node B returns `Command(goto="node_a")` without a counter guard.  
    *Consequence*: `GraphRecursionError` after 25 supersteps.  
    *Fix*: Track iteration counters in `update={"step": state["step"] + 1}` and route to `END` when threshold is exceeded.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-using Command and Obscuring Graph Visualizations")
    st.markdown("""
    *Mechanism*: Building a 50-node graph where every transition is handled by `Command(goto=...)` with zero registered edges.  
    *Consequence*: Static graph visualization diagrams (Graphviz, LangGraph Studio) appear as disconnected nodes because transitions are invisible to static analysis.  
    *Fix*: Use type annotations `Command[Literal[...]]` so LangGraph can infer edges for visualization.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Returning None Instead of Command or Dict")
    st.markdown("""
    *Mechanism*: A node contains an `if / else` block where the `else` branch returns nothing (`None`).  
    *Consequence*: Runtime `ValueError: Node callable must return dict or Command`.  
    *Fix*: Always ensure all execution paths return a valid `Command` or `{}`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Command API**\\nA modern LangGraph primitive unifying state mutations and dynamic graph routing in one object.")
with col_fc2:
    st.info("**Command(update=...)**\\nThe state delta dictionary applied to the graph channels at superstep synchronization.")
with col_fc3:
    st.info("**Command(goto=...)**\\nThe destination node identifier (or list of nodes) to execute in the next superstep.")
with col_fc4:
    st.info("**goto Precedence**\\nA runtime rule where `Command(goto=...)` overrides any static edges declared in the graph.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural advantages of Command over decoupled conditional edges.", value=True)
c2 = st.checkbox("I know how to co-locate state updates and routing using Command(update=..., goto=...).", value=True)
c3 = st.checkbox("I understand how to dispatch parallel fan-out nodes using a list of goto destinations.", value=True)
c4 = st.checkbox("I know that Command(goto=...) takes precedence over static graph edges.", value=True)
c5 = st.checkbox("I can use typing.Literal with Command to enforce compile-time destination validation.", value=True)
c6 = st.checkbox("I understand how Command eliminates redundant intermediate state variables.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 7.1 'Understanding Command' with full architectural and practical comprehension!")
