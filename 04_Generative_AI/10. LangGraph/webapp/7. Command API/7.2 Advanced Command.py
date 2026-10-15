import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="7.2 Advanced Command - LangGraph Curriculum",
    page_icon="🧭",
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
# MODULE-SCOPE HIERARCHICAL RESUME ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class AdvancedCommandEngine:
    """
    Simulates advanced Command capabilities:
    Human interrupt resumption (Command(resume=...)),
    and hierarchical subgraph-to-parent navigation (Command(graph=...)).
    """
    def __init__(self):
        self.scopes = ["CURRENT_GRAPH", "PARENT_GRAPH"]
        self.active_subgraphs = {
            "research_subgraph": {"parent": "root_agent", "nodes": ["fetch_web", "summarize"]},
            "coding_subgraph": {"parent": "root_agent", "nodes": ["generate_ast", "run_linter"]}
        }

    def simulate_resume_step(self, interrupted_action: str, human_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates app.invoke(Command(resume=human_payload)) resumption.
        """
        is_approved = human_payload.get("approved", False)
        reviewer = human_payload.get("reviewer_id", "admin_01")
        
        return {
            "action_evaluated": interrupted_action,
            "resumed_with": human_payload,
            "approval_status": "APPROVED" if is_approved else "REJECTED",
            "next_target_node": "execute_transaction" if is_approved else "handle_rejection",
            "audit_entry": f"Resumed via Command(resume=...) by {reviewer}. Decision: {is_approved}"
        }

    def simulate_parent_navigation(self, current_subgraph: str, target_parent_node: str, return_delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates Command(goto=target, graph=Command.PARENT)
        """
        return {
            "origin_subgraph": current_subgraph,
            "target_node_in_parent": target_parent_node,
            "escaped_to": "root_agent",
            "delta_merged_to_parent": return_delta,
            "status": "SUBGRAPH_EXIT_SUCCESS"
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Hierarchical Multi-Graph Routing and Resumption with Advanced Command", divider="rainbow")

st.markdown("""
Beyond basic node-to-node routing, the **Command API** provides advanced primitives for:
1. **Human-in-the-Loop Resumption**: `Command(resume=...)` provides inputs to nodes paused by `interrupt()`.
2. **Hierarchical Navigation**: `Command(graph=Command.PARENT, goto="...")` allows a nested child subgraph
   to navigate back into its parent graph or jump across sibling subgraphs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    advanced_cmd_dot = """
    digraph AdvancedCommandBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_parent_graph {
            label="1. Parent Orchestration Graph (Root)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Parent_Supervisor [label="Parent Supervisor\\nRoutes task into Subgraph", fillcolor="#334155"];
            Parent_Aggregator [label="Parent Aggregator\\nTarget of Subgraph Command(graph=PARENT)", fillcolor="#059669"];
        }

        subgraph cluster_child_subgraph {
            label="2. Child Specialized Subgraph (Nested)";
            style="filled";
            color="#0F172A";
            fontcolor="#A855F7";
            Sub_Worker1 [label="Child Worker Node 1\\nPerforms localized sub-task", fillcolor="#4338CA"];
            Sub_Interrupt [label="Child Approval Node\\nTriggers interrupt('Review draft?')", fillcolor="#D97706"];
            Sub_Egress [label="Child Exit Node\\nreturns Command(goto='Parent_Aggregator',\\n graph=Command.PARENT)", fillcolor="#0284C7"];
        }

        subgraph cluster_human_ingress {
            label="3. External Human Ingress";
            style=dashed;
            color="#EAB308";
            fontcolor="#FACC15";
            Human_Client [label="Human Reviewer (UI / API)\\napp.invoke(Command(resume={'approved': True}))", fillcolor="#1E293B", shape=ellipse];
        }

        Parent_Supervisor -> Sub_Worker1 [label="Delegate to Subgraph"];
        Sub_Worker1 -> Sub_Interrupt;
        Sub_Interrupt -> Human_Client [label="interrupt() pauses thread"];
        Human_Client -> Sub_Interrupt [label="Command(resume=...) unpauses"];
        Sub_Interrupt -> Sub_Egress [label="Proceed upon approval"];
        Sub_Egress -> Parent_Aggregator [label="Command(graph=Command.PARENT) Escapes Subgraph"];
    }
    """
    st.graphviz_chart(advanced_cmd_dot, use_container_width=True)

st.markdown("""
### Enterprise Significance: Modular Agent Nesting
Without `Command(graph=Command.PARENT)`, child subgraphs are black boxes: they can only execute from start to finish
and return their final state. Advanced Command allows a child to **interrupt for human input** and directly signal
parent coordinators when emergency escalation or early termination is required.
""")

# ==============================================================================
# SECTION 2: COMMAND(RESUME=...)
# ==============================================================================
st.header("2. Command(resume=...): Resuming Interrupted Threads and Human-in-the-Loop Payloads", divider="rainbow")

st.markdown("""
When a node invokes `interrupt(prompt_value)`, execution halts immediately. The thread state is saved to the checkpointer.
To resume execution, the external client invokes the graph passing `Command(resume=...)`:
""")

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.markdown("""
    #### 1. Inside the Node: Calling `interrupt()`
    ```python
    from langgraph.types import interrupt

    def approval_node(state: State) -> dict:
        # Pauses graph execution and returns payload to client
        human_input = interrupt({
            "prompt": "Approve transaction?",
            "amount": state["amount"]
        })
        # When resumed, interrupt() returns the resume payload!
        return {"approved": human_input["approved"]}
    ```
    """)
with col_res2:
    st.markdown("""
    #### 2. From the Client: Calling `Command(resume=...)`
    ```python
    from langgraph.types import Command

    # Client receives interrupted state, prompts user, and resumes:
    app.invoke(
        Command(resume={"approved": True, "notes": "LGTM"}),
        config={"configurable": {"thread_id": "tx_42"}}
    )
    ```
    """)

# ==============================================================================
# SECTION 3: COMMAND(GRAPH=...)
# ==============================================================================
st.header("3. Command(graph=...): Scope Targeting Across Subgraph Hierarchies", divider="rainbow")

st.markdown("""
By default, `Command(goto="node_name")` looks for `node_name` inside the **current graph**.
The `graph` parameter allows specifying a different graph scope:
""")

st.markdown("""
- **`Command(graph=Command.PARENT, goto="parent_node")`**: Targets a node in the immediate enclosing parent graph.
- **`Command(graph=Command.ROOT, goto="root_node")`**: Targets a node in the top-level root graph across arbitrarily deep nesting.
""")

code_scope_demo = """from langgraph.types import Command

def child_exit_node(state: SubgraphState) -> Command:
    # Navigates straight out of this child subgraph into the parent!
    return Command(
        update={"subgraph_findings": state["local_summary"]},
        goto="parent_synthesis_node",
        graph=Command.PARENT
    )
"""
st.code(code_scope_demo, language="python")

# ==============================================================================
# SECTION 4: PARENT GRAPH NAVIGATION
# ==============================================================================
st.header("4. Parent Graph Navigation: Ascending from Child Subgraphs to Root Graphs", divider="rainbow")

st.markdown("""
Parent graph navigation solves the problem of **early exit** and **exception escalation**.
If a specialized SQL validation subgraph discovers a database corruption error, it does not need to finish
its full internal cycle; it can immediately bail out and route to `error_containment` in the parent graph.
""")

# Contextual diagram for parent escalation
subgraph_escalation_dot = """
digraph EscalationFlow {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Sub_Check [label="Child Subgraph Node: Check Security", fillcolor="#1E293B"];
    Decision [label="SQL Injection Detected?", shape=diamond, fillcolor="#DC2626"];
    Sub_Continue [label="Child Subgraph: Execute Query", fillcolor="#334155"];
    Parent_Alert [label="Parent Graph: Quarantine & Alert Node", fillcolor="#991B1B"];

    Sub_Check -> Decision;
    Decision -> Sub_Continue [label="No: Safe"];
    Decision -> Parent_Alert [label="Yes: Command(graph=PARENT, goto='quarantine')"];
}
"""
with st.expander("🚨 View Subgraph Escalation to Parent Graph Flow", expanded=True):
    st.graphviz_chart(subgraph_escalation_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: SUBGRAPH ROUTING
# ==============================================================================
st.header("5. Subgraph Routing: Descending and Inter-Subgraph Delegations", divider="rainbow")

st.markdown("""
In hierarchical multi-agent teams, a supervisor node in the parent graph can delegate directly to specific nodes
within subgraphs, or subgraphs can delegate horizontally to peer subgraphs via parent mediation.
""")

st.markdown("""
#### Best Practice for Subgraph State Mapping:
When crossing graph boundaries using `Command`, ensure that the keys returned in `update={...}`
match the channels defined in the target graph's schema. If you update keys that exist only in the child schema,
the parent graph will reject the update.
""")

# ==============================================================================
# SECTION 6: COMMAND TYPING
# ==============================================================================
st.header("6. Command Typing: Type-Safe Literals and Compiler Navigation Validation", divider="rainbow")

st.markdown("""
LangGraph's `Command` class is a generic type `Command[Targets]`.
By parameterizing `Command` with a `typing.Literal` of valid destination node names,
you enable static type checkers (mypy, pyright) to catch invalid `goto` strings at build time.
""")

code_typed_command = """from typing import Literal
from langgraph.types import Command

# Enforce valid targets via Literal:
ValidTargets = Literal["retrieve_docs", "call_llm", "human_review", "__end__"]

def strict_router(state: MyState) -> Command[ValidTargets]:
    if state["needs_docs"]:
        return Command(goto="retrieve_docs") # OK!
    
    # Static linter will raise an error if you type:
    # return Command(goto="typo_node") -> Type error!
    return Command(goto="__end__")
"""
st.code(code_typed_command, language="python")

# ==============================================================================
# SECTION 7: COMMAND-BASED WORKFLOWS
# ==============================================================================
st.header("7. Command-Based Workflows: Autonomous Multi-Agent Choreography", divider="rainbow")

st.markdown("""
In a pure **Command-Based Architecture**, static edges are used only for initialization (`START -> supervisor`),
while all subsequent transitions are negotiated dynamically through `Command` returns.
This pattern matches the true autonomy of multi-agent systems: agents decide who to consult next based on the evolving context.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Hierarchical Multi-Agent Graph with Resume and Subgraph Navigation", divider="rainbow")

st.markdown("""
Below is a complete, production-grade script demonstrating hierarchical multi-agent orchestration,
human resumption using `Command(resume=...)`, and cross-subgraph parent navigation.
""")

prod_advanced_code = '''from typing import TypedDict, Optional, Dict, Any
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Step 1: Parent State & Child Subgraph State
class ParentState(TypedDict):
    ticket_id: str
    task_scope: str
    subgraph_output: Optional[str]
    is_authorized: bool
    final_resolution: str

class ChildWorkerState(TypedDict):
    task_scope: str
    intermediate_result: str

# Step 2: Child Subgraph Nodes
def child_worker_node(state: ChildWorkerState) -> Dict[str, Any]:
    return {"intermediate_result": f"Executed deep research on {state['task_scope']}"}

def child_escalation_node(state: ChildWorkerState) -> Command:
    # Escapes child subgraph directly to parent aggregator!
    return Command(
        update={"subgraph_output": state["intermediate_result"]},
        goto="parent_finalize",
        graph=Command.PARENT
    )

# Step 3: Build Child Subgraph
child_builder = StateGraph(ChildWorkerState)
child_builder.add_node("worker", child_worker_node)
child_builder.add_node("escalate", child_escalation_node)
child_builder.add_edge(START, "worker")
child_builder.add_edge("worker", "escalate")
compiled_child_graph = child_builder.compile()

# Step 4: Parent Nodes
def parent_triage_node(state: ParentState) -> Command:
    # Delegate to child subgraph
    return Command(goto="child_subgraph")

def parent_finalize_node(state: ParentState) -> Dict[str, Any]:
    output = state.get("subgraph_output", "N/A")
    return {
        "final_resolution": f"Parent completed resolution based on: {output}"
    }

# Step 5: Build Hierarchical Parent Graph
def build_hierarchical_system():
    parent_builder = StateGraph(ParentState)
    
    parent_builder.add_node("triage", parent_triage_node)
    parent_builder.add_node("child_subgraph", compiled_child_graph)
    parent_builder.add_node("parent_finalize", parent_finalize_node)
    
    parent_builder.add_edge(START, "triage")
    parent_builder.add_edge("parent_finalize", END)
    
    # Checkpointer is required for thread durability & interrupts
    memory = MemorySaver()
    return parent_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    app = build_hierarchical_system()
    init_payload = {
        "ticket_id": "TICKET_9012",
        "task_scope": "Multi-Tenant Security Architecture",
        "subgraph_output": None,
        "is_authorized": False,
        "final_resolution": "PENDING"
    }
    cfg = {"configurable": {"thread_id": "thread_sec_01"}}
    res = app.invoke(init_payload, config=cfg)
    print("Final Resolution:", res["final_resolution"])
    print("Output from Subgraph:", res["subgraph_output"])
'''
st.code(prod_advanced_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Advanced Command Scope and Human Resumption Simulator", divider="rainbow")

st.markdown("""
Test advanced Command mechanics: simulate resuming an interrupted approval thread with `Command(resume=...)`
or escaping a child subgraph to target parent nodes with `Command(graph=Command.PARENT)`.
""")

adv_engine = AdvancedCommandEngine()

tab1, tab2 = st.tabs(["1. Human-in-the-Loop Resumption", "2. Hierarchical Subgraph Escape"])

with tab1:
    st.subheader("Simulate `Command(resume=...)`")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        action_name = st.text_input("Interrupted Action Pending Review", value="Transfer $50,000 to Vendor Account")
        human_decision = st.checkbox("Human Approves Action?", value=True)
        reviewer_tag = st.text_input("Reviewer Identity", value="security_lead_sarah")

        if st.button("🚀 Invoke app.invoke(Command(resume=...))", use_container_width=True):
            resume_payload = {"approved": human_decision, "reviewer_id": reviewer_tag}
            st.session_state.resume_trace = adv_engine.simulate_resume_step(action_name, resume_payload)
            st.success("Thread resumed from checkpoint!")

    with col_r2:
        if "resume_trace" in st.session_state:
            st.markdown("**Resumption Execution Result:**")
            st.json(st.session_state.resume_trace)
        else:
            st.info("Configure human feedback and click Resume.")

with tab2:
    st.subheader("Simulate `Command(graph=Command.PARENT)`")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        chosen_sub = st.selectbox("Current Subgraph Scope", list(adv_engine.active_subgraphs.keys()))
        target_parent = st.text_input("Target Node in Parent Graph", value="parent_aggregator")
        delta_val = st.text_input("State Delta to Merge to Parent", value="Verified telemetry logs")

        if st.button("🧭 Escape Subgraph to Parent", use_container_width=True):
            st.session_state.subgraph_escape = adv_engine.simulate_parent_navigation(
                chosen_sub,
                target_parent,
                {"subgraph_result": delta_val}
            )
            st.success("Successfully navigated to parent graph!")

    with col_s2:
        if "subgraph_escape" in st.session_state:
            st.markdown("**Hierarchical Navigation Packet:**")
            st.json(st.session_state.subgraph_escape)
        else:
            st.info("Trigger navigation to view parent packet routing.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does Command(resume=...) work under the hood with LangGraph checkpointers?", expanded=False):
    st.markdown("""
    **Answer**:
    When `interrupt()` is called, LangGraph captures the active state, assigns a checkpoint ID, and pauses execution.
    When `app.invoke(Command(resume=val), config={"configurable": {"thread_id": "..."}})` is called:
    1. LangGraph loads the latest checkpoint matching `thread_id`.
    2. It restores the suspended node's execution context.
    3. The return value of `interrupt(...)` inside the node becomes `val`.
    4. The node resumes and runs to completion.
    """)

with st.expander("Q2: What is the difference between Command.PARENT and Command.ROOT?", expanded=False):
    st.markdown("""
    **Answer**:
    - `Command.PARENT`: Targets the immediate enclosing parent graph one level up in the hierarchy.
    - `Command.ROOT`: Targets the topmost parent graph in the multi-level hierarchy, regardless of how deeply nested the child subgraph is (e.g. root -> supervisor -> team -> worker).
    """)

with st.expander("Q3: Can Command(resume=...) update state channels at the same time as resuming an interrupt?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. The client can invoke:
    ```python
    app.invoke(
        Command(resume={"approved": True}, update={"admin_comment": "Approved by CTO"}),
        config={"configurable": {"thread_id": "thread_123"}}
    )
    ```
    LangGraph merges the `update` dictionary into the state channels while simultaneously providing the `resume` payload to the unpaused `interrupt()` call.
    """)

with st.expander("Q4: Why should you avoid updating child-specific state keys when escaping to a parent with Command(graph=PARENT)?", expanded=False):
    st.markdown("""
    **Answer**:
    When navigating across graph boundaries, the `update` dictionary is merged into the **target graph's state**.
    If you emit keys that are defined in the child schema but absent from the parent schema, the parent graph compiler will raise an `InvalidUpdateError` for un-mapped state channels.
    """)

with st.expander("Q5: How does Command typing enable compile-time verification in CI/CD pipelines?", expanded=False):
    st.markdown("""
    **Answer**:
    By using `Command[Literal["node_a", "node_b", "__end__"]]`, type checkers like `mypy` verify that every `goto` string matches an actual valid node identifier. If an engineer renames a node in the graph but forgets to update a string inside a node's `Command(goto=...)`, `mypy` flags the mismatch during CI before deployment.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Calling Command(resume=...) on a Thread That Was Never Interrupted")
    st.markdown("""
    *Mechanism*: Passing `Command(resume=...)` to a thread that is either completed or currently running.  
    *Consequence*: Crash with `ValueError: Cannot resume execution; no active interrupt found for thread`.  
    *Fix*: Check thread status or inspect state with `app.get_state(config)` before issuing resume commands.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Emitting Incompatible State Keys to Parent Graphs")
    st.markdown("""
    *Mechanism*: Using `Command(graph=Command.PARENT, update={'child_private_key': 42})` where the parent schema does not have `child_private_key`.  
    *Consequence*: Crash with `InvalidUpdateError` due to undeclared state channels.  
    *Fix*: Ensure keys in `update` exist in the target parent graph's state schema.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Missing thread_id in Config During Resumption")
    st.markdown("""
    *Mechanism*: Calling `app.invoke(Command(resume=...))` without passing `config={"configurable": {"thread_id": "..."}}`.  
    *Consequence*: Checkpointer cannot locate the paused state; LangGraph starts a brand new uninitialized thread.  
    *Fix*: Always supply the exact same `thread_id` that was used when the interrupt occurred.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Bypassing Subgraph Ingress/Egress Isolation")
    st.markdown("""
    *Mechanism*: Using `Command(graph=Command.PARENT)` haphazardly across dozens of subgraphs.  
    *Consequence*: Creates tightly coupled spaghetti code where subgraphs depend on the internal structure of their parents.  
    *Fix*: Use parent navigation sparingly, primarily for early termination and emergency exception handling.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Command(resume=...)**\\nThe external invocation payload that resumes a graph paused by an interrupt() call.")
with col_fc2:
    st.info("**Command.PARENT**\\nA graph scope constant targeting the immediate enclosing parent graph.")
with col_fc3:
    st.info("**Command.ROOT**\\nA graph scope constant targeting the topmost root graph in a nested multi-level hierarchy.")
with col_fc4:
    st.info("**interrupt()**\\nA pausing function that captures state and waits for Command(resume=...) from the client.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand how Command(resume=...) provides inputs to unpause interrupt() calls.", value=True)
c2 = st.checkbox("I know how to use Command(graph=Command.PARENT) to navigate from child subgraphs to parents.", value=True)
c3 = st.checkbox("I understand the difference between Command.PARENT and Command.ROOT scopes.", value=True)
c4 = st.checkbox("I know how to type Command with Literal to enforce compile-time destination checks.", value=True)
c5 = st.checkbox("I can ensure state update compatibility when crossing graph boundaries.", value=True)
c6 = st.checkbox("I understand how to inspect thread state before issuing resumption commands.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 7.2 'Advanced Command' with full architectural and practical comprehension!")
