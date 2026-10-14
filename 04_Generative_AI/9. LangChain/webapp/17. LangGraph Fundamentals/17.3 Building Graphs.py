import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockBuildingGraphsStudio:
    """
    Simulates the construction, compilation, invocation, and streaming of a StateGraph.
    Demonstrates both stream_mode="values" and stream_mode="updates".
    """
    def __init__(self):
        pass

    def execute_stream_simulation(
        self,
        query: str,
        stream_mode: str,
        enable_human_approval: bool
    ) -> Dict[str, Any]:
        events = []
        state = {
            "query": query,
            "analyst_notes": [],
            "risk_score": 0.0,
            "status": "pending",
            "approved": False
        }

        # Step 1: Intake Node
        delta_1 = {"analyst_notes": ["Intake: Query categorized as security infrastructure audit."]}
        state["analyst_notes"].extend(delta_1["analyst_notes"])
        events.append({
            "superstep": 1,
            "node": "intake_node",
            "delta": delta_1,
            "full_state": dict(state)
        })

        # Step 2: Risk Scoring Node
        q_lower = query.lower()
        risk = 0.85 if any(w in q_lower for w in ["drop", "delete", "purge", "credential", "bypass"]) else 0.25
        delta_2 = {"risk_score": risk, "status": "risk_evaluated"}
        state["risk_score"] = risk
        state["status"] = "risk_evaluated"
        events.append({
            "superstep": 2,
            "node": "risk_evaluator",
            "delta": delta_2,
            "full_state": dict(state)
        })

        # Step 3: Human Approval / Gate Node
        if risk > 0.50:
            if enable_human_approval:
                delta_3 = {"status": "human_approved", "approved": True, "analyst_notes": ["Gatekeeper: Human operator approved privileged action."]}
                state["status"] = "human_approved"
                state["approved"] = True
                state["analyst_notes"].extend(delta_3["analyst_notes"])
            else:
                delta_3 = {"status": "action_blocked", "approved": False, "analyst_notes": ["Gatekeeper: High risk action rejected."]}
                state["status"] = "action_blocked"
                state["approved"] = False
                state["analyst_notes"].extend(delta_3["analyst_notes"])

            events.append({
                "superstep": 3,
                "node": "approval_gate_node",
                "delta": delta_3,
                "full_state": dict(state)
            })

        # Step 4: Final Synthesis Node
        if state["status"] in ["risk_evaluated", "human_approved"]:
            ans = f"Action Completed: Audit performed for '{query}'. Operational risk is {state['risk_score']} (Within safe parameters)."
        else:
            ans = f"Action Terminated: Query '{query}' triggered high security risk and was blocked by policy gate."

        delta_4 = {"status": "completed", "final_answer": ans}
        state["status"] = "completed"
        state["final_answer"] = ans
        events.append({
            "superstep": 4,
            "node": "synthesis_node",
            "delta": delta_4,
            "full_state": dict(state)
        })

        return {
            "stream_mode": stream_mode,
            "events": events,
            "final_state": state
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="17.3 Building Graphs — LangChain Course",
    page_icon="🛠️",
    layout="wide"
)

st.title("17.3 Building Graphs")
st.caption("Practical Guide to StateGraph: Defining Schemas, Creating Nodes, Wiring Conditional Routing, Compiling, Invoking, and Streaming")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: Construction, Routing, Compilation & Streaming Pipeline", divider="rainbow")
st.markdown("""
Building robust agent state machines follows a strict developer lifecycle:
1. **Define the State Schema**: Declare channels and reducer functions.
2. **Implement Node Callables**: Write pure functions returning state updates.
3. **Assemble the Graph Topology**: Wire direct and conditional edges using `StateGraph`.
4. **Compile the Graph**: Attach checkpointers, thread storage, and interrupt breakpoints.
5. **Execute via Streaming**: Stream intermediate node values or deltas in real-time.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph BuildingGraphsArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_phase1 {
            label = "Phase 1: Schema & Graph Builder Definition";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            schema_box [label="Define State Schema\\nclass State(TypedDict): messages, metrics", fillcolor="#1E293B", color="#0284C7"];
            builder_box [label="builder = StateGraph(State)\\nInstantiates graph builder instance", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_phase2 {
            label = "Phase 2: Topology Wiring & Conditional Routing";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            add_nodes [label="builder.add_node('intake', fn)\\nbuilder.add_node('worker', fn)\\nbuilder.add_node('tools', tool_node)", fillcolor="#1E293B", color="#38BDF8"];
            wire_edges [label="builder.add_edge(START, 'intake')\\nbuilder.add_conditional_edges('worker', router)", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_phase3 {
            label = "Phase 3: Compilation & Persistence";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            compile_box [label="app = builder.compile(checkpointer=MemorySaver())\\nValidates cycles & returns CompiledStateGraph", fillcolor="#0F172A", color="#D97706"];
        }

        subgraph cluster_phase4 {
            label = "Phase 4: Runtime Execution & Streaming Modes";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            invoke_mode [label="app.invoke(input, config)\\nSynchronous execution to END", fillcolor="#064E3B", color="#10B981"];
            stream_mode [label="app.stream(input, stream_mode='values' | 'updates')\\nReal-time event streaming per superstep", fillcolor="#064E3B", color="#10B981"];
        }

        schema_box -> builder_box;
        builder_box -> add_nodes;
        add_nodes -> wire_edges;
        wire_edges -> compile_box [label="builder.compile()"];
        compile_box -> invoke_mode [label="invoke()"];
        compile_box -> stream_mode [label="stream()"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: STATEGRAPH ---
st.header("2. The StateGraph Class", divider="rainbow")
st.markdown("""
`StateGraph` is the primary class used to construct graphs in LangGraph. When instantiating `StateGraph`, you pass 
the state schema class as its argument. This parameterized type definition establishes the schema governing all channels.
""")

st.code("""
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class MyState(TypedDict):
    query: str
    counter: int

# Instantiation parameterized by MyState
builder = StateGraph(MyState)
""", language="python")

st.divider()

# --- SECTION 3: DEFINING STATE ---
st.header("3. Defining State & Channel Reducers", divider="rainbow")
st.markdown("""
State can be modeled with `TypedDict` or Pydantic `BaseModel`. For conversational applications, the `messages` key 
must use the `add_messages` reducer to prevent new messages from overwriting conversation history.
""")

st.code("""
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

class EnterpriseAgentState(TypedDict):
    # Appends new messages and updates existing ones by message ID
    messages: Annotated[List[AnyMessage], add_messages]
    # Appends log strings
    audit_trail: Annotated[List[str], operator.add]
    # Simple overwrite channel
    session_id: str
""", language="python")

st.divider()

# --- SECTION 4: CREATING NODES ---
st.header("4. Creating Nodes", divider="rainbow")
st.markdown("""
Nodes are Python callables (sync or async). Every node takes `state: State` (and optionally `config: RunnableConfig`) 
and returns a dictionary containing state keys to update.
""")

st.code("""
from langchain_core.runnables import RunnableConfig

def triage_node(state: EnterpriseAgentState, config: RunnableConfig):
    # Extract config parameters like user_id or thread_id
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    
    # Return update delta
    return {
        "audit_trail": [f"User {user_id} initiated triage at superstep."],
        "session_id": "sess_9941"
    }

# Register node on builder
builder.add_node("triage", triage_node)
""", language="python")

st.divider()

# --- SECTION 5: CONNECTING NODES ---
st.header("5. Connecting Nodes with Direct Edges", divider="rainbow")
st.markdown("""
Edges define the control flow. Direct edges represent deterministic, unconditional transitions.
""")

st.code("""
from langgraph.graph import START, END

# Direct transition from START to intake
builder.add_edge(START, "triage")

# Direct transition from triage to worker
builder.add_edge("triage", "process_worker")

# Direct transition from worker to END
builder.add_edge("process_worker", END)
""", language="python")

st.divider()

# --- SECTION 6: CONDITIONAL ROUTING ---
st.header("6. Conditional Routing & Path Maps", divider="rainbow")
st.markdown("""
Conditional edges introduce dynamic runtime decisions. A router function inspects the state and returns the key 
of the next node or a list of nodes for parallel execution.
""")

st.code("""
from typing import Literal

def check_routing_condition(state: EnterpriseAgentState) -> Literal["escalate_lead", "auto_resolve"]:
    if len(state["audit_trail"]) > 5:
        return "escalate_lead"
    return "auto_resolve"

builder.add_conditional_edges(
    "process_worker",
    check_routing_condition,
    {
        "escalate_lead": "human_escalation_node",
        "auto_resolve": END
    }
)
""", language="python")

st.divider()

# --- SECTION 7: COMPILING GRAPHS ---
st.header("7. Compiling Graphs (`compile()`)", divider="rainbow")
st.markdown("""
Calling `.compile()` transforms a `StateGraph` builder into an executable `CompiledStateGraph`.
""")

st.code("""
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# Compilation with persistence checkpointer and human-in-the-loop breakpoint
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_escalation_node"]  # Pauses execution for human review!
)
""", language="python")

st.divider()

# --- SECTION 8: INVOKING GRAPHS ---
st.header("8. Invoking Graphs (`invoke()` and `ainvoke()`)", divider="rainbow")
st.markdown("""
`CompiledStateGraph` implements the standard Runnable interface. When running with a checkpointer, you must pass 
a `thread_id` inside the `configurable` dictionary to isolate conversational state across sessions.
""")

st.code("""
# Thread configuration isolates state per conversation
config = {"configurable": {"thread_id": "user_session_42"}}

# Synchronous execution
final_state = app.invoke(
    {"messages": [HumanMessage(content="Hello!")]},
    config=config
)

print(final_state["messages"][-1].content)
""", language="python")

st.divider()

# --- SECTION 9: STREAMING GRAPHS ---
st.header("9. Streaming Graphs (`stream()` and Stream Modes)", divider="rainbow")
st.markdown("""
LangGraph provides powerful streaming capabilities with two primary modes:
1. **`stream_mode="values"`**: Yields the **complete state snapshot** after each superstep completes.
2. **`stream_mode="updates"`**: Yields only the **state delta dictionary** emitted by the specific node that ran in that superstep.
""")

col_st1, col_st2 = st.columns(2)
with col_st1:
    st.subheader("Values Streaming (`stream_mode='values'`)")
    st.code("""
# Streams full state after each superstep
for state_snapshot in app.stream(inputs, config, stream_mode="values"):
    print("Messages count:", len(state_snapshot["messages"]))
    """, language="python")

with col_st2:
    st.subheader("Updates Streaming (`stream_mode='updates'`)")
    st.code("""
# Streams only the node name and delta output
for update_event in app.stream(inputs, config, stream_mode="updates"):
    for node_name, delta in update_event.items():
        print(f"Node {node_name} emitted:", delta)
    """, language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: End-to-End Enterprise StateGraph with Streaming", divider="rainbow")
st.markdown("""
Below is a full production-ready StateGraph implementation featuring TypedDict state, tool-calling agent loop, 
persistent memory checkpointer, and streaming execution.
""")

st.code("""
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class ProductionAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 2. Tool Definition
@tool
def query_system_load() -> str:
    \"\"\"Returns real-time cluster health and CPU load.\"\"\"
    return "Cluster Health: OK. CPU load: 18%. Memory: 34%."

tools = [query_system_load]
tool_node = ToolNode(tools)

# 3. Model with Tools Bound
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def call_model(state: ProductionAgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: ProductionAgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# 4. Graph Construction
builder = StateGraph(ProductionAgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")  # Cycle back!

# 5. Compile with Checkpointer
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# 6. Stream Execution Utility
def run_streaming_agent(user_prompt: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=user_prompt)]}
    
    for update in app.stream(inputs, config, stream_mode="updates"):
        print("Superstep Update:", update)
""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Graph Builder & Streaming Studio", divider="rainbow")
st.markdown("""
Observe how a compiled `StateGraph` executes step-by-step. Compare **Values Streaming** vs **Updates Streaming**, 
and toggle **Human Approval Gates** to inspect how conditional routing dynamically pauses or completes tasks.
""")

# Instantiate module-scope simulator
build_studio = MockBuildingGraphsStudio()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    query_in = st.selectbox(
        "Select User Action Request:",
        [
            "Perform standard audit of cloud firewall rules.",
            "PURGE all customer credential tables and bypass MFA.",
            "Run automated database read-replica health diagnostics."
        ]
    )

    stream_mode_select = st.radio(
        "Stream Mode Strategy:",
        ["stream_mode='updates' (State Deltas)", "stream_mode='values' (Full State Snapshots)"],
        index=0
    )

    human_gate_toggle = st.toggle("Simulate Human Approval for High-Risk Actions", value=True)
    execute_stream_btn = st.button("⚡ Run Graph Streaming Engine", type="primary", use_container_width=True)

with wb_c2:
    st.subheader("Streaming Event Stream")

    if execute_stream_btn or query_in:
        chosen_mode = "updates" if "updates" in stream_mode_select else "values"
        sim_res = build_studio.execute_stream_simulation(
            query=query_in,
            stream_mode=chosen_mode,
            enable_human_approval=human_gate_toggle
        )

        st.markdown(f"### 🏁 Final State Status: `{sim_res['final_state']['status']}`")
        if sim_res['final_state'].get("approved", True):
            st.success(sim_res['final_state']["final_answer"])
        else:
            st.error(sim_res['final_state']["final_answer"])

        st.markdown(f"### 📡 Streamed Superstep Events ({chosen_mode.upper()} MODE)")
        for ev in sim_res["events"]:
            with st.expander(f"Superstep {ev['superstep']}: Node [{ev['node']}]", expanded=True):
                if chosen_mode == "updates":
                    st.write("**Delta Update Emitted**:")
                    st.json(ev["delta"])
                else:
                    st.write("**Full State Snapshot**:")
                    st.json(ev["full_state"])

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the difference between `stream_mode='values'` and `stream_mode='updates'` in LangGraph?",
     "`stream_mode='values'` yields the full state dictionary after every superstep completes. It is ideal when the UI needs to re-render the entire state (such as full conversation message history). `stream_mode='updates'` yields only the partial dictionary returned by the specific node that ran during that superstep, which minimizes network payload and enables tracking discrete node actions in real time."),
    ("Why is `thread_id` mandatory when invoking a compiled graph with a checkpointer?",
     "Checkpointers manage persistent conversation threads. The `thread_id` serves as the primary partition key in the checkpoint database. Without a thread_id, the checkpointer cannot identify which historical thread to load state from or where to commit the current superstep's snapshot, causing state collisions across concurrent users."),
    ("How does LangGraph's `interrupt_before` parameter work during compilation?",
     "When you compile with `interrupt_before=['node_name']`, LangGraph pauses execution immediately before `node_name` runs. The graph saves the checkpoint and returns control to the caller. A human operator can inspect the state, edit the state if necessary, and resume execution by calling `graph.invoke(None, config)` with the same thread_id."),
    ("Can a StateGraph node be an existing LCEL chain?",
     "Yes. Because LangChain LCEL chains implement the Runnable interface, any LCEL chain whose input matches the State schema and whose output returns a dictionary of state updates can be added directly as a node: `builder.add_node('rag_chain', rag_chain)`."),
    ("What is the difference between `add_messages` and a standard Python list in state schemas?",
     "A standard Python list without a reducer is an overwrite channel: returning `{'messages': [new_msg]}` replaces all previous messages with `[new_msg]`. `add_messages` is a specialized reducer that appends new messages while updating existing messages by their unique `id` attribute, preserving full conversational memory.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Mutable Default Values in TypedDict Schemas. Defining default mutable objects (like `logs: list = []`) directly in state classes causes shared reference bugs across threads. Always initialize state with fresh dictionaries on invocation.")
    st.error("**Trap 2 (Breaking Error)**: Forgetting Thread Config in Checkpointed Graphs. Calling `app.invoke(inputs)` on a graph compiled with a checkpointer without passing `config={'configurable': {'thread_id': '...'}}` raises a ValueError at runtime.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Streaming `values` Mode Over High-Volume Websockets. In a graph with 50+ messages, streaming `values` mode transmits the full message history on every single superstep, multiplying network bandwidth 50x. Use `stream_mode='updates'` or `messages` for streaming.")
    st.warning("**Trap 4 (Quality Warning)**: Modifying State Inside Conditional Routing Functions. Conditional edge functions must be pure read-only routing functions. Mutating state attributes inside a router function produces non-deterministic execution bugs.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: StateGraph(State)"):
        st.markdown("**Builder class that defines nodes, edges, and state schemas before compilation.**")
    with st.expander("🎴 Flashcard 2: stream_mode='updates'"):
        st.markdown("**Yields partial dictionary deltas emitted by active nodes per superstep.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: thread_id"):
        st.markdown("**Mandatory config key that partitions persistent state checkpoints per conversation session.**")
    with st.expander("🎴 Flashcard 4: interrupt_before"):
        st.markdown("**Compilation parameter that halts execution before target nodes for human review.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand how to instantiate and parameterize a StateGraph with a TypedDict schema.", value=True)
chk2 = st.checkbox("I can create functional nodes and wire both direct and conditional edges.", value=True)
chk3 = st.checkbox("I know how to compile a graph with checkpointers and breakpoint interrupts.", value=True)
chk4 = st.checkbox("I understand why thread_id is required for persistent state resumption.", value=True)
chk5 = st.checkbox("I can differentiate between stream_mode='values' and stream_mode='updates'.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 building graphs technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 17.3 Building Graphs!")
