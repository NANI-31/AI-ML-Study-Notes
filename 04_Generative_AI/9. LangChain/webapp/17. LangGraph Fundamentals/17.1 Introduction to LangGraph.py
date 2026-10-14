import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockLangGraphIntroState:
    """Represents a simplified state schema tracking messages, tool calls, and iteration cycles."""
    def __init__(self, query: str):
        self.query = query
        self.messages: List[Dict[str, str]] = [{"role": "user", "content": query}]
        self.iteration: int = 0
        self.tool_calls: List[str] = []
        self.is_finished: bool = False
        self.final_response: str = ""

class MockLangGraphIntroSimulator:
    """
    Simulates a cyclic LangGraph state machine:
    START -> Agent Model -> Conditional Edge (Tool Call vs Final Answer) -> Tool Node -> Agent Model -> END.
    """
    def __init__(self):
        pass

    def run_cyclic_graph(self, query: str, max_iterations: int = 4) -> Dict[str, Any]:
        state = MockLangGraphIntroState(query)
        step_history: List[Dict[str, Any]] = []

        q_lower = query.lower()
        needs_tool = any(w in q_lower for w in ["database", "sql", "weather", "search", "cluster", "latency", "status"])

        # Node 1: START
        step_history.append({
            "step": 0,
            "node": "START",
            "state_snapshot": {"messages": len(state.messages), "iteration": state.iteration},
            "description": "Graph execution begins. Input query loaded into state channel."
        })

        while not state.is_finished and state.iteration < max_iterations:
            state.iteration += 1

            # Node 2: Agent Reasoning Node
            if state.iteration == 1 and needs_tool:
                state.tool_calls.append("execute_system_query")
                state.messages.append({
                    "role": "assistant",
                    "content": "Action: execute_system_query(target='infrastructure_metrics')"
                })
                step_history.append({
                    "step": len(step_history),
                    "node": "agent_model",
                    "state_snapshot": {"iteration": state.iteration, "tool_pending": True},
                    "description": "Agent inspected state and decided a tool call is required to resolve query."
                })

                # Conditional Edge: route to tool_node
                step_history.append({
                    "step": len(step_history),
                    "node": "should_continue (conditional_edge)",
                    "state_snapshot": {"decision": "tool_node"},
                    "description": "Conditional routing edge evaluated tool_calls present -> routed to tool_node."
                })

                # Node 3: Tool Execution Node
                state.messages.append({
                    "role": "tool",
                    "content": "Result: Aurora PostgreSQL cluster latency=4.2ms, CPU load=22%, replicas=3"
                })
                step_history.append({
                    "step": len(step_history),
                    "node": "tool_node",
                    "state_snapshot": {"iteration": state.iteration, "tool_executed": True},
                    "description": "Tool executed successfully and emitted tool response message into state channel."
                })

                # Cyclic Edge: loop back to agent_model
                step_history.append({
                    "step": len(step_history),
                    "node": "cyclic_edge",
                    "state_snapshot": {"target": "agent_model"},
                    "description": "Edge loops back from tool_node into agent_model with updated message history."
                })

            else:
                # Agent Synthesizes Final Answer
                if needs_tool:
                    ans = f"System Report: Analysis shows healthy operational parameters for query '{query}': latency is 4.2ms with 22% CPU across 3 replicas."
                else:
                    ans = f"Direct Response: '{query}' was resolved using reasoning without external tool execution."
                state.messages.append({"role": "assistant", "content": ans})
                state.final_response = ans
                state.is_finished = True

                step_history.append({
                    "step": len(step_history),
                    "node": "agent_model",
                    "state_snapshot": {"iteration": state.iteration, "tool_pending": False},
                    "description": "Agent analyzed tool results and generated final user answer."
                })

                # Conditional Edge: route to END
                step_history.append({
                    "step": len(step_history),
                    "node": "should_continue (conditional_edge)",
                    "state_snapshot": {"decision": "END"},
                    "description": "Conditional routing edge evaluated no tool calls -> routed to END."
                })

        step_history.append({
            "step": len(step_history),
            "node": "END",
            "state_snapshot": {"messages": len(state.messages), "final": True},
            "description": "Execution reached END. Final state returned to caller."
        })

        return {
            "query": query,
            "final_response": state.final_response,
            "total_iterations": state.iteration,
            "history": step_history
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="17.1 Introduction to LangGraph — LangChain Course",
    page_icon="🕸️",
    layout="wide"
)

st.title("17.1 Introduction to LangGraph")
st.caption("The Next-Generation Agent Runtime: Cyclic State Machines, StateGraph Architecture, and Enterprise Agent Orchestration")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Cyclic State Machine & Agent Orchestration Engine", divider="rainbow")
st.markdown("""
Traditional LLM frameworks model pipelines as **Directed Acyclic Graphs (DAGs)**: chains execute sequentially from A to B 
with no way to loop backward. Real-world agents, however, require **cyclic execution**: reasoning, selecting a tool, 
inspecting the result, and looping back to reason again until the task is complete.

**LangGraph** is LangChain's lower-level orchestration runtime designed specifically for **cyclic, stateful, multi-actor applications**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangGraphIntroArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_start {
            label = "Step 1: Graph Entry Point";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            start_node [label="START\\n(Initial State: query, empty messages)", fillcolor="#1E293B", color="#0284C7"];
        }

        subgraph cluster_cycle {
            label = "Step 2: Cyclic Reasoning & Execution Loop";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            
            agent_node [label="Node: 'agent' (LLM Call)\\nInputs: Current State\\nOutputs: State Updates (AIMessage)", fillcolor="#0F172A", color="#38BDF8"];
            cond_edge [label="Conditional Edge: should_continue()\\nInspects AIMessage.tool_calls", fillcolor="#D97706", color="#F59E0B"];
            tool_node [label="Node: 'tools' (ToolNode Execution)\\nExecutes external tools in parallel\\nEmits ToolMessage updates", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_persistence {
            label = "Step 3: State Persistence & Memory";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            checkpointer [label="Persistent Checkpointer\\n(SqliteSaver / PostgresSaver)\\nSaves snapshot per superstep", fillcolor="#1E293B", color="#64748B"];
        }

        subgraph cluster_end {
            label = "Step 4: Terminal Exit Point";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            end_node [label="END\\nReturns Final State to Client", fillcolor="#064E3B", color="#10B981"];
        }

        start_node -> agent_node [label="invoke(input)"];
        agent_node -> cond_edge;
        
        cond_edge -> tool_node [label="has tool_calls", color="#F59E0B"];
        cond_edge -> end_node [label="no tool_calls / finished", color="#10B981"];
        
        tool_node -> agent_node [label="Cyclic Loop: return tool output", color="#38BDF8", style="bold"];

        agent_node -> checkpointer [label="checkpoint write", style="dotted"];
        tool_node -> checkpointer [label="checkpoint write", style="dotted"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT IS LANGGRAPH? ---
st.header("2. What is LangGraph?", divider="rainbow")
st.markdown("""
**LangGraph** is an open-source library developed by LangChain to build resilient, multi-actor applications with LLMs 
by modeling agent workflows as **state machines**.

### Core Tenets of LangGraph:
1. **First-Class Cycles**: Loops are native primitives. Nodes can transition to other nodes or cycle back onto themselves indefinitely.
2. **Explicit Shared State**: Every component reads from and writes to a central, type-safe state schema.
3. **Durable Persistence**: Built-in checkpointers serialize graph state at every step, enabling **human-in-the-loop approvals**, 
   **time-travel debugging**, and fault-tolerant resumption.
4. **Fine-Grained Control**: Unlike black-box agent loops (such as legacy `AgentExecutor`), developers control exact node transitions, 
   branching conditions, and state reducer rules.
""")

st.divider()

# --- SECTION 3: WHY LANGGRAPH EXISTS ---
st.header("3. Why LangGraph Exists: Beyond Linear Chains", divider="rainbow")
st.markdown("""
Earlier agent abstractions in LangChain v0.1 (such as `LLMChain` and `AgentExecutor`) were built on rigid linear assumptions:
""")

col_wh1, col_wh2 = st.columns(2)
with col_wh1:
    st.error("""
    **Limitations of Legacy AgentExecutor**:
    - **Black-Box Loop**: The reasoning loop `Thought -> Action -> Observation` was hardcoded inside C++ style while-loops; 
      customizing termination or error recovery was cumbersome.
    - **No Human-in-the-Loop**: Halting execution to wait for human approval required fragile monkey-patching.
    - **Ephemeral State**: State was kept in RAM. If the server crashed mid-agent execution, all context was lost.
    - **DAG Constraints**: LCEL chains are strictly acyclic; creating recursive reflection loops required awkward recursive hacks.
    """)
with col_wh2:
    st.success("""
    **LangGraph's Architectural Solutions**:
    - **Explicit Graph Topology**: Nodes, edges, and state transitions are explicitly declared and visualized.
    - **Native Interrupts**: The `interrupt()` API pauses execution at any node and waits for external approval.
    - **Persistent Checkpointing**: State is committed to disk or PostgreSQL on every superstep, making execution crash-proof.
    - **First-Class Cycles**: Re-prompting, reflection loops, and multi-agent debate are clean, native cycles.
    """)

st.divider()

# --- SECTION 4: LANGCHAIN VS LANGGRAPH ---
st.header("4. LangChain vs LangGraph Architectural Comparison", divider="rainbow")
st.markdown("""
LangGraph does **not replace LangChain**; they operate at complementary layers of the modern Generative AI software stack.
""")

comp_data = {
    "Architectural Dimension": [
        "Core Metaphor",
        "Control Flow",
        "State Management",
        "Persistence & Resumption",
        "Human-in-the-Loop (HITL)",
        "Primary Use Cases",
        "Recommended LangChain Stack Role"
    ],
    "LangChain (LCEL)": [
        "Chains & Runnables (pipe operator |)",
        "Strictly Acyclic Directed Graphs (DAG)",
        "Stateless: inputs transform into outputs",
        "Ephemeral: runtime context lost on exit",
        "Not natively supported",
        "Single-turn RAG, extraction, transformations",
        "Component primitives (models, prompts, tools)"
    ],
    "LangGraph": [
        "State Machine Graphs (StateGraph)",
        "Cyclic Graphs (Native loops & branches)",
        "Stateful: central State schema with reducers",
        "Durable: thread checkpointing to SQL / Redis",
        "First-class interrupts and resume commands",
        "Autonomous agents, multi-agent teams, coding bots",
        "Orchestration runtime for multi-step agents"
    ]
}
st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

st.divider()

# --- SECTION 5: GRAPH-BASED WORKFLOWS ---
st.header("5. Graph-Based Workflows", divider="rainbow")
st.markdown("""
A LangGraph application is structured as a **StateGraph** composed of three fundamental building blocks:
1. **Nodes**: Python functions (or LCEL runnables) that accept current state and return state updates.
2. **Edges**: Direct connections specifying that execution flows from Node A directly to Node B.
3. **Conditional Edges**: Dynamic routing functions that inspect current state and determine which node to visit next.
""")

st.code("""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class WorkflowState(TypedDict):
    input_text: str
    processed_text: str

def clean_node(state: WorkflowState):
    return {"processed_text": state["input_text"].strip().lower()}

builder = StateGraph(WorkflowState)
builder.add_node("clean", clean_node)
builder.add_edge(START, "clean")
builder.add_edge("clean", END)
graph = builder.compile()
""", language="python")

st.divider()

# --- SECTION 6: STATEFUL APPLICATIONS ---
st.header("6. Stateful Applications & State Reducers", divider="rainbow")
st.markdown("""
State in LangGraph is managed through **channels**. By default, when a node returns a dictionary key, it overwrites 
the existing value in the state. However, for conversation histories, we want to **append new messages** rather than overwrite.

This is achieved using **State Reducers**, most commonly `add_messages` from `langgraph.graph.message`.
""")

st.code("""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # The add_messages reducer appends new messages and updates existing ones by ID
    messages: Annotated[list[AnyMessage], add_messages]
    sender: str
""", language="python")

st.divider()

# --- SECTION 7: AGENT ORCHESTRATION ---
st.header("7. Agent Orchestration Patterns", divider="rainbow")
st.markdown("""
LangGraph provides the foundation for advanced agent topologies:
- **ReAct Agent**: A single agent looping between model reasoning and tool execution.
- **Supervisor Pattern**: A centralized manager agent delegating tasks to worker specialist agents.
- **Hierarchical Teams**: Subgraphs operating as isolated teams with local state, reporting back to an executive graph.
- **Reflection / Critic Loops**: A generator agent writes code; a critic agent tests the code and loops back with bug fixes.
""")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: End-to-End Cyclic StateGraph Pipeline", divider="rainbow")
st.markdown("""
Below is a complete, production-ready implementation of a cyclic ReAct agent built with LangGraph v0.2+ 
utilizing `StateGraph`, `ToolNode`, and memory checkpointing.
""")

st.code("""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. Define Tools
@tool
def lookup_database_sla(service_name: str) -> str:
    \"\"\"Looks up enterprise availability SLA for a given database service.\"\"\"
    return f"SLA for {service_name}: 99.995% uptime with 12-second automated failover."

tools = [lookup_database_sla]
tool_node = ToolNode(tools)

# 2. Define State Schema
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Model with Tool Binding
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 4. Conditional Edge Function
def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")  # Cyclic loop back!

# 6. Compile with Checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 7. Execute with Thread ID
# config = {"configurable": {"thread_id": "thread-101"}}
# output = app.invoke({"messages": [HumanMessage(content="What is our Aurora SLA?")]}, config=config)
""", language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: LangGraph Cycle & State Simulator", divider="rainbow")
st.markdown("""
Step through the execution of a **cyclic LangGraph state machine**. Observe how state transitions between nodes, 
how conditional edges route decisions, and how cyclic edges return tool outputs back to the reasoning model.
""")

# Instantiate module-scope simulator
graph_simulator = MockLangGraphIntroSimulator()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    test_query_in = st.selectbox(
        "Select User Inquiry:",
        [
            "Check current Aurora database cluster latency and replica status.",
            "Explain what a cyclic state machine is in conceptual terms.",
            "Look up production cloud infrastructure metrics."
        ]
    )

    max_iter_slider = st.slider("Maximum Iteration Cycle Limit:", min_value=1, max_value=5, value=3)
    run_graph_btn = st.button("🕸️ Execute StateGraph Simulation", type="primary", use_container_width=True)

with wb_c2:
    st.subheader("Graph Execution Telemetry")

    if run_graph_btn or test_query_in:
        graph_run = graph_simulator.run_cyclic_graph(test_query_in, max_iterations=max_iter_slider)

        st.markdown("### 🤖 Final Synthesized Output")
        st.success(graph_run["final_response"])
        st.caption(f"**Total Supersteps / Iterations Executed**: `{graph_run['total_iterations']}`")

        st.markdown("### 🔄 Superstep Transition Log")
        for step_info in graph_run["history"]:
            with st.expander(f"Step {step_info['step']}: Node [{step_info['node']}]", expanded=True):
                st.write(f"**Action**: {step_info['description']}")
                st.json(step_info["state_snapshot"])

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why can't standard LCEL chains support agent reflection loops cleanly?",
     "LCEL is designed for Directed Acyclic Graphs (DAGs) where data flows strictly downstream from one Runnable to the next. While a chain can call an external function that contains an internal while-loop, the chain itself cannot model cyclic transitions, partial state checkpointing, or mid-cycle human interruptions. LangGraph introduces native cyclic state transitions, where any node can return an edge to an upstream node."),
    ("What is the role of a state reducer in LangGraph, and what does `add_messages` do?",
     "A state reducer defines how updates from a node are merged into the existing state. By default, LangGraph replaces key values with the returned update. The `add_messages` reducer modifies this behavior for message lists: if a message has a new ID, it is appended; if an incoming message shares an existing ID, it replaces the existing message. This enables append-only histories and in-place message editing."),
    ("What is a LangGraph superstep?",
     "A superstep is a single atomic execution phase inspired by the Pregel graph processing model. In a superstep, all active nodes run in parallel. At the end of the superstep, their state updates are resolved and merged by channel reducers, the checkpointer persists the state, and outgoing edges determine which nodes will run in the next superstep."),
    ("How does LangGraph achieve fault tolerance in long-running multi-step agents?",
     "LangGraph integrates persistent checkpointers (e.g. PostgresSaver). After every superstep, the complete state snapshot is written to disk/database alongside thread metadata. If an agent fails mid-execution due to an infrastructure outage or API rate limit, the application can reload the latest checkpoint by thread_id and resume execution from the exact point of interruption."),
    ("What is the difference between direct edges and conditional edges?",
     "A direct edge connects Node A to Node B unconditionally: whenever Node A completes, Node B always executes next. A conditional edge connects Node A to a routing function: the routing function inspects the state returned by Node A and dynamically returns the name of the next node (e.g. routing to 'tools' if tool calls are present, or 'END' if the task is finished).")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Creating Infinite Cycles Without Iteration Caps. If an agent fails to parse tool output, it may loop between model and tools endlessly, exhausting API credits. Always configure `recursion_limit` in the graph execution config (default is 25).")
    st.error("**Trap 2 (Breaking Error)**: Mutating State in Place Instead of Returning Dictionaries. In LangGraph nodes, modifying `state['messages'].append(...)` directly violates pure functional invariants and causes bugs in time travel. Nodes must always return a dictionary of updates: `return {'messages': [new_msg]}`.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Forgetting `add_messages` Reducer on Conversation State. If you define a message list without `Annotated[list, add_messages]`, nodes that return `{'messages': [msg]}` will overwrite the entire conversation history, wiping past context.")
    st.warning("**Trap 4 (Quality Warning)**: Passing Non-Serializable Objects in State Schemas. Checkpointers serialize state to JSON or binary storage. Storing active database connections, file handles, or raw sockets in state will cause checkpointer serialization crashes.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: LangGraph"):
        st.markdown("**Low-level orchestration runtime for building cyclic, stateful, multi-actor LLM agents.**")
    with st.expander("🎴 Flashcard 2: Superstep"):
        st.markdown("**Atomic parallel execution phase where nodes execute, state updates merge, and checkpoints commit.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: add_messages Reducer"):
        st.markdown("**State reducer that appends new messages or updates existing messages by ID instead of overwriting.**")
    with st.expander("🎴 Flashcard 4: Checkpointer"):
        st.markdown("**Persistence engine saving state snapshots per superstep to enable fault-tolerance and human approval.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why LangGraph exists to overcome the acyclic constraints of linear chains.", value=True)
chk2 = st.checkbox("I can articulate the architectural differences between LangChain (LCEL) and LangGraph.", value=True)
chk3 = st.checkbox("I understand the roles of State, Nodes, Edges, and Conditional Edges in a StateGraph.", value=True)
chk4 = st.checkbox("I know how the add_messages reducer manages conversation history across supersteps.", value=True)
chk5 = st.checkbox("I understand how persistent checkpointers provide fault-tolerant resume capabilities.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 LangGraph fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 17.1 Introduction to LangGraph!")
