import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.1 Graph Observability - LangSmith Master Curriculum",
    page_icon="🕸️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class GraphObservabilityEngine:
    """
    Production LangGraph Observability & State Machine Tracer.
    Simulates distributed state machine execution, cyclic iteration telemetry,
    channel reducer auditing (add_messages), persistent checkpoint versioning,
    conditional edge routing, and tool call span instrumentation in LangSmith.
    """
    def __init__(self, project_name: str = "langgraph-state-observability"):
        self.project_name = project_name
        self.recorded_runs: List[Dict[str, Any]] = []
        self._checkpoint_counter = 1000

    def execute_graph_workflow(
        self,
        thread_id: str,
        user_query: str,
        max_revisions: int = 2,
        inject_tool_error: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a simulated LangGraph state machine with cyclic critique and revision.
        Captures discrete node execution spans, state snapshots at each checkpoint,
        edge routing decisions, and tool invocation telemetry.
        """
        self._checkpoint_counter += 1
        run_id = f"graph_run_{self._checkpoint_counter}"

        # Initial State Schema
        current_state = {
            "messages": [f"User: {user_query}"],
            "query": user_query,
            "research_notes": "",
            "critique_score": 0.0,
            "revision_count": 0,
            "final_answer": ""
        }

        checkpoints = []
        node_spans = []
        edge_transitions = []
        step_idx = 0

        # Step 0: Initial Checkpoint at START
        step_idx += 1
        cp_0 = {
            "checkpoint_id": f"cp_{self._checkpoint_counter}_{step_idx}",
            "step": step_idx,
            "node": "__start__",
            "state_snapshot": dict(current_state),
            "channel_diff": {"messages": f"+1 message (User: {user_query})"}
        }
        checkpoints.append(cp_0)

        # Cyclic Execution Loop
        is_finished = False
        loop_counter = 0

        while not is_finished and loop_counter <= max_revisions:
            loop_counter += 1

            # Node 1: Researcher Node
            step_idx += 1
            node1_latency = 145.0 + (loop_counter * 12.0)
            research_update = f"Draft research findings for '{user_query}' (Cycle {loop_counter})"
            current_state["research_notes"] = research_update
            current_state["messages"].append(f"Researcher: {research_update}")

            span_research = {
                "step": step_idx,
                "node_name": "researcher_node",
                "run_type": "chain",
                "latency_ms": node1_latency,
                "inputs": {"query": user_query, "revision_count": loop_counter - 1},
                "outputs": {"research_notes": research_update},
                "status": "success"
            }
            node_spans.append(span_research)
            edge_transitions.append({"from": "__start__" if loop_counter == 1 else "critic_node", "to": "researcher_node", "type": "direct"})

            # Checkpoint after Researcher
            checkpoints.append({
                "checkpoint_id": f"cp_{self._checkpoint_counter}_{step_idx}",
                "step": step_idx,
                "node": "researcher_node",
                "state_snapshot": dict(current_state),
                "channel_diff": {"research_notes": research_update, "messages": "+1 message"}
            })

            # Node 2: Tool Node (Simulated Web Search / Knowledge Retrieval)
            step_idx += 1
            tool_latency = 85.0
            tool_status = "error" if (inject_tool_error and loop_counter == 1) else "success"
            tool_output = "Error 503: External API Unavailable" if tool_status == "error" else f"Verified internal knowledge for {user_query}"
            current_state["messages"].append(f"Tool[Search]: {tool_output}")

            span_tool = {
                "step": step_idx,
                "node_name": "tool_node",
                "run_type": "tool",
                "latency_ms": tool_latency,
                "inputs": {"tool_input": f"search({user_query})"},
                "outputs": {"tool_result": tool_output},
                "status": tool_status
            }
            node_spans.append(span_tool)
            edge_transitions.append({"from": "researcher_node", "to": "tool_node", "type": "direct"})

            checkpoints.append({
                "checkpoint_id": f"cp_{self._checkpoint_counter}_{step_idx}",
                "step": step_idx,
                "node": "tool_node",
                "state_snapshot": dict(current_state),
                "channel_diff": {"messages": f"+1 ToolMessage (status={tool_status})"}
            })

            # Node 3: Critic Node
            step_idx += 1
            critic_latency = 120.0
            # Critique score improves with revisions
            score = 0.55 if loop_counter == 1 else 0.88
            current_state["critique_score"] = score
            current_state["revision_count"] = loop_counter
            critique_msg = f"Critique Score: {score:.2f} (Revision {loop_counter})"
            current_state["messages"].append(f"Critic: {critique_msg}")

            span_critic = {
                "step": step_idx,
                "node_name": "critic_node",
                "run_type": "chain",
                "latency_ms": critic_latency,
                "inputs": {"notes": current_state["research_notes"]},
                "outputs": {"critique_score": score},
                "status": "success"
            }
            node_spans.append(span_critic)
            edge_transitions.append({"from": "tool_node", "to": "critic_node", "type": "direct"})

            checkpoints.append({
                "checkpoint_id": f"cp_{self._checkpoint_counter}_{step_idx}",
                "step": step_idx,
                "node": "critic_node",
                "state_snapshot": dict(current_state),
                "channel_diff": {"critique_score": score, "revision_count": loop_counter}
            })

            # Conditional Edge Evaluation
            if score >= 0.80 or loop_counter >= max_revisions:
                edge_transitions.append({
                    "from": "critic_node",
                    "to": "synthesizer_node",
                    "type": f"conditional (score={score:.2f} >= 0.80 or limit reached)"
                })
                is_finished = True
            else:
                edge_transitions.append({
                    "from": "critic_node",
                    "to": "researcher_node",
                    "type": f"conditional (score={score:.2f} < 0.80 -> REVISE)"
                })

        # Node 4: Synthesizer Node (Terminal)
        step_idx += 1
        synth_latency = 210.0
        final_text = f"Synthesized verified response for '{user_query}' after {loop_counter} research iteration(s)."
        current_state["final_answer"] = final_text
        current_state["messages"].append(f"Synthesizer: {final_text}")

        span_synth = {
            "step": step_idx,
            "node_name": "synthesizer_node",
            "run_type": "chain",
            "latency_ms": synth_latency,
            "inputs": {"notes": current_state["research_notes"], "score": current_state["critique_score"]},
            "outputs": {"final_answer": final_text},
            "status": "success"
        }
        node_spans.append(span_synth)
        edge_transitions.append({"from": "synthesizer_node", "to": "__end__", "type": "direct"})

        checkpoints.append({
            "checkpoint_id": f"cp_{self._checkpoint_counter}_{step_idx}",
            "step": step_idx,
            "node": "synthesizer_node",
            "state_snapshot": dict(current_state),
            "channel_diff": {"final_answer": final_text}
        })

        execution_record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user_query": user_query,
            "total_iterations": loop_counter,
            "total_steps": step_idx,
            "total_latency_ms": sum(s["latency_ms"] for s in node_spans),
            "final_score": current_state["critique_score"],
            "checkpoints": checkpoints,
            "spans": node_spans,
            "transitions": edge_transitions,
            "final_state": current_state
        }

        self.recorded_runs.append(execution_record)
        return execution_record


# Instantiate engine
graph_tracer = GraphObservabilityEngine()

# --- HEADER SECTION ---
st.title("🕸️ Chapter 24.1: LangGraph Observability & State Machine Tracing")
st.markdown("""
Welcome to the observability manual for **LangGraph state machines** traced with **LangSmith**.
While standard chains follow linear, unidirectional Directed Acyclic Graphs (DAGs), production AI agents
are cyclical, stateful, and non-deterministic.
LangGraph models agents as state machines where nodes compute state updates, edges determine conditional transitions,
and checkpointers persist state at every step.
LangSmith provides full-spectrum observability into cyclic loops, channel reducers, checkpoint diffs, and tool node executions.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LangGraph State Machine Execution & Distributed Checkpoint Tracing", divider="rainbow")

st.markdown("""
The architecture below illustrates how LangSmith captures cyclic graph iterations.
Every graph execution runs under a durable `thread_id`. Each step executes a node function,
computes channel updates using reducers (like `add_messages`), writes a checkpoint snapshot to disk or memory,
evaluates conditional routing logic, and streams execution telemetry to LangSmith.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangGraphObservability {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_invocation {
            label="1. Invocation Boundary & Thread Scope";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            thread [label="Client Invocations\nthread_id: 'sess-8492'\nRunnableConfig: {metadata, tags}", fillcolor="#1E293B", color="#38BDF8"];
            start_node [label="StateGraph Entry\n__start__ Node", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_state_core {
            label="2. Cyclic State Machine Core";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            research [label="Node: Researcher\nQuery Formulation & Synthesis", fillcolor="#1E293B", color="#38BDF8"];
            tool_node [label="Node: ToolNode\nExternal Tools / RAG VectorDB", fillcolor="#1E293B", color="#38BDF8"];
            critic [label="Node: Critic\nRubric & Quality Scoring", fillcolor="#1E293B", color="#38BDF8"];
            cond_edge [label="Conditional Edge Router\nBranch: critique_score >= 0.80 ?", shape=diamond, fillcolor="#0F172A", color="#F59E0B"];
            synth [label="Node: Synthesizer\nFinal Document Polish", fillcolor="#1E293B", color="#38BDF8"];
            end_node [label="StateGraph Terminal\n__end__ Node", fillcolor="#1E293B", color="#10B981"];
        }

        subgraph cluster_checkpointer {
            label="3. Checkpointer Persistence Layer";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            cp_store [label="Checkpointer (MemorySaver / SqliteSaver)\nImmutable State Snapshots\nThread Versions: v1 -> v2 -> v3", fillcolor="#0F172A", color="#38BDF8"];
            time_travel [label="Time-Travel & Human-In-The-Loop\nFork Thread at Checkpoint v2\nState Rewind & State Mutation", fillcolor="#0F172A", color="#A855F7"];
        }

        subgraph cluster_langsmith_ui {
            label="4. LangSmith Telemetry & Trace Hierarchy";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_graph [label="LangSmith Graph Execution Tree\nRoot Run (CompiledStateGraph)\nChild Runs (Nodes, Tools, Reducers)", fillcolor="#0F172A", color="#10B981"];
            ls_channels [label="Channel Reducer Audit\nmessages: [HumanMessage, AIMessage, ToolMessage]\nDelta Diffs Highlighted", fillcolor="#0F172A", color="#38BDF8"];
        }

        thread -> start_node;
        start_node -> research [label="invoke()"];
        research -> tool_node [label="emit ToolCalls"];
        tool_node -> critic [label="emit ToolMessage"];
        critic -> cond_edge [label="eval state"];

        cond_edge -> research [label="REVISE (score < 0.80)", color="#F59E0B"];
        cond_edge -> synth [label="ACCEPT (score >= 0.80)", color="#10B981"];
        synth -> end_node;

        research -> cp_store [label="Save Checkpoint", style=dotted, color="#94A3B8"];
        tool_node -> cp_store [label="Save Checkpoint", style=dotted, color="#94A3B8"];
        critic -> cp_store [label="Save Checkpoint", style=dotted, color="#94A3B8"];
        synth -> cp_store [label="Save Checkpoint", style=dotted, color="#94A3B8"];

        cp_store -> time_travel [label="Branch / Fork"];
        start_node -> ls_graph [label="Root Run Span"];
        research -> ls_graph [label="Node Span"];
        tool_node -> ls_graph [label="Tool Span"];
        cp_store -> ls_channels [label="State Diffs"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Six Core Telemetry Dimensions of LangGraph in LangSmith
The following table provides the operational breakdown of each syllabus topic:
""")

lg_topics_df = pd.DataFrame([
    {
        "Dimension": "1. Graph Execution",
        "LangSmith Span Type": "chain (Root: CompiledStateGraph)",
        "Key Telemetry Captured": "thread_id, recursion_limit, total steps, aggregate latency, status (COMPLETED/INTERRUPTED)",
        "Production Relevance": "Top-level session tracking, billing allocation per thread, user workflow auditing"
    },
    {
        "Dimension": "2. Node Execution",
        "LangSmith Span Type": "chain / node (Child Run)",
        "Key Telemetry Captured": "Function name, inputs (channel state subset), outputs (channel update dict), execution latency",
        "Production Relevance": "Isolates bottlenecks to specific agent roles (e.g., slow research node vs fast critic)"
    },
    {
        "Dimension": "3. State Updates",
        "LangSmith Span Type": "metadata / channel events",
        "Key Telemetry Captured": "Reducer execution (add_messages, dict union), immutable state diffs, channel keys modified",
        "Production Relevance": "Detects state corruption, unbounded message array bloat, and unexpected key overwrites"
    },
    {
        "Dimension": "4. Edge Transitions",
        "LangSmith Span Type": "metadata / routing span",
        "Key Telemetry Captured": "Source node, target node, conditional function output string, branch path evaluated",
        "Production Relevance": "Diagnoses infinite loops, incorrect router conditions, and unexpected premature endings"
    },
    {
        "Dimension": "5. Checkpoints",
        "LangSmith Span Type": "checkpointer event",
        "Key Telemetry Captured": "checkpoint_id, thread_id, parent_checkpoint_id, version hash, serialized snapshot",
        "Production Relevance": "Enables time-travel debugging, audit replay, and crash recovery across multi-day tasks"
    },
    {
        "Dimension": "6. Tool Calls",
        "LangSmith Span Type": "tool (Child Run of ToolNode)",
        "Key Telemetry Captured": "Tool name, input arguments JSON, return payload, ToolMessage ID, status (SUCCESS/ERROR)",
        "Production Relevance": "Monitors tool failure rates, argument hallucination, and downstream API latency"
    }
])
st.dataframe(lg_topics_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: GRAPH EXECUTION
# ==============================================================================
st.header("2. Graph Execution (Compiled Graph Invocations, Thread IDs & Top-Level Workflow Spans)", divider="rainbow")

st.markdown("""
When a `StateGraph` is compiled into a `CompiledStateGraph` via `app = graph.compile(checkpointer=memory)`,
invoking the graph creates a top-level root run in LangSmith.
Unlike simple chains, a graph execution can run across multiple turns, pause for human approval,
or resume days later.
""")

st.markdown("### Graph Invocations & Configuration Structure")
st.markdown("""
1. **Thread Identity (`thread_id`)**: Every persistent conversation or multi-step session must specify a unique `thread_id` inside `RunnableConfig`. This binds all subsequent steps to the same historical state thread.
2. **Recursion Bounds (`recursion_limit`)**: LangGraph enforces a hard safety limit on the maximum number of node transitions (default 25). If a cyclic agent exceeds this limit without terminating, a `GraphRecursionError` is raised.
3. **Execution Modes**: LangGraph supports streaming execution via `app.stream()` in multiple modes:
   * `'values'`: Emits the full state dictionary after every step.
   * `'updates'`: Emits only the channel delta emitted by the node that just executed.
   * `'messages'`: Streams raw token chunks directly from LLM calls inside nodes.
""")

st.code('''
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Instantiate persistence checkpointer
checkpointer = MemorySaver()

# Compile the graph
app = workflow.compile(checkpointer=checkpointer)

# Invoke with thread configuration
config = {
    "configurable": {
        "thread_id": "session-cust-9921",
        "checkpoint_ns": ""
    },
    "metadata": {"environment": "production", "tenant": "Alpha"}
}

# In LangSmith, this creates a root run titled 'LangGraph' with thread_id attached
events = app.stream({"messages": [("user", "Analyze balance sheet")]}, config=config, stream_mode="updates")
for event in events:
    print(event)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production enterprise architectures:
* The root graph span in LangSmith serves as the umbrella transaction for distributed billing and latency tracking.
* If a user cancels a query mid-flight, LangGraph's cooperative cancellation signals propagate down the execution tree, marking child spans as `CANCELLED` in LangSmith.
* Teams configure automated alerts that flag graphs whose total execution step count approaches `recursion_limit - 3`.
""")

st.divider()

# ==============================================================================
# SECTION 3: NODE EXECUTION
# ==============================================================================
st.header("3. Node Execution (Individual State Functions, Step Execution Latency & Input/Output Channel Snapshots)", divider="rainbow")

st.markdown("""
Nodes in LangGraph are standard Python functions (or runnables) that accept the current state and return a dictionary of state updates.
In LangSmith, every node invocation is automatically instrumented as a discrete child run.
""")

st.markdown("### Node Telemetry Anatomy")
st.markdown("""
* **Scoped State Ingestion**: When a node executes, LangSmith records the exact snapshot of channels available to that node.
* **Granular Latency Attribution**: Because nodes encapsulate distinct agent capabilities (e.g., drafting vs critiquing vs tool executing), engineers can instantly identify which specific agent persona consumes the most latency.
* **Isolated Crash Boundaries**: If a node throws an unhandled exception, only that node's span is marked `ERROR`, allowing engineers to inspect the exact input state that caused the crash.
""")

st.code('''
def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Node function: receives current state, invokes LLM, returns state update.
    LangSmith automatically captures this function name, its input state,
    execution duration, and the returned dictionary.
    """
    query = state["query"]
    iteration = state.get("revision_count", 0)
    
    # LLM inference child span is automatically nested under researcher_node
    response = model.invoke([("system", "Conduct technical research"), ("user", query)])
    
    return {
        "research_notes": response.content,
        "revision_count": iteration + 1
    }
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In high-throughput multi-agent systems:
* Nodes often run concurrently via LangGraph's parallel node dispatch (fan-out).
* LangSmith visualizes concurrent node spans side-by-side in the timeline waterfall, showing simultaneous execution before joining at a downstream reducer node.
""")

st.divider()

# ==============================================================================
# SECTION 4: STATE UPDATES
# ==============================================================================
st.header("4. State Updates (Channel Reducers: add_messages, Dict Merges & State Immutability Auditing)", divider="rainbow")

st.markdown("""
State management in LangGraph is governed by **Channels** and **Reducers**.
Unlike mutable global variables, LangGraph state channels are updated through deterministic mathematical reduction functions.
""")

st.markdown("### Channel Reducer Mechanics")
st.markdown("""
1. **Default Dict Overwrite**: By default, returning `{"query": "new"}` replaces the existing value of `query` in the state dictionary.
2. **The `add_messages` Reducer**: The most common reducer in agentic systems:
   * Appends new messages to the existing list.
   * Automatically handles deduplication and in-place updates if a message shares an existing `id`.
   * Preserves polymorphically typed message sequences (`HumanMessage`, `AIMessage`, `ToolMessage`).
3. **Custom Reducers**: Functions such as `operator.add` or custom set merges that govern how parallel nodes combine outputs into a single channel without race conditions.
""")

st.code('''
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ProductionAgentState(TypedDict):
    # Channel with add_messages reducer: appends messages, merges by ID
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Standard overwrite channel: each update replaces previous value
    critique_score: float
    
    # Custom reducer: accumulates distinct search queries
    search_queries: Annotated[List[str], lambda existing, new: existing + new]
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
State corruption is one of the most difficult bugs to isolate in distributed agent systems.
In LangSmith:
* Every node output is compared against the preceding state to generate an **interactive state diff**.
* Engineers can click on any step in the LangSmith trace view to see exactly which keys were modified, added, or overwritten.
* This makes it trivial to catch bugs where an agent accidentally wipes out conversational history by returning an empty list instead of relying on `add_messages`.
""")

st.divider()

# ==============================================================================
# SECTION 5: EDGE TRANSITIONS
# ==============================================================================
st.header("5. Edge Transitions (Conditional Edge Routing, Decision Branch Tracing & END Transitions)", divider="rainbow")

st.markdown("""
Edges in LangGraph define the control flow of the state machine.
They fall into two categories:
* **Fixed Edges**: Direct transitions (`builder.add_edge("node_a", "node_b")`).
* **Conditional Edges**: Dynamic routing decisions (`builder.add_conditional_edges("critic", route_decision)`).
""")

st.markdown("### Debugging Conditional Routing in LangSmith")
st.markdown("""
In cyclic graphs, conditional routing bugs are the number one cause of runaway costs and infinite loops:
* **The Return Value Mismatch**: The routing function returns `"continue"`, but the routing dictionary maps `{"retry": "node_a", "done": END}`.
* **Oscillating Decisions**: Node A and Node B repeatedly route back to each other because their state conditions never converge.
* **Premature Termination**: A conditional edge routes to `END` before required data has been synthesized or validated.
""")

st.code('''
def critique_router(state: AgentState) -> str:
    """
    Evaluates current state and returns destination node name.
    """
    score = state.get("critique_score", 0.0)
    revisions = state.get("revision_count", 0)
    
    # Guardrail against infinite loops
    if score >= 0.85 or revisions >= 3:
        return "synthesizer"
    return "researcher"

# Binding conditional edge to the StateGraph builder
builder.add_conditional_edges(
    "critic_node",
    critique_router,
    {
        "synthesizer": "synthesizer_node",
        "researcher": "researcher_node"
    }
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith, conditional edge transitions are explicitly visualized within the run timeline:
* The routing function's input state, evaluated return value, and chosen edge destination are recorded as metadata on the parent step.
* When viewing a completed trace, engineers can step through the sequence of transitions to see the exact trajectory the agent navigated through the graph.
""")

st.divider()

# ==============================================================================
# SECTION 6: CHECKPOINTS
# ==============================================================================
st.header("6. Checkpoints (Persistent State Snapshots, MemorySaver / SqliteSaver & Time-Travel Debugging)", divider="rainbow")

st.markdown("""
Checkpoints are the foundational persistence primitive of LangGraph.
At every super-step of graph execution, the checkpointer serializes the entire state dictionary and saves it
along with the `thread_id`, `checkpoint_id`, and parent lineage pointer.
""")

st.markdown("### What Checkpointing Enables in Production")
st.markdown("""
1. **Durable Fault Tolerance**: If a worker node crashes or the container restarts mid-execution, the graph resumes seamlessly from the exact state of the latest checkpoint.
2. **Time-Travel Debugging**: Developers can inspect, replay, or fork the agent from any historical step in the conversation thread.
3. **Human-in-the-Loop Interrupts**: The graph can pause before dangerous operations (like executing a SQL drop or financial transaction), wait for human approval, and resume upon confirmation.
""")

st.code('''
from langgraph.checkpoint.sqlite import SqliteSaver

# In production: durable SQLite or Postgres checkpointer
with SqliteSaver.from_conn_string("state_checkpoint.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
    # Fetch historical checkpoint for a thread
    thread_config = {"configurable": {"thread_id": "thread-user-404"}}
    latest_state = app.get_state(thread_config)
    print("Latest State Snapshot:", latest_state.values)
    print("Next Executable Node:", latest_state.next)
    
    # Time-travel: resume from an earlier checkpoint version
    historical_config = {"configurable": {"thread_id": "thread-user-404", "checkpoint_id": "1ef4b..."}}
    app.invoke(None, config=historical_config)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
The integration between LangGraph checkpointers and LangSmith is bi-directional:
* In LangSmith, every checkpoint event is timestamped and indexed alongside the corresponding span.
* Engineers can click **"Fork Thread"** directly in the LangSmith UI, allowing them to test prompt modifications or model parameter changes against real customer historical states without modifying production data.
""")

st.divider()

# ==============================================================================
# SECTION 7: TOOL CALLS
# ==============================================================================
st.header("7. Tool Calls (ToolNode Tracing, Multi-Tool Invocations & ToolMessage Ingestion)", divider="rainbow")

st.markdown("""
In LangGraph, tools are typically managed by a dedicated `ToolNode`.
When a model node emits `tool_calls` in its `AIMessage`, the graph routes execution to the `ToolNode`.
The `ToolNode` executes the requested tools and returns `ToolMessage` instances containing the results.
""")

st.markdown("### ToolNode Observability in LangSmith")
st.markdown("""
* **Parallel Tool Execution**: If an LLM requests three tool calls simultaneously (e.g., three separate search queries), `ToolNode` executes them concurrently. LangSmith captures each tool execution as an independent child span under the parent `ToolNode` run.
* **Argument Telemetry**: LangSmith logs the exact JSON arguments emitted by the model alongside the structured output returned by the tool.
* **ToolMessage Lineage**: Each `ToolMessage` contains a `tool_call_id` that pairs it deterministically with the original `tool_call` from the LLM, enabling LangSmith to trace the request-response correlation.
""")

st.code('''
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def calculate_vat(amount: float, country_code: str) -> float:
    """Calculate VAT rate for financial reconciliation."""
    rates = {"DE": 0.19, "FR": 0.20, "US": 0.08}
    return round(amount * rates.get(country_code, 0.15), 2)

# ToolNode encapsulates execution and automatic LangSmith child span creation
tool_node = ToolNode([calculate_vat])
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production agent architectures:
* Tool errors (e.g., rate limits, invalid keys) do not crash the `ToolNode`.
* The `ToolNode` captures the error string and embeds it inside the `ToolMessage.content`.
* This ensures that the state machine continues to the next node (typically the model), allowing the agent to observe the tool failure and formulate an alternative approach or retry.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: LangGraph Observability & State Machine Tracer (GraphObservabilityEngine)", divider="rainbow")

st.markdown("""
Below is the production telemetry engine that tracks cyclic iterations, node latencies,
channel reducer operations, persistent checkpoints, and edge transitions.
""")

st.code('''
class GraphObservabilityEngine:
    """
    Production LangGraph Observability & State Machine Tracer.
    Simulates distributed state machine execution, cyclic iteration telemetry,
    channel reducer auditing (add_messages), persistent checkpoint versioning,
    and tool call span instrumentation in LangSmith.
    """
    def __init__(self, project_name: str = "langgraph-state-observability"):
        self.project_name = project_name
        self.recorded_runs: List[Dict[str, Any]] = []
        self._checkpoint_counter = 1000

    def execute_graph_workflow(
        self,
        thread_id: str,
        user_query: str,
        max_revisions: int = 2,
        inject_tool_error: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a simulated LangGraph state machine with cyclic critique and revision.
        Captures discrete node execution spans, state snapshots at each checkpoint,
        edge routing decisions, and tool invocation telemetry.
        """
        # Emulates step-by-step state machine execution and checkpointing
        ...
''', language="python")

st.info("The engine is fully instantiated in module scope and powers the interactive state machine inspector below.")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Cyclic State Machine Trace Inspector & Time-Travel Debugger", divider="rainbow")

st.markdown("""
Simulate and inspect a complete LangGraph cyclic agent workflow.
Observe how nodes execute, how channels update via reducers, how persistent checkpoints are saved,
and how LangSmith visualizes the resulting trace hierarchy.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Graph Execution Controls")
    thread_input = st.text_input("Thread ID (Conversation Scope):", value="thread_session_701")
    agent_query = st.text_input("User Research Goal:", value="Assess cybersecurity risks of autonomous AI agents")

    max_cycles = st.slider(
        "Maximum Allowed Revisions (Cycle Bound):",
        min_value=1,
        max_value=3,
        value=2,
        help="Controls the cyclic loop bound before forcing convergence to the Synthesizer node"
    )

    inject_fault = st.toggle(
        "Inject Tool API Error (Cycle 1)",
        value=False,
        help="Simulates an external tool API 503 error to inspect error logging and recovery"
    )

    run_graph_btn = st.button("🚀 Run LangGraph State Machine Workflow", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith State Machine Trace Inspector")

    if run_graph_btn or len(graph_tracer.recorded_runs) == 0:
        run_record = graph_tracer.execute_graph_workflow(
            thread_id=thread_input,
            user_query=agent_query,
            max_revisions=max_cycles,
            inject_tool_error=inject_fault
        )
    else:
        run_record = graph_tracer.recorded_runs[-1]

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Thread ID", run_record["thread_id"])
    m2.metric("Total Latency", f"{run_record['total_latency_ms']:.1f} ms")
    m3.metric("Execution Cycles", run_record["total_iterations"])
    m4.metric("Checkpoints Saved", len(run_record["checkpoints"]))

    # Node Spans Table
    st.markdown("#### Node Execution Spans (LangSmith Waterfall)")
    spans_df = pd.DataFrame(run_record["spans"])
    st.dataframe(spans_df[["step", "node_name", "run_type", "latency_ms", "status"]], use_container_width=True)

    # Edge Transitions
    st.markdown("#### Dynamic Edge Routing Trajectory")
    transitions_df = pd.DataFrame(run_record["transitions"])
    st.dataframe(transitions_df, use_container_width=True)

    # Checkpoint Inspector & Time-Travel
    st.markdown("#### Persistent Checkpoint Store (State Snapshots)")
    selected_step = st.selectbox(
        "Select Checkpoint to Inspect Snapshot:",
        options=[cp["step"] for cp in run_record["checkpoints"]],
        format_func=lambda s: f"Step {s}: Node '{run_record['checkpoints'][s-1]['node']}' (ID: {run_record['checkpoints'][s-1]['checkpoint_id']})"
    )

    cp_target = run_record["checkpoints"][selected_step - 1]
    with st.container(border=True):
        st.markdown(f"**Checkpoint ID**: `{cp_target['checkpoint_id']}` | **Step**: `{cp_target['step']}` | **Active Node**: `{cp_target['node']}`")
        st.markdown("**Channel Diffs at this Checkpoint:**")
        st.json(cp_target["channel_diff"])
        st.markdown("**Complete State Snapshot at this Version:**")
        st.json(cp_target["state_snapshot"])

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does LangSmith represent cyclic graph execution differently from standard linear chains?"):
    st.markdown("""
    **Answer:**
    In a standard linear chain, the LangSmith trace is a static tree where each component appears exactly once.
    In a cyclic LangGraph state machine:
    1. The root run represents the `CompiledStateGraph`.
    2. Nodes that execute multiple times across iterative cycles appear as distinct child spans sequentially ordered by execution timestamp.
    3. Each node span records its iteration index, inputs, and outputs, allowing engineers to visualize how the state evolved over time across loops.
    4. LangSmith correlates all cycles under the same persistent `thread_id`.
    """)

with st.expander("Q2: What is the role of channel reducers in LangGraph, and how do you debug them in LangSmith?"):
    st.markdown("""
    **Answer:**
    Channel reducers determine how node return values update the graph's shared state.
    * The default reducer overwrites existing values.
    * The `add_messages` reducer appends messages and deduplicates or replaces messages that share the same `id`.
    * In LangSmith, engineers debug reducers by inspecting the **State Diff** between successive node spans. If a node accidentally returns an unexpected dictionary structure or an empty list, the state diff highlights the unexpected deletion or overwrite immediately.
    """)

with st.expander("Q3: How do LangGraph checkpointers enable time-travel debugging?"):
    st.markdown("""
    **Answer:**
    Checkpointers (such as `MemorySaver`, `SqliteSaver`, or `PostgresSaver`) save immutable state snapshots at each super-step.
    Each snapshot is indexed by `(thread_id, checkpoint_id)`.
    To time-travel:
    1. Retrieve the list of checkpoint IDs for a given thread via `app.get_state_history(config)`.
    2. Pass an earlier `checkpoint_id` in the `RunnableConfig`.
    3. Re-invoke the graph or modify the state at that version using `app.update_state()`.
    This creates a new branch of execution without corrupting or deleting the previous thread history.
    """)

with st.expander("Q4: What causes a GraphRecursionError, and how do you protect production deployments?"):
    st.markdown("""
    **Answer:**
    A `GraphRecursionError` occurs when a graph exceeds its configured `recursion_limit` (default 25 node transitions).
    This typically happens when a conditional edge gets trapped in an infinite cycle (e.g., Critic constantly rejects drafts, but Researcher fails to improve them).
    **Protection strategies**:
    1. Always implement an explicit iteration counter channel (e.g., `revision_count: int`) in the state.
    2. Include a hard cycle ceiling in the conditional router (`if count >= max_iterations: return 'synthesizer'`).
    3. Monitor step counts in LangSmith alerts to detect diverging agents before users experience timeouts.
    """)

with st.expander("Q5: How does a ToolNode handle tool errors in LangGraph without crashing the entire state machine?"):
    st.markdown("""
    **Answer:**
    Prebuilt `ToolNode` instances wrap tool execution in try-except logic. When a tool raises an exception:
    1. The `ToolNode` catches the error.
    2. It creates a `ToolMessage` with `status="error"` and the exception traceback as its text content.
    3. The `ToolMessage` is appended to the message channel via `add_messages`.
    4. The state machine transitions back to the model node, enabling the LLM to inspect the failure and decide whether to retry or choose an alternative tool.
    """)

st.markdown("### Production Traps & Failure Modes")

lg_tr_col1, lg_tr_col2 = st.columns(2)

with lg_tr_col1:
    st.error("""
    **Trap 1: Unbounded State Growth from add_messages**
    * **Failure**: Accumulating hundreds of raw tool responses and LLM messages in the state across long-running sessions without pruning.
    * **Consequence**: Prompts quickly blow through model context windows, resulting in exponential token costs and eventual API context errors.
    * **Remedy**: Use `trim_messages` or insert a message pruning node that filters out historical tool payloads before calling the LLM.
    """)

    st.error("""
    **Trap 2: Missing thread_id in Production Invocations**
    * **Failure**: Invoking a compiled graph with an active checkpointer but omitting `{"configurable": {"thread_id": "..."}}` in the config.
    * **Consequence**: The checkpointer cannot partition state, raising a `ValueError` or failing to persist conversation memory across requests.
    * **Remedy**: Enforce strict validation that every incoming API request provides a verified, authenticated `thread_id`.
    """)

with lg_tr_col2:
    st.warning("""
    **Trap 3: Mutable State In-Place Modification**
    * **Failure**: Modifying state dictionaries in-place inside node functions (e.g., `state["messages"].append(msg)`) instead of returning a new update dictionary.
    * **Consequence**: Bypasses channel reducers, corrupts checkpointer serialization, and prevents LangSmith from computing accurate state diffs.
    * **Remedy**: Treat state as strictly immutable; always return a fresh update dictionary (e.g., `return {"messages": [msg]}`).
    """)

    st.warning("""
    **Trap 4: Router Functions with Side Effects**
    * **Failure**: Placing LLM calls, database queries, or state modifications inside conditional edge routing functions.
    * **Consequence**: Conditional edge functions are meant to be pure deterministic evaluators. Placing side effects inside them breaks time-travel replay and creates un-traced child executions.
    * **Remedy**: Keep conditional routers pure; perform all computation and LLM calls inside dedicated nodes before routing.
    """)

st.markdown("### Core Technical Flashcards")

fclg_c1, fclg_c2 = st.columns(2)

with fclg_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary difference between a DAG and a LangGraph StateGraph?")
        st.markdown("**Back**: A DAG is unidirectional and acyclic (no loops allowed), whereas a LangGraph StateGraph natively supports cyclic loops, conditional branching, and persistent state checkpointing.")

    with fclg_c2:
        st.markdown("**Front**: What does the `add_messages` reducer do when two messages share the same `id`?")
        st.markdown("**Back**: It updates and overwrites the existing message with the new message in-place rather than appending a duplicate, enabling real-time streaming updates and edits.")

with fclg_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does LangGraph enforce safety against runaway agent loops?")
        st.markdown("**Back**: Via the `recursion_limit` parameter in `RunnableConfig` (default 25). If total node transitions reach this threshold, execution halts with a `GraphRecursionError`.")

    with fclg_c2:
        st.markdown("**Front**: What are the three primary components of a LangGraph Checkpoint?")
        st.markdown("**Back**: 1. `thread_id` (session scope), 2. `checkpoint_id` (version identifier), and 3. `state snapshot` (the serialized channel values at that exact super-step).")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LangGraph observability and state machine tracing:")

lg_chk1 = st.checkbox("I understand how LangSmith traces cyclic graph runs under a unified root run and thread_id.", value=True)
lg_chk2 = st.checkbox("I can isolate individual node latencies, input states, and output updates in LangSmith traces.", value=True)
lg_chk3 = st.checkbox("I understand how channel reducers (add_messages) work and how to audit state diffs.", value=True)
lg_chk4 = st.checkbox("I can debug conditional edge routing decisions and prevent runaway recursion_limit errors.", value=True)
lg_chk5 = st.checkbox("I know how persistent checkpointers enable fault tolerance, time-travel, and state rewinds.", value=True)
lg_chk6 = st.checkbox("I understand how ToolNode captures tool execution errors as ToolMessages without crashing the graph.", value=True)

if lg_chk1 and lg_chk2 and lg_chk3 and lg_chk4 and lg_chk5 and lg_chk6:
    st.success("Mastery Certified: You have mastered LangGraph observability, cyclic state machine tracing, and checkpoint auditing in LangSmith.")
