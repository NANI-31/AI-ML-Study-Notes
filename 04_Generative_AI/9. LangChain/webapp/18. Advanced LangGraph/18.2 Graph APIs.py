import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockGraphAPIsStudio:
    """
    Simulates advanced LangGraph APIs:
    Graph API (StateGraph), Functional API (@entrypoint/@task),
    Send API (Dynamic Map-Reduce), Command API (Combined Update & Route), and Subgraphs.
    """
    def __init__(self):
        pass

    def run_api_demo(self, selected_api: str, payload: str) -> Dict[str, Any]:
        execution_trace = []

        if "Functional API" in selected_api:
            execution_trace.append({"step": "@entrypoint main_workflow", "action": f"Started workflow with input: '{payload}'"})
            execution_trace.append({"step": "@task step_one.invoke()", "action": "Executed task 'validate_schema' in background thread", "result": "OK"})
            execution_trace.append({"step": "@task step_two.invoke()", "action": "Executed task 'enrich_metadata' with cached dependencies", "result": "ENRICHED"})
            execution_trace.append({"step": "@entrypoint return", "action": "Yielded final payload without StateGraph builder"})
            summary = "Functional API executed imperatively using standard Python control flow."

        elif "Command API" in selected_api:
            execution_trace.append({"step": "Node 'triage_node'", "action": f"Evaluated payload: '{payload}'"})
            execution_trace.append({"step": "Command() Emitted", "action": "Node returned Command(update={'priority': 'HIGH'}, goto='security_review')"})
            execution_trace.append({"step": "Direct Routing", "action": "Execution jumped directly to 'security_review' bypassing conditional edge table"})
            execution_trace.append({"step": "Node 'security_review'", "action": "Executed security validation and routed to END"})
            summary = "Command API combined state mutation and control flow routing in a single return statement."

        elif "Send API" in selected_api:
            partitions = [f"{payload} [Part 1]", f"{payload} [Part 2]", f"{payload} [Part 3]"]
            execution_trace.append({"step": "Dispatcher Node", "action": f"Returned {len(partitions)} Send() objects"})
            for idx, p in enumerate(partitions, 1):
                execution_trace.append({"step": f"Worker #{idx}", "action": f"Processed partition '{p}' in parallel superstep"})
            execution_trace.append({"step": "Reducer Node", "action": "Aggregated 3 worker outputs using operator.add channel reducer"})
            summary = "Send API dispatched dynamic parallel worker instances across data partitions."

        elif "Subgraphs" in selected_api:
            execution_trace.append({"step": "Parent Graph START", "action": "Parent graph initiated execution"})
            execution_trace.append({"step": "Invoke Subgraph 'ResearchTeam'", "action": "Parent delegated task to encapsulated ResearchTeam subgraph"})
            execution_trace.append({"step": "Subgraph Internal Superstep 1", "action": "Subgraph WebResearcher gathered external documentation"})
            execution_trace.append({"step": "Subgraph Internal Superstep 2", "action": "Subgraph Synthesizer compiled report"})
            execution_trace.append({"step": "Parent Graph Resumed", "action": "Parent received final subgraph output into parent state channel"})
            summary = "Hierarchical Subgraph executed with isolated internal state before returning to parent graph."

        else: # Standard Graph API (StateGraph)
            execution_trace.append({"step": "builder.add_node()", "action": "Registered nodes and edges"})
            execution_trace.append({"step": "builder.compile()", "action": "Compiled StateGraph with memory checkpointer"})
            execution_trace.append({"step": "app.invoke()", "action": f"Executed declarative state machine on input '{payload}'"})
            summary = "Standard StateGraph executed across discrete nodes and supersteps."

        return {
            "api": selected_api,
            "trace": execution_trace,
            "summary": summary
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="18.2 Graph APIs — LangChain Course",
    page_icon="🔌",
    layout="wide"
)

st.title("18.2 Graph APIs")
st.caption("Mastering Modern LangGraph APIs: StateGraph, Functional API (@entrypoint/@task), Send API, Command API, and Hierarchical Subgraphs")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Modern LangGraph API Surface & Paradigm Shifts", divider="rainbow")
st.markdown("""
LangGraph provides two complementary programming paradigms for building stateful agent workflows:
1. **Graph API (`StateGraph`)**: Declarative state machines with explicit nodes, edges, and channel reducers.
2. **Functional API (`@entrypoint` and `@task`)**: Imperative Python workflows using standard if-statements, for-loops, 
   and functions with automatic task checkpointing.

Alongside these paradigms, the **`Command` API** and **`Send` API** unify state mutation with dynamic control-flow routing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GraphAPIsArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_declarative {
            label = "Paradigm A: Declarative Graph API";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            sg_box [label="StateGraph(StateSchema)\\nbuilder.add_node()\\nbuilder.add_edge()\\nbuilder.compile()", fillcolor="#0F172A", color="#38BDF8"];
            subgraph_box [label="Subgraphs\\nEncapsulated child graphs\\nIsolated state channels", fillcolor="#1E293B", color="#64748B"];
        }

        subgraph cluster_functional {
            label = "Paradigm B: Imperative Functional API";
            style = "rounded";
            color = "#D97706";
            fontcolor = "#FBBF24";
            entry_box [label="@entrypoint main_workflow\\nOrchestrates flow via native Python if/for", fillcolor="#78350F", color="#F59E0B"];
            task_box [label="@task subtask_fn\\nAuto-checkpointed task execution unit", fillcolor="#78350F", color="#F59E0B"];
        }

        subgraph cluster_primitives {
            label = "Dynamic Control Flow Primitives";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            cmd_box [label="Command API\\nreturn Command(update=..., goto=...)\\nCombines state update + routing", fillcolor="#1E293B", color="#10B981"];
            send_box [label="Send API\\nreturn [Send('worker', item) for item in list]\\nDynamic Map-Reduce dispatch", fillcolor="#1E293B", color="#10B981"];
        }

        subgraph cluster_runtime {
            label = "Unified Pregel Execution Engine";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            pregel_core [label="Checkpointed Pregel Engine\\nThread isolation, time-travel, durable resume", fillcolor="#064E3B", color="#10B981"];
        }

        sg_box -> subgraph_box [label="embeds as node"];
        subgraph_box -> pregel_core;
        
        entry_box -> task_box [label="calls .invoke()"];
        task_box -> pregel_core;
        
        cmd_box -> pregel_core;
        send_box -> pregel_core;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: GRAPH API ---
st.header("2. The Graph API (`StateGraph`)", divider="rainbow")
st.markdown("""
The **Graph API** is the declarative foundation of LangGraph. You define an explicit state schema, add discrete node 
functions, and wire direct and conditional edges before compiling into a runnable application.
""")

st.code("""
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class MyState(TypedDict):
    query: str
    result: str

def process_node(state: MyState):
    return {"result": state["query"].upper()}

builder = StateGraph(MyState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()
""", language="python")

st.divider()

# --- SECTION 3: FUNCTIONAL API ---
st.header("3. The Functional API (`@entrypoint` and `@task`)", divider="rainbow")
st.markdown("""
Introduced in LangGraph v0.2+, the **Functional API** allows developers to build complex workflows without defining 
explicit `StateGraph` builders, schemas, or edges. Workflows are written as standard Python functions decorated 
with `@entrypoint` and `@task`.

### Core Benefits:
- **No Boilerplate Schema**: Write normal Python function arguments and return types.
- **Native Control Flow**: Use standard Python `if/else`, `try/except`, and `for-loops`.
- **Automatic Checkpointing**: Each `@task` is checkpointed; if a step fails or interrupts, only un-executed tasks rerun.
""")

st.code("""
from langgraph.func import entrypoint, task

@task
def scrape_web(query: str) -> str:
    # Checkpointed task execution unit
    return f"Web findings for {query}"

@task
def analyze_findings(data: str) -> str:
    return f"Analysis summary of: {data}"

@entrypoint()
def research_pipeline(query: str):
    # Standard Python imperative orchestration!
    raw_data = scrape_web(query).result()
    report = analyze_findings(raw_data).result()
    return report

# Invocation
# output = research_pipeline.invoke("Quantum Computing 2026")
""", language="python")

st.divider()

# --- SECTION 4: SEND API ---
st.header("4. The Send API: Dynamic Map-Reduce Fan-Out", divider="rainbow")
st.markdown("""
The **`Send` API** decouples edge routing from static topology. When a node processes a list of items with variable length, 
it can dynamically dispatch a separate parallel worker node instance for each item.
""")

st.code("""
from langgraph.constants import Send

def dispatch_worker_tasks(state: CorpusState):
    # Returns a list of Send objects: Send(target_node_name, item_payload)
    return [
        Send("summarize_chunk", {"chunk_id": c.id, "text": c.text})
        for c in state["document_chunks"]
    ]

builder.add_conditional_edges("splitter_node", dispatch_worker_tasks, ["summarize_chunk"])
""", language="python")

st.divider()

# --- SECTION 5: COMMAND API ---
st.header("5. The Command API: Combining State Updates & Routing", divider="rainbow")
st.markdown("""
In traditional LangGraph, a node returns a state update dictionary, and a separate conditional edge router decides 
where to transition next. 

The **`Command` API** eliminates this separation: a node can return a `Command` object that **simultaneously updates state 
and specifies the destination node (`goto`)**.
""")

st.code("""
from langgraph.types import Command
from langgraph.graph import END

def triage_incident(state: SecurityState) -> Command[Literal["sec_ops", "dev_ops", END]]:
    if state["severity"] == "P1":
        # Simultaneously updates state and jumps directly to sec_ops node!
        return Command(
            update={"audit_trail": ["P1 Severity: Escalated to SecOps"]},
            goto="sec_ops"
        )
    return Command(
        update={"audit_trail": ["Standard triage: resolved"]},
        goto=END
    )

builder.add_node("triage", triage_incident)
# No separate conditional edge required!
""", language="python")

st.divider()

# --- SECTION 6: STATE UPDATES ---
st.header("6. State Updates & Dynamic Mutation (`update_state()`)", divider="rainbow")
st.markdown("""
Outside the normal graph execution loop, applications often need to inspect or modify a thread's state 
(e.g., when a human reviews a draft, edits prompt instructions, or rolls back state).

The **`app.update_state(config, values)`** method allows external callers to inject updates into an active thread checkpoint.
""")

st.code("""
# External thread state modification
config = {"configurable": {"thread_id": "ticket_104"}}

# Overwrite or append state manually
app.update_state(
    config,
    {"analyst_review": "Approved by SecLead with MFA bypass exemption."},
    as_node="human_review_node"  # Injects update as if emitted by human_review_node
)

# Resume execution from updated checkpoint
# app.invoke(None, config=config)
""", language="python")

st.divider()

# --- SECTION 7: SUBGRAPHS ---
st.header("7. Subgraphs: Hierarchical Multi-Agent Encapsulation", divider="rainbow")
st.markdown("""
Large enterprise applications shouldn't dump 50 nodes into a single monolithic graph. **Subgraphs** enable 
hierarchical encapsulation:
- Each subgraph defines its own `StateGraph` with a private, isolated state schema.
- The parent graph registers the compiled subgraph as a single standard node: `parent_builder.add_node("research_team", compiled_subgraph)`.
- Promotes modularity, independent unit testing, and team boundaries.
""")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Multi-Agent Subgraph System with Command & Send APIs", divider="rainbow")
st.markdown("""
Below is an enterprise-grade Python implementation composing subgraphs, the `Command` API, and the `Send` API.
""")

st.code("""
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.constants import Send

# 1. Subgraph: Specialized Analytics Team
class SubState(TypedDict):
    data_points: List[float]
    average: float

def compute_avg(state: SubState):
    avg = sum(state["data_points"]) / max(1, len(state["data_points"]))
    return {"average": avg}

sub_builder = StateGraph(SubState)
sub_builder.add_node("calc", compute_avg)
sub_builder.add_edge(START, "calc")
sub_builder.add_edge("calc", END)
analytics_subgraph = sub_builder.compile()

# 2. Parent Graph State
class ParentState(TypedDict):
    user_query: str
    metrics: Annotated[List[float], operator.add]
    summary_report: str

# 3. Parent Node utilizing Command API
def route_with_command(state: ParentState) -> Command[Literal["analytics_team", END]]:
    if "metrics" in state["user_query"].lower():
        return Command(
            update={"metrics": [10.5, 20.0, 30.5]},
            goto="analytics_team"
        )
    return Command(
        update={"summary_report": "Direct answer: Zero metrics needed."},
        goto=END
    )

parent_builder = StateGraph(ParentState)
parent_builder.add_node("router", route_with_command)
# Embed the Subgraph as a single node!
parent_builder.add_node("analytics_team", analytics_subgraph)

parent_builder.add_edge(START, "router")
parent_builder.add_edge("analytics_team", END)

parent_app = parent_builder.compile()
""", language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Graph APIs & Control Flow Studio", divider="rainbow")
st.markdown("""
Test and compare the modern LangGraph API surface. Select an API paradigm to observe how 
**Functional API (@entrypoint)**, **Command API**, **Send API**, and **Subgraphs** handle execution and state updates.
""")

# Instantiate module-scope studio
apis_studio = MockGraphAPIsStudio()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    api_choice = st.selectbox(
        "Select LangGraph API Paradigm to Test:",
        [
            "Command API (Combined State Update & Routing)",
            "Functional API (@entrypoint and @task)",
            "Send API (Dynamic Map-Reduce Fan-Out)",
            "Subgraphs (Hierarchical Multi-Agent Teams)",
            "Standard Graph API (StateGraph Builder)"
        ]
    )

    sample_payload = st.text_input(
        "Input Payload:",
        value="Evaluate infrastructure access permissions for Aurora cluster"
    )

    run_api_btn = st.button("⚡ Execute API Pipeline", type="primary", use_container_width=True)

with wb_c2:
    st.subheader("API Execution Trace")

    if run_api_btn or sample_payload:
        api_res = apis_studio.run_api_demo(api_choice, sample_payload)

        st.markdown(f"### 🏁 Summary: `{api_res['summary']}`")
        st.markdown("### 📋 Superstep Trace")
        for t in api_res["trace"]:
            with st.expander(f"Step: {t['step']}", expanded=True):
                st.write(f"**Action**: {t['action']}")
                if "result" in t:
                    st.caption(f"**Result**: `{t['result']}`")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What are the key advantages of the LangGraph Functional API over the Graph API?",
     "The Functional API (@entrypoint and @task) enables developers to write agent workflows in an imperative, idiomatic Python style without declaring explicit TypedDict schemas or GraphBuilder objects. Control flow uses standard if/else branching and for-loops. Furthermore, @task provides automatic checkpoint caching: if a downstream task fails, rerunning the entrypoint skips previously completed tasks without re-executing expensive LLM calls."),
    ("How does the `Command` API simplify graph topologies compared to standard conditional edges?",
     "In standard LangGraph, dynamic transitions require two elements: a node function returning a state dictionary and a separate conditional edge routing function that inspects the state. The Command API unifies both: a node returns `Command(update={...}, goto='target_node')`, eliminating the need for boilerplate conditional edge definitions and making node behavior self-contained."),
    ("When should an engineering team decompose an agent into Subgraphs?",
     "Use Subgraphs when: (1) A specialized group of nodes has its own private state schema that shouldn't pollute the parent state, (2) Different teams are developing distinct agent capabilities independently (e.g. Research Team vs Coding Team), or (3) Reusable agent subroutines need to be imported across multiple distinct parent graphs."),
    ("How does `app.update_state()` enable Human-in-the-Loop workflows?",
     "When an agent hits an `interrupt_before` breakpoint, execution halts and saves a checkpoint. `app.update_state(config, updates, as_node='human_review')` allows a human reviewer to inject edits, approve tool arguments, or provide clarifying instructions directly into the checkpoint database before calling `app.invoke(None, config)` to resume execution."),
    ("Can a node return multiple `Command` objects?",
     "No. A node returns a single `Command` object specifying updates and a single `goto` destination (or a list of destinations for parallel fan-out). For dynamic variable-length fan-out across multiple tasks with separate custom payloads, use the `Send` API instead.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using `goto` in Command API Pointing to Unregistered Node. If a node returns `Command(goto='billing')` and 'billing' was never added to the graph or registered in the node's type annotation, LangGraph raises a GraphConfigError at compile time or a runtime RoutingError.")
    st.error("**Trap 2 (Breaking Error)**: Calling `.result()` on `@task` Inside Standard Python Code. In the Functional API, calling `.result()` on a task outside an `@entrypoint` context raises a TaskRuntimeError. Tasks must always be orchestrated from within an entrypoint function.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Subgraph State Leaks into Parent State. If a subgraph returns keys that match parent state keys without channel reducers, the subgraph will silently overwrite parent state. Always explicitly map parent-to-child state transformations.")
    st.warning("**Trap 4 (Quality Warning)**: Manually Updating State Without `as_node` Context. When using `app.update_state()`, failing to pass `as_node` can cause downstream conditional edges to evaluate against undefined node origins.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Command API"):
        st.markdown("**Combines state delta updates and next node routing in a single return statement.**")
    with st.expander("🎴 Flashcard 2: Functional API"):
        st.markdown("**Imperative workflow paradigm using @entrypoint and @task with automatic task checkpointing.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Subgraphs"):
        st.markdown("**Encapsulated child state machines embedded as modular nodes inside parent workflows.**")
    with st.expander("🎴 Flashcard 4: update_state()"):
        st.markdown("**Method allowing external human or API modification of active thread checkpoints.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between the Graph API and the Functional API.", value=True)
chk2 = st.checkbox("I know how the Command API unifies state updates and control flow routing.", value=True)
chk3 = st.checkbox("I understand how the Send API enables dynamic map-reduce task fan-out.", value=True)
chk4 = st.checkbox("I can implement hierarchical multi-agent architectures using isolated Subgraphs.", value=True)
chk5 = st.checkbox("I know how to use update_state() to inject human feedback into persistent checkpoints.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 graph APIs technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 18.2 Graph APIs!")
