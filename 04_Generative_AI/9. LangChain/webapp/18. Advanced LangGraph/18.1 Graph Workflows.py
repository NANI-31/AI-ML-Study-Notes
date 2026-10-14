import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockGraphWorkflowsStudio:
    """
    Simulates advanced LangGraph workflows:
    Sequential, Branching, Reflection Loops, Conditional Routing, Parallel, and Map-Reduce.
    """
    def __init__(self):
        pass

    def execute_workflow(self, workflow_type: str, input_items: List[str]) -> Dict[str, Any]:
        trace = []

        if workflow_type == "Sequential":
            trace.append({"step": "1. Ingest", "action": "Ingested raw payload", "items": len(input_items)})
            trace.append({"step": "2. Transform", "action": "Formatted strings into uppercase schema", "items": len(input_items)})
            trace.append({"step": "3. Output", "action": "Generated final deterministic response", "items": len(input_items)})
            result = f"Sequential pipeline processed {len(input_items)} records linearly."

        elif workflow_type == "Reflection / Self-Correction Loop":
            trace.append({"step": "1. Draft Code", "action": "LLM generated initial SQL query", "status": "drafted"})
            trace.append({"step": "2. Critic Node", "action": "Linter detected syntax vulnerability (Unescaped quote)", "status": "critiqued"})
            trace.append({"step": "3. Refinement Loop", "action": "Looping back to draft node with linter feedback", "status": "looped"})
            trace.append({"step": "4. Final Draft", "action": "LLM produced parameterized query; tests passed", "status": "verified"})
            result = "Reflection loop completed in 2 iterations with automated self-correction."

        elif workflow_type == "Parallel Fan-Out & Join":
            trace.append({"step": "1. Dispatch", "action": "Dispatched tasks to 2 parallel workers", "status": "dispatched"})
            trace.append({"step": "2. Worker Security", "action": "Scanned authorization boundaries in parallel", "latency_ms": 12.4})
            trace.append({"step": "2. Worker Performance", "action": "Benchmarked IOPS latency in parallel", "latency_ms": 11.8})
            trace.append({"step": "3. Join Node", "action": "Aggregated worker results into single report", "status": "merged"})
            result = "Parallel workers executed concurrently in a single superstep and merged at join node."

        else: # Map-Reduce
            trace.append({"step": "1. Map Dispatch (Send API)", "action": f"Dispatched {len(input_items)} independent map workers", "status": "mapped"})
            for idx, it in enumerate(input_items, 1):
                trace.append({"step": f"Map Worker #{idx}", "action": f"Summarized document partition '{it}'", "score": 0.92})
            trace.append({"step": "2. Reduce Aggregator", "action": "Synthesized mapped chunks into executive summary", "status": "reduced"})
            result = f"Map-Reduce aggregated {len(input_items)} document partitions via dynamic Send API."

        return {
            "workflow_type": workflow_type,
            "trace": trace,
            "result": result
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="18.1 Graph Workflows — LangChain Course",
    page_icon="🔄",
    layout="wide"
)

st.title("18.1 Graph Workflows")
st.caption("Advanced Graph Design Patterns: Sequential, Branching, Reflection Loops, Conditional, Parallel, and Map-Reduce Workflows")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: Advanced Workflow Topologies & Design Patterns", divider="rainbow")
st.markdown("""
Modern agent systems extend beyond simple reactive chains. LangGraph allows developers to model 
sophisticated computational topologies:
- **Sequential**: Linear pipelines with deterministic order.
- **Branching**: Static multi-channel pathways.
- **Loops**: Iterative reflection, critique, and self-correction cycles.
- **Conditional**: Dynamic runtime routing based on model evaluation.
- **Parallel**: Simultaneous execution of independent nodes within a single superstep.
- **Map-Reduce**: Dynamic fan-out across variable-length lists using the `Send` API followed by reduction.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GraphWorkflowsArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_loop {
            label = "Pattern A: Reflection & Self-Correction Loop";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            draft_node [label="Generator Node\\nWrites initial code / answer", fillcolor="#0F172A", color="#38BDF8"];
            critique_node [label="Critic Node (Tests / Linter)\\nEvaluates correctness & generates feedback", fillcolor="#0F172A", color="#38BDF8"];
            loop_edge [label="Conditional Routing\\nPass -> END | Fail -> Loop Back", fillcolor="#D97706", color="#F59E0B"];
        }

        subgraph cluster_mapreduce {
            label = "Pattern B: Map-Reduce via Send API";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            dispatcher [label="Dispatcher Node\\nSplits corpus into N items", fillcolor="#1E293B", color="#64748B"];
            map_1 [label="Map Worker 1", fillcolor="#0F172A", color="#38BDF8"];
            map_2 [label="Map Worker 2", fillcolor="#0F172A", color="#38BDF8"];
            map_n [label="Map Worker N", fillcolor="#0F172A", color="#38BDF8"];
            reducer [label="Reduce Aggregator Node\\nMerges mapped summaries", fillcolor="#064E3B", color="#10B981"];
        }

        draft_node -> critique_node [label="draft output"];
        critique_node -> loop_edge;
        loop_edge -> draft_node [label="Iterative Refinement Loop", color="#38BDF8", style="bold"];

        dispatcher -> map_1 [label="Send(worker, item_1)"];
        dispatcher -> map_2 [label="Send(worker, item_2)"];
        dispatcher -> map_n [label="Send(worker, item_n)"];
        
        map_1 -> reducer [label="delta"];
        map_2 -> reducer [label="delta"];
        map_n -> reducer [label="delta"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: SEQUENTIAL WORKFLOWS ---
st.header("2. Sequential Workflows", divider="rainbow")
st.markdown("""
A **Sequential Workflow** executes nodes in a fixed, linear order. Each node receives the state produced by its 
predecessor, computes an update, and unconditionally yields to the next node.
""")

st.code("""
# Sequential pipeline: A -> B -> C -> END
builder.add_edge(START, "data_cleaner")
builder.add_edge("data_cleaner", "feature_extractor")
builder.add_edge("feature_extractor", "classifier")
builder.add_edge("classifier", END)
""", language="python")

st.divider()

# --- SECTION 3: BRANCHING WORKFLOWS ---
st.header("3. Branching Workflows", divider="rainbow")
st.markdown("""
A **Branching Workflow** splits execution into multiple independent pathways based on static configuration or 
domain rules (e.g. routing support inquiries to Billing, Tech Support, or Security).
""")

st.code("""
# Branching from intake to specialized department nodes
builder.add_edge("intake", "billing_node")
builder.add_edge("intake", "technical_node")
# Both branches execute concurrently in the next superstep
""", language="python")

st.divider()

# --- SECTION 4: LOOPS ---
st.header("4. Loops: Iterative Refinement & Reflection", divider="rainbow")
st.markdown("""
**Cycles and Loops** are the hallmark of advanced agency. In a reflection loop:
1. An **actor node** produces an initial output (e.g., Python code or a SQL query).
2. A **critic node** validates the output against a linter or unit tests.
3. If errors are found, execution cycles back to the actor node with critique feedback.
4. If tests pass, execution exits to `END`.
""")

st.code("""
def route_reflection(state: CodingState):
    if state["tests_passed"] or state["iteration"] >= 3:
        return END
    return "coder_node"  # Loop back!

builder.add_conditional_edges("critic_node", route_reflection, ["coder_node", END])
builder.add_edge("coder_node", "critic_node")
""", language="python")

st.divider()

# --- SECTION 5: CONDITIONAL WORKFLOWS ---
st.header("5. Conditional Workflows", divider="rainbow")
st.markdown("""
**Conditional Workflows** use dynamic router callables that inspect current state attributes (e.g. confidence scores, 
sentiment, or tool calls) to select the next destination node at runtime.
""")

st.code("""
def route_by_confidence(state: TriageState) -> str:
    if state["confidence"] < 0.60:
        return "human_review"
    elif state["category"] == "urgent":
        return "pagerduty_alert"
    return "standard_archive"

builder.add_conditional_edges("evaluator", route_by_confidence)
""", language="python")

st.divider()

# --- SECTION 6: PARALLEL WORKFLOWS ---
st.header("6. Parallel Workflows & Channel Synchronization", divider="rainbow")
st.markdown("""
In a **Parallel Workflow**, multiple nodes execute simultaneously within a single superstep. 
To prevent update conflicts, all shared state keys written by parallel nodes must be protected by **channel reducers**.
""")

st.code("""
# Parallel Fan-Out: START triggers both scanner nodes concurrently
builder.add_edge(START, "sec_scanner")
builder.add_edge(START, "perf_scanner")

# Fan-In: Both scanners converge on aggregator before terminating
builder.add_edge("sec_scanner", "aggregator")
builder.add_edge("perf_scanner", "aggregator")
builder.add_edge("aggregator", END)
""", language="python")

st.divider()

# --- SECTION 7: MAP-REDUCE WORKFLOWS ---
st.header("7. Map-Reduce Workflows & the Send API", divider="rainbow")
st.markdown("""
When processing a variable-length list of items (e.g. 50 documents in a corpus), static parallel edges are insufficient 
because the number of workers is dynamic.

The **`Send` API** enables dynamic map-reduce:
1. A dispatcher node inspects a list of N items and returns `[Send("worker_node", {"item": it}) for it in items]`.
2. LangGraph dynamically spawns N parallel worker node instances.
3. Each worker node returns its result, which is merged by an `operator.add` reducer into the parent state.
4. A **reducer node** aggregates the mapped outputs.
""")

st.code("""
from langgraph.constants import Send

def continue_to_subtasks(state: DocumentState):
    # Dynamically spawn one summarize_chunk worker per document chunk
    return [
        Send("summarize_chunk", {"chunk_text": chunk})
        for chunk in state["document_chunks"]
    ]

builder.add_conditional_edges("split_node", continue_to_subtasks, ["summarize_chunk"])
builder.add_edge("summarize_chunk", "reduce_summaries")
builder.add_edge("reduce_summaries", END)
""", language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Map-Reduce & Self-Correction Pipeline", divider="rainbow")
st.markdown("""
The production script below demonstrates how to construct a robust Map-Reduce summarization pipeline 
using the `Send` API and channel reducers.
""")

st.code("""
from typing import Annotated, List
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

# 1. State Schemas
class OverallState(TypedDict):
    topics: List[str]
    summaries: Annotated[List[str], operator.add]
    final_report: str

class WorkerState(TypedDict):
    topic: str

# 2. Node Functions
def dispatch_topics(state: OverallState):
    # Returns dynamic Send objects for each topic in state
    return [Send("research_worker", {"topic": t}) for t in state["topics"]]

def research_worker(state: WorkerState):
    # Runs in parallel per topic
    return {"summaries": [f"Research findings for topic: {state['topic']}"]}

def reducer_node(state: OverallState):
    combined = " \\n".join(state["summaries"])
    return {"final_report": f"Executive Synthesis:\\n{combined}"}

# 3. Graph Assembly
builder = StateGraph(OverallState)
builder.add_node("research_worker", research_worker)
builder.add_node("reducer_node", reducer_node)

builder.add_conditional_edges(START, dispatch_topics, ["research_worker"])
builder.add_edge("research_worker", "reducer_node")
builder.add_edge("reducer_node", END)

app = builder.compile()

# Example Invocation:
# output = app.invoke({"topics": ["Kubernetes", "Redis", "Aurora"]})
""", language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Advanced Workflow Topology Studio", divider="rainbow")
st.markdown("""
Simulate how different graph workflows execute. Compare **Sequential**, **Reflection Loops**, 
**Parallel Fan-Out & Join**, and **Map-Reduce (Send API)** to observe superstep progression and state resolution.
""")

# Instantiate module-scope studio
workflows_studio = MockGraphWorkflowsStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    wf_choice = st.selectbox(
        "Select Graph Workflow Topology Pattern:",
        [
            "Map-Reduce (Dynamic Send API)",
            "Reflection / Self-Correction Loop",
            "Parallel Fan-Out & Join",
            "Sequential"
        ]
    )

    items_input = st.multiselect(
        "Sample Items for Processing:",
        options=["Aurora DB Cluster", "Redis Rate Limiter", "Vault mTLS PKI", "Kubernetes Ingress"],
        default=["Aurora DB Cluster", "Redis Rate Limiter", "Vault mTLS PKI"]
    )

    run_wf_btn = st.button("🚀 Run Workflow Execution", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Workflow Execution Telemetry")

    if run_wf_btn or wf_choice:
        wf_res = workflows_studio.execute_workflow(wf_choice, items_input)

        st.markdown(f"### 🏁 Result: `{wf_res['result']}`")
        st.markdown("### 🔄 Superstep Transition Trace")
        for t in wf_res["trace"]:
            with st.expander(f"Phase: {t['step']}", expanded=True):
                st.write(f"**Action**: {t['action']}")
                st.json(t)

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does the LangGraph `Send` API differ from standard conditional edges?",
     "A standard conditional edge routes execution to a statically known node name based on state. The `Send` API allows dynamic map-reduce fan-out: a conditional edge can return a variable-length list of `Send('target_node', target_state)` objects. If the state contains 10 items, it dispatches 10 parallel worker nodes with customized individual payloads, which is impossible with static edge declarations."),
    ("Why is a recursion limit required when implementing reflection and self-correction loops?",
     "Reflection loops cycle back from critic to generator nodes. If the model is unable to satisfy validation criteria due to a persistent bug or flawed instruction, the loop will run indefinitely without a termination constraint. Configuring `recursion_limit` (e.g. 15 or 25 supersteps) or tracking `iteration` in state ensures the graph safely halts and raises a `GraphRecursionError`."),
    ("What prevents race conditions when multiple parallel nodes write to the same state channel?",
     "LangGraph implements the Pregel model where nodes in a superstep execute in parallel without reading each other's live writes. At the end of the superstep, all emitted state updates are combined using that channel's designated reducer function (such as `operator.add`). If two parallel nodes write to an un-reduced key, LangGraph raises an `InvalidUpdateError`."),
    ("What is the difference between a fan-in join node and standard sequential transitions?",
     "In a sequential transition, Node B runs immediately after Node A completes. In a fan-in join, multiple upstream nodes (e.g. Worker 1 and Worker 2) must both complete their execution in the current superstep before the join node is scheduled in the next superstep, ensuring synchronized aggregation."),
    ("Can subgraphs be embedded as nodes inside parent workflows?",
     "Yes. A compiled `StateGraph` implements the standard Runnable protocol and can be added as a node inside a parent graph: `parent_builder.add_node('subteam', compiled_subgraph)`. This enables hierarchical multi-agent architectures where teams of agents operate with local isolated state.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Writing to Un-reduced State from `Send` API Map Workers. In map-reduce workflows, all map workers return outputs in the same superstep. If their output key lacks an `operator.add` reducer, an InvalidUpdateError will crash the graph. Always use `Annotated[list, operator.add]` for map-reduce outputs.")
    st.error("**Trap 2 (Breaking Error)**: Modifying State Inside the `Send` Dispatcher Function. The router function returning `[Send(...)]` must be pure and read-only. Attempting to mutate state inside the routing callable creates un-tracked state divergence.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Unbounded Reflection Loops Without Critic Guidance. Looping back to the generator node with generic feedback like 'Fix errors' often causes the model to repeat identical mistakes. Always pass explicit error stack traces or linter outputs in the feedback channel.")
    st.warning("**Trap 4 (Quality Warning)**: Over-partitioning in Map-Reduce Workflows. Spawning hundreds of parallel `Send` calls on small text snippets incurs massive LLM API concurrency overhead and rate limits. Batch items into balanced chunks before dispatching.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Reflection Loop"):
        st.markdown("**Iterative refinement cycle between generator and critic nodes with automated error recovery.**")
    with st.expander("🎴 Flashcard 2: Send API"):
        st.markdown("**Primitive enabling dynamic Map-Reduce fan-out across variable-length item collections.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Fan-In Join"):
        st.markdown("**Synchronized node waiting for all parallel upstream workers to complete before aggregating state.**")
    with st.expander("🎴 Flashcard 4: recursion_limit"):
        st.markdown("**Execution constraint that terminates runaway reflection cycles after a fixed superstep budget.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between Sequential, Branching, and Cyclic workflows.", value=True)
chk2 = st.checkbox("I know how to implement reflection and self-correction loops with recursion limits.", value=True)
chk3 = st.checkbox("I understand how parallel fan-out and synchronized join nodes operate across supersteps.", value=True)
chk4 = st.checkbox("I can implement dynamic Map-Reduce workflows using the LangGraph Send API.", value=True)
chk5 = st.checkbox("I know how channel reducers prevent race conditions during concurrent worker execution.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 graph workflows technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 18.1 Graph Workflows!")
