import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="36.3 Advanced Orchestration - LangChain Curriculum",
    page_icon="🎼",
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
    .orch-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-flow {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AdvancedOrchestrationEngine:
    """
    Simulates LangGraph cyclic state workflows, parallel fan-out/fan-in,
    map-reduce batch processing, and human-in-the-loop interrupts.
    """
    def __init__(self):
        self.workflow_states = {
            "idle": "Graph ready for invocation.",
            "executing_map": "Spawning 4 parallel worker nodes for document chunk analysis.",
            "interrupted": "Workflow paused at interrupt(). Awaiting human sign-off.",
            "resumed": "Human approval received. Reducing outputs to final synthesis.",
            "completed": "State successfully committed to END."
        }

    def simulate_map_reduce(self, chunk_count: int) -> Dict[str, Any]:
        # Latency of parallel map is max single chunk (~120ms) + small merge overhead
        map_latency = 120 + (chunk_count * 5)
        reduce_latency = 150
        total_time = map_latency + reduce_latency

        return {
            "chunks_mapped": chunk_count,
            "parallel_map_latency_ms": map_latency,
            "reduce_synthesis_latency_ms": reduce_latency,
            "total_execution_ms": total_time,
            "speedup_vs_sequential": round((chunk_count * 120 + 150) / total_time, 2)
        }

# Instantiate engine
orch_engine = AdvancedOrchestrationEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Graph Orchestration, Dynamic Map-Reduce, & Human-in-the-Loop Interrupts", divider="rainbow")

st.markdown("""
Modern generative AI architectures demand complex control flow far beyond simple linear DAGs. 
**Advanced Orchestration** in LangGraph enables **cyclic feedback loops**, **dynamic map-reduce** 
fan-out across massive datasets, **parallel asynchronous branch execution**, and deterministic 
**human interrupts** for sensitive administrative approvals.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Orchestration_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "Workflow Ingress & Fan-Out Mapping";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            start_node [label="START Node\\n(Receives complex task payload)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            planner_node [label="Task Decomposition (Planner)\\n(Splits task into 3 sub-documents)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_parallel {
            label = "Parallel Map Execution (Fan-Out)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            worker_1 [label="Worker Node 1: Section A\\n(Parallel LLM Analysis: 120ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            worker_2 [label="Worker Node 2: Section B\\n(Parallel LLM Analysis: 140ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            worker_3 [label="Worker Node 3: Section C\\n(Parallel LLM Analysis: 110ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_reduce {
            label = "Map-Reduce Fan-In & Human-in-the-Loop Verification";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            reducer_agg [label="Fan-In Reducer\\n(Aggregates section outputs into draft brief)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            human_interrupt [label="Human Approval: interrupt()\\n(Pauses execution; awaits admin sign-off)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
            publish_node [label="Publish / Action Node\\n(Executes final write operation)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            end_node [label="END Node\\n(Workflow completed)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        start_node -> planner_node;
        planner_node -> worker_1 [label="parallel dispatch"];
        planner_node -> worker_2 [label="parallel dispatch"];
        planner_node -> worker_3 [label="parallel dispatch"];

        worker_1 -> reducer_agg;
        worker_2 -> reducer_agg;
        worker_3 -> reducer_agg;

        reducer_agg -> human_interrupt;
        human_interrupt -> publish_node [label="approved by reviewer"];
        publish_node -> end_node;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>The Orchestration Power of LangGraph:</strong> Unlike traditional DAG engines (Airflow, Prefect) that 
    forbid cyclical execution, LangGraph natively supports cycles (agent loops), parallel branches that rejoin via reducers, 
    and first-class checkpoints that pause execution indefinitely for human feedback.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: GRAPH WORKFLOWS
# ==============================================================================
st.header("2. Graph Workflows: Cyclic Graphs vs Linear DAGs, Entrypoints, and Loops", divider="rainbow")

st.markdown("""
Traditional LLM chains are strictly unidirectional: Step A leads to Step B, which leads to Step C. 
If Step C generates an error or fails validation, a linear chain cannot backtrack. 
**Cyclic Graph Workflows** model reasoning as a state machine:
""")

st.markdown("""
- **`START` Node**: The declared entrypoint where user inputs enter the graph.
- **Node Execution**: Python functions that transform state and return updates.
- **Cyclic Feedback Loops**: If code generation in Node B fails automated tests in Node C, a conditional edge routes execution back to Node B for self-correction.
- **`END` Node**: The terminal state where execution finishes and final state is returned.
""")

# ==============================================================================
# SECTION 3: CONDITIONAL ROUTING
# ==============================================================================
st.header("3. Conditional Routing: Conditional Edges and Path Mapping Dictionaries", divider="rainbow")

st.markdown("""
**Conditional Edges** (`add_conditional_edges`) dynamically route execution between nodes 
based on runtime state evaluations:
""")

st.code('''from langgraph.graph import StateGraph, START, END

def decide_next_step(state: dict) -> str:
    \"\"\"Routing function returning destination node name.\"\"\"
    if state.get("requires_search"):
        return "web_search_node"
    elif state.get("requires_calculation"):
        return "calculator_node"
    return "final_synthesis"

# Wire conditional edge with explicit path mapping
# builder.add_conditional_edges(
#     "intent_router",
#     decide_next_step,
#     {
#         "web_search_node": "web_search_node",
#         "calculator_node": "calculator_node",
#         "final_synthesis": "final_synthesis"
#     }
# )
''', language="python")

# ==============================================================================
# SECTION 4: PARALLEL EXECUTION
# ==============================================================================
st.header("4. Parallel Execution: Concurrent Node Execution and Fan-Out Fan-In Topologies", divider="rainbow")

st.markdown("""
When multiple tasks are mutually independent (e.g., retrieving facts from 3 different vector databases), 
running them sequentially creates unnecessary delay. LangGraph supports **native parallel fan-out**:
""")

st.markdown("""
1. **Fan-Out**: An edge originates from a single source node and targets multiple destination nodes simultaneously (`builder.add_edge("planner", "search_a")`, `builder.add_edge("planner", "search_b")`).
2. **Concurrent Execution**: Both `search_a` and `search_b` execute in parallel via the asynchronous event loop.
3. **Fan-In Reducer**: Both nodes target `aggregator`. LangGraph waits for all parallel nodes to finish, combines their outputs using the channel reducer, and invokes `aggregator`.
""")

# ==============================================================================
# SECTION 5: MAP-REDUCE
# ==============================================================================
st.header("5. Map-Reduce: Dynamic Task Decomposition and Aggregation Synthesis", divider="rainbow")

st.markdown("""
In large-scale document analysis, processing a 200-page report in a single prompt causes 
context overflow and 'lost in the middle' syndrome. **Map-Reduce Orchestration** solves this:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### 1. Map Phase (Parallel)
    - The document is chunked into 10 sections.
    - LangGraph dynamically spawns 10 parallel sub-tasks via `Send(node_name, chunk)`.
    - Each sub-node independently summarizes its assigned section in parallel.
    """)

with col2:
    st.markdown("""
    ### 2. Reduce Phase (Synthesis)
    - All 10 section summaries are gathered into an accumulator channel.
    - A single synthesizer node combines the summaries into an executive briefing.
    """)

# ==============================================================================
# SECTION 6: HUMAN INTERRUPTS
# ==============================================================================
st.header("6. Human Interrupts: interrupt() Control Primitives and Resumption", divider="rainbow")

st.markdown("""
For sensitive actions (e.g., initiating wire transfers, deleting database tables, publishing marketing campaigns), 
agents must never operate fully autonomously. LangGraph v0.2+ introduces the native **`interrupt()`** primitive:
""")

st.code('''from langgraph.types import interrupt

def process_refund_node(state: dict) -> dict:
    refund_amount = state["refund_amount"]
    
    # If transaction exceeds $500, pause execution for human sign-off!
    if refund_amount > 500.00:
        human_approval = interrupt({
            "message": f"Refund of ${refund_amount} requires manager approval.",
            "refund_amount": refund_amount,
            "user_id": state["user_id"]
        })
        
        if not human_approval.get("approved"):
            return {"status": "REJECTED_BY_MANAGER"}

    # Proceed with action
    return {"status": "REFUND_EXECUTED"}
''', language="python")

st.markdown("""
When `interrupt()` is called:
1. Graph execution **immediately halts**.
2. State is durably committed to the database checkpointer.
3. When the human reviewer submits approval via UI/API, the graph **resumes execution from the exact line of code**!
""")

# ==============================================================================
# SECTION 7: DURABLE EXECUTION
# ==============================================================================
st.header("7. Durable Execution: Multi-Step Fault Tolerance and Heartbeat Monitors", divider="rainbow")

st.markdown("""
**Durable Execution** guarantees that once a workflow starts, it will run to completion 
regardless of infrastructure failures, network blips, or deployment updates:
""")

st.markdown("""
- **Idempotency Keys**: Nodes that interact with external billing or notification APIs attach idempotency keys to ensure retries do not execute duplicate actions.
- **Heartbeat Leases**: Worker tasks renew a heartbeat in Redis every 10 seconds. If a worker dies, a standby worker claims the thread and resumes from the last checkpoint.
""")

# ==============================================================================
# SECTION 8: LONG-RUNNING AGENTS
# ==============================================================================
st.header("8. Long-Running Agents: Asynchronous Background Tasks and Webhooks", divider="rainbow")

st.markdown("""
Some agent tasks require days or weeks to conclude (e.g., awaiting user feedback, scheduled daily reports, 
or multi-stage hiring candidate evaluations).
""")

long_run_df = pd.DataFrame([
    {"Pattern": "Webhook Callback Ingress", "Mechanism": "Graph pauses at interrupt(); external webhook POSTs /threads/{id}/resume to continue execution."},
    {"Pattern": "Cron-Triggered Schedules", "Mechanism": "LangGraph Cloud runs graphs on cron expressions (e.g., '0 8 * * *') to generate daily digest emails."},
    {"Pattern": "Time-Delayed Pollers", "Mechanism": "Agent polls an external API status with exponential backoff while persisting state across sleep intervals."}
])
st.dataframe(long_run_df, use_container_width=True)

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Map-Reduce & Human Interrupt StateGraph Pipeline", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**orchestration engine featuring parallel map-reduce processing and human approval interrupts**.
""")

st.code('''from typing import Dict, Any, List

class ProductionOrchestrationEngine:
    def __init__(self):
        self.pending_interrupts: Dict[str, dict] = {}
        self.completed_runs: Dict[str, dict] = {}

    def execute_map_reduce_workflow(self, thread_id: str, document_sections: List[str]) -> Dict[str, Any]:
        """Simulates parallel map-reduce execution."""
        # Map phase: process each section in parallel
        mapped_summaries = [f"Summary of: {sec[:30]}..." for sec in document_sections]
        
        # Check if sensitive content requires human interrupt
        if len(document_sections) > 5:
            self.pending_interrupts[thread_id] = {
                "thread_id": thread_id,
                "reason": "Large batch processing (> 5 sections) requires human verification.",
                "mapped_summaries": mapped_summaries
            }
            return {
                "thread_id": thread_id,
                "status": "INTERRUPTED",
                "pending_review": True
            }

        # Reduce phase: synthesize summaries
        final_brief = " | ".join(mapped_summaries)
        result = {"thread_id": thread_id, "status": "COMPLETED", "executive_brief": final_brief}
        self.completed_runs[thread_id] = result
        return result

    def resume_interrupted_thread(self, thread_id: str, approved: bool) -> Dict[str, Any]:
        """Resumes execution after human reviewer sign-off."""
        if thread_id not in self.pending_interrupts:
            raise ValueError("Thread ID not paused at interrupt.")

        paused_data = self.pending_interrupts.pop(thread_id)
        if not approved:
            return {"thread_id": thread_id, "status": "REJECTED_BY_HUMAN"}

        # Proceed with reduce synthesis
        final_brief = " | ".join(paused_data["mapped_summaries"])
        result = {"thread_id": thread_id, "status": "COMPLETED", "executive_brief": final_brief}
        self.completed_runs[thread_id] = result
        return result

orchestrator = ProductionOrchestrationEngine()

# Run batch requiring approval (> 5 sections)
sections = [f"Section #{i}: Technical infrastructure details" for i in range(1, 8)]
run_status = orchestrator.execute_map_reduce_workflow("th_exec_9001", sections)
print("Run 1 Status:", run_status)

# Human signs off and resumes thread
resumed_status = orchestrator.resume_interrupted_thread("th_exec_9001", approved=True)
print("Resumed Run Status:", resumed_status)
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Human-in-the-Loop Interrupt Simulator & Map-Reduce Benchmarker", divider="rainbow")

st.markdown("""
Simulate Map-Reduce parallel speedups and test human interrupt approval 
mechanics in real time.
""")

wb_view = st.selectbox(
    "Select Orchestration Tool:",
    ["1. Map-Reduce Parallel Throughput Benchmarker", "2. Human-in-the-Loop Interrupt Simulator", "3. Cyclic Feedback Loop Inspector"]
)

if wb_view == "1. Map-Reduce Parallel Throughput Benchmarker":
    st.subheader("Map-Reduce Concurrency Calculator")
    chunks = st.slider("Number of Document Sections to Map:", 2, 20, 8, 1)

    mr_stats = orch_engine.simulate_map_reduce(chunks)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Execution Time", f"{mr_stats['total_execution_ms']}ms")
    with m2:
        st.metric("Parallel Speedup", f"{mr_stats['speedup_vs_sequential']}x")
    with m3:
        st.metric("Mapped Sections", f"{mr_stats['chunks_mapped']}")

    st.success(f"Parallel map reduced processing time from {chunks * 120 + 150}ms to {mr_stats['total_execution_ms']}ms!")

elif wb_view == "2. Human-in-the-Loop Interrupt Simulator":
    st.subheader("interrupt() Approval Flow Sandbox")
    refund_val = st.number_input("Proposed Refund Amount (USD):", min_value=50.0, max_value=2000.0, value=750.0, step=50.0)

    if refund_val > 500.00:
        st.error(f"WORKFLOW PAUSED: Refund of USD {refund_val:.2f} exceeds threshold (USD 500.00). State checkpointed at interrupt().")
        st.markdown("#### Human Reviewer Actions:")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Authorize Refund"):
                st.success(f"Approved! Execution resumed from checkpoint. Transaction of USD {refund_val:.2f} processed.")
        with col_b:
            if st.button("Reject Refund"):
                st.warning("Rejected! Execution resumed with status 'REJECTED_BY_MANAGER'.")
    else:
        st.success(f"Auto-approved! Refund of USD {refund_val:.2f} is under threshold (USD 500.00). Executed without interrupt.")

else:
    st.subheader("Cyclic Feedback Self-Correction State Machine")
    st.markdown("Cycle flow for automated code generation:")
    st.info("""
    1. Node `coder`: Generates Python script based on user prompt.
    2. Node `tester`: Runs automated pytest suite against generated script.
    3. Conditional Edge `evaluate_tests`:
       - If all tests PASS -> Route to `END`.
       - If any test FAILS -> Route back to `coder` with error logs in state (Cycle repeats, max 3 loops).
    """)
    st.success("Cyclic state machines allow agents to self-correct autonomously before returning results to users.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does LangGraph handle cyclic execution loops without infinite recursion?"):
    st.markdown("""
    **Answer:**  
    LangGraph supports cyclic graphs natively by allowing edges to point back to previous nodes. To prevent infinite loops, 
    developers configure `recursion_limit` (defaulting to 25 steps) in the run config. If a cycle exceeds this threshold, 
    the runtime raises a `GraphRecursionError`, protecting operational budgets.
    """)

with st.expander("Q2: How does the interrupt() primitive differ from traditional webhook polling?"):
    st.markdown("""
    **Answer:**  
    Traditional polling requires maintaining active threads, sleeping processes, or complex state serialization 
    in bespoke databases. The `interrupt()` primitive is a first-class language feature: it halts the graph, 
    persists state automatically via the checkpointer, and cleanly terminates the worker process. The workflow 
    can resume hours or weeks later when the client dispatches a resumption request with feedback.
    """)

with st.expander("Q3: What is the Send API in LangGraph, and how does it enable dynamic map-reduce?"):
    st.markdown("""
    **Answer:**  
    `Send(node_name, arg)` allows conditional edges to emit a dynamic list of tasks to execute. In a map-reduce pattern, 
    a router node inspects state, partitions a dataset into N chunks, and returns `[Send("summarizer", chunk_i) for chunk_i in chunks]`. 
    LangGraph spawns N concurrent node executions and merges their results into the designated reducer channel.
    """)

with st.expander("Q4: Why are idempotency keys critical for durable workflow execution?"):
    st.markdown("""
    **Answer:**  
    If a worker node crashes immediately after executing an external action (such as charging a credit card) but before 
    committing the checkpoint to the database, a resuming worker will replay that node. Idempotency keys ensure the external 
    payment gateway identifies the replay and returns the original transaction receipt rather than charging the customer twice.
    """)

with st.expander("Q5: How does parallel fan-out / fan-in improve throughput compared to sequential execution?"):
    st.markdown("""
    **Answer:**  
    Sequential execution sums the latencies of all tasks (T1 + T2 + T3). Parallel fan-out executes independent tasks 
    concurrently via `asyncio`, reducing total execution time to max(T1, T2, T3) + overhead. For multi-retriever or 
    multi-tool agent tasks, this delivers a 2x to 5x latency reduction.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Deploying Cyclic State Machines Without Recursion Limits**  
A self-correcting agent whose test suite always fails will cycle indefinitely, racking up massive API bills 
until manual intervention. Always set a strict `recursion_limit`.
""")

st.error("""
**Trap 2: Executing Non-Idempotent External Actions Inside Retried Nodes**  
Placing credit card charges or email dispatches inside a graph node without idempotency keys risks duplicate transactions 
whenever a network glitch triggers a retry.
""")

st.warning("""
**Trap 3: Using interrupt() Without a Persistent Database Checkpointer**  
Invoking `interrupt()` with `MemorySaver` keeps state only in RAM. If the server restarts before the human provides approval, 
the entire workflow state is lost forever.
""")

st.warning("""
**Trap 4: Overusing Map-Reduce for Small Contexts**  
Spawning 10 parallel LLM calls to process a 500-word document incurs significant orchestration overhead and token costs. 
Map-reduce is only cost-effective for documents that exceed single-prompt context limits.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>interrupt()</h4>
        <p>A native LangGraph control primitive pausing execution and persisting state to await external human sign-off.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Map-Reduce Orchestration</h4>
        <p>A pattern partitioning large inputs across parallel worker nodes (map) before combining outputs into a unified synthesis (reduce).</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Cyclic Graph</h4>
        <p>A state machine topology allowing backward edges and feedback loops for iterative self-correction.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Send API</h4>
        <p>A LangGraph primitive allowing conditional edges to dispatch dynamic lists of concurrent tasks to target nodes.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand cyclic state machine topologies and how to enforce recursion limits.", value=True)
chk_2 = st.checkbox("I know how to implement conditional edges with path mapping dictionaries.", value=True)
chk_3 = st.checkbox("I can implement parallel fan-out and fan-in workflows using channel reducers.", value=True)
chk_4 = st.checkbox("I understand dynamic map-reduce orchestration using the Send API.", value=True)
chk_5 = st.checkbox("I know how to implement human-in-the-loop approvals using the interrupt() primitive.", value=True)
chk_6 = st.checkbox("I understand durable execution, idempotency keys, and crash fault-tolerance.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced Orchestration competencies verified! Chapter 36 fully completed.")
