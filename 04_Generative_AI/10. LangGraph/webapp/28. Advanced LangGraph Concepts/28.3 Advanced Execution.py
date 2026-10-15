import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="28.3 Advanced Execution - LangGraph Curriculum",
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
    .badge-exec {
        background-color: #172554;
        color: #38BDF8;
        border: 1px solid #0284C7;
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
class AdvancedExecutionSimulator:
    """
    Simulates dynamic Map-Reduce execution using Send API:
    - Splits document batches into variable worker tasks
    - Executes parallel fan-out
    - Aggregates outputs at superstep fan-in barrier
    """
    def __init__(self):
        pass

    def execute_map_reduce_simulation(self, chunk_count: int, worker_latency_ms: int) -> Dict[str, Any]:
        tasks = [f"task_chunk_{i+1}" for i in range(chunk_count)]
        sequential_duration = chunk_count * worker_latency_ms
        # Parallel duration overlaps with small barrier overhead
        parallel_duration = worker_latency_ms + 25
        speedup = sequential_duration / max(parallel_duration, 1)
        
        return {
            "chunk_count": chunk_count,
            "tasks_spawned": tasks,
            "sequential_duration_ms": sequential_duration,
            "parallel_duration_ms": parallel_duration,
            "speedup_factor": round(speedup, 2)
        }

exec_sim = AdvancedExecutionSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Execution Models, Map-Reduce Topologies, and Durable Runtimes", divider="rainbow")

st.markdown("""
Advanced LangGraph execution models push beyond static node topologies to dynamically adaptive architectures. 
By utilizing the **Send API**, graphs can inspect runtime state, split large collections into dynamic batches, 
fan-out variable numbers of concurrent worker nodes simultaneously, and coalesce their outputs at a synchronized fan-in barrier. 
Combined with durable checkpointers, exponential retry policies, and semantic caching, this architecture enables resilient, 
high-throughput distributed agent processing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedExecutionBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_dispatch {
            label="1. Ingress & Dynamic Fan-Out Dispatcher";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            input_doc [label="Ingress Documents / Batch\\nArray of N Unprocessed Items", fillcolor="#0F172A"];
            splitter_node [label="Splitter Node (Map Stage)\\nReturns [Send('worker', item_1), Send('worker', item_2)...]", fillcolor="#312E81"];
        }

        subgraph cluster_workers {
            label="2. Concurrent Worker Pool (Dynamic Fan-Out)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            w1 [label="Worker Instance 1\\nItem 1 Processing", fillcolor="#1E293B"];
            w2 [label="Worker Instance 2\\nItem 2 Processing", fillcolor="#1E293B"];
            wn [label="Worker Instance N\\nItem N Processing", fillcolor="#1E293B"];
        }

        subgraph cluster_reduce {
            label="3. Fan-In Barrier & Reducer Aggregation";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            fan_in_barrier [label="Fan-In Synchronization Barrier\\nChannel Reducers Aggregate All Worker Deltas", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            reducer_node [label="Synthesizer Node (Reduce Stage)\\nProduces Consolidated Final Deliverable", fillcolor="#0F172A"];
        }

        input_doc -> splitter_node;
        splitter_node -> w1 [label="Send(item_1)"];
        splitter_node -> w2 [label="Send(item_2)"];
        splitter_node -> wn [label="Send(item_N)"];

        w1 -> fan_in_barrier;
        w2 -> fan_in_barrier;
        wn -> fan_in_barrier;

        fan_in_barrier -> reducer_node;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Parallel Execution
# ==============================================================================
st.header("2. Parallel Execution: Asynchronous Concurrency, Asyncio Tasks, and Superstep Scheduling", divider="rainbow")

st.markdown("""
In LangGraph, parallel execution is achieved by scheduling multiple nodes to run in the same superstep. 
The runtime dispatches node coroutines concurrently using `asyncio.create_task` under the hood.

### Execution Guarantees:
- **Non-Blocking Throughput**: While Worker 1 awaits a 3-second LLM API response, Worker 2 and Worker 3 execute concurrently.
- **Synchronized Convergence**: The graph never advances to the next superstep until **all** concurrent workers complete.
""")

# ==============================================================================
# SECTION 3: Map-Reduce
# ==============================================================================
st.header("3. Map-Reduce: Splitting Complex Workloads, Worker Task Queuing, and Aggregator Synthesis", divider="rainbow")

st.markdown("""
Map-Reduce is a foundational pattern for processing large datasets in LangGraph:
1. **Map (Split)**: A node inspects an input (such as a 100-page document or a list of 50 customer tickets) 
   and splits it into discrete units of work.
2. **Execute (Workers)**: Multiple worker nodes process individual units in parallel.
3. **Reduce (Aggregate)**: A synthesis node collects all worker outputs from a reduced channel and produces 
   the final summarized report.
""")

# ==============================================================================
# SECTION 4: Dynamic Fan-Out
# ==============================================================================
st.header("4. Dynamic Fan-Out: Spawning Variable Worker Nodes at Runtime via Send() API", divider="rainbow")

st.markdown("""
In static graphs, the number of parallel branches is hardcoded at compile time. 
With LangGraph's **`Send` API**, the graph dynamically fans out to an arbitrary number of workers determined at runtime.

### Syntax & Mechanics:
A conditional edge returns a list of `Send("node_name", arg_dict)` objects:
""")

st.code('''from langgraph.types import Send

def continue_to_workers(state: OverallState) -> list:
    items = state.get("items_to_process", [])
    # Dynamically spawn one worker node per item
    return [Send("worker_node", {"item": item}) for item in items]

# Binding dynamic fan-out to graph builder
builder.add_conditional_edges("splitter_node", continue_to_workers, ["worker_node"])
''', language="python")

# ==============================================================================
# SECTION 5: Dynamic Fan-In
# ==============================================================================
st.header("5. Dynamic Fan-In: Barrier Synchronization and Channel Aggregation Across Variable Workers", divider="rainbow")

st.markdown("""
When variable workers complete, their outputs must converge cleanly at a fan-in barrier.

### Rules for Dynamic Fan-In:
- **Shared Reduced Channel**: Workers must write to a channel configured with a reducer (such as `Annotated[List[str], operator.add]`).
- **Barrier Edge**: In LangGraph, adding a standard edge from the worker node to the aggregator node (`builder.add_edge("worker_node", "aggregator")`) 
  automatically creates a synchronization barrier that waits for **all dynamic `Send` instances** to finish before invoking the aggregator.
""")

# ==============================================================================
# SECTION 6: Retry Policies
# ==============================================================================
st.header("6. Retry Policies: Tenacity Integration, Jittered Backoff, and Idempotent Node Replay", divider="rainbow")

st.markdown("""
Network hiccups and provider rate limits require resilient retry policies inside node functions:
- **Exponential Backoff**: Increase retry intervals exponentially (1s, 2s, 4s, 8s).
- **Full Jitter**: Add randomized delay to prevent thundering herds across concurrent pods.
- **Idempotency**: Ensure that retrying a node does not generate duplicate database side effects.
""")

# ==============================================================================
# SECTION 7: Caching
# ==============================================================================
st.header("7. Caching: Deterministic In-Memory and Redis Caching for Expensive Supersteps", divider="rainbow")

st.markdown("""
Expensive tool calls (such as search queries, document embeddings, or code executions) should be cached:
- **In-Memory LRU Cache**: For deterministic pure functions within local process lifecycles.
- **Distributed Redis Cache**: Shared semantic and key-value caching across multiple Kubernetes agent pods.
""")

# ==============================================================================
# SECTION 8: Durable Execution
# ==============================================================================
st.header("8. Durable Execution: Fault-Tolerant State Commitment and Process Crash Recovery", divider="rainbow")

st.markdown("""
Durable execution guarantees that an agent workflow can survive hardware crashes, container restarts, 
and network partition events:
- **Per-Superstep Snapshots**: Every superstep commit is an atomic transaction in PostgreSQL.
- **Resumption**: If a worker pod crashes at step 4 of 10, a replacement pod loads the checkpoint for step 3 
  and continues execution without losing historical work.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Dynamic Map-Reduce Send() Pipeline with Resilient Caching", divider="rainbow")

st.markdown("""
The following complete module implements a dynamic Map-Reduce pipeline using the `Send` API to process 
variable document chunks concurrently.
""")

st.code('''from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

class OverallState(TypedDict):
    documents: List[str]
    summaries: Annotated[List[str], operator.add]
    final_report: str

class WorkerInput(TypedDict):
    doc_text: str

def splitter_node(state: OverallState) -> dict:
    return {"summaries": []}

def map_documents_to_workers(state: OverallState) -> list:
    # Dynamic fan-out using Send API
    return [Send("summarize_worker", {"doc_text": doc}) for doc in state["documents"]]

def summarize_worker(state: WorkerInput) -> dict:
    # Worker processes single document chunk
    doc = state["doc_text"]
    summary = f"Summary of: {doc[:30]}..."
    return {"summaries": [summary]}

def reducer_node(state: OverallState) -> dict:
    # Aggregates all worker outputs at fan-in barrier
    combined = " | ".join(state["summaries"])
    return {"final_report": f"Consolidated Executive Report: {combined}"}

builder = StateGraph(OverallState)
builder.add_node("splitter", splitter_node)
builder.add_node("summarize_worker", summarize_worker)
builder.add_node("reducer", reducer_node)

builder.add_edge(START, "splitter")
builder.add_conditional_edges("splitter", map_documents_to_workers, ["summarize_worker"])
builder.add_edge("summarize_worker", "reducer")
builder.add_edge("reducer", END)

production_map_reduce_app = builder.compile(checkpointer=MemorySaver())
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Dynamic Fan-Out Workload Calculator & Send() Pipeline Sandbox", divider="rainbow")

st.markdown("""
Calculate latency speedups and simulate dynamic `Send` fan-out execution across variable document workloads.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Dynamic Fan-Out Parameters")
    num_chunks = st.slider("Document Chunks to Process (Dynamic N)", min_value=2, max_value=12, value=5)
    chunk_latency = st.slider("Average Worker Task Latency (ms)", min_value=50, max_value=600, value=200, step=25)
    
    metrics = exec_sim.execute_map_reduce_simulation(num_chunks, chunk_latency)
    
    st.markdown(f"**Sequential Execution Duration:** `{metrics['sequential_duration_ms']} ms`")
    st.markdown(f"**Parallel Send() Duration:** `{metrics['parallel_duration_ms']} ms`")
    st.metric("Concurrency Speedup Factor", f"{metrics['speedup_factor']}x Faster")

with col_wb2:
    st.subheader("Dynamic Send() Task Queue")
    st.markdown(f"**Spawned {len(metrics['tasks_spawned'])} Dynamic Worker Instances:**")
    st.json(metrics["tasks_spawned"])
    st.success("All dynamic worker tasks converged at the fan-in barrier and merged into the consolidated report.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What is the Send API in LangGraph and what problem does it solve?"):
    st.markdown("""
    **Answer:**  
    The `Send(node_name, state_arg)` primitive enables **dynamic fan-out**. In standard graphs, parallel branches are statically fixed. 
    `Send` allows a conditional routing function to return a list of destination tasks whose count and input payloads are computed dynamically 
    at runtime (e.g., spawning one worker per search result or document chunk).
    """)

with st.expander("Q2: How does LangGraph handle the fan-in barrier when Send spawns a variable number of workers?"):
    st.markdown("""
    **Answer:**  
    LangGraph tracks all dynamic task instances dispatched in that superstep. It buffers their execution and only schedules the downstream 
    node when **all** instances of the target node complete. The target channel must use a reducer (e.g., `operator.add`) to accumulate the outputs.
    """)

with st.expander("Q3: What happens if one dynamic worker fails during a parallel Map-Reduce execution?"):
    st.markdown("""
    **Answer:**  
    If an unhandled exception occurs in one worker, the superstep barrier fails, and the graph halts. In production, wrap worker logic 
    in retry policies (via Tenacity) or catch exceptions inside the worker to return an error status object, allowing the remaining workers to finish.
    """)

with st.expander("Q4: Why is durable execution essential for long-running Map-Reduce agent workflows?"):
    st.markdown("""
    **Answer:**  
    If an agent processing 50 documents crashes at document 45, without durable checkpoints the entire workflow would need to re-execute from document 1, 
    wasting significant compute and API tokens. Durable execution commits intermediate progress to disk, enabling immediate recovery.
    """)

with st.expander("Q5: How does semantic caching differ from standard key-value caching in agent systems?"):
    st.markdown("""
    **Answer:**  
    Standard caching requires exact string matches on queries. Semantic caching embeds the incoming query and performs vector similarity search; 
    if similarity exceeds a threshold (e.g., 0.96), it returns the cached response even if phrasing differs.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Forgetting Channel Reducers on Send() Fan-In Destinations**  
If dynamic workers write to an unreduced state channel, LangGraph raises `InvalidUpdateError` as soon as the second worker finishes. 
Always annotate the destination channel with a reducer like `operator.add`.
""")

st.error("""
**Trap 2: Spawning Unbounded Dynamic Workers Overwhelming Downstream APIs**  
Using `Send` to spawn 500 parallel workers simultaneously triggers massive concurrency bursts that exceed model provider rate limits (429) 
or exhaust local thread pools. Always batch items or use concurrency semaphores.
""")

st.warning("""
**Trap 3: Passing Heavy State Payloads Inside Send() Arguments**  
Passing full megabyte document contexts in each `Send` payload increases memory usage. Pass document IDs or pointers where possible.
""")

st.warning("""
**Trap 4: Unhandled Worker Exceptions Aborting Entire Map Stages**  
Failing to catch exceptions in worker nodes causes the entire superstep barrier to abort. Wrap worker calls in try/except blocks that return structured error dicts.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Send API</h4>
        <p>LangGraph primitive enabling dynamic fan-out by spawning variable worker tasks at runtime with custom state arguments.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Map-Reduce Topology</h4>
        <p>A distributed pattern splitting large workloads across dynamic workers (Map) and consolidating results at a barrier (Reduce).</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Dynamic Fan-In Barrier</h4>
        <p>LangGraph superstep synchronization waiting for all dynamically dispatched Send instances to finish before proceeding.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Durable Execution</h4>
        <p>Guaranteed resumption of interrupted workflows from the last committed checkpoint snapshot following process crashes.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to implement dynamic fan-out using LangGraph's Send API.", value=True)
chk_2 = st.checkbox("I understand how the fan-in barrier synchronizes variable dynamic worker instances.", value=True)
chk_3 = st.checkbox("I can design Map-Reduce workflows to process large document batches in parallel.", value=True)
chk_4 = st.checkbox("I understand how to protect parallel workers with retry policies and exponential backoff.", value=True)
chk_5 = st.checkbox("I know how to implement in-memory and semantic caching for expensive supersteps.", value=True)
chk_6 = st.checkbox("I understand durable execution and how checkpoints guarantee crash recovery.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced execution models and dynamic Map-Reduce mastery confirmed! Ready for Advanced Persistence.")
