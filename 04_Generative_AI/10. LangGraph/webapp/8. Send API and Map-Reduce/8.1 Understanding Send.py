import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="8.1 Understanding Send - LangGraph Curriculum",
    page_icon="🚀",
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
# MODULE-SCOPE DYNAMIC SEND SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class DynamicSendSimulator:
    """
    Simulates LangGraph's Send primitive:
    Dynamically generating N parallel worker tasks with isolated state payloads.
    """
    def __init__(self):
        self.worker_template = "worker_node"

    def generate_send_dispatches(self, items: List[str]) -> List[Dict[str, Any]]:
        """
        Emulates a conditional edge returning [Send('worker_node', {'item': i}) for i in items]
        """
        dispatches = []
        for idx, item in enumerate(items, 1):
            dispatches.append({
                "worker_id": f"worker_{idx}",
                "target_node": self.worker_template,
                "payload": {"item_id": idx, "content": item.strip()},
                "status": "DISPATCHED"
            })
        return dispatches

    def execute_workers(self, dispatches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulates parallel BSP execution of worker nodes.
        """
        results = []
        for d in dispatches:
            payload = d["payload"]
            length = len(payload["content"].split())
            results.append({
                "worker_id": d["worker_id"],
                "input_item": payload["content"],
                "word_count": length,
                "summary": f"Analyzed item '{payload['content']}' ({length} words).",
                "emitted_delta": {"summaries": [f"Summary {payload['item_id']}: {length} words"]}
            })
        return results


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Dynamic Worker Dispatch and Map-Reduce with the Send API", divider="rainbow")

st.markdown("""
In static graphs, the number of parallel branches is fixed at compile time (e.g. `add_edge('dispatch', 'node_a')` and `add_edge('dispatch', 'node_b')`).
However, real-world AI applications frequently encounter **variable-length collections**:
a user uploads 10 PDF documents, an LLM generates 5 sub-queries, or an ensemble model needs 7 candidate answers.
The **Send API** (`from langgraph.types import Send`) enables **dynamic map-reduce**:
a conditional edge evaluates state at runtime and emits a dynamically sized list of `Send("node_name", worker_payload)` objects,
spawning parallel workers with isolated payloads in the next superstep.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    send_blueprint_dot = """
    digraph SendArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress & Item Splitting";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nUser Input: N Documents", fillcolor="#1E293B", shape=ellipse];
            Splitter [label="Chunking / Planning Node\\nProduces state['items'] = [Doc1, Doc2, ..., DocN]", fillcolor="#0284C7"];
        }

        subgraph cluster_send_edge {
            label="2. Dynamic Send Edge (Conditional Dispatch)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Send_Edge [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#4338CA"><b>Conditional Edge returning List[Send]</b></td></tr>
                <tr><td>Send 1</td><td>Send('process_doc', {'doc': Doc1})</td></tr>
                <tr><td>Send 2</td><td>Send('process_doc', {'doc': Doc2})</td></tr>
                <tr><td>Send N</td><td>Send('process_doc', {'doc': DocN})</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_dynamic_workers {
            label="3. Parallel Worker Execution (BSP Superstep)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Worker1 [label="Worker Instance 1\\nRuns process_doc(Doc1)", fillcolor="#0D9488"];
            Worker2 [label="Worker Instance 2\\nRuns process_doc(Doc2)", fillcolor="#0D9488"];
            WorkerN [label="Worker Instance N\\nRuns process_doc(DocN)", fillcolor="#0D9488"];
        }

        subgraph cluster_reduce_phase {
            label="4. Barrier Aggregation (Reduce Phase)";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            Reducer_Node [label="Aggregator Node\\nConsolidates Annotated[list, operator.add] deltas", fillcolor="#059669"];
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Splitter;
        Splitter -> Send_Edge [label="Evaluate items in state"];
        Send_Edge -> Worker1 [label="Dynamic Spawn"];
        Send_Edge -> Worker2 [label="Dynamic Spawn"];
        Send_Edge -> WorkerN [label="Dynamic Spawn"];
        Worker1 -> Reducer_Node [label="Emit partial summary"];
        Worker2 -> Reducer_Node [label="Emit partial summary"];
        WorkerN -> Reducer_Node [label="Emit partial summary"];
        Reducer_Node -> END;
    }
    """
    st.graphviz_chart(send_blueprint_dot, use_container_width=True)

st.markdown("""
### Core Value: Dynamic Elastic Parallelism
With `Send`, your graph topology adapts automatically to the scale of incoming data.
Whether processing 1 document or 100 documents, the graph architecture remains identical,
and workers run concurrently in a single BSP superstep without writing custom looping logic.
""")

# ==============================================================================
# SECTION 2: WHAT IS SEND?
# ==============================================================================
st.header("2. What is Send? The Dynamic Edge and Payload Dispatch Primitive", divider="rainbow")

st.markdown("""
`Send` (`from langgraph.types import Send`) is a lightweight container specifying:
1. **Target Node Identifier** (`str`): The name of the registered node to execute.
2. **Custom State Payload** (`dict`): The specific input dictionary provided to that worker instance.
""")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("""
    #### Anatomy of a Send Call:
    ```python
    from langgraph.types import Send

    Send(
        node="summarize_chunk", # Registered node name
        arg={"chunk_text": "..."} # Custom isolated state for worker
    )
    ```
    """)
with col_s2:
    st.markdown("""
    #### How It Differs from Normal Edges:
    - **Normal Edges**: Target nodes receive the *entire central state* snapshot.
    - **Send Objects**: Target nodes receive a *customized sub-payload* tailored exclusively to that worker!
    """)

# ==============================================================================
# SECTION 3: WHY SEND IS NEEDED
# ==============================================================================
st.header("3. Why Send is Needed: Overcoming Static Edge Constraints in Multi-Item Workflows", divider="rainbow")

st.markdown("""
Before `Send`, handling a dynamic list of items required one of two flawed anti-patterns:
""")

col_anti1, col_anti2 = st.columns(2)
with col_anti1:
    st.error("Anti-Pattern 1: In-Node Sequential Loop")
    st.code("""def process_all_docs(state):
    # Runs docs sequentially!
    # If 10 docs take 2s each -> 20s total latency!
    summaries = []
    for doc in state["docs"]:
        summaries.append(llm.invoke(doc))
    return {"summaries": summaries}
""", language="python")
    st.caption("Destroys parallelism; scales poorly with latency.")

with col_anti2:
    st.error("Anti-Pattern 2: Hardcoded Parallel Branches")
    st.code("""# Fixed compile-time topology
builder.add_edge("split", "worker_1")
builder.add_edge("split", "worker_2")
builder.add_edge("split", "worker_3")
# What if the user submits 4 documents? Or only 1?
""", language="python")
    st.caption("Inflexible; fails when document count varies dynamically.")

st.success("✅ The Send Solution: A conditional edge generates `[Send('worker', {'doc': d}) for d in state['docs']]`, executing all documents concurrently in one superstep!")

# ==============================================================================
# SECTION 4: DYNAMIC PARALLEL EXECUTION
# ==============================================================================
st.header("4. Dynamic Parallel Execution: Runtime Worker Instantiation", divider="rainbow")

st.markdown("""
When a conditional edge returns a list of `Send` objects, LangGraph's BSP runtime treats each `Send` as an active task.
All worker instances execute concurrently in the next superstep.
""")

code_send_router = """from langgraph.types import Send

def dispatch_documents_router(state: OverallState) -> list[Send]:
    documents = state["documents"]
    # Spawn a worker for every document dynamically!
    return [
        Send("analyze_document", {"doc_id": doc["id"], "text": doc["text"]})
        for doc in documents
    ]

# Bind router to conditional edge
builder.add_conditional_edges("ingest_documents", dispatch_documents_router)
"""
st.code(code_send_router, language="python")

# ==============================================================================
# SECTION 5: DYNAMIC STATE
# ==============================================================================
st.header("5. Dynamic State: Isolated Worker Payloads and Localized State Subsets", divider="rainbow")

st.markdown("""
A critical innovation of `Send` is **payload isolation**.
The worker node does NOT need to accept the massive `OverallState` schema;
it can declare its own lightweight, focused input schema matching the `arg` dictionary!
""")

code_isolated_worker = """from typing import TypedDict

# 1. Worker has its own focused state contract!
class WorkerInputState(TypedDict):
    doc_id: str
    text: str

# 2. Worker node receives only its assigned document
def analyze_document_node(state: WorkerInputState) -> dict:
    doc_length = len(state["text"])
    # Emits delta that merges into the OverallState reducer channel
    return {
        "aggregated_results": [{"id": state["doc_id"], "length": doc_length}]
    }
"""
st.code(code_isolated_worker, language="python")

# ==============================================================================
# SECTION 6: DYNAMIC EDGES
# ==============================================================================
st.header("6. Dynamic Edges: Conditional Functions Emitting Send Objects", divider="rainbow")

st.markdown("""
Conditional edges can mix return types:
a router function can return a list of `Send` objects, a standard node name string, or `END`.
""")

code_flexible_router = """def intelligent_triage(state: TriageState) -> list[Send] | str:
    queries = state["sub_queries"]
    
    if not queries:
        return "fallback_answer" # Standard node transition
        
    # Emit dynamic Send workers
    return [Send("search_worker", {"query": q}) for q in queries]
"""
st.code(code_flexible_router, language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Topic Research Dispatcher Powered by Send", divider="rainbow")

st.markdown("""
Below is a complete, production-tested script demonstrating `Send` in a Multi-Query Research pipeline.
It splits a topic into sub-questions, executes parallel search workers, and reduces findings into a final report.
""")

prod_send_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# Step 1: Schemas
class OverallResearchState(TypedDict):
    topic: str
    sub_questions: List[str]
    worker_findings: Annotated[List[Dict[str, str]], operator.add]
    final_synthesis: str

class ResearchWorkerState(TypedDict):
    question: str

# Step 2: Nodes
def planner_node(state: OverallResearchState) -> Dict[str, Any]:
    topic = state["topic"]
    # Decompose into 3 targeted sub-questions
    questions = [
        f"What are the core technical benefits of {topic}?",
        f"What are the major performance trade-offs of {topic}?",
        f"What are enterprise production best practices for {topic}?"
    ]
    return {"sub_questions": questions}

def worker_node(state: ResearchWorkerState) -> Dict[str, Any]:
    # Each worker executes independently with its own question!
    q = state["question"]
    mock_finding = {
        "question": q,
        "answer": f"Verified insight answering: '{q}'"
    }
    return {"worker_findings": [mock_finding]}

def reducer_node(state: OverallResearchState) -> Dict[str, Any]:
    findings = state["worker_findings"]
    summary = f"Synthesized {len(findings)} research streams for topic '{state['topic']}'."
    return {"final_synthesis": summary}

# Step 3: Dynamic Send Router
def dispatch_research_workers(state: OverallResearchState) -> List[Send]:
    return [
        Send("research_worker", {"question": q})
        for q in state["sub_questions"]
    ]

# Step 4: Build Graph
def build_send_graph():
    builder = StateGraph(OverallResearchState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("research_worker", worker_node)
    builder.add_node("reducer", reducer_node)
    
    builder.add_edge(START, "planner")
    
    # Conditional edge returns List[Send]!
    builder.add_conditional_edges("planner", dispatch_research_workers)
    
    # All parallel worker instances join at reducer
    builder.add_edge("research_worker", "reducer")
    builder.add_edge("reducer", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_send_graph()
    initial_input = {
        "topic": "LangGraph Send API",
        "sub_questions": [],
        "worker_findings": [],
        "final_synthesis": ""
    }
    result = app.invoke(initial_input)
    print("Final Synthesis:", result["final_synthesis"])
    print("Collected Findings Count:", len(result["worker_findings"]))
'''
st.code(prod_send_code, language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Dynamic Send Fan-Out Simulator", divider="rainbow")

st.markdown("""
Observe dynamic parallel dispatch in real-time. Enter a comma-separated list of documents or queries,
and watch the simulator generate isolated `Send` packets and execute parallel worker nodes.
""")

sim = DynamicSendSimulator()

col_sw_l, col_sw_r = st.columns([1, 1.2])

with col_sw_l:
    st.subheader("Dynamic Item Generator")
    raw_items = st.text_area(
        "Enter Items to Dispatch (One per line)",
        value="Architecture of LangGraph Checkpointers\nChannel Reducers and BSP Synchronization\nSend API Map-Reduce Workflows\nHuman-in-the-Loop Interrupt Semantics"
    )

    items_list = [line.strip() for line in raw_items.strip().split("\n") if line.strip()]
    st.metric("Detected Dynamic Worker Count", len(items_list))

    if st.button("🚀 Emit Send Packets & Execute Workers", use_container_width=True):
        dispatches = sim.generate_send_dispatches(items_list)
        results = sim.execute_workers(dispatches)
        st.session_state.send_dispatches = dispatches
        st.session_state.send_results = results
        st.success(f"Dispatched and executed {len(items_list)} parallel workers!")

with col_sw_r:
    st.subheader("Dynamic Send Dispatch Packets")
    if "send_dispatches" in st.session_state:
        st.markdown("**Generated `List[Send]` Objects:**")
        for d in st.session_state.send_dispatches:
            st.code(f"Send('{d['target_node']}', {d['payload']})", language="python")
    else:
        st.info("Click 'Emit Send Packets' to view generated dispatch instructions.")

if "send_results" in st.session_state:
    st.markdown("### Aggregated Worker Reducer Results")
    st.dataframe(pd.DataFrame(st.session_state.send_results), use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What is the primary architectural difference between normal fan-out and the Send API?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Normal Fan-Out**: The number of parallel branches is hardcoded statically in the graph topology at compile time via `add_edge(A, B)` and `add_edge(A, C)`. All workers receive the exact same overall state snapshot.
    - **Send API**: The number of parallel branches is determined dynamically at runtime based on state data (e.g. `len(documents)`). Each worker receives an isolated custom payload (`Send("worker", {"doc": d})`).
    """)

with st.expander("Q2: How does the reduce phase work when multiple Send workers emit state updates?", expanded=False):
    st.markdown("""
    **Answer**:
    Because all `Send` workers run in parallel during the same BSP superstep, their output channels **must** use channel reducers (such as `Annotated[list, operator.add]`).
    When workers complete, LangGraph applies the reducer to merge all worker deltas into the central state. The downstream aggregator node then runs in the subsequent superstep.
    """)

with st.expander("Q3: Can a worker node spawned by Send declare a state schema different from the parent graph?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. The worker node only needs to accept the schema of the dictionary passed in `Send(node, arg)`.
    However, the dictionary returned by the worker must match the channel names defined in the overall graph state so reducers can merge the output.
    """)

with st.expander("Q4: Can a conditional edge return both Send objects and standard string transitions?", expanded=False):
    st.markdown("""
    **Answer**:
    A conditional edge must return either:
    1. A single string destination or list of string destinations.
    2. A `Send` object or list of `Send` objects.
    Mixing `Send` objects and bare strings in the same return list is invalid in LangGraph.
    """)

with st.expander("Q5: What happens if the items list is empty and the router returns an empty list [] of Send objects?", expanded=False):
    st.markdown("""
    **Answer**:
    If the router returns `[]` (an empty list of `Send`), LangGraph spawns zero workers.
    Control flow does not enter the worker nodes. If there is a direct edge from the router to an aggregator or fallback, execution proceeds there; otherwise, execution terminates cleanly.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing Channel Reducer on Send Worker Aggregation")
    st.markdown("""
    *Mechanism*: Spawning 10 `Send` workers where workers return updates to an unannotated overwrite channel (e.g. `{"summary": "..."}`).  
    *Consequence*: Crash with `InvalidUpdateError` due to 10 parallel conflicting writes.  
    *Fix*: Always use an append reducer: `summaries: Annotated[list, operator.add]`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Target Node Name Mismatch in Send Constructor")
    st.markdown("""
    *Mechanism*: Returning `Send("summarize", {...})` where the registered node is named `"summarizer"`.  
    *Consequence*: Crash with `ValueError: Node 'summarize' not found in graph`.  
    *Fix*: Use Python constants for node names across `add_node` and `Send`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Unbounded Send Spawning Leading to API Rate Limit Exhaustion")
    st.markdown("""
    *Mechanism*: Uploading a 500-page document and spawning 500 parallel LLM workers via `Send` simultaneously.  
    *Consequence*: Massive HTTP 429 RateLimitError spikes, quota exhaustion, and worker timeouts.  
    *Fix*: Batch items into chunks (e.g. 5–10 items per worker) or enforce semaphore rate-limits.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Heavy State Replication in Send Payloads")
    st.markdown("""
    *Mechanism*: Passing 10 MB of shared context inside the `arg` dictionary of every individual `Send` call.  
    *Consequence*: 10 workers x 10 MB = 100 MB memory spike in a single superstep.  
    *Fix*: Store shared context in central state; pass only unique item IDs in the `Send` payload.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Send API**\\nA primitive for dynamically spawning parallel worker nodes with custom state payloads at runtime.")
with col_fc2:
    st.info("**Map Phase**\\nThe stage where a collection is partitioned into individual tasks and dispatched via `List[Send]`.")
with col_fc3:
    st.info("**Reduce Phase**\\nThe stage where an aggregator node consolidates worker outputs merged via channel reducers.")
with col_fc4:
    st.info("**Payload Isolation**\\nThe capability of `Send` to provide worker nodes with lightweight sub-states rather than full state.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand why Send is necessary for dynamic-count parallel execution.", value=True)
c2 = st.checkbox("I know how to construct Send(node, payload) inside conditional edge functions.", value=True)
c3 = st.checkbox("I understand the requirement for Annotated channel reducers to aggregate Send worker outputs.", value=True)
c4 = st.checkbox("I know how to design isolated state schemas for worker nodes spawned by Send.", value=True)
c5 = st.checkbox("I understand how to prevent API rate limit exhaustion when mapping over large collections.", value=True)
c6 = st.checkbox("I can implement a complete Map-Reduce pipeline using LangGraph's Send API.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 8.1 'Understanding Send' with full architectural and practical comprehension!")
