import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="8.2 Map-Reduce - LangGraph Curriculum",
    page_icon="🗺️",
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
# MODULE-SCOPE MAP-REDUCE ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class MapReduceExecutionEngine:
    """
    Simulates the classical distributed Map-Reduce lifecycle:
    Partitioning (Map Ingress) -> Parallel Worker Tasks -> Barrier Reduce Aggregation.
    """
    def __init__(self):
        self.aggregation_modes = ["Concatenation", "Consensus Voting", "Hierarchical Summary"]

    def run_map_reduce(self, documents: List[str], agg_mode: str) -> Dict[str, Any]:
        """
        Executes map phase over documents and reduces them into a unified output.
        """
        # Map Phase: Parallel simulated workers
        map_outputs = []
        for idx, doc in enumerate(documents, 1):
            char_len = len(doc)
            words = len(doc.split())
            key_points = [w for w in doc.split() if len(w) > 5][:3]
            map_outputs.append({
                "worker_id": f"map_worker_{idx}",
                "input_snippet": doc[:40] + ("..." if len(doc) > 40 else ""),
                "metrics": {"chars": char_len, "words": words},
                "extracted_points": key_points,
                "partial_summary": f"Chunk {idx} focuses on {', '.join(key_points) if key_points else 'general topics'}."
            })

        # Reduce Phase: Consolidation
        if agg_mode == "Concatenation":
            final_text = " | ".join(m["partial_summary"] for m in map_outputs)
        elif agg_mode == "Consensus Voting":
            all_pts = [p for m in map_outputs for p in m["extracted_points"]]
            final_text = f"Consensus topics identified across {len(documents)} workers: {list(set(all_pts))}"
        else: # Hierarchical Summary
            total_words = sum(m["metrics"]["words"] for m in map_outputs)
            final_text = f"Consolidated Executive Summary across {len(documents)} documents ({total_words} total words processed)."

        return {
            "worker_count": len(documents),
            "map_phase_results": map_outputs,
            "reduce_phase_summary": final_text,
            "aggregation_strategy": agg_mode
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Map-Reduce Distributed Pattern in LangGraph", divider="rainbow")

st.markdown("""
**Map-Reduce** is the quintessential distributed computing architecture for big data and generative AI.
In LangGraph, Map-Reduce translates into three formal stages:
1. **Map Ingress (Splitter)**: A planner node decomposes an input corpus into N autonomous tasks.
2. **Parallel Map Phase (`Send`)**: N independent worker instances execute in parallel during a single superstep, evaluating against isolated sub-states.
3. **Reduce Phase (Aggregator)**: Channel reducers gather worker outputs across a barrier synchronization join, and a reducer node synthesizes the consolidated answer.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    mapreduce_blueprint_dot = """
    digraph MapReduceArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress & Partitioning (Planner)";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nCorpus / Unstructured Data", fillcolor="#1E293B", shape=ellipse];
            Partition_Node [label="Partitioning Node\\nSplits corpus into chunks\\nstate['chunks'] = [c1, c2, ..., cN]", fillcolor="#0284C7"];
        }

        subgraph cluster_map_phase {
            label="2. Parallel Map Phase (Dynamic Send Workers)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Worker1 [label="Worker 1: summarize_chunk(c1)\\nEmits: {'summaries': [s1]}", fillcolor="#4338CA"];
            Worker2 [label="Worker 2: summarize_chunk(c2)\\nEmits: {'summaries': [s2]}", fillcolor="#4338CA"];
            WorkerN [label="Worker N: summarize_chunk(cN)\\nEmits: {'summaries': [sN]}", fillcolor="#4338CA"];
        }

        subgraph cluster_reduce_phase {
            label="3. Barrier Reduce Phase (Synthesis)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Reducer_Hub [label="Channel Reducer Barrier\\nsummaries: Annotated[list, operator.add]\\nCollects [s1, s2, ..., sN]", fillcolor="#0D9488"];
            Reduce_Node [label="Synthesizer Node (Reduce)\\nFuses partial summaries into master report", fillcolor="#059669"];
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Partition_Node;
        Partition_Node -> Worker1 [label="Send('worker', {'chunk': c1})"];
        Partition_Node -> Worker2 [label="Send('worker', {'chunk': c2})"];
        Partition_Node -> WorkerN [label="Send('worker', {'chunk': cN})"];
        Worker1 -> Reducer_Hub;
        Worker2 -> Reducer_Hub;
        WorkerN -> Reducer_Hub;
        Reducer_Hub -> Reduce_Node [label="Consolidated list ready"];
        Reduce_Node -> END;
    }
    """
    st.graphviz_chart(mapreduce_blueprint_dot, use_container_width=True)

st.markdown("""
### Why Map-Reduce is Critical for LLMs
Language models possess finite context windows and suffer from the **'Lost in the Middle'** attention degradation
when prompts exceed tens of thousands of tokens. Map-Reduce enables boundless scaling: documents are summarized in parallel chunks,
and only high-signal extracted insights are passed to the final reduce stage.
""")

# ==============================================================================
# SECTION 2: MAP PHASE
# ==============================================================================
st.header("2. Map Phase: Partitioning Collections into Independent Tasks", divider="rainbow")

st.markdown("""
The **Map Phase** begins by decomposing a complex problem into independent, non-overlapping atomic units.
In LangGraph, this is implemented using a conditional edge that returns a list of `Send` objects.
""")

code_map_phase = """from langgraph.types import Send

def map_dispatcher(state: OverallState) -> list[Send]:
    chunks = state["text_chunks"]
    # Return N dynamic Send operations:
    return [
        Send("map_worker_node", {"chunk_id": idx, "text": chunk})
        for idx, chunk in enumerate(chunks)
    ]
"""
st.code(code_map_phase, language="python")

# ==============================================================================
# SECTION 3: PARALLEL WORKERS
# ==============================================================================
st.header("3. Parallel Workers: Concurrent Execution and Resource Management", divider="rainbow")

st.markdown("""
All worker instances generated by the map phase execute concurrently within a single LangGraph superstep.
In async runtimes (`ainvoke()`, `astream()`), LangGraph dispatches workers as concurrent asyncio coroutines.
""")

st.markdown("""
#### Resource Management Best Practices:
- **Rate-Limiting**: If spawning 50+ workers that call OpenAI/Anthropic APIs, configure client-side token bucket limiters to prevent HTTP 429 throttling.
- **Worker Timeouts**: Ensure individual worker nodes have internal timeouts to prevent a single slow API call from stalling the entire barrier join.
""")

# ==============================================================================
# SECTION 4: WORKER-SPECIFIC STATE
# ==============================================================================
st.header("4. Worker-Specific State: Scoped Schemas and Memory Isolation", divider="rainbow")

st.markdown("""
A foundational design principle of Map-Reduce is that workers should only have access to their own partition.
Workers should NOT have access to sibling data or the entire document corpus.
""")

code_worker_state = """from typing import TypedDict

# Isolated worker input contract
class MapWorkerInput(TypedDict):
    chunk_id: int
    text: str

def map_worker_node(state: MapWorkerInput) -> dict:
    # Worker is scoped exclusively to its assigned chunk
    extracted = extract_entities(state["text"])
    return {
        "collected_entities": extracted # Appends to parent reducer channel
    }
"""
st.code(code_worker_state, language="python")

# ==============================================================================
# SECTION 5: REDUCE PHASE
# ==============================================================================
st.header("5. Reduce Phase: Barrier Synchronization and Reducer Aggregations", divider="rainbow")

st.markdown("""
The **Reduce Phase** begins only after every single map worker has finished.
LangGraph's BSP barrier guarantees that the downstream aggregator node never runs on partial or incomplete data.
""")

st.markdown("""
#### The Role of Channel Reducers in Reduce:
The parent graph schema MUST declare the target channel with a reducer:
```python
class OverallState(TypedDict):
    # operator.add merges worker outputs into a single consolidated list!
    collected_entities: Annotated[List[str], operator.add]
```
""")

# ==============================================================================
# SECTION 6: AGGREGATING RESULTS
# ==============================================================================
st.header("6. Aggregating Results: Structured Reductions, Consensus, and Hierarchical Merges", divider="rainbow")

st.markdown("""
Depending on your AI task, the reduce node can apply different aggregation strategies:
""")

col_ag1, col_ag2, col_ag3 = st.columns(3)
with col_ag1:
    st.markdown("""
    #### 1. Concatenation
    Appends summaries or facts into a coherent multi-section report.
    """)
with col_ag2:
    st.markdown("""
    #### 2. Consensus Voting
    Compares candidate answers across workers and selects the majority verdict.
    """)
with col_ag3:
    st.markdown("""
    #### 3. Hierarchical Merge
    If the collected list is still too large, recursively maps over sub-summaries.
    """)

# ==============================================================================
# SECTION 7: DYNAMIC NUMBER OF WORKERS
# ==============================================================================
st.header("7. Dynamic Number of Workers: Handling Elastic Workloads from 0 to N", divider="rainbow")

st.markdown("""
Because `Send` is evaluated dynamically at runtime, your graph handles elastic workloads gracefully:
- **0 Items**: Router returns `[]`, bypassing workers and proceeding to early termination.
- **1 Item**: Router returns 1 `Send`, executing with minimal overhead.
- **100 Items**: Router returns 100 `Send` instances, executing massive parallel fan-out.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Document Map-Reduce Summarizer", divider="rainbow")

st.markdown("""
Below is a complete production pipeline showing chunk partitioning, dynamic `Send` map execution,
and barrier reduce aggregation.
""")

prod_mapreduce_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# Step 1: State Contracts
class CorpusState(TypedDict):
    corpus: str
    chunks: List[str]
    partial_summaries: Annotated[List[str], operator.add]
    master_summary: str

class ChunkWorkerState(TypedDict):
    chunk_index: int
    content: str

# Step 2: Nodes
def partition_node(state: CorpusState) -> Dict[str, Any]:
    raw = state["corpus"]
    # Partition into paragraphs
    paragraphs = [p.strip() for p in raw.split("\\n\\n") if p.strip()]
    return {"chunks": paragraphs}

def worker_node(state: ChunkWorkerState) -> Dict[str, Any]:
    # Parallel Map Worker
    idx = state["chunk_index"]
    words = len(state["content"].split())
    summary = f"Section {idx} ({words} words): Key insights extracted."
    return {"partial_summaries": [summary]}

def reduce_node(state: CorpusState) -> Dict[str, Any]:
    # Barrier Reduce Aggregator
    all_summaries = state["partial_summaries"]
    consolidated = " | ".join(all_summaries)
    return {"master_summary": f"Executive Summary across {len(all_summaries)} sections: {consolidated}"}

# Step 3: Map Router
def map_chunks_router(state: CorpusState) -> List[Send]:
    return [
        Send("worker", {"chunk_index": i + 1, "content": chunk})
        for i, chunk in enumerate(state["chunks"])
    ]

# Step 4: Build Graph
def create_mapreduce_graph():
    builder = StateGraph(CorpusState)
    
    builder.add_node("partition", partition_node)
    builder.add_node("worker", worker_node)
    builder.add_node("reduce", reduce_node)
    
    builder.add_edge(START, "partition")
    builder.add_conditional_edges("partition", map_chunks_router)
    builder.add_edge("worker", "reduce")
    builder.add_edge("reduce", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = create_mapreduce_graph()
    sample_text = """Paragraph 1: Generative AI workflows require robust state management.

Paragraph 2: LangGraph implements the Bulk Synchronous Parallel execution model.

Paragraph 3: The Send API enables elastic map-reduce scaling across dynamic workloads."""

    out = app.invoke({
        "corpus": sample_text,
        "chunks": [],
        "partial_summaries": [],
        "master_summary": ""
    })
    print("Master Summary:", out["master_summary"])
    print("Partial Summaries Count:", len(out["partial_summaries"]))
'''
st.code(prod_mapreduce_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Map-Reduce Pipeline Simulator and Execution Inspector", divider="rainbow")

st.markdown("""
Test the Map-Reduce lifecycle interactively. Paste or modify document passages, choose an aggregation strategy,
and inspect the parallel map outputs versus the reduced consolidation.
""")

engine = MapReduceExecutionEngine()

col_mr_l, col_mr_r = st.columns([1, 1.2])

with col_mr_l:
    st.subheader("Configure Input Passages")
    raw_passages = st.text_area(
        "Passages to Map-Reduce (Separate with blank lines)",
        value="""Passage 1: Vector databases index high-dimensional embeddings for similarity retrieval.

Passage 2: Chunking strategies must preserve semantic boundaries to avoid fragmenting ideas.

Passage 3: Cross-encoder rerankers dramatically boost top-k retrieval precision."""
    )
    
    chosen_agg = st.selectbox("Reduce Aggregation Strategy", engine.aggregation_modes)
    passages_list = [p.strip() for p in raw_passages.strip().split("\n\n") if p.strip()]
    st.metric("Total Map Workers to Spawn", len(passages_list))

    if st.button("🚀 Execute Map-Reduce Job", use_container_width=True):
        st.session_state.mr_run = engine.run_map_reduce(passages_list, chosen_agg)
        st.success("Map-Reduce job executed successfully!")

with col_mr_r:
    st.subheader("Reduce Phase Consolidated Output")
    if "mr_run" in st.session_state:
        st.info(st.session_state.mr_run["reduce_phase_summary"])
        
        st.markdown("**Map Phase Worker Outputs:**")
        df_workers = pd.DataFrame([
            {
                "Worker ID": m["worker_id"],
                "Snippet": m["input_snippet"],
                "Words": m["metrics"]["words"],
                "Partial Summary": m["partial_summary"]
            }
            for m in st.session_state.mr_run["map_phase_results"]
        ])
        st.dataframe(df_workers, use_container_width=True)
    else:
        st.info("Click 'Execute Map-Reduce Job' to run the pipeline.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does Map-Reduce in LangGraph prevent 'Lost in the Middle' context degradation?", expanded=False):
    st.markdown("""
    **Answer**:
    When an LLM receives a prompt containing 50 pages of raw text, empirical studies show that information in the middle 60% of the prompt is overlooked.
    Map-Reduce chunks the text and processes each chunk through a dedicated worker LLM call. The reduce node only receives the distilled high-signal insights, ensuring that every section receives equal cognitive attention.
    """)

with st.expander("Q2: What is the barrier synchronization guarantee in LangGraph Map-Reduce?", expanded=False):
    st.markdown("""
    **Answer**:
    In accordance with Bulk Synchronous Parallelism (BSP), if a map phase spawns 10 workers, the reduce node will not execute until **all 10 workers have emitted their deltas** and the channel reducers have merged the outputs. There is zero risk of the reduce node executing on partial data.
    """)

with st.expander("Q3: How do you handle a worker failure inside a Map-Reduce pipeline?", expanded=False):
    st.markdown("""
    **Answer**:
    1. Attach a `RetryPolicy` to the worker node to handle transient API exceptions automatically.
    2. In the worker node, catch exceptions and return a structured failure delta: `{"partial_summaries": [{"status": "FAILED", "id": chunk_id}]}`.
    3. The reduce node inspects the summaries and gracefully synthesizes the report using the remaining successful chunks.
    """)

with st.expander("Q4: Can a Map-Reduce pipeline recursively reduce when there are too many map outputs?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. If the map phase produces 100 summaries (still too large for one prompt), you can implement **Hierarchical Tree Summarization**: a conditional edge checks `if len(summaries) > 10: return Send('sub_reducer', ...)`. This reduces 100 items to 10, and then 10 to 1.
    """)

with st.expander("Q5: Why must the reduce channel use an Annotated reducer rather than a standard list?", expanded=False):
    st.markdown("""
    **Answer**:
    Because all map workers run concurrently during the exact same superstep, multiple nodes are returning updates for the same key at the exact same moment. Without `Annotated[list, operator.add]`, LangGraph would detect conflicting concurrent writes and crash with `InvalidUpdateError`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting operator.add on the Reduction Channel")
    st.markdown("""
    *Mechanism*: Declaring `summaries: List[str]` in state instead of `Annotated[List[str], operator.add]`.  
    *Consequence*: Crash with `InvalidUpdateError` as soon as two workers complete in parallel.  
    *Fix*: Always wrap reduction collections in `Annotated[list, operator.add]`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Single Worker Timeout Stalling Entire Barrier")
    st.markdown("""
    *Mechanism*: 9 out of 10 workers complete in 1s, but 1 worker hangs indefinitely without a timeout.  
    *Consequence*: The reduce node never executes, causing the graph invocation to hang until client timeout.  
    *Fix*: Set strict client-side timeouts on LLM and HTTP calls inside the worker node.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Chunking into Millions of Micro-Tasks")
    st.markdown("""
    *Mechanism*: Splitting a 1,000-word essay into sentence-level chunks, spawning 100 workers.  
    *Consequence*: High overhead, excessive API connection setup latency, and high cost.  
    *Fix*: Chunk into substantial paragraphs (500–1,500 tokens) to balance parallelism with throughput.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Information Loss in Non-Extractive Reducers")
    st.markdown("""
    *Mechanism*: Instructing the reduce node to compress 20 detailed technical reports into 2 sentences.  
    *Consequence*: Critical nuances, citations, and quantitative figures are completely dropped.  
    *Fix*: Prompt the reducer to synthesize structured tables or preserve key quantitative citations.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Map-Reduce**\\nA distributed architecture partitioning tasks across parallel workers and aggregating results via reducers.")
with col_fc2:
    st.info("**Map Ingress**\\nThe planning node that splits a corpus into discrete chunks for dynamic dispatch.")
with col_fc3:
    st.info("**Barrier Join**\\nThe BSP guarantee where the reduce node waits until all active map workers finish.")
with col_fc4:
    st.info("**Hierarchical Reduction**\\nA multi-tier tree reduction pattern for summarizing massive document collections.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural three-stage lifecycle of Map-Reduce in LangGraph.", value=True)
c2 = st.checkbox("I know how to partition collections and dispatch workers using Send().", value=True)
c3 = st.checkbox("I understand why worker-specific state isolation protects concurrency.", value=True)
c4 = st.checkbox("I know how to configure Annotated[list, operator.add] to collect worker deltas.", value=True)
c5 = st.checkbox("I can implement multiple reduction strategies (Concatenation, Consensus, Hierarchical).", value=True)
c6 = st.checkbox("I understand how to handle worker timeouts to prevent stalling the barrier join.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 8.2 'Map-Reduce' with full architectural and practical comprehension!")
