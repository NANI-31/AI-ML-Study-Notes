import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="25.2 Performance - LangGraph Curriculum",
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
    .badge-perf {
        background-color: #14532D;
        color: #4ADE80;
        border: 1px solid #16A34A;
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
class PerformanceOptimizationSimulator:
    """
    Simulates performance optimizations in LangGraph workflows:
    - Sequential vs parallel node execution speedup
    - Semantic caching hit rates
    - Context window trimming token reduction
    """
    def __init__(self):
        pass

    def calculate_parallel_speedup(
        self,
        node_count: int,
        avg_latency_ms: int,
        use_parallel: bool
    ) -> Dict[str, Any]:
        if use_parallel:
            # Overlap execution with 15ms barrier overhead
            total_time_ms = avg_latency_ms + 15
            concurrency = node_count
        else:
            total_time_ms = node_count * avg_latency_ms
            concurrency = 1
        return {
            "node_count": node_count,
            "total_time_ms": total_time_ms,
            "concurrency": concurrency,
            "speedup_factor": (node_count * avg_latency_ms) / max(total_time_ms, 1)
        }

    def simulate_token_savings(
        self,
        turns: int,
        tokens_per_turn: int,
        enable_trimming: bool,
        enable_prompt_caching: bool
    ) -> Dict[str, Any]:
        # Unoptimized: full cumulative history resent each turn
        cumulative_raw = sum((i + 1) * tokens_per_turn for i in range(turns))
        
        if enable_trimming:
            # Sliding window of 3 turns
            effective_window = 3
            optimized_tokens = sum(min(i + 1, effective_window) * tokens_per_turn for i in range(turns))
        else:
            optimized_tokens = cumulative_raw

        cache_savings = 0
        if enable_prompt_caching:
            # 50% discount on cached system/prefix prompt tokens
            cache_savings = int(optimized_tokens * 0.45)
            effective_tokens = optimized_tokens - cache_savings
        else:
            effective_tokens = optimized_tokens

        reduction_pct = ((cumulative_raw - effective_tokens) / max(cumulative_raw, 1)) * 100
        return {
            "turns": turns,
            "raw_tokens": cumulative_raw,
            "effective_tokens": effective_tokens,
            "saved_tokens": cumulative_raw - effective_tokens,
            "reduction_pct": round(reduction_pct, 1)
        }

perf_sim = PerformanceOptimizationSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: High-Performance Graph Execution, Concurrency, and Optimization Pipeline", divider="rainbow")

st.markdown("""
High-performance LangGraph systems optimize execution along four primary dimensions: **latency** (via parallel fan-out and streaming), 
**cost** (via caching and token minimization), **resource efficiency** (via state schema pruning), and **throughput** (via asynchronous event dispatch). 
By designing graphs around non-blocking superstep barriers and aggressive prompt caching, production systems achieve 
sub-second response latencies and 60-80% token cost reductions.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph PerformanceBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_caching {
            label="1. Ingress & Caching Layer";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            query_in [label="Incoming Query Request", fillcolor="#0F172A"];
            cache_lookup [label="Semantic Cache (Redis / GPTCache)\\nCosine Similarity >= 0.96", fillcolor="#1E293B"];
            heuristic_gate [label="Heuristic Fast Path (Regex / Rule Engine)\\nSkips LLM if intent is deterministic", fillcolor="#14532D", fontcolor="#86EFAC"];
        }

        subgraph cluster_parallel {
            label="2. Parallel Graph Execution (Fan-Out / Fan-In)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            fan_out [label="Fan-Out Dispatcher\\n(START -> Node A, B, C)", fillcolor="#312E81"];
            worker_a [label="Worker A: Web Retriever\\nAsync HTTP (120ms)", fillcolor="#1E293B"];
            worker_b [label="Worker B: Vector DB Search\\nAsync Index Query (85ms)", fillcolor="#1E293B"];
            worker_c [label="Worker C: SQL Data Fetcher\\nAsync Pool Query (95ms)", fillcolor="#1E293B"];
            barrier [label="Superstep Barrier (Fan-In)\\nChannels Merged via Reducers (125ms total)", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_optimization {
            label="3. Model & Memory Optimization";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            trimmer [label="Context Window Trimmer\\nKeeps last N turns / Summarizes older history", fillcolor="#1E293B"];
            llm_stream [label="Prompt Cached LLM Stream\\nastream_mode='messages' (TTFT ~250ms)", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        query_in -> cache_lookup;
        cache_lookup -> heuristic_gate [label="Cache Miss"];
        heuristic_gate -> fan_out [label="Complex Request"];

        fan_out -> worker_a;
        fan_out -> worker_b;
        fan_out -> worker_c;

        worker_a -> barrier;
        worker_b -> barrier;
        worker_c -> barrier;

        barrier -> trimmer;
        trimmer -> llm_stream;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Parallel Execution
# ==============================================================================
st.header("2. Parallel Execution: Fan-Out Supersteps, Async Node Concurrency, and Barrier Synchronization", divider="rainbow")

st.markdown("""
LangGraph natively executes nodes in parallel whenever multiple edges originate from the same source node 
or when multiple nodes are scheduled for the same superstep.

### How Parallel Execution Works:
1. **Fan-Out Dispatch**: The engine evaluates outgoing edges and schedules multiple target nodes simultaneously.
2. **Async Task Execution**: Independent node coroutines run concurrently via `asyncio.gather` under the hood.
3. **Superstep Barrier**: Execution pauses until all concurrent nodes complete. Their individual state updates 
   are aggregated and merged using channel reducers before the next superstep begins.
""")

with st.expander("🔍 View Sequential vs Parallel Execution Comparison", expanded=True):
    fanout_dot = """
    digraph FanOutComparison {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_seq {
            label="Sequential Pipeline (Total Latency: 300ms)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            s1 [label="Fetch Web (100ms)"];
            s2 [label="Fetch SQL (100ms)"];
            s3 [label="Fetch Vector (100ms)"];
            s1 -> s2 -> s3;
        }

        subgraph cluster_par {
            label="Parallel Fan-Out (Total Latency: 110ms - 2.7x Faster)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            p_start [label="Dispatcher (START)"];
            p1 [label="Fetch Web (100ms)"];
            p2 [label="Fetch SQL (100ms)"];
            p3 [label="Fetch Vector (100ms)"];
            p_barrier [label="Fan-In Barrier"];

            p_start -> p1;
            p_start -> p2;
            p_start -> p3;
            p1 -> p_barrier;
            p2 -> p_barrier;
            p3 -> p_barrier;
        }
    }
    """
    st.graphviz_chart(fanout_dot, use_container_width=True)

st.code('''# Building parallel fan-out in LangGraph
workflow = StateGraph(PipelineState)

# Both nodes branch directly from START
workflow.add_edge(START, "fetch_web_data")
workflow.add_edge(START, "fetch_database_data")

# Both nodes converge at the synthesizer barrier
workflow.add_edge("fetch_web_data", "synthesizer")
workflow.add_edge("fetch_database_data", "synthesizer")
''', language="python")

# ==============================================================================
# SECTION 3: Streaming
# ==============================================================================
st.header("3. Streaming: Token-by-Token Delivery, astream Modes, and Real-Time Interactive Responsiveness", divider="rainbow")

st.markdown("""
Streaming transforms user-perceived performance by drastically reducing **Time to First Token (TTFT)**. 
Instead of waiting 3-5 seconds for an entire agent reasoning loop to complete, streaming outputs the initial 
tokens within 250 milliseconds.

### Production Streaming Modes:
- **`messages`**: Streams individual LLM token chunks as they are generated by the model provider.
- **`updates`**: Yields state updates emitted by nodes immediately upon completing their superstep.
- **`values`**: Yields the entire state dictionary after every superstep barrier.
- **`custom`**: Allows nodes to stream custom progress events (e.g., `"Searching database (2/5)..."`).
""")

st.code('''# Asynchronous message streaming for web endpoints
async for msg_chunk, metadata in app.astream(
    {"messages": [HumanMessage(content="Explain quantum computing")]},
    config,
    stream_mode="messages"
):
    if msg_chunk.content:
        # Stream token directly to HTTP response socket
        await response.write(msg_chunk.content)
''', language="python")

# ==============================================================================
# SECTION 4: Caching
# ==============================================================================
st.header("4. Caching: Prompt Caching, Semantic Cache Layers, and Deterministic Tool Output Memoization", divider="rainbow")

st.markdown("""
Caching operates at three distinct tiers within a LangGraph architecture:

1. **Semantic Cache (Ingress)**: Compares incoming user queries against a vector cache of previously answered questions. 
   If cosine similarity exceeds a threshold (e.g., 0.96), return the cached response immediately, bypassing the graph entirely.
2. **Provider Prompt Caching (Model Layer)**: Providers such as Anthropic and OpenAI support prompt caching for static prefixes 
   (system prompts, documentation chunks, tool schemas). Cached input tokens receive a 50-90% cost reduction and 80% latency reduction.
3. **Tool Memoization (Node Layer)**: Deterministic tool queries (such as currency conversion, geographic coordinate lookups, 
   or static reference data) should be memoized in Redis with a TTL to prevent redundant network calls.
""")

# ==============================================================================
# SECTION 5: Reducing LLM Calls
# ==============================================================================
st.header("5. Reducing LLM Calls: Heuristic Routing, Regex Early-Exits, and Deterministic Rules Engines", divider="rainbow")

st.markdown("""
Every LLM call introduces 500-2000 milliseconds of latency and financial cost. Production systems intercept routine queries 
before invoking foundation models:

- **Rule-Based Pre-Routing**: Use lightweight regex or string matching for deterministic commands (e.g., `"help"`, `"reset"`, `"cancel"`).
- **Small Model Classification**: Use small, fast models (e.g., `gpt-4o-mini` or `claude-3-5-haiku`) for routing and intent classification, 
  reserving frontier models for complex multi-step reasoning.
- **Deterministic Tool Calling**: When the user query unambiguously targets a specific tool (e.g., `"track package #12345"`), 
  dispatch the tool directly without asking an LLM to generate the tool call schema.
""")

# ==============================================================================
# SECTION 6: Reducing Token Usage
# ==============================================================================
st.header("6. Reducing Token Usage: Context Trimming, Message Summarization, and Sliding Windows", divider="rainbow")

st.markdown("""
As conversation threads grow, resending hundreds of past messages in the state array inflates cost and slows down model inference.

### Token Reduction Strategies:
- **`trim_messages`**: Utility that retains only the last N tokens or M messages, guaranteeing that system prompts and tool call pairs remain intact.
- **Periodic Summarization**: When message count exceeds a threshold (e.g., 20 messages), trigger a summarizer node that collapses older turns into a concise running summary and purges the raw messages.
""")

st.code('''from langchain_core.messages import trim_messages

def agent_node_with_trimming(state: AgentState) -> dict:
    # Retain only the last 3000 tokens of conversation history
    trimmed = trim_messages(
        state["messages"],
        max_tokens=3000,
        strategy="last",
        token_counter=len,  # Or model tokenizer
        include_system=True,
        allow_partial=False
    )
    response = model.invoke(trimmed)
    return {"messages": [response]}
''', language="python")

# ==============================================================================
# SECTION 7: State Optimization
# ==============================================================================
st.header("7. State Optimization: Minimal State Schemas, Shallow Copies, and Pruning Ephemeral Channels", divider="rainbow")

st.markdown("""
In LangGraph, state is serialized and written to the checkpointer at every superstep. Inefficient state structures 
degrade persistence throughput and inflate database IOPS:

1. **Avoid Bloated Schemas**: Only store data required by downstream nodes or user presentation. 
   Do not pass large raw JSON payloads between nodes if only two fields are used.
2. **Ephemeral Scratchpads**: Use separate temporary channels that are cleared at the end of a superstep to avoid 
   persisting transient intermediate calculations into long-term checkpoints.
3. **Immutability & Zero Deep Copies**: Avoid expensive deep-copy operations inside node functions; return minimal delta dictionaries.
""")

# ==============================================================================
# SECTION 8: Memory Optimization
# ==============================================================================
st.header("8. Memory Optimization: Thread-Scoped Memory vs Cross-Session Long-Term Memory Compression", divider="rainbow")

st.markdown("""
Managing agent memory efficiently requires separating short-term thread memory from long-term memory:

- **Short-Term Memory (Thread-Scoped)**: Kept compact via message trimming and checkpoint TTL expiration. 
  Archived or purged after sessions conclude.
- **Long-Term Memory (User-Scoped)**: Extracted asynchronously into structured semantic profiles (e.g., user preferences, 
  recurring entities) stored in an external key-value store, rather than loading cumulative history into the active graph state.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Performance-Optimized Graph Pipeline with Caching and Token Budgeting", divider="rainbow")

st.markdown("""
The following complete module implements an optimized agent graph featuring parallel data fetching, 
sliding window message trimming, and prompt caching flags.
""")

st.code('''from typing import TypedDict, Annotated, List
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages

class OptimizedState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    web_context: str
    db_context: str
    token_budget: int

# Parallel Node 1
async def fetch_web_node(state: OptimizedState) -> dict:
    await asyncio.sleep(0.08)  # Simulated async fetch
    return {"web_context": "Web search: Market trending up 4.2%"}

# Parallel Node 2
async def fetch_db_node(state: OptimizedState) -> dict:
    await asyncio.sleep(0.09)  # Simulated async DB query
    return {"db_context": "DB Query: User balance = 14,250 USD"}

# Synthesizer Node with context trimming
def synthesizer_node(state: OptimizedState) -> dict:
    # Trim conversation history to prevent token explosion
    compact_messages = trim_messages(
        state["messages"],
        max_tokens=2000,
        strategy="last",
        include_system=True
    )
    
    # In production, pass compact_messages + contexts to LLM
    final_text = f"Synthesized from web ({state['web_context']}) and db ({state['db_context']})"
    return {"messages": [AIMessage(content=final_text)]}

def create_optimized_graph():
    builder = StateGraph(OptimizedState)
    
    builder.add_node("fetch_web", fetch_web_node)
    builder.add_node("fetch_db", fetch_db_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    # Fan-Out from START in parallel
    builder.add_edge(START, "fetch_web")
    builder.add_edge(START, "fetch_db")
    
    # Fan-In at synthesizer barrier
    builder.add_edge("fetch_web", "synthesizer")
    builder.add_edge("fetch_db", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    return builder.compile()
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Performance & Token Budget Simulator, Parallel Speedup Calculator", divider="rainbow")

st.markdown("""
Calculate latency speedups from parallel execution and estimate token and cost savings from prompt caching and context trimming.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Parallel Execution Speedup Calculator")
    parallel_nodes = st.slider("Concurrent Node Count (Fan-Out)", min_value=2, max_value=8, value=3)
    avg_latency = st.slider("Average Node Latency (ms)", min_value=50, max_value=500, value=150, step=25)
    
    seq_metrics = perf_sim.calculate_parallel_speedup(parallel_nodes, avg_latency, use_parallel=False)
    par_metrics = perf_sim.calculate_parallel_speedup(parallel_nodes, avg_latency, use_parallel=True)
    
    st.markdown(f"**Sequential Execution Duration:** `{seq_metrics['total_time_ms']} ms`")
    st.markdown(f"**Parallel Execution Duration:** `{par_metrics['total_time_ms']} ms`")
    st.metric("Latency Speedup Factor", f"{par_metrics['speedup_factor']:.2f}x Faster")

with col_wb2:
    st.subheader("Token Optimization & Cost Estimator")
    conv_turns = st.slider("Conversation Turns", min_value=2, max_value=20, value=8)
    tokens_per_turn = st.slider("Tokens Generated per Turn", min_value=200, max_value=1500, value=500, step=50)
    
    opt_trim = st.toggle("Enable Sliding Window Trimming (Last 3 Turns)", value=True)
    opt_cache = st.toggle("Enable Provider Prompt Caching (Prefix Discount)", value=True)
    
    token_results = perf_sim.simulate_token_savings(conv_turns, tokens_per_turn, opt_trim, opt_cache)
    
    st.metric("Raw Cumulative Tokens", f"{token_results['raw_tokens']:,}")
    st.metric("Optimized Cumulative Tokens", f"{token_results['effective_tokens']:,}")
    st.success(f"Total Token Reduction: {token_results['reduction_pct']}% saved ({token_results['saved_tokens']:,} tokens)")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How does LangGraph handle state conflicts when two nodes execute in parallel during the same superstep?"):
    st.markdown("""
    **Answer:**  
    If two parallel nodes write to the same state channel, that channel **must** define a reducer (such as `Annotated[list, add_messages]` or `Annotated[dict, custom_merge]`). 
    At the superstep barrier, LangGraph invokes the reducer sequentially with each node's return value. If an updated channel has no reducer, LangGraph raises an `InvalidUpdateError`.
    """)

with st.expander("Q2: What is the architectural difference between stream_mode='messages' and stream_mode='updates'?"):
    st.markdown("""
    **Answer:**  
    - `stream_mode="messages"` yields fine-grained token chunks emitted during LLM generation, ideal for typing effects and low TTFT.  
    - `stream_mode="updates"` yields dictionary deltas emitted by nodes only after a complete node or superstep finishes, ideal for updating UI cards or progress indicators.
    """)

with st.expander("Q3: How does Anthropic or OpenAI prompt caching interact with dynamic LangGraph state?"):
    st.markdown("""
    **Answer:**  
    Prompt caching requires an exact prefix match on input tokens. If dynamic values (such as timestamps or counters) are inserted at the beginning of the prompt, 
    the cache key is invalidated. Best practice is to place static system prompts and tool schemas at the beginning, followed by cached document blocks, 
    and append dynamic state messages strictly at the end.
    """)

with st.expander("Q4: Why can't you simply run a Python thread pool inside standard synchronous LangGraph nodes for parallelism?"):
    st.markdown("""
    **Answer:**  
    Manual thread pools inside nodes obscure execution from the LangGraph engine, preventing the checkpointer from creating deterministic superstep checkpoints, 
    bypassing channel reducers, and corrupting execution lineage and time-travel rollbacks. Native LangGraph graph branching must be used instead.
    """)

with st.expander("Q5: What is the performance impact of storing large documents in graph state vs storing document IDs?"):
    st.markdown("""
    **Answer:**  
    Storing raw text documents in state channels causes high serialization overhead on every superstep, increases database IOPS for the checkpointer, 
    and balloons network payload sizes. Best practice is to pass document IDs or signed URLs in state and retrieve content on-demand.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Unreduced Channels in Parallel Fan-Out Graphs**  
Configuring two nodes to execute in parallel when both return a dictionary updating the same unreduced state key causes `InvalidUpdateError` 
at runtime, halting graph execution.
""")

st.error("""
**Trap 2: Cache Key Invalidation from Dynamic Timestamps in System Prompts**  
Injecting current UTC timestamps into the system prompt prefix invalidates provider prompt caches on every request, completely eliminating caching cost benefits.
""")

st.warning("""
**Trap 3: Memory Leaks from Unbounded Message Channels**  
Allowing conversation threads to grow without message trimming or summarization leads to exponential token costs and eventual context window overflow exceptions.
""")

st.warning("""
**Trap 4: Blocking the Event Loop with Synchronous HTTP Requests**  
Using synchronous `requests.get()` inside async graph nodes blocks the main asyncio thread, nullifying the throughput advantages of concurrent supersteps.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Fan-Out / Fan-In</h4>
        <p>Branching execution to multiple nodes in parallel from a single source node, converging at a common barrier node with channel reducers.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Time to First Token (TTFT)</h4>
        <p>The duration between user request dispatch and the receipt of the first streamed completion token, critical for perceived performance.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Provider Prompt Caching</h4>
        <p>Model layer optimization caching static prompt prefixes to provide up to 90% discount on input tokens and 80% latency reduction.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>trim_messages</h4>
        <p>A LangChain utility function used to enforce token budgets by intelligently truncating message lists while keeping system context intact.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to design parallel fan-out and fan-in workflows with channel reducers.", value=True)
chk_2 = st.checkbox("I understand the difference between messages, updates, and values streaming modes.", value=True)
chk_3 = st.checkbox("I can implement prompt caching strategies by structuring static prefix tokens correctly.", value=True)
chk_4 = st.checkbox("I know how to reduce LLM calls using heuristic pre-routing and fast classifier models.", value=True)
chk_5 = st.checkbox("I can manage context windows and token budgets using trim_messages and summarization.", value=True)
chk_6 = st.checkbox("I understand how to optimize state schemas and memory by storing references instead of raw payloads.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Performance and concurrency optimization mastery confirmed! Ready for security and authorization engineering.")
