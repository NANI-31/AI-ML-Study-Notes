import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="39.2 Performance Best Practices - LangChain",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for performance telemetry
st.markdown("""
<style>
    .perf-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-speedup {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class PerformanceOptimizationSimulator:
    """
    Simulates high-throughput, low-latency performance optimizations:
    cache hits, parallel tool dispatches, streaming time-to-first-token (TTFT),
    and prompt token reduction strategies.
    """
    def __init__(self):
        self.scenarios = {
            "Naive Unoptimized Pipeline": {
                "llm_calls": 5,
                "caching_active": False,
                "tool_execution": "Sequential (Synchronous)",
                "streaming_active": False,
                "prompt_tokens": 4200,
                "ttft_ms": 3200,
                "total_latency_ms": 6800,
                "estimated_cost_per_1k": 0.14
            },
            "Optimized Enterprise Architecture": {
                "llm_calls": 2,
                "caching_active": True,
                "tool_execution": "Parallel (RunnableParallel / asyncio.gather)",
                "streaming_active": True,
                "prompt_tokens": 850,
                "ttft_ms": 280,
                "total_latency_ms": 1150,
                "estimated_cost_per_1k": 0.03
            }
        }

    def simulate_pipeline(self, scenario_key: str) -> Dict[str, Any]:
        return self.scenarios.get(scenario_key, self.scenarios["Optimized Enterprise Architecture"])


# --- LESSON HEADER ---
st.title("39.2 LangChain Performance Best Practices")
st.markdown("""
Generative AI applications are constrained by two primary engineering bottlenecks: high latency and compounding token costs.
Achieving sub-second response times and operational scale requires eliminating redundant model calls through caching,
streaming tokens to minimize perceived latency, executing independent tools in parallel, and aggressively optimizing
prompt lengths and retrieval chunk sizes.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: High-Throughput & Low-Latency AI Systems Optimization", divider="rainbow")
st.markdown("""
A high-performance LangChain architecture establishes multi-tiered optimizations across the entire execution lifecycle.
Incoming queries hit exact or semantic caches first; un-cached queries are triaged with lightweight deterministic heuristics;
necessary model calls stream tokens immediately; and independent tools execute concurrently via asynchronous gather operations.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Performance_Best_Practices_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_ingress_cache {
            label = "Tier 1: Ingress & Caching Gateway";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            IncomingQuery [label="User Query", fillcolor="#0F172A", color="#3B82F6"];
            ExactCache [label="Redis Exact Match Cache\\n(O(1) Hash Lookup: SHA256(prompt))", fillcolor="#1E293B", color="#60A5FA"];
            SemanticCache [label="Vector Semantic Cache\\n(Cosine similarity >= 0.96 threshold)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_triage_routing {
            label = "Tier 2: Triage & Model Routing";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            TriageRouter [label="Deterministic Intent Router\\n(Regex / Keyword / Lightweight SLM)", fillcolor="#0369A1", color="#38BDF8"];
            PromptOptimizer [label="Prompt Token Pruner\\n(System Prompt Minifier & Dynamic Context)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_parallel_execution {
            label = "Tier 3: Asynchronous Parallel Engine";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ParallelDispatcher [label="RunnableParallel / asyncio.gather\\n(Dispatches Tool A, B, and C simultaneously)", fillcolor="#0369A1", color="#38BDF8"];
            HybridRetriever [label="Optimized Hybrid Retrieval\\n(BM25 + Dense Vectors + Top-K Pruning)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_streaming_delivery {
            label = "Tier 4: Streaming & Delivery";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            TokenStreamer [label="Real-Time Token Streamer\\n(astream / TTFT < 300ms)", fillcolor="#065F46", color="#10B981"];
            ClientUI [label="Responsive Client (React / Streamlit)\\n(Immediate First-Token Display)", fillcolor="#065F46", color="#10B981"];
        }

        IncomingQuery -> ExactCache [label="1. Check Cache"];
        ExactCache -> ClientUI [label="Cache Hit (10ms)"];
        ExactCache -> SemanticCache [label="Cache Miss"];
        SemanticCache -> ClientUI [label="Semantic Match (50ms)"];
        SemanticCache -> TriageRouter [label="Full Miss"];
        TriageRouter -> PromptOptimizer [label="Prepare Minified Context"];
        PromptOptimizer -> ParallelDispatcher [label="Parallel Tool Needs"];
        PromptOptimizer -> HybridRetriever [label="Parallel Retrieval"];
        ParallelDispatcher -> TokenStreamer [label="Feed Synthesizer"];
        HybridRetriever -> TokenStreamer [label="Feed Context"];
        TokenStreamer -> ClientUI [label="Streamed Response"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: AVOID UNNECESSARY LLM CALLS ---
st.header("2. Eliminating Unnecessary LLM Invocations: Deterministic Heuristics & Triage Routers", divider="rainbow")
st.markdown("""
The fastest and cheapest LLM call is the one that is never made. In enterprise applications, up to 40% of user queries
(greetings, basic FAQ lookups, navigation requests, or invalid inputs) can be handled deterministically using regex pattern matching,
static dictionaries, or lightweight rules engines before reaching the foundation model.
""")

col_llm1, col_llm2 = st.columns(2)
with col_llm1:
    st.markdown("### Triage Strategies")
    st.markdown("""
    - **Keyword & Regex Fast-Paths**: Trap common inputs like 'help', 'pricing', 'status', and greetings with zero LLM latency.
    - **Guardrail Pre-Validation**: Run local regex or Pydantic validation before LLM calls to reject malformed inputs without spending tokens.
    - **Model Tiering / Cascading**: Route simple routing tasks to smaller, faster models (e.g. `gpt-4o-mini` or local SLMs) rather than monolithic frontier models.
    """)
with col_llm2:
    st.code("""
from typing import Optional

STATIC_FAQ = {
    "what are your support hours?": "Support is available Monday-Friday, 9 AM to 6 PM EST.",
    "where is your office located?": "Our headquarters is located at 100 Enterprise Way, Boston, MA."
}

def fast_triage_router(query: str) -> Optional[str]:
    normalized = query.strip().lower()
    # 1. Deterministic dictionary check (O(1) execution)
    if normalized in STATIC_FAQ:
        return STATIC_FAQ[normalized]
    # 2. Basic greeting regex check
    if normalized in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you with your enterprise account today?"
    # Return None to signify full LLM agent execution required
    return None
    """, language="python")

# --- SECTION 3: USE STREAMING ---
st.header("3. End-to-End Streaming Pipelines: Token-Level Streaming and Time-to-First-Token (TTFT)", divider="rainbow")
st.markdown("""
Users perceive latency based on Time-to-First-Token (TTFT), not the time taken to complete the entire generation.
A 30-token response that streams its first token in 250 milliseconds feels instantaneous, whereas waiting 3 seconds
for the full completion causes user frustration. LangChain enables streaming across all Runnables via `.stream()` and `.astream()`.
""")

st.code("""
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", streaming=True)
prompt = ChatPromptTemplate.from_template("Explain the benefits of {topic} concisely.")
parser = StrOutputParser()

chain = prompt | llm | parser

# Asynchronous token-by-token streaming
async def stream_tokens(topic: str):
    async for chunk in chain.astream({"topic": topic}):
        # Immediate yield to WebSocket, SSE, or UI buffer
        print(chunk, end="", flush=True)

# In FastAPI: Return StreamingResponse(chain.astream(...), media_type="text/event-stream")
""", language="python")

# --- SECTION 4: USE CACHING ---
st.header("4. Multi-Tiered Caching: In-Memory Exact Match vs Vector Semantic Caching", divider="rainbow")
st.markdown("""
Caching prevents redundant computation for repeated or semantically equivalent questions.
LangChain supports global exact match caching (`InMemoryCache`, `RedisCache`) as well as semantic distance caching (`OpenAICache` / VectorDB).
""")

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.markdown("### Exact Match vs Semantic Caching")
    st.markdown("""
    - **Exact Match (`RedisCache`)**: Keys are exact hashes of prompt messages. Microsecond response time, zero false positives.
    - **Semantic Cache (`RedisSemanticCache`)**: Computes vector embedding of prompt and returns cached answers if cosine similarity > 0.95.
    - **Best Practice**: Use exact match caching for customer service queries and static document Q&A to minimize vector store overhead.
    """)
with col_c2:
    st.code("""
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Enable global LLM caching in LangChain
# Subsequent identical invocations return in 0.1ms with zero API cost
set_llm_cache(InMemoryCache())

# In production with Redis:
# from langchain_community.cache import RedisCache
# import redis
# redis_client = redis.Redis.from_url("redis://localhost:6379/0")
# set_llm_cache(RedisCache(redis_ = redis_client))
    """, language="python")

# --- SECTION 5: USE PARALLEL EXECUTION ---
st.header("5. Concurrent & Parallel Tool Execution: RunnableParallel and Non-Blocking I/O", divider="rainbow")
st.markdown("""
Executing multiple tools sequentially creates compounding latency. If an agent must search a database (500ms),
query a web API (800ms), and retrieve documents (400ms), sequential execution takes 1,700ms.
Executing them concurrently via `RunnableParallel` or `asyncio.gather` reduces total duration to the slowest single operation (800ms).
""")

st.code("""
import asyncio
from langchain_core.runnables import RunnableParallel, RunnableLambda

def fetch_weather(location: str) -> str:
    # Simulated 0.5s network I/O
    return f"Weather in {location}: 72F, Sunny"

def fetch_news(location: str) -> str:
    # Simulated 0.6s network I/O
    return f"Headlines in {location}: Tech conference opens"

def fetch_events(location: str) -> str:
    # Simulated 0.4s network I/O
    return f"Events in {location}: Marathon this weekend"

# Compose parallel execution pipeline
parallel_research = RunnableParallel(
    weather=RunnableLambda(fetch_weather),
    news=RunnableLambda(fetch_news),
    events=RunnableLambda(fetch_events)
)

# Runs all 3 functions concurrently across thread pool / async loop
results = parallel_research.invoke("Boston")
# Output: {'weather': '...', 'news': '...', 'events': '...'}
""", language="python")

# --- SECTION 6: OPTIMIZE PROMPTS ---
st.header("6. Prompt Token Optimization: System Prompt Trimming and Token Budgets", divider="rainbow")
st.markdown("""
Every unnecessary token in a system prompt increases Time-to-First-Token and adds cost on every turn.
System prompts should be systematically pruned of conversational pleasantries, using concise declarative syntax
and strict token budgets.
""")

col_pt1, col_pt2 = st.columns(2)
with col_pt1:
    st.markdown("### Optimization Techniques")
    st.markdown("""
    - **Declarative Conciseness**: Replace 'Please make sure that you always remember to...' with 'Constraint: Never...'.
    - **Dynamic Few-Shot Pruning**: Inject only the top-2 most semantically relevant examples using `SemanticSimilarityExampleSelector` rather than 20 static examples.
    - **Context Window Trimming**: Use `trim_messages()` to discard older conversation turns while preserving system instructions.
    """)
with col_pt2:
    st.code("""
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# Automatically trim conversation history to fit within a 2,000-token budget
trimmed_history = trim_messages(
    messages,
    max_tokens=2000,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o"),
    include_system=True,  # Always preserve system prompt
    allow_partial=False
)
    """, language="python")

# --- SECTION 7: OPTIMIZE RETRIEVAL ---
st.header("7. Retrieval Pipeline Optimization: Hybrid Indexing, RRF, and Reranking Pruning", divider="rainbow")
st.markdown("""
Retrieving too many documents (e.g. top-k = 20) floods the LLM context window with irrelevant noise,
increasing generation latency and triggering 'lost in the middle' degradation.
High-performance retrieval uses hybrid search combined with reciprocal rank fusion (RRF) and a cross-encoder reranker
to prune top-k down to the top-3 strictly relevant chunks.
""")

st.code("""
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever

# 1. Sparse keyword retriever + 2. Dense vector retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

vector_retriever = faiss_store.as_retriever(search_kwargs={"k": 10})

# 3. Ensemble with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 4. Contextual Compressor: Downsample 20 candidate chunks to top-3 highest precision chunks
# Eliminates 80% of prompt context tokens before generation
""", language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: High-Performance Concurrent LangChain Execution Engine", divider="rainbow")
st.markdown("""
Below is a complete, production-grade template implementing exact match caching, asynchronous parallel tool gathering,
and token streaming.
""")

st.code("""
# high_performance_pipeline.py
import asyncio
from typing import AsyncGenerator
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 1. Global in-memory cache
set_llm_cache(InMemoryCache())

# 2. Fast streaming model configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# 3. Parallel asynchronous I/O tools
async def fetch_db_stats(query: str):
    await asyncio.sleep(0.05)
    return {"records_analyzed": 1420}

async def fetch_audit_log(query: str):
    await asyncio.sleep(0.05)
    return {"status": "compliant"}

# 4. Asynchronous streaming generator
async def execute_fast_pipeline(prompt_text: str) -> AsyncGenerator[str, None]:
    # Concurrent tool dispatch
    tools_results = await asyncio.gather(
        fetch_db_stats(prompt_text),
        fetch_audit_log(prompt_text)
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Synthesize context: Stats: {stats}, Audit: {audit}. Question: {q}"
    )
    chain = prompt | llm
    
    # Stream tokens to caller
    async for chunk in chain.astream({
        "stats": tools_results[0],
        "audit": tools_results[1],
        "q": prompt_text
    }):
        yield chunk.content
""", language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Latency & Cost Optimization Profiler Simulator", divider="rainbow")
st.markdown("""
Compare the latency, token consumption, and cost telemetry between an unoptimized naive pipeline
and a high-performance optimized LangChain architecture.
""")

sim = PerformanceOptimizationSimulator()

col_cfg, col_prof = st.columns([1, 1])

with col_cfg:
    st.subheader("⚙️ Pipeline Architecture Configuration")
    chosen_scenario = st.selectbox(
        "Select Pipeline Architecture Configuration:",
        list(sim.scenarios.keys())
    )
    profile_btn = st.button("🚀 Run Profiler & Benchmark Latency", type="primary")

with col_prof:
    st.subheader("📊 Performance & Telemetry Scorecard")
    if profile_btn:
        res = sim.simulate_pipeline(chosen_scenario)
        speedup = "<span class='badge-speedup'>OPTIMAL EFFICIENCY</span>" if "Optimized" in chosen_scenario else "<span style='color:#EF4444;font-weight:bold'>HIGH LATENCY</span>"
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Time-to-First-Token (TTFT)", f"{res['ttft_ms']} ms")
            st.metric("Total Latency", f"{res['total_latency_ms']} ms")
        with col_m2:
            st.metric("Prompt Tokens Consumed", f"{res['prompt_tokens']}")
            st.metric("Cost per 1k Requests", f"${res['estimated_cost_per_1k']:.2f}")

        st.markdown(
            f"<div class='perf-card'>"
            f"<b>Profile Status:</b> {speedup}<br><br>"
            f"<b>LLM Invocations per Turn:</b> {res['llm_calls']}<br>"
            f"<b>Caching Enabled:</b> {res['caching_active']}<br>"
            f"<b>Tool Execution Strategy:</b> {res['tool_execution']}<br>"
            f"<b>Streaming Active:</b> {res['streaming_active']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if "Optimized" in chosen_scenario:
            st.success("Pipeline demonstrates 6x speedup and 78% cost reduction.")
        else:
            st.warning("Significant latency bottlenecks detected: missing caching and sequential tools.")
    else:
        st.info("Select a configuration and profile to view benchmark metrics.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is Time-to-First-Token (TTFT) and why is it the primary UX metric in GenAI?"):
    st.markdown("""
    **Answer**: TTFT measures the duration between when a user submits a prompt and when the application displays the first generated token.
    Because reading happens at roughly 200-300 words per minute, once streaming begins with low TTFT (< 400ms), users perceive the application
    as instantaneous, even if the total completion takes several seconds. Without streaming, users experience the entire wait as frozen dead-time.
    """)

with st.expander("Q2: How does `RunnableParallel` optimize agent tool execution latency?"):
    st.markdown("""
    **Answer**: By default, sequentially executing three tools taking 500ms, 800ms, and 400ms takes 1,700ms total. `RunnableParallel` wraps
    the tools in asynchronous coroutines and dispatches them concurrently via the event loop, resolving when the slowest tool finishes (800ms),
    yielding a 53% latency reduction.
    """)

with st.expander("Q3: When should you use exact caching versus semantic vector caching?"):
    st.markdown("""
    **Answer**: **Exact caching** (e.g. SHA256 hashed prompt in Redis) is microsecond fast ($O(1)$) and carries zero risk of false positive cache hits,
    making it ideal for exact FAQ queries and deterministic extraction. **Semantic caching** uses vector embeddings to match similar questions
    (e.g. 'How do I cancel?' vs 'Where to cancel subscription?'), but incurs vector computation latency (20-50ms) and risks answering with stale context
    if the cosine similarity threshold is tuned too loosely (< 0.95).
    """)

with st.expander("Q4: How does retrieving excessive document chunks (high top-k) degrade generation speed?"):
    st.markdown("""
    **Answer**: Passing 20 chunks instead of 3 increases prompt context length by thousands of tokens. This increases LLM prefill computation time,
    increases Time-to-First-Token, inflates per-token API costs, and increases hallucination risk due to attention dilution ('lost in the middle' effect).
    Rerankers should be used to compress candidate lists down to top-3 relevant passages.
    """)

with st.expander("Q5: What is the purpose of `trim_messages()` in long-running conversational agents?"):
    st.markdown("""
    **Answer**: In multi-turn chat sessions, conversation history expands indefinitely. `trim_messages()` programmatically truncates older messages
    based on a strict token budget (e.g. 2,000 tokens), preserving the system prompt and the most recent context while preventing context-length
    exhaustion errors and runaway token costs.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Sequential Tool Execution)**: Invoking external tools in a linear loop (`for tool in tools: tool.invoke()`). Latency scales linearly with tool count; always use `asyncio.gather` or `RunnableParallel`.")
    st.warning("**Trap 2 (Setting Semantic Cache Thresholds Too Low)**: Configuring semantic cache similarity at 0.85. The cache returns answers for completely different questions, causing silent hallucinations.")
with col_trap2:
    st.error("**Trap 3 (Buffering Entire Streams on the Backend)**: Consuming a streaming generator on the server and waiting for completion before sending the response to the client. This negates all streaming UX benefits.")
    st.warning("**Trap 4 (Unbounded System Prompt Bloat)**: Adding every historical edge case into the system prompt until it exceeds 4,000 tokens, permanently bloating TTFT and per-request operational costs.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the most effective way to optimize Time-to-First-Token?")
        st.caption("Flip for answer:")
        st.info("**Back**: Enable streaming (`streaming=True`, `.astream()`) and trim unnecessary system prompt tokens.")
    with st.container(border=True):
        st.markdown("**Front**: Which LangChain primitive dispatches independent tasks concurrently?")
        st.caption("Flip for answer:")
        st.info("**Back**: `RunnableParallel` (executes branches concurrently across threads or async coroutines).")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What function enforces token budgets on message history in LangChain?")
        st.caption("Flip for answer:")
        st.info("**Back**: `trim_messages()` from `langchain_core.messages`.")
    with st.container(border=True):
        st.markdown("**Front**: What is the primary operational advantage of exact caching over semantic caching?")
        st.caption("Flip for answer:")
        st.info("**Back**: Zero embedding latency overhead ($O(1)$ lookup) and zero false-positive cache collisions.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I know how to implement deterministic triage routers to eliminate unnecessary LLM calls.", value=True)
c2 = st.checkbox("I can build end-to-end token streaming pipelines using .astream() and StreamingResponse.", value=True)
c3 = st.checkbox("I understand the operational differences between Redis exact caching and vector semantic caching.", value=True)
c4 = st.checkbox("I can execute independent tools concurrently using RunnableParallel and asyncio.gather.", value=True)
c5 = st.checkbox("I know how to manage conversation token budgets using trim_messages().", value=True)
c6 = st.checkbox("I understand why contextual compression and rerankers prevent retrieval context dilution.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 39.3: Reliability.")
