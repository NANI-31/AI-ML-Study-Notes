import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.3 Runnable Methods - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("5.3 Runnable Methods: The Six Execution Protocols (Sync, Async, Stream & Batch)")
st.caption("Chapter 5: LangChain Expression Language and Runnables | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
The superpower of the modern LangChain Expression Language is that every pipeline built with LCEL 
instantly and automatically inherits **six execution protocols**. 

Developers do not have to write separate implementations for synchronous scripts, asynchronous web servers, 
concurrent background batching jobs, and progressive UI streaming. The unified `Runnable` interface exposes 
these capabilities natively across every stage of your pipeline.

In this lesson, you will master:
- **Synchronous Execution**: `.invoke()`, blocking request lifecycles, and direct result unpacking.
- **Concurrent Batching**: `.batch()` and `.abatch()` using internal threadpools and `asyncio.gather()`.
- **Progressive Streaming**: `.stream()` and `.astream()` for sub-second Time-To-First-Token (TTFT).
- **Asynchronous Non-Blocking I/O**: `.ainvoke()` for high-throughput production web backends (FastAPI).
- **Protocol Selection Matrix**: Choosing the exact method for real-time user chats, batch offline processing, and background jobs.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockRunnableMethodBenchmarkEngine:
    """Simulates performance and telemetry differences across the 6 Runnable execution methods in module scope."""
    def __init__(self):
        self.method_profiles = {
            "invoke()": {"type": "Synchronous Single", "concurrency": 1, "ttft_ms": 380, "total_ms": 380, "throughput_qps": 2.6},
            "ainvoke()": {"type": "Asynchronous Single", "concurrency": 1, "ttft_ms": 370, "total_ms": 370, "throughput_qps": 28.5},
            "stream()": {"type": "Synchronous Stream", "concurrency": 1, "ttft_ms": 85, "total_ms": 410, "throughput_qps": 2.4},
            "astream()": {"type": "Asynchronous Stream", "concurrency": 1, "ttft_ms": 80, "total_ms": 390, "throughput_qps": 25.0},
            "batch()": {"type": "Concurrent ThreadPool", "concurrency": 5, "ttft_ms": 390, "total_ms": 480, "throughput_qps": 10.4},
            "abatch()": {"type": "Asynchronous Gather", "concurrency": 10, "ttft_ms": 375, "total_ms": 420, "throughput_qps": 42.0}
        }

    def get_benchmark_df(self) -> pd.DataFrame:
        data = []
        for method, p in self.method_profiles.items():
            data.append({
                "Runnable Method": method,
                "Execution Model": p["type"],
                "Concurrency Factor": f"{p['concurrency']}x",
                "Time-to-First-Token": f"{p['ttft_ms']} ms",
                "Total Latency": f"{p['total_ms']} ms",
                "Relative Throughput": f"{p['throughput_qps']} QPS"
            })
        return pd.DataFrame(data)

    def simulate_execution(self, method: str, input_count: int, prompt_text: str) -> Dict[str, Any]:
        profile = self.method_profiles.get(method, self.method_profiles["invoke()"])
        
        simulated_tokens = [
            "Cognitive", " pipelines", " in", " LangChain", " execute", " via", " unified", " Runnable", " protocols."
        ]
        
        return {
            "method": method,
            "inputs_processed": input_count,
            "effective_latency_ms": profile["total_ms"] if input_count <= 1 else int(profile["total_ms"] * (1 + (input_count * 0.15))),
            "ttft_ms": profile["ttft_ms"],
            "tokens": simulated_tokens,
            "summary": f"Executed {input_count} workload(s) via {method}. Latency bounded by {profile['type']} model."
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Six Core Runnable Invocation Protocols & Event Lifecycle", divider="rainbow")

st.markdown("""
The diagram below illustrates the six core execution methods of the `Runnable` interface, 
divided across Synchronous, Asynchronous, Streaming, and Batching axes.
""")

dot_blueprint = """
digraph RunnableMethodsTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_sync {
        label="1. Synchronous Execution Domain (Blocking)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_invoke [label=".invoke(input)\nReturns: OutputType\nBlocks calling thread until complete", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_stream [label=".stream(input)\nReturns: Iterator[OutputChunk]\nYields progressive chunks via generator", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_batch [label=".batch(List[input])\nReturns: List[OutputType]\nExecutes concurrently via ThreadPoolExecutor", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_async {
        label="2. Asynchronous Execution Domain (Non-Blocking / Asyncio)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_ainvoke [label="await .ainvoke(input)\nReturns: OutputType\nNon-blocking coroutine for FastAPI / web servers", fillcolor="#1E293B", fontcolor="#34D399"];
        node_astream [label="async for chunk in .astream(input)\nReturns: AsyncIterator[OutputChunk]\nNon-blocking token generator for WebSockets & SSE", fillcolor="#1E293B", fontcolor="#34D399"];
        node_abatch [label="await .abatch(List[input])\nReturns: List[OutputType]\nConcurrent asyncio.gather() non-blocking batching", fillcolor="#1E293B", fontcolor="#34D399"];
    }

    // Connect counterpart protocols
    node_invoke -> node_ainvoke [label="async counterpart", color="#38BDF8", style="dashed"];
    node_stream -> node_astream [label="async counterpart", color="#38BDF8", style="dashed"];
    node_batch -> node_abatch [label="async counterpart", color="#38BDF8", style="dashed"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Notice the symmetry between synchronous and asynchronous protocols. 
> Every synchronous method (`invoke`, `stream`, `batch`) has an exact asynchronous counterpart 
> prefixed with `a` (`ainvoke`, `astream`, `abatch`). When building an LCEL chain, you get all six methods 
> for free without writing a single line of method-specific boilerplate.
""")

# --- SECTION 2: INVOKE() ---
st.header("2. invoke(): Synchronous Blocking Execution, Return Types & Telemetry Inspection", divider="rainbow")

st.markdown("""
`.invoke()` is the most fundamental execution method. It takes a single input, executes the pipeline 
synchronously on the calling thread, blocks until the final component produces an output, and returns the result.

### When to Use `.invoke()`
- Command-line scripts and CLI tools.
- Offline data science analysis in Jupyter notebooks.
- Unit testing and deterministic validation runs.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Define {concept} in 15 words.") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Synchronous blocking call
result = chain.invoke({"concept": "Database Sharding"})
print(result)
''', language="python")

# --- SECTION 3: BATCH() ---
st.header("3. batch(): Concurrent Multi-Input Processing, ThreadPool Concurrency & Batch Telemetry", divider="rainbow")

st.markdown("""
When processing a list of items (e.g., classifying 100 customer emails or summarizing 50 support tickets), 
calling `.invoke()` in a Python `for` loop is painfully slow because requests are executed serially (O(N) network latency).

**`.batch()`** takes a list of inputs and processes them **concurrently using an internal `ThreadPoolExecutor`**.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Classify sentiment: {text}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

inputs = [
    {"text": "The payment gateway timed out."},
    {"text": "Fast delivery, very satisfied!"},
    {"text": "Unable to connect to Redis replica."}
]

# Executes concurrently across multiple worker threads
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}  # Limits concurrent worker threads
)
print(results)  # ['NEGATIVE', 'POSITIVE', 'NEGATIVE']
''', language="python")

# --- SECTION 4: STREAM() ---
st.header("4. stream(): Synchronous Progressive Token Generation, Iterators & Backpressure", divider="rainbow")

st.markdown("""
Foundation models generate completions token-by-token. Waiting for a 500-token completion to finish 
before showing anything to the user creates 2 to 5 seconds of blank-screen latency.

**`.stream()`** returns a standard Python generator that yields chunks (`AIMessageChunk` or strings) 
in real-time as they are emitted from the model.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Write a poem about {topic}:") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Progressive token consumption
for token_chunk in chain.stream({"topic": "Event-Driven Architecture"}):
    print(token_chunk, end="", flush=True)
''', language="python")

# --- SECTION 5: AINVOKE() ---
st.header("5. ainvoke(): Asynchronous Non-Blocking Execution & FastAPI Integration", divider="rainbow")

st.markdown("""
In modern Python web backends built with **FastAPI**, **Starlette**, or **Aiohttp**, 
calling a blocking synchronous method like `.invoke()` freezes the entire single-threaded asyncio event loop. 
All other concurrent web users will be blocked from receiving responses!

**`.ainvoke()`** is an asynchronous coroutine that uses non-blocking HTTP clients (`httpx.AsyncClient`), 
allowing web servers to handle thousands of concurrent requests smoothly.
""")

st.code('''from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

chain = (
    ChatPromptTemplate.from_template("Audit infrastructure ticket: {ticket}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

class TicketRequest(BaseModel):
    ticket_text: str

@app.post("/audit")
async def audit_endpoint(req: TicketRequest):
    # Non-blocking async call preserves event loop concurrency
    result = await chain.ainvoke({"ticket": req.ticket_text})
    return {"analysis": result}
''', language="python")

# --- SECTION 6: ABATCH() ---
st.header("6. abatch(): Asynchronous Concurrent Batching with asyncio.gather & Rate Limiting", divider="rainbow")

st.markdown("""
`.abatch()` is the asynchronous equivalent of `.batch()`. Instead of managing threads in an OS threadpool, 
it uses **`asyncio.gather()`** to dispatch asynchronous tasks concurrently across the event loop.
""")

st.code('''import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Summarize: {doc}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

async def process_batch_docs(docs: list):
    inputs = [{"doc": d} for d in docs]
    # Asynchronously processes all inputs concurrently
    results = await chain.abatch(
        inputs,
        config={"max_concurrency": 10}
    )
    return results

# asyncio.run(process_batch_docs(["Doc 1...", "Doc 2..."]))
''', language="python")

# --- SECTION 7: ASTREAM() ---
st.header("7. astream(): Asynchronous Token Streaming, Server-Sent Events (SSE) & WebSockets", divider="rainbow")

st.markdown("""
When building interactive web chat frontends, tokens must stream asynchronously from the web server 
to the browser via **Server-Sent Events (SSE)** or **WebSockets**.

**`.astream()`** returns an `AsyncIterator`, allowing web frameworks like FastAPI to stream tokens 
progressively using `StreamingResponse`.
""")

st.code('''from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

chain = (
    ChatPromptTemplate.from_template("Explain architecture of: {topic}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

@app.get("/stream")
async def stream_endpoint(topic: str):
    async def token_generator():
        async for chunk in chain.astream({"topic": topic}):
            yield chunk

    return StreamingResponse(token_generator(), media_type="text/event-stream")
''', language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Asynchronous High-Throughput Server Gateway", divider="rainbow")

st.markdown("""
Below is an enterprise-grade production module demonstrating how to expose a single LCEL pipeline 
across asynchronous single-query (`ainvoke`), progressive SSE streaming (`astream`), and concurrent batching (`abatch`).
""")

st.code('''from typing import List, Dict, Any, AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class EnterpriseExecutionGateway:
    """Production gateway exposing a single LCEL chain across all asynchronous protocols."""

    def __init__(self, api_key: str):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise cloud security compliance auditor."),
            ("human", "Inspect policy for service '{service_name}':\\n{policy_document}")
        ])
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
        self.parser = StrOutputParser()

        # Canonical LCEL Pipeline
        self.pipeline = self.prompt | self.model | self.parser

    async def execute_single(self, service: str, policy: str) -> str:
        """Asynchronous single invocation (.ainvoke)."""
        return await self.pipeline.ainvoke({"service_name": service, "policy_document": policy})

    async def execute_stream(self, service: str, policy: str) -> AsyncIterator[str]:
        """Asynchronous token streaming (.astream)."""
        async for chunk in self.pipeline.astream({"service_name": service, "policy_document": policy}):
            yield chunk

    async def execute_bulk(self, policy_items: List[Dict[str, str]], max_workers: int = 5) -> List[str]:
        """Asynchronous concurrent batching (.abatch)."""
        return await self.pipeline.abatch(
            policy_items,
            config={"max_concurrency": max_workers}
        )
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Interactive Multi-Protocol Invocation & Latency Benchmark Lab", divider="rainbow")

st.markdown("""
Compare the empirical performance profiles of the 6 Runnable execution methods below. 
Observe the trade-offs between single invocations, concurrent batching, and progressive streaming.
""")

bench_engine = MockRunnableMethodBenchmarkEngine()

st.subheader("📊 Cross-Protocol Architectural Decision Matrix")
matrix_df = bench_engine.get_benchmark_df()
st.dataframe(matrix_df, use_container_width=True)

st.divider()

st.subheader("⚡ Live Protocol Simulator")
col_p1, col_p2 = st.columns(2)
with col_p1:
    selected_method = st.selectbox(
        "Select Runnable Method to Test",
        ["invoke()", "ainvoke()", "stream()", "astream()", "batch()", "abatch()"],
        index=0
    )
    workload_count = st.slider("Workload Units (Items in Batch)", min_value=1, max_value=20, value=5 if "batch" in selected_method else 1)

with col_p2:
    prompt_wb = st.text_input("Sample Prompt", value="Describe distributed consensus latency.")

if st.button("🚀 Execute Protocol Simulation", type="primary"):
    sim_res = bench_engine.simulate_execution(selected_method, workload_count, prompt_wb)
    
    st.subheader("📊 Simulation Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("Method Tested", sim_res["method"])
    m2.metric("Time-To-First-Token (TTFT)", f"{sim_res['ttft_ms']} ms")
    m3.metric("Effective Latency", f"{sim_res['effective_latency_ms']} ms")

    if "stream" in selected_method:
        st.info("Progressive token emission simulated:")
        placeholder = st.empty()
        accumulated = ""
        for tok in sim_res["tokens"]:
            accumulated += tok
            placeholder.code(accumulated, language="markdown")
    else:
        st.write("### Returned Output")
        st.code(sim_res["summary"], language="markdown")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why will calling .invoke() inside a FastAPI async route severely degrade web server throughput?",
     "FastAPI runs an asynchronous event loop on a single thread. Calling synchronous .invoke() blocks that thread for the duration of the network request (300ms to 2000ms), freezing the event loop and preventing all other concurrent HTTP requests from being processed. Calling await .ainvoke() yields execution to the loop during network I/O."),
    ("How does .batch() implement concurrency under the hood in LangChain?",
     ".batch() wraps executions inside a Python ThreadPoolExecutor. It takes an iterable of inputs, submits each input to an available worker thread, and returns the list of results once all threads complete. Concurrency limits are controlled via config={'max_concurrency': N}."),
    ("What is the difference between .stream() and .astream() in terms of Python mechanics?",
     ".stream() returns a synchronous generator (Iterator[T]) consumed via `for chunk in chain.stream()`. .astream() returns an asynchronous generator (AsyncIterator[T]) consumed via `async for chunk in chain.astream()`. .astream() is required for non-blocking streaming in asynchronous web backends."),
    ("How does streaming propagate when an LCEL chain contains multiple sequential steps?",
     "In a pipeline like `prompt | model | parser`, streaming starts as soon as the model yields its first token chunk. The parser transforms that chunk and yields it immediately to downstream consumers without waiting for the full generation to complete, preserving minimal Time-To-First-Token."),
    ("What does the config={'max_concurrency': N} parameter protect against during .batch() execution?",
     "Without max_concurrency, .batch() attempts to dispatch all items simultaneously. For large batches (e.g., 500 items), this creates 500 concurrent HTTP requests, triggering severe rate limit throttling (HTTP 429) from model providers and exhausting local socket connection pools.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Synchronous .invoke() in Async Handlers. Calling `.invoke()` inside `async def` routes completely blocks ASGI workers. Always use `await .ainvoke()` in asynchronous web frameworks.")
    st.error("**Trap 2 (Breaking Error)**: Unbounded .batch() Concurrency. Passing 200 items to `.batch()` without setting `max_concurrency` causes instant HTTP 429 rate limit crashes from OpenAI and Anthropic.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Streaming Overhead in Server-to-Server Jobs. Using `.stream()` for backend batch jobs that only care about the final text introduces unnecessary per-chunk CPU overhead. Use `.invoke()` or `.batch()` for non-UI workloads.")
    st.warning("**Trap 4 (Quality Warning)**: Incomplete Chunk Accumulation. Assuming `chunk` contains the full accumulated string instead of an incremental delta causes parsing errors. Accumulate chunks (`full_text += chunk`) when reading streams.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Async Invocation Rule"):
        st.markdown("**Always use await chain.ainvoke() inside async def web handlers.**")
    with st.expander("🎴 Flashcard 2: Concurrency Throttling"):
        st.markdown("**config={'max_concurrency': 5} prevents HTTP 429 rate limit exhaustion in .batch().**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Streaming Generator"):
        st.markdown("**.stream() yields chunks progressively without waiting for full generation.**")
    with st.expander("🎴 Flashcard 4: Web Streaming Protocol"):
        st.markdown("**FastAPI StreamingResponse + chain.astream() powers real-time SSE streaming.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can differentiate the 6 execution methods of the Runnable interface.", value=True)
chk2 = st.checkbox("I understand why .ainvoke() is mandatory for FastAPI web applications.", value=True)
chk3 = st.checkbox("I know how to bound concurrency in .batch() using max_concurrency.", value=True)
chk4 = st.checkbox("I understand how progressive streaming reduces Time-To-First-Token (TTFT).", value=True)
chk5 = st.checkbox("I can implement Server-Sent Events using chain.astream() and StreamingResponse.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 5.3 Runnable Methods. Ready for 5.4 Advanced Runnable Concepts!")
