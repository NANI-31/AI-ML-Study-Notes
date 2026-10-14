import streamlit as st
import json

st.set_page_config(
    page_title="27.2 Latency Optimization - Prompt Engineering",
    page_icon="⚡",
    layout="wide"
)

# Header Section
st.title("27.2 Latency Optimization: Slashing TTFT & Tail Latency")
st.caption("Chapter 27: Performance and Cost Optimization | Domain: 1. Prompt Engineering")
st.markdown("""
In user-facing production software, latency is the difference between a delight and a bounce. 
Large generative AI models naturally incur significant execution delays. 
Systematic latency engineering focuses on **reducing unnecessary roundtrips, concurrent execution, 
token streaming, model routing cascades, speculative decoding, and sub-10ms semantic caching**.
""")

st.divider()

# Section 1: The Anatomy of LLM Latency
st.header("1. Anatomy of End-to-End Latency", divider="rainbow")
st.markdown("""
Total request latency breaks down into distinct phases across the network and compute stack:
""")

st.graphviz_chart('''
digraph LatencyAnatomy {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Net1 [label="Network Ingress\\n(~30-80ms)", fillcolor="#E1BEE7"];
    Queue [label="Provider Queueing\\n(~20-200ms)", fillcolor="#FFCDD2"];
    Prefill [label="Prompt Prefill (TTFT)\\n(~100-400ms)\\nProcesses input tokens", fillcolor="#BBDEFB"];
    Decode [label="Autoregressive Decoding\\n(~20-40ms per token)\\nGenerates completion tokens", fillcolor="#FFF9C4"];
    Net2 [label="Network Egress\\n(~30-80ms)", fillcolor="#E1BEE7"];
    
    Net1 -> Queue -> Prefill -> Decode -> Net2;
}
''', use_container_width=False)

st.markdown("""
* **Time to First Token (TTFT)** = Network Ingress + Provider Queue + Prompt Prefill.
* **Generation Duration** = Completion Tokens * Time Per Output Token (TPOT).
* **Total Latency** = TTFT + Generation Duration + Egress.
""")

st.divider()

# Section 2: Five Core Latency Optimization Strategies
st.header("2. Five Core Architectural Interventions", divider="rainbow")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Eliminate Unnecessary Model Calls")
    st.markdown("""
    * **Heuristic Short-Circuits**: Use regex, keyword sets, or fast local models to handle greetings, common FAQs, and basic queries without invoking expensive LLM clusters.
    * **Semantic Caching**: Exact and vector match caches return responses in under 15ms.
    """)

    st.subheader("2. Cascading Model Routing")
    st.markdown("""
    * **Pattern**: Route 80% of traffic to lightweight, ultra-fast models (e.g. `gpt-4o-mini`, `claude-3-haiku` with TTFT ~150ms). 
    Only escalate to frontier models (`gpt-4o`, `claude-3-5-sonnet`) when confidence scores drop below a threshold.
    """)

with col2:
    st.subheader("3. Token Streaming via Server-Sent Events (SSE)")
    st.markdown("""
    * **User Experience Impact**: Even if total generation takes 4 seconds, streaming the first word in 250ms makes the application feel instant to the human user.
    """)

    st.subheader("4. Speculative Decoding & Parallel Tool Execution")
    st.markdown("""
    * **Concurrent Tools**: If an agent determines it needs 3 database lookups, dispatch all 3 asynchronously via `asyncio.gather()` rather than sequentially in 3 roundtrips.
    """)

st.divider()

# Section 3: Concurrent Tool Execution Pattern
st.header("3. Asynchronous Concurrent Tool Dispatch", divider="rainbow")
st.markdown("""
When an LLM emits multiple tool calls, running them sequentially multiplies latency. Running them concurrently with `asyncio` collapses wait time:
""")

st.code('''# Asynchronous Concurrent Tool Execution
import asyncio

async def fetch_order(order_id: str):
    await asyncio.sleep(0.3)  # Simulates 300ms DB latency
    return {"order_id": order_id, "status": "Shipped"}

async def fetch_user_profile(user_id: str):
    await asyncio.sleep(0.25)  # Simulates 250ms DB latency
    return {"user_id": user_id, "tier": "Gold"}

async def fetch_inventory(sku: str):
    await asyncio.sleep(0.35)  # Simulates 350ms DB latency
    return {"sku": sku, "in_stock": True}

async def execute_tools_in_parallel():
    # Dispatches all 3 tools concurrently!
    # Total elapsed time = 350ms (max) instead of 900ms (sum)!
    order, profile, inventory = await asyncio.gather(
        fetch_order("ORD-101"),
        fetch_user_profile("USR-55"),
        fetch_inventory("SKU-99")
    )
    return {"order": order, "profile": profile, "inventory": inventory}''', language="python")

st.divider()

# Section 4: Interactive Latency Profiler Simulator
st.header("4. Interactive Latency Profiler & Budget Workbench", divider="rainbow")
st.markdown("Configure architecture parameters to visualize predicted P50 and P99 latency percentiles.")

col_p1, col_p2 = st.columns(2)

with col_p1:
    model_choice = st.selectbox("Inference Model:", ["GPT-4o (Standard Frontier)", "GPT-4o-mini (Lightweight Fast)", "Self-Hosted vLLM (Local Speculative Decoding)"])
    enable_cache = st.checkbox("Enable Semantic Cache (35% hit rate)", value=True)
    enable_stream = st.checkbox("Enable Real-Time Token Streaming", value=True)
    out_tokens = st.slider("Expected Output Tokens", 50, 1000, 250, step=50)

with col_p2:
    # Telemetry estimation
    if "mini" in model_choice:
        base_ttft = 180
        tpot = 18  # ms per token
    elif "vLLM" in model_choice:
        base_ttft = 90
        tpot = 12
    else:
        base_ttft = 350
        tpot = 32
        
    calc_total_gen = out_tokens * tpot
    calc_e2e = base_ttft + calc_total_gen
    perceived_latency = base_ttft if enable_stream else calc_e2e
    
    if enable_cache:
        # Weighted avg with 35% cache at 15ms
        effective_perceived = int(perceived_latency * 0.65 + 15 * 0.35)
    else:
        effective_perceived = perceived_latency

    st.subheader("Latency Forecast Breakdown")
    m1, m2 = st.columns(2)
    m1.metric("Time to First Token (TTFT)", f"{base_ttft} ms")
    m2.metric("Perceived User Latency", f"{effective_perceived} ms", delta="Streaming Active" if enable_stream else "Blocked on Completion")
    
    st.caption(f"Full generation time: {calc_e2e} ms ({out_tokens} tokens @ {1000/tpot:.0f} tok/sec).")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is Speculative Decoding and how does it reduce inference latency?"):
        st.write("""
        Speculative decoding uses a small, high-speed draft model (e.g., a 1B parameter model) to rapidly generate 
        a sequence of $K$ candidate tokens. The large target model (e.g., 70B parameter model) then evaluates all $K$ tokens 
        simultaneously in a single forward pass in parallel. Tokens accepted by the target model are retained; 
        the first rejected token triggers a correction. 
        This achieves a 2x to 3x speedup in wall-clock decoding time without altering the mathematical output distribution.
        """)

    with st.expander("Q: What is the difference between TTFT and TPOT and which matters more for UX?"):
        st.write("""
        * **TTFT (Time to First Token)**: Measures responsiveness. Governed by prompt prefill and network roundtrips.
        * **TPOT (Time Per Output Token)**: Measures reading speed. Governed by memory bandwidth and model parameter size.
        For interactive chat user experiences, **TTFT matters most** because humans perceive the system as fast 
        as long as tokens begin appearing within ~300ms, even if the complete answer takes several seconds to stream.
        """)

with qa2:
    with st.expander("Trap to Avoid: Neglecting Cold-Start Container Latencies"):
        st.write("""
        When self-hosting open-source LLMs on serverless GPU infrastructure (like RunPod, Modal, or AWS SageMaker Serverless), 
        cold starts can take 20 to 90 seconds while GPU weights load into VRAM. 
        Always configure provisioned concurrency or warm instance pools for production endpoints to prevent severe P99 latency spikes.
        """)

    with st.expander("Flashcards: Latency Optimization"):
        fc_sel = st.selectbox(
            "Select Latency Term:",
            ["TTFT vs TPOT", "Speculative Decoding", "Concurrent Tool Gathering", "Semantic Cache Latency"]
        )
        if fc_sel == "TTFT vs TPOT":
            st.info("TTFT measures initial responsiveness; TPOT measures the continuous velocity of token generation.")
        elif fc_sel == "Speculative Decoding":
            st.info("Using a small draft model to generate tokens that a large model verifies in parallel.")
        elif fc_sel == "Concurrent Tool Gathering":
            st.info("Executing multiple external API lookups simultaneously using asynchronous gathering.")
        else:
            st.info("Bypassing the LLM entirely for frequent queries to return cached answers in < 15ms.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the components of TTFT and how to optimize them.")
    st.checkbox("I can implement asynchronous parallel tool execution with asyncio.")
with c2:
    st.checkbox("I know how speculative decoding accelerates autoregressive generation.")
    st.checkbox("I can design cascading model routers to preserve low latency for simple queries.")
