import streamlit as st
import json
import time

st.set_page_config(
    page_title="23.3 Production API Usage - Prompt Engineering",
    page_icon="🚀",
    layout="wide"
)

# Header Section
st.title("23.3 Production API Architecture: Resilience, Streaming, & Cost")
st.caption("Chapter 23: Prompt Engineering with APIs | Domain: 1. Prompt Engineering")
st.markdown("""
Calling an API in a local Jupyter notebook is trivial; keeping an LLM service running reliably at 50,000 requests per minute 
demands enterprise-grade systems engineering. Production deployments require **exponential backoff with jitter, 
rate-limit token buckets, real-time token streaming, multi-tier prompt caching, structured telemetry, and automated cost governance**.
""")

st.divider()

# Section 1: Error Handling, Retries & Rate Limits
st.header("1. Error Handling, Exponential Backoff, and Rate Limits", divider="rainbow")
st.markdown("""
LLM cloud APIs operate under aggressive multi-tenant throttling. Applications must handle standard HTTP error status codes gracefully:
""")

col_err1, col_err2 = st.columns(2)
with col_err1:
    st.subheader("HTTP Error Status Codes")
    st.markdown("""
    * **429 Too Many Requests**: Exceeded Requests Per Minute (RPM) or Tokens Per Minute (TPM). Requires client-side backoff or request queuing.
    * **500 / 503 Internal Server Error**: Downstream GPU cluster overload. Retriable with jitter.
    * **400 Bad Request / Invalid JSON**: Schema or token context window exceeded. Non-retriable; requires payload truncation.
    * **401 Unauthorized**: Invalid or expired API key. Instant alert to DevOps.
    """)

with col_err2:
    st.subheader("Exponential Backoff with Full Jitter")
    st.markdown("""
    Retrying failed requests at fixed intervals (e.g. every 2 seconds) causes the 'thundering herd' problem, 
    re-saturating the server. 
    Full jitter randomizes wait times:
    * `Wait Time = Random(0, Min(Max_Delay, Base_Delay * 2^attempt))`
    * Spreads retries evenly across time, maximizing recovery probability.
    """)

st.code('''# Resilient LLM API Caller with Exponential Backoff and Jitter
import time
import random

def call_llm_with_resilience(prompt_messages, max_retries=4, base_delay=1.0, max_delay=30.0):
    for attempt in range(max_retries):
        try:
            # Simulated API call
            # response = client.chat.completions.create(model="gpt-4o", messages=prompt_messages)
            return {"status": "SUCCESS", "content": "Processed response payload"}
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "500" in err_msg or "503" in err_msg:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with full jitter
                backoff_cap = min(max_delay, base_delay * (2 ** attempt))
                sleep_duration = random.uniform(0.5, backoff_cap)
                time.sleep(sleep_duration)
            else:
                # 400 Bad Request or 401 Unauthorized are non-retriable
                raise e''', language="python")

st.divider()

# Section 2: Token Streaming & TTFT Optimization
st.header("2. Token Streaming and Time-to-First-Token (TTFT)", divider="rainbow")
st.markdown("""
For conversational and interactive user interfaces, waiting 5 to 10 seconds for a complete 800-token completion 
destroys user experience. Token streaming utilizes Server-Sent Events (SSE) to deliver tokens chunk-by-chunk:
""")

st.graphviz_chart('''
digraph StreamingWorkflow {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Client [label="Frontend Client\\n(Browser/Streamlit)", fillcolor="#E1BEE7"];
    Server [label="API Inference Gateway\\n(vLLM / OpenAI)", fillcolor="#BBDEFB"];
    
    Client -> Server [label="POST /chat/completions\\nstream=true"];
    Server -> Client [label="Chunk 1: 'Here' (TTFT ~250ms)", color="#388E3C"];
    Server -> Client [label="Chunk 2: ' is'", color="#388E3C"];
    Server -> Client [label="Chunk 3: ' the'", color="#388E3C"];
    Server -> Client [label="Chunk 4: ' answer.'", color="#388E3C"];
    Server -> Client [label="data: [DONE]", color="#D32F2F"];
}
''', use_container_width=False)

st.code('''# Streamlit / Python Token Streaming Pattern
def stream_response_generator(prompt: str):
    # stream = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], stream=True)
    mock_tokens = ["Prompt ", "engineering ", "in ", "production ", "requires ", "rock-solid ", "streaming."]
    for token in mock_tokens:
        time.sleep(0.08)  # Simulated token generation latency
        yield token

# In Streamlit UI:
# st.write_stream(stream_response_generator("Tell me about streaming."))''', language="python")

st.divider()

# Section 3: Caching, Observability, and Cost Governance
st.header("3. Caching Architectures, Logging, & Cost Governance", divider="rainbow")
st.markdown("""
Running high-volume LLM pipelines without caching and cost monitoring is financially hazardous:
""")

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.subheader("Multi-Tier Caching")
    st.markdown("""
    * **Provider Prompt Caching**: Providers (OpenAI, Anthropic, DeepSeek) automatically cache shared system prompts 
    and long few-shot prefixes, cutting input token costs by 50% to 90% with zero cache invalidation effort.
    * **Semantic Application Cache (Redis / GPTCache)**: Evaluates vector cosine similarity between incoming queries. 
    If a query has 96%+ semantic match to a cached answer, returns cached response in < 15ms at $0.00 cost.
    """)

with col_c2:
    st.subheader("Logging & Observability Telemetry")
    st.markdown("""
    Every production request must log:
    * `timestamp`, `session_id`, `user_id`
    * `model_name`, `system_prompt_version`
    * `prompt_tokens`, `completion_tokens`, `cached_tokens`
    * `ttft_ms`, `total_latency_ms`
    * `total_cost_usd`
    """)

st.divider()

# Section 4: Interactive Production Telemetry Dashboard
st.header("4. Interactive Production Telemetry & Cost Dashboard", divider="rainbow")
st.markdown("Simulate a production API endpoint with live streaming, rate limiting, and cache hit monitoring.")

st.subheader("Live Token Streaming Demonstration")
user_input_sim = st.text_input("Enter Query to Test Stream:", value="Explain prompt caching in 20 words.")

if st.button("Execute Streaming API Call", type="primary"):
    def mock_stream():
        text = "Prompt caching stores frequent prefix tokens in GPU memory, drastically reducing latency and input costs by up to 90%."
        for word in text.split():
            yield word + " "
            time.sleep(0.05)
            
    st.write_stream(mock_stream())

st.divider()

st.subheader("Production Traffic & Cache Economics Simulator")
sim_rpm = st.slider("Target Requests Per Minute (RPM)", 100, 5000, 1000, step=100)
cache_hit_rate = st.slider("Semantic Cache Hit Rate (%)", 0, 80, 45, step=5)

# Economics
queries_per_month = sim_rpm * 60 * 24 * 30
cached_queries = queries_per_month * (cache_hit_rate / 100.0)
uncached_queries = queries_per_month - cached_queries

avg_cost_uncached = 0.004  # $0.004 per uncached query
avg_cost_cached = 0.0001    # Redis lookup cost

total_monthly_spend = (uncached_queries * avg_cost_uncached) + (cached_queries * avg_cost_cached)
baseline_spend_no_cache = queries_per_month * avg_cost_uncached
savings = baseline_spend_no_cache - total_monthly_spend

col_ec1, col_ec2, col_ec3 = st.columns(3)
col_ec1.metric("Monthly Requests", f"{queries_per_month:,}")
col_ec2.metric("Monthly Cost (With Cache)", f"${total_monthly_spend:,.2f}")
col_ec3.metric("Monthly Savings", f"${savings:,.2f}", delta=f"{cache_hit_rate}% saved")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is Prompt Caching at the API provider level and how do you structure prompts to maximize it?"):
        st.write("""
        Prompt caching allows the inference server to reuse pre-computed Key-Value (KV) attention caches across requests.
        To maximize cache hits:
        1. Place static, invariant content (system instructions, tool definitions, few-shot examples) at the **very beginning** of the prompt.
        2. Place dynamic, per-request content (user query, timestamps, retrieved docs) at the **end**.
        3. Do not inject volatile metadata (like current millisecond timestamps) into the system prompt.
        """)

    with st.expander("Q: How does a Token Bucket algorithm prevent 429 rate limit errors on client applications?"):
        st.write("""
        A token bucket algorithm maintains a local bucket that continuously refills with tokens at the allowed rate (e.g., 500 TPM). 
        Before sending a request, the client estimates token usage and deducts it from the bucket. 
        If insufficient tokens remain, the client delays the request locally rather than dispatching it to the API and getting hit with a 429 HTTP error.
        """)

with qa2:
    with st.expander("Trap to Avoid: Ingesting Streams Without Exception Handlers"):
        st.write("""
        Streaming connections can drop midway through generation due to network hiccups or provider timeouts. 
        If an application renders tokens directly without buffering or try/except blocks around stream chunks, 
        users end up with severed half-sentences. Always buffer stream segments and handle socket disconnects with graceful retries.
        """)

    with st.expander("Flashcards: Production API Terms"):
        fc_sel = st.selectbox(
            "Select Architecture Term:",
            ["Exponential Backoff with Jitter", "KV Cache Reuse", "TTFT (Time to First Token)", "Semantic Cache"]
        )
        if fc_sel == "Exponential Backoff with Jitter":
            st.info("Retrying failed calls with randomized increasing delays to avoid thundering herd server collapse.")
        elif fc_sel == "KV Cache Reuse":
            st.info("Reusing attention matrices stored in GPU VRAM for identical prefix prompts, slashing compute and latency.")
        elif fc_sel == "TTFT (Time to First Token)":
            st.info("The latency from client dispatch to the arrival of the first completion token.")
        else:
            st.info("Using vector similarity on past queries to return cached responses instantly without invoking an LLM.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can implement exponential backoff with full jitter to handle 429 errors.")
    st.checkbox("I know how to structure prompts to maximize provider KV prompt cache hits.")
with c2:
    st.checkbox("I understand how token streaming via SSE reduces perceived user latency.")
    st.checkbox("I can calculate the financial ROI of semantic caching in high-volume applications.")
