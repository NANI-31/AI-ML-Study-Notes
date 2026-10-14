import streamlit as st
import json

st.set_page_config(
    page_title="24.3 OpenAI Advanced Applications - Prompt Engineering",
    page_icon="🚀",
    layout="wide"
)

# Header Section
st.title("24.3 Advanced OpenAI Systems: Retrieval, Batches, & Production Architecture")
st.caption("Chapter 24: Prompt Engineering with OpenAI | Domain: 1. Prompt Engineering")
st.markdown("""
At enterprise scale, applications move beyond single-turn API calls into **fully managed Assistants with vector search, 
50%-discount asynchronous Batch API pipelines, event-driven streaming engines, automated evaluation harnesses, 
and aggressive prompt caching architectures**.
""")

st.divider()

# Section 1: Managed Retrieval & Assistants API
st.header("1. Managed Retrieval: Assistants API & File Search", divider="rainbow")
st.markdown("""
OpenAI's **Assistants API** provides managed state, automatic session thread history, and built-in vector search 
via the `file_search` tool:
""")

col_ast1, col_ast2 = st.columns(2)
with col_ast1:
    st.subheader("Managed Vector Store Architecture")
    st.markdown("""
    * **Automatic Chunking & Embeddings**: Files uploaded to a vector store are automatically chunked, embedded via `text-embedding-3-large`, and indexed in OpenAI's managed infrastructure.
    * **Automatic Query Rewriting**: In multi-turn threads, the Assistant rewrites user follow-ups to maintain conversational context during vector retrieval.
    * **Citations & Grounding**: File search automatically annotates output with file IDs and snippet offsets.
    """)

with col_ast2:
    st.subheader("Assistants Setup Pattern")
    st.code('''# Assistants API with File Search
assistant = client.beta.assistants.create(
    name="Enterprise Legal Assistant",
    instructions="You are a legal research assistant. Answer questions strictly based on uploaded policy files.",
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": ["vs_abc123xyz"]
        }
    }
)''', language="python")

st.divider()

# Section 2: Async Batches API (50% Cost Discount)
st.header("2. Asynchronous Batches API: 50% Cost Savings", divider="rainbow")
st.markdown("""
For non-real-time workloads—such as overnight document classification, weekly dataset labeling, back-testing prompts, 
or offline embeddings—OpenAI offers the **Batch API**.
* **50% Cost Reduction**: All input and output tokens are billed at exactly half the standard price.
* **Higher Rate Limits**: Batches bypass standard RPM/TPM limits and run on idle GPU cluster capacity.
* **24-Hour SLA**: Results are typically ready in 2 to 6 hours with a guaranteed 24-hour turnaround.
""")

st.code('''# Asynchronous Batch Submission Pattern
import json

# 1. Format requests as JSONL
# {"custom_id": "req-001", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [...]}}

# 2. Upload file
batch_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

# 3. Create batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch created with ID: {batch_job.id}")''', language="python")

st.divider()

# Section 3: Cost Optimization & Automatic Prompt Caching
st.header("3. Automatic KV Prompt Caching & Model Distillation", divider="rainbow")
st.markdown("""
OpenAI provides **automatic prompt caching** on supported models (GPT-4o, GPT-4o-mini). 
Whenever a prompt prefix contains 1,024 or more identical tokens, OpenAI reuses the pre-computed attention cache:
* **Input Token Discount**: Cached prompt tokens receive a **50% discount** automatically.
* **Latency Reduction**: Bypasses prefill computation, delivering up to 80% faster TTFT.
* **Zero Configuration**: No special headers needed—simply structure static instructions at the start of prompts!
""")

st.graphviz_chart('''
digraph PromptCacheOptimization {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    StaticPrefix [label="Static Prefix (>= 1,024 tokens)\\nSystem instructions, schemas, few-shot examples\\n(CACHE HIT: 50% OFF & 80% FASTER)", fillcolor="#C8E6C9"];
    DynamicSuffix [label="Dynamic Suffix (Variable tokens)\\nUser question, current document, timestamp\\n(STANDARD BILLING)", fillcolor="#FFF9C4"];
    InferenceEngine [label="Inference KV Cache\\nReuse", fillcolor="#BBDEFB"];
    
    StaticPrefix -> InferenceEngine;
    DynamicSuffix -> InferenceEngine;
}
''', use_container_width=False)

st.divider()

# Section 4: Interactive Batch vs Realtime Economics Calculator
st.header("4. Batch vs Real-Time Economics Calculator", divider="rainbow")
st.markdown("Calculate potential monthly savings by transitioning offline workloads to OpenAI's Batch API and leveraging prompt caching.")

col_b1, col_b2 = st.columns(2)

with col_b1:
    offline_volume = st.number_input("Monthly Offline Batch Inquiries", value=500000, step=100000)
    avg_input_toks = st.slider("Avg Input Tokens", 500, 10000, 2000, step=500)
    avg_output_toks = st.slider("Avg Output Tokens", 100, 2000, 400, step=100)
    cache_rate = st.slider("Prompt Cache Hit Rate on Static Prefix (%)", 0, 90, 60, step=10)

with col_b2:
    # GPT-4o standard: $2.50/M in, $10.00/M out
    # Batch is 50% off. Cached input is an additional 50% off cached portion.
    std_input_cost = (offline_volume * avg_input_toks / 1_000_000) * 2.50
    std_output_cost = (offline_volume * avg_output_toks / 1_000_000) * 10.00
    total_standard = std_input_cost + std_output_cost
    
    # Batch pricing: 50% off
    batch_base = total_standard * 0.50
    # Additional savings from cache
    batch_with_cache = batch_base * (1 - (cache_rate / 100.0 * 0.25))
    
    saved_dollars = total_standard - batch_with_cache
    
    st.subheader("Financial Comparison")
    st.metric("Standard Real-Time Cost", f"${total_standard:,.2f}")
    st.metric("Batch API + Caching Cost", f"${batch_with_cache:,.2f}")
    st.metric("Total Monthly Savings", f"${saved_dollars:,.2f}", delta=f"-{(saved_dollars/total_standard)*100:.1f}%")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: How does OpenAI's automatic prompt caching differ from semantic caching in Redis?"):
        st.write("""
        * **OpenAI Prompt Caching**: Operates at the **token level on GPU Key-Value attention tensors**. 
        Requires exact prefix token match (1,024+ tokens). Reuses the prefill phase computation while still generating a novel completion for the suffix.
        * **Semantic Caching (Redis/GPTCache)**: Operates at the **semantic query level**. 
        Computes cosine similarity of the entire user question and returns an identical stored answer if above a threshold. 
        Zero tokens are sent to OpenAI, but answers may be stale or lack dynamic personalization.
        """)

    with st.expander("Q: When should an enterprise use the Assistants API versus building a custom LangChain/LlamaIndex RAG pipeline?"):
        st.write("""
        * **Assistants API**: Best for rapid prototyping and enterprise teams that want fully managed thread state, vector storage, 
        and automatic chunking without maintaining Redis, Pinecone, or background worker clusters.
        * **Custom RAG (LangChain/LlamaIndex)**: Essential when you require custom hybrid search (BM25 + Dense), 
        specialized rerankers (Cohere), multi-tenant ACL permissions, private on-prem vector databases, or non-OpenAI LLM failover.
        """)

with qa2:
    with st.expander("Trap to Avoid: Polling Assistants API Runs Synchronously"):
        st.write("""
        In early versions of the Assistants API, developers polled `client.beta.threads.runs.retrieve()` in tight while loops, 
        wasting compute and hitting rate limits. Modern production code uses the streaming helper:
        `with client.beta.threads.runs.stream(...) as stream: stream.until_done()`, which uses SSE event listeners 
        to stream tokens as soon as they are ready.
        """)

    with st.expander("Flashcards: Advanced OpenAI Features"):
        fc_sel = st.selectbox(
            "Select Architecture Term:",
            ["Batch API 50% SLA", "Automatic KV Cache", "File Search Tool", "Stream Event Handlers"]
        )
        if fc_sel == "Batch API 50% SLA":
            st.info("Asynchronous endpoint offering 50% price cuts for jobs that can complete within 24 hours.")
        elif fc_sel == "Automatic KV Cache":
            st.info("Provider-level attention cache reuse on static prefixes >= 1,024 tokens.")
        elif fc_sel == "File Search Tool":
            st.info("Managed vector database and chunking engine built into the Assistants API.")
        else:
            st.info("Client-side event listeners that process streaming chunks reactively without manual polling.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know when to use the Batch API to achieve 50% cost reductions.")
    st.checkbox("I can structure prompt prefixes to maximize OpenAI automatic KV cache reuse.")
with c2:
    st.checkbox("I understand the trade-offs between managed Assistants API and custom RAG pipelines.")
    st.checkbox("I can handle streaming runs reactively using stream event listeners.")
