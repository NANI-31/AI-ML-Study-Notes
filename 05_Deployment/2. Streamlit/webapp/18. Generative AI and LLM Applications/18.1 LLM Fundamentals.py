import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="18.1 LLM Fundamentals",
    page_icon="🧠",
    layout="wide"
)

st.title("18.1 Large Language Model (LLM) Fundamentals in Streamlit")
st.caption("Foundations of Generative AI engineering: Transformer mechanics, ChatML schema, prompt design, token economics, context windows, sampling hyper-parameters, and response streaming.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The LLM Application Stack in Streamlit

Streamlit is the standard interface for Generative AI applications. 
Connecting interactive user interfaces to large language models requires understanding how the client manages chat history, token buffers, and sampling hyper-parameters:

""")

st.graphviz_chart(
    r"""
    digraph LLM_Fundamentals {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        ui [label="1. Streamlit Chat Interface\n• st.chat_input()\n• st.chat_message()\n• st.write_stream()", fillcolor="#DBEAFE", color="#1D4ED8"];
        state [label="2. Conversation State Manager\n• st.session_state['messages']\n• System, User, Assistant ChatML History", fillcolor="#FEF3C7", color="#D97706"];
        gateway [label="3. LLM Gateway (OpenAI / Anthropic / Ollama)\n• Tokenizer (BPE / SentencePiece)\n• Autoregressive Decoder Stack (T, Top-P)\n• SSE Token Stream", fillcolor="#D1FAE5", color="#059669"];

        ui -> state [label="Submits user turn"];
        state -> gateway [label="Formatted ChatML context"];
        gateway -> ui [label="SSE Streaming Tokens"];
    }
    """
)


tabs = st.tabs([
    "1. LLM Mechanics",
    "2. LLM APIs",
    "3. Prompt Engineering",
    "4. System vs User Roles",
    "5. Token Economics",
    "6. Context Windows",
    "7. Temperature & Sampling",
    "8. Streaming & Responses"
])

# =============================================================================
# 1. LLM MECHANICS
# =============================================================================
with tabs[0]:
    st.header("1. What Are Large Language Models?")
    st.markdown(r"""
    Large Language Models (LLMs) are **autoregressive neural networks** based on the Transformer architecture.
    Trained on vast text corpora to predict the next token:
    $$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 LLM Lifecycle Stages**")
        st.markdown(r"""
        1. **Pre-training**: Massive unsupervised self-learning on trillions of tokens (Web, books, code). Learns world facts, grammar, and syntax.
        2. **Supervised Fine-Tuning (SFT)**: High-quality instruction-following demonstration datasets (ChatML format).
        3. **Alignment (RLHF / DPO)**: Reinforcement Learning from Human Feedback aligns model outputs with human intent, safety, and helpfulness.
        4. **Reasoning / Inference-Time Search**: Extended chain-of-thought (CoT) and search exploration (e.g. OpenAI o1/o3, DeepSeek-R1).
        """)

    with col2:
        st.markdown("**🖥️ Model Family Archetypes**")
        with st.container(border=True):
            st.markdown(r"""
            ### Major Model Archetypes

            | Archetype | Examples | Optimal Use Case |
            | :--- | :--- | :--- |
            | **Proprietary Frontier**| GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro | Complex reasoning, coding, multimodal vision. |
            | **Open Weights** | Llama 3.1 70B, Mistral Large, Qwen 2.5 | Private on-prem deployment, custom fine-tuning. |
            | **Reasoning Specialized**| OpenAI o1, DeepSeek-R1 | Competitive mathematics, deep formal logic, multi-step planning. |
            | **Edge / Small (SLM)** | Phi-3.5 (3.8B), Gemma 2 (2B) | Mobile devices, low-latency edge gateways. |
            """)

# =============================================================================
# 2. LLM APIS
# =============================================================================
with tabs[1]:
    st.header("2. LLM APIs & Integration Clients")
    st.markdown(r"""
    Modern LLM APIs communicate over HTTPS using the standard `/v1/chat/completions` REST interface.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import requests

def query_llm_gateway(messages: list[dict], model: str, temperature: float):
    # Standard OpenAI-compatible API payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512
    }
    headers = {
        "Authorization": f"Bearer {st.secrets.get('OPENAI_API_KEY', 'sk-demo')}",
        "Content-Type": "application/json"
    }
    
    # In production:
    # response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    # return response.json()["choices"][0]["message"]["content"]
    
    # Mock response for resilient demo
    return f"Simulated analysis generated by {model} (T={temperature})."
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            gateway_provider = st.selectbox("API Provider", ["OpenAI (v1/chat)", "Anthropic (v1/messages)", "Ollama (Localhost:11434)"], key="api_prov_sel")
            selected_model = st.selectbox("Target Model", ["gpt-4o", "claude-3-5-sonnet", "llama3.1:70b"], key="api_mod_sel")
            
            st.caption(f"Active Client: `{gateway_provider}` → `{selected_model}`")
            st.success("Universal REST payload format supported across cloud gateways.")

# =============================================================================
# 3. PROMPT ENGINEERING
# =============================================================================
with tabs[2]:
    st.header("3. Systematic Prompt Engineering Frameworks")
    st.markdown(r"""
    Prompt engineering guides model behavior through structured context framing:
    - **Role Definition**: Assigning expertise persona.
    - **Few-Shot Exemplars**: Demonstrating input-output pairs.
    - **Chain-of-Thought (CoT)**: Forcing step-by-step intermediate reasoning.
    - **Output Constraints**: Restricting format (e.g. valid JSON only).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def build_prompt_template(user_query: str, domain: str) -> str:
    template = f"""You are a senior {domain} specialist.

CRITICAL CONSTRAINTS:
1. Provide answers formatted strictly as structured markdown with bullet points.
2. If uncertain, state "Insufficient information" rather than guessing.
3. Cite foundational principles where applicable.

USER QUESTION:
{user_query}

DETAILED RESPONSE:"""
    return template

query = st.text_input("Enter Question", "What is an embedding vector?")
rendered_prompt = build_prompt_template(query, "Machine Learning")
st.text_area("Compiled Prompt", rendered_prompt, height=180)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            dom = st.selectbox("Specialist Domain", ["Machine Learning", "FinTech Compliance", "Bioinformatics"], key="pe_dom_sel")
            q_in = st.text_input("Enter Question", "What is an embedding vector?", key="pe_query_in")

            built_prompt = f"""You are a senior {dom} specialist.

CRITICAL CONSTRAINTS:
1. Provide answers formatted strictly as structured markdown.
2. If uncertain, state "Insufficient information".

USER QUESTION:
{q_in}

RESPONSE:"""
            st.caption("Compiled Prompt String:")
            st.text_area("Prompt Preview", built_prompt, height=150, key="pe_out_ta")

# =============================================================================
# 4. SYSTEM VS USER ROLES
# =============================================================================
with tabs[3]:
    st.header("4. System, User, and Assistant Message Roles")
    st.markdown(r"""
    Modern LLMs operate over multi-turn conversations formatted as **ChatML message lists**:
    - `system`: Top-level steering instructions and behavioral boundaries.
    - `user`: Inquiries and instructions issued by the end-user.
    - `assistant`: Previous model responses providing conversational continuity.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# ChatML Multi-Turn Conversation History
messages = [
    {
        "role": "system", 
        "content": "You are a concise enterprise data analyst. Always respond in tables."
    },
    {
        "role": "user", 
        "content": "Compare Q1 and Q2 revenue: Q1 was $1.2M, Q2 was $1.5M."
    },
    {
        "role": "assistant", 
        "content": "| Quarter | Revenue | Growth |\n|---|---|---|\n| Q1 | $1.2M | - |\n| Q2 | $1.5M | +25% |"
    },
    {
        "role": "user", 
        "content": "What was the growth delta?"
    }
]

st.json(messages)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            messages = [
                {"role": "system", "content": "You are a concise enterprise data analyst."},
                {"role": "user", "content": "Compare Q1 and Q2 revenue: Q1 was $1.2M, Q2 was $1.5M."},
                {"role": "assistant", "content": "Quarter: Q1 ($1.2M) -> Q2 ($1.5M). Delta: +$300k (+25%)."},
                {"role": "user", "content": "What was the growth percentage?"}
            ]
            st.caption("Structured ChatML Message History:")
            st.json(messages)
            st.caption("System prompt sets persona; user/assistant turns maintain dialogue state.")

# =============================================================================
# 5. TOKEN ECONOMICS
# =============================================================================
with tabs[4]:
    st.header("5. Token Usage & Cost Economics")
    st.markdown(r"""
    LLMs do not process raw characters or words; they process **Tokens** (sub-word fragments via BPE).
    - Rule of thumb: $1 \text{ token} \approx 0.75 \text{ words}$ (or $\sim 4 \text{ English characters}$).
    - Pricing separates **Input/Prompt Tokens** from **Output/Completion Tokens** (outputs cost $3\times - 4\times$ more due to autoregressive generation).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

prompt_text = st.text_area("Input Prompt", "Explain gradient descent optimization in detail.")

# Approximate token calculation
approx_tokens = int(len(prompt_text.split()) * 1.33)
input_cost_per_million = 2.50   # $2.50 per 1M prompt tokens
output_cost_per_million = 10.00  # $10.00 per 1M completion tokens

est_completion_tokens = 300
total_cost = (
    (approx_tokens / 1e6) * input_cost_per_million +
    (est_completion_tokens / 1e6) * output_cost_per_million
)

c1, c2, c3 = st.columns(3)
c1.metric("Input Tokens", approx_tokens)
c2.metric("Output Tokens", est_completion_tokens)
c3.metric("Est. Query Cost", f"${total_cost:.5f}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            p_text = st.text_area("Sample Prompt", "Explain gradient descent optimization in detail for machine learning.", key="tok_p_ta")
            
            approx_toks = int(len(p_text.split()) * 1.33)
            est_comp_toks = 350
            cost = ((approx_toks / 1e6) * 2.50) + ((est_comp_toks / 1e6) * 10.00)

            c1, c2, c3 = st.columns(3)
            c1.metric("Input Tokens", approx_toks)
            c2.metric("Est. Output Tokens", est_comp_toks)
            c3.metric("Cost per Query", f"${cost:.5f}")

# =============================================================================
# 6. CONTEXT WINDOWS
# =============================================================================
with tabs[5]:
    st.header("6. Context Windows & Truncation Strategies")
    st.markdown(r"""
    The **Context Window** defines the maximum total tokens (input + output) a model can attend to in a single forward pass.
    Exceeding the context window triggers `BadRequestError: context_length_exceeded`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Context Management Strategies**")
        st.markdown(r"""
        ### Managing Large Histories

        1. **Sliding Window Memory**: Retain only the last $K$ turns (e.g., last 6 messages) plus the persistent system prompt.
        2. **Summarization Memory**: Periodically summarize older messages with an auxiliary LLM call.
        3. **RAG (Retrieval-Augmented Generation)**: Store large documents in a Vector DB and retrieve only the top-$K$ relevant chunks ($<4\text{k tokens}$).
        """)

    with col2:
        st.markdown("**🖥️ Context Window Gauge**")
        with st.container(border=True):
            active_model_ctx = 128000  # 128k context window
            curr_tokens = st.slider("Simulated Conversation Tokens", 1000, 150000, 42000, step=5000, key="ctx_tok_sl")

            pct_used = curr_tokens / active_model_ctx
            st.metric("Context Utilization", f"{curr_tokens:,} / {active_model_ctx:,} tokens", f"{pct_used:.1%}")
            st.progress(min(pct_used, 1.0))

            if pct_used > 0.85:
                st.warning("⚠️ Approaching context horizon! Triggering automatic conversation summarization.")
            else:
                st.success("✅ Context depth healthy.")

# =============================================================================
# 7. TEMPERATURE & SAMPLING
# =============================================================================
with tabs[6]:
    st.header("7. Temperature, Top-P, and Generation Settings")
    st.markdown(r"""
    Sampling parameters govern the probability distribution over candidate tokens:
    - **Temperature ($T$)**: Controls output randomness. Low ($0.0 - 0.2$) produces deterministic, analytical outputs. High ($0.7 - 1.2$) produces creative variety.
    - **Top-P (Nucleus Sampling)**: Restricts token choices to the smallest cumulative probability mass $\le P$.
    - **Frequency & Presence Penalties**: Penalize repeated tokens to prevent circular loops.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

c1, c2 = st.columns(2)
with c1:
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.95, step=0.05)
with c2:
    max_tokens = st.number_input("Max Output Tokens", 64, 4096, 512, step=64)
    seed = st.number_input("Seed (Deterministic)", value=42)

st.write(f"Inference Configuration: T={temperature}, Top-P={top_p}, MaxTokens={max_tokens}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                t_val = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05, key="samp_t_sl")
                top_p_val = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.95, step=0.05, key="samp_tp_sl")
            with sc2:
                max_t = st.number_input("Max Output Tokens", 64, 4096, 512, step=64, key="samp_max_t")
                seed_val = st.number_input("Seed (Deterministic)", value=42, key="samp_seed")

            st.caption("Active Hyperparameter Set:")
            st.json({
                "temperature": t_val,
                "top_p": top_p_val,
                "max_tokens": max_t,
                "seed": seed_val
            })

# =============================================================================
# 8. STREAMING & RESPONSES
# =============================================================================
with tabs[7]:
    st.header("8. Streaming Model Responses (`st.write_stream`)")
    st.markdown(r"""
    Waiting for an entire 1,000-token response introduces 10+ seconds of perceived latency.
    Using **Server-Sent Events (SSE)** and `st.write_stream()`, tokens render immediately as they are generated by the model.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import time

def stream_tokens_generator(prompt: str):
    response = (
        "Large language models generate tokens sequentially using autoregressive decoding. "
        "Streaming with st.write_stream drastically improves perceived latency and user experience!"
    )
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.04)

if st.button("Generate Streaming Response"):
    st.write_stream(stream_tokens_generator("Explain streaming"))
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def stream_tokens_demo():
                sample_text = (
                    "Large language models generate tokens sequentially using autoregressive decoding. "
                    "Streaming with st.write_stream drastically improves perceived latency and user experience!"
                )
                for word in sample_text.split(" "):
                    yield word + " "
                    time.sleep(0.03)

            if st.button("Generate Streaming Response", key="stream_demo_btn"):
                st.write_stream(stream_tokens_demo())
            else:
                st.info("Click the button above to observe real-time token streaming via `st.write_stream`.")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### LLM Fundamentals Best Practices

| Parameter / Technique | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Temperature** | Setting $T=1.0$ for SQL code generation | Use $T=0.0 - 0.2$ for deterministic, factual, and code tasks. |
| **Perceived Latency** | Blocking execution with unstreamed responses | Stream tokens incrementally with `st.write_stream`. |
| **History Growth** | Appending unlimited turns to conversation list | Implement a sliding window or periodic LLM summarizer. |
| **Model Credentials** | Hardcoding API keys in code | Store keys in `.streamlit/secrets.toml`. |
""")
