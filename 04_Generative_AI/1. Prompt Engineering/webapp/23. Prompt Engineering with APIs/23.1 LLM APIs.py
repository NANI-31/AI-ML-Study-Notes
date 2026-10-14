import streamlit as st
import json

st.set_page_config(
    page_title="23.1 LLM APIs - Prompt Engineering",
    page_icon="🔌",
    layout="wide"
)

# Header Section
st.title("23.1 LLM APIs: Architecture, Protocols, and Foundations")
st.caption("Chapter 23: Prompt Engineering with APIs | Domain: 1. Prompt Engineering")
st.markdown("""
In production applications, prompts do not live in chat web interfaces. They are transmitted programmatically 
across **REST and gRPC HTTP APIs** to hosted inference clusters. Mastering LLM APIs involves understanding 
request/response schemas, bearer token authentication, secure secret management, and model routing decisions.
""")

st.divider()

# Section 1: Anatomy of an LLM API Request & Response
st.header("1. Anatomy of an LLM API Request & Response", divider="rainbow")
st.markdown("""
Modern LLM APIs follow standardized JSON schemas over HTTP POST. 
Below is the standard payload structure exchanged between your application and the inference gateway:
""")

col_req, col_res = st.columns(2)
with col_req:
    st.subheader("HTTP Request Body (JSON)")
    st.code('''POST /v1/chat/completions HTTP/1.1
Host: api.openai.com
Authorization: Bearer sk-prod-secret-key-1234
Content-Type: application/json

{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise financial analyst."
    },
    {
      "role": "user",
      "content": "Summarize Q3 EBITDA growth for Acme Corp."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 500,
  "response_format": { "type": "json_object" }
}''', language="http")

with col_res:
    st.subheader("HTTP Response Body (JSON)")
    st.code('''HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "chatcmpl-9x7K91mPqL81vA",
  "object": "chat.completion",
  "created": 1728749100,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\\"company\\": \\"Acme Corp\\", \\"q3_ebitda_growth\\": \\"14.2%\\"}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 28,
    "total_tokens": 70
  }
}''', language="json")

st.divider()

# Section 2: Authentication & Secure Key Management
st.header("2. Authentication and Secure Key Management", divider="rainbow")
st.markdown("""
API keys grant direct access to billed inference resources. Leaked keys can result in tens of thousands of dollars 
in unauthorized compute spend within hours. Enterprise applications adhere to zero-trust secret handling practices:
""")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.subheader("Environment Variables")
    st.markdown("""
    * Store keys in `.env` or system environment (`os.getenv("OPENAI_API_KEY")`).
    * **Never** hardcode keys into source files, Jupyter notebooks, or Git repositories.
    * Add `.env` to `.gitignore` on project initialization.
    """)

with col_s2:
    st.subheader("Secret Managers")
    st.markdown("""
    * Enterprise deployments retrieve API credentials at runtime from AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault.
    * Implement automated key rotation schedules (e.g. 90-day cycles).
    """)

with col_s3:
    st.subheader("Least-Privilege Scoping")
    st.markdown("""
    * Use project-scoped or team-scoped restricted API tokens rather than root organization admin keys.
    * Enforce hard monthly spending limits and IP CIDR allowlists.
    """)

st.code('''# Secure API Initialization Pattern
import os
from openai import OpenAI

# The client automatically reads the OPENAI_API_KEY environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("CRITICAL: OPENAI_API_KEY environment variable is missing.")

client = OpenAI(
    api_key=api_key,
    timeout=30.0,      # Strict request timeout in seconds
    max_retries=2      # Automated retry limit on 429/500 errors
)''', language="python")

st.divider()

# Section 3: Model Selection Strategy Matrix
st.header("3. Strategic Model Selection Matrix", divider="rainbow")
st.markdown("""
Selecting the right model is an engineering balance between **reasoning capability, latency, context window, and financial cost**:
""")

st.graphviz_chart('''
digraph ModelSelection {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Task [label="Incoming Production Task", fillcolor="#E1BEE7"];
    Dec [label="Does task require deep\\nmulti-step logic or code?", shape=diamond, fillcolor="#FFF9C4"];
    
    Tier1 [label="Frontier Reasoning Tier\\n(e.g., GPT-4o, Claude 3.5 Sonnet)\\nHigh Cost / Higher Latency", fillcolor="#FFCDD2"];
    
    Dec2 [label="Is latency critical\\n(< 300ms SLA)?", shape=diamond, fillcolor="#FFF9C4"];
    Tier2 [label="Edge / High-Speed Tier\\n(e.g., GPT-4o-mini, Claude 3 Haiku)\\nUltra Low Cost / Rapid TTFT", fillcolor="#C8E6C9"];
    Tier3 [label="Specialized / Fine-Tuned 8B\\n(vLLM hosted internal model)\\nZero Cloud Data Transfer", fillcolor="#BBDEFB"];
    
    Task -> Dec;
    Dec -> Tier1 [label="Yes (Complex)"];
    Dec -> Dec2 [label="No (Standard)"];
    Dec2 -> Tier2 [label="Yes (Cloud API)"];
    Dec2 -> Tier3 [label="No (Internal Data Privacy)"];
}
''', use_container_width=False)

st.divider()

# Section 4: Interactive API Request Simulator
st.header("4. Interactive API Request Simulator Workbench", divider="rainbow")
st.markdown("Configure payload parameters and inspect the generated HTTP request and response token breakdown.")

col_sim_in, col_sim_out = st.columns(2)

with col_sim_in:
    sim_model = st.selectbox("Select Target Model:", ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "llama-3.1-70b"])
    sim_role = st.selectbox("System Persona:", ["Strict JSON Extractor", "Senior Code Reviewer", "Support Agent"])
    sim_query = st.text_input("User Prompt:", value="Extract customer ID and sentiment from: 'ID #4412: Great experience!'")
    sim_fmt = st.checkbox("Enforce JSON Object Response Format", value=True)

with col_sim_out:
    # Synthetic token and latency calculation
    in_tokens = len(sim_query.split()) * 2 + 35
    out_tokens = 25 if sim_fmt else 45
    est_latency = 420 if "mini" in sim_model or "haiku" in sim_model else 850
    
    st.subheader("Simulated API Telemetry")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Input Tokens", in_tokens)
    col_m2.metric("Output Tokens", out_tokens)
    col_m3.metric("Est. Latency", f"{est_latency} ms")
    
    st.markdown("**Simulated Completion Payload**:")
    sim_completion = {
        "customer_id": "4412",
        "sentiment": "POSITIVE",
        "confidence": 0.99
    } if sim_fmt else "Customer ID is 4412 and customer sentiment is positive."
    
    if sim_fmt:
        st.json(sim_completion)
    else:
        st.write(sim_completion)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_col1, qa_col2 = st.columns(2)

with qa_col1:
    with st.expander("Q: What is `finish_reason` in an LLM API response and why is it critical?"):
        st.write("""
        `finish_reason` communicates why generation terminated:
        * `"stop"`: Model completed its response naturally or encountered a stop token.
        * `"length"`: Generation was cut off prematurely by `max_tokens`. (Indicates truncated output or broken JSON!)
        * `"tool_calls"`: Model paused to request tool/function execution.
        * `"content_filter"`: Generation halted by safety guardrails.
        Production parsers must always check `finish_reason == "stop"` before attempting to parse completions.
        """)

    with st.expander("Q: What are the primary differences between REST API vs WebSocket/gRPC for LLMs?"):
        st.write("""
        * **REST (Server-Sent Events / SSE)**: Standard unidirectional streaming over HTTP/1.1 or HTTP/2. Ideal for streaming assistant tokens to web UIs.
        * **WebSockets / gRPC**: Full duplex bidirectional streams. Essential for real-time speech-to-speech audio interactions (e.g. OpenAI Realtime API) 
        where the client must stream microphone chunks while simultaneously listening for interruptions.
        """)

with qa_col2:
    with st.expander("Trap to Avoid: Hardcoding API Keys in Client-Side Code"):
        st.write("""
        Never embed an LLM provider API key in frontend single-page apps (React, Vue, mobile apps). 
        Anyone can inspect browser network traffic and extract the token. Always proxy requests through 
        your secure backend server which authenticates the user session and attaches the private key.
        """)

    with st.expander("Flashcards: API Terminology"):
        sel_card = st.selectbox(
            "Select API Term:",
            ["finish_reason='length'", "Bearer Token", "SSE (Server-Sent Events)", "Token Quota"]
        )
        if sel_card == "finish_reason='length'":
            st.info("The output was forcefully truncated because the model hit max_tokens before completing.")
        elif sel_card == "Bearer Token":
            st.info("The standard HTTP Authorization header format: 'Authorization: Bearer <API_KEY>'.")
        elif sel_card == "SSE (Server-Sent Events)":
            st.info("The HTTP protocol used to stream token chunks from the inference server to the client.")
        else:
            st.info("Rate limits enforced by providers measured in RPM (Requests Per Minute) and TPM (Tokens Per Minute).")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the JSON request and response contracts for chat completion APIs.")
    st.checkbox("I know how to check finish_reason before attempting to parse response JSON.")
with c2:
    st.checkbox("I can implement secure environment-variable-based key authentication.")
    st.checkbox("I can choose between frontier reasoning models and high-speed mini models based on SLAs.")
