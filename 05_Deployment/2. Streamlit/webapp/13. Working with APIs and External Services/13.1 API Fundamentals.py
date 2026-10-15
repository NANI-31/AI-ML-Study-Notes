import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlencode

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="API Fundamentals in Streamlit",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 REST API Fundamentals & Web Services — Complete Guide")
st.markdown(
    """
Modern Generative AI, machine learning, and enterprise Streamlit applications rarely operate 
in total isolation. They interact with external services — such as **LLM inference gateways 
(OpenAI, Anthropic, vLLM, Ollama)**, vector databases, weather services, payment rails, 
and internal enterprise microservices — via **HTTP/REST APIs**. 
Mastering REST fundamentals, HTTP methods, status code semantics, headers, and JSON serialization 
is essential for connecting Streamlit to external intelligence.
"""
)

st.divider()

# ==============================================================================
# 1. What is an API? & 2. REST API Concepts
# ==============================================================================
st.header("1. What is an API? & 2. REST Architecture Principles", divider="rainbow")
st.markdown(
    r"""
An **API (Application Programming Interface)** is a standardized communication contract 
between software applications. In web development, **REST (Representational State Transfer)** 
is the dominant architectural standard built on standard HTTP protocol rules:

| REST Principle | Meaning in Web Applications | Production AI Example |
| :--- | :--- | :--- |
| **Statelessness** | Every HTTP request contains all context needed; the server retains no session state. | Each prompt completion request passes authentication, model, and message history explicitly. |
| **Resource-Oriented URIs**| Endpoints are named after plural nouns, not verbs. | `GET /v1/models` (List models), `POST /v1/chat/completions` (Create chat completion). |
| **Uniform Interface** | Standard HTTP verbs (`GET`, `POST`, `DELETE`) indicate action semantics. | `DELETE /v1/fine_tuning/jobs/ft-123` cancels a training run. |
| **Standard Representations**| Standard data interchange payloads. | Requests and responses use RFC-compliant `application/json`. |
"""
)

col_arc1, col_arc2 = st.columns(2)

with col_arc1:
    st.markdown("**💻 Python Source Code (REST Client Architecture)**")
    code_client = '''import requests

# Standard REST Endpoint Pattern:
BASE_URL = "https://api.openai.com/v1"

# 1. List Available Models (GET Resource)
resp_models = requests.get(
    f"{BASE_URL}/models",
    headers={"Authorization": "Bearer sk-..."}
)

# 2. Dispatch Completion (POST Resource with JSON Body)
resp_chat = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={
        "Authorization": "Bearer sk-...",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Explain APIs"}]
    }
)'''
    st.code(code_client, language="python")

with col_arc2:
    st.markdown("**🖥️ REST Request / Response Cycle Diagram**")
    rest_dag = """
    digraph {
        rankdir=LR;
        node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#f8fafc", color="#94a3b8"];
        edge [color="#64748b", fontname="Helvetica", fontsize=9];

        Client [label="Streamlit App\\n(Python requests / httpx)", fillcolor="#e0f2fe", color="#0284c7"];
        Gateway [label="API Gateway / Load Balancer\\n(TLS, Auth, Rate Limiter)", fillcolor="#fef3c7", color="#d97706"];
        Inference [label="LLM Inference Cluster\\n(vLLM, Anthropic, OpenAI)", fillcolor="#fae8ff", color="#c026d3"];

        Client -> Gateway [label="HTTP POST /v1/completions\\n(JSON Payload)"];
        Gateway -> Inference [label="Authorized Forward"];
        Inference -> Gateway [label="Token Generation"];
        Gateway -> Client [label="HTTP 200 OK\\n(JSON Response)"];
    }
    """
    st.graphviz_chart(rest_dag, width="stretch")

st.divider()

# ==============================================================================
# 3. HTTP Methods & 4. GET and POST Requests
# ==============================================================================
st.header("3. HTTP Methods & 4. GET vs. POST Request Deep-Dive", divider="rainbow")
st.markdown(
    r"""
The choice of HTTP method communicates the intent of your operation to the server:

| HTTP Verb | Safe? | Idempotent? | Request Body? | Typical GenAI / Data Science Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`GET`** | ✅ Yes | ✅ Yes | ❌ No | Fetch model lists, query token credit balances, download datasets. |
| **`POST`** | ❌ No | ❌ No | ✅ Yes | Dispatch prompts, execute inference, submit user feedback, upload files. |
| **`PUT`** | ❌ No | ✅ Yes | ✅ Yes | Replace an entire prompt template configuration or model card. |
| **`PATCH`**| ❌ No | ❌ No | ✅ Yes | Partially update a model deployment status (e.g. `status: 'Production'`). |
| **`DELETE`**| ❌ No | ✅ Yes | ❌ Rarely | Delete a cached embedding index, cancel a fine-tuning run. |
"""
)

col_gp1, col_gp2 = st.columns(2)

with col_gp1:
    st.markdown("**💻 Python Source Code (GET vs. POST Syntax)**")
    code_get_post = '''import requests

# GET: Parameters passed in URL Query String (?key=value)
# Safe, idempotent, cached by default:
res_get = requests.get(
    "https://api.weather.gov/alerts",
    params={"area": "CA", "severity": "Severe"},
    timeout=10
)

# POST: Data passed in HTTP Request Body
# Non-idempotent, payload encrypted in TLS tunnel:
res_post = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={"x-api-key": "sk-ant-..."},
    json={
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}]
    },
    timeout=30
)'''
    st.code(code_get_post, language="python")

with col_gp2:
    st.markdown("**🖥️ Live Interactive Method Payload Inspector**")
    method_demo = st.radio("Select HTTP Method", ["GET (Read Operation)", "POST (Create / Infer)"], horizontal=True, key="rad_meth_demo")

    if "GET" in method_demo:
        st.info("🌐 **HTTP GET Request Structure**")
        st.code("GET /v1/models?limit=5&category=chat HTTP/1.1\nHost: api.ai-hub.org\nAuthorization: Bearer sk-token", language="http")
        st.caption("Payload is serialized directly into the URL query parameters; no body payload exists.")
    else:
        st.success("📤 **HTTP POST Request Structure**")
        sample_post_body = json.dumps({"model": "claude-3.7", "prompt": "Refactor code", "temperature": 0.2}, indent=2)
        st.code(f"POST /v1/chat/completions HTTP/1.1\nHost: api.ai-hub.org\nContent-Type: application/json\nContent-Length: {len(sample_post_body)}\n\n{sample_post_body}", language="http")
        st.caption("Payload is packed inside the HTTP Request Body as an arbitrary JSON document.")

st.divider()

# ==============================================================================
# 5. HTTP Status Codes
# ==============================================================================
st.header("5. HTTP Status Codes & Error Semantics", divider="rainbow")
st.markdown(
    r"""
HTTP status codes inform the client whether a request succeeded, failed due to client error, 
or crashed due to server issues:

| Status Code Series | Category | Meaning in AI / Cloud Applications |
| :--- | :--- | :--- |
| **`200 OK`** | Success | Standard successful response returning requested data or completion. |
| **`201 Created`** | Success | Resource successfully generated (e.g. fine-tune job spawned). |
| **`400 Bad Request`**| Client Error | Malformed JSON schema, missing mandatory prompt argument. |
| **`401 Unauthorized`**| Client Error | Invalid or expired API token (`Bearer sk-...`). |
| **`403 Forbidden`** | Client Error | Valid token, but account lacks required permissions/tier. |
| **`404 Not Found`** | Client Error | Requested model ID or endpoint path does not exist. |
| **`429 Rate Limited`**| Client Error | **Crucial in GenAI**: RPM (requests per min) or TPM (tokens per min) exceeded! |
| **`500 Internal Error`**| Server Error| Upstream LLM worker crash or unhandled runtime failure. |
| **`504 Gateway Timeout`**| Server Error| Model generation took longer than edge proxy timeout (e.g. >60s). |
"""
)

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**💻 Python Source Code (Status Code Validation)**")
    code_status = '''response = requests.post(API_URL, json=payload, timeout=20)

# Check status code explicitly:
if response.status_code == 200:
    data = response.json()
    st.success("Generation completed!")
elif response.status_code == 429:
    retry_after = response.headers.get("Retry-After", 10)
    st.warning(f"Rate limited! Backing off for {retry_after}s...")
elif response.status_code == 401:
    st.error("Invalid API credentials. Check secrets.toml!")
else:
    # Raise HTTPError for any unhandled 4xx/5xx code:
    response.raise_for_status()'''
    st.code(code_status, language="python")

with col_sc2:
    st.markdown("**🖥️ Live Interactive Status Code Simulator**")
    sim_status = st.selectbox(
        "Select Simulated Response Code:",
        [
            "200 OK (Success)",
            "400 Bad Request (Schema Error)",
            "401 Unauthorized (Invalid Key)",
            "429 Rate Limit Exceeded (TPM Capped)",
            "504 Gateway Timeout (Inference Hang)"
        ],
        key="sel_sim_status"
    )

    with st.container(border=True):
        if "200" in sim_status:
            st.success("🟢 **HTTP 200 OK**: Request resolved successfully with token response.")
        elif "400" in sim_status:
            st.warning("🟠 **HTTP 400 Bad Request**: Missing `'messages'` array in JSON body.")
        elif "401" in sim_status:
            st.error("🔴 **HTTP 401 Unauthorized**: Provided API key is inactive or revoked.")
        elif "429" in sim_status:
            st.warning("⚠️ **HTTP 429 Rate Limit**: Token quota exhausted (Retry-After: 12s).")
        elif "504" in sim_status:
            st.error("💥 **HTTP 504 Gateway Timeout**: Upstream model worker hung past 60s.")

st.divider()

# ==============================================================================
# 6. JSON Responses (Deserialization & Parsing)
# ==============================================================================
st.header("6. Parsing Nested JSON Responses", divider="rainbow")
st.markdown(
    r"""
Modern AI APIs return structured, deeply nested JSON documents. 
Using `.json()` automatically deserializes the response byte stream into a native Python dictionary.
"""
)

col_js1, col_js2 = st.columns(2)

with col_js1:
    st.markdown("**💻 Python Source Code (Extracting Generated Text)**")
    code_json = '''# Standard OpenAI / vLLM Response Payload Schema:
response = requests.post(url, json=body)
data = response.json()

# Extract generated completion:
assistant_message = data["choices"][0]["message"]["content"]

# Extract token telemetry:
prompt_tokens = data["usage"]["prompt_tokens"]
completion_tokens = data["usage"]["completion_tokens"]
total_tokens = data["usage"]["total_tokens"]

st.write(assistant_message)
st.metric("Total Tokens Incurred", total_tokens)'''
    st.code(code_json, language="python")

with col_js2:
    st.markdown("**🖥️ Live Interactive JSON Explorer**")
    mock_llm_json = {
        "id": "chatcmpl-9b872f01a",
        "object": "chat.completion",
        "created": 1791782176,
        "model": "claude-3-7-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "A REST API is an architectural style for providing standards between computer systems on the web."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 22,
            "total_tokens": 36
        }
    }

    with st.container(border=True):
        st.markdown("##### 📄 Raw JSON API Payload:")
        st.json(mock_llm_json)
        extracted_text = mock_llm_json["choices"][0]["message"]["content"]
        tokens_used = mock_llm_json["usage"]["total_tokens"]
        st.info(f"**Extracted Content**: \"{extracted_text}\" (Used: **{tokens_used}** tokens)")

st.divider()

# ==============================================================================
# 7. Query Parameters & 8. Request Headers
# ==============================================================================
st.header("7. Query Parameters & 8. Request Headers Architecture", divider="rainbow")
st.markdown(
    r"""
### Query Parameters vs. Request Headers

- **Query Parameters**: Key-value pairs appended after the `?` in the URL (`?limit=10&sort=desc`). Used for filtering, pagination, and sorting.
- **Request Headers**: Metadata transmitted alongside the HTTP request envelope. Used for **authentication tokens**, **content negotiation**, and **distributed tracing**.

| Essential Request Header | Standard Value Example | Purpose |
| :--- | :--- | :--- |
| **`Authorization`** | `Bearer sk-proj-49821...` | Authenticates request with an API key / JWT token |
| **`Content-Type`** | `application/json` | Informs server of request body format |
| **`Accept`** | `application/json` or `text/event-stream` | Informs server of desired response format (e.g. streaming chunks) |
| **`User-Agent`** | `StreamlitApp/1.65 (AI-Suite)` | Identifies the client software |
| **`X-Request-ID`** | `req_0192a83f-841b` | Distributed tracing ID for correlating server logs |
"""
)

col_hdr1, col_hdr2 = st.columns(2)

with col_hdr1:
    st.markdown("**💻 Python Source Code (Passing Headers & Query Params)**")
    code_params = '''import requests

endpoint = "https://api.enterprise-ai.com/v1/benchmarks"

# Query parameters for filtering:
params = {
    "provider": "Anthropic",
    "min_accuracy": 0.90,
    "page": 1,
    "limit": 10
}

# Request headers for authentication & tracing:
headers = {
    "Authorization": "Bearer sk-enterprise-secret",
    "Content-Type": "application/json",
    "X-Client-Version": "Streamlit-v1.65",
    "X-Request-ID": "trace-892401"
}

resp = requests.get(endpoint, params=params, headers=headers)'''
    st.code(code_params, language="python")

with col_hdr2:
    st.markdown("**🖥️ Live Interactive URL & Header Assembler**")
    base_ep = "https://api.ai-models.net/v1/leaderboard"
    p_provider = st.selectbox("Provider Parameter (`provider`)", ["All", "Anthropic", "OpenAI", "DeepSeek"], key="sel_qp_prov")
    p_limit = st.slider("Page Size (`limit`)", 5, 50, 10, key="sl_qp_limit")

    qp_dict = {"limit": p_limit}
    if p_provider != "All":
        qp_dict["provider"] = p_provider

    generated_url = f"{base_ep}?{urlencode(qp_dict)}"

    with st.container(border=True):
        st.markdown("##### 🔗 Compiled Request URL:")
        st.code(generated_url, language="bash")
        st.markdown("##### 🗂️ Header Map:")
        st.json({
            "Authorization": "Bearer sk-proj-••••••••••••38f1",
            "Content-Type": "application/json",
            "X-Streamlit-Client": "Active-Session"
        })

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 REST API Fundamentals Checklist", divider="rainbow")
st.markdown(
    r"""
| Component | Key Best Practice | Example |
| :--- | :--- | :--- |
| **HTTP Method** | Match verb to semantic intent | `GET` for reads, `POST` for prompt execution |
| **Authentication**| Pass tokens in `Authorization` header | `headers={"Authorization": f"Bearer {token}"}` |
| **Payload Format** | Send JSON with `json=dict` | `requests.post(url, json={"prompt": "..."})` |
| **Timeouts** | Always specify explicit timeouts | `timeout=(5.0, 30.0)` (5s connect, 30s read) |
| **Status Codes** | Guard against errors defensively | Handle `429` (rate limits) & `401` (invalid auth) |
| **Query Params** | Let requests encode parameters | Pass `params={"key": "val"}` instead of manual string formatting |
"""
)
