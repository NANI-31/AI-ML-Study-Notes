import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import json
import asyncio

st.set_page_config(
    page_title="13.4 Advanced API Integration",
    page_icon="🔐",
    layout="wide"
)

st.title("13.4 Advanced API Integration in Streamlit")
st.caption("Enterprise-grade API patterns: authentication, secrets protection, bearer tokens, rate-limiting, cursor pagination, retries with backoff, async I/O, and legal compliance.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Enterprise API Integration Architecture

Moving from prototype API calls to enterprise-grade integration requires resilience across 5 critical dimensions:
1. **Security & Secrets Governance**: Securing credentials via `st.secrets` without leaking keys to git or client-side JavaScript.
2. **Authentication Lifecycle**: Managing Bearer tokens, token refresh handshakes, and rotation.
3. **Throttling & Traffic Shaping**: Honoring upstream `429 Too Many Requests`, parsing `Retry-After` headers, and token-bucket queuing.
4. **Resilience & Fault Tolerance**: Implementing exponential backoff with jitter and circuit-breaker patterns.
5. **High-Throughput Asynchronous Fetching**: Executing concurrent I/O using `httpx` / `asyncio` to eliminate sequential latency bottlenecks.

""")

st.graphviz_chart(
    r'''
    digraph ResilientAPI {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        client [label="Streamlit Client\nst.button('Fetch API Data')", fillcolor="#DBEAFE", color="#1D4ED8"];
        gateway [label="Resilience Gateway\n• Exponential Backoff & Jitter\n• Token Bucket Rate Limiter\n• Circuit Breaker (Closed / Half-Open / Open)", fillcolor="#FEF3C7", color="#D97706"];
        upstream [label="External Cloud API\nUpstream Microservice", fillcolor="#D1FAE5", color="#059669"];

        client -> gateway [label="Dispatches request"];
        gateway -> upstream [label="Protected HTTP call"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Auth & API Keys",
    "2. Bearer Tokens & JWT",
    "3. Protecting Secrets",
    "4. Rate Limits & Quotas",
    "5. Pagination Patterns",
    "6. Retry & Backoff",
    "7. Error Handling",
    "8. Async API Requests",
    "9. Compliance & Terms"
])

# =============================================================================
# 1. AUTH & API KEYS
# =============================================================================
with tabs[0]:
    st.header("1. API Authentication & API Key Schemes")
    st.markdown(r"""
    Public APIs enforce identity through various authentication schemes.
    The most common are **Header-based API keys**, **Query parameter keys**, and **Custom authorization headers**.
    Header-based authentication is strictly preferred over query strings to avoid leaking keys into proxy logs and browser history.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import requests

def query_authenticated_service(api_key: str, auth_type: str):
    headers = {"User-Agent": "StreamlitEnterpriseApp/1.0"}
    params = {}

    if auth_type == "Header Key (x-api-key)":
        headers["x-api-key"] = api_key
    elif auth_type == "Authorization Header":
        headers["Authorization"] = f"ApiKey {api_key}"
    elif auth_type == "Query Parameter (?api_key=...)":
        params["api_key"] = api_key  # Less secure

    # Simulated request verification
    is_valid = len(api_key) >= 12 and api_key.startswith("sk-")
    return {
        "status_code": 200 if is_valid else 401,
        "authenticated": is_valid,
        "auth_scheme": auth_type,
        "masked_key": f"{api_key[:4]}...{api_key[-3:]}" if len(api_key) > 7 else "****"
    }

key_input = st.text_input("API Key", "sk-live-94a8e23f001c", type="password")
scheme = st.selectbox("Injection Scheme", ["Header Key (x-api-key)", "Authorization Header", "Query Parameter (?api_key=...)"])

res = query_authenticated_service(key_input, scheme)
if res["authenticated"]:
    st.success(f"Authenticated successfully using {res['auth_scheme']} ({res['masked_key']})")
else:
    st.error("Authentication failed: Invalid key format (must start with 'sk-' and be >=12 chars).")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def query_authenticated_service(api_key: str, auth_type: str):
                is_valid = len(api_key) >= 12 and api_key.startswith("sk-")
                return {
                    "status_code": 200 if is_valid else 401,
                    "authenticated": is_valid,
                    "auth_scheme": auth_type,
                    "masked_key": f"{api_key[:4]}...{api_key[-3:]}" if len(api_key) > 7 else "****"
                }

            k_input = st.text_input("API Key", "sk-live-94a8e23f001c", type="password", key="auth_key_input")
            s_type = st.selectbox("Injection Scheme", ["Header Key (x-api-key)", "Authorization Header", "Query Parameter (?api_key=...)"], key="auth_scheme_select")

            auth_res = query_authenticated_service(k_input, s_type)
            if auth_res["authenticated"]:
                st.success(f"Authenticated successfully using {auth_res['auth_scheme']} ({auth_res['masked_key']})")
            else:
                st.error("Authentication failed: Invalid key format (must start with 'sk-' and be >=12 chars).")

# =============================================================================
# 2. BEARER TOKENS & JWT
# =============================================================================
with tabs[1]:
    st.header("2. Bearer Tokens & OAuth2 Authorization")
    st.markdown(r"""
    Modern LLM APIs (OpenAI, Anthropic, Hugging Face) and OAuth2 microservices use the standard HTTP header:
    $$\text{Authorization: Bearer } \langle\text{token}\rangle$$
    Bearer tokens are often short-lived JSON Web Tokens (JWT) containing cryptographically signed claims.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import datetime

def simulate_bearer_token_validation(bearer_token: str):
    # Simulated JWT structure parsing
    claims = {
        "iss": "https://auth.enterprise.ai",
        "sub": "user_dev_0918",
        "role": "model-engineer",
        "exp": datetime.datetime.now() + datetime.timedelta(minutes=45),
        "scopes": ["models:read", "inference:write", "fine-tuning:read"]
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    return headers, claims

token = st.text_input("Bearer Token (JWT)", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-41a...", type="password")
req_headers, token_claims = simulate_bearer_token_validation(token)

st.markdown("**Generated HTTP Request Headers:**")
st.code(f"Authorization: Bearer {token[:12]}...[REDACTED]\nContent-Type: application/json", language="http")

st.markdown("**Decoded JWT Claims & Scopes:**")
st.json({
    "Issuer": token_claims["iss"],
    "Subject": token_claims["sub"],
    "Role": token_claims["role"],
    "Expires": token_claims["exp"].strftime("%Y-%m-%d %H:%M:%S UTC"),
    "Granted Scopes": token_claims["scopes"]
})
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def simulate_bearer_token_validation(bearer_token: str):
                claims = {
                    "iss": "https://auth.enterprise.ai",
                    "sub": "user_dev_0918",
                    "role": "model-engineer",
                    "exp": datetime.datetime.now() + datetime.timedelta(minutes=45),
                    "scopes": ["models:read", "inference:write", "fine-tuning:read"]
                }
                return claims

            b_token = st.text_input("Bearer Token (JWT)", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-41a...", type="password", key="bearer_tok_input")
            t_claims = simulate_bearer_token_validation(b_token)

            st.caption("Generated HTTP Header:")
            st.code(f"Authorization: Bearer {b_token[:12]}...[REDACTED]\nContent-Type: application/json", language="http")

            st.caption("Decoded JWT Scopes & Claims:")
            st.json({
                "Issuer": t_claims["iss"],
                "Subject": t_claims["sub"],
                "Role": t_claims["role"],
                "Expires": t_claims["exp"].strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Scopes": t_claims["scopes"]
            })

# =============================================================================
# 3. PROTECTING SECRETS
# =============================================================================
with tabs[2]:
    st.header("3. Protecting Secrets in Streamlit (`st.secrets`)")
    st.markdown(r"""
    Streamlit provides native secrets management via `.streamlit/secrets.toml`.
    Secrets are read automatically into `st.secrets` without exposing keys to source control or end users.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import os

# Pattern: Fallback from st.secrets to OS environment variables
def get_secret(key_name: str, default: str = "") -> str:
    # 1. Try Streamlit native secrets
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name])
    except Exception:
        pass
    # 2. Fallback to OS environment
    return os.environ.get(key_name, default)

# Example TOML file layout (never commit to git):
toml_example = """# .streamlit/secrets.toml
[openai]
api_key = "sk-proj-7a91bf40c883e..."
organization_id = "org-enterprise-corp"

[anthropic]
api_key = "sk-ant-api03-bb81e..."
"""

st.markdown("**Configuration File Format:**")
st.code(toml_example, language="toml")

# Retrieve secret safely
simulated_key = "sk-proj-live-mock-902184"
masked_display = f"{simulated_key[:7]}...{simulated_key[-4:]}"
st.info(f"Active Secret Loaded: `{masked_display}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            toml_example = """# .streamlit/secrets.toml
[openai]
api_key = "sk-proj-7a91bf40c883e..."
organization_id = "org-enterprise-corp"

[anthropic]
api_key = "sk-ant-api03-bb81e..."
"""
            st.caption("Local Secret Storage (`.streamlit/secrets.toml`):")
            st.code(toml_example, language="toml")

            st.caption("Production Secrets Best Practices:")
            with st.container(border=True):
                st.markdown("✅ **Rule 1**: Add `.streamlit/secrets.toml` to `.gitignore` unconditionally.")
                st.markdown("✅ **Rule 2**: In production (Streamlit Cloud / AWS / Docker), inject via ENV variables or Cloud Secrets.")
                st.markdown("✅ **Rule 3**: Never print raw `st.secrets` in error tracebacks or `st.write()`.")

            simulated_key = "sk-proj-live-mock-902184"
            masked_disp = f"{simulated_key[:7]}...{simulated_key[-4:]}"
            st.success(f"Secret Loaded & Masked: `{masked_disp}`")

# =============================================================================
# 4. RATE LIMITS & QUOTAS
# =============================================================================
with tabs[3]:
    st.header("4. Rate Limits & Quota Management")
    st.markdown(r"""
    APIs communicate limits via HTTP headers (`x-ratelimit-remaining`, `x-ratelimit-reset`, `Retry-After`).
    When rate limits are exceeded, the API returns **HTTP 429 Too Many Requests**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import time

def simulate_rate_limiter_state(requests_fired: int, limit_per_min: int = 100):
    remaining = max(0, limit_per_min - requests_fired)
    pct_used = (requests_fired / limit_per_min) * 100
    is_throttled = remaining == 0
    retry_after = 24 if is_throttled else 0
    
    return {
        "limit": limit_per_min,
        "remaining": remaining,
        "percent_used": pct_used,
        "is_throttled": is_throttled,
        "retry_after_sec": retry_after
    }

rpm_slider = st.slider("Requests Made This Minute", 0, 120, 85)
quota = simulate_rate_limiter_state(rpm_slider, limit_per_min=100)

c1, c2, c3 = st.columns(3)
c1.metric("Quota Limit", f"{quota['limit']} req/min")
c2.metric("Remaining", f"{quota['remaining']} calls")
c3.metric("Utilization", f"{quota['percent_used']:.1f}%")

st.progress(min(quota['percent_used'] / 100, 1.0))

if quota["is_throttled"]:
    st.error(f"🚨 HTTP 429 Too Many Requests: Rate limit exceeded! Retry after {quota['retry_after_sec']}s.")
elif quota["percent_used"] > 80:
    st.warning("⚠️ Warning: Consumed >80% of minute quota. Consider debouncing.")
else:
    st.success("✅ Quota healthy: Within standard operating envelope.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def simulate_rate_limiter_state(requests_fired: int, limit_per_min: int = 100):
                remaining = max(0, limit_per_min - requests_fired)
                pct_used = (requests_fired / limit_per_min) * 100
                is_throttled = remaining == 0
                retry_after = 24 if is_throttled else 0
                return {
                    "limit": limit_per_min,
                    "remaining": remaining,
                    "percent_used": pct_used,
                    "is_throttled": is_throttled,
                    "retry_after_sec": retry_after
                }

            rpm_val = st.slider("Requests Made This Minute", 0, 120, 85, key="rl_slider")
            q_info = simulate_rate_limiter_state(rpm_val, limit_per_min=100)

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Quota Limit", f"{q_info['limit']} req/min")
            rc2.metric("Remaining", f"{q_info['remaining']} calls")
            rc3.metric("Utilization", f"{q_info['percent_used']:.1f}%")

            st.progress(min(q_info['percent_used'] / 100, 1.0))

            if q_info["is_throttled"]:
                st.error(f"🚨 HTTP 429 Too Many Requests: Rate limit exceeded! Retry after {q_info['retry_after_sec']}s.")
            elif q_info["percent_used"] > 80:
                st.warning("⚠️ Warning: Consumed >80% of minute quota.")
            else:
                st.success("✅ Quota healthy: Within normal limits.")

# =============================================================================
# 5. PAGINATION PATTERNS
# =============================================================================
with tabs[4]:
    st.header("5. API Pagination Strategies")
    st.markdown(r"""
    Large datasets are served via pagination. Three standard paradigms exist:
    - **Offset / Limit**: `?offset=20&limit=10` (Simple, but slow on large databases).
    - **Page Number**: `?page=3&per_page=10` (Common in CMS and REST endpoints).
    - **Cursor-based**: `?cursor=eyJpZCI6MTAxfQ==` (O(1) indexing, standard in Twitter/Stripe/Slack APIs).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Simulated paginated API records
ALL_RECORDS = [
    {"id": f"RUN-{i:03d}", "model": f"Mistral-7B-v0.{i%3}", "accuracy": round(0.85 + (i%15)*0.01, 3), "status": "Finished"}
    for i in range(1, 46)
]

PAGE_SIZE = 5
total_pages = (len(ALL_RECORDS) + PAGE_SIZE - 1) // PAGE_SIZE

col_prev, col_info, col_next = st.columns([1, 2, 1])

if "curr_page" not in st.session_state:
    st.session_state.curr_page = 1

with col_prev:
    if st.button("⬅️ Previous", disabled=(st.session_state.curr_page <= 1)):
        st.session_state.curr_page -= 1
        st.rerun()

with col_info:
    st.markdown(f"<div style='text-align:center;'>Page {st.session_state.curr_page} of {total_pages} ({len(ALL_RECORDS)} items)</div>", unsafe_allow_html=True)

with col_next:
    if st.button("Next ➡️", disabled=(st.session_state.curr_page >= total_pages)):
        st.session_state.curr_page += 1
        st.rerun()

start_idx = (st.session_state.curr_page - 1) * PAGE_SIZE
end_idx = start_idx + PAGE_SIZE
page_data = pd.DataFrame(ALL_RECORDS[start_idx:end_idx])

st.dataframe(page_data, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            ALL_RECORDS = [
                {"id": f"RUN-{i:03d}", "model": f"Mistral-7B-v0.{i%3}", "accuracy": round(0.85 + (i%15)*0.01, 3), "status": "Finished"}
                for i in range(1, 46)
            ]
            P_SIZE = 5
            t_pages = (len(ALL_RECORDS) + P_SIZE - 1) // P_SIZE

            if "pag_demo_page" not in st.session_state:
                st.session_state.pag_demo_page = 1

            cp1, cp2, cp3 = st.columns([1, 2, 1])
            with cp1:
                if st.button("⬅️ Prev", key="pag_prev_btn", disabled=(st.session_state.pag_demo_page <= 1)):
                    st.session_state.pag_demo_page -= 1
                    st.rerun()
            with cp2:
                st.markdown(f"<div style='text-align:center; padding-top:6px;'>Page {st.session_state.pag_demo_page} of {t_pages}</div>", unsafe_allow_html=True)
            with cp3:
                if st.button("Next ➡️", key="pag_next_btn", disabled=(st.session_state.pag_demo_page >= t_pages)):
                    st.session_state.pag_demo_page += 1
                    st.rerun()

            s_idx = (st.session_state.pag_demo_page - 1) * P_SIZE
            e_idx = s_idx + P_SIZE
            st.dataframe(pd.DataFrame(ALL_RECORDS[s_idx:e_idx]), width="stretch", hide_index=True)

# =============================================================================
# 6. RETRY & BACKOFF
# =============================================================================
with tabs[5]:
    st.header("6. Retry Strategies & Exponential Backoff")
    st.markdown(r"""
    Transient network errors (HTTP 502, 503, 504, 429) must not fail immediately.
    Instead, use **Exponential Backoff with Full Jitter**:
    $$T_{\text{sleep}} = 2^{\text{attempt}} \times \text{base\_delay} + \text{uniform}(0, \text{jitter})$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import time

def simulate_backoff_retry(max_retries: int = 4, base_delay: float = 0.5):
    attempt_log = []
    for attempt in range(1, max_retries + 1):
        # Calculate exponential delay with randomized jitter
        jitter = np.random.uniform(0.05, 0.2)
        sleep_sec = (2 ** (attempt - 1)) * base_delay + jitter
        
        # Simulate failure for first 2 attempts, success on 3rd
        status = 503 if attempt < 3 else 200
        outcome = "Failed (503 Service Unavailable)" if status != 200 else "Success (200 OK)"
        
        attempt_log.append({
            "Attempt": attempt,
            "Delay_Calculated": f"{sleep_sec:.2f}s",
            "HTTP_Code": status,
            "Outcome": outcome
        })
        if status == 200:
            break
            
    return attempt_log

retries = st.slider("Max Configured Retries", 2, 6, 4)
log = simulate_backoff_retry(max_retries=retries)

st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def simulate_backoff_retry(max_retries: int = 4, base_delay: float = 0.5):
                attempt_log = []
                for attempt in range(1, max_retries + 1):
                    jitter = np.random.uniform(0.05, 0.2)
                    sleep_sec = (2 ** (attempt - 1)) * base_delay + jitter
                    status = 503 if attempt < 3 else 200
                    outcome = "Failed (503 Service Unavailable)" if status != 200 else "Success (200 OK)"
                    attempt_log.append({
                        "Attempt": attempt,
                        "Delay Calculated": f"{sleep_sec:.2f}s",
                        "HTTP Status": status,
                        "Outcome": outcome
                    })
                    if status == 200:
                        break
                return attempt_log

            r_val = st.slider("Max Configured Retries", 2, 6, 4, key="backoff_retries")
            b_log = simulate_backoff_retry(max_retries=r_val)
            st.dataframe(pd.DataFrame(b_log), width="stretch", hide_index=True)
            st.caption("Exponential backoff prevents thundering herd problems on struggling servers.")

# =============================================================================
# 7. ERROR HANDLING
# =============================================================================
with tabs[6]:
    st.header("7. Resilient Error Handling & Circuit Breakers")
    st.markdown(r"""
    A robust Streamlit app must gracefully degrade when third-party APIs fail.
    The **Circuit Breaker** pattern tracks consecutive failures and temporarily disables calls to failing endpoints, 
    serving cached stale data or fallback offline modes.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED (Healthy), OPEN (Tripped), HALF-OPEN
        
    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

cb = CircuitBreaker(failure_threshold=3)

# Simulate failure tracking
failures = st.slider("Simulated Consecutive Failures", 0, 5, 2)
for _ in range(failures):
    cb.record_failure()

if cb.state == "OPEN":
    st.error("🚨 Circuit Breaker OPEN: Upstream endpoint failing repeatedly! Requests blocked; serving local cache.")
else:
    st.success(f"✅ Circuit Breaker CLOSED (Healthy). Failure count: {cb.failure_count} / {cb.failure_threshold}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            class CircuitBreaker:
                def __init__(self, failure_threshold: int = 3):
                    self.failure_threshold = failure_threshold
                    self.failure_count = 0
                    self.state = "CLOSED"
                def record_failure(self):
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"

            cb_demo = CircuitBreaker(failure_threshold=3)
            f_count = st.slider("Simulated Consecutive Failures", 0, 5, 2, key="cb_failures")
            for _ in range(f_count):
                cb_demo.record_failure()

            if cb_demo.state == "OPEN":
                st.error("🚨 Circuit Breaker OPEN: Upstream endpoint failing repeatedly! Requests blocked; serving local fallback cache.")
            else:
                st.success(f"✅ Circuit Breaker CLOSED (Healthy). Failures: {cb_demo.failure_count} / {cb_demo.failure_threshold}")

# =============================================================================
# 8. ASYNC API REQUESTS
# =============================================================================
with tabs[7]:
    st.header("8. Asynchronous API Requests (`asyncio` & `httpx`)")
    st.markdown(r"""
    Sequential HTTP requests add cumulative network latency ($3 \times 300\text{ms} = 900\text{ms}$).
    With asynchronous execution (`asyncio.gather`), requests fire concurrently in parallel, reducing total runtime to the single slowest request ($300\text{ms}$).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import asyncio
import time

async def fetch_mock_endpoint(endpoint_name: str, delay_sec: float):
    await asyncio.sleep(delay_sec)
    return {
        "Endpoint": endpoint_name,
        "Latency_Simulated": f"{delay_sec*1000:.0f} ms",
        "Status": "200 OK"
    }

async def run_concurrent_fetches():
    tasks = [
        fetch_mock_endpoint("Model Catalog API", 0.35),
        fetch_mock_endpoint("Vector Index Status", 0.45),
        fetch_mock_endpoint("User Permissions API", 0.20)
    ]
    return await asyncio.gather(*tasks)

if st.button("Run Concurrent Asynchronous Dispatch"):
    t0 = time.perf_counter()
    results = asyncio.run(run_concurrent_fetches())
    total_time = (time.perf_counter() - t0) * 1000
    
    st.metric("Total Execution Time", f"{total_time:.1f} ms", delta="Concurrent Speedup (~60%)")
    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            async def fetch_mock_endpoint(endpoint_name: str, delay_sec: float):
                await asyncio.sleep(delay_sec)
                return {
                    "Endpoint": endpoint_name,
                    "Latency Simulated": f"{delay_sec*1000:.0f} ms",
                    "Status": "200 OK"
                }

            async def run_concurrent_fetches():
                tasks = [
                    fetch_mock_endpoint("Model Catalog API", 0.05),
                    fetch_mock_endpoint("Vector Index Status", 0.08),
                    fetch_mock_endpoint("User Permissions API", 0.03)
                ]
                return await asyncio.gather(*tasks)

            if st.button("Run Concurrent Asynchronous Dispatch", key="async_demo_btn"):
                t0 = time.perf_counter()
                results = asyncio.run(run_concurrent_fetches())
                tot_time = (time.perf_counter() - t0) * 1000
                st.metric("Total Wall-Clock Time", f"{tot_time:.1f} ms", delta="~60% concurrency savings")
                st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
            else:
                st.info("Click button above to simulate concurrent `asyncio.gather` execution.")

# =============================================================================
# 9. COMPLIANCE & TERMS
# =============================================================================
with tabs[8]:
    st.header("9. Respecting Third-Party API Terms of Service")
    st.markdown(r"""
    Enterprise developers must comply with API provider terms, privacy regulations, and ethical guidelines.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Essential HTTP Headers for Legal & Attribution Compliance
headers = {
    "User-Agent": "EnterpriseMLApp/2.1 (contact@enterprise.ai; +https://enterprise.ai/legal)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate"
}

# Principles of Ethical API Usage:
# 1. Provide an identifiable, non-generic User-Agent
# 2. Never scrape HTML when a public API is provided
# 3. Adhere to caching headers (Cache-Control, max-age)
# 4. Respect robots.txt and rate limits
# 5. Do not store or redistribute copyrighted third-party training data
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Production Compliance Checklist

            | Area | Requirement | Implementation in Streamlit |
            | :--- | :--- | :--- |
            | **User-Agent Attribution** | Identify application name & contact info. | Custom header in `requests.Session`. |
            | **Rate Limit Politeness** | Do not exceed queries-per-second constraints. | Client-side throttling / caching. |
            | **Data Retention** | Purge transient personal or copyright data. | Avoid saving raw PII to disk; use session-only state. |
            | **IP Masking & Proxies** | Avoid circumventing rate-limiting bans. | Coordinate quota increases with vendor support. |
            """)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Advanced API Integration Cheatsheet

| Task | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **API Keys** | Hardcoding `api_key = "sk-..."` in code | Store in `.streamlit/secrets.toml` or OS env vars; access via `st.secrets`. |
| **Bearer Tokens** | Passing tokens in URL query strings | Send in `Authorization: Bearer <token>` HTTP header. |
| **Rate Limits** | Crashing with unhandled HTTP 429 | Parse `Retry-After` header, back off exponentially, and cache responses. |
| **Pagination** | Fetching 10,000 items in a single giant query | Implement Cursor / Offset pagination with Streamlit next/prev controls. |
| **Multiple Calls** | Running 5 API requests sequentially | Use `httpx.AsyncClient` + `asyncio.gather` for parallel non-blocking I/O. |
""")
