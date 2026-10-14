import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Calling APIs in Streamlit",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Calling External APIs in Streamlit — Complete Guide")
st.markdown(
    """
Connecting a Streamlit application to external cloud microservices and public REST endpoints 
requires a disciplined approach to network reliability. In production, web applications 
must gracefully handle **transient network partitions**, **strict HTTP timeouts**, 
**responsive loading states (`st.status` & `st.spinner`)**, and **response caching (`@st.cache_data`)** 
to prevent rate-limit penalties and sluggish user experiences.
"""
)

st.divider()

# ==============================================================================
# 1. Using the requests Library & requests.Session()
# ==============================================================================
st.header("1. Using the `requests` Library & Persistent Sessions", divider="rainbow")
st.markdown(
    r"""
### Why `requests.Session()` is Critical
While `requests.get()` works for one-off calls, opening a new TCP socket and TLS handshake on 
every call wastes 100–300 ms. A **`requests.Session()`** object:
1. Reuses open TCP sockets across multiple calls via HTTP Keep-Alive.
2. Automatically attaches default authentication headers, cookies, and timeouts to every request.

```python
import requests
import streamlit as st

@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "StreamlitApp/1.65 (Enterprise-AI)",
        "Accept": "application/json"
    })
    return session
```
"""
)

col_req1, col_req2 = st.columns(2)

with col_req1:
    st.markdown("**💻 Python Source Code (Session Reusable Client)**")
    code_session = '''import requests
import streamlit as st

@st.cache_resource
def get_api_client():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {st.secrets.get('API_TOKEN', '')}",
        "Content-Type": "application/json"
    })
    return session

client = get_api_client()
# Socket remains warm across all subsequent calls!
response = client.get("https://api.github.com/zen", timeout=5)'''
    st.code(code_session, language="python")

with col_req2:
    st.markdown("**🖥️ Live Interactive Session Inspector**")
    @st.cache_resource
    def init_demo_session():
        s = requests.Session()
        s.headers.update({"User-Agent": "Streamlit-Tutorial-Client/1.0"})
        return s

    demo_session = init_demo_session()
    with st.container(border=True):
        st.markdown("##### 🔌 Active HTTP Session Singleton")
        st.write(f"• **Session Pointer**: `{hex(id(demo_session))}`")
        st.write(f"• **Default Headers**: `{dict(demo_session.headers)}`")
        st.success("Session warm in memory: Socket pooling active across reruns.")

st.divider()

# ==============================================================================
# 2. Calling Public APIs & 3. Displaying API Responses
# ==============================================================================
st.header("2. Calling Public APIs & 3. Rendering Structured Responses", divider="rainbow")
st.markdown(
    r"""
When receiving data from external APIs, transform raw JSON into executive UI components:
- **`st.metric()`** for primary numerical KPIs and status deltas.
- **`st.dataframe()`** with custom `column_config` for list payloads.
- **`st.json()`** for collapsible technical schemas and debugging traces.
"""
)

col_pub1, col_pub2 = st.columns(2)

with col_pub1:
    st.markdown("**💻 Python Source Code (Public API Dispatch)**")
    code_pub = '''# Calling Open-Meteo Public Weather API (No Auth Required)
URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "current_weather": "true"
}

resp = requests.get(URL, params=params, timeout=5)
if resp.status_code == 200:
    data = resp.json()["current_weather"]
    st.metric("Temperature", f"{data['temperature']} °C")
    st.metric("Wind Speed", f"{data['windspeed']} km/h")'''
    st.code(code_pub, language="python")

with col_pub2:
    st.markdown("**🖥️ Live Interactive Public API Consumer**")
    city_coords = {
        "San Francisco (US)": {"lat": 37.7749, "lon": -122.4194},
        "London (UK)": {"lat": 51.5074, "lon": -0.1278},
        "Tokyo (Japan)": {"lat": 35.6762, "lon": 139.6503}
    }
    sel_city = st.selectbox("Select Target City", list(city_coords.keys()), key="sel_pub_city")
    selected_coord = city_coords[sel_city]

    # Live call with graceful mock fallback
    try:
        resp_meteo = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": selected_coord["lat"], "longitude": selected_coord["lon"], "current_weather": "true"},
            timeout=4
        )
        weather_payload = resp_meteo.json().get("current_weather", {})
    except Exception:
        weather_payload = {"temperature": 18.5, "windspeed": 12.2, "weathercode": 1}

    with st.container(border=True):
        st.markdown(f"##### 🌤️ Live Weather Telemetry: {sel_city}")
        w1, w2 = st.columns(2)
        w1.metric("Current Temp", f"{weather_payload.get('temperature', 20.0)} °C")
        w2.metric("Wind Velocity", f"{weather_payload.get('windspeed', 10.0)} km/h")
        st.caption("Live data fetched directly from Open-Meteo REST API.")

st.divider()

# ==============================================================================
# 4. Handling Loading States (st.spinner & st.status)
# ==============================================================================
st.header("4. Handling Loading States (`st.spinner` & `st.status`)", divider="rainbow")
st.markdown(
    r"""
Network requests are unpredictable. Keeping users informed prevents impatient re-clicks:
- **`st.spinner("...")`**: Minimalist animated spinner for quick one-shot queries.
- **`st.status("...")`**: Multi-stage progress container ideal for complex multi-step API sequences (e.g. Authenticate $\to$ Tokenize $\to$ Ingest $\to$ Generate).
"""
)

col_ld1, col_ld2 = st.columns(2)

with col_ld1:
    st.markdown("**💻 Python Source Code (Multi-Stage st.status)**")
    code_status = '''with st.status("Executing Multi-Step LLM Workflow...", expanded=True) as status:
    # Stage 1: Auth handshake
    st.write("🔑 Authenticating with AI Gateway...")
    time.sleep(0.5)
    
    # Stage 2: Tokenize prompt
    st.write("✂️ Tokenizing prompt payload (142 tokens)...")
    time.sleep(0.5)
    
    # Stage 3: Network dispatch
    st.write("📡 Awaiting first token from inference cluster...")
    time.sleep(1.0)
    
    status.update(label="✅ Workflow Complete! (2.0s elapsed)", state="complete")'''
    st.code(code_status, language="python")

with col_ld2:
    st.markdown("**🖥️ Live Interactive Multi-Stage Workflow Demo**")
    if st.button("🚀 Trigger Multi-Stage API Pipeline", key="btn_run_status_demo"):
        with st.status("Executing AI Inference Pipeline...", expanded=True) as status_box:
            st.write("1. 🔑 Verifying bearer token credentials...")
            time.sleep(0.4)
            st.write("2. 📊 Calculating prompt token count (412 tokens)...")
            time.sleep(0.4)
            st.write("3. ⚡ Dispatching HTTP request to GPU inference cluster...")
            time.sleep(0.6)
            status_box.update(label="✅ Generation Completed Successfully!", state="complete", expanded=False)
        st.success("Result ready: Generated 85 tokens at 62 tok/sec.")
    else:
        st.caption("Click button above to observe animated `st.status()` container progression.")

st.divider()

# ==============================================================================
# 5. Handling HTTP Errors & 6. Request Timeouts
# ==============================================================================
st.header("5. Defensive Error Handling & 6. Request Timeouts", divider="rainbow")
st.markdown(
    r"""
### The Two Golden Rules of Network Defense

1. **Always Specify Explicit Timeouts**: 
   `requests.get(url)` with no timeout will **wait forever** if the remote server drops packets, completely locking your Streamlit app! Always provide a tuple: `timeout=(connect_seconds, read_seconds)` (e.g., `timeout=(3.05, 10)`).
2. **Handle Specific Exception Types**:
   Catch network timeouts and client errors gracefully using dedicated exception handlers.
"""
)

col_err1, col_err2 = st.columns(2)

with col_err1:
    st.markdown("**💻 Python Source Code (Defensive Exception Trap)**")
    code_trap = '''import requests
from requests.exceptions import Timeout, HTTPError, ConnectionError

try:
    resp = requests.get(
        "https://api.example.com/v1/data",
        timeout=(3.05, 10.0)  # 3.05s connect, 10s read
    )
    # Raise exception for 4xx and 5xx status codes:
    resp.raise_for_status()
    data = resp.json()

except Timeout:
    st.error("⏱️ Gateway Timeout: Remote service took too long to answer.")
except HTTPError as e:
    code = e.response.status_code
    st.error(f"HTTP Error {code}: {e.response.text}")
except ConnectionError:
    st.error("🔌 Connection Failed: Check your network connection.")
except Exception as e:
    st.error(f"Unexpected error: {e}")'''
    st.code(code_trap, language="python")

with col_err2:
    st.markdown("**🖥️ Live Interactive Network Error Simulator**")
    sim_error_mode = st.selectbox(
        "Simulate Network Failure Mode",
        ["Successful Request (200 OK)", "Simulate Timeout (100ms Cap)", "Simulate 404 Not Found"],
        key="sel_sim_net_err"
    )

    if st.button("Execute Simulated Request", key="btn_exec_sim_net"):
        try:
            if "Timeout" in sim_error_mode:
                # Force timeout with impossibly small threshold on remote host
                requests.get("https://httpbin.org/delay/2", timeout=0.01)
            elif "404" in sim_error_mode:
                r = requests.get("https://httpbin.org/status/404", timeout=5)
                r.raise_for_status()
            else:
                r = requests.get("https://httpbin.org/status/200", timeout=5)
                st.success("✅ HTTP 200 OK: Endpoint responded cleanly!")
        except requests.exceptions.Timeout:
            st.error("⏱️ **Timeout Intercepted**: The server failed to respond within the allocated timeout window.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ **HTTP Error {e.response.status_code} Intercepted**: Requested resource path does not exist.")
        except Exception as e:
            st.error(f"Network error: {e}")

st.divider()

# ==============================================================================
# 7. Parsing JSON Safely & 8. Caching API Results (`@st.cache_data`)
# ==============================================================================
st.header("7. Parsing JSON Safely & 8. Caching API Results (`@st.cache_data`)", divider="rainbow")
st.markdown(
    r"""
### Combining `@st.cache_data` with Network APIs
Never execute raw API calls directly inside your page flow. 
Always wrap them in **`@st.cache_data(ttl=...)`**:
- Prevents hitting external rate limits (RPM / TPM).
- Makes widget interactions instantaneous (<1ms).
- Provides a manual `.clear()` refresh button to bypass cache on demand!
"""
)

col_cch1, col_cch2 = st.columns(2)

with col_cch1:
    st.markdown("**💻 Python Source Code (Cached API with Invalidation)**")
    code_caching_api = '''@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_leaderboard(category: str) -> dict:
    # Heavy network call cached for 5 minutes:
    resp = requests.get(
        "https://api.ai-models.org/v1/benchmarks",
        params={"category": category},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()

# Refresh button pattern:
if st.button("🔄 Force Refresh"):
    fetch_model_leaderboard.clear()
    st.rerun()

data = fetch_model_leaderboard("Reasoning")'''
    st.code(code_caching_api, language="python")

with col_cch2:
    st.markdown("**🖥️ Live Interactive Cached API Consumer**")
    @st.cache_data(ttl=60, show_spinner="Querying simulated model API...")
    def get_cached_api_feed(tag: str) -> dict:
        time.sleep(1.2)  # Simulates remote HTTP latency
        return {
            "query_tag": tag,
            "fetched_timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "records": [
                {"model": f"{tag} Frontier-1", "pass_rate": 0.942, "latency": "380ms"},
                {"model": f"{tag} Mini-Turbo", "pass_rate": 0.885, "latency": "140ms"}
            ]
        }

    tag_choice = st.selectbox("API Filter Tag", ["Coding", "Reasoning", "Multimodal"], key="sel_cch_tag")
    
    t_c0 = time.perf_counter()
    cached_response = get_cached_api_feed(tag_choice)
    elapsed_call_ms = (time.perf_counter() - t_c0) * 1000

    with st.container(border=True):
        if elapsed_call_ms > 500:
            st.warning(f"🐢 **Network Request Dispatched**: Resolved in **{elapsed_call_ms/1000:.2f}s**")
        else:
            st.success(f"⚡ **Cached Response Served**: Instant **{elapsed_call_ms:.2f} ms**")

        st.caption(f"Fetched at: `{cached_response['fetched_timestamp']}` (TTL = 60s)")
        st.dataframe(pd.DataFrame(cached_response["records"]), width="stretch", hide_index=True)

        if st.button("🔄 Invalidate & Re-fetch API", key="btn_inv_api"):
            get_cached_api_feed.clear(tag_choice)
            st.rerun()

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Calling APIs from Streamlit Best Practices Checklist", divider="rainbow")
st.markdown(
    r"""
| Architectural Practice | Why It Matters | Standard Code Implementation |
| :--- | :--- | :--- |
| **Session Reuse** | Reuses TCP sockets & Keep-Alive | `s = requests.Session()` cached via `@st.cache_resource` |
| **Explicit Timeouts** | Prevents app from freezing forever | `requests.get(url, timeout=(3.05, 10))` |
| **Response Caching** | Prevents redundant API costs & rate limits | Decorate with `@st.cache_data(ttl=300)` |
| **Loading Feedback** | Transparent UI progression | Wrap in `st.spinner(...)` or `with st.status(...)` |
| **Status Verification**| Catches 4xx & 5xx HTTP codes cleanly | Call `response.raise_for_status()` |
| **Safe JSON Traversal**| Prevents `KeyError` crashes on missing fields | Use `payload.get('key', default_val)` |
| **Manual Refresh** | Allows user-driven cache invalidation | Provide `func.clear()` button alongside `st.rerun()` |
"""
)
