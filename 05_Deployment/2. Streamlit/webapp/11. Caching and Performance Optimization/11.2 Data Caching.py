import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from io import StringIO

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Data Caching with @st.cache_data",
    page_icon="💾",
    layout="wide",
)

st.title("💾 Data Caching with `@st.cache_data` — Complete Guide")
st.markdown(
    """
The `@st.cache_data` decorator is Streamlit's primary tool for caching **serializable data computations**.
Whether querying remote databases, calling external REST / LLM APIs, transforming tabular DataFrames, 
or parsing large CSV/Parquet files, `@st.cache_data` memoizes the returned result and guarantees 
**mutation safety** by returning a clean copy on every call.
"""
)

st.divider()

# ==============================================================================
# 1. Understanding @st.cache_data & Parameter Signatures
# ==============================================================================
st.header("1. Anatomy of `@st.cache_data`", divider="rainbow")
st.markdown(
    r"""
```python
@st.cache_data(
    ttl: float | timedelta | str | None = None,
    max_entries: int | None = None,
    show_spinner: bool | str = True,
    persist: "disk" | None = None,
    experimental_allow_widgets: bool = False,
    hash_funcs: dict[Type, Callable] | None = None
)
def compute_data(*args, **kwargs):
    ...
```

- **`ttl`**: Time-To-Live expiration (e.g. `60`, `timedelta(minutes=15)`, `"1d"`).
- **`max_entries`**: Upper bound of distinct function calls stored (evicts oldest via LRU).
- **`show_spinner`**: Controls whether to show a spinner during cache misses (pass a custom string like `"Querying Databricks..."`).
- **`persist="disk"`**: Persists cached data across server restarts by writing to disk (`.streamlit/cache`).
"""
)

# ==============================================================================
# 2. Caching Pandas DataFrames
# ==============================================================================
st.header("2. Caching Pandas DataFrames (Synthetic Benchmarks)", divider="rainbow")
st.markdown(
    """
DataFrames are the most common artifact passed through `@st.cache_data`. 
Because Streamlit pickles the DataFrame and returns a copy, in-memory filtering 
and column mutations in your page never corrupt the cached source of truth.
"""
)

col_df1, col_df2 = st.columns(2)

with col_df1:
    st.markdown("**💻 Python Source Code (Cached DataFrame Ingestion)**")
    code_df = '''@st.cache_data(show_spinner="Generating 50,000 synthetic rows...")
def load_large_benchmark(n_rows: int = 50000):
    time.sleep(1.2)  # Simulating heavy aggregation / DB read
    return pd.DataFrame({
        "Model": np.random.choice(["Claude 3.7", "GPT-4.5", "DeepSeek R1"], size=n_rows),
        "Prompt_Tokens": np.random.randint(100, 4000, size=n_rows),
        "Latency_ms": np.random.normal(320, 45, size=n_rows)
    })

# First run: 1.2s delay | Subsequent reruns: <1ms!
df = load_large_benchmark(50000)
st.dataframe(df.head())'''
    st.code(code_df, language="python")

with col_df2:
    st.markdown("**🖥️ Live Interactive Benchmark Execution**")
    @st.cache_data(show_spinner="Synthesizing benchmark telemetry...")
    def get_large_benchmark(n_rows: int) -> pd.DataFrame:
        time.sleep(1.0)
        np.random.seed(42)
        return pd.DataFrame({
            "Model": np.random.choice(["Claude 3.7", "GPT-4.5", "DeepSeek R1"], size=n_rows),
            "Prompt_Tokens": np.random.randint(100, 4000, size=n_rows),
            "Latency_ms": np.random.normal(320, 45, size=n_rows).round(1)
        })

    row_choice = st.selectbox("Select Sample Size", [10000, 25000, 50000], key="sel_df_rows")
    t0 = time.perf_counter()
    bench_data = get_large_benchmark(row_choice)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    with st.container(border=True):
        if elapsed_ms > 500:
            st.warning(f"🐢 **Cache MISS**: Generated **{len(bench_data):,} rows** in **{elapsed_ms/1000:.2f}s**")
        else:
            st.success(f"⚡ **Cache HIT**: Retrieved **{len(bench_data):,} rows** in **{elapsed_ms:.2f} ms**!")
        st.dataframe(bench_data.head(3), width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 3. Caching API Responses
# ==============================================================================
st.header("3. Caching External REST & LLM API Responses", divider="rainbow")
st.markdown(
    """
External API endpoints frequently enforce strict rate limits or charge per request. 
Caching API responses prevents redundant network round-trips when parameters haven't changed.
"""
)

col_api1, col_api2 = st.columns(2)

with col_api1:
    st.markdown("**💻 Python Source Code (Cached API Client)**")
    code_api = '''@st.cache_data(ttl=300, show_spinner="Querying Model Endpoint...")
def query_model_leaderboard(category: str):
    time.sleep(1.5)  # Simulate external HTTP GET call
    return {
        "category": category,
        "updated_at": datetime.now().isoformat(),
        "top_model": "DeepSeek R1" if category == "Reasoning" else "Claude 3.7",
        "score": 93.6 if category == "Reasoning" else 92.0
    }

resp = query_model_leaderboard("Coding")
st.json(resp)'''
    st.code(code_api, language="python")

with col_api2:
    st.markdown("**🖥️ Live Interactive API Cache Demo**")
    @st.cache_data(ttl=60, show_spinner="Calling simulated REST API...")
    def fetch_api_leaderboard(category: str) -> dict:
        time.sleep(1.0)
        return {
            "query_category": category,
            "cached_timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "leaderboard_first_place": "DeepSeek R1" if category == "Reasoning" else "Claude 3.7 Sonnet",
            "eval_score": 93.6 if category == "Reasoning" else 92.4
        }

    cat_sel = st.selectbox("API Query Parameter (Category)", ["Coding", "Reasoning", "Mathematics"], key="sel_api_cat")
    t_api0 = time.perf_counter()
    api_payload = fetch_api_leaderboard(cat_sel)
    t_api_ms = (time.perf_counter() - t_api0) * 1000

    with st.container(border=True):
        if t_api_ms > 500:
            st.warning(f"🐢 **API Request Dispatched**: Resolved in **{t_api_ms/1000:.2f}s**")
        else:
            st.success(f"⚡ **Cached Response Served**: Instant **{t_api_ms:.2f} ms**")
        st.json(api_payload)

st.divider()

# ==============================================================================
# 4. Caching File Processing (CSVs, JSON, Parquet)
# ==============================================================================
st.header("4. Caching File Processing & Parsing", divider="rainbow")
st.markdown(
    r"""
Parsing raw file byte streams or string buffers into structured DataFrames is computationally intensive.
Pass the raw file content or upload buffer into `@st.cache_data` to ensure you parse the bytes **only once**!

> **Critical Gotcha with `UploadedFile`**: Because `UploadedFile` instances contain internal state 
> (like stream pointers), **pass `file.getvalue()`** (the raw bytes or string) instead of the file object itself!
"""
)

col_fp1, col_fp2 = st.columns(2)

with col_fp1:
    st.markdown("**💻 Python Source Code (Passing Raw Bytes)**")
    code_file = '''# ✅ BEST PRACTICE: Pass raw bytes to ensure clean hashing!
@st.cache_data
def parse_uploaded_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    time.sleep(1.0)  # Simulates parsing a 50MB CSV
    return pd.read_csv(io.BytesIO(file_bytes))

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    # Hash is computed on the raw bytes payload!
    df = parse_uploaded_csv_bytes(uploaded.getvalue())
    st.dataframe(df)'''
    st.code(code_file, language="python")

with col_fp2:
    st.markdown("**🖥️ Live Interactive Raw-String Parser Simulation**")
    @st.cache_data
    def parse_raw_telemetry_csv(csv_text: str) -> pd.DataFrame:
        time.sleep(0.8)
        return pd.read_csv(StringIO(csv_text))

    raw_sample_csv = "Model,Score,Latency\nClaude-3.7,92.4,380\nGPT-4.5,90.2,410\nDeepSeek-R1,93.6,920"
    
    t_parse0 = time.perf_counter()
    parsed_df = parse_raw_telemetry_csv(raw_sample_csv)
    t_parse_ms = (time.perf_counter() - t_parse0) * 1000

    with st.container(border=True):
        st.write(f"• Parse Time: **{t_parse_ms:.2f} ms**")
        st.dataframe(parsed_df, width="stretch", hide_index=True)
        st.caption("Subsequent reruns bypass StringIO parsing entirely.")

st.divider()

# ==============================================================================
# 5. Time-To-Live (TTL) Settings & 6. Maximum Cache Entries
# ==============================================================================
st.header("5. Time-to-Live (TTL) & 6. Maximum Cache Entries (`max_entries`)", divider="rainbow")
st.markdown(
    r"""
### Managing Cache Freshness and Memory Caps

- **`ttl`**: Discards stale entries after a duration:
  - `ttl=60` (Seconds)
  - `ttl="15m"` or `ttl="2h"` (String notation)
  - `ttl=timedelta(days=1)` (Datetime object)
- **`max_entries`**: Sets an upper limit on the number of cached items in the dictionary. When exceeded, Streamlit evicts the **Least Recently Used (LRU)** item.
"""
)

col_lru1, col_lru2 = st.columns(2)

with col_lru1:
    st.markdown("**💻 Python Source Code (TTL & LRU Eviction)**")
    code_lru = '''@st.cache_data(
    ttl="10m",          # Stays fresh for 10 minutes
    max_entries=3,      # Holds at most 3 distinct query results!
    show_spinner=False
)
def fetch_token_metrics(model_id: str):
    time.sleep(1.0)
    return {"model": model_id, "fetched_at": time.time()}

# If called with Model A, B, C, D:
# Model A will be evicted from memory when D is added!'''
    st.code(code_lru, language="python")

with col_lru2:
    st.markdown("**🖥️ Live Interactive LRU & TTL Demonstration**")
    @st.cache_data(ttl=20, max_entries=2)
    def fetch_capped_data(model_name: str) -> dict:
        time.sleep(0.8)
        return {
            "model": model_name,
            "generated_timestamp": datetime.now().strftime("%H:%M:%S")
        }

    selected_lru_model = st.selectbox(
        "Select Model (Max Entries = 2, TTL = 20s)",
        ["Model Alpha", "Model Beta", "Model Gamma"],
        key="sel_lru_demo"
    )

    t_lru0 = time.perf_counter()
    lru_payload = fetch_capped_data(selected_lru_model)
    t_lru_ms = (time.perf_counter() - t_lru0) * 1000

    with st.container(border=True):
        if t_lru_ms > 400:
            st.warning(f"🐢 **Cache MISS**: Loaded `{selected_lru_model}` in **{t_lru_ms/1000:.2f}s**")
        else:
            st.success(f"⚡ **Cache HIT**: Retrieved from LRU memory in **{t_lru_ms:.2f} ms**")
        st.write(f"Payload: `{lru_payload}`")
        st.caption("Switch between all 3 models: because `max_entries=2`, the least recently used model is automatically evicted!")

st.divider()

# ==============================================================================
# 7. Hashing Behavior & 8. Clearing Cached Data
# ==============================================================================
st.header("7. Advanced Hashing (`hash_funcs`) & 8. Clearing Cached Data", divider="rainbow")
st.markdown(
    r"""
### Advanced Hashing with `hash_funcs`
When passing unhashable or custom classes (e.g. custom database client instances or open network sockets), 
Streamlit raises an `UnhashableParamError`. Tell Streamlit how to extract a unique signature using `hash_funcs`:

```python
class CustomClient:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

@st.cache_data(hash_funcs={CustomClient: lambda client: client.tenant_id})
def get_tenant_quota(client: CustomClient):
    ...
```

### Clearing Cached Data
1. **Clear specific function**: `my_func.clear()`
2. **Clear specific parameter invocation**: `my_func.clear("Coding")`
3. **Clear all data caches**: `st.cache_data.clear()`
"""
)

col_clr1, col_clr2 = st.columns(2)

with col_clr1:
    st.markdown("**💻 Python Source Code (Granular Clearing)**")
    code_clear = '''# 1. Clear function completely:
if st.button("Wipe Leaderboard"):
    fetch_api_leaderboard.clear()

# 2. Clear ONLY the 'Coding' category cache:
if st.button("Wipe Coding Only"):
    fetch_api_leaderboard.clear("Coding")

# 3. Global wipe:
if st.button("Flush Everything"):
    st.cache_data.clear()'''
    st.code(code_clear, language="python")

with col_clr1:
    st.markdown("**Granular Invalidation vs. Global Flush:**")
    st.markdown(
        r"""
- Using `func.clear(*args)` only invalidates entries matching those arguments.
- Other cached parameters remain active and hot in memory!
"""
    )

with col_clr2:
    st.markdown("**🖥️ Live Interactive Cache Management Console**")
    with st.container(border=True):
        st.markdown("##### 🎛️ Active Function Cache Controls")
        
        c_btn1, c_btn2, c_btn3 = st.columns(3)
        with c_btn1:
            if st.button("🔄 Clear Active API", key="btn_clr_active"):
                fetch_api_leaderboard.clear(cat_sel)
                st.success(f"Cleared `{cat_sel}`!")
                st.rerun()
        with c_btn2:
            if st.button("🗑️ Clear All Categories", key="btn_clr_func"):
                fetch_api_leaderboard.clear()
                st.success("Cleared function cache!")
                st.rerun()
        with c_btn3:
            if st.button("🧹 Flush All Caches", key="btn_clr_global"):
                st.cache_data.clear()
                st.success("Wiped all `@st.cache_data`!")
                st.rerun()

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 `@st.cache_data` Parameter Reference", divider="rainbow")
st.markdown(
    r"""
| Parameter | Default | Supported Types | Best Practice |
| :--- | :--- | :--- | :--- |
| **`ttl`** | `None` (Infinite) | `float`, `int`, `str`, `timedelta` | Use `"10m"` for fast-changing data, `"1d"` for daily tables |
| **`max_entries`** | `None` (Infinite) | `int` | Cap to `100–500` to prevent RAM exhaustion on large DataFrames |
| **`show_spinner`** | `True` | `bool` or `str` | Provide descriptive feedback: `"Fetching analytics from BigQuery..."` |
| **`persist`** | `None` | `"disk"` | Use for heavy datasets that should survive server restarts |
| **`hash_funcs`** | `None` | `dict[Type, Callable]` | Override hashing for unhashable custom objects or connections |
| **`func.clear()`** | N/A | Method | Invalidate individual functions without dropping the entire app cache |
"""
)
