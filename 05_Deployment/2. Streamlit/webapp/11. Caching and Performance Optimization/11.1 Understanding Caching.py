import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Understanding Caching in Streamlit",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Understanding Caching & Performance — Complete Guide")
st.markdown(
    """
Streamlit's execution model is uniquely reactive: **every time a user interacts with a widget, 
the entire script executes top-to-bottom**. Without caching, expensive operations — such as 
querying remote databases, calculating embeddings, running Monte Carlo simulations, or loading 
billion-parameter model weights — would execute repeatedly on every single click, resulting 
in unusable latency and astronomical server costs.
"""
)

st.divider()

# ==============================================================================
# 1. What is Caching? & 2. Why Caching Matters in Streamlit
# ==============================================================================
st.header("1. What is Caching? & 2. Why Caching Matters in Streamlit", divider="rainbow")
st.markdown(
    r"""
### The Core Concept
**Caching** is an optimization technique that stores the results of expensive function calls 
in high-speed memory. When the function is called again with the **exact same inputs**, 
Streamlit skips executing the function body entirely and instantly returns the precomputed result.

| Execution Mode | First Run / New Inputs | Subsequent Reruns (Cached) |
| :--- | :--- | :--- |
| **Uncached Execution** | ⏱️ 3.5 seconds (Database / ML inference) | ⏱️ 3.5 seconds (Re-executes completely) |
| **Cached Execution** | ⏱️ 3.5 seconds (Executes once & memoizes) | ⚡ **< 1 millisecond** (Instant memory lookup) |
"""
)

col_wh1, col_wh2 = st.columns(2)

with col_wh1:
    st.markdown("**💻 Python Source Code (Cached vs. Uncached)**")
    code_comp = '''import time
import streamlit as st

# ❌ Uncached: Re-runs on every slider adjustment!
def slow_load_data(n_samples):
    time.sleep(2.0)  # Simulating slow network query
    return [i * 2 for i in range(n_samples)]

# ✅ Cached: Runs once; instantaneous thereafter!
@st.cache_data
def fast_load_data(n_samples):
    time.sleep(2.0)
    return [i * 2 for i in range(n_samples)]'''
    st.code(code_comp, language="python")

with col_wh2:
    st.markdown("**🖥️ Live Interactive Latency Benchmark**")
    @st.cache_data
    def benchmark_cached_operation(n: int) -> float:
        time.sleep(1.5)  # Simulate expensive work
        return float(np.sum(np.arange(n, dtype=np.int64)))

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🚀 Run Cached Operation (N=500k)", key="btn_run_cache_bench"):
            t0 = time.perf_counter()
            res = benchmark_cached_operation(500000)
            elapsed = time.perf_counter() - t0
            if elapsed > 1.0:
                st.warning(f"🐢 **Cache MISS (First Call)**: Computed in **{elapsed:.2f}s**")
            else:
                st.success(f"⚡ **Cache HIT (Instant)**: Retrieved in **{elapsed*1000:.2f} ms**!")
            st.metric("Computed Sum", f"{res:,.0f}")
    with b_col2:
        st.caption("Click the button twice! The first click takes ~1.5 seconds to populate the cache. The second click retrieves the result in under 1 millisecond!")

st.divider()

# ==============================================================================
# 3. Script Reruns & Expensive Operations
# ==============================================================================
st.header("3. Script Reruns & The Cost of Expensive Operations", divider="rainbow")
st.markdown(
    """
Every widget change triggers a script rerun. Without caching, a simple slider adjustment 
forces your server to re-download datasets, re-train models, or re-parse JSON files.
"""
)

col_rr1, col_rr2 = st.columns(2)

with col_rr1:
    st.markdown("**💻 Python Source Code (Filtering Cached Datasets)**")
    code_filter = '''@st.cache_data
def fetch_telemetry_dataset():
    """Heavy operation: Only runs once across all users!"""
    time.sleep(1.5)  # Simulates remote API pull
    return pd.DataFrame({
        "Model": ["Claude 3.7", "GPT-4.5", "DeepSeek R1", "Llama 3.3"] * 250,
        "Tokens": np.random.randint(50, 4000, 1000)
    })

# Instant on rerun:
df = fetch_telemetry_dataset()

# Cheap operation: Runs instantly during rerun
min_tokens = st.slider("Filter by Min Tokens", 100, 3000, 500)
filtered = df[df["Tokens"] >= min_tokens]'''
    st.code(code_filter, language="python")

with col_rr2:
    st.markdown("**🖥️ Rendered Interactive Demonstration**")
    @st.cache_data
    def get_cached_telemetry():
        time.sleep(1.0)
        np.random.seed(42)
        return pd.DataFrame({
            "Model": ["Claude 3.7", "GPT-4.5", "DeepSeek R1", "Llama 3.3"] * 100,
            "Tokens": np.random.randint(50, 4000, 400)
        })

    t_start = time.perf_counter()
    df_cached = get_cached_telemetry()
    t_load = (time.perf_counter() - t_start) * 1000

    filter_cutoff = st.slider("Adjust Filter Threshold (Triggers Rerun)", 100, 3000, 1000, 100, key="slider_rerun_demo")
    df_sub = df_cached[df_cached["Tokens"] >= filter_cutoff]

    with st.container(border=True):
        st.write(f"• Dataset fetch time: **{t_load:.1f} ms** *(Cached)*")
        st.write(f"• Active records: **{len(df_sub)}** (Filtered from {len(df_cached)})")
        st.caption("Notice how adjusting the slider updates the count instantly without re-fetching data!")

st.divider()

# ==============================================================================
# 4. Cache Keys: How Streamlit Computes Hashes
# ==============================================================================
st.header("4. Cache Keys: How Streamlit Computes Function Hashes", divider="rainbow")
st.markdown(
    r"""
Streamlit calculates a cryptographic hash key using **four components**:

```text
Cache Key Hash = HASH(
    1. The byte code of the function itself,
    2. The function name and module path,
    3. The exact input argument values,
    4. Any global variables referenced inside the function body
)
```

- If you **change the code** inside the function body $\to$ Hash changes $\to$ **Cache MISS**.
- If you **pass different arguments** $\to$ Hash changes $\to$ **Cache MISS**.
- If inputs and code are identical $\to$ Hash matches $\to$ **Cache HIT**!
"""
)

col_hk1, col_hk2 = st.columns(2)

with col_hk1:
    st.markdown("**💻 Code Example: How Parameters Affect the Key**")
    code_hash = '''@st.cache_data
def compute_embeddings(text: str, model_id: str):
    # Cache key depends on BOTH 'text' AND 'model_id'
    time.sleep(1.0)
    return len(text) * 10

# Call 1: compute_embeddings("Hello", "text-embedding-3") -> Key A (MISS)
# Call 2: compute_embeddings("Hello", "text-embedding-3") -> Key A (HIT)
# Call 3: compute_embeddings("Hello", "bge-large-en")     -> Key B (MISS)'''
    st.code(code_hash, language="python")

with col_hk2:
    st.markdown("**🖥️ Live Interactive Cache Key Simulator**")
    @st.cache_data
    def simulate_key_lookup(param_str: str, param_num: int) -> str:
        time.sleep(1.0)
        return f"Generated hash for: '{param_str}' + {param_num} at {datetime.now().strftime('%H:%M:%S')}"

    inp_str = st.selectbox("Parameter 1 (Model)", ["qwen-2.5", "claude-3.7"], key="cache_k_str")
    inp_num = st.selectbox("Parameter 2 (Dimensions)", [512, 1024], key="cache_k_num")

    t_k0 = time.perf_counter()
    k_res = simulate_key_lookup(inp_str, inp_num)
    k_elapsed = time.perf_counter() - t_k0

    with st.container(border=True):
        if k_elapsed > 0.8:
            st.warning(f"🔑 **Cache MISS** (Took {k_elapsed:.2f}s) — New Parameter Combination!")
        else:
            st.success(f"⚡ **Cache HIT** (Took {k_elapsed*1000:.2f}ms) — Existing Key Reused!")
        st.write(f"Result: `{k_res}`")

st.divider()

# ==============================================================================
# 5. Cache Invalidation & 6. Cache Expiration (TTL & Max Entries)
# ==============================================================================
st.header("5. Cache Invalidation & 6. Cache Expiration (`ttl` & `max_entries`)", divider="rainbow")
st.markdown(
    r"""
Cached results cannot live forever. Streamlit provides explicit expiration and limit controls:
- **`ttl=int | float | timedelta`** (Time-To-Live): Automatically invalidates cache after a set duration.
- **`max_entries=int`**: Caps memory usage by evicting Least Recently Used (LRU) entries when full.

```python
@st.cache_data(ttl="10m", max_entries=500)
def fetch_live_stock_prices(ticker: str):
    # Evicted automatically after 10 minutes OR when 500 entries are exceeded!
    return api.get_quote(ticker)
```
"""
)

col_ttl1, col_ttl2 = st.columns(2)

with col_ttl1:
    st.markdown("**💻 Python Source Code (Programmatic Invalidation)**")
    code_inval = '''@st.cache_data(ttl=60)  # Expires after 60 seconds
def load_user_profile(user_id):
    return query_db(user_id)

# 1. Clear cache for ONE specific function:
if st.button("Refresh Profile"):
    load_user_profile.clear()  # Invalidates this function only!
    st.rerun()

# 2. Clear ALL data caches across the app:
if st.button("Flush Everything"):
    st.cache_data.clear()      # Wipes all @st.cache_data caches!
    st.rerun()'''
    st.code(code_inval, language="python")

with col_ttl2:
    st.markdown("**🖥️ Live Interactive Invalidation Console**")
    @st.cache_data(ttl=15)
    def get_timestamped_data() -> str:
        return f"Snapshot generated at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"

    ts_data = get_timestamped_data()
    st.info(f"🕒 **Cached Value**: `{ts_data}` *(TTL = 15s)*")

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("🔄 Clear Function Cache (`.clear()`)", key="btn_clear_func"):
            get_timestamped_data.clear()
            st.rerun()
    with btn_c2:
        if st.button("🧹 Flush All Caches (`st.cache_data.clear()`)", key="btn_clear_all"):
            st.cache_data.clear()
            st.rerun()

st.divider()

# ==============================================================================
# 7. Cache Memory Usage & 8. Cache Debugging
# ==============================================================================
st.header("7. Cache Memory Footprint & 8. Cache Debugging Best Practices", divider="rainbow")
st.markdown(
    r"""
### The Mutating Return Value Gotcha
In `@st.cache_data`, Streamlit returns a **deep copy** of cached objects to prevent 
mutations from corrupting the master cache. However, returning massive DataFrames 
causes substantial copy overhead!

### Debugging Checklist
1. **Never cache functions with side-effects**: 
   A cached function should be pure. Never put `st.write()` or `st.metric()` inside a cached function, 
   as those UI calls will only render on cache misses!
2. **Handle unhashable types with `hash_funcs`**:
   If an argument cannot be hashed (e.g., custom threading locks, unpicklable objects), 
   provide a custom hash function:
   ```python
   @st.cache_data(hash_funcs={CustomDBConnection: lambda conn: conn.session_id})
   def query(conn): ...
   ```
3. **Use the Streamlit In-App Cache Manager**: 
   Click `Settings (≡)` $\to$ `Clear cache` or press **`C`** on your keyboard to instantly wipe all cache state during development.
"""
)

col_dbg1, col_dbg2 = st.columns(2)

with col_dbg1:
    st.markdown("**💻 Anti-Pattern: Side Effects Inside Cached Functions**")
    code_anti = '''# ❌ ANTI-PATTERN: UI call inside cached function!
@st.cache_data
def bad_function():
    st.success("Data loaded!")  # ⚠️ Only shows on cache MISS!
    return [1, 2, 3]            # On cache HIT, UI is completely silent!

# ✅ RECOMMENDED PATTERN: Pure function with caller UI:
@st.cache_data
def good_function():
    return [1, 2, 3]            # Pure computation only

# Handle UI in caller:
data = good_function()
st.success("Data ready!")'''
    st.code(code_anti, language="python")

with col_dbg2:
    st.markdown("**🖥️ Live Interactive Memory & Mutation Safety Demo**")
    @st.cache_data
    def get_base_list():
        return ["Embedding A", "Embedding B"]

    base_list = get_base_list()
    st.write(f"Original Cached Output: `{base_list}`")
    
    if st.button("Attempt In-Place Mutation (`.append`)", key="btn_mutate_demo"):
        base_list.append("Malicious In-Place Mutation")
        st.warning(f"Mutated Local Copy: `{base_list}`")
        st.success(f"Original Cache Unharmed on Next Call: `{get_base_list()}`")
        st.caption("Streamlit's `@st.cache_data` safely returned a copy, preventing corruption of the stored cache entry.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Caching Diagnostic Quick Reference", divider="rainbow")
st.markdown(
    r"""
| Parameter / Concept | Function / Syntax | Primary Purpose |
| :--- | :--- | :--- |
| **Data Caching** | `@st.cache_data` | Computations, DataFrames, API JSON responses |
| **Resource Caching** | `@st.cache_resource` | Database connections, ML model weights, singletons |
| **Time-To-Live** | `ttl="15m"` or `ttl=900` | Evicts stale data automatically after duration |
| **Entry Ceiling** | `max_entries=200` | Limits memory usage using LRU eviction |
| **Single Invalidation** | `func.clear(*args)` | Clears cache for specific function or parameter slice |
| **Global Invalidation** | `st.cache_data.clear()` | Purges all `@st.cache_data` storage across the app |
| **Development Wipe** | Press **`C`** key in browser | Instant cache reset in Streamlit developer mode |
"""
)
