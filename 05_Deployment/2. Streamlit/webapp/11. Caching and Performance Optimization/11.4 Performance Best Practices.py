import streamlit as st
import time
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from io import BytesIO
from PIL import Image

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Performance Best Practices",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Streamlit Performance & Optimization — Complete Guide")
st.markdown(
    """
Building a fast, responsive Streamlit application requires deliberate architecture. 
Because Streamlit re-executes top-to-bottom on user interactions, small inefficiencies 
compound quickly into sluggish UIs and server Out-Of-Memory (OOM) crashes. 
This guide details the **10 core performance engineering pillars** for production apps.
"""
)

st.divider()

# ==============================================================================
# 1. Avoiding Unnecessary Computation & 2. Reducing Repeated API Calls
# ==============================================================================
st.header("1. Avoiding Unnecessary Computation & 2. Reducing API Calls", divider="rainbow")
st.markdown(
    r"""
### The Golden Rule of Execution Scoping
Never place expensive calculations or external API queries in the global script flow 
unless guarded by **`@st.cache_data`**, **`@st.cache_resource`**, or **form submission barriers (`st.form`)**.

```python
# ❌ Anti-Pattern: Runs an expensive LLM embedding call on EVERY keystroke!
query = st.text_input("Enter search")
embedding = call_openai_embeddings(query)

# ✅ Recommended Pattern: Guard with a Form or Cache!
with st.form("search_form"):
    query = st.text_input("Enter search")
    submitted = st.form_submit_button("Search")
    if submitted:
        embedding = get_cached_embedding(query)
```
"""
)

col_unc1, col_unc2 = st.columns(2)

with col_unc1:
    st.markdown("**💻 Python Source Code (Debounced Form Barrier)**")
    code_form_guard = '''with st.form("llm_api_barrier"):
    st.markdown("##### Batch Parameter Submission")
    model = st.selectbox("Model", ["Claude-3.7", "GPT-4.5"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    prompt = st.text_area("Prompt", "Explain quantum computing")
    
    # Prevents rerunning API calls during slider adjustment:
    execute_btn = st.form_submit_button("🚀 Dispatch to LLM")
    
if execute_btn:
    result = dispatch_cached_api(model, temperature, prompt)'''
    st.code(code_form_guard, language="python")

with col_unc2:
    st.markdown("**🖥️ Live Interactive Form Submission Barrier**")
    with st.form("live_api_guard_form"):
        st.caption("Adjusting these widgets produces ZERO server rerun overhead until submitted:")
        g_model = st.selectbox("Target Model", ["Claude 3.7", "GPT-4.5", "DeepSeek R1"], key="p_guard_mod")
        g_temp = st.slider("Temperature Sampling", 0.0, 1.0, 0.7, 0.05, key="p_guard_temp")
        g_btn = st.form_submit_button("🚀 Dispatch Request")

    if g_btn:
        st.success(f"Dispatched request for `{g_model}` at temperature `{g_temp}` atomically!")
    else:
        st.info("Form barrier active: Widget interactions are completely free.")

st.divider()

# ==============================================================================
# 3. Efficient DataFrame Processing & 4. Loading Data Lazily
# ==============================================================================
st.header("3. Efficient DataFrame Processing & 4. Lazy Data Loading", divider="rainbow")
st.markdown(
    r"""
### Downcasting & Vectorization
- **Use Parquet instead of CSV**: Parquet preserves columnar compression and data types, loading up to **10× faster** than CSV.
- **Filter at the Source (Pushdown Predicates)**: Filter rows and select only required columns *before* pulling into Pandas (`SELECT colA, colB FROM table WHERE date > ...`).
- **Downcast Numerical Types**: Convert 64-bit floats to 32-bit floats and string categories to `category` dtypes to slash memory by 75%.
"""
)

col_df1, col_df2 = st.columns(2)

with col_df1:
    st.markdown("**💻 Python Source Code (Memory Downcasting)**")
    code_downcast = '''def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to reduce RAM footprint."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    return df'''
    st.code(code_downcast, language="python")

with col_df2:
    st.markdown("**🖥️ Live Interactive Memory Profiling Demo**")
    n_sample = 20000
    df_raw = pd.DataFrame({
        "ID": np.arange(n_sample, dtype=np.int64),
        "Score": np.random.uniform(0.0, 100.0, size=n_sample).astype(np.float64),
        "Category": np.random.choice(["Tier A", "Tier B", "Tier C"], size=n_sample).astype(object)
    })
    raw_mem = df_raw.memory_usage(deep=True).sum() / 1024

    df_opt = df_raw.copy()
    df_opt["ID"] = pd.to_numeric(df_opt["ID"], downcast="integer")
    df_opt["Score"] = pd.to_numeric(df_opt["Score"], downcast="float")
    df_opt["Category"] = df_opt["Category"].astype("category")
    opt_mem = df_opt.memory_usage(deep=True).sum() / 1024

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Standard DataFrame RAM", f"{raw_mem:.1f} KB")
        m2.metric("Optimized DataFrame RAM", f"{opt_mem:.1f} KB")
        m3.metric("RAM Reduction", f"-{(1 - opt_mem/raw_mem)*100:.1f}%", delta_color="inverse")
        st.caption("Downcasting 64-bit primitives to 32-bit and categories slashes memory instantly.")

st.divider()

# ==============================================================================
# 5. Limiting Large Table Rendering (Virtualization & Pagination)
# ==============================================================================
st.header("5. Limiting Table Rendering (`st.dataframe` vs `st.table`)", divider="rainbow")
st.markdown(
    r"""
Never render massive datasets using static `st.table()`! 
- **`st.table()`**: Renders all rows directly as raw HTML DOM elements. A 10,000-row table creates 50,000+ DOM nodes, freezing the browser!
- **`st.dataframe()`**: Employs **virtual scrolling** (Glide Data Grid). Only the ~20 rows currently visible on screen are rendered into the DOM.
- **Server-Side Pagination**: For millions of records, slice using `.iloc[start:end]` or SQL `LIMIT / OFFSET`.
"""
)

col_tbl1, col_tbl2 = st.columns(2)

with col_tbl1:
    st.markdown("**💻 Python Source Code (Interactive Pagination)**")
    code_page = '''PAGE_SIZE = 20
total_pages = int(np.ceil(len(large_df) / PAGE_SIZE))
page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

start_idx = (page_num - 1) * PAGE_SIZE
end_idx = start_idx + PAGE_SIZE

# Slices only 20 rows into memory for rendering:
st.dataframe(large_df.iloc[start_idx:end_idx], width="stretch")'''
    st.code(code_page, language="python")

with col_tbl2:
    st.markdown("**🖥️ Live Interactive Pagination Widget**")
    full_dataset = pd.DataFrame({
        "Index": [f"REC-{i:05d}" for i in range(1000)],
        "Value": np.random.normal(50, 15, 1000).round(2),
        "Timestamp": [datetime.now().strftime("%H:%M:%S") for _ in range(1000)]
    })

    PAGE_SIZE = 5
    tot_pages = len(full_dataset) // PAGE_SIZE
    active_page = st.number_input("Select Page (5 rows/page)", 1, tot_pages, 1, key="num_page_sel")
    s_idx = (active_page - 1) * PAGE_SIZE
    e_idx = s_idx + PAGE_SIZE

    with st.container(border=True):
        st.dataframe(full_dataset.iloc[s_idx:e_idx], width="stretch", hide_index=True)
        st.caption(f"Showing rows {s_idx + 1} to {e_idx} of {len(full_dataset)} total rows.")

st.divider()

# ==============================================================================
# 6. Optimizing Image Sizes & 7. Optimizing Chart Rendering
# ==============================================================================
st.header("6. Optimizing Image Payloads & 7. Chart Rendering", divider="rainbow")
st.markdown(
    r"""
### High-Efficiency Media Delivery
- **Resize Images Server-Side**: Never send a 12MB 4K PNG to `st.image()` if it renders at 300px width. Downscale with Pillow first!
- **Downsample Scatter Plots**: Rendering 100,000 scatter markers creates an enormous client-side SVG/JSON payload. Subsample points or use 2D density heatmaps (`px.density_heatmap`).
"""
)

col_med1, col_med2 = st.columns(2)

with col_med1:
    st.markdown("**💻 Python Source Code (Image Resizing & Subsampling)**")
    code_media = '''from PIL import Image

def optimize_image_payload(img: Image.Image, max_dim: int = 400) -> Image.Image:
    """Downsample image dimensions before web transfer."""
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

# Subsampling 100k data points for scatter charts:
def sample_points(df, max_pts=2000):
    if len(df) > max_pts:
        return df.sample(n=max_pts, random_state=42)
    return df'''
    st.code(code_media, language="python")

with col_med2:
    st.markdown("**🖥️ Live Interactive Scatter Downsampler**")
    total_raw_points = 50000
    st.caption(f"Raw Dataset contains **{total_raw_points:,}** telemetry points.")
    
    max_pts_sel = st.slider("Max Display Points (Subsampled)", 500, 5000, 1500, 500, key="sl_subsample")
    
    np.random.seed(42)
    sub_df = pd.DataFrame({
        "Lat": np.random.randn(max_pts_sel),
        "Lon": np.random.randn(max_pts_sel)
    })
    
    st.scatter_chart(sub_df, x="Lat", y="Lon", height=240)
    st.caption(f"Rendered **{max_pts_sel:,}** points cleanly without browser stutter.")

st.divider()

# ==============================================================================
# 8. Profiling Application Performance & 9. Managing Memory Usage
# ==============================================================================
st.header("8. Profiling Application Performance & 9. Managing Memory", divider="rainbow")
st.markdown(
    r"""
### Benchmarking Execution Bottlenecks
Use Python's `time.perf_counter()` or context managers to profile exactly which sections 
of your script consume the most time during reruns.
"""
)

col_prf1, col_prf2 = st.columns(2)

with col_prf1:
    st.markdown("**💻 Python Source Code (Execution Timer Context Manager)**")
    code_profile = '''import time
from contextlib import contextmanager

@contextmanager
def timer(section_name: str):
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    st.caption(f"⏱️ `{section_name}`: **{elapsed:.1f} ms**")

with timer("Feature Engineering Pipeline"):
    df_transformed = run_heavy_transforms(df)

with timer("Plotly Visual Canvas"):
    st.plotly_chart(fig)'''
    st.code(code_profile, language="python")

with col_prf2:
    st.markdown("**🖥️ Live Interactive Section Profiler**")
    from contextlib import contextmanager

    @contextmanager
    def benchmark_timer(name: str):
        t0 = time.perf_counter()
        yield
        el = (time.perf_counter() - t0) * 1000
        st.write(f"⏱️ **{name}**: `{el:.2f} ms`")

    with st.container(border=True):
        st.markdown("##### 📊 Real-Time Execution Breakdown")
        with benchmark_timer("Data Filter Pass"):
            time.sleep(0.04)
        with benchmark_timer("Aggregation Math"):
            time.sleep(0.02)
        with benchmark_timer("JSON Serialization"):
            time.sleep(0.01)

st.divider()

# ==============================================================================
# 10. Avoiding Unbounded Caching (Memory Leak Prevention)
# ==============================================================================
st.header("10. Avoiding Unbounded Caching (Preventing Server OOM)", divider="rainbow")
st.markdown(
    r"""
### The Danger of Unbounded Cache Growth
If a function decorated with `@st.cache_data` receives arbitrary, continuous floating-point inputs 
(e.g., timestamps or random seeds), **Streamlit creates a new cache entry on every single execution**:

```python
# ❌ DANGEROUS: Every timestamp creates a new entry! Will crash server RAM!
@st.cache_data
def get_user_data(timestamp):
    ...

# ✅ FIX 1: Enforce max_entries ceiling
@st.cache_data(max_entries=200, ttl=600)
def get_user_data(user_id):
    ...

# ✅ FIX 2: Round continuous floating-point arguments to discrete buckets
@st.cache_data
def evaluate_threshold(threshold_float):
    bucket = round(threshold_float, 2)  # Quantize input!
    ...
```
"""
)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Performance Engineering Checklist", divider="rainbow")
st.markdown(
    r"""
| Category | Recommended Pattern | Impact |
| :--- | :--- | :--- |
| **Execution Scoping** | Use `st.form` for multi-widget input bundles | Prevents wasteful intermediate reruns |
| **Data Ingestion** | Use `.parquet` over `.csv` | Up to 10× faster load, preserves dtypes |
| **RAM Footprint** | Downcast float64 to float32 & category dtypes | Reduces in-memory DataFrame size by 50–75% |
| **Table Rendering** | Use `st.dataframe` with pagination | Prevents DOM bloating and browser tab freezing |
| **Chart Points** | Subsample scatter points ($\le 2,500$ points) | Smooth 60fps pan/zoom interactivity |
| **Cache Bounding** | Always set `max_entries` on high-cardinality functions | Eliminates Out-Of-Memory (OOM) crashes |
| **Model Weights** | `@st.cache_resource` for ML neural networks | Zero-copy shared VRAM pointer |
"""
)
